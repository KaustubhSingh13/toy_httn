import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector 
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
import qr_decomposition as qr

from pauli import pauli

'''
I will be implementing the circuit in Fig. 6e of arXiv:2007.009582v2 Quantum simulation with hybrid tensor networks
This had one quantum tensor with n_quantum legs and one of the quantum legs is contracted with a classical tensor of rank 2
The quantum tensor has no classical indices.

I will be finding the ground state energy of the Ising model using this hTTN.
'''

def QuantumTensor(n_legs):
    '''
    A QuantumTensor is implemented as a QuantumCircuit object.

    In the PRX (hybrid tree tensor networks for quantum simulation) paper the quantum circuit
    is obtained by first running a sweep with a classical TTN and then the upper tensors 
    are mapped to QuantumCirucit. This can be done. See ref [82,83] of the PRX paper.

    Here I will be going with the simplest practical ansatz I can think of.

    A quantum tensor will be a tuple containing
    The quantum circuit
    A list of P matrices (see pg.7 of the PRX paper) which are needed for isometrisation
    '''
    qc = QuantumCircuit(n_legs, n_legs)

    # I will have to create a way to distinguish between classical and quantum indices.
    # could inherit some properties of indices to both.
    # This is needed because contraction laws are different
    # Additionally there has to be a difference between classical legs of quantum vs classical tensors

    theta = ParameterVector('theta',n_legs)
    for i in range(n_legs):
        qc.ry(theta[i], i)

    if n_legs > 1:
        for i in range(n_legs - 1):
            qc.cx(i, i+1)
    
    # for now I am initializing the P matrices to be identity. 
    # these will be updated when we isometrise some tensors.
    return qc, [np.eye(2) for _ in range(n_legs)]       

def ClassicalTensor():
    return np.random.randn(2,2)

def ising_operators(num_sites, h = 1, J = 1, pbc = True):
    '''
    H = -h\sum_{i} \sigma_x^(i) -J\sum_{i} \sigma_z^(i)\sigma_z^(i+1)

    The canonical form of the Hamiltonian is given in Eq. (1) of PRB 90 125154 (2014)

    with (N+1) = (0) for periodic boundary conditions

    this format just returns the Pauli string. 
    depending on if the index this is contracted with is clasical or quantum, 
    we will either consturct a SprasePauliOp or give the direct Pauli matrix.
    '''
    local_terms = ['X' for site_idx in range(num_sites)]
    local_coupling = -h
    
    left_interaction_terms = [None for idx in range(num_sites)]
    right_interaction_terms = [None for idx in range(num_sites)]

    if num_sites > 1:
        interaction_coupling = -J

        for site_idx in range(num_sites - 1): left_interaction_terms[site_idx] = 'Z'      # see PRB 90, 125154 (2014) Eq. 1
        for site_idx in range(1, num_sites): right_interaction_terms[site_idx] = 'Z'

        if pbc and num_sites != 2:          # To avoid double counting
            left_interaction_terms[- 1] = 'Z'
            right_interaction_terms[0] = 'Z'
    
    out = {
            'local' : (local_coupling, local_terms),
            'interaction' : (interaction_coupling, left_interaction_terms, right_interaction_terms)
           }
    
    '''
    for now I have fixed the number of 'terms' you can have.
    say for instance your Hamiltonian defined on one site is X^(0) + Y^(0). (pauli matrices)
    One might say that there are two terms in this Hamiltonian.
    I am defining this to be one term: (X^(0) + Y^(0)). 
    This presents a problem when the number of summands is not equal to the number of terms (like above)
    This is not there for the Ising model but this is something to look out for.
    Need to present the summands of the Hamiltonian with more control
    on which operator acts on which index.

    Also need to find a way to account for more than one body interactions
    Also need to fix this so that it works for long range interactions like in LMG.
    Also should be able to handle cases when there are no interaction terms
    '''
    return out

def _pauli_list_from_operator(hamiltonian_out):
    '''
    Takes the output of something like the ising_operator
    and returns the associated list containing Pauli strings
    compatible with SprasePauliOp.from_list
    '''
    
    local_coupling, local_terms = hamiltonian_out['local']
    interaction_coupling, left_interaction_terms, right_interaction_terms = hamiltonian_out['interaction']
    
    string_length = len(local_terms)            

    assert len(local_terms) == len(left_interaction_terms) == len(right_interaction_terms), 'Elements of each term must map to a single site.'

    def _pad_identities(bare_pauli_str, site_idx):
        # entire function would not support long range interactions
        out = 'I'*site_idx + bare_pauli_str + 'I'*(string_length - len(bare_pauli_str) - site_idx)
        return out
    
    # adding the local terms
    out_list = [(_pad_identities(bare_pauli_str = op, site_idx = idx), local_coupling) for idx, op in enumerate(local_terms)]
    
    # adding the interaction terms
    for site_idx in range(string_length-1):
        left_pauli_op = left_interaction_terms[site_idx]
        right_pauli_op = right_interaction_terms[site_idx+1]
        
        out_list.append((_pad_identities(left_pauli_op + right_pauli_op, site_idx), interaction_coupling))
    
    # by hand adding the interaction terms incase we have pbc. 
    # the _pad_identities function is insufficent when left_pauli_op and right_pauli_op are not adjacent. Needs another function to handle that.
    if right_interaction_terms[0] != None:
        # for the ising model this will always only happen when left_interaction_terms[-1] != None.
        # This is veryy crude. Fix.
        left_pauli_op = left_interaction_terms[-1]
        right_pauli_op = right_interaction_terms[0]
        out_list.append((right_pauli_op + 'I'*(string_length - 2) + left_pauli_op, interaction_coupling))

    return out_list

def _pauli_character_to_array(sigma):   # how can this be vectorised so that it takes in a list as an input and applies this operation parallely to all the sigmas inside that list?
    '''
    returns the matrix corresponding to a pauli character
    if sigma is not a string, returns sigma
    '''
    if isinstance(sigma, str):
        return pauli[sigma]
    return sigma

def _array_to_pauli_list(arr):
    '''
    Takes in an ndarray of shape (2,2)
    returns the corresponding pauli list

    np.ndarray([[1,1],[1,1]]) -> [ ('X',1), ('I',1) ]
    '''
    a, b, c, d = arr[0,0], arr[0,1], arr[1,0], arr[1,1]
    I_coeff = .5*(a + c)
    Z_coeff = .5*(a-c)
    # do Y_coeff and X_coeff as well

def _expectation_circuit(params, circuit, operator:SparsePauliOp, estimator = StatevectorEstimator(), shots = None):
    '''
    Evaluates the expectation value of the state formed by circuit
    for the operator defined by operator.

    If the circuit does not have paramaeters or is already bound,
    put params = None
    '''
    if shots is not None:
        estimator.options.default_shots = shots

    pub = (circuit, operator, list([params]))
    job = estimator.run([pub])
    result = job.result()
    expect_value = float(result[0].data.evs[0])
    return expect_value

def _ndarry_list_to_pauli_list(arr):
    '''
    takes a list containing 2 by 2 matrices 
    and gives the corresponding pauli list
    '''


def expectation_tensor(params, q_tensor, operators, estimator = StatevectorEstimator(), shots = None):
    '''
    Evaluated the expectation value of the operator 
    with respect to the quantum tensor q_tensor.

    operators here is assumed to be a list containing ndarryas of ndim 2

    This would involve contractions with the P matrices as well.

    \bra{\psi}P^\dagger O P \ket{psi}
    Essetinally this is taking the expectation value of P^\dagger O P
    '''
    q_circuit, P_list = q_tensor
    P_contracted_operator = []
    for i,O_i in enumerate(operators):                                                      # how can I parallelize this loop
        P_i = P_list[i]
        P_i_dagger = P_i.T.conj()
        P_contracted_operator.append(np.einsum('ij,jk,kl->il',P_i_dagger,O_i,P_i))          # can this be made faster? maybe use tensordot.
    # INCOMPLETE
    # now I will have to convert P_contracted_operator to SparsePauliOp and then use _expectation_circuit
    # In general, wherever I have used q_tensor, I will have to fix it to accomodate the P matrices as well.

def _pop_at_idx(pauli_list, idx):
    '''
    pauli_list = [('XII', -1), ('IXI', -1), ('IIX', -1), ('ZZI', -1), ('IZZ', -1), ('ZIZ', -1)]
    idx = 1
    
    ->
    
    [('XI', -1), ('II', -1), ('IX', -1), ('ZI', -1), ('IZ', -1), ('ZZ', -1)]
    '''
    pauli_list_popped = pauli_list.copy()
    for i,term in enumerate(pauli_list_popped):
        term = list(term)
        term[0] = term[0][:idx] + term[0][idx+1:]
        pauli_list_popped[i] = tuple(term)
    return pauli_list_popped

def _push_at_idx(pauli_list, sigma:str, idx):
    '''
    pauli_list = [('XII', -1), ('IXI', -1), ('IIX', -1), ('ZZI', -1), ('IZZ', -1), ('ZIZ', -1)]
    idx = 1
    sigma = Y

    ->
    
    [('XYII', -1), ('IYXI', -1), ('IYIX', -1), ('ZYZI', -1), ('IYZZ', -1), ('ZYIZ', -1)]
    '''
    pauli_list_with_sigma = pauli_list.copy()
    for i,term in enumerate(pauli_list_with_sigma):
        term = list(term)
        term[0] = term[0][:idx] + sigma + term[0][idx:]
        pauli_list_with_sigma[i] = tuple(term)
    return pauli_list_with_sigma

def _E(sigma:str, q_tensor, params, idx, pauli_list, estimator=StatevectorEstimator(), shots=None)->float:  # for testing the function inside open_link_contraction
    '''
    Evaluates the expectation value of the operators defined in pauli_list with sigma inserted at idx.
    See Eq. A21 of arXiv:2007.009582v2
    Requires all the inputs of open_link_contraction
    '''
    # inserting sigma (a pauli string) at index idx
    pauli_list_with_sigma = _push_at_idx(pauli_list, sigma, idx)
    
    operator = SparsePauliOp.from_list(pauli_list_with_sigma)
    return expectation_tensor(params, q_tensor, operator, estimator, shots)

def open_link_contraction(q_tensor, params, idx, pauli_list, estimator=StatevectorEstimator(), shots = None )->np.ndarray:          # Think of ways of testing this.
    '''
    See Fig.1, especially Fig. 1e and section 3 of Appendix A, especially Eq. A22 of arXiv:2007.009582v2 and Appendix A of the PRX paper

    See my notes for the derivation.
    
    Evaluates the open link contraction of the quantum tensor q_tensor 
    with respect to the quantum index idx.

    Gives the resultant matrix M
    by evaluating the expectation value of operators (these operators are specified by the pauli_list)
    which are defined on all the indices, leaving the index idx open
    
    Assumes that the q_tensor has no classical indices.
    '''
    assert q_tensor.num_qubits - 1 >= idx, f"Can't have index {idx} on a quantum circuit of {q_tensor.num_qubits} qubits."
    assert np.sum([len(pauli_op[0])+1-q_tensor.num_qubits for pauli_op in pauli_list]) == 0, 'Each pauli string in pauli_list must have one less than number of operators as the number of qubits in q_tensors'     # make sure that these assert values are working as expected. maybe using allclose would be better.
    # Calculating the expectation values E(I), E(X), E(Y) and E(Z) as defined in Eq. A21 of arXiv:2007.009582v2
    def _E(sigma:str)->float:
        '''
        Evaluates the expectation value of the operators defined in pauli_list with sigma inserted at idx.
        See Eq. A21 of arXiv:2007.009582v2
        Requires all the inputs of open_link_contraction
        '''
        # inserting sigma (a pauli string) at index idx
        pauli_list_with_sigma = _push_at_idx(pauli_list, sigma, idx)
        
        operator = SparsePauliOp.from_list(pauli_list_with_sigma)
        return _expectation(params, q_tensor, operator, estimator, shots)

    # Computing M as done in A22 of arXiv:2007.009582v2
    M = np.zeros([2,2],dtype = complex)
    for sigma in pauli:
        if sigma != 'Y':
            M += _E(sigma)*pauli[sigma]
        else:
            M -= _E(sigma)*pauli[sigma]

    return M*.5


def contract_loop(c_tensor, operators = ['I', 'I']):
    '''
    Assumes that c_tensor.shape = (2,2).
    
    Evaluates (A is the c_tensor and O^0 = operators[0])

    A_{i j} O^0_{i i'} O^1_{j j'} A^*_{i' j'}.
    '''
    c_tensor_conj = np.conj(c_tensor)

    O0 = _pauli_character_to_array(operators[0])
    O1 = _pauli_character_to_array(operators[1])
    
    return np.einsum('ij,ik,jl,kl',c_tensor,O0,O1,c_tensor_conj, optimize = 'greedy')    # if this throws a dtype error, complexify c_tensor.

def norm_of_network(q_tensor:QuantumCircuit, params, c_tensor, estimator = StatevectorEstimator(), shots = None):
    '''
    This assumes that c_tensor.shape = (2,2)

     _________    _________
    |         |  |         |
    |q_tensor |__|c_tensor |
    |_________|  |_________|
    ||  ...   |       |

    The last index of q_tensor is contracted with
    the 0th index of c_tensor.
    '''
    num_qubits = q_tensors.num_qubits
    identity_list = [ ('I'*(num_qubits-1), 1) ]
    M_identity = open_link_contraction(q_tensor, params, num_qubits-1, identity_list, estimator, shots)
    norm = contract_loop(c_tensor, operators = [M_identity, 'I'])
    
    # is it possible to change c_tensor here itself? 
    # would c_tensor = c_tensor / norm change c_tensor for good?
    return norm


def optimise_classical_tensor(c_tensor):
    pass

num_quantum_legs = 4


quantum_tensor = QuantumTensor(num_quantum_legs)               # we will compare this against the classical ttn defined on 8 sites
op = ising_operators(3)
pauli_list = _pauli_list_from_operator(op)
params = [0,0,0,0]



