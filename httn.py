import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector 
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
import qr_decomposition as qr

from pauli import pauli
from operators import MultiSiteOperator, term
from qr_decomposition import qr_tensors, combine, projection_to_closest_unitary

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

def ising_operators(num_sites, 
                    h = 1, J = 1, pbc = True) -> MultiSiteOperator:
    '''
    H = -h\sum_{i} \sigma_x^(i) -J\sum_{i} \sigma_z^(i)\sigma_z^(i+1)

    The canonical form of the Hamiltonian is given in Eq. (1) of PRB 90 125154 (2014)

    with (N+1) = (0) for periodic boundary conditions

    this format just returns the Pauli string. 
    depending on if the index this is contracted with is clasical or quantum, 
    we will either consturct a SprasePauliOp or give the direct Pauli matrix.
    '''
    mso = MultiSiteOperator(num_sites)
    mso.add_uniform_single_site_terms(-pauli['X']*h)
    mso.add_uniform_local_two_site_terms(-pauli['Z']*J)
    return mso

def _expectation_circuit(params, circuit, operator:SparsePauliOp, 
                         estimator = StatevectorEstimator(), shots = None):
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

def expectation_tensor(params, q_tensor, operator: MultiSiteOperator, 
                       estimator = StatevectorEstimator(), shots = None):
    '''
    Evaluated the expectation value of the operator 
    with respect to the quantum tensor q_tensor.

    operator here is a MultiSiteOperator obejct.

    This would involve contractions with the P matrices as well.

    \bra{\psi}P^\dagger O P \ket{psi}
    Essetinally this is taking the expectation value of P^\dagger O P
    '''
    q_circuit, P_matrices = q_tensor
    sandwiched_operator = operator.sandwich(P_matrices)

    spo = sandwiched_operator.to_SparsePauliOp()

    return _expectation_circuit(params, q_circuit, spo, 
                                estimator, shots)
    
def _pop_at_idx(pauli_list, idx):                       # CLEANUP : put these helper functions in a different file. Either that or create a PauliList class that handles all these.
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


def _E(sigma:str, q_tensor, params, idx, mso: MultiSiteOperator,            
       estimator=StatevectorEstimator(), shots=None)->float:            # TODO TEST 

    '''
    Evaluates the expectation value of the operators defined in mso with sigma inserted at idx.
    See Eq. A21 of arXiv:2007.009582v2
    Requires all the inputs of open_link_contraction
    '''
    q_circuit, P_matrices = q_tensor
    # inserting sigma (a pauli string) at index idx
    mat = pauli[sigma]
    if mso is None:                                 # when we have all identities
        trm = term({idx: mat}, q_circuit.num_qubits)
        mso_with_sigma = MultiSiteOperator(q_circuit.num_qubits, {1:{trm}})
        return expectation_tensor(params, q_tensor, mso_with_sigma,
                                  estimator, shots)
        
    mso_with_sigma = mso.push_at_idx(mat, idx)                    # would we not require a pop before push? TODO CHECK.
     
    return expectation_tensor(params, q_tensor, mso_with_sigma, 
                              estimator, shots)


def open_link_contraction(q_tensor, params, idx, mso: MultiSiteOperator = None,        # TODO TEST
                          estimator=StatevectorEstimator(), shots = None )->np.ndarray:          # Think of ways of testing this. TEST
    '''
    See Fig.1, especially Fig. 1e and section 3 of Appendix A, especially Eq. A22 of arXiv:2007.009582v2 and Appendix A of the PRX paper

    See my notes for the derivation.
    
    Evaluates the open link contraction of the quantum tensor q_tensor 
    with respect to the quantum index idx.

    Gives the resultant matrix M
    by evaluating the expectation value of operators (these operators are specified by the mso)
    which are defined on all the indices, leaving the index idx open
    
    Assumes that the q_tensor has no classical indices.
    '''
    q_circuit, P_matrices = q_tensor                    
    assert q_circuit.num_qubits - 1 >= idx, f"Can't have index {idx} on a quantum circuit of {q_circuit.num_qubits} qubits."
    #assert np.sum([len(pauli_op[0])+1-q_circuit.num_qubits for pauli_op in pauli_list]) == 0, 'Each pauli string in pauli_list must have one less than number of operators as the number of qubits in q_circuit'     # make sure that these assert values are working as expected. maybe using allclose would be better.
    # Calculating the expectation values E(I), E(X), E(Y) and E(Z) as defined in Eq. A21 of arXiv:2007.009582v2
    def _E(sigma:str) -> float:
           
        '''
        Evaluates the expectation value of the operators defined in mso with sigma inserted at idx.
        See Eq. A21 of arXiv:2007.009582v2
        Requires all the inputs of open_link_contraction
        '''
        q_circuit, P_matrices = q_tensor
        # inserting sigma (a pauli string) at index idx
        mat = pauli[sigma]
        if mso is None:                                 # when we have all identities
            trm = term({idx: mat}, q_circuit.num_qubits)
            mso_with_sigma = MultiSiteOperator(q_circuit.num_qubits, {1:{trm}})
            return expectation_tensor(params, q_tensor, mso_with_sigma,
                                      estimator, shots)
            
        mso_with_sigma = mso.push_at_idx(mat, idx)                    # would we not require a pop before push? TODO CHECK.
         
        return expectation_tensor(params, q_tensor, mso_with_sigma, 
                                  estimator, shots)

    # Computing M as done in A22 of arXiv:2007.009582v2
    '''

    M = np.zeros([2,2],dtype = complex)

    for sigma in pauli:
        if sigma != 'Y':
            M += _E(sigma)*pauli[sigma]
        else:
            M -= _E(sigma)*pauli[sigma]
    '''

    M = _E('I')*pauli['I'] + _E('X')*pauli['X'] - _E('Y')*pauli['Y'] + _E('Z')*pauli['Z']   # would present a minor speed up if we didnt' loop through the pauli matrices.
    return M*.5


def contract_loop(c_tensor, operators = None):
    '''
    Assumes that c_tensor.shape = (2,2).
    
    Evaluates (A is the c_tensor and O^0 = operators[0])

    A_{i j} O^0_{i i'} O^1_{j j'} A^*_{i' j'}.
    '''
    if operators is None:
        operators =  ['I', 'I']

    c_tensor_conj = np.conj(c_tensor)

    O0 = _pauli_character_to_array(operators[0])
    O1 = _pauli_character_to_array(operators[1])
    
    return np.einsum('ij,ik,jl,kl',c_tensor,O0,O1,c_tensor_conj, optimize = 'greedy')    # if this throws a dtype error, complexify c_tensor.


def norm_of_network(q_tensor, params, c_tensor, 
                    estimator = StatevectorEstimator(), shots = None):
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
    q_circuit, P_matrices = q_tensor            # TODO incomplete`

    num_qubits = q_tensors.num_qubits
    identity_list = [ ('I'*(num_qubits-1), 1) ]
    M_identity = open_link_contraction(q_circuit, params, num_qubits-1, identity_list, estimator, shots)
    norm = contract_loop(c_tensor, operators = [M_identity, 'I'])
    
    # is it possible to change c_tensor here itself? 
    # would c_tensor = c_tensor / norm change c_tensor for good?
    return norm


def isometrise_quantum_tensor(q_tensor, params, iso_idx,
                              estimator = StatevectorEstimator(), shots = None):
    '''
    isometrising the quantum tensor.

    iso_idx is the index with respect to which the quantum_tensor is isomterised.

    returns the new (isometrised) quantum tensor
    and the non isomteric part of the quantum tensor

    See section III A of the PRX paper.
    '''
    q_circuit, P_matrices = q_tensor
    num_qubits = q_circuit.num_qubits
    
    M = open_link_contraction(q_tensor, params, idx = iso_idx, mso = None,
                              estimator = estimator, shots = shots) 
    M = (M + M.conj().T)*.5                         # M should be +ve semi definite as it is. Doing this for increased numerical stability.
    D,U = np.linalg.eig(M)                          # M = (U*D) @ U_dagger = (U @ np.diag(E)) @ U_dagger 
    sqrt_D     = np.sqrt(D)
    sqrt_D_inv = np.linalg.pinv(np.diag(sqrt_D))    # using the pseudo inverse to avoid numerical instability, as done in pg 7 of the PRX paper (below Eq. 13)

    quantum_r = (U.T * sqrt_D).T                    # = np.diag(sqrt_D) @ U
    quantum_r_inv = np.linalg.inv(U) @ sqrt_D_inv
    
    P_matrices_out = P_matrices.copy()
    P_matrices_out[iso_idx] = quantum_r_inv @ P_matrices[iso_idx]
    return (q_circuit, P_matrices_out), quantum_r
    
def isometrise_classical_tensor(c_tensor, iso_idx):
    '''
    sets the quantum tensor as the centre of isometrisation
    by isometrising the classical tensor

    returns the new (isometrised) c_tensor 
    and the new q_tensor (with new P_matrices)
    '''
    
    classical_q, classical_r = qr_tensors(c_tensor, iso_idx)          # because 0th index of the ctensor is contracted with the last index of the qtensor

    # P_matrices[-1] = P_matrices[-1] @ classical_r.T               

    # the last index of the qtensor is contracted with the 0th index of teh classical tensor
    # the 0th index of classical_r is contracted with classical_q
    # and the 1st index of classical_r is contracted with the P matrix.

    return classical_q, classical_r
    
def unitarise_all_P_matrices(q_tensor):
    '''
    projects all the P matrices to the closest unitary
    by the Frobenius norm.
    '''
    q_circuit, P_matrices = q_tensor 

    for idx, P in enumerate(P_matrices):                        # PARALLELIZE: LOW
        P_matrices[idx] = projection_to_closest_unitary(P)

    return (q_circuit, P_matrices)

def effective_hamiltonian_quantum_tensor():
    pass
def effective_hamiltonian_classical_tensor():
    pass

def optimise_quantum_tensor():
    pass

def optimise_classical_tensor(c_tensor):
    pass

if __name__ == '__main__':
    num_quantum_legs = 4
    h,J =1,1


    qt = QuantumTensor(num_quantum_legs)               # we will compare this against the classical ttn defined on 8 sites
    par1 = [0 for _ in range(num_quantum_legs)]
    par2 = [np.random.randn() for _ in range(num_quantum_legs)]

    mso = MultiSiteOperator(3)
    mso.add_uniform_single_site_terms(pauli['X']*(-h))
    mso.add_uniform_local_two_site_terms(pauli['Z']*(-J))       

    
