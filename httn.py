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
from open_link_contraction import *
import vqe 

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

'''
    *********************************************************************
    This is on branch quantum_quantum_httn
    
    here I will be implementing a httn with two quantum tensors,
    each with three quantum indices. 
    
    The hTTN structure is as shown below.
     ____________     ____________
    |            |   |            |
    | q_tensor_1 |___| q_tensor_2 |
    |____________|   |____________|
      |        |       |        |
      |        |       |        |   

      all indices are quantum in both the quantum tensors.
      The 2nd index of q_tensor_1 is contracted with the 
          0th index of q_tensor_2. 
  
    *********************************************************************
'''

def _check_and_add(term_set, trm):
    '''
    Checks if there is a term in term_set 
    which has support on the same sites as trm.
    If so, then adds a term equal to the sum of the two terms and removes the preexisting term
    if not, then simply adds trm to term_set.
    '''
    sites_with_support = trm.keys()

    for trmm in term_set:                       # TODO Speedup: This can be O(1) instead of O(N) if we use hash for the set right.
        if trmm.keys() == sites_with_support:
            out_trm = trmm + trm
            term_set.discard(trmm)
            term_set.add(out_trm)
            return term_set
    term_set.add(trm)
    return term_set

def effective_hamiltonian_q_tensor_1(q_tensor_1, param_1, q_tensor_2, param_2, 
                                     hamiltonian:MultiSiteOperator, 
                                     estimator = StatevectorEstimator(), shots = None,
                                     get_energy = True):
    '''
    gives the local effective hamiltonian for quantum_tensor_1
    '''
    qc_2, p_matrices_2 = q_tensor_2
    
    #   FOR NOW I AM HARDCODING TWO BODY INTERACTIONS
    out_terms = {1:set(), 2:set()}
    for num_supported_sites, trm_set in hamiltonian.terms.items():
        for trm in trm_set:
            if set(trm.keys()) & {2,3} == set(trm.keys()):      
                # if the sites with support in trm are in {2,3}
                # we contract the loop with that part of the hamiltonian
                # for now i am popping the matrices at 0 and 1 even though i have ensured there aren't any. this is to avoid any bt with the term.to_pauli_list() method. 
                # next time look for a more elegant and efficient solution
                M = open_link_contraction_term(q_tensor = q_tensor_2, params = param_2, idx = 0, trm = trm.pop_at_idx(0,1),
                                               estimator = estimator, shots = shots)

                M_with_p = p_matrices_2[0] @ M @ p_matrices_2[0].conj().T
                new_term = term( site_ops = {2: M_with_p}, num_sites = 3
                        )   # because q_tensor_1 has 3 indices, and M is contracted on index 2 (the last index)
                out_terms[1] = _check_and_add(out_terms[1],new_term)
                #out_terms[1].add(new_term)

            elif set(trm.keys()) & {2,3} != set():
                # if there is a non zero intersection in the sites with support
                # we would have to create a term with support on two sites for the effective hamiltonain
                M = open_link_contraction_term(q_tensor = q_tensor_2, params = param_2, idx = 0, trm = trm.pop_at_idx(0,1),

                                               estimator = estimator, shots = shots)

                M_with_p = p_matrices_2[0] @ M @ p_matrices_2[0].conj().T

                # so far these lines are the same the previous if statement
                # reudce this repetetiveness in code when you have the will to live by having cleverer conditionals
                other_idx = list(set(trm.keys()) & {0,1})[0]    # assuming only two body interactions

                new_term = term( site_ops = {other_idx: trm[other_idx], 2: M_with_p}, 
                                 num_sites = 3
                                )   # because q_tensor_1 has 3 indices, and M is contracted on index 2 (the last index) for more than 2 body interactions we would have to fix this line and the following line

                out_terms[2] = _check_and_add(out_terms[2],new_term)
                #out_terms[2].add(new_term)

            else:
                # incase there is no intersection of the sites with support with {2,3} whatsoever
                # there has to be a more elegant way of doing this using the methods i have defined
                # but i am too tired for all that
                new_site_ops = trm.site_ops
                new_trm = term(new_site_ops, num_sites  = 3)

                out_terms[new_trm.num_supported_sites] = _check_and_add(out_terms[new_trm.num_supported_sites],new_trm)
                #out_terms[new_trm.num_supported_sites].add(new_trm)
    
    mso_out = MultiSiteOperator(num_sites = 3, terms = out_terms)

    if get_energy:
        energy = expectation_tensor(params_1, q_tensor_1, mso_out)
        return mso_out, energy

    return mso_out, None

def effective_hamiltonian_q_tensor_2(q_tensor_1, param_1, q_tensor_2, param_2,
                                     hamiltonian:MultiSiteOperator, 
                                     estimator = StatevectorEstimator(), shots = None,
                                     get_energy = True):
    '''
    gives the local effective hamiltonian for quantum_tensor_1
    '''
    qc_1, p_matrices_1 = q_tensor_1

    
    #   FOR NOW I AM HARDCODING TWO BODY INTERACTIONS
    out_terms = {1:set(), 2:set()}
    for num_supported_sites, trm_set in hamiltonian.terms.items():
        for trm in trm_set:
            if set(trm.keys()) & {0,1} == set(trm.keys()):      
                # if the sites with support in trm are in {2,3}
                # we contract the loop with that part of the hamiltonian
                # for now i am popping the matrices at 0 and 1 even though i have ensured there aren't any. this is to avoid any bt with the term.to_pauli_list() method. 
                # next time look for a more elegant and efficient solution
                M = open_link_contraction_term(q_tensor = q_tensor_1, params = param_1, idx = 2, trm = trm.pop_at_idx(2,3),
                                               estimator = estimator, shots = shots)

                M_with_p = p_matrices_1[2] @ M @ p_matrices_1[2].conj().T
                new_term = term( site_ops = {0: M_with_p}, num_sites = 3
                        )   # because q_tensor_1 has 3 indices, and M is contracted on index 2 (the last index)
                out_terms[1] = _check_and_add(out_terms[1],new_term)
                #out_terms[1].add(new_term)

            elif set(trm.keys()) & {0,1} != set():
                # if there is a non zero intersection in the sites with support
                # we would have to create a term with support on two sites for the effective hamiltonain
                M = open_link_contraction_term(q_tensor = q_tensor_1, params = param_1, idx = 2, trm = trm.pop_at_idx(2,3),

                                               estimator = estimator, shots = shots)

                M_with_p = p_matrices_1[2] @ M @ p_matrices_1[2].conj().T

                # so far these lines are the same the previous if statement
                # reudce this repetetiveness in code when you have the will to live by having cleverer conditionals
                other_idx = list(set(trm.keys()) & {2,3})[0] -1 # This is veryyy inelegant and haphazard. this would not generalise at all. 
                                                                # TODO: add a dict which matches the site index with the tensor indices it is contracted with so we dont have to hardcode these things.
                                                                # it works for effective_hamiltonian_q_tensor_1 because there the site index of the terms is equal to the tensor index it is contracted with.
                new_term = term( site_ops = {other_idx: trm[other_idx], 0: M_with_p}, 
                                 num_sites = 3
                                )   # because q_tensor_1 has 3 indices, and M is contracted on index 2 (the last index) for more than 2 body interactions we would have to fix this line and the following line

                out_terms[2] = _check_and_add(out_terms[2],new_term)
                #out_terms[2].add(new_term)

            else:
                # incase there is no intersection of the sites with support with {2,3} whatsoever
                # there has to be a more elegant way of doing this using the methods i have defined
                # but i am too tired for all that
                
                new_site_ops = {site_idx -1: mat for site_idx,mat in trm.items()} # this is veryy haphazard and hardcoded. it would not generalise well at all
                new_trm = term(new_site_ops, num_sites  = 3)


                out_terms[new_trm.num_supported_sites] = _check_and_add(out_terms[new_trm.num_supported_sites],new_trm)
                #out_terms[new_trm.num_supported_sites].add(new_trm)
    
    mso_out = MultiSiteOperator(num_sites = 3, terms = out_terms)

    if get_energy:
        energy = expectation_tensor(params_2, q_tensor_2, mso_out)
        return mso_out, energy

    return mso_out, None



def optimise_quantum_tensor(q_tensor, param, effective_hamiltonian:MultiSiteOperator,
                            estimator = StatevectorEstimator(), shots = None):
    qc, p_matrices = q_tensor
    ham = effective_hamiltonian.sandwich(p_matrices)
    spo = ham.to_SparsePauliOp()
    res = vqe.run_vqe(qc, param, spo,
                      estimator = estimator, shots = shots)
    new_params = res.x
    return new_params 


def run_sweep(q_tensor_1, init_params_1,            # TODO: ADD normalisation functions
              q_tensor_2, init_params_2,
              hamiltonian: MultiSiteOperator, num_sweeps = 10,
              get_energy = True):
    """
    we will first set quantum_tensor_1 to be the isometrisation centre
    and then optimise it before shifting the isometrisation centre 
    to quantum_tensor_2 and then optimising that. 

    We will then shift the isometrisation centre back to quantum_tensor_1
    and repeat. 
    """
    energies = []

    # first setting q_tensor_1 to be the isometrisation centre
    q_tensor_2, R_2 = isometrise_quantum_tensor(q_tensor_2, init_params_2, 0)
    
    q_circuit_1, p_matrices_1 = q_tensor_1 
    p_matrices_1[2] = p_matrices_1[2] @ R_2.T           # absorbing R_2 into q_tensor_1
    q_tensor_1 = (q_circuit_1, p_matrices_1)
    q_tensor_1 = unitarise_all_P_matrices(q_tensor_1)   # unitarising the p matrices.

    params_1 = init_params_1 
    params_2 = init_params_2

    for sweep_idx in range(num_sweeps):
        # obtaining the effective_hamiltonain for q_tensor_1
        eff_ham_1,energy = effective_hamiltonian_q_tensor_1(q_tensor_1, params_1, 
                                                            q_tensor_2, params_2, hamiltonian,
                                                            get_energy = get_energy)
        energies.append(energy)

        # optimising the effective_hamiltonian for q_tensor_1
        params_1 = optimise_quantum_tensor(q_tensor_1, params_1, eff_ham_1)

        # shifting the isometrisation centre to q_tensor_2
        q_tensor_1, R_1 = isometrise_quantum_tensor(q_tensor_1, params_1, 2)
        
        q_circuit_2, p_matrices_2 = q_tensor_2 
        p_matrices_2[0] = p_matrices_2[0] @ R_1.T           # absorbing R_1 into q_tensor_2
        q_tensor_2 = (q_circuit_2, p_matrices_2)
        q_tensor_2 = unitarise_all_P_matrices(q_tensor_2)   # unitarising the p matrices.
        
        # obtaining the effective hamiltonian for q_tensor_2
        eff_ham_2,energy = effective_hamiltonian_q_tensor_2(q_tensor_1, params_1,
                                                            q_tensor_2, params_2, hamiltonian,
                                                            get_energy = get_energy)
        energies.append(energy)

        # optimising the effective_hamiltonain for q_tensor_2
        params_2 = optimise_quantum_tensor(q_tensor_2, params_2, eff_ham_2)
        
        # shifting the isometrisation centre back to q_tensor_1
        q_tensor_2, R_2 = isometrise_quantum_tensor(q_tensor_2, params_2, 0)
        
        q_circuit_1, p_matrices_1 = q_tensor_1 
        p_matrices_1[2] = p_matrices_1[2] @ R_2.T           # absorbing R_2 into q_tensor_1
        q_tensor_1 = (q_circuit_1, p_matrices_1)
        q_tensor_1 = unitarise_all_P_matrices(q_tensor_1)   # unitarising the p matrices.
    
    return energies
        

# implementing the Ising model Hamiltonain
hamiltonian_mso = MultiSiteOperator(4)
h,J = 1,1
hamiltonian_mso.add_uniform_single_site_terms(-h*pauli['X'])
hamiltonian_mso.add_uniform_local_two_site_terms(pauli['Z'], pbc = False)
hamiltonian_mso.multiply_by_scalar(num_supported_sites = 2, scalar = -1)

# initialising the quanutm tensors from the random normal distribution
q_tensor_1 = QuantumTensor(3)
params_1 = np.random.randn(q_tensor_1[0].num_parameters)

q_tensor_2 = QuantumTensor(3)
params_2 = np.random.randn(q_tensor_2[0].num_parameters)



if __name__ == '__main__':
    num_quantum_legs = 4
    h,J =1,1


    qt = QuantumTensor(num_quantum_legs)               # we will compare this against the classical ttn defined on 8 sites
    par1 = [0 for _ in range(num_quantum_legs)]
    par2 = [np.random.randn() for _ in range(num_quantum_legs)]

    trm = term({0:pauli['X'], 1:np.ones([2,2])},num_sites = 2)
    mso = MultiSiteOperator(3)
    mso.add_uniform_single_site_terms(pauli['X']*(-h))
    mso.add_uniform_local_two_site_terms(pauli['Z']*(-J))       

    
