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

def effective_hamiltonian_quantum_tensor(c_tensor, hamiltonain:MultiSiteOperator):
    '''
    The hTTN structure is as shown below.
     _________    _________
    |         |  |         |
    |q_tensor |__|c_tensor |
    |_________|  |_________|
    ||  ...   |       |

    '''
    
    

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

    trm = term({0:pauli['X'], 1:np.ones([2,2])},num_sites = 2)
    mso = MultiSiteOperator(3)
    mso.add_uniform_single_site_terms(pauli['X']*(-h))
    mso.add_uniform_local_two_site_terms(pauli['Z']*(-J))       

    
