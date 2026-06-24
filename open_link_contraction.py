import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from pauli import pauli
#from operators import MultiSiteOperator, term

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

def expectation_tensor(params, q_tensor, operator, 
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
    
def expectation_tensor_term(params, q_tensor, trm, 
                       estimator = StatevectorEstimator(), shots = None):
    '''
    Evaluated the expectation value of the operator 
    with respect to the quantum tensor q_tensor.

    trm here is a term object.

    This would involve contractions with the P matrices as well.

    \bra{\psi}P^\dagger O P \ket{psi}
    Essetinally this is taking the expectation value of P^\dagger O P
    '''
    q_circuit, P_matrices = q_tensor
    sandwiched_trm = trm.sandwich(P_matrices)

    spo = sandwiched_trm.to_SparsePauliOp()

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


def _E(sigma:str, q_tensor, params, idx, mso,            
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


def open_link_contraction(q_tensor, params, idx, mso = None,        # TODO TEST
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

def open_link_contraction_term(q_tensor, params, idx, trm,       
                          estimator=StatevectorEstimator(), shots = None )->np.ndarray:          
    q_circuit, P_matrices = q_tensor
    assert q_circuit.num_qubits-1 == trm.num_sites, f"q_circuit has {q_circuit.num_qubits} qubits but term has {trm.num_sites} sites."

    def _E(sigma:str) -> float:
           
        '''
        Evaluates the expectation value of the operators defined in mso with sigma inserted at idx.
        See Eq. A21 of arXiv:2007.009582v2
        Requires all the inputs of open_link_contraction
        '''
        q_circuit, P_matrices = q_tensor
        # inserting sigma (a pauli string) at index idx
        mat = pauli[sigma]
            
        trm_with_sigma = trm.push_at_idx(mat, idx)                   
         
        return expectation_tensor(params, q_tensor, trm_with_sigma, 
                                  estimator, shots)

    M = _E('I')*pauli['I'] + _E('X')*pauli['X'] - _E('Y')*pauli['Y'] + _E('Z')*pauli['Z']   # would present a minor speed up if we didnt' loop through the pauli matrices.
    return .5 * M

