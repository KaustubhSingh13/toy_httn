import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

'''
If you don't want to do all this 
to import a file from the parent directory then run
python3 -m toy_httn.testing.testing_pauli_interface.py
from the project root.

In that case you will have to replace
from pauli import pauli 
by
from ..pauli import pauli
'''
import numpy as np
import time
from qiskit.quantum_info import SparsePauliOp
import matplotlib.pyplot as plt

from pauli import pauli
from operators import *

def term_to_pauli_list_test(trm:term):
    '''
    tests term.to_matrix()
    and term.to_pauli_list()
    '''
    pauli_list = trm.to_pauli_list()
    return np.allclose(trm.to_matrix(), SparsePauliOp.from_list(pauli_list).to_matrix()) 

def mso_to_pauli_list_test(mso:MultiSiteOperator):
    '''
    tests MultiSiteOperator.to_pauli_list
    and MultiSiteOperator.to_matrix
    '''
    pauli_list = trm.to_pauli_list()
    return np.allclose(mso.to_pauli_list(), SparsePauliOp.from_list(pauli_list).to_matrix())

def _not_repeated_randint(lo, hi, num):
    '''
    gives a list of num elements
    wher each element is an int, chosen on random 
    from between lo (inclusive) to hi (exclusive)
    '''
    assert num <= (hi-lo), f"can't pick {num} integers from the interval [{lo},{hi}) which are all different." 
    out = []
    for _ in range(num):
        elem = np.random.randint(lo, hi)
        if elem not in out:
            out.append(elem)
        else:
            remaining_elems = list(set(range(lo,hi)) -  set(out))
            idx = np.random.randint(0, len(remaining_elems))
            out.append(remaining_elems[idx])
    return out

def create_random_term(num_sites, num_supported_sites, 
                       lo = -10, hi = 10, dtype = complex, hermitian = True):
    '''
    Instantiates a random term object
    defined on num_sites sites
    with support on num_supported_sites.

    Each 2 by 2 matrix will have random elements
    between lo and hi. 

    if dtype is complex,
    then the real and imag parts will be sampled 
    from ints between
    '''
    assert num_sites >= num_supported_sites, 'num_sites must be greater than or equal to the num_supported_sites.'
    supported_site_idxs = _not_repeated_randint(0, num_sites, num_supported_sites)
    
    trm = {}
    for idx in supported_site_idxs:

        # when dtype is complex, sample real and imag parts separately
        if dtype is complex:
            A_real = np.random.uniform(lo, hi, size = (2,2))
            A_imag = np.random.uniform(lo, hi, size = (2,2))
            A = A_real + 1j*A_imag

            # extracting the Hermitian part
            if hermitian:
                trm[idx] = .5 * (A + A.T.conj())
            else:
                trm[idx] = A
        else:
            A = np.random.uniform(lo, hi, shape = (2,2))
            if hermitian:
                trm[idx] = .5 * (A + A.T)
            else:
                trm[idx] = A
    return term(trm)

def create_random_MultiSiteOperator(num_sites, num_terms_dict, 
                                    lo = -10, hi = 10, dtype = complex, hermitian = True):
    '''
    num_terms_dict = {1: 2, 2: 7, 5:9} 
    => 
    there are 2 terms with support on 1 site
              7 terms with support on 2 sites
              9 terms with support on 5 sites

    within each term, the matrices will have elements
    chosen on random between lo and hi
    '''
    largest_num_supported_sites = max(num_terms_dict.keys())
    assert num_sites >= largest_num_supported_sites , "can't have support on {largest_num_supported_sites} on a lattice with {num_sites} sites"
    mso_terms = {}

    for num_supported_sites, num_terms in num_terms_dict.items():
        mso_terms[num_supported_sites] = set()
        for _ in range(num_terms):
            trm = create_random_term(num_sites, num_supported_sites, 
                                     lo, hi, dtype, hermitian)
            mso_terms[num_supported_sites].add(trm)

    mso = MultiSiteOperator(num_sites)
    mso.add_all_terms(mso_terms)
    return mso

# for now MultiSiteOperator.to_pauli_list() only works for Hermitian operators.
# I will check if including a hermitian flag significantly impacts speed.

def check_term_to_pauli_list(runs = 5, max_num_sites = 10,
                             lo = -10, hi = 10, dtype = complex, hermitian = True, 
                             verbose = True, give_times = True):
    '''
    Rigorous testing for term.to_pauli_list()

    For each site between 1 and max_num_sites
        for each run
            creates a random term with support sites 
            ranging from 1 to the current site number
            converts it to a matrix
            and to the pauli list which is then converted to a SparsePauliOp
            and checks if that matrix is equal to the matrix
            given by SparsePauliOp.to_matrix()

    Also reports the average time 
    '''
    print('\n\n')
    print('\t\t\t\t\t my time (s)\tSparsePauliOp time (s)\t\tPauli list conversion time (s)')
    pauli_conversion_times, qiskit_matrix_conversion_times, matrix_conversion_times =[],[],[]
    pauli_conversion_var, qiskit_matrix_conversion_var, matrix_conversion_var =[],[],[]
    for num_sites in range(1,max_num_sites+1):
        fx_pauli_conversion_times, fx_matrix_conversion_times, fx_qiskit_matrix_conversion_times = [], [], []
        fx_pauli_conversion_var, fx_matrix_conversion_var, fx_qiskit_matrix_conversion_var = [], [], []
        if verbose:
            print(f'num_sites = {num_sites}')
        for num_supported_sites in range(1,num_sites+1):

            pc, mc, qmc = [], [], []    # this will contain run times for fixed num_sites and num_supported_sites for all the runs
            for run_idx in range(runs):
                trm = create_random_term(num_sites, num_supported_sites,
                                         lo, hi, dtype, hermitian)
                
                pl_st = time.perf_counter()
                pauli_list = trm.to_pauli_list()
                pl_et = time.perf_counter()
                
                my_mat_st = time.perf_counter()
                my_mat = trm.to_matrix()
                my_mat_et = time.perf_counter()
                
                spo = SparsePauliOp.from_list(pauli_list)
                qiskit_mat_st = time.perf_counter()
                qiskit_mat = spo.to_matrix()
                qiskit_mat_et = time.perf_counter()

                if not np.allclose(my_mat, qiskit_mat):
                    return False, trm

                pc.append(pl_et-pl_st)
                mc.append(my_mat_et-my_mat_st)
                qmc.append(qiskit_mat_et-qiskit_mat_st)
            
            mmc = np.mean(mc)
            qmc = np.mean(qmc)

            fx_pauli_conversion_times.append(np.mean(pc))
            fx_matrix_conversion_times.append(mmc)
            fx_qiskit_matrix_conversion_times.append(np.mean(qmc))

            fx_pauli_conversion_var.append(np.var(pc))
            fx_matrix_conversion_var.append(np.var(mc))
            fx_qiskit_matrix_conversion_var.append(np.var(qmc))
            
            if verbose:
                diff = 'mine faster' if mmc<qmc else 'qiskit faster'
                print(f'\t num_supported_sites = {num_supported_sites} done\t {mmc:.2e}\t{qmc:.2e}\t{diff}\t{np.mean(pc):.2e}')


        pauli_conversion_times.append(fx_pauli_conversion_times)
        matrix_conversion_times.append(fx_matrix_conversion_times)
        qiskit_matrix_conversion_times.append(fx_qiskit_matrix_conversion_times)

        pauli_conversion_var.append(fx_pauli_conversion_var)
        matrix_conversion_var.append(fx_matrix_conversion_var)
        qiskit_matrix_conversion_var.append(fx_qiskit_matrix_conversion_var)
        if verbose:
            print('\n')

    if give_times:
        return True, [
                        (pauli_conversion_times,         pauli_conversion_var), 
                        (matrix_conversion_times,        matrix_conversion_var), 
                        (qiskit_matrix_conversion_times, qiskit_matrix_conversion_var)
                     ] 
    return True

def _random_num_terms_dict(num_sites, max_num_terms):
    '''
    creates a random num_terms_dict
    with max site index being num_sites - 1.
    
    num_terms_dict = {1: 2, 2: 7, 5:9} 
    => 
    there are 2 terms with support on 1 site
              7 terms with support on 2 sites
              9 terms with support on 5 sites
    '''
    num_supported_sites = _not_repeated_randint(1,num_sites+1, np.random.randint(1,num_sites+1))
    out = {}
    for val in num_supported_sites:
        out[val] = np.random.randint(1,max_num_terms+1)
    return out
    

def check_MultiSiteOperator_to_pauli_list(runs = 10, max_num_sites = 5, max_num_terms = 5,
                                          lo = -10, hi = 10, dtype = complex, hermitian = True,
                                          verbose = True, give_times = True):
    '''
    for each run
        for num_sites up to max_num_sites
            creates a random num_terms_dict
            and instantians a random MultiSiteOperator with that num_terms_dict
            gets the corresponding pauli list
            uses that pauli list to create a SparsePauliOp
            checks the MultiSiteOperator.to_matrix() against SparsePauliOp.to_matrix()

    This tests the to_matrix() method and the to_pauli_list() method
    '''
    if verbose:
        print('my time (s)\tSparsePauliOp time (s)\t\tPauli list conversion time (s)\tnum_terms_dict')

    pauli_conversion_times, qiskit_matrix_conversion_times, matrix_conversion_times =[],[],[]
    
    for num_sites in range(1,max_num_sites+1):
        for run_idx in range(runs):

            num_terms_dict = _random_num_terms_dict(num_sites, max_num_terms)
            mso = create_random_MultiSiteOperator(num_sites,num_terms_dict,
                                              lo, hi, dtype, hermitian)
            
            pl_st = time.perf_counter()
            pauli_list = mso.to_pauli_list()
            pl_et = time.perf_counter()

            spo = SparsePauliOp.from_list(pauli_list)

            my_mat_st = time.perf_counter()
            my_mat = mso.to_matrix()
            my_mat_et = time.perf_counter()

            qiskit_mat_st = time.perf_counter()
            qiskit_mat = spo.to_matrix()
            qiskit_mat_et = time.perf_counter()
            
            if not np.allclose(my_mat, qiskit_mat):
                return False, mso

            mmc = my_mat_et - my_mat_st
            qmc = qiskit_mat_et - qiskit_mat_st
            pauli_conversion_times.append(pl_et-pl_st)
            matrix_conversion_times.append(my_mat_et-my_mat_st)
            qiskit_matrix_conversion_times.append(qiskit_mat_et-qiskit_mat_st)

            if verbose:
                diff = 'mine faster' if mmc<qmc else 'qiskit faster'
                print(f"{mmc:.2e}\t{qmc:.2e}\t{diff}\t{pauli_conversion_times[-1]:.2e}\t{num_terms_dict}")
        print(f"\nnum_sites = {num_sites} done\n\n")

    if give_times:
        return True, [pauli_conversion_times, matrix_conversion_times, qiskit_matrix_conversion_times]
    return True
            
    
