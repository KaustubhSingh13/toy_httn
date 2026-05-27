import numpy as np
from pauli import pauli

def _op_to_matrix(op):
    # for ease of use, allowing op = (coefficient, pauli_character) to be passed
    if not isinstance(op, np.ndarray):
        if op[1].upper() in pauli.keys():       # op[1] = pauli character, op[0] = coef
            matrix = pauli[op[1].upper()]*op[0]
        else:
            raise ValueError(f"{op[1]} is not a valid pauli character.")
    else:
        matrix = np.array(op)
    return matrix

class Term:
    """
    Represents a single operator term, e.g. {0: Z_matrix, 2: X_matrix} → ZIXI...
    Makes the dict hashable so it can live in a set.
    """
    def __init__(self, site_ops: dict):
        # site_ops: {site_index (int): operator (np.ndarray)}
        self.site_ops = site_ops                    # keep the dict for easy access
        self._hash = hash(
            frozenset(
                (site, arr.tobytes()) for site, arr in site_ops.items()
            )
        )

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        if self.site_ops.keys() != other.site_ops.keys():
            return False
        return all(
            np.array_equal(self.site_ops[k], other.site_ops[k])
            for k in self.site_ops
        )

    def __repr__(self):
        return f"{self.site_ops}"

class MultiSiteOperator:
    '''
    A lightweight implementation of many body operators.

    Would be compatible with TTNs and hTTNs.
    '''

    def __init__(self, num_sites, num_term_types):
        '''
        num_terms_list = [
                            num_single_site_terms,
                            num_two_site_terms,
                            num_three_site terms,
                            ...
                         ]

        For transverse field Ising model,
        on N sites

        num_term_types = 2  (single site and two site terms)
        num_sites = N
        '''
        self.num_term_types = num_term_types
        self.num_sites = num_sites
        
        self.terms = [set() for _ in range(num_term_types)]    # would it be faster if this was a dict?
        '''
        self.terms[0] would be a set of all the single site terms
        self.terms[1] would be a set of all two site terms and so on

        A term will be a dict 
        with int indices
        and values of ndarrays of ndim 2

        term = {0:'Z', 2:'X'}
        corresponds to ZIXII...I
        I have given the pauli str instead of ndarray for brevity
        '''

    def add_uniform_single_site_terms(self, op):
        '''
        Adds a single particle operator repeatedly across all the sites
        '''
        matrix = _op_to_matrix(op)

        for site_idx in range(self.num_sites):
            term = Term({site_idx: matrix})
            self.terms[0].add(term)                         # self.terms[i] is a set

    def add_uniform_local_two_site_terms(self, op, pbc = False):
        '''
        N = self.num_sites
        adds 
        O^(0)O^(1) + O^(1)O^(2) + ... + O^(N-2)O^(N-1)                  if pbc = False
        O^(0)O^(1) + O^(1)O^(2) + ... + O^(N-2)O^(N-1) + O^(N-1)O^(0)   if pbc = True
        '''
        matrix = _op_to_matrix(op) 
        assert num_sites > 1, f'Need at least two sites to add two sites terms, you have {self.num_sites}'
        for site_idx in range(self.num_sites - 1):
            term = Term({site_idx: matrix, site_idx+1: matrix})
            self.terms[1].add(term)

        if pbc:
            term = Term({self.num_sites-1: matrix, 0: matrix})
            self.terms[1].add(term)

    # we can have add_uniform_local_k_site_terms
    # add_global_k_site_terms (all to all k site interactions)
    # LMG has all to all two site interactions

    # for the __str__ function, return the analytical version 
    # like X^(0) + X^(1) + Z^(0)Z^(1) etc

    # could also include a function that add terms or instantiates using pauli_lists or SparsePauliOp
    # include a function that returns the pauli list 
