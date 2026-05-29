import numpy as np
from pauli import pauli

def op_to_matrix(op):
    # for ease of use, allowing op = (coefficient, pauli_character) to be passed
    # will be used in add* methods of MultiSiteOperator
    if not isinstance(op, np.ndarray):
        if op[1].upper() in pauli.keys():       # op[1] = pauli character, op[0] = coef
            matrix = pauli[op[1].upper()]*op[0]
        else:
            raise ValueError(f"{op[1]} is not a valid pauli character.")
    else:
        matrix = np.array(op)
    return matrix               # tested.

def array_to_pauli_list(arr):
    '''
    Takes in an ndarray of shape (2,2)
    returns the corresponding Pauli list

    np.ndarray([[1,1],[1,1]]) -> [ ['X',1], ['I',1] ]
    '''
    assert arr.shape == (2,2), f'arr.shape is {arr.shape}, has to be (2,2).'
    a, b, c, d = arr[0,0], arr[0,1], arr[1,0], arr[1,1]
    I_coeff = .5 *(a+d)
    Z_coeff = .5 *(a-d)

    # if arr is Hermitian, we can simplify X_coeff and Y_coeff.
    # this speedup would be at the cost of accuracy/ stability.
    # test what happens when you do that.
    X_coeff = .5 *(b+c)               # b.real
    Y_coeff = .5j*(b-c)               #-b.imag
    
    # should I make all these real here itself?
    out = [  
        [ 'I', I_coeff ],
        [ 'X', X_coeff ],
        [ 'Y', Y_coeff ],
        [ 'Z', Z_coeff ]
    ]
    
    filtered_out = []
    # keeping coeffs which are close to zero
    # raising error if there is a large imaginary part.
    for i, p_chr in enumerate(out):
        if p_chr[1].imag > 1e-6:
            raise ValueError(f'The coefficient of {p_chr[0]} has an imaginary part of {p_chr[1].imag}')
        if np.abs(p_chr[1]) > 1e-6:
            filtered_out.append([p_chr[0], p_chr[1].real])

    return filtered_out             # tested.

def _push_at_idx_string(p_string_0, p_string_1, idx):
    len_pauli_string = len(p_string_0)
    
    if len_pauli_string >= idx:
        new_pauli_string = p_string_0[:idx] + p_string_1 + p_string_0[idx:]

    else:
        idtt_padding = 'I' * (idx - len_pauli_string)
        new_pauli_string = p_string_0 + idtt_padding + p_string_1
    return new_pauli_string

def _pauli_list_to_pauli_dict(p_list):
    '''
    Converts
    pauli_list = [('XII', -1), ('IXI', -1), ('IIX', -1), ('ZZI', -1), ('IZZ', -1), ('ZIZ', -1)]

    -->


    pauli_dict = ['XII': -1, 'IXI': -1, 'IIX': -1, 'ZZI': -1, 'IZZ': -1, 'ZIZ': -1]
    '''
    return {elem[0]:elem[1] for elem in p_list}

def _pauli_dict_to_pauli_list(p_dict):
    '''
    inverse of _pauli_list_to_pauli_dict
    '''
    return [[pauli_string, coeff] for pauli_string, coeff in p_dict.items()]

def _add_pauli_lists(p_list_0, p_list_1):
    # If you create a pauli list/ dict class, this can be the __add__ method.
    '''
    p_list_0 = [['XI',1] ['IX',3]]
    p_list_1 = [['XI',2] ['IZ',3]]
    
    -->

    [['XI',3],['IX',3], ['IZ',3]]
    '''
    p_dict_0 = _pauli_list_to_pauli_dict(p_list_0)
    p_dict_1 = _pauli_list_to_pauli_dict(p_list_1)
    
    basis_set_0 = set(p_dict_0.keys())
    basis_set_1 = set(p_dict_1.keys())
    
    shared_pauli_strings    = basis_set_0 & basis_set_1     # set intersection
    exclusive_pauli_strings = basis_set_0 ^ basis_set_1     # set symmetric difference
    # look into how ^ is implemented. If internally it calculates the set intersection 
    # and then substracts it from the union then diy because you already calculated ^.
    summed_dict = {} 
    for ps in shared_pauli_strings:
        coeff_sum = p_dict_0[ps] + p_dict_1[ps]
        if abs(coeff_sum)>1e-6:
            summed_dict = {ps: p_dict_0[ps] + p_dict_1[ps]}

    # would it be faster to take the difference of the intersection with
    # basis_set_0 and basis_set_1 getting two different sets and then running two differnt for loops?
    # we would not have to put in the conditiional in that case.
    for ps in exclusive_pauli_strings:
        if ps in p_dict_0:
            summed_dict[ps] = p_dict_0[ps]
        else:
            summed_dict[ps] = p_dict_1[ps]
    return _pauli_dict_to_pauli_list(summed_dict)

def _push_at_idx_list(pauli_list, p_string:str, idx):
    '''
    pauli_list = [('XII', -1), ('IXI', -1), ('IIX', -1), ('ZZI', -1), ('IZZ', -1), ('ZIZ', -1)]
    idx = 1
    p_string = Y

    ->
    
    [['XYII', -1], ['IYXI', -1], ['IYIX', -1], ['ZYZI', -1], ['IYZZ', -1], ['ZYIZ', -1]]

    if idx exceeds the length of the Pauli strings
    then it will add identities and then place sigma at the end.
    '''
    pauli_list_with_sigma = pauli_list.copy()
    len_pauli_string = len(pauli_list[0][0])        # I am assuming all strings are of the same length and the list is non empty

    for i,term in enumerate(pauli_list_with_sigma):
        term = list(term)
        term[0] = _push_at_idx_string(term[0], p_string, idx)
        pauli_list_with_sigma[i] = term

    return pauli_list_with_sigma                

def tensor_pauli_lists(p_list0, p_list1, idx):
    '''
    Gives the tensor product of two pauli lists where
    p_list0 is a pauli list representing a multi site operator
    p_list1 is a pauli list representing a single site operator
    at site index idx
    '''
    out = []

    # amounts to expanding the paraenthesis
    for elem_0 in p_list0:                           # this could be parallelised.
        pstring_0, coeff_0 = elem_0

        for elem_1 in p_list1:                   
            pstring_1, coeff_1 = elem_1

            pstring = _push_at_idx_string(pstring_0, pstring_1, idx)
            coeff = coeff_0 * coeff_1

            out.append([pstring, coeff])

    return out

class term:
    """
    Represents a single operator term, e.g. {0: Z_matrix, 2: X_matrix} → ZIXI...
    Makes the dict hashable so it can live in a set.

    would it be better to just inherit the dict class?
    probably not since we dont want to mess with immutability.
    """
    def __init__(self, site_ops: dict):
        # site_ops: {site_index (int): operator (np.ndarray)}
        self.site_ops = site_ops                    # keep the dict for easy access
        self.num_sites = max(site_ops.keys()) + 1
        self._hash = hash(
            frozenset(
                (site, arr.tobytes()) for site, arr in site_ops.items()
            )
        )

        # have a better error message when 
        # the values in site_ops are not np.ndarrays
    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, term):
            return False
        if self.site_ops.keys() != other.site_ops.keys():
            return False
        return all(
            np.array_equal(self.site_ops[k], other.site_ops[k])
            for k in self.site_ops
        )

    def __iter__(self):
        return iter(self.site_ops)

    def items(self):
        return self.site_ops.items()

    def values(self):
        return self.site_ops.values()

    def __getitem__(self, site):
        # it would be nice if we could change the KeyError
        # to also reflect what the keys (site_idxs) actually are.
        return self.site_ops[site]

    def __len__(self):
        return len(self.site_ops)
    
    def __contains__(self, site):
        return site in self.site_ops

    def keys(self):
        return self.site_ops.keys()

    def to_pauli_list(self, num_sites = None):
        # will this be recomputed on each call?
        # if so, is there a way of storing this untill there is a change?
        '''
        Takes in a term
        returns the corresponding Pauli list

        {0:[[1,1],[1,1]], 1:[[1,0],[0,-1]]} -> [('XZ',1), ('IZ',1)]
        '''

        indiv_pauli_decomp = {}
        
        # decomposing the term into pauli matrices along with site index
        for site_idx, arr in self.site_ops.items():
            indiv_pauli_decomp[site_idx] = array_to_pauli_list(arr)
        '''
        (2X^(0) + 3Z^(0)) \otimes (5Y^(2) - I^(2))
        <-->
        indiv_pauli_decomp = {
            0: [ ['X', 2], ['Z', 3] ],
            2: [ ['Y', 5], ['I',-1] ]
        }
        <-->
        [
            ['XIY', 10],
            ['XII', -2],
            ['ZIY', 15],
            ['ZII', -3]
        ]
        '''
        indiv_pauli_decomp = {site_idx: indiv_pauli_decomp[site_idx] for site_idx in sorted(indiv_pauli_decomp.keys())}
        
        #return(indiv_pauli_decomp)
        
        # Initial I padding
        smallest_site_idx = list(indiv_pauli_decomp.keys())[0] 
        initial_idtt_padding = 'I' * smallest_site_idx

        pauli_list_out = [
                [initial_idtt_padding + pauli_term[0], pauli_term[1] ] for pauli_term in indiv_pauli_decomp[smallest_site_idx]
        ]

        for site_idx, pauli_list in list(indiv_pauli_decomp.items())[1:]:   # skipping the 0th element
            pauli_list_out = tensor_pauli_lists(pauli_list_out, pauli_list, site_idx)

        # tail I padding
        if num_sites is not None:
            final_idtt_padding = 'I'*(num_sites - self.num_sites)
            return [[pt[0]+final_idtt_padding, pt[1]] for pt in pauli_list_out]           # test more.
        return pauli_list_out
    def __repr__(self):
        # maybe change this to give the pauli list form
        return f"{self.site_ops}"
    
    # create a way of instantiating a term through a pauli string.
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
        matrix = op_to_matrix(op)

        for site_idx in range(self.num_sites):
            term = term({site_idx: matrix})
            self.terms[0].add(term)                         # self.terms[i] is a set

    def add_uniform_local_two_site_terms(self, op, pbc = False):
        '''
        N = self.num_sites
        adds 
        O^(0)O^(1) + O^(1)O^(2) + ... + O^(N-2)O^(N-1)                  if pbc = False
        O^(0)O^(1) + O^(1)O^(2) + ... + O^(N-2)O^(N-1) + O^(N-1)O^(0)   if pbc = True
        '''
        matrix = op_to_matrix(op) 
        assert num_sites > 1, f'Need at least two sites to add two sites terms, you have {self.num_sites}'
        for site_idx in range(self.num_sites - 1):
            term = term({site_idx: matrix, site_idx+1: matrix})
            self.terms[1].add(term)

        if pbc:
            term = term({self.num_sites-1: matrix, 0: matrix})
            self.terms[1].add(term)
    
    def unfold(self):
        '''
        returns a set with all the term objects
        '''
        out = []
        for trm_set in self.terms:
            for trm in trm_set:
                out.append(trm)
        return set(out)

    def to_pauli_list(self):        # Not working correctly. Run for mso.
        # will have to define how to take the sum of pauli lists whcih have pauli strings of the same length
        pauli_list_out = [] 

        # converting terms which have support on the same number of sites
        # into pauli lists and then adding them up
        for trm_set in self.terms:
            _sum = []
            for trm in trm_set:
                print(_sum,trm.to_pauli_list(self.num_sites))
                _sum = _add_pauli_lists(_sum, trm.to_pauli_list(self.num_sites))
            print('\n')
            pauli_list_out += _sum

        return pauli_list_out
        
    # we can have add_uniform_local_k_site_terms
    # add_global_k_site_terms (all to all k site interactions)
    # LMG has all to all two site interactions

    # for the __str__ function, return the analytical version 
    # like X^(0) + X^(1) + Z^(0)Z^(1) etc

    # could also include a function that add terms or instantiates using pauli_lists or SparsePauliOp
    # include a function that returns the pauli list 
    # include a function that gives the full dense matrix.
    # that can be used to check the to_pauli_list function against the SparsePauliOp.to_matrix

if __name__ == '__main__':
    '''
    These is for debugging.
    Will be deleted later.
    '''
    x = pauli['X']
    y = pauli['Y']
    z = pauli['Z']
    so = {4:np.ones([2,2]), 1:z, 2:(x+y+z+np.ones([2,2])+ np.eye(2)*7)}

    pauli_list = [('XII', -1), ('IXI', -1), ('IIX', -1), ('ZZI', -1), ('IZZ', -1), ('ZIZ', -1)]

    mso_terms = [
            { term({0:x}), term({1:x}), term({2:x})
                },
            { term({0: z, 1:z}), term({1:z, 2:z})
                }
    ]
    # There needs to be a better interface for instantiating MultiSiteOperator.

    mso = MultiSiteOperator(3,1)
    mso.terms = mso_terms
