import numpy as np
from pauli import pauli

# *************************************************************************

# I have hardcoded 2 level systems in this implementation (eg. spin chains)
# In a later version, generalise this to more than 2 levels.

# *************************************************************************

def op_to_matrix(op):
    # for ease of use, allowing op = (pauli_character, coefficient) to be passed
    # will be used in add* methods of MultiSiteOperator
    if not isinstance(op, np.ndarray):
        if op[0].upper() in pauli.keys():       # op[0] = pauli character, op[1] = coef
            matrix = pauli[op[0].upper()]*op[1]
        else:
            raise ValueError(f"{op[0]} is not a valid pauli character.")
    else:
        matrix = np.array(op)
    return matrix               # tested.

def array_to_pauli_list_general(arr):
    '''
    Takes in an ndarray of shape (2,2)
    returns the corresponding Pauli list

    np.ndarray([[1,1],[1,1]]) -> [ ['X',1], ['I',1] ]
    
    Works for all matrices in GL(2)
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
    # keeping coeffs which are not close to zero
    for i, p_chr in enumerate(out):
        if np.abs(p_chr[1]) > 1e-6:
            filtered_out.append([p_chr[0], p_chr[1]])

    return filtered_out             # tested.

def array_to_pauli_list_hermitian(arr):
    '''
    Takes in an ndarray of shape (2,2)
    returns the corresponding Pauli list

    np.ndarray([[1,1],[1,1]]) -> [ ['X',1], ['I',1] ]
    
    For Hermitian matrices. 
    Little bit faster than the general method.
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
    # keeping coeffs which are not close to zero
    # raising error if there is a large imaginary part.
    for i, p_chr in enumerate(out):
        if abs(p_chr[1].imag) > 1e-6:
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

def _add_pauli_lists(p_list_0, p_list_1):                                           # IMPORTANT SIMPLE SPEED UP: DONT ADD TERMS. JUST APPEND.
    p_dict_0 = _pauli_list_to_pauli_dict(p_list_0)                                  # Do we even need to group terms together like this?
    p_dict_1 = _pauli_list_to_pauli_dict(p_list_1)                                  # sure it looks prettier, but SparsePauliOp does not care if we pass [['X',2]] or [['X',1],['X',1]]. 
                                                                                    # both are treated the same by SparsePauliOp. 
    basis_set_0 = set(p_dict_0.keys())                                              # This grouping together business could be taking time 
    basis_set_1 = set(p_dict_1.keys())                                              # as it is MultiSiteOperator.to_pauli_list() is quite time consuming (order 1 sec for large num sites).
                                                                                    # TEST IF USING APPEND INSTEAD OF ADDING HAS NOTABLE IMPACE ON SPEED.
    shared_pauli_strings = basis_set_0 & basis_set_1
    exclusive_0          = basis_set_0 - basis_set_1
    exclusive_1          = basis_set_1 - basis_set_0

    summed_dict = {}

    for ps in shared_pauli_strings:
        coeff_sum = p_dict_0[ps] + p_dict_1[ps]
        if abs(coeff_sum) > 1e-6:
            summed_dict[ps] = coeff_sum         

    for ps in exclusive_0:
        summed_dict[ps] = p_dict_0[ps]

    for ps in exclusive_1:
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
    Represents a single operator term, e.g. {0: Z_matrix, 2: X_matrix} -> ZIX
    Makes the dict hashable so it can live in a set.

    would it be better to just inherit the dict class?
    probably not since we dont want to mess with immutability.
    """
    def __init__(self, site_ops: dict, num_sites = None):
        # site_ops: {site_index (int): operator (np.ndarray)}
        self.site_ops = site_ops                    # keep the dict for easy access
        supported_site_idxs = site_ops.keys()
        self.supported_site_idxs = supported_site_idxs
        if num_sites is None:
            self.num_sites = max(supported_site_idxs) + 1
        else:
            self.num_sites = num_sites
        
        self.num_supported_sites = len(supported_site_idxs)
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

    def to_pauli_list(self):
        # will this be recomputed on each call?
        # if so, is there a way of storing this untill there is a change?
        '''
        Takes in a term
        returns the corresponding Pauli list

        {0:[[1,1],[1,1]], 1:[[1,0],[0,-1]]} -> [('XZ',1), ('IZ',1)]
        '''
        num_sites = self.num_sites

        indiv_pauli_decomp = {}
        
        # decomposing the term into pauli matrices along with site index
        for site_idx, arr in self.site_ops.items():                             # PARALLELIZE: MEDIUM-HIGH
            indiv_pauli_decomp[site_idx] = array_to_pauli_list_hermitian(arr)   # IF COMPLEX DTYPE TROUBLE, CHANGE THIS TO array_to_pauli_list_general()
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
        largest_site_idx =  list(indiv_pauli_decomp.keys())[-1] 
        initial_idtt_padding = 'I' * smallest_site_idx

        pauli_list_out = [
                [initial_idtt_padding + pauli_term[0], pauli_term[1] ] for pauli_term in indiv_pauli_decomp[smallest_site_idx]
        ]

        for site_idx, pauli_list in list(indiv_pauli_decomp.items())[1:]:   # skipping the 0th element
            pauli_list_out = tensor_pauli_lists(pauli_list_out, pauli_list, site_idx)   #PARALLELIZE: MEDIUM-HIGH

        # tail I padding
        final_idtt_padding = 'I'*(num_sites - largest_site_idx - 1)
        return [[pt[0]+final_idtt_padding, pt[1]] for pt in pauli_list_out]         

    def to_matrix(self):
        '''
        returns the Kronecker producted matrix
        '''
        if 0 in self.supported_site_idxs:
            out = self.site_ops[0]
        else:
            out = np.eye(2)

        for i in range(1,self.num_sites):
            if i in self.supported_site_idxs:
                out = np.kron(out, self.site_ops[i])
            else:
                out = np.kron(out,np.eye(2))
            
        return out

    def sandwich(self, P_matrices: list):               # maybe this could be absorbed into a different method later. 
        '''
        P_matrices is a list containing ndarrays of ndim = bond dim of quantum tensors
        of length self.num_sites
        This contracts P^\dagger term P across all sites

        I am assuming that the P matrices are all unitary. See Section D. (ii) of the PRX paper.
        This would preserve the number of sites with support.
        
        Returns a new term object
        '''
        site_ops_out = {}
        for site_idx, mat in self.site_ops.items():             # PARALLELIZE: MEDIUM
            p = P_matrices[site_idx]
            site_ops_out[site_idx] = p.conj().T @ mat @ p       # @ calls np.matmul which uses BLAS. Fast enough. PARALLELIZE: LOW (increases with the number of qubits)

        term_out = term(site_ops = site_ops_out, num_sites = self.num_sites)
        return term_out

    def __repr__(self):
        # maybe change this to give the pauli list form
        return f"{self.site_ops}"
    
    # create a way of instantiating a term through a pauli string.
class MultiSiteOperator:
    '''
    A lightweight implementation of many body operators.

    Would be compatible with TTNs and hTTNs.
    '''

    def __init__(self, num_sites):
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
        self.num_sites = num_sites
        
        #self.terms = [set() for _ in range(num_term_types)]    # would it be faster if this was a dict?
        self.terms = {}
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
    def add_term(self, trm):
        '''
        adds the term trm to the MultiSiteOperator
        ensures that the num_sites attribuite is inherited
        '''
        
        # One issue is that if we add the same term twice, 
        # it won't count the second one. FIX. 
        # throw error if trm has support on more sites than self. FIX.

        idx = trm.num_supported_sites 
        trm.num_sites = self.num_sites
        
        if idx not in self.terms.keys():
            self.terms[idx] = set()

        self.terms[idx].add(trm)

    def add_all_terms(self, trms_dict):
        '''
        updates self.terms by passing the terms dict 

        mso_terms = {
                1: { term({0:x}), term({1:x}), term({2:x})
                },

                2: { term({0: z, 1:z}), term({1:z, 2:z})
                }
        }
        '''
        for num_supported_sites, trm_set in trms_dict.items():              # PARALLELIZE: MEDIUM-HIGH
            for trm in trm_set:
                self.add_term(trm)


    def add_uniform_single_site_terms(self, op):
        '''
        Adds a single particle operator repeatedly across all the sites
        '''
        matrix = op_to_matrix(op)

        if 1 not in self.terms.keys():
            self.terms[1] = set()

        for site_idx in range(self.num_sites):                              # PARALLELIZE: VERY LOW
            trm = term({site_idx: matrix},num_sites = self.num_sites)
            self.terms[1].add(trm)                         # self.terms[i] is a set

    def add_uniform_local_two_site_terms(self, op, pbc = False):
        '''
        N = self.num_sites
        adds 
        O^(0)O^(1) + O^(1)O^(2) + ... + O^(N-2)O^(N-1)                  if pbc = False
        O^(0)O^(1) + O^(1)O^(2) + ... + O^(N-2)O^(N-1) + O^(N-1)O^(0)   if pbc = True
        '''
        matrix = op_to_matrix(op) 
        assert self.num_sites > 1, f'Need at least two sites to add two sites terms, you have {self.num_sites}'

        if not 2 in self.terms.keys():
            self.terms[2] = set()

        for site_idx in range(self.num_sites - 1):                          # PARALLELIZE: VERY LOW
            trm = term({site_idx: matrix, site_idx+1: matrix}, num_sites = self.num_sites)
            self.terms[2].add(trm)

        if pbc:
            trm = term({self.num_sites-1: matrix, 0: matrix},num_sites = self.num_sites)
            self.terms[2].add(trm)
    
    def unfold(self):                                                       #FIX: redundant?
        '''
        returns a set with all the term objects
        '''
        out = []
        for num_supported_sites, trm_set in self.terms.items():             # PARALLELIZE?
            for trm in trm_set:         
                out.append(trm)
        return set(out)
    
    def flush(self):
        self.terms = {}

    def to_pauli_list(self):                                                                                # SEE IF THIS COULD BE SPED UP. TAKES TOO LONG. 
        
        pauli_list_out = [] 

        # converting terms which have support on the same number of sites
        # into pauli lists and then adding them up
        for num_supported_sites,trm_set in self.terms.items():              # PARALLELIZE: HIGH
            _sum = []
            for trm in trm_set:
                _sum = _add_pauli_lists(_sum, trm.to_pauli_list())

            pauli_list_out += _sum

        return pauli_list_out

    def to_matrix(self):
        '''
        converts the MultiSiteOperator object
        to a Kronecker producted matrix.
        '''
        N = self.num_sites
        _sum = np.zeros([2**N, 2**N], dtype = complex)        # Change this dtype to complex if casting issues occur.
        unfolded_terms = self.unfold()              # FIX: unfold as it is uses a double for loop. you are using a for looop again here. SPEEDUP.
        for trm in unfolded_terms:                                      # PARALLELIZE: MEDIUM
            _sum += trm.to_matrix()

        return _sum

    @property
    def shape(self) -> dict:
        '''
        returns a dict
        with the same keys as self.terms
        and self.shape[num_supported_sites] being the number of terms which
        have support on num_supported_sites

        In the testing\testing_pauli_interface.py, 
        a random shape generated by _random_num_terms_dict(num_sites, max_num_terms)

        shape <---> num_terms_dict  (shape is a better name.) 

        FIX change the name from num_terms_dict to shape in all testing files including README.md
        '''
        shape_out = {}
        for num_supported_sites, term_set in self.terms.items():        # PARALLELIZE: LOW
            shape_out[num_supported_sites] = len(term_set)

        return shape_out
    
    def sandwich(self, P_matrices:list):
        '''
        P_matrices is a list containing ndarrays of ndim = bond dim of quantum tensors

        for now I am assuming that the P matrices are all unitary
        this reduces the number of contractions we have to perform

        evaluates P^\dagger O P
        The P matrices are needed for implicitly isometrizing the hTTN. 
        See the PRX paper.

        returns a new MultiSiteOperator object
        '''
        assert len(P_matrices) == self.num_sites, 'The number of P matrices must be equal to the number of sites'
        terms_dict_out = {}                                # would it be better (faster or more memory efficient) to do this in place instead of return a new MultiSiteOperator object?
        for num_supported_sites, terms_set in self.terms.items():           # PARALLELIZE: HIGH
            _trm_set = set()
            for trm in terms_set:
                new_trm = trm.sandwich(P_matrices)
                _trm_set.add(new_trm) 
            terms_dict_out[num_supported_sites] = _trm_set

        mso_out = MultiSiteOperator(self.num_sites)
        mso_out.terms = terms_dict_out                      # would have been faster than using .add_all_terms() because mso_out.terms had nothing and all terms getting added would be consistent (having the same num_sites) since we copied from a pre-existing mso.
        return mso_out
    
    def __str__(self):
        return str(self.to_pauli_list())
    # we can have add_uniform_local_k_site_terms
    # add_global_k_site_terms (all to all k site interactionsr
    # LMG has all to all two site interactions

    # for the __repr__ function, return the analytical version 
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
    so1 = {0: np.array([[3,1],[2,1]]) 

           }

    pauli_list = [('XII', -1), ('IXI', -1), ('IIX', -1), ('ZZI', -1), ('IZZ', -1), ('ZIZ', -1)]

    mso_terms = {
            1: { term({0:x}), term({1:x}), term({2:x})
                },
            2: { term({0: z, 1:z}), term({1:z, 2:z})
                }
            }
    # There needs to be a better interface for instantiating MultiSiteOperator.

    mso = MultiSiteOperator(3)

    mso1 = MultiSiteOperator(4)
    mso2 = MultiSiteOperator(4)
    trm = term({0:y, 1:np.eye(2), 3:z})
    trm1 = term({0:y, 1:np.ones((2,2)), 2:z})
