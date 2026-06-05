run `testing_pauli_interface.py` to check the `term.to_pauli_list()` and `MultiSiteOperator.to_pauli_list()` methods.

This is done by
1. First instantiate a term (or MultiSiteOperator) object.
2. Use the `.to_pauli_list()` method to get the corresponding Pauli list.
3. Use the `.to_matrix()` method to get the Kronecker producted matrix.
4. Create a `SparsePauliOp` object from the Pauli list in 2. via `SparsePauliOp.from_list()`.
5. Compare the arrays obtained in 3. against the array `SparsePauliOp.to_matrix()` from 4..

**Testing `term` objects**

To see this for yourself, for a `term` object:

```Python
$ python3 -i testing_pauli_interface.py 
>>> trm = create_random_term(num_sites = 5,                                      # See the docstring
...                          num_supported_sites = 5)
>>>
>>> pauli_list = trm.to_pauli_list()
>>> spo = SparsePauliOp.from_list(pauli_list)
>>>
>>> my_mat = trm.to_matrix()
>>> spo_mat = spo.to_matrix()
>>> 
>>> np.allclose(my_mat, spo_mat)
True
```

To run this repeatedly for `num_sites` ranging from `1` to `10`,

```Python
$ python3 -i testing_pauli_interface.py 
>>> _bool, timing = check_term_to_pauli_list(runs = 5, max_num_sites = 10,       # See the docstring
...                          verbose = True, give_times = True)
```

If `_bool` is `True` then the all the `term` objects passes the test.  
If `_bool` is `False` then the `timing` will be equal to `term` object that failed the test.

**Testing `MultiSiteOperator` objects**

Similarly, we can test this for a `MultiSiteOperator` object:

```Python
$ python3 -i testing_pauli_interface.py 
>>> terms_dict = _random_num_terms_dict(num_sites = 5, max_num_terms = 6)       # See the docstring
>>> mso = create_random_MultiSiteOperator(num_sites = 5, num_terms_dict = terms_dict)
>>>
>>> pauli_list = mso.to_pauli_list()
>>> spo = SparsePauliOp.from_list(pauli_list)
>>> 
>>> my_mat = mso.to_matrix()
>>> spo_mat = spo.to_matrix()
>>> 
>>> np.allclose(my_mat, spo_mat)
True
```

To run this repeatedly,

```Python
$ python3 -i testing_pauli_interface.py 
>>> _bool, timing = check_MultiSiteOperator_to_pauli_list(runs = 5, max_num_sites = 5, 
...                          verbose = True, give_times = True)
```

If `_bool` is `True` then the all the `MultiSiteOperator` objects passes the test.  
If `_bool` is `False` then the `timing` will be equal to `MultiSiteOperator` object that failed the test.
