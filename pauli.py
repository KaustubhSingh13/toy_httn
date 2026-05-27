import numpy as np
'''
Is it faster to call from a dict or a list?
refactor all the code to make it compatible with a list
if a list implementation is faster
'''
pauli = {
            'I' :   np.eye(2),
            'X' :   np.array([[0,1],[1,0]], dtype = float),
            'Y' :   np.array([[0, -1j],[1j,0]], dtype = complex),
            'Z' :   np.array([[1,0],[0,-1]], dtype = float),
        }

"""
pauli = [
                    np.eye(2),
                    np.array([[0,1],[1,0]], dtype = float),
                    np.array([[0, -1j],[1j,0]], dtype = complex),
                    np.array([[1,0],[0,-1]], dtype = float),
        ]
# In general, avoid using pauli Y (complex dtype). See what are the associated drawbacks.
"""
