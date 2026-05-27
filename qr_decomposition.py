import numpy as np
#Here I am going to swap the isometric index with the last index before reshaping. 
def qr_tensors(t,index):
    '''
    Parameters:

    T : The tensor you want to QR decompose
    i : The position of the index you want the tensor to be isometrised with respect to

    Returns:

    (Q,R) such that np.tensordot(Q,R, axes = [[i,0]] ) = T
    and Q satisfying the isometry condition with respect to the i^{th} index.

    where Q is a ndarray with Q.shape = T.shape
    R is a ndarray with R.shape = (n_i, n_i) where n_i is the number of values the i^{th} index can take 
    '''
    
    t_shape = t.shape; isometric_index_dim = t_shape[index]
    
    contracted_index_dim = int(np.prod(t_shape)/isometric_index_dim)
    ''' 
    if contracted_index_dim < isometric_index_dim:              # <=> If the dimensionality of the vector space is less than the number of demanded orthonormal vectors.
        raise ValueError(f"It is not possible to isometrise the given tensor with respect to the {index}th index")
    '''
    
    t1 = np.swapaxes(t,len(t_shape)-1,index) # Swapping the last index with the iso index.

    tnew = t1.reshape(contracted_index_dim,isometric_index_dim)  # This ensures that tnew is not fat and the resultant q is isometrised wrt the 1st index.
    q,r = np.linalg.qr(tnew)
    
    q = np.swapaxes(q.reshape(t1.shape),len(t_shape)-1,index)    # This reshapes q to t.shape and ensures that the q is isometrised with respect to the desried index by swapping the index positions. 
    
    return q,r

#The usecases of this function are when you want to contract q and r to check if a given tensor has been decomposed properly and when you want to abosorb (contract) r with the next tensor when you are shifting the isometrisation centre or when you are isometrising the tensor network.
def combine(t,r,t_index,r_index):
    '''
    Contracts the t_index of t with the r_index of r 
    to give the resultant tensor with the appropriate index placement.

    T_{ijk} R_{ja} = X_{iak}
    T_{ikj} R_{ja} = X_{ika}
    '''
    assert t.shape[t_index] == r.shape[r_index], f'The dim of index {t_index} of T is not the same as the dim of the {r_index} index of r.'

    out = np.tensordot(t,r,axes=[t_index,r_index])
    return np.moveaxis(out,len(t.shape)-1,t_index)  # len(t.shape)-1 because the other (non-contracted index) of r would be placed at the end by tensordot convention. We want to preserve the shape of t after contracting it with r.

