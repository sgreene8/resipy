"""

Utilities for performing sparse vector operations

"""

def merge_sparse_vectors(ind1, ind2, val1, val2, sorted=False):
    """Given two sparse vectors, the indices of the first and second
        represented by ind1 and ind2, respectively, and the elements of the
        first and second represented by val1 and val2, respectively, return a
        representation of the sparse vector obtained by adding the two, i.e.
        with no repeated indices.
        
        For optimal performance, the number of nonzero elements in the first
        sparse vector should be significantly greater than that in the second.
        
        Parameters
        ----------
        ind1 : (numpy.ndarray, int)
            indices of first sparse vector
        ind2 : (numpy.ndarray, int)
            indices of second sparse vector
        val1 : (numpy.ndarray, float)
            elements of first sparse vector
        val1 : (numpy.ndarray, float)
            elements of second sparse vector
        sorted : (boolean)
            if True, will assume that ind1 is sorted in ascending order

        Returns
        -------
        (numpy.ndarray, unsigned int)
            indices of nonzero elements in resulting vector
        (numpy.ndarray, float)
            values of nonzero elements in resulting vector
    """
    
    if not(sorted):
        srt_idx = ind1.argsort()
        ind1 = ind1[srt_idx]
        val1 = val1[srt_idx]
    
    ins_idx = numpy.searchsorted(ind1, ind2)
    valid_idx = ins_idx != ind1.shape[0]
    match_idx = numpy.equal(ind2[valid_idx], ind1[ins_idx[valid_idx]])
    numpy.add.at(val1, ins_idx[valid_idx][match_idx], val2[valid_idx][match_idx])
    
    new_ind_1 = ind2[numpy.logical_not(valid_idx)]
    new_val_1 = val2[numpy.logical_not(valid_idx)]
    uni_ind_1, uni_inv = numpy.unique(new_ind_1, return_inverse=True)
    uni_val_1 = numpy.zeros_like(uni_ind_1, dtype=val1.dtype)
    numpy.add.at(uni_val_1, uni_inv, new_val_1)
    
    new_ind_2 = ind2[valid_idx][numpy.logical_not(match_idx)]
    new_val_2 = val2[valid_idx][numpy.logical_not(match_idx)]
    uni_ind_2, uni_inv = numpy.unique(new_ind_2, return_inverse=True)
    uni_val_2 = numpy.zeros_like(uni_ind_2, dtype=val1.dtype)
    numpy.add.at(uni_val_2, uni_inv, new_val_2)
    
    new_ind = numpy.concatenate((ind1, uni_ind_1, uni_ind_2))
    new_val = numpy.concatenate((val1, uni_val_1, uni_val_2))
    
    nonz_ind = new_val != 0
    
    new_ind = new_ind[nonz_ind]
    new_val = new_val[nonz_ind]
    
    srt_idx = new_ind.argsort()
    return new_ind[srt_idx], new_val[srt_idx]
