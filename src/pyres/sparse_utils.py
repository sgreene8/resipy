"""

Utilities for performing sparse vector operations

"""
import numpy


def add_vectors(ind1, ind2, val1, val2, sorted1=False):
    """Add 2 sparse vectors.

        Parameters
        ----------
        ind1 : (numpy.ndarray, int)
            indices of first sparse vector
        ind2 : (numpy.ndarray, int)
            indices of second sparse vector, may have duplicates
        val1 : (numpy.ndarray, float)
            elements of first sparse vector
        val2 : (numpy.ndarray, float)
            elements of second sparse vector
        sorted1 : (boolean)
            Indicates whether ind1 is sorted in ascending order

        Returns
        -------
        (numpy.ndarray)
            indices of nonzero elements in resulting vector
        (numpy.ndarray, float)
            values of nonzero elements in resulting vector
    """

    if not(sorted1):
        srt_idx = ind1.argsort()
        ind1 = ind1[srt_idx]
        val1 = val1[srt_idx]

    ins_idx = numpy.searchsorted(ind1, ind2)
    valid_idx = ins_idx != ind1.shape[0]
    match_idx = numpy.equal(ind2[valid_idx], ind1[ins_idx[valid_idx]])
    numpy.add.at(val1, ins_idx[valid_idx][match_idx],
                 val2[valid_idx][match_idx])

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
    if sorted1:
        numpy.add.at(val1, ins_idx[valid_idx][match_idx],
                     -val2[valid_idx][match_idx])

    nonz_ind = new_val != 0

    new_ind = new_ind[nonz_ind]
    new_val = new_val[nonz_ind]

    srt_idx = new_ind.argsort()
    return new_ind[srt_idx], new_val[srt_idx]


def dot_vectors(ind1, ind2, val1, val2, sorted1=False, sorted2=False):
    """Calculate the dot product of 2 vectors in sparse format.

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
        sorted1 : (boolean)
            Indicates whether ind1 is sorted in ascending order
        sorted2 : (boolean)
            Indicates whether ind2 is sorted in ascending order

        Returns
        -------
        (float) :
            Dot product of vectors
    """

    if not(sorted1):
        if not(sorted2):
            if ind1.shape[0] < ind2.shape[0]:
                srt_idx = ind1.argsort()
                ind1 = ind1[srt_idx]
                val1 = val1[srt_idx]
                sorted1 = True
            else:
                srt_idx = ind2.argsort()
                ind2 = ind2[srt_idx]
                val2 = val2[srt_idx]
    if sorted1:
        ind_a = ind1
        val_a = val1
        ind_b = ind2
        val_b = val2
    else:
        ind_a = ind2
        val_a = val2
        ind_b = ind1
        val_b = val1

    num_a = ind_a.shape[0]
    idx_in_a = numpy.searchsorted(ind_a, ind_b)
    valid_idx = idx_in_a != num_a
    connected = ind_b[valid_idx] == ind_a[idx_in_a[valid_idx]]
    return numpy.sum(val_a[idx_in_a[valid_idx][connected]] * val_b[valid_idx][connected])
