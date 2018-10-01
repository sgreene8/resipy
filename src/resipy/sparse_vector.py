# -*- coding: utf-8 -*-
"""
Definition of the SparseVector class.
"""

import numpy
import misc_c_utils


def _uniquify(idx, val, ins_idx):
    # Remove repeated indices in sparse representation of vector
    # and update ins_idx accordingly
    # The returned indices (uni_idx) are sorted
    if ins_idx is None:
    	uni_idx, uni_inv = numpy.unique(idx, return_inverse=True)
    else:
    	uni_idx, orig_idx, uni_inv = numpy.unique(idx, return_index=True, return_inverse=True)
    	ins_idx = ins_idx[orig_idx]

    uni_val = numpy.zeros_like(uni_idx, dtype=val.dtype)
    numpy.add.at(uni_val, uni_inv, val)
    return uni_idx, uni_val, ins_idx


class SparseVector(object):
    """
    Sparse representation of a vector with included methods for manipulating vectors.

    Parameters
    ----------
    indices : (numpy.ndarray, int or uint)
            Indices of nonzero elements in the vector. Stored sorted in ascending order.
    values : (numpy.ndarray, float)
            Values of nonzero elements in the vector.
    """

    def __init__(self, idx, val):
        """
        Initializer for the SparseVector class
        """
        self.indices = idx
        self.values = val

    def add(self, idx, val):
        """
        Add elements to an existing sparse vector.

        Parameters
        ----------
        idx : (numpy.ndarray, int or uint)
                Indices of elements to add. Need not be sorted.
        val : (numpy.ndarray, float)
                Values of elements to add
        """
        insert_idx = numpy.searchsorted(self.indices, idx)
        # elements that do not need to be appended to vector
        is_inside = self.indices.shape[0] != insert_idx
        inside_idx = idx[is_inside]
        insert_inside = insert_idx[is_inside]
        val_inside = val[is_inside]

        # Elements that match up with existing elements
        match_idx = numpy.equal(self.indices[insert_inside], inside_idx)
        numpy.add.at(
            self.values, insert_inside[match_idx], val_inside[match_idx])

        # Elements that do not match up and need to be inserted before the end of the array
        notmatch_idx = numpy.logical_not(match_idx)
        idx_to_insert = inside_idx[notmatch_idx]
        val_to_insert = val_inside[notmatch_idx]
        target_idx = insert_inside[notmatch_idx]
        idx_to_insert, val_to_insert, target_idx = _uniquify(idx_to_insert, val_to_insert, target_idx)
        self.indices = numpy.insert(self.indices, target_idx, idx_to_insert)
        self.values = numpy.insert(self.values, target_idx, val_to_insert)

        nonz_ind = self.values != 0
        self.indices = self.indices[nonz_ind]
        self.values = self.values[nonz_ind]

        # Elements that must be appended to end of array
        not_inside = numpy.logical_not(is_inside)
        app_idx = idx[not_inside]
        app_val = val[not_inside]
        app_idx, app_val, dummy = _uniquify(app_idx, app_val, None)
        self.indices = numpy.append(self.indices, app_idx)
        self.values = numpy.append(self.values, app_val)

    def dot(self, vec):
        """
        Calculate dot product with another SparseVector

        Parameters
        ----------
        vec : (SparseVector)
                vector for dot product

        Returns
        -------
        (float) :
                dot product
        """

        vals = self.values.astype(numpy.float64)
        return misc_c_utils.dot_sorted(self.indices, vals, vec.indices, vec.values)

    def one_norm(self):
        """
        Returns
        -------
        (float) :
                one-norm of vector
        """
        return numpy.sum(numpy.abs(self.values))

    def save(self, path):
        """
        Saves vector to file. Output file for the indices is at path + '_idx.npy', and for
        the values is at path + '_val.npy'
        Parameters
        ----------
        path : (str)
                Path and file name prefix at which to save file

        Returns
        -------
        """
        numpy.save(path + '_idx.npy', self.indices)
        numpy.save(path + '_val.npy', self.values)
