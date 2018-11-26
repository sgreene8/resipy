# -*- coding: utf-8 -*-
"""
Definition of the SparseVector class.
"""

import numpy
import misc_c_utils


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
        if idx.shape[0] == 0:
            return
        srt_idx = idx.argsort()
        idx = idx[srt_idx]
        val = val[srt_idx]

        if self.values.dtype is numpy.dtype('int32'):
            new_idx, new_val = misc_c_utils.merge_sorted_int(self.indices, self.values, idx, val)
        elif self.values.dtype is numpy.dtype('float64'):
            new_idx, new_val = misc_c_utils.merge_sorted_doub(self.indices, self.values, idx, val)
        else:
            raise TypeError("Dtype %s not yet supported in SparseVector class", self.values.dtype)
        self.indices = new_idx
        self.values = new_val


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
