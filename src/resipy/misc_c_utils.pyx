#cython: boundscheck=False, wraparound=False

import numpy
cimport numpy
cimport cython

def dot_sorted(long long[:] ind1, double[:] val1, long long[:] ind2, double[:] val2):
	"""Calculate dot product of 2 vectors in sparse format.

	Parameters
	----------
	ind1 : (numpy.ndarray, int64)
		Sorted indices of first vector
	val1 : (numpy.ndarray, float64)
		Values of first vector
	ind2 : (numpy.ndarray, int64)
		Sorted indices of second vector
	val2 : (numpy.ndarray, float64)
		Values of second vector

	Returns
	-------
	(double) :
		dot product of vectors
	"""

	cdef unsigned int pos1 = 0, pos2 = 0
	cdef unsigned int len1 = ind1.shape[0]
	cdef unsigned int len2 = ind2.shape[0]
	cdef unsigned int idx1, idx2
	cdef double dot_prod = 0


	while pos1 < len1 and pos2 < len2:
		idx1 = ind1[pos1]
		idx2 = ind2[pos2]
		if idx1 < idx2:
			pos1 += 1
		elif idx1 > idx2:
			pos2 += 1
		else:
			dot_prod += val1[pos1] * val2[pos2]
			pos1 += 1
			pos2 += 1

	return dot_prod


def ind_from_count(unsigned int[:] counts):
	"""Returns a vector of indices 0, 1, 2, ... in order, each repeated a number
    of times as specified in the counts argument.

    Parameters
    ----------
    counts : (numpy.ndarray, uint32)
        number of times to repeat each index
    
    Returns
    -------
    (numpy.ndarray, uint32) : 
        vector of indices, with a size of numpy.sum(counts)

    Example
    -------
    >>> fci_c_utils.ind_from_count(numpy.array([3, 1, 4, 1]))
    array([0, 0, 0, 1, 2, 2, 2, 2, 3])
    """

	cdef unsigned long tot_num = numpy.sum(counts)
	cdef unsigned long num_counts = counts.shape[0]
	cdef numpy.ndarray[numpy.uint32_t] indices = numpy.zeros(tot_num, 
                                                              dtype=numpy.uint32)
	cdef unsigned long i, j, pos = 0
    
	for i in range(num_counts):
		for j in range(counts[i]):
			indices[pos] = i
			pos += 1
	return indices


