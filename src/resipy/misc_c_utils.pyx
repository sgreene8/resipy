# cython: boundscheck=False, wraparound=False

import numpy
cimport numpy
cimport cython
cimport openmp
from cython.parallel import prange, threadid, parallel


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

    cdef size_t pos1 = 0, pos2 = 0
    cdef size_t len1 = ind1.shape[0]
    cdef size_t len2 = ind2.shape[0]
    cdef long long idx1, idx2
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


def merge_sorted_int(long long[:] ind1, int[:] val1, long long[:] ind2, int[:] val2):
    """Merge two vectors in sparse format whose indices are already sorted.

    Parameters
    ----------
    ind1 : (numpy.ndarray, int64)
        Sorted indices of first vector
    val1 : (numpy.ndarray, int32)
        Values of first vector
    ind2 : (numpy.ndarray, int64)
        Sorted indices of second vector
    val2 : (numpy.ndarray, int32)
        Values of second vector

    Returns
    -------
    (numpy.ndarray, int64)
        Sorted indices of merged vector
    (numpy.ndarray, int32)
        Values of merged vector
    """
    cdef size_t pos1 = 0, pos2 = 0
    cdef size_t len1 = ind1.shape[0]
    cdef size_t len2 = ind2.shape[0]
    cdef long long idx1 = ind1[0]
    cdef long long idx2 = ind2[0]
    cdef size_t new_pos = 0
    cdef size_t new_idx
    cdef numpy.ndarray[numpy.int64_t] new_ind = numpy.zeros(len1 + len2, dtype=numpy.int64)
    cdef numpy.ndarray[numpy.int32_t] new_val = numpy.zeros(len1 + len2, dtype=numpy.int32)

    cdef long long max_idx = ind1[len1 - 1]
    if max_idx < ind2[len2 - 1]:
        max_idx = ind2[len2 - 1]
    max_idx += 1

    # Make sure new_idx is not equal to idx1 or idx2
    new_idx = idx1 + 1
    if new_idx == idx2:
        new_idx = idx2 + 1

    while pos1 < len1 or pos2 < len2:
        if idx1 == new_idx:
            new_val[new_pos] += val1[pos1]
            pos1 += 1
            if pos1 < len1:
                idx1 = ind1[pos1]
            else:
                idx1 = max_idx
        elif idx2 == new_idx:
            new_val[new_pos] += val2[pos2]
            pos2 += 1
            if pos2 < len2:
                idx2 = ind2[pos2]
            else:
                idx2 = max_idx
        else:
            if new_val[new_pos] != 0:
                new_pos += 1
            if idx1 < idx2:
                new_idx = idx1
                new_val[new_pos] = val1[pos1]
                pos1 += 1
                if pos1 < len1:
                    idx1 = ind1[pos1]
                else:
                    idx1 = max_idx
            else:
                new_idx = idx2
                new_val[new_pos] = val2[pos2]
                pos2 += 1
                if pos2 < len2:
                    idx2 = ind2[pos2]
                else:
                    idx2 = max_idx
            new_ind[new_pos] = new_idx

    if new_val[new_pos] != 0:
        new_pos += 1

    return new_ind[:new_pos], new_val[:new_pos]


def merge_sorted_doub(long long[:] ind1, double[:] val1, long long[:] ind2, double[:] val2):
    """Merge two vectors in sparse format whose indices are already sorted.

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
    (numpy.ndarray, int64)
        Sorted indices of merged vector
    (numpy.ndarray, float64)
        Values of merged vector
    """
    cdef size_t pos1 = 0, pos2 = 0
    cdef size_t len1 = ind1.shape[0]
    cdef size_t len2 = ind2.shape[0]
    cdef long long idx1 = ind1[0]
    cdef long long idx2 = ind2[0]
    cdef size_t new_pos = 0
    cdef size_t new_idx
    cdef numpy.ndarray[numpy.int64_t] new_ind = numpy.zeros(len1 + len2, dtype=numpy.int64)
    cdef numpy.ndarray[numpy.float64_t] new_val = numpy.zeros(len1 + len2)

    cdef long long max_idx = ind1[len1 - 1]
    if max_idx < ind2[len2 - 1]:
        max_idx = ind2[len2 - 1]
    max_idx += 1

    # Make sure new_idx is not equal to idx1 or idx2
    new_idx = idx1 + 1
    if new_idx == idx2:
        new_idx = idx2 + 1

    while pos1 < len1 or pos2 < len2:
        if idx1 == new_idx:
            new_val[new_pos] += val1[pos1]
            pos1 += 1
            if pos1 < len1:
                idx1 = ind1[pos1]
            else:
                idx1 = max_idx
        elif idx2 == new_idx:
            new_val[new_pos] += val2[pos2]
            pos2 += 1
            if pos2 < len2:
                idx2 = ind2[pos2]
            else:
                idx2 = max_idx
        else:
            if new_val[new_pos] != 0:
                new_pos += 1
            if idx1 < idx2:
                new_idx = idx1
                new_val[new_pos] = val1[pos1]
                pos1 += 1
                if pos1 < len1:
                    idx1 = ind1[pos1]
                else:
                    idx1 = max_idx
            else:
                new_idx = idx2
                new_val[new_pos] = val2[pos2]
                pos2 += 1
                if pos2 < len2:
                    idx2 = ind2[pos2]
                else:
                    idx2 = max_idx
            new_ind[new_pos] = new_idx

    new_pos += 1

    return new_ind[:new_pos], new_val[:new_pos]


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

    cdef size_t tot_num = numpy.sum(counts)
    cdef size_t num_counts = counts.shape[0]
    cdef numpy.ndarray[numpy.uint32_t] indices = numpy.zeros(tot_num,
                                                             dtype=numpy.uint32)
    cdef size_t i, j, pos = 0

    for i in range(num_counts):
        for j in range(counts[i]):
            indices[pos] = i
            pos += 1
    return indices


def seq_from_count(unsigned int[:] counts):
    """Returns a concatenation of sequences of 0, 1, 2, ...,
        each with a length of counts[i]

    Parameters
    ----------
    counts : (numpy.ndarray, uint32)
        length of each sequence of integers

    Returns
    -------
    (numpy.ndarray, uint32) :
        vector of sequences, with a size of numpy.sum(counts)

    Example
    -------
    >>> fci_c_utils.ind_from_count(numpy.array([3, 1, 4, 1]))
    array([0, 1, 2, 0, 0, 1, 2, 3, 0])
    """

    cdef size_t tot_num = numpy.sum(counts)
    cdef size_t num_counts = counts.shape[0]
    cdef numpy.ndarray[numpy.uint32_t] indices = numpy.zeros(tot_num,
                                                             dtype=numpy.uint32)
    cdef size_t i, j, pos = 0

    for i in range(num_counts):
        for j in range(counts[i]):
            indices[pos] = j
            pos += 1
    return indices


@cython.wraparound(True)
def setup_alias(probs):
    '''Calculate the probabilities Q(i) and aliases A(i) needed to perform multinomial sampling,
    by the alias method, as described in Appendix D of Holmes et al. (2016).

    Parameters
    ----------
    probs : (numpy.ndarray, float64)
        Probability array with any number of dimensions. Sum along the last index must be 1

    Returns
    -------
    (numpy.ndarray, uint32) :
        has the same shape as probs, aliases of states in probs array
    (numpy.ndarray, float64) :
        has the same same as probs, probability of returning the state instead
        of its alias.
    '''
    p_shape = probs.shape
    probs.shape = (-1, p_shape[-1])
    alias, Q = _alias_2D(probs)
    probs.shape = p_shape
    alias.shape = p_shape
    Q.shape = p_shape

    return alias, Q


def linsearch_1D(double[:] search_list, double[:] search_vals):
    """Equivalent to numpy.searchsorted(search_list, search_vals), except
    leverages the fact that search_vals is sorted for efficiency.

    Parameters
    ----------
    search_list : (numpy.ndarray, float64)
        sorted list to search
    search_vals : (numpy.ndarray, float64)
        elements to find in search_list

    Returns
    -------
    (numpy.ndarray, uint32)
        position of each search_vals[i] in search_list
    """
    cdef size_t n_search = search_vals.shape[0]
    cdef size_t search_idx
    cdef unsigned int search_pos = 0
    cdef numpy.ndarray[numpy.uint32_t] ret_idx = numpy.zeros(n_search, dtype=numpy.uint32)

    for search_idx in range(n_search):
        while search_list[search_pos] < search_vals[search_idx]:
            search_pos += 1
        ret_idx[search_idx] = search_pos
    return ret_idx


def linsearch_1D_cts(double[:] search_list, double[:] search_vals):
    cdef size_t n_search = search_vals.shape[0]
    cdef unsigned int search_pos = 0
    cdef numpy.ndarray[numpy.uint32_t] ret_cts = numpy.zeros(search_list.shape[0], dtype=numpy.uint32)

    for search_idx in range(n_search):
        while search_list[search_pos] < search_vals[search_idx]:
            search_pos += 1
        ret_cts[search_pos] += 1
    return ret_cts


def linsearch_2D(double[:, :] search_lists, unsigned int[:] row_idx,
                 double[:] search_vals):
    """Equivalent to numpy.searchsorted(search_lists[row_idx[i]], search_vals[i])
        for all i, except that search_vals[i] cannot exceed search_lists[row_idx[i], -1]

    Parameters
    ----------
    search_lists : (numpy.ndarray, float64)
        lists to search. Elements in each row must be in order.
    row_idx : (numpy.ndarray, uint32)
        indices of rows to search
    search_vals : (numpy.ndarray, float64)
        values to search

    Returns
    -------
    (numpy.ndarray, uint32)
        column index for each search value
    """
    cdef size_t n_search = search_vals.shape[0]
    cdef size_t search_idx = 0
    cdef unsigned int curr_row = row_idx[0]
    cdef unsigned int curr_col = 0
    cdef double col_pos = 0.
    cdef numpy.ndarray[numpy.uint32_t] ret_idx = numpy.zeros(n_search, dtype=numpy.uint32)

    while search_idx < n_search:
        if curr_row < row_idx[search_idx]:
            curr_row = row_idx[search_idx]
            curr_col = 0
            col_pos = 0
        while col_pos <= search_vals[search_idx]:
            col_pos += search_lists[curr_row, curr_col]
            curr_col += 1
        ret_idx[search_idx] = curr_col - 1
        search_idx += 1
    return ret_idx


def _alias_2D(double[:, :] probs):
    cdef unsigned int n_s, n_b, s, b
    cdef size_t num_sys = probs.shape[0]
    cdef unsigned int num_states = probs.shape[1]
    cdef size_t i
    cdef unsigned int j
    cdef numpy.ndarray[numpy.uint32_t, ndim=2] aliases = numpy.zeros([num_sys, num_states], dtype=numpy.uint32)
    cdef numpy.ndarray[numpy.float64_t, ndim=2] Q = numpy.zeros([num_sys, num_states], dtype=numpy.float64)

    cdef unsigned int thread_idx
    cdef unsigned int n_threads = openmp.omp_get_max_threads()
    cdef numpy.ndarray[numpy.uint32_t, ndim=2] smaller = numpy.zeros([n_threads, num_states], dtype=numpy.uint32)
    cdef numpy.ndarray[numpy.uint32_t, ndim=2] bigger = numpy.zeros([n_threads, num_states], dtype=numpy.uint32)

    for j in prange(num_sys, nogil=True, schedule=static, num_threads=n_threads):
        n_s = 0
        n_b = 0
        thread_idx = threadid()
        for i in range(num_states):
            aliases[j, i] = i
            Q[j, i] = num_states * probs[j, i]
            if Q[j, i] < 1.:
                smaller[thread_idx, n_s] = i
                n_s = n_s + 1
            else:
                bigger[thread_idx, n_b] = i
                n_b = n_b + 1

        while (n_s > 0 and n_b > 0):
            s = smaller[thread_idx, n_s - 1]
            b = bigger[thread_idx, n_b - 1]
            aliases[j, s] = b
            Q[j, b] += Q[j, s] - 1
            if Q[j, b] < 1:
                smaller[thread_idx, n_s - 1] = b
                n_b = n_b - 1
            else:
                n_s = n_s - 1
        for i in range(num_states):
            if Q[j, i] > 1.:
                Q[j, i] = 1.

    return aliases, Q
