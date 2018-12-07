#!/usr/bin/env python2
"""
Utilities for performing stochastic matrix and vector compressions.
"""

import numpy
import misc_c_utils
import near_uniform


def fri_subd(vec, num_div, sub_weights, n_samp):
    """ Perform FRI compression on a vector whose first elements,
    vec[i] are each subdivided into equal segments, and
    whose last elements are each divided into unequal segments.

    Parameters
    ----------
    vec : (numpy.ndarray, float)
        vector on which to perform compression. Elements can
        be negative, and it need not be normalized. vec.shape[0]
        must equal num_div.shape[0] + sub_weights.shape[0]
    num_div : (numpy.ndarray, unsigned int)
        the first num_div.shape[0] elements of vec are subdivided
        into equal segments, the number of which for each element
        is specified in this array
    sub_weights : (numpy.ndarray, float)
        the weights of the unequal subdivisions of the last
        sub_weights.shape[0] elements of vec. Must be row-
        normalized.

    Returns
    -------
    (numpy.ndarray, uint32)
        2-D array of indices of nonzero elements in the vector. The
        0th column specifies the index in the vec array, while
        the 1st specifies the index within the subdivision. Not
        necessarily sorted, although indices in the uniform part
        of the array are grouped first, followed by indices in the
        nonuniform part.
    (numpy.ndarray, float64)
        values of nonzero elements in the compressed vector
    """
    new_idx = numpy.zeros([n_samp, 2], dtype=numpy.uint32)
    new_vals = numpy.zeros(n_samp)

    weights = numpy.abs(vec)
    sub_cp = numpy.copy(sub_weights)

    preserve_uni, preserve_nonuni = _keep_idx(weights, num_div, sub_cp, n_samp)
    preserve_counts = numpy.zeros_like(num_div, dtype=numpy.uint32)
    preserve_counts[preserve_uni] = num_div[preserve_uni]
    uni_rpt = misc_c_utils.ind_from_count(preserve_counts)
    n_exact_uni = uni_rpt.shape[0]
    new_idx[:n_exact_uni, 0] = uni_rpt
    uni_seq = misc_c_utils.seq_from_count(preserve_counts)
    new_idx[:n_exact_uni, 1] = uni_seq
    new_vals[:n_exact_uni] = vec[uni_rpt] / num_div[uni_rpt]

    nonuni_exact_idx = numpy.nonzero(preserve_nonuni)
    n_exact_nonuni = nonuni_exact_idx[0].shape[0]

    n_samp -= (n_exact_uni + n_exact_nonuni)
    num_uni_wt = num_div.shape[0]

    sub_renorm = numpy.sum(sub_cp, axis=1)
    weights[num_uni_wt:] *= sub_renorm
    sub_renorm.shape = (-1, 1)
    sub_renorm = 1. / sub_renorm
    sub_cp *= sub_renorm
    one_norm = weights.sum()

    if abs(one_norm) > 1e-10:
        sampl_idx = sys_subd(weights, num_div, sub_cp, n_samp)
        end_idx = n_exact_uni + n_samp
        new_idx[n_exact_uni:end_idx] = sampl_idx
        new_vals[n_exact_uni:end_idx] = numpy.sign(vec[sampl_idx[:, 0]]) * one_norm / n_samp
    else:
        end_idx = n_exact_uni

    end_idx2 = end_idx + n_exact_nonuni
    new_idx[end_idx:end_idx2, 0] = nonuni_exact_idx[0] + num_uni_wt
    new_idx[end_idx:end_idx2, 1] = nonuni_exact_idx[1]
    new_vals[end_idx:end_idx2] = vec[num_uni_wt + nonuni_exact_idx[0]] * sub_weights[nonuni_exact_idx]
    return new_idx[:end_idx2], new_vals[:end_idx2]


def fri_1D(vec, n_samp):
    """Compress a vector in full (non-sparse format) using the
        FRI scheme, potentially preserving some elements exactly.

    Parameters
    ----------
    vec : (numpy.ndarray)
        vector to compress
    n_samp : (unsigned int)
        desired number of nonzero entries in the output vector

    Returns
    -------
    (numpy.ndarray, unsigned int)
        indices of nonzero elements in compressed vector, in order
    (numpy.ndarray, float)
        values of nonzero elements in compressed vector
    """
    weights = numpy.abs(vec)
    new_vec = numpy.zeros(weights.shape[0])

    counts = numpy.ones_like(vec, dtype=numpy.uint32)
    sub_wts = numpy.empty((0, 2))
    preserve_idx, empty_ret = _keep_idx(weights, counts, sub_wts, n_samp)
    preserve_vals = vec[preserve_idx]
    new_vec[preserve_idx] = preserve_vals

    n_samp -= preserve_vals.shape[0]
    one_norm = weights.sum()

    if abs(one_norm) > 1e-10:
        sampl_idx = sys_resample(weights, n_samp, ret_idx=True)
        new_vec[sampl_idx] = one_norm / n_samp * numpy.sign(vec[sampl_idx])

    nonz_idx = numpy.nonzero(new_vec)[0]
    return nonz_idx, new_vec[nonz_idx]


def _keep_idx(weights, num_div, sub_weights, n_samp):
    # Calculate indices of elements in weights that are preserved exactly
    # Elements in weights are subdivided according to num_div and sub_weights
    num_uni = num_div.shape[0]
    uni_keep = numpy.zeros(num_uni, dtype=numpy.bool_)
    nonuni_keep = numpy.zeros_like(sub_weights, dtype=numpy.bool_)
    one_norm = weights.sum()
    any_kept = True
    uni_weights = weights[:num_uni] / num_div
    nonuni_weights = weights[num_uni:]
    nonuni_weights.shape = (-1, 1)
    nonuni_weights = nonuni_weights * sub_weights
    while any_kept and one_norm > 1e-9:
        add_uni = one_norm / n_samp <= uni_weights
        uni_weights[add_uni] = 0
        uni_keep[add_uni] = True
        num_add_uni = num_div[add_uni].sum()
        n_samp -= num_add_uni
        one_norm -= weights[:num_uni][add_uni].sum()

        if one_norm > 0:
            add_nonuni = one_norm / n_samp <= nonuni_weights
            chosen_weights = nonuni_weights[add_nonuni]
            nonuni_weights[add_nonuni] = 0
            nonuni_keep[add_nonuni] = True
            num_add_nonuni = chosen_weights.shape[0]
            n_samp -= num_add_nonuni
            one_norm -= chosen_weights.sum()
        else:
            num_add_nonuni = 0

        any_kept = num_add_uni > 0 or num_add_nonuni > 0

    sub_weights[nonuni_keep] = 0
    weights[:num_uni][uni_keep] = 0
    return uni_keep, nonuni_keep


def sys_resample(vec, nsample, ret_idx=False, ret_counts=False):
    """Choose nsample elements of vector vec according to systematic resampling
        algorithm (eq. 44-46 in SIAM Rev. 59 (2017), 547-587)

    Parameters
    ----------
    vec : (numpy.ndarray, float)
        the weights for each index
    nsample : (unsigned int)
        the number of samples to draw
    ret_idx : (bool, optional)
        If True, return a vector containing the indices (possibly repeated) of
        chosen indices
    ret_counts : (bool, optional)
        If True, return a 1-D vector of the same shape as vec with the number of
        chosen samples at each position

    Returns
    -------
    (tuple)
        Contains 0, 1, or 2 numpy vectors depending on the values of input parameters
        ret_idx and ret_counts
    """

    if nsample == 0:
        return numpy.array([], dtype=int)

    rand_points = (numpy.linspace(0, 1, num=nsample, endpoint=False) +
                   numpy.random.uniform(high=1. / nsample))
    intervals = numpy.cumsum(vec)
    # normalize if necessary
    if intervals[-1] != 1.:
        intervals /= intervals[-1]
    ret_tuple = ()
    if ret_idx:
        ret_tuple = ret_tuple + (misc_c_utils.linsearch_1D(intervals, rand_points),)
    if ret_counts:
        ret_tuple = ret_tuple + (misc_c_utils.linsearch_1D_cts(intervals, rand_points),)
    return ret_tuple


def sys_subd(weights, counts, sub_weights, nsample):
    """Performs systematic resampling on a vector of weights subdivided
    according to counts and sub_weights
    Parameters
    ----------
    weights : (numpy.ndarray, float)
        vector of weights to be subdivided. weights.shape[0] must equal
        counts.shape[0] + sub_weights.shape[0]
    counts : (numpy.ndarray, unsigned int)
        the first counts.shape[0] elements of weights are subdivided into
        equal subintervals
    sub_weights : (numpy.ndarray, float)
        sub_weights[i] corresponds to the subdivisions of weights[i].
        Must be row-normalized
    n_sample : (unsigned int)
        number of samples to draw
    Returns
    -------
    (numpy.ndarray, unsigned int)
        2-D array of chosen indices. The 0th column is the index in
        the weights vector, and the 1st is the index with the
        subdivision.
    """

    if nsample == 0:
        return numpy.empty((0, 2), dtype=numpy.uint32)

    rand_points = (numpy.arange(0, 1, 1. / nsample) +
                   numpy.random.uniform(high=1. / nsample))
    rand_points = rand_points[:nsample]
    big_intervals = numpy.cumsum(weights)
    one_norm = big_intervals[-1]
    # normalize if necessary
    if abs(one_norm - 1.) > 1e-10:
        big_intervals /= one_norm

    ret_idx = numpy.zeros([nsample, 2], dtype=numpy.uint32)
    weight_idx = misc_c_utils.linsearch_1D(big_intervals, rand_points)
    ret_idx[:, 0] = weight_idx
    rand_points[weight_idx > 0] -= big_intervals[weight_idx[weight_idx > 0] - 1]
    rand_points *= one_norm / weights[weight_idx]
    rand_points[rand_points >= 1.] = 0.999999

    n_uni_wts = counts.shape[0]
    num_uni = numpy.searchsorted(weight_idx, n_uni_wts)
    ret_idx[:num_uni, 1] = rand_points[:num_uni] * counts[weight_idx[:num_uni]]

    subweight_idx = misc_c_utils.linsearch_2D(sub_weights, weight_idx[num_uni:] - n_uni_wts,
                                              rand_points[num_uni:])
    ret_idx[num_uni:, 1] = subweight_idx
    return ret_idx


def round_binomially(vec, num_round):
    """Round non-integer entries in vec to integer entries in b such that
    b[i] ~ (binomial(num_round[i], vec[i] - floor(vec[i])) + floor(vec[i])
            * num_round)
    Parameters
    ----------
    vec : (numpy.ndarray, float)
        non-integer numbers to be rounded
    num_round : (numpy.ndarray, unsigned int)
        parameter of the binomial distribution for each entry in vec, must
        have same shape as vec
    Returns
    -------
    (numpy.ndarray, int)
        integer array of results
    """

    flr_vec = numpy.floor(vec)
    flr_vec = flr_vec.astype(numpy.int32)
    n = num_round.astype(numpy.uint32)
    b = flr_vec * num_round + numpy.random.binomial(n, vec - flr_vec).astype(numpy.int32)
    return b


def round_bernoulli(vec, mt_ptrs):
    """Round non-integer entries in vec to integer entries in b such that
    b[i] ~ binomial(1, vec[i] - floor(vec[i])) + floor(vec[i])
    Parameters
    ----------
    vec : (numpy.ndarray, float)
        non-integer numbers to be rounded
    mt_ptrs : (numpy.ndarray, uint64)
        List of addresses to MT state objects to use for RN generation
    Returns
    -------
    (numpy.ndarray, int)
        integer array of results
    """

    flr_vec = numpy.floor(vec)
    flr_vec = flr_vec.astype(numpy.int32)
    b = flr_vec + near_uniform.par_bernoulli(vec - flr_vec, mt_ptrs)
    return b


def sample_alias(alias, Q, row_idx, mt_ptrs):
    """Perform multinomial sampling using the alias method for an array of
    probability distributions.
    Parameters
    ----------
    alias : (numpy.ndarray, unsigned int)
        alias indices as calculated in cyth_helpers2.setup_alias
    Q : (numpy.ndarray, float)
        alias probabilities as calculated in cyth_helpers2.setup_alias
    row_idx : (numpy.ndarray, unsigned int)
        Row index in alias/Q of each value to sample. Can be obtained from
        desired numbers of samples using cyth_helpers2.ind_from_count()
    mt_ptrs : (numpy.ndarray, uint64)
        List of addresses to MT state objects to use for RN generation
    Returns
    -------
    (numpy.ndarray, unsigned char)
        1-D array of chosen column indices of each sample
    """

    n_states = alias.shape[1]
    tot_samp = row_idx.shape[0]
    r_ints = numpy.random.randint(n_states, size=tot_samp)
    orig_success = near_uniform.par_bernoulli(Q[row_idx, r_ints], mt_ptrs)
    orig_idx = orig_success == 1
    alias_idx = numpy.logical_not(orig_idx)

    choices = numpy.zeros(tot_samp, dtype=numpy.uint)
    choices[orig_idx] = r_ints[orig_idx]
    choices[alias_idx] = alias[row_idx[alias_idx], r_ints[alias_idx]]
    choices = choices.astype(numpy.uint8)
    return choices


def proc_fri_sd_choices(sampl_idx, n_sing, all_arrs, sing_arrs, doub_arrs):
    """Given the results from an FRI compression in which the weights for single excitations
    are uniformly subdivided and those for double excitations are subdivided nonuniformly,
    rearrange arrays of observables such that they are indexed by the sample index from the compression,
    instead of the index in the uncompressed array.
    Parameters
    ----------
    sampl_idx : (numpy.ndarray, unsigned int)
        Array of chosen top-level indices from the FRI compression
    n_sing : (unsigned int)
        Number of single excitation elements in the uncompressed array
    all_arrs : (python list of numpy.ndarrays)
        Each array contains numbers corresponding to each element in the original uncompressed array
    sing_arrs : (python list of numpy.ndarrays)
        Each array contains numbers corresponding only to single excitation elements in the orinal
        uncompressed array
    doub_arrs : (python list of numpy.ndarrays)
        Each array contains numbers corresponding only to double excitation elements in the orinal
        uncompressed array
    Returns
    -------
    (python list of numpy.ndarrays), (python list of numpy.ndarrays), (python list of numpy.ndarrays)
        Each array contains numbers corresponding to the same parameters as in the inputted arrays,
        except they are now indexed according to the sampled indices in the compressed array.
    (unsigned int)
        Number of single excitation elements in the compressed array    
    """
    # sampl_idx isn't necessarily sorted, but that's ok because single excitation elements come first
    new_n_sing = numpy.searchsorted(sampl_idx, n_sing)
    new_all = []
    for arr in all_arrs:
        new_all.append(arr[sampl_idx])
    new_sing = []
    for arr in sing_arrs:
        new_sing.append(arr[sampl_idx[:new_n_sing]])
    doub_idx = sampl_idx[new_n_sing:] - n_sing
    new_doub = []
    for arr in doub_arrs:
        new_doub.append(arr[doub_idx])
    return new_all, new_sing, new_doub, new_n_sing
