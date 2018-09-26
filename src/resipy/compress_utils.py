"""
Utilities for performing stochastic matrix and vector compressions.
"""

import numpy


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
    flr_vec = flr_vec.astype(int)
    b = flr_vec * num_round + numpy.random.binomial(num_round, vec - flr_vec)
    return b


def sample_alias(alias, Q, row_idx):
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

        Returns
        -------
        (numpy.ndarray, unsigned char)
            1-D array of chosen column indices of each sample
    """

    n_states = alias.shape[1]
    tot_samp = row_idx.shape[0]
    r_ints = numpy.random.randint(n_states, size=tot_samp)
    orig_success = numpy.random.binomial(1, Q[row_idx, r_ints])
    orig_idx = orig_success == 1
    alias_idx = numpy.logical_not(orig_idx)

    choices = numpy.zeros(tot_samp, dtype=numpy.uint)
    choices[orig_idx] = r_ints[orig_idx]
    choices[alias_idx] = alias[row_idx[alias_idx], r_ints[alias_idx]]
    choices = choices.astype(numpy.uint8)
    return choices


def sys_resample(vec, nsample):
    """Choose nsample elements of vector vec according to systematic resampling
        algorithm (eq. 44-46 in SIAM Rev. 59 (2017), 547-587)

        Parameters
        ----------
        vec : (numpy.ndarray, float)
            the elements (probabilities) to be sampled. vec is assumed to have
            a 1-norm of 1
        nsample : (unsigned int)
            the number of samples to draw

        Returns
        -------
        (numpy.ndarray, unsigned int)
            indices of chosen elements (duplicates are possible)
    """

    if nsample == 0:
        return numpy.array([], dtype=int)

    rand_points = (numpy.linspace(0, 1, num=nsample, endpoint=False) +
                   numpy.random.uniform(high=1. / nsample))
    intervals = numpy.cumsum(vec)
    # normalize if necessary
    if intervals[-1] != 1.:
        intervals /= intervals[-1]
    return numpy.searchsorted(intervals, rand_points)


def compress_sparse_vector(vec_idx, vec_vals, m_nonzero, take_abs=True):
    """Compresses a sparse vector according to the FRI framework.

        Parameters
        ----------
        vec_idx : (numpy.ndarray, unsigned int)
            the indices of nonzero elements in the input vector. Multi-indexing
            is allowed, with each row denoting a multi-index. It is assumed
            that there are no duplicate rows in vec_idx.
        vec_vals : (numpy.ndarray, float)
            the values of nonzero elements in the input vector. vec_vals is
            MODIFIED inside this function.
        m_nonzero : (unsigned int)
            the desired number of nonzero elements in the output vector
        take_abs : (bool)
            If False, it is assumed that all of the elements of vec_vals are
            real and nonnegative.

        Returns
        -------
        (numpy.ndarray)
            indices of nonzero elements in the resulting vector
        (numpy.ndarray)
            values of nonzero elements in the resulting vector
    """

    initial_n = vec_idx.shape[0]
    num_to_sample = numpy.minimum(m_nonzero, initial_n)

    if take_abs:
        vec_abs = numpy.abs(vec_vals)
    else:
        vec_abs = vec_vals

    idx_shape = vec_idx.shape
    # Allocate arrays for indices and elements of the output vector
    if len(idx_shape) == 1:
        out_idx = numpy.zeros(num_to_sample, dtype=int)
    else:
        out_idx = numpy.zeros([num_to_sample, idx_shape[1]], dtype=int)
    out_vals = numpy.zeros(num_to_sample)

    big_idx = _get_largest_idx(vec_abs, num_to_sample)
    tau = big_idx.shape[0]

    out_idx[:tau] = vec_idx[big_idx]
    out_vals[:tau] = vec_vals[big_idx]

    vec_abs[big_idx] = 0
    vec_norm = vec_abs.sum()
    chosen_idx = sys_resample(vec_abs / vec_norm, num_to_sample - tau)
    resamp_weight = vec_norm / (num_to_sample - tau)

    if take_abs:
        out_vals[tau:] = numpy.sign(vec_vals[chosen_idx]) * resamp_weight
    else:
        out_vals[tau:] = resamp_weight
    out_idx[tau:] = vec_idx[chosen_idx]

    return out_idx, out_vals


def _get_largest_idx(vec_abs_el, n_sample):
    # Calculate the indices of the elements in the weights vector that are to
    # preserved exactly, according to eq. 42 in the FRI paper.

    vec_len = vec_abs_el.shape[0]
    srt_idx = vec_abs_el.argsort()  # ascending order
    srt_weights = vec_abs_el[srt_idx]
    lhs = numpy.cumsum(srt_weights)  # LHS of inequality in eq 42
    rhs = numpy.linspace(n_sample - vec_len + 1, n_sample, num=vec_len,
                         dtype=int) * srt_weights  # RHS of inequality in eq 42
    compare_vec = lhs >= rhs
    tau = vec_len - numpy.sum(compare_vec)
    return srt_idx[(vec_len - tau):]
