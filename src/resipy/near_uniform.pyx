# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Utilities for sampling the Near-Uniform distribution described in
Booth et al. (2014) (doi:10.1080/00268976.2013.877165).
"""

import numpy
from cython.parallel import prange, threadid, parallel
cimport numpy
cimport cython
from libc.math cimport sqrt, fabs
from libc.time cimport time
from libc.stdio cimport printf


cdef extern from "dc.h":
    ctypedef struct mt_struct:
        pass
    mt_struct *get_mt_parameter_id_st(int, int, int, unsigned int)
    unsigned int genrand_mt(mt_struct *) nogil
    void sgenrand_mt(unsigned int seed, mt_struct *mts)


cdef struct s_orb_pair:
    unsigned char orb1
    unsigned char orb2
    unsigned char spin1
    unsigned char spin2
ctypedef s_orb_pair orb_pair

DEF RAND_MAX = 4294967296.


def initialize_mt(unsigned int num_threads):
    ''' Setup an array of Mersenne twister state objects to use for
        multithreaded sampling.

    Parameters
    ----------
    num_threads : (unsigned int)
        The number of state objects to initialize

    Returns
    -------
    (numpy.ndarray, uint64)
        Vector of addresses of each of the state objects

    '''

    cdef unsigned int i
    cdef mt_struct *mts
    cdef numpy.ndarray[numpy.uint64_t] ini_ptrs = numpy.zeros(num_threads, dtype=numpy.uint64)

    for i in range(num_threads):
        mts = get_mt_parameter_id_st(32, 521, i, 4172)
        sgenrand_mt(1000 * i + time(NULL), mts)
        ini_ptrs[i] = < unsigned long > mts
    return ini_ptrs


def bin_n_sing_doub(col_nsamp, p_doub):
    """Binomially partition the samples for each column into single and double excitations.

    Parameters
    ----------
    col_nsamp : (numpy.ndarray, uint32)
        Number of off-diagonal elements to choose from each column
    p_doub : (float)
        Probability of choosing a double, instead of a single excitation
    """

    doub_samp = numpy.random.binomial(col_nsamp, p_doub)
    doub_samp = doub_samp.astype(numpy.uint32)
    sing_samp = col_nsamp - doub_samp
    sing_samp = sing_samp.astype(numpy.uint32)
    return doub_samp, sing_samp


def doub_multin(long long[:] dets, unsigned char[:, :] occ_orbs,
                unsigned char[:] orb_symm, unsigned char[:, :] lookup_tabl,
                unsigned int[:] num_sampl, unsigned long[:] mt_ptrs):
    ''' Uniformly chooses num_sampl[i] double excitations for each determinant
        dets[i] according to the symmetry-adapted rules described in Sec. 5.2
        of Booth et al. (2014) using independent multinomial sampling.

    Parameters
    ----------
    dets : (numpy.ndarray, int64)
        Bit string representations of all determinants to be sampled
    occ_orbs : (numpy.ndarray, uint8)
        The numbers in each row correspond to the indices of occupied
        orbitals in each determinant, calculated from fci_c_utils.gen_orb_lists
    orb_symm : (numpy.ndarray, uint8)
        irreducible representation of each spatial orbital
    lookup_tabl : (numpy.ndarray, uint8)
        Table of orbitals with each type of symmetry, as generated
        by fci_utils.gen_symm_lookup()
    num_sampl : (numpy.ndarray, uint32)
        number of double excitations to choose for each determinant
    mt_ptrs : (numpy.ndarray, uint64)
        List of addresses to MT state objects to use for RN generation

    Returns
    -------
    (numpy.ndarray, uint8) :
        chosen occupied (0th and 1st columns) and unoccupied (2nd and 3rd
        columns) orbitals
    (numpy.ndarray, float64) :
        probability of each choice
    (numpy.ndarray, uint32) :
        index of the origin determinant of each chosen excitation in the
        dets array
    '''

    cdef size_t num_dets = dets.shape[0]
    cdef unsigned int num_elec = occ_orbs.shape[1]
    cdef unsigned int num_orb = orb_symm.shape[0]
    cdef size_t det_idx
    cdef unsigned int i, a_symm, b_symm, a_spin, b_spin, sym_prod
    cdef unsigned int unocc1, unocc2, m_a_allow, m_a_b_allow, m_b_a_allow
    cdef orb_pair occ
    cdef long long curr_det
    cdef size_t tot_sampl = 0
    cdef double prob
    cdef numpy.ndarray[numpy.uint32_t] start_idx = numpy.zeros(num_dets, dtype=numpy.uint32)
    cdef unsigned int thread_idx, n_threads = mt_ptrs.shape[0]
    cdef unsigned int n_symm = lookup_tabl.shape[0]
    cdef unsigned int[:, :, :] unocc_sym_counts = numpy.zeros([n_threads, n_symm, 2], dtype=numpy.uint32)
    cdef mt_struct *mts

    for det_idx in range(num_dets):
        start_idx[det_idx] = tot_sampl
        tot_sampl += num_sampl[det_idx]

    cdef numpy.ndarray[numpy.uint8_t, ndim=2] chosen_orbs = numpy.zeros([tot_sampl, 4],
                                                                        dtype=numpy.uint8)
    cdef numpy.ndarray[numpy.float64_t] prob_vec = numpy.zeros(tot_sampl)
    cdef numpy.ndarray[numpy.uint32_t] idx_arr = numpy.zeros(tot_sampl, dtype=numpy.uint32)

    for det_idx in prange(num_dets, nogil=True, schedule=static, num_threads=n_threads):
        if (num_sampl[det_idx] == 0):
            continue
        tot_sampl = start_idx[det_idx]
        curr_det = dets[det_idx]
        thread_idx = threadid()
        mts = < mt_struct * > mt_ptrs[thread_idx]

        _count_symm_virt(<unsigned int (*)[2]> &unocc_sym_counts[thread_idx, 0, 0], &occ_orbs[det_idx, 0],
                         num_elec, lookup_tabl, orb_symm)

        # Start sampling
        for i in range(num_sampl[det_idx]):
            occ = _choose_occ_pair(&occ_orbs[det_idx, 0], num_elec, mts)

            sym_prod = orb_symm[occ.orb1 % num_orb] ^ orb_symm[occ.orb2 % num_orb]

            m_a_allow = _count_doub_virt(occ, orb_symm, num_elec,
                                         <unsigned int (*)[2]> &unocc_sym_counts[thread_idx, 0, 0], n_symm)

            if m_a_allow == 0:
                prob_vec[tot_sampl + i] = -1.
                continue

            unocc1 = _doub_choose_virt1(occ, curr_det, orb_symm,
                                        <unsigned int (*)[2]> &unocc_sym_counts[thread_idx, 0, 0], m_a_allow,
                                        mts)
            a_spin = unocc1 / num_orb  # spin of 1st virtual orbital chosen
            b_spin = occ.spin1 ^ occ.spin2 ^ a_spin  # 2nd virtual spin
            a_symm = orb_symm[unocc1 % num_orb]
            b_symm = sym_prod ^ a_symm

            m_a_b_allow = (unocc_sym_counts[thread_idx, b_symm, b_spin] -
                           (sym_prod == 0 and a_spin == b_spin))

            # Choose second unoccupied orbital
            unocc2 = _doub_choose_virt2(b_spin * num_orb, curr_det,
                                        &lookup_tabl[b_symm, 0],
                                        unocc1, m_a_b_allow, mts)

            # Calculate probability of choosing this excitation
            m_b_a_allow = (unocc_sym_counts[thread_idx, a_symm, a_spin] -
                           (sym_prod == 0 and a_spin == b_spin))

            prob = 2. / num_elec / (num_elec - 1) / (m_a_allow) * (1. / m_a_b_allow
                                                                   + 1. / m_b_a_allow)

            chosen_orbs[tot_sampl + i, 0] = occ.orb2
            chosen_orbs[tot_sampl + i, 1] = occ.orb1
            if unocc1 < unocc2:
                chosen_orbs[tot_sampl + i, 2] = unocc1
                chosen_orbs[tot_sampl + i, 3] = unocc2
            else:
                chosen_orbs[tot_sampl + i, 2] = unocc2
                chosen_orbs[tot_sampl + i, 3] = unocc1
            prob_vec[tot_sampl + i] = prob
            idx_arr[tot_sampl + i] = det_idx

    successes = prob_vec > 0
    chosen_orbs = chosen_orbs[successes]
    prob_vec = prob_vec[successes]
    idx_arr = idx_arr[successes]
    return chosen_orbs, prob_vec, idx_arr


def virt_symm(unsigned char[:, :] occ_orbs, unsigned char[:] orb_symm,
              unsigned char[:, :] lookup_tabl):
    '''Count the number of virtual orbitals with each irrep in each
        determinant in an array.

    Parameters
    ----------
    dets : (numpy.ndarray, int64)
        Bit string representations of the determinants in question
    occ_orbs : (numpy.ndarray, uint8)
        The numbers in each row correspond to the indices of occupied
        orbitals in each determinant
    orb_symm : (numpy.ndarray, uint8)
        irreducible representation of each spatial orbital
    lookup_tabl : (numpy.ndarray, uint8)
        Table of orbitals with each type of symmetry, as generated
        by fci_utils.gen_symm_lookup()
    Returns
    -------
    (numpy.ndarray, uint8)
        3-D array with the 0th index representing the determinant index,
        the 1st representing the spin, and the 2nd the spatial irrep
        index.
    '''
    cdef size_t num_dets = occ_orbs.shape[0]
    cdef unsigned int n_symm = lookup_tabl.shape[0]
    cdef size_t det_idx
    cdef numpy.ndarray[numpy.uint32_t, ndim=3] virt_counts = numpy.zeros([num_dets, n_symm, 2], dtype=numpy.uint32)

    for det_idx in range(num_dets):
        _count_symm_virt(<unsigned int (*)[2]> &virt_counts[det_idx, 0, 0], &occ_orbs[det_idx, 0],
                         occ_orbs.shape[1], lookup_tabl, orb_symm)
    return virt_counts


def sing_allow(unsigned int[:, :, :] virt_counts, unsigned char[:, :] occ_orbs,
               unsigned char[:] orb_symm):
    '''Identify the symmetry-allowed single excitations from each determinant
        in an array.

    Parameters
    ----------
    virt_counts : (numpy.ndarray, uint8)
        The number of virtual orbitals with each irrep in each determinant
    occ_orbs : (numpy.ndarray, uint8)
        The numbers in each row correspond to the indices of occupied
        orbitals in each determinant, calculated from fci_c_utils.gen_orb_lists
    orb_symm : (numpy.ndarray, uint8)
        irreducible representation of each spatial orbital
    Returns
    -------
    (numpy.ndarray, uint8)
        2-D array with the 0th column as number of occupied orbitals
        that can be chosen for each determinant and the remaining columns
        as the indices of these allowed orbitals
    (numpy.ndarray, uint8)
        2-D array with number of virtual orbitals that can be chosen
        for each occupied orbital
    '''
    cdef size_t num_dets = occ_orbs.shape[0]
    cdef unsigned int num_elec = occ_orbs.shape[1]
    cdef unsigned int num_orb = orb_symm.shape[0]
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] occ_allow = numpy.zeros([num_dets, num_elec + 1], dtype=numpy.uint8)
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] n_virt = numpy.zeros([num_dets, num_elec], dtype=numpy.uint8)
    cdef size_t det_idx
    cdef unsigned int elec_idx, num_allowed, symm_allowed
    cdef unsigned char occ_symm

    for det_idx in range(num_dets):
        num_allowed = 0
        for elec_idx in range(num_elec):
            occ_symm = orb_symm[occ_orbs[det_idx, elec_idx] % num_orb]
            symm_allowed = virt_counts[det_idx, occ_symm, elec_idx / (num_elec / 2)]
            n_virt[det_idx, elec_idx] = symm_allowed
            if symm_allowed > 0:
                num_allowed += 1
                occ_allow[det_idx, num_allowed] = elec_idx
        occ_allow[det_idx, 0] = num_allowed
    return occ_allow, n_virt


def symm_pair_wt(unsigned int[:, :, :] virt_counts, unsigned char[:, :] occ_orbs,
                 unsigned char[:] orb_symm, unsigned int[:] det_idx,
                 unsigned int[:] opair_tri):
    """Calculate the weight of each virtual irrep pair, and number of choices
    within each irrep pair, for each of an array of choices of occupied pairs.

    Parameters
    ----------
    virt_counts : (numpy.ndarray, uint32)
        The number of virtual orbitals with each irrep in each determinant in
        the solution vector
    occ_orbs : (numpy.ndarray, uint8)
        The numbers in each row correspond to the indices of occupied
        orbitals in each determinant, calculated from fci_c_utils.gen_orb_lists
    orb_symm : (numpy.ndarray, uint8)
        irreducible representation of each spatial orbital
    det_idx : (numpy.ndarray, uint32)
        The index of the determinant of each of the occupied pair choices in
        the solution vector.
    opair_tri : (numpy.ndarray, uint32)
        The triangle index of each chosen occupied pair

    Returns
    -------
    (numpy.ndarray, float64)
        2-D array of weights of each virtual irrep pair for each occupied pair.
    (numpy.ndarray, uint8)
        2-D array of number of choices within each virtual irrep pair for each
        occupied pair
    (numpy.ndarray, uint8)
        orbitals of each occupied pair
    """
    cdef size_t n_samples = opair_tri.shape[0]
    cdef unsigned int num_orb = orb_symm.shape[0]
    cdef unsigned int n_symm = virt_counts.shape[1]
    cdef unsigned int num_elec = occ_orbs.shape[1]
    cdef size_t sampl_idx
    cdef unsigned char[4][8] xor_idx
    xor_idx[0][:] = [0, 1, 2, 3, 4, 5, 6, 7]
    xor_idx[1][:] = [1, 3, 5, 7, 0, 0, 0, 0]
    xor_idx[2][:] = [2, 3, 6, 7, 0, 0, 0, 0]
    xor_idx[3][:] = [4, 5, 6, 7, 0, 0, 0, 0]
    cdef orb_pair occ
    cdef unsigned int curr_det, n_symm1, n_symm2, sym_prod, symm_idx
    cdef unsigned int tot_virt
    cdef unsigned int num_symm_pair, xor_row_idx, occ_same
    cdef numpy.ndarray[numpy.float64_t, ndim=2] irrep_weights = numpy.zeros([n_samples, n_symm])
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] irrep_cts = numpy.zeros([n_samples, n_symm], dtype=numpy.uint8)
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] doub_orbs = numpy.zeros([n_samples, 2], dtype=numpy.uint8)
    cdef unsigned int m_a_allow

    for sampl_idx in range(n_samples):
        curr_det = det_idx[sampl_idx]
        occ = _symm_pair_wt(&occ_orbs[curr_det, 0], num_elec, opair_tri[sampl_idx],
                            orb_symm, < unsigned int (*)[2] > &virt_counts[curr_det, 0, 0], xor_idx, n_symm,
                            &irrep_weights[sampl_idx, 0], &irrep_cts[sampl_idx, 0])
        doub_orbs[sampl_idx, 1] = occ.orb1
        doub_orbs[sampl_idx, 0] = occ.orb2
    return irrep_weights, irrep_cts, doub_orbs


cdef orb_pair _symm_pair_wt(unsigned char *occ_orbs, unsigned int n_elec, unsigned int tri_idx,
                            unsigned char[:] orb_symm, unsigned int v_counts[][2],
                            unsigned char xor_idx[][8], unsigned int n_symm,
                            double *ret_weights, unsigned char *ret_counts) nogil:
    # Calculate weights for all pairs of irreps whose symmetry product is equal to that of a pair of specified occupied orbitals
    cdef orb_pair occ = _tri_to_occ_pair(occ_orbs, n_elec, tri_idx)
    cdef unsigned int num_orb = orb_symm.shape[0]
    cdef unsigned int sym_prod = orb_symm[occ.orb1 % num_orb] ^ orb_symm[occ.orb2 % num_orb]
    cdef unsigned int m_a_allow = _count_doub_virt(occ, orb_symm, n_elec, v_counts, n_symm)
    if m_a_allow == 0:
        occ.orb1 = 0
        occ.orb2 = 0
        return occ
    cdef unsigned int num_symm_pair, xor_row_idx
    cdef unsigned int symm_idx, n_symm1, n_symm2

    # Get pointer to list to use for enumerating symmetry products
    if ((occ.spin1 != occ.spin2) or sym_prod == 0):
        num_symm_pair = n_symm
        xor_row_idx = 0
    else:
        num_symm_pair = n_symm / 2
        if (sym_prod == 1):
            xor_row_idx = 1
        elif (sym_prod == 2 or sym_prod == 3):
            xor_row_idx = 2
        else:
            xor_row_idx = 3

    if sym_prod == 0 and (occ.spin1 == occ.spin2):
        for symm_idx in range(num_symm_pair):
            n_symm1 = v_counts[xor_idx[xor_row_idx][symm_idx]][occ.spin1]
            if n_symm1 > 1:
                ret_weights[symm_idx] = 1. * n_symm1 / m_a_allow
                ret_counts[symm_idx] = n_symm1 * (n_symm1 - 1) / 2
            else:
                ret_weights[symm_idx] = 0
        for symm_idx in range(num_symm_pair, n_symm):
            ret_weights[symm_idx] = 0
    else:
        for symm_idx in range(num_symm_pair):
            n_symm1 = v_counts[xor_idx[xor_row_idx][symm_idx]][occ.spin1]
            n_symm2 = v_counts[sym_prod ^ xor_idx[xor_row_idx][symm_idx]][occ.spin2]
            if n_symm1 != 0 and n_symm2 != 0:
                ret_weights[symm_idx] = 1. * (n_symm1 + n_symm2) / m_a_allow
                ret_counts[symm_idx] = n_symm1 * n_symm2
            else:
                ret_weights[symm_idx] = 0
        for symm_idx in range(num_symm_pair, n_symm):
            ret_weights[symm_idx] = 0
    return occ


def sing_multin(long long[:] dets, unsigned char[:, :] occ_orbs,
                unsigned char[:] orb_symm, unsigned char[:, :] lookup_tabl,
                unsigned int[:] num_sampl, unsigned long[:] mt_ptrs):
    ''' Uniformly chooses num_sampl[i] single excitations for each determinant
        dets[i] according to the symmetry-adapted rules described in Sec. 5.1 of
        Booth et al. (2014).

    Parameters
    ----------
    dets : (numpy.ndarray, int64)
        Bit string representations of all determinants
    occ_orbs : (numpy.ndarray, uint8)
        The numbers in each row correspond to the indices of occupied
        orbitals in each determinant, calculated from fci_c_utils.gen_orb_lists
    orb_symm : (numpy.ndarray, uint8)
        irreducible representation of each spatial orbital
    lookup_tabl : (numpy.ndarray, uint8)
        Table of orbitals with each type of symmetry, as generated
        by fci_utils.gen_byte_table()
    num_sampl : (numpy.ndarray, uint64)
        number of single excitations to choose for each determinant
    mt_ptrs : (numpy.ndarray, uint64)
        List of addresses to MT state objects to use for RN generation

    Returns
    -------
    (numpy.ndarray, uint8) :
        chosen occupied (0th column) and unoccupied (1st column) orbitals
    (numpy.ndarray, float64) :
        probability of each choice
    (numpy.ndarray, uint32) :
        index of origin determinant of each choice in the dets array
    '''

    cdef size_t num_dets = occ_orbs.shape[0]
    cdef unsigned int num_elec = occ_orbs.shape[1]
    cdef unsigned int num_orb = orb_symm.shape[0]
    cdef size_t det_idx
    cdef unsigned int j, delta_s, num_allowed
    cdef unsigned int occ_orb, occ_symm, occ_spin, virt_orb, sampl_idx
    cdef unsigned int elec_idx
    cdef unsigned int tot_sampl = 0
    cdef long long curr_det
    cdef numpy.ndarray[numpy.uint32_t] start_idx = numpy.zeros(num_dets, dtype=numpy.uint32)
    cdef unsigned int thread_idx
    cdef unsigned int n_threads = mt_ptrs.shape[0]
    cdef mt_struct * mts
    cdef unsigned int[:, :] m_allow = numpy.zeros([n_threads, num_elec], dtype=numpy.uint32)
    cdef unsigned int n_symm = lookup_tabl.shape[0]
    cdef unsigned int[:, :, :] unocc_sym_counts = numpy.zeros([n_threads, n_symm, 2], dtype=numpy.uint32)

    for det_idx in range(num_dets):
        start_idx[det_idx] = tot_sampl
        tot_sampl += num_sampl[det_idx]

    cdef numpy.ndarray[numpy.uint8_t, ndim=2] chosen_orbs = numpy.zeros([tot_sampl, 2],
                                                                        dtype=numpy.uint8)
    cdef numpy.ndarray[numpy.float64_t] prob_vec = numpy.zeros(tot_sampl)
    cdef numpy.ndarray[numpy.uint32_t] idx_arr = numpy.zeros(tot_sampl, dtype=numpy.uint32)

    for det_idx in prange(num_dets, nogil=True, schedule=static, num_threads=n_threads):
        if (num_sampl[det_idx] == 0):
            continue
        thread_idx = threadid()
        mts = <mt_struct *> mt_ptrs[thread_idx]
        curr_det = dets[det_idx]
        sampl_idx = start_idx[det_idx]

        _count_symm_virt(< unsigned int (*)[2]> &unocc_sym_counts[thread_idx, 0, 0], &occ_orbs[det_idx, 0],
                         num_elec, lookup_tabl, orb_symm)
        delta_s = 0  # number of electrons with no symmetry-allowed excitations
        for elec_idx in range(num_elec):
            occ_symm = orb_symm[occ_orbs[det_idx, elec_idx] % num_orb]
            num_allowed = unocc_sym_counts[thread_idx, occ_symm, elec_idx / (num_elec / 2)]
            m_allow[thread_idx, elec_idx] = num_allowed
            if num_allowed == 0:
                delta_s = delta_s + 1

        for j in range(num_sampl[det_idx]):
            if (delta_s == num_elec):
                prob_vec[sampl_idx + j] = -1.
                continue
            elec_idx = _sing_choose_occ(m_allow[thread_idx], mts)
            occ_orb = occ_orbs[det_idx, elec_idx]
            occ_symm = orb_symm[occ_orb % num_orb]
            occ_spin = occ_orb / num_orb

            virt_orb = _sing_choose_virt(curr_det, lookup_tabl[occ_symm],
                                         occ_spin * num_orb, mts)

            prob_vec[sampl_idx + j] = (1. / m_allow[thread_idx, elec_idx] /
                                       (num_elec - delta_s))
            chosen_orbs[sampl_idx + j, 0] = occ_orb
            chosen_orbs[sampl_idx + j, 1] = virt_orb
            idx_arr[sampl_idx + j] = det_idx

    successes = prob_vec > 0
    chosen_orbs = chosen_orbs[successes]
    prob_vec = prob_vec[successes]
    idx_arr = idx_arr[successes]
    return chosen_orbs, prob_vec, idx_arr


def fri_parallel(long long[:] dets, unsigned char[:, :] occ_orbs,
                unsigned char[:] orb_symm, unsigned char[:, :] lookup_tabl,
                unsigned int[:] num_sampl, unsigned long[:] mt_ptrs,
                double doub_prob):
    ''' Compress each column of the near-uniform excitation weight matrix
        independently using systematic compression.

    Parameters
    ----------
    dets : (numpy.ndarray, int64)
        Bit string representations of all determinants
    occ_orbs : (numpy.ndarray, uint8)
        The numbers in each row correspond to the indices of occupied
            orbitals in each determinant, calculated from fci_c_utils.gen_orb_lists
    orb_symm : (numpy.ndarray, uint8)
        irreducible representation of each spatial orbital
    lookup_tabl : (numpy.ndarray, uint8)
        Table of orbitals with each type of symmetry, as generated
        by fci_utils.gen_symm_lookup()
    num_sampl : (numpy.ndarray, uint32)
        number of nonzero entries to keep in compression of each column
    mt_ptrs : (numpy.ndarray, uint64)
        List of addresses to MT state objects to use for RN generation
    doub_prob : (double)
        The probability of choosing a double vs. single excitation from the
        definition of the near-uniform excitation weight matrix.

    Returns
    -------
    (numpy.ndarray, uint8) :
        chosen occupied (0th and 1st columns) and unoccupied (2nd and 3rd
        columns) orbitals for each double excitation
    (numpy.ndarray, float64) :
        Values of excitation weight matrix elements divided by their
        compressed values, for double excitations
    (numpy.ndarray, uint32) :
        index in the dets array of the origin determinant for each double
        excitation
    (numpy.ndarray, uint8) :
        chosen occupied (0th column) and unoccupied (1st column) orbitals
        for each single excitation
    (numpy.ndarray, float64) :
        Values of excitation weight matrix elements divided by their
        compressed values, for single excitations
    (numpy.ndarray, uint32) :
        index in the dets array of the origin determinant for each single
        excitation
    '''
    cdef size_t num_dets = occ_orbs.shape[0]
    cdef unsigned int num_elec = occ_orbs.shape[1]
    cdef unsigned int num_orb = orb_symm.shape[0]
    cdef size_t det_idx
    cdef unsigned int j
    cdef unsigned int occ_orb, occ_symm, sample_idx
    cdef unsigned int elec_idx
    cdef unsigned int tot_sampl = 0
    cdef long long curr_det
    cdef numpy.ndarray[numpy.uint32_t] start_idx = numpy.zeros(num_dets, dtype=numpy.uint32)
    cdef unsigned int thread_idx
    cdef unsigned int n_threads = mt_ptrs.shape[0]
    cdef mt_struct * mts
    cdef unsigned int[:, :, :] m_allow = numpy.zeros([n_threads, num_elec, 2], dtype=numpy.uint32)
    cdef unsigned int n_sing_allow, num_allowed
    cdef unsigned int n_symm = lookup_tabl.shape[0]
    cdef unsigned int[:, :, :] unocc_sym_counts = numpy.zeros([n_threads, n_symm, 2], dtype=numpy.uint32)
    cdef unsigned int max_samples = 0
    cdef orb_pair occ
    cdef double loc_p_doub

    cdef unsigned char[4][8] xor_idx
    xor_idx[0][:] = [0, 1, 2, 3, 4, 5, 6, 7]
    xor_idx[1][:] = [1, 3, 5, 7, 0, 0, 0, 0]
    xor_idx[2][:] = [2, 3, 6, 7, 0, 0, 0, 0]
    xor_idx[3][:] = [4, 5, 6, 7, 0, 0, 0, 0]

    for det_idx in range(num_dets):
        start_idx[det_idx] = tot_sampl
        j = num_sampl[det_idx]
        if j > max_samples:
            max_samples = j
        tot_sampl += j

    cdef numpy.ndarray[numpy.uint8_t, ndim=2] doub_orbs = numpy.zeros([tot_sampl, 4],
                                                                      dtype=numpy.uint8)
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] sing_orbs = numpy.zeros([tot_sampl, 2],
                                                                      dtype=numpy.uint8)
    cdef numpy.ndarray[numpy.float64_t] sing_probs = numpy.zeros(tot_sampl)
    cdef numpy.ndarray[numpy.uint32_t] sing_det_idx = numpy.zeros(tot_sampl, dtype=numpy.uint32)
    cdef numpy.ndarray[numpy.float64_t] doub_probs = numpy.zeros(tot_sampl)
    cdef numpy.ndarray[numpy.uint32_t] doub_det_idx = numpy.zeros(tot_sampl, dtype=numpy.uint32)

    cdef unsigned char[:, :, :, :] samples = numpy.zeros([n_threads, 2, max_samples, 4], dtype=numpy.uint8)
    cdef double[:, :, :] fri_weights = numpy.zeros([n_threads, 2, max_samples], dtype=numpy.float64)
    cdef unsigned int[:, :, :] fri_idx = numpy.zeros([n_threads, max_samples, 2], dtype=numpy.uint32)
    cdef unsigned int[:, :] fri_counts = numpy.zeros([n_threads, max_samples], dtype=numpy.uint32)
    cdef double[:, :, :, :] fri_sub_weights = numpy.zeros([n_threads, 2, max_samples, n_symm], dtype=numpy.float64)
    cdef double[:, :] fri_probs = numpy.zeros([n_threads, max_samples], dtype=numpy.float64)
    cdef unsigned char[:, :, :] doub_orb_cts = numpy.zeros([n_threads, max_samples, n_symm], dtype=numpy.uint8)
    cdef unsigned int curr_n_samp, n_sing, n_doub, samp_idx
    cdef unsigned char *samp_ptr
    cdef int renorm_fri, ret_val
    cdef double one_norm

    #for det_idx in prange(num_dets, nogil=True, schedule=static, num_threads=n_threads):
    for det_idx in range(num_dets):
        if (num_sampl[det_idx] < 2):
            continue
        thread_idx = threadid()
        mts = < mt_struct * > mt_ptrs[thread_idx]
        curr_det = dets[det_idx]
        sample_idx = start_idx[det_idx]
        _count_symm_virt(< unsigned int (*)[2]> &unocc_sym_counts[thread_idx, 0, 0], &occ_orbs[det_idx, 0],
                         num_elec, lookup_tabl, orb_symm)
        n_sing_allow = 0  # number of electrons with symmetry-allowed single excitations
        for elec_idx in range(num_elec):
            occ_orb = occ_orbs[det_idx, elec_idx]
            occ_symm = orb_symm[occ_orb % num_orb]
            num_allowed = unocc_sym_counts[thread_idx, occ_symm, elec_idx / (num_elec / 2)]
            if num_allowed > 0:
                m_allow[thread_idx, n_sing_allow, 0] = occ_orb
                m_allow[thread_idx, n_sing_allow, 1] = num_allowed
                n_sing_allow = n_sing_allow + 1

        # select occ orbitals (singles) and occ pairs (doubles)
        if n_sing_allow > 0:
            loc_p_doub = doub_prob
            fri_counts[thread_idx, 0] = n_sing_allow
        else:
            loc_p_doub = 1
            fri_counts[thread_idx, 0] = 1
        fri_weights[thread_idx, 0, 0] = 1 - loc_p_doub
        fri_weights[thread_idx, 0, 1] = loc_p_doub
        fri_counts[thread_idx, 1] = num_elec * (num_elec - 1) / 2
        curr_n_samp = _fri_subd(&fri_weights[thread_idx, 0, 0], &fri_counts[thread_idx, 0], NULL,
                               2, 0, 0, < unsigned int (*)[2] > &fri_idx[thread_idx, 0, 0],
                               &fri_weights[thread_idx, 1, 0], num_sampl[det_idx], < mt_struct * > mts)

        # select unocc (singles) and symm pairs (doubles)
        n_sing = 0
        for samp_idx in range(curr_n_samp):
            if fri_idx[thread_idx, samp_idx, 0] == 0: # single
                elec_idx = fri_idx[thread_idx, samp_idx, 1]
                # samples[thread_idx, 0, n_sing, 0] = m_allow[thread_idx, elec_idx, 0] # orbital
                samples[thread_idx, 0, n_sing, 0] = elec_idx
                fri_counts[thread_idx, n_sing] = m_allow[thread_idx, elec_idx, 1] # num unocc
                fri_weights[thread_idx, 0, n_sing] = fri_weights[thread_idx, 1, samp_idx]
                n_sing = n_sing + 1
        n_doub = 0
        for samp_idx in range(curr_n_samp):
            if fri_idx[thread_idx, samp_idx, 0] == 1: # double
                elec_idx = fri_idx[thread_idx, samp_idx, 1]
                occ = _symm_pair_wt(&occ_orbs[det_idx, 0], num_elec, elec_idx,
                                    orb_symm, < unsigned int (*)[2] > &unocc_sym_counts[thread_idx, 0, 0], xor_idx, n_symm, 
                                    &fri_sub_weights[thread_idx, 0, n_sing + n_doub, 0], &doub_orb_cts[thread_idx, n_sing + n_doub, 0])
                if occ.orb1 == 0 and occ.orb2 == 0:
                    fri_weights[thread_idx, 0, n_sing + n_doub] = 0
                    renorm_fri = 1
                    n_doub = n_doub + 1
                    continue
                for j in range(n_symm):
                    fri_sub_weights[thread_idx, 1, n_sing + n_doub, j] = fri_sub_weights[thread_idx, 0, n_sing + n_doub, j]
                samples[thread_idx, 0, n_sing + n_doub, 0] = occ.orb2
                samples[thread_idx, 0, n_sing + n_doub, 1] = occ.orb1
                fri_weights[thread_idx, 0, n_sing + n_doub] = fri_weights[thread_idx, 1, samp_idx]
                n_doub = n_doub + 1

        if renorm_fri:
            renorm_fri = 0
            one_norm = 0
            for j in range(n_doub + n_sing):
                one_norm = one_norm + fri_weights[thread_idx, 0, j]
            for j in range(n_doub + n_sing):
                fri_weights[thread_idx, 0, j] = fri_weights[thread_idx, 0, j] / one_norm

        curr_n_samp = _fri_subd(&fri_weights[thread_idx, 0, 0], &fri_counts[thread_idx, 0], &fri_sub_weights[thread_idx, 0, n_sing, 0],
                               n_sing, n_doub, n_symm, < unsigned int (*)[2] > &fri_idx[thread_idx, 0, 0], &fri_weights[thread_idx, 1, 0],
                               num_sampl[det_idx], < mt_struct * > mts)

        # select virtual orbital pair (doubles)
        samp_idx = 0
        j = fri_idx[thread_idx, samp_idx, 0]
        while j < n_sing and samp_idx < curr_n_samp: # singles
            samples[thread_idx, 1, samp_idx, 0] = samples[thread_idx, 0, j, 0]
            samples[thread_idx, 1, samp_idx, 1] = fri_idx[thread_idx, samp_idx, 1]
            fri_counts[thread_idx, samp_idx] = 1
            samp_idx = samp_idx + 1
            j = fri_idx[thread_idx, samp_idx, 0]
        n_sing = samp_idx
        while samp_idx < curr_n_samp:
            j = fri_idx[thread_idx, samp_idx, 0]
            samples[thread_idx, 1, samp_idx, 0] = samples[thread_idx, 0, j, 0]
            samples[thread_idx, 1, samp_idx, 1] = samples[thread_idx, 0, j, 1]
            samples[thread_idx, 1, samp_idx, 2] = fri_idx[thread_idx, samp_idx, 1]
            fri_counts[thread_idx, samp_idx] = doub_orb_cts[thread_idx, j, fri_idx[thread_idx, samp_idx, 1]]
            fri_probs[thread_idx, samp_idx] = fri_sub_weights[thread_idx, 1, j, fri_idx[thread_idx, samp_idx, 1]]
            ret_val = _find_symm_virt(curr_det, lookup_tabl, &orb_symm[0], &samples[thread_idx, 1, samp_idx, 0],
                            samples[thread_idx, 1, samp_idx, 2], 0, < unsigned int (*)[2]> &unocc_sym_counts[thread_idx, 0, 0], num_orb,
                            &doub_orbs[sample_idx, 2], xor_idx)
            if ret_val == -1:
                printf('prob: %lf, fri_value: %lf, old fri: %lf, old sub: %lf, n_samp: %u\n', fri_probs[thread_idx, samp_idx], fri_weights[thread_idx, 1, samp_idx], fri_weights[thread_idx, 0, j], fri_sub_weights[thread_idx, 0, j, fri_idx[thread_idx, samp_idx, 1]], curr_n_samp)

                elec_idx = samples[thread_idx, 1, samp_idx, 1] * (samples[thread_idx, 1, samp_idx, 1] - 1) / 2 + samples[thread_idx, 1, samp_idx, 0]
                occ = _symm_pair_wt(&occ_orbs[det_idx, 0], num_elec, elec_idx,
                            orb_symm, < unsigned int (*)[2] > &unocc_sym_counts[thread_idx, 0, 0], xor_idx, n_symm,
                            &fri_sub_weights[thread_idx, 0, j, 0], &doub_orb_cts[thread_idx, j, 0])
                if (occ.orb2 != samples[thread_idx, 0, n_sing + n_doub, 0] or occ.orb1 != samples[thread_idx, 0, n_sing + n_doub, 1]):
                    printf('conversion messed up\n')
                printf('recalculated weight is: %lf\n', fri_sub_weights[thread_idx, 0, j, fri_idx[thread_idx, samp_idx, 1]])
                raise ValueError('early div by 0')
            samp_idx = samp_idx + 1

        curr_n_samp = _fri_subd(&fri_weights[thread_idx, 1, 0], &fri_counts[thread_idx, 0], NULL,
                               curr_n_samp, 0, 0, < unsigned int (*)[2] > &fri_idx[thread_idx, 0, 0], &fri_weights[thread_idx, 0, 0],
                               num_sampl[det_idx], < mt_struct * > mt_ptrs[thread_idx])

        n_doub = 0
        for samp_idx in range(curr_n_samp):
            j = fri_idx[thread_idx, samp_idx, 0]
            if j < n_sing:
                sing_det_idx[sample_idx + samp_idx - n_doub] = det_idx
                occ_orb = m_allow[thread_idx, samples[thread_idx, 1, j, 0], 0]
                sing_orbs[sample_idx + samp_idx - n_doub, 0] = occ_orb
                occ_symm = orb_symm[occ_orb % num_orb]
                samp_ptr = &samples[thread_idx, 0, 0, 0]
                _list_virt_symm(curr_det, &lookup_tabl[occ_symm, 0], (occ_orb / num_orb) * num_orb, samp_ptr)
                sing_orbs[sample_idx + samp_idx - n_doub, 1] = samp_ptr[samples[thread_idx, 1, j, 1]]
                sing_probs[sample_idx + samp_idx - n_doub] = ((1 - doub_prob) / n_sing_allow / 
                                                              m_allow[thread_idx, samples[thread_idx, 1, j, 0], 1] / 
                                                              fri_weights[thread_idx, 0, samp_idx])
            else:
                doub_det_idx[sample_idx + n_doub] = det_idx
                doub_orbs[sample_idx + n_doub, 0] = samples[thread_idx, 1, j, 0]
                doub_orbs[sample_idx + n_doub, 1] = samples[thread_idx, 1, j, 1]
                ret_val = _find_symm_virt(curr_det, lookup_tabl, &orb_symm[0], &doub_orbs[sample_idx + n_doub, 0],
                                samples[thread_idx, 1, j, 2], fri_idx[thread_idx, samp_idx, 1],
                                < unsigned int (*)[2]> &unocc_sym_counts[thread_idx, 0, 0], num_orb, 
                                &doub_orbs[sample_idx + n_doub, 2], xor_idx)
                if ret_val == -1:
                    printf('det: %ld, occ1: %u, occ2: %u, symm: %u, counts: %u, virt: %u, weight: %lf\n', curr_det, doub_orbs[sample_idx + n_doub, 0], doub_orbs[sample_idx + n_doub, 1], samples[thread_idx, 1, j, 2], fri_counts[thread_idx, j], fri_idx[thread_idx, samp_idx, 1], fri_weights[thread_idx, 0, samp_idx])
                    raise ValueError('divide by zero')
                doub_probs[sample_idx + n_doub] = (loc_p_doub * 2 / num_elec / (num_elec - 1) * fri_probs[thread_idx, j]
                                                    / fri_counts[thread_idx, j] / fri_weights[thread_idx, 0, samp_idx])
                n_doub = n_doub + 1

    doub_non_null = doub_probs > 0
    sing_non_null = sing_probs > 0

    return doub_orbs[doub_non_null], doub_probs[doub_non_null], doub_det_idx[doub_non_null], sing_orbs[sing_non_null], sing_probs[sing_non_null], sing_det_idx[sing_non_null]


cdef unsigned int _fri_subd(double *weights, unsigned int *num_div, double *sub_weights,
                           size_t n_ct, size_t n_wt, size_t n_subwt, unsigned int new_idx[][2],
                           double *new_weights, unsigned int n_samp, mt_struct *mt_ptr) nogil:
    # perform systematic compression on a vector with elements subdivided further,
    # either uniformly (into num_div portions) or nonuniformly (as in sub_weights)
    # weights and sub_weights are modified in this function
    cdef double one_norm = 1
    cdef size_t wt_idx, subwt_idx
    cdef size_t front_samp_idx = 0
    cdef size_t back_samp_idx = n_samp
    cdef size_t local_n_samp = n_samp
    cdef int any_kept = 1
    cdef double curr_weight, curr_subwt
    cdef double rand_num, sub_rand

    while any_kept and fabs(one_norm) > 1e-10: # Decide which to preserve exactly
        any_kept = 0
        for wt_idx in range(n_ct):
            curr_weight = weights[wt_idx] / num_div[wt_idx]
            if curr_weight >= one_norm / local_n_samp:
                for subwt_idx in range(num_div[wt_idx]):
                    new_idx[front_samp_idx][0] = wt_idx
                    new_idx[front_samp_idx][1] = subwt_idx
                    new_weights[front_samp_idx] = curr_weight
                    front_samp_idx += 1
                local_n_samp -= num_div[wt_idx]
                one_norm -= weights[wt_idx]
                weights[wt_idx] = 0
                any_kept = 1
                break
        wt_idx = 0
        while wt_idx < n_wt and any_kept == 0:
            for subwt_idx in range(n_subwt):
                curr_weight = weights[wt_idx + n_ct] * sub_weights[wt_idx * n_subwt + subwt_idx]
                if curr_weight >= one_norm / local_n_samp:
                    new_idx[back_samp_idx - 1][0] =  wt_idx + n_ct
                    new_idx[back_samp_idx - 1][1] = subwt_idx
                    new_weights[back_samp_idx - 1] = curr_weight
                    back_samp_idx -= 1
                    local_n_samp -= 1
                    one_norm -= curr_weight
                    sub_weights[wt_idx * n_subwt + subwt_idx] = 0
                    any_kept = 1
                    break
            wt_idx = wt_idx + 1
    # renormalize sub_weights
    one_norm = 0
    for wt_idx in range(n_ct):
        one_norm += weights[wt_idx]
    for wt_idx in range(n_wt):
        curr_weight = 0
        for subwt_idx in range(n_subwt):
            curr_weight += sub_weights[wt_idx * n_subwt + subwt_idx]
        weights[wt_idx + n_ct] *= curr_weight
        one_norm += weights[wt_idx + n_ct]
        if curr_weight != 0:
            for subwt_idx in range(n_subwt):
                sub_weights[wt_idx * n_subwt + subwt_idx] /= curr_weight

    # Systematic resampling
    if fabs(one_norm) > 1e-10:
        rand_num = genrand_mt(mt_ptr) / RAND_MAX * one_norm / local_n_samp
        wt_idx = 1
        curr_weight = weights[0]
        while rand_num < one_norm:
            while curr_weight < rand_num:
                curr_weight += weights[wt_idx]
                wt_idx += 1
            wt_idx -= 1
            new_idx[front_samp_idx][0] = wt_idx
            sub_rand = (rand_num - curr_weight + weights[wt_idx]) / weights[wt_idx]

            if wt_idx < n_ct:
                new_idx[front_samp_idx][1] = < unsigned int > (sub_rand * num_div[wt_idx])
            else:
                curr_subwt = sub_weights[(wt_idx - n_ct) * n_subwt]
                subwt_idx = 1
                while curr_subwt <= sub_rand:
                    curr_subwt += sub_weights[(wt_idx - n_ct) * n_subwt + subwt_idx]
                    subwt_idx += 1
                new_idx[front_samp_idx][1] = subwt_idx - 1
            new_weights[front_samp_idx] = one_norm / local_n_samp
            front_samp_idx += 1
            rand_num += one_norm / local_n_samp
            wt_idx += 1
    if back_samp_idx > front_samp_idx:
        while back_samp_idx < n_samp:
            new_idx[front_samp_idx][0] = new_idx[back_samp_idx][0]
            new_idx[front_samp_idx][1] = new_idx[back_samp_idx][1]
            new_weights[front_samp_idx] = new_weights[back_samp_idx]
            front_samp_idx += 1
            back_samp_idx += 1
        return front_samp_idx
    else:
        return n_samp


cdef unsigned int _sing_choose_occ(unsigned int[:] counts, mt_struct *mt_ptr
                                   ) nogil:
    # Choose an occupied orbital with a nonzero number of symmetry-allowed
    # excitations
    cdef unsigned int elec_idx, n_allow = 0
    cdef unsigned int n_elec = counts.shape[0]
    # Rejection sampling
    while n_allow == 0:
        elec_idx = _choose_uint(mt_ptr, n_elec)
        n_allow = counts[elec_idx]
    return elec_idx


cdef unsigned int _sing_choose_virt(long long det, unsigned char[:] symm_row,
                                    unsigned int spin_shift, mt_struct *mt_ptr
                                    ) nogil:
    # Uniformly choose a virtual orbital with the specified symmetry
    cdef int symm_idx = -1
    cdef unsigned int orbital
    # Rejection sampling
    while (symm_idx == -1):
        symm_idx = _choose_uint(mt_ptr, symm_row[0])
        orbital = spin_shift + symm_row[symm_idx + 1]
        if det & (< long long > 1 << orbital):
            # orbital is occupied, choose again
            symm_idx = -1
    return orbital


def id_doub_virt(long long[:] dets, unsigned char[:, :] lookup_tabl,
                 unsigned char[:] orb_symm, unsigned char[:, :] occ_pairs,
                 unsigned int[:] irrep_idx, unsigned int[:] virt_idx,
                 unsigned int[:, :, :] virt_counts):
    '''Identify the virtual orbitals corresponding to the irrep pair indices
        and the virtual pair indices for double excitations.
    Parameters
    ----------
    dets : (numpy.ndarray, int64)
        Bit string representation of determinant corresponding to each choice
    lookup_tabl : (numpy.ndarray, uint8)
        Table of orbitals with each type of symmetry, as generated
        by fci_utils.gen_symm_lookup()
    occ_pairs : (numpy.ndarray, uint8)
        2-D array containing the occupied orbitals of each choice
    irrep_idx : (numpy.ndarray, uint32)
        Index of the irrep pair chosen for each excitation
    virt_idx : (numpy.ndarray, uint32)
        Index of the virtual orbital pair within each irrep pair
    virt_counts : (numpy.ndarray, uint32)
        The number of virtual orbitals with each irrep in each determinant in
        dets
    Returns
    -------
    (numpy.ndarray, uint8)
        The virtual orbitals identified for each excitation
    '''
    cdef size_t det_idx
    cdef size_t num_dets = dets.shape[0]
    cdef unsigned int n_orb = orb_symm.shape[0]
    cdef unsigned int n_symm = lookup_tabl.shape[0]
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] virt_orbs = numpy.zeros([num_dets, 2],  dtype=numpy.uint8)
    cdef long long curr_det
    cdef unsigned char[4][8] xor_idx
    xor_idx[0][:] = [0, 1, 2, 3, 4, 5, 6, 7]
    xor_idx[1][:] = [1, 3, 5, 7, 0, 0, 0, 0]
    xor_idx[2][:] = [2, 3, 6, 7, 0, 0, 0, 0]
    xor_idx[3][:] = [4, 5, 6, 7, 0, 0, 0, 0]
    cdef unsigned int symm_prod, occ_same, spin1, spin2, occ_orb1, occ_orb2
    cdef unsigned int xor_row_idx, num_symm_pair, v1, v2, spin_a, spin_b
    cdef unsigned char symm1, symm2

    for det_idx in range(num_dets):
        curr_det = dets[det_idx]
        _find_symm_virt(curr_det, lookup_tabl, &orb_symm[0], &occ_pairs[det_idx, 0],
                        irrep_idx[det_idx], virt_idx[det_idx], 
                        < unsigned int (*)[2] > &virt_counts[det_idx, 0, 0], n_orb, 
                        & virt_orbs[det_idx, 0], xor_idx)
    return virt_orbs

cdef int _find_symm_virt(long long curr_det, unsigned char[:, :] lookup_tabl, unsigned char *orb_symm,
                          unsigned char *occ_pair, unsigned int irrep_idx, unsigned int virt_idx,
                          unsigned int virt_counts[][2], unsigned int n_orb, unsigned char *virt_pair,
                          unsigned char xor_idx[][8]) nogil:
    cdef unsigned int n_symm = lookup_tabl.shape[0]
    cdef unsigned char occ_orb2 = occ_pair[0]
    cdef unsigned char occ_orb1 = occ_pair[1]
    cdef unsigned int spin1 = occ_orb1 / n_orb
    cdef unsigned int spin2 = occ_orb2 / n_orb
    cdef unsigned int symm_prod = orb_symm[occ_orb1 % n_orb] ^ orb_symm[occ_orb2 % n_orb]
    cdef unsigned int xor_row_idx, num_symm_pair, v1, v2, spin_a, spin_b
    cdef unsigned char symm1, symm2

    if ((spin1 != spin2) or symm_prod == 0):
        num_symm_pair = n_symm
        xor_row_idx = 0
    else:
        num_symm_pair = n_symm / 2
        if (symm_prod == 1):
            xor_row_idx = 1
        elif (symm_prod == 2 or symm_prod == 3):
            xor_row_idx = 2
        else:
            xor_row_idx = 3

    symm1 = xor_idx[xor_row_idx][irrep_idx]
    symm2 = symm_prod ^ symm1

    if symm_prod == 0 and (spin1 == spin2):
        v1 = <unsigned int > ((sqrt(virt_idx * 8. + 1) - 1) / 2)
        v2 = <unsigned int > (virt_idx - v1 * (v1 + 1.) / 2)
        v1 += 1  # v2 < v1

        # virt_pair[0] = _find_virt(curr_det, & lookup_tabl[symm1, 1],
        #                           spin1 * n_orb, v1)
        # virt_pair[1] = _find_virt(curr_det, & lookup_tabl[symm1, 1],
        #                           spin1 * n_orb, v2)
    else:
        xor_row_idx = virt_counts[symm1][spin1]
        if xor_row_idx == 0:
            return -1
        v1 = virt_idx % xor_row_idx
        v2 = virt_idx / xor_row_idx

        # virt_pair[0] = _find_virt(curr_det, & lookup_tabl[symm1, 1],
        #                           spin1 * n_orb, v1)
        # virt_pair[1] = _find_virt(curr_det, & lookup_tabl[symm2, 1],
        #                           spin2 * n_orb, v2)

    v1 = _find_virt(curr_det, & lookup_tabl[symm1, 1], spin1 * n_orb, v1)
    v2 = _find_virt(curr_det, & lookup_tabl[symm2, 1], spin2 * n_orb, v2)
    if v1 < v2:
        virt_pair[0] = v1
        virt_pair[1] = v2
    else:
        virt_pair[0] = v2
        virt_pair[1] = v1
    return 0


cdef unsigned int _find_virt(long long det, unsigned char * symm_row,
                             unsigned int spin_shift, unsigned int symm_idx) nogil:
    # Find the (symm_idx)th virtual orbital in det with a particular symmetry
    cdef unsigned int col_idx = 0
    cdef unsigned int symm_counter = 0
    cdef unsigned int orbital

    while symm_counter <= symm_idx:
        orbital = spin_shift + symm_row[col_idx]
        if not(det & (< long long > 1 << orbital)):
            symm_counter += 1
        col_idx += 1
    return orbital


def virt_symm_idx(long long[:] dets, unsigned char[:, :] lookup_tabl,
                  unsigned char[:] chosen_symm, unsigned char[:] spin_shifts):
    '''Build lists of all virtual orbitals in each of an array of determinants
    with a particular spin and spatial symmetry.
    Parameters
    ----------
    dets : (numpy.ndarray, int64)
        Bit string representation of each determinant
    lookup_tabl : (numpy.ndarray, uint8)
        Table of orbitals with each type of symmetry, as generated
        by fci_utils.gen_symm_lookup()
    chosen_symm : (numpy.ndarray, uint8)
        Desired spatial symmetry for each list of virtual orbitals
    spin_shifts : (numpy.ndarray, uint32)
        Desired spin symmetry (0 or 1) multiplied by number of spatial orbitals
    Returns
    -------
    (numpy.ndarray, uint8)
        2-D array with values in each row corresponding to the found virtual orbitals
        (spin included)
    '''
    cdef size_t det_idx
    cdef size_t num_dets = dets.shape[0]
    cdef unsigned int n_orb = lookup_tabl.shape[1]
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] virt_orbs = numpy.zeros([num_dets, n_orb], dtype=numpy.uint8)
    cdef long long curr_det
    cdef unsigned int curr_shift
    cdef unsigned int symm_idx
    cdef unsigned int symm_counter
    cdef unsigned char curr_symm
    cdef unsigned char orbital

    for det_idx in range(num_dets):
        curr_symm = chosen_symm[det_idx]
        curr_shift = spin_shifts[det_idx]
        _list_virt_symm(dets[det_idx], &lookup_tabl[curr_symm, 0], curr_shift, &virt_orbs[det_idx, 0])
        # curr_det = dets[det_idx]
        # symm_counter = 0
        # for symm_idx in range(lookup_tabl[curr_symm, 0]):
        #     orbital = curr_shift + lookup_tabl[curr_symm, symm_idx + 1]
        #     if not(curr_det & (< long long > 1 << orbital)):
        #         virt_orbs[det_idx, symm_counter] = orbital
        #         symm_counter += 1
    return virt_orbs

cdef void _list_virt_symm(long long curr_det, unsigned char *lookup_row,
                          unsigned char spin_shift, unsigned char *virt_orbs) nogil:
    # enumerate virtual orbitals with specified irrep
    cdef unsigned int symm_counter = 0
    cdef unsigned int symm_idx
    cdef unsigned char orbital
    for symm_idx in range(lookup_row[0]):
        orbital = spin_shift + lookup_row[symm_idx + 1]
        if not(curr_det & (< long long > 1 << orbital)):
            virt_orbs[symm_counter] = orbital
            symm_counter += 1


def virt_symm_bool(long long[:] dets, unsigned char[:, :] lookup_tabl,
                   unsigned char[:] chosen_symm, unsigned char[:] spin_shifts,
                   unsigned int n_orb):
    '''Calculate whether each of the spatial orbitals in a HF basis are unoccupied
    in each of a set of determinants, and whether they have a particular spatial
    symmetry.
    Parameters
    ----------
    dets : (numpy.ndarray, int64)
        Bit string representation of each determinant
    lookup_tabl : (numpy.ndarray, uint8)
        Table of orbitals with each type of symmetry, as generated
        by fci_utils.gen_symm_lookup()
    chosen_symm : (numpy.ndarray, uint8)
        Irrep of spatial symmetry to test for the orbitals for each determinant
    spin_shifts : (numpy.ndarray, uint32)
        Spin symmetry (0 or 1) to test for each determinant, multiplied by number
        of spatial orbitals
    n_orb : (unsigned int)
        Number of orbitals in the spatial HF basis
    Returns
    -------
    (numpy.ndarray, bool)
        2-D array indicating whether each spatial orbital is unoccupied and conforms
        to the spatial and spin symmetries specified in the arguments.
    '''
    cdef size_t det_idx
    cdef size_t num_dets = dets.shape[0]
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] virt_orbs = numpy.zeros([num_dets, n_orb], dtype=numpy.uint8)
    cdef long long curr_det
    cdef unsigned int curr_shift
    cdef unsigned int symm_idx
    cdef unsigned char curr_symm
    cdef unsigned char orbital

    for det_idx in range(num_dets):
        curr_symm = chosen_symm[det_idx]
        curr_det = dets[det_idx]
        curr_shift = spin_shifts[det_idx]
        for symm_idx in range(lookup_tabl[curr_symm, 0]):
            orbital = lookup_tabl[curr_symm, symm_idx + 1]
            virt_orbs[det_idx, orbital] = not(curr_det & (< long long > 1 << (orbital + curr_shift)))
    return virt_orbs.astype(numpy.bool_)


cdef void _count_symm_virt(unsigned int counts[][2], unsigned char *occ_orbs,
                           unsigned int n_elec, unsigned char[:, :] symm_table,
                           unsigned char[:] orb_irreps) nogil:
    # Count number of unoccupied orbitals with each irrep
    cdef unsigned int i
    cdef unsigned int n_symm = symm_table.shape[0]
    # cdef unsigned int n_elec = occ_orbs.shape[0]
    cdef unsigned int n_orb = orb_irreps.shape[0]

    for i in range(n_symm):
        counts[i][0] = symm_table[i, 0]
        counts[i][1] = symm_table[i, 0]
    for i in range(n_elec / 2):
        counts[orb_irreps[occ_orbs[i]]][0] -= 1
    for i in range(n_elec / 2, n_elec):
        counts[orb_irreps[occ_orbs[i] - n_orb]][1] -= 1


cdef orb_pair _choose_occ_pair(unsigned char *occ_orbs, unsigned int num_elec, mt_struct *mt_ptr
                               ) nogil:
    # Randomly & uniformly choose a pair of occupied orbitals
    # cdef unsigned int num_elec = occ_orbs.shape[0]
    cdef unsigned int rand_pair = _choose_uint(mt_ptr, num_elec * (num_elec - 1)
                                               / 2)
    return _tri_to_occ_pair(occ_orbs, num_elec, rand_pair)


cdef orb_pair _tri_to_occ_pair(unsigned char *occ_orbs, unsigned int num_elec, unsigned int tri_idx) nogil:
    # Use triangle inversion to convert an electron pair index into an orb_pair object
    cdef orb_pair pair
    cdef unsigned int orb_idx1 = < unsigned int > ((sqrt(tri_idx * 8. + 1) - 1) / 2)
    cdef unsigned int orb_idx2 = < unsigned int > (tri_idx - orb_idx1 * (orb_idx1
                                                                       + 1.) / 2)
    orb_idx1 += 1
    pair.orb1 = occ_orbs[orb_idx1]
    pair.orb2 = occ_orbs[orb_idx2]
    pair.spin1 = orb_idx1 / (num_elec / 2)
    pair.spin2 = orb_idx2 / (num_elec / 2)
    return pair


cdef unsigned int _count_doub_virt(orb_pair occ, unsigned char[:] orb_irreps,
                                   unsigned int num_elec,
                                   unsigned int virt_counts[][2],
                                   unsigned int n_symm) nogil:
    # Count number of spin- and symmetry-allowed unoccupied orbitals
    # given a chosen pair of occupied orbitals
    cdef unsigned int n_allow
    cdef unsigned int n_orb = orb_irreps.shape[0]
    cdef unsigned char sym_prod = (orb_irreps[occ.orb1 % n_orb] ^
                                   orb_irreps[occ.orb2 % n_orb])
    cdef int same_symm = sym_prod == 0 and occ.spin1 == occ.spin2
    cdef unsigned int i

    if occ.spin1 == occ.spin2:
        n_allow = n_orb - num_elec / 2
    else:
        n_allow = 2 * n_orb - num_elec

    for i in range(n_symm):
        if (virt_counts[i ^ sym_prod][occ.spin2] == same_symm):
            n_allow -= virt_counts[i][occ.spin1]
        if (occ.spin1 != occ.spin2 and virt_counts[i ^ sym_prod][occ.spin1]
                == same_symm):
            n_allow -= virt_counts[i][occ.spin2]
    return n_allow


cdef unsigned int _doub_choose_virt1(orb_pair occ, long long det,
                                     unsigned char[:] irreps,
                                     unsigned int virt_counts[][2],
                                     unsigned int num_allowed,
                                     mt_struct *mt_ptr) nogil:
    # Choose the first virtual orbital for a double excitation uniformly from
    # among the allowed orbitals
    cdef int virt_choice
    cdef unsigned int orbital
    cdef unsigned int a_spin, b_spin
    cdef unsigned int n_orb = irreps.shape[0]
    cdef unsigned char sym_prod = (irreps[occ.orb1 % n_orb] ^
                                   irreps[occ.orb2 % n_orb])
    cdef unsigned int n_virt2, nv2
    cdef unsigned int a_symm

    if num_allowed <= 3:  # choose the orbital index and then search for it
        virt_choice = _choose_uint(mt_ptr, num_allowed)

        if (occ.spin1 == occ.spin2):
            a_spin = occ.spin1
            b_spin = a_spin
        else:
            a_spin = 0
            b_spin = 1
        # begin search for virtual orbital
        orbital = 0
        while (virt_choice >= 0 and orbital < n_orb):
            # check that this orbital is unoccupied
            if (not(det & (< long long > 1 << (orbital + a_spin * n_orb)))):
                a_symm = irreps[orbital]
                n_virt2 = (virt_counts[sym_prod ^ a_symm][b_spin] -
                           (sym_prod == 0 and a_spin == b_spin))
                if n_virt2 != 0:
                    virt_choice -= 1
            orbital += 1
        if (virt_choice >= 0):
            # Different spins and orbital not found; keep searching
            a_spin = 1
            b_spin = 0
            while (virt_choice >= 0 and orbital < 2 * n_orb):
                # check that this orbital is unoccupied
                if (not(det & (< long long > 1 << orbital))):
                    a_symm = irreps[orbital - n_orb]
                    n_virt2 = (virt_counts[sym_prod ^ a_symm][b_spin] -
                               (sym_prod == 0 and a_spin == b_spin))
                    if n_virt2 != 0:
                        virt_choice -= 1
                orbital += 1
            orbital -= n_orb
        virt_choice = orbital - 1 + a_spin * n_orb
    else:  # choose the orbital by rejection
        n_virt2 = 0
        while (n_virt2 == 0):
            if (occ.spin1 == occ.spin2):
                a_spin = occ.spin1
                b_spin = a_spin
                virt_choice = _choose_uint(mt_ptr, n_orb) + a_spin * n_orb
            else:
                virt_choice = _choose_uint(mt_ptr, 2 * n_orb)
                a_spin = virt_choice / n_orb
                b_spin = 1 - a_spin
            if not(det & (< long long> 1 << virt_choice)):  # check if unoccupied
                a_symm = irreps[virt_choice % n_orb]
                n_virt2 = (virt_counts[sym_prod ^ a_symm][b_spin] -
                           (sym_prod == 0 and a_spin == b_spin))
    return virt_choice


cdef unsigned int _doub_choose_virt2(unsigned int spin_shift, long long det,
                                     unsigned char *symm_row,
                                     unsigned int virt1, unsigned int n_allow,
                                     mt_struct *mt_ptr) nogil:
    # Choose the second virtual orbial uniformly
    cdef int orb_idx = _choose_uint(mt_ptr, n_allow)
    cdef unsigned int orbital
    cdef unsigned int symm_idx = 1
    # Search for chosen orbital
    while (orb_idx >= 0):
        orbital = symm_row[symm_idx] + spin_shift
        if (not(det & (< long long > 1 << orbital)) and orbital != virt1):
            orb_idx -= 1
        symm_idx += 1
    return orbital


cdef unsigned int _choose_uint(mt_struct *mt_ptr, unsigned int nmax) nogil:
    # Choose an integer uniformly on the interval [0, nmax)
    return < unsigned int > (genrand_mt(mt_ptr) / RAND_MAX * nmax)


def par_bernoulli(double[:] p, unsigned long[:] mt_ptrs):
    cdef size_t n_samp = p.shape[0]
    cdef size_t samp_idx
    cdef unsigned int n_threads = mt_ptrs.shape[0]
    cdef mt_struct *mt_ptr
    cdef unsigned int thread_idx
    cdef numpy.ndarray[numpy.uint8_t] samples = numpy.zeros(n_samp, dtype=numpy.uint8)
    cdef double rn

    for samp_idx in prange(n_samp, nogil=True, schedule=static, num_threads=n_threads):
        thread_idx = threadid()
        mt_ptr = <mt_struct * > mt_ptrs[thread_idx]
        rn = genrand_mt(mt_ptr) / RAND_MAX
        samples[samp_idx] = rn < p[samp_idx]
    return samples
