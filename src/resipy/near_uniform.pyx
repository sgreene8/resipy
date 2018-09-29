#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy
from cython.parallel import prange, threadid, parallel
cimport numpy
cimport cython
from libc.time cimport time
from libc.math cimport sqrt

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
        ini_ptrs[i] = <unsigned long> mts
    return ini_ptrs

def doub_multin(long long[:] dets, unsigned char[:,:] occ_orbs, 
                       unsigned char[:] orb_symm, unsigned char[:,:] lookup_tabl,
                       unsigned int[:] num_sampl, unsigned long[:] mt_ptrs):
    ''' Uniformly chooses num_sampl[i] double excitations for each determinant 
        dets[i] according to the symmetry-adapted rules described in Sec. 5.2
        of Booth et al. (2014) using independent multinomial sampling.

        Parameters
        ----------
        dets : (numpy.ndarray, int64)
            Bit string representations of all determinants
        occ_orbs : (numpy.ndarray, uint8)
            The numbers in each row correspond to the indices of occupied
            orbitals in each determinant, calculated from gen_orb_lists.
        orb_symm : (numpy.ndarray, unit8)
            irreducible representation of each spatial orbital
        lookup_tabl : (numpy.ndarray, uint8)
            Table of orbitals with each type of symmetry, as generated
            by fci_helpers2.gen_byte_table()
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
    '''

    cdef unsigned int num_dets = dets.shape[0]
    cdef unsigned int num_elec = occ_orbs.shape[1]
    cdef unsigned int num_orb = orb_symm.shape[0]
    cdef unsigned int i, det_idx
    cdef unsigned int a_symm, b_symm, a_spin, b_spin, sym_prod
    cdef unsigned int unocc1, unocc2, m_a_allow, m_a_b_allow, m_b_a_allow
    cdef orb_pair occ
    cdef long long curr_det
    cdef unsigned int tot_sampl = 0
    cdef double prob
    cdef numpy.ndarray[numpy.uint32_t] start_idx = numpy.zeros(num_dets, dtype=numpy.uint32)
    cdef unsigned int thread_idx, n_threads = mt_ptrs.shape[0]
    cdef unsigned int n_symm = lookup_tabl.shape[0]
    cdef unsigned int[:, :, :] unocc_sym_counts = numpy.zeros([n_threads, 2, n_symm], dtype=numpy.uint32)
    cdef mt_struct *mts

    for i in range(num_dets):
        start_idx[i] = tot_sampl
        tot_sampl += num_sampl[i]
    
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] chosen_orbs = numpy.zeros([tot_sampl, 4],
                                                                          dtype=numpy.uint8)
    cdef numpy.ndarray[numpy.float64_t] prob_vec = numpy.zeros(tot_sampl)

    for det_idx in prange(num_dets, nogil=True, schedule=static, num_threads=n_threads):
        if (num_sampl[det_idx] == 0):
            continue
        tot_sampl = start_idx[det_idx]
        curr_det = dets[det_idx]
        thread_idx = threadid()
        mts = <mt_struct *>mt_ptrs[thread_idx]

        _count_symm_virt(unocc_sym_counts[thread_idx], occ_orbs[det_idx],
                           lookup_tabl, orb_symm)

        # Start sampling
        for i in range(num_sampl[det_idx]):
            occ = _choose_occ_pair(occ_orbs[det_idx], mts)

            sym_prod = orb_symm[occ.orb1 % num_orb] ^ orb_symm[occ.orb2 % num_orb]

            m_a_allow = _count_doub_virt(occ, orb_symm, num_elec,
                                         unocc_sym_counts[thread_idx])

            if m_a_allow == 0:
                prob_vec[tot_sampl + i] = -1.
                continue
            
            unocc1 = _doub_choose_virt1(occ, curr_det, orb_symm,
                                        unocc_sym_counts[thread_idx], m_a_allow,
                                        mts)
            a_spin = unocc1 / num_orb # spin of 1st virtual orbital chosen
            b_spin = occ.spin1 ^ occ.spin2 ^ a_spin # 2nd virtual spin
            a_symm = orb_symm[unocc1 % num_orb]
            b_symm = sym_prod ^ a_symm
            
            m_a_b_allow = (unocc_sym_counts[thread_idx, b_spin, b_symm] - 
                           (sym_prod == 0 and a_spin == b_spin))
            
            # Choose second unoccupied orbital
            unocc2 = _doub_choose_virt2(b_spin * num_orb, curr_det,
                                        lookup_tabl[b_symm],
                                        unocc1, m_a_b_allow, mts)
            
            # Calculate probability of choosing this excitation
            m_b_a_allow = (unocc_sym_counts[thread_idx, a_spin, a_symm] -
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

    return chosen_orbs, prob_vec



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
            orbitals in each determinant, calculated from gen_orb_lists.
        orb_symm : (numpy.ndarray, unit8)
            irreducible representation of each spatial orbital
        lookup_tabl : (numpy.ndarray, uint8)
            Table of orbitals with each type of symmetry, as generated
            by fci_helpers2.gen_byte_table()
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
    '''
    
    cdef unsigned int num_dets = occ_orbs.shape[0]
    cdef unsigned int num_elec = occ_orbs.shape[1]
    cdef unsigned int num_orb = orb_symm.shape[0]
    cdef unsigned int j, delta_s, num_allowed, det_idx
    cdef unsigned int occ_orb, occ_symm, occ_spin, virt_orb, sampl_idx
    cdef unsigned int elec_idx
    cdef unsigned int tot_sampl = 0
    cdef long long curr_det
    cdef numpy.ndarray[numpy.uint32_t] start_idx = numpy.zeros(num_dets, dtype=numpy.uint32)
    cdef unsigned int thread_idx
    cdef unsigned int n_threads = mt_ptrs.shape[0]
    cdef mt_struct *mts
    cdef unsigned int[:, :] m_allow = numpy.zeros([n_threads, num_elec], dtype=numpy.uint32)
    cdef unsigned int n_symm = lookup_tabl.shape[0]
    cdef unsigned int[:, :, :] unocc_sym_counts = numpy.zeros([n_threads, 2, n_symm], dtype=numpy.uint32)
    
    for det_idx in range(num_dets):
        start_idx[det_idx] = tot_sampl
        tot_sampl += num_sampl[det_idx]
    
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] chosen_orbs = numpy.zeros([tot_sampl, 2], 
                                                                          dtype=numpy.uint8)
    cdef numpy.ndarray[numpy.float64_t] prob_vec = numpy.zeros(tot_sampl)
    
    for det_idx in prange(num_dets, nogil=True, schedule=static, num_threads=n_threads):
        if (num_sampl[det_idx] == 0):
            continue
        thread_idx = threadid()
        mts = <mt_struct *>mt_ptrs[thread_idx]
        curr_det = dets[det_idx]
        sampl_idx = start_idx[det_idx]

        _count_symm_virt(unocc_sym_counts[thread_idx], occ_orbs[det_idx],
                         lookup_tabl, orb_symm)
        delta_s = 0 # number of electrons with no symmetry-allowed excitations
        for elec_idx in range(num_elec):
            occ_symm = orb_symm[occ_orbs[det_idx, elec_idx] % num_orb]
            num_allowed = unocc_sym_counts[thread_idx, elec_idx / (num_elec / 2), 
                                           occ_symm]
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
    
    return chosen_orbs, prob_vec


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
        if det & (<long long>1 << orbital):
            # orbital is occupied, choose again
            symm_idx = -1
    return orbital


cdef void _count_symm_virt(unsigned int[:, :] counts, unsigned char[:] occ_orbs,
                           unsigned char[:, :] symm_table,
                           unsigned char[:] orb_irreps) nogil:
    # Count number of unoccupied orbitals with each irrep
    cdef unsigned int i
    cdef unsigned int n_symm = symm_table.shape[0]
    cdef unsigned int n_elec = occ_orbs.shape[0]
    cdef unsigned int n_orb = orb_irreps.shape[0]

    for i in range(n_symm):
        counts[0, i] = symm_table[i, 0]
        counts[1, i] = symm_table[i, 0]
    for i in range(n_elec / 2):
        counts[0, orb_irreps[occ_orbs[i]]] -= 1
    for i in range(n_elec / 2, n_elec):
        counts[1, orb_irreps[occ_orbs[i] - n_orb]] -= 1


cdef orb_pair _choose_occ_pair(unsigned char[:] occ_orbs, mt_struct *mt_ptr
                               ) nogil:
    # Randomly & uniformly choose a pair of occupied orbitals 
    # using triangle inversion
    cdef unsigned int num_elec = occ_orbs.shape[0]
    cdef unsigned int rand_pair = _choose_uint(mt_ptr, num_elec * (num_elec - 1) 
                                                / 2)
    cdef orb_pair pair
    cdef unsigned int orb_idx1 = <unsigned int>((sqrt(rand_pair * 8. + 1) - 1) 
                                                / 2)
    cdef unsigned int orb_idx2 = <unsigned int>(rand_pair - orb_idx1 * (orb_idx1 
                                                                       + 1.) / 2)
    orb_idx1 += 1
    pair.orb1 = occ_orbs[orb_idx1]
    pair.orb2 = occ_orbs[orb_idx2]
    pair.spin1 = orb_idx1 / (num_elec / 2)
    pair.spin2 = orb_idx2 / (num_elec / 2)
    return pair


cdef unsigned int _count_doub_virt(orb_pair occ, unsigned char[:] orb_irreps,
                                   unsigned int num_elec,
                                   unsigned int[:, :] virt_counts) nogil:
    # Count number of spin- and symmetry-allowed unoccupied orbitals
    # given a chosen pair of occupied orbitals
    cdef unsigned int n_allow
    cdef unsigned int n_orb = orb_irreps.shape[0]
    cdef unsigned int n_symm = virt_counts.shape[1]
    cdef unsigned char sym_prod = (orb_irreps[occ.orb1 % n_orb] ^ 
                                   orb_irreps[occ.orb2 % n_orb])
    cdef int same_symm = sym_prod == 0 and occ.spin1 == occ.spin2
    cdef unsigned int i

    if occ.spin1 == occ.spin2:
        n_allow = n_orb - num_elec / 2
    else:
        n_allow = 2 * n_orb - num_elec

    for i in range(n_symm):
        if (virt_counts[occ.spin2, i ^ sym_prod] == same_symm):
            n_allow -= virt_counts[occ.spin1, i]
        if (occ.spin1 != occ.spin2 and virt_counts[occ.spin1, i ^ sym_prod] 
                                        == same_symm):
            n_allow -= virt_counts[occ.spin2, i]
    return n_allow


cdef unsigned int _doub_choose_virt1(orb_pair occ, long long det,
                                     unsigned char[:] irreps,
                                     unsigned int[:, :] virt_counts, 
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

    if num_allowed <= 3: # choose the orbital index and then search for it
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
            if (not(det & (<long long>1 << (orbital + a_spin * n_orb)))):
                a_symm = irreps[orbital]
                n_virt2 = (virt_counts[b_spin, sym_prod ^ a_symm] - 
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
                if (not(det & (<long long>1 << orbital))):
                    a_symm = irreps[orbital - n_orb]
                    n_virt2 = (virt_counts[b_spin, sym_prod ^ a_symm] - 
                               (sym_prod == 0 and a_spin == b_spin))
                    if n_virt2 != 0:
                        virt_choice -= 1
                orbital += 1
            orbital -= n_orb
        virt_choice = orbital - 1 + a_spin * n_orb
    else: # choose the orbital by rejection
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
            if not(det & (<long long>1 << virt_choice)): # check if unoccupied
                a_symm = irreps[virt_choice % n_orb]
                n_virt2 = (virt_counts[b_spin, sym_prod ^ a_symm] - 
                           (sym_prod == 0 and a_spin == b_spin))
    return virt_choice


cdef unsigned int _doub_choose_virt2(unsigned int spin_shift, long long det,
                                     unsigned char[:] symm_row,
                                     unsigned int virt1, unsigned int n_allow,
                                     mt_struct *mt_ptr) nogil:
    # Choose the second virtual orbial uniformly
    cdef int orb_idx = _choose_uint(mt_ptr, n_allow)
    cdef unsigned int orbital
    cdef unsigned int symm_idx = 1
    # Search for chosen orbital
    while (orb_idx >= 0):
        orbital = symm_row[symm_idx] + spin_shift
        if (not(det & (<long long>1 << orbital)) and orbital != virt1):
            orb_idx -= 1
        symm_idx += 1
    return orbital


cdef unsigned int _choose_uint(mt_struct *mt_ptr, unsigned int nmax) nogil:
    # Choose an integer uniformly on the interval [0, nmax)
    return <unsigned int> (genrand_mt(mt_ptr) / RAND_MAX * nmax)
