#!/usr/bin/env python2
"""
Utilities for compressing the Heat-Bath distribution described in
Holmes et al. (2016) (doi: 10.1021/acs.jctc.5b01170).
"""

import numpy
import fci_utils
import compress_utils
import misc_c_utils
import near_uniform


def set_up(frozen, eris):
    """Set up the Heat-Bath probability distributions for choosing the occupied
    orbitals for a double excitation.

    Parameters
    ----------
    frozen : (unsigned int)
        Number of core electrons frozen in this system
    eris : (numpy.ndarray, float)
        4-D array of 2-electron integrals in the spatial
        Slater determinant basis from a Hartree-Fock calculation.

    Returns
    -------
    (numpy.ndarray, float64) :
        one-electron probability tensor (S in Holmes et al.), with shape (M)
    (numpy.ndarray, float64) :
        electron pair selection probability tensor (D in Holmes et al.), with
        shape (M, M)
    (numpy.ndarray, float64) :
        Square root of absolute value of exchange integrals. 2-D tensor, with
        the 0th and 1st indices corresponding to orbitals in the integral as
        |<0 1 | 1 0>|. Frozen orbitals are not included, and spin is not taken
        into account
    """

    start = frozen / 2
    eris_spin = fci_utils.eris_space2spin(eris[start:, start:, start:, start:])
    eris_abs = numpy.abs(eris_spin)
    eris_dim = eris_abs.shape[0]

    exch_idx1 = numpy.arange(eris_dim / 2, dtype=numpy.uint32)
    exch_idx1.shape = (-1, 1)
    exch_idx1 = numpy.tile(exch_idx1, (1, eris_dim / 2))
    exch_idx2 = exch_idx1.transpose()
    exch_abs = eris_abs[exch_idx1, exch_idx2, exch_idx2, exch_idx1]

    for i in range(eris_dim):
        eris_abs[:, i, i, :] = 0
        eris_abs[i, :, :, i] = 0
        eris_abs[i, :, i, :] = 0
        eris_abs[:, i, :, i] = 0
    d_tens = numpy.sum(eris_abs, axis=(2, 3))
    s_tens = numpy.sum(d_tens, axis=1)

    return s_tens, d_tens, numpy.sqrt(exch_abs)


def _cs_virt_wts(sampl_occ, det_idx, exch_tens, occ_orbs):
    # Calculate the normalized Cauchy-Schwartz weights for virtual orbitals
    # given selected occupied orbitals for a set of determinants
    n_orb = exch_tens.shape[0]
    n_samp = sampl_occ.shape[0]
    n_elec = occ_orbs.shape[1]

    spin = sampl_occ / n_orb
    occ_idx = numpy.arange(n_elec / 2, dtype=numpy.uint32)
    occ_idx = numpy.tile(occ_idx, (n_samp, 1))
    occ_idx += spin[:, numpy.newaxis] * n_elec / 2
    tmp_idx = numpy.arange(n_samp, dtype=numpy.uint32)
    tmp_idx.shape = (-1, 1)

    virt_wts = exch_tens[sampl_occ % n_orb]
    det_idx.shape = (-1, 1)
    virt_wts[tmp_idx, occ_orbs[det_idx, occ_idx] % n_orb] = 0
    det_idx.shape = -1

    # Normalize
    norms = 1. / virt_wts.sum(axis=1)
    norms.shape = (-1, 1)
    virt_wts *= norms

    return virt_wts


def _cs_symm_wts(sampl_virt, sampl_occ, symm, all_dets, exch_tens,
                 lookup_tabl):
    # Calculate the normalized Cauchy-Schwartz weights for a given
    # symmetry, excluding virtual orbitals that have already been sampled
    n_orb = exch_tens.shape[0]
    spin_unocc = sampl_virt / n_orb
    spin_occ = sampl_occ / n_orb
    same_spin = spin_occ == spin_unocc
    spin_occ *= n_orb

    symm_options = near_uniform.virt_symm_bool(all_dets, lookup_tabl,
                                               symm, spin_occ, n_orb)
    symm_wts = exch_tens[sampl_occ % n_orb]
    # Exclude orbitals that don't have correct symmetry
    symm_wts[numpy.logical_not(symm_options)] = 0

    # Exclude virtual orbitals already selected
    tmp_idx = numpy.arange(sampl_virt.shape[0], dtype=numpy.uint32)
    symm_wts[tmp_idx[same_spin], sampl_virt[same_spin] % n_orb] = 0

    # Normalize, and identify non-null excitations
    norms = symm_wts.sum(axis=1)
    nonnull = norms > 1e-8
    norms = 1. / norms[nonnull]
    symm_wts = symm_wts[nonnull] * norms[:, numpy.newaxis]

    return symm_wts, nonnull


def fri_comp(sol_vec, n_nonz, s_tens, d_tens, exch_tens, p_doub, occ_orbs, orb_symm, lookup_tabl):
    """Perform FRI-type compression on the Near-Uniform (for singles) and heat-bath Power-Pitzer 
    (for doubles) distribution, exploiting its hierarchical structure for efficiency.

    Parameters
    ----------
    sol_vec : (SparseVector object)
        the current solution vector
    n_nonz : (unsigned int)
        the desired number of nonzero matrix elements after the compression
    s_tens : (numpy.ndarray, float64)
        one-electron probability tensor from the set_up subroutine
    d_tens : (numpy.ndarray, float64)
        two-electron probability tensor from the set_up subroutine
    exch_tens : (numpy.ndarray, float64)
        2-D array of exchange integrals from the set_up subroutine
    p_doub : (double)
        the probability of choosing a double excitation vs a single excitation
    occ_orbs : (numpy.ndarray, uint8)
        The numbers in each row correspond to the indices of occupied
        orbitals in each determinant, calculated from fci_c_utils.gen_orb_lists
    orb_symm : (numpy.ndarray, uint8)
        irreducible representation of each spatial orbital
    lookup_tabl : (numpy.ndarray, uint8)
        Table of orbitals with each type of symmetry, as generated
        by fci_utils.gen_byte_table()

    Returns
    -------
    (numpy.ndarray, uint8) :
        chosen occupied (0th and 1st columns) and unoccupied (2nd and 3rd
        columns) orbitals for double excitations
    (numpy.ndarray, float64) :
        probability of selecting each chosen double excitation
    (numpy.ndarray, uint32) :
        index of the origin determinant of each chosen double excitation
         in the dets array
    (numpy.ndarray, uint8) :
        chosen occupied (0th column) and unoccupied (1st column)
        orbitals for single excitations
    (numpy.ndarray, float64) :
        probability of selecting each chosen single excitation
    (numpy.ndarray, uint32) :
        index of the origin determinant of each chosen single excitation
         in the dets array
    """


def doub_multin(s_tens, d_tens, exch_tens, dets, occ_orbs, orb_symm, lookup_tabl, n_sampl):
    """Sample double excitations multinomially from the heat-bath Power-Pitzer
    distribution defined in Neufeld and Thom (2018).

    Parameters
    ----------
    s_tens : (numpy.ndarray, float64)
        one-electron probability tensor from the set_up subroutine
    d_tens : (numpy.ndarray, float64)
        two-electron probability tensor from the set_up subroutine
    exch_tens : (numpy.ndarray, float64)
        2-D array of exchange integrals from the set_up subroutine
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
    n_sampl : (numpy.ndarray, uint32)
        number of double excitations to choose for each determinant

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
    """
    det_idx = misc_c_utils.ind_from_count(n_sampl)
    tot_sampl = det_idx.shape[0]
    n_orb = s_tens.shape[0] / 2

    doub_occ, occ_probs = _multi_occ_pair(s_tens, d_tens, occ_orbs, n_sampl, det_idx)

    doub_unocc = numpy.zeros([tot_sampl, 2], dtype=numpy.uint8)

    tmp_idx = numpy.arange(tot_sampl, dtype=numpy.uint32)
    virt1_wts = _cs_virt_wts(doub_occ[:, 0], det_idx, exch_tens, occ_orbs)

    alias, Q = misc_c_utils.setup_alias(virt1_wts)
    unocc1_orbs = compress_utils.sample_alias(alias, Q, tmp_idx)
    unocc_probs = virt1_wts[tmp_idx, unocc1_orbs]
    spin1 = (doub_occ[:, 0] / n_orb) * n_orb
    doub_unocc[:, 0] = unocc1_orbs + spin1

    virt2_symm = (orb_symm[doub_occ[:, 0] % n_orb] ^ orb_symm[doub_occ[:, 1] % n_orb]
                  ^ orb_symm[unocc1_orbs])

    spin2 = (doub_occ[:, 1] / n_orb) * n_orb
    virt2_wts, nonnull = _cs_symm_wts(doub_unocc[:, 0], doub_occ[:, 1], virt2_symm, dets[det_idx], exch_tens, lookup_tabl)

    doub_occ = doub_occ[nonnull]
    doub_unocc = doub_unocc[nonnull]
    spin1 = spin1[nonnull]
    spin2 = spin2[nonnull]
    det_idx = det_idx[nonnull]
    occ_probs = occ_probs[nonnull]
    unocc_probs = unocc_probs[nonnull]
    unocc1_orbs = unocc1_orbs[nonnull]
    tot_sampl = unocc_probs.shape[0]

    alias, Q = misc_c_utils.setup_alias(virt2_wts)
    tmp_idx = numpy.arange(tot_sampl, dtype=numpy.uint32)
    unocc2_orbs = compress_utils.sample_alias(alias, Q, tmp_idx)
    doub_unocc[:, 1] = unocc2_orbs + spin2
    unocc_probs *= virt2_wts[tmp_idx, unocc2_orbs]

    doub_unocc.sort(axis=1)
    sampl_orbs = numpy.append(doub_occ, doub_unocc, axis=1)

    # Calculate probability of choosing orbitals for same-spin excitations in
    # reverse order
    same_spin = spin1 == spin2
    new_det_idx = det_idx[same_spin]
    dets = dets[new_det_idx]
    doub_occ = doub_occ[same_spin]
    unocc1_orbs = unocc1_orbs[same_spin]
    unocc2_orbs = unocc2_orbs[same_spin]
    virt1_wts_rev = _cs_virt_wts(doub_occ[:, 1], new_det_idx, exch_tens, occ_orbs)
    virt2_symm = (orb_symm[doub_occ[:, 0] % n_orb] ^ orb_symm[doub_occ[:, 1] % n_orb] ^
                  orb_symm[unocc2_orbs])
    virt2_wts_rev, nonnull = _cs_symm_wts(unocc2_orbs + spin2[same_spin], doub_occ[:, 0], virt2_symm, dets, exch_tens, lookup_tabl)

    tmp_idx = numpy.arange(doub_occ.shape[0], dtype=numpy.uint32)
    unocc_probs[same_spin] += virt1_wts_rev[tmp_idx, unocc2_orbs] * virt2_wts_rev[tmp_idx, unocc1_orbs]

    return sampl_orbs, unocc_probs * occ_probs, det_idx


def _multi_occ_pair(s_tens, d_tens, occ_orbs, n_sampl, idx):
    # Multinomially choose the occupied orbitals for a double excitation
    tot_sampl = idx.shape[0]
    chosen_orbs = numpy.zeros([tot_sampl, 2], dtype=numpy.uint8)
    chosen_probs = numpy.zeros([tot_sampl, 2])

    n_hf = n_sampl[0]
    hf_o1_probs = s_tens[occ_orbs[0]]
    norms = hf_o1_probs.sum()
    hf_o1_probs /= norms
    hf_o1_probs.shape = (1, -1)
    alias, Q = misc_c_utils.setup_alias(hf_o1_probs)
    hf_o1_idx = compress_utils.sample_alias(alias, Q, idx[:n_hf])
    chosen_orbs[:n_hf, 0] = occ_orbs[0, hf_o1_idx]
    chosen_probs[:n_hf, 0] = hf_o1_probs[0, hf_o1_idx]

    o1_probs = s_tens[occ_orbs[1:]]
    norms = o1_probs.sum(axis=1)
    norms.shape = (-1, 1)
    o1_probs /= norms
    alias, Q = misc_c_utils.setup_alias(o1_probs)
    o1_idx = compress_utils.sample_alias(alias, Q, idx[n_hf:] - 1)
    chosen_orbs[n_hf:, 0] = occ_orbs[idx[n_hf:], o1_idx]
    chosen_probs[n_hf:, 0] = o1_probs[idx[n_hf:] - 1, o1_idx]

    # Choose the second occupied orbitals
    n_elec = occ_orbs.shape[1]
    row_idx = occ_orbs[0]
    col_idx = row_idx
    row_idx.shape = (-1, 1)
    row_idx = numpy.tile(row_idx, (1, n_elec))
    col_idx.shape = -1
    col_idx = numpy.tile(col_idx, (n_elec, 1))

    hf_o2_probs = d_tens[row_idx, col_idx]
    norms = hf_o2_probs.sum(axis=1)
    norms.shape = (-1, 1)
    hf_o2_probs /= norms
    alias, Q = misc_c_utils.setup_alias(hf_o2_probs)
    hf_o2_idx = compress_utils.sample_alias(alias, Q, hf_o1_idx)
    chosen_orbs[:n_hf, 1] = occ_orbs[0, hf_o2_idx]
    chosen_probs[:n_hf, 1] = hf_o1_probs[0, hf_o2_idx] * hf_o2_probs[hf_o2_idx, hf_o1_idx]
    chosen_probs[:n_hf, 0] *= hf_o2_probs[hf_o1_idx, hf_o2_idx]

    occ_expand = occ_orbs[idx[n_hf:]]
    o2_probs = d_tens[chosen_orbs[n_hf:, 0:1], occ_expand]
    norms = o2_probs.sum(axis=1)
    norms.shape = (-1, 1)
    o2_probs /= norms
    alias, Q = misc_c_utils.setup_alias(o2_probs)
    tmp_idx = numpy.arange(tot_sampl - n_hf, dtype=numpy.uint32)
    o2_idx = compress_utils.sample_alias(alias, Q, tmp_idx)
    chosen_orbs[n_hf:, 1] = occ_expand[tmp_idx, o2_idx]
    chosen_probs[n_hf:, 1] = o1_probs[idx[n_hf:] - 1, o2_idx]
    chosen_probs[n_hf:, 0] *= o2_probs[tmp_idx, o2_idx]
    hf_o2_probs = d_tens[chosen_orbs[n_hf:, 1:2], occ_expand]
    norms = hf_o2_probs.sum(axis=1)
    norms.shape = (-1, 1)
    hf_o2_probs /= norms
    chosen_probs[n_hf:, 1] *= hf_o2_probs[tmp_idx, o1_idx]

    chosen_orbs.sort(axis=1)

    return chosen_orbs, chosen_probs.sum(axis=1)
