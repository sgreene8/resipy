#!/usr/bin/env python2
"""
Subroutine for applying FRI-type compression to the
Near-Uniform distribution.
"""

import numpy
import compress_utils
import near_uniform
import fci_c_utils


def cmp_hier_strat(sol_vec, n_sample, p_doub, occ_orb,
                   orb_symm, symm_lookup, hf_num, rngen_ptrs):
    """Perform FRI-type compression on the Near-Uniform distribution,
        column-by-column, preserving colummns exactly as determined by
        number of samples vs. number of nonzero elements.

        Parameters
        ----------
        sol_vec : (SparseVector object)
            the current solution vector
        n_sample : (unsigned int)
            the desired number of nonzero matrix elements in each column after
            compression
        p_doub : (double)
            the probability of choosing a double excitation vs a single excitation
        occ_orb : (numpy.ndarray, uint8)
            The numbers in each row correspond to the indices of occupied
            orbitals in each determinant, calculated from fci_c_utils.gen_orb_lists
        orb_symm : (numpy.ndarray, uint8)
            irreducible representation of each spatial orbital
        symm_lookup : (numpy.ndarray, uint8)
            Table of orbitals with each type of symmetry, as generated
            by fci_utils.gen_byte_table()

        Returns
        -------
        (numpy.ndarray, uint8) :
            chosen occupied (0th and 1st columns) and unoccupied (2nd and 3rd
            columns) orbitals for double excitations
        (numpy.ndarray, float64) :
            probability of selecting each chosen double excitation (weight
            divided by compressed weight)
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
    vec_weights = numpy.abs(sol_vec.values)
    one_norm = vec_weights.sum()
    kept_sing_orb = numpy.empty([0, 2], dtype=numpy.uint8)
    kept_doub_orb = numpy.empty([0, 4], dtype=numpy.uint8)
    kept_sing_idx = numpy.empty(0, dtype=numpy.uint32)
    kept_doub_idx = numpy.empty(0, dtype=numpy.uint32)

    num_nonz = vec_weights.shape[0]

    max_idx = 0
    target_n_col = (n_sample - num_nonz) * vec_weights[max_idx] / one_norm
    # target_n_col = n_sample * vec_weights[max_idx] / one_norm
    all_col_doub, det_idx = fci_c_utils.all_doub_ex(sol_vec.indices[0:1], occ_orb[0:1], orb_symm)
    all_col_sing = numpy.empty([0, 2], dtype=numpy.uint8)
    n_col = hf_num

    while target_n_col + 1 > n_col and one_norm > 1e-10:
        kept_sing_orb = numpy.append(kept_sing_orb, all_col_sing, axis=0)
        kept_doub_orb = numpy.append(kept_doub_orb, all_col_doub, axis=0)
        kept_sing_idx = numpy.append(kept_sing_idx, numpy.full(all_col_sing.shape[0], max_idx, dtype=numpy.uint32))
        kept_doub_idx = numpy.append(kept_doub_idx, numpy.full(all_col_doub.shape[0], max_idx, dtype=numpy.uint32))

        one_norm -= vec_weights[max_idx]
        n_sample -= n_col
        vec_weights[max_idx] = 0
        num_nonz -= 1

        max_idx = numpy.argmax(vec_weights)
        target_n_col = (n_sample - num_nonz) * vec_weights[max_idx] / one_norm
        # target_n_col = n_sample * vec_weights[max_idx] / one_norm
        curr_det = sol_vec.indices[max_idx:(max_idx + 1)]
        curr_occ = occ_orb[max_idx:(max_idx + 1)]
        all_col_doub, det_idx = fci_c_utils.all_doub_ex(curr_det, curr_occ, orb_symm)
        n_col = det_idx.shape[0]
        all_col_sing, det_idx = fci_c_utils.all_sing_ex(curr_det, curr_occ, orb_symm)
        n_col += det_idx.shape[0]
    print('number preserved exactly', vec_weights.shape[0] - num_nonz)

    doub_probs = numpy.ones(kept_doub_orb.shape[0])
    sing_probs = numpy.ones(kept_sing_orb.shape[0])

    one_norm = vec_weights.sum()
    if one_norm > 1e-10:
        n_col, = compress_utils.sys_resample(vec_weights / one_norm, n_sample - num_nonz, ret_counts=True)
        n_col[vec_weights != 0] += 1
#         n_col, = compress_utils.sys_resample(vec_weights / one_norm, n_sample, ret_counts=True)
    else:
        return kept_doub_orb, doub_probs, kept_doub_idx, kept_sing_orb, sing_probs, kept_sing_idx

    single_col = n_col == 1
    single_counts = numpy.zeros_like(sol_vec.indices, dtype=numpy.uint32)
    single_counts[single_col] = 1
    n_col[single_col] = 0
#    single_counts = n_col
    single_doub, single_sing = near_uniform.bin_n_sing_doub(single_counts, p_doub)

    one_doub_orb, one_doub_prob, one_doub_idx = near_uniform.doub_multin(
        sol_vec.indices, occ_orb, orb_symm, symm_lookup, single_doub, rngen_ptrs)
    one_doub_prob *= p_doub  # * single_counts[one_doub_idx]
    one_sing_orb, one_sing_prob, one_sing_idx = near_uniform.sing_multin(
        sol_vec.indices, occ_orb, orb_symm, symm_lookup, single_sing, rngen_ptrs)
    one_sing_prob *= (1 - p_doub)  # * single_counts[one_sing_idx]

    doub_orb = numpy.append(kept_doub_orb, one_doub_orb, axis=0)
    doub_probs = numpy.append(doub_probs, one_doub_prob)
    doub_idx = numpy.append(kept_doub_idx, one_doub_idx)
    sing_orb = numpy.append(kept_sing_orb, one_sing_orb, axis=0)
    sing_probs = numpy.append(sing_probs, one_sing_prob)
    sing_idx = numpy.append(kept_sing_idx, one_sing_idx)

    fri_doub_orb, fri_doub_probs, fri_doub_idx, fri_sing_orb, fri_sing_probs, fri_sing_idx = near_uniform.fri_parallel(sol_vec.indices,
                                                                                                                       occ_orb, orb_symm,
                                                                                                                       symm_lookup, n_col,
                                                                                                                       rngen_ptrs, p_doub)

    doub_orb = numpy.append(doub_orb, fri_doub_orb, axis=0)
    doub_probs = numpy.append(doub_probs, fri_doub_probs)
    doub_idx = numpy.append(doub_idx, fri_doub_idx)
    sing_orb = numpy.append(sing_orb, fri_sing_orb, axis=0)
    sing_probs = numpy.append(sing_probs, fri_sing_probs)
    sing_idx = numpy.append(sing_idx, fri_sing_idx)

    return doub_orb, doub_probs, doub_idx, sing_orb, sing_probs, sing_idx


def cmp_hier(sol_vec, n_sample, p_doub, occ_orb,
             orb_symm, symm_lookup):
    """Perform FRI-type compression on the Near-Uniform distribution,
    exploiting its hierarchical structure for efficiency.

    Parameters
    ----------
    sol_vec : (SparseVector object)
        the current solution vector
    n_sample : (unsigned int)
        the desired number of nonzero matrix elements after the compression
    p_doub : (double)
        the probability of choosing a double excitation vs a single excitation
    occ_orb : (numpy.ndarray, uint8)
        The numbers in each row correspond to the indices of occupied
        orbitals in each determinant, calculated from fci_c_utils.gen_orb_lists
    orb_symm : (numpy.ndarray, uint8)
        irreducible representation of each spatial orbital
    symm_lookup : (numpy.ndarray, uint8)
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
    seq_idx = numpy.arange(n_sample, dtype=numpy.int32)
    symm_virt = near_uniform.virt_symm(occ_orb, orb_symm, symm_lookup)
    occ_allow, virt_allow = near_uniform.sing_allow(symm_virt, occ_orb,
                                                    orb_symm)
    # First layer of compression: singles vs. doubles
    sing_doub = numpy.array([[1 - p_doub], [p_doub]])
    num_dets = sol_vec.values.shape[0]
    vec_reweights = numpy.abs(sol_vec.values)
    new_weights = sing_doub * vec_reweights
    new_weights.shape = -1  # singles first, then doubles

    fri_idx, fri_vals = compress_utils.fri_1D(new_weights, n_sample)
    n_fried = fri_idx.shape[0]
    det_idx = fri_idx % num_dets  # index of determinant
    det_idx = det_idx.astype(numpy.uint32)
    n_sing = numpy.searchsorted(fri_idx, num_dets)
    n_doub = n_fried - n_sing

    # Second layer of compression: occupied orbitals, or occupied pairs, for each choice
    counts = occ_allow[det_idx[:n_sing], 0].astype(numpy.uint32)
    disallowed_ex = counts == 0
    fri_vals[:n_sing][disallowed_ex] = 0
    counts[disallowed_ex] = 1  # to avoid 0/0 errors

    n_elec = occ_orb.shape[1]
    n_occ_pair = n_elec * (n_elec - 1) / 2
    counts = numpy.append(counts, n_occ_pair * numpy.ones(n_doub, numpy.uint32))

    fri_idx, fri_vals = compress_utils.fri_subd(fri_vals, counts, numpy.empty([0, 0]), n_sample)
    sampl_idx = fri_idx[:, 0]
    # Group nonzero elements in FRI vector by single/double excitations
    sing_idx = sampl_idx < n_sing
    doub_idx = numpy.logical_not(sing_idx)
    new_det_idx = det_idx[sampl_idx[sing_idx]]
    n_sing = new_det_idx.shape[0]
    occ_idx = occ_allow[new_det_idx, fri_idx[sing_idx, 1] + 1]
    new_det_idx = numpy.append(new_det_idx, det_idx[sampl_idx[doub_idx]])
    det_idx = new_det_idx
    occ_idx = numpy.append(occ_idx, fri_idx[doub_idx, 1])  # index of occupied orbital or occ pair
    fri_vals = numpy.append(fri_vals[sing_idx], fri_vals[doub_idx])

    # Third layer of compression: allowed virtual orbitals for singles, symmetry pairs for doubles
    doub_wts, doub_nvirt, doub_occ = near_uniform.symm_pair_wt(symm_virt, occ_orb, orb_symm,
                                                               det_idx[n_sing:], occ_idx[n_sing:])
    null_doub = numpy.logical_and(doub_occ[:, 0] == 0, doub_occ[:, 1] == 0)
    null_idx = numpy.nonzero(null_doub)[0]
    fri_vals[null_idx + n_sing] = 0

    fri_idx, fri_vals = compress_utils.fri_subd(fri_vals, virt_allow[det_idx[:n_sing], occ_idx[:n_sing]],
                                                doub_wts, n_sample)
    all_arrs, sing_arrs, doub_arrs, n_sing = compress_utils.proc_fri_sd_choices(
        fri_idx[:, 0], n_sing, [det_idx, occ_idx], [], [doub_occ, doub_wts, doub_nvirt])
    det_idx, occ_idx = all_arrs
    doub_occ, doub_wts, doub_nvirt = doub_arrs
    n_doub = doub_occ.shape[0]

    virt_idx = fri_idx[:, 1]
    doub_wts = doub_wts[seq_idx[:n_doub], virt_idx[n_sing:]]
    doub_nvirt = doub_nvirt[seq_idx[:n_doub], virt_idx[n_sing:]]

    orb_counts = numpy.append(numpy.ones(n_sing), doub_nvirt)
    weights = numpy.empty((0, 2))
    # Fourth layer of compression: virtual orbital pair for doubles
    fri_idx, fri_vals = compress_utils.fri_subd(fri_vals, orb_counts, weights, n_sample)
    sampl_idx = fri_idx[:, 0]
    # Group nonzero elements in FRI vector by single/double excitations
    sing_idx = sampl_idx < n_sing
    doub_idx = numpy.logical_not(sing_idx)

    # Handle single excitations
    sing_det_idx = det_idx[sampl_idx[sing_idx]]
    sing_orb_idx = occ_idx[sampl_idx[sing_idx]]
    doub_idx_shift = n_sing
    n_sing = sing_det_idx.shape[0]
    sing_orb = numpy.zeros([n_sing, 2], dtype=numpy.uint8)
    sing_occ = occ_orb[sing_det_idx, sing_orb_idx]
    sing_orb[:, 0] = sing_occ
    n_orb = orb_symm.shape[0]
    virt_choices = near_uniform.virt_symm_idx(sol_vec.indices[sing_det_idx], symm_lookup,
                                              orb_symm[sing_occ % n_orb], (sing_occ / n_orb) * n_orb)
    sing_orb[:, 1] = virt_choices[seq_idx[:n_sing], virt_idx[sampl_idx[sing_idx]]]
    sing_probs = ((1 - p_doub) / occ_allow[sing_det_idx, 0] / virt_allow[sing_det_idx, sing_orb_idx]
                  * vec_reweights[sing_det_idx] / fri_vals[sing_idx])

    # Handle double excitations
    doub_det_idx = det_idx[sampl_idx[doub_idx]]
    n_doub = doub_det_idx.shape[0]
    doub_orb = numpy.zeros([n_doub, 4], dtype=numpy.uint8)
    doub_occ = doub_occ[sampl_idx[doub_idx] - doub_idx_shift]
    doub_orb[:, :2] = doub_occ
    doub_unocc = near_uniform.id_doub_virt(sol_vec.indices[doub_det_idx], symm_lookup,
                                           orb_symm, doub_occ, virt_idx[sampl_idx[doub_idx]], fri_idx[doub_idx, 1],
                                           symm_virt[doub_det_idx])
    doub_unocc.sort(axis=1)
    doub_orb[:, 2:] = doub_unocc
    doub_probs = (p_doub / n_occ_pair * doub_wts[sampl_idx[doub_idx] - doub_idx_shift] / doub_nvirt[sampl_idx[doub_idx] - doub_idx_shift]
                  * vec_reweights[doub_det_idx] / fri_vals[doub_idx])

    return doub_orb, doub_probs, doub_det_idx, sing_orb, sing_probs, sing_det_idx
