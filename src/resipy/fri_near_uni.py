#!/usr/bin/env python2
"""
Utilities for applying FRI-type compression to the
Near-Uniform distribution.
"""

import numpy
import compress_utils
import near_uniform


def cmp_hier(sol_vec, n_sample, p_doub, occ_orb,
                 orb_symm, symm_lookup):
    symm_virt = near_uniform.virt_symm(occ_orb, orb_symm, symm_lookup)
    occ_allow, virt_allow = near_uniform.sing_allow(symm_virt, occ_orb,
                                                    orb_symm)
    # First layer of compression: singles vs. doubles
    sing_doub = numpy.array([[1 - p_doub], [p_doub]])
    num_dets = sol_vec.values.shape[0]
    new_weights = sing_doub * sol_vec.values
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
    n_fried = fri_idx.shape[0]
    sampl_idx = fri_idx[:, 0]
    # Group nonzero elements in FRI vector by single/double excitations
    sing_idx = sampl_idx < n_sing
    doub_idx = numpy.logical_not(sing_idx)
    new_det_idx = det_idx[sampl_idx[sing_idx]]
    n_sing = new_det_idx.shape[0]
    n_doub = n_fried - n_sing
    occ_idx = occ_allow[new_det_idx, fri_idx[sing_idx, 1] + 1]
    new_det_idx = numpy.append(new_det_idx, det_idx[sampl_idx[doub_idx]])
    det_idx = new_det_idx
    occ_idx = numpy.append(occ_idx, fri_idx[doub_idx, 1])  # index of occupied orbital or occ pair
    fri_vals = numpy.append(fri_vals[sing_idx], fri_vals[doub_idx])

    # Third layer of compression: allowed virtual orbitals for singles, symmetry pairs for doubles
    doub_wts, doub_nvirt, doub_occ = near_uniform.symm_pair_wt(symm_virt, occ_orb, orb_symm,
                                                               det_idx[n_sing:], occ_idx[n_sing:])

    fri_idx, fri_vals = compress_utils.fri_subd(fri_vals, virt_allow[det_idx[:n_sing], occ_idx[:n_sing]],
                                                    doub_wts, n_sample)
    n_fried = fri_idx.shape[0]
    sampl_idx = fri_idx[:, 0]
    # Group nonzero elements in FRI vector by single/double excitations
    sing_idx = sampl_idx < n_sing
    doub_idx = numpy.logical_not(sing_idx)
    new_det_idx = det_idx[sampl_idx[sing_idx]]
    doub_idx_shift = n_sing
    n_sing = new_det_idx.shape[0]
    n_doub = n_fried - n_sing
    new_det_idx = numpy.append(new_det_idx, det_idx[sampl_idx[doub_idx]])
    det_idx = new_det_idx
    occ_idx = numpy.append(occ_idx[sampl_idx[sing_idx]], occ_idx[sampl_idx[doub_idx]])
    virt_idx = numpy.append(fri_idx[sing_idx, 1], fri_idx[doub_idx, 1])
    fri_vals = numpy.append(fri_vals[sing_idx], fri_vals[doub_idx])
    doub_occ = doub_occ[sampl_idx[doub_idx] - doub_idx_shift]
    doub_wts = doub_wts[sampl_idx[doub_idx] - doub_idx_shift, virt_idx[n_sing:]]
    doub_nvirt = doub_nvirt[sampl_idx[doub_idx] - doub_idx_shift, virt_idx[n_sing:]]

    orb_counts = numpy.append(numpy.ones(n_sing), doub_nvirt)
    weights = numpy.empty((0, 2))
    # Fourth layer of compression: virtual orbital pair for doubles
    fri_idx, fri_vals = compress_utils.fri_subd(fri_vals, orb_counts, weights, n_sample)
    n_fried = fri_idx.shape[0]
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
    sing_orb[:, 1] = near_uniform.id_sing_virt(sol_vec.indices[sing_det_idx], symm_lookup,
                                               orb_symm[sing_occ % n_orb], (sing_occ / n_orb) * n_orb, virt_idx[sampl_idx[sing_idx]])
    sing_probs = ((1 - p_doub) / occ_allow[sing_det_idx, 0] / virt_allow[sing_det_idx, sing_orb_idx]
                  * sol_vec.values[sing_det_idx] / fri_vals[sing_idx])

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
                  * sol_vec.values[doub_det_idx] / fri_vals[doub_idx])

    return doub_orb, doub_probs, doub_det_idx, sing_orb, sing_probs, sing_det_idx
