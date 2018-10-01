#!/usr/bin/env python2
"""
Script for running an FCIQMC calculation.
For help, run  python fciqmc.py -h
"""

import numpy
import argparse
from resipy import fci_utils
from resipy import fci_c_utils
from resipy import sparse_utils
from resipy import near_uniform
from resipy import compress_utils
from resipy import io_utils


def main():
    args = _parse_args()

    h_core, eris, symm, hf_en = io_utils.read_in_hf(args.hf_path, args.frozen)

    # Generate lookup tables for later use
    byte_nums, byte_idx = fci_utils.gen_byte_table()
    symm_lookup = fci_utils.gen_symm_lookup(8, symm)

    n_orb = symm.shape[0]
    hf_det = fci_utils.gen_hf_bitstring(n_orb, args.n_elec - args.frozen)

    # Initialize solution vector
    sol_dets = numpy.array([hf_det], dtype=numpy.int64)
    sol_vals = numpy.array([1.])
    occ_orbs = fci_c_utils.gen_orb_lists(sol_dets, 2 * n_orb, args.n_elec -
                                         args.frozen, byte_nums, byte_idx)

    results = io_utils.setup_results(args.result_int, args.result_dir,
                                     args.rayleigh, False)

    if args.prob_dist == "near_uniform":
        n_doub_ref = fci_utils.count_doubex(occ_orbs[0], symm, symm_lookup)
        n_sing_ref = fci_utils.count_singex(
            hf_det, occ_orbs[0], symm, symm_lookup)
        p_doub = n_doub_ref * 1.0 / (n_sing_ref + n_doub_ref)

    rngen_ptrs = near_uniform.initialize_mt(args.procs)

    # Elements in the HF column of FCI matrix
    hf_col_dets, hf_col_matrel = fci_utils.gen_hf_ex(
        hf_det, occ_orbs[0], n_orb, symm, eris, args.frozen)

    for iterat in range(args.max_iter):
        # number of samples to draw from each column
        n_col = numpy.ceil(args.H_sample * numpy.abs(sol_vals)).astype(int)

        if args.prob_dist == "near_uniform":
            n_doub_col, n_sing_col = near_uniform.bin_n_sing_doub(n_col, p_doub)
            # Sample double excitations
            doub_orbs, doub_probs, doub_idx = near_uniform.doub_multin(
                sol_dets, occ_orbs, symm, symm_lookup, n_doub_col, rngen_ptrs)
            doub_matrel = fci_utils.doub_matr_el_nosgn(
                doub_orbs, eris, args.frozen)
            # Retain nonzero elements
            doub_nonz = doub_matrel != 0
            doub_idx = doub_idx[doub_nonz]
            doub_orbs = doub_orbs[doub_nonz]
            doub_matrel = doub_matrel[doub_nonz]
            # Calculate determinants and matrix elements
            doub_dets, doub_signs = fci_utils.doub_dets_parity(
                sol_dets[doub_idx], doub_orbs)
            doub_matrel *= args.epsilon / doub_probs / p_doub / \
                n_col[doub_idx] * doub_signs * -sol_vals[doub_idx]
            # Start forming next iterate
            spawn_dets = doub_dets
            spawn_vals = doub_matrel

            # Sample single excitations
            sing_orbs, sing_probs, sing_idx = near_uniform.sing_multin(sol_dets, occ_orbs, symm, symm_lookup,  n_sing_col, rngen_ptrs)
            sing_dets, sing_matrel = fci_c_utils.single_dets_matrel(
                sol_dets[sing_idx], sing_orbs, eris, h_core, occ_orbs[sing_idx], args.frozen)
            # Retain nonzero elements
            sing_nonz = sing_matrel != 0
            sing_idx = sing_idx[sing_nonz]
            sing_dets = sing_dets[sing_nonz]
            sing_matrel = sing_matrel[sing_nonz]
            # Calculate determinants and matrix elements
            sing_matrel *= args.epsilon / sing_probs / \
                (1 - p_doub) / n_col[sing_idx] * -sol_vals[sing_idx]
            # Add to next iterate
            spawn_dets = numpy.append(spawn_dets, sing_dets)
            spawn_vals = numpy.append(spawn_vals, sing_matrel)
        # Diagonal matrix elements
        diag_matrel = fci_c_utils.diag_matrel(
            occ_orbs, h_core, eris, args.frozen) - hf_en
        diag_matrel = 1 - args.epsilon * diag_matrel
        diag_matrel *= sol_vals

        # Add vectors in sparse format
        next_dets, next_vals = sparse_utils.add_vectors(
            sol_dets, spawn_dets, diag_matrel, spawn_vals)
        one_norm = numpy.sum(numpy.abs(next_vals))
        io_utils.calc_results(results, next_dets, next_vals, 0, iterat,
                              hf_col_dets, hf_col_matrel)
        next_vals /= one_norm
        sol_dets, sol_vals = compress_utils.compress_sparse_vector(
            next_dets, next_vals, args.sparsity)
        occ_orbs = fci_c_utils.gen_orb_lists(sol_dets, 2 * n_orb, args.n_elec -
                                             args.frozen, byte_nums, byte_idx)


def _sample_singles(vec_dets, vec_occ, orb_symm, symm_lookup, n_col, rn_vec):
    """
    For the near-uniform distribution, multinomially choose the single
    excitations for each column.
    """
    det_idx = fci_c_utils.ind_from_count(n_col)
    orb_choices, ex_probs = near_uniform.sing_multin(vec_dets, vec_occ, orb_symm,
                                                     symm_lookup, n_col, rn_vec)
    return orb_choices, ex_probs, det_idx


def _parse_args():
    """
    Subroutine to parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Perform an FRI Calculation")
    parser.add_argument(
        'hf_path', type=str, help="Path to the directory that contains the HF output files eris.npy, hcore.npy, symm.npy, and hf_en.txt")
    parser.add_argument('n_elec', type=int,
                        help="Number of electrons in the molecule")
    parser.add_argument('epsilon', type=float, help="Imaginary time step")
    parser.add_argument('H_sample', type=int,
                        help="Total number of off-diagonal samples to draw from the Hamiltonian matrix")
    parser.add_argument('sparsity', type=int,
                        help="Target number of nonzero elements in the solution vector")
    parser.add_argument('prob_dist', choices=["near_uniform"],
                        help="Probability distribution to use to select Hamiltonian off-diagonal elements")
    parser.add_argument('sampl_mode', choices=["multinomial", "all"],
                        help="Method for sampling elements from this probability distribution.")
    parser.add_argument('-f', '--frozen', type=int, default=0,
                        help="Number of core electrons frozen")
    parser.add_argument('-p', '--procs', type=int, default=8,
                        help="Number of processors to use for multithreading")
    parser.add_argument('-r', '--result_int', type=int, default=1000,
                        help="Period with which to write results to disk")
    parser.add_argument('-y', '--result_dir', type=str, default=".",
                        help="Directory in which to save output files")
    parser.add_argument('--rayleigh', action="store_true",
                        help="Calculate Rayleigh quotient every 10 iterations")
    parser.add_argument('-i', '--max_iter', type=int, default=800000,
                        help="Number of iterations to simulate in the trajectory.")

    args = parser.parse_args()
    # process arguments and perform error checking

    if args.hf_path[-1] != '/':
        args.hf_path += '/'

    if args.result_dir[-1] != '/':
        args.result_dir += '/'

    if args.n_elec <= 0:
        raise ValueError(
            "Number of electrons in the system (%d) must be > 0" % args.n_elec)

    if args.frozen > args.n_elec:
        raise ValueError("Number of core electrons to freeze (%d) is greater than number of electrons in the system (%d)." % (
            args.frozen, args.n_elec))

    if args.frozen < 0:
        raise ValueError(
            "Number of electrons to freeze (%d) must be >= 0." % args.frozen)

    if args.epsilon < 0:
        raise ValueError(
            "The imaginary time step (%f) must be > 0." % args.epsilon)

    return args


if __name__ == "__main__":
    main()
