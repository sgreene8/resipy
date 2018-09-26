#!/usr/bin/env python2
"""
Script for running an FCIQMC calculation.
For help, run  python fciqmc.py -h
"""

import numpy
import argparse
from pyres import fci_utils
from pyres import fci_c_utils
from pyres import sparse_utils
from pyres import near_uniform
from pyres import compress_utils
from pyres import io_utils


def main():
    args = _parse_args()

    h_core, eris, symm, hf_en = _read_in_hf(args)

    # Generate lookup tables for later use
    byte_nums, byte_idx = fci_utils.gen_byte_table()
    symm_lookup = fci_utils.gen_symm_lookup(8, symm)

    n_orb = symm.shape[0]
    hf_det = fci_utils.gen_hf_bitstring(n_orb, args.n_elec - args.frozen)

    # Initialize solution vector
    sol_dets = numpy.array([hf_det], dtype=numpy.int64)
    sol_vals = numpy.array([args.walkers], dtype=numpy.int32)
    occ_orbs = fci_c_utils.gen_orb_lists(sol_dets, 2 * n_orb, args.n_elec -
                                         args.frozen, byte_nums, byte_idx)
    n_walk = numpy.sum(numpy.abs(sol_vals))  # one-norm of solution vector
    last_nwalk = 0  # number of walkers at the time of previous shift update
    # energy shift for controlling normalization
    en_shift = args.initial_shift

    results = io_utils.setup_results(args.result_int, args.result_dir,
                                     args.rayleigh, True, args.interval)

    if args.prob_dist == "near_uniform":
        n_doub_ref = fci_utils.count_doubex(occ_orbs[0], symm, symm_lookup)
        n_sing_ref = fci_utils.count_singex(
            hf_det, occ_orbs[0], symm, symm_lookup)
        p_doub = n_doub_ref * 1.0 / (n_sing_ref + n_doub_ref)

    rngen_ptrs = near_uniform.initialize_mt(args.procs)
    numpy.random.seed(0)

    # Elements in the HF column of FCI matrix
    hf_col_dets, hf_col_matrel = fci_utils.gen_hf_ex(hf_det, occ_orbs[0], n_orb, symm, eris, args.frozen)

    for iterat in range(args.max_iter):
        # number of samples to draw from each column
        n_col = numpy.abs(sol_vals)

        if args.prob_dist == "near_uniform":
            n_doub_col, n_sing_col = _choose_sing_doub(n_col, p_doub)
            # Sample double excitations
            doub_orbs, doub_probs, doub_idx = _sample_doubles(
                sol_dets, occ_orbs, symm, symm_lookup, n_doub_col, rngen_ptrs)
            # Compress chosen elements
            doub_matrel = fci_utils.doub_matr_el_nosgn(
                doub_orbs, eris, args.frozen)
            doub_matrel *= args.epsilon / doub_probs / p_doub
            doub_matrel = compress_utils.round_binomially(doub_matrel, 1)
            # Retain nonzero elements
            doub_nonz = doub_matrel != 0
            doub_idx = doub_idx[doub_nonz]
            doub_orbs = doub_orbs[doub_nonz]
            doub_matrel = doub_matrel[doub_nonz]
            # Calculate determinants and matrix elements
            doub_dets, doub_signs = fci_utils.doub_dets_parity(
                sol_dets[doub_idx], doub_orbs)
            doub_matrel *= doub_signs * -numpy.sign(sol_vals[doub_idx])
            # Start forming next iterate
            spawn_dets = doub_dets
            spawn_vals = doub_matrel

            # Sample single excitations
            sing_orbs, sing_probs, sing_idx = _sample_singles(
                sol_dets, occ_orbs, symm, symm_lookup, n_sing_col, rngen_ptrs)
            sing_dets, sing_matrel = fci_c_utils.single_dets_matrel(
                sol_dets[sing_idx], sing_orbs, eris, h_core, occ_orbs[sing_idx], args.frozen)
            # Compress chosen elements
            sing_matrel *= args.epsilon / sing_probs / (1 - p_doub)
            sing_matrel = compress_utils.round_binomially(sing_matrel, 1)
            # Retain nonzero elements
            sing_nonz = sing_matrel != 0
            sing_idx = sing_idx[sing_nonz]
            sing_dets = sing_dets[sing_nonz]
            sing_matrel = sing_matrel[sing_nonz]
            # Calculate determinants and matrix elements
            sing_matrel *= -numpy.sign(sol_vals[sing_idx])
            # Add to next iterate
            spawn_dets = numpy.append(spawn_dets, sing_dets)
            spawn_vals = numpy.append(spawn_vals, sing_matrel)
        # Diagonal matrix elements
        diag_matrel = fci_c_utils.diag_matrel(
            occ_orbs, h_core, eris, args.frozen) - en_shift - hf_en
        diag_matrel = 1 - args.epsilon * diag_matrel
        diag_matrel *= numpy.sign(sol_vals)
        diag_matrel = compress_utils.round_binomially(diag_matrel, n_col)
        # Retain nonzero elements
        diag_nonz = diag_matrel != 0
        next_dets = sol_dets[diag_nonz]
        next_vals = diag_matrel[diag_nonz]

        # Add vectors in sparse format
        next_dets, next_vals = sparse_utils.add_vectors(next_dets, spawn_dets, next_vals, spawn_vals,
                                                                 sorted1=True)
        occ_orbs = fci_c_utils.gen_orb_lists(next_dets, 2 * n_orb, args.n_elec -
                                             args.frozen, byte_nums, byte_idx)
        n_walk = numpy.sum(numpy.abs(next_vals))
        en_shift, last_nwalk = adjust_shift(en_shift, n_walk, last_nwalk, args.walker_target, args.damping)
        io_utils.calc_results(results, next_dets, next_vals, en_shift, iterat,
                              hf_col_dets, hf_col_matrel)
        sol_dets = next_dets
        sol_vals = next_vals


def adjust_shift(shift, n_walkers, last_walkers, target_walkers, damp_factor):
    """
    Adjust the constant energy shift used to control normalization
    """
    if last_walkers:
        shift -= damp_factor * numpy.log(float(n_walkers) / last_walkers)
        last_walkers = n_walkers
    if not(last_walkers) and n_walkers > target_walkers:
        last_walkers = n_walkers
    return shift, last_walkers


def _sample_singles(vec_dets, vec_occ, orb_symm, symm_lookup, n_col, rn_vec):
    """
    For the near-uniform distribution, multinomially choose the single
    excitations for each column.
    """
    det_idx = fci_c_utils.ind_from_count(n_col)
    orb_choices, ex_probs = near_uniform.sing_multin(vec_dets, vec_occ, orb_symm,
                                                     symm_lookup, n_col, rn_vec)
    return orb_choices, ex_probs, det_idx


def _sample_doubles(vec_dets, vec_occ, orb_symm, symm_lookup, n_col, rn_vec):
    """
    For the near-uniform distribution, multinomially choose the double
    excitations for each column.
    """
    det_idx = fci_c_utils.ind_from_count(n_col)
    orb_choices, ex_probs = near_uniform.doub_multin(vec_dets, vec_occ, orb_symm,
                                                     symm_lookup, n_col, rn_vec)
    successes = ex_probs > 0
    orb_choices = orb_choices[successes]
    ex_probs = ex_probs[successes]
    det_idx = det_idx[successes]

    return orb_choices, ex_probs, det_idx


def _choose_sing_doub(col_nsamp, p_doub):
    """
    For the near-uniform distribution, binomially partition the samples for
    each column into single and double excitations.
    """

    doub_samp = numpy.random.binomial(col_nsamp, p_doub)
    doub_samp = doub_samp.astype(numpy.uint32)
    sing_samp = col_nsamp - doub_samp
    sing_samp = sing_samp.astype(numpy.uint32)
    return doub_samp, sing_samp


def _read_in_hf(arg_dict):
    """
    Read in and process the output files from a pyscf HF calculation
    """
    h_core = numpy.load(arg_dict.hf_path + 'hcore.npy')

    eris = numpy.load(arg_dict.hf_path + 'eris.npy')

    symm = numpy.load(arg_dict.hf_path + 'symm.npy')
    symm = symm[(arg_dict.frozen / 2):]

    hf_en = numpy.genfromtxt(arg_dict.hf_path + 'hf_en.txt')

    return h_core, eris, symm, hf_en


def _parse_args():
    """
    Subroutine to parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Perform an FCIQMC Calculation")
    parser.add_argument(
        'hf_path', type=str, help="Path to the directory that contains the HF output files eris.npy, hcore.npy, symm.npy, and hf_en.txt")
    parser.add_argument('n_elec', type=int,
                        help="Number of electrons in the molecule")
    parser.add_argument('epsilon', type=float, help="Imaginary time step")
    parser.add_argument('walker_target', type=int,
                        help="Target number of walkers, must be greater than the plateau value for this system")
    parser.add_argument('prob_dist', choices=["near_uniform"],
                        help="Probability distribution to use to select Hamiltonian off-diagonal elements")
    parser.add_argument('-f', '--frozen', type=int, default=0,
                        help="Number of core electrons frozen")
    parser.add_argument('-s', '--initial_shift', type=float, default=0.,
                        help="Initial energy shift (S) for controlling normalization")
    parser.add_argument('-a', '--interval', type=int, default=10,
                        help="Period with which to update the energy shift (A).")
    parser.add_argument('-d', '--damping', type=float, default=0.05,
                        help="Damping parameter for shift updates (xi)")
    parser.add_argument('-p', '--procs', type=int, default=8,
                        help="Number of processors to use for multithreading")
    parser.add_argument('-w', '--walkers', type=int, default=100,
                        help="Initial number of walkers to put on HF determinant")
    parser.add_argument('-r', '--result_int', type=int, default=1000,
                        help="Period with which to write results to disk")
    parser.add_argument('-y', '--result_dir', type=str, default=".",
                        help="Directory in which to save output files")
    parser.add_argument('--rayleigh', action="store_true",
                        help="Calculate Rayleigh quotient every A iterations")
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

    if args.walker_target <= 0:
        raise ValueError(
            "The target number of walkers (%d) must be > 0." % args.walker_target)

    if args.result_int % args.interval != 0:
        raise ValueError("The interval for saving results (%d) must be an integer multiple of the interval for updating the shift (%d)." % (
            args.result_int, args.interval))

    return args


if __name__ == "__main__":
    main()
