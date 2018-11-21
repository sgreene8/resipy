#!/usr/bin/env python2
"""
Script for running an FCIQMC calculation.
For help, run  python fri.py -h
"""

import numpy
import argparse
from resipy import fci_utils
from resipy import fci_c_utils
from resipy import sparse_vector
from resipy import compress_utils
from resipy import io_utils
from resipy import near_uniform


def main():
    args = _parse_args()
    _describe_args(args)

    h_core, eris, symm, hf_en = io_utils.read_in_hf(args.hf_path, args.frozen)

    # Generate lookup tables for later use
    byte_nums, byte_idx = fci_utils.gen_byte_table()
    symm_lookup = fci_utils.gen_symm_lookup(8, symm)

    n_orb = symm.shape[0]
    hf_det = fci_utils.gen_hf_bitstring(n_orb, args.n_elec - args.frozen)

    # Initialize solution vector
    ini_idx = numpy.array([hf_det], dtype=numpy.int64)
    sol_vec = sparse_vector.SparseVector(ini_idx, numpy.array([1.]))
    occ_orbs = fci_c_utils.gen_orb_lists(sol_vec.indices, 2 * n_orb, args.n_elec -
                                         args.frozen, byte_nums, byte_idx)
    last_norm = 1.
    en_shift = 0.

    results = io_utils.setup_results(args.result_int, args.result_dir,
                                     args.rayleigh, False)

    if args.sampl_mode != "all":
        n_doub_ref = fci_utils.count_doubex(occ_orbs[0], symm, symm_lookup)
        n_sing_ref = fci_utils.count_singex(
            hf_det, occ_orbs[0], symm, symm_lookup)
        p_doub = n_doub_ref * 1.0 / (n_sing_ref + n_doub_ref)
    if args.sampl_mode != "all" and args.dist == "heat-bath_PP":
        from resipy import heat_bath
        occ1_probs, occ2_probs, exch_probs = heat_bath.set_up(args.frozen, eris)
    if args.sampl_mode == "fri" and args.dist == "near_uniform":
        from resipy import fri_near_uni

    rngen_ptrs = near_uniform.initialize_mt(args.procs)

    # Elements in the HF column of FCI matrix
    hf_col = fci_utils.gen_hf_ex(
        hf_det, occ_orbs[0], n_orb, symm, eris, args.frozen)

    for iterat in range(args.max_iter):
        if args.sampl_mode == "all":
            # Choose all double excitations
            doub_orbs, doub_idx = fci_c_utils.all_doub_ex(
                sol_vec.indices, occ_orbs, symm)
            doub_probs = numpy.ones_like(doub_idx, dtype=numpy.float64)
            # Choose all single excitations
            sing_orbs, sing_idx = fci_c_utils.all_sing_ex(
                sol_vec.indices, occ_orbs, symm)
            sing_probs = numpy.ones_like(sing_idx, dtype=numpy.float64)
        elif args.sampl_mode == "multinomial":
            n_col, = compress_utils.sys_resample(numpy.abs(sol_vec.values), args.H_sample, ret_counts=True)
            n_doub_col, n_sing_col = near_uniform.bin_n_sing_doub(
                n_col, p_doub)

        if args.sampl_mode == "multinomial":
            # Sample single excitations
            sing_orbs, sing_probs, sing_idx = near_uniform.sing_multin(
                sol_vec.indices, occ_orbs, symm, symm_lookup, n_sing_col, rngen_ptrs)
            sing_probs *= (1 - p_doub) * n_col[sing_idx]
        if args.dist == "near_uniform" and args.sampl_mode == "multinomial":
            # Sample double excitations
            doub_orbs, doub_probs, doub_idx = near_uniform.doub_multin(
                sol_vec.indices, occ_orbs, symm, symm_lookup, n_doub_col, rngen_ptrs)
            doub_probs *= p_doub * n_col[doub_idx]
        elif args.dist == "near_uniform" and args.sampl_mode == "fri":
            # Compress both excitations
            doub_orbs, doub_probs, doub_idx, sing_orbs, sing_probs, sing_idx = fri_near_uni.cmp_hier(sol_vec, args.H_sample, p_doub,
                                                                                                     occ_orbs, symm, symm_lookup)
        elif args.dist == "heat-bath_PP" and args.sampl_mode == "multinomial":
            # Sample double excitations
            doub_orbs, doub_probs, doub_idx = heat_bath.doub_multin(
                occ1_probs, occ2_probs, exch_probs, sol_vec.indices, occ_orbs, symm, symm_lookup, n_doub_col, rngen_ptrs)
            doub_probs *= p_doub * n_col[doub_idx]
        elif args.dist == "heat-bath_PP" and args.sampl_mode == "fri":
            pass
        else:
            raise RuntimeError("This sampling mode is not yet implemented for the specified distribution.")

        doub_matrel = fci_c_utils.doub_matr_el_nosgn(
            doub_orbs, eris, args.frozen)
        # Retain nonzero elements
        doub_nonz = doub_matrel != 0
        doub_idx = doub_idx[doub_nonz]
        doub_orbs = doub_orbs[doub_nonz]
        doub_matrel = doub_matrel[doub_nonz]
        doub_probs = doub_probs[doub_nonz]
        # Calculate determinants and matrix elements
        doub_dets, doub_signs = fci_utils.doub_dets_parity(
            sol_vec.indices[doub_idx], doub_orbs)
        # origin_idx = numpy.searchsorted(deter_dets, sol_vec.indices[doub_idx])
        doub_matrel *= args.epsilon / doub_probs * \
            doub_signs * -sol_vec.values[doub_idx]
        # Start forming next iterate
        spawn_dets = doub_dets
        spawn_vals = doub_matrel

        sing_dets, sing_matrel = fci_c_utils.single_dets_matrel_nosgn(
            sol_vec.indices[sing_idx], sing_orbs, eris, h_core, occ_orbs[sing_idx], args.frozen)
        # Retain nonzero elements
        sing_nonz = sing_matrel != 0
        sing_idx = sing_idx[sing_nonz]
        sing_dets = sing_dets[sing_nonz]
        sing_matrel = sing_matrel[sing_nonz]
        sing_probs = sing_probs[sing_nonz]
        sing_orbs = sing_orbs[sing_nonz]
        # Calculate determinants and matrix elements
        sing_signs = fci_c_utils.excite_signs(sing_orbs[:, 1], sing_orbs[:, 0], sing_dets)
        sing_matrel *= args.epsilon / sing_probs * -sol_vec.values[sing_idx] * sing_signs
        # Add to next iterate
        spawn_dets = numpy.append(spawn_dets, sing_dets)
        spawn_vals = numpy.append(spawn_vals, sing_matrel)

        # Diagonal matrix elements
        diag_matrel = fci_c_utils.diag_matrel(
            occ_orbs, h_core, eris, args.frozen) - hf_en - en_shift
        diag_matrel = 1 - args.epsilon * diag_matrel
        diag_matrel *= sol_vec.values
        next_vec = sparse_vector.SparseVector(sol_vec.indices, diag_matrel)

        # Add vectors in sparse format
        next_vec.add(spawn_dets, spawn_vals)
        one_norm = next_vec.one_norm()
        io_utils.calc_results(results, next_vec, 0, iterat, hf_col)
        # next_vec.values /= one_norm
        en_shift -= 0.05 * numpy.log(one_norm / last_norm)
        last_norm = one_norm

        cmp_idx, cmp_vals = compress_utils.fri_1D(next_vec.values, args.sparsity)
        cmp_dets = next_vec.indices[cmp_idx]
        sol_vec = sparse_vector.SparseVector(cmp_dets, cmp_vals)
        occ_orbs = fci_c_utils.gen_orb_lists(cmp_dets, 2 * n_orb, args.n_elec -
                                             args.frozen, byte_nums, byte_idx)


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
    parser.add_argument('--H_sample', type=int,
                        help="Total number of off-diagonal samples to draw from the Hamiltonian matrix")
    parser.add_argument('sparsity', type=int,
                        help="Target number of nonzero elements in the solution vector")
    parser.add_argument('sampl_mode', choices=["all", "multinomial", "fri"],
                        help="Method for sampling off-diagonal Hamiltonian elements.")
    parser.add_argument('--dist', choices=["near_uniform", "heat-bath_PP"],
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
        args.epsilon *= -1

    if args.sampl_mode != "all" and args.dist is None:
        raise ValueError("The probability distribution must be specified if sample_mode is not 'all'.")

    if args.sampl_mode != "all" and args.H_sample is None:
        raise ValueError("The number of off-diagonal Hamiltonian elements to sample must be specified if sample_mode is not 'all'.")

    return args


def _describe_args(arg_dict):
    path = arg_dict.result_dir + 'params.txt'
    with open(path, "w") as file:
        file.write("FRI calculation\n")
        file.write("HF path: " + arg_dict.hf_path)
        file.write("\nepsilon (imaginary time step): {}\n".format(arg_dict.epsilon))
        file.write("sparsity: {}\n".format(arg_dict.sparsity))
        file.write("Sampling mode: {}\n".format(arg_dict.sampl_mode))
        if arg_dict.sampl_mode != "all":
            file.write("Number of matrix samples: {}\n".format(arg_dict.H_sample))


if __name__ == "__main__":
    main()
