#!/usr/bin/env python2
"""
    Script for applying the Power method with deterministic compression
    based on the magnitudes of vector element.
"""

import numpy
import argparse
from resipy import fci_utils
from resipy import fci_c_utils
from resipy import sparse_vector
from resipy import compress_utils
from resipy import io_utils


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
    if args.restart:
        ini_idx = numpy.load(args.restart + 'vec_idx.npy')
        ini_val = numpy.load(args.restart + 'vec_val.npy').astype(numpy.float64)
        
        cmp_idx = compress_utils.deterministic(ini_val, args.sparsity)
        ini_idx = ini_idx[cmp_idx]
        ini_val = ini_val[cmp_idx]
    else:
        ini_idx = numpy.array([hf_det], dtype=numpy.int64)
        ini_val = numpy.array([1.])

    sol_vec = sparse_vector.SparseVector(ini_idx, ini_val)
    occ_orbs = fci_c_utils.gen_orb_lists(sol_vec.indices, args.n_elec - args.frozen,
                                         byte_nums, byte_idx)

    results = io_utils.setup_results(args.result_int, args.result_dir,
                                     args.rayleigh, 0, 'all')

    # Elements in the HF column of FCI matrix
    hf_col = fci_utils.gen_hf_ex(
        hf_det, occ_orbs[0], n_orb, symm, eris, args.frozen)

    for iterat in range(args.max_iter):
        # Choose all double excitations
        doub_orbs, doub_idx = fci_c_utils.all_doub_ex(
            sol_vec.indices, occ_orbs, symm)
        doub_probs = numpy.ones_like(doub_idx, dtype=numpy.float64)
        # Choose all single excitations
        sing_orbs, sing_idx = fci_c_utils.all_sing_ex(
            sol_vec.indices, occ_orbs, symm)
        sing_probs = numpy.ones_like(sing_idx, dtype=numpy.float64)

        mat_eval = doub_probs.shape[0] + sing_probs.shape[0]

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
            occ_orbs, h_core, eris, args.frozen) - hf_en
        diag_vals = 1 - args.epsilon * (diag_matrel)
        diag_vals *= sol_vec.values
        next_vec = sparse_vector.SparseVector(sol_vec.indices, diag_vals)

        # Add vectors in sparse format
        next_vec.add(spawn_dets, spawn_vals)
        io_utils.calc_results(results, next_vec, 0, iterat, hf_col, mat_eval)

        cmp_idx = compress_utils.deterministic(next_vec.values, args.sparsity)
        cmp_dets = next_vec.indices[cmp_idx]
        cmp_vals = next_vec.values[cmp_idx]
        sol_vec = sparse_vector.SparseVector(cmp_dets, cmp_vals)
        sol_vec.values /= sol_vec.one_norm()
        occ_orbs = fci_c_utils.gen_orb_lists(sol_vec.indices, args.n_elec - args.frozen,
                                             byte_nums, byte_idx)


def _parse_args():
    """
    Subroutine to parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Power iteration with deterministic compression")
    parser.add_argument(
        'hf_path', type=str, help="Path to the directory that contains the HF output files eris.npy, hcore.npy, symm.npy, and hf_en.txt")
    parser.add_argument('n_elec', type=int,
                        help="Number of electrons in the molecule")
    parser.add_argument('epsilon', type=float, help="Imaginary time step")
    parser.add_argument('sparsity', type=int,
                        help="Target number of nonzero elements in the solution vector")
    parser.add_argument('-f', '--frozen', type=int, default=0,
                        help="Number of core electrons frozen")
    parser.add_argument('-p', '--procs', type=int, default=8,
                        help="Number of processors to use for multithreading")
    parser.add_argument('-r', '--result_int', type=int, default=1000,
                        help="Period with which to write results to disk")
    parser.add_argument('-y', '--result_dir', type=str, default=".",
                        help="Directory in which to save output files")
    parser.add_argument('--rayleigh', type=int, default=0,
                        help="Interval at which to calculate exact quadratic rayleigh quotient (0 means it is not calculated).")
    parser.add_argument('-i', '--max_iter', type=int, default=800000,
                        help="Number of iterations to simulate in the trajectory.")
    parser.add_argument('-l', '--restart', type=str,
                        help="Directory from which to load the vec_idx.npy and vec_val.npy files to initialize the solution vector.")

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

    return args


def _describe_args(arg_dict):
    path = arg_dict.result_dir + 'params.txt'
    with open(path, "w") as file:
        file.write("Power iteration with deterministic compression\n")
        file.write("HF path: " + arg_dict.hf_path)
        file.write("\nepsilon (imaginary time step): {}\n".format(arg_dict.epsilon))
        file.write("sparsity: {}\n".format(arg_dict.sparsity))
        file.write("Rayleigh quotient interval: {}\n".format(arg_dict.rayleigh))


if __name__ == "__main__":
    main()
