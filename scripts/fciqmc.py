#!/usr/bin/env python2
"""
Script for running an FCIQMC calculation.
For help, run  python fciqmc.py -h
"""

import numpy
import argparse
from resipy import fci_utils
from resipy import fci_c_utils
from resipy import sparse_vector
from resipy import near_uniform
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
        vec_val = numpy.load(args.restart + 'vec_val.npy')
        en_shift = numpy.genfromtxt(args.restart + 'S.txt')[-1]
        last_nwalk = numpy.abs(vec_val).sum()
    else:
        ini_idx = numpy.array([hf_det], dtype=numpy.int64)
        ini_val = numpy.array([args.walkers], dtype=numpy.int32)
        # energy shift for controlling normalization
        en_shift = args.initial_shift
        last_nwalk = 0  # number of walkers at the time of previous shift update

    sol_vec = sparse_vector.SparseVector(ini_idx, ini_val)
    occ_orbs = fci_c_utils.gen_orb_lists(sol_vec.indices, 2 * n_orb, args.n_elec -
                                         args.frozen, byte_nums, byte_idx)

    results = io_utils.setup_results(args.result_int, args.result_dir,
                                     args.rayleigh, True, args.interval)

    n_doub_ref = fci_utils.count_doubex(occ_orbs[0], symm, symm_lookup)
    n_sing_ref = fci_utils.count_singex(
        hf_det, occ_orbs[0], symm, symm_lookup)
    p_doub = n_doub_ref * 1.0 / (n_sing_ref + n_doub_ref)
    if args.prob_dist == "heat-bath_PP":
        from resipy import heat_bath
        occ1_probs, occ2_probs, exch_probs = heat_bath.set_up(args.frozen, eris)

    rngen_ptrs = near_uniform.initialize_mt(args.procs)

    # Elements in the HF column of FCI matrix
    hf_col = fci_utils.gen_hf_ex(
        hf_det, occ_orbs[0], n_orb, symm, eris, args.frozen)

    for iterat in range(args.max_iter):
        # number of samples to draw from each column
        n_col = numpy.abs(sol_vec.values)

        n_doub_col, n_sing_col = near_uniform.bin_n_sing_doub(
            n_col, p_doub)
        # Sample single excitations
        sing_orbs, sing_probs, sing_idx = near_uniform.sing_multin(
            sol_vec.indices, occ_orbs, symm, symm_lookup,  n_sing_col, rngen_ptrs)

        # Sample double excitations
        if args.prob_dist == "near_uniform":
            doub_orbs, doub_probs, doub_idx = near_uniform.doub_multin(
                sol_vec.indices, occ_orbs, symm, symm_lookup, n_doub_col, rngen_ptrs)
        elif args.prob_dist == "heat-bath_PP":
            doub_orbs, doub_probs, doub_idx = heat_bath.doub_multin(
                occ1_probs, occ2_probs, exch_probs, sol_vec.indices, occ_orbs, symm, symm_lookup, n_doub_col, rngen_ptrs)
        else:
            print("Invalid probability distribution chosen")

        # Compress double elements
        doub_matrel = fci_c_utils.doub_matr_el_nosgn(doub_orbs, eris, args.frozen)
        doub_matrel *= args.epsilon / doub_probs / p_doub
        doub_matrel = compress_utils.round_bernoulli(doub_matrel, rngen_ptrs)

        doub_nonz = doub_matrel != 0
        doub_idx = doub_idx[doub_nonz]
        doub_orbs = doub_orbs[doub_nonz]
        doub_matrel = doub_matrel[doub_nonz]

        doub_dets, doub_signs = fci_utils.doub_dets_parity(
            sol_vec.indices[doub_idx], doub_orbs)
        doub_matrel *= doub_signs * -numpy.sign(sol_vec.values[doub_idx])

        # Compress single elements
        sing_dets, sing_matrel = fci_c_utils.single_dets_matrel_nosgn(
            sol_vec.indices[sing_idx], sing_orbs, eris, h_core, occ_orbs[sing_idx], args.frozen)
        sing_matrel *= args.epsilon / sing_probs / (1 - p_doub)
        sing_matrel = compress_utils.round_bernoulli(sing_matrel, rngen_ptrs)

        sing_nonz = sing_matrel != 0
        sing_idx = sing_idx[sing_nonz]
        sing_dets = sing_dets[sing_nonz]
        sing_matrel = sing_matrel[sing_nonz]
        sing_orbs = sing_orbs[sing_nonz]

        sing_signs = fci_c_utils.excite_signs(sing_orbs[:, 1], sing_orbs[:, 0], sing_dets)
        sing_matrel *= -numpy.sign(sol_vec.values[sing_idx]) * sing_signs

        spawn_dets = numpy.append(doub_dets, sing_dets)
        spawn_vals = numpy.append(doub_matrel, sing_matrel)

        # Diagonal matrix elements
        diag_matrel = fci_c_utils.diag_matrel(
            occ_orbs, h_core, eris, args.frozen) - en_shift - hf_en
        diag_matrel = 1 - args.epsilon * diag_matrel
        diag_matrel *= numpy.sign(sol_vec.values)
        diag_matrel = compress_utils.round_binomially(diag_matrel, n_col, rngen_ptrs)
        # Retain nonzero elements
        diag_nonz = diag_matrel != 0
        next_vec = sparse_vector.SparseVector(
            sol_vec.indices[diag_nonz], diag_matrel[diag_nonz])

        # Add vectors in sparse format
        next_vec.add(spawn_dets, spawn_vals)
        occ_orbs = fci_c_utils.gen_orb_lists(next_vec.indices, 2 * n_orb, args.n_elec -
                                             args.frozen, byte_nums, byte_idx)
        n_walk = next_vec.one_norm()
        if iterat % args.interval == 0:
            en_shift, last_nwalk = _adjust_shift(
                en_shift, n_walk, last_nwalk, args.walker_target, args.damping / args.interval / args.epsilon)

        io_utils.calc_results(results, next_vec, en_shift, iterat, hf_col)
        sol_vec = next_vec


def _adjust_shift(shift, n_walkers, last_walkers, target_walkers, damp_factor):
    """
    Adjust the constant energy shift used to control normalization
    """
    if last_walkers:
        shift -= damp_factor * numpy.log(float(n_walkers) / last_walkers)
        last_walkers = n_walkers
    if not(last_walkers) and n_walkers > target_walkers:
        last_walkers = n_walkers
    return shift, last_walkers


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
    parser.add_argument('prob_dist', choices=["near_uniform", "heat-bath_PP"],
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
    parser.add_argument('-l', '--restart', type=str, help="Directory from which to load the vec_idx.npy and vec_val.npy files to initialize the solution vector.")

    args = parser.parse_args()
    # process arguments and perform error checking

    if args.hf_path[-1] != '/':
        args.hf_path += '/'

    if args.result_dir[-1] != '/':
        args.result_dir += '/'

    if args.restart and args.restart[-1] != '/':
        args.restart += '/'

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

    return args


def _describe_args(arg_dict):
    path = arg_dict.result_dir + 'params.txt'
    with open(path, "w") as file:
        file.write("FCIQMC calculation\n")
        file.write("HF path: " + arg_dict.hf_path + "\n")
        if arg_dict.restart:
            file.write("Restarting calculation from {}\n".format(arg_dict.restart))
        file.write("epsilon (imaginary time step): {}\n".format(arg_dict.epsilon))
        file.write("Target number of walkers: {}\n".format(arg_dict.walker_target))


if __name__ == "__main__":
    main()
