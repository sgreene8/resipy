#!/usr/bin/env python2
"""
Script for running an FCIQMC calculation.
For help, run  python fciqmc.py -h
"""

import numpy
import argparse
from pyres import fci_utils
from pyres import sparse_utils
from pyres import near_uniform
from pyres import compress_utils

def main():
    args = _parse_args()
    
#    h_core, eris, symm, hf_en = _read_in_hf(args)


def _read_in_hf(arg_dict):
    """
    Read in and process the output files from a pyscf HF calculation
    """
    h_core = numpy.load(arg_dict.hf_path + 'hcore.npy')

    eris = numpy.load(arg_dict.hf_path + 'eris.npy')
    
    symm = numpy.load(arg_dict.hf_path + 'symm.npy')
    symm = symm[(arg_dict.frozen / 2):]
    
#    hf_en = 


def _parse_args():
    """
    Subroutine to parse command line arguments
    """

    parser = argparse.ArgumentParser(description=
                                     "Perform an FCIQMC Calculation")
    parser.add_argument('hf_path', type=str, help="Path to the directory that contains the HF output files eris.npy, hcore.npy, symm.npy, and hf_en.txt")
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
    parser.add_argument('-a', '--interval', type=int, default=10.,
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
    
    args = parser.parse_args()
    # process arguments and perform error checking
    
    if args.hf_path[-1] != '/':
        args.hf_path += '/'
    
    if args.n_elec <= 0:
        raise ValueError("Number of electrons in the system (%d) must be > 0" % args.n_elec)
    
    if args.frozen > args.n_elec:
        raise ValueError("Number of core electrons to freeze (%d) is greater than number of electrons in the system (%d)." % (args.frozen, args.n_elec))
    
    if args.frozen < 0:
        raise ValueError("Number of electrons to freeze (%d) must be >= 0." % args.frozen)
    
    if args.epsilon < 0:
        raise ValueError("The imaginary time step (%f) must be > 0." % args.epsilon)
    
    if args.walker_target <= 0:
        raise ValueError("The target number of walkers (%d) must be > 0." % args.walker_target)
    
    return args

if __name__ == "__main__":
    main()
