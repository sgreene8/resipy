#!/usr/bin/env python2
"""
Subroutines for file I/O operations
"""
import numpy
from resipy import fci_c_utils
from resipy import fci_utils
from resipy import misc_c_utils
import sys


def setup_results(buf_len, res_dir, ray_int, shift_int, samp_mode):
    """Initialize the file handles for writing results and their corresponding
        buffer arrays.

    Parameters
    ----------
    buf_len : (unsigned int)
        The number of iterations between writing results to disk.
    res_dir : (str)
        The directory in which results should be saved
    ray_int : (unsigned int)
        The interval (number of steps) at which the Rayleigh quotient is to be calculated.
    fciqmc : (bool)
        A flag that specifies whether this is a fciqmc calculation, and
        therefore whether to save the number of walkers and solution
        vector sparsity.
    shift_int : (unsigned int)
        The number of iterations between updates
        to the energy shift.
    samp_mode: (str)
        The method by which the Hamiltonian matrix is compressed ('fciqmc', 'multinomial',
        'fri', 'fri_strat', or 'all'). If 'fciqmc', the number of walkers and sparsity will
        be saved. If 'all', the number of matrix evalauations will be saved.
    """
    r_dict = {}
    if samp_mode == 'fciqmc':
        r_dict['n_walk'] = [open(res_dir + 'N.txt', 'ab', 0),
                            numpy.zeros(buf_len / shift_int, dtype=numpy.int32)]
        r_dict['sparsity'] = [open(res_dir + 'sparsity.txt', 'ab', 0),
                              numpy.zeros(buf_len / shift_int, dtype=numpy.int32)]
    r_dict['ray_int'] = ray_int
    if ray_int != 0:
        r_dict['ray_num'] = [open(res_dir + 'raynum.txt', 'ab', 0),
                             numpy.zeros(buf_len / ray_int)]
        r_dict['ray_den'] = [open(res_dir + 'rayden.txt', 'ab', 0),
                         numpy.zeros(buf_len / ray_int)]
    if samp_mode  == 'all':
        r_dict['mat_eval'] = [open(res_dir + 'mat_eval.txt', 'ab', 0),
                              numpy.zeros(buf_len, dtype=numpy.uint32)]

    r_dict['shift'] = [open(res_dir + 'S.txt', 'ab', 0),
                       numpy.zeros(buf_len / shift_int)]
    r_dict['shift_int'] = shift_int
    r_dict['proj_num'] = [open(res_dir + 'projnum.txt', 'ab', 0),
                          numpy.zeros(buf_len)]
    r_dict['proj_den'] = [open(res_dir + 'projden.txt', 'ab', 0),
                          numpy.zeros(buf_len)]
    r_dict['buf_len'] = buf_len
    r_dict['vec_file'] = res_dir + 'vec'
    return r_dict


def calc_ray_quo(r_dict, sol_vec, occ_orbs, symm, iter_num, diag_el, h_core, eris, n_frozen):
    """Calculate the quadratic rayleigh quotient using the current and previous iterates.

    Parameters
    ----------
    r_dict : (dict)
        dictionary containing results arrays, file handles, and other info,
        generated by the setup_results subroutine
    sol_vec : (SparseVector)
        Vector for which to calculate the quadratic rayleigh quotient
    occ_orbs : (numpy.ndarray, uint8)
        orbitals occupied in each determinant in sol_vec
    symm : (numpy.ndarray, uint8)
        Irreducible representations of the spatial orbitals in the basis
    iter_num : (unsigned int)
        index of the current iteration in the trajectory
    diag_el : (numpy.ndarray, float64)
        diagonal Hamiltonian matrix elements corresponding to each of the determinants
        in the solution vector
    hcore : (numpy.ndarray, float64)
        1-electron integrals from  Hartree-Fock calculation
    eris : (numpy.ndarray, float64)
        2-electron integrals from Hartree-Fock calculation
    n_frozen : (unsigned int)
        number of core electrons frozen in the calculation
    """
    buf_len = r_dict['buf_len']
    ray_int = r_dict['ray_int']
    res_idx = (iter_num % buf_len) / ray_int

    self_overlap = numpy.linalg.norm(sol_vec.values)**2
    r_dict['ray_den'][1][res_idx] = self_overlap

    numer = fci_c_utils.ray_off_diag(sol_vec.indices, sol_vec.values.astype(numpy.float64), occ_orbs,
                                     h_core, eris, n_frozen, symm) + numpy.sum(diag_el * sol_vec.values**2)
    r_dict['ray_num'][1][res_idx] = numer


def check_ray_quo(sol_vec, occ_orbs, symm, diag_el, h_core, eris, n_frozen):
    self_overlap = numpy.linalg.norm(sol_vec.values)**2
    off_diag, num = fci_c_utils.ray_off_diag(sol_vec.indices, sol_vec.values.astype(numpy.float64), occ_orbs,
                                        h_core, eris, n_frozen, symm)
    num_diag = numpy.sum(diag_el * sol_vec.values**2)
    return num_diag, off_diag, self_overlap, num


def calc_results(r_dict, vec, shift, iter_num, hf_col, mat_eval=0):
    """Estimate the correlation energy from the current iterate and write results
        to file, if necessary.

    Parameters
    ----------
    r_dict : (dict)
        dictionary containing results arrays, file handles, and other info,
        generated by the setup_results subroutine
    vec : (SparseVector)
        Current iterate
    shift : (float)
        for FCIQMC calculations, the current value of the energy shift
    iter_num : (unsigned int)
        index of the current iteration in the trajectory
    hf_col : (SparseVector)
        HF column of the FCI matrix, including the HF determinant itself
    mat_eval : (unsigned int)
        Number of off-diagonal Hamiltonian matrix elements sampled in most recent iteration.
    """
    buf_len = r_dict['buf_len']
    res_idx = iter_num % buf_len
    if vec.indices[0] != hf_col.indices[0]:
        r_dict['proj_den'][1][res_idx] = 0
    else:
        r_dict['proj_den'][1][res_idx] = vec.values[0]
    r_dict['proj_num'][1][res_idx] = vec.dot(hf_col)
    print(iter_num, r_dict['proj_num'][1]
          [res_idx] / r_dict['proj_den'][1][res_idx])

    shift_int = r_dict['shift_int']
    if ((iter_num + 1) % shift_int) == 0:
        r_dict['shift'][1][res_idx / shift_int] = shift
        if 'n_walk' in r_dict:
            r_dict['n_walk'][1][res_idx /
                                shift_int] = vec.one_norm()
        if 'sparsity' in r_dict:
            r_dict['sparsity'][1][res_idx / shift_int] = vec.indices.shape[0]

    if 'mat_eval' in r_dict:
        r_dict['mat_eval'][1][res_idx] = mat_eval

    if ((iter_num + 1) % buf_len) == 0:
        sys.stdout.flush()
        numpy.savetxt(r_dict['proj_num'][0], r_dict['proj_num'][1])
        numpy.savetxt(r_dict['proj_den'][0], r_dict['proj_den'][1])
        vec.save(r_dict['vec_file'])
        if 'shift' in r_dict:
            numpy.savetxt(r_dict['shift'][0], r_dict['shift'][1])
        if 'n_walk' in r_dict:
            numpy.savetxt(r_dict['n_walk'][0], r_dict['n_walk'][1])
        if 'sparsity' in r_dict:
            numpy.savetxt(r_dict['sparsity'][0], r_dict['sparsity'][1])
        if 'ray_num' in r_dict:
            numpy.savetxt(r_dict['ray_num'][0], r_dict['ray_num'][1])
        if 'ray_den' in r_dict:
            numpy.savetxt(r_dict['ray_den'][0], r_dict['ray_den'][1])
        if 'mat_eval' in r_dict:
            numpy.savetxt(r_dict['mat_eval'][0], r_dict['mat_eval'][1])


def read_in_hf(hf_path, n_frozen):
    """Read in and process the output files from a pyscf HF calculation

        Parameters
        ----------
        hf_path : (str)
            Directory containing output files from the HF calculation
        n_frozen : (unsigned int)
            Desired number of electrons to be frozen in the calculation

        Returns
        -------
        (numpy.ndarray) :
            2-D array containing the 1-electron integrals in the spatial orbital basis
        (numpy.ndarray) :
            4-D array containing the 2-electron integrals in the spatial orbital basis
        (numpy.ndarray) :
            1-D array containing the irreducible representations of each spatial orbital
        (float) :
            Hartree-Fock electronic energy
    """
    h_core = numpy.load(hf_path + 'hcore.npy')

    eris = numpy.load(hf_path + 'eris.npy')

    symm = numpy.load(hf_path + 'symm.npy')
    symm = symm[(n_frozen / 2):]

    hf_en = numpy.genfromtxt(hf_path + 'hf_en.txt')

    return h_core, eris, symm, hf_en
