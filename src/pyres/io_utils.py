#!/usr/bin/env python2
"""
Subroutines for file I/O operations
"""
import numpy
import sparse_utils


def setup_results(buf_len, res_dir, calc_ray, fciqmc, shift_int):
    """Initialize the file handles for writing results and their corresponding
        buffer arrays.

        Parameters
        ----------
        buf_len : (unsigned int)
            The number of iterations between writing results to disk.
        res_dir : (str)
            The directory in which results should be saved
        calc_ray : (bool)
            A flag that specifies whether to save the Rayleigh quotient
        fciqmc : (bool)
            A flag that specifies whether this is a fciqmc calculation, and
            therefore whether to save the energy shift, number of walkers, and
            solution vector sparsity.
        shift_int : (unsigned int)
            For fciqmc calculations, the number of iterations between updates
            to the energy shift.
    """
    r_dict = {}
    if fciqmc:
        r_dict['shift'] = [open(res_dir + 'S.txt', 'ab', 0),
                           numpy.zeros(buf_len / shift_int)]
        r_dict['n_walk'] = [open(res_dir + 'N.txt', 'ab', 0),
                            numpy.zeros(buf_len / shift_int, dtype=numpy.int32)]
        r_dict['sparsity'] = [open(res_dir + 'sparsity.txt', 'ab', 0),
                              numpy.zeros(buf_len / shift_int, dtype=numpy.int32)]
        r_dict['shift_int'] = shift_int
    if calc_ray:
        r_dict['ray_num'] = [open(res_dir + 'raynum.txt', 'ab', 0),
                             numpy.zeros(buf_len / shift_int)]
        r_dict['ray_den'] = [open(res_dir + 'rayden.txt', 'ab', 0),
                             numpy.zeros(buf_len / shift_int)]
    r_dict['proj_num'] = [open(res_dir + 'projnum.txt', 'ab', 0),
                          numpy.zeros(buf_len)]
    r_dict['proj_den'] = [open(res_dir + 'projden.txt', 'ab', 0),
                          numpy.zeros(buf_len)]
    r_dict['buf_len'] = buf_len
    r_dict['dets_file'] = res_dir + 'sol_dets.npy'
    r_dict['vals_file'] = res_dir + 'sol_vals.npy'
    return r_dict


def calc_results(r_dict, vec_dets, vec_vals, shift, iter_num, hf_dets, hf_matrel):
    """Estimate the correlation energy from the current iterate and write results
        to file, if necessary.

        Parameters
        ----------
        r_dict : (dict)
            dictionary containing results arrays, file handles, and other info,
            generated by the setup_results subroutine
        vec_dets : (numpy.ndarray, int)
            Slater determinant indices in sparse representation of current iterate
        vec_vals : (numpy.ndarray, float)
            values in sparse representation of current iterate
        shift : (float)
            for FCIQMC calculations, the current value of the energy shift
        iter_num : (unsigned int)
            index of the current iteration in the trajectory
        hf_dets :  (numpy.ndarray, int64)
            bit string representations of Slater determinants corresponding to nonzero
            elements in the HF column of the FCI matrix, including the HF determinant
            itself
        hf_matrel : (numpy.ndarray, float)
            Nonzero FCI matrix elements in the HF column of the FCI matrix corresponding
            to hf_dets
    """
    buf_len = r_dict['buf_len']
    shift_int = r_dict['shift_int']
    res_idx = iter_num % buf_len
    if vec_dets[0] != hf_dets[0]:
        r_dict['proj_den'][1][res_idx] = 0
    else:
        r_dict['proj_den'][1][res_idx] = vec_vals[0]
    r_dict['proj_num'][1][res_idx] = sparse_utils.dot_vectors(
        vec_dets, hf_dets, vec_vals, hf_matrel, sorted1=True, sorted2=True)  # check this
    print(iter_num, r_dict['proj_num'][1]
          [res_idx] / r_dict['proj_den'][1][res_idx])

    if 'shift' in r_dict:
        r_dict['shift'][1][res_idx / shift_int] = shift
    if 'n_walk' in r_dict:
        r_dict['n_walk'][1][res_idx /
                            shift_int] = numpy.sum(numpy.abs(vec_vals))
    if 'sparsity' in r_dict:
        r_dict['sparsity'][1][res_idx / shift_int] = vec_dets.shape[0]

    if (iter_num + 1) % buf_len:
        numpy.savetxt(r_dict['proj_num'][0], r_dict['proj_num'][1])
        numpy.savetxt(r_dict['proj_den'][0], r_dict['proj_den'][1])
        numpy.save(r_dict['dets_file'], vec_dets)
        numpy.save(r_dict['vals_file'], vec_vals)
        if 'shift' in r_dict:
            numpy.savetxt(r_dict['shift'][0], r_dict['shift'][1])
        if 'n_walk':
            numpy.savetxt(r_dict['n_walk'][0], r_dict['n_walk'][1])
        if 'sparsity':
            numpy.savetxt(r_dict['sparsity'][0], r_dict['sparsity'][1])