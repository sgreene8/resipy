"""
Utilities for calculating matrix elements of the FCI Hamiltonian and
manipulating bit-string representations of Slater determinants.

A Slater determinant for a system with 2*M Hartree-Fock spin orbitals is
represented as a string of 2*M 1's and 0's. The rightmost M bits correspond to
spin-up orbitals, and the leftmost M to spin-down.

"""

import numpy
import fci_c_utils


def excite_signs(cre_ops, des_ops, bit_strings, num_orb):
    """Calculate the parities of single excitations for an array of bit strings,
        i.e. determine the sign of cre_ops^+ des_ops |bitstrings>. Same as the
        pyscf subroutine pyscf.fci.cistring.cre_des_sign, except can operate on
        numpy vectors.

        Parameters
        ----------
        cre_ops : (numpy.ndarray, unsigned int)
            orbital indices of creation operators
        des_ops : (numpy.ndarray, unsigned int)
            orbital indices of destruction operators
        num_orb : (int)
            number of spin orbitals encoded by each bit string

        Returns
        -------
        (numpy.ndarray, int8)
            signs of excitations, +1 or -1
    """

    credes_max = numpy.maximum(cre_ops, des_ops)
    credes_min = numpy.minimum(cre_ops, des_ops)
    mask = (1 << credes_max) - (1 << (credes_min + 1))
    
    # number of 1's in a binary representation of each number 0-15
    byte_counts = numpy.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4],
                              dtype=numpy.int8)
    # the number of half-bytes (4 bits) encoded in each bit string
    num_hexa = numpy.ceil(num_orb / 4)
    num_hexa = num_hexa.astype(int)
    
    mask &= bit_strings
    
    # divide mask into 4-bit units
    mask.shape = (-1, 1)
    shifts = 4 * numpy.arange(num_hexa)
    mask_split = mask >> shifts
    mask_split &= 15
    
    # count number of 1's in each 4-bit unit
    num_jump = byte_counts[mask_split]
    num_jump = num_jump.sum(axis=1)

    return (-1) ** num_jump


def gen_byte_table():
    """Generate lookup tables used to decompose a byte into a list of positions
        of 1's.
    """

    pos = numpy.zeros((256, 8), dtype=numpy.uint8)
    idx = numpy.linspace(0, 255, num=256, dtype=numpy.uint8)
    idx.shape = (-1, 1)
    bits = numpy.unpackbits(idx, axis=1)
    nums = numpy.sum(bits, axis=1)
    nums = nums.astype(numpy.uint8)
    for i in range(1, 256):
        pos[i, 0:nums[i]] = 7 - numpy.nonzero(bits[i, :])[0][::-1]
    return nums, pos


def gen_symm_lookup(n_symm_el, orb_symm):
    """Generate a list of all spatial orbitals of each irrep in the point
        group.

        Parameters
        ----------
        n_symm_el : (unsigned int)
            number of distinct irreps in this group
        orb_symm : (numpy.ndarray, unsigned int)
            irrep of each orbital

        Returns
        -------
        (numpy.ndarray, unsigned int)
            a matrix with rows indexed by the irrep indices.
            The 0th number in each row is the number of orbitals with this
            irrep. The remaining numbers are the indices of those orbitals.
    """
    symm_cp = orb_symm.copy()
    symm_cp.shape = (-1, 1)
    max_same = numpy.amax(numpy.sum(symm_cp == orb_symm, axis=1))
    symm_table = numpy.zeros((n_symm_el, max_same + 1), dtype=numpy.uint8)
    for i in range(n_symm_el):
        matching_orbs = numpy.nonzero(orb_symm == i)[0]
        n_match = matching_orbs.shape[0]
        symm_table[i, 0] = n_match
        symm_table[i, 1:(1+n_match)] = matching_orbs
    return symm_table


def doub_matr_el_nosgn(chosen_idx, eris, n_frozen):
    """Calculate the matrix elements for double excitations without accounting
        for the parity of the excitations.

        Parameters
        ----------
        chosen_idx : (numpy.ndarray, unsigned int)
            the chosen indices of the two occupied orbitals (0th and 1st
            columns) and two virtual orbitals (2nd and 3rd columns) in each
            excitation
        eris : (numpy.ndarray, float)
            4-D array of 2-electron integrals in spatial MO basis
        n_frozen : (unsigned int)
            number of core electrons frozen in the calculation

        Returns
        -------
        (numpy.ndarray, float)
            matrix elements for all excitations
    """

    n_orb = eris.shape[0]
    chosen_orbs = chosen_idx + n_frozen/2
    chosen_orbs[chosen_orbs >= n_orb] += n_frozen/2
    same_sp = chosen_orbs[:, 0] / n_orb == chosen_orbs[:, 1] / n_orb
    spatial = chosen_orbs % n_orb
    matrix_el = eris[spatial[:, 0], spatial[:, 1],
                     spatial[:, 2], spatial[:, 3]]
    matrix_el[same_sp] -= eris[spatial[same_sp, 0], spatial[same_sp, 1],
                               spatial[same_sp, 3], spatial[same_sp, 2]]
    return matrix_el


def doub_dets_parity(dets, chosen_idx):
    """Given a set of double excitations from certain determinants, calculate
        the parity and the new determinant resulting from each excitation.
        (Parity in the sense of eq. 7 in Holmes et al. (2016))

        Parameters
        ----------
        dets : (numpy.ndarray, int)
            the initial determinants for which to calculate excitations
        chosen_idx : (numpy.ndarray, unsigned int)
            the chosen indices of the two occupied orbitals (0th and 1st
            columns) and two virtual orbitals (2nd and 3rd columns) in each
            excitation

        Returns
        -------
        (numpy.ndarray, int)
            resulting determinants
        (numpy.ndarray, int)
            resulting parities (+/-1)
    """

    one = numpy.ones(1, dtype=numpy.int64)
    excited_dets = dets ^ (one << chosen_idx[:, 0])
    excited_dets ^= (one << chosen_idx[:, 1])
    signs = excite_signs(chosen_idx[:, 2], chosen_idx[:, 0], excited_dets)
    signs *= excite_signs(chosen_idx[:, 3], chosen_idx[:, 1], excited_dets)
    excited_dets ^= (one << chosen_idx[:, 2])
    excited_dets ^= (one << chosen_idx[:, 3])
    return excited_dets, signs


def count_singex(det, occ_orbs, orb_symm, lookup_tabl):
    """Count the number of spin- and symmetry-allowed single excitations from a
        given determinant.

        Parameters
        ----------
        det : (int)
            bit-string representation of the origin determinant
        occ_orbs : (numpy.ndarray, unsigned int)
            list of occupied orbitals in det
        orb_symm : (numpy.ndarray, unsigned int)
            symmetry of each spatial orbital in the basis
        lookup_tabl : (numpy.ndarray, unsigned int)
            List of orbitals with each type of symmetry, as generated by
            gen_symm_lookup()

        Returns
        -------
        (int) :
            number of allowed single excitations
    """
    num_orb = orb_symm.shape[0]
    num_ex = 0

    for elec_orb in occ_orbs:
        elec_symm = orb_symm[elec_orb % num_orb]
        elec_spin = elec_orb / num_orb
        for symm_idx in range(lookup_tabl[elec_symm, 0]):
            if not(det & (1L << (lookup_tabl[elec_symm, symm_idx + 1] +
                                 num_orb * elec_spin))):
                num_ex += 1

    return num_ex


def count_doubex(occ_orbs, orb_symm, lookup_tabl):
    """Count the number of spin- and symmetry-allowed double excitations from a
        given determinant.

        Parameters
        ----------
        occ_orbs : (numpy.ndarray, unsigned int)
            list of occupied orbitals in the origin determinant
        orb_symm : (numpy.ndarray, unsigned int)
            symmetry of each spatial orbital in the basis
        lookup_tabl : (numpy.ndarray, unsigned int)
            List of orbitals with each type of symmetry, as generated by
            gen_symm_lookup()

        Returns
        -------
        (int) :
            number of allowed double excitations
    """

    num_orb = orb_symm.shape[0]
    num_ex = 0
    num_elec = occ_orbs.shape[0]
    n_symm = lookup_tabl.shape[0]
    unocc_sym_counts = numpy.zeros([2, n_symm], dtype=numpy.uint8)
    unocc_sym_counts[0, :] = lookup_tabl[:, 0]
    unocc_sym_counts[1, :] = lookup_tabl[:, 0]
    numpy.add.at(unocc_sym_counts[0, :], orb_symm[occ_orbs[:num_elec/2]], -1)
    numpy.add.at(unocc_sym_counts[1, :], orb_symm[occ_orbs[num_elec/2:] - num_orb], -1)

    for elec_i in range(num_elec):
        occ1 = occ_orbs[elec_i]
        occ1_spin = occ1 / num_orb
        for elec_j in range(elec_i + 1, num_elec):
            occ2 = occ_orbs[elec_j]
            occ2_spin = occ2 / num_orb
            symm_prod = orb_symm[occ1 % num_orb] ^ orb_symm[occ2 % num_orb]
            same_symm = symm_prod == 0 and occ1_spin == occ2_spin
            for j in range(n_symm):
                # number of unoccupied orbs w/ spin = occ1_spin and symm = j
                u1_poss = unocc_sym_counts[occ1_spin, j]
                # number of unoccupied orbs w/ spin = occ2_spin and symm ^ j =
                # symm_prod
                u2_poss = unocc_sym_counts[occ2_spin, j ^ symm_prod] - same_symm
                num_ex += u1_poss * u2_poss / (1. + (occ1_spin == occ2_spin))
    return num_ex


def gen_hf_bitstring(n_orb, n_elec):
    """Generate the bit string representation of the Hartree-Fock determinant.
    
        Parameters
        ----------
        n_orb : (unsigned int)
            the number of spatial orbitals in the basis
        n_elec : (unsigned int)
            the number of electrons in the system
        
        Returns
        -------
        (int64) :
            bit string representation of the determinant    
    """
    
    ones = 2**(n_elec / 2) - 1
    hf_state = ones << n_orb
    hf_state |= ones
    return hf_state


def gen_hf_ex(hf_det, hf_occ, n_orb, orb_symm, eris, n_frozen):
    """Generate all symmetry-allowed double excitations from the Hartree-Fock determinant.

        Parameters
        ----------
        hf_det : (int64)
            bit-string representation of the HF determinant
        hf_occ : (numpy.ndarray, uint8)
            orbitals occupied in the HF determinant
        n_orb : (unsigned int)
            number of spatial orbitals in the basis
        orb_symm : (numpy.ndarray, unsigned int)
            symmetry of each spatial orbital in the basis
        eris : (numpy.ndarray, float)
            4-D array of 2-electron integrals in spatial MO basis
        n_frozen : (unsigned int)
            number of core electrons frozen in the calculation

        Returns
        -------
        (numpy.ndarray, int64) :
            bit string representations of excited determinants
        (numpy.ndarray, float64) :
            FCI matrix elements for excited determinants
    """

    det_arr = numpy.array([hf_det], numpy.int64)
    occ_arr = hf_occ.copy()
    occ_arr.shape = (1, -1)
    ex_orbs = fci_c_utils.all_doub_ex(det_arr, occ_arr, n_orb)
    symm_allow = ((orb_symm[ex_orbs[:, 0] % n_orb] ^ orb_symm[ex_orbs[:, 1] % n_orb]) ==
                  (orb_symm[ex_orbs[:, 2] % n_orb] ^ orb_symm[ex_orbs[:, 3] % n_orb]))
    ex_orbs = ex_orbs[symm_allow]
    matr_el = doub_matr_el_nosgn(ex_orbs, eris, n_frozen)
    ex_dets, ex_signs = doub_dets_parity(det_arr, ex_orbs)
    ex_dets = numpy.insert(ex_dets, 0, hf_det)
    matr_el *= ex_signs
    matr_el = numpy.insert(matr_el, 0, 0.)
    return ex_dets, matr_el
