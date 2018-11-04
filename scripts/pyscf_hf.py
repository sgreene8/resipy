#!/usr/bin/env python2
"""
Perform a Hartree-Fock calculation using pyscf and write the results to disk.
"""

from pyscf import gto, scf, ao2mo, symm
import argparse
import numpy
import os


def main():
    supported_mols = ["B2", "N2", "Ne", "Be", "HF", "Li", "O", "H2", "H2O"]
    args = _parse_args(supported_mols)
    mol_name = args.molecule

    mol = gto.Mole()

    if mol_name == supported_mols[0]:
        # Geometry is specified in angstroms
        mol.atom = [['B', (0, 0, -0.5)], ['B', (0, 0, 0.5)]]
        mol.basis = 'sto3g'
        mol.symmetry = 'd2h'

    elif mol_name == supported_mols[1]:
        mol.atom = [['N', (0, 0, -1.1115)], ['N', (0, 0, 1.1115)]]
        mol.basis = 'sto3g'
        mol.symmetry = 'd2h'

    elif mol_name == supported_mols[2]:
        mol.atom = [['Ne', (0, 0, 0)]]
        mol.basis = 'aug-cc-pvdz'
        mol.symmetry = 'd2h'

    elif mol_name == supported_mols[3]:
        mol.atom = [['Be', (0, 0, 0)]]
        mol.basis = 'cc-pv5z'
        mol.symmetry = 'd2h'

    elif mol_name == supported_mols[4]:
        mol.atom = [['H', (0, 0, -0.870056)], ['F', (0, 0, 0.0461636)]]
        mol.basis = 'cc-pcvdz'
        mol.symmetry = 'd2h'

    elif mol_name == supported_mols[5]:
        mol.atom = [['Li', (0, 0, 0)]]
        mol.basis = 'sto3g'
        mol.charge = +1
        mol.symmetry = 'd2h'

    elif mol_name == supported_mols[6]:
        mol.atom = [['O', (0, 0, 0)]]
        mol.basis = 'cc-pvdz'
        mol.symmetry = 'd2h'

    elif mol_name == supported_mols[7]:
        mol.atom = [['H', (0, 0, -0.5)], ['H', (0, 0, 0.5)]]
        mol.basis = 'sto3g'
        mol.symmetry = 'd2h'

    elif mol_name == supported_mols[8]:
        mol.atom = [['O', (0, 0, 0)], [
            'H', (-0.342668, 0.913347, 0)], ['H', (0.342668, 0.913347, 0)]]
        mol.basis = 'cc-pvdz'
        mol.symmetry = 'c2v'

    mol.build()

    rhf_solver = scf.RHF(mol)

    # Run RHF calculation
    rhf_solver.kernel()

    # Calculate matrix elements of one-electron operator h(1) (Szabo & Ostlund, eq 3.150) in MO basis
    h_core = rhf_solver.mo_coeff.T.dot(
        rhf_solver.get_hcore()).dot(rhf_solver.mo_coeff)
    n_orb = h_core.shape[0]

    # Calculate electron repulsion integrals in MO basis
    eris = ao2mo.full(mol.intor('int2e_sph', aosym='s4'),
                      rhf_solver.mo_coeff, compact=False)
    # Convert to physicists' format
    eris.shape = (n_orb, n_orb, n_orb, n_orb)
    eris = eris.transpose(0, 2, 1, 3)

    # Generate irreducible representations
    irreps = symm.label_orb_symm(
        mol, mol.irrep_id, mol.symm_orb, rhf_solver.mo_coeff)
    irreps = irreps.astype(numpy.uint8)

    # Save results to disk
    res_path = args.out_dir + mol_name
    if not(os.path.isdir(res_path)):
        try:
            os.mkdir(res_path)
        except OSError:
            raise OSError("Hartee-Fock directory (%s) not found." % args.out_dir)

    numpy.save(res_path + '/hcore', h_core)
    numpy.save(res_path + '/eris', eris)
    numpy.save(res_path + '/symm', irreps)
    with open(res_path + '/hf_en.txt', 'w') as f:
        f.write('%.10f\n' % rhf_solver.energy_elec()[0])


def _parse_args(names):
    """
    Subroutine to parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Perform an Hartree-Fock Calculation with pyscf")
    parser.add_argument('molecule', choices=names,
                        help="Molecule/atom for which to perform HF")
    parser.add_argument('-o', '--out_dir', type=str, default='../tests/HF_results/',
                        help="Directory to which to write the output")

    args = parser.parse_args()
    if args.out_dir[-1] != '/':
        args.out_dir += '/'

    return args


if __name__ == "__main__":
    main()
