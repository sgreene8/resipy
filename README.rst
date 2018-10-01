========
Overview
========

RESiPy: Randomized Electronic Structure in Python

This library provides scripts for running FCIQMC and FRI calculations
to calculate the ground-state energy of molecular/atomic systems.

* Free software: MIT license

Installation
============

Start by compiling the requisite Mersenne-Twister library (included) by
running:

> cd dcmt

> make

This will give you the libdcmt.a file, which is needed to compile the resipy
module.

Further information about the included Mersenne-Twister library may be found at
https://github.com/MersenneTwister-Lab/dcmt

Next, compile the resipy module by running (in the resipy root directory, using
pip for example):

> pip install .

or

> pip install . --user

The resipy module requires the one- and two-electron integrals from a Hartree-
Fock calculation. Some sample integrals for the B2 dimer and Ne atom are
included in tests/HF_results.

These integrals may also be generated using PySCF
(https://github.com/sunqm/pyscf). A script for performing HF calculations in
PySCF is included at /pyscf_hf.py


Documentation
=============

The fciqmc.py and fri.py scripts may be used to run calculations. Information
about the command-line arguments for these scripts may be found by running:

> python fciqmc.py -h

and

> python fri.py - h

For example, a set of arguments that I found work well for the included B2
dimer are:

> python fciqmc.py ../tests/HF_results/B2 10 0.01 20000 near_uniform

By default, this will produce 5 output files. The ground-state energy may be
estimated using the projected energy estimator by dividing the values at each
iteration in projnum.txt by those in projden.txt.



