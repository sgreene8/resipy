========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/pyres/badge/?style=flat
    :target: https://readthedocs.org/projects/pyres
    :alt: Documentation Status

.. |version| image:: https://img.shields.io/pypi/v/pyres.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/pyres

.. |commits-since| image:: https://img.shields.io/github/commits-since/sgreene8/pyres/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/sgreene8/pyres/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/pyres.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/pyres

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/pyres.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/pyres

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pyres.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/pyres


.. end-badges

AImplementations of randomized electronic structure methods.

* Free software: MIT license

Installation
============

::

    pip install pyres

Documentation
=============

https://pyres.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
