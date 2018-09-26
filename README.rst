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

.. |docs| image:: https://readthedocs.org/projects/resipy/badge/?style=flat
    :target: https://readthedocs.org/projects/resipy
    :alt: Documentation Status

.. |version| image:: https://img.shields.io/pypi/v/resipy.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/resipy

.. |commits-since| image:: https://img.shields.io/github/commits-since/sgreene8/resipy/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/sgreene8/resipy/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/resipy.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/resipy

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/resipy.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/resipy

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/resipy.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/resipy


.. end-badges

Implementations of randomized electronic structure methods in Python.

* Free software: MIT license

Installation
============

::

    pip install resipy

Documentation
=============

https://resipy.readthedocs.io/

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
