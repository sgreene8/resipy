"""
Script for compiling the fci_c_utils Cython module.

Usage: python compile_fcicutils.py build_ext --inplace
"""


from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("fci_c_utils",
              ["fci_c_utils.pyx"],
              libraries=["m", "dcmt"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp'],
               library_dirs = ["./"]
              )
]

setup(
  name = "fci_c_utils",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs = [numpy.get_include()]
)

