"""
Script for compiling the near_uniform Cython module.

Usage: python compile_nearuni.py build_ext --inplace
"""


from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("near_uniform",
              ["near_uniform.pyx"],
              libraries=["m", "dcmt"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp'],
               library_dirs = ["./"]
              )
]

setup(
  name = "near_uniform",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs = [numpy.get_include()]
)

