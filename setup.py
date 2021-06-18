from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

from setuptools import setup
from Cython.Build import cythonize

setup(
    name='UGPFM_fast',
    ext_modules=cythonize("UGPFM_fast.pyx"),
    include_dirs=[numpy.get_include()],
    #zip_safe=False,
)
