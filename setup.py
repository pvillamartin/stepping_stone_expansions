from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import cython_gsl
import numpy as np

extensions = [
    Extension("stepping_stone.range_expansion", sources=["stepping_stone/range_expansion.pyx"], language='c',
              libraries = cython_gsl.get_libraries(),
              library_dirs = [cython_gsl.get_library_dir()],
              include_dirs = [cython_gsl.get_cython_include_dir(), np.get_include()])
]

setup(
    name="Range Expansions",
    version='0.6',
    author='Bryan T. Weinstein',
    author_email = 'bweinstein@seas.harvard.edu',
    include_dirs = [cython_gsl.get_include(), np.get_include()],
    packages=['stepping_stone'],
    ext_modules = cythonize(extensions, annotate=True),
)
