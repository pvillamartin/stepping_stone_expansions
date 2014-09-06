from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import cython_gsl

extensions = [
    Extension("range_expansions", sources=["src/range_expansions.pyx"], language="c++",
              libraries = cython_gsl.get_libraries(),
              library_dirs = [cython_gsl.get_library_dir()],
              include_dirs = [cython_gsl.get_cython_include_dir()])
]

setup(
    name="Range Expansions",
    version='0.5',
    author='Bryan T. Weinstein',
    author_email = 'bweinstein@seas.harvard.edu',
    include_dirs = [cython_gsl.get_include()],
    ext_modules = cythonize(extensions, annotate=True),
    py_modules = ['range_expansion_convenience']
)
