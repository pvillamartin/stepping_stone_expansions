from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import cython_gsl

extensions = [
    Extension("range_expansions", sources=["src/range_expansions.pyx"], language="c++",
              extra_compile_args=['-std=c++11'],
              libraries = cython_gsl.get_libraries(),
              library_dirs = [cython_gsl.get_library_dir()],
              include_dirs = [cython_gsl.get_cython_include_dir()])
]

setup(
    name="Range Expansions",
    include_dirs = [cython_gsl.get_include()],
    ext_modules = cythonize(extensions, annotate=True, profile=False)
)
