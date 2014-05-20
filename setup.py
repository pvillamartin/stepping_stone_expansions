from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("random_cython_port", sources=["src/random_cython_port.pyx", "src/random_cython.cpp"], language="c++",
              extra_compile_args=['-std=c++11', '-O2']),
    Extension("range_expansions", sources=["src/range_expansions.pyx"], language="c++",
              extra_compile_args=['-std=c++11', '-O2'])
]

setup(
    name="Range Expansions",
    ext_modules = cythonize(extensions, annotate=True, profile=True)
)
