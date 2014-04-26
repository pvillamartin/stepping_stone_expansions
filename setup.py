from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("range_expansions", sources=["range_expansions.pyx"],
              extra_compile_args=['-O2']),
    Extension("random_cython", sources=["random_cython.cpp"],
              language="c++",
              extra_compile_args=['-std=c++11', '-O2']),
        Extension("random_test", sources=["random_test.pyx"],
              language="c++",
              extra_compile_args=['-std=c++11', '-O2'])
]

setup(
    name="Range Expansions",
    ext_modules = cythonize(extensions)
)
