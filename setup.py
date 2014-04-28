from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    # Extension("src/range_expansions", sources=["src/range_expansions.pyx"],
    #           extra_compile_args=['-O2']),
    # Extension("src/random_cython", sources=["src/random_cython.cpp"],
    #           language="c++",
    #           extra_compile_args=['-std=c++11', '-O2']),
    # Extension("src/random_cython_pyx", sources=["src/random_cython_port.pxd"],
    #           language="c++",
    #           extra_compile_args=['-std=c++11', '-O2'])
    Extension("random_cython_port", sources=["src/random_cython_port.pyx", "src/random_cython.cpp"], language="c++",
              extra_compile_args=['-std=c++11', '-O2']),
    Extension("range_expansions", sources=["src/range_expansions.pyx"], language="c++",
              extra_compile_args=['-std=c++11', '-O2'])
]

setup(
    name="Range Expansions",
    ext_modules = cythonize(extensions)
)
