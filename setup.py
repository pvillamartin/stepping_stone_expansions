from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("src/range_expansions", sources=["range_expansions.pyx"],
              extra_compile_args=['-O2'])
]

setup(
    name="Range Expansions",
    ext_modules = cythonize(extensions)
)
