from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("range_expansions", ["range_expansions.pyx"], language="c",
                  extra_compile_args=['-O2'])
    ]
)
