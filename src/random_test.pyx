# distutils: language = c++
# distutils: sources = src/random_cython.cpp

from random_test_def cimport uniform_random

cdef class py_uniform_random:

    cdef uniform_random *thisptr
    def __cinit__(self, int low, int high, double seed):
        self.thisptr = new uniform_random(low, high, seed)
    def __dealloc__(self):
        del self.thisptr

    def get_random(self):
        return self.thisptr.get_random()