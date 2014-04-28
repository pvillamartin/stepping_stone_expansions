cdef extern from 'random_cython.h' namespace '':
    cdef cppclass uniform_random:
        uniform_random(int low, int high, double seed) except +
        int get_random()

cdef class py_uniform_random:
    cdef uniform_random *thisptr
    cpdef get_random(self)