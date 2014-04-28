cdef extern from 'random_cython.h' namespace '':
    cdef cppclass uniform_random:
        uniform_random(int low, int high, double seed) except +
        int get_random()
