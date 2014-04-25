__author__ = 'bryan'

cdef extern from "<random>" namespace "std":
    cdef cppclass default_random_engine:
        default_random_engine() except +

    cdef cppclass uniform_int_distribution:
        uniform_int_distribution(int a, int b) except +
        int operator()(default_random_engine)

# cdef class libstd_random:
#     cdef default_random_engine *engine_ptr
#     cdef uniform_int_distribution *uniform_ptr
#
#     def __cinit__(self, a, b):
#         self.engine_ptr = new default_random_engine()
#         self.uniform_ptr= new uniform_int_distribution(a, b)
#
#     def get_random(self):
#         return self.uniform_ptr(self.engine_ptr)