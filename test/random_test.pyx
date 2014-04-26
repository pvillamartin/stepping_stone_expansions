from cpp.random_cython cimport uniform_random

cdef uniform_random *dist = new uniform_random(0, 5, 3)
try:
    print dist.get_random()
finally:
    del dist # delete heap allocated object