__author__ = 'bryan'

cimport numpy as np
import numpy as np
cimport random_cython_port
from random_cython_port cimport py_uniform_random
import random
import sys

cdef class Individual:

    cdef readonly long allele_id

    def __init__(Individual self, long allele_id):
        self.allele_id = allele_id

cdef class Deme:
# Assumes population in each deme is fixed!
# Otherwise random number generator breaks down.

    cdef readonly Individual[:] members
    cdef readonly long num_alleles
    cdef readonly long[:] binned_alleles
    cdef readonly long num_members
    cdef py_uniform_random r

    def __init__(Deme self,  long num_alleles, Individual[:] members):
        self.members = members
        self.num_members = len(members)
        self.num_alleles = num_alleles
        self.binned_alleles = self.bin_alleles()

        cdef double seed

        seed = random.randint(0, sys.maxint)

        self.r = py_uniform_random(0, self.num_members - 1, seed)

    cdef reproduce(Deme self):
        cdef int to_reproduce
        cdef int to_die

        to_reproduce = self.r.get_random()
        to_die = self.r.get_random()
        # Update allele array
        self.binned_alleles[self.members[to_die].allele_id] -= 1
        self.binned_alleles[self.members[to_reproduce].allele_id] += 1
        # Update the members
        # This is a little silly, i.e. doing this in two steps, but
        # it doesn't seem to work otherwise
        cdef Individual reproducer = self.members[to_reproduce]
        self.members[to_die] = reproducer

    cpdef get_alleles(Deme self):
        return [individual.allele_id for individual in self.members]

    cdef bin_alleles(Deme self):
        return np.bincount(self.get_alleles(), minlength=self.num_alleles)

def simulate_deme(Deme deme, long numAlleles, long numIterations=100):
    cdef long[:,:] history

    history = np.empty((numIterations, numAlleles), dtype=np.long)
    cdef long i
    for i in range(numIterations):
        history[i, :] = deme.binned_alleles
        deme.reproduce()
    return history