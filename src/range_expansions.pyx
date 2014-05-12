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
    cdef readonly long m_swap
    cdef py_uniform_random r

    def __init__(Deme self,  long num_alleles, Individual[:] members,int m_swap=0):
        self.members = members
        self.num_members = len(members)
        self.num_alleles = num_alleles
        self.binned_alleles = self.bin_alleles()
        self.m_swap = m_swap

        cdef double seed
        seed = random.randint(0, sys.maxint)

        self.r = py_uniform_random(0, self.num_members - 1, seed)

    cpdef reproduce(Deme self):
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

    cpdef swap_members(Deme self, Deme other):
        cdef int i

        cdef int self_swap_index
        cdef int other_swap_index
        cdef Individual self_swap
        cdef Individual other_swap

        self_swap_index = self.r.get_random()
        other_swap_index = other.r.get_random()

        self_swap = self.members[self_swap_index]
        other_swap = other.members[other_swap_index]

        ## Update allele array of BOTH demes
        self.binned_alleles[self_swap.allele_id] -= 1
        self.binned_alleles[other_swap.allele_id] += 1

        other.binned_alleles[other_swap.allele_id] -= 1
        other.binned_alleles[self_swap.allele_id] += 1

        ## Update members
        self.members[self_swap_index] = other_swap
        other.members[other_swap_index] = self_swap

    cpdef get_alleles(Deme self):
        return [individual.allele_id for individual in self.members]

    cdef bin_alleles(Deme self):
        return np.bincount(self.get_alleles(), minlength=self.num_alleles)

    cpdef check_allele_frequency(Deme self):
        '''A diagnostic test that makes sure that the result returned by
        bin_alleles is the same as the current allele frequency. If it is false,
        there is a problem in the code somewhere.'''

        return np.array_equal(self.binned_alleles, self.bin_alleles())

def simulate_deme(Deme deme, long num_generations=100):
    cdef long[:,:] history
    cdef double[:] fractional_generation
    cdef long num_iterations

    num_iterations = num_generations * deme.num_members

    fractional_generation = np.empty(num_iterations, dtype=np.float)
    history = np.empty((num_iterations, deme.num_alleles), dtype=np.long)

    cdef long i
    for i in range(num_iterations):
        fractional_generation[i] = float(i)/deme.num_members
        history[i, :] = deme.binned_alleles
        deme.reproduce()

    return fractional_generation, history