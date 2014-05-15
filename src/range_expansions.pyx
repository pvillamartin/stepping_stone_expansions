__author__ = 'bryan'

cimport numpy as np
import numpy as np
cimport random_cython_port
from random_cython_port cimport py_uniform_random
import random
import sys
from libcpp cimport bool

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
    cdef public Deme[:] neighbors

    def __init__(Deme self,  long num_alleles, Individual[:] members,int m_swap=0):
        self.members = members
        self.num_members = len(members)
        self.num_alleles = num_alleles
        self.binned_alleles = self.bin_alleles()
        self.m_swap = m_swap

        cdef double seed
        seed = random.randint(0, sys.maxint)

        self.r = py_uniform_random(0, self.num_members - 1, seed)
        self.neighbors=None

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

    cdef swap_members(Deme self, Deme other):
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

def simulate_deme_line(int num_demes = 10, long num_individuals=100,
                       long num_alleles=2, long num_generations=100, long m_swap=10, bool debug=False):

    cdef int i
    cdef Individual[:] ind_list
    cdef Deme d

    temp_deme_list = []

    for i in range(num_demes):
        ind_list = np.array([Individual(i) for i in np.random.randint(low=0, high=num_alleles, size=num_individuals)])
        d = Deme(num_alleles, ind_list, m_swap)
        temp_deme_list.append(d)

    cdef Deme[:] deme_list = np.array(temp_deme_list, dtype=Deme)

    # We assume m is the same for each deme, each deme has the same population,
    # and that there is a known finite number of alleles at the start

    cdef long num_iterations
    cdef double[:] fractional_generation
    cdef long[:, :, :] history

    num_iterations = num_generations * num_individuals + 1
    fractional_generation = np.empty(num_iterations, dtype=np.float)
    history = np.empty((num_demes, num_iterations, num_alleles), dtype=np.long)

    # On each swap, you swap one with both of your neighbors
    # The generation time is set by the number of members
    # This should be an integer!
    cdef long swap_every = 2*num_individuals / m_swap

    cdef long[:] iterations = np.arange(num_iterations, dtype=long)

    for i in range(num_demes):
        if i ==0:
            deme_list[i].neighbors = np.array([deme_list[num_demes - 1], deme_list[1]], dtype=Deme)
        if i==num_demes - 1:
            deme_list[i].neighbors = np.array([deme_list[num_demes - 2], deme_list[0]], dtype=Deme)
        else:
            deme_list[i].neighbors = np.array([deme_list[i -1], deme_list[i+1]], dtype=Deme)

    cdef int d_num
    cdef long[:] current_alleles
    cdef Deme current_deme
    cdef Deme[:] neighbors
    cdef Deme n
    cdef Deme tempDeme

    cdef long swap_index
    cdef long[:] swap_order

    for i in iterations:
        fractional_generation[i] = float(i)/num_individuals
        for d_num in range(num_demes):
            current_alleles = deme_list[d_num].binned_alleles
            history[d_num, i, :] = current_alleles
            tempDeme = deme_list[d_num]
            tempDeme.reproduce()
        if ((i + 1) % swap_every) == 0:
            # Swap between all the neighbors once choosing the order randomly
            swap_order = np.random.permutation(num_demes)
            for swap_index in swap_order:
                current_deme = deme_list[swap_index]
                neighbors = current_deme.neighbors
                for n in neighbors:
                    current_deme.swap_members(n)
    # Check that gene frequencies are correct!

    cdef long num_correct = 0
    if debug:
        for d in deme_list:
            if d.check_allele_frequency():
                num_correct += 1
            else:
                print 'Incorrect allele frequencies!'
        print 'Num correct:' , num_correct, 'out of', len(deme_list)

    return fractional_generation, history