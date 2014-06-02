# cython: profile=True

__author__ = 'bryan'

cimport numpy as np
import numpy as np
cimport random_cython_port
from random_cython_port cimport py_uniform_random
import random
import sys
from libcpp cimport bool
import pandas as pd
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas as pd

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
    cdef readonly double fraction_swap
    cdef py_uniform_random r
    cdef public Deme[:] neighbors

    def __init__(Deme self,  long num_alleles, Individual[:] members, double fraction_swap = 0.0):
        self.members = members
        self.num_members = len(members)
        self.num_alleles = num_alleles
        self.binned_alleles = self.bin_alleles()
        self.fraction_swap = fraction_swap

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

cdef class Simulate_Deme_Line:

    cdef readonly Deme[:] deme_list
    cdef readonly long[:,:,:] history
    cdef readonly position_map
    cdef readonly long num_demes
    cdef readonly long num_individuals
    cdef readonly long num_alleles
    cdef readonly long num_generations
    cdef readonly double fraction_swap
    cdef readonly bool debug

    def __init__(Simulate_Deme_Line self, long num_demes=100, long num_individuals=100, long num_alleles=2,
        long num_generations=100, double fraction_swap=0.1, bool debug = False):

        #### Set the properties of the simulation ###

        self.num_demes = num_demes
        self.num_individuals = num_individuals
        self.num_alleles = num_alleles
        self.num_generations = num_generations
        self.fraction_swap = fraction_swap
        self.debug = debug

        cdef double invalid_length

        if self.debug:
            invalid_length = np.sqrt(self.fraction_swap * self.num_individuals * self.num_generations)
            print 'Invalid length from walls is ~' , invalid_length

        temp_deme_list = []

        cdef int i
        cdef Individual[:] ind_list
        cdef Deme d

        for i in range(num_demes):
            ind_list = np.array([Individual(j) for j in np.random.randint(low=0, high=num_alleles, size=num_individuals)])
            d = Deme(num_alleles, ind_list, fraction_swap = fraction_swap)
            temp_deme_list.append(d)


        self.deme_list = np.array(temp_deme_list, dtype=Deme)

        input_dict = {}
        input_dict['demes'] = np.asarray(self.deme_list)
        positions = np.arange(self.num_demes) - self.num_demes / 2
        input_dict['position'] = positions

        self.position_map = pd.DataFrame(input_dict)

        # We assume m is the same for each deme, each deme has the same population,
        # and that there is a known finite number of alleles at the start

        cdef long num_iterations

        num_iterations = num_generations * num_individuals + 1
        self.history = np.empty((num_generations + 1, num_demes, num_alleles), dtype=np.long)

        # Set up the network structure; make sure not to double count!
        # Also do not create a circle, just create a line
        for i in range(num_demes - 1):
            self.deme_list[i].neighbors = np.array([self.deme_list[i + 1]], dtype=Deme)

        cdef double swap_every
        if fraction_swap == 0:
            swap_every = -1.0
        else:
            swap_every = 1.0/fraction_swap

        cdef long swap_count = 0
        cdef long num_times_swapped = 0
        cdef double remainder = 0

        # Only useful when you swap more than once per iteration
        cdef double num_times_to_swap = 1.0/swap_every
        cdef int cur_gen

        for i in range(num_iterations):
            # Bookkeeping
            swap_count += 1 # So at the start of the loop this has a minimum of 1

            # Record every generation
            if i % self.num_individuals == 0:
                cur_gen = i / self.num_individuals
                for d_num in range(self.num_demes):
                    self.history[cur_gen, d_num, :] = self.deme_list[d_num].binned_alleles

            # Reproduce
            self.reproduce(i)

            # Swap when appropriate
            if swap_every >= 1: # Swap less frequently than reproduction
                if swap_count >= swap_every:
                    swap_count = 0
                    num_times_swapped += 1
                    self.swap_with_neighbors()

            elif swap_every > 0: # Swap more frequently than reproduction
                while swap_count <= num_times_to_swap:
                    self.swap_with_neighbors()
                    swap_count += 1
                    num_times_swapped += 1

                #swap_count will always be too high as you just exited the for loop
                remainder += num_times_to_swap - (swap_count - 1)
                swap_count = 0
                if remainder >= 1:
                    remainder -= 1
                    self.swap_with_neighbors()
                    num_times_swapped += 1

        # Check that gene frequencies are correct!

        cdef long num_correct = 0
        if debug:
            for d in self.deme_list:
                if d.check_allele_frequency():
                    num_correct += 1
                else:
                    print 'Incorrect allele frequencies!'
            print 'Num correct:' , num_correct, 'out of', len(self.deme_list)

            # Check number of times swapped
            print 'Fraction swapped:' , num_times_swapped / float(self.num_generations*self.num_individuals)
            print 'Desired fraction:' , self.fraction_swap


    cdef swap_with_neighbors(Simulate_Deme_Line self):
        '''Be careful not to double swap! Each deme swaps once per edge.'''
        cdef long[:] swap_order
        cdef long swap_index
        cdef Deme current_deme
        cdef Deme[:] neighbors
        cdef Deme n

        # Swap between all the neighbors once choosing the order randomly
        swap_order = np.random.permutation(self.num_demes)
        cdef int i
        cdef int j
        for i in range(swap_order.shape[0]):
            current_deme = self.deme_list[swap_order[i]]
            neighbors = current_deme.neighbors
            for j in range(neighbors.shape[0]):
                current_deme.swap_members(neighbors[j])

    cdef reproduce(Simulate_Deme_Line self, long i):
        cdef int d_num
        cdef long[:] current_alleles
        cdef Deme tempDeme

        for d_num in range(self.num_demes):
            tempDeme = self.deme_list[d_num]
            tempDeme.reproduce()

    def count_sectors(Simulate_Deme_Line self, double cutoff = 0.1):
        '''Run this after the simulation has concluded to count the number of sectors'''
        # All you have to do is to count what the current domain type is and when it changes.
        # This is complicated by the fact that everything is fuzzy and that there can be
        # multiple colors.
        cdef int i

        data_list = []

        for i in range(self.deme_list.shape[0]):

            current_alleles = np.asarray(self.deme_list[i].binned_alleles)
            allele_frac = current_alleles / self.num_individuals
            dominant_sectors = allele_frac > cutoff

            data_list.append(dominant_sectors)

        return np.array(data_list)

    def animate(Simulate_Deme_Line self, generation_spacing = 1, interval = 1):
        '''Animates at the desired generation spacing using matplotlib'''
        history = np.asarray(self.history)
        # Only get data for every generation
        history_pieces = history[:, ::self.num_individuals, :]

        # Set up canvas to be plotted
        fig = plt.figure()
        ax = plt.axes(xlim = (0, self.num_demes), ylim = (0, 1))
        line, = ax.plot([], [])

        # Begin plotting

        x_values = np.arange(self.num_demes)
        fractional_pieces = history_pieces / float(self.num_individuals)

        num_frames = self.num_generations / generation_spacing

        def init():
            line.set_data(x_values, fractional_pieces[:, 0, 0])
            return line,

        def animate_frame(i):
            line.set_data(x_values, fractional_pieces[:, generation_spacing * i, 0])
            return line,

        return animation.FuncAnimation(fig, animate_frame, blit=True, init_func = init,
                                       frames=num_frames, interval=interval)
