#cython: profile=False
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

# Things will actually crash if nonecheck is set to true...as neighbors is initially set to none

__author__ = 'bryan'

cimport cython

cimport numpy as np
import numpy as np
import random
import sys
from libcpp cimport bool
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas as pd

from cython_gsl cimport *

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
    cdef readonly long num_individuals
    cdef readonly double fraction_swap
    cdef public Deme[:] neighbors

    def __init__(Deme self,  long num_alleles, Individual[:] members not None, double fraction_swap = 0.0):
        self.members = members
        self.num_individuals = len(members)
        self.num_alleles = num_alleles
        self.binned_alleles = self.bin_alleles()
        self.fraction_swap = fraction_swap

        self.neighbors=None

    cdef reproduce(Deme self, int to_reproduce, int to_die):

        # Update allele array

        cdef Individual individual_to_die =  self.members[to_die]
        cdef Individual individual_to_reproduce = self.members[to_reproduce]

        cdef int allele_to_die = individual_to_die.allele_id
        cdef int allele_to_reproduce = individual_to_reproduce.allele_id

        self.binned_alleles[allele_to_die] -= 1
        self.binned_alleles[allele_to_reproduce] += 1
        # Update the members
        # This is a little silly, i.e. doing this in two steps, but
        # it doesn't seem to work otherwise
        cdef Individual reproducer = self.members[to_reproduce]
        self.members[to_die] = reproducer

    cdef swap_members(Deme self, Deme other, int self_swap_index, int other_swap_index):
        cdef int i

        cdef Individual self_swap
        cdef Individual other_swap

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

cdef class Simulate_Neutral_Deme:
    cdef readonly Deme deme
    cdef readonly long num_generations
    cdef readonly unsigned long int seed

    cdef readonly long[:,:] history
    cdef readonly double[:] fractional_generation
    cdef readonly unsigned int num_iterations
    cdef readonly double record_every_fracgen

    cdef unsigned int record_every

    def __init__(Simulate_Neutral_Deme self, Deme deme, long num_generations, unsigned long int seed = 0, record_every_fracgen = -1.0):

        self.deme = deme
        self.num_generations = num_generations
        self.seed = seed
        self.record_every_fracgen = record_every_fracgen

        if self.record_every_fracgen == -1.0:
            self.record_every_fracgen = 1./self.deme.num_individuals

        # Calculate how many iterations you must wait before recording
        self.record_every = int(deme.num_individuals * self.record_every_fracgen)

        # Take into account how often we record
        self.num_iterations = self.num_generations * self.deme.num_individuals / self.record_every

        self.fractional_generation = np.empty(self.num_iterations + 1, dtype=np.double)
        self.history = np.empty((self.num_iterations + 1, deme.num_alleles), dtype=np.long)

    cpdef simulate(Simulate_Neutral_Deme self):

        # Prepare random number generation
        np.random.seed(self.seed)

        cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(r, self.seed)

        cdef unsigned long to_reproduce
        cdef unsigned long to_die

        cdef unsigned int count = 0

        cdef long i

        cdef long cur_num_individuals = self.deme.num_individuals

        for i in range(self.num_iterations):
            if (i % self.record_every) == 0:
                self.fractional_generation[count] = float(i)/self.deme.num_individuals
                self.history[count, :] = self.deme.binned_alleles
                count += 1

            to_reproduce = get_neutral_reproduce(r, cur_num_individuals)
            to_die = get_neutral_die(r, cur_num_individuals)
            self.deme.reproduce(to_reproduce, to_die)

        self.fractional_generation[count] = self.num_iterations/self.deme.num_individuals
        self.history[count, :] = self.deme.binned_alleles

        gsl_rng_free(r)

cdef inline unsigned long int get_neutral_reproduce(gsl_rng *r, long num_individuals) nogil:
        return gsl_rng_uniform_int(r, num_individuals)

cdef inline unsigned long int get_neutral_die(gsl_rng *r, long num_individuals) nogil:
        return gsl_rng_uniform_int(r, num_individuals)


cdef class Simulate_Neutral_Deme_Line:

    cdef readonly Deme[:] initial_deme_list
    cdef readonly Deme[:] deme_list
    cdef readonly long num_demes
    cdef readonly long num_individuals
    cdef readonly long num_alleles
    cdef readonly double fraction_swap

    cdef readonly long num_generations
    cdef readonly double record_every
    cdef readonly unsigned int num_iterations

    cdef readonly bool debug
    cdef readonly unsigned long int seed

    cdef readonly long[:,:,:] history
    cdef readonly double[:] frac_gen

    def __init__(Simulate_Neutral_Deme_Line self, Deme[:] initial_deme_list, long num_alleles=2,
        long num_generations=100, double fraction_swap=0.1, double record_every = 1.0, unsigned long int seed=0,
        bool debug = False):
        '''  The user should input the list of demes. It is too annoying otherwise. There can be a utility
        function to generate common setups though.

        We assume m is the same for each deme, each deme has the same population,
        and that there is a known finite number of alleles at the start
        '''

        self.initial_deme_list = initial_deme_list
        self.deme_list = initial_deme_list.copy()

        self.num_individuals = initial_deme_list[0].num_individuals
        self.seed = seed

        #### Set the properties of the simulation ###

        self.num_demes = initial_deme_list.shape[0]

        self.num_alleles = num_alleles
        self.num_generations = num_generations
        self.fraction_swap = fraction_swap
        self.record_every = record_every
        self.debug = debug

        cdef int num_records = int(self.num_generations / self.record_every) + 1

        self.frac_gen = np.empty(num_records)

        cdef double invalid_length

        if self.debug:
            invalid_length = np.sqrt(self.fraction_swap * self.num_individuals * self.num_generations)
            print 'Invalid length from walls is ~' , invalid_length

        self.num_iterations = self.num_generations * self.num_individuals + 1
        self.history = np.empty((num_records, self.num_demes, num_alleles), dtype=np.long)


    def link_demes(Simulate_Neutral_Deme_Line self):
        '''Set up the network structure; make sure not to double count!
        Create periodic or line BC's here, your choice'''

        cdef long i

        for i in range(self.num_demes):
            if i != (self.num_demes - 1):
                self.deme_list[i].neighbors = np.array([self.deme_list[i + 1]], dtype=Deme)
            else:
                self.deme_list[i].neighbors = np.array([self.deme_list[0]], dtype=Deme)


    cpdef initialize_line(Simulate_Neutral_Deme_Line self, long[:] initial_condition):
        '''Create the same IC in each deme for convenience.'''
        temp_deme_list = []

        cdef int i
        cdef Individual[:] ind_list
        cdef Deme d

        # Put the same IC in each deme for now

        if initial_condition is None:
            for i in range(self.num_demes):
                ind_list = np.array([Individual(j) for j in np.random.randint(low=0, high=self.num_alleles, size=self.num_individuals)])
                d = Deme(self.num_alleles, ind_list, fraction_swap = self.fraction_swap)
                temp_deme_list.append(d)
        else:
            for i in range(self.num_demes):
                ind_list = np.array([Individual(j) for j in initial_condition])
                d = Deme(self.num_alleles, ind_list, fraction_swap = self.fraction_swap)
                temp_deme_list.append(d)

        self.deme_list = np.array(temp_deme_list, dtype=Deme)


    cdef swap_with_neighbors(Simulate_Neutral_Deme_Line self, gsl_rng *r):
        '''Be careful not to double swap! Each deme swaps once per edge.'''
        cdef long[:] swap_order
        cdef long swap_index
        cdef Deme current_deme
        cdef Deme[:] neighbors
        cdef Deme n

        # Create a permutation
        cdef int N = self.num_demes
        cdef gsl_permutation * p
        p = gsl_permutation_alloc (N)
        gsl_permutation_init (p)
        gsl_ran_shuffle(r, p.data, N, sizeof(size_t))

        cdef size_t *p_data = gsl_permutation_data(p)

        # Swap between all the neighbors once choosing the order randomly
        cdef int i
        cdef int j

        cdef int self_swap_index, other_swap_index
        cdef Deme otherDeme
        cdef int num_neighbors

        cdef int current_perm_index

        for i in range(N):
            current_perm_index = p_data[i]
            current_deme = self.deme_list[current_perm_index]
            neighbors = current_deme.neighbors
            num_neighbors = neighbors.shape[0]
            for j in range(num_neighbors):
                otherDeme = neighbors[j]
                self_swap_index = gsl_rng_uniform_int(r, current_deme.num_individuals)
                other_swap_index = gsl_rng_uniform_int(r, otherDeme.num_individuals)
                current_deme.swap_members(otherDeme, self_swap_index, other_swap_index)

        gsl_permutation_free(p)

    cdef reproduce(Simulate_Neutral_Deme_Line self, gsl_rng *r):
        cdef int d_num
        cdef long[:] current_alleles
        cdef Deme tempDeme

        cdef unsigned long int to_reproduce, to_die

        for d_num in range(self.num_demes):
            tempDeme = self.deme_list[d_num]
            to_reproduce = gsl_rng_uniform_int(r, tempDeme.num_individuals)
            to_die = gsl_rng_uniform_int(r, tempDeme.num_individuals)
            tempDeme.reproduce(to_reproduce, to_die)

    cpdef simulate(self):

        self.link_demes()

        cdef double swap_every
        if self.fraction_swap == 0:
            swap_every = -1.0
        else:
            swap_every = 1.0/self.fraction_swap

        cdef long swap_count = 0
        cdef long num_times_swapped = 0
        cdef double remainder = 0

        # Only useful when you swap more than once per iteration
        cdef double num_times_to_swap = 1.0/swap_every
        cdef int cur_gen

        # Use fast random number generation in mission critical methods
        # Make sure to delete this at the end to avoid memory leaks...
        cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

        # Now set seeds
        np.random.seed(self.seed)
        gsl_rng_set(r, self.seed)

        # Figure out how many iterations you should go before recording
        cdef int record_every_iter = int(self.record_every * self.num_individuals)

        cdef int num_times_recorded = 0

        cdef unsigned int i

        for i in range(self.num_iterations):
            # Bookkeeping
            swap_count += 1 # So at the start of the loop this has a minimum of 1

            # Record every "record_every"
            if (i % record_every_iter == 0) or (i == (self.num_iterations - 1)):
                self.frac_gen[num_times_recorded] = float(i) / self.num_individuals
                for d_num in range(self.num_demes):
                    self.history[num_times_recorded, d_num, :] = self.deme_list[d_num].binned_alleles
                num_times_recorded += 1

            # Reproduce
            self.reproduce(r)

            # Swap when appropriate
            if swap_every >= 2: # Swap less frequently than reproduction
                if swap_count >= swap_every:
                    swap_count = 0
                    num_times_swapped += 1
                    self.swap_with_neighbors(r)

            elif swap_every > 0: # Swap more frequently than reproduction
                while swap_count <= num_times_to_swap:
                    self.swap_with_neighbors(r)
                    swap_count += 1
                    num_times_swapped += 1

                #swap_count will always be too high as you just exited the for loop
                remainder += num_times_to_swap - (swap_count - 1)
                swap_count = 0
                if remainder >= 1:
                    remainder -= 1
                    self.swap_with_neighbors(r)
                    num_times_swapped += 1

        # Check that gene frequencies are correct!

        cdef long num_correct = 0
        if self.debug:
            for d in self.deme_list:
                if d.check_allele_frequency():
                    num_correct += 1
                else:
                    print 'Incorrect allele frequencies!'
            print 'Num correct:' , num_correct, 'out of', len(self.deme_list)

            # Check number of times swapped
            print 'Fraction swapped:' , num_times_swapped / float(self.num_generations*self.num_individuals)
            print 'Desired fraction:' , self.fraction_swap

        # DONE! Deallocate as necessary.
        gsl_rng_free(r)

    ####### Utility Classes #######

    def count_sectors(Simulate_Neutral_Deme_Line self, double cutoff = 0.1):
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

    cpdef F_ij(Simulate_Neutral_Deme_Line self, long i, long j, x1):

        m = self.position_map
        start_deme_index = m[m['position'] == x1].index[0]

        delta_positions = self.position_map.copy()

        delta_positions['position'] -= x1

        # Now calculate the heterozygosity at each time for each deme which have a
        # given position
        fij = np.empty((self.num_generations, self.num_demes))

        frac_history = np.asarray(self.history)/float(self.num_individuals)

        for gen_index in range(self.num_generations):
            fij[gen_index, :] = frac_history[gen_index, start_deme_index, i] * frac_history[gen_index, :, j]

        return fij, delta_positions

    def animate(Simulate_Neutral_Deme_Line self, generation_spacing = 1, interval = 1):
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

    def get_allele_history(Simulate_Neutral_Deme_Line self, long allele_num):

        history = np.asarray(self.history)
        fractional_history = history/float(self.num_individuals)

        num_entries = len(self.frac_gen)

        pixels = np.empty((num_entries, self.num_demes))

        cdef int i

        for i in range(num_entries):
            pixels[i, :] = fractional_history[i, :, allele_num]

        return pixels


