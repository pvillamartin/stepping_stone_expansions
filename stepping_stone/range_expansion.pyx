#cython: profile=False
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

# Things will actually crash if nonecheck is set to true...as neighbors is initially set to none

__author__ = 'bryan'

cimport cython
import numpy as np
cimport numpy as np
from libcpp cimport bool
from matplotlib import animation
import matplotlib.pyplot as plt
import math #yes??

from cython_gsl cimport *
from libc.stdlib cimport free

cdef class Individual:
    """
    Individuals are placed in demes.
    """

    cdef:
        # Note that all rates should be non-dimensionalized in terms of generation time
        readonly long allele_id     # Basically a marker for color. If there are q alleles, the allele indices must be [0, 1, ..., q-1]
        readonly double growth_rate # Growth rate is only used if we include selection, i.e. selection demes

    def __init__(Individual self, long allele_id = 0, growth_rate = 1.0):
        self.allele_id = allele_id
        self.growth_rate = growth_rate

cdef class Deme:
    """
    The neutral deme. Individuals are placed in demes via the "members" memoryview. We currently
    assume that the population size "num_individuals" is fixed. We ignore mutation, selection, etc.
    Creating those parameters and placing them in a neutral deme will do nothing.
    """

    cdef:
        ##### Inputs ####
        readonly long num_alleles           # The number of alleles at the beginning of the simulation; assumes no creation of alleles
        readonly Individual[:] members      # A list of individuals currently in the deme
        readonly double fraction_swap       # The fraction of the individuals swapped with neighbors each generation
        #################

        #### Other Attributes ######
        readonly long[:] binned_alleles     # A count of each type of allele in the deme.
        readonly long num_individuals       # The number of individuals in the deme, N; assumes this is fixed
        public Deme[:] neighbors            # A list of neighboring demes. You have to initialize this.
        readonly double[:] growth_rate_list # A list of growth rates in the deme. Important for dealing with selection
        readonly int num_iterations         # The number of iterations the deme has undergone
        readonly TIME_PER_ITERATION         # The time per generation is defined as 1/N; it is a constant, hence the capitals

        readonly int last_index_to_die      # The index of the last member to die in the members list. Important for subclassing
        readonly int last_index_to_reproduce # The index of the last member to reproduce in the members list. Important for subclassing.
        ############################

    def __init__(Deme self,  long num_alleles, Individual[:] members not None, double fraction_swap = 0.0):
        self.members = members
        self.num_individuals = len(members)
        self.num_alleles = num_alleles
        self.binned_alleles = self.bin_alleles()
        self.fraction_swap = fraction_swap

        self.neighbors=None
        self.num_iterations = -1 # First iteration has a value of zero
        self.TIME_PER_ITERATION = 1./self.num_individuals

        self.last_index_to_die = -1 # garbage value
        self.last_index_to_reproduce = -1 # garbage value

        # Initialize the growth rate array
        cdef Individual ind
        temp_growth_list = []
        for ind in self.members:
            temp_growth_list.append(ind.growth_rate)
        self.growth_rate_list = np.array(temp_growth_list, dtype = np.double)

    cdef void reproduce_die_step(Deme self, gsl_rng *r):
        """
        Choose an individual to reproduce and an individual to die. Make them do so, update accordingly.

        :param r: from cython_gsl, random number generator. Used for fast random number generation
        """

        self.num_iterations +=1

        cdef unsigned int to_reproduce = self.get_reproduce(r)
        cdef unsigned int to_die = self.get_die(r)

        # Update allele array

        cdef Individual individual_to_die =  self.members[to_die]
        cdef Individual individual_to_reproduce = self.members[to_reproduce]

        cdef int allele_to_die = individual_to_die.allele_id
        cdef int allele_to_reproduce = individual_to_reproduce.allele_id

        # Update the binned alleles
        self.binned_alleles[allele_to_die] -= 1
        self.binned_alleles[allele_to_reproduce] += 1
        # Update the growth rate array; take the small (hopefully) hit in speed
        # for the neutral case to get additional flexibility
        cdef double surviving_growth_rate = self.growth_rate_list[to_reproduce]
        self.growth_rate_list[to_die] = surviving_growth_rate

        # Update the members
        # This is a little silly, i.e. doing this in two steps, but
        # it doesn't seem to work otherwise
        cdef Individual reproducer = self.members[to_reproduce]
        self.members[to_die] = reproducer

        # Keep track of who reproduced and who died
        self.last_index_to_die = to_die
        self.last_index_to_reproduce = to_reproduce

    cdef unsigned long int get_reproduce(Deme self, gsl_rng *r):
        """
        Choose an indiviual to reproduce randomly. Important for subclassing.
        :param r: from cython_gsl, random number generator. Used for fast random number generation.
        :return:  The index of the individual to reproduce in the members memoryview.
        """
        return gsl_rng_uniform_int(r, self.num_individuals)

    cdef unsigned long int get_die(Deme self, gsl_rng *r):
        """
        Choose an individual to die randomly. Important for subclassing.
        :param r: from cython_gsl, random number generator. Used for fast random number generation.
        :return: The index of the individual to die in the members memoryview.
        """
        return gsl_rng_uniform_int(r, self.num_individuals)


    cdef void swap_members(Deme self, Deme other, gsl_rng *r):
        """
        Swaps an individual randomly with another deme. Important for subclassing.

        :param other: The deme you are going to swap with.
        :param r: from cython_gsl, random number generator. Used for fast random number generation.
        """
        cdef:
            int i

            unsigned int self_swap_index = self.get_swap_index(r)
            unsigned int other_swap_index = other.get_swap_index(r)

            Individual self_swap
            Individual other_swap

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

        ## Update growth rates
        cdef other_fitness = other.growth_rate_list[other_swap_index]
        cdef double self_fitness  = self.growth_rate_list[self_swap_index]
        self.growth_rate_list[self_swap_index] = other_fitness
        self.growth_rate_list[other_swap_index] = self_fitness

    cdef unsigned long int get_swap_index(Deme self, gsl_rng *r):
        """
        Randomly chooses which individual to swap.
        :param r: from cython_gsl, random number generator. Used for fast random number generation.
        :return: index of the individual to swap.
        """
        return gsl_rng_uniform_int(r, self.num_individuals)

    cpdef get_alleles(Deme self):
        """
        A helper function to get the allele of each individual currently in the system.
        :return: A list of alleles
        """
        return [individual.allele_id for individual in self.members]

    cdef bin_alleles(Deme self):
        """
        :return: the number of type 0 alleles in index 0 of an array, number of type 1 alleles in index 1, etc.
        """
        return np.bincount(self.get_alleles(), minlength=self.num_alleles)

    cpdef check_allele_frequency(Deme self):
        """
        A diagnostic test that makes sure that the result returned by bin_alleles is the same as the current
        allele frequency. If it is false, there is a problem in the code somewhere.
        :return: True if the allele frequency is what it should be based on current individuals.
        """

        return np.array_equal(self.binned_alleles, self.bin_alleles())

cdef class Selection_Deme(Deme):
    """
    A deme that implements selection. Growth rates are now important. A subclass of the neutral deme.
    """

    def __init__(Selection_Deme self, *args, **kwargs):
        Deme.__init__(self, *args, **kwargs)

    # The only thing we have to update is the reproduce/die weighting function with selection.
    # No increased chance to die based on growth rate.
    cdef unsigned long int get_reproduce(Selection_Deme self, gsl_rng *r):
        """
        Choose an individual to reproduce based on their growth rates. Faster growing individuals have a higher
        chance to reproduce.

        :param r: from cython_gsl, random number generator. Used for fast random number generation.
        :return: Index of the individual to reproduce in the members array.
        """
        cdef double rand_num = gsl_rng_uniform(r)

        cdef double cur_sum = 0
        cdef unsigned int index = 0

        # Normalize the fitnesses
        cdef double[:] normalized_weights = self.growth_rate_list / np.sum(self.growth_rate_list)

        cdef double normalized_sum = 0

        for index in range(self.num_individuals):
            cur_sum += normalized_weights[index]

            if cur_sum > rand_num:
                return index

        print 'Selection sampling has exploded!'
        print 'Returning -1...something bad is going to happen.'
        return -1

cdef class Selection_aij_Deme(Selection_Deme):

    """
    A deme that implements selection in the way fitness(allele_i)=a_min+sum_j(aij* allele_frecuency_j).
    The fitness is now for each allele of each deme, not for each individual!
    """

    cdef:
        #Inputs
        readonly double[:,:] aij                    #The interaction matrix

    def __init__(Selection_aij_Deme self, *args, double[:,:] aij not None, **kwargs):
        Selection_Deme.__init__(self, *args, **kwargs)
        self.aij=aij

        # Initialize the fitness array
        self.get_fitness()

    cdef void get_fitness(Selection_aij_Deme self):
        """
        Gets deme's fitness (that depends on alleles frequency)

        :param r: from cython_gsl, random number generator. Used for fast random number generation
        """
        # Update the fitness array
        cdef long[:] cur_alleles = self.binned_alleles
        cdef double num_alleles = np.float(self.num_alleles)
        cdef double[:] frac_alleles = np.asarray(cur_alleles)/num_alleles

        cdef double[:] sum=np.dot(self.aij,frac_alleles)

        cdef int index
        for index in range(self.num_alleles):
            self.growth_rate_list[index]=self.members[index].growth_rate+sum[index]

    cdef void reproduce_die_step(Selection_aij_Deme self, gsl_rng *r):
        """
        Choose an individual to reproduce and an individual to die. Make them do so, update accordingly.

        :param r: from cython_gsl, random number generator. Used for fast random number generation
        """

        Selection_Deme.reproduce_die_step(self, r)
        self.get_fitness()

    cdef void swap_members(Selection_aij_Deme self, Deme other, gsl_rng *r):
        """
        Swaps an individual randomly with another deme. Important for subclassing.

        :param other: The deme you are going to swap with.
        :param r: from cython_gsl, random number generator. Used for fast random number generation.
        """

        Selection_Deme.swap_members(self,other,r)

        ## Update fitness
        self.get_fitness()
        (<Selection_aij_Deme>other).get_fitness()

cdef class Selection_Ratchet_Deme(Selection_Deme):
    """
    Implements deleterious mutations with a probability of "mutation_probability" per generation. s is the factor that
    is multiplied by your current growth rate when you suffer a deleterious mutation. Used to study mueller's ratchet.
    """
    cdef:
        #### Inputs ####
        readonly double mutation_probability    # The probability per generation to suffer a deleterious mutation
        readonly double s                       # The factor that your fitness is multiplied by when you suffer a deleterious mutation.
        ################

        #### Other attributes ####
        readonly double mutations_per_iteration # On average, how many mutations we expect per iteration
        readonly double mutation_remainder      # Important counter in order to implement the correct number of mutations per iteration
        readonly double mutation_count          # The number of mutations that have accumulated
        readonly double min_s                   # The minimum fitness at which to stop decrementing growth rate.
        ##########################

    def __init__(Selection_Ratchet_Deme self, *args, mutation_probability = 0.1, s=0.01, min_s = 10.**-300., **kwargs):
        Selection_Deme.__init__(self, *args, **kwargs)
        self.mutation_probability = mutation_probability
        self.s = s
        self.min_s = min_s

        self.mutations_per_iteration = self.mutation_probability * self.TIME_PER_ITERATION
        self.mutation_remainder = 0
        self.mutation_count = 0

    cdef void reproduce_die_step(Selection_Ratchet_Deme self, gsl_rng *r):
        """
        Reproduce an die as usual, but potentially acquire deleterious mutations.

        :param r: from cython_gsl, random number generator. Used for fast random number generation.
        """

        # Reproduce and die as usual.
        Selection_Deme.reproduce_die_step(self, r)
        # Now implement mutation: assume at most one mutation per birth for now
        cdef double roll = gsl_rng_uniform(r)

        if roll < self.mutation_probability:
            self.mutate(r)
            self.mutation_count += 1

    cdef void mutate(Selection_Ratchet_Deme self, gsl_rng *r):
        """
        Mutate the last individual that was born.

        :param r: from cython_gsl, random number generator. Used for fast random number generation.
        """
        cdef unsigned int index_to_mutate = self.last_index_to_die

        # Update the individual and the selection list
        cdef Individual member_to_mutate = self.members[index_to_mutate]
        member_to_mutate.growth_rate *= (1-self.s)
        if member_to_mutate.growth_rate < self.min_s:
            member_to_mutate.growth_rate = self.min_s
        self.growth_rate_list[index_to_mutate] = member_to_mutate.growth_rate

cdef class Simulate_Deme:
    """
    Responsible for simulating a single deme.
    """

    #### Inputs ####
    cdef readonly Deme deme                     # The input deme.
    cdef readonly long num_generations          # How many generations to make the simulation go
    cdef readonly unsigned long int seed        # Seed for the simulations. Must be between 0 and 2**32 - 1
    cdef readonly double record_every_fracgen   # How often (per generation) that you want to record
    ################

    #### Records of the simulation ####
    cdef readonly long[:,:] history             # The number of individuals with each allele vs. time
    cdef readonly double[:, :] fitness_history  # The history of the population vs. time
    cdef readonly double[:] fractional_generation   # The time of each recording.
    ###################################

    #### Helper attributes ####
    cdef readonly unsigned int num_iterations
    cdef readonly unsigned int record_every
    cdef readonly double cur_gen
    ###########################

    def __init__(Simulate_Deme self, Deme deme, long num_generations,
                 unsigned long int seed = 0, double record_every_fracgen = -1):

        self.cur_gen = 0
        self.deme = deme
        self.num_generations = num_generations
        self.seed = seed
        self.record_every_fracgen = record_every_fracgen

        if self.record_every_fracgen == -1:
            self.record_every_fracgen = 1.

        # Calculate how many iterations you must wait before recording
        self.record_every = int(deme.num_individuals * self.record_every_fracgen)

        # The number of iterations is independent of how often we record
        self.num_iterations = (self.num_generations + 1) * self.deme.num_individuals
        # Take into account the zeroth state
        cdef int num_to_record = (self.num_iterations / self.record_every) + 1

        self.fractional_generation = -1*np.ones(num_to_record, dtype=np.double)
        self.history = -1*np.ones((num_to_record, deme.num_alleles), dtype=np.long)
        self.fitness_history = -1*np.ones((num_to_record, deme.num_individuals), dtype=np.double)

    cpdef void simulate(Simulate_Deme self):
        """
        Simulates the deme's evolution for "num_generations."
        """

        # Prepare random number generation
        np.random.seed(self.seed)

        cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(r, self.seed)

        cdef unsigned long to_reproduce
        cdef unsigned long to_die
        cdef long i
        cdef long cur_num_individuals = self.deme.num_individuals
        cdef unsigned int count = 0

        cdef double generations_per_step = 1./self.deme.num_individuals

        for i in range(self.num_iterations):
            self.cur_gen = float(i)/self.deme.num_individuals
            if (i % self.record_every) == 0:
                self.fractional_generation[count] = self.cur_gen
                self.history[count, :] = self.deme.binned_alleles
                self.fitness_history[count, :] = self.deme.growth_rate_list
                count += 1

            self.deme.reproduce_die_step(r)

        self.fractional_generation[count] = self.num_iterations/self.deme.num_individuals
        self.history[count, :] = self.deme.binned_alleles

        gsl_rng_free(r)

cdef class Simulate_Deme_Line:
    """
    Responsible for simulating the stepping stone model: a line of connected demes! We assume swapping rate is the
    same for each deme, each deme has the same population, and that there is a known finite number of alleles at the
    start.
    """

    ####Input####
    cdef readonly Deme[:] initial_deme_list     # The initial state of the simulation
    cdef readonly Deme[:] deme_list             # The current state of the demes being evolved
    cdef readonly long num_alleles              # The total number of alleles in the system initially
    cdef readonly double fraction_swap          # How much of your population you swap with your neighbor each generation
    cdef readonly long num_generations          # How many generations to run the simulation
    cdef readonly double record_every           # How often to record the simulation state
    cdef readonly unsigned long int seed        # Seed for the simulation
    cdef readonly bool debug                    # A debug flag, useful for tracking down problems
    #############

    #### Outputs from the simulation ####
    cdef readonly double[:,:,:] fitness_history # [Time, Deme, array of fitnesses]
    cdef readonly long[:,:,:] history           # [Time, deme, array of binned alleles]
    cdef readonly double[:] frac_gen            # The number of elapsed generations when recorded.
    #####################################

    #### Helper variables ####
    cdef readonly double cur_gen
    cdef readonly long num_demes
    cdef readonly long num_individuals # Assumes that the number of individuals per deme is constant!
    cdef readonly unsigned int num_iterations
    ##########################

    def __init__(Simulate_Deme_Line self, Deme[:] initial_deme_list, long num_alleles=2,
        long num_generations=100, double fraction_swap=0.1, double record_every = 1.0, unsigned long int seed=0,
        bool debug = False):

        self.cur_gen = 0

        self.initial_deme_list = initial_deme_list.copy()
        self.deme_list = initial_deme_list

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
        self.history = -1*np.ones((num_records, self.num_demes, num_alleles), dtype=np.long)
        self.fitness_history = -1*np.ones((num_records, self.num_demes, self.num_individuals), dtype=np.double)
        # Don't forget to link the demes!

        self.link_demes()

    cpdef void link_demes(Simulate_Deme_Line self):
        """
        Set up the network structure.
        """

        cdef long i

        # You must have neighbors on left & right or bizarre things happen...
        cdef long max_index = self.num_demes -1

        for i in range(self.num_demes):
            if i == max_index:
                self.deme_list[i].neighbors = np.array([self.deme_list[0], self.deme_list[i - 1]], dtype=Deme)
            elif i== 0:
                self.deme_list[i].neighbors = np.array([self.deme_list[max_index], self.deme_list[i+1]], dtype=Deme)
            else:
                self.deme_list[i].neighbors = np.array([self.deme_list[i - 1], self.deme_list[i + 1]], dtype=Deme)

    cdef void swap_with_neighbors(Simulate_Deme_Line self, gsl_rng *r):
        """
        Loops through every deme on the line and randomly chooses a neighbor to swap with. This is a *terrible*
        way to do things currently; it would be better if things were replaced with choosing a random deme to swap.

        :param r: from cython_gsl, random number generator. Used for fast random number generation.
        """

        #TODO: Don't swap every neighbor at once, that is fraught with peril. Probably need a new update step in this simulation...

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

        cdef:
            size_t *p_data = gsl_permutation_data(p)

            # Swap between all the neighbors once choosing the order randomly
            int i

            int self_swap_index, other_swap_index
            Deme otherDeme
            int num_neighbors

            int current_perm_index
            int neighbor_choice

        for i in range(self.num_demes):
            current_perm_index = p_data[i]
            current_deme = self.deme_list[current_perm_index]
            neighbors = current_deme.neighbors
            num_neighbors = neighbors.shape[0]
            # Choose a neighbor at random to swap with
            neighbor_choice = gsl_rng_uniform_int(r, num_neighbors)
            otherDeme = neighbors[neighbor_choice]

            current_deme.swap_members(otherDeme, r)

        gsl_permutation_free(p)

    cdef void reproduce_line(Simulate_Deme_Line self, gsl_rng *r):
        """
        Loops over every deme on the line and makes them reproduce.
        :param r:  from cython_gsl, random number generator. Used for fast random number generation.
        """
        cdef int d_num
        cdef Deme tempDeme

        for d_num in range(self.num_demes):
            tempDeme = self.deme_list[d_num]
            tempDeme.reproduce_die_step(r)

    cpdef void simulate(self):
        """
        Run the simulation.
        """

        cdef double swap_every
        if self.fraction_swap == 0:
            swap_every = -1.0
        else:
            swap_every = 1.0/self.fraction_swap # When to swap

        cdef long swap_count = 0
        cdef long num_times_swapped = 0
        cdef double remainder = 0

        # Only useful when you swap more than once per iteration
        cdef double num_times_to_swap = 1.0/swap_every

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
            self.cur_gen = float(i)/self.num_individuals

            # Record every "record_every"
            if (i % record_every_iter == 0) or (i == (self.num_iterations - 1)):
                self.frac_gen[num_times_recorded] = self.cur_gen
                for d_num in range(self.num_demes):
                    self.history[num_times_recorded, d_num, :] = self.deme_list[d_num].binned_alleles
                    self.fitness_history[num_times_recorded, d_num, :] = self.deme_list[d_num].growth_rate_list
                num_times_recorded += 1

            #TODO: This should be done stochastically for each deme! i.e. a stochastic life cycle or something.

            # Reproduce
            self.reproduce_line(r)

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

        # Check that gene frequencies are correct if debugging.
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

    ####### Classes to get simulation history ####
    def get_allele_history(Simulate_Deme_Line self, long allele_num):

        history = np.asarray(self.history)
        fractional_history = history/float(self.num_individuals)

        num_entries = len(self.frac_gen)

        pixels = np.empty((num_entries, self.num_demes))

        cdef int i

        for i in range(num_entries):
            pixels[i, :] = fractional_history[i, :, allele_num]

        return pixels

    def get_fitness_history(Simulate_Deme_Line self):
        fit = np.array(self.fitness_history)
        return fit.mean(axis=2)

    def get_color_array(Simulate_Deme_Line self):

        cmap = plt.get_cmap('gist_rainbow')
        cmap.N = self.num_alleles

        # Hue will not be taken into account
        color_array = cmap(np.linspace(0, 1, self.num_alleles))

        alleleList = []
        for i in range(self.num_alleles):
            alleleList.append(self.get_allele_history(i))

        image = np.zeros((alleleList[0].shape[0], alleleList[0].shape[1], 4))

        for i in range(self.num_alleles):
            currentAllele = alleleList[i]

            redArray = currentAllele * color_array[i, 0]
            greenArray = currentAllele * color_array[i, 1]
            blueArray = currentAllele * color_array[i, 2]
            aArray = currentAllele * color_array[i, 3]

            image[:, :, 0] += redArray
            image[:, :, 1] += greenArray
            image[:, :, 2] += blueArray
            image[:, :, 3] += aArray

        #There is likely a faster way to do this involving the history, and multiplying it by cmap or something

        return image


    ####### Not sure if they work below...lol #######

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

    cpdef F_ij(Simulate_Deme_Line self, long i, long j, x1):

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
            print 'potato'
            return line,

        def animate_frame(i):
            line.set_data(x_values, fractional_pieces[:, generation_spacing * i, 0])
            return line,

        return animation.FuncAnimation(fig, animate_frame, blit=True, init_func = init,
                                       frames=num_frames, interval=interval)

    def get_color_array_by_fitness(Simulate_Deme_Line self):
        # We just have to cycle through the history and calculate the average selective advantage
        # at each point. I might have to do this as we go, however...
        print 'Not done yet'
        #TODO Make these plots looking at fitness instead of allele. It is a more robust measure of what is going on.


# cdef class Disordered_Diffusion_Deme(Simulate_Deme_Line):
#     """
#     Implements asymmetric diffusion between demes.
#     """
#     cdef:
#         #Variables
#         double Deme[:] deme_right_walls_height
#         double Deme[:] deme_prob_jump_right
#
#     def __init__(Disordered_Diffusion_Deme self, *args, **kwargs):
#         Deme.__init__(Simulate_Deme_Line self, *args, **kwargs)
#         self.fix_jum_probabilites()
#
#     cdef void fix_jump_probabilities(Disordered_Diffusion_Deme self, gsl_rng *r):
#
#         cdef:
#             double selected_deme
#             double aux_prob_left
#             double aux_prob_right
#             double left_neighbor
#
#         for index in range(self.num_demes):
#             selected_deme=self.deme_list[index]
#             self.deme_right_walls_height[selected_deme] = gsl_rng_uniform(r)
#
#         for index in range(self.num_demes):
#             selected_deme=self.deme_list[index]
#             left_neighbor=selected_deme.neighbors[0]
#             aux_prob_left=1 - self.deme_right_walls_height[left_neighbor]
#             aux_prob_right=1- self.deme_right_walls_height[selected_deme]
#             sum_prob=aux_prob_left+aux_prob_right
#             self.deme_prob_jump_right[selected_deme] = aux_prob_right/sum_prob
#
#     cdef void swap_with_neighbors(Disordered_Diffusion_Deme self, gsl_rng *r):
#         """
#         Loops through every deme on the line and randomly chooses a neighbor to swap with. This is a *terrible*
#         way to do things currently; it would be better if things were replaced with choosing a random deme to swap.
#
#         :param r: from cython_gsl, random number generator. Used for fast random number generation.
#         """
#
#         #TODO: Don't swap every neighbor at once, that is fraught with peril. Probably need a new update step in this simulation...
#
#         cdef long[:] swap_order
#         cdef long swap_index
#         cdef Deme current_deme
#         cdef Deme[:] neighbors
#         cdef Deme n
#
#         # Create a permutation
#         cdef int N = self.num_demes
#         cdef gsl_permutation * p
#         p = gsl_permutation_alloc (N)
#         gsl_permutation_init (p)
#         gsl_ran_shuffle(r, p.data, N, sizeof(size_t))
#
#         cdef:
#             size_t *p_data = gsl_permutation_data(p)
#
#             # Swap between all the neighbors once choosing the order randomly
#             int i
#
#             int self_swap_index, other_swap_index
#             Deme otherDeme
#             int num_neighbors
#
#             int current_perm_index
#             int neighbor_choice
#
#         for i in range(self.num_demes):
#             current_perm_index = p_data[i]
#             current_deme = self.deme_list[current_perm_index]
#             neighbors = current_deme.neighbors
#             num_neighbors = neighbors.shape[0]
#             # Choose a neighbor at random to swap with
#             neighbor_choice = gsl_rng_uniform_int(r, num_neighbors) #Now change this!
#             otherDeme = neighbors[neighbor_choice]
#
#             current_deme.swap_members(otherDeme, r)
#
#         gsl_permutation_free(p)