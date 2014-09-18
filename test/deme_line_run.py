__author__ = 'bryan'

import numpy as np
import range_expansions as re

num_demes = 10**3
num_individuals = 50
num_alleles = 2
num_generations = 100
fraction_swap = 0.1
seed = np.random.randint(0, 2**32 - 1)

hetero_condition = num_demes**2/(fraction_swap**2 * num_individuals**4)

print hetero_condition

sim = re.Simulate_Deme_Line(num_demes = num_demes, num_individuals = num_individuals,
                            num_alleles = num_alleles, num_generations = num_generations,
                            fraction_swap = fraction_swap, seed = seed,
                            debug = False, record_every=.5)