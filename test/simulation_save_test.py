__author__ = 'bryan'

import numpy as np
import range_expansions as re

num_demes = 1000
num_individuals = 10
num_generations = 1000
fraction_swap = 0.2
num_alleles = 2

simulation = re.Simulate_Deme_Line(num_demes=num_demes, num_individuals=num_individuals,
                                   num_alleles=num_alleles, fraction_swap = fraction_swap, num_generations=num_generations,
                                   debug=True)
# Save the simulation

np.save('simulation_test.npy', np.asarray(simulation.history))
muffin = np.load('simulation_test.npy')

print muffin

print 'Loaded successfully!'