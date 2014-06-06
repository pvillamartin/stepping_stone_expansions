__author__ = 'bryan'

import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import range_expansions as re
import skimage as ski
import skimage.io

num_demes = 1000
num_individuals = 10
num_generations = 1000
fraction_swap = 0.2
num_alleles = 2

simulation = re.Simulate_Deme_Line(num_demes=num_demes, num_individuals=num_individuals,
                                   num_alleles=num_alleles, fraction_swap = fraction_swap, num_generations=num_generations,
                                   debug=True)
# Generate a picture of all the demes

history = np.asarray(simulation.history)

pixels = np.empty((history.shape[0], num_demes))

for i in range(num_generations):
    pixels[i, :] = history[i, :,0]/float(num_individuals)

ski.io.imshow(pixels, origin='lower')
plt.show()

#sectors = simulation.count_sectors()

#print sectors