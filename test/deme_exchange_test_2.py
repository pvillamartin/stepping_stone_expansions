__author__ = 'bryan'

import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import range_expansions as re
import skimage as ski
import skimage.io

num_demes = 200
num_individuals = 500
num_generations = 200
m_swap = 10

frac_gen, history = re.simulate_deme_line(num_demes=num_demes, num_individuals=num_individuals,
                                          num_alleles=2, m_swap = m_swap, num_generations=num_generations,
                                          debug=True)

# Generate a picture of all the demes

frac_gen = np.asarray(frac_gen)
history = np.asarray(history)

pixels = np.empty((len(frac_gen)/num_individuals, num_demes))

for i in range(num_demes):
    cur_history = history[i, 1::num_individuals, :]
    pixels[:, i] = cur_history[:,0]/float(num_individuals)

ski.io.imshow(pixels, origin='lower')
plt.show()