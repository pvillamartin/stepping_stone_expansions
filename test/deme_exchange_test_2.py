__author__ = 'bryan'

import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import range_expansions as re
import seaborn as sns

frac_gen, history = re.simulate_deme_line(num_demes=100, num_individuals=100, num_alleles=3, m_swap = 10)

# Generate a scatterplot of all the demes

plt.plot(frac_gen, history[0, :, :])
plt.show()