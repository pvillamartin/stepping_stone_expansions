__author__ = 'bryan'

import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import range_expansions as re
import seaborn as sns

num_individuals=500
numAlleles = num_individuals
numGenerations = 100
m=0

ind_list = np.array([re.Individual(i) for i in np.random.randint(low=0, high=numAlleles, size=num_individuals)])
d = re.Deme(numAlleles, ind_list, 0)

frac_gen, history = re.simulate_deme(d, numGenerations)

# Checking that the allele frequency is what is predicted

print 'Allele frequency is correct:' , d.check_allele_frequency()

plt.plot(frac_gen, history)
plt.show()