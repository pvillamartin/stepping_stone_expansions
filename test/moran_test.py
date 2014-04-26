__author__ = 'bryan'

import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import range_expansions as re
import seaborn as sns

num_individuals=1000
numAlleles = 100
numIterations = 10**5

ind_list = np.array([re.Individual(i) for i in np.random.randint(low=0, high=numAlleles, size=num_individuals)])
d = re.Deme(numAlleles, ind_list)

history = re.simulate_deme(d, numAlleles, numIterations)
plt.plot(history)
plt.show()