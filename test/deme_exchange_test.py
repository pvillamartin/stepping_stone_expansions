__author__ = 'bryan'

import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import range_expansions as re
import seaborn as sns

num_individuals= 500
numAlleles = num_individuals
numGenerations = 100
m = num_individuals/10

numDemes = 2
demeList = []

for i in range(numDemes):
    ind_list = np.array([re.Individual(i) for i in np.random.randint(low=0, high=numAlleles, size=num_individuals)])
    d = re.Deme(numAlleles, ind_list, m)
    demeList.append(d)

# Now try to create a simulation
# We assume m is the same for each deme
# We also assume that each deme has the same population
# We also assume that there is a known finite number of alleles at the start
num_members = demeList[0].num_members
num_alleles = demeList[0].num_alleles
m_swap = demeList[0].m_swap

num_iterations = numGenerations * num_members + 1
fractional_generation = np.empty(num_iterations, dtype=np.float)
history = np.empty((numDemes, num_iterations, num_alleles), dtype=np.long)

# On each swap, you swap one with both of your neighbors
# The generation time is set by the number of members
swap_every = 2*num_members / (m_swap)

iterations = np.arange(num_iterations)

# Create the desired network structure

for i in iterations:
    fractional_generation[i] = float(i)/num_members
    for d_num in range(numDemes):
        history[d_num, i, :] = demeList[d_num].binned_alleles
        demeList[d_num].reproduce_line()
    if np.mod(i + 1, swap_every) == 0:
        # Swap between all the neighbors, ideally utilizing some sort of
        # network structure
        for d_num in demeList:
            pass