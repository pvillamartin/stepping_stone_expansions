__author__ = 'bryan'

import numpy as np
import range_expansions as re

def simulate_deme_many_times(initial_condition, num_alleles, num_individuals, num_generations, num_times,
                               record_every_fracgen):

    number_of_records = int(num_generations / record_every_fracgen + 1)

    sim_list = np.empty((num_times, number_of_records, num_alleles))

    frac_gen = None

    for i in range(num_times):
        # Set the seed
        seed = np.random.randint(0, 2**32 -1)
        # Set half of the individuals to 0 and the other half to 1, fo=0.5

        ind_list = np.array([re.Individual(j) for j in initial_condition])
        deme = re.Deme(num_alleles, ind_list)

        frac_gen, history = re.simulate_deme(deme, num_generations, seed, record_every_fracgen)
        frac_gen = np.asarray(frac_gen)
        history = np.asarray(history)
        frac_history = history/float(num_individuals)

        sim_list[i, :, :] = frac_history

    sim_list = np.asarray(sim_list)
    frac_gen = np.asarray(frac_gen)

    return sim_list, frac_gen