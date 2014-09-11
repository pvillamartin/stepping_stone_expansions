__author__ = 'bryan'

import numpy as np
import range_expansions as re
import ternary
import scipy as sp
from matplotlib import pyplot as plt

interpn = sp.interpolate.interpn

def simulate_deme_many_times(initial_condition, num_alleles, num_generations, num_times, record_every_fracgen):

    number_of_records = int(num_generations / record_every_fracgen + 1)

    sim_list = np.empty((num_times, number_of_records, num_alleles))

    num_individuals = len(initial_condition)

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

class Simulate_3_Alleles_Deme:
    '''A convenience class to do 3 color stepping stone models. Also
        cotains methods to plot the results on a triangle.'''

    def __init__(self, initial_condition, num_generations, num_simulations, record_every=None):
        self.num_alleles = 3
        self.initial_condition = initial_condition
        self.num_individuals = initial_condition.shape[0]
        self.num_generations = num_generations
        self.num_simulations = num_simulations

        self.record_every = record_every
        if self.record_every is None:
            self.record_every = 1./(initial_condition.shape[0])

        # Does the simulation

        self.sim_list, self.frac_gen = simulate_deme_many_times(self.initial_condition, self.num_alleles,
                                                                self.num_generations, self.num_simulations,
                                                                self.record_every)

        # Collects the data in a convenient way
        self.edges, self.centers = self.get_hist_edges_and_center()
        self.histogrammed_data = self.get_3d_histogram_in_time()

    def get_hist_edges_and_center(self):
        edges = np.arange(-1./(2*self.num_individuals), 1 + 2 *(1./(2*self.num_individuals)),
                          1./self.num_individuals)
        centers = (edges[:-1] + edges[1:])/2.

        return edges, centers

    def get_3d_histogram_in_time(self):
        num_bins = self.centers.shape[0]
        num_records = self.frac_gen.shape[0]
        histogrammed_data = np.empty((num_records, num_bins, num_bins, num_bins))

        for i in range(num_records):
            histogrammed_data[i, :, :, :] = np.histogramdd(self.sim_list[:, i, :], bins=[self.edges, self.edges, self.edges])[0]
            histogrammed_data[i, :, :, :] /= float(self.num_simulations)
        return histogrammed_data

    def interp_histogram_at_iteration(self, iteration, points):
        # Just returns the nearest neighbor. I use the same grid in python_ternary so no interpolation is done,
        # this just saves me from having to rewrite EVERYTHING.
        return interpn((self.centers, self.centers, self.centers),
        self.histogrammed_data[iteration], points, method='nearest', bounds_error=True)

    def plot_heatmap_at_iteration(self, iteration, **options):
        '''I line up the actual grid with interpolating grid so there is no interpolation, actually.

        Options:
            min_max_scale: [min, max] of the scale
            colorbar_decimals: number of decimals used in the colorbar
        '''
        ax = plt.subplot()
        ternary.plot_heatmap(lambda x: self.interp_histogram_at_iteration(iteration, x),
                             steps = self.num_individuals, boundary=True, **options)
        scale = self.num_individuals
        ternary.draw_boundary(scale=scale, ax=ax)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.ylim([-0.13*scale, scale])

        plt.text(.15 * scale, .43*scale, r'$f_1$', fontsize=35 )
        plt.text(.78 * scale, .43*scale, r'$f_3$', fontsize=35 )
        plt.text(.46*scale, -0.1*scale, r'$f_2$', fontsize=35)
        plt.gca().yaxis.set_visible(False)
        plt.gca().xaxis.set_visible(False)
        plt.grid(False)

        # Plot the fractional generation
        generation_formatted = '%.2f' % self.frac_gen[iteration]
        plt.text(.77*scale, .925*scale, 'Generation: ' + generation_formatted, fontsize=15)

