__author__ = 'bryan'

from random_test import py_uniform_random
import numpy as np

muffin = np.random.rand()*10**6

derp = py_uniform_random(0, 100, muffin)
print derp.get_random()