# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:44:52 2023

@author: Amin
"""

import numpy as np

from pymoo.problems import get_problem
from pymoo.visualization.fitness_landscape import FitnessLandscape

problem = get_problem("ackley", n_var=2, a=5, b=1/5, c=2 * np.pi)
# problem = get_problem("rastrigin", n_var=2, A=5)
# problem = get_problem("zakharov", n_var=2)
# problem = get_problem("rosenbrock", n_var=2)

FitnessLandscape(problem, angle=(45, 45), _type="surface").show()
