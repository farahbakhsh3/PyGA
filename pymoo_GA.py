# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:54:05 2023

@author: Amin
"""

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("g1")
problem = get_problem("rosenbrock", n_var=3)

algorithm = GA(
    pop_size=100,
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
