# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:44:19 2023

@author: Amin
"""

from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import matplotlib.pyplot as plt


def objective(x, y):
 return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * 
  pi * x)+cos(2 * pi * y))) + e + 20


def rastrigin_function(x, y):
    A = 10
    n = len(x)
    return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


r_min, r_max = -5.0, 5.0
xaxis = arange(r_min, r_max, .05)
yaxis = arange(r_min, r_max, .05)
x, y = meshgrid(xaxis, yaxis)
results = objective(x, y)
figure = plt.figure()
axis = figure.add_subplot(projection='3d')
axis.plot_surface(x, y, results, cmap='jet', shade= "false")
plt.show()
plt.contour(x,y,results)
plt.show()
plt.scatter(x, y, results)
plt.show()
