# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:27:37 2020

@author: jaces
"""
import numpy                         as np                                    #For making operations in lists
from scipy.stats                     import poisson                           #To generate poisson distribution.

sample = poisson.rvs(mu=2.2, size=10000, random_state=13)
x = np.unique(sample)
y      = poisson.pmf(x, 2.2)
for value, percent in zip(x, y):
    print(value, percent)

z = np.unique(sample, return_counts=True)
print(z) 