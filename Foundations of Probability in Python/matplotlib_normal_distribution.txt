# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:25:35 2020

@author: jaces
"""

import numpy                         as np                                    #For making operations in lists
import matplotlib.pyplot             as plt                                   #For creating charts

some_data = np.random.normal(size=1000000)
H, edges = np.histogram(some_data, bins=100)

plt.figure(figsize=(10, 4))
plt.title('The "Normal" Distribution with Mean & St. Devs.')
plt.plot(edges[:-1], H)
plt.plot([np.mean(some_data), np.mean(some_data)], [0, max(H)], linestyle="--", color="r")
plt.fill_between([np.mean(some_data) - 3*np.std(some_data), np.mean(some_data) + 3*np.std(some_data)],
                 [0, 0],
                 [1.1*max(H), 1.1*max(H)], linestyle="--", color='k', alpha=0.25)
plt.fill_between([np.mean(some_data) - 2*np.std(some_data), np.mean(some_data) + 2*np.std(some_data)],
                 [0, 0],
                 [1.1*max(H), 1.1*max(H)], linestyle="--", color='k', alpha=0.25)
plt.fill_between([np.mean(some_data) - np.std(some_data), np.mean(some_data) + np.std(some_data)],
                 [0, 0],
                 [1.1*max(H), 1.1*max(H)], linestyle="--", color='k', alpha=0.25)
plt.xlim(-4, 4)
plt.ylim(0, 1.05*max(H))
plt.xlabel("Some Measurement"); plt.ylabel("Frequency of Measurement")
plt.show()