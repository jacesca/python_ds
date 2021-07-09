# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:29:24 2020

@author: jaces
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d

fig = plt.figure()
ax = fig.gca(projection='3d')

# load some test data for demonstration and plot a wireframe
X, Y, Z = axes3d.get_test_data(0.1)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
    print("View {} of {}".format(angle, 360))
    
print("END.")