# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:10:26 2020

@author: jacesca@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## (1). Init Values
n = 15
x_data = [ii for ii in range(n)]
y_data = [ii**2 for ii in x_data]


## (2). Preparing the figure.    
#fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ln, = plt.plot([], [], '-o')
ax.set_xlim(np.min(x_data), np.max(x_data))
ax.set_ylim(np.min(y_data), np.max(y_data))


## (3) Construct the animation.
def init():
    ln.set_data([], [])
    ax.set_xlim(np.min(x_data), np.max(x_data))
    ax.set_ylim(np.min(y_data), np.max(y_data))
    return ln, ax

def update(i):
    x = x_data[0:i]
    y = y_data[0:i]
    ln.set_data(x, y)
    ax.set_xlim(np.amin(x), np.amax(x)+0.01)
    return ln, ax


## (4) Start the animation.
ani = FuncAnimation(fig, update, frames=n,
                    init_func=init, repeat=True)
plt.show()