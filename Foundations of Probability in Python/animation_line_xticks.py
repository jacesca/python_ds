# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:28:16 2020

@author: jacesca@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], '-o')


def init():
    global xdata
    global ydata
    xdata, ydata = [], []
    ln.set_data([], [])
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)


def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    ax.set_xlim(np.amin(xdata), np.amax(xdata))


ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init)
plt.show()