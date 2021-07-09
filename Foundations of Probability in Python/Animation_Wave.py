# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:27:02 2020

@author: jaces
"""

###############################################################################
##  Importing libraries.
###############################################################################
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


###############################################################################
##  Preparing the figure.
##  We simply create a figure window with a single axis in the figure. Then 
##  we create our empty line object which is essentially the one to be modified 
##  in the animation. The line object will be populated with data later.
###############################################################################
plt.style.use('seaborn-pastel')

fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-1.25, 1.25))
line, = ax.plot([], [], lw=3)

plt.title("y = sin[2π (x - 0.01 i)] ", weight='bold', color='red', fontsize=20)
plt.suptitle("Animated Math Function", color='darkblue', fontsize=15)
plt.xlabel('X', fontsize=20, color='red')
plt.ylabel('Y', fontsize=20, color='red', rotation=0)
plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);

###############################################################################
##  Construct the animation.
##  First, we create the init function that will make the animation happen. 
##  The init function initializes the data and also sets the axis limits.
##  Second, we finally define the animation function which takes in the 
##  frame number(i) as the parameter and creates a sine wave(or any other 
##  animation) which a shift depending upon the value of i. This function here 
##  returns a tuple of the plot objects which have been modified which tells 
##  the animation framework what parts of the plot should be animated.
###############################################################################
def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

###############################################################################
##  Start the animation.
##  We create the actual animation object. The blit parameter ensures that 
##  only those pieces of the plot are re-drawn which have been changed.
###############################################################################
anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
plt.show()

###############################################################################
##  Save the image.
###############################################################################
writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
anim.save('wave.gif', writer=writer)
plt.style.use('default')
