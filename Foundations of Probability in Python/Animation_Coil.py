# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:08:22 2020

@author: jaces
"""

###############################################################################
##  Importing libraries.
###############################################################################
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


###############################################################################
##  Setting the data.
###############################################################################
# lists to store x and y axis points 
xdata, ydata = [], [] 

###############################################################################
##  Preparing the figure.
###############################################################################
plt.style.use('dark_background')

fig = plt.figure() 
ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50)) 
line, = ax.plot([], [], lw=2) 

# setting a title for the plot 
plt.title('Creating a growing coil with matplotlib!') 
# hiding the axis details 
plt.axis('off') 

###############################################################################
##  Construct the animation.
###############################################################################
# initialization function 
def init(): 
	# creating an empty plot/frame 
	line.set_data([], []) 
	return line, 

# animation function 
def animate(i): 
	# t is a parameter 
	t = 0.1*i 
	
	# x, y values to be plotted 
	x = t*np.sin(t) 
	y = t*np.cos(t) 
	
	# appending new points to x, y axes points list 
	xdata.append(x) 
	ydata.append(y) 
	line.set_data(xdata, ydata) 
	return line, 

###############################################################################
##  Start the animation.
###############################################################################
# call the animator	 
anim = FuncAnimation(fig, animate, init_func=init, frames=500, interval=20, blit=True, repeat=True) 
plt.show()

###############################################################################
##  Save the image.
###############################################################################
writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com')) #For .gif. 
anim.save('coil.gif', writer=writer)
plt.style.use('default')
