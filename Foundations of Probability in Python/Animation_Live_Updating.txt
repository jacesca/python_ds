# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:29:54 2020

@author: jaces
"""

###############################################################################
##  Importing libraries.
###############################################################################
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


###############################################################################
##  Setting the data.
###############################################################################
file = 'stock.csv'

###############################################################################
##  Preparing the figure.
###############################################################################
fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(1,1,1)

###############################################################################
##  Construct the animation.
###############################################################################
def animate(i):
    data = open(file,'r').read()
    lines = data.split('\n')
    xs = []
    ys = []
   
    for line in lines:
        x, y = line.split(',') # Delimiter is comma    
        xs.append(float(x))
        ys.append(float(y))
   
    
    ax1.clear()
    ax1.plot(xs, ys)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Live graph with matplotlib', fontsize=14, weight='bold', color='darkred')

###############################################################################
##  Start the animation.
###############################################################################
ani = FuncAnimation(fig, animate, interval=1000)
plt.suptitle('Animation Live Updating', color='darkblue')
plt.show()
plt.style.use('default')