# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:52:13 2020

@author: jaces
"""
###############################################################################
##  Importing libraries.
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


###############################################################################
##  Fixing random state for reproducibility
###############################################################################
np.random.seed(19680801)


###############################################################################
##  Reading data.
###############################################################################
# Create rain data
n_drops = 50
rain_drops = np.zeros(n_drops, dtype=[('position', float, (2,)),
                                      ('size',     float),
                                      ('growth',   float),
                                      ('color',    float, (4,))])

###############################################################################
##  Preparing the figure.
###############################################################################
# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])

###############################################################################
#  Initialize the raindrops in random positions and with
###############################################################################
# random growth rates.
rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
rain_drops['growth'] = np.random.uniform(50, 200, n_drops)
#rain_drops['color'][:,0:3] = np.random.randint(0,2, (n_drops,3))
rain_drops['color'] = np.random.randint(0,2, (n_drops,4))

# Construct the scatter which we will update during animation
# as the raindrops develop.
scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
                  s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
                  facecolors='none')


###############################################################################
##  The animation function in which we define what happens in each frame of 
##  your video. 
###############################################################################
def update(frame_number):
    # Get an index which we can use to re-spawn the oldest raindrop.
    current_index = frame_number % n_drops

    # Make all colors more transparent as time progresses.
    #rain_drops['color'][:, 3] -= 1.0/len(rain_drops) #Opacity of the color
    rain_drops['color'][:, 3] = 1 #Solid color
    rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)

    # Make all circles bigger.
    rain_drops['size'] += rain_drops['growth']

    # Pick a new position for oldest rain drop, resetting its size,
    # color and growth factor.
    rain_drops['position'][current_index] = np.random.uniform(0, 1, 2)
    rain_drops['size'][current_index] = 5
    #rain_drops['color'][current_index] = (0, 0, 0, 1)
    rain_drops['color'][current_index] = (np.random.randint(2), np.random.randint(2), np.random.randint(2), 1)
    rain_drops['growth'][current_index] = np.random.uniform(50, 200)

    # Update the scatter collection, with the new colors, sizes and positions.
    scat.set_edgecolors(rain_drops['color'])
    scat.set_sizes(rain_drops['size'])
    scat.set_offsets(rain_drops['position']) #Translate to the new position.


###############################################################################
##  To start the animation.
###############################################################################
# Construct the animation, using the update function as the animation director.
animation = FuncAnimation(fig, update, interval=10)
plt.show()
###############################################################################
##  Save the image.
###############################################################################
writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
animation.save('raindrops.gif', writer=writer)
plt.style.use('default')

