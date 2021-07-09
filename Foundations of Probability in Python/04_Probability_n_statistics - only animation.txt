# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:40:32 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 4: Probability meets statistics
    No that you know how to calculate probabilities and important properties of probability 
    distributions, we'll introduce two important results: the law of large numbers and the 
    central limit theorem. This will expand your understanding on how the sample mean 
    converges to the population mean as more data is available and how the sum of random 
    variables behaves under certain conditions. We will also explore connections between 
    linear and logistic regressions as applications of probability and statistics in data 
    science.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import logging
import numpy                         as np                                    #For making operations in lists

from matplotlib                      import pyplot as plt
from matplotlib.animation            import FuncAnimation, PillowWriter
from scipy.stats                     import binom
#from scipy.stats                    import describe

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

logger = logging.getLogger('matplotlib') #Trackint any error
logger.setLevel(logging.INFO)

SEED = 42
np.random.seed(SEED) 


print("****************************************************")
topic = "1. From sample mean to population mean"; print("** %s\n" % topic)

## (1.1). Samples of n fair coin flips
n = 250 

## (1.2). Preparing the figure.
fig = plt.figure()
ax = plt.axes(xlim=(0, n), ylim=(0, 1))
line, = ax.plot([], [], lw=2, color='darkblue', label='Sample mean')
label = ax.text(n/2, 0.6, "", ha='center', fontsize=17, backgroundcolor='rosybrown', color='darkblue')
plt.axhline(y=0.5, lw=2, color='red', label='Population Mean')
plt.title("Fair coin flips Tendency", weight='bold', color='red', fontsize=20)
plt.suptitle("From sample mean to population mean", color='darkblue', fontsize=15)
plt.xlabel("Size of coin flips' sample", fontsize=20, color='red')
plt.ylabel('Sample Mean', fontsize=20, color='red', rotation=90)
plt.legend(loc='best')
plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);


## (1.3) Construct the animation.
def init():
    line.set_data([], [])
    label.set_text("                 ")
    return line, label


def animate(i):
    x = [ii for ii in range(1,i+1)]
    y = [binom.rvs(n=1, p=0.5, size=ii, random_state=SEED).mean() for ii in x]
    line.set_data(x, y)
    label.set_text("     {:.5f}     ".format(binom.rvs(n=1, p=0.5, size=(i+1), random_state=SEED).mean()))
    return line, label


## (1.4) Start the animation.
anim = FuncAnimation(fig, animate, init_func=init, frames=n, interval=100, blit=True, repeat=True)
plt.show()

## (1.5) Save the image.
#writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
#anim.save('04_Probability_n_statistics.gif', writer=writer)
##anim.save("04_Probability_n_statistics.gif", fps=20, writer='pillow')
plt.style.use('default')


print("****************************************************")
print("** END                                            **")
print("****************************************************")

