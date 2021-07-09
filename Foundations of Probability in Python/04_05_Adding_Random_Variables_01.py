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

Documentation:
    https://math.stackexchange.com/questions/593318/factorial-but-with-addition
    https://www.programming-idioms.org/idiom/67/binomial-coefficient-n-choose-k/1426/python
"""

###############################################################################
## Importing libraries
###############################################################################
import logging
import math
import matplotlib.patches            as patches
import matplotlib.path               as path
import matplotlib.pyplot             as plt
import numpy                         as np                                    #For making operations in lists
import pandas                        as pd
import seaborn                       as sns

from matplotlib.animation            import FuncAnimation
from matplotlib.animation            import PillowWriter
from scipy.stats                     import binom
from scipy.stats                     import describe
from scipy.stats                     import poisson

#import scipy.special                as ss                                    #To call a binomial coefficient function binom   
#ss.binom(4,2) == triangle_number(3)  == 3+2+1 == 6

###############################################################################
## Preparing the environment
###############################################################################
logger = logging.getLogger('matplotlib') #Trackint any error
logger.setLevel(logging.INFO)

SEED = 42
np.random.seed(SEED) 

def poisson_bar(mu, SEED, size=10000, y_text=0.001, text_percent=True): 
    """Make a poisson graph. Parameter needed:
        mu    -->Poisson parameter.
        SEED  -->Random seed.
        size  -->10,000 by default. Size of the sample to plot.
        sample-->If not given, the function will generete it.
        y     -->pmf of the sample.
        y_text-->the height add to y for printing the pmf in the plot.
    Return sample, unique values in the sample (x) and its pmf (y)."""
    sample = poisson.rvs(mu=mu, size=size, random_state=SEED) #if sample.size == 0 else sample
    mean     = np.mean(sample)
    median   = np.median(sample)
    #x, freq = np.unique(sample, return_counts=True)
    x        = np.unique(sample)
    y        = poisson.pmf(x, mu)
    
    #Plot the sample
    plt.bar(x, y)
    plt.xticks(x)
    plt.xlabel('k (Sample)'); plt.ylabel('poisson.pmf(k, mu={})'.format(mu)); # Labeling the axis.
    plt.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)
    plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
    plt.legend(loc='best', fontsize='small')
    if text_percent:
        for value, percent in zip(x, y):
            plt.text(value, percent+y_text, "{:,.2%}".format(percent), fontsize=8, ha='center')
    return sample, x, y

def triangle_number(n, k=2):
    """
    Factorial but with adition! [n + (n-1) + (n-2) + ... + 1]
    It's called n th triangle number and it can be written as (n+1 2), as a binomial coefficient.
    Parameters
    ----------
    n : Value of n.
    k : Coefficient in term of expansion, for our case k=2

    Returns
    -------
    [n + (n-1) + (n-2) + ... + 1].
    """
    return math.factorial(n+1) // math.factorial(k) // math.factorial(n+1-k)

###############################################################################
## Main part of the code
###############################################################################
print("****************************************************")
topic = "5. Adding random variables"; print("** %s\n" % topic)

mu=2
#size=100; n_sample_means=10  #Short example
size=1000; n_sample_means=350  #Large example
seed_sample_generation=20
seed_sample_mean=42


###############################################################################
# Poisson population plot
subtopic = "Printing poisson population plot..."; print("** %s\n" % subtopic)
fig = plt.figure()
population, x, y = poisson_bar(mu, SEED=seed_sample_generation, size=size)
max_value = x.max()
poisson_values = np.linspace(0, max_value, max_value+1, dtype=int) #Out: [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]

plt.title("Number of accidents per day", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=.15, bottom=None, right=None, top=.85, wspace=None, hspace=None);
plt.show()


###############################################################################
# Sample Means Plot
subtopic = "Printing sample means plot..."; print("** %s\n" % subtopic)
np.random.seed(seed_sample_mean)
sample_means = []
for _ in range(n_sample_means):
    # Select 10 from population
    sample = np.random.choice(population, 10)        
    # Calculate sample mean of sample
    sample_means.append(describe(sample).mean)
plt.figure()
plt.xlabel("Sample mean values")
plt.ylabel("Frequency")
plt.title("Sample means histogram", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.hist(sample_means)
plt.show()


###############################################################################
# Distribution Sample Plot (Only works for integers)
subtopic = "Printing distribution sample plot..."; print("** %s\n" % subtopic)
x_distro = np.array([])
y_distro = np.array([])
n_distro = [triangle_number(ii+1) for ii in range(size)]
for n in range(1, size+1):
    s = np.unique(poisson.rvs(mu=mu, size=n, random_state=seed_sample_generation))
    x_distro = np.append(x_distro, np.repeat(n, len(s))) 
    y_distro = np.append(y_distro, s)
#print(x_distro[0:10], y_distro[0:10])

plt.figure()
plt.xlabel("Size of Sample")
plt.ylabel("Population Values")
plt.yticks(poisson_values)
plt.title("Number of accidents per day", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.scatter(x_distro, y_distro, s=1, marker='.')
plt.show()


###############################################################################
# First animation
###############################################################################
## The data add the values taken in each size of the population, as it they 
## were possible values each.
## This is not the information required in the class, but the graph is an 
## example of how an animation works.
###############################################################################
subtopic = "First animation..."; print("** %s\n" % subtopic)
## (1). Read the data & Init Variables

## (2). Preparing the figure.
fig, ax = plt.subplots(1, 1)
fig.suptitle(topic, color='darkblue', fontsize=15)
plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=.5, hspace=None);

ax.set_title('Sample from Population', weight='bold', color='red', fontsize=20)
ax.set_xlim(0, size); ax.set_ylim(-0.5, max_value+.5); ax.set_yticks(poisson_values)
ax.set_xlabel('n sample'); ax.set_ylabel('Population values');

scat, = ax.plot([], [], ls='None', marker='.', markeredgecolor='darkblue', markeredgewidth=.5, markerfacecolor='lavender', markersize=30, alpha=0.5)
  

## (3) Construct the animation.
def init():
    scat.set_data([], [])
    return scat, 


def animate(i):
    x = x_distro[x_distro<=(i+1)]
    y = y_distro[np.where(x_distro<=(i+1))]
    
    scat.set_data(x, y)
    return scat, 

## (4) Start the animation.
anim = FuncAnimation(fig, animate, init_func=init, frames=size, interval=10, blit=False, repeat=True, repeat_delay=10)
plt.show()


## (5) Save the image.
#writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
#anim.save('04_05_01_Distro_Error.gif', writer=writer)


