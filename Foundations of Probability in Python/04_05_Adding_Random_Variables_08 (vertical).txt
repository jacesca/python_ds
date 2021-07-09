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
max_freq = int(round(np.max(y)*size,0))
poisson_values = np.linspace(0, max_value, max_value+1, dtype=int) #Out: [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]

plt.title("Number of accidents per day", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=.15, bottom=None, right=None, top=.85, wspace=None, hspace=None);
plt.show()


###############################################################################
# Distribution Sample Plot (Only works for integers)
subtopic = "Printing Population plot..."; print("** %s\n" % subtopic)

plt.figure()
plt.xlabel("n day")
plt.ylabel("Ocurred accidents per day")
plt.yticks(poisson_values)
plt.title("Population registered during 1,000 days", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.plot(range(1,size+1), population, ls='None', marker='.', markersize=3)
plt.show()


###############################################################################
# Sample Means Plot
subtopic = "Printing sample means plot..."; print("** %s\n" % subtopic)
np.random.seed(seed_sample_mean)
population_index = np.arange(1,size+1)
sample_means = []
x_distro = np.array([])
y_distro = np.array([])
n_distro = np.array([])
for n in range(n_sample_means):
    # Select 10 from population
    choices= np.random.choice(population_index, 10)        
    sample = population[choices]        
    sample_means.append(describe(sample).mean)
    n_distro = np.append(n_distro, np.repeat(n, len(sample)))
    x_distro = np.append(x_distro, choices) 
    y_distro = np.append(y_distro, sample)

plt.figure()
plt.xlabel("Sample mean values")
plt.ylabel("Frequency")
plt.title("Sample means histogram", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.hist(sample_means)
plt.show()



###############################################################################
# Fourth animation
subtopic = "Frequency values..."; print("** %s\n" % subtopic)
## (1). Read the data & Init Variables

## (2). Preparing the figure.
fig, ax = plt.subplots(1, 1)
fig.suptitle(topic, color='darkblue', fontsize=15)
plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);

hp_n, hp_bins, hp_patches = ax.hist(population, bins=max_value)

ax.set_title('Population Histogram', weight='bold', color='red', fontsize=20)
ax.set_ylabel('Frequency'); ax.set_xlabel('Population values');
ax.set_ylim(0, max_freq); 
ax.set_xticks(poisson_values)


## (3) Construct the animation.
def animate(i):
    n, bins = np.histogram(population[:i+1], bins=poisson_values)
    for rect, h in zip(hp_patches, n):
        rect.set_height(h)
    return hp_patches 
    """
    plt.cla()
    hp_n, hp_bins, hp_patches = ax.hist(population[:i+1], bins=max_value+1)
    """
    

## (4) Start the animation.
anim = FuncAnimation(fig, animate, frames=size, interval=10, blit=False, repeat=True, repeat_delay=100)
plt.show()


## (5) Save the image.
#writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
#anim.save('04_05_08_Hist_Population.gif', writer=writer)


