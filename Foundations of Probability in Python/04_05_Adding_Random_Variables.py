# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:45:05 2020

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
import matplotlib.ticker             as ticker
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
#logger = logging.getLogger('matplotlib') #Trackint any error
#logger.setLevel(logging.INFO)

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
def static_image(mu, size, n_sample_means, seed_population, seed_sample_mean):
    print("****************************************************")
    topic = "4.5.0 Static images"; print("** %s\n" % topic)
    
    mu=2
    #size=100; n_sample_means=10  #Short example
    size=1000; n_sample_means=350  #Large example
    seed_population=20
    seed_sample_mean=42
    
    
    ###############################################################################
    # Poisson plot
    subtopic = "Printing poisson plot..."; print("** %s" % subtopic)
    fig = plt.figure()
    population, x, y = poisson_bar(mu, SEED=seed_population, size=size)
    max_value = x.max()
    max_freq = int(round(np.max(y)*size,0))
    poisson_values = np.linspace(0, max_value, max_value+1, dtype=int) #Out: [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]
    
    plt.title("Number of accidents per day (Poisson Plot)", color='red')
    plt.suptitle(topic, color='navy');  # Setting the titles.
    plt.subplots_adjust(left=.15, bottom=None, right=None, top=.85, wspace=None, hspace=None);
    plt.show()
    
    
    ###############################################################################
    # Populatio Plot
    subtopic = "Printing population plot..."; print("** %s" % subtopic)
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
    np.random.seed(seed_sample_mean)
    population_index = np.arange(1,size+1)
    sample_means = []
    sample_means_int = []
    x_sample = np.array([])
    y_sample = np.array([])
    n_sample = np.array([])
    for n in range(n_sample_means):
        # Select 10 from population
        choices= np.random.choice(population_index, 10)        
        sample = population[choices]        
        sample_means.append(describe(sample).mean)
        sample_means_int.append(int(describe(sample).mean))
        n_sample = np.append(n_sample, np.repeat(n, len(sample)))
        x_sample = np.append(x_sample, choices) 
        y_sample = np.append(y_sample, sample)
        
    plt.figure()
    plt.xlabel("Sample mean values")
    plt.ylabel("Frequency")
    plt.title("Sample means histogram", color='red')
    plt.suptitle(topic, color='navy');  # Setting the titles.
    plt.hist(sample_means)
    plt.show()

    return population, population_index, max_value, max_freq, poisson_values, n_sample, x_sample, y_sample, sample_means, sample_means_int
    


def animation_error(mu, size, n_sample_means, seed_population, seed_sample_mean, max_value, poisson_values):
    print("****************************************************")
    topic = "4.5.1 Animation from the incorrect interpretation"; print("** %s\n" % topic)
    ###############################################################################
    # Distribution Sample Plot (Only works for integers)
    subtopic = "Printing distribution sample plot..."; print("** %s\n" % subtopic)
    x_distro = np.array([])
    y_distro = np.array([])
    #n_distro = [triangle_number(ii+1) for ii in range(size)]
    for n in range(1, size+1):
        s = np.unique(poisson.rvs(mu=mu, size=n, random_state=seed_population))
        x_distro = np.append(x_distro, np.repeat(n, len(s))) 
        y_distro = np.append(y_distro, s)
    #print(x_distro[0:10], y_distro[0:10])
        
    ##plt.figure()
    ##plt.xlabel("Size of Sample")
    ##plt.ylabel("Population Values")
    ##plt.yticks(poisson_values)
    ##plt.title("Number of accidents per day", color='red')
    ##plt.suptitle(topic, color='navy');  # Setting the titles.
    ##plt.scatter(x_distro, y_distro, s=1, marker='.')
    ##plt.show()
    
    ###############################################################################
    # First animation
    ###############################################################################
    ## The data add the values taken in each size of the population, as it they 
    ## were possible values each.
    ## This is not the information required in the class, but the graph is an 
    ## example of how an animation works.
    ###############################################################################
    subtopic = "First animation..."; print("** %s" % subtopic)
    ## (1). Read the data & Init Variables
    
    ## (2). Preparing the figure.
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(topic, color='darkblue', fontsize=15)
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
    
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
    anim = FuncAnimation(fig, animate, init_func=init, frames=size, interval=10, blit=True, repeat=True, repeat_delay=10)
    plt.show()


    ## (5) Save the image.
    #writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
    #anim.save('04_05_01_Distro_Error.gif', writer=writer)


def animation_population(mu, size, n_sample_means, seed_population, seed_sample_mean, population, max_value, poisson_values):
    print("****************************************************")
    topic = "4.5.2 Animation of Population record during 1,000 days"; print("** %s\n" % topic)
    ## (1). Read the data & Init Variables
    
    ## (2). Preparing the figure.
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(topic, color='darkblue', fontsize=15)
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
    
    ax.set_title('Sample from Population', weight='bold', color='red', fontsize=20)
    ax.set_xlim(0, size); ax.set_ylim(-0.5, max_value+.5); ax.set_yticks(poisson_values)
    ax.set_xlabel('n sample'); ax.set_ylabel('Population values');
    
    scat, = ax.plot([], [], ls='None', marker='.', markerfacecolor='darkblue', markersize=3)
    
    ## (3) Construct the animation.
    def init():
        scat.set_data([], [])
        return scat, 
    
    def animate(i):
        x = range(1,(i+1))
        y = population[:i]
        
        scat.set_data(x, y)
        return scat, 
    
    ## (4) Start the animation.
    anim = FuncAnimation(fig, animate, init_func=init, frames=size, interval=10, blit=True, repeat=True, repeat_delay=10)
    plt.show()
    
    ## (5) Save the image.
    #writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
    #anim.save('04_05_02_Population.gif', writer=writer)


def animation_population_and_sample(mu, size, n_sample_means, seed_population, seed_sample_mean, 
                                    population, population_index,
                                    n_sample, x_sample, y_sample,
                                    max_value, poisson_values):
    print("****************************************************")
    topic = "4.5.3 Animation of Selected Sample"; print("** %s\n" % topic)
    
    ## (1). Read the data & Init Variables
    ## (2). Preparing the figure.
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(topic, color='darkblue', fontsize=15)
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
    
    ax.set_title('Sample from Population', weight='bold', color='red', fontsize=20)
    ax.set_xlim(0, size); ax.set_ylim(-0.5, max_value+.5); ax.set_yticks(poisson_values)
    ax.set_xlabel('n sample'); ax.set_ylabel('Population values');
    
    ax.plot(population_index, population, ls='None', marker='.', markerfacecolor='darkblue', markersize=3)
    scat, = ax.plot([], [], ls='None', marker='s', markerfacecolor='darkred', markersize=3)
    plt.legend(['Population', 'Selected Sample'], loc='upper right')
        
    ## (3) Construct the animation.
    def init():
        scat.set_data([], [])
        return scat, 
    
    
    def animate(i):
        x = x_sample[np.where(n_sample==i)]
        y = y_sample[np.where(n_sample==i)]
        
        scat.set_data(x, y)
        return scat, 
    
    ## (4) Start the animation.
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_sample_means, interval=100, blit=True, repeat=True, repeat_delay=100)
    plt.show()
    
    
    ## (5) Save the image.
    #writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
    #anim.save('04_05_03_Population_and_sample.gif', writer=writer)
    

def animation_frequency_population(size, population, max_value, poisson_values):
    print("****************************************************")
    topic = "4.5.4 Data Frequency found in the Population"; print("** %s\n" % topic)
    
    ## (1). Read the data & Init Variables
    frequency_values = []
    for n in range(size):
        v, f = np.unique(population[:n+1], return_counts=True)
        f = [int(f[np.where(v==k)]) if np.isin(k,v) else k*0 for k in poisson_values]
        frequency_values.append(f)
        
    ## (2). Preparing the figure.
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(topic, color='darkblue', fontsize=15)
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
    
    ax.set_title('Population Histogram', weight='bold', color='red', fontsize=20)
    ax.set_xlim(0, np.max(f)+0.5); ax.set_ylim(-0.5, max_value+.5); ax.set_yticks(poisson_values)
    ax.set_xlabel('Frequency'); ax.set_ylabel('Population values');
    
    bar_h = plt.barh(list(poisson_values), f)
    
    
    ## (3) Construct the animation.
    def animate(i):
        y = frequency_values[i] 
        for k, b in enumerate(bar_h):
            b.set_width(y[k])
        return bar_h 
    
    ## (4) Start the animation.
    anim = FuncAnimation(fig, animate, frames=size, interval=10, blit=True, repeat=True, repeat_delay=100)
    plt.show()
    
    
    ## (5) Save the image.
    #writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
    #anim.save('04_05_04_Population_Frequency.gif', writer=writer)
    
    
def animation_description_of_population(size, population, max_value, poisson_values):
    print("****************************************************")
    topic = "4.5.5 Two animation at once"; print("** %s\n" % topic)
    
    ## (1). Read the data & Init Variables
    frequency_values = []
    for n in range(size):
        v, f = np.unique(population[:n+1], return_counts=True)
        f = [int(f[np.where(v==k)]) if np.isin(k,v) else k*0 for k in poisson_values]
        frequency_values.append(f)
        
        
    ## (2). Preparing the figure.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(topic, color='darkblue', fontsize=15)
    plt.subplots_adjust(left=.1, bottom=.15, right=.9, top=.85, wspace=.2, hspace=None);
    
    ax1.set_title('Sample from Population', weight='bold', color='red', fontsize=20)
    ax1.set_xlim(0, size); ax1.set_ylim(-0.5, max_value+.5); ax1.set_yticks(poisson_values)
    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.set_xlabel('n sample');  ax1.set_ylabel('Population values');
    scat, = ax1.plot([], [], ls='None', marker='.', markerfacecolor='darkblue', markersize=3)
    
    ax2.set_title('Population Histogram', weight='bold', color='red', fontsize=20)
    ax2.set_xlim(0,  np.max(f)+0.5); ax2.set_ylim(-0.5, max_value+.5); ax2.set_yticks(poisson_values)
    ax2.set_xlabel('Frequency'); ax2.set_ylabel('Population values');
    bar_h = ax2.barh(list(poisson_values), f)
    
    
    ## (3) Construct the animation.
    def animate(i):
        x = range(1,(i+1)); y = population[:i];
        width = frequency_values[i] 
        
        scat.set_data(x, y)
        for k, b in enumerate(bar_h):
            b.set_width(width[k])
        return [each_bar for each_bar in bar_h] + [scat,]
        #return scat, bar_h --> It doesn't work.
    
    
    ## (4) Start the animation.
    anim = FuncAnimation(fig, animate, frames=size, interval=10, blit=True, repeat=True, repeat_delay=100)
    plt.show()
    
    
    ## (5) Save the image.
    #writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
    #anim.save('04_05_05_Description_of_population.gif', writer=writer)
    
    

def animation_description_of_population_and_sample(size, n_sample_means, population, population_index, 
                                                   n_sample, x_sample, y_sample, 
                                                   max_value, poisson_values):
    print("****************************************************")
    topic = "4.5.6 Population and Sample"; print("** %s\n" % topic)
    
    ## (1). Read the data & Init Variables
    frequency_values = []
    for n in range(size):
        v, f = np.unique(population[:n+1], return_counts=True)
        f = [int(f[np.where(v==k)]) if np.isin(k,v) else k*0 for k in poisson_values]
        frequency_values.append(f)
        
        
    ## (2). Preparing the figure.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(topic, color='darkblue', fontsize=15)
    plt.subplots_adjust(left=.1, bottom=.15, right=.9, top=.85, wspace=.2, hspace=None);
    
    ax1.set_title('Sample from Population', weight='bold', color='red', fontsize=20)
    ax1.set_xlim(0, size); ax1.set_ylim(-0.5, max_value+.5); ax1.set_yticks(poisson_values)
    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.set_xlabel('n sample');  ax1.set_ylabel('Population values');
    ax1.plot(population_index, population, ls='None', marker='.', markerfacecolor='darkblue', markersize=3)
    scat, = ax1.plot([], [], ls='None', marker='s', markerfacecolor='darkred', markersize=3)
    ax1.legend(['Population','Sample'], loc='best')
    
    ax2.set_title('Population Histogram', weight='bold', color='red', fontsize=20)
    ax2.set_xlim(0,  np.max(f)+0.5); ax2.set_ylim(-0.5, max_value+.5); ax2.set_yticks(poisson_values)
    ax2.set_xlabel('Frequency'); ax2.set_ylabel('Population values');
    bar_h = ax2.barh(list(poisson_values), f)
    
    
    ## (3) Construct the animation.
    def animate(i):
        j = int(i*n_sample_means/size)
        x = x_sample[np.where(n_sample==j)]
        y = y_sample[np.where(n_sample==j)]
        width = frequency_values[i] 
        
        scat.set_data(x, y)
        for k, b in enumerate(bar_h):
            b.set_width(width[k])
        return [each_bar for each_bar in bar_h] + [scat,]
        #return scat, bar_h --> It doesn't work.
    
    
    ## (4) Start the animation.
    anim = FuncAnimation(fig, animate, frames=size, interval=10, blit=True, repeat=True, repeat_delay=100)
    plt.show()
    
    
    ## (5) Save the image.
    #writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
    #anim.save('04_05_06_animation_description_of_population_and_sample.gif', writer=writer)
    
    

def animation_complete_description(size, n_sample_means, population, population_index, 
                                   n_sample, x_sample, y_sample, sample_means_int, 
                                   max_value, poisson_values):
    print("****************************************************")
    topic = "4.5.7 Population and Sample"; print("** %s\n" % topic)
    
    ## (1). Read the data & Init Variables
    frequency_values = []
    for n in range(size):
        v, f = np.unique(population[:n+1], return_counts=True)
        f = [int(f[np.where(v==k)]) if np.isin(k,v) else k*0 for k in poisson_values]
        frequency_values.append(f)
        
    frequency_samples = []
    for n in range(n_sample_means):
        vs, fs = np.unique(sample_means_int[:n+1], return_counts=True)
        fs = [int(fs[np.where(vs==k)]) if np.isin(k,vs) else k*0 for k in poisson_values]
        frequency_samples.append(fs)
    
    
    ## (2). Preparing the figure.
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 5.5))
    fig.suptitle(topic, color='darkblue', fontsize=15)
    plt.subplots_adjust(left=.1, bottom=.15, right=.9, top=.85, wspace=.2, hspace=.5);
    
    ax1.set_title('Sample from Population', weight='bold', color='red', fontsize=12)
    ax1.set_xlim(0, size); ax1.set_ylim(-0.5, max_value+.5); ax1.set_yticks(poisson_values)
    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.set_xlabel('n sample', fontsize=9);  ax1.set_ylabel('Population values', fontsize=9);
    ax1.plot(population_index, population, ls='None', marker='.', markerfacecolor='darkblue', markersize=3)
    scat, = ax1.plot([], [], ls='None', marker='s', markerfacecolor='darkred', markersize=3)
    ax1.legend(['Population','Sample'], loc='best')
    
    ax2.set_title('Population Histogram', weight='bold', color='red', fontsize=12)
    ax2.set_xlim(0,  np.max(f)+0.5); ax2.set_ylim(-0.5, max_value+.5); ax2.set_yticks(poisson_values)
    ax2.set_xlabel('Frequency', fontsize=9); ax2.set_ylabel('Population values', fontsize=9);
    bar_h = ax2.barh(list(poisson_values), f)
    
    ax3.set_title('Selected Samples', weight='bold', color='red', fontsize=12)
    ax3.set_xlim(0, n_sample_means); ax3.set_ylim(-0.5, max_value+.5); ax3.set_yticks(poisson_values)
    ax3.set_xlabel('n sample', fontsize=9);  ax3.set_ylabel('Sample Mean Values', fontsize=9);
    ax3.tick_params(labelsize=8)
    samp, = ax3.plot([], [], ls='None', marker='.', color='darkred', markersize=3)
    
    ax4.set_title('Sample Mean Histogram', weight='bold', color='red', fontsize=12)
    ax4.set_xlim(0,  np.max(fs)+0.5); ax4.set_ylim(-0.5, max_value+.5); ax4.set_yticks(poisson_values)
    ax4.set_xlabel('Frequency', fontsize=9); ax4.set_ylabel('Sample Mean values', fontsize=9);
    ax4.tick_params(labelsize=8)
    bar_s = ax4.barh(list(poisson_values), fs)

    ## (3) Construct the animation.
    def animate(i):
        j = int(i*n_sample_means/size)
        x = x_sample[np.where(n_sample==j)]
        y = y_sample[np.where(n_sample==j)]
        scat.set_data(x, y)
        
        x = x_sample[np.where(n_sample<=j)]
        y = y_sample[np.where(n_sample<=j)]
        samp.set_data(x, y)
        
        h = frequency_values[i] 
        for k, b in enumerate(bar_h):
            b.set_width(h[k])
        
        h = frequency_samples[i] 
        for k, b in enumerate(bar_s):
            b.set_width(h[k])
        
        return [each_bar for each_bar in bar_h] + [scat,] + [each_bar for each_bar in bar_s] + [samp,]
        
    
    ## (4) Start the animation.
    anim = FuncAnimation(fig, animate, frames=size, interval=10, blit=True, repeat=True, repeat_delay=100)
    plt.show()
    
    
    ## (5) Save the image.
    #writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
    #anim.save('04_05_07_animation_complete_description.gif', writer=writer)
    
    

def animation_hist_population(size, population, max_value, max_freq, poisson_values):
    print("****************************************************")
    topic = "4.5.8 Population Histogram"; print("** %s\n" % topic)
    
    ## (1). Read the data & Init Variables
    ## (2). Preparing the figure.
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(topic, color='darkblue', fontsize=15)
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
    
    hp_n, hp_bins, hp_patches = ax.hist(population, bins=max_value, orientation='horizontal')
    
    ax.set_title('Population Histogram', weight='bold', color='red', fontsize=20)
    ax.set_xlabel('Frequency'); ax.set_ylabel('Population values');
    ax.set_xlim(0, max_freq)
    ax.set_yticks(poisson_values)
    
    label = ax.text(.8, .9, "                ", ha='center', transform=ax.transAxes, fontsize=14, backgroundcolor='#a36666', color='snow')
        
    ## (3) Construct the animation.
    def animate(i):
        n, bins = np.histogram(population[:i+1], bins=poisson_values)
        for rect, h in zip(hp_patches, n):
            rect.set_width(h)
        label.set_text(" Día: {:.0f}/1000 ".format(i+1))
        return [each_bar for each_bar in hp_patches] + [label]
        
    ## (4) Start the animation.
    anim = FuncAnimation(fig, animate, frames=size, interval=10, blit=True, repeat=True, repeat_delay=100)
    plt.show()
    
    ## (5) Save the image.
    #writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
    #anim.save('04_05_08_Hist_Population.gif', writer=writer)

    

    
def animation_complete_description_hist(size, n_sample_means, population, population_index,
                                        n_sample, x_sample, y_sample, sample_means,
                                        max_value, max_freq, poisson_values):
    print("****************************************************")
    topic = "4.5.9 Complete Description of Population and Sample"; print("** %s\n" % topic)
    
    ## (1). Read the data & Init Variables
    ## (2). Preparing the figure.
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 5.5))
    fig.suptitle(topic, color='darkblue', fontsize=15)
    plt.subplots_adjust(left=.1, bottom=.15, right=.9, top=.85, wspace=.2, hspace=.5);
    
    ax1.set_title('Sample from Population', weight='bold', color='red', fontsize=12)
    ax1.set_xlim(0, size); ax1.set_ylim(-0.5, max_value+.5); ax1.set_yticks(poisson_values)
    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.set_xlabel('n sample', fontsize=9);  ax1.set_ylabel('Population values', fontsize=9);
    ax1.plot(population_index, population, ls='None', marker='.', markerfacecolor='darkblue', markersize=3)
    scat, = ax1.plot([], [], ls='None', marker='s', markerfacecolor='darkred', markersize=3)
    ax1.legend(['Population','Sample'], loc='best')
    
    ax2.set_title('Population Histogram', weight='bold', color='red', fontsize=12)
    hp_n, hp_bins, hp_patches = ax2.hist(population, bins=poisson_values, orientation='horizontal')
    ax2.set_xlabel('Frequency', fontsize=9); ax2.set_ylabel('Population values', fontsize=9);
    ax2.set_xlim(0, max_freq)
    ax2.set_yticks(poisson_values)
    label = ax2.text(.8, .8, "                ", ha='center', transform=ax2.transAxes, fontsize=10, backgroundcolor='#a36666', color='snow')
    
    ax3.set_title('Selected Samples', weight='bold', color='red', fontsize=12)
    ax3.set_xlim(0, n_sample_means); ax3.set_ylim(-0.5, max_value+.5); ax3.set_yticks(poisson_values)
    ax3.set_xlabel('n sample', fontsize=9);  ax3.set_ylabel('Sample Mean Values', fontsize=9);
    ax3.tick_params(labelsize=8)
    samp, = ax3.plot([], [], ls='None', marker='.', color='darkred', markersize=3)
    
    ax4.set_title('Sample Mean Histogram', weight='bold', color='red', fontsize=12)
    hs_n, hs_bins, hs_patches = ax4.hist(sample_means, bins=poisson_values, orientation='horizontal')
    ax4.set_xlabel('Frequency', fontsize=9); ax2.set_ylabel('Sample Mean values', fontsize=9);
    ax4.set_xlim(0, max_freq)
    ax4.set_yticks(poisson_values)
    
    
    
    ## (3) Construct the animation.
    def animate(i):
        j = int(i*n_sample_means/size)
        x = x_sample[np.where(n_sample==j)]
        y = y_sample[np.where(n_sample==j)]
        scat.set_data(x, y)
        
        x = x_sample[np.where(n_sample<=j)]
        y = y_sample[np.where(n_sample<=j)]
        samp.set_data(x, y)
        
        n, bins = np.histogram(population[:i+1], bins=poisson_values)
        for rect, h in zip(hp_patches, n):
            rect.set_width(h)
        label.set_text(" Día: {:.0f}/1000 ".format(i+1))
        
        n, bins = np.histogram(sample_means[:j+1], bins=poisson_values)
        for rect, h in zip(hs_patches, n):
            rect.set_width(h)
        
        return [each_bar for each_bar in hp_patches] + [label] + [scat,] + [samp,] + [each_bar for each_bar in hs_patches]
        
    
    ## (4) Start the animation.
    anim = FuncAnimation(fig, animate, frames=size, interval=10, blit=True, repeat=True, repeat_delay=100)
    plt.show()
    
    
    ## (5) Save the image.
    #writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
    #anim.save('04_05_09_hist_complete_description.gif', writer=writer)
    
    


def main():
    print("****************************************************")
    topic = "5. Adding random variables"; print("** %s\n" % topic)
    
    mu=2
    #size=100; n_sample_means=10  #Short example
    size=1000; n_sample_means=350  #Large example
    seed_population=20
    seed_sample_mean=42

    population, population_index, max_value, max_freq, poisson_values, n_sample, x_sample, y_sample, sample_means, sample_means_int = static_image(
                                        mu=mu, 
                                        size=size, n_sample_means=n_sample_means, 
                                        seed_population=seed_population, 
                                        seed_sample_mean=seed_sample_mean)
    
    animation_error(mu=mu, 
                    size=size, n_sample_means=n_sample_means, 
                    seed_population=seed_population, seed_sample_mean=seed_sample_mean,
                    max_value=max_value, poisson_values=poisson_values)
    
    animation_population(mu=mu, 
                         size=size, n_sample_means=n_sample_means, 
                         seed_population=seed_population, seed_sample_mean=seed_sample_mean,
                         population=population,
                         max_value=max_value, poisson_values=poisson_values)
    
    animation_population_and_sample(mu=mu, 
                                    size=size, n_sample_means=n_sample_means, 
                                    seed_population=seed_population, seed_sample_mean=seed_sample_mean,
                                    population=population, population_index=population_index,
                                    n_sample=n_sample, x_sample=x_sample, y_sample=y_sample,
                                    max_value=max_value, poisson_values=poisson_values)
    
    animation_frequency_population(size=size, population=population, 
                                   max_value=max_value, poisson_values=poisson_values)
    
    animation_description_of_population(size=size, population=population, 
                                        max_value=max_value, poisson_values=poisson_values)
    
    animation_description_of_population_and_sample(size=size, n_sample_means=n_sample_means,
                                                   population=population, population_index=population_index,
                                                   n_sample=n_sample, x_sample=x_sample, y_sample=y_sample,
                                                   max_value=max_value, poisson_values=poisson_values)
    
    animation_complete_description(size=size, n_sample_means=n_sample_means,
                                   population=population, population_index=population_index, 
                                   n_sample=n_sample, x_sample=x_sample, y_sample=y_sample, sample_means_int=sample_means_int, 
                                   max_value=max_value, poisson_values=poisson_values)
    
    animation_hist_population(size=size, population=population, 
                              max_value=max_value, max_freq=max_freq,
                              poisson_values=poisson_values)
    
    animation_complete_description_hist(size=size, n_sample_means=n_sample_means, 
                                        population=population, population_index=population_index,
                                        n_sample=n_sample, x_sample=x_sample, y_sample=y_sample, sample_means=sample_means,
                                        max_value=max_value, max_freq=max_freq, poisson_values=poisson_values)
    
    
if __name__ == '__main__':
    main()