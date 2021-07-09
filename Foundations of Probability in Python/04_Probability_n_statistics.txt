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
import matplotlib.pyplot             as plt
import matplotlib.ticker             as ticker
import numpy                         as np                                    #For making operations in lists
import pandas                        as pd

from matplotlib.animation            import FuncAnimation
from matplotlib.animation            import PillowWriter
from scipy.stats                     import binom
from scipy.stats                     import describe
from scipy.stats                     import geom                              #Generate gometric distribution
from scipy.stats                     import poisson
from sklearn.linear_model            import LinearRegression                  #Calculate a linear least-squares regression for two sets of measurements. To get the parameters (slope and intercept) from a model
from sklearn.linear_model            import LogisticRegression                #Logistic Regression (aka logit, MaxEnt) classifier.
from scipy.special                   import expit                             #Inverse of the logistic function.
from scipy.stats                     import linregress                        #Fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

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
    
    
def roll_dice(size=1, sides=6, SEED=None):
    """
    Generate dice roll simulations
    
    Parameters
    ----------
    size  : int, optional. The number of dice rolls to simulate. The default is 1.
    sides : int, optional. The number of sides of the dice. The default is 6.
    SEED  : None or int. If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``. Default is None.
    
    Returns
    -------
    list, a list with num_rolls simulations of dice rolls
    """
    if SEED != None:
        np.random.seed(SEED)
    return np.random.randint(1, sides+1, size)
    
    
###############################################################################
## Main part of the code
###############################################################################
def From_Sample_Mean_to_Population_Mean(n, SEED=SEED):
    print("****************************************************")
    topic = "1. From sample mean to population mean"; print("** %s\n" % topic)
    ## (1.1). Init Values
    #n = 250 
    
    ## (1.2). Preparing the figure.
    fig = plt.figure()
    ax = plt.axes(xlim=(0, n), ylim=(0, 1))
    line, = ax.plot([], [], lw=2, color='darkblue', label='Sample mean')
    label = ax.text(n/2, 0.6, "", ha='center', fontsize=17, backgroundcolor='rosybrown', color='darkblue')
    plt.axhline(y=0.5, lw=2, color='red', label='Population Mean')
    plt.title("Fair coin flips Tendency", weight='bold', color='red', fontsize=20)
    plt.suptitle(topic, color='darkblue', fontsize=15)
    plt.xlabel("Size of coin flips' sample", fontsize=20, color='gray')
    plt.ylabel('Sample Mean', fontsize=20, color='gray', rotation=90)
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
    #anim.save('04_01_Fair_Coin.gif', writer=writer)    
    
    
def Generating_a_Sample(p, SEED=SEED):
    print("****************************************************")
    topic = "2. Generating a sample"; print("** %s\n" % topic)
    
    # Generate a sample of 250 newborn children
    sample = binom.rvs(n=1, p=p, size=250, random_state=SEED)

    # Show the sample values
    print("A hospital's planning department is investigating different treatments \
           for newborns. As a data scientist you are hired to simulate the sex of \
           250 newborn children, and you are told that on average 50.50% are males.")
    print(sample)
    return sample, p

def Calculating_the_Sample_Mean(sample, size=[10,50,250], SEED=SEED):
    print("****************************************************")
    topic = "3. Calculating the sample mean"; print("** %s\n" % topic)
    
    for n in size: 
        print("Sample mean of the first {:3d} samples: {}".format(n, describe(sample[0:n]).mean))

def Plotting_the_Sample_Mean(sample, p, SEED=SEED):
    print("****************************************************")
    topic = "4. Plotting the sample mean"; print("** %s\n" % topic)
    
    title    = "Male Newborns simulate samples"
    x_label  = "Size of Male Newborns sample"
    y_label  = "Sample Mean"
    mean_pop = binom.mean(n=1, p=p)
    
    # Calculate sample mean and store it on averages array
    averages = []
    for i in range(2, len(sample)+1):
        averages.append(describe(sample[0:i]).mean)
    
    fig = plt.figure()
    # Add population mean line and sample mean plot
    plt.axhline(mean_pop, color='red')
    plt.plot(averages, '-')
    
    # Add legend
    plt.legend(("Population mean","Sample mean"), loc='upper right')
    plt.title(title, weight='bold', color='red', fontsize=20)
    plt.suptitle(topic, color='darkblue', fontsize=15)
    plt.xlabel(x_label, fontsize=20, color='gray')
    plt.ylabel(y_label, fontsize=20, color='gray', rotation=90)
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
    plt.show()
    
    
    
    ## (1.1). n elements in the Sample
    n = len(sample)
    
    ## (1.2). Preparing the figure.
    fig = plt.figure()
    ax = plt.axes(xlim=(0, n), ylim=(0, 1))
    line, = ax.plot([], [], lw=2, color='darkblue', label='Sample mean')
    label = ax.text(n/2, 0.8, "", ha='center', fontsize=17, backgroundcolor='rosybrown', color='darkblue')
    plt.axhline(y=mean_pop, lw=2, color='red', label='Population Mean')
    plt.title(title, weight='bold', color='red', fontsize=20)
    plt.suptitle(topic, color='darkblue', fontsize=15)
    plt.xlabel(x_label, fontsize=20, color='gray')
    plt.ylabel(y_label, fontsize=20, color='gray', rotation=90)
    plt.legend(loc='best')
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
    
    
    ## (1.3) Construct the animation.
    def init():
        line.set_data([], [])
        label.set_text("                 ")
        return line, label
    
    
    def animate(i):
        x = [ii for ii in range(1,i+1)]
        y = averages[0:i]
        line.set_data(x, y)
        label.set_text("     {:.5f}     ".format(binom.rvs(n=1, p=p, size=(i+1), random_state=SEED).mean()))
        return line, label
    
    
    ## (1.4) Start the animation.
    anim = FuncAnimation(fig, animate, init_func=init, frames=n, interval=100, blit=True, repeat=True)
    plt.show()
    
    ## (1.5) Save the image.
    #anim.save("04_04_Male_Newborns_samples.gif", fps=20, writer='pillow')
        
def Adding_Random_Variables(mu, size, n_sample_means, seed_sample_generation=SEED, seed_sample_mean=SEED):
    print("****************************************************")
    topic = "5. Adding random variables"; print("** %s\n" % topic)
    
    ###########################################################################
    # Poisson population plot
    ###########################################################################
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
    # Population Plot
    ###############################################################################
    
    plt.figure()
    plt.xlabel("n day")
    plt.ylabel("Ocurred accidents per day")
    plt.yticks(poisson_values)
    plt.title("Population registered during 1,000 days", color='red')
    plt.suptitle(topic, color='navy');  # Setting the titles.
    plt.plot(range(1,size+1), population, ls='None', marker='.', markersize=3)
    plt.show()
        
    ###########################################################################
    # Sample Means Plot
    ###########################################################################
    np.random.seed(seed_sample_mean)
    population_index = np.arange(1,size+1)
    sample_means = []
    x_sample = np.array([])
    y_sample = np.array([])
    n_sample = np.array([])
    for n in range(n_sample_means):
        # Select 10 from population
        choices= np.random.choice(population_index, 10)        
        sample = population[choices]        
        sample_means.append(describe(sample).mean)
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
    
    ###########################################################################
    # The animation
    ###########################################################################
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
    ax1.tick_params(labelsize=8)
    ax1.legend(['Population','Sample'], loc='best')
    
    ax2.set_title('Population Histogram', weight='bold', color='red', fontsize=12)
    hp_n, hp_bins, hp_patches = ax2.hist(population, bins=poisson_values, orientation='horizontal')
    ax2.set_xlabel('Frequency', fontsize=9); ax2.set_ylabel('Population values', fontsize=9);
    ax2.set_xlim(0, max_freq)
    ax2.set_yticks(poisson_values)
    ax2.tick_params(labelsize=8)
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
    ax4.tick_params(labelsize=8)
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
        
    
def Sample_Means(size, n_sample_means, SEED=SEED):
    print("****************************************************")
    topic = "6. Sample means"; print("** %s\n" % topic)
    
    p=.5 #Fair coin
    population = binom.rvs(n=10, p=p, size=size)
    mean_pop = binom.mean(n=10, p=p)
    
    # Create list for sample means
    sample_means = []
    for _ in range(n_sample_means):
    	# Take 20 values from the population
        	sample = np.random.choice(population, 20)
          	# Calculate the sample mean
        	sample_means.append(describe(sample).mean)
        
    # Plot the histogram
    fig = plt.figure(figsize=(11,3))
    fig.suptitle(topic, color='darkblue', fontsize=10)
    
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Population\nBinomial Distribution', weight='bold', color='red', fontsize=14)
    ax.set_xlabel('Population Values', fontsize=9);  
    ax.set_ylabel('Frequencies', fontsize=9);
    ax.axvline(mean_pop, color='r', linestyle='--', linewidth=2)
    ax.hist(population)
    
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Sample Mean', weight='bold', color='red', fontsize=14)
    ax.set_xlabel("Sample mean values", fontsize=9); 
    ax.set_ylabel("Frequency", fontsize=9)
    ax.axvline(mean_pop, color='r', linestyle='--', linewidth=2)
    ax.hist(sample_means)
    
    plt.subplots_adjust(top=.9);
    plt.tight_layout() #To fix overlapping the supplots elements.
    plt.show()
    
    print("The sample mean follows a normal distribution. This is the central limit theorem in action.")



def Sample_Means_Follow_a_Normal_Distribution(size, n_sample_means, SEED=SEED):
    print("****************************************************")
    topic = "7. Sample means follow a normal distribution"; print("** %s\n" % topic)
    # Init values
    p=.5 
    mu=2
    
    # Prepare the figure
    fig = plt.figure(figsize=(11,5.5))
    fig.suptitle(topic, color='darkblue', fontsize=10)
    
    # Generate the population and sample from geom distribution
    population_g   = geom.rvs(p=p, size=size)
    mean_pop_g     = geom.mean(p=p)
    sample_means_g = []
    for _ in range(n_sample_means):
        sample = np.random.choice(population_g, 20)  # Take 20 values from the population
        sample_means_g.append(describe(sample).mean) # Calculate the sample mean
    
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('Population\nGeom Distribution', weight='bold', color='red', fontsize=14)
    ax.set_xlabel('Population Values', fontsize=9);  
    ax.set_ylabel('Frequencies', fontsize=9);
    ax.axvline(mean_pop_g, color='r', linestyle='--', linewidth=2)
    ax.tick_params(labelsize=8)
    ax.hist(population_g)
    
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title('Sample Mean\n ', weight='bold', color='red', fontsize=14)
    ax.set_xlabel("Sample mean values", fontsize=9); 
    ax.set_ylabel("Frequency", fontsize=9)
    ax.axvline(mean_pop_g, color='r', linestyle='--', linewidth=2)
    ax.tick_params(labelsize=8)
    ax.hist(sample_means_g)
    
    # Generate the population and sample from poisson distribution
    population_p   = poisson.rvs(mu=mu, size=size)
    mean_pop_p     = poisson.mean(mu=mu)
    sample_means_p = []
    for _ in range(n_sample_means):
        sample = np.random.choice(population_p, 20)  # Take 20 values from the population
        sample_means_p.append(describe(sample).mean) # Calculate the sample mean
    
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title('Population\nPoisson Distribution', weight='bold', color='red', fontsize=14)
    ax.set_xlabel('Population Values', fontsize=9);  
    ax.set_ylabel('Frequencies', fontsize=9);
    ax.axvline(mean_pop_p, color='r', linestyle='--', linewidth=2)
    ax.tick_params(labelsize=8)
    ax.hist(population_p)
    
    ax = fig.add_subplot(2, 2, 4)
    ax.set_title('Sample Mean\n ', weight='bold', color='red', fontsize=14)
    ax.set_xlabel("Sample mean values", fontsize=9); 
    ax.set_ylabel("Frequency", fontsize=9)
    ax.axvline(mean_pop_p, color='r', linestyle='--', linewidth=2)
    ax.tick_params(labelsize=8)
    ax.hist(sample_means_p)
    
    # Plot the histogram
    plt.subplots_adjust(top=.9);
    plt.tight_layout() #To fix overlapping the supplots elements.
    plt.show()
    
    print("The sample mean follows a normal distribution. This is the central limit theorem in action.")
    
    
def Adding_Dice_Rolls(size, SEED=SEED):
    print("****************************************************")
    topic = "8. Adding dice rolls"; print("** %s\n" % topic)

    # Prepare the figure
    fig = plt.figure(figsize=(12,3))
    fig.suptitle(topic, color='darkblue', fontsize=10)
    
    # A sample of 2000 dice rolls 
    np.random.seed(SEED) 
    sample1 = roll_dice(size=size) # Generate the sample
    bins = range(sample1.min(), sample1.max()+2) #range(1, 8)
    x_ticks = np.linspace(sample1.min(), sample1.max()+1, dtype=int) #[1, 2, 3, 4, 5, 6, 7]
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('A sample of {} once dice rolls'.format(size), weight='bold', color='red', fontsize=10)
    ax.set_xlabel('Values', fontsize=7);  
    ax.set_ylabel('Frequencies', fontsize=7);
    ax.hist(sample1, bins=bins, width=0.9)
    ax.tick_params(labelsize=6)
    ax.set_xticks(x_ticks)
    
    # Generate two samples of 2000 dice rolls
    np.random.seed(SEED) 
    sample1 = roll_dice(size=size)
    sample2 = roll_dice(size=size)
    sum_of_1_and_2 = np.add(sample1, sample2) # Add the first two samples
    bins = range(sum_of_1_and_2.min(), sum_of_1_and_2.max()+2) #range(2, 14)
    x_ticks = np.linspace(sum_of_1_and_2.min(), sum_of_1_and_2.max()+1, dtype=int) #[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('A sample of {} twice dice rolls'.format(size), weight='bold', color='red', fontsize=10)
    ax.set_xlabel('Values', fontsize=7);  
    ax.set_ylabel('Frequencies', fontsize=7);
    ax.hist(sum_of_1_and_2, bins=bins, width=0.9)
    ax.tick_params(labelsize=6)
    ax.set_xticks(x_ticks)
    
    # Generate three samples of 2000 dice rolls
    np.random.seed(42)
    sample1 = roll_dice(size=size)
    sample2 = roll_dice(size=size)
    sample3 = roll_dice(size=size)
    sum_of_1_and_2 = np.add(sample1, sample2) # Add the first two samples
    sum_of_3_samples = np.add(sum_of_1_and_2, sample3) # Add the first two with the third sample
    bins = range(sum_of_3_samples.min(), sum_of_3_samples.max()+2) #range(3, 20)
    x_ticks = np.linspace(sum_of_3_samples.min(), sum_of_3_samples.max()+1, dtype=int) #[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('A sample of {} thrice dice rolls'.format(size), weight='bold', color='red', fontsize=10)
    ax.set_xlabel('Values', fontsize=7);  
    ax.set_ylabel('Frequencies', fontsize=7);
    ax.hist(sum_of_3_samples, bins=bins, width=0.9)
    ax.tick_params(labelsize=6)
    ax.set_xticks(x_ticks)
    
    # Plot the histogram
    plt.subplots_adjust(left=.1, bottom=.15, right=.9, top=.8, wspace=.4);                                    #To set the marginsplt.show() 
    
    print("The central limit theorem states that the sum of equally distributed random variables converges to a normal distribution.")



def Linear_Regression(hours_of_study, scores, hours_to_predict_score, SEED=SEED):
    print("****************************************************")
    topic = "9. Linear regression"; print("** %s\n" % topic)

    #hours_of_study = [ 4,  8,  8, 12,  8,  9,  6, 11, 13, 13, 19, 16, 17, 17, 21, 21, 23, 27, 30, 24]
    #scores         = [52, 54, 61, 63, 63, 60, 61, 70, 75, 77, 76, 79, 81, 83, 85, 86, 88, 90, 95, 93]
    
    #hours_of_study = [ 4,  6,  8,  8,  8,  9, 11, 12, 13, 13, 16, 17, 17, 19, 21, 21, 23, 24, 27, 30]
    #scores         = [52, 61, 54, 61, 63, 60, 70, 63, 75, 77, 79, 81, 83, 76, 85, 86, 88, 93, 90, 95]

    # sklearn linear model
    model = LinearRegression()
    model.fit(np.array(hours_of_study).reshape(-1,1), scores)
    
    # Get parameters
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Print parameters
    print('slope:', slope)         # Slope of the regression line.
    print('intercept:', intercept) # Intercept of the regression line.

    # Score prediction
    score = model.predict(np.array([[hours_to_predict_score]]))
    print('score predicted for value {:5,.1f} hours:'.format(hours_to_predict_score), score)
    
    fig = plt.figure()
    fig.suptitle(topic, color='darkblue', fontsize=10)
    
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Relation between hours of study and scores', weight='bold', color='red', fontsize=10)
    ax.set_xlabel('Hours of study');  
    ax.set_ylabel('Scores');
    ax.scatter(hours_of_study, scores, s=30, c='indianred', label='Data')
    ax.plot(hours_of_study, model.predict(np.reshape(hours_of_study, (-1,1))), color='darkslategray', lw=2, label='Model')
    ax.legend(loc='best')
    plt.show()
    
    
        
def Fitting_a_Model(hours_of_study, scores, SEED=SEED):
    print("****************************************************")
    topic = "10. Fitting a model"; print("** %s\n" % topic)
    
    #hours_of_study = [ 4,  8,  8, 12,  8,  9,  6, 11, 13, 13, 19, 16, 17, 17, 21, 21, 23, 27, 30, 24]
    #scores         = [52, 54, 61, 63, 63, 60, 61, 70, 75, 77, 76, 79, 81, 83, 85, 86, 88, 90, 95, 93]
    
    # Get the model parameters
    slope, intercept, r_value, p_value, std_err = linregress(hours_of_study, scores)
    
    # Print the linear model parameters
    print('slope:', slope) # Slope of the regression line.
    print('intercept:', intercept) # Intercept of the regression line.
    print('correlation:', r_value) # Correlation coefficient.
    print('p_value:', p_value) # Two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero, using Wald Test with t-distribution of the test statistic.
    print('std_err:', std_err) # Standard error of the estimated gradient.
    return slope, intercept
    
    
def Predicting_Test_Scores(slope, intercept, many_hours_to_predict_score, SEED=SEED):
    print("****************************************************")
    topic = "11. Predicting test scores"; print("** %s\n" % topic)
    
    # Get the predicted test score for given hours of study
    for hours_to_predict_score in many_hours_to_predict_score:
        score = slope*hours_to_predict_score + intercept
        print('score predicted for value {:5,.1f}:'.format(hours_to_predict_score), score)
        
    
def Studying_Residuals(SEED=SEED):
    print("****************************************************")
    topic = "12. Studying residuals"; print("** %s\n" % topic)
    ## To implement a linear model you must study the residuals, which are the distances between 
    ## the predicted outcomes and the data.
    ## Three conditions must be met:
    ## 1.	The mean should be 0.
    ## 2.	The variance must be constant.
    ## 3.	The distribution must be normal.
    
    # Prepare the data
    hours_of_study_A        = [ 4,  9,  7, 12,  3,  9,  6, 11, 13, 13, 19, 16, 17, 17, 13, 21, 23, 27, 30, 24]
    test_scores_A           = [52, 56, 59, 60, 61, 62, 63, 73, 75, 77, 76, 79, 81, 83, 85, 87, 89, 89, 89, 93]
    hours_of_study_B        = [ 4,  9,  7, 12,  3,  9,  6, 11, 13, 13, 19, 16, 17, 17, 13, 21, 23, 27, 30, 24, 17, 17, 19, 19, 19, 19]
    test_scores_B           = [58, 70, 60, 65, 57, 63, 63, 73, 65, 77, 58, 62, 62, 90, 85, 95, 97, 95, 65, 65, 70, 75, 65, 75, 85, 93]
    hours_of_study_values_A = np.linspace(1, 30.5, 60)
    hours_of_study_values_B = np.linspace(1, 30.5, 60)
    
    # sklearn linear model
    model_A = LinearRegression()
    model_A.fit(np.array(hours_of_study_A).reshape(-1,1), test_scores_A)
    model_B = LinearRegression()
    model_B.fit(np.array(hours_of_study_B).reshape(-1,1), test_scores_B)
    
    
    # Make the graphic analysis
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 5.5))
    fig.suptitle(topic, color='darkblue', fontsize=15)
    
    # Model A
    ax1.set_title('Model of case A', weight='bold', color='red', fontsize=10)
    ax1.set_xlabel('Hours of study', fontsize=7);  
    ax1.set_ylabel('Scores', fontsize=7);
    ax1.tick_params(labelsize=6)
    ax1.scatter(hours_of_study_A, test_scores_A, s=30, c='indianred', label='Data')
    ax1.plot(hours_of_study_values_A, model_A.predict(hours_of_study_values_A.reshape(-1,1)), color='darkslategray', lw=2, label='Model')
    ax1.legend(loc='best', fontsize=7)
    
    # Model A - Residuals
    residuals_A = model_A.predict(np.reshape(hours_of_study_A, (-1,1))) - test_scores_A
    ax2.set_title('Residual plot of Model A (Mean={:,.2f})'.format(describe(residuals_A).mean), weight='bold', color='red', fontsize=10)
    ax2.set_xlabel('Hours of study', fontsize=7);  
    ax2.set_ylabel('Residuls', fontsize=7);
    ax2.tick_params(labelsize=6)
    ax2.scatter(hours_of_study_A, residuals_A, s=30, c='olivedrab', label='Residuals')
    ax2.hlines(0, 0, 30, colors='r', linestyles='--', label='Reference line')
    ax2.legend(loc='lower right', fontsize=7)

    # Model A - Residuals Histogram
    ax3.set_title('Histogram of Residual Model A)', weight='bold', color='red', fontsize=10)
    ax3.set_xlabel('Residuals Values', fontsize=7);  
    ax3.set_ylabel('Frequencies', fontsize=7);
    ax3.tick_params(labelsize=6)
    ax3.hist(residuals_A)

    # Model B
    ax4.set_title('Model of case B', weight='bold', color='red', fontsize=10)
    ax4.set_xlabel('Hours of study', fontsize=7);  
    ax4.set_ylabel('Scores', fontsize=7);
    ax4.tick_params(labelsize=6)
    ax4.scatter(hours_of_study_B, test_scores_B, s=30, c='indianred', label='Data')
    ax4.plot(hours_of_study_values_B, model_B.predict(hours_of_study_values_B.reshape(-1,1)), color='darkslategray', lw=2, label='Model')
    ax4.legend(loc='best', fontsize=7)
    
    # Model B - Residuals
    residuals_B = model_A.predict(np.reshape(hours_of_study_B, (-1,1))) - test_scores_B
    ax5.set_title('Residual plot of Model B (Mean={:,.2f})'.format(describe(residuals_B).mean), weight='bold', color='red', fontsize=10)
    ax5.set_xlabel('Hours of study', fontsize=7);  
    ax5.set_ylabel('Residuls', fontsize=7);
    ax5.tick_params(labelsize=6)
    ax5.scatter(hours_of_study_B, residuals_B, s=30, c='olivedrab', label='Residuals')
    ax5.hlines(0, 0, 30, colors='r', linestyles='--', label='Reference line')
    ax5.legend(loc='best', fontsize=7)

    # Model B - Residuals Histogram
    ax6.set_title('Histogram of Residual Model B)', weight='bold', color='red', fontsize=10)
    ax6.set_xlabel('Residuals Values', fontsize=7);  
    ax6.set_ylabel('Frequencies', fontsize=7);
    ax6.tick_params(labelsize=6)
    ax6.hist(residuals_B)

    plt.subplots_adjust(left=.1, bottom=.1, right=.9, top=.87, wspace=.4, hspace=.4);                                    #To set the marginsplt.show() 
    plt.show()


    
def Logistic_Regression(hours_of_study, outcomes, hours_to_predict, SEED=SEED):
    print("****************************************************")
    topic = "13. Logistic regression"; print("** %s\n" % topic)

    #hours_of_study   = [ 4,  8,  8, 12,  8,  9,  6, 11, 13, 13, 19, 16, 17, 17, 21, 21, 23, 27, 30, 24]
    #outcomes         = [False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
    #hours_to_predict = 10
    
    hours_of_study   = np.reshape(hours_of_study, (-1,1))
    hours_to_predict = [[hours_to_predict]]
    
    # sklearn logistic model
    model = LogisticRegression(C=1e9, random_state=SEED)
    model.fit(hours_of_study, outcomes)
    
    # Get parameters
    beta1 = model.coef_[0][0] #slope
    beta0 = model.intercept_[0] #intercept
    
    # Print parameters
    print('beta1 (slope):', beta1) # Slope of the logistic regression.
    print('beta0 (intercept):', beta0) # Intercept of the Logistic regression.
    
    # Predicting outcomes based on hours of study
    outcome = model.predict(hours_to_predict)
    print("Predicted outcome for {} hours of study: {}".format(hours_to_predict[0][0], ('Pass' if outcome else 'Fail')))
    
    # Probability calculation
    outcome_probability = model.predict_proba(hours_to_predict)[:,1]
    print("{} hours of study has {:,.2f} probability to Pass the test.".format(hours_to_predict[0][0], float(outcome_probability)))
    
    print("Minimal hours of study to pass (0.5 probability): ", -beta0/beta1)
    
    
def Fitting_a_Logistic_Model(hours_of_study, outcomes, SEED=SEED):
    print("****************************************************")
    topic = "14. Fitting a logistic model"; print("** %s\n" % topic)

    #hours_of_study   = [ 4,  8,  8, 12,  8,  9,  6, 11, 13, 13, 19, 16, 17, 17, 21, 21, 23, 27, 30, 24]
    #outcomes         = [False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
    
    hours_of_study   = np.reshape(hours_of_study, (-1,1))
    
    # sklearn logistic model
    model = LogisticRegression(C=1e9, random_state=SEED)
    model.fit(hours_of_study, outcomes)
    
    # Get parameters
    beta1 = model.coef_[0][0]
    beta0 = model.intercept_[0]
    
    # Print parameters
    print('beta1 (slope):', beta1) # Slope of the logistic regression.
    print('beta0 (intercept):', beta0) # Intercept of the Logistic regression.
    print("Minimal hours of study to pass (0.5 probability): ", -beta0/beta1)


    
def Predicting_If_Students_Will_Pass(hours_of_study, outcomes, many_hours_to_predict, SEED=SEED):
    print("****************************************************")
    topic = "15. Predicting if students will pass"; print("** %s\n" % topic)

    #hours_of_study        = [ 4,  8,  8, 12,  8,  9,  6, 11, 13, 13, 19, 16, 17, 17, 21, 21, 23, 27, 30, 24]
    #outcomes              = [False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
    #many_hours_to_predict = [10, 11, 12, 13, 14]
    
    hours_of_study   = np.reshape(hours_of_study, (-1,1))
    many_hours_to_predict   = np.reshape(many_hours_to_predict, (-1,1))
    
    # sklearn logistic model
    model = LogisticRegression(C=1e9, random_state=SEED)
    model.fit(hours_of_study, outcomes)
    
    # Get parameters
    beta1 = model.coef_[0][0]
    beta0 = model.intercept_[0]
    
    # Pass values to predict
    predicted_outcomes  = model.predict(many_hours_to_predict)
    probability_to_pass = model.predict_proba(many_hours_to_predict)[:,1]
    for i in range(len(many_hours_to_predict)):
        print('score predicted for {:3,.0f} hours of study is: {}.'.format(many_hours_to_predict[i][0], ('Pass' if predicted_outcomes[i] else 'Fail')))
        print("Probability of passing test: {}\n ".format(probability_to_pass[i]))
    
    print("Minimal hours of study to pass (0.5 probability): ", -beta0/beta1)

def Passing_Two_Tests(SEED=SEED):
    print("****************************************************")
    topic = "16. Passing two tests"; print("** %s\n" % topic)

    ###########################################################################
    subtopic = "PREPARING THE MODELS A AND B"; print("{}\n{}\n{}".format("-"*52,subtopic,"-"*52))
    ###########################################################################
    hours_of_study_subject_A   = [ 8,  9,  7, 12,  6, 11,  7, 11, 13, 13, 19, 16, 17, 17, 16, 17, 18, 16, 20, 19]
    hours_of_study_subject_B   = [ 4,  5,  4,  6,  3,  6,  4,  6,  7,  7, 10,  8,  9,  9,  8,  9,  9,  8, 10, 10]
    outcome_A                  = [False, True, False, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
    outcome_B                  = [False, True, False, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True]

    #Prepare the data
    t_hours_of_study_subject_A = np.reshape(hours_of_study_subject_A, (-1,1))
    t_hours_of_study_subject_B = np.reshape(hours_of_study_subject_B, (-1,1))
    
    # Prepare the logistic model
    model_A                    = LogisticRegression(C=1e9, random_state=SEED)
    model_B                    = LogisticRegression(C=1e9, random_state=SEED)
    
    model_A.fit(t_hours_of_study_subject_A, outcome_A)
    model_B.fit(t_hours_of_study_subject_B, outcome_B)
    
    # Inflexion point    
    print("Minimal hours of study to pass (0.5 probability) subject A: ", -model_A.intercept_[0]/model_A.coef_[0][0])
    print("Minimal hours of study to pass (0.5 probability) subject B: ", -model_B.intercept_[0]/model_B.coef_[0][0])
    
    
    
    ###########################################################################
    subtopic = "PREDICTING TO DIFFERENT HOURS OF STUDY"; print("\n{}\n{}\n{}".format("-"*52,subtopic,"-"*52))
    ###########################################################################
    # Specify values to predict
    hours_of_study_test_A      = [  6,  7,  8,  9, 10]
    hours_of_study_test_B      = [  3,  4,  5,  6]
    
    #Prepare the data
    t_hours_of_study_test_A    = np.reshape(hours_of_study_test_A, (-1,1))
    t_hours_of_study_test_B    = np.reshape(hours_of_study_test_B, (-1,1))
    
    # Prediciting the outcomes
    predicted_outcomes_A       = model_A.predict(t_hours_of_study_test_A)
    predicted_outcomes_B       = model_B.predict(t_hours_of_study_test_B)
    
    # Printing the outcomes
    data_model_A = pd.DataFrame({'Hours of Study': hours_of_study_test_A,
                                 "Predictions   ": predicted_outcomes_A})
    data_model_B = pd.DataFrame({'Hours of Study': hours_of_study_test_B,
                                 "Predictions"   : predicted_outcomes_B})
    print("Prediction for subject A: \n{}\n".format(data_model_A))
    print("Prediction for subject B: \n{}\n".format(data_model_B))
    
    
    
    ###########################################################################
    subtopic = "PREDICTING TO A SPECIFIC VALUE"; print("\n{}\n{}\n{}".format("-"*52,subtopic,"-"*52))
    ###########################################################################
    # Specify values to predict
    specific_hours_study_A     = 8.6
    specific_hours_study_B     = 4.7
    
    #Prepare the data
    specific_hours_study_A     = np.asarray([specific_hours_study_A]).reshape(-1,1)
    specific_hours_study_B     = np.asarray([specific_hours_study_B]).reshape(-1,1)
    
    # Prediciting the outcomes
    specific_score_study_A     = model_A.predict_proba(specific_hours_study_A)[:,1]
    specific_score_study_B     = model_B.predict_proba(specific_hours_study_B)[:,1]
    
    # Printing the prediction
    print("The probability of passing test A with {} hours of study is {}.".format(specific_hours_study_A, specific_score_study_A))
    print("The probability of passing test B with {} hours of study is {}.".format(specific_hours_study_B, specific_score_study_B))

    
    
    
    ###########################################################################
    subtopic = "JOINT PROBABILITY"; print("\n{}\n{}\n{}".format("-"*52,subtopic,"-"*52))
    ###########################################################################
    study_hours_A              = np.linspace(0, 13.5, 28)
    study_hours_B              = np.flip(np.linspace(0.5, 14, 28))
        
    #Prepare the data
    t_study_hours_A            = study_hours_A.reshape(-1,1)
    t_study_hours_B            = study_hours_B.reshape(-1,1)
    
    # Probability calculation for each value of study_hours
    prob_passing_A             = model_A.predict_proba(t_study_hours_A)[:,1]
    prob_passing_B             = model_B.predict_proba(t_study_hours_B)[:,1]
    
    # Calculate the probability of passing both tests
    prob_passing_A_and_B = prob_passing_A * prob_passing_B

    # Maximum probability value
    max_prob = max(prob_passing_A_and_B)

    # Position where we get the maximum value
    max_position = np.where(prob_passing_A_and_B == max_prob)[0][0]

    # Study hours for each test
    print("Study {:1.0f} hours for the first and {:1.0f} hours for the second test and you will pass both tests with {:6.2%} of probability.".format(study_hours_A[max_position], study_hours_B[max_position], max_prob))


    
def Wrapping_Up(SEED=SEED):
    print("****************************************************")
    topic = "17. Wrapping up"; print("** %s\n" % topic)



def main():
    From_Sample_Mean_to_Population_Mean(n=250)
    
    sample, p = Generating_a_Sample(p=.505)
    Calculating_the_Sample_Mean(sample)
    Plotting_the_Sample_Mean(sample, p)
    
    Adding_Random_Variables(mu=2, size=1000, n_sample_means=350, seed_sample_generation=20, seed_sample_mean=42)
    
    Sample_Means(size=1000, n_sample_means=1500)
    Sample_Means_Follow_a_Normal_Distribution(size=1000, n_sample_means=3000)
    Adding_Dice_Rolls(size=2000)
    
    hours_of_study = [4, 8, 8, 12, 8, 9, 6, 11, 13, 13, 19, 16, 17, 17, 21, 21, 23, 27, 30, 24]
    scores = [52, 54, 61, 63, 63, 60, 61, 70, 75, 77, 76, 79, 81, 83, 85, 86, 88, 90, 95, 93]
    Linear_Regression(hours_of_study=hours_of_study, scores=scores, hours_to_predict_score=15)
    
    slope, intercept = Fitting_a_Model(hours_of_study=hours_of_study, scores=scores)
    Predicting_Test_Scores(slope=slope, intercept=intercept, many_hours_to_predict_score=[10, 9, 12])
    Studying_Residuals()
    
    hours_of_study   = [ 4,  8,  8, 12,  8,  9,  6, 11, 13, 13, 19, 16, 17, 17, 21, 21, 23, 27, 30, 24]
    outcomes         = [False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
    Logistic_Regression(hours_of_study=hours_of_study, outcomes=outcomes, hours_to_predict = 10)
    
    Fitting_a_Logistic_Model(hours_of_study=hours_of_study, outcomes=outcomes)
    Predicting_If_Students_Will_Pass(hours_of_study=hours_of_study, outcomes=outcomes, many_hours_to_predict=[10, 11, 12, 13, 14])
    Passing_Two_Tests()
    Wrapping_Up()
    

if __name__ == '__main__':
    main()