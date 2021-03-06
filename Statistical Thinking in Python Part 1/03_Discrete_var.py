# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:28:15 2019

@author: jacqueline.cortez

Capítulo 3. Thinking probabilistically-- Discrete variables
Introduction:
    Statistical inference rests upon probability. Because we can very rarely say 
    anything meaningful with absolute certainty from data, we use probabilistic 
    language to make quantitative statements about data. In this chapter, you will 
    learn how to think probabilistically about discrete quantities, those that can 
    only take certain values, like integers. It is an important first step in building 
    the probabilistic language necessary to think statistically.
"""

# Import packages
#import pandas as pd                  #For loading tabular data
import numpy as np                   #For making operations in lists
import matplotlib.pyplot as plt      #For creating charts
import seaborn as sns                #For visualizing data
#import scipy.stats as stats          #For accesign to a vary of statistics functiosn
#import statsmodels as sm             #For stimations in differents statistical models
#import scykit-learn                  #For performing machine learning  
#import tabula                        #For extracting tables from pdf
#import nltk                          #For working with text data
#import math                          #For accesing to a complex math operations
#import random                        #For generating random numbers
#import calendar                      #For accesing to a vary of calendar operations

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching

#from bokeh.io import curdoc, output_file, show                                      #For interacting visualizations
#from bokeh.plotting import figure, ColumnDataSource                                 #For interacting visualizations
#from bokeh.layouts import row, widgetbox, column, gridplot                          #For interacting visualizations
#from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper        #For interacting visualizations
#from bokeh.models import Slider, Select, Button, CheckboxGroup, RadioGroup, Toggle  #For interacting visualizations
#from bokeh.models.widgets import Tabs, Panel                                        #For interacting visualizations
#from bokeh.palettes import Spectral6                                                #For interacting visualizations

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters() #Require to explicitly register matplotlib converters.

#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data) # Number of data points: n
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n # y-data for the ECDF: y
    
    return x, y


print("****************************************************")
tema = '5. Generating random numbers using the np.random module'; print("** %s\n" % tema)

np.random.seed(42) # Seed the random number generator
random_numbers = np.empty(100000) # Initialize random numbers: random_numbers
for i in range(100000): # Generate random numbers by looping over range(100000)
    random_numbers[i] = np.random.random()

sns.set() # Set default Seaborn style

np.random.seed(42) # Seed the random number generator
easy_random_numbers = np.random.random(size=100000)

plt.subplot(2,1,1)
_ = plt.hist(random_numbers) # Plot a histogram
_ = plt.xlabel('Random number between 0 and 1')
_ = plt.ylabel('Number of times')
_ = plt.title('Get a number between 0 and 1')

plt.subplot(2,1,2)
_ = plt.hist(easy_random_numbers, color='green') # Plot a histogram
_ = plt.xlabel('Random number between 0 and 1')
_ = plt.ylabel('Number of times')
_ = plt.title('Get a number between 0 and 1 (Easiest way)')

# Show the plot
_ = plt.suptitle(tema)
plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.7)
plt.show() # Show the plot
plt.style.use('default')


print("****************************************************")
tema = '6. The np.random module and Bernoulli trials'; print("** %s\n" % tema)

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    #n_success = 0 # Initialize number of successes: n_success
    #
    #for i in range(n): # Perform trials
    #    random_number = np.random.random() # Choose random number between zero and one: random_number
    #    if random_number<p: # If less than p, it's a success so add one to n_success
    #        n_success += 1
    #return n_success
    #Easiest way:
    #return np.sum(np.random.random(size=n)<p)
    return sum(np.random.random(size=n)<p)

print("****************************************************")
tema = '7. How many defaults might we expect? (1,000 of times)'; print("** %s\n" % tema)

plt.figure()
sns.set() # Set default Seaborn style

np.random.seed(42) # Seed random number generator
size=1000 # Initialize the number of defaults: n_defaults
n_defaults=np.empty(size)

for i in range(size): # Compute the number of defaults
    n_defaults[i] = perform_bernoulli_trials(n=100, p=0.05)

# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, density=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')
_ = plt.title('Probability of get a default in the loans')
_ = plt.suptitle(tema)

# Show the plot
#plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '8. Will the bank fail? (1,000 times)'; print("** %s\n" % tema)

plt.figure()
sns.set() # Set default Seaborn style

# Compute ECDF: x, y
x, y = ecdf(n_defaults)

# Plot the ECDF with labeled axes
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('number of defaulted per 100 loans')
_ = plt.ylabel('probability')
_ = plt.title('Probability of default out of 100 loans')
_ = plt.suptitle(tema)

# Show the plot
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
n_lose_money = np.sum(n_defaults>=10)

# Compute and print probability of losing money
print('Probability of losing money = {}\n'.format(n_lose_money / len(n_defaults)))

# Show the plot
#plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '10. Sampling out of the Binomial distribution (10,000 times)'; print("** %s\n" % tema)

plt.figure()
sns.set() # Set default Seaborn style
np.random.seed(42) # Seed random number generator
size=10000 # Initialize the number of defaults: n_defaults

n_defaults = np.random.binomial(100, 0.05, size) # Take 10,000 samples out of the binomial distribution: n_defaults
x, y = ecdf(n_defaults) # Compute CDF: x, y

# Plot the CDF with axis labels
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('CDF')
_ = plt.title('Probability of default out of 100 loans')
_ = plt.suptitle(tema)

# Show the plot
#plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '11. Plotting the Binomial PMF (10,000 times)'; print("** %s\n" % tema)

plt.figure()
sns.set() # Set default Seaborn style
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5 # Compute bin edges: bins

_ = plt.hist(n_defaults, bins=bins, density=True) # Generate histogram
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')
_ = plt.title('Probability of default out of 100 loans')
_ = plt.suptitle(tema)

# Show the plot
plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')

#Graphing a PMF correctly
plt.figure()
sns.set() # Set default Seaborn style
y = np.bincount(n_defaults)/len(n_defaults) # Get the frequency
x = np.nonzero(y)[0] 

_ = plt.plot(x, y, 'ro', ms=8, mec='r')
_ = plt.vlines(x, 0, y, colors='r', linestyles='-', lw=2)
_ = plt.title('Custom made discrete distribution(PMF)')
_ = plt.ylabel('Probability')
_ = plt.suptitle(tema)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '13. Relationship between Binomial and Poisson distributions'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

samples_poisson = np.random.poisson(10, size=10000) # Draw 10,000 samples out of Poisson distribution: samples_poisson
print('Poisson: ', np.mean(samples_poisson),np.std(samples_poisson)) # Print the mean and standard deviation

# Specify values of n and p to consider for Binomial: n, p
n = [ 20, 100, 1000, 10000, 100000]
p = [0.5, 0.1, 0.01, 0.001, 0.0001]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(5):
    samples_binomial = np.random.binomial(n[i], p[i], size=10000)

    # Print results
    print('(n={}, p={}), Binom: {} {}'.format(n[i], p[i], 
                                              np.mean(samples_binomial),
                                              np.std(samples_binomial)))
    

print("\n****************************************************")
tema = '15. Was 2015 anomalous?'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator
n_nohitters = np.random.poisson(251/115,10000) # Draw 10,000 samples out of Poisson distribution: n_nohitters
n_large = np.sum(n_nohitters>=7) # Compute number of samples that are seven or greater: n_large
p_large = n_large/len(n_nohitters) # Compute probability of getting seven or more: p_large

print('Probability of seven or more no-hitters: {}\n'.format(p_large)) # Print the result


print("****************************************************")
print("** END                                            **")
print("****************************************************")