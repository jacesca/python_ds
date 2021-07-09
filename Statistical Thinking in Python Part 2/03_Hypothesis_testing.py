# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:42:11 2019

@author: jacqueline.cortez

Capítulo 3. Introduction to hypothesis testing
Introduction:
    You now know how to define and estimate parameters given a model. 
    But the question remains: how reasonable is it to observe your data 
    if a model is true? This question is addressed by hypothesis tests. 
    They are the icing on the inference cake. After completing this chapter, 
    you will be able to carefully construct and test hypotheses using 
    hacker statistics.
"""

# Import packages
import pandas as pd                   #For loading tabular data
import numpy as np                    #For making operations in lists
#import matplotlib as mpl              #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt       #For creating charts
import seaborn as sns                 #For visualizing data
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

#Setting the numpy options
#np.set_printoptions(precision=3) #precision set the precision of the output:
#np.set_printoptions(suppress=True) #suppress suppresses the use of scientific notation for small numbers

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    corr_mat = np.corrcoef(x, y) # Compute correlation matrix: corr_mat
    return corr_mat[0,1]         # Return entry [0,1]


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data) # Number of data points: n
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n # y-data for the ECDF: y
    return x, y


def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    t1 = np.random.exponential(tau1, size=size) # Draw samples out of first exponential distribution: t1
    t2 = np.random.exponential(tau2, size=size) # Draw samples out of second exponential distribution: t2
    return t1 + t2


def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""
    return func(np.random.choice(data, size=len(data)))


def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    bs_replicates = np.empty(size) # Initialize array of replicates: bs_replicates
    for i in range(size): # Generate replicates
        bs_replicates[i] = bootstrap_replicate_1d(data, func)
    return bs_replicates


def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    inds = np.arange(len(x)) # Set up array of indices to sample from: inds
    bs_slope_reps = np.empty(size) # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_intercept_reps = np.empty(size)
    for i in range(size): # Generate replicates
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
    return bs_slope_reps, bs_intercept_reps


def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    data = np.concatenate((data1, data2)) # Concatenate the data sets: data
    permuted_data = np.random.permutation(data) # Permute the concatenated array: permuted_data
    perm_sample_1 = permuted_data[:len(data1)] # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_2 = permuted_data[len(data1):]
    return perm_sample_1, perm_sample_2


def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    perm_replicates = np.empty(size) # Initialize array of replicates: perm_replicates
    for i in range(size):
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2) # Generate permutation sample
        perm_replicates[i] = func(perm_sample_1, perm_sample_2) # Compute the test statistic
    return perm_replicates


def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    diff = data_1.mean()-data_2.mean() # The difference of means of data_1, data_2: diff
    return diff



file = "sheffield_weather_station.data"
sheffield_df = pd.read_fwf(file, header = None, skiprows=9, #skipfooter=1,
                           names = ['year','month','tmax (degC)','tmin (degC)','af (days)', 'rain (mm)', 'sun (hr)'],
                          )
rain_june = sheffield_df[sheffield_df.month==6]['rain (mm)'].values
rain_november = sheffield_df[sheffield_df.month==11]['rain (mm)'].values


file = "frog_tongue.csv"
frog_df = pd.read_csv(file, comment='#')
frog_df.drop(frog_df[frog_df.ID.isin(['I','III'])].index, axis='index', inplace=True)
frog_df["ID"] = frog_df.ID.replace(["II", "IV"],["A", "B"])
frog_df['impact force (N)'] = frog_df['impact force (mN)']/1000
force_a = frog_df[frog_df.ID=='A']['impact force (N)'].values
force_b = frog_df[frog_df.ID=='B']['impact force (N)'].values



print("****************************************************")
tema = '2. Generating a permutation sample'; print("** %s\n" % tema)
tema = '3. Visualizing permutation sampling'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator
sns.set() # Set default Seaborn style

for i in range(50):
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november) # Generate permutation samples
    x_1, y_1 = ecdf(perm_sample_1) # Compute ECDFs
    x_2, y_2 = ecdf(perm_sample_2)
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red', alpha=0.02) # Plot ECDFs of permutation sample
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
_ = plt.title("Permuting the June and November data of Sheffield station")
_ = plt.suptitle(tema)

plt.show()
plt.style.use('default')


print("****************************************************")
tema = '8. Look before you leap: EDA before hypothesis testing'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()

_ = sns.swarmplot('ID', 'impact force (N)', data=frog_df) # Make bee swarm plot
_ = plt.xlabel('frog')# Label axes
_ = plt.ylabel('impact force (N)')
_ = plt.title("Swarmplot from frog data")
_ = plt.suptitle(tema)

# Show the plot
plt.show()


plt.figure()

_ = sns.boxplot('ID', 'impact force (N)', data=frog_df) # Make bee swarm plot
_ = plt.xlabel('frog')# Label axes
_ = plt.ylabel('impact force (N)')
_ = plt.title("Boxplot from frog data")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '9. Permutation test on frog data'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

empirical_diff_means = diff_of_means(force_a, force_b) # Compute difference of mean impact force from experiment: empirical_diff_means
perm_replicates = draw_perm_reps(force_a, force_b, diff_of_means, size=10000) # Draw 10,000 permutation replicates: perm_replicates
p = np.sum(perm_replicates >= empirical_diff_means)*1.0 / len(perm_replicates) # Compute p-value: p
print('p-value =', p) # Print the result
print("A mean = ", force_a.mean())
print("B mean = ", force_b.mean())
"""
The p-value tells you that there is about a 0.6\% chance that you would get the difference 
of means observed in the experiment if frogs were exactly the same. A p-value below 0.01 
is typically said to be 'statistically significant', but: warning! warning! warning! You 
have computed a p-value; it is a number. I encourage you not to distill it to a yes-or-no 
phrase. p = 0.006 and p = 0.000000006 are both said to be 'statistically significant', but 
they are definitely not the same!")
"""

print("****************************************************")
tema = '11. A one-sample bootstrap hypothesis test'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

translated_force_b = force_b - force_b.mean() + 0.55 # Make an array of translated impact forces: translated_force_b
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000) # Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
p = np.sum(bs_replicates <= np.mean(force_b)) / len(bs_replicates) # Compute fraction of replicates that are less than the observed Frog B force: p
print('p = ', p) # Print the p-value
"""
Great work! The low p-value suggests that the null hypothesis that Frog B and Frog C 
have the same mean impact force is false.
"""


print("****************************************************")
tema = '12. A two-sample bootstrap hypothesis test for difference of means'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

forces_concat = np.concatenate((force_a, force_b))
empirical_diff_means = force_a.mean() - force_b.mean()
mean_force = np.mean(forces_concat) # Compute mean of all forces: mean_force
# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force
# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, 10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, 10000)
# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b
# Compute and print p-value: p
p = sum(bs_replicates>=empirical_diff_means) / len(bs_replicates)
print('p-value =', p)


print("****************************************************")
print("** END                                            **")
print("****************************************************")