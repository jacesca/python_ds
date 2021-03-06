# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:42:11 2019

@author: jacqueline.cortez

Capítulo 5. Putting it all together: a case study
Introduction:
    Every year for the past 40-plus years, Peter and Rosemary Grant have gone to the 
    Galápagos island of Daphne Major and collected data on Darwin's finches. Using your 
    skills in statistical inference, you will spend this chapter with their data, and 
    witness first hand, through data, evolution in action. It's an exhilarating way to 
    end the course!
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


def frac_yea(data_1, data_2):
    """Compute fraction of Democrat yea votes."""
    frac = sum(data_1) / len(data_1)
    return frac


def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for a single statistic."""
    inds = np.arange(len(x)) # Set up array of indices to sample from: inds
    bs_replicates = np.empty(size) # Initialize replicates: bs_replicates
    for i in range(size): # Generate replicates
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)
    return bs_replicates


def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples.
       We have two sets of data parent and y, np.cov(parent, offspring 
       returns a 2D array where entries [0,1] and [1,0] are the 
       covariances. 
       Entry [0,0] is the variance of the data in parent.
       Entry [1,1] is the variance of the data in offspring. 
    """
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0,1] / covariance_matrix[0,0]




file = "05_finch_beaks_1975.csv" 
finch_beaks_1975 = pd.read_csv(file, skiprows=1, names=['band', 'species', 'blength', 'bdepth'])
finch_beaks_1975['year'] = 1975 
bd_1975 = finch_beaks_1975[finch_beaks_1975.species=='scandens'].bdepth.values
bl_1975 = finch_beaks_1975[finch_beaks_1975.species=='scandens'].blength.values

file = "05_finch_beaks_2012.csv" 
finch_beaks_2012 = pd.read_csv(file, skiprows=1, names=['band', 'species', 'blength', 'bdepth'])
finch_beaks_2012['year'] = 2012 
bd_2012 = finch_beaks_2012[finch_beaks_2012.species=='scandens'].bdepth.values
bl_2012 = finch_beaks_2012[finch_beaks_2012.species=='scandens'].blength.values
scandens_bdepth = finch_beaks_1975[finch_beaks_1975.species=='scandens'][['year','bdepth']].append(finch_beaks_2012[finch_beaks_2012.species=='scandens'][['year','bdepth']])


file = "05_scandens_beak_depth_heredity.csv" 
scandens_heredity = pd.read_csv(file)
bd_parent_scandens = scandens_heredity.mid_parent.values
bd_offspring_scandens = scandens_heredity.mid_offspring.values


file = "05_fortis_beak_depth_heredity.csv" 
fortis_heredity = pd.read_csv(file)
fortis_heredity['mid_parent'] = (fortis_heredity['Male BD']+fortis_heredity['Female BD'])/2
bd_parent_fortis = fortis_heredity.mid_parent.values
bd_offspring_fortis = fortis_heredity['Mid-offspr'].values



print("****************************************************")
tema = '2. EDA of beak depths of Darwin\'s finches'; print("** %s\n" % tema)

#Swarmplot
sns.set() # Set default Seaborn style
_ = sns.swarmplot('year','bdepth', data=scandens_bdepth) # Create bee swarm plot
_ = plt.xlabel('year') # Label the axes
_ = plt.ylabel('beak depth (mm)')
_ = plt.title("How the beak depth of the finch species \nGeospiza scandens has changed over time")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot
plt.style.use('default')


#Boxplot
sns.set() # Set default Seaborn style
plt.figure()
_ = sns.boxplot('year','bdepth', data=scandens_bdepth) # Create bee swarm plot
_ = plt.xlabel('year') # Label the axes
_ = plt.ylabel('beak depth (mm)')
_ = plt.title("How the beak depth of the finch species \nGeospiza scandens has changed over time")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot
plt.style.use('default')


print("****************************************************")
tema = '3. ECDFs of beak depths'; print("** %s\n" % tema)

# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

sns.set() # Set default Seaborn style
plt.figure()
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')
_ = plt.margins(0.02)
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')
_ = plt.title("How the beak depth of the finch species \nGeospiza scandens has changed over time")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot
plt.style.use('default')



print("****************************************************")
tema = '4. Parameter estimates of beak depths'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator
num_exp = 10000 # Initialize permutation replicates: perm_replicates

# Compute the difference of the sample means: mean_diff
mean_diff = bd_2012.mean() - bd_1975.mean()

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, num_exp)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, num_exp)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

mu = bs_diff_replicates.mean()
sigma = bs_diff_replicates.std()
ci = np.percentile(bs_diff_replicates,[2.5, 97.5])
# Compute and print p-value: p
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print the results
print('Difference of means =', mean_diff, 'mm')
print('95% confidence interval =', ci, 'mm')
print('Boostrap samples mean = ',mu, 'mm')


#Swarmplot
sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(bs_diff_replicates , bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(mu, color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=2)
_ = plt.xlabel('differences observed in Mean (2012 - 1975)')
_ = plt.ylabel('PDF')
_ = plt.text(-0.1, 4,"Boostrap replicate \n>= {0:,.2f}) = {1:,.6f}".format(mean_diff,p), color='red', fontsize=9)
_ = plt.text(-0.1, 3,"Boostrap samples \nmean diff= {0:,.2f}\nCI = [{1:,.2f}, {2:,.2f}]".format(mu,ci[0],ci[1]), color='black', fontsize=9)
_ = plt.text(-0.1, 2,"Ho: There's no diff. \nbetween both years.", color='black', fontsize=9)
_ = plt.title("How the beak depth of the finch species \nGeospiza scandens has changed over time")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '5. Hypothesis test: Are beaks deeper in 2012?'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator


num_exp = 10000
mean_diff = bd_2012.mean() - bd_1975.mean()
# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))
# Shift the samples
bd_1975_shifted = bd_1975 - bd_1975.mean() + combined_mean
bd_2012_shifted = bd_2012 - bd_2012.mean() + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, num_exp)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, num_exp)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

mu = bs_diff_replicates.mean()
sigma = bs_diff_replicates.std()
ci = np.percentile(bs_diff_replicates,[2.5, 97.5])
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates) # Compute the p-value


sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(bs_diff_replicates , bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(bs_diff_replicates.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=2)
_ = plt.axvline(ci[0], color='gray', linestyle='solid', linewidth=1)
_ = plt.axvline(ci[1], color='gray', linestyle='solid', linewidth=1)
_ = plt.xlabel('Differences observed in Mean (2012 - 1975)')
_ = plt.ylabel('PDF')
_ = plt.text( 0.05, 4,"p-value = \n(Boostrap replicate \n>= {0:,.6f}) \n{1:,.6f}".format(mean_diff,p), color='red', fontsize=9)
_ = plt.text(-0.35, 4,"Boostrap samples \nmean = {0:,.6f}\nCI = [{1:,.6f}, \n{2:,.6f}]".format(mu,ci[0],ci[1]), color='black', fontsize=9)
_ = plt.text(-0.35, 3,"Ho: The means \nobserved in \nboth years is really \nthe same.", color='black', fontsize=9)
_ = plt.text(-0.35, 2,"A p-value less than \n0.01 means \nstatistically \nsignificance.", color='red', fontsize=9)
_ = plt.text(-0.35, 1,"We conclude that \nthe beak depth \nof birds changed \nover the time.", color='blue', fontsize=9)
_ = plt.title("What is the probability that we would get the observed difference \nin mean beak depth if the means were the same?")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '7. EDA of beak length and depth'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
# Make scatter plot of 1975 data
_ = plt.scatter(bl_1975, bd_1975, marker='.', linestyle='None', color='blue', alpha=0.5)
# Make scatter plot of 2012 data
_ = plt.scatter(bl_2012, bd_2012, marker='.', linestyle='None', color='red', alpha=0.5)
# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')
_ = plt.title("Differences observed over the time")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '8. Linear regressions'; print("** %s" % tema)
tema = '9. Displaying the linear regression results'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

num_exp = 1000

# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, 1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, 1)

# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = draw_bs_pairs_linreg(bl_1975, bd_1975, size=num_exp)
bs_slope_reps_2012, bs_intercept_reps_2012 = draw_bs_pairs_linreg(bl_2012, bd_2012, size=num_exp)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])

slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])


# Generate array of x-values for bootstrap lines: x
x = np.array([np.min((bl_1975.min(),bl_2012.min())), np.max((bl_1975.max(),bl_2012.max()))])
x = np.array([11.0, 16.0])
y_1975 = slope_1975*x + intercept_1975
y_2012 = slope_2012*x + intercept_2012

sns.set() # Set default Seaborn style
plt.figure()
_ = plt.scatter(bl_1975, bd_1975, marker='.', linestyle='None', color='blue', alpha=0.5) # Make scatter plot of 1975 data
_ = plt.scatter(bl_2012, bd_2012, marker='.', linestyle='None', color='red', alpha=0.5) # Make scatter plot of 2012 data
_ = plt.legend(('1975', '2012'), loc='upper left')
for i in range(100): # Plot the bootstrap lines
    _ = plt.plot(x, bs_slope_reps_1975[i]*x + bs_intercept_reps_1975[i], linewidth=0.5, alpha=0.2, color='blue')
    _ = plt.plot(x, bs_slope_reps_2012[i]*x + bs_intercept_reps_2012[i], linewidth=0.5, alpha=0.2, color='red')
# Plot the data
_ = plt.plot(x, y_1975, linewidth=1, color='blue')
_ = plt.plot(x, y_2012, linewidth=1, color='red')
_ = plt.xlabel('beak length (mm)') # Label axes and make legend
_ = plt.ylabel('beak depth (mm)')
_ = plt.title("Differences observed over the time")
_ = plt.suptitle(tema)
_ = plt.text(11, 10.2,"1975: beaks depth = {0:,.2f}, beaks length = {1:,.2f}".format(bd_1975.mean(), bl_1975.mean()), color='black', fontsize=9)
_ = plt.text(11, 10.0,"2012: beaks depth = {0:,.2f}, beaks length = {1:,.2f}".format(bd_2012.mean(), bl_2012.mean()), color='black', fontsize=9)
_ = plt.text(12, 7.7,"1975: slope = {0:,.6f}, conf int = [{1:,.6f}, {2:,.6f}]".format(slope_1975, slope_conf_int_1975[0], slope_conf_int_1975[1]), color='black', fontsize=9)
_ = plt.text(12, 7.5,"1975: intercept = {0:,.6f}, conf int = [{1:,.6f}, {2:,.6f}]".format(intercept_1975, intercept_conf_int_1975[0], intercept_conf_int_1975[1]), color='black', fontsize=9)
_ = plt.text(12, 7.3,"2012: slope = {0:,.6f}, conf int = [{1:,.6f}, {2:,.6f}]".format(slope_2012, slope_conf_int_2012[0], slope_conf_int_2012[1]), color='black', fontsize=9)
_ = plt.text(12, 7.1,"2012: intercept = {0:,.6f}, conf int = [{1:,.6f}, {2:,.6f}]".format(intercept_2012, intercept_conf_int_2012[0], intercept_conf_int_2012[1]), color='black', fontsize=9)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.margins(0.02)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '10. Beak length to depth ratio'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

num_exp = 10000

# Compute length-to-depth ratios
ratio_1975 = bl_1975/bd_1975
ratio_2012 = bl_2012/bd_2012
# Compute means
mean_ratio_1975 = ratio_1975.mean()
mean_ratio_2012 = ratio_2012.mean()
# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, num_exp)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, num_exp)
# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975, 'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012, 'conf int =', conf_int_2012)


mu_1975    = bs_replicates_1975.mean()
sigma_1975 = bs_replicates_1975.std()
ci_1975    = np.percentile(bs_replicates_1975,[0.5, 99.5])
p_1975     = np.sum(bs_replicates_1975 <= mean_ratio_1975) / len(bs_replicates_1975) # Compute the p-value

mu_2012    = bs_replicates_2012.mean()
sigma_2012 = bs_replicates_2012.std()
ci_2012    = np.percentile(bs_replicates_2012,[0.5, 99.5])
p_2012     = np.sum(bs_replicates_2012 <= mean_ratio_2012) / len(bs_replicates_2012) # Compute the p-value


sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(bs_replicates_1975, bins=50, density=True, color='blue', histtype='step', label='1975') # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma_1975)) * np.exp(-0.5 * (1 / sigma_1975 * (bins - mu_1975))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='blue', linewidth=2) # add a 'best fit' line
_ = plt.axvline(bs_replicates_1975.mean(), color='black', linestyle='dashed', linewidth=2)
_ = plt.axvline(mean_ratio_1975, color='blue', linestyle='dashed', linewidth=1)
_ = plt.axvline(ci_1975[0], color='gray', linestyle='solid', linewidth=1)
_ = plt.axvline(ci_1975[1], color='gray', linestyle='solid', linewidth=1)

n, bins, patches = plt.hist(bs_replicates_2012, bins=50, density=True, color='red', histtype='step', label='2012') # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma_2012)) * np.exp(-0.5 * (1 / sigma_2012 * (bins - mu_2012))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='red', linewidth=2) # add a 'best fit' line
_ = plt.axvline(bs_replicates_2012.mean(), color='black', linestyle='dashed', linewidth=2)
_ = plt.axvline(mean_ratio_2012, color='red', linestyle='dashed', linewidth=1)
_ = plt.axvline(ci_2012[0], color='gray', linestyle='solid', linewidth=1)
_ = plt.axvline(ci_2012[1], color='gray', linestyle='solid', linewidth=1)

_ = plt.text(1.5, 40,"1975: mean ratio \n= {0:,.6f}, \nconf int = [{1:,.6f}, \n{2:,.6f}]".format(mean_ratio_1975, ci_1975[0], ci_1975[1]), color='blue', fontsize=9)
_ = plt.text(1.5, 30,"2012: mean ratio \n= {0:,.6f}, \nconf int = [{1:,.6f}, \n{2:,.6f}]".format(mean_ratio_2012, ci_2012[0], ci_2012[1]), color='red', fontsize=9)

_ = plt.legend()
_ = plt.xlabel('Beak length / beak depth')
_ = plt.ylabel('PDF')
_ = plt.title("Ratio between beak length and depth over time")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '13. EDA of heritability'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
# Make scatter plots
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis, marker='.', linestyle='none', color='blue', alpha=0.5)
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens, marker='.', linestyle='none', color='red', alpha=0.5)
# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')
# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')
_ = plt.title("Average offspring beak depth versus average \nparental beak depth for both species. ")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '14. Correlation of offspring and parental data'; print("** %s" % tema)
tema = '15. Pearson correlation of offspring and parental data'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

num_exp = 1000

# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, num_exp)
bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, num_exp)

# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', r_scandens, conf_int_scandens)
print('G. fortis:', r_fortis, conf_int_fortis)

#Scanders case
mu = bs_replicates_scandens.mean()
sigma = bs_replicates_scandens.std()
ci = np.percentile(bs_replicates_scandens,[2.5, 97.5])
p = np.sum(bs_replicates_scandens >= r_scandens) / len(bs_replicates_scandens) # Compute the p-value

sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(bs_replicates_scandens, bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(bs_replicates_scandens.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(r_scandens, color='red', linestyle='dashed', linewidth=2)
_ = plt.axvline(ci[0], color='gray', linestyle='solid', linewidth=1)
_ = plt.axvline(ci[1], color='gray', linestyle='solid', linewidth=1)
_ = plt.xlabel('Pearson correl observed in Scander species')
_ = plt.ylabel('PDF')
_ = plt.text(0.27, 5.5,"p-value = \n(Boostrap replicate) \n>= {0:,.6f}) \n{1:,.6f}".format(r_scandens,p), color='red', fontsize=9)
_ = plt.text(0.11, 3,"Pearson correl = \n{0:,.6f}.\n\nBoostrap samples \npearson correl = \n{1:,.6f}.\n\n95% CI = [{2:,.6f}, \n{3:,.6f}].".format(r_scandens,mu,ci[0],ci[1]), color='black', fontsize=9)
_ = plt.title("Relation observed between generetaion from Scander Species")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



#Fortis case
mu = bs_replicates_fortis.mean()
sigma = bs_replicates_fortis.std()
ci = np.percentile(bs_replicates_fortis,[2.5, 97.5])
p = np.sum(bs_replicates_fortis >= r_fortis) / len(bs_replicates_fortis) # Compute the p-value

sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(bs_replicates_fortis, bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(bs_replicates_fortis.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(r_fortis, color='red', linestyle='dashed', linewidth=2)
_ = plt.axvline(ci[0], color='gray', linestyle='solid', linewidth=1)
_ = plt.axvline(ci[1], color='gray', linestyle='solid', linewidth=1)
_ = plt.xlabel('Pearson correl observed in Fortis species')
_ = plt.ylabel('PDF')
_ = plt.text(0.674, 15,"p-value = \n(Boostrap replicate) \n>= {0:,.6f}) \n{1:,.6f}".format(r_fortis,p), color='red', fontsize=9)
_ = plt.text(0.613, 7.5,"Pearson correl = \n{0:,.6f}.\n\nBoostrap samples \npearson correl = \n{1:,.6f}.\n\n95% CI = [{2:,.6f}, \n{3:,.6f}].".format(r_fortis,mu,ci[0],ci[1]), color='black', fontsize=9)
_ = plt.title("Relation observed between generetaion from Fortis Species")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '16. Measuring heritability'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

num_exp = 1000

# Compute the heritability
heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)
heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of heritability
replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, heritability, size=num_exp)
replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, heritability, size=num_exp)

# Compute 95% confidence intervals
conf_int_scandens = np.percentile(replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', heritability_scandens, conf_int_scandens)
print('G. fortis:', heritability_fortis, conf_int_fortis)


#Scanders case
mu = replicates_scandens.mean()
sigma = replicates_scandens.std()
ci = np.percentile(replicates_scandens,[2.5, 97.5])
p = np.sum(replicates_scandens >= heritability_scandens) / len(replicates_scandens) # Compute the p-value

sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(replicates_scandens, bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(replicates_scandens.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(heritability_scandens, color='red', linestyle='dashed', linewidth=2)
_ = plt.axvline(ci[0], color='gray', linestyle='solid', linewidth=1)
_ = plt.axvline(ci[1], color='gray', linestyle='solid', linewidth=1)
_ = plt.xlabel('Heriatibility observed in Scander species')
_ = plt.ylabel('PDF')
_ = plt.text(0.369, 4,"p-value = \n(Boostrap replicate) \n>= {0:,.6f}) \n{1:,.6f}".format(heritability_scandens,p), color='red', fontsize=9)
_ = plt.text(0.175, 1,"Heriatibility = \n{0:,.6f}.\n\nBoostrap samples \nheriatibility = \n{1:,.6f}.\n\n95% CI = \n[{2:,.6f}, \n{3:,.6f}].".format(heritability_scandens,mu,ci[0],ci[1]), color='black', fontsize=9)
_ = plt.title("Heriatibility observed between generetaion from Scander Species")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



#Fortis case
mu = replicates_fortis.mean()
sigma = replicates_fortis.std()
ci = np.percentile(replicates_fortis,[2.5, 97.5])
p = np.sum(replicates_fortis >= heritability_fortis) / len(replicates_fortis) # Compute the p-value

sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(replicates_fortis, bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(replicates_fortis.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(heritability_fortis, color='red', linestyle='dashed', linewidth=2)
_ = plt.axvline(ci[0], color='gray', linestyle='solid', linewidth=1)
_ = plt.axvline(ci[1], color='gray', linestyle='solid', linewidth=1)
_ = plt.xlabel('Pearson correl observed in Fortis species')
_ = plt.ylabel('PDF')
_ = plt.text(0.662, 10,"p-value = \n(Boostrap replicate) \n>= {0:,.6f}) \n{1:,.6f}".format(heritability_fortis,p), color='red', fontsize=9)
_ = plt.text(0.605, 4,"Heriatibility \n= {0:,.6f}.\n\nBoostrap \nsamples \nheriatibility \n= {1:,.6f}.\n\n95% CI = \n[{2:,.6f}, \n{3:,.6f}].".format(heritability_fortis,mu,ci[0],ci[1]), color='black', fontsize=9)
_ = plt.title("Heriatibility observed between generetaion from Fortis Species")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '17. Is beak depth heritable at all in G. scandens?'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

num_exp = 10000
heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)

# Initialize array of replicates: perm_replicates
perm_replicates = np.empty(num_exp)
# Draw replicates
for i in range(num_exp):
    # Permute parent beak depths
    bd_parent_permuted = np.random.permutation(bd_parent_scandens)
    perm_replicates[i] = heritability(bd_parent_permuted, bd_offspring_scandens)


#Scanders case
mu = perm_replicates.mean()
sigma = perm_replicates.std()
ci = np.percentile(perm_replicates,[2.5, 97.5])
p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates) # Compute the p-value

sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(perm_replicates, bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(perm_replicates.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(heritability_scandens, color='red', linestyle='dashed', linewidth=2)
_ = plt.axvline(ci[0], color='gray', linestyle='solid', linewidth=1)
_ = plt.axvline(ci[1], color='gray', linestyle='solid', linewidth=1)
_ = plt.xlabel('Heriatibility observed in permuted replicates')
_ = plt.ylabel('PDF')
_ = plt.text( 0.27, 3,"p-value = \n(Boostrap replicate) \n>= {0:,.6f}) \n{1:,.6f}".format(heritability_scandens,p), color='red', fontsize=9)
_ = plt.text( 0.24, 1,"Heriatibility = \n{0:,.6f}.\n\nBoostrap \npermuted samples \nheriatibility = \n{1:,.6f}.\n\n95% CI = \n[{2:,.6f}, \n{3:,.6f}].".format(heritability_scandens,mu,ci[0],ci[1]), color='black', fontsize=9)
_ = plt.text(-0.45, 3,"Ho: Variability \nwas made by \nchance.", color='blue', fontsize=9)
_ = plt.text(-0.45, 2,"A p-value less \nthan 0.01 means \nstatistically \nsignificance.", color='red', fontsize=9)
_ = plt.text(-0.45, 1,"We conclude \nthat the \nvariability \nis not made \nby chance.", color='blue', fontsize=9)
_ = plt.title("It could be that this observed heritability was just achieved by chance and \nbeak depth is actually not really heritable in the species.")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
print("** END                                            **")
print("****************************************************")