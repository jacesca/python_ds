# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:42:11 2019

@author: jacqueline.cortez

Capítulo 2. Bootstrap confidence intervals
Introduction:
    To "pull yourself up by your bootstraps" is a classic idiom meaning that you achieve 
    a difficult task by yourself with no help at all. In statistical inference, you want 
    to know what would happen if you could repeat your data acquisition an infinite number 
    of times. This task is impossible, but can we use only the data we actually have to 
    get close to the same result as an infinitude of experiments? The answer is yes! The 
    technique to do it is aptly called bootstrapping. This chapter will introduce you to 
    this extraordinarily powerful tool.
"""

# Import packages
import pandas as pd                   #For loading tabular data
import numpy as np                    #For making operations in lists
import matplotlib as mpl              #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
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


file = "michelson_speed_of_light.csv"
speed_of_light = pd.read_csv(file)
speed_of_light = speed_of_light["velocity of light in air (km/s)"].values

file = "sheffield_weather_station.data"
sheffield_df = pd.read_fwf(file, header = None, skiprows=9, #skipfooter=1,
                           names = ['year','month','tmax (degC)','tmin (degC)','af (days)', 'rain (mm)', 'sun (hr)'],
                          )
rainfall = sheffield_df.groupby('year')['rain (mm)'].sum().values

file='no-hitters.csv'
nohitters_df = pd.read_csv(file, sep=';', parse_dates=['Date'])
nohitters_df.sort_values('Date', inplace=True)
nohitters_df['Days Since Previous'] = (nohitters_df['Date'] -
                                       nohitters_df['Date'].shift(1)).astype('timedelta64[D]')
nohitters_df.reset_index(inplace=True, drop=True)
nohitters_df.drop(labels=0,axis=0,inplace=True)
nohitter_times = nohitters_df['Days Since Previous'].values

file = "Female_Educatiov_vs_Fertility.csv"
fertility_df = pd.read_csv(file, skiprows=1, names=['country','continent','fem_literacity','fertility','population'])
fertility = fertility_df.fertility.values
female_literacy = fertility_df.fem_literacity.values
female_iliteracy = 100-fertility_df.fem_literacity.values

file = '2008_swing_states.csv'
swing_states_df = pd.read_csv(file)
total_votes = swing_states_df.total_votes.values
dem_share = swing_states_df.dem_share.values


print("****************************************************")
tema = '1. Generating bootstrap replicates'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

bs = 100
bs_mean = np.arange(bs)
for i in range(bs):
    bs_sample = np.random.choice(speed_of_light, size=100)
    bs_mean[i] = np.mean(bs_sample)

#Calculating mean and std from boostrap samples
mean = np.mean(bs_mean)
std = np.std(bs_mean)

#Get a new theorical normal sample from the computed mean and std
samples = np.random.normal(mean, std, size=10000)

#Getting the ecdf
x, y = ecdf(bs_mean)
x_theor, y_theor = ecdf(samples)

sns.set() # Set default Seaborn style
fig, ax = plt.subplots(1, 1)
_ = ax.plot(x_theor, y_theor)
_ = ax.plot(x, y, marker='.', linestyle='none')
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
#ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
_ = plt.xlabel('Speed Mean of light in air (km/s)')
_ = plt.ylabel('CDF')
_ = plt.title("Resampling the data of Albert A. Michelson'data")
_ = plt.suptitle(tema)

# Show the plot
#plt.ylim(-0.01, 0.42)
#plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '4. Visualizing bootstrap samples'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator
sns.set() # Set default Seaborn style

plt.figure()

size = len(rainfall)
for i in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=size)

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')
_ = plt.title("Boostrap samples of rainfall (mm)")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '7. Bootstrap replicates of the mean and the SEM'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.mean, 10000)

sem = np.std(rainfall) / np.sqrt(len(rainfall)) # Compute and print SEM
bs_std = np.std(bs_replicates) # Compute and print standard deviation of bootstrap replicates

sns.set() # Set default Seaborn style
plt.figure()

# add a 'best fit' line
mu = bs_replicates.mean()
sigma = bs_replicates.std()

# Make a histogram of the results
n, bins, patches = plt.hist(bs_replicates, bins=50, density=True)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--') # add a 'best fit' line
_ = plt.axvline(bs_replicates.mean(), color='red', linestyle='dashed', linewidth=2)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')
_ = plt.text(809,0.03,"SEM of data: {0:,.6f}\nSTD of bootstrap samples: {1:.6f}".format(sem,bs_std), color='red', fontsize=8)
_ = plt.title("Boostrap samples of rainfall (mm)")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')

    
print("****************************************************")
tema = '9. Bootstrap replicates of other statistics'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

# Take 10,000 bootstrap replicates of the var: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var, 10000)
bs_replicates = bs_replicates / 100

sns.set() # Set default Seaborn style
plt.figure()

# add a 'best fit' line
mu = bs_replicates.mean()
sigma = bs_replicates.std()

# Make a histogram of the results
n, bins, patches = plt.hist(bs_replicates, bins=50, density=True)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--') # add a 'best fit' line
_ = plt.axvline(bs_replicates.mean(), color='red', linestyle='dashed', linewidth=2)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')
#_ = plt.text(809,0.03,"SEM of data: {0:,.6f}\nSTD of bootstrap samples: {1:.6f}".format(sem,bs_std), color='red', fontsize=8)
_ = plt.title("Different Variance from Boostrap samples of rainfall (mm)")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')
    

print("****************************************************")
tema = '10. Confidence interval on the rate of no-hitters'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times, np.mean, 10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates, [2.5, 97.5])

sns.set() # Set default Seaborn style
plt.figure()

# add a 'best fit' line
mu = bs_replicates.mean()
sigma = bs_replicates.std()

# Plot the histogram of the replicates
n, bins, patches = plt.hist(bs_replicates, bins=50, density=True)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--') # add a 'best fit' line
_ = plt.axvline(bs_replicates.mean(), color='red', linestyle='dashed', linewidth=2)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')
_ = plt.text(120,0.0253,"95% confidence interval = \n[{}, \n {}] \ngames.".format(conf_int[0],conf_int[1]), color='red', fontsize=10)
_ = plt.title("Mean expected (Boostrap Samples)")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '11. Pairs bootstrap'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(total_votes/1000, dem_share, 1000)
conf_int = np.percentile(bs_slope_reps, [2.5, 97.5]) # Compute and print 95% CI for slope

sns.set() # Set default Seaborn style
plt.figure()

# add a 'best fit' line
mu = bs_slope_reps.mean()
sigma = bs_slope_reps.std()

# Plot the histogram of the replicates
n, bins, patches = plt.hist(bs_slope_reps, bins=50, density=True)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--') # add a 'best fit' line
_ = plt.axvline(bs_slope_reps.mean(), color='red', linestyle='dashed', linewidth=2)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
_ = plt.text(0.047,80,"95% confidence interval = \n[{}, \n {}] \nslope.".format(conf_int[0],conf_int[1]), color='red', fontsize=10)
_ = plt.title("Slope expected (Boostrap Samples total votes vs democratic share)")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '12. A function to do pairs bootstrap'; print("** %s\n" % tema)
tema = '13. Pairs bootstrap of literacy/fertility data'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(female_iliteracy, fertility, 1000)
conf_int = np.percentile(bs_slope_reps, [2.5, 97.5]) # Compute and print 95% CI for slope

sns.set() # Set default Seaborn style
plt.figure()

# add a 'best fit' line
mu = bs_slope_reps.mean()
sigma = bs_slope_reps.std()

# Plot the histogram of the replicates
n, bins, patches = plt.hist(bs_slope_reps, bins=50, density=True)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--') # add a 'best fit' line
_ = plt.axvline(bs_slope_reps.mean(), color='red', linestyle='dashed', linewidth=2)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
_ = plt.text(0.039,140,"95% confidence interval = \n[{}, \n {}] \nslope.".format(conf_int[0],conf_int[1]), color='red', fontsize=10)
_ = plt.title("Slope expected (Boostrap Samples female illiteracy vs fertility)")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')


sns.set() # Set default Seaborn style
plt.figure()

sns.distplot(bs_slope_reps, bins=50, kde=False, norm_hist=True, label="Boostrap Sample")
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
theorical=np.random.normal(mu,sigma,100000)
sns.distplot(theorical, color='red', hist=False, label='Theorical', hist_kws=dict(edgecolor='k', linewidth=1))
plt.suptitle(tema)
# Show the plot
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '14. Plotting bootstrap regressions'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

# Generate array of x-values for bootstrap lines: x
x = np.array([0,100])

sns.set() # Set default Seaborn style
plt.figure()

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x, 
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(female_iliteracy, fertility, marker='.', linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
_ = plt.title('Expecter Linear Regression (Female illiteracy vs fertility)')
_ = plt.suptitle(tema)
plt.margins(0.02)

# Show the plot
plt.show()
plt.style.use('default')


print("****************************************************")
print("** END                                            **")
print("****************************************************")