# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:42:11 2019

@author: jacqueline.cortez

Capítulo 4. Hypothesis test example
Introduction:
    As you saw from the last chapter, hypothesis testing can be a bit tricky. 
    You need to define the null hypothesis, figure out how to simulate it, and 
    define clearly what it means to be "more extreme" in order to compute the 
    p-value. Like any skill, practice makes perfect, and this chapter gives you 
    some good practice with hypothesis tests.
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


file='mlb_nohitters.csv'
nohitters_df = pd.read_csv(file, parse_dates=['date'])
nohitters_df.sort_values('date', inplace=True)
nohitters_df['Games Since Previous'] = (nohitters_df['game_number'] -
                                       nohitters_df['game_number'].shift(1)-1).fillna(-1)
nohitters_df.reset_index(inplace=True, drop=True)
nohitters_df["period"] = (nohitters_df['date'] < '1920-01-01').replace([True, False],["Before new rules", "After new rules"])
nht_dead = nohitter_times = nohitters_df[nohitters_df.date<'1920-01-01']['Games Since Previous'].values
nht_live = nohitter_times = nohitters_df[nohitters_df.date>='1920-01-01']['Games Since Previous'].values


file = "Female_Educatiov_vs_Fertility.csv"
fertility_df = pd.read_csv(file, skiprows=1, names=['country','continent','fem_literacity','fertility','population'])
fertility = fertility_df.fertility.values
female_literacy = fertility_df.fem_literacity.values
illiteracy = 100-fertility_df.fem_literacity.values


file = "bee_sperm.csv"
bee_sperm_df = pd.read_csv(file, comment='#')
bee_sperm_df['Live Sperm per half ml'] =bee_sperm_df.AliveSperm.values/1000/500            
control = bee_sperm_df[bee_sperm_df.Treatment == 'Control']['Live Sperm per half ml'].values
treated = bee_sperm_df[bee_sperm_df.Treatment == 'Pesticide']['Live Sperm per half ml'].values


print("****************************************************")
tema = '2. The vote for the Civil Rights Act in 1964'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)
frac_dems = sum(dems)/len(dems)
frac_reps = sum(reps)/len(reps)

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea, 10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)

sns.set() # Set default Seaborn style

# add a 'best fit' line
mu = perm_replicates .mean()
sigma = perm_replicates .std()

# Plot the histogram of the replicates
n, bins, patches = plt.hist(perm_replicates , bins=50, density=True)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(perm_replicates.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(frac_dems, color='red', linestyle='dashed', linewidth=2)
_ = plt.axvline(frac_reps, color='blue', linestyle='dashed', linewidth=2)
_ = plt.xlabel('percentages of dems votes')
_ = plt.ylabel('PDF')
_ = plt.text(0.725,22.8,"Frac dems = {0:,.6f}".format(frac_dems), color='red', fontsize=8)
_ = plt.text(0.725,22,"Frac reps = {0:,.6f}".format(frac_reps), color='blue', fontsize=8)
_ = plt.text(0.629,25,"p-value \n(frac dems<={0:,.6f}) \n= {1:,.6f}".format(frac_dems,p), color='black', fontsize=8)
_ = plt.text(0.725,17,"Ho: There's no difference\nbetween dems and reps.", color='black', fontsize=8)
_ = plt.title("Did party affiliation make a difference in the vote?")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')


#Evaluating the other extreme
np.random.seed(42) # Seed random number generator
perm_replicates = draw_perm_reps(reps, dems, frac_yea, 10000) # Acquire permutation samples: perm_replicates
p = np.sum(perm_replicates >= 136/171) / len(perm_replicates) # Compute and print p-value: p
mu = perm_replicates .mean() # add a 'best fit' line
sigma = perm_replicates .std()

sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(perm_replicates , bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(perm_replicates.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(frac_dems, color='red', linestyle='dashed', linewidth=2)
_ = plt.axvline(frac_reps, color='blue', linestyle='dashed', linewidth=2)
_ = plt.xlabel('percentages of reps votes')
_ = plt.ylabel('PDF')
_ = plt.text(0.72,17.7,"Frac dems = {0:,.6f}".format(frac_dems), color='red', fontsize=8)
_ = plt.text(0.72,17,"Frac reps = {0:,.6f}".format(frac_reps), color='blue', fontsize=8)
_ = plt.text(0.73,20,"p-value \n(frac reps>={0:,.6f}) \n= {1:,.6f}".format(frac_reps,p), color='black', fontsize=8)
_ = plt.text(0.72,15,"Ho: There's no difference\nbetween dems and reps.", color='black', fontsize=8)
_ = plt.title("Did party affiliation make a difference in the vote?")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '4. A time-on-website analog'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

#Swarmplot
sns.set() # Set default Seaborn style
plt.figure()
_ = sns.swarmplot('period', 'Games Since Previous', data=nohitters_df) # Make bee swarm plot
_ = plt.xlabel('Period of times')# Label axes
_ = plt.ylabel('Games between no-hitters')
_ = plt.title("Swarmplot from no-hitters database")
_ = plt.suptitle(tema)
# Show the plot
plt.show()
plt.style.use('default')


#Boxplot
sns.set() # Set default Seaborn style
plt.figure()
_ = sns.boxplot('period', 'Games Since Previous', data=nohitters_df) # Make bee swarm plot
_ = plt.xlabel('Period of times')# Label axes
_ = plt.ylabel('Games between no-hitters')
_ = plt.title("Boxplot from no-hitters database")
_ = plt.suptitle(tema)
# Show the plot
plt.show()
plt.style.use('default')


#ECDF
x_1, y_1 = ecdf(nht_dead)
x_2, y_2 = ecdf(nht_live)
sns.set() # Set default Seaborn style
plt.figure()
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red', label="Before new rules games")
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue', label="After new rules games")
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('ECDF')
_ = plt.title("no-hitters database")
_ = plt.suptitle(tema)
plt.legend()
plt.margins(0.02)
plt.show()
plt.style.use('default')


# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)
# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, 10000)
# Compute and print the p-value: p
p = np.sum(perm_replicates<=nht_diff_obs)*1.0/len(perm_replicates)

#Understanding the boostraps samples
mu = perm_replicates.mean() # add a 'best fit' line
sigma = perm_replicates .std()
ci = np.percentile(perm_replicates,[2.5,97.5])



sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(perm_replicates , bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(perm_replicates.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(nht_diff_obs, color='red', linestyle='dashed', linewidth=2)
_ = plt.xlabel('differences observed in boostraps samples')
_ = plt.ylabel('PDF')
_ = plt.text( 100,0.00365,"Dead - live period", color='black', fontsize=8)
_ = plt.text( 100,0.0035,"Observed difference = {0:,.2f}".format(nht_diff_obs), color='red', fontsize=8)
_ = plt.text( 100,0.00335,"Boostraps samples mean = {0:,.2f}".format(mu), color='black', fontsize=8)
_ = plt.text( 100,0.0032,"Boostraps CI = [{0:,.2f}, {1:,.2f}]".format(ci[0],ci[1]), color='black', fontsize=8)
_ = plt.text(-330,0.0035,"The new observed value \n<= {0:,.2f}) \n= {1:,.6f}".format(nht_diff_obs,p), color='black', fontsize=8)
_ = plt.text( 100,0.0028,"Ho: There's no difference.", color='black', fontsize=8)
_ = plt.title("Determine if these rule changes resulted in a slower rate of no-hitters")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '8. Hypothesis test on Pearson correlation'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

num_exp = 10000 # Initialize permutation replicates: perm_replicates
r_obs = pearson_r(illiteracy, fertility) # Compute observed correlation: r_obs
perm_replicates = np.empty(num_exp)

for i in range(num_exp): # Draw replicates
    illiteracy_permuted = np.random.permutation(illiteracy) # Permute illiteracy measurments: illiteracy_permuted
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility) # Compute Pearson correlation
    
mu = perm_replicates.mean()
sigma = perm_replicates.std()
ci = np.percentile(perm_replicates, [2.5,97.5])
p = np.sum(perm_replicates>=r_obs)/len(perm_replicates) # Compute p-value: p

sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(perm_replicates , bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(perm_replicates.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(r_obs, color='red', linestyle='dashed', linewidth=2)
_ = plt.xlabel('differences observed in Peorson Correl (boostraps samples)')
_ = plt.ylabel('PDF')
_ = plt.text(0.2, 5,"The new pearson correl value \n<= {0:,.2f}) = {1:,.6f}".format(r_obs,p), color='red', fontsize=10)
_ = plt.text(0.2, 4,"Boostraps samples mean = {0:,.2f}\nCI = [{1:,.2f}, {2:,.2f}]".format(mu,ci[0],ci[1]), color='black', fontsize=10)
_ = plt.text(0.2, 3,"Ho: There's no correlation between \nilliteracy and fertility.", color='black', fontsize=10)
_ = plt.title("May fertility be totally independent of its illiteracy?")
_ = plt.suptitle(tema)

# Show the plot
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '9. Do neonicotinoid insecticides have \nunintended consequences?'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

#Swarmplot
sns.set() # Set default Seaborn style
plt.figure()
_ = sns.swarmplot('Treatment', 'Live Sperm per half ml', data=bee_sperm_df) # Make bee swarm plot
_ = plt.xlabel('Group of Bees')# Label axes
_ = plt.ylabel('Live Sperm per half ml')
_ = plt.title("How the pesticide treatment affected the \ncount of live sperm per half milliliter of semen")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.80, wspace=None, hspace=None)
plt.show() # Show the plot
plt.style.use('default')


#Boxplot
sns.set() # Set default Seaborn style
plt.figure()
_ = sns.boxplot('Treatment', 'Live Sperm per half ml', data=bee_sperm_df) # Make bee swarm plot
_ = plt.xlabel('Group of Bees')# Label axes
_ = plt.ylabel('Live Sperm per half ml')
_ = plt.title("How the pesticide treatment affected the \ncount of live sperm per half milliliter of semen")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.80, wspace=None, hspace=None)
plt.show() # Show the plot
plt.style.use('default')


#ecdf
x_control, y_control = ecdf(control) # Compute x,y values for ECDFs
x_treated, y_treated = ecdf(treated)

sns.set() # Set default Seaborn style
plt.figure()
plt.plot(x_control, y_control, marker='.', linestyle='none') # Plot the ECDFs
plt.plot(x_treated, y_treated, marker='.', linestyle='none')
plt.margins(0.02) # Set the margins
plt.legend(('control', 'treated'), loc='lower right') # Add a legend
plt.xlabel('millions of alive sperm per mL') # Label axes and show plot
plt.ylabel('ECDF')
plt.title("How the pesticide treatment affected the \ncount of live sperm per half milliliter of semen")
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.80, wspace=None, hspace=None)
plt.show() # Show the plot
plt.style.use('default')


print("****************************************************")
tema = '10. Bootstrap hypothesis test on bee sperm counts'; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

num_exp = 10000 # Initialize permutation replicates: perm_replicates

# Compute the difference in mean sperm count: diff_means
diff_means = control.mean()-treated.mean()
control_mean = control.mean()
treated_mean = treated.mean()

# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate((control,treated)))
# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count
# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted, np.mean, size=num_exp)
bs_reps_treated = draw_bs_reps(treated_shifted, np.mean, size=num_exp)
# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control - bs_reps_treated

mu = bs_replicates.mean()
sigma = bs_replicates.std()
ci = np.percentile(bs_replicates,[2.5, 97.5])
# Compute and print p-value: p
p = np.sum(bs_replicates >= diff_means) / len(bs_replicates)

#hist
sns.set() # Set default Seaborn style
plt.figure()
n, bins, patches = plt.hist(bs_replicates , bins=50, density=True) # Plot the histogram of the replicates
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
_ = plt.plot(bins, y, '--', color='black') # add a 'best fit' line
_ = plt.axvline(bs_replicates.mean(), color='black', linestyle='dashed', linewidth=1)
_ = plt.axvline(diff_means, color='red', linestyle='dashed', linewidth=2)
_ = plt.xlabel('differences observed in Mean (control - treated)')
_ = plt.ylabel('PDF')
_ = plt.text(0.45, 1.4,"Boostrap replicate \n>= {0:,.2f}) = {1:,.6f}".format(diff_means,p), color='red', fontsize=9)
_ = plt.text(0.45, 1.15,"Boostrap samples \nmean = {0:,.2f}\nCI = [{1:,.2f}, {2:,.2f}]".format(mu,ci[0],ci[1]), color='black', fontsize=9)
_ = plt.text(0.45, 0.9,"Ho: There's no diff. \nbetween group of \nBees.", color='black', fontsize=9)
_ = plt.title("How the pesticide treatment affected the \ncount of live sperm per half milliliter of semen")
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
# Show the plot
plt.show()
plt.style.use('default')

print("****************************************************")
print("** END                                            **")
print("****************************************************")