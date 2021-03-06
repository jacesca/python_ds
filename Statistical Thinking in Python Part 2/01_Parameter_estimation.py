# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:42:11 2019

@author: jacqueline.cortez

Capítulo 1. Parameter estimation by optimization
Introduction:
    When doing statistical inference, we speak the language of probability. 
    A probability distribution that describes your data has parameters. 
    So, a major goal of statistical inference is to estimate the values of 
    these parameters, which allows us to concisely and unambiguously describe 
    our data and draw conclusions from it. In this chapter, you will learn how 
    to find the optimal parameters, those that best describe your data.
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


file='no-hitters.csv'
nohitters_df = pd.read_csv(file, sep=';', parse_dates=['Date'])
nohitters_df.sort_values('Date', inplace=True)
nohitters_df['Days Since Previous'] = (nohitters_df['Date'] -
                                       nohitters_df['Date'].shift(1)).astype('timedelta64[D]')
nohitters_df.reset_index(inplace=True, drop=True)
nohitters_df.drop(labels=0,axis=0,inplace=True)
nohitter_times = nohitters_df['Days Since Previous'].values

file = '2008_swing_states.csv'
swing_states_df = pd.read_csv(file)
total_votes = swing_states_df.total_votes.values
dem_share = swing_states_df.dem_share.values

file = "Female_Educatiov_vs_Fertility.csv"
fertility_df = pd.read_csv(file, skiprows=1, names=['country','continent','fem_literacity','fertility','population'])
fertility = fertility_df.fertility.values
female_literacy = fertility_df.fem_literacity.values
female_iliteracy = 100-fertility_df.fem_literacity.values

print("****************************************************")
tema = '2. How often do we get no-hitters?'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator
sns.set() # Set default Seaborn style

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time, bins=50, density=True, histtype='step')
_ = plt.xlabel('Days between no-hitters')
_ = plt.ylabel('PDF')
_ = plt.title('The PDF Graph')
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()

plt.style.use('default') #Return to dafault style



print("****************************************************")
tema = '3. Do the data follow our story?'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style

x, y = ecdf(nohitter_times) # Create an ECDF from real data: x, y
x_theor, y_theor = ecdf(inter_nohitter_time) # Create a CDF from theoretical samples: x_theor, y_theor

plt.figure()

# Overlay the plots
_ = plt.plot(x_theor, y_theor, label='Theorical data')
_ = plt.plot(x, y, marker='.', linestyle='none', label='Register events')
_ = plt.legend()

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Days between no-hitters')
plt.ylabel('CDF')
_ = plt.title('The CDF Graph')
_ = plt.suptitle(tema)

# Show the plot
plt.show()

plt.style.use('default') #Return to dafault style



print("****************************************************")
tema = '4. How is this parameter optimal?'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style

samples_half = np.random.exponential(tau/2,10000) # Take samples with half tau: samples_half
samples_double = np.random.exponential(2*tau, 10000) # Take samples with double tau: samples_double
# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

plt.figure()

# Overlay the plots
_ = plt.plot(x, y, marker='.', linestyle='none', label='Register events')
_ = plt.plot(x_theor, y_theor, label='Theorical data with Tau')
_ = plt.plot(x_half, y_half, label='Theorical data with Half Tau')
_ = plt.plot(x_double, y_double, label='Theorical data with Double Tau')
_ = plt.legend()

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Days between no-hitters')
plt.ylabel('CDF')
_ = plt.title('The PDF Graph')
_ = plt.suptitle(tema)

# Show the plot
plt.show()

plt.style.use('default') #Return to dafault style


print("****************************************************")
tema = '5. Linear regression by least squares'; print("** %s\n" % tema)

# Show the Pearson correlation coefficient
pearson = pearson_r(total_votes/1000, dem_share)
slope, intercept = np.polyfit(total_votes/1000, dem_share, 1)
x = np.linspace(np.min(total_votes/1000), np.max(total_votes/1000), 1000)
y = slope*x + intercept

sns.set() # Set default Seaborn style
plt.figure()

# Plot the illiteracy rate versus fertility
#_ = plt.plot(total_votes/1000, dem_share, marker='.', linestyle='none')
_ = plt.scatter(total_votes/1000, dem_share, marker='.')
_ = plt.plot(x, y, color='red')
_ = plt.xlabel('total voles (thousands)')
_ = plt.ylabel('percent of votes for obama')
_ = plt.text(300,30,"Pearson correlation: {0:,.4f}\nSlope: {1:.4f}\nIntercept: {2:,.4f}".format(pearson,slope,intercept), color='red')
_ = plt.title('2008 US swing states election results')
_ = plt.suptitle(tema)
plt.margins(0.02)
plt.show()

plt.style.use('default') #Return to dafault style


print("****************************************************")
tema = '6. EDA of literacy/fertility data'; print("** %s\n" % tema)
tema = '7. Linear regression'; print("** %s\n" % tema)

# Show the Pearson correlation coefficient
pearson = pearson_r(female_iliteracy, fertility)
a, b = np.polyfit(female_iliteracy, fertility, 1)
x = np.array([0,100])
y = a*x + b

sns.set() # Set default Seaborn style
plt.figure()

# Plot the illiteracy rate versus fertility
_ = plt.plot(female_iliteracy, fertility, marker='.', linestyle='none')
_ = plt.plot(x, y)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')
_ = plt.text(30,1.25,"Pearson correlation: {0:,.4f}\nSlope: {1:.4f}\nIntercept: {2:,.4f}".format(pearson,a,b), color='red')
_ = plt.title('Data around the World')
_ = plt.suptitle(tema)
plt.margins(0.02)
plt.show()

plt.style.use('default') #Return to dafault style


print("****************************************************")
tema = '8. How is it optimal?'; print("** %s\n" % tema)

slope = a
a_vals = np.linspace(0, 0.1, 10000) # Specify slopes to consider: a_vals
rss = np.empty_like(a_vals) # Initialize sum of square of residuals: rss

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*female_iliteracy - b)**2)

slope_manual = a_vals[np.argmin(rss)]

sns.set() # Set default Seaborn style
plt.figure()

# Plot the RSS
_ = plt.plot(a_vals, rss, '-')
_ = plt.xlabel('slope (children per woman / percent illiterate)')
_ = plt.ylabel('sum of square of residuals')
_ = plt.text(0.016,400,"The python calculated slope: {0:,.6f}\nThe manual found slope: {1:.6f}".format(slope,slope_manual), color='red')
_ = plt.title('Finding the perfect slope')
_ = plt.suptitle(tema)

plt.show()

plt.style.use('default') #Return to dafault style


print("****************************************************")
tema = '11. Linear regression on appropriate Anscombe data'; print("** %s\n" % tema)

x = np.array([10.,  8., 13.,  9., 11., 14.,  6.,  4., 12.,  7.,  5.])
y = np.array([ 8.04,  6.95,  7.58,  8.81,  8.33,  9.96,  7.24,  4.26, 10.84, 4.82,  5.68])

pearson = pearson_r(x, y) # Compute the Pearson correlation coefficient
a, b = np.polyfit(x, y, 1) # Perform linear regression: a, b

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = a * x_theor + b

sns.set() # Set default Seaborn style
plt.figure()

# Plot the Anscombe data and theoretical line
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.plot(x_theor, y_theor)
_ = plt.xlabel('x')
_ = plt.ylabel('y')
_ = plt.text(3,9.5,"Pearson correlation: {0:,.4f}\nSlope: {1:.6f}\nIntercept: {2:.2f}".format(pearson,a,b), color='red')
_ = plt.title('Verifying the linear regression with the EDA')
_ = plt.suptitle(tema)
plt.show()

plt.style.use('default') #Return to dafault style


print("****************************************************")
tema = '12. Linear regression on all Anscombe data'; print("** %s\n" % tema)

anscombe_x = [np.array([10.,  8., 13.,  9., 11., 14.,  6.,  4., 12.,  7.,  5.]),
              np.array([10.,  8., 13.,  9., 11., 14.,  6.,  4., 12.,  7.,  5.]),
              np.array([10.,  8., 13.,  9., 11., 14.,  6.,  4., 12.,  7.,  5.]),
              np.array([ 8.,  8.,  8.,  8.,  8.,  8.,  8., 19.,  8.,  8.,  8.])]

anscombe_y = [np.array([ 8.04,  6.95,  7.58,  8.81,  8.33,  9.96,  7.24,  4.26, 10.84, 4.82,  5.68]),
              np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.1 , 6.13, 3.1 , 9.13, 7.26, 4.74]),
              np.array([ 7.46,  6.77, 12.74,  7.11,  7.81,  8.84,  6.08,  5.39,  8.15, 6.42,  5.73]),
              np.array([ 6.58,  5.76,  7.71,  8.84,  8.47,  7.04,  5.25, 12.5 ,  5.56, 7.91,  6.89])]

sns.set() # Set default Seaborn style
plt.figure()
i = 1
point = [(4,9), (7,3.5), (5,10), (8,10.5)]
for x, y in zip(anscombe_x, anscombe_y):
    pearson = pearson_r(x, y) # Compute the Pearson correlation coefficient
    a, b = np.polyfit(x, y, 1) # Perform linear regression: a, b
    # Generate theoretical x and y data: x_theor, y_theor
    x_theor = np.array([np.min(x-1), np.max(x)+1])
    y_theor = a * x_theor + b
    plt.subplot(2,2,i)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.plot(x_theor, y_theor)
    _ = plt.xlabel('x')
    _ = plt.ylabel('y')
    _ = plt.text(point[i-1][0],point[i-1][1],"Pearson correlation: {0:,.4f}\nSlope: {1:.6f}\nIntercept: {2:.2f}".format(pearson,a,b), color='red', fontsize=6)
    _ = plt.title('Set {}'.format(i))
    i+=1

_ = plt.suptitle(tema)
plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=0.5, hspace=0.7)
plt.show()

plt.style.use('default') #Return to dafault style


print("****************************************************")
print("** END                                            **")
print("****************************************************")