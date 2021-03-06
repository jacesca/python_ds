# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:42:11 2019

@author: jacqueline.cortez

Capítulo 4. Thinking probabilistically-- Continuous variables
Introduction:
    In the last chapter, you learned about probability distributions of discrete variables. 
    Now it is time to move on to continuous variables, such as those that can take on any 
    fractional value. Many of the principles are the same, but there are some subtleties. 
    At the end of this last chapter of the course, you will be speaking the probabilistic 
    language you need to launch into the inference techniques covered in the sequel to this 
    course.
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
np.set_printoptions(suppress=True) #suppress suppresses the use of scientific notation for small numbers

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

def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size=size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size=size)

    return t1 + t2

file = 'belmont.csv'
belmont_df = pd.read_csv(file)
belmont_df['segundo'] = (belmont_df.Time.str.split(':').str.get(0).astype(float)*60) + belmont_df.Time.str.split(':').str.get(1).astype(float)
belmont_outliers = pd.DataFrame(belmont_df[belmont_df.Year.isin([1970, 1973])])
belmont_outliers['outlier'] = ['fastest','slowest']
#belmont_outliers['outlier'] = pd.Series(['slowest','fastest'])
#belmont_outliers['outlier'] = belmont_outliers['Year'].apply(lambda x: 'slowest' if x==1970 else 'fastest')
belmont_original_data=belmont_df.segundo.values
belmont_df.drop(index=belmont_df[belmont_df.Year.isin([1970, 1973])].index, axis=0, inplace=True)
belmont_no_outliers = belmont_df.segundo.values

file = "michelson_speed_of_light.csv"
speed_of_light = pd.read_csv(file)
speed_of_light = speed_of_light["velocity of light in air (km/s)"].values

#Source of data: https://xyotta.com/v1/index.php/Nuclear_events_database
file='NuclearPowerAccidents2016.csv'
nuclear_incident_df = pd.read_csv(file, parse_dates=['Date'])
nuclear_incident_df.sort_values('Date', inplace=True)
nuclear_incident_df['Days Since Previous'] = (nuclear_incident_df['Date'] -
                                               nuclear_incident_df['Date'].shift(1)).astype('timedelta64[D]')
nuclear_incident_df.reset_index(inplace=True, drop=True)
nuclear_incident_df.drop(labels=0,axis=0,inplace=True)
inter_times = nuclear_incident_df['Days Since Previous'].values

print("****************************************************")
tema = '4. Introduction to the Normal distribution'; print("** %s\n" % tema)

SEED=42
np.random.seed(SEED) # Seed random number generator

#Get param from the real data
mean = np.mean(speed_of_light)
std = np.std(speed_of_light)
#Get a normal sample from the computed mean and std
samples = np.random.normal(mean, std, size=10000)
x, y = ecdf(speed_of_light)
x_theor, y_theor = ecdf(samples)

sns.set() # Set default Seaborn style
fig, ax = plt.subplots(1, 1)
_ = ax.plot(x_theor, y_theor)
_ = ax.plot(x, y, marker='.', linestyle='none')
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
#ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
_ = plt.xlabel('Speed of light in air (km/s)')
_ = plt.ylabel('CDF')
_ = plt.title("Verifying the normal distribution of Albert A. Michelson'data")
_ = plt.suptitle(tema)

# Show the plot
#plt.ylim(-0.01, 0.42)
#plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '5. The Normal PDF'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1  = np.random.normal(20,  1, size=100000)
samples_std3  = np.random.normal(20,  3, size=100000)
samples_std10 = np.random.normal(20, 10, size=100000)

sns.set() # Set default Seaborn style
plt.figure()

# Make histograms
_ = plt.hist(samples_std1, density=True, histtype='step', bins=100)
_ = plt.hist(samples_std3, density=True, histtype='step', bins=100)
_ = plt.hist(samples_std10, density=True, histtype='step', bins=100)

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
_ = plt.xlabel('Samples')
_ = plt.ylabel('PDF')
_ = plt.title('A hacked statistic for Normal PDF')
_ = plt.suptitle(tema)

# Show the plot
plt.ylim(-0.01, 0.42)
#plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '6. The Normal CDF'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()

# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

# Plot CDFs
_ = plt.plot(x_std1, y_std1, marker='.', linestyle='none')
_ = plt.plot(x_std3, y_std3, marker='.', linestyle='none')
_ = plt.plot(x_std10, y_std10, marker='.', linestyle='none')

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
_ = plt.xlabel('Samples')
_ = plt.ylabel('PDF')
_ = plt.title('A hacked statistic for Normal PDF')
_ = plt.suptitle(tema)

# Show the plot
#plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '9. Are the Belmont Stakes results Normally distributed?\nBelmont since 1926 until 2016'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

sns.set_style("darkgrid")

#First, we explore the original data in boxplot
fig, ax = plt.subplots()
ax.set_title('Exploring Data (First step)') # Add the title
ax.boxplot([belmont_original_data]) # Add a boxplot for the column 
ax.set_xticklabels(['Belmont race data']) # Add x-axis tick labels
ax.set_ylabel("Belmont winning time (sec.)") # Add a y-axis label
plt.suptitle(tema)
plt.text(1.03,154,"Slower than normal! (outlier)", color='red')
plt.text(1.03,144,"Faster than normal! (outlier)", color='red')
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
sns.set_style("white")


#Second, we compute the data without outliers, get a theorical sample and plot the PDF
mu = np.mean(belmont_no_outliers) # Compute mean and standard deviation: mu, sigma
sigma = np.std(belmont_no_outliers)
samples = np.random.normal(mu, sigma, size=10000) # Sample out of a normal distribution with this mu and sigma: samples

plt.figure()
_ = plt.hist(samples, density=True, histtype='step', bins=100, label="Theorical data")# Make histograms
_ = plt.hist(belmont_no_outliers, density=True, histtype='step', label="Belmont data without outliers")# Make histograms
_ = plt.hist(belmont_original_data, density=True, histtype='step', label="Belmont original data")# Make histograms
_ = plt.legend() # Make a legend, set limits and show plot
_ = plt.xlabel('Comparisons between data')
_ = plt.ylabel('PDF')
_ = plt.title('Verifyng the normality of Belmont Data (Second Step)')
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()


#Third Look the CDF
plt.figure()
x_theor, y_theor = ecdf(samples) # Get the CDF of the samples and of the data
x, y = ecdf(belmont_no_outliers)

_ = plt.plot(x_theor, y_theor) # Plot the CDFs and show the plot
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
_ = plt.title('Plot the CDF (Third step)')
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()



plt.style.use('default')

print("****************************************************")
tema = '10. What are the chances of a horse matching or beating Secretariat\'s record?'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

speed = belmont_outliers.segundo.min()
samples = np.random.normal(mu, sigma, size=1000000) # Take a million samples out of the Normal distribution: samples

prob = np.sum(samples<=speed)/len(samples) # Compute the fraction that are faster than 144 seconds: prob

# Print the result
print('Probability of besting Secretariat: {}\n'.format(prob))


print("****************************************************")
tema = '11. The Exponential distribution'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

#Calculate the mean of the data
mean = np.mean(inter_times)

#Prepare a theorical exponential distribution sample
samples = np.random.exponential(mean, size=10000)

#Get de ECDF of both data set
x, y = ecdf(inter_times)
x_theor, y_theor = ecdf(samples)

sns.set() # Set default Seaborn style

#Plotting the PDF graph
plt.figure()
_ = plt.hist(samples, density=True, histtype='step', bins=100, label="Theorical data")# Make histograms
_ = plt.hist(inter_times, density=True, histtype='step', label="Register events")# Make histograms
_ = plt.legend() # Make a legend, set limits and show plot
_ = plt.xlabel('Days between nuclear incident')
_ = plt.ylabel('PDF')
_ = plt.title('The PDF Graph')
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()

#Plotting the CDF graph
plt.figure()
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.legend(('Theorical data', 'Register events'))
_ = plt.xlabel('Time in days')
_ = plt.ylabel('CDF')
_ = plt.title('The CDF Graph')
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()

plt.style.use('default') #Return to dafault style


print("****************************************************")
tema = '12. Distribution of no-hitters and cycles'; print("** %s\n" % tema)

np.random.seed(42) # Seed random number generator

# Draw samples of waiting times: waiting_times
samples = successive_poisson(764, 715, 100000)
x_theor, y_theor = ecdf(samples)

sns.set() # Set default Seaborn style

#Plotting the PDF graph
plt.figure()
_ = plt.hist(samples, bins=100, density=True, histtype='step')
_ = plt.xlabel("total waiting time (games)")
_ = plt.ylabel("PDF")
_ = plt.title('The PDF Graph')
_ = plt.suptitle(tema)
plt.show()

#Plotting the CDF graph
plt.figure()
_ = plt.plot(x_theor, y_theor)
_ = plt.xlabel('Time in days')
_ = plt.ylabel('CDF')
_ = plt.title('The CDF Graph')
_ = plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()

plt.style.use('default') #Return to dafault style


print("****************************************************")
print("** END                                            **")
print("****************************************************")