# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:50:56 2019

@author: jacqueline.cortez

Capítulo 1. Graphical exploratory data analysis
Introduction:
    Look before you leap! A very important proverb, indeed. 
	Prior to diving in headlong into sophisticated statistical 
	inference techniques, you should first explore your data 
	by plotting them and computing simple summary statistics. 
	This process, called exploratory data analysis, is a crucial 
	first step in statistical analysis of data. So it is a fitting 
	subject for the first chapter of Statistical Thinking in Python.
Excercise 01-05
"""

# Import packages
import pandas as pd                  #For loading tabular data
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

iris = sns.load_dataset('iris') #load a pre-data 
#print(iris.head())

setosa_petal_length     = iris[iris.species == 'setosa'].petal_length.values
versicolor_petal_length = iris[iris.species == 'versicolor'].petal_length.values
virginica_petal_length  = iris[iris.species == 'virginica'].petal_length.values

file = '2008_swing_states.csv'
swing_states = pd.read_csv(file)
print("Reading the data ({})...\n".format(file.upper()))

#print("\nAnother dataset inside seaborn: ")
#print(sns.get_dataset_names())
# [ 'anscombe', 'attention', 'brain_networks', 'car_crashes', 'diamonds', 'dots', 
#   'exercise', 'flights', 'fmri', 'gammas', 'iris', 'mpg', 'planets', 'tips', 'titanic']
#Source of data--> https://github.com/mwaskom/seaborn-data

print("****************************************************")
tema = '4 Plotting a histogram'; print("** %s\n" % tema)

bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


plt.subplot(2,2,1)
_ = plt.hist(swing_states.dem_share) # Plot histogram of versicolor petal lengths
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('number of counties')
_ = plt.title('2008 Election Result')


plt.subplot(2,2,2)
_ = plt.hist(swing_states.dem_share, bins=bin_edges) # Plot histogram of versicolor petal lengths
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('number of counties')
_ = plt.title('2008 Election Result')


plt.subplot(2,2,3)
_ = plt.hist(swing_states.dem_share, bins=20) # Plot histogram of versicolor petal lengths
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('number of counties')
_ = plt.title('2008 Election Result')


plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0.3, hspace=0.5)
plt.show() # Show histogram

print("****************************************************")
tema = '5. Plotting a histogram of iris data'; print("** %s\n" % tema)
tema = '6. Axis labels!'; print("** %s\n" % tema)
tema = '7. Adjusting the number of bins in a histogram'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style

# Number of bins is the square root of number of data points: n_bins
n_bins = int(np.sqrt(len(versicolor_petal_length)))

plt.figure() 
_ = plt.hist(versicolor_petal_length, bins=n_bins) # Plot histogram of versicolor petal lengths
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')
_ = plt.title('Versicolor Petal Legnth')
_ = plt.suptitle(tema)

#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.show() # Show histogram
plt.style.use('default')


print("****************************************************")
tema = '9. Bee swarm plot'; print("** %s\n" % tema)

sns.set_style("darkgrid")

plt.figure() 
_ = sns.swarmplot(x='species', y='petal_length', data=iris) # Create bee swarm plot with Seaborn's default settings
_ = plt.xlabel('Species')
_ = plt.ylabel('Petal length (cm)')
_ = plt.title('Iris Petal Length')
_ = plt.suptitle(tema)

#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.show()
sns.set_style("white")


print("****************************************************")
tema = '11. Plotting all of your data Empirical cumulative distribution functions'; print("** %s\n" % tema)
sns.set_style("darkgrid")

x = np.sort(swing_states.dem_share)
y = np.arange(1, len(x)+1)/len(x)

x_pa = np.sort(swing_states[swing_states.state == 'PA'].dem_share)
y_pa = np.arange(1, len(x_pa)+1)/len(x_pa)

x_oh = np.sort(swing_states[swing_states.state == 'OH'].dem_share)
y_oh = np.arange(1, len(x_oh)+1)/len(x_oh)

x_fl = np.sort(swing_states[swing_states.state == 'FL'].dem_share)
y_fl = np.arange(1, len(x_fl)+1)/len(x_fl)

plt.figure() 
plt.subplot(2,1,1)
#plt.margins(0.02)
_ = plt.plot(x, y, marker='.', linestyle='none') 
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('ECDF')
_ = plt.title('2008 Election Result')


plt.subplot(2,1,2)
#plt.margins(0.02)
_ = plt.plot(x_pa, y_pa, marker='.', linestyle='none', label='Pensilvania') 
_ = plt.plot(x_oh, y_oh, marker='.', linestyle='none', label='Ohio') 
_ = plt.plot(x_fl, y_fl, marker='.', linestyle='none', label='Florida') 
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('ECDF')
_ = plt.title('2008 Election Result')
plt.legend()

plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0.5)
plt.show()

sns.set_style("white")

print("****************************************************")
tema = '12. Computing the ECDF'; print("** %s\n" % tema)

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data) # Number of data points: n
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n # y-data for the ECDF: y
    
    return x, y


print("****************************************************")
tema = '13. Plotting the ECDF'; print("** %s\n" % tema)


x_vers, y_vers = ecdf(versicolor_petal_length) # Compute ECDF for versicolor data: x_vers, y_vers

plt.figure() 
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none') # Generate plot
_ = plt.xlabel('Petal length (cm)')
_ = plt.ylabel('ECDF')
_ = plt.title('Iris Petal Length')
_ = plt.suptitle(tema)

#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0.5)
plt.show()


print("****************************************************")
tema = '14. Comparison of ECDFs'; print("** %s\n" % tema)

# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

sns.set() # Set default Seaborn style

# Plot all ECDFs on the same plot
plt.figure() 
_ = plt.plot(x_set, y_set, marker='.', linestyle='none')
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none')

# Annotate the plot
_ = plt.xlabel('Petal length (cm)')
_ = plt.ylabel('ECDF')
_ = plt.title('Iris Petal Length')
_ = plt.suptitle(tema)

# Display the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')

print("****************************************************")
print("** END                                            **")
print("****************************************************")