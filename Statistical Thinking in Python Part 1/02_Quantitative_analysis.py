# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:41:01 2019

@author: jacqueline.cortez

Capítulo 2. Quantitative exploratory data analysis
Introduction:
    In the last chapter, you learned how to graphically explore data. In this 
    chapter, you will compute useful summary statistics, which serve to concisely 
    describe salient features of a data set with a few numbers.
Excercise 01-05
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


iris = sns.load_dataset('iris') #load a pre-data 
#print(iris.head())

setosa_petal_length     = iris[iris.species == 'setosa'].petal_length.values
versicolor_petal_length = iris[iris.species == 'versicolor'].petal_length.values
virginica_petal_length  = iris[iris.species == 'virginica'].petal_length.values

versicolor_petal_width  = iris[iris.species == 'versicolor'].petal_width.values

x_vers, y_vers = ecdf(versicolor_petal_length)

#print("\nAnother dataset inside seaborn: ")
#print(sns.get_dataset_names())
# [ 'anscombe', 'attention', 'brain_networks', 'car_crashes', 'diamonds', 'dots', 
#   'exercise', 'flights', 'fmri', 'gammas', 'iris', 'mpg', 'planets', 'tips', 'titanic']
#Source of data--> https://github.com/mwaskom/seaborn-data

print("****************************************************")
tema = '2. Computing means'; print("** %s\n" % tema)

mean_length_seto = np.mean(setosa_petal_length) # Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length) # Compute the mean: mean_length_vers
mean_length_virg = np.mean(virginica_petal_length) # Compute the mean: mean_length_vers

print('  I. setosa:    ', mean_length_seto, 'cm') # Print the result with some nice formatting
print(' II. versicolor:', mean_length_vers, 'cm') 
print('III. virginica: ', mean_length_virg, 'cm\n') 


print("****************************************************")
tema = '5. Computing percentiles'; print("** %s\n" % tema)

percentiles = np.array([2.5, 25, 50, 75, 97.5]) # Specify array of percentiles: percentiles

ptiles_seto = np.percentile(setosa_petal_length, percentiles) # Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)
ptiles_virg = np.percentile(virginica_petal_length, percentiles)

print("  I. Percentiles {} of setosa's sample:     {}.".format(percentiles, ptiles_seto)) # Print the result
print(" II. Percentiles {} of versicolor's sample: {}.".format(percentiles, ptiles_vers))
print("III. Percentiles {} of virginica's sample:  {}.\n".format(percentiles, ptiles_virg))


print("****************************************************")
tema = '6. Comparing percentiles to ECDF'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style

# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')
_ = plt.title('Versicolor Petal Length')
_ = plt.suptitle(tema)

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red', linestyle='none')

# Show the plot
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '7. Box-and-whisker plot'; print("** %s\n" % tema)

sns.set_style("darkgrid")

# Plot the ECDF
plt.figure()
_ = sns.boxplot(x='species', y='petal_length', data=iris) # Create box plot with Seaborn's default settings
_ = plt.xlabel('Species')
_ = plt.ylabel('Petal length (cm)')
_ = plt.title('Iris Petal Length')
_ = plt.suptitle(tema)

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red', linestyle='none')

# Show the plot
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0.5)
plt.show()
sns.set_style("white")


print("****************************************************")
tema = '9. Computing the variance'; print("** %s\n" % tema)

differences = versicolor_petal_length - np.mean(versicolor_petal_length) # Array of differences to mean: differences
diff_sq = differences ** 2                                               # Square the differences: diff_sq
variance_explicit = np.mean(diff_sq)                                     # Compute the mean square difference: variance_explicit
variance_np =  np.var(versicolor_petal_length)                           # Compute the variance using NumPy: variance_np
print(variance_explicit, variance_np)                                    # Print the results


print("****************************************************")
tema = '10. The standard deviation and the variance'; print("** %s\n" % tema)

variance = np.var(versicolor_petal_length)  # Compute the variance: variance
print(np.sqrt(variance))                    # Print the square root of the variance
print(np.std(versicolor_petal_length),'\n') # Print the standard deviation


print("****************************************************")
tema = '12. Scatter plots'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style

# Plot the ECDF
plt.figure()
_ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')
_ = plt.xlabel('Petal length (cm)')
_ = plt.ylabel('Petal width (cm)')
_ = plt.title('Iris Petal Length vs Width')
_ = plt.suptitle(tema)

# Show the plot
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '14. Computing the covariance'; print("** %s\n" % tema)

covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width) # Compute the covariance matrix: covariance_matrix
print(covariance_matrix, '\n')                                              # Print covariance matrix
petal_cov = covariance_matrix[0,1]                                          # Extract covariance of length and width of petals: petal_cov
print(petal_cov, '\n')                                                      # Print the length/width covariance


print("****************************************************")
tema = '15. Computing the Pearson correlation coefficient'; print("** %s\n" % tema)

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    corr_mat = np.corrcoef(x, y) # Compute correlation matrix: corr_mat
    return corr_mat[0,1]         # Return entry [0,1]

r = pearson_r(versicolor_petal_length, versicolor_petal_width) # Compute Pearson correlation coefficient for I. versicolor: r
print(r, '\n') # Print the result


print("****************************************************")
print("** END                                            **")
print("****************************************************")