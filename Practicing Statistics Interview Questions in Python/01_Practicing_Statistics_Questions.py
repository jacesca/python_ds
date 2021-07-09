# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:18:03 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 1: Probability and Sampling Distributions
    This chapter kicks the course off by reviewing conditional probabilities, Bayes' theorem, and central 
    limit theorem. Along the way, you will learn how to handle questions that work with commonly referenced 
    probability distributions.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot              as plt                                  #For creating charts
import numpy                          as np                                   #For making operations in lists
import seaborn                        as sns                                  #For visualizing data

from scipy.stats                     import bernoulli                         #Generate bernoulli data
from scipy.stats                     import binom                             #Generate binomial data
from scipy.stats                     import norm                              #Generate normal data

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 123
np.random.seed(SEED)

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data) # Number of data points: n
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n # y-data for the ECDF: y
    
    return x, y

print("****************************************************")
topic = "5. Samples from a rolled die"; print("** %s\n" % topic)

# Create a sample of 10 die rolls
# numpy.random.randint(low, high=None, size=None, dtype='l')
# Return random integers from low (inclusive) to high (exclusive).
small = np.random.randint(1, 7, 10)

# Calculate and print the mean of the sample
small_mean = small.mean()
print("The mean in a sample of 10 die rolls:", small_mean)

# Create a sample of 1000 die rolls
large = np.random.randint(1, 7, 1000)

# Calculate and print the mean of the large sample
large_mean = large.mean()
print("The mean in a sample of 1000 die rolls:", large_mean)

print("\nwhich theorem is at work here?")
print("LAW OF LARGE NUMBERS.")
print("The LAW OF LARGE NUMBERS states that as the size of a sample is increased, the estimate of the sample mean will be more accurately reflect the population mean.")

print("****************************************************")
topic = "6. Simulating central limit theorem"; print("** %s\n" % topic)

# Create a list of 1000 sample means of size 30
means1000 = [np.random.randint(1, 7, 30).mean() for i in range(1000)]

plt.figure()
# Create and show a histogram of the means
plt.subplot(2,1,1)
plt.hist(means1000)
plt.xlabel('Average getted', fontsize=8); plt.ylabel('Number of times', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("Means in a 1000 samples of 30 die rolls", color='red')

# Adapt code for 100 samples of size 30
means100 = [np.random.randint(1, 7, 30).mean() for i in range(100)]

plt.subplot(2,1,2)
# Create and show a histogram of the means
plt.hist(means100)
plt.xlabel('Average getted', fontsize=8); plt.ylabel('Number of times', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("Means in a 100 samples of 30 die rolls", color='red')

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5);
plt.show()

median_1000 = np.median(means1000)
mean_1000 = np.mean(means1000)
median_100 = np.median(means100)
mean_100 = np.mean(means100)

#Reoeting using seaborn library
sns.set_style('darkgrid')
plt.figure()
plt.subplot(2,1,1)
sns.distplot(means1000, bins=10)
plt.axvline(x=mean_1000, color='b', label='Mean', linestyle='-', linewidth=2)
plt.axvline(x=median_1000, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
plt.xlabel('Average getted', fontsize=8); plt.ylabel('Number of times', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.legend(loc='best', fontsize='small')
plt.title("Means in a 1000 samples of 30 die rolls (Seaborn)", color='red')
plt.subplot(2,1,2)
sns.distplot(means100, bins=10)
plt.axvline(x=mean_100, color='b', label='Mean', linestyle='-', linewidth=2)
plt.axvline(x=median_100, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
plt.xlabel('Average getted', fontsize=8); plt.ylabel('Number of times', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("Means in a 100 samples of 30 die rolls (Seaborn)", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5);
plt.legend(loc='best', fontsize='small')
plt.show()
plt.style.use('default')


print("****************************************************")
topic = "8. Bernoulli distribution"; print("** %s\n" % topic)

#rvs(p, loc=0, size=1, random_state=None)-->Random variates.
data = bernoulli.rvs(p=0.5, size=100)

# Plot distribution
plt.figure()
plt.subplot(2,1,1)
plt.hist(data)
plt.axhline(y=50, color='b', label='Mean', linestyle='--', linewidth=1)
plt.xlabel('Result obtained', fontsize=8); plt.ylabel('Number of times', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("Results in a 100 samples of flipping a fair coin", color='red')

# Generate bernoulli data
data = bernoulli.rvs(p=0.5, size=1000)

# Plot distribution
plt.subplot(2,1,2)
plt.hist(data)
plt.axhline(y=500, color='b', label='Mean', linestyle='--', linewidth=1)
plt.xlabel('Result obtained', fontsize=8); plt.ylabel('Number of times', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("Results in a 1000 samples of flipping a fair coin", color='red')

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "9. Binomial distribution"; print("** %s\n" % topic)

#For this exercise, consider a game where you are trying to make a ball in a basket. 
#You are given 10 shots and you know that you have an 80% chance of making a given shot. 
#To simplify things, assume each shot is an independent event.

data = binom.rvs(n=10, p=0.8, size=1000)
mu = data.mean()
sigma = data.std()
median = np.median(data)
theorical = np.random.normal(mu,sigma,100000)

# Plot the distribution
plt.figure()
plt.hist(data)
plt.xlabel('Success in 10 shots', fontsize=8); plt.ylabel('Number of times', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("Results in a 1000 samples of 10 BKB shots each (p = 0.80)", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.show()

# Assign and print probability of 8 or less successes
prob1 = binom.cdf(k=8, n=10, p=0.8)
print("Probability of 8 or less successes:", prob1)

# Assign and print probability of exactly 8 successes
prob2 = binom.pmf(k=8, n=10, p=0.8)
print("Probability of exactly 8 successes:", prob2)

# Assign and print probability of all 10 successes
prob3 = binom.pmf(k=10, n=10, p=0.8)
print("Probability of all 10 successes:", prob3)

# Assign and print probability of 10 or less successes
prob4 = binom.cdf(k=10, n=10, p=0.8)
print("Probability of 10 or less successes:", prob4)



#Plot the CDF and PDF of the samples.
plt.figure()
# Plot the PDF
plt.subplot(2,1,1)
plt.hist(data, density=True)
plt.xlabel('Success in 10 shots', fontsize=8); plt.ylabel('Number of times', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("PDF - Results in a 1000 samples of 10 BKB shots each", color='red')

# Plot the CDF
plt.subplot(2,1,2)
n, bins, patches = plt.hist(data, density=True, cumulative=True, label='Empirical')
plt.xlabel('Success in 10 shots', fontsize=8); plt.ylabel('Number of times', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("CDF - Results in a 1000 samples of 10 BKB shots each", color='red')

# Add a line showing the expected distribution.
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
y = y.cumsum()
y /= y[-1]
plt.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')
plt.legend(loc='best', fontsize='small')

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5);
plt.show()



#Repeting using seaborn library for showing CDF and PDF
sns.set_style('darkgrid')
plt.figure()
plt.subplot(2,1,1)
sns.distplot(data, kde=False, norm_hist=True, label='Empirical')
sns.distplot(theorical, color='black', hist=False, label='Theorical', hist_kws=dict(edgecolor='k', linewidth=1))
plt.axvline(x=mu, color='b', label='Mean', linestyle='-', linewidth=2)
plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
plt.xlabel('Slope', fontsize=8); plt.ylabel('PDF', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.legend(loc='best', fontsize='small')
plt.title("PDF - Results in a 1000 samples of 10 BKB shots each (Seaborn)", color='red')
plt.subplot(2,1,2)
sns.distplot(data, kde=False, hist_kws={"density":True, "cumulative":True}, label='Empirical')
# Add a line showing the expected distribution.
_, bins = np.histogram(data)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
y = y.cumsum()
y /= y[-1]
plt.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')
plt.legend(loc='best', fontsize='small')
plt.xlabel('Slope', fontsize=8); plt.ylabel('CDF', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("CDF - Results in a 1000 samples of 10 BKB shots each (Seaborn)", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5);
plt.show()
plt.style.use('default')



#Repeting one more time for showing CDF and PDF (Graphing a PMF correctly)
plt.figure()
sns.set_style('darkgrid')
#PDF
plt.subplot(2,1,1)
y = np.bincount(data)/len(data) # Get the frequency
x = np.nonzero(y)[0]
y = y[y!=0]
plt.plot(x, y, 'ko', ms=6, mec='k')
plt.vlines(x, 0, y, colors='k', linestyles='-', lw=2)
plt.axvline(x=mu, color='b', label='Mean', linestyle='-', linewidth=1)
plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=1) # Add vertical lines for the median and mean
plt.xlabel('Number of success in 10 BKB shots', fontsize=8); plt.ylabel('Probability (PMF)', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.legend(loc='best', fontsize='small')
plt.title("Probability Mass Function (PMF)", color='red')
#CDF
plt.subplot(2,1,2)
x, y = ecdf(data) #Ex. x=[2,2,3,3,3,4,5], y=[0.14, 0.28, 0.43, 0.57, 0.71, 0.86, 1]
u, i = np.unique(x, return_index=True) #Ex. i=[0,2,5,6] -->Return the indices of the original array that give the unique values
i = i-1 #Ex. i=['-1,1,4,5] -->To find the index of change.
i = i[i>=0] #Ex. i=[1,4,5] -->To delete negative inexistent indices.
i = np.append(i, len(x)-1) #Ex. i=[1,4,5,6] -->To add the last element index
plt.plot(x[i], y[i], 'ko', ms=6, mec='k')
plt.vlines(x[i], 0, y[i], colors='k', linestyles='-', lw=2)
plt.axvline(x=mu, color='b', label='Mean', linestyle='-', linewidth=1)
plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=1) # Add vertical lines for the median and mean
plt.xlabel('Number of success in 10 BKB shots', fontsize=8); plt.ylabel('Probability (CDF)', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.legend(loc='best', fontsize='small')
plt.title("Cumulative Distribution function (CDF)", color='red')
# Add a line showing the expected distribution.
_, bins = np.histogram(data)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
y = y.cumsum()
y /= y[-1]
plt.plot(bins, y, 'k--', linewidth=1.5)

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5);
plt.show()
plt.style.use('default')



print("****************************************************")
topic = "10. Normal distribution"; print("** %s\n" % topic)

data = norm.rvs(size=1000)
mu = data.mean()
sigma = data.std()
median = np.median(data)
theorical = np.random.normal(mu,sigma,100000)

# Compute and print true probability for greater than 2
true_prob = 1 - norm.cdf(2)
print("Given a standardized normal distribution, what is the probability of an observation greater than 2?", true_prob)

# Compute and print sample probability for greater than 2
sample_prob = sum(obs > 2 for obs in data) / len(data)
print("Looking at our sample, what is the probability of an observation greater than 2?", sample_prob)

#Using seaborn library for showing CDF and PDF
sns.set_style('darkgrid')
plt.figure()
plt.subplot(2,1,1)
sns.distplot(data, kde=False, norm_hist=True, label='Empirical')
sns.distplot(theorical, color='black', hist=False, label='Theorical', hist_kws=dict(edgecolor='k', linewidth=1))
plt.axvline(x=mu, color='b', label='Mean', linestyle='-', linewidth=2)
plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
plt.xlabel('Samples', fontsize=8); plt.ylabel('Probability (PDF)', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.legend(loc='best', fontsize='small')
plt.title("Probability Density Function (PDF)", color='red')
plt.subplot(2,1,2)
sns.distplot(data, kde=False, hist_kws={"density":True, "cumulative":True}, label='Empirical')
# Add a line showing the expected distribution.
_, bins = np.histogram(data)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
y = y.cumsum()
y /= y[-1]
plt.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')
plt.legend(loc='best', fontsize='small')
plt.xlabel('Samples', fontsize=8); plt.ylabel('Probability (CDF)', fontsize=8); # Labeling the axis.
plt.xticks(fontsize=8); plt.yticks(fontsize=8);
plt.title("Cumulative Distribution function (CDF)", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5);
plt.show()
plt.style.use('default')

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import contextily                                                             #To add a background web map to our plot
#import inspect                                                                #Used to get the code inside a function
#import folium                                                                 #To create map street folium.__version__'0.10.0'
#import geopandas         as gpd                                               #For working with geospatial data 
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.dates  as mdates                                            #For providing sophisticated date plotting capabilities
#import matplotlib.pyplot as plt                                               #For creating charts
#import numpy             as np                                                #For making operations in lists
#import pandas            as pd                                                #For loading tabular data
#import seaborn           as sns                                               #For visualizing data
#import pprint                                                                 #Import pprint to format disctionary output
#import missingno         as msno                                              #Missing data visualization module for Python

#import os                                                                     #To raise an html page in python command
#import tempfile                                                               #To raise an html page in python command
#import webbrowser                                                             #To raise an html page in python command  

#import calendar                                                               #For accesing to a vary of calendar operations
#import datetime                                                               #For accesing datetime functions
#import math                                                                   #For accesing to a complex math operations
#import nltk                                                                   #For working with text data
#import random                                                                 #For generating random numbers
#import re                                                                     #For regular expressions
#import tabula                                                                 #For extracting tables from pdf
#import timeit                                                                 #For Measure execution time of small code snippets
#import time                                                                   #To measure the elapsed wall-clock time between two points
#import scykit-learn                                                           #For performing machine learning  
#import warnings
#import wikipedia

#from collections                     import defaultdict                       #Returns a new dictionary-like object
#from datetime                        import date                              #For obteining today function
#from datetime                        import datetime                          #For obteining today function
#from functools                       import reduce                            #For accessing to a high order functions (functions or operators that return functions)
#from glob                            import glob                              #For using with pathnames matching
#from itertools                       import combinations                      #For iterations
#from itertools                       import cycle                             #Used in the function plot_labeled_decision_regions()
#from math                            import ceil                              #Used in the function plot_labeled_decision_regions()
#from math                            import floor                             #Used in the function plot_labeled_decision_regions()
#from math                            import radian                            #For accessing a specific math operations
#from math                            import sqrt
#from matplotlib                      import colors                            #To create custom cmap
#from matplotlib.ticker               import StrMethodFormatter                #Import the necessary library to delete the scientist notation
#from mpl_toolkits.mplot3d            import Axes3D
#from numpy.random                    import randint                           #numpy.random.randint(low, high=None, size=None, dtype='l')-->Return random integers from low (inclusive) to high (exclusive).  
#from pandas.api.types                import CategoricalDtype                  #For categorical data
#from pandas.plotting                 import parallel_coordinates              #For Parallel Coordinates
#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
#from shapely.geometry                import LineString                        #(Geospatial) To create a Linestring geometry column 
#from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 
#from shapely.geometry                import Polygon                           #(Geospatial) To create a point geometry column 
#from string                          import Template                          #For working with string, regular expressions


#from bokeh.io                        import curdoc                            #For interacting visualizations
#from bokeh.io                        import output_file                       #For interacting visualizations
#from bokeh.io                        import show                              #For interacting visualizations
#from bokeh.plotting                  import ColumnDataSource                  #For interacting visualizations
#from bokeh.plotting                  import figure                            #For interacting visualizations
#from bokeh.layouts                   import column                            #For interacting visualizations
#from bokeh.layouts                   import gridplot                          #For interacting visualizations
#from bokeh.layouts                   import row                               #For interacting visualizations
#from bokeh.layouts                   import widgetbox                         #For interacting visualizations
#from bokeh.models                    import Button                            #For interacting visualizations
#from bokeh.models                    import CategoricalColorMapper            #For interacting visualizations
#from bokeh.models                    import CheckboxGroup                     #For interacting visualizations
#from bokeh.models                    import ColumnDataSource                  #For interacting visualizations
#from bokeh.models                    import HoverTool                         #For interacting visualizations
#from bokeh.models                    import RadioGroup                        #For interacting visualizations
#from bokeh.models                    import Select                            #For interacting visualizations
#from bokeh.models                    import Slider                            #For interacting visualizations
#from bokeh.models                    import Toggle                            #For interacting visualizations
#from bokeh.models.widgets            import Panel                             #For interacting visualizations
#from bokeh.models.widgets            import Tabs                              #For interacting visualizations
#from bokeh.palettes                  import Spectral6                         #For interacting visualizations


#import keras                                                                  #For DeapLearning
#import keras.backend as k                                                     #For DeapLearning
#from keras.applications.resnet50     import decode_predictions                #For DeapLearning
#from keras.applications.resnet50     import preprocess_input                  #For DeapLearning
#from keras.applications.resnet50     import ResNet50                          #For DeapLearning
#from keras.callbacks                 import EarlyStopping                     #For DeapLearning
#from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
#from keras.datasets                  import fashion_mnist                     #For DeapLearning
#from keras.datasets                  import mnist                             #For DeapLearning
#from keras.layers                    import BatchNormalization                #For DeapLearning
#from keras.layers                    import Concatenate                       #For DeapLearning
#from keras.layers                    import Conv2D                            #For DeapLearning
#from keras.layers                    import Dense                             #For DeapLearning
#from keras.layers                    import Dropout                           #For DeapLearning
#from keras.layers                    import Embedding                         #For DeapLearning
#from keras.layers                    import Flatten                           #For DeapLearning
#from keras.layers                    import GlobalMaxPooling1D                #For DeapLearning
#from keras.layers                    import Input                             #For DeapLearning
#from keras.layers                    import LSTM                              #For DeapLearning
#from keras.layers                    import MaxPool2D                         #For DeapLearning
#from keras.layers                    import SpatialDropout1D                  #For DeapLearning
#from keras.layers                    import Subtract                          #For DeapLearning
#from keras.models                    import load_model                        #For DeapLearning
#from keras.models                    import Model                             #For DeapLearning
#from keras.models                    import Sequential                        #For DeapLearning
#from keras.optimizers                import Adam                              #For DeapLearning
#from keras.optimizers                import SGD                               #For DeapLearning
#from keras.preprocessing             import image                             #For DeapLearning
#from keras.preprocessing.text        import Tokenizer                         #For DeapLearning
#from keras.preprocessing.sequence    import pad_sequences                     #For DeapLearning
#from keras.utils                     import plot_model                        #For DeapLearning
#from keras.utils                     import to_categorical                    #For DeapLearning
#from keras.wrappers.scikit_learn     import KerasClassifier                   #For DeapLearning


#import networkx          as nx                                                #For Network Analysis in Python
#import nxviz             as nv                                                #For Network Analysis in Python
#from nxviz                           import ArcPlot                           #For Network Analysis in Python
#from nxviz                           import CircosPlot                        #For Network Analysis in Python 
#from nxviz                           import MatrixPlot                        #For Network Analysis in Python 


#import scipy.stats as stats                                                   #For accesign to a vary of statistics functiosn
#from scipy.cluster.hierarchy         import dendrogram                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import fcluster                          #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import linkage                           #For learning machine - unsurpervised
#from scipy.ndimage                   import gaussian_filter                   #For working with images
#from scipy.ndimage                   import median_filter                     #For working with images
#from scipy.signal                    import convolve2d                        #For learning machine - deep learning
#from scipy.sparse                    import csr_matrix                        #For learning machine 
#from scipy.special                   import expit as sigmoid                  #For learning machine 
#from scipy.stats                     import anderson                          #For Anderson-Darling Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import bernoulli                         #Generate bernoulli data
#from scipy.stats                     import binom                             #Generate binomial data
#from scipy.stats                     import chi2_contingency                  #For Chi-Squared Test. Tests whether two categorical variables are related or independent
#from scipy.stats                     import f_oneway                          #For Analysis of Variance Test. Tests whether the means of two or more independent samples are significantly different.
#from scipy.stats                     import friedmanchisquare                 #For Friedman Test. Tests whether the distributions of two or more paired samples are equal or not.
#from scipy.stats                     import kendalltau                        #For Kendall's Rank Correlation Test. To check if two samples are related.
#from scipy.stats                     import kruskal                           #For Kruskal-Wallis H Test. Tests whether the distributions of two or more independent samples are equal or not.
#from scipy.stats                     import mannwhitneyu                      #For Mann-Whitney U Test. Tests whether the distributions of two independent samples are equal or not.
#from scipy.stats                     import norm                              #Generate normal data
#from scipy.stats                     import normaltest                        #For D'Agostino's K^2 Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import pearsonr                          #For learning machine. For Pearson's Correlation test. To check if two samples are related.
#from scipy.stats                     import randint                           #For learning machine 
#from scipy.stats                     import shapiro                           #For Shapiro-Wilk Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import spearmanr                         #For Spearman's Rank Correlation Test.  To check if two samples are related.
#from scipy.stats                     import ttest_ind                         #For Student's t-test. Tests whether the means of two independent samples are significantly different.
#from scipy.stats                     import ttest_rel                         #For Paired Student's t-test. Tests whether the means of two paired samples are significantly different.
#from scipy.stats                     import wilcoxon                          #For Wilcoxon Signed-Rank Test. Tests whether the distributions of two paired samples are equal or not.


#from skimage                         import exposure                          #For working with images
#from skimage                         import measure                           #For working with images
#from skimage.filters.thresholding    import threshold_otsu                    #For working with images
#from skimage.filters.thresholding    import threshold_local                   #For working with images 


#from sklearn                         import datasets                          #For learning machine
#from sklearn.cluster                 import KMeans                            #For learning machine - unsurpervised
#from sklearn.decomposition           import NMF                               #For learning machine - unsurpervised
#from sklearn.decomposition           import PCA                               #For learning machine - unsurpervised
#from sklearn.decomposition           import TruncatedSVD                      #For learning machine - unsurpervised
#from sklearn.ensemble                import AdaBoostClassifier                #For learning machine - surpervised
#from sklearn.ensemble                import BaggingClassifier                 #For learning machine - surpervised
#from sklearn.ensemble                import GradientBoostingRegressor         #For learning machine - surpervised
#from sklearn.ensemble                import RandomForestClassifier            #For learning machine
#from sklearn.ensemble                import RandomForestRegressor             #For learning machine - unsurpervised
#from sklearn.ensemble                import VotingClassifier                  #For learning machine - unsurpervised
#from sklearn.feature_selection       import chi2                              #For learning machine
#from sklearn.feature_selection       import SelectKBest                       #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                   #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                 #For learning machine
#from sklearn.feature_extraction.text import TfidfVectorizer                   #For learning machine - unsurpervised
#from sklearn.impute                  import SimpleImputer                     #For learning machine
#from sklearn.linear_model            import ElasticNet                        #For learning machine
#from sklearn.linear_model            import Lasso                             #For learning machine
#from sklearn.linear_model            import LinearRegression                  #For learning machine
#from sklearn.linear_model            import LogisticRegression                #For learning machine
#from sklearn.linear_model            import Ridge                             #For learning machine
#from sklearn.manifold                import TSNE                              #For learning machine - unsurpervised
#from sklearn.metrics                 import accuracy_score                    #For learning machine
#from sklearn.metrics                 import classification_report             #For learning machine
#from sklearn.metrics                 import confusion_matrix                  #For learning machine
#from sklearn.metrics                 import mean_squared_error as MSE         #For learning machine
#from sklearn.metrics                 import roc_auc_score                     #For learning machine
#from sklearn.metrics                 import roc_curve                         #For learning machine
#from sklearn.model_selection         import cross_val_score                   #For learning machine
#from sklearn.model_selection         import GridSearchCV                      #For learning machine
#from sklearn.model_selection         import KFold                             #For learning machine
#from sklearn.model_selection         import RandomizedSearchCV                #For learning machine
#from sklearn.model_selection         import train_test_split                  #For learning machine
#from sklearn.multiclass              import OneVsRestClassifier               #For learning machine
#from sklearn.neighbors               import KNeighborsClassifier as KNN       #For learning machine
#from sklearn.pipeline                import FeatureUnion                      #For learning machine
#from sklearn.pipeline                import make_pipeline                     #For learning machine - unsurpervised
#from sklearn.pipeline                import Pipeline                          #For learning machine
#from sklearn.preprocessing           import FunctionTransformer               #For learning machine
#from sklearn.preprocessing           import Imputer                           #For learning machine
#from sklearn.preprocessing           import MaxAbsScaler                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing           import Normalizer                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing           import normalize                         #For learning machine - unsurpervised
#from sklearn.preprocessing           import scale                             #For learning machine
#from sklearn.preprocessing           import StandardScaler                    #For learning machine
#from sklearn.svm                     import SVC                               #For learning machine
#from sklearn.tree                    import DecisionTreeClassifier            #For learning machine - supervised
#from sklearn.tree                    import DecisionTreeRegressor             #For learning machine - supervised


#import statsmodels                   as sm                                    #For stimations in differents statistical models
#import statsmodels.api               as sm                                    #Make a prediction model
#import statsmodels.formula.api       as smf                                   #Make a prediction model    
#from statsmodels.graphics.tsaplots   import plot_acf                          #For autocorrelation function
#from statsmodels.graphics.tsaplots   import plot_pacf                         #For simulating data and for plotting the PACF. Partial Autocorrelation Function measures the incremental benefit of adding another lag.
#from statsmodels.tsa.arima_model     import ARIMA                             #Similar to use ARMA but on original data (before differencing)
#from statsmodels.tsa.arima_model     import ARMA                              #To estimate parameters from data simulated (AR model)
#from statsmodels.tsa.arima_process   import ArmaProcess                       #For simulating data and for plotting the PACF, For Simulate Autoregressive (AR) Time Series 
#from statsmodels.tsa.stattools       import acf                               #For autocorrelation function
#from statsmodels.tsa.stattools       import adfuller                          #For Augmented Dickey-Fuller unit root test. Test for Random Walk. Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive. Tests that you can use to check if a time series is stationary or not.
#from statsmodels.tsa.stattools       import coint                             #Test for cointegration
#from statsmodels.tsa.stattools       import kpss                              #For Kwiatkowski-Phillips-Schmidt-Shin test. Tests whether a time series is trend stationary or not.


#import tensorflow              as tf                                          #For DeapLearning



# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#pd.set_option('display.max_rows', -1)                                         #Shows all rows

#register_matplotlib_converters()                                              #Require to explicitly register matplotlib converters.

#Setting images params
#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09
#plt.rcParams.update({'figure.max_open_warning': 0})                           #To solve the max images open
#plt.rcParams["axes.labelsize"] = 8                                            #Font
#plt.rc('xtick',labelsize=8)
#plt.rc('ytick',labelsize=6)
#plt.rcParams['figure.max_open_warning'] = 60                                  #params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 5), 'axes.labelsize': 'x-large', 'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
#plt.rcParams["legend.fontsize"] = 8
#plt.style.use('dark_background')
#plt.style.use('default')
#plt.xticks(fontsize=7); plt.yticks(fontsize=8);
#plt.xticks(rotation=45)
#ax.xaxis.set_major_locator(plt.MaxNLocator(3))                                #Define the number of ticks to show on x axis.
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())                          #Other way to Define the number of ticks to show on x axis.
#ax.xaxis.set_major_formatter(mdates.AutoDateFormatter())                      #Other way to Define the number of ticks to show on x axis.


#plt.xticks(rotation=45)                                                       #rotate x-axis labels by 45 degrees
#plt.yticks(rotation=90)                                                       #rotate y-axis labels by 90 degrees
#plt.savefig("sample.jpg")                                                     #save image of `plt`

#To supress the scientist notation in plt
#from matplotlib.ticker import StrMethodFormatter                              #Import the necessary library to delete the scientist notation
#ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation

#from matplotlib.axes._axes import _log as matplotlib_axes_logger              #To avoid warnings
#matplotlib_axes_logger.setLevel('ERROR')
#matplotlib_axes_logger.setLevel(0)                                            #To restore default

#ax.tick_params(labelsize=6)                                                   #axis : {'x', 'y', 'both'}
#ax.tick_params(axis='x', rotation=45)                                         #Set rotation atributte

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.
#np.random.seed(SEED)

#tf.compat.v1.set_random_seed(SEED)                                            #Instead of tf.set_random_seed, because it is deprecated.

#sns.set(font_scale=0.8)                                                       #Font
#sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot
#sns.set(color_codes=True)                                                     #Habilita el uso de los codigos de color
#sns.set()                                                                     #Seaborn defult style
#sns.set_style(this_style)                                                     #['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
#sns.despine(left=True)                                                        #Remove the spines (all borders)
#sns.palettes.SEABORN_PALETTES                                                 #Despliega todas las paletas disponibles 
#sns.palplot(sns.color_palette())                                              #Display a palette
#sns.color_palette()                                                           #The current palette
#sns.set(style=”whitegrid”, palette=”pastel”, color_codes=True)
#sns.mpl.rc(“figure”, figsize=(10,6))
#sns.distplot(data, bins=10, hist_kws={"density": True})                       #Example of hist_kws parameter 
#sns.distplot(data, hist=False, hist_kws=dict(edgecolor='k', linewidth=1))     #Example of hist_kws parameter 


#warnings.filterwarnings('ignore', 'Objective did not converge*')              #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
#warnings.filterwarnings('default', 'Objective did not converge*')             #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


#Create categorical type data to use
#cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  ordered=True)
# Change the data type of 'rating' to category
#weather['rating'] = weather.rating.astype(cats)


#print("The area of your rectangle is {}cm\u00b2".format(area))                 #Print the superscript 2

### Show a basic html page
#tmp=tempfile.NamedTemporaryFile()
#path=tmp.name+'.html'
#f=open(path, 'w')
#f.write("<html><body><h1>Test</h1></body></html>")
#f.close()
#webbrowser.open('file://' + path)
