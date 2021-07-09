# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:40:32 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 3: Important probability distributions
    Until now we've been working with binomial distributions, but there are many probability 
    distributions a random variable can take. In this chapter we'll introduce three more that 
    are related to the binomial distribution: the normal, Poisson, and geometric distributions.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import numpy                         as np                                    #For making operations in lists
import matplotlib.pyplot             as plt                                   #For creating charts
import seaborn                       as sns                                   #For visualizing data

from scipy.stats                     import poisson                           #To generate poisson distribution.

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 13
np.random.seed(SEED) 

def poisson_distplot(mu, SEED, size=10000, width=.7): 
    """Make a poisson graph. Parameter needed:
        mu   -->Poisson parameter.
        SEED -->Random seed.
        size -->10,000 by default. Size of the sample to plot.
        width-->0.7 by default. Width of the bar in the graph
    Return sample of the poisson distribution with the mu defined."""
    sample = poisson.rvs(mu=mu, size=size, random_state=SEED)
    mean   = np.mean(sample)
    median = np.median(sample)
    x      = np.unique(sample)
    
    #Plot the sample
    sns.distplot(sample, kde=False, hist_kws={"width":width})
    plt.xticks(x)
    plt.xlabel('k (Sample)'); plt.ylabel('Frequency (mu={})'.format(mu)); # Labeling the axis.
    plt.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)
    plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
    plt.legend(loc='best', fontsize='small')
    return sample


def poisson_plot(mu, SEED, size=10000): #, sample=np.array([])): 
    """Make a poisson graph. Parameter needed:
        mu    -->Poisson parameter.
        SEED  -->Random seed.
        size  -->10,000 by default. Size of the sample to plot.
        sample-->If not given, the function will generete it.
    Return sample of the poisson distribution with the mu defined and its pmf."""
    #sample   = poisson.rvs(mu=mu, size=size, random_state=SEED) if sample.size == 0 else sample
    sample   = poisson.rvs(mu=mu, size=size, random_state=SEED)
    mean     = np.mean(sample)
    median   = np.median(sample)
    #x, freq = np.unique(sample, return_counts=True)
    x        = np.unique(sample)
    y        = poisson.pmf(x, mu)
    
    #Plot the sample
    plt.plot(x, y, 'ko', ms=8)
    plt.vlines(x=x, ymin=0, ymax=y, colors='k', linestyles='-', lw=3)
    plt.xticks(x)
    plt.xlabel('k (Sample)'); plt.ylabel('poisson.pmf(k, mu={})'.format(mu)); # Labeling the axis.
    plt.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)
    plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
    plt.legend(loc='best', fontsize='small')
    return x, y


def poisson_bar(mu, SEED, size=10000, y_text=0.001, text_percent=True): 
    """Make a poisson graph. Parameter needed:
        mu    -->Poisson parameter.
        SEED  -->Random seed.
        size  -->10,000 by default. Size of the sample to plot.
        sample-->If not given, the function will generete it.
        y     -->pmf of the sample.
        y_text-->the height add to y for printing the pmf in the plot.
    Return sample of the poisson distribution with the mu defined and its pmf."""
    sample = poisson.rvs(mu=mu, size=size, random_state=SEED) #if sample.size == 0 else sample
    mean     = np.mean(sample)
    median   = np.median(sample)
    #x, freq = np.unique(sample, return_counts=True)
    x        = np.unique(sample)
    y        = poisson.pmf(x, mu)
    
    #Plot the sample
    plt.bar(x, y)
    plt.xticks(x)
    plt.xlabel('k (Sample)'); plt.ylabel('poisson.pmf(k, mu={})'.format(mu)); # Labeling the axis.
    plt.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)
    plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
    plt.legend(loc='best', fontsize='small')
    if text_percent:
        for value, percent in zip(x, y):
            plt.text(value, percent+y_text, "{:,.2%}".format(percent), fontsize=8, ha='center')
    return x, y




print("****************************************************")
topic = "9. Poisson distributions"; print("** %s\n" % topic)

#Plot the sample
plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
mu=0.8
poisson_bar(mu=mu, SEED=SEED, text_percent=False)
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title("mu={:,.2f}".format(mu), color='red')
plt.subplot(2,2,2)
mu=1.8
poisson_bar(mu=mu, SEED=SEED, text_percent=False)
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title("mu={:,.2f}".format(mu), color='red')
plt.subplot(2,2,3)
mu=3.0
poisson_bar(mu=mu, SEED=SEED, text_percent=False)
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title("mu={:,.2f}".format(mu), color='red')
plt.subplot(2,2,4)
mu=6.0
poisson_bar(mu=mu, SEED=SEED, text_percent=False)
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title("mu={:,.2f}".format(mu), color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=.15, bottom=None, right=None, top=None, wspace=.5, hspace=.5);
plt.show()

#Sample generation
mu = 2.2 #Average number of calls per minute

#Plot the sample, first choice
plt.figure()
sample = poisson_distplot(mu=mu, SEED=SEED)
plt.title("Calls per minute (using displot)", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=.15, bottom=None, right=None, top=.85, wspace=.5, hspace=None);
plt.show()

#Plot the sample, second choice
plt.figure()
sample, y = poisson_plot(mu, SEED)
plt.title("Calls per minute (using plot)", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=.15, bottom=None, right=None, top=.85, wspace=.5, hspace=None);
plt.show()

#Plot the sample, third choice
plt.figure()
poisson_bar(mu, SEED)
plt.title("Calls per minute (using bar)", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=.15, bottom=None, right=None, top=.85, wspace=.5, hspace=None);
plt.show()


print("What is the probability of having 3 calls in a minute?")
# Calculate the probability mass
probability = poisson.pmf(k=3, mu=mu)
print("{:,.2%}".format(probability))

print("\nWhat is the probability of having no calls in a minute?")
# Calculate the probability mass
probability = poisson.pmf(k=0, mu=mu)
print("{:,.2%}".format(probability))

print("\nWhat is the probability of having 6 calls in a minute?")
# Calculate the probability mass
probability = poisson.pmf(k=6, mu=mu)
print("{:,.2%}".format(probability))

print("\nWhat is the probability of having 2 or less calls in a minute?")
# Calculate the probability mass
probability = poisson.cdf(k=2, mu=mu)
print("{:,.2%}".format(probability))

print("\nWhat is the probability of having 5 or less calls in a minute?")
# Calculate the probability mass
probability = poisson.cdf(k=5, mu=mu)
print("{:,.2%}".format(probability))

print("\nWhat is the probability of having more than 2 calls in a minute?")
# Calculate the probability mass
probability = poisson.sf(k=2, mu=mu)
print("{:,.2%}".format(probability))

print("\nHow many calls accumulates 50% of probability?")
# Calculate the probability mass
probability = poisson.ppf(q=.5, mu=mu)
print("{:,.2f}".format(probability))

print("****************************************************")
topic = "10. ATM example"; print("** %s\n" % topic)

mu = 1 #Average number of calls per minute

#Plot the sample
plt.figure()
poisson_bar(mu, SEED)
plt.title("Average number of customers at ATM per minute", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=.15, bottom=None, right=None, top=.85, wspace=.5, hspace=None);
plt.show()

print("Calculate the probability of having more than one customer visiting the ATM in this 5-minute period.")
# Probability of more than 1 customer
probability = poisson.sf(k=1, mu=mu)
# Print the result
print("{:,.2%}".format(probability))


print("****************************************************")
topic = "11. Highway accidents example"; print("** %s\n" % topic)

mu = 2 #Average number of accidents per day

#Plot the sample
plt.figure()
poisson_bar(mu, SEED)
plt.title("Number of accidents per day", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=.15, bottom=None, right=None, top=.85, wspace=.5, hspace=None);
plt.show()


print("Determine and print the probability of there being 5 accidents on any day.")
P_five_accidents = poisson.pmf(k=5, mu=mu)   # Probability of 5 accidents any day
print("{:,.2%}\n".format(P_five_accidents)) # Print the result

print("Determine and print the probability of having 4 or 5 accidents on any day.")
P_four_accident = poisson.pmf(k=4, mu=mu)
P_five_accidents = poisson.pmf(k=5, mu=mu)
print("{:,.2%}\n".format(P_four_accident + P_five_accidents))

print("Determine and print the probability of having more than 3 accidents on any day.")
P_more_than_3 = poisson.sf(k=3, mu=mu) # Probability of more than 3 accidents any day
print("{:,.2%}\n".format(P_more_than_3))

print("Determine and print the number of accidents that is likely to happen with 0.75 probability.")
accidents = poisson.ppf(q=.75, mu=mu) # Number of accidents with 0.75 probability
print("{:,.1f} accidentes per day\n".format(accidents))


print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import contextily                                                             #To add a background web map to our plot
#import inspect                                                                #Used to get the code inside a function
#import itertools                                                              #For iterations
#import folium                                                                 #To create map street folium.__version__'0.10.0'
#import geopandas                     as gpd                                   #For working with geospatial data 
#import math                                                                   #https://docs.python.org/3/library/math.html
#import matplotlib                    as mpl                                   #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.dates              as mdates                                #For providing sophisticated date plotting capabilities
#import matplotlib.pyplot             as plt                                   #For creating charts
#import numpy                         as np                                    #For making operations in lists
#import pandas                        as pd                                    #For loading tabular data
#import seaborn                       as sns                                   #For visualizing data
#import pprint                                                                 #Import pprint to format disctionary output
#import missingno                     as msno                                  #Missing data visualization module for Python

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
#from scipy.stats                     import describe                          #To get the arithmetic statistics.
#from scipy.stats                     import f_oneway                          #For Analysis of Variance Test. Tests whether the means of two or more independent samples are significantly different.
#from scipy.stats                     import find_repeats                      #To find repeated data in a sample. Statistical terms
#from scipy.stats                     import friedmanchisquare                 #For Friedman Test. Tests whether the distributions of two or more paired samples are equal or not.
#from scipy.stats                     import kendalltau                        #For Kendall's Rank Correlation Test. To check if two samples are related.
#from scipy.stats                     import kruskal                           #For Kruskal-Wallis H Test. Tests whether the distributions of two or more independent samples are equal or not.
#from scipy.stats                     import mannwhitneyu                      #For Mann-Whitney U Test. Tests whether the distributions of two independent samples are equal or not.
#from scipy.stats                     import norm                              #Generate normal data
#from scipy.stats                     import normaltest                        #For D'Agostino's K^2 Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import pearsonr                          #For learning machine. For Pearson's Correlation test. To check if two samples are related.
#from scipy.stats                     import poisson                           #To generate poisson distribution.
#from scipy.stats                     import randint                           #For learning machine 
#from scipy.stats                     import relfreq                           #To calculate the relative frequency of each outcome
#from scipy.stats                     import sem                               #For statistic thinking 
#from scipy.stats                     import shapiro                           #For Shapiro-Wilk Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import spearmanr                         #For Spearman's Rank Correlation Test.  To check if two samples are related.
#from scipy.stats                     import t                                 #For statistic thinking 
#from scipy.stats                     import ttest_ind                         #For Student's t-test. Tests whether the means of two independent samples are significantly different.
#from scipy.stats                     import ttest_rel                         #For Paired Student's t-test. Tests whether the means of two paired samples are significantly different.
#from scipy.stats                     import wilcoxon                          #For Wilcoxon Signed-Rank Test. Tests whether the distributions of two paired samples are equal or not.


#from skimage                         import exposure                          #For working with images
#from skimage                         import measure                           #For working with images
#from skimage.filters.thresholding    import threshold_otsu                    #For working with images
#from skimage.filters.thresholding    import threshold_local                   #For working with images 


#from sklearn                         import datasets                          #For learning machine
#from sklearn                         import preprocessing
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
#from sklearn.metrics                 import mean_absolute_error as MAE        #For learning machine
#from sklearn.metrics                 import mean_squared_error as MSE         #For learning machine
#from sklearn.metrics                 import precision_score                   #Compute the precision of the model. Precision is the number of true positives over the number of true positives plus false positives and is linked to the rate of the type 1 error.
#from sklearn.metrics                 import recall_score                      #Compute the recall of the model. Recall is the number of true positives over the number of true positives plus false negatives and is linked to the rate of type 2 error.
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
#from sklearn.preprocessing           import LabelEncoder                      #Create the encoder and print our encoded new_vals
#from sklearn.preprocessing           import MaxAbsScaler                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing           import MinMaxScaler                      #Used for normalize data in a dataframe
#from sklearn.preprocessing           import Normalizer                        #For dataframe. For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing           import normalize                         #For arrays. For learning machine - unsurpervised
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
#from statsmodels.sandbox.stats.multicomp import multipletests                 #To adjust the p-value when you run multiple tests.
#from statsmodels.stats.power         import TTestIndPower                     #Explain how the effect, power and significance level affect the sample size. Create results object for t-test analysis
#from statsmodels.stats.power         import zt_ind_solve_power                #To determinate sample size. Assign and print the needed sample size
#from statsmodels.stats.proportion    import proportion_confint                #Fon confidence interval-->proportion_confint(number of successes, number of trials, alpha value represented by 1 minus our confidence level)
#from statsmodels.stats.proportion    import proportion_effectsize             #To determinate sample size. Standardize the effect size
#from statsmodels.stats.proportion    import proportions_ztest                 #To run the Z-score test, when you know the population standard deviation
#from statsmodels.tsa.arima_model     import ARIMA                             #Similar to use ARMA but on original data (before differencing)
#from statsmodels.tsa.arima_model     import ARMA                              #To estimate parameters from data simulated (AR model)
#from statsmodels.tsa.arima_process   import ArmaProcess                       #For simulating data and for plotting the PACF, For Simulate Autoregressive (AR) Time Series 
#from statsmodels.tsa.stattools       import acf                               #For autocorrelation function
#from statsmodels.tsa.stattools       import adfuller                          #For Augmented Dickey-Fuller unit root test. Test for Random Walk. Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive. Tests that you can use to check if a time series is stationary or not.
#from statsmodels.tsa.stattools       import coint                             #Test for cointegration
#from statsmodels.tsa.stattools       import kpss                              #For Kwiatkowski-Phillips-Schmidt-Shin test. Tests whether a time series is trend stationary or not.

#import tensorflow              as tf                                          #For DeapLearning


###When to use z-score or t-tests --> Usually you use a t-test when you do not know the population standard 
###                                   deviation σ, and you use the standard error instead. You usually use the 
###                                   z-test when you do know the population standard deviation. Although it is 
###                                   true that the central limit theorem kicks in at around n=30. I think that 
###                                   formally, the convergence in distribution of a sequence of t′s to a normal 
###                                   is pretty good when n>30.
#############################################
##Finding z-crtical values 95% one side    ##
##100*(1-alpha)% Confidence Level          ##
#############################################
#import scipy.stats as sp
#alpha=0.05                                                                    
#sp.norm.ppf(1-alpha, loc=0, scale=1)                                          # One-sided; ppf=percent point function
##Out --> 1.6448536269514722
#############################################
##Finding z-crtical values 95% two sides   ##
## 100*(1-alpha)% Confidence Level         ##
#############################################
#import scipy.stats as sp
#alpha=0.05
#sp.norm.ppf(1-alpha/2, loc=0, scale=1)                                        # Two-sided
##Out --> 1.959963984540054
#############################################
##Finding t-crtical values 95% one side    ##
# 100*(1-alpha)% Confidence Level          ##
#############################################
#import scipy.stats as sp
#alpha=0.05
#sp.t.ppf(1-alpha, df=4)                                                       # One-sided; df=degrees of fredom
##Out --> 2.13184678133629
#############################################
##Finding t-crtical values 95% two sides   ##
# 100*(1-alpha)% Confidence Level          ##
#############################################
#import scipy.stats as sp
#alpha=0.05
#sp.t.ppf(1-alpha/2, df=4) # Two-sided; df=degrees of fredom
##Out --> 2.7764451051977987
##Source --> http://lecture.riazulislam.com/uploads/3/9/8/5/3985970/python_practice_3_2019_1.pdf



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
#plt.xlabel('Average getted', fontsize=8); 
#plt.ylabel('Number of times', fontsize=8, rotation=90); # Labeling the axis.
#plt.gca().set_yticklabels(['{:,.2f}'.format(x) for x in plt.gca().get_yticks()]) #To format y axis
#plt.legend().set_visible(False)
#ax.legend().set_visible(False)
#ax.xaxis.set_major_locator(plt.MaxNLocator(3))                                #Define the number of ticks to show on x axis.
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())                          #Other way to Define the number of ticks to show on x axis.
#ax.xaxis.set_major_formatter(mdates.AutoDateFormatter())                      #Other way to Define the number of ticks to show on x axis.
#ax = plt.gca()                                                                #To get the current active axes: x_ax = ax.coords[0]; y_ax = ax.coords[1]

#handles, labels = ax.get_legend_handles_labels()                              #To make a single legend for many subplots with matplotlib
#fig.legend(handles, labels)                                                   #To make a single legend for many subplots with matplotlib

#plt.axis('off')                                                               #Turn off the axis in a graph, subplots.
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