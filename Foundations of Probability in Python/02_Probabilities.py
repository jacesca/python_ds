# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:36:32 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 2: Calculate some probabilities
    In this chapter you'll learn to calculate various kinds of probabilities, such as the 
    probability of the intersection of two events and the sum of probabilities of two events, 
    and to simulate those situations. You'll also learn about conditional probability and how 
    to apply Bayes' rule.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import numpy                         as np                                    #For making operations in lists

from scipy.stats                     import binom                             #Generate binomial data
from scipy.stats                     import find_repeats                      #To find repeated data in a sample. Statistical terms
from scipy.stats                     import relfreq                           #To calculate the relative frequency of each outcome


print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 1
np.random.seed(SEED) 
    

print("****************************************************")
topic = "3. Measuring a sample"; print("** %s\n" % topic)

# Generate a sample that represents 1000 throws of two fair coin  
sample_of_two_coin_flips = binom.rvs(n=2, p=.5, size=1000)

# 1. From the provided samples in sample_of_two_coin_flips, get the probability of having 2 heads out of the 1,000 trials.
count_2_heads = find_repeats(sample_of_two_coin_flips).counts[2] # Count how many times you got 2 heads from the sample data
prob_2_heads = count_2_heads / len(sample_of_two_coin_flips) # Divide the number of heads by the total number of draws
print("Get the probability of having 2 heads out of the 1,000 trials: {:,.2%}".format(prob_2_heads))

# 2. Calculate the relative frequency from sample_of_two_coin_flips, set numbins as 3, and extract frequency.
# Get the relative frequency from sample_of_two_coin_flips. Set numbins as 3. Extract frequency
rel_freq = relfreq(sample_of_two_coin_flips, numbins=3).frequency
print("\nCalculate the relative frequency from sample_of_two_coin_flips, set numbins as 3, and extract frequency: ", rel_freq)

# 3. Calculate the probability of getting 0, 1, or 2 from a binomial distribution with n=2 and p=0.5.
# Probability of getting 0, 1, or 2 from the distribution
probabilities = binom.pmf([0, 1, 2], n=2, p=0.5)
print("\nCalculate the probability of getting 0, 1, or 2 from a binomial distribution with n=2 and p=0.5: ", probabilities)

    
print("****************************************************")
topic = "4. Joint probabilities"; print("** %s\n" % topic)

#           ENGINE              GEARBOX
# FAILS       0.01                0.005 
# WORKS       0.99                0.995

# Individual probabilities
P_Eng_fails = 0.01
P_Eng_works = 0.99
P_GearB_fails = 0.005
P_GearB_works = 0.995


print("\nCalculate the probability that the engine and gear box both work.")
P_both_works = P_Eng_works*P_GearB_works # Joint probability calculation
print(P_both_works)


print("\nCalculate the probability that one fails -- either engine or gear box -- but not both.")
P_only_GearB_fails = P_GearB_fails*P_Eng_works # Joint probability calculation
P_only_Eng_fails = P_Eng_fails*P_GearB_works
P_one_fails = P_only_Eng_fails + P_only_GearB_fails # Calculate result
print(P_one_fails)


print("\nWhat is the probability that either both work or both fail?")
P_EngW_GearBW = P_Eng_works*P_GearB_works # Joint probability calculation
P_EngF_GearBF = P_Eng_fails*P_GearB_fails
P_fails_or_works = P_EngW_GearBW + P_EngF_GearBF # Calculate result
print(P_fails_or_works,"\n")
 
    
print("****************************************************")
topic = "5. Deck of cards"; print("** %s\n" % topic)
 
print("Calculate the probability of not getting an Ace.")
P_Ace = 4/52 # Ace probability
P_not_Ace = 1 - P_Ace # Not Ace probability
print(P_not_Ace)


print("\nCalculate the probability of getting a red card.")
P_Hearts = 13/52 # Figure probabilities
P_Diamonds = 13/52 # Probability of red calculation
P_Red = P_Hearts + P_Diamonds
print(P_Red)


print("\nCalculate the probability of getting a Jack or a spade.")
P_Jack = 4/52 # Figure probabilities
P_Spade = 13/52
P_Jack_n_Spade = 1/52 # Joint probability
P_Jack_or_Spade = P_Jack + P_Spade - P_Jack_n_Spade # Probability of Jack or spade
print(P_Jack_or_Spade)


print("\nCalculate the probability of getting a King or a Queen.")
P_King = 4/52 # Figure probabilities
P_Queen = 4/52
P_King_n_Queen = 0 # Joint probability
P_King_or_Queen = P_King + P_Queen - P_King_n_Queen # Probability of King or Queen
print(P_King_or_Queen)


    
print("****************************************************")
topic = "7. Delayed flights"; print("** %s\n" % topic)

#        Delayed	On time		Total
# Tuesday     24	    114	      138
# Friday	  11	    127	      138
# Total	      35	    241	      276

On_time = 241          # Needed quantities
Total_departures = 276 
Delayed_on_Tuesday = 24
On_Tuesday = 138
Delayed_on_Friday = 11
On_Friday = 138
 
print("What is the probability of a flight being on time?")
P_On_time = On_time / Total_departures # Needed quantities
print("{:,.2%}".format(P_On_time))

print("\nEvery departure is on time with probability P_On_time. What is the probability of a flight being delayed?")
P_Delayed = 1 - P_On_time # Probability calculation
print("{:,.2%}".format(P_Delayed))

print("\nGiven that it's Tuesday, what is the probability of a flight being delayed (P(Delayed|Tuesday))?")
P_Delayed_g_Tuesday = Delayed_on_Tuesday / On_Tuesday # Probability calculation
print("{:,.2%}".format(P_Delayed_g_Tuesday))

print("\nGiven that it's Friday, what is the probability of a flight being delayed (P(Delayed|Friday))?")
P_Delayed_g_Friday = Delayed_on_Friday / On_Friday
print("{:,.2%}".format(P_Delayed_g_Friday))


print("****************************************************")
topic = "8. Contingency table"; print("** %s\n" % topic)
    
#           Red	    Black    Total
# Ace    	  2	        2	     4
# Non Ace	 24	       24	    48
# Total	     26	       26	    52

P_Red = 26/52 # Individual probabilities
P_Red_n_Ace = 2/52
P_Ace_given_Red = P_Red_n_Ace / P_Red # Conditional probability calculation
print("Calculate P(Ace|Red): {:,.2%}.".format(P_Ace_given_Red))


P_Ace = 4/52   # Individual probabilities
P_Ace_n_Black = 2/52
P_Black_given_Ace = P_Ace_n_Black / P_Ace  # Conditional probability calculation
print("Calculate P(Black|Ace): {:,.2%}.".format(P_Black_given_Ace))


P_Black = 26/52  # Individual probabilities
P_Black_n_Non_ace = 24/52
P_Non_ace_given_Black = P_Black_n_Non_ace / P_Black # Conditional probability calculation
print("Calculate P(Non Ace|Black): {:,.2%}.".format(P_Non_ace_given_Black))


P_Non_ace = 48/52  # Individual probabilities
P_Non_ace_n_Red = 24/52
P_Red_given_Non_ace = P_Non_ace_n_Red / P_Non_ace   # Conditional probability calculation
print("Calculate P(Red|Non Ace): {:,.2%}.".format(P_Red_given_Non_ace))

    
print("****************************************************")
topic = "9. More cards"; print("** %s\n" % topic)

P_first_Jack = 4/52  # Needed probabilities
P_Jack_given_Jack = 3/51
P_two_Jacks = P_first_Jack * P_Jack_given_Jack  # Joint probability calculation
print("Calculate the probability of getting two Jacks (P(Jack and Jack)): {:,.2%}.".format(P_two_Jacks))


P_Spade = 13/52  # Needed probabilities
P_Spade_n_Ace = 1/52
P_Ace_given_Spade = P_Spade_n_Ace / P_Spade  # Conditional probability calculation
print("Calculate P(Ace|Spade): {:,.2%}.".format(P_Ace_given_Spade))


P_Face_card = 12/52  # Needed probabilities
P_Face_card_n_Queen = 4/52
P_Queen_given_Face_card = P_Face_card_n_Queen / P_Face_card  # Conditional probability calculation
print("Calculate P(Queen|Face card): {:,.2%}.".format(P_Queen_given_Face_card))


print("****************************************************")
topic = "11. Formula 1 engines"; print("** %s\n" % topic)

# Needed probabilities
P_A = 0.7
P_last5000_g_A = 0.99
P_B = 0.3
P_last5000_g_B = 0.95

# Total probability calculation
P_last_5000 = (P_A*P_last5000_g_A)  + (P_B*P_last5000_g_B)

print("What is the chance that an engine will last more than 5,000 km?")
print("{:,.2%}.".format(P_last_5000))

    
print("****************************************************")
topic = "12. Voters"; print("** %s\n" % topic)

# Individual probabilities
P_X = .43
P_Y = .25
P_Z = .32

# Conditional probabilities
P_Support_g_X = .53
P_Support_g_Y = .67
P_Support_g_Z = .32

print("What is the probability that the voter lives in state X and supports John Doe?")
P_X_n_Support = P_X * P_Support_g_X      # Total probability calculation
print("{:,.2%}.".format(P_X_n_Support))

print("\nWhat is the probability that the voter lives in state Z and does not support John Doe?")
P_NoSupport_g_Z = 1 - P_Support_g_Z      # Conditional probabilities
P_Z_n_NoSupport = P_Z * P_NoSupport_g_Z  # Total probability calculation
print("{:,.2%}.".format(P_Z_n_NoSupport))

print("\nWhat is the total percentage of voters that support John Doe?")
P_Support = P_X * P_Support_g_X + P_Y * P_Support_g_Y + P_Z * P_Support_g_Z  # Total probability calculation
print("{:,.2%}.".format(P_Support))

  
print("****************************************************")
topic = "15. Factories and parts"; print("** %s\n" % topic)

# Individual probabilities & conditional probabilities
P_V1 = .50
P_V2 = .25
P_V3 = .25
P_D_g_V1 = .01
P_D_g_V2 = .02
P_D_g_V3 = .03

# Probability of Damaged
P_Damaged = (P_V1 * P_D_g_V1) + (P_V2 * P_D_g_V2) + (P_V3 * P_D_g_V3)

print("What is the probability that the part was manufactured by V1? --> P(V1|D)")
# Bayes' rule for P(V1|D)
P_V1_g_D = (P_V1 * P_D_g_V1) / P_Damaged
print("{:,.2%}.".format(P_V1_g_D))

print("\nWhat is the probability that it was manufactured by V2?--> P(V2|D)")
# Bayes' rule for P(V2|D)
P_V2_g_D = (P_V2 * P_D_g_V2) / P_Damaged
print("{:,.2%}.".format(P_V2_g_D))

print("\nWhat is the probability that the part was manufactured by V3?--> P(V3|D)")
# Bayes' rule for P(V3|D)
P_V3_g_D = (P_V3 * P_D_g_V3) / P_Damaged
print("{:,.2%}.".format(P_V3_g_D))
    

print("****************************************************")
topic = "16. Swine flu blood test"; print("** %s\n" % topic)
    
print("What is the probability that you have swine flu? --> P(PSF|PP)")
# CONFUSION MATRIX
# --------------|-----------------------|---------------------
# PREDICTION	   REALITY
#---------------|-----------------------|---------------------
#                  TRUE (PSF)             FALSE (PNSF)
# TRUE (PP)        100%                   1% (false positive)
# FALSE(PN)        0% (false negative)    99%
# --------------|-----------------------|---------------------
# PROPOSITION:
# PSF = Have a swine flue -------------> 1/9000
# PNSF= Don't have a swine flue
# PP = Test positive
# PN = Test negative
# P(PP|PNSF) = 1% -----------> P(PN|PNSF) = 99%%
# P(PN|PSF)  = 0% -----------> P(PP|PSF)  = 100%
#
#              P(PSF)P(PP|PSF)
# P(PSF|PP) = -----------------
#                   P(PP)
# Individual probabilities & conditional probabilities
# Individual probabilities & conditional probabilities
P_Swine_flu = 1/350
P_no_Swine_flu = 1 - P_Swine_flu
P_Positive_g_Swine_flu = 1
P_Positive_g_no_Swine_flu = 0.01
# Probability of Positive
P_Positive = (P_Swine_flu * P_Positive_g_Swine_flu) + (P_no_Swine_flu * P_Positive_g_no_Swine_flu)
# Bayes' rule for P(Swine_flu|Positive)
P_Swine_flu_g_Positive = (P_Swine_flu * P_Positive_g_Swine_flu) / P_Positive
print("{:,.2%}.".format(P_Swine_flu_g_Positive))


print("\nYou went to Miami and 1 in 350 people came back with swine flu. Calculate the new probability that you'll test positive. --> P(PSF|PP)")
# CONFUSION MATRIX
# --------------|-----------------------|---------------------
# PREDICTION	   REALITY
#---------------|-----------------------|---------------------
#                  TRUE (PSF)             FALSE (PNSF)
# TRUE (PP)        100%                   1% (false positive)
# FALSE(PN)        0% (false negative)    99%
# --------------|-----------------------|---------------------
# PROPOSITION:
# PSF = Have a swine flue -------------> 1/350
# PNSF= Don't have a swine flue
# PP = Test positive
# PN = Test negative
# P(PP|PNSF) = 1% -----------> P(PN|PNSF) = 99%%
# P(PN|PSF)  = 0% -----------> P(PP|PSF)  = 100%
#
#              P(PSF)P(PP|PSF)
# P(PSF|PP) = -----------------
#                   P(PP)
# Individual probabilities & conditional probabilities
P_Swine_flu = 1./350
P_no_Swine_flu = 1 - P_Swine_flu
P_Positive_g_Swine_flu = 1
P_Positive_g_no_Swine_flu = 0.01
# Probability of Positive
P_Positive = (P_Swine_flu * P_Positive_g_Swine_flu) + (P_no_Swine_flu * P_Positive_g_no_Swine_flu)
# Bayes' rule for P(Swine_flu|Positive)
P_Swine_flu_g_Positive = (P_Swine_flu * P_Positive_g_Swine_flu) / P_Positive
print("{:,.2%}.".format(P_Swine_flu_g_Positive))


print("\nIf the probability of a false positive is 2%, what is the new probability that you have swine flu after your vacation? --> P(PSF|PP)")
# CONFUSION MATRIX
# --------------|-----------------------|---------------------
# PREDICTION	   REALITY
#---------------|-----------------------|---------------------
#                  TRUE (PSF)             FALSE (PNSF)
# TRUE (PP)        100%                   1% (false positive)
# FALSE(PN)        0% (false negative)    99%
# --------------|-----------------------|---------------------
# PROPOSITION:
# PSF = Have a swine flue -------------> 1/350
# PNSF= Don't have a swine flue
# PP = Test positive
# PN = Test negative
# P(PP|PNSF) = 2% -----------> P(PN|PNSF) = 98%%
# P(PN|PSF)  = 0% -----------> P(PP|PSF)  = 100%
#
#              P(PSF)P(PP|PSF)
# P(PSF|PP) = -----------------
#                   P(PP)
# Individual probabilities & conditional probabilities
P_Swine_flu = 1./350
P_no_Swine_flu = 1 - P_Swine_flu
P_Positive_g_Swine_flu = 1
P_Positive_g_no_Swine_flu = .02
# Probability of Positive
P_Positive = P_Swine_flu * P_Positive_g_Swine_flu + P_no_Swine_flu * P_Positive_g_no_Swine_flu
# Bayes' rule for P(Swine_flu|Positive)
P_Swine_flu_g_Positive = (P_Swine_flu * P_Positive_g_Swine_flu) / P_Positive
print("{:,.2%}.".format(P_Swine_flu_g_Positive))

    
print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import contextily                                                             #To add a background web map to our plot
#import inspect                                                                #Used to get the code inside a function
#import itertools                                                              #For iterations
#import folium                                                                 #To create map street folium.__version__'0.10.0'
#import geopandas         as gpd                                               #For working with geospatial data 
#import math                                                                   #https://docs.python.org/3/library/math.html
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