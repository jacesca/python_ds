# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:46:51 2019

@author: jacqueline.cortez

Capítulo 1. Getting Started in Python
Introduction:
    Welcome to the wonderful world of Data Analysis in Python! In this chapter, you'll 
    learn the basics of Python syntax, load your first Python modules, and use functions 
    to get a suspect list for the kidnapping of Bayes, DataCamp's prize-winning Golden 
    Retriever.
"""

# Import packages
import pandas as pd                  #For loading tabular data
#import numpy as np                   #For making operations in lists
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
print("Filling the report...\n")

bayes_age    = 4.0           # Fill in Bayes' age (4.0)
favorite_toy = 'Mr. Squeaky' # Bayes' favorite toy
owner        = 'DataCamp'    # Bayes' owner
birthday     = '2019-02-14'
case_id      = 'DATACAMP!123-456?'

print("Read the file that contains the frequency of each letter in the ransom note for Bayes....\n")
r = pd.read_fwf('ransom.data')
print('{}\n'.format(r))

print("Graphing the data....\n")
sns.set()
plt.plot(r.letter.tolist(), r.frequency.tolist())
plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")