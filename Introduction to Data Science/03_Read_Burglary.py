# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:23:31 2019

@author: jacqueline.cortez

Capítulo 3. Plotting Data with matplotlib
Introduction:
    Get ready to visualize your data! You'll create line plots with another Python 
    module: matplotlib. Using line plots, you'll analyze the letter frequencies from 
    the ransom note and several handwriting samples to determine the kidnapper.
"""
# Import packages
import pandas as pd                  #For loading tabular data
#import numpy as np                   #For making operations in lists
import matplotlib.pyplot as plt      #For creating charts
#import seaborn as sns                #For visualizing data
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

file = 'Burglary per year in US State 1985-2014.csv'
print("Reading the data ({})...\n".format(file.upper()))

burglary_sta = pd.read_csv(file, skiprows=4, index_col='Agency')
burglary_sta.drop(['Unnamed: 32'], axis=1, inplace=True)
burglary_sta = burglary_sta.groupby('State').mean()
burglary_sta = burglary_sta.transpose()

print("** Making the graph\n")

# Plot lines
plt.plot(burglary_sta.index, burglary_sta.CA, label="California")
plt.plot(burglary_sta.index, burglary_sta.WA, label="Washington")
plt.plot(burglary_sta.index, burglary_sta.NY, label="New York")

plt.title('Burglary Statistics per Year') # Add a title
plt.xlabel('Year') # Add x-axis label
plt.xticks(rotation=90)
plt.ylabel('Burglaries register') # Add y-axis label
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=0.9, wspace=None, hspace=None)

plt.legend() # Add a legend
plt.show() # Display the plot


print("****************************************************")
print("** END                                            **")
print("****************************************************")
