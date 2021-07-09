# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:18:02 2019

@author: jacqueline.cortez

Capítulo 1. Highlighting your data
Introduction:
    How do you show all of your data while making sure that viewers don't miss an important point or points? 
    Here we discuss how to guide your viewer through the data with color-based highlights and text. We also introduce 
    a dataset on common pollutant values across the United States.
"""

# Import packages
import pandas as pd                   #For loading tabular data
#import numpy as np                    #For making operations in lists
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
#import re                             #For regular expressions

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                        #For obteining today function
#from string import Template                                                          #For working with string, regular expressions

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

file = "pollution_wide.csv" 
pollution = pd.read_csv(file)

#houston_pollution = pollution[pollution.city  ==  'Houston']
houston_pollution = pollution[pollution.city  ==  'Houston'].copy()
max_O3 = houston_pollution.O3.max() # Find the highest observed O3 value
houston_pollution['point type'] = ['Highest O3 Day' if O3  ==  max_O3 else 'Others' for O3 in houston_pollution.O3] # Make a column that denotes which day had highest O3



print("****************************************************")
tema = '2. Hardcoding a highlight'; print("** %s\n" % tema)

# Make array orangred for day 330 of year 2014, otherwise lightgray
houston_colors = ['orangered' if (day  ==  330) & (year  ==  2014) else 'lightgray' for day,year in zip(houston_pollution.day, houston_pollution.year)]
#print(set(zip(houston_pollution.day, houston_pollution.year)))       

sns.set() # Set default Seaborn style
sns.regplot(x = 'NO2', y = 'SO2', data = houston_pollution, fit_reg = False, scatter_kws = {'facecolors': houston_colors, 'alpha': 0.7})
plt.title("City of Houston")
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '3. Programmatically creating a highlight'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
sns.scatterplot(x = 'NO2', y = 'SO2', hue = 'point type', data = houston_pollution) # Encode the hue of the points with the O3 generated column
plt.title("City of Houston")
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '5. Comparing with two KDEs'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
sns.kdeplot(pollution[pollution.year == 2012].O3, shade = True, label = '2012') # Filter dataset to the year 2012
sns.kdeplot(pollution[pollution.year != 2012].O3, shade = True, label = 'other years') # Filter dataset to everything except the year 2012
plt.xlabel('O3') # Label the axes
plt.ylabel('KDE (Kernel Density Estimator)')
plt.title("Pollution in 2012 compare with other years")
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '6. Improving your KDEs'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
sns.distplot(pollution[pollution.city == 'Vandenberg Air Force Base'].O3, label = 'Vandenberg', hist = False, color = 'steelblue', rug = True)
sns.distplot(pollution[pollution.city != 'Vandenberg Air Force Base'].O3, label = 'Other cities', hist = False, color = 'gray')
plt.xlabel('O3') # Label the axes
plt.ylabel('KDE (Kernel Density Estimator)')
plt.title("Pollution in Vandenberg Air Force Base")
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '7. Beeswarms'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
pollution_mar = pollution[pollution.month == 3]
sns.swarmplot(y = "city", x = 'O3', data = pollution_mar, size = 3)
plt.title('March Ozone levels by city')
plt.suptitle(tema)
plt.subplots_adjust(left=0.40, bottom=None, right=0.95, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '9. A basic text annotation'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
sns.scatterplot(x = 'CO', y = 'SO2', data = pollution[pollution.month  ==  8]) # Draw basic scatter plot of pollution data for August
plt.text(0.57, 41, "Cincinnati had highest observed\nSO2 value on Aug 11, 2013",  fontdict = {'ha': 'left', 'size': 'small'}) # Label highest SO2 value with text annotation
plt.title("Relation between CO and SO2")
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '10. Arrow annotations'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
# Query and filter to New Years in Long Beach
jan_pollution = pollution.query("(month  ==  1) & (year  ==  2012)")
lb_newyears = jan_pollution.query("(day  ==  1) & (city  ==  'Long Beach')")
sns.scatterplot(x = 'CO', y = 'NO2', data = jan_pollution)

# Point arrow to lb_newyears & place text in lower left 
plt.annotate('Long Beach New Years', xy = (lb_newyears.CO, lb_newyears.NO2), xytext = (2, 15), 
             # Shrink the arrow to avoid occlusion
             arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03},
             backgroundcolor = 'white')
plt.title("Pollution in January 2012")
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '11. Combining annotations and color'; print("** %s\n" % tema)

# Make a vector where Long Beach is orangered; else lightgray
is_lb = ['orangered' if city  ==  'Long Beach' else 'lightgray' for city in pollution['city']]

sns.set() # Set default Seaborn style
plt.figure()
# Map facecolors to the list is_lb and set alpha to 0.3
sns.regplot(x = 'CO', y = 'O3', data = pollution, fit_reg = False, scatter_kws = {'facecolors':is_lb, 'alpha': 0.3})
# Add annotation to plot
plt.text(1.6, 0.072, 'April 30th, Bad Day')
plt.title("Pollution in Long Beach City")
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
print("** END                                            **")
print("****************************************************")