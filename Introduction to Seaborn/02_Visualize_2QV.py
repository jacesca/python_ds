# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:44:47 2019

@author: jacqueline.cortez

Capítulo 2: Visualizing Two Quantitative Variables
Introduction:
    In this chapter, you will create and customize plots that visualize the 
    relationship between two quantitative variables. To do this, you will use 
    scatter plots and line plots to explore how the level of air pollution in a 
    city changes over the course of a day and how horsepower relates to fuel 
    efficiency in cars. You will also see another big advantage of using Seaborn - 
    the ability to easily create subplots in a single figure!
"""


# Import packages
import pandas as pd
#import numpy as np
#import tabula 
#import math
import matplotlib.pyplot as plt
import seaborn as sns
#import scipy.stats as stats
#import random
#import calendar

#from pandas.plotting import register_matplotlib_converters #for conversion as datetime index in x-axis
#from math import radians
#from functools import reduce#import pandas as pd
#from pandas.api.types import CategoricalDtype #For categorical data
#from glob import glob
#from bokeh.io import output_file, show
#from bokeh.plotting import figure

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

#file = 'https://assets.datacamp.com/production/repositories/3996/datasets/61e08004fef1a1b02b62620e3cd2533834239c90/student-alcohol-consumption.csv'
file = 'student_data.csv'
print("Reading the data ({})...\n".format(file.upper()))
student_data = pd.read_csv(file)

file = "auto-mpg.csv"
print("Reading the data ({})...\n".format(file.upper()))
mpg = pd.read_csv(file, quotechar='"', skiprows=1,
                  names=['mpg','cylinders','displacement','horsepower','weight',
                         'acceleration','model_year','origin','name','color','size'])

tema = '2. Creating subplots with col and row'
print("****************************************************")
print("** %s\n" % tema)

# A normal scatter plot
g = sns.scatterplot(x="absences", y="G3", data=student_data)
g.set_title('Relationship between study time and final grade (scatter)') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


# A normal relplot plot
g = sns.relplot(x="absences", y="G3", data=student_data, kind='scatter')
#g.set_titles('Relationship between study time and final grade (relplot)') # Doesn't work because there is no subplot yet
plt.subplots_adjust(top=0.9)#, left=None, bottom=None, right=None, wspace=None, hspace=None)
plt.title('Relationship between study time and final grade (relplot)') # Doesn't work because overlap
plt.suptitle(tema) #Same as g.fig.suptitle(tema)
plt.show() # Show plot


# A relplot plot organized in cols
sns.relplot(x="absences", y="G3", data=student_data, kind='scatter', col='study_time')
plt.subplots_adjust(top=0.85)#, left=None, bottom=None, right=None, wspace=None, hspace=None)
plt.suptitle('{}\nRelationship between study time and final grade (relplot)'.format(tema))
plt.show() # Show plot


# A relplot plot organized in rows
sns.relplot(x="absences", y="G3", data=student_data, kind='scatter', 
                 row='study_time')
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.3)
plt.suptitle('{}\nRelationship between study time and final grade (relplot)'.format(tema))
plt.show() # Show plot


tema = '3. Creating two-factor subplots'
print("****************************************************")
print("** %s\n" % tema)

# Adjust further to add subplots based on family support
sns.relplot(x="G1", y="G3", data=student_data, kind="scatter", 
            col="schoolsup",  col_order=["yes", "no"],
            row='famsup', row_order=['yes','no'])
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.3)
plt.suptitle('{}\nRelationship between schoolsup and famsup'.format(tema))
plt.show() # Show plot


tema = '5. Changing the size of scatter plot points'
print("****************************************************")
print("** %s\n" % tema)

# Create scatter plot of horsepower vs. mpg
sns.relplot(x="horsepower", y="mpg", data=mpg, kind="scatter", 
            size="cylinders", hue='cylinders')
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.90, wspace=None, hspace=None)
plt.title('Relationship between horsepower and mpg') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '6. Changing the style of scatter plot points'
print("****************************************************")
print("** %s\n" % tema)

# Create a scatter plot of acceleration vs. mpg
sns.relplot(kind='scatter', x='acceleration', y='mpg', data=mpg,
            style='origin', hue='origin')
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.90, wspace=None, hspace=None)
plt.title('Relationship between acceleration and mpg') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '8. Interpreting line plots'
print("****************************************************")
print("** %s\n" % tema)

sns.relplot(x='model_year', y='mpg', data=mpg, kind='line')
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.90, wspace=None, hspace=None)
plt.title('Relationship between acceleration and mpg') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '9. Visualizing standard deviation with line plots'
print("****************************************************")
print("** %s\n" % tema)

sns.relplot(x="model_year", y="mpg", data=mpg, kind="line") # Make the shaded area show the standard deviation
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.90, wspace=None, hspace=None)
plt.title('Confidence interval of the mpg evolution through years') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


sns.relplot(x="model_year", y="mpg", data=mpg, kind="line", ci='sd') # Make the shaded area show the standard deviation
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.90, wspace=None, hspace=None)
plt.title('Standard Desviation of the mpg evolution through years') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '10. Plotting subgroups in line plots'
print("****************************************************")
print("** %s\n" % tema)

# Add markers and make each line have the same style
sns.relplot(x="model_year", y="horsepower", data=mpg, kind="line", 
                 ci=None, style="origin", hue="origin", dashes=False, markers=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.90, wspace=None, hspace=None)
plt.title('Horsepower evolution through years') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


print("****************************************************")
print("** END                                            **")
print("****************************************************")