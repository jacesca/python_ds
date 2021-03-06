# -*- coding: utf-8 -*-
"""
Created on Mon May 20 07:37:56 2019

@author: jacqueline.cortez

Capítulo 1. Introduction to Seaborn
Introduction:
    What is Seaborn, and when should you use it? In this chapter, you will find out! 
    Plus, you will learn how to create scatter plots and count plots with both lists 
    of data and pandas DataFrames. You will also be introduced to one of the big 
    advantages of using Seaborn - the ability to easily add a third variable to your 
    plots by using color to represent different subgroups.
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

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

file = 'Databank_2015.csv'
print("Reading the data ({})...\n".format(file.upper()))
world_data = pd.read_csv(file, index_col='country')

print("Getting the required data...\n")

gdp              = world_data.gdp.tolist()
phones           = world_data.mobile_subs.tolist()
percent_literate = world_data.literacy_rate.tolist()
region           = world_data.region.tolist()
sub_region       = world_data.sub_region.tolist()


#file = 'http://assets.datacamp.com/production/repositories/3996/datasets/ab13162732ae9ca1a9a27e2efd3da923ed6a4e7b/young-people-survey-responses.csv'
file = 'Survey_spiderscare.csv'
print("Reading the data ({})...\n".format(file.upper()))
spider_data = pd.read_csv(file)


#file = 'https://assets.datacamp.com/production/repositories/3996/datasets/61e08004fef1a1b02b62620e3cd2533834239c90/student-alcohol-consumption.csv'
file = 'student_data.csv'
print("Reading the data ({})...\n".format(file.upper()))
student_data = pd.read_csv(file)


tema = '2. Making a scatter plot with lists'
print("****************************************************")
print("** %s\n" % tema)

g = sns.scatterplot(x=gdp, y=phones) # Create scatter plot with GDP on the x-axis and number of phones on the y-axis
g.set_title('GDP vs Phones') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


plt.figure() # Change this scatter plot to have percent literate on the y-axis
g = sns.scatterplot(x=gdp, y=percent_literate)
g.set_title('GDP vs Percent Literate') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '3. Making a scatter plot with lists'
print("****************************************************")
print("** %s\n" % tema)


plt.figure() 
g = sns.countplot(y=sub_region) # Create count plot with sub_region on the y-axis
g.set_title('Counting sub-region') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


plt.figure() 
g = sns.countplot(x=region) # Create count plot with region on the x-axis
g.set_title('Counting region') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '6. Making a count plot with a DataFrame'
print("****************************************************")
print("** %s\n" % tema)

plt.figure() 
g = sns.countplot(x='Spiders', data=spider_data) # Create a count plot with "Spiders" on the x-axis
g.set_title('How many young people surveyed report being scared of spiders?') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '7. Adding a third variable with hue'
print("****************************************************")
print("** %s\n" % tema)

tips = sns.load_dataset('tips') #load a pre-data 
print(tips.head())

#print("\nAnother dataset inside seaborn: ")
#print(sns.get_dataset_names())
#['anscombe', 'attention', 'brain_networks', 'car_crashes', 'diamonds', 'dots', 'exercise', 'flights', 'fmri', 'gammas', 'iris', 'mpg', 'planets', 'tips', 'titanic']
#Source of data--> https://github.com/mwaskom/seaborn-data

tema = '8. Hue and scatter plots'
print("****************************************************")
print("** %s\n" % tema)

plt.figure() 
g = sns.scatterplot(x="absences", y="G3", data=student_data,# Change the legend order in the scatter plot
                     hue="location", hue_order=['Rural', 'Urban'])
g.set_title('Relationship between absences and final grade') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '9. Hue and count plots'
print("****************************************************")
print("** %s\n" % tema)

plt.figure() 
g = sns.countplot(x='school', data=student_data, hue='location', # Create a count plot of school with location subgroups
              palette={'Rural': 'green', 'Urban': 'blue'})
g.set_title('How many students live in urban vs. rural areas?') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


print("****************************************************")
print("** END                                            **")
print("****************************************************")



#Warning generated when the next line is executed:
#    print(sns.get_dataset_names())
#    
#Another dataset inside seaborn: 
#C:\Anaconda3\lib\site-packages\seaborn\utils.py:376: UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("lxml"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.
#
#The code that caused this warning is on line 376 of the file C:\Anaconda3\lib\site-packages\seaborn\utils.py. To get rid of this warning, pass the additional argument 'features="lxml"' to the BeautifulSoup constructor.