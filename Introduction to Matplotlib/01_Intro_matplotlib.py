# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:50:56 2019

@author: jacqueline.cortez

Capítulo 1. Introduction to Matplotlib
Introduction:
    This chapter introduces the Matplotlib visualization library and 
    demonstrates how to use it with data.
Excercise 01-10
"""

# Import packages
import pandas as pd
#import numpy as np
#import tabula 
#import math
import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
#import random

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

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** 2. Using the matplotlib.pyplot interface\n")

fig, ax = plt.subplots() # Create a Figure and an Axes with plt.subplots
ax.set_title('2. Using the matplotlib.pyplot interface') # Add the title
plt.show() # Call the show function to show the result
#print (ax.shape) --> Error, un único subplots


print("****************************************************")
print("** 3. Adding data to an Axes object")
print("** 5. Customizing data appearance")
print("** 6. Customizing axis labels and adding titles\n")

print("Preparin data...\n")
seattle_file = 'SEATTLE_weather.csv'
seattle_weather = pd.read_csv(seattle_file, index_col='date', parse_dates = True)
#seattle_year = seattle_weather.resample('M').TMAX.mean()
#seattle_year.index = seattle_year.index.strftime('%b').str.upper()
#seattle_year = seattle_year.reset_index()
seattle_year = seattle_weather.resample('M').agg({'PRCP': 'sum', 'TMAX': 'max', 
                                                  'TAVG': 'mean','TMIN': 'min'})
seattle_year['month'] = seattle_year.index.strftime('%b')

austin_file = 'AUSTIN_weather.csv'
austin_weather = pd.read_csv(austin_file, index_col='date', parse_dates = True)
austin_year = austin_weather.resample('M').agg({'PRCP': 'sum', 'TMAX': 'max', 
                                                'TAVG': 'mean','TMIN': 'min'})
austin_year['month'] = austin_year.index.strftime('%b')


print("Graphing...\n")

fig, ax = plt.subplots() # Create a Figure and an Axes with plt.subplots
ax.plot(seattle_year['month'], seattle_year['TAVG'],
        color='b', marker='o', linestyle='--') # Plot MLY-PRCP-NORMAL from seattle_weather against the MONTH

ax.plot(austin_year['month'], austin_year['TAVG'],
        color='r', marker='^', linestyle='--')# Plot MLY-PRCP-NORMAL from austin_weather against MONTH


ax.set_xlabel('Time (months)') # Customize the x-axis label
ax.set_ylabel('Average temperature (Celsius degrees)') # Customize the y-axis label
ax.set_title('(3, 5, 6). Weather patterns in Austin and Seattle') # Add the title

plt.show()# Call the show function
#print (ax.shape) --> Error, un único subplots


print("****************************************************")
print("** 7. Small multiples")

fig, ax = plt.subplots(3,2) # Create a Figure and an Axes with plt.subplots
fig.suptitle('7. Small multiples') # Add the title
plt.show() # Call the show function to show the result
print (ax.shape)


print("****************************************************")
print("** 9. Creating small multiples with plt.subplots")

fig, ax = plt.subplots(2, 2) # Create a Figure and an array of subplots with 2 rows and 2 columns
fig.suptitle('9. Creating small multiples with plt.subplots') # Add the title

ax[0, 0].plot(seattle_year["month"], seattle_year["PRCP"]) # Addressing the top left Axes as index 0, 0, plot Seattle precipitation
ax[0, 1].plot(seattle_year["month"], seattle_year["TAVG"]) # In the top right (index 0,1), plot Seattle temperatures
ax[1, 0].plot(austin_year["month"], austin_year["PRCP"]) # In the bottom left (1, 0) plot Austin precipitations
ax[1, 1].plot(austin_year["month"], austin_year["TAVG"]) # In the bottom right (1, 1) plot Austin temperatures

plt.show()



print("****************************************************")
print("** 9. Creating small multiples with plt.subplots (again)")

fig, ax = plt.subplots(2, 2) # Create a Figure and an array of subplots with 2 rows and 2 columns
fig.suptitle('9. Creating small multiples with plt.subplots (again)') # Add the title

ax[0, 0].plot(seattle_year["month"], seattle_year["PRCP"]) # Addressing the top left Axes as index 0, 0, plot Seattle precipitation
ax[0, 1].plot(austin_year["month"], austin_year["PRCP"]) # In the bottom left (1, 0) plot Austin precipitations
ax[1, 0].plot(seattle_year["month"], seattle_year["TAVG"]) # In the top right (index 0,1), plot Seattle temperatures
ax[1, 1].plot(austin_year["month"], austin_year["TAVG"]) # In the bottom right (1, 1) plot Austin temperatures

plt.show()


print("****************************************************")
print("** 10. Small multiples with shared y axis")

# Create a figure and an array of axes: 2 rows, 1 column with shared y axis
fig, ax = plt.subplots(2, 1, sharey=True)
fig.suptitle('10. Small multiples with shared y axis') # Add the title

# Plot Seattle precipitation data in the top axes
ax[0].plot(seattle_year["month"], seattle_year["TAVG"], color = 'b')
ax[0].plot(seattle_year["month"], seattle_year["TMIN"], color = 'b', linestyle = '--')
ax[0].plot(seattle_year["month"], seattle_year["TMAX"], color = 'b', linestyle = '--')

# Plot Austin precipitation data in the bottom axes
ax[1].plot(austin_year["month"], austin_year["TAVG"], color = 'r')
ax[1].plot(austin_year["month"], austin_year["TMIN"], color = 'r', linestyle = '--')
ax[1].plot(austin_year["month"], austin_year["TMAX"], color = 'r', linestyle = '--')

plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")

"""
#resample aggregate function use multiple columns
df_x.resample('5Min').agg({'price': 'mean', 'vol': 'sum'}).head()


#https://docs.python.org/3/library/datetime.html
http://strftime.org/
#Get month full name
austin_year.index.strftime('%B')

#Get 3 first letters of month name
austin_year.index.strftime('%b')

#Convert to upper and list
seattle_year.index.strftime('%b').str.upper().tolist()
"""