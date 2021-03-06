# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:50:18 2019

@author: jacqueline.cortez

Capítulo 2. Plotting time-series
Introduction:
    Time-series data are data that are recorded. Visualizing this kind of data helps 
    clarify trends and understand relationships between different data.
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

from pandas.plotting import register_matplotlib_converters
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
register_matplotlib_converters() #Require to explicitly register matplotlib converters.

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** 2. Read data with a time index\n")

print("Reading global CO2 emissions...\n")
file = 'CO2_globales_0-2014.csv'
co2 = pd.read_csv(file, parse_dates = [['year', 'month', 'day']], 
                  skiprows = range(1,22201),
                  usecols=['year', 'month', 'day', 'data_mean_global'],
                  index_col=0)
#data_mean_nh --> north hemisphere
#data_mean_sh --> south hemisphere

print("Reading global temperatures anomalies...\n")
file = 'CO2_temperature_1850-1919.txt'
temp = pd.read_fwf(file, header = None,
                   names = ['ym', 'temp', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                   usecols = ['ym', 'temp'],
                   skipfooter = 50)
temp['year'] = temp.ym.str.split("/").str.get(0)
temp['month'] = temp.ym.str.split("/").str.get(1)
temp['date'] = pd.to_datetime(temp.year+'-'+temp.month+'-'+'15')
temp.set_index('date', inplace=True)
temp.drop(["ym",'year','month'], axis='columns', inplace=True)

print("Concatening the two previews dataframes...")
climate_change = pd.concat([co2, temp], axis=1)
climate_change.columns = ['co2','relative_temp']
print(climate_change.head())

print("\n****************************************************")
print("** 3. Plot time-series data\n")

#register_matplotlib_converters() #Require to explicitly register matplotlib converters.
fig, ax = plt.subplots()

ax.set_title('3. Plot time-series data') # Add the title
ax.plot(climate_change.index, climate_change.relative_temp) # Add the time-series for "relative_temp" to the plot

ax.set_xlabel('Time') # Set the x-axis label
ax.set_ylabel('Relative temperature (Celsius)')# Set the y-axis label

plt.show() # Show the figure


print("\n****************************************************")
print("** 4. Using a time index to zoom in\n")

seventies = climate_change['1970-01-01':'1979-12-31'] # Create variable seventies with data from "1970-01-01" to "1979-12-31"

fig, ax = plt.subplots() # Use plt.subplots to create fig and ax
ax.set_title('4. Using a time index to zoom in') # Add the title

ax.plot(seventies.index, seventies.co2) # Add the time-series for "co2" data from seventies to the plot
plt.show() # Show the figure



print("\n****************************************************")
print("** 6. Plotting two variables\n")

fig, ax = plt.subplots() # Initalize a Figure and Axes
ax.set_title('6. Plotting two variables') # Add the title

ax.plot(climate_change.index, climate_change.co2, color='blue') # Plot the CO2 variable in blue
ax2 = ax.twinx() # Create a twin Axes that shares the x-axis
ax2.plot(climate_change.index, climate_change.relative_temp, color='red') # Plot the relative temperature in red

plt.show()


print("\n****************************************************")
print("** 7. Defining a function that plots time-series data\n")
print("** 8. Using a plotting function\n")

def plot_timeseries(axes, x, y, color, xlabel, ylabel): # Define a function called plot_timeseries
  axes.plot(x, y, color=color)         # Plot the inputs x,y in the provided color
  axes.set_xlabel(xlabel)              # Set the x-axis label
  axes.set_ylabel(ylabel, color=color) # Set the y-axis label
  axes.tick_params('y', colors=color)  # Set the colors tick params for y-axis
  
fig, ax = plt.subplots() # Initalize a Figure and Axes
ax.set_title('8. Using a plotting function\n') # Add the title

plot_timeseries(ax, climate_change.index, climate_change.co2, "blue", 'Time (years)', 'CO2 levels') # Plot the CO2 levels time-series in blue
ax2 = ax.twinx() # Create a twin Axes object that shares the x-axis
plot_timeseries(ax2, climate_change.index, climate_change.relative_temp, "red", 'Time', 'Relative temperature (Celsius)') # Plot the relative temperature data in red

plt.show()

print("\n****************************************************")
print("** 10. Annotating a plot of time-series data\n")

fig, ax = plt.subplots() # Initalize a Figure and Axes
ax.set_title('10. Annotating a plot of time-series data\n') # Add the title

# Plot the relative temperature data
ax.plot(climate_change.index, climate_change.relative_temp)

ax.set_xlabel('Time') # Set the x-axis label
ax.set_ylabel('Relative temperature (Celsius)') # Set the y-axis label

ax.annotate('>0.5 degree', (pd.Timestamp('1990-03-15'), 0.5)) # Annotate the date at which temperatures exceeded 0.5 degree
ax.annotate('>0.75 degree', (pd.Timestamp('1998-02-15'), 0.75)) # Annotate the date at which temperatures exceeded 0.75 degree

plt.show()

print("\n****************************************************")
print("** 11. Plotting time-series: putting it all together\n")

fig, ax = plt.subplots() # Initalize a Figure and Axes
ax.set_title('11. Plotting time-series: putting it all together\n') # Add the title

plot_timeseries(ax, climate_change.index, climate_change.co2, 'blue', 'Time (years)', 'CO2 levels') # Plot the CO2 levels time-series in blue
ax2 = ax.twinx() # Create an Axes object that shares the x-axis
plot_timeseries(ax2, climate_change.index, climate_change.relative_temp, 'red', 'Time (years)', 'Relative temp (Celsius)') # Plot the relative temperature data in red

# Annotate point with relative temperature >1 degree
ax2.annotate(">0.75 degree", 
             xy=(pd.Timestamp('1998-02-15'), 0.75), 
             xytext=(pd.Timestamp('1980-01-15'),-0.5), 
             arrowprops={'arrowstyle':'fancy', 'color':'gray'})

ax2.annotate(">0.5 degree", 
             xy=(pd.Timestamp('1990-03-15'), 0.5), 
             xytext=(pd.Timestamp('1955-01-15'),-0.75), 
             arrowprops={'arrowstyle':'fancy', 'color':'gray'})

plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")