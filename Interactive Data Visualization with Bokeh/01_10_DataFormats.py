# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:50:56 2019

@author: jacqueline.cortez

Capítulo 1. Basic plotting with Bokeh
Introduction:
    An introduction to basic plotting with Bokeh. 
    You will create your first plots, learn about different 
    data formats Bokeh understands, and make visual 
    customizations for selections and mouse hovering.
Excercise 10-14
"""

# Import packages
import pandas as pd
import numpy as np
#import tabula 
#import math
#import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
#import random
#import json

#from math import radians
#from functools import reduce#import pandas as pd
#from pandas.api.types import CategoricalDtype #For categorical data
#from glob import glob
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Data formats\n")

x = np.linspace(0, 10, 1000)
y = np.sin(x) + np.random.random(1000)*0.2 #np.random.random()-->Return random floats in the half-open interval [0.0, 1.0).

plot = figure()
plot.line(x, y)
output_file('01_10_numpy.html')
show(plot)


print("\n****************************************************")
print("Plotting data from NumPy arrays\n")

x = np.linspace(0, 5, 100) # Create array using np.linspace: x
y = np.cos(x) # Create array using np.cos: y

p = figure()
p.circle(x, y) # Add circles at x and y
output_file('01_11_numpy.html') # Specify the name of the output file and show the result
show(p)


print("\n****************************************************")
print("Plotting data from Pandas DataFrames\n")

# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "auto-mpg.data"
auto = pd.read_csv(file, sep="\s+", header=None, quotechar='"', na_values="?", 
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration","model_year", "origin", "car_name"], 
                   dtype={"mpg": np.float64, "cylinders": np.int64, "displacement": np.float64, "horsepower": np.float64, "weight": np.float64, "acceleration": np.float64,"model_year": np.int64, "origin": str, "car_name": str})

# Step 2 - Drop all rows that are missing 'driver_gender'
print("Cleaning the data...")
auto.dropna(subset=["horsepower"], inplace=True)
auto.reset_index(inplace=True)
auto["origin"] = auto.origin.replace(["1", "2", "3"],["North America", "Europe", "Asia"]) # Re-encode the values for origin (1, 2, 3) into “North America”, “Europe”, and “Asia” 
auto["origin"] = auto.origin.astype("category") # Fixing data types

# Configuring a no-existent color column
auto_color = ['blue', 'red', 'white', 'black', 'dark green', 'dark gray', 'dark red', 'light gray', 'gold']
auto["color"] = np.random.randint(len(auto_color), size=len(auto)) #color_number
auto["color"] = auto.color.replace(range(len(auto_color)),auto_color) # Re-encode the right color name

# Step 3- Make the graph
print("Creating the graph...")
p = figure(x_axis_label='Horse Power', y_axis_label='MPG') # Create the figure: p
p.circle(x=auto.horsepower, y=auto.mpg, color=auto.color, size=10, alpha=0.5) # Plot mpg vs hp by color
output_file('01_12_auto-df.html') # Specify the name of the output file and show the result
show(p)

print("\n****************************************************")
print("The Bokeh ColumnDataSource (continued)\n")

# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "100m_Olympic_Medal.csv"
medal = pd.read_csv(file, quotechar='"')

# Step 2- Make the graph
print("Creating the graph...")
p = figure(x_axis_label='Year', y_axis_label='Time')

# Create a ColumnDataSource from df: source
source = ColumnDataSource(medal)

# Add circle glyphs to the figure p
p.circle(x='Year', y='Time', source=source, color='color', size=8, alpha=0.5)

# Specify the name of the output file and show the result
output_file('01_14_olympic_medal.html')
show(p)

print("****************************************************")
print("** END                                            **")
print("****************************************************")