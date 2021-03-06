# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:50:56 2019

@author: jacqueline.cortez

Cap?tulo 1. Basic plotting with Bokeh
Introduction:
    An introduction to basic plotting with Bokeh. 
    You will create your first plots, learn about different 
    data formats Bokeh understands, and make visual 
    customizations for selections and mouse hovering.
Excercise 01-05
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

#from math import radians
#from functools import reduce#import pandas as pd
#from pandas.api.types import CategoricalDtype #For categorical data
#from glob import glob
from bokeh.io import output_file, show
from bokeh.plotting import figure

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Plotting with glyphs\n")

#from bokeh.io import output_file, show
#from bokeh.plotting import figure

#Example 1
plot = figure(plot_width=400, tools="pan,box_zoom")
plot.circle([1,2,3,4,5], [8,6,5,2,3])

output_file('01_01_circle_ex1.html')
show(plot)

#Example 1
plot = figure()
plot.circle(x=10, y=[2,5,8,12], size=[10,20,30,40])

output_file('01_01_circle_ex2.html')
show(plot)

print("****************************************************")
print("A simple scatter plot\n")
#import numpy as np

# Importing fertility and female_literacy as a regular list
file = "Female_Educatiov_vs_Fertility.csv"
fertility = list(np.loadtxt(file, skiprows=1, usecols=4, delimiter=",", dtype=np.float32))
female_literacy = list(np.loadtxt(file, skiprows=1, usecols=3, delimiter=",", dtype=np.float32))

print("First 5 values of fertility: {}.".format(fertility[0:5]))
print("First 5 values of female_literacy: {}.".format(female_literacy[0:5]))

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('01_03_simple_scatter_plot.html')

# Display the plot
show(p)

print("****************************************************")
print("A scatter plot with different shapes\n")
#import pandas as pd

# Read the complete data
file = "Female_Educatiov_vs_Fertility.csv"
df_fem = pd.read_csv(file, index_col=0)

print("Exploring the complete data...")
print(df_fem.head())
print(df_fem.continent.unique())

# Getting the require lists
print("\nGetting the require lists...")
fertility_latinamerica = list(df_fem[df_fem.continent=='LAT']['fertility'].values)
female_literacy_latinamerica = list(df_fem[df_fem.continent=='LAT']['fem_literacity'].values)
fertility_africa = list(df_fem[df_fem.continent=='AF']['fertility'].values)
female_literacy_africa = list(df_fem[df_fem.continent=='AF']['fem_literacity'].values)

print("Latinoamerica: {}.".format(len(fertility_latinamerica)))
print("Africa: {}.".format(len(fertility_africa)))

print("\nMaking the boken plot...")

# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(x=fertility_latinamerica, y=female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(x=fertility_africa, y=female_literacy_africa)

# Specify the name of the file
output_file('01_04_scatter_dif_shape.html')

# Display the plot
show(p)

print("\n****************************************************")
print("Customizing your scatter plots\n")

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('01_05_scatter_custom.html')

# Display the plot
show(p)

print("****************************************************")
print("** END                                            **")
print("****************************************************")