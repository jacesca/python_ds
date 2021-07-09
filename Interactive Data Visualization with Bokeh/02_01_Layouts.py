# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:23:50 2019

@author: jacqueline.cortez

Capítulo 2. Layouts, Interactions, and Annotations
Introduction:
    Learn how to combine mutiple Bokeh plots into different kinds of layouts on a page, 
    how to easily link different plots together in various ways, and how to add annotations 
    such as legends and hover tooltips.
Excercise: 01-04
"""


# Import packages
import pandas as pd
#import numpy as np
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
#from bokeh.models import HoverTool, CategoricalColorMapper
from bokeh.layouts import row, column

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Creating rows of plots\n")

# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "Female_Educatiov_vs_Fertility.csv"
female = pd.read_csv(file, quotechar='"')

# Step 2- Make the graph
print("Creating the graph...")

source = ColumnDataSource(female) # Convert df to a ColumnDataSource: source

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', 
            y_axis_label='female literacy (% population)',
            title="Female literacy vr Fertility in the world")
p1.circle('fertility', 'fem_literacity', source=source) # Add a circle glyph to p1

# Create the second figure: p2
p2 = figure(x_axis_label='population', 
            y_axis_label='female literacy (% population)',
            title="Population vs Female literacy in the world")
p2.circle('population', 'fem_literacity', source=source) # Add a circle glyph to p2

# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('02_02_fert_row.html')
show(layout)



print("****************************************************")
print("Creating columns of plots\n")

source = ColumnDataSource(female) # Convert df to a ColumnDataSource: source

# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', 
            y_axis_label='female literacy (% population)',
            title="Female literacy vr Fertility in the world")
p1.circle('fertility', 'fem_literacity', source=source) # Add a circle glyph to p1

# Create the second figure: p2
p2 = figure(x_axis_label='population', 
            y_axis_label='female literacy (% population)',
            title="Population vs Female literacy in the world")
p2.circle('population', 'fem_literacity', source=source) # Add a circle glyph to p2

# Put plots p1 and p2 in a column: layout
layout = column(p1, p2)

# Specify the name of the output_file and show the result
output_file('02_03_fert_column.html')
show(layout)



print("****************************************************")
print("Nesting rows and columns of plots\n")

# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "auto-mpg.csv"
auto = pd.read_csv(file, quotechar='"')

mean_mpg = auto.groupby('yr').mpg.mean().reset_index()

# Step 2- Make the graph
print("Creating the graph...")

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(auto)
source_avg = ColumnDataSource(mean_mpg)

# Create figure avg_mpg
avg_mpg = figure(x_axis_label='Year', y_axis_label='Mean Miles per Gallon', 
                 x_axis_type="datetime",
                 title='Auto from 1970 to 1982')
avg_mpg.line(x='yr', y='mpg', source=source_avg) # Add a circle glyph to avg_mpg


# Create figure avg_mpg
mpg_hp = figure(x_axis_label='Horse Power', y_axis_label='Miles per Gallon', 
                title='Auto from 1970 to 1982')
mpg_hp.circle('hp', 'mpg', source=source) # Add a circle glyph to mpg_hp


# Create figure avg_mpg
mpg_weight = figure(x_axis_label='Weight', y_axis_label='Miles per Gallon', 
                title='Auto from 1970 to 1982')
mpg_weight.circle('weight', 'mpg', source=source) # Add a circle glyph to mpg_weight


# Put together

# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('02_04_layout_custom.html')
show(layout)

print("****************************************************")
print("** END                                            **")
print("****************************************************")