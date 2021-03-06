# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:34:16 2019

@author: jacqueline.cortez

Capítulo 1. Basic plotting with Bokeh
Introduction:
    An introduction to basic plotting with Bokeh. 
    You will create your first plots, learn about different 
    data formats Bokeh understands, and make visual 
    customizations for selections and mouse hovering.
Excercise 15-18
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
from bokeh.models import HoverTool, CategoricalColorMapper

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Selection and non-selection glyphs\n")

# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "100m_Olympic_Medal.csv"
medal = pd.read_csv(file, quotechar='"')

# Step 2- Make the graph
# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Year', y_axis_label='Time', 
           tools='box_select,reset,save,wheel_zoom,box_zoom,pan')

# Create a ColumnDataSource from df: source
source = ColumnDataSource(medal)

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle(x='Year', y='Time', source=source, color='color', size=8, selection_color='red', nonselection_alpha=0.1)

# Specify the name of the output file and show the result
output_file('01_16_box_select.html')
show(p)

print("\n****************************************************")
print("Hover glyphs\n")

# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "glucose.csv"
glucosa = pd.read_csv(file, quotechar='"')

# Step 2- Make the graph
print("Creating the graph...")
# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Time of day', y_axis_label='Blood glucose (mg/dL)', 
           title='Blood glucose', x_axis_type="datetime",
           plot_width=900, plot_height=400)

# Plot date along the x-axis and price along the y-axis
p.line(x=range(288), y=glucosa.glucose, 
       line_dash='dashed', line_color='gray')

# Add circle glyphs to figure p
p.circle(x=range(288), y=glucosa.glucose, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('01_17_hover_glyph.html')
show(p)


print("\n****************************************************")
print("Colormapping\n")

# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "auto-mpg.csv"
auto = pd.read_csv(file, quotechar='"')

# Step 2- Make the graph
print("Creating the graph...")
# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Weights (lb)', y_axis_label='Miler per gallon', 
           title='Auto from 1970 to 1982')

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(auto)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle(x='weight', y='mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('01_18_colormap.html')
show(p)

print("****************************************************")
print("** END                                            **")
print("****************************************************")