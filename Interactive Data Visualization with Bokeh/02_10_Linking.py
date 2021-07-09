# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:23:50 2019

@author: jacqueline.cortez

Capítulo 2. Layouts, Interactions, and Annotations
Introduction:
    Learn how to combine mutiple Bokeh plots into different kinds of layouts on a page, 
    how to easily link different plots together in various ways, and how to add annotations 
    such as legends and hover tooltips.
Excercise: 11-12
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
from bokeh.layouts import row#, column
from bokeh.layouts import gridplot
#from bokeh.models.widgets import Panel, Tabs

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Linked axes\n")

# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "Female_Educatiov_vs_Fertility.csv"
female = pd.read_csv(file, quotechar='"')


# Step 2 - Make the graph
print("Creating the graph...")

# Create figure p_lat
lat = female[female.continent=='LAT'] #Get the data
source_lat = ColumnDataSource(lat) # Convert df to a ColumnDataSource: source
p_lat = figure(x_axis_label='Fertility (children per woman)', 
               y_axis_label='Female literacy (% population)', 
               plot_width=250, plot_height=250,
               title='Latin American')
p_lat.circle('fertility', 'fem_literacity', source=source_lat) # Add a circle glyph to mpg_hp


# Create figure p_af
af = female[female.continent=='AF'] #Get the data
source_af = ColumnDataSource(af) # Convert df to a ColumnDataSource: source
p_af = figure(x_axis_label='Fertility (children per woman)', 
              y_axis_label='Female literacy (% population)', 
              plot_width=250, plot_height=250,
              title='Africa')
p_af.circle('fertility', 'fem_literacity', source=source_af) # Add a circle glyph to mpg_hp


# Create figure p_asi
asi = female[female.continent=='ASI'] #Get the data
source_asi = ColumnDataSource(asi) # Convert df to a ColumnDataSource: source
p_asi = figure(x_axis_label='Fertility (children per woman)', 
               y_axis_label='Female literacy (% population)', 
               plot_width=250, plot_height=250,
               title='Asia')
p_asi.circle('fertility', 'fem_literacity', source=source_asi) # Add a circle glyph to mpg_hp


# Create figure p_eur
eur = female[female.continent=='EUR'] #Get the data
source_eur = ColumnDataSource(eur) # Convert df to a ColumnDataSource: source
p_eur = figure(x_axis_label='Fertility (children per woman)', 
               y_axis_label='Female literacy (% population)', 
               plot_width=250, plot_height=250,
               title='Europe')
p_eur.circle('fertility', 'fem_literacity', source=source_eur) # Add a circle glyph to mpg_hp


# Step 3 - Configure the grid
row1 = [p_lat, p_af] # Create a list containing plots p1 and p2: row1
row2 = [p_asi, p_eur] # Create a list containing plots p3 and p4: row2
layout = gridplot([row1, row2]) # Create a gridplot using row1 and row2: layout



# Step 4 - Link the plots
p_af.x_range = p_lat.x_range # Link the x_range of p2 to p1: p2.x_range
p_af.y_range = p_lat.y_range # Link the y_range of p2 to p1: p2.y_range
p_asi.x_range = p_lat.x_range
p_asi.y_range = p_lat.y_range
p_eur.x_range = p_lat.x_range
p_eur.y_range = p_lat.y_range


# Step 5 - Specify the name of the output_file and show the result
output_file('02_11_linked_range.html')
show(layout)

print("****************************************************")
print("Linked brushing\n")

# Step 1 - Read the file into a DataFrame named ri
source = ColumnDataSource(female) # Create ColumnDataSource: source

# Step 2 - Make the graph
print("Creating the graph...")


# Create the first figure: p1
p1 = figure(x_axis_label='Fertility (children per woman)', y_axis_label='female literacy (% population)',
            title='Fertility vs Female Literacy', 
            tools='box_select,lasso_select,reset,save,wheel_zoom,box_zoom,pan')
p1.circle('fertility', 'fem_literacity', source=source,
          selection_color='red',
          nonselection_fill_alpha=0.2,
          nonselection_fill_color='gray')

# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            title='Fertility vs Population', 
            tools='box_select,lasso_select,reset,save,wheel_zoom,box_zoom,pan')
p2.circle('fertility', 'population', source=source,
          selection_color='red',
          nonselection_fill_alpha=0.2,
          nonselection_fill_color='gray')


# Step 3 - Make the Layout
layout = row([p1,p2]) # Create row layout of figures p1 and p2: layout

# Step 4 - Link the plots
p2.x_range = p1.x_range # Link the x_range of p2 to p1: p2.x_range

# Step 5 - Specify the name of the output_file and show the result
output_file('02_12_linked_brush.html') # Specify the name of the output_file and show the result
show(layout)

print("****************************************************")
print("** END                                            **")
print("****************************************************")