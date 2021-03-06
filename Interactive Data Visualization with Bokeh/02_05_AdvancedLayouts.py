# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:23:50 2019

@author: jacqueline.cortez

Cap?tulo 2. Layouts, Interactions, and Annotations
Introduction:
    Learn how to combine mutiple Bokeh plots into different kinds of layouts on a page, 
    how to easily link different plots together in various ways, and how to add annotations 
    such as legends and hover tooltips.
Excercise: 05-09
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
#from bokeh.layouts import row, column
from bokeh.layouts import gridplot
from bokeh.models.widgets import Panel, Tabs

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Creating gridded layouts\n")

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

# Create a list containing plots p1 and p2: row1
row1 = [p_lat, p_af]

# Create a list containing plots p3 and p4: row2
row2 = [p_asi, p_eur]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])

# Specify the name of the output_file and show the result
output_file('02_07_grid.html')
show(layout)



print("****************************************************")
print("Displaying tabbed layouts\n")

# Step 1 - Read the file into a DataFrame named ri
# Step 2 - Make the graph

# Create figure p_lat
source_lat = ColumnDataSource(lat) # Convert df to a ColumnDataSource: source
p_lat = figure(x_axis_label='Fertility (children per woman)', 
               y_axis_label='Female literacy (% population)', 
               title='Latin American')
p_lat.circle('fertility', 'fem_literacity', source=source_lat,
             color='green', size=10) # Add a circle glyph to mpg_hp


# Create figure p_af
source_af = ColumnDataSource(af) # Convert df to a ColumnDataSource: source
p_af = figure(x_axis_label='Fertility (children per woman)', 
              y_axis_label='Female literacy (% population)', 
              title='Africa')
p_af.circle('fertility', 'fem_literacity', source=source_af,
            color='brown', size=10) # Add a circle glyph to mpg_hp


# Create figure p_asi
source_asi = ColumnDataSource(asi) # Convert df to a ColumnDataSource: source
p_asi = figure(x_axis_label='Fertility (children per woman)', 
               y_axis_label='Female literacy (% population)', 
               title='Asia')
p_asi.circle('fertility', 'fem_literacity', source=source_asi,
             color='gold', size=10) # Add a circle glyph to mpg_hp


# Create figure p_eur
source_eur = ColumnDataSource(eur) # Convert df to a ColumnDataSource: source
p_eur = figure(x_axis_label='Fertility (children per woman)', 
               y_axis_label='Female literacy (% population)', 
               title='Europe')
p_eur.circle('fertility', 'fem_literacity', source=source_eur,
             color='blue', size=10) # Add a circle glyph to mpg_hp


# Step 3 - Creating the tabs and layouts
tab1 = Panel(child=p_lat, title='Latin America') 
tab2 = Panel(child=p_af, title='Africa')
tab3 = Panel(child=p_asi, title='Asia')
tab4 = Panel(child=p_eur, title='Europe')

layout = Tabs(tabs=[tab1, tab2, tab3, tab4]) # Create a Tabs layout: layout

# Specify the name of the output_file and show the result
output_file('02_09_tabs.html')
show(layout)

print("****************************************************")
print("** END                                            **")
print("****************************************************")