# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:23:50 2019

@author: jacqueline.cortez

Capítulo 2. Layouts, Interactions, and Annotations
Introduction:
    Learn how to combine mutiple Bokeh plots into different kinds of layouts on a page, 
    how to easily link different plots together in various ways, and how to add annotations 
    such as legends and hover tooltips.
Excercise: 13-17
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
from bokeh.models import HoverTool#, CategoricalColorMapper
#from bokeh.layouts import row, column
#from bokeh.layouts import gridplot
#from bokeh.models.widgets import Panel, Tabs

# Setting the pandas options
#pd.set_option("display.max_columns",20)
pd.options.display.float_format = '{:,.0f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("How to create legends\n")

# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "Female_Educatiov_vs_Fertility.csv"
female = pd.read_csv(file, quotechar='"')
female['pop'] = female.population.apply('{:,.0f} hab.'.format)

lat = female[female.continent=='LAT'] #Get the data
af = female[female.continent=='AF'] #Get the data

source_lat = ColumnDataSource(lat) # Convert df to a ColumnDataSource: source
source_af = ColumnDataSource(af) # Convert df to a ColumnDataSource: source

# Step 2 - Make the graph
print("Making the plots...\n")
p = figure(x_axis_label='Fertility (children per woman)', 
           y_axis_label='Female literacy (% population)', 
           title='Latin American vs Africa')

p.circle('fertility', 'fem_literacity', source=source_lat, 
         size=10, color='green', legend='Latin America') # Add the first circle glyph to the figure p

p.circle('fertility', 'fem_literacity', source=source_af, 
         size=10, color='brown', legend='Africa') # Add the second circle glyph to the figure p


# Step 3 - Show the graph
print("Showing the graph...\n")
output_file('02_14_fert_lit_groups.html') # Specify the name of the output_file 
show(p)# Show the result




print("****************************************************")
print("Positioning and styling legends\n")

# Step 1 - Read the file into a DataFrame named ri
print("Getting the data...")
source_lat = ColumnDataSource(lat) # Convert df to a ColumnDataSource: source
source_af = ColumnDataSource(af) # Convert df to a ColumnDataSource: source


# Step 2 - Make the graph
print("Making the plots...\n")
p = figure(x_axis_label='Fertility (children per woman)', 
           y_axis_label='Female literacy (% population)', 
           title='Latin American vs Africa')

p.circle('fertility', 'fem_literacity', source=source_lat, 
         size=10, color='green', legend='Latin America') # Add the first circle glyph to the figure p

p.circle('fertility', 'fem_literacity', source=source_af, 
         size=10, color='brown', legend='Africa') # Add the second circle glyph to the figure p

p.legend.location = 'bottom_left' # Assign the legend to the bottom left: p.legend.location
p.legend.background_fill_color = 'lightgray' # Fill the legend background with the color 'lightgray': p.legend.background_fill_color


# Step 3 - Configure the grid
print("Showing the graph...\n")
output_file('02_15_fert_lit_groups.html')
show(p)


print("****************************************************")
print("Adding a hover tooltip\n")

# Step 1 - Read the file into a DataFrame named ri
print("Getting the data...")
source_lat = ColumnDataSource(lat) # Convert df to a ColumnDataSource: source
source_af = ColumnDataSource(af) # Convert df to a ColumnDataSource: source



# Step 2 - Make the graph
print("Making the plots...\n")
p = figure(x_axis_label='Fertility (children per woman)', 
           y_axis_label='Female literacy (% population)', 
           title='Latin American vs Africa')

p.circle('fertility', 'fem_literacity', source=source_lat, 
         size=10, color='green', legend='Latin America') # Add the first circle glyph to the figure p

p.circle('fertility', 'fem_literacity', source=source_af, 
         size=10, color='brown', legend='Africa') # Add the second circle glyph to the figure p

hover = HoverTool(tooltips=[('Country:', '@country (@continent)'),
                            ('Population:', '@pop')]) # Create a HoverTool object: hover
p.add_tools(hover) # Add the HoverTool object to figure p

p.legend.location = 'bottom_left' # Assign the legend to the bottom left: p.legend.location
p.legend.background_fill_color = 'lightgray' # Fill the legend background with the color 'lightgray': p.legend.background_fill_color


# Step 3 - Show the graph
output_file('02_17_hover.html')
show(p)

print("****************************************************")
print("** END                                            **")
print("****************************************************")