# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:25:03 2019

@author: jacqueline.cortez

Capítulo 3. Building interactive apps with Bokeh
Introduction:
    Bokeh server applications let you connect all of the powerful Python libraries 
    for analytics and data science, such as NumPy and Pandas, to rich interactive 
    Bokeh visualizations. Learn about Bokeh's built-in widgets, how to add them to 
    Bokeh documents alongside plots, and how to connect everything to real python 
    code using the Bokeh server.
Excercise: 06-09
"""

#How to execute:
#En linea de comando hacer:
#    CD C:\Users\jacqueline.cortez\Documents\Data Science\Python\15 Interactive Data Visualization with Bokeh
#    bokeh serve --show 03_06_ConnectingSliders.py


# Import packages
#import pandas as pd
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
#from numpy.random import random
from bokeh.io import curdoc#, show, output_file
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import widgetbox, column#, row, gridplot
from bokeh.models import Slider, HoverTool#, CategoricalColorMapper
#from bokeh.models.widgets import Panel, Tabs

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.0f}'.format 
#pd.reset_option("all")

print("Ctrl + C to stop the server")

#print("****************************************************")
#print("** BEGIN                                          **")
#print("****************************************************")
#print("How to combine Bokeh models into layouts\n")

# Step 1 - Define the data
x = np.linspace(0.3, 10, 300)
#y = -3 + x**(1/2) + np.sin(x)
y = np.sin(1/x)
source = ColumnDataSource(data={'x': x, 'y': y}) # Create ColumnDataSource: source

# Step 2 - Make the graph
plot = figure(plot_width=900, plot_height=400)
plot.line('x', 'y', source=source, 
          line_dash='dashed', line_color='gray') # Add a line to the plot
plot.circle('x', 'y', source=source, size=10, fill_color='grey', alpha=0.1, 
            line_color=None, hover_fill_color='firebrick', hover_alpha=0.5,
            hover_line_color='white')

slider = Slider(start=1, end=10, value=1, step=1, title='Scale') # Add circle glyphs to figure plot
hover = HoverTool(tooltips=[('X:', '@x'),
                            ('Y:', '@y')], 
                  mode='vline') # Create a HoverTool: hover
plot.add_tools(hover) # Add the hover tool to the figure p



#print("Learn about widget callbacks\n")

# Step 3 - Define functions
def callback(attr, old, new): # Define a callback function: callback
    scale = slider.value # Read the current value of the slider: scale
    new_y = np.sin(scale/x) # Compute the updated y using np.sin(scale/x): new_y
    source.data = {'x': x, 'y': new_y} # Update source with the new data values
    
slider.on_change('value', callback) # Attach the callback to the 'value' property of slider


# Step 4 - Create the space to show
layout = column(widgetbox(slider), plot) # Create a column layout: layout
curdoc().add_root(layout) # Add the layout to thes current document


#print("****************************************************")
#print("** END                                            **")
#print("****************************************************")