# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:36:43 2019

@author: jacqueline.cortez

Capítulo 3. Building interactive apps with Bokeh
Introduction:
    Bokeh server applications let you connect all of the powerful Python libraries 
    for analytics and data science, such as NumPy and Pandas, to rich interactive 
    Bokeh visualizations. Learn about Bokeh's built-in widgets, how to add them to 
    Bokeh documents alongside plots, and how to connect everything to real python 
    code using the Bokeh server.
Excercise: 01-05
"""
#How to execute:
#En linea de comando hacer:
#    CD C:\Users\jacqueline.cortez\Documents\Data Science\Python\15 Interactive Data Visualization with Bokeh
#    bokeh serve --show 03_01_BokehServer.py


# Import packages
#import pandas as pd
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
from bokeh.io import curdoc#, show, output_file
from bokeh.plotting import figure#, ColumnDataSource
#from bokeh.models import HoverTool, CategoricalColorMapper
#from bokeh.layouts import row, column
#from bokeh.layouts import gridplot
from bokeh.layouts import widgetbox
#from bokeh.models.widgets import Panel, Tabs
from bokeh.models import Slider

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.0f}'.format 
#pd.reset_option("all")

print("Ctrl + C to stop the server")

#print("****************************************************")
#print("** BEGIN                                          **")
#print("****************************************************")
#print("Using the current document\n")

plot = figure() # Create a new plot: plot
plot.line(x=[1,2,3,4,5], y=[2,5,4,6,7]) # Add a line to the plot

curdoc().add_root(plot) # Add the plot to the current document


#print("****************************************************")
#print("Add a single slider\n")

slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2) # Create a slider: slider
layout = widgetbox(slider) # Create a widgetbox layout: layout

curdoc().add_root(layout)# Add the layout to the current document



#print("****************************************************")
#print("Multiple sliders in one document\n")

slider1 = Slider(title='slider1', start=0, end=10, step=0.1, value=2) # Create first slider: slider1
slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20) # Create second slider: slider2

layout = widgetbox(slider1, slider2) # Add slider1 and slider2 to a widgetbox
curdoc().add_root(layout)# Add the layout to the current document



#print("****************************************************")
#print("** END                                            **")
#print("****************************************************")