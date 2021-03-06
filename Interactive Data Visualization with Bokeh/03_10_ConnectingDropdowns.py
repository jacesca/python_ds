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
Excercise: 10-15
"""

#How to execute:
#En linea de comando hacer:
#    CD C:\Users\jacqueline.cortez\Documents\Data Science\Python\15 Interactive Data Visualization with Bokeh
#    bokeh serve --show 03_10_ConnectingDropdowns.py


# Import packages
import pandas as pd
import numpy as np
#import tabula 
#import math
#import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
#import random, normal, lognormal
#import json

#from math import radians
#from functools import reduce#import pandas as pd
#from pandas.api.types import CategoricalDtype #For categorical data
#from glob import glob
#from numpy.random import random
from bokeh.io import curdoc#, show, output_file
from bokeh.plotting import figure#, ColumnDataSource
from bokeh.layouts import row, widgetbox, column#, gridplot
from bokeh.models import ColumnDataSource, Select, Button, CheckboxGroup, RadioGroup, Toggle#, Slider, HoverTool, CategoricalColorMapper
from bokeh.models.widgets import Tabs, Panel

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.0f}'.format 
#pd.reset_option("all")

print("Ctrl + C to stop the server")

#print("****************************************************")
#print("** BEGIN                                          **")
#print("****************************************************")
#print("Updating data sources from dropdown callbacks\n")

# Step 1 - Define the data 
file = "Female_Educatiov_vs_Fertility.csv"
female = pd.read_csv(file, quotechar='"')
female['pop'] = female.population.apply('{:,.0f} hab.'.format)

fertility = female['fertility'] #Get the data
female_literacy = female['fem_literacity'] #Get the data
population = female['population']

source = ColumnDataSource(data={'x' : fertility, 'y' : female_literacy}) # Create ColumnDataSource: source


# Step 2 - Make the graph
plot = figure() # Create a new plot: plot
plot.circle('x', 'y', source=source)# Add circles to the plot


# Step 3 - Define behaives
def update_plot(attr, old, new):
    if new == 'female_literacy': 
        source.data = {'x' : fertility, 'y' : female_literacy}
    else:
        source.data = {'x' : fertility, 'y' : population}

select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy') # Create a dropdown Select widget: select    
select.on_change('value', update_plot) # Attach the update_plot callback to the 'value' property of select

tab1 = Panel(child=row(select, plot), title='Dropdown callbacks')


#print("****************************************************")
#print("Updating data sources from dropdown callbacks\n")

# Step 1 - Define the data 
# Step 2 - Make the graph
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Step 3 - Define behaives
def callback(attr, old, new):
    if select1.value == "A":
        select2.options = ['1', '2', '3']
        select2.value = '1'
    else:
        select2.options = ['100', '200', '300']
        select2.value = '100'

select1.on_change('value', callback)

tab2 = Panel(child=widgetbox(select1, select2), title='Updating data')


#print("****************************************************")
#print("Updating data sources from dropdown callbacks\n")

# Step 1 - Define the data 
N = 200
x = np.linspace(0, 10, 200)
y = np.sin(x)
source = ColumnDataSource(data={'x' : x, 'y' : y}) # Create ColumnDataSource: source


# Step 2 - Make the graph
plot = figure() # Create a new plot: plot
plot.circle('x', 'y', source=source)# Add circles to the plot

button = Button(label='Update Data') # Create a Button with label 'Update Data'


# Step 3 - Define behaives
def update():# Define an update callback with no arguments: update
    y = np.sin(x) + np.random.random(N) # Compute new y values: y
    source.data = {'x':x, 'y':y} # Update the ColumnDataSource data dictionary
    
button.on_click(update) # Add the update callback to the button

tab3 = Panel(child=column(widgetbox(button), plot), title='Button widgets')


#print("****************************************************")
#print("Button styles\n")

# Step 1 - Define the data 
# Step 2 - Make the graph
toggle = Toggle(button_type='success', label='Toggle button') # Add a Toggle: toggle
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3']) # Add a CheckboxGroup: checkbox
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3']) # Add a RadioGroup: radio


# Step 3 - Define behaives
tab4 = Panel(child=widgetbox(toggle, checkbox, radio), title='Button styles')


#print("****************************************************")

# Step 4 - Create the space to show
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])
curdoc().add_root(layout)



#print("****************************************************")
#print("** END                                            **")
#print("****************************************************")