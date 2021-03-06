# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:00:21 2019

@author: jacqueline.cortez

Capítulo 4. Putting It All Together! A Case Study}
Introduction:
    In this final chapter, you'll build a more sophisticated Bokeh data exploration 
    application from the ground up, based on the famous Gapminder data set.
"""

#How to execute:
#En linea de comando hacer:
#    CD C:\Users\jacqueline.cortez\Documents\Data Science\Python\15 Interactive Data Visualization with Bokeh
#    bokeh serve --show 04_Bokeh_CaseStudy.py


# Import packages
import pandas as pd
#import numpy as np
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
from bokeh.io import curdoc#, output_file, show
from bokeh.plotting import figure#, ColumnDataSource
from bokeh.layouts import row, widgetbox#, column, gridplot
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, Slider, Select#, Button, CheckboxGroup, RadioGroup, Toggle
#from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import Spectral6

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.0f}'.format 
#pd.reset_option("all")

print("Ctrl + C to stop the server")

#print("****************************************************")
#print("** BEGIN                                          **")
#print("****************************************************")
#print("Introducing the project dataset\n")
#print("Some exploratory plots of the data\n")
#print("Beginning with just a plot\n")
#print("Enhancing the plot with some shading\n")
#print("Adding a slider to vary the year\n")
#print("Customizing based on user input\n")
#print("Adding a hover tool\n")
#print("Adding dropdowns to the app\n")

# Step 1 - Define the data 
file = "gapminder_tidy.csv"
gap_minder = pd.read_csv(file, quotechar='"')

year_min, year_max = gap_minder.Year.min(), gap_minder.Year.max() # Save the minimum and maximum values of year

gap_minder.set_index('Year', inplace=True)
source = ColumnDataSource(data={'x'       : gap_minder.loc[year_min].fertility, 
                                'y'       : gap_minder.loc[year_min].life,
                                'country' : gap_minder.loc[year_min].Country,
                                'pop'     : (gap_minder.loc[year_min].population / 20000000) + 2,
                                'region'  : gap_minder.loc[year_min].region
                                })

xmin, xmax = min(gap_minder.fertility), max(gap_minder.fertility) # Save the minimum and maximum values of the fertility column: xmin, xmax
ymin, ymax = min(gap_minder.life), max(gap_minder.life) # Save the minimum and maximum values of the life expectancy column: ymin, ymax

regions_list = gap_minder.region.unique().tolist() # Make a list of the unique values from the region column: regions_list



# Step 2 - Make the graph
hover = HoverTool(tooltips=[('Country:', '@country [@region]'),
                            ('Fertility:', '@x'),
                            ('Life:', '@y'),
                            ('Population:', '@pop Millones')]) # Create a HoverTool object: hover

p = figure(title='Gapminder Data for {}'.format(year_min), 
           #x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)', # Create the figure: p
           x_range=(xmin, xmax), y_range=(ymin, ymax),
           plot_height=400, plot_width=700,
           #tools=[HoverTool(tooltips='@country [@region]')]
           #tools="box_select,reset,save,wheel_zoom,box_zoom,pan"
           tools=[hover, 'box_select', 'reset,save', 'wheel_zoom', 'box_zoom', 'pan'])

p.add_tools(hover) # Add the HoverTool object to figure p

color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6) # Make a color mapper: color_mapper

p.circle(x='x', y='y', fill_alpha=0.8, source=source,
         color=dict(field='region', transform=color_mapper), # Add the color mapper to the circle glyph
         legend='region') # Add a circle glyph to the figure p
p.xaxis.axis_label ='Fertility (children per woman)' # Set the x-axis label
p.yaxis.axis_label = 'Life Expectancy (years)' # Set the y-axis label
p.legend.location = 'top_right' # Set the legend.location attribute of the plot to 'top_right'

slider = Slider(start=year_min, end=year_max, step=1, value=year_min, title='Year') # Make a slider object: slider

x_select = Select(options = ['fertility', 'life', 'child_mortality', 'gdp'], # Create a dropdown Select widget for the x data: x_select
                  value   = 'fertility', 
                  title   = 'x-axis data')
y_select = Select(options = ['fertility', 'life', 'child_mortality', 'gdp'], # Create a dropdown Select widget for the y data: y_select
                  value   = 'life', 
                  title   = 'y-axis data')
r_select = Select(options = ['All'] + regions_list, 
                  value   = 'All',
                  title   = 'Region')

# Step 3 - Define behaives
def update_plot(attr, old, new): # Define the callback function: update_plot
    yr = slider.value # Read the current value off the slider and 2 dropdowns: yr, x, y
    x = x_select.value
    y = y_select.value
    r = r_select.value
    
    p.xaxis.axis_label = x # Label axes of plot
    p.yaxis.axis_label = y
    
    if r == 'All':
        new_data = {'x'       : gap_minder.loc[yr][x],
                    'y'       : gap_minder.loc[yr][y],
                    'country' : gap_minder.loc[yr].Country,
                    'pop'     : (gap_minder.loc[yr].population / 1000000),
                    'region'  : gap_minder.loc[yr].region}

        p.x_range.start = gap_minder[x].min()
        p.x_range.end = gap_minder[x].max()
        p.y_range.start = gap_minder[y].min()
        p.y_range.end = gap_minder[y]. max()

    else:
        new_data = {'x'       : gap_minder[gap_minder.region == r].loc[yr][x],
                    'y'       : gap_minder[gap_minder.region == r].loc[yr][y],
                    'country' : gap_minder[gap_minder.region == r].loc[yr].Country,
                    'pop'     : (gap_minder[gap_minder.region == r].loc[yr].population / 1000000),
                    'region'  : gap_minder[gap_minder.region == r].loc[yr].region}

        p.x_range.start = gap_minder[gap_minder.region == r][x].min()
        p.x_range.end = gap_minder[gap_minder.region == r][x].max()
        p.y_range.start = gap_minder[gap_minder.region == r][y].min()
        p.y_range.end = gap_minder[gap_minder.region == r][y]. max()

    source.data = new_data # Assign new_data to: source.data
    
    
    p.title.text = 'Gapminder data for %d' % yr # Add title to figure: plot.title.text, Regular expressions

slider.on_change('value', update_plot) # Attach the callback to the 'value' property of slider
x_select.on_change('value', update_plot) # Attach the update_plot callback to the 'value' property of x_select
y_select.on_change('value', update_plot) # Attach the update_plot callback to the 'value' property of y_select
r_select.on_change('value', update_plot)

# Step 4 - Create the space to show
#output_file('04_gapminder.html') # Output the file and show the figure
#show(p)
layout = row(widgetbox(slider, x_select, y_select, r_select), p)
curdoc().add_root(layout)
curdoc().title = 'Gapminder'

#layout = Tabs(tabs=[tab1, tab2, tab3, tab4])
#curdoc().add_root(layout)



#print("****************************************************")
#print("** END                                            **")
#print("****************************************************")