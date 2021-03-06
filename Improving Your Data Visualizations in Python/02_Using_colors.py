# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:18:02 2019

@author: jacqueline.cortez

Capítulo 2. Using color in your visualizations
Introduction:
    Color is a powerful tool for encoded values in data visualization. However, with this power comes danger. 
    In this chapter, we talk about how to choose an appropriate color palette for your visualization based upon 
    the type of data it is showing.
"""

# Import packages
import pandas as pd                   #For loading tabular data
import numpy as np                    #For making operations in lists
#import matplotlib as mpl              #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt       #For creating charts
import seaborn as sns                 #For visualizing data
#import scipy.stats as stats          #For accesign to a vary of statistics functiosn
#import statsmodels as sm             #For stimations in differents statistical models
#import scykit-learn                  #For performing machine learning  
#import tabula                        #For extracting tables from pdf
#import nltk                          #For working with text data
#import math                          #For accesing to a complex math operations
#import random                        #For generating random numbers
#import calendar                      #For accesing to a vary of calendar operations
#import re                             #For regular expressions

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                        #For obteining today function
#from string import Template                                                          #For working with string, regular expressions

#from bokeh.io import curdoc, output_file, show                                      #For interacting visualizations
#from bokeh.plotting import figure, ColumnDataSource                                 #For interacting visualizations
#from bokeh.layouts import row, widgetbox, column, gridplot                          #For interacting visualizations
#from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper        #For interacting visualizations
#from bokeh.models import Slider, Select, Button, CheckboxGroup, RadioGroup, Toggle  #For interacting visualizations
#from bokeh.models.widgets import Tabs, Panel                                        #For interacting visualizations
#from bokeh.palettes import Spectral6                                                #For interacting visualizations

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters() #Require to explicitly register matplotlib converters.

#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09

#Setting the numpy options
#np.set_printoptions(precision=3) #precision set the precision of the output:
#np.set_printoptions(suppress=True) #suppress suppresses the use of scientific notation for small numbers

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

file = "pollution_wide.csv" 
pollution = pd.read_csv(file)


cinci_2014 = pollution.query("city  ==  'Cincinnati' & year  ==  2014") # Filter the data


nov_pollution = pollution[(pollution.year==2015) & (pollution.month==11)][['city','day', 'CO']].copy()
#mean_nov_CO = nov_pollution.groupby('day').CO.mean()
#nov_pollution = pd.merge(nov_pollution, mean_nov_CO, how='left', on='day', suffixes=('_city','_mean'))
#nov_pollution['CO'] = nov_pollution.CO_city - nov_pollution.CO_mean
mean_nov_CO = nov_pollution.CO.mean()
desv_nov_CO = nov_pollution.CO.std()
nov_pollution['day'] = nov_pollution.day - 305
nov_pollution['CO'] = (nov_pollution.CO - mean_nov_CO)/desv_nov_CO
nov_2015_CO = pd.pivot_table(nov_pollution, values='CO', index='city', columns='day')


oct_pollution = pollution[(pollution.year==2015) & (pollution.month==10)][['city','day', 'O3']].copy()
#mean_nov_CO = nov_pollution.groupby('day').CO.mean()
#nov_pollution = pd.merge(nov_pollution, mean_nov_CO, how='left', on='day', suffixes=('_city','_mean'))
#nov_pollution['CO'] = nov_pollution.CO_city - nov_pollution.CO_mean
mean_oct_o3 = oct_pollution.O3.mean()
desv_oct_o3 = oct_pollution.O3.std()
oct_pollution['day'] = oct_pollution.day - 274
oct_pollution['O3'] = (oct_pollution.O3 - mean_oct_o3)/desv_oct_o3
oct_2015_o3 = pd.pivot_table(oct_pollution, values='O3', index='city', columns='day')


pollution_jan13 = pollution.query('year  ==  2013 & month  ==  1') # Filter our data to Jan 2013

#pollution['Date'] = pd.to_datetime((pollution.year*1000+pollution.day).apply(str), format='%Y%j')
#city_pol_month = pollution.set_index('Date')
#city_pol_month = city_pol_month[['CO','NO2','O3','SO2']].resample('M').mean()

#city_pol_CO = pollution.groupby(['city','month']).CO.mean().reset_index()
#city_pol_CO['city'] = city_pol_CO.city + " CO"
#mu = city_pol_CO.CO.mean()
#sigma = city_pol_CO.CO.std()
#city_pol_CO['CO'] =  (city_pol_CO.CO - mu)/sigma
#city_pol_CO.columns = ['city_pol', 'month', 'value']
#city_pol_NO2 = pollution.groupby(['city','month']).NO2.mean().reset_index()
#city_pol_NO2['city'] = city_pol_NO2.city + " NO2"
#mu = city_pol_NO2.NO2.mean()
#sigma = city_pol_NO2.NO2.std()
#city_pol_NO2['NO2'] =  (city_pol_NO2.NO2 - mu)/sigma
#city_pol_NO2.columns = ['city_pol', 'month', 'value']
#city_pol_O3 = pollution.groupby(['city','month']).O3.mean().reset_index()
#city_pol_O3['city'] = city_pol_O3.city + " O3"
#mu = city_pol_O3.O3.mean()
#sigma = city_pol_O3.O3.std()
#city_pol_O3['O3'] =  (city_pol_O3.O3 - mu)/sigma
#city_pol_O3.columns = ['city_pol', 'month', 'value']
#city_pol_SO2 = pollution.groupby(['city','month']).SO2.mean().reset_index()
#city_pol_SO2['city'] = city_pol_SO2.city + " SO2"
#mu = city_pol_SO2.SO2.mean()
#sigma = city_pol_SO2.SO2.std()
#city_pol_SO2['SO2'] =  (city_pol_SO2.SO2 - mu)/sigma
#city_pol_SO2.columns = ['city_pol', 'month', 'value']
#city_pol_month = city_pol_CO.append(city_pol_NO2).append(city_pol_O3).append(city_pol_SO2)

def city_pol_pd(df, col):
    city_df = df.groupby(['city','month'])[col].mean().reset_index()
    city_df['city'] = city_df.city + " " + col
    mu = city_df[col].mean()
    sigma = city_df[col].std()
    city_df[col] =  (city_df[col] - mu)/sigma
    city_df.columns = ['city_pol', 'month', 'value']
    return city_df


city_pol_month = pd.DataFrame(columns=['city_pol', 'month', 'value'])
for col in ['CO', 'NO2', 'O3', 'SO2']:
    city_pol_month = city_pol_month.append(city_pol_pd(pollution, col))


max_pollutant_values = pollution[['city', 'year', 'month', 'day', 'CO', 'NO2', 'O3', 'SO2']]
max_pollutant_values = max_pollutant_values.groupby(['city', 'year', 'month'])[['CO', 'NO2', 'O3', 'SO2']].max().reset_index()
max_pollutant_values = pd.melt(max_pollutant_values, id_vars=['city', 'year', 'month'], var_name="pollutant", value_vars=['CO', 'NO2', 'O3', 'SO2'])


print("****************************************************")
tema = '2. Getting rid of unnecessary color'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
# Hard to read scatter of CO and NO2 w/ color mapped to city
sns.scatterplot('CO', 'NO2', alpha = 0.2, hue = 'city', data = pollution)
plt.title("Relationship between CO to NO2 values across cities (1/2)")
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


sns.set() # Set default Seaborn style
#plt.figure()
# Setup a facet grid to separate the cities apart
sns.set(font_scale=0.8)
#sns.plotting_context(font_scale=1.5)
g = sns.FacetGrid(data = pollution, col = 'city', col_wrap = 3)
# Map sns.scatterplot to create separate city scatter plots
g.map(sns.scatterplot, 'CO', 'NO2', alpha = 0.2)
plt.suptitle("{}\nRelationship between CO to NO2 values across cities (2/2)".format(tema))
plt.subplots_adjust(left=None, bottom=0.10, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "3. Fixing Seaborn's bar charts"; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
sns.set(font_scale=0.8)
plt.figure()
sns.barplot(y = 'city', x = 'CO', estimator = np.mean, ci = False, data = pollution)
plt.title("Level of CO across cities (1/3)")
plt.suptitle(tema)
plt.subplots_adjust(left=0.32, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


sns.set() # Set default Seaborn style
sns.set(font_scale=0.8)
plt.figure()
sns.barplot(y = 'city', x = 'CO', estimator = np.mean, ci = False, data = pollution, edgecolor = 'black')
plt.title("Level of CO across cities (2/3)")
plt.suptitle(tema)
plt.subplots_adjust(left=0.32, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


sns.set() # Set default Seaborn style
sns.set(font_scale=0.8)
plt.figure()
sns.barplot(y = 'city', x = 'CO', estimator = np.mean, ci = False, data = pollution, color = 'cadetblue')
plt.title("Level of CO across cities (3/3)")
plt.suptitle(tema)
plt.subplots_adjust(left=0.32, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "4. Continuous color palettes"; print("** %s\n" % tema)

blue_scale = sns.light_palette('steelblue')
sns.palplot(blue_scale)
plt.title("Steelblue Light Palette")
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.55, wspace=None, hspace=None)
plt.show()

red_scale = sns.dark_palette('orangered')
sns.palplot(red_scale)
plt.title("Orangered Dark Palette")
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.55, wspace=None, hspace=None)
plt.show()


print("****************************************************")
tema = "5. Making a custom continuous palette"; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
#sns.set(font_scale=0.8)
plt.figure()
color_palette = sns.light_palette('orangered', as_cmap = True) # Define a custom continuous color palette
sns.scatterplot(x = 'CO', y = 'NO2', hue = 'O3', data = cinci_2014, palette = color_palette) # Plot mapping the color of the points with custom palette
plt.title("Pollution in Cincinnati, 2014")
plt.suptitle(tema)
#plt.subplots_adjust(left=0.32, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = "6. Customizing a diverging palette heatmap"; print("** %s\n" % tema)

print("Min value in Nov 2015: {}".format(nov_2015_CO.min().min()))
print("Max value in Nov 2015: {}".format(nov_2015_CO.max().max()))

sns.set() # Set default Seaborn style
#sns.set(font_scale=0.8)
plt.figure()
color_palette = sns.diverging_palette(250, 0, as_cmap = True) # Define a custom palette
sns.heatmap(nov_2015_CO, cmap = color_palette, center = 0, vmin = -4, vmax = 4) # Pass palette to plot and set axis ranges
plt.yticks(rotation = 0)
plt.tick_params(labelsize = 9)
plt.title("Level of CO in November 2015")
plt.suptitle(tema)
plt.subplots_adjust(left=0.35, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "7. Adjusting your palette according to context"; print("** %s\n" % tema)

print("Min value in Oct 2015: {}".format(oct_2015_o3.min().min()))
print("Max value in Oct 2015: {}".format(oct_2015_o3.max().max()))

sns.set() # Set default Seaborn style
#sns.set(font_scale=0.8)
fig = plt.figure()
fig.patch.set_facecolor('xkcd:black')
plt.style.use("dark_background") # Dark plot background
color_palette = sns.diverging_palette(250, 0, center = 'dark', as_cmap = True) # Modify palette for dark background
sns.heatmap(oct_2015_o3, cmap = color_palette, center = 0) # Pass palette to plot and set center
plt.yticks(rotation = 0)
plt.tick_params(labelsize = 9)
plt.title("Level of O3 in October 2015")
plt.suptitle(tema)
plt.subplots_adjust(left=0.35, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "9. Using a custom categorical palette"; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
#sns.set(font_scale=0.8)
plt.figure()
sns.lineplot(x = "day", y = "CO", hue = "city", palette = "Set2", linewidth = 3, data = pollution_jan13) # Color lines by the city and use custom ColorBrewer palette
plt.title("Level of CO in January 2013")
plt.suptitle(tema)
#plt.subplots_adjust(left=0.35, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "10. Dealing with too many categories"; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
#sns.set(font_scale=0.8)
plt.figure()
wanted_combos = ['Vandenberg Air Force Base NO2', 'Long Beach CO', 'Cincinnati SO2'] # Choose the combos that get distinct colors
city_pol_month['color_cats'] = [x if x in wanted_combos else 'other' for x in city_pol_month['city_pol']] # Assign a new column to DataFrame for isolating the desired combos
sns.lineplot(x = "month", y = "value", hue = 'color_cats', units = 'city_pol', estimator = None, palette = 'Set2', data = city_pol_month) # Plot lines with color driven by new column and lines driven by original categories
plt.title("Pollutant trajectory along the year")
plt.suptitle(tema)
#plt.subplots_adjust(left=0.35, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = "11. Coloring ordinal categories"; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
#sns.set(font_scale=0.8)
plt.figure()
pollution['CO quartile'] = pd.qcut(pollution['CO'], q = 4, labels = False) # Divide CO into quartiles
des_moines = pollution.query("city  ==  'Des Moines'") # Filter to just Des Moines
sns.scatterplot(x = 'SO2', y = 'NO2', hue = 'CO quartile', data = des_moines, palette = 'GnBu') # Color points with by quartile and use ColorBrewer palette
plt.title("Pollutant in Des Moines")
plt.suptitle(tema)
#plt.subplots_adjust(left=0.35, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = "12. Choosing the right variable to encode with color"; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
#sns.set(font_scale=0.8)
cities = ['Fairbanks', 'Long Beach', 'Vandenberg Air Force Base', 'Denver']
city_maxes = max_pollutant_values[max_pollutant_values.city.isin(cities)] # Filter data to desired cities
sns.catplot(x = 'city', hue = 'year', y = 'value', row = 'pollutant', data = city_maxes, palette = 'BuGn', sharey = False, kind = 'bar', ci=None) # Swap city and year encodings
plt.tick_params(labelsize = 7)
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.90, wspace=None, hspace=0.5)
plt.show()
plt.style.use('default')


print("****************************************************")
print("** END                                            **")
print("****************************************************")