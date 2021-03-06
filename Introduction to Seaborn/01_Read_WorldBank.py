# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:39:47 2019

@author: jacqueline.cortez
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
#import calendar

#from pandas.plotting import register_matplotlib_converters #for conversion as datetime index in x-axis
#from math import radians
#from functools import reduce#import pandas as pd
#from pandas.api.types import CategoricalDtype #For categorical data
#from glob import glob
#from bokeh.io import output_file, show
#from bokeh.plotting import figure

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters() #Require to explicitly register matplotlib converters.

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Initializing variables...\n")
year     = 2015
file_gdp = 'Databank_PhoneGDP_1960-2018.csv'
file_lit = 'Databank_literacy_rate_1475-2015.csv'
file_reg = 'Databank_RegionCountry_ISO3166.csv'


print("****************************************************")
print("Reading the data from the world bank ({})...\n".format(file_gdp))

world_bank = pd.read_csv(file_gdp, header=None, skiprows=1, na_values='NA',
                         skipfooter=5, engine='python', 
                         names=['year','yr','country','country_code','gdp',
                                'mobile_subs', 'mobile_rate'],
                         index_col='country_code')

# Delete rows with null values
world_bank.dropna(axis=0, how='any', inplace=True)

print("Getting data for {} year...\n".format(year))
world_bank = world_bank[world_bank.year == year]

print(world_bank.info())

print("\n****************************************************")
print("Reading the data from UNESCO ({})...\n".format(file_lit))

unesco = pd.read_csv(file_lit, header=None, skiprows=1, na_values='NA',
                     names=['Country','country_code','Year','literacy_rate'],
                     index_col='country_code')

# Delete rows with null values
unesco.dropna(axis=0, how='any', inplace=True)

print('Getting the last value avaliable\n')
unesco = unesco.groupby('country_code').last()

print(unesco.info())


print("\n****************************************************")
print("Reading the data from ISO ({})...\n".format(file_reg))

region = pd.read_csv(file_reg, header=None, skiprows=1, 
                     names=['country_name', 'c2', 'country_code', 'n3', 'iso', 
                            'region', 'sub_region', 'ir', 'rc', 'src', 'irc'],
                     usecols=['country_name', 'country_code', 'region', 'sub_region'],
                     index_col='country_code')
print(region.info())


print("\n****************************************************")
print("Getting together...\n")

world = pd.merge(world_bank, unesco, on='country_code')
world = pd.merge(world, region, on='country_code')

world.drop(['yr', 'Country', 'Year', 'country_name'], axis='columns', inplace=True)

print(world.info())

print("\n****************************************************")
print("Saving the data...\n")

out_file = "Databank_%d.csv" % year
world.to_csv(out_file, index=False)
print(world.head())
print("columns: {}".format(world.columns.tolist()))
print("Data saved as {} file.\n".format(out_file))

print("****************************************************")
print("** END                                            **")
print("****************************************************")