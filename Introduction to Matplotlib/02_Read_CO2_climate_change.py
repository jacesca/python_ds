# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:53:27 2019

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

#from pandas.plotting import register_matplotlib_converters
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
print("** Read data with a time index\n")

print("Reading global CO2 emissions...\n")
file = 'CO2_globales_0-2014.csv'
co2 = pd.read_csv(file, parse_dates = [['year', 'month', 'day']], 
                  skiprows = range(1,22201),
                  usecols=['year', 'month', 'day', 'data_mean_global'],
                  index_col=0)
#data_mean_nh --> north hemisphere
#data_mean_sh --> south hemisphere

print("Reading global temperatures anomalies...\n")
file = 'CO2_temperature_1850-1919.txt'
temp = pd.read_fwf(file, header = None,
                   names = ['ym', 'temp', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                   usecols = ['ym', 'temp'],
                   skipfooter = 50)
temp['year'] = temp.ym.str.split("/").str.get(0)
temp['month'] = temp.ym.str.split("/").str.get(1)
temp['date'] = pd.to_datetime(temp.year+'-'+temp.month+'-'+'15')
temp.set_index('date', inplace=True)
temp.drop(["ym",'year','month'], axis='columns', inplace=True)

print("Concatening the two previews dataframes...")
climate_change = pd.concat([co2, temp], axis=1)
climate_change.columns = ['co2','relative_temp']
print(climate_change.head())

out_file = "CO2_climate_change_1850-2014.csv"
climate_change.to_csv(out_file)
print(climate_change.head())
print("columns: {}".format(climate_change.columns.tolist()))
print("Data saved as {} file.\n".format(out_file))

print("****************************************************")
print("** END                                            **")
print("****************************************************")