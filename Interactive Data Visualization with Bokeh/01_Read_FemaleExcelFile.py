# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:27:37 2019

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

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Reading the excel file...\n")

file = 'Female_Educatiov_vs_Fertility.xls'
data = pd.ExcelFile(file)

print("Hojas del archivo {}: {}.\n".format(file, data.sheet_names))

print("****************************************************")
print("Getting the required data...\n")
# Obtiene los datos de la primer hoja
df = data.parse(0, skiprows=8, header=None, skipfooter=20,
                 names=['country', 'continent', 'fem_literacity', 'fertility', 'population'])

print("Cleaning...\n")

# Write the lambda function using replace to delete annoying comma, Regular expression
df['country'] = df.country.apply(lambda x: x.replace(',', ''))

# Explore the data
print(df.head(),'\n')
print(df.info())

print("****************************************************")
print("Saving the dataframe...\n")

# Saving the dataframe
out_file = "Female_Educatiov_vs_Fertility.csv"
df.to_csv(out_file)
print("Data saved as {} file.\n".format(out_file))

print("****************************************************")
print("** END                                            **")
print("****************************************************")