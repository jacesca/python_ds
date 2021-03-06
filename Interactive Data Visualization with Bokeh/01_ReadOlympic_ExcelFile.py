# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:21:30 2019

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

file = 'Summer_Olympic_Medallists_1896_to_2008.xls'
data = pd.ExcelFile(file)

print("Hojas del archivo {}: {}.\n".format(file, data.sheet_names))

print("****************************************************")
print("Getting the required data...\n")

# Obtiene los datos de la primer hoja
df = data.parse('ALL MEDALISTS', skiprows=5, header=None, #skipfooter=20,
                 names=['city', 'edition', 'sport', 'discipline', 'athlete',
                        'country', 'gender', 'event', 'event_gender','medal'])

# Explore the data
print(df.head(),'\n')
print(df.info())

print("****************************************************")
print("Saving the dataframe...\n")

# Saving the dataframe
out_file = "Summer_Olympic_Medallists_1896_to_2008.csv"
df.to_csv(out_file)
print("Data saved as {} file.\n".format(out_file))

print("****************************************************")
print("** END                                            **")
print("****************************************************")

"""
# Step 1 - Read the file into a DataFrame named ri
print("Reading the file...")
file = "Summer_Olympic_Medallists_1896_to_2008.csv"
medal = pd.read_csv(file, quotechar='"', index_col=0,
                    dtype={'edition': np.int32})

# Step 2 - Filter and prepare
print("Filtering and preparing the data...")
medal = medal[medal.event=='100m']
medal["color"] = medal.medal.replace(["Gold", "Silver", "Bronze"],["gold", "silver", "brown"]) # Re-encode the color column
"""