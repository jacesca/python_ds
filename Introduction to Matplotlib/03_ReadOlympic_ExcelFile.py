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
print("Reading the excel file: Summer Olympic Medallists from 1896 to 2008...\n")

file = 'Summer_Olympic_Medallists_1896_to_2008.xls'
data = pd.ExcelFile(file)

print("Hojas del archivo {}: {}.\n".format(file, data.sheet_names))

print("Getting the required data...\n")

# Obtiene los datos de la tercera hoja
df = data.parse('ALL MEDALISTS', skiprows=5, header=None, #skipfooter=20,
                 names=['city', 'edition', 'sport', 'discipline', 'athlete',
                        'NOC', 'gender', 'event', 'event_gender','medal'])
# Explore the data
print(df.head(),'\n')

#Create categorical type data to use
#cats = CategoricalDtype(categories=['Gold', 'Silver', 'Bronze'],ordered=True)
# Change the data type of 'rating' to category
#df['medal'] = df.medal.astype(cats)

# Obtiene los datos de la quinta hoja
df_country = data.parse('COUNTRY TOTALS', skiprows=2, skipfooter=145,
                        usecols=[0,1])
df_country.columns = ['NOC','country']

# Fix missing 'NOC' values of hosts
print("Fixing missing NOC values...")
df_country.loc[101, 'country'] = 'Russian Empire' #for NOC = 'RU1'

# Concatenating to get the country name
df = pd.merge(df,df_country, on='NOC', how='left')

print("\nFinal result...")
print(df.info())

print("****************************************************")
print("Reading the excel file: 2012 London Olympic Medallist...\n")

file = 'All London 2012 athletes.xlsx'
data = pd.ExcelFile(file)

print("Hojas del archivo {}: {}.\n".format(file, data.sheet_names))

print("Getting the required data...\n")

# Obtiene los datos de la tercera hoja
df2012 = data.parse('ALL ATHLETES', skiprows=1, header=None, #skipfooter=20,
                    names=['name', 'country', 'age', 'height', 'weight', 'sex', 'birth', 'nacionality', 
                           'gold', 'silver', 'bronze', 'total', 'sport', 'event'])
# Explore the data
print(df2012.head(),'\n')

# Dropping rows with 0 medals
print("Dropping rows with 0 medals...")
df2012.drop(index = df2012[df2012.total==0].index, inplace=True)
print("Rows with 0 medals won: {}\n".format(df2012[df2012.total==0].shape[0]))

# Filling height missing values
print("Filling height missing values...\n")
missing_values = {'Alexander Kristoff': 78, 'Yanet Bermoy Acosta': 48, 'Ivan Cambar Rodriguez': 77, 
                  'Asley Gonzalez': 179, 'Vijay Kumar': 170, 'Gagan Narang': 180, 
                  'Idalys Ortiz': 180, 'Bartosz Piasecki': 196, 'Leuris Pupo': 168,
                  'Valent Sinkovic': 184}
df2012['height'] = df2012['name'].map(missing_values).fillna(df2012['height'])

# Filling weight missing values
# Don' found (using aprox. values): Diana Maria Chelaru
print("Filling weight missing values...\n")
missing_values = {'Kseniia Afanaseva': 48, 'Alexander Kristoff': 78, 'Diana Laura Bulimar': 32,
                  'Diana Maria Chelaru': 57, 'Yibing Chen': 58, 'Catalina Ponor': 51,
                  'Dong Dong': 57, 'Gabrielle Douglas': 50, 'Zhe Feng': 58,
                  'Anastasia Grishina': 37, 'Larisa Andreea Iordache': 37, 'Sandra Raluca Izbasa': 52,
                  'Ryohei Kato': 54, 'Victoria Komova': 48, 'Vijay Kumar': 70,
                  'Danell Leyva': 61, 'Chunlong Lu': 57, 'Mc Kayla Maroney': 46,
                  'Aliya Mustafina': 48, 'Gagan Narang': 115, 'Marcel Nguyen': 55,
                  'Sam Oldham': 62, 'Maria Paseka': 48, 'Bartosz Piasecki': 80,
                  'Leuris Pupo': 78, 'Daniel Purvis': 63, 'Kyla Ross': 45,
                  'Valent Sinkovic': 93, 'Louis Smith': 76, 'Kazuhito Tanaka': 56, 
                  'Yusuke Tanaka': 58, 'Kristian Thomas': 78, 'Dmitry Ushakov': 64,
                  'Max Whitlock': 56, 'Jordyn Wieber': 52, 'Koji Yamamuro': 58,
                  'Chenglong Zhang': 65, 'Kai Zou': 55}
df2012['weight'] = df2012['name'].map(missing_values).fillna(df2012['weight'])

# Fixing date birth
print("Fixing wrongs date birth...\n")
i = df2012[df2012.birth == '08/01/1985 (KOR)'].index
df2012.loc[i, 'birth'] = df2012.loc[i, 'birth'].str[0:10]
i = df2012[df2012.birth == '05/02/1979 (ITA)'].index
df2012.loc[i, 'birth'] = df2012.loc[i, 'birth'].str[0:10]

# Delete the nacionality column, too many missing values
print("Deleting the nacionality column (too many missing values)...\n")
df2012.drop(columns = 'nacionality', inplace=True)

df2012.sort_index(inplace=True)

print("\nFinal result...")
print(df2012.info())


print("****************************************************")
print("Saving the dataframes...\n")

# Saving the dataframe
out_file = "Summer_Olympic_Medallists_1896_to_2008.csv"
df.to_csv(out_file, index=False)
print(df.head())
print("columns: {}".format(df.columns.tolist()))
print("Data saved as {} file.\n\n".format(out_file))

# Reshaping one more time
df = pd.pivot_table(df, values='athlete', index=['edition','country','NOC'], columns='medal', aggfunc='count').reset_index()
df.dropna(how='all', inplace=True)
df.fillna(0, inplace=True) #Chage the null values into 0
df['total'] = df.Gold + df.Silver + df.Bronze
out_file = "Summer_Olympic_1896_to_2008_medals.csv"
df.to_csv(out_file, index=False)
print(df.head())
print("columns: {}".format(df.columns.tolist()))
print("Data saved as {} file.\n\n".format(out_file))

out_file = "Summer_Olympic_2012.csv"
df2012.to_csv(out_file, index=False)
print(df2012.head())
print("columns: {}".format(df2012.columns.tolist()))
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