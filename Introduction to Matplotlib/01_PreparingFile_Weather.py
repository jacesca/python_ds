# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:13:15 2019

@author: jacqueline.cortez
"""

# Import packages
#from glob import glob
import pandas as pd
import numpy as np

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Wich file (AUSTIN_USW00013904.csv or SEATTLE_USW00094290.csv):")

file = str(input()).strip()
weather = pd.read_csv(file)#, usecols=['station','date','element','value','city'])

city = weather['city'].unique()[0]
station = weather['station'].unique()[0]
print("****************************************************")
print("{} station: '{}'.".format(city, station))

# Setting the datetime column, regular expressions
print("Fixing the date format...")
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')

print("Pivotting the table...")
weather = weather.pivot(index='date', columns='element', values='value')
weather.reset_index(inplace=True)

print("Getting the right measure for temp...")
weather['TMIN'] /= 10 #Avoiding tenths of degrees C
weather['TAVG'] = weather['TAVG'] / 10 if 'TAVG' in weather.columns.tolist() else (weather['TMIN']+weather['TMAX'])/2/10
weather['TMAX'] /= 10

print("Getting the right measure for precipitation...")
weather['PRCP'] /= 10 #Avoiding tenths of mm

print("Adding the station and city as reference")
weather['station'] = station
weather['city'] = city

print("Exploring the final DataFrame...")
print(weather.head())
print(weather.info())

# Saving the dataframe Regular expressions
out_file = "%s_weather.csv" % city
weather.to_csv(out_file, index=False)
print("DataFrame saved into '{}' file.".format(out_file))

print("****************************************************")
print("** END                                            **")
print("****************************************************")

