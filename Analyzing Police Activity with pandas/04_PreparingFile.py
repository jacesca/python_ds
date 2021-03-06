# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:13:15 2019

@author: jacqueline.cortez
"""

# Import packages
#from glob import glob
import pandas as pd
#import numpy as np

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Reading the file...")

file = 'StationRI_2004_2016.csv'
weather = pd.read_csv(file, usecols=['station','date','element','value'])

stationRI = weather['station'].unique()[0]
print("Rhode Island station: '{}'.".format(stationRI))

# Setting the datetime column, regular expressions
print("Fixing the date format...")
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')

print("Pivotting the table...")
weather = weather.pivot(index='date', columns='element', values='value')
weather['station'] = stationRI
weather.reset_index(inplace=True)

print("Getting the right measure for temp...")
weather['TMIN'] /= 10 #Avoiding tenths of degrees C
weather['TAVG'] /= 10
weather['TMAX'] /= 10

print("Exploring the final DataFrame...")
print(weather.head())
print(weather.info())

# Saving the dataframe
out_file = "RIWeather_2004_2016.csv"
weather.to_csv(out_file)
print("DataFrame saved into '{}' file.".format(out_file))

print("****************************************************")
print("** END                                            **")
print("****************************************************")

