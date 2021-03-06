# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:34:04 2019

@author: jacqueline.cortez

https://www.ncdc.noaa.gov/ghcnd-data-access
https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt
ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/

"""

# Import packages
from glob import glob
import pandas as pd
#import numpy as np

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Introduce the station to explore:")
station = str(input()).strip()

print("Introduce the name of the city related:")
city = str(input()).strip()

print("****************************************************")
print("Reading the wheather file...")
#Get the file names to read
filenames = glob("weather_*.csv")

# Initialize empty DataFrame: data
weather = pd.DataFrame()

# Read each name of filenames list
for file in filenames:
    # Prepare the chunk object
    w_reader = pd.read_csv(file, parse_dates = True, 
                           header = None,
                           names = ['station', 'date', 'element', 'value',
                                    'mflag', 'qflag', 'sflag', 'time'],
                           usecols = ['station','date','element','value'],
                           chunksize = 500000)
    # Initialize the counted rows
    counted_row = 0
    counted_rowt = 0

    # Iterate over each DataFrame chunk
    for temp_w in w_reader:
        counted_row += temp_w.shape[0]
        
        # Getting the Seattle Station
        temp_w = temp_w[temp_w['station']==station]
        counted_rowt += temp_w.shape[0]
        if temp_w.shape[0] > 0 :
            weather = weather.append(temp_w)
            
    # Giving information of the total rows read from the file
    print("{} rows read into '{}'. {} recorded".format(counted_row, file, counted_rowt))

# Re-initialize the index
weather.reset_index(inplace=True)

# Dropping time column
weather.drop(["index"], axis='columns', inplace=True)
weather['city'] = city

# Exploring the final dataframe
print("\n****************************************************")
print("Exploring the %s DataFrame..." % station) #Regular expressions
print(weather.head())
#print(seattle_weather.info())

# Saving the dataframe
out_file = "{}_{}.csv".format(city,station)
weather.to_csv(out_file, index=False)
print("\n****************************************************")
print("DataFrame saved into '{}' file.".format(out_file))

print("****************************************************")
print("** END                                            **")
print("****************************************************")

"""
08/04.05 Cleaning and tidying datetime data.txt


df_dropped['date'] = df_dropped["date"].astype(str)
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))
date_string = df_dropped["date"]+df_dropped["Time"]
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')
df_clean = df_dropped.set_index(date_times)
"""