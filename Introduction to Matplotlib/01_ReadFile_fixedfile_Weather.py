# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:34:04 2019

@author: jacqueline.cortez
"""

# Import packages
from glob import glob
import pandas as pd
#import numpy as np

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Reading the station file...")
file = "weather-stations.txt"
#fwidths = [1, 13, 22, 32, 39, 42, 73, 77, 81]

f_station = pd.read_fwf(file, #widths=fwidths)
                        header = None,
                        names = ['station', 'latitude', 'longitude', 'elevation', 'state',
                                 'name', 'gsn_flag', 'hcn_flag', 'wmo_id'],
                        usecols = ['station', 'latitude', 'longitude', 'elevation', 'state'],
                        index_col = 'station')
seattle_station = f_station[f_station.state.str.contains('SEATTLE')]
austin_station = f_station[f_station.state.str.contains('AUSTIN')]
                           
print(f_station.head())

print("****************************************************")
print("Reading the wheather file...")
#Get the file names to read
filenames = glob("weather_*.csv")

# Initialize empty DataFrame: data
seattle_weather = pd.DataFrame()
austin_weather = pd.DataFrame()

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
        temp_seattle = temp_w[temp_w['station'].isin(seattle_station.index.tolist())]
        counted_rowt += temp_seattle.shape[0]
        if temp_seattle.shape[0] > 0 :
            seattle_weather = seattle_weather.append(temp_seattle)
    
        # Getting the Austin Station
        temp_austin = temp_w[temp_w['station'].isin(austin_station.index.tolist())]
        counted_rowt += temp_austin.shape[0]
        if temp_austin.shape[0] > 0 :
            austin_weather = austin_weather.append(temp_austin)
            
    # Giving information of the total rows read from the file
    print("{} rows read into '{}'. {} recorded".format(counted_row, file, counted_rowt))

# Re-initialize the index
seattle_weather.reset_index(inplace=True)
austin_weather.reset_index(inplace=True)

# Dropping time column
seattle_weather.drop(["index"], axis='columns', inplace=True)
austin_weather.drop(["index"], axis='columns', inplace=True)


# Exploring the final dataframe
print("\n****************************************************")
print("Exploring the Seattle DataFrame...")
print(seattle_weather.head())
#print(seattle_weather.info())

print("\n****************************************************")
print("Exploring the Austin DataFrame...")
print(austin_weather.head())
#print(austin_weather.info())

# Saving the dataframe
seattle_file = "seattle_weather_all.csv"
austin_file = "austin_weather_all.csv"
seattle_weather.to_csv(seattle_file, index=False)
austin_weather.to_csv(austin_file, index=False)
print("\n****************************************************")
print("DataFrame saved into '{}' and '{}' files.".format(seattle_file, austin_file))

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