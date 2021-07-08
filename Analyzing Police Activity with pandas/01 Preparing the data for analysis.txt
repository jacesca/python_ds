# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:14:00 2019

@author: jacqueline.cortez

Introduction:
    Before beginning your analysis, it is critical that you first 
    examine and clean the dataset, to make working with it a more 
    efficient process. In this chapter, you will practice fixing 
    data types, handling missing values, and dropping columns and 
    rows while learning about the Stanford Open Policing Project 
    dataset.
"""

# Import packages
import pandas as pd
import numpy as np

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.0f}'.format 
#pd.reset_option("all")

# Defining a function to print complement info about df
def show_df_info(df):
    df_info = {}
    #df_info["columns"] = df.columns.values
    df_info["null_values"] = df.isnull().sum()
    df_info["not_null_values"] = df.notnull().sum()
    df_info["type_column"] = df.dtypes
    df_info["unique_values"] = df.nunique(dropna=False)
    #df_info["memory_use"] = df.memory_usage()[1:]
    print("Shape: {}".format(df.shape)) #Regular expressions
    print(pd.DataFrame(df_info))
    print("Index {}: From {} to {}".format(df.index.dtype, df.index.min(), df.index.max()))
    print("Total memory used: {0:,.0f}".format(df.memory_usage().sum()))

# Read 'police.csv' into a DataFrame named ri
file = "policing_rhodeisland 2005-2015.csv"
ri = pd.read_csv(file, parse_dates=True, 
                 usecols=["date", "time", "subject_sex", "subject_race", "reason_for_stop", "search_conducted", "reason_for_search", "outcome", "arrest_made", "contraband_drugs", "zone"])

"""
# Reading the complete file

pd.set_option("display.max_columns",20)

ri = pd.read_csv(file)
print(ri.head())
print(ri.columns)

pd.reset_option("all")

"""
# Fixing columns name
ri.columns = ['stop_date', 'stop_time', 'district', 'driver_race', 'driver_gender', 'is_arrested', 'stop_outcome', 'drugs_related_stop', 'search_conducted', 'search_type',  'violation_raw']

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("PRE-VISUALIZING THE FILE:")
print("Columns: {}".format(ri.columns.values))
print(ri.head()) #Getting the first 5 rows
show_df_info(ri)

# Drop all rows that are missing 'driver_gender'
ri.dropna(subset=["driver_gender"], inplace=True)
ri.reset_index(inplace=True)

print("****************************************************")
print("AFTER DELETING NULL VALUES IN driver_gender:")
show_df_info(ri)

print("****************************************************")
print("SHOWING UNIQUE VALUES:")
print("district: ", ri.driver_race.unique())
print("driver_race: ", ri.driver_race.unique())
print("driver_gender: ", ri.driver_gender.unique())
print("is_arrested: ", ri.is_arrested.unique())
print("stop_outcome: ", ri.stop_outcome.unique())
#print(ri.stop_outcome.value_counts(dropna=False)) #counting unique values including null values
print("drugs_related_stop: ", ri.drugs_related_stop.unique())
print("search_conducted: ", ri.search_conducted.unique())
print("violation_raw: ", ri.violation_raw.unique())

# Filling the missing values
ri["drugs_related_stop"] = ri.drugs_related_stop.fillna(False)
ri["stop_outcome"] = ri.stop_outcome.fillna("no action")
#ri["search_type"] = ri.search_type.fillna("no action")

# Fixing data types
ri["district"] = ri.district.astype("category")
ri["driver_race"] = ri.driver_race.astype("category")
ri["driver_gender"] = ri.driver_gender.astype("category")
ri["is_arrested"] = ri.is_arrested.astype("bool")
ri["stop_outcome"] = ri.stop_outcome.astype("category")
ri["drugs_related_stop"] = ri.drugs_related_stop.astype("bool")
ri["search_conducted"] = ri.search_conducted.astype("bool")
ri["violation_raw"] = ri.violation_raw.astype("category")

# Preparing the date column
#ri["stop_datetime"] = pd.to_datetime(ri.stop_date+' '+ri.stop_time)
ri["stop_datetime"] = pd.to_datetime(ri.stop_date.str.cat(ri.stop_time, sep=" "))

# Set the index to be the column stop_date
ri.set_index("stop_datetime", inplace=True)

print("****************************************************")
print("ADDING A RANDOM COLUMN AS 'stop_duration':")

# Crear a pandas series
#stop_duration = pd.Series(np.random.randint(3, size=len(ri)))
ri["duration"] = np.random.randint(3, size=len(ri)) #age
ri["stop_duration"] = ri.duration.map({0:'0-15 Min', 1:'16-30 Min', 2:'30+ Min'}) 

print("****************************************************")
print("DROPPING COLUMNS")

# Dropping time column
ri.drop(["index", "duration"], axis='columns', inplace=True)

print("****************************************************")
print("AFTER FYXING DATA TYPES:")
show_df_info(ri)
print(ri.head()) #Getting the first 5 rows

# Saving the dataframe
out_file = "rhode_island_2005_2015.csv"
ri.to_csv(out_file)

print("****************************************************")
print("** END                                            **")
print("****************************************************")