# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:49:36 2019

@author: jacqueline.cortez
"""

# Import packages
#import numpy as np
import pandas as pd

# reading LifeExp and GDP
dict_file = {"East Asia & Pacific": ["World_A_EastAsiaPacific_2017_1960.csv", "yellow"],
             "Europe & Central Asia": ["World_A_EuropeCentralAsia_2017_1960.csv", "pink"],
             "Latin America & Caribbean": ["World_A_LatinAmericaCaribbean_2017_1960.csv", "blue"],
             "Middle East & North Africa": ["World_A_MiddleEastNorthAfrica_2017_1960.csv", "black"],
             "North America": ["World_A_NorthAmerica_2017_1960.csv", "red"],
             "South Asia": ["World_A_SouthAsia_2017_1960.csv", "green"],
             "Sub-Saharan Africa": ["World_A_Sub-SaharanAfrica_2017_1960.csv", "brown"]}
    
w_2017_1960 = pd.DataFrame({})
    
for region, data in dict_file.items():
    r = pd.read_csv(data[0], 
                    names=["pais","p","serie","s","1960","2017"], 
                    skiprows=1,
                    #index_col="pais",
                    skipfooter=5,
                    usecols=["pais", "serie", "1960", "2017"],
                    na_values="NA",
                    encoding='utf-8',
                    engine='python') #This line is because skipfooter

    print("** {}:".format(region.upper()))
    print("Initial Shape: {}".format(r.shape))

    # Evaluating no duplicates rows
    #if (r[r.duplicated(["pais", "serie"])].shape[0] == 0):
    #    print("No duplicated data...\n")
    # Evaluating no null values in series
    #if r[r["serie"].isnull()].shape[0] == 0:
    #    print("No error data in 'serie' column...\n")

    r_2017 = r.pivot(index="pais", columns="serie", values="2017") # Getting data from 2017
    r_2017.columns = ["2017_LifeExp_F", "2017_LifeExp", "2017_LifeExp_M", "2017_PIB", "2017_POP_M", "2017_POP_F", "2017_POP"]
    
    r_1960 = r.pivot(index="pais", columns="serie", values="1960") # Getting data from 1969
    r_1960.columns = ["1969_LifeExp_F", "1969_LifeExp", "1969_LifeExp_M", "1969_PIB", "1969_POP_M", "1969_POP_F", "1969_POP"]
    
    #Concatenating in one df
    r_2017_1960 = pd.concat([r_1960, r_2017], axis=1)
    r_2017_1960["Region"] = region
    r_2017_1960["Color"] = data[1]
    print("Final Shape: {}\n".format(r_2017_1960.shape))

    w_2017_1960 = w_2017_1960.append(r_2017_1960, sort=True)

print("Initial Shape of 1960 y 2017 Data:{}".format(w_2017_1960.shape))

# Delete rows with null values in any column
w_2017_1960.dropna(axis='index', how='any', inplace=True)
            
print("Final Shape of 1960 y 2017 Data:{}".format(w_2017_1960.shape))
print("Columns of 1960 y 2017 Data:\n{}".format(w_2017_1960.columns.values))
print("Head of 1960 y 2017 Data:\n{}".format(w_2017_1960.head()))
