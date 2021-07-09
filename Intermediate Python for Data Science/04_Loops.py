# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:37:55 2019

@author: jacqueline.cortez
"""

# Import packages
import pandas as pd
import numpy as np

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Basic while loop\n")

# Initialize offset
offset=8

# Code the while loop
while offset!=0 :
    print("correcting...")
    offset=offset-1
    print(offset)

print("\n****************************************************")
print("** Add conditionals\n")

# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0 :
        offset = offset - 1
    else :
        offset = offset + 1
    print(offset)

print("\n****************************************************")
print("** Loop over a list\n")

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for element in areas :
    print(element)

print("\n****************************************************")
print("** Indexes and values (1)\n")

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate() and update print()
for i,a in enumerate(areas) :
    print("room "+str(i)+": "+str(a))

print("\n****************************************************")
print("** Indexes and values (2)\n")

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for index, area in enumerate(areas) :
    print("room " + str(index+1) + ": " + str(area))

print("\n****************************************************")
print("** Loop over list of lists\n")

# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
for room in house :
    print("the "+room[0]+" is "+str(room[1])+" sqm")

print("\n****************************************************")
print("** Loop over dictionary\n")

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for key,value in europe.items() :
    print("the capital of "+key+" is "+value)

print("\n****************************************************")
print("** Loop over Numpy array\n")

# height is available as a regular list
file = "SOCR_Data_MLB_HeightsWeights.csv"
np_height = np.loadtxt(file, skiprows=1030, usecols=3, delimiter=";", dtype=np.int32)
np_baseball = np.loadtxt(file, skiprows=1030, usecols=[3,4], delimiter=";", dtype=np.int32)

# For loop over np_height
for height in np_height:
    print(str(height)+ " inches")

# For loop over np_baseball
for height in np.nditer(np_baseball):
    print(height)

print("\n****************************************************")
print("** Loop over DataFrame (1)\n")

# Import cars data
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab, row in cars.iterrows():
    print(lab)
    print(row)

print("\n****************************************************")
print("** Loop over DataFrame (2)\n")

# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab+": "+str(row["cars_per_cap"]))
    
print("\n****************************************************")
print("** Add column (1)\n")

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab,"COUNTRY"]=row["country"].upper()

# Print cars
print(cars)

print("\n****************************************************")
print("** Add column (2)\n")

# Use .apply(str.upper)
cars["COUNTRY2"]=cars["country"].apply(str.upper)

print(cars)

print("\n****************************************************")
print("** END                                            **")
print("****************************************************")