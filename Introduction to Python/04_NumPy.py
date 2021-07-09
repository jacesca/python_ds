# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:37:16 2019

@author: jacqueline.cortez
"""

# Import the numpy package as np
import numpy as np

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("YOUR FIRST NUMPY ARRAY\n")

# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Create a numpy array from baseball: np_baseball
np_baseball=np.array(baseball)

# Print out type of np_baseball
print(type(np_baseball))

print("")
print("****************************************************")
print("BASEBALL PLAYERS' HEIGHT\n")

# height is available as a regular list
file = "SOCR_Data_MLB_HeightsWeights.csv"
height_in = list(np.loadtxt(file, skiprows=1, usecols=3, delimiter=";", dtype=np.int32))

# Create a numpy array from height_in: np_height_in
np_height_in=np.array(height_in)

# Print out np_height_in
print(np_height_in)

# Convert np_height_in to m: np_height_m
np_height_m=np_height_in*0.0254

# Print np_height_m
print(np_height_m)

print("")
print("****************************************************")
print("BASEBALL PLAYER'S BMI\n")

# height and weight are available as regular lists
#file = "SOCR_Data_MLB_HeightsWeights.csv"
#height_in = list(np.loadtxt(file, skiprows=1, usecols=3, delimiter=";", dtype=np.int32))
weight_lb = list(np.loadtxt(file, skiprows=1, usecols=4, delimiter=";", dtype=np.int32))

# Create array from height_in with metric units: np_height_m
np_height_m = np.array(height_in) * 0.0254

# Create array from weight_lb with metric units: np_weight_kg
np_weight_kg = np.array(weight_lb) * 0.453592

# Calculate the BMI: bmi
bmi=np_weight_kg/(np_height_m**2)

# Print out bmi
print(bmi)

print("")
print("****************************************************")
print("LIGHTWEIGHT BASEBALL PLAYERS\n")

# height and weight are available as a regular lists
#file = "SOCR_Data_MLB_HeightsWeights.csv"
#height_in = list(np.loadtxt(file, skiprows=1, usecols=3, delimiter=";", dtype=np.int32))
#weight_lb = list(np.loadtxt(file, skiprows=1, usecols=4, delimiter=";", dtype=np.int32))

# Calculate the BMI: bmi
np_height_m = np.array(height_in) * 0.0254
np_weight_kg = np.array(weight_lb) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light=bmi<21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(np.array(bmi[light]))

print("")
print("****************************************************")
print("SUBSETTING NUMPY ARRAYS\n")

# height and weight are available as a regular lists
#file = "SOCR_Data_MLB_HeightsWeights.csv"
#height_in = list(np.loadtxt(file, skiprows=1, usecols=3, delimiter=";", dtype=np.int32))
#weight_lb = list(np.loadtxt(file, skiprows=1, usecols=4, delimiter=";", dtype=np.int32))

# Store weight and height lists as numpy arrays
np_weight_lb = np.array(weight_lb)
np_height_in = np.array(height_in)

# Print out the weight at index 50
print(np_weight_lb[50])

# Print out sub-array of np_height_in: index 100 up to and including index 110
print(np_height_in[100:111])

print("")
print("****************************************************")
print("YOUR FIRST 2D NUMPY ARRAY\n")

# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Create a 2D numpy array from baseball: np_baseball
np_baseball=np.array(baseball)

# Print out the type of np_baseball
print(type(np_baseball))

# Print out the shape of np_baseball
print(np_baseball.shape)

print("")
print("****************************************************")
print("BASEBALL DATA IN 2D FORM\n")

# baseball is available as a regular list of lists
#file = "SOCR_Data_MLB_HeightsWeights.csv"
baseball = list(np.loadtxt(file, skiprows=1, usecols=[3,4], delimiter=";", dtype=np.int32))

# Create a 2D numpy array from baseball: np_baseball
np_baseball=np.array(baseball)

# Print out the shape of np_baseball
print(np_baseball.shape)

print("")
print("****************************************************")
print("SUBSETTING 2D NUMPY ARRAYS\n")

# baseball is available as a regular list of lists
#file = "SOCR_Data_MLB_HeightsWeights.csv"
#baseball = list(np.loadtxt(file, skiprows=1, usecols=[3,4], delimiter=";", dtype=np.int32))

# Create np_baseball (2 cols)
np_baseball = np.array(baseball)

# Print out the 50th row of np_baseball
print(np_baseball[49,:])

# Select the entire second column of np_baseball: np_weight
np_weight=np_baseball[:,1]

# Print out height of 124th player
print(np_baseball[123,0])

print (type(np_baseball))
print (np_baseball.shape)

print("")
print("****************************************************")
print("2D ARITHMETIC\n")

# baseball is available as a regular list of lists
#file = "SOCR_Data_MLB_HeightsWeights.csv"
baseball = list(np.loadtxt(file, skiprows=1, usecols=[3,4,5], delimiter=";", dtype=np.float32))

# updated is available as 2D numpy array
#updated = np.array(zip(np.random.rand(1034), np.random.rand(1034)))
#h = np.round(np.random.rand(1034),2) #height
h = np.random.rand(1034) #height
w = np.random.rand(1034) #weight
#a = np.random.choice(np.arange(5), 1034) #age
#a = np.random.randint(5, size=1034) #age
a = [1 for x in range(1034)]
u = list(zip(h, w, a)) #combinando height, weight and age
updated = np.array(u) #getting the numpy array
#print(updated)

# Create np_baseball (3 cols)
np_baseball = np.array(baseball)
print("Baseball data (original): ")
print(np_baseball)
print("")

# Print out addition of np_baseball and updated
print("Baseball data updated: ")
print(np_baseball+updated)
print("")

# Create numpy array: conversion
conversion=np.array([0.0254,0.453592,1])

# Print out product of np_baseball and conversion
print("Convertion into meters and kilograms (Original Array): ")
print(np_baseball*conversion)

print("")
print("****************************************************")
print("AVERAGE VERSUS MEDIAN\n")

# np_baseball is available
#file = "SOCR_Data_MLB_HeightsWeights.csv"
np_baseball = np.loadtxt(file, skiprows=1, usecols=[3,4,5], delimiter=";", dtype=np.float32)

# Create np_height from np_baseball
np_height=np_baseball[:,0]

# Print out the mean of np_height
print(np.mean(np_height))

# Print out the median of np_height
print(np.median(np_height))

print("")
print("****************************************************")
print("EXPLORE THE BASEBALL DATA\n")

# np_baseball is available
#file = "SOCR_Data_MLB_HeightsWeights.csv"
#np_baseball = np.loadtxt(file, skiprows=1, usecols=[3,4,5], delimiter=";", dtype=np.float32)

# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation:  \n{}".format(str(corr)))

print("")
print("****************************************************")
print("BLEND IT ALL TOGETHER\n")

# heights and positions are available as lists
#file = "SOCR_Data_MLB_HeightsWeights.csv"
heights = list(np.loadtxt(file, skiprows=1, usecols=3, delimiter=";", dtype=np.int32))
positions = list(np.loadtxt(file, skiprows=1, usecols=2, delimiter=";", dtype=str))

# Convert positions and heights to numpy arrays: np_positions, np_heights
np_positions=np.array(positions)
np_heights=np.array(heights)

# Define the position 
position_i = "Catcher"

# Heights of the interested position: gk_heights
gk_heights=np_heights[np_positions==position_i]

# Heights of the other players: other_heights
other_heights=np_heights[np_positions!=position_i]

# Print out the median height of the interested position. 
print("Median height of {}: {}".format(position_i,str(np.median(gk_heights))))

# Print out the median height of other players. 
print("Median height of other players: " + str(np.median(other_heights)))

print("")
print("****************************************************")
print("** END                                            **")
print("****************************************************")

