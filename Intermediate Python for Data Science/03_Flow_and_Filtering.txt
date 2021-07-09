# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:23:58 2019

@author: jacqueline.cortez
"""

# Import packages
import pandas as pd
import numpy as np

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Equality\n")

# Comparison of booleans
print(True==False)

# Comparison of integers
print(-5*15!=75)

# Comparison of strings
print("pyscript"=="PyScript")

# Compare a boolean with an integer
print(True==1)

print("\n****************************************************")
print("** Greater and less than\n")

# Comparison of integers
x = -3 * 6
print(x>=-10)

# Comparison of strings
y = "test"
print("test"<=y)

# Comparison of booleans
print(True>False)

print("\n****************************************************")
print("** Compare arrays\n")

my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house>=18)

# my_house less than your_house
print(my_house<your_house)


print("\n****************************************************")
print("** and, or, not (1)\n")

# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(my_kitchen>10 and my_kitchen<18)

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen<14 or my_kitchen>17)

# Double my_kitchen smaller than triple your_kitchen?
print(2*my_kitchen<3*your_kitchen)

print("\n****************************************************")
print("** Boolean operators with Numpy\n")

my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house>18.5,my_house<10))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house<11,your_house<11))

print("\n****************************************************")
print("** if\n")

# Define variables
room = "kit"
area = 14.0

# if statement for room
if room == "kit" :
    print("looking around in the kitchen.")

# if statement for area
if area>15:
    print("big place!")

print("\n****************************************************")
print("** Add else\n")

# Define variables
room = "kit"
area = 14.0

# if-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
else :
    print("looking around elsewhere.")

# if-else construct for area
if area > 15 :
    print("big place!")
else :
    print("pretty small.")

print("\n****************************************************")
print("** Customize further: elif\n")

# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area > 10 :
    print("medium size, nice!")
else :
    print("pretty small.")

print("\n****************************************************")
print("** Driving right (1)\n")

# Import cars data
cars = pd.read_csv('cars.csv', index_col = 0)

# Extract drives_right column as Series: dr
dr=cars["drives_right"]

# Use dr to subset cars: sel
sel = cars[dr]

# Print sel
print(sel)

print("\n****************************************************")
print("** Driving right (2)\n")

# Convert code to a one-liner
sel = cars[cars['drives_right']]

# Print sel
print(sel)

print("\n****************************************************")
print("** Cars per capita (1)\n")

# Create car_maniac: observations that have a cars_per_cap over 500
print(cars[cars["cars_per_cap"]>500])

print("\n****************************************************")
print("** Cars per capita (2)\n")

# Create medium: observations with cars_per_cap between 100 and 500
medium=cars[np.logical_and(cars["cars_per_cap"]>100,cars["cars_per_cap"]<500)]

# Print medium
print(medium)

print("\n****************************************************")
print("** END                                            **")
print("****************************************************")