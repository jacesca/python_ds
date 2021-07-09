# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:30:34 2019

@author: jacqueline.cortez
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("CREATE A LIST\n")

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas=[hall,kit,liv,bed,bath]

# Print areas
print(areas)

print("")
print("****************************************************")
print("CREATE LIST WITH DIFFERENT TYPES\n")

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Adapt list areas
areas = ["hallway", hall, "kitchen", kit, "living room", liv, "bedroom", bed, "bathroom", bath]

# Print areas
print(areas)

print("")
print("****************************************************")
print("LIST OF LISTS\n")

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom",bed],
         ["bathroom",bath]]

# Print out house
print(house)

# Print out the type of house
print(type(house))

print("")
print("****************************************************")
print("SUBSET AND CONQUER\n")

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas
print(areas[1])

# Print out last element from areas
print(areas[9])

# Print out the area of the living room
print(areas[5])

print("")
print("****************************************************")
print("SUBSET AND CALCULATE\n")

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area=areas[3]+areas[7]

# Print the variable eat_sleep_area
print(eat_sleep_area)

print("")
print("****************************************************")
print("SLICING AND DICING\n")

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Use slicing to create downstairs
downstairs=areas[0:6]

# Use slicing to create upstairs
upstairs=areas[6:10]

# Print out downstairs and upstairs
print(downstairs)
print(upstairs)

print("")
print("****************************************************")
print("SLICING AND DICING (2)\n")

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Alternative slicing to create downstairs
downstairs=areas[:6]

# Alternative slicing to create upstairs
upstairs=areas[6:]

print(downstairs)
print(upstairs)

print("")
print("****************************************************")
print("REPLACE LIST ELEMENTS\n")

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Correct the bathroom area
areas[-1]=10.50

# Change "living room" to "chill zone"
areas[4]="chill zone"

print(areas)

print("")
print("****************************************************")
print("EXTEND A LIST\n")

# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]

# Add poolhouse data to areas, new list is areas_1
areas_1=areas+["poolhouse", 24.5]

# Add garage data to areas_1, new list is areas_2
areas_2=areas_1+["garage",15.45]

print(areas_1)
print(areas_2)

print("")
print("****************************************************")
print("INNER WORKINGS OF LISTS\n")

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy
areas_copy = list(areas)

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)

print("")
print("****************************************************")
print("** END                                            **")
print("****************************************************")
