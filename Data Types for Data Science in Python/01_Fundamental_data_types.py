# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:24:50 2021

@author: jaces
"""
# Import libraries
# Import libraries
import numpy as np
import pandas as pd

from pprint import pprint


# Read data from file into list of list
baby_records = list(np.genfromtxt('data/baby_names.csv', delimiter=',', skip_header=1,
                                  encoding='utf-8', dtype=None))

df = pd.read_csv('data/baby_names.csv')

girl_names = list(df[df.GENDER == 'FEMALE'].NAME.unique())
boy_names = list(df[df.GENDER == 'MALE'].NAME.unique())

baby_names_2011 = set(df[(df.BRITH_YEAR.isin([2011, 2012]))].NAME.str.title())
baby_names_2014 = set(df[(df.BRITH_YEAR.isin([2013, 2014]))].NAME.str.title())


print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 01. Fundamental data types')
print('*********************************************************')
print('** 01.01 Introduction and lists')
print('*********************************************************')

# Accessing single items in list
cookies = ['chocolate chip', 'peanut butter', 'sugar']
cookies.append('Tirggel')
print(cookies)
print(cookies[2])

# Combining Lists
cakes = ['strawberry', 'vanilla']
desserts = cookies + cakes
print(desserts)

# Finding Elements in a List
position = cookies.index('sugar')
print(position)
print(cookies[position])

# Removing Elements in a List
name = cookies.pop(position)
print(name)
print(cookies)

# Iterating over lists
for cookie in cookies:
    print(cookie)

# Sorting lists
print(cookies)
sorted_cookies = sorted(cookies)
print(sorted_cookies)

print('*********************************************************')
print('** 01.02 Manipulating lists for fun and profit')
print('*********************************************************')
# Create a list containing the names: baby_names
baby_names = ['Ximena', 'Aliza', 'Ayden', 'Calvin']

# Extend baby_names with 'Rowen' and 'Sandeep'
baby_names.extend(['Rowen', 'Sandeep'])

# Print baby_names
print(baby_names)

# Find the position of 'Aliza': position
position = baby_names.index('Aliza')

# Remove 'Aliza' from baby_names
_ = baby_names.pop(position)

# Print baby_names
print(baby_names)

print('*********************************************************')
print('** 01.03 Looping over lists')
print('*********************************************************')
# Create the empty list: baby_names
baby_names = []

# Loop over records 
for row in baby_records:
    # Add the name to the list
    baby_names.append(row[3])
    
# Sort the names in alphabetical order
sorted_baby_names = sorted(baby_names)

# Print first 30 names
print(sorted_baby_names[:30])

# Print last 10 names
print(sorted_baby_names[-10:])

print('*********************************************************')
print('** 01.04 Meet the Tuples')
print('*********************************************************')
print('** 01.05 Data type usage')
print('*********************************************************')
print('** 01.06 Using and unpacking tuples')
print('*********************************************************')
# Pair up the girl and boy names: pairs
pairs = zip(girl_names, boy_names)

# Iterate over pairs
result = []
for idx, pair in enumerate(pairs):
    # Unpack pair: girl_name, boy_name
    girl_name, boy_name = pair
    # Print the rank and names associated with each rank
    result.append('Rank {}: {} and {}'.format(idx, girl_name, boy_name))
pprint(result[:10])

print('*********************************************************')
print('** 01.07 Making tuples by accident')
print('*********************************************************')
# Create the normal variable: normal
normal = 'simple'

# Create the mistaken variable: error
error = 'trailing comma',

# Print the types of the variables
print(type(normal))
print(type(error))

print('*********************************************************')
print('** 01.08 Sets for unordered and unique data')
print('*********************************************************')
# Creating Sets
cookies_eaten_today = ['chocolate chip', 'peanut butter',
                       'chocolate chip', 'oatmeal cream', 'chocolate chip']
types_of_cookies_eaten = set(cookies_eaten_today)
print(types_of_cookies_eaten)

# Modifying Sets
types_of_cookies_eaten.add('biscotti')
types_of_cookies_eaten.add('chocolate chip')
print(types_of_cookies_eaten)

# Updating Sets
cookies_hugo_ate = ['chocolate chip', 'anzac']
types_of_cookies_eaten.update(cookies_hugo_ate)
print(types_of_cookies_eaten)

# Removing data from sets
types_of_cookies_eaten.discard('biscotti')
print(types_of_cookies_eaten)
print(types_of_cookies_eaten.pop())
print(types_of_cookies_eaten.pop())
print(types_of_cookies_eaten)

# Two sets
cookies_jason_ate = set(['chocolate chip', 'oatmeal cream',
'peanut butter'])
cookies_hugo_ate = set(['chocolate chip', 'anzac'])
print('Jason: ', cookies_jason_ate)
print('Hugo : ', cookies_hugo_ate)

# Set Operations - Similarities
cookies_jason_ate = set(['chocolate chip', 'oatmeal cream',
'peanut butter'])
cookies_hugo_ate = set(['chocolate chip', 'anzac'])
print('Eaten by Jason and Hugo: ', cookies_jason_ate.union(cookies_hugo_ate))

# Set Operations - Differences
print('Not eaten by Hugo: ', cookies_jason_ate.difference(cookies_hugo_ate))
print('Not eaten by Jason: ', cookies_hugo_ate.difference(cookies_jason_ate))


print('*********************************************************')
print('** 01.09 Finding all the data and the overlapping data between sets')
print('*********************************************************')
# Find the union: all_names
all_names = baby_names_2011.union(baby_names_2014)

# Print the count of names in all_names
print(len(all_names))

# Find the intersection: overlapping_names
overlapping_names = baby_names_2011.intersection(baby_names_2014)

# Print the count of names in overlapping_names
print(len(overlapping_names))

print('*********************************************************')
print('** 01.10 Determining set differences')
print('*********************************************************')
# Create the empty set: baby_names_2011
baby_names_2011 = set()

# Loop over records and add the names from 2011 to the baby_names_2011 set
for row in baby_records:
    # Check if the first column is '2011'
    if row[0] == 2011:
        # Add the fourth column to the set
        baby_names_2011.add(row[3])

# Find the difference between 2011 and 2014: differences
differences = baby_names_2011.difference(baby_names_2014)

# Print the differences
print(len(differences))

print('*********************************************************')
print('END')
print('*********************************************************')