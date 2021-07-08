# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 00:00:29 2021

@author: jaces
"""
# Import libraries
import pandas as pd
import csv

from collections import defaultdict
from pprint import pprint

# Read data from file into list of list
df = pd.read_csv('data/baby_names.csv')
df.sort_values(by = ['RANK', 'NAME'], inplace = True)
df['NAME'] = df.NAME.str.title()
print(df.head())

baby_names_2012 = defaultdict(list)
for n, c in df[(df.BRITH_YEAR == 2012)].NAME.value_counts().items():
    baby_names_2012[c].append(n)
# in:
# baby_names_2012
# out: 
# defaultdict(list,
#            {6: ['ARIEL', 'JORDAN', 'AVERY'],
#             5: ['RILEY', 'DYLAN', 'RYAN'],
#             4: ['ISABELLA', 'MIA', 'SEBASTIAN', 'ADAM', ...]
#             3: [EVA', 'ALICE', 'HAZEL', 'CONNOR', ...]
#             2: ['KELLY', 'ERIK', 'ARMANI', 'AMELIE', ...]
#             1: ['MOISES', 'TIANA', 'CARSON', 'BROOKE', ...])

female_baby_names_2012 = df[(df.BRITH_YEAR == 2012) & (df.ETHNICTY.isin(['WHITE NON HISP', 'WHITE NON HISPANIC'])) &
                            (df.GENDER == 'FEMALE')].set_index('RANK').sort_index().NAME.str.title().to_dict()
male_baby_names = {2012: {},
                   2013: df[(df.BRITH_YEAR == 2013) & (df.ETHNICTY.isin(['WHITE NON HISP', 'WHITE NON HISPANIC'])) &
                            (df.GENDER == 'MALE')].set_index('RANK').sort_index().NAME.to_dict(),
                   2014: df[(df.BRITH_YEAR == 2014) & (df.ETHNICTY.isin(['WHITE NON HISP', 'WHITE NON HISPANIC'])) &
                            (df.GENDER == 'MALE')].set_index('RANK').sort_index().NAME.to_dict()}
male_baby_names_2011 = df[(df.BRITH_YEAR == 2011) & (df.ETHNICTY.isin(['WHITE NON HISP', 'WHITE NON HISPANIC'])) &
                            (df.GENDER == 'MALE')].set_index('RANK').sort_index().NAME.str.title().to_dict()
"""
female_baby_names = df[(df.GENDER == 'FEMALE') &
                       (df.RANK < 11)].pivot_table(index='RANK', 
                                                   columns='BRITH_YEAR', values='NAME', aggfunc='last').to_dict()
"""
female_baby_names = {2011: df[(df.BRITH_YEAR == 2011) & (df.ETHNICTY.isin(['WHITE NON HISP', 'WHITE NON HISPANIC'])) &
                            (df.GENDER == 'FEMALE') & (df.RANK < 11)].set_index('RANK').sort_index().NAME.to_dict(),
                     2012: df[(df.BRITH_YEAR == 2012) & (df.ETHNICTY.isin(['WHITE NON HISP', 'WHITE NON HISPANIC'])) &
                            (df.GENDER == 'FEMALE') & (df.RANK < 11)].set_index('RANK').sort_index().NAME.to_dict(),
                     2013: df[(df.BRITH_YEAR == 2013) & (df.ETHNICTY.isin(['WHITE NON HISP', 'WHITE NON HISPANIC'])) &
                            (df.GENDER == 'FEMALE') & (df.RANK < 11)].set_index('RANK').sort_index().NAME.to_dict(),
                     2014: df[(df.BRITH_YEAR == 2014) & (df.ETHNICTY.isin(['WHITE NON HISP', 'WHITE NON HISPANIC'])) &
                            (df.GENDER == 'FEMALE') & (df.RANK < 11)].set_index('RANK').sort_index().NAME.to_dict()}

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 02. Dictionaries - the root of Python')
print('*********************************************************')
print('** 02.01 Using dictionaries')
print('*********************************************************')
print('** 02.02 Creating and looping through dictionaries')
print('*********************************************************')
# Create an empty dictionary: names_by_rank
names_by_rank = {}

# Loop over the girl names
for rank, name in female_baby_names_2012.items():
    # Add each name to the names_by_rank dictionary using rank as the key
    names_by_rank[rank] = name
    
# Sort the names_by_rank dict by rank in descending order and slice the first 10 items
for rank in sorted(names_by_rank, reverse=True)[:10]:
    # Print each item
    print(names_by_rank[rank])

print('*********************************************************')
print('** 02.03 Safely finding by key')
print('*********************************************************')
# Safely print rank 7 from the names dictionary
print(female_baby_names_2012.get(7))

# Safely print the type of rank 100 from the names dictionary
print(type(female_baby_names_2012.get(105)))

# Safely print rank 105 from the names dictionary or 'Not Found'
print(female_baby_names_2012.get(105, 'Not Found'))

print('*********************************************************')
print('** 02.04 Dealing with nested data')
print('*********************************************************')
# Print a list of keys from the boy_names dictionary
print(male_baby_names.keys())

# Print a list of keys from the boy_names dictionary for the year 2013
print(male_baby_names[2013].keys())

# Loop over the dictionary
for year in male_baby_names:
    # Safely print the year and the third ranked name or 'Unknown'
    print(year, male_baby_names[year].get(3, 'unknown'))
    
print('*********************************************************')
print('** 02.05 Altering dictionaries')
print('*********************************************************')
print('** 02.06 Adding and extending dictionaries')
print('*********************************************************')
# Assign the names_2011 dictionary as the value to the 2011 key of boy_names
male_baby_names[2011] = male_baby_names_2011

# Update the 2012 key in the boy_names dictionary
male_baby_names[2012].update([(1, 'Casey'), (2, 'Aiden')])

# Loop over the years in the boy_names dictionary 
for year in male_baby_names:
    # Sort the data for each year by descending rank and get the lowest one
    lowest_ranked =  sorted(male_baby_names[year], reverse=True)[0]
    # Safely print the year and the least popular name or 'Not Available'
    print(year, male_baby_names[year].get(lowest_ranked, 'Not Available'))

print('*********************************************************')
print('** 02.07 Popping and deleting from dictionaries')
print('*********************************************************')
# Remove 2011 from female_names and store it: female_names_2011
female_names_2011 = female_baby_names.pop(2011)

# Safely remove 2015 from female_names with an empty dictionary as the default: female_names_2015
female_names_2015 = female_baby_names.pop(2015, {})

# Delete 2012 from female_names
del female_baby_names[2012]

# Print female_names
pprint(female_baby_names)

print('*********************************************************')
print('** 02.08 Pythonically using dictionaries')
print('*********************************************************')
print('** 02.09 Working with dictionaries more pythonically')
print('*********************************************************')
# Iterate over the 2014 nested dictionary
for rank, name in list(male_baby_names[2014].items())[:10]:
    # Print rank and name
    print(rank, name)
    
# Iterate over the 2012 nested dictionary
for rank, name in female_baby_names.get(2012,{}).items():
    # Print rank and name
    print(rank, name)

print('*********************************************************')
print('** 02.10 Checking dictionaries for data')
print('*********************************************************')
female_baby_names[2012] = {}

# Check to see if 2011 is in baby_names
if 2011 in female_baby_names:
    # Print 'Found 2011'
    print('Found 2011')
    
# Check to see if rank 1 is in 2012
if 1 in female_baby_names[2012]:
    # Print 'Found Rank 1 in 2012' if found
    print('Found Rank 1 in 2012')
else:
    # Print 'Rank 1 missing from 2012' if not found
    print('Rank 1 missing from 2012')
    
# Check to see if Rank 5 is in 2013
if 5 in female_baby_names[2013]:
   # Print 'Found Rank 5'
   print('Found Rank 5')
   
print('*********************************************************')
print('** 02.11 Working with CSV files')
print('*********************************************************')
# Reading from a file using CSV reader
with open('data/ART_GALLERY.csv', 'r') as f:
    for row in csv.reader(f):
        print(row)

# Creating a dictionary from a file
with open('data/ART_GALLERY.csv', 'r') as f:
    for row in csv.DictReader(f):
        pprint(row)
    
print('*********************************************************')
print('** 02.12 Reading from a file using CSV reader')
print('*********************************************************')
baby_names = {}
csvfile = 'data/baby_names.csv'

# Create a python file object in read mode for the baby_names.csv file: csvfile
with open(csvfile, 'r') as f:
    # Loop over a csv reader on the file object
    for i, row in enumerate(csv.reader(f)):
        if i > 10:
            break
        else:
            # Print each row 
            print(row)
            # Add the rank and name to the dictionary
            baby_names[row[5]] = row[3]

# Print the dictionary keys
print(baby_names.keys())

print('*********************************************************')
print('** 02.13 Creating a dictionary from a file')
print('*********************************************************')
baby_names = {}
csvfile = 'data/baby_names.csv'

# Create a python file object in read mode for the `baby_names.csv` file: csvfile
with open(csvfile, 'r') as f:
    # Loop over a DictReader on the file
    for i, row in enumerate(csv.DictReader(f), start=1):
        if i > 10:
            break;
        # Print each row 
        print(row)
        # Add the rank and name to the dictionary: baby_names
        baby_names[row['RANK']] = row['NAME']

    # Print the dictionary keys
    print(baby_names.keys())

print('*********************************************************')
print('END')
print('*********************************************************')