# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:23:54 2021

@author: jaces
"""
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from collections import namedtuple

from pprint import pprint
from datetime import datetime

import pandas as pd


# Read data from file into list of list
df = pd.read_csv('data/cta_daily_station_totals.csv')
print(df.head())


#stations = ['stationname'] + df.stationname.to_list()
stations = df.stationname.to_list()
entries = list(df[['date', 'stationname', 'rides']].to_records(index=False))
entries_Austin_Forest_Park = list(df[df.stationname == 'Austin-Forest Park'][['date', 'rides']].to_records(index=False))

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 03. Meet the collections module')
print('*********************************************************')
print('** 03.01 Counting made easy')
print('*********************************************************')
nyc_eatery_types = ['Mobile Food Truck', 'Mobile Food Truck', 'Mobile Food Truck', 'Mobile Food Truck', 'Mobile Food Truck', 
                    'Mobile Food Truck', 'Mobile Food Truck', 'Mobile Food Truck', 'Mobile Food Truck', 'Mobile Food Truck', 
                    'Food Cart', 'Food Cart', 'Food Cart', 'Food Cart', 'Food Cart', 'Food Cart', 'Food Cart', 'Snack Bar', 
                    'Snack Bar', 'Snack Bar', 'Snack Bar', 'Snack Bar', 'Restaurant', 'Restaurant', 'Restaurant', 
                    'Fruit & Vegetable Cart']

nyc_eatery_count_by_types = Counter(nyc_eatery_types)
print(nyc_eatery_count_by_types)

print(nyc_eatery_count_by_types['Restaurant'])

print('*********************************************************')
print('** 03.02 Using Counter on lists')
print('*********************************************************')
# Print the first ten items from the stations list
print(stations[:10])

# Create a Counter of the stations list: station_count
station_count = Counter(stations)

# Print the station_count
pprint(station_count)

print('*********************************************************')
print('** 03.03 Finding most common elements')
print('*********************************************************')
# Find the 5 most common elements
print(station_count.most_common(5))

print('*********************************************************')
print('** 03.04 Dictionaries of unknown structure - Defaultdict')
print('*********************************************************')
print('** 03.05 Creating dictionaries of an unknown structure')
print('*********************************************************')
# Create an empty dictionary: ridership
ridership = {}

# Iterate over the entries
for date, stop, riders in entries:
    # Check to see if date is already in the ridership dictionary
    if date not in ridership:
        # Create an empty list for any missing date
        ridership[date] = []
    # Append the stop and riders as a tuple to the date keys list
    ridership[date].append((stop, riders))
    
# Print the ridership for '03/09/2016'
pprint(ridership['03/09/2016'][:10])

print('*********************************************************')
print("** 03.06 Safely appending to a key's value list")
print('*********************************************************')
# Create a defaultdict with a default type of list: ridership
ridership = defaultdict(list)

# Iterate over the entries
for date, stop, riders in entries:
    # Use the stop as the key of ridership and append the riders to its value
    ridership[stop].append(riders)
    
# Print the first 10 items of the ridership dictionary
print(list(ridership.items())[:1])

print('*********************************************************')
print('** 03.07 Maintaining Dictionary Order with OrderedDict')
print('*********************************************************')
print('** 03.08 Working with OrderedDictionaries')
print('*********************************************************')
# Create an OrderedDict called: ridership_date
ridership_date = OrderedDict()

# Iterate over the entries
for date, riders in entries_Austin_Forest_Park:
    day = datetime.strptime(date, "%m/%d/%Y").strftime("%a")
    date_day = (date, day)
    
    # If a key does not exist in ridership_date, set it to 0
    if date_day not in ridership_date:
        ridership_date[date_day] = 0
        
    # Add riders to the date key in ridership_date
    ridership_date[date_day] += riders
    
# Print the first 31 records
pprint(list(ridership_date.items())[:31])

print('*********************************************************')
print('** 03.09 Powerful Ordered popping')
print('*********************************************************')
# Print the first key in ridership_date
print(list(ridership_date.keys())[0])

# Pop the first item from ridership_date and print it
print(ridership_date.popitem(last=False))

# Print the last key in ridership_date
print(list(ridership_date.keys())[-1])

# Pop the last item from ridership_date and print it
print(ridership_date.popitem())


print('*********************************************************')
print("** 03.10 What do you mean I don't have any class? Namedtuple")
print('*********************************************************')
print('** 03.11 Creating namedtuples for storing data')
print('*********************************************************')
# Create the namedtuple: DateDetails
DateDetails = namedtuple('DateDetails', ['date', 'stop', 'riders'])

# Create the empty list: labeled_entries
labeled_entries = []

# Iterate over the entries list
for date, stop, riders in entries:
    # Append a new DateDetails namedtuple instance for each entry to labeled_entries
    labeled_entries.append(DateDetails(date, stop, riders))
    
# Print the first 5 items in labeled_entries
pprint(labeled_entries[:5])

print('*********************************************************')
print('** 03.12 Leveraging attributes on namedtuples')
print('*********************************************************')
# Iterate over the first twenty items in labeled_entries
for item in labeled_entries[:20]:
    # Print each item's stop, date and riders
    print(item.stop, item.date, item.riders)

print('*********************************************************')
print('END')
print('*********************************************************')