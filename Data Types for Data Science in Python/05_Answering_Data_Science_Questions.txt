# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:47:44 2021

@author: jaces
"""
# Import libraries
import csv
import calendar

from pprint import pprint
from collections import Counter
from collections import defaultdict
from datetime import datetime

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 05 Answering Data Science Questions')
print('*********************************************************')
print('** 05.01 Counting within Date Ranges')
print('*********************************************************')
print('** 05.02 Reading your data with CSV Reader and Establishing your Data Containers')
print('*********************************************************')
# Create the file object: csvfile
file = 'data/crime_sampler.csv'
with open(file, 'r') as csvfile:
    # Create a list: crime_data, with the date, type of crime, location description, and arrest on the file object
    crime_data = [(row[0], row[2], row[4], row[5]) for row in csv.reader(csvfile)]
    
# Remove the first element from crime_data
del crime_data[0]

# Print the first 10 records
pprint(crime_data[:10])

print('*********************************************************')
print('** 05.03 Find the Months with the Highest Number of Crimes')
print('*********************************************************')
# Create a Counter Object: crimes_by_month
crimes_by_month = Counter()

# Loop over the crime_data list
for crime in crime_data:
    
    # Convert the first element of each item into a Python Datetime Object: date
    date = datetime.strptime(crime[0], '%m/%d/%Y %I:%M:%S %p')
    
    # Increment the counter for the month of the row by one
    crimes_by_month[date.month] += 1
    
# Print the 3 most common months for crime
print(crimes_by_month.most_common(3))

print('*********************************************************')
print('** 05.04 Transforming your Data Containers to Month and Location')
print('*********************************************************')
# Create a dictionary that defaults to a list: locations_by_month
locations_by_month = defaultdict(list)

# Loop over the crime_data list
for row in crime_data:
    # Convert the first element to a date object
    date = datetime.strptime(row[0], '%m/%d/%Y %I:%M:%S %p')
    
    # If the year is 2016 
    if date.year == 2016:
        # Set the dictionary key to the month and append the location (fifth element) to the values list
        locations_by_month[date.month].append(row[2])

# Print the dictionary
for m in sorted(locations_by_month):
    print('In {}, we registered {} locations.'.format(datetime(1900, m, 1).strftime('%b'), len(locations_by_month[m])))
    print(f'In {calendar.month_abbr[m]}, we registered {len(locations_by_month[m])} locations.')

print('*********************************************************')
print('** 05.05 Find the Most Common Crimes by Location Type by Month in 2016')
print('*********************************************************')
# Loop over the items from locations_by_month using tuple expansion of the month and locations
for month, locations in locations_by_month.items():
    # Make a Counter of the locations
    location_count = Counter(locations)
    # Print the month and the most common location
    print(month, ':', location_count.most_common(5))
    

# Create a dictionary that defaults to a list: locations_by_month
locations_by_month2 = defaultdict(Counter)

# Loop over the crime_data list
for row in crime_data:
    # Convert the first element to a date object
    date = datetime.strptime(row[0], '%m/%d/%Y %I:%M:%S %p')
    
    # If the year is 2016 
    if date.year == 2016:
        # Set the dictionary key to the month and append the location (fifth element) to the values list
        locations_by_month2[date.month].update({row[2]: 1})

# Print the dictionary
for m in sorted(locations_by_month2):
    print('In {}, most common crime locations: '.format(datetime(1900, m, 1).strftime('%b')))
    print(locations_by_month2[m].most_common(5))

print('*********************************************************')
print('** 05.06 Dictionaries with Time Windows for Keys')
print('*********************************************************')
print('** 05.07 Reading your Data with DictReader and Establishing your Data Containers')
print('*********************************************************')
# Create the CSV file: csvfile
with open(file, 'r') as csvfile:
    # Create a dictionary that defaults to a list: crimes_by_district
    crimes_by_district = defaultdict(list)
    
    # Loop over a DictReader of the CSV file
    for row in csv.DictReader(csvfile):
        # Pop the district from each row: district
        district = row.pop('District')
        # Append the rest of the data to the list for proper district in crimes_by_district
        crimes_by_district[int(district)].append(row)

for district in sorted(crimes_by_district):
    print(f'In {district:>2}, it has been registered {len(crimes_by_district[district]):>4} crime(s).')

pprint(crimes_by_district[district][0])

print('*********************************************************')
print('** 05.08 Determine the Arrests by District by Year')
print('*********************************************************')
# Loop over the crimes_by_district using expansion as district and crimes
for district, crimes in dict(sorted(crimes_by_district.items())).items():
    # Create an empty Counter object: year_count
    year_count = Counter()
    
    # Loop over the crimes:
    for crime in crimes:
        # If there was an arrest
        if crime['Arrest'] == 'true':
            # Convert the Date to a datetime and get the year
            year = datetime.strptime(crime['Date'], '%m/%d/%Y %I:%M:%S %p').year
            # Increment the Counter for the year
            year_count[year] += 1
            
    # Print the district and  the counter
    print(f'In district {district:>2}: {year_count}')
    
print('*********************************************************')
print('** 05.09 Unique Crimes by City Block')
print('*********************************************************')
crimes_by_district_and_block = defaultdict(dict)
for district, crimes in dict(sorted(crimes_by_district.items())).items():
    
    crimes_by_block = defaultdict(set)
    for crime in crimes:
        crimes_by_block[crime['Block']].add(crime['Primary Type'])
    crimes_by_district_and_block[district] = crimes_by_block
    
# Print data for district = 20
district = 1
for i, block in enumerate(sorted(crimes_by_district_and_block[district]), start=1):
    if i > 5: break
    print('District No.{}: Block "{}" registered {} different types of crime(s).'.format(
                district, block,
                len(crimes_by_district_and_block[district][block])))
    
# Create a unique list of crimes for the first block: n_state_st_crimes
n_state_st_crimes = set(crimes_by_district_and_block[1]['001XX N STATE ST'])

# Print the list
print(n_state_st_crimes)

# Create a unique list of crimes for the second block: w_terminal_st_crimes
w_terminal_st_crimes = set(crimes_by_district_and_block[16]['0000X W TERMINAL ST'])

# Print the list
print(w_terminal_st_crimes)

# Find the differences between the two blocks: crime_differences
crime_differences = n_state_st_crimes.difference(w_terminal_st_crimes)

# Print the differences
print(crime_differences)

print('*********************************************************')
print('** 05.10 Final thoughts')
print('*********************************************************')
print('END')
print('*********************************************************')