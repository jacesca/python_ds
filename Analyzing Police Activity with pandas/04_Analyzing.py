# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:45:55 2019

@author: jacqueline.cortez

Introduction:
    In this chapter, you will use a second dataset to explore the impact 
    of weather conditions on police behavior during traffic stops. 
    You will practice merging and reshaping datasets, assessing whether 
    a data source is trustworthy, working with categorical data, and 
    other advanced skills.
"""

# Import the pandas library as pd
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype #For categorical data

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")

# Read the file into a DataFrame named ri
file_w = "RIWeather_2004_2016.csv"
weather = pd.read_csv(file_w, parse_dates=True)
# Drop the unnecesary column
weather.drop(["Unnamed: 0"], axis='columns', inplace=True)

# Read the file into a DataFrame named ri
file_p = "rhode_island_2005_2015_min.csv"
ri = pd.read_csv(file_p, parse_dates=True, index_col='stop_datetime')

# Print the head of the dataframe
print(weather.head())

print("****************************************************")
print("** Plotting the temperature\n")

print(weather[['TMIN','TAVG','TMAX']].describe())

# Create a box plot of the temperature columns
weather[['TMIN','TAVG','TMAX']].plot(kind="box")

# Display the plot
plt.show()

print("****************************************************")
print("Plotting the temperature difference")

# Create a 'TDIFF' column that represents temperature difference
weather['TDIFF'] = weather['TMAX'] - weather['TMIN']

# Describe the 'TDIFF' column
#print(weather['TDIFF'].describe())

# Create a histogram with 20 bins to visualize 'TDIFF'
weather.TDIFF.plot(kind='hist', bins=20)
#weather['TDIFF'].hist(bins=20)

# Show vertical grids
plt.grid(axis='x')

# Display the plot
plt.show()

print("****************************************************")
print("Counting bad weather conditions")

# Copy 'WT01' through 'WT22' to a new DataFrame
WT = weather.loc[:, 'WT01':'WT22']

# Calculate the sum of each row in 'WT'
weather['bad_conditions'] = WT.sum(axis='columns')

# Replace missing values in 'bad_conditions' with '0'
weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')

# Create a histogram to visualize 'bad_conditions'
weather.bad_conditions.plot(kind='hist', bins=20)

# Display the plot
plt.show()

print("****************************************************")
print("Rating the weather conditions\n")

# Count the unique values in 'bad_conditions' and sort the index
print(weather.bad_conditions.value_counts().sort_index())

# Create a dictionary that maps integers to strings
mapping = {0:'good', 
           1:'bad', 2:'bad', 3:'bad', 4:'bad',
           5:'worse', 6:'worse', 7:'worse', 8:'worse', 9:'worse'}

# Convert the 'bad_conditions' integers to strings using the 'mapping'
weather['rating'] = weather.bad_conditions.map(mapping)

# Count the unique values in 'rating'
print(weather.rating.value_counts(dropna=False))

print("****************************************************")
print("Changing the data type to category\n")

"""
FutureWarning: 
    specifying 'categories' or 'ordered' in .astype() is deprecated; 
    pass a CategoricalDtype instead

# Create a list of weather ratings in logical order
cats = ['good', 'bad', 'worse']

# Change the data type of 'rating' to category
weather['rating'] = weather.rating.astype('category', ordered=True, categories=cats)
"""

#Create categorical type data to use
cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  
                        ordered=True)

# Change the data type of 'rating' to category
weather['rating'] = weather.rating.astype(cats)

# Examine the head of 'rating'
print(weather['rating'].head())

print("****************************************************")
print("Preparing the DataFrames\n")

# Reset the index of 'ri'
ri.reset_index(inplace=True)

# Examine the head of 'ri'
print(ri.head())

# Create a DataFrame from the 'DATE' and 'rating' columns
weather_rating = weather.loc[:,['date','rating']]

# Examine the head of 'weather_rating'
print(weather_rating.head())

print("****************************************************")
print("Merging the DataFrames\n")

# Examine the shape of 'ri' and 'weather_rating' dataframes
print("'ri' and 'weather_rating' shape respective: {} and {}.".format(ri.shape, weather_rating.shape))

# Merge 'ri' and 'weather_rating' using a left join
ri_weather = pd.merge(left=ri, right=weather_rating, left_on='stop_date', right_on='date', how='left')

# Examine the shape of 'ri_weather'
print("The shape of the merge result 'ri_weather': {}.\n".format(ri_weather.shape))

# Set 'stop_datetime' as the index of 'ri_weather'
ri_weather.set_index('stop_datetime', inplace=True)

print("****************************************************")
print("Comparing arrest rates by weather rating\n")

# Calculate the overall arrest rate
print("Overall arrest rate: {:,.2f} %".format(ri_weather.is_arrested.mean()*100))

# Calculate the arrest rate for each 'rating'
print("\nArrest rate (%) for each weather type:")
arrest_rate = ri_weather.groupby('rating').is_arrested.mean()*100
print(arrest_rate)
arrest_rate.plot(kind='bar')
plt.xlabel("Weather")
plt.ylabel("Arrest Rate (%)")
plt.show()

# Calculate the arrest rate for each 'violation' and 'rating'
arrest_rate = ri_weather.groupby(['violation_raw', 'rating']).is_arrested.mean()*100

print("****************************************************")
print("Selecting from a multi-indexed Series\n")

print("Arrest rate (%) per violation and weather type:")
print(arrest_rate)

# Print the arrest rate for Violation of City/Town Ordinance in bad weather
arrest_rate_moving = arrest_rate.loc['Violation of City/Town Ordinance','bad']
print("\nArrest rate for moving violations in bad weather: {} %".format(arrest_rate_moving))

# Print the arrest rates for speeding violations in all three weather conditions
print("Arrest rate (%) for speeding violations in all three weather conditions:")
print(arrest_rate.loc['Speeding'])

#idx = pd.IndexSlice
#print(arrest_rate.loc[idx[:,'bad'],:])

print("\n****************************************************")
print("Reshaping the arrest rate data\n")

# First method
arrest_rate = ri_weather.groupby(['violation_raw', 'rating']).is_arrested.mean()*100
arrest_rate = arrest_rate.unstack() # Unstack the 'arrest_rate' Series into a DataFrame
print(arrest_rate)

# Second way: Create the same DataFrame using a pivot table
arrest_rate = ri_weather.pivot_table(index='violation_raw', columns='rating', values='is_arrested')*100 
print("\n{}".format(arrest_rate))

# Showing the graph
arrest_rate.plot(kind='bar', color=['blue','yellow','red'], stacked=False, logy=False)
plt.ylabel("Arrest Rate (%)")
plt.show()

print("****************************************************")
# Print the columns of the dataframe
print("The weather of Rhode Island dataframe info:")
print(weather.columns)
#print(weather.info())

# Print the columns of the dataframe
print("\nThe needed data set of Rhode Island's weather:")
print(weather_rating.columns)
#print(weather.info())

print("\nThe road stop police of Rhode Island dataframe info:")
print(ri.columns)
#print(ri.info())

print("****************************************************")
print("** END                                            **")
print("****************************************************")
