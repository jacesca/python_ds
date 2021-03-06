# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:56:47 2019

@author: jacqueline.cortez

Introduction:
    Are you more likely to get arrested at a certain time of day? 
    Are drug-related stops on the rise? In this chapter, you will 
    answer these and other questions by analyzing the dataset 
    visually, since plots can help you to understand trends in 
    a way that examining the raw data cannot.
"""

# Import the pandas library as pd
import pandas as pd
import matplotlib.pyplot as plt

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.0f}'.format 
#pd.reset_option("all")


print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")

# Read the file into a DataFrame named ri
file = "rhode_island_2005_2015.csv"
ri = pd.read_csv(file, parse_dates=True, index_col='stop_datetime')
#ri["f"] = 1

# Print the head and the columns of the dataframe
print(ri.head())
print("Columns: {}".format(ri.columns.values))

print("****************************************************")
print("CALCULATING THE HOURLY ARREST RATE\n")

# Calculate the overall arrest rate
print("Overall arrest rate: {}%\n".format(ri.is_arrested.mean()*100))

# Calculate the hourly arrest rate order by desc rate
print("Hourly arrest rate:")
print(ri.groupby(ri.index.hour)["is_arrested"].mean().sort_values(ascending=False))

# Save the hourly arrest rate
hourly_arrest_rate = ri.groupby(ri.index.hour)["is_arrested"].mean()
print("Type of the index of hourly_arrest_rate: {}".format(hourly_arrest_rate.index.dtype))

print("")
print("****************************************************")
print("PLOTTING THE HOURLY ARREST RATE\n")
# Create a line plot of 'hourly_arrest_rate'
hourly_arrest_rate.plot()

# Add the xlabel, ylabel, and title
plt.xlabel("Hour")
plt.ylabel("Arrest Rate")
plt.xticks(range(0, 24, 2)) #Make a list of 2 in 2
plt.title("Arrest Rate by Time of Day")

# Display the plot
plt.show()

print("")
print("****************************************************")
print("PLOTTING DRUG-RELATED STOP\n")
# Calculate the annual rate of drug-related stops
print(ri.drugs_related_stop.resample("A").mean())

# Save the annual rate of drug-related stops
annual_drug_rate = ri.drugs_related_stop.resample("A").mean()

# Create a line plot of 'annual_drug_rate'
annual_drug_rate.plot()
plt.ylabel("Anual Drug-related Stop Rate")
# Create a datetime range, ex --> pd.date_range(start='2005/12/31', end='2015/12/31', dtype='datetime64[ns]', freq='D'freq='A') 
# Showing all xticks.
plt.xticks(pd.date_range(start='2005/12/31', end='2015/12/31', freq='A'),
           range(2005,2016), rotation=90)
plt.title("Drug-related Stop Rate by Year")

# Display the plot
plt.show()

print("")
print("****************************************************")
print("COMPARING DRUG AND SEARCH RATE\n")
# Calculate and save the annual search rate
annual_search_rate = ri.search_conducted.resample("A").mean()

# Concatenate 'annual_drug_rate' and 'annual_search_rate'
annual = pd.concat([annual_drug_rate, annual_search_rate], axis="columns")

# Create subplots from 'annual'
axes = annual.plot(subplots=True)
axes[0].set_title("Comparing Drug and Search Rate by Year")
axes[0].set_ylabel('Rate')
axes[1].set_ylabel('Rate')
plt.xticks(pd.date_range(start='2005/12/31', end='2015/12/31', freq='A'),
           range(2005,2016), rotation=90)

# Display the subplots
plt.show()

print("****************************************************")
print("TALLYING VIOLATIONS BY DISTRICT\n")

# Create a frequency table of districts and violations
all_zones = pd.crosstab(ri.district, ri.violation_raw)
print(all_zones)
all_zones.plot(kind='bar')
plt.show()

# Select rows 'Zone K1' through 'Zone K3'
k_zones = all_zones.loc[all_zones.index.str.contains("K"),:]
print(k_zones)
k_zones.plot(kind='bar')
plt.show()

print("****************************************************")
print("PLOTTING VIOLATIONS BY DISTRICT\n")

# Create a stacked bar plot of 'k_zones'
k_zones.plot(kind='bar', stacked=True)

# Display the plot
plt.show()

print("****************************************************")
print("CONVERTING STOP DURATIONS TO NUMBERS\n")

# Print the unique values in 'stop_duration'
print(ri.stop_duration.unique())

# Create a dictionary that maps strings to integers
mapping = {'0-15 Min': 8, '16-30 Min': 23, '30+ Min': 45}

# Convert the 'stop_duration' strings to integers using the 'mapping'
ri['stop_minutes'] = ri.stop_duration.map(mapping)

# Print the unique values in 'stop_minutes'
print(ri['stop_minutes'].unique())

print("****************************************************")
print("PLOTTING STOP LENGHT")

# Save the resulting Series as 'stop_length'
stop_length = ri.groupby('violation_raw').stop_minutes.mean()

# Sort 'stop_length' by its values and create a horizontal bar plot
stop_length.sort_values().plot(kind='barh')

# Display the plot
plt.show()

print("****************************************************")
print("THE FILE INFO")
print("Columns: {}".format(ri.columns.values))

# Saving the dataframe
out_file = "rhode_island_2005_2015_min.csv"
ri.to_csv(out_file)
print("Data saved as {} file.\n".format(out_file))

print("****************************************************")
print("** END                                            **")
print("****************************************************")














