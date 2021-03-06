# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:52:57 2019

@author: jacqueline.cortez

Capítulo 3. Quantitative comparisons and statistical visualizations
Introduction:
    Visualizations can be used to compare different data in a quantitative manner. 
    This chapter shows several methods for quantitative visualizations.
"""

# Import packages
import pandas as pd
#import numpy as np
#import tabula 
#import math
import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
#import random
import calendar

#from pandas.plotting import register_matplotlib_converters #for conversion as datetime index in x-axis
#from math import radians
#from functools import reduce#import pandas as pd
#from pandas.api.types import CategoricalDtype #For categorical data
#from glob import glob
#from bokeh.io import output_file, show
#from bokeh.plotting import figure

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters() #Require to explicitly register matplotlib converters.

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** 2. Bar chart\n")

print("Reading medals file ...\n")
file = 'Summer_Olympic_1896_to_2008_medals.csv'
medals = pd.read_csv(file, index_col=[0,1])

print("Choosing year 2008 ...\n")
medals_2008 = medals.loc[2008].sort_values('total', ascending=False)
medals_2008_tiny = medals_2008.iloc[0:10]

fig, ax = plt.subplots()
ax.set_title('2. Bar chart') # Add the title

ax.bar(medals_2008_tiny.index, medals_2008_tiny.Gold) # Plot a bar-chart of gold medals as a function of country
ax.set_xticklabels(medals_2008_tiny.index, rotation=90) # Set the x-axis tick labels to the country names
ax.set_ylabel('Number of medals') # Set the y-axis label

plt.show()


print("****************************************************")
print("** 3. Stacked bar chart\n")

fig, ax = plt.subplots()
ax.set_title('3. Stacked bar chart') # Add the title

ax.bar(medals_2008_tiny.index, medals_2008_tiny.Gold, 
       label='Gold', color='gold') # Add bars for "Gold" with the label "Gold"
ax.bar(medals_2008_tiny.index, medals_2008_tiny.Silver, 
       bottom=medals_2008_tiny.Gold, 
       label='Silver', color='silver') # Stack bars for "Silver" on top with label "Silver"
ax.bar(medals_2008_tiny.index, medals_2008_tiny.Bronze, 
       bottom=medals_2008_tiny.Gold+medals_2008_tiny.Silver, 
       label='Bronze', color='Brown') # Stack bars for "Bronze" on top of that with label "Bronze"

ax.set_xticklabels(medals_2008_tiny.index, rotation=90) # Set the x-axis tick labels to the country names
ax.legend() # Display the legend

plt.show()

print("****************************************************")
print("** 3. Stacked bar chart (From DataFrame)\n")

#fig, ax = plt.subplots()
medals_2008_tiny[['Gold','Silver','Bronze']].plot(kind='bar', stacked=True, 
                                                  title="3. Stacked bar chart (From DataFrame)\n",
                                                  color=['Gold','Silver','Brown'],
                                                  sort_columns = True, width=0.9)

plt.show()

print("****************************************************")
print("** 5. Creating histograms\n")

print("Reading the data of the medallist from 2012 summer olympic game...\n")
file = 'Summer_Olympic_2012.csv'
olympic2012 = pd.read_csv(file, parse_dates=['birth'], index_col='name')
olympic2012['sex'] = olympic2012.sex.astype('category')
print(olympic2012.info())

print("Getting the required data...\n")
mens_rowing = olympic2012[(olympic2012.sport == 'Rowing') & (olympic2012.sex == 'M')]
mens_gymnastics = olympic2012[(olympic2012.sport == 'Gymnastics - Artistic') & (olympic2012.sex == 'M')]


fig, ax = plt.subplots()
ax.set_title('5. Creating histograms') # Add the title

ax.hist(mens_rowing.weight) # Plot a histogram of "Weight" for mens_rowing
ax.hist(mens_gymnastics.weight) # Compare to histogram of "Weight" for mens_gymnastics

ax.set_xlabel('Weight (kg)') # Set the x-axis label to "Weight (kg)"
ax.set_ylabel('# of observations') # Set the y-axis label to "# of observations"

plt.show()



print("****************************************************")
print("** 6. Step  histogram\n")

fig, ax = plt.subplots()
ax.set_title('6. Step  histogram') # Add the title

ax.hist(mens_rowing.weight, label='Rowing', histtype='step', bins=5) # Plot a histogram of "Weight" for mens_rowing
ax.hist(mens_gymnastics.weight, label='Gymnastics', histtype='step', bins=5) # Compare to histogram of "Weight" for mens_gymnastics

ax.set_xlabel("Weight (kg)")
ax.set_ylabel("# of observations")

ax.legend(loc='upper left') # Add the legend and show the Figure
plt.show()

print("****************************************************")
print("** 8. Adding error-bars to a bar chart\n")

fig, ax = plt.subplots()
ax.set_title('8. Adding error-bars to a bar chart') # Add the title

ax.bar("Rowing", mens_rowing.height.mean(), yerr=mens_rowing.height.std()) # Add a bar for the rowing "Height" column mean/std
ax.bar("Gymnastics", mens_gymnastics.height.mean(), yerr=mens_gymnastics.height.std()) # Add a bar for the gymnastics "Height" column mean/std
ax.set_ylabel("Height (cm)") # Label the y-axis

plt.show()

print("****************************************************")
print("** 9. Adding error-bars to a plot\n")

print("Reading the data from the weather of Seattle and Austin...\n")
file = 'NOAA_2010_Seattle_and_Austin.csv'
weather = pd.read_csv(file)

seattle_weather = weather.loc[0:11].reset_index()
austin_weather = weather.loc[12:23].reset_index()

seattle_weather['MONTH'] = seattle_weather['DATE'].apply(lambda x: calendar.month_abbr[x]) #calendar.month_name[i]--> get the names
austin_weather['MONTH'] = austin_weather['DATE'].apply(lambda x: calendar.month_abbr[x]) 

fig, ax = plt.subplots()
ax.set_title('9. Adding error-bars to a plot') # Add the title

ax.errorbar(seattle_weather.MONTH, seattle_weather['MLY-TAVG-NORMAL'], # Add Seattle temperature data in each month with error bars
            seattle_weather['MLY-TAVG-STDDEV'], 
            label='Seattle')
ax.errorbar(austin_weather.MONTH, austin_weather['MLY-TAVG-NORMAL'], # Add Austin temperature data in each month with error bars
            austin_weather['MLY-TAVG-STDDEV'], 
            label='Austin') 
ax.set_xlabel('Year 2010') # Set the x-axis label
ax.set_ylabel('Temperature (Fahrenheit)') # Set the y-axis label

ax.legend()

plt.show()



print("****************************************************")
print("** 10. Creating boxplots\n")

fig, ax = plt.subplots()
ax.set_title('10. Creating boxplots') # Add the title

ax.boxplot([mens_rowing.height, mens_gymnastics.height]) # Add a boxplot for the "Height" column in the DataFrames

ax.set_xticklabels(['Rowing', 'Gymnastics']) # Add x-axis tick labels
ax.set_ylabel("Height (cm)") # Add a y-axis label

plt.show()



print("****************************************************")
print("** 12. Simple scatter plot\n")

print("Reading the data from the CO2 climate chage...\n")
file = 'CO2_climate_change_1850-2014.csv'
climate_change = pd.read_csv(file, parse_dates=True, index_col=0)

fig, ax = plt.subplots()
ax.set_title('12. Simple scatter plot') # Add the title

ax.scatter(climate_change.co2, climate_change.relative_temp) # Add data: "co2" on x-axis, "relative_temp" on y-axis
ax.set_xlabel('CO2 (ppm)') # Set the x-axis label to "CO2 (ppm)"
ax.set_ylabel('Relative temperature (C)') # Set the y-axis label to "Relative temperature (C)"

plt.show()



print("****************************************************")
print("** 13. Encoding time by color\n")

fig, ax = plt.subplots()
ax.set_title('13. Encoding time by color') # Add the title

ax.scatter(climate_change.co2, climate_change.relative_temp, c=climate_change.index) # Add data: "co2", "relative_temp" as x-y, index as color
ax.set_xlabel('CO2 (ppm)') # Set the x-axis label to "CO2 (ppm)"
ax.set_ylabel('Relative temperature (C)') # Set the y-axis label to "Relative temperature (C)"


plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")