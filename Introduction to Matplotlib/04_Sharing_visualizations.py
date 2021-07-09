# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:42:49 2019

@author: jacqueline.cortez

Capítulo 4. Introduction to Matplotlib
Introduction:
    This chapter shows how to share your visualizations with others: how 
    to save your figures as files, how to control their look and feel, 
    and how to automate their creation based on input data.
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
print("** 3. Switching between styles (ggplot and Solarize_Light2)\n")

print("Reading the data from the weather of Seattle and Austin...\n")
file = 'NOAA_2010_Seattle_and_Austin.csv'
weather = pd.read_csv(file)

seattle_weather = weather.loc[0:11].reset_index()
austin_weather = weather.loc[12:23].reset_index()

seattle_weather['MONTH'] = seattle_weather['DATE'].apply(lambda x: calendar.month_abbr[x]) #calendar.month_name[i]--> get the names
austin_weather['MONTH'] = austin_weather['DATE'].apply(lambda x: calendar.month_abbr[x]) 


# Use the "ggplot" style and create new Figure/Axes
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_title('3. Switching between styles (ggplot)') # Add the title

ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
#fig.set_size_inches([3, 5]) # Set figure dimensions and save as a PNG
plt.show()


# Use the "Solarize_Light2" style and create new Figure/Axes
plt.style.use('default')
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
ax.set_title('3. Switching between styles (Solarize_Light2)') # Add the title

ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()


print("****************************************************")
print("** 5. Saving a file several times\n")

fig.savefig('austin_weather.png') # Save as a PNG file
fig.savefig('austin_weather_300dpi.png', dpi=300) # Save as a PNG file with 300 dpi


print("****************************************************")
print("** 6. Save a figure with different sizes\n")

fig.set_size_inches([3, 5]) # Set figure dimensions and save as a PNG
fig.savefig('austin_weather_3_5.png')

fig.set_size_inches([5, 3]) # Set figure dimensions and save as a PNG
fig.savefig('austin_weather_5_3.png')



print("****************************************************")
print("** 8. Unique values of a column\n")
print("** 9. Automate your visualization\n")

print("Reading the data of the medallist from 2012 summer olympic game...\n")
file = 'Summer_Olympic_2012.csv'
olympic2012 = pd.read_csv(file, parse_dates=['birth'], index_col='name')
olympic2012['sex'] = olympic2012.sex.astype('category')

print("Getting the required data...\n")

sports_column = olympic2012.sport # Extract the "Sport" column
sports = sports_column.unique() # Find the unique values of the "Sport" column

print("Identifier sports: {}\n".format(sports)) # Print out the unique sports values

# Use the "Solarize_Light2" style and create new Figure/Axes
plt.style.use('default')
fig, ax = plt.subplots()
ax.set_title('9. Automate your visualization') # Add the title

for sport in sports: # Loop over the different sports branches
    sport_df = olympic2012[olympic2012.sport==sport]   # Extract the rows only for this sport
    ax.bar(sport, sport_df.weight.mean(), yerr=sport_df.weight.std())   # Add a bar for the "Weight" mean with std y error bar

ax.set_ylabel("Height (cm)")
ax.set_xticklabels(sports, rotation=90)

# Save the figure to file
plt.show()
fig.savefig('olympic2012_sports_weights.png')

print("****************************************************")
print("** END                                            **")
print("****************************************************")