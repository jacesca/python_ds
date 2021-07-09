# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:02:10 2019

@author: jacqueline.cortez

Work for result inside
"""

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#import sys
#sys.setdefaultencoding('utf-8')

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.0f}'.format 
#pd.reset_option("all")

def reading_countriesfiles():
    # reading LifeExp and GDP
    dict_file = {"East Asia & Pacific": ["World_A_EastAsiaPacific_2017_1960.csv", "yellow"],
                 "Europe & Central Asia": ["World_A_EuropeCentralAsia_2017_1960.csv", "pink"],
                 "Latin America & Caribbean": ["World_A_LatinAmericaCaribbean_2017_1960.csv", "blue"],
                 "Middle East & North Africa": ["World_A_MiddleEastNorthAfrica_2017_1960.csv", "black"],
                 "North America": ["World_A_NorthAmerica_2017_1960.csv", "red"],
                 "South Asia": ["World_A_SouthAsia_2017_1960.csv", "green"],
                 "Sub-Saharan Africa": ["World_A_Sub-SaharanAfrica_2017_1960.csv", "brown"]}
        
    w_2017_1960 = pd.DataFrame({})
        
    for region, (file, color) in dict_file.items():
        r = pd.read_csv(file, 
                        names=["pais","p","serie","s","1960","2017"], 
                        skiprows=1,
                        #index_col="pais",
                        skipfooter=5,
                        usecols=["pais", "serie", "1960", "2017"],
                        na_values="NA",
                        encoding='utf-8',
                        engine='python') #This line is because skipfooter
    
        #print("** {}:".format(region.upper()))
        #print("Initial Shape: {}".format(r.shape))
    
        # Evaluating no duplicates rows
        #if (r[r.duplicated(["pais", "serie"])].shape[0] == 0):
        #    print("No duplicated data...\n")
        # Evaluating no null values in series
        #if r[r["serie"].isnull()].shape[0] == 0:
        #    print("No error data in 'serie' column...\n")
    
        r_2017 = r.pivot(index="pais", columns="serie", values="2017") # Getting data from 2017
        r_2017.columns = ["2017_LifeExp_F", "2017_LifeExp", "2017_LifeExp_M", "2017_PIB", "2017_POP_M", "2017_POP_F", "2017_POP"]
        
        r_1960 = r.pivot(index="pais", columns="serie", values="1960") # Getting data from 1969
        r_1960.columns = ["1960_LifeExp_F", "1960_LifeExp", "1960_LifeExp_M", "1960_PIB", "1960_POP_M", "1960_POP_F", "1960_POP"]
        
        #Concatenating in one df
        r_2017_1960 = pd.concat([r_1960, r_2017], axis=1)
        r_2017_1960["Region"] = region
        r_2017_1960["Color"] = color
        #print("Final Shape: {}\n".format(r_2017_1960.shape))

        w_2017_1960 = w_2017_1960.append(r_2017_1960, sort=True)

    #print("Initial Shape of 1960 y 2017 Data:{}".format(w_2017_1960.shape))

    # Delete rows with null values in any column
    w_2017_1960.dropna(axis='index', how='any', inplace=True)
            
    #print("Final Shape of 1960 y 2017 Data:{}".format(w_2017_1960.shape))
    #print("Columns of 1960 y 2017 Data:\n{}".format(w_2017_1960.columns.values))
    #print("Head of 1960 y 2017 Data:\n{}".format(w_2017_1960.head()))

    return w_2017_1960

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Line plot (1)\n")

# reading pop and year
file = "WorldPopulation.csv"
pop = list(np.loadtxt(file, skiprows=1, usecols=1, delimiter=";", dtype=np.float64))
year = list(np.loadtxt(file, skiprows=1, usecols=0, delimiter=";", dtype=np.int32))

# Print the last item from year and pop
print(year[len(year)-1])
print('{:,.0f}'.format(pop[-1]))

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year,pop)

# Display the plot with plt.show()
plt.show()
plt.clf()

print("****************************************************")
print("** Line plot (3)\n")

#Reading the files
world_2017_1960 = reading_countriesfiles()

#Getting the two list
gdp_cap = list(world_2017_1960['2017_PIB'])
life_exp = list(world_2017_1960['2017_LifeExp'])

# Make a line plot, gdp_cap on the x-axis, life_exp on the y-axis
plt.plot(gdp_cap, life_exp)

# Display the plot
plt.show()
plt.clf()

print("****************************************************")
print("** Scatter Plot (1)")

# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)

# Put the x-axis on a logarithmic scale
plt.xscale("log")

# Show plot
plt.show()

print("****************************************************")
print("** Scatter Plot (2)")

#Getting the pop list
pop = list(world_2017_1960['2017_POP'])

# Build Scatter plot
plt.scatter(pop, life_exp)

# Show plot
plt.show()
plt.clf()

print("****************************************************")
print("** Build a histogram (1)")

# Create histogram of life_exp data
plt.hist(life_exp)

# Display histogram
plt.show()
plt.clf()

print("****************************************************")
print("** Build a histogram (2): bins")
# Build histogram with 5 bins
plt.hist(life_exp, bins=5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(life_exp, bins=20)

# Show and clean up again
plt.show()
plt.clf()

print("****************************************************")
print("** Build a histogram (3): compare")

# Histogram of life_exp, 15 bins
plt.hist(life_exp, bins=15)

# Show and clear plot
plt.show()
plt.clf()

#Getting the two list
life_exp1960 = list(world_2017_1960['1960_LifeExp'])

# Histogram of life_exp1950, 15 bins
plt.hist(life_exp1960, bins=15, histtype="step")

# Show and clear plot again
plt.show()
plt.clf()

print("****************************************************")
print("** Labels")

# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2017'

# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)

# Add title
plt.title(title)

# After customizing, display the plot
plt.show()
plt.clf()

print("****************************************************")
print("** Ticks")

# Scatter plot
plt.scatter(gdp_cap, life_exp)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in Million USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2017')

# Definition of tick_val and tick_lab
tick_val = [1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000]
tick_lab = ['1,000', '10,000', '100,000', '1,000,000', '10,000,000']

# Adapt the ticks on the x-axis
plt.xticks(tick_val, tick_lab)

# After customizing, display the plot
plt.show()
plt.clf()

print("****************************************************")
print("** Sizes")

# Store pop as a numpy array: np_pop
np_pop = np.array(pop)

# Double np_pop
np_pop =  np_pop/1000000*5

# Update: set s argument to np_pop, big size
plt.scatter(gdp_cap, life_exp, s = np_pop)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in Million USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2017')
plt.xticks(tick_val,tick_lab)

# Display the plot
plt.show()
plt.clf()

print("****************************************************")
print("** Colors")

# Setting the colors
col = world_2017_1960['Color']

# Specify c and alpha inside plt.scatter()
plt.scatter(x = gdp_cap, y = life_exp, s = np_pop, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in Million USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2017')
plt.xticks(tick_val, tick_lab)

# Show the plot
plt.show()
plt.clf()

print("****************************************************")
print("** Additional Customizations")

# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np_pop, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in Million USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2017')
plt.xticks(tick_val, tick_lab)

# Additional customizations
plt.text(10000000000000, 75, 'China', fontsize=12)
plt.text((1000000000000 + 10000000000000)/5.5, 68, 'India', color='white', fontsize=12)

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()

print("\n****************************************************")
print("** END                                            **")
print("****************************************************")