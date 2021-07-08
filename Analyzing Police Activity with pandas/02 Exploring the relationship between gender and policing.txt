# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:36:23 2019

@author: jacqueline.cortez

Introduction:
    Does the gender of a driver have an impact on 
    police behavior during a traffic stop? 
    In this chapter, you will explore that question 
    while practicing filtering, grouping, method chaining, 
    Boolean math, string methods, and more!
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
ri["f"] = 1

# Print the head and the columns of the dataframe
print(ri.head())
print("Columns: {}".format(ri.columns.values))

print("****************************************************")
print("EXAMINING TRAFFIC VIOLATIONS")
#ri_crime = ri.groupby("violation_raw")["violation_raw"].count()
rule_broken = ri.violation_raw.value_counts(normalize=True)*100 #Get the frecuency of the column
print(rule_broken)
# Graph the frecuency
rule_broken.plot(kind="bar", color="green", alpha=0.5, width=0.8)
plt.title("Violation Raw [2005-2015 Rhode Island]", color="red")
plt.ylabel("Percentage %")
plt.xlabel("Violation Raw")
plt.show()

print("****************************************************")
print("COMPARING VIOLATIONS BY GENDER")
print("Gender register: {}".format(ri.driver_gender.unique())) #Getting the unique values
# Getting the frecuency by gender

#First way : frecuency count and normalize en cada delito
rbb_gender = pd.DataFrame({})
rbb_gender["female"] = ri[ri.driver_gender == "female"].violation_raw.value_counts(normalize=True)*100
rbb_gender["male"] = ri[ri.driver_gender == "male"].violation_raw.value_counts(normalize=True)*100
print(rbb_gender)

"""
#Second way : Frecuency count and normalize para la poblacion total por delito
#           normalize : boolean, {‘all’, ‘index’, ‘columns’}, or {0,1}, default False
#           Normalize by dividing all values by the sum of values.
#                 If passed ‘all’ or True, will normalize over all values.
#                 If passed ‘index’ will normalize over each row.
#                 If passed ‘columns’ will normalize over each column.
#                 If margins is True, will also normalize margin values.

rbb_gender = pd.crosstab(ri.violation_raw, ri.driver_gender, normalize="columns")
print(rbb_gender)
"""
#rule_broken = ri.groupby(["driver_gender","violation_raw"])["f"].count() 
#rule_broken = rule_broken.swaplevel(1,0).unstack(level="driver_gender")
#print(rule_broken)
# Graph the frecuency
rbb_gender.plot(kind="bar", logy=False, width=0.9, color=["pink","navy"])
plt.title("Violation Raw by Gender [2005-2015 Rhode Island]", color="red")
plt.ylabel("Percentage %")
plt.xlabel("Violation Raw")
plt.show()

print("****************************************************")
print("COMPARING SPEEDING OUTCOMES BY GENDER")
rbb_gender = pd.DataFrame({})
speeding = ri[ri.violation_raw == "Speeding"]
rbb_gender["female"] = speeding[speeding.driver_gender == "female"].stop_outcome.value_counts(normalize=True)*100
rbb_gender["male"] = speeding[speeding.driver_gender == "male"].stop_outcome.value_counts(normalize=True)*100
print(rbb_gender)
# Graph the frecuency
rbb_gender.plot(kind="bar", logy=False, width=0.9, color=["pink","navy"])
plt.title("Speeding outcomes by Gender [2005-2015 Rhode Island]", color="red")
plt.ylabel("Percentage %")
plt.xlabel("Stop Outcomes")
plt.show()

print("****************************************************")
print("COMPARING VEHICLES SEARCHES RATES\n")
print("Vehícules searches rates: {}% ".format(round(ri.search_conducted.mean()*100,2)))
print("Type of search_conducted column: {}\n".format(ri.search_conducted.dtype))

print("****************************************************")
print("COMPARING VEHICLES SEARCHES RATES BY GENDER\n")
# Plotting the graph
(ri.groupby("driver_gender").search_conducted.mean()*100).plot(kind="bar", logy=False, width=0.9, color=["pink","navy"])
plt.title("Vehicles searches by Gender [2005-2015 Rhode Island]", color="red")
plt.ylabel("Percentage %")
plt.xlabel("Driver Gender")
plt.show()

print("****************************************************")
print("COMPARING VEHICLES SEARCHES  VS VIOLATIONS BY GENDER")
print("Gender register: {}".format(ri.driver_gender.unique())) #Getting the unique values
# Getting the frecuency by gender
rbb_gender = pd.DataFrame({})
rbb_gender["female"] = ri[ri.driver_gender == "female"].groupby("violation_raw").search_conducted.mean()*100
rbb_gender["male"] = ri[ri.driver_gender == "male"].groupby("violation_raw").search_conducted.mean()*100
print(rbb_gender)
# Graph the frecuency
rbb_gender.plot(kind="bar", logy=False, width=0.9, color=["pink","navy"])
plt.title("Vehicles Searches vs Violation Raw by Gender [2005-2015 Rhode Island]", color="red")
plt.ylabel("Vehicles Searches Percentage %")
plt.xlabel("Violation Raw")
plt.show()

print("****************************************************")
print("HOW MANY TIMES WAS 'Terry Frisk' THE ONLY SEARCH TYPE?")
print("Values for search_type:\n")
print(ri[ri.search_conducted == True].search_type.value_counts())
print("")
print("how many times was 'Terry Frisk' the only search type? {} veces".format(ri[ri.search_type.str.contains('Terry Frisk', na=False)].search_conducted.sum()))

print("****************************************************")
print("COMPARING FRISK RATES BY GENDER")
# Check if 'search_type' contains the string 'Terry Frisk'
ri["frisk"] = ri.search_type.str.contains('Terry Frisk', na=False)
# Plotting the graph
(ri[ri.search_conducted == True].groupby("driver_gender").frisk.mean()*100).plot(kind="bar", logy=False, width=0.9, color=["pink","navy"])
plt.title("Frisk rates by Gender [2005-2015 Rhode Island]", color="red")
plt.ylabel("Percentage %")
plt.xlabel("Driver Gender")
plt.show()

print("****************************************************")
print("The file info")
print("Columns: {}".format(ri.columns.values))

print("****************************************************")
print("** END                                            **")
print("****************************************************")