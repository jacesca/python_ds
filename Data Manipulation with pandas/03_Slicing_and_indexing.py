# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:37:38 2020

@author: jacesca@gmail.com
Chapter 3: Slicing and indexing
    Indexes are supercharged row and column names. Learn how they can be combined 
    with slicing for powerful DataFrame subsetting.
Source: https://learn.datacamp.com/courses/data-manipulation-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
import pandas as pd

###############################################################################
## Preparing the environment
###############################################################################
    
# Global variables
#SEED = 42
#np.random.seed(SEED) 

#Global configuration
#np.set_printoptions(formatter={'float': '{:,.3f}'.format})
#pd.options.display.float_format = '{:,.3f}'.format 

# Read the data
animal_shelter = pd.read_pickle("animal_shelter.pkl.bz2", compression='bz2')
temperatures = pd.read_csv("temperatures.csv", sep=';', index_col='date', parse_dates=True)

            
###############################################################################
## Main part of the code
###############################################################################
def Explicit_indexes():
    print("****************************************************")
    topic = "1. Explicit indexes"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    print(f'Dogs dataset (head):\n{animal_shelter.head(2)}\n')
    
    print('-----------------------------------DROPPING AN INDEX')
    animal_shelter.reset_index(drop=True, inplace=True)
    print(animal_shelter.head(2), '\n')
    
    print('---------------------------SETTING A COLUMN AS INDEX')
    animal_shelter.set_index('Name', inplace=True)
    print(animal_shelter.head(2), '\n')
    
    print('-----------------------INDEX MAKE SUBSETTING SIMPLER')
    print("Using columns to filter:")
    print(animal_shelter[animal_shelter['Breed'].isin(['Labrador Retriever', 'German Shepherd'])].head(2), "\n")
    print("Using index to filter:")
    print(animal_shelter.loc[['Bella', 'Max']].head(2), "\n")
    
    print('------------------------------------MULTILEVEL INDEX')
    animal_shelter.set_index(['Breed', 'Color'], inplace=True)
    print(animal_shelter.head(2), '\n')
    
    print('---------------MULTILEVEL INDEX - SUBSET OUTER LEVEL')
    print(animal_shelter.loc[["Chihuahua", "Shih Tzu"]].head(2), '\n')
    
    print('---------------MULTILEVEL INDEX - SUBSET INNER LEVEL')
    print(animal_shelter.loc[[('Pit Bull','Black'), ('Shih Tzu','Sable')], 'Animal ID'].head(2), '\n')
    
    print('---------------------------------------SORTING INDEX')
    print(animal_shelter.sort_index().head(2), '\n')
    
    print('-----------------------CUSTOMIZING THE SORTING INDEX')
    print(animal_shelter.sort_index(level=["Color", "Breed"], ascending=[True, False]).head(2), '\n')

    """
    print('---------------------------------MORE INDEX EXAMPLES')
    print("All dataset\n>> animal_shelter.loc[:, :].head()\n")
    print(animal_shelter.loc[:, :].head())
    print("\n\nSelecting one column\n>> animal_shelter.loc[:, 'Animal ID'].head()\n")
    print(animal_shelter.loc[:, 'Animal ID'].head())
    print("\n\nSelecting two columns\n>> animal_shelter.loc[:, ['Animal ID', 'sex']].head()\n")
    print(animal_shelter.loc[:, ['Animal ID', 'sex']].head())
    print("\n\nSelecting two pairs from index\n>> animal_shelter.loc[[('Pit Bull Mix','Black'), ('Shih Tzu','Sable')], 'Animal ID'].head()\n")
    print(animal_shelter.loc[[('Pit Bull','Black'), ('Shih Tzu','Sable')], 'Animal ID'].head())
    
    print('\n\n\n---MULTILEVEL INDEX - SUBSET INNER LEVEL - USING IDX')
    idx = pd.IndexSlice
    print("\n\nSelecting two items inside inner level\n>> animal_shelter.loc[idx[:,('Black','Tan')], 'Animal ID'].head()\n")
    print(animal_shelter.loc[idx[:,('Black','Tan')], 'Animal ID'].head())
    print("\n\nSelecting one items inside outer level\n>> animal_shelter.loc[idx['Pit Bull Mix',:], 'Animal ID'].head()\n")
    print(animal_shelter.loc[idx['Pit Bull',:], 'Animal ID'].head())
    print("\n\nSelecting two items inside outer level\n>> animal_shelter.loc[idx[('Pit Bull Mix', 'Shih Tzu'),:], 'Animal ID'].head()\n")
    print(animal_shelter.loc[idx[('Pit Bull', 'Shih Tzu'),:], 'Animal ID'].head())
    print("\n\nSelecting two items inside outer and inner levels, making a lot of combination\n>> animal_shelter.loc[idx[('Basset Hound', 'Basset Hound Mix'),('White/Brown','White/Gold')], :].head()\n")
    print(animal_shelter.loc[idx[('Basset Hound', 'Basset Hound'),('White','White')], :].head())
    print("\n\nSelecting two pairs from index\n>> animal_shelter.loc[(idx['Pit Bull Mix','Black'], idx['Shih Tzu','Sable']), 'Animal ID'].head()\n")
    print(animal_shelter.loc[(idx['Pit Bull','Black'], idx['Shih Tzu','Sable']), 'Animal ID'].head())
    """
    
    
    
def Setting_removing_indexes():
    print("\n\n****************************************************")
    topic = "2. Setting & removing indexes"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    # Look at temperatures
    print("Shape: ",temperatures.shape)
    print(temperatures.head(), '\n')
    
    print('-------------------------------SETTING CITY AS INDEX')
    # Index temperatures by city
    temperatures_ind = temperatures.reset_index().set_index('city')
    # Look at temperatures_ind
    print(temperatures_ind.head(), '\n')
    
    print('-----------------------------JUST RESETING THE INDEX')
    # Reset the index, keeping its contents
    print(temperatures_ind.reset_index().head(), '\n')
    
    print('----------------------RESETING AND DROPING THE INDEX')
    # Reset the index, dropping its contents
    print(temperatures_ind.reset_index(drop=True).head())
    return temperatures_ind    
    
    
def Subsetting_with_loc(temperatures_ind):
    print("\n\n****************************************************")
    topic = "3. Subsetting with .loc[]"; print("** %s" % topic)
    print("****************************************************")
    
    # Make a list of cities to subset on
    cities = ['Moscow', 'Saint Petersburg']
    
    print('---------------------USING SQUARE BKACLETS TO SUBSET')
    # Subset temperatures using square brackets
    print("temperatures[temperatures.city.isin(cities)]")
    print(temperatures[temperatures.city.isin(cities)].head(), '\n')
    
    print('--------------------------------USING .LOC TO SUBSET')
    # Subset temperatures_ind using .loc[]
    print("temperatures_ind.loc[cities]")
    print(temperatures_ind.loc[cities].head())
        
    
    
def Setting_multi_level_indexes():
    print("\n\n****************************************************")
    topic = "4. Setting multi-level indexes"; print("** %s" % topic)
    print("****************************************************")
    
    # Index temperatures by country & city
    temperatures_ind = temperatures.reset_index().set_index(['country', 'city'])
    
    # List of tuples: Brazil, Rio De Janeiro & Pakistan, Lahore
    rows_to_keep = [('Brazil', 'Rio De Janeiro'), ('Pakistan', 'Lahore')]
    
    # Subset for rows to keep
    print(temperatures_ind.loc[rows_to_keep])
    return temperatures_ind
    
    
    
def Sorting_by_index_values(temperatures_ind):
    print("\n\n****************************************************")
    topic = "5. Sorting by index values"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    # Look at temperatures
    print(temperatures_ind, '\n')
    
    print('-------------------------------SORT INDEX BY DEFAULT')
    # Sort temperatures_ind by index values
    print(temperatures_ind.sort_index(), '\n')
    
    print('----------------------------------SORT INDEX BY CITY')
    # Sort temperatures_ind by index values at the city level
    print(temperatures_ind.sort_index(level=['city']), '\n')
    
    print('-------------SORT INDEX BY COUNTRY ASC AND CITY DESC')
    # Sort temperatures_ind by country then descending city
    print(temperatures_ind.sort_index(level=['country', 'city'], ascending=[True, False]))
        
    
    
def Slicing_and_subsetting_with_loc_and_iloc():
    print("\n\n****************************************************")
    topic = "6. Slicing and subsetting with .loc and .iloc"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    dogs_srt = animal_shelter[animal_shelter['Animal Type'] == 'Dog'].sort_index()
    print(dogs_srt, '\n')

    print('-----------------------SLICING THE OUTER INDEX LEVEL')
    print(dogs_srt.loc["Akita":"Alaskan Husky"], '\n')
    #idx = pd.IndexSlice
    #print(dogs_srt.loc[idx["Cairn Terrier/Chihuahua Shorthair":"Cairn Terrier/Dachshund Wirehair"]], '\n')
    
    print('-----------------------SLICING THE INNER INDEX LEVEL')
    print(dogs_srt.loc[("Akita", "White"):("Alaskan Husky", "Cream")], '\n')
    #idx = pd.IndexSlice
    #print(dogs_srt.loc[idx[("Cairn Terrier/Chihuahua Shorthair", "Tan"):("Cairn Terrier/Dachshund", "Tan/White")]], '\n')

    print('-------------------------------------SLICING COLUMNS')
    print(dogs_srt.loc[:, 'Animal Type':'sex'], '\n')
    
    print('-----------------------------------------SLICE TWICE')
    print(dogs_srt.loc[("Akita", "White"):("Alaskan Husky", "Cream"), 'Animal Type':'sex'], '\n')
    
    print('-----------------------------------------DATES INDEX')
    dogs = animal_shelter[animal_shelter['Animal Type'] == 'Dog'].set_index("DateTime").sort_index()
    print(dogs, '\n')
    
    print('------------------------------------SLICING BY DATES')
    print(dogs.loc['2014-08-25':'2015-09-16'], '\n')
    
    print('----------------------------SLICING BY PARTIAL DATES')
    print(dogs.loc['2014':'2015'], '\n')
    
    print('---------------------SUBSETTING BY ROW/COLUMN NUMBER')
    print(dogs.iloc[2:5, 1:4])
    
    
    
def Slicing_index_values(temperatures_ind):
    print("\n\n****************************************************")
    topic = "7. Slicing index values"; print("** %s" % topic)
    print("****************************************************")
    
    # Sort the index of temperatures_ind
    temperatures_srt = temperatures_ind.sort_index()
    
    print('-----------------------------FROM PAKISTAN TO RUSSIA')
    # Subset rows from Pakistan to Russia
    print(temperatures_srt.loc['Pakistan':'Russia'], '\n')
    
    print('--------------------FROM LAHORE TO MOSCOW (NONSENSE)')
    # Try to subset rows from Lahore to Moscow
    print(temperatures_srt.loc['Lahore':'Moscow'], '\n')
    
    print('-------------FROM PAKISTAN, LAHORE TO RUSSIA, MOSCOW')
    # Subset rows from Pakistan, Lahore to Russia, Moscow
    print(temperatures_srt.loc[('Pakistan','Lahore'):('Russia','Moscow')])
    return temperatures_srt
    
    
    
def Slicing_in_both_directions(temperatures_srt):
    print("\n\n****************************************************")
    topic = "8. Slicing in both directions"; print("** %s" % topic)
    print("****************************************************")
    
    print('-----------------------------------------SLICE INDEX')
    # Subset rows from India, Hyderabad to Iraq, Baghdad
    print(temperatures_srt.loc[('India', 'Hyderabad'):('Iraq', 'Baghdad')])
    
    print('---------------------------------------SLICE COLUMNS')
    # Subset columns from date to avg_temp_c
    print(temperatures_srt.loc[:, 'date':'avg_temp_c'])
    
    print('-----------------------------SLICE COLUMNS AND INDEX')
    # Subset in both directions at once
    print(temperatures_srt.loc[('India', 'Hyderabad'):('Iraq', 'Baghdad'), 'date':'avg_temp_c'])
        
    
    
def Slicing_time_series():
    print("\n\n****************************************************")
    topic = "9. Slicing time series"; print("** %s" % topic)
    print("****************************************************")
    
    
    print('-------------------------------ROWS IN 2010 AND 2011')
    temperatures.reset_index(inplace=True)
    # Use Boolean conditions to subset temperatures for rows in 2010 and 2011
    print(temperatures[(temperatures.date.dt.year == 2010) | (temperatures.date.dt.year == 2011)], '\n')
    
    print('-----------------------INDEX SLICE FROM 2010 TO 2011')
    # Set date as an index
    temperatures_ind = temperatures.set_index('date')
    # Use .loc[] to subset temperatures_ind for rows in 2010 and 2011
    print(temperatures_ind.loc['2010':'2011'], '\n')
    
    print('---------------INDEX SLICE FROM 2010 AGO TO 2011 FEB')
    # Use .loc[] to subset temperatures_ind for rows from Aug 2010 to Feb 2011
    print(temperatures_ind.loc['2010-08':'2011-02'])
        
    
    
def Subsetting_by_row_column_number():
    print("\n\n****************************************************")
    topic = "10. Subsetting by row/column number"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------------------23RD ROW, 2ND COLUMN')
    # Get 23rd row, 2nd column (index 22, 1)
    print(temperatures.iloc[22, 1])
    
    print('----------------------------------------FIRST 5 ROWS')
    # Use slicing to get the first 5 rows
    print(temperatures.iloc[0:5])
    
    print('-------------------------------------COLUMNS 3 AND 4')
    # Use slicing to get columns 3 to 4
    print(temperatures.iloc[:, 2:4])
    
    print('-------------------------SLICE BOTH ROWS AND COLUMNS')
    # Use slicing in both directions at once
    print(temperatures.iloc[:5, 2:4])
        
    
    
def Working_with_pivot_tables():
    print("\n\n****************************************************")
    topic = "11. Working with pivot tables"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------------------PIVOTING THE DOG PACK')
    dog_pack = animal_shelter[animal_shelter['Animal Type'] == 'Dog'].reset_index().sort_values(by='Breed')
    dogs_age = dog_pack.pivot_table(values="Age upon Intake", index="Breed", columns="Color", fill_value=0)
    print(dogs_age.head(), '\n')
    
    print('-------------------.LOC[] + SLICING IS A POWER COMBO')
    print(dogs_age.loc["Chow Chow":"Dalmatian"], '\n')
    
    print('-----------------------------------THE AXIS ARGUMENT')
    print(dogs_age.mean(axis="index"), '\n')
    
    print('---------------------------------AXIS ACROSS COLUMNS')
    print(dogs_age.mean(axis="columns"))
    
    
    
def Pivot_temperature_by_city_and_year():
    print("\n\n****************************************************")
    topic = "12. Pivot temperature by city and year"; print("** %s" % topic)
    print("****************************************************")
    
    # Add a year column to temperatures
    temperatures['year'] = temperatures.date.dt.year
    # Pivot avg_temp_c by country and city vs year
    temp_by_country_city_vs_year = temperatures.pivot_table(values='avg_temp_c', index=['country','city'], columns='year')
    # See the result
    print(temp_by_country_city_vs_year.head())
    return temp_by_country_city_vs_year
    
    

def Subsetting_pivot_tables(temp_by_country_city_vs_year):
    print("\n\n****************************************************")
    topic = "13. Subsetting pivot tables"; print("** %s" % topic)
    print("****************************************************")
    
    # Subset for Egypt to India
    print(temp_by_country_city_vs_year.loc['Egypt':'India'])
    # Subset for Egypt, Cairo to India, Delhi
    print(temp_by_country_city_vs_year.loc[('Egypt','Cairo'):('India','Delhi')])
    # Subset in both directions at once
    print(temp_by_country_city_vs_year.loc[('Egypt','Cairo'):('India','Delhi'), 2005:2010])
    
    
    
def Calculating_on_a_pivot_table(temp_by_country_city_vs_year):
    print("\n\n****************************************************")
    topic = "14. Calculating on a pivot table"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------TEMPERATURE BY YEAR')
    # Get the worldwide mean temp by year
    mean_temp_by_year = temp_by_country_city_vs_year.mean()
    # Filter for the year that had the highest mean temp
    print(mean_temp_by_year[mean_temp_by_year == mean_temp_by_year.max()])
    
    print('---------------------------------TEMPERATURE BY CITY')
    # Get the mean temp by city
    mean_temp_by_city = temp_by_country_city_vs_year.mean(axis=1)
    # Filter for the city that had the lowest mean temp
    print(mean_temp_by_city[mean_temp_by_city == mean_temp_by_city.min()])
    
    
    
        
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Explicit_indexes()
    temperatures_city = Setting_removing_indexes()
    Subsetting_with_loc(temperatures_city)
    temperatures_country_city = Setting_multi_level_indexes()
    Sorting_by_index_values(temperatures_country_city)
    
    Slicing_and_subsetting_with_loc_and_iloc()
    temperatures_country_city = Slicing_index_values(temperatures_country_city)
    Slicing_in_both_directions(temperatures_country_city)
    Slicing_time_series()
    Subsetting_by_row_column_number()
    
    Working_with_pivot_tables()
    temp_by_country_city_vs_year = Pivot_temperature_by_city_and_year()
    Subsetting_pivot_tables(temp_by_country_city_vs_year)
    Calculating_on_a_pivot_table(temp_by_country_city_vs_year)
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()
    #np.set_printoptions(formatter = {'float': None}) #Return to default
    #pd.options.display.float_format = None
