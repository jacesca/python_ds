# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:37:38 2020

@author: jacesca@gmail.com
Chapter 2: Aggregating Data
    In this chapter, you’ll calculate summary statistics on DataFrame columns, 
    and master grouped summary statistics and pivot tables.
Source: https://learn.datacamp.com/courses/data-manipulation-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import re


###############################################################################
## Preparing the environment
###############################################################################
    
# Global variables
#SEED = 42
#np.random.seed(SEED) 

#Global configuration
np.set_printoptions(formatter={'float': '{:,.3f}'.format})
pd.options.display.float_format = '{:,.3f}'.format 

# Read the data
sales = pd.read_pickle("walmart_sales.pkl.bz2", compression='bz2')

animal_shelter = pd.read_excel('austin animal intakes oct 2013 nov 2016.xlsx', 
                                sheet_name  =  'Austin_Animal_Center_Intakes',
                                parse_dates = ['DateTime', ],
                                na_values   = ['Unknown', ''],# ['Unknown', '', 'NULL'], --> NULL is not necesary, because is already a default identifier.
                                converters  = {'MonthYear'       : lambda x: pd.to_datetime(x).strftime('%Y-%m'),
                                               'Age upon Intake' : lambda x: pd.to_numeric(re.sub(r' year\w*', '', x), errors='coerce')},
                                dtype       = {#'Breed'           : 'category',
                                               'Color'           : 'category',
                                               'Animal Type'     : 'category',
                                               'Intake Condition': 'category',
                                               'Intake Type'     : 'category',
                                               'Sex upon Intake' : 'category'})
animal_shelter.dropna(inplace=True)
animal_shelter['Breed'] = animal_shelter['Breed'].str.replace('Tan ','')
animal_shelter['Breed'] = animal_shelter['Breed'].apply(lambda x: re.sub(r'Black\s*\/*','',x))
animal_shelter['Breed'] = animal_shelter['Breed'].apply(lambda x: re.sub(r'\s*Mix\s*|Tan\s|Blue\s','',x))
animal_shelter['dummy'] = animal_shelter['Breed'].str.split('\/')
animal_shelter['Breed'] = animal_shelter["dummy"].str.get(0)
animal_shelter['dummy'] = animal_shelter['Color'].str.split('\/')
animal_shelter['Color'] = animal_shelter["dummy"].str.get(0)
animal_shelter['dummy'] = animal_shelter['Color'].str.split(' ')
animal_shelter['Color'] = animal_shelter["dummy"].str.get(0)
animal_shelter['Name'] = animal_shelter['Name'].str.replace('*','')
animal_shelter['sex'] = ['Female' if Female else 'Male' for Female in animal_shelter['Sex upon Intake'].str.contains('Female')]
animal_shelter['sex'] = ['Female' if Female else 'Male' for Female in animal_shelter['Sex upon Intake'].str.contains('Female')]
animal_shelter.drop(columns='dummy', inplace=True)

dog_pack = animal_shelter[animal_shelter['Animal Type'] == 'Dog'].copy()
dog_pack['Breed'] = dog_pack['Breed'].astype('category')   
dog_pack['Color'] = dog_pack['Color'].astype('category')   
dog_pack = dog_pack.to_pickle("dogs_shelter.pkl.bz2", compression='bz2')

animal_shelter['Breed'] = animal_shelter['Breed'].astype('category')   
animal_shelter['Color'] = animal_shelter['Color'].astype('category')   
animal_shelter.to_pickle('animal_shelter.pkl.bz2', compression='bz2')

###############################################################################
## Main part of the code
###############################################################################
def Mean_and_median():
    print("****************************************************")
    topic = "2. Mean and median"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------------------HEAD')
    # Print the head of the sales DataFrame
    print(sales.head(), '\n')
    
    print('----------------------------------------GENERAL INFO')
    # Print the info about the sales DataFrame
    print(sales.info(), '\n')
    
    print('---------------------------------MEAN OF WEEKLY SALES')
    # Print the mean of weekly_sales
    print(f"{sales.weekly_sales.mean():,.2f}\n")
    
    print('-------------------------------MEDIAN OF WEEKLY SALES')
    # Print the median of weekly_sales
    print(f"{sales.weekly_sales.median():,.2f}\n")
    
    
    
def Summarizing_dates():
    print("\n\n****************************************************")
    topic = "3. Summarizing dates"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------------------------------MAX DATE')
    # Print the maximum of the date column
    print(f"{sales.date.max():%Y-%m-%d}")
    
    print('--------------------------------------------MIN DATE')
    # Print the minimum of the date column
    print(f"{sales.date.min():%Y-%m-%d}")
    
    
    
def Efficient_summaries():
    print("\n\n****************************************************")
    topic = "4. Efficient summaries"; print("** %s" % topic)
    print("****************************************************")
    
    # A custom IQR function
    def iqr(column):
        return column.quantile(0.75) - column.quantile(0.25)

    print('---------------------------AGG 1 FUNCTION - 1 COLUMN')
    # Print IQR of the temperature_c column
    print(sales.temperature_c.agg(iqr), '\n')

    print('-----------------------AGG 1 FUNCTION - MANY COLUMNS')
    # Update to print IQR of temperature_c, fuel_price_usd_per_l, & unemployment
    print(sales[["temperature_c", 'fuel_price_usd_per_l', 'unemployment']].agg(iqr), '\n')
    
    print('--------------------AGG MANY FUNCTION - MANY COLUMNS')
    # Update to print IQR and median of temperature_c, fuel_price_usd_per_l, & unemployment
    print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg([iqr, np.median]))



def Cumulative_statistics():
    print("\n\n****************************************************")
    topic = "5. Cumulative statistics"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------DATA FILTERED BY DEPARTM 1 AND STORE 1')
    sales_1_1 = sales[(sales.department == 1) & (sales.store == 1)]
    print(sales_1_1.head(), '\n')
    
    print('----------------------------------SORT TABLE BY DATE')
    # Sort sales_1_1 by date
    sales_1_1 = sales_1_1.sort_values(by='date')
    
    print('-----------------------------ADDING CUM_WEEKLY_SALES')
    # Get the cumulative sum of weekly_sales, add as cum_weekly_sales col
    sales_1_1['cum_weekly_sales'] = sales_1_1.weekly_sales.cumsum()
    
    print('--------------------------------ADDING CUM_MAX_SALES')
    # Get the cumulative max of weekly_sales, add as cum_max_sales col
    sales_1_1['cum_max_sales'] = sales_1_1.weekly_sales.cummax()
    
    print('-------------------------------EXPLORING THE RESULTS')
    # See the columns you calculated
    print(sales_1_1[["date", "weekly_sales", "cum_weekly_sales", "cum_max_sales"]])
    
    
    
def Dropping_duplicates():
    print("\n\n****************************************************")
    topic = "7. Dropping duplicates"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    # Print the head of the sales DataFrame
    print(f"{sales.shape[0]} rows in sales.\n")
    
    print('-----USING DROP_DUPLICATES TO FIND UNIQUE STORE-TYPE')
    # Drop duplicate store/type combinations
    store_types = sales.drop_duplicates(subset=['store', 'type'])
    print(f'{store_types.shape[0]} rows found. \n{store_types.head()}\n')
    
    print('-----USING DROP_DUPLICATES TO FIND UNIQUE STORE-DEPT')
    # Drop duplicate store/department combinations
    store_depts = sales.drop_duplicates(subset=['store', 'department'])
    print(f'{store_depts.shape[0]} rows found. \n{store_depts.head()}\n')
    
    print('--------USING DOP DUPLICATES TO FIND UNIQUE HOLIDAYS')
    # Subset the rows that are holiday weeks and drop duplicate dates
    holiday_dates = sales[sales.is_holiday==True].drop_duplicates(subset='date')
    print(f'{holiday_dates.shape[0]} rows found. \n{holiday_dates.date}\n')
    return store_types, store_depts
    
    
    
def Counting_categorical_variables(store_types, store_depts):
    print("\n\n****************************************************")
    topic = "8. Counting categorical variables"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------COUNTING NUMBER OF STORES IN EACH TYPE')
    # Count the number of stores of each type
    store_counts = store_types['type'].value_counts()
    print(store_counts)
    
    print('-------------------------PROPORTION NUMBER OF STORES')
    # Get the proportion of stores of each type
    store_props = store_types['type'].value_counts(normalize=True)
    print(store_props)
    
    print('---------------------------COUNTING NUMBER OF DEPTOS')
    # Count the number of each department number and sort
    dept_counts_sorted = store_depts.department.value_counts(sort=True)
    print(dept_counts_sorted)
    
    print('--------------------------------PROPORTION OF DEPTOS')
    # Get the proportion of departments of each number and sort
    dept_props_sorted = store_depts.department.value_counts(sort=True, normalize=True)
    print(dept_props_sorted)
    
    
    
def What_percent_of_sales_occurred_at_each_store_type():
    print("\n\n****************************************************")
    topic = "10. What percent of sales occurred at each store type?"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------------TOTAL WEEKLY SALES')
    # Calc total weekly sales
    sales_all = sales["weekly_sales"].sum()
    print(f'{sales_all:,.2f}')
    
    print('---------------------------------------------SALES A')
    # Subset for type A stores, calc total weekly sales
    sales_A = sales[sales["type"] == "A"]["weekly_sales"].sum()
    print(f'{sales_A:,.2f}')
    
    print('---------------------------------------------SALES B')
    # Subset for type B stores, calc total weekly sales
    sales_B = sales[sales['type'] == 'B']['weekly_sales'].sum()
    print(f'{sales_B:,.2f}')
    
    print('---------------------------------------------SALES C')
    # Subset for type C stores, calc total weekly sales
    sales_C = sales[sales['type'] == 'C']['weekly_sales'].sum()
    print(f'{sales_C:,.2f}')
    
    print('------------------------------------------PROPORTION')
    # Get proportion for each type
    sales_propn_by_type = [sales_A, sales_B, sales_C] / sales_all
    print(sales_propn_by_type)

    
    
def Calculations_with_groupby():
    print("\n\n****************************************************")
    topic = "11. Calculations with .groupby()"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------GROUP BY TYPE AND GET PROPORTION')
    # Group by type; calc total weekly sales
    sales_by_type = sales.groupby("type")["weekly_sales"].sum()
    
    # Get proportion for each type
    sales_propn_by_type = sales_by_type / sales_by_type.sum()
    print(sales_propn_by_type)

    print('---------------------------GROUP BY TYPE AND HOLIDAY')
    # Group by type and is_holiday; calc total weekly sales
    sales_by_type_is_holiday = sales.groupby(['type', 'is_holiday'])['weekly_sales'].sum()
    print(sales_by_type_is_holiday)
    
    
def Multiple_grouped_summaries():
    print("\n\n****************************************************")
    topic = "12. Multiple grouped summaries"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------------WEEKLY SALES GROUP BY TYPE')
    # For each store type, aggregate weekly_sales: get min, max, mean, and median
    sales_stats = sales.groupby('type')['weekly_sales'].agg(['min', 'max', np.mean, np.median])
    # Print sales_stats
    print(sales_stats, '\n')
    
    print('---------------------------------------GROUP BY TYPE')
    # For each store type, aggregate unemployment and fuel_price_usd_per_l: get min, max, mean, and median
    unemp_fuel_stats = sales.groupby(['type'])[['unemployment', 'fuel_price_usd_per_l']].agg([np.min, np.max, np.mean, np.median])
    # Print unemp_fuel_stats
    print(unemp_fuel_stats)


    
    
def Pivot_tables():
    print("****************************************************")
    topic = "13. Pivot tables"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------------------------------GROUP BY')
    print(animal_shelter.groupby('Color')['Age upon Intake'].mean())
    
    print('\n-----------------------------PIVOT (MEAN BY DEFAULT)')
    print(animal_shelter.pivot_table(values='Age upon Intake', index='Color'))
    
    print('\n--------------------------------PIVOT (MAX FUNCTION)')
    print(animal_shelter.pivot_table(values='Age upon Intake', index='Color', aggfunc=np.max))
    
    print('\n---------------PIVOT (MEAN, MAX AND MEDIAN FUNCTION)')
    print(animal_shelter.pivot_table(values='Age upon Intake', index='Color', aggfunc=['mean', 'max', np.median]))
    
    print('\n------------------------------PIVOT ON TWO VARIABLES')
    print(animal_shelter.pivot_table(values='Age upon Intake', index='Color', columns='Breed'))
    
    print('\n------------------------------FILLING MISSING VALUES')
    print(animal_shelter.pivot_table(values='Age upon Intake', index='Color', columns='Breed', fill_value=0))
    
    print('\n---------------------------------------------SUMMING')
    print(animal_shelter.pivot_table(values='Age upon Intake', index='Color', columns='Breed', fill_value=0, margins=True))
    
    
    
def Pivoting_on_one_variable():
    print("\n\n****************************************************")
    topic = "14. Pivoting on one variable"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------------WEEKLY SALES - SIMPLE PIVOT')
    # Pivot for mean weekly_sales for each store type
    mean_sales_by_type = sales.pivot_table(values='weekly_sales', index='type')
    # Print mean_sales_by_type
    print(mean_sales_by_type)
        
    print('-----------WEEKLY SALES - MORE FUNCTIONS IN THE PIVOT')
    # Pivot for mean and median weekly_sales for each store type
    mean_med_sales_by_type = sales.pivot_table(values='weekly_sales', index='type', aggfunc=[np.mean, np.median])
    # Print mean_med_sales_by_type
    print(mean_med_sales_by_type)
    
    print('---WEEKLY SALES - ADDING ANOTHER VARIABLE AS COLUMNS')
    # Pivot for mean weekly_sales by store type and holiday 
    mean_sales_by_type_holiday = sales.pivot_table(values='weekly_sales', index='type', columns='is_holiday')
    # Print mean_sales_by_type_holiday
    print(mean_sales_by_type_holiday)
    
    print('---------WEEKLY SALES - ADDING MORE COLUMNS AS INDEX')
    # Pivot for mean weekly_sales by store type and holiday 
    mean_sales_by_type_holiday = sales.pivot_table(values='weekly_sales', index=['store', 'type'], columns='is_holiday')
    # Print mean_sales_by_type_holiday
    print(mean_sales_by_type_holiday)
    
    
    
def Fill_in_missing_values_and_sum_values_with_pivot_tables():
    print("\n\n****************************************************")
    topic = "15. Fill in missing values and sum values with pivot tables"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------USING FILL_VALUE')
    # Print mean weekly_sales by department and type; fill missing values with 0
    print(sales.pivot_table(values='weekly_sales', index='department', columns='type', fill_value=0))
    
    print('---------------------------------------USING MARGINS')
    # Print the mean weekly_sales by department and type; fill missing values with 0s; sum all rows and cols
    print(sales.pivot_table(values="weekly_sales", index="department", columns="type", fill_value=0, margins=True))

    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Mean_and_median()
    Summarizing_dates()
    Efficient_summaries()
    Cumulative_statistics()
    store_types, store_depts = Dropping_duplicates()
    Counting_categorical_variables(store_types, store_depts)
    What_percent_of_sales_occurred_at_each_store_type()
    Calculations_with_groupby()
    Multiple_grouped_summaries()
    Pivot_tables()
    Pivoting_on_one_variable()
    Fill_in_missing_values_and_sum_values_with_pivot_tables()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()
    np.set_printoptions(formatter = {'float': None}) #Return to default
    pd.options.display.float_format = None
