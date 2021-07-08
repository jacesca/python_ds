# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:29:35 2020

@author: jacesca@gmail.com
Chapter 1: The Problem With Missing Data
    Get familiar with missing data and how it impacts your analysis! Learn about 
    different null value operations in your dataset, how to find missing data and 
    summarizing missingness in your data.
Source: https://learn.datacamp.com/courses/data-manipulation-with-pandas
Help:
    https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py
"""
###############################################################################
## Importing libraries
###############################################################################
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd


###############################################################################
## Preparing the environment
###############################################################################
# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 8, 'font.size': 8})
suptitle_param   = dict(color='darkblue', fontsize=10)
title_param      = {'color': 'darkred', 'fontsize': 12}

# Reading the data
college = pd.read_csv('college.csv', sep=';')
pima_diabetes = pd.read_csv('pima-indians-diabetes data.csv')
pima_diabetes.loc[pima_diabetes.BMI.isnull(), 'BMI'] = 0
airquality = pd.read_csv('air-quality.csv', parse_dates=['Date'], index_col='Date')
      
            
###############################################################################
## Main part of the code
###############################################################################
def Null_value_operations():
    print("****************************************************")
    topic = "3. Null value operations"; print("** %s" % topic)
    print("****************************************************")
    
    # Print the sum of two None's
    try             : print("Add operation output of 'None': ", None + None)
    except TypeError: print("'None' does not support Arithmetic Operations!")

    # Print the sum of two np.nan's
    try             : print("Add operation output of 'np.nan': ", np.nan + np.nan) #output: nan
    except TypeError: print("'np.nan' does not support Arithmetic Operations!!")
    
    # Print the output of logical OR of two None's
    try             : print("OR operation output of 'None': ", None or None) #output: None
    except TypeError: print("'None' does not support Logical Operations!!")
    
    # Print the output of logical OR of two np.nan's
    try             : print("OR operation output of 'np.nan': ", np.nan or np.nan) #output: nan
    except TypeError: print("'np.nan' does not support Logical Operations!!")
    
    
    
def Finding_Null_values():
    print("****************************************************")
    topic = "4. Finding Null values"; print("** %s" % topic)
    print("****************************************************")
    
    # Print the comparison of two 'None's
    try             : print("'None' comparison output: ", None == None) #Output: True
    except TypeError: print("'None' does not support this operation!!")
    
    # Print the comparison of two 'np.nan's
    try             : print("'np.nan' comparison output: ", np.nan == np.nan) #Output: False
    except TypeError: print("'np.nan' does not support this operation!!")
    
    # Check if 'None' is 'NaN'
    try             : print("Is 'None' same as nan? ", np.isnan(None))
    except TypeError: print("Function 'np.isnan()' does not support this Type!")
    
    # Check if 'np.nan' is 'NaN'
    try             : print("Is 'np.nan' same as nan? ", np.isnan(np.nan)) #Output: True
    except TypeError: print("Function 'np.isnan()' does not support this Type!!")
    
    
    
def Handling_missing_values():
    print("****************************************************")
    topic = "5. Handling missing values"; print("** %s" % topic)
    print("****************************************************")
    
    diabetes = pima_diabetes.copy()
    
    print('---------------------------------------------EXPLORE')
    print(f"General info of the dataset: \n{diabetes.info()}\n")
    print(f"Describe: \n{diabetes.describe()}\n")
    print(f"Rows with BMI==0: \n{diabetes[diabetes.BMI == 0]}\n")
    
    print('---------------------REPLACE MISSING VALUES WITH NAN')
    diabetes.loc[diabetes.BMI == 0, 'BMI'] = np.nan
    print(diabetes.BMI[np.isnan(diabetes.BMI)], '\n')
    
    
        
def Detecting_missing_values():
    print("****************************************************")
    topic = "6. Detecting missing values"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    print(college.head(), '\n')
    # Print the info of college
    print(college.info(), '\n')
    
    print('-------------------------------UNIQUE VALUES IN CSAT')
    # Store unique values of 'csat' column to 'csat_unique'
    csat_unique = college.csat.unique()
    # Print the sorted values of csat_unique
    print(np.sort(csat_unique), '\n')
    
    
    
def Replacing_missing_values():
    print("****************************************************")
    topic = "7. Replacing missing values"; print("** %s" % topic)
    print("****************************************************")
    
    global college
    
    print('---------------------------------------------EXPLORE')
    # Read the dataset 'college.csv' with na_values set to '.'
    college = pd.read_csv('college.csv', sep=';', na_values='.')
    print(college.head(), '\n')
    
    print('---------------------------INFORMATION ABOUT COLUMNS')
    # Print the info of college
    print(college.info(), '\n')
    
    
    
def Replacing_hidden_missing_values():
    print("****************************************************")
    topic = "8. Replacing hidden missing values"; print("** %s" % topic)
    print("****************************************************")
    
    diabetes = pima_diabetes.copy()
    
    print('----------------------------------------DESCRIOTIONS')
    # Print the description of the data
    print(diabetes.describe())
    
    print('----------------------------------ROWS WITH BMI == 0')
    # Store all rows of column 'BMI' which are equal to 0 
    zero_bmi = diabetes.BMI[diabetes.BMI == 0]
    print(zero_bmi)
    
    print('-------------------------TRANSFORMING TO NULL VALUES')
    # Set the 0 values of column 'BMI' to np.nan
    diabetes.loc[diabetes.BMI == 0, 'BMI'] = np.nan
    # Print the 'NaN' values in the column BMI
    print(diabetes.BMI[np.isnan(diabetes.BMI)])
    
    
    
def Analyze_the_amount_of_missingness():
    print("****************************************************")
    topic = "9. Analyze the amount of missingness"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    print(airquality.head(), '\n')
    
    print('-------------------FINDING MISSING VALUES PER COLUMN')
    print(airquality.isna().sum(), '\n')
    
    print('---------------------------PERCENTAGE OF MISSINGNESS')
    print(airquality.isna().mean() * 100)
    
    print('-----------------------------------------NULLITY BAR')
    fig, ax = plt.subplots()
    msno.bar(airquality, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Missing values', fontsize=8)
    ax.set_title('Airquality Dataset - Nullity Bar', **title_param)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('--------------------------------------NULLITY MATRIX')
    #fig, ax = plt.subplots()
    ax = msno.matrix(airquality, sparkline=True, fontsize=8, figsize=(6.4, 4.8)) #ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Airquality Dataset - Nullity Matrix', **title_param)
    plt.subplots_adjust(left=None, bottom=.1, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    print('-----------------NULLITY MATRIX FOR TIME SERIES DATA')
    #fig, ax = plt.subplots()
    ax = msno.matrix(airquality, freq='M', sparkline=True, fontsize=8, figsize=(6.4, 4.8)) #ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Airquality Dataset - Nullity Matrix For Time Series Data', **title_param)
    plt.subplots_adjust(left=.2, bottom=.1, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('-------------------------------NULLITY MATRIX TUNING')
    #fig, ax = plt.subplots()
    ax = msno.matrix(airquality.loc['May-1976': 'Jul-1976'], freq='M', sparkline=True, fontsize=8, figsize=(6.4, 4.8)) #ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Airquality Dataset - From May-1976 to Jul-1976', **title_param)
    plt.subplots_adjust(left=.2, bottom=.1, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Analyzing_missingness_percentage():
    print("****************************************************")
    topic = "10. Analyzing missingness percentage"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------NULL VALUES IN AIRQUALITY')
    # Create a nullity DataFrame airquality_nullity
    airquality_nullity = airquality.isnull()
    print(airquality_nullity.head())
    
    print('-----------------------------TOTAL OF MISSING VALUES')
    # Calculate total of missing values
    missing_values_sum = airquality_nullity.sum()
    print('Total Missing Values:\n', missing_values_sum)
    
    print('-----------------------PERCENTAGES OF MISSING VALUES')
    # Calculate percentage of missing values
    missing_values_percent = airquality_nullity.mean() * 100
    print('Percentage of Missing Values:\n', missing_values_percent)
    
    
    
def Visualize_missingness():
    print("****************************************************")
    topic = "11. Visualize missingness"; print("** %s" % topic)
    print("****************************************************")
    
    fig, axis = plt.subplots(2,2, figsize=(12.1, 5.5))
    print('-----------------------------------------NULLITY BAR')
    ax = axis[0, 0]
    msno.bar(airquality, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Missing values', fontsize=8)
    ax.set_title('Airquality Dataset - Nullity Bar', **title_param)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    print('--------------------------------------NULLITY MATRIX')
    ax = axis[0, 1]
    ax = msno.matrix(airquality, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Airquality Dataset - Nullity Matrix', **title_param)
    
    print('-----------------NULLITY MATRIX FOR TIME SERIES DATA')
    ax = axis[1, 0]
    ax = msno.matrix(airquality, freq='M', sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Airquality Dataset - Nullity Matrix For Time Series Data', **title_param)
    
    print('-------------------------------NULLITY MATRIX TUNING')
    ax = axis[1, 1]
    ax = msno.matrix(airquality.loc['May-1976':'Jul-1976'], freq='M', sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Airquality Dataset - From May-1976 to Jul-1976', **title_param)
    
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.8, wspace=.4, hspace=1.2); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
        
        
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Null_value_operations()
    Finding_Null_values()
    Handling_missing_values()
    Detecting_missing_values()
    Replacing_missing_values()
    Replacing_hidden_missing_values()
    Analyze_the_amount_of_missingness()
    Analyzing_missingness_percentage()
    Visualize_missingness()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    np.set_printoptions(formatter = {'float': None}) #Return to default
    plt.style.use('default')