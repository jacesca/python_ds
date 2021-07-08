# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:11:28 2020

@author: jacesca@gmail.com
Chapter 3: Advanced data problems
    In this chapter, you’ll dive into more advanced data cleaning problems, such 
    as ensuring that weights are all written in kilograms instead of pounds. 
    You’ll also gain invaluable skills that will help you verify that values 
    have been added correctly and that missing values don’t negatively impact 
    your analyses.
Source: https://learn.datacamp.com/courses/data-cleaning-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import datetime as dt
import matplotlib.pyplot as plt
import missingno as msno #Help on missingno library --> https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py
import numpy as np
import pandas as pd


###############################################################################
## Preparing the environment
###############################################################################
    
# Global variables
SEED = 42
np.random.seed(SEED) 

# Global params

plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 8, 'font.size': 8})
suptitle_param   = dict(color='darkblue', fontsize=10)
title_param      = {'color': 'darkred', 'fontsize': 12}


# Read the data
banking          = pd.read_csv('banking_dirty.csv', index_col=0)
banking_currency = pd.read_fwf('banking_currency.data', index_col=0)
banking_missing  = pd.read_fwf('banking_missing.data', index_col=0)
banking_missing2 = pd.read_fwf('banking_missing2.data', index_col=0)

airquality       = pd.read_csv('airquality.csv')
     
###############################################################################
## Main part of the code
###############################################################################
def Uniformity(seed=SEED):
    print("****************************************************")
    topic = "1. Uniformity"; print("** %s" % topic)
    
    # Prepare the data
    temperatures = pd.DataFrame({'Date'       : pd.date_range(start='3/3/2019', end='31/3/2019'), 
                                 'Temperature': np.concatenate((np.random.randint(low=12, high=22, size=4),
                                                                np.random.randint(low=53, high=72, size=1), # F°
                                                                np.random.randint(low=12, high=22, size=3),
                                                                np.random.randint(low=53, high=72, size=1), # F°
                                                                np.random.randint(low=12, high=22, size=12),
                                                                np.random.randint(low=53, high=72, size=1), # F°
                                                                np.random.randint(low=12, high=22, size=7)))})
    
    print('-------------------------------------------EXPLORING')
    print(temperatures.head())
    
    print('------------------------------------FINDING OUTLIERS')
    fig, axis = plt.subplots(1, 2, figsize=(10, 4))
    ax = axis[0]
    temperatures.plot(kind='scatter', x='Date', y='Temperature', rot=45, ax=ax)
    ax.set_xlabel('Dates')
    ax.set_ylabel('Temperature in Celsius')
    ax.set_title('Temperature in Celsius March 2019\nNYCTemperature in Celsius March 2019 - NYC', **title_param)
    ax.set_xlim(temperatures.Date.min(), temperatures.Date.max())
    ax = axis[1]
    temperatures.boxplot(column='Temperature', ax=ax)
    ax.set_title('EDA - Temperatures', **title_param)
    ax.set_ylabel('Values')
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    fig.suptitle("{}\nBEFORE TREATMENT".format(topic), **suptitle_param)
    plt.show()
    
    print('---------------------------TREATING TEMPERATURE DATA')
    temp_fah = temperatures.loc[temperatures['Temperature'] > 40, 'Temperature']
    print("outliers: \n{}\n".format(temp_fah))
    temp_cels = (temp_fah - 32) * (5/9)
    temperatures.loc[temperatures['Temperature'] > 40, 'Temperature'] = temp_cels
    print("data transformed: \n{}".format(temperatures.head()))
    
    # Confirm with EDA
    fig, axis = plt.subplots(1, 2, figsize=(10, 4))
    ax = axis[0]
    temperatures.plot(kind='scatter', x='Date', y='Temperature', rot=45, ax=ax)
    ax.set_xlabel('Dates')
    ax.set_ylabel('Temperature in Celsius')
    ax.set_title('Temperature in Celsius March 2019\nNYCTemperature in Celsius March 2019 - NYC', **title_param)
    ax.set_xlim(temperatures.Date.min(), temperatures.Date.max())
    ax = axis[1]
    temperatures.boxplot(column='Temperature', ax=ax)
    ax.set_title('EDA - Temperatures', **title_param)
    ax.set_ylabel('Values')
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    fig.suptitle("{}\nAFTER TREATMENT".format(topic), **suptitle_param)
    plt.show()
    
    
    print('-------------------MAKING SOME ASSERTION TO VALIDATE')
    # Assert conversion is correct
    try:
        assert temperatures['Temperature'].max() < 40
        print("All temperatures are in C°, ok!...")
    except: print("Error: not uniformed unit in temperature")
    
    
    
def Uniform_currencies(seed=SEED):
    print("\n\n****************************************************")
    topic = "3. Uniform currencies"; print("** %s" % topic)
    
    print('-------------------------------------------EXPLORING')
    print("currencies: {}".format(banking_currency.acct_cur.unique()))
    print(banking_currency.head(), '\n')
    
    print('--------------------------FINDING NOT UNIFORMED DATA')
    # Find values of acct_cur that are equal to 'euro'
    acct_eu = banking_currency.acct_cur == 'euro'
    print(banking_currency[acct_eu].head(2), '\n')

    print('------------------------------MAKING TRANSFORMATIONS')
    # Convert acct_amount where it is in euro to dollars
    banking_currency.loc[acct_eu, 'acct_amount'] = banking_currency.loc[acct_eu, 'acct_amount'] * 1.1

    # Unify acct_cur column by changing 'euro' values to 'dollar'
    banking_currency.loc[acct_eu, 'acct_cur'] = 'dollar'
    print(banking_currency[acct_eu].head(2), '\n')

    print('-------------------MAKING SOME ASSERTION TO VALIDATE')
    # Assert that only dollar currency remains
    try:
        assert banking_currency['acct_cur'].unique() == 'dollar'
        print("only dollar currency remains, ok!...")
    except:
        print("Error: there are not only dollars!...")

        
def Uniform_dates(seed=SEED):
    print("\n\n****************************************************")
    topic = "4. Uniform dates"; print("** %s" % topic)
    
    print('-------------------------------------------EXPLORING')
    # Print the header of account_opened
    print(banking.account_opened.head(), '\n')
    
    print('----------------------------TRANSFORMING TO DATETIME')
    # Convert account_opened to datetime
    banking['account_opened'] = pd.to_datetime(banking.account_opened,
                                               infer_datetime_format = True, # Infer datetime format
                                               errors = 'coerce')  # Return missing value for error
    print(banking.account_opened.head(), '\n')
    
    print('-------------------------------GETTING ONLY THE YEAR')
    # Get year of account opened
    banking['acct_year'] = banking['account_opened'].dt.strftime('%Y')
    
    # Print acct_year
    print(banking[['account_opened','acct_year']].head(), '\n')
    
    
        
def Cross_field_validation(seed=SEED):
    print("\n\n****************************************************")
    topic = "5. Cross field validation"; print("** %s" % topic)
    
    # Prepare the data
    flights = pd.DataFrame({'flight_number'   : ['DL14', 'BA248', 'MEA124', 'AFR939', 'TKA101'],
                            'economy_class'   : [100, 130, 100, 140, 130],
                            'business_class'  : [ 60, 100,  50,  70, 100],
                            'first_class'     : [ 40,  70,  50,  90,  20],
                            'total_passengers': [200, 300, 200, 300, 300]})
    family = pd.DataFrame({'Member'  : ['Father', 'Mother', 'Daughter', 'Baby'],
                           'Birthday': ['1973-05-30', '1976-06-06', '2000-09-09', '2008-01-19'],
                           'Age'     : [47, 25, 20, 12]})
    
    print('------------------------------------------FIRST CASE')
    print(flights, '\n')
    
    sum_classes = flights[['economy_class', 'business_class', 'first_class']].sum(axis = 1)
    passenger_equ = sum_classes == flights['total_passengers']
    
    # Find and filter out rows with inconsistent passengers
    print("inconsistent pass: \n{}\n".format(flights[~passenger_equ]))
    print("consistent pass: \n{}\n".format(flights[passenger_equ]))
    
    print('-----------------------------------------SECOND CASE')
    # Convert to datetime and get today's date
    family['Birthday'] = pd.to_datetime(family['Birthday'])
    
    # For each row in the Birthday column, calculate year difference
    #-------------------------------------------USING PANDAS LIBRARY
    today = pd.Timestamp.now()
    age_manual = today.year - family['Birthday'].dt.year
    print("Manual age, using pd.Timestamp: {}".format(age_manual.values))
    
    #-----------------------------------------USING DATETIME LIBRARY
    today = dt.date.today()
    age_manual = today.year - family['Birthday'].dt.year
    print("Manual age, using datetime: {}".format(age_manual.values))
    
    # Find instances where ages match
    age_equ = age_manual == family['Age']

    # Find and filter out rows with inconsistent age
    print("inconsistent age: \n{}\n".format(family[~age_equ]))
    print("consistent age: \n{}\n".format(family[age_equ])) 
    
    

def Hows_our_data_integrity(seed=SEED):
    print("\n\n****************************************************")
    topic = "7. How's our data integrity?"; print("** %s" % topic)
    
    
    # Prepare the data
    banking['birth_date'] = pd.to_datetime(banking.birth_date,
                                           dayfirst = True, # Infer datetime format
                                           errors = 'coerce')  # Return missing value for error
    
    print('-------------------------------------------EXPLORING')
    # Print the header of account_opened
    print(banking.head(), '\n')
    
    print('-----------CROSS FIELD CHECKING VALUES OF INV_AMOUNT')    
    # Store fund columns to sum against
    fund_columns = ['fund_A', 'fund_B', 'fund_C', 'fund_D']
    
    # Find rows where fund_columns row sum == inv_amount
    inv_equ = banking[fund_columns].sum(axis=1) == banking.inv_amount

    # Store consistent and inconsistent data
    consistent_inv = banking[inv_equ]
    inconsistent_inv = banking[~inv_equ]

    # Store consistent and inconsistent data
    print("Number of consistent investments: ", consistent_inv.shape[0])
    print("Number of inconsistent investments: ", inconsistent_inv.shape[0])
    
    print('------------------CROSS FIELD CHECKING VALUES OF AGE')
    # Store today's date and find ages
    today = dt.date.today()
    ages_manual = today.year - banking.birth_date.dt.year
    
    # Find rows where age column == ages_manual
    age_equ = ages_manual == banking.Age
    
    # Store consistent and inconsistent data
    consistent_ages = banking[age_equ]
    inconsistent_ages = banking[~age_equ]
    
    # Store consistent and inconsistent data
    print("Number of consistent ages: ", consistent_ages.shape[0])
    print("Number of inconsistent ages: ", inconsistent_ages.shape[0])
    
    
    
def Completeness(seed=SEED):
    print("\n\n****************************************************")
    topic = "8. Completeness"; print("** %s" % topic)
    
    print('-------------------------------------------EXPLORING')
    print(airquality.info(), '\n')
    print(airquality.head(), '\n')
    
    print('------------------------------FINDING MISSING VALUES')
    # Return missing values: .isna() == .isnull()
    print(airquality.isna().head(), "\n")
    
    print('--------------------------GET SUMMARY OF MISSINGNESS')
    # Get summary of missingness
    print(airquality.isna().sum(), "\n")
    
    print('---------------------------VISUALIZING MISSING VALUES')
    # Visualize missingness
    fig, axis = plt.subplots(1, 3, figsize=(12.1, 4))
    ax = axis[0]
    msno.matrix(airquality, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Visualizing missing values\nas Matrix', **title_param)
    
    ax = axis[1]
    msno.bar(airquality, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.set_title('Visualizing missing values\nas Bar\n', **title_param)
    
    ax = axis[2]
    msno.heatmap(airquality, fontsize=8, n=6, vmin=0, vmax=1, ax=ax)
    ax.set_title('Visualize only missing values\nas Heatmap\n\n\n', **title_param)
    
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.7, wspace=.7, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('-----------------------------------------ISOLATE DATA')
    # Isolate missing and complete values aside
    missing_values = np.logical_and(airquality['Ozone'].isna(), airquality['Solar.R'].isna())
    missing = airquality[missing_values]
    print("Rows with missing values;\n{}\n".format(missing.head()))
    complete = airquality[~missing_values]
    print("Complete rows;\n{}\n".format(complete.head()))
    
    print('-------------------------------DESCRIBE COMPLETE DATA')
    print(complete.describe(), '\n')
    
    print('--------------------------------DESCRIBE MISSIGN DATA')
    print(missing.describe(), '\n')
    
    print('-------------------------------------FINDING PATTERNS')
    sorted_airquality = airquality.sort_values(by = ['Temp'])
    
    fig, ax = plt.subplots()
    msno.matrix(sorted_airquality, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Order by Temp', **title_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.7, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('-----------------------DROPPING VALUES - FIRST METHOD')
    # Drop missing values
    airquality_dropped = airquality.dropna(subset = ['Ozone', 'Solar.R'])
    print("Summary of missingness: \n{}\n".format(airquality_dropped.isna().sum()))
    
    print('--REPLACING WITH STATISTICAL MEASURES - SECOND METHOD')
    ozone_mean = airquality['Ozone'].mean()
    solar_mean = airquality['Solar.R'].mean()
    airquality_imputed = airquality.fillna({'Ozone'  : ozone_mean,
                                            'Solar.R': solar_mean})
    print(airquality_imputed.head())
    
    
            
def Missing_investors(seed=SEED):
    print("\n\n****************************************************")
    topic = "10. Missing investors"; print("** %s" % topic)
    
    # Prepare the data
    banking_missing['account_opened'] = pd.to_datetime(banking_missing.account_opened,
                                                       dayfirst = True, # Infer datetime format
                                                       errors = 'coerce')  # Return missing value for error
    banking_missing['last_transaction'] = pd.to_datetime(banking_missing.last_transaction,
                                                         dayfirst = True, # Infer datetime format
                                                         errors = 'coerce')  # Return missing value for error
    
    print('-------------------------------------------EXPLORING')
    print(banking_missing.info(), '\n')
    
    print('------------------------------FINDING MISSING VALUES')
    # Print number of missing values in banking
    print("Summary of missing values: \n{}".format(banking_missing.isna().sum()))
    
    # Visualize missingness matrix
    fig, axis = plt.subplots(1, 3, figsize=(12.1, 4))
    ax = axis[0]
    msno.matrix(banking_missing, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Visualizing missing values\nas Matrix', **title_param)
    
    ax = axis[1]
    msno.bar(banking_missing, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.set_title('Visualizing missing values\nas Bar\n', **title_param)
    
    ax = axis[2]
    msno.heatmap(banking_missing, fontsize=8, n=6, vmin=0, vmax=1, ax=ax)
    ax.set_title('Visualize only missing values\nas Heatmap\n\n\n', **title_param)
    
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.7, wspace=.7, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('-----------------------------------------ISOLATE DATA')
    # Isolate missing and non missing values of inv_amount
    missing_investors = banking_missing[banking_missing.inv_amount.isna()]
    investors = banking_missing[~banking_missing.inv_amount.isna()]
    
    print("Describe missing data: \n{}\n".format(missing_investors.describe()))
    print("Describe complete data: \n{}\n".format(investors.describe()))

    print('-------------------------------------FINDING PATTERNS')
    # Sort banking by age and visualize
    banking_sorted = banking_missing.sort_values(by='age')
    
    # Visualize missingness matrix
    fig, ax = plt.subplots()
    msno.matrix(banking_sorted, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Order by Age', **title_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.7, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    

def Follow_the_money(seed=SEED):
    print("\n\n****************************************************")
    topic = "11. Follow the money"; print("** %s" % topic)
    
    # Prepare the data
    banking_missing2['account_opened'] = pd.to_datetime(banking_missing2.account_opened,
                                                        dayfirst = True, # Infer datetime format
                                                        errors = 'coerce')  # Return missing value for error
    banking_missing2['last_transaction'] = pd.to_datetime(banking_missing2.last_transaction,
                                                          dayfirst = True, # Infer datetime format
                                                          errors = 'coerce')  # Return missing value for error
    
    print('-------------------------------------------EXPLORING')
    print(banking_missing2.info(), '\n')
    
    print('------------------------------FINDING MISSING VALUES')
    # Print number of missing values in banking
    print("Summary of missing values: \n{}".format(banking_missing2.isna().sum()))
    
    # Visualize missingness matrix
    fig, axis = plt.subplots(1, 3, figsize=(12.1, 4))
    ax = axis[0]
    msno.matrix(banking_missing2, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Visualizing missing values\nas Matrix', **title_param)
    
    ax = axis[1]
    msno.bar(banking_missing2, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.set_title('Visualizing missing values\nas Bar\n', **title_param)
    
    ax = axis[2]
    msno.heatmap(banking_missing2, fontsize=8, n=6, vmin=0, vmax=1, ax=ax)
    ax.set_title('Visualize only missing values\nas Heatmap\n\n\n', **title_param)
    
    plt.subplots_adjust(left=None, bottom=.3, right=None, top=.6, wspace=.7, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('--------------------DROPPING ROWS WUTH MISSING VALUES')
    # Drop missing values of cust_id
    banking_fullid = banking_missing2.dropna(subset = ['cust_id'])
    
    # Compute estimated acct_amount
    acct_imp = banking_fullid.inv_amount * 5
    
    # Impute missing acct_amount with corresponding acct_imp
    banking_imputed = banking_fullid.fillna({'acct_amount':acct_imp})
    
    # Print number of missing values
    print(banking_imputed.isna().sum())
    


def main(seed=SEED):
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Uniformity()
    Uniform_currencies()
    Uniform_dates()
    Cross_field_validation()
    Hows_our_data_integrity()
    Completeness()
    Missing_investors()
    Follow_the_money()
        
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()
    plt.style.use('default')