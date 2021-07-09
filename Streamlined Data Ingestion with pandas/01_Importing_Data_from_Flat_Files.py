# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 1: Importing Data from Flat Files
    Practice using pandas to get just the data you want from flat files, learn 
    how to wrangle data types and handle errors, and look into some U.S. tax data 
    along the way.
Source: https://learn.datacamp.com/courses/streamlined-data-ingestion-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter



###############################################################################
## Preparing the environment
###############################################################################
# Global variables
csv_file = 'vt_tax_data_2016.csv'
csv_file_corrupt = 'vt_tax_data_2016_corrupt.csv'

# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 8})
suptitle_param = dict(color='darkblue', fontsize=9)
title_param    = {'color': 'darkred', 'fontsize': 10}


        
###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_flat_files():
    print("****************************************************")
    topic = "1. Introduction to flat files"; print("** %s" % topic)
    print("****************************************************")
    topic = "2. Get data from CSVs"; print("** %s" % topic)
    print("****************************************************")
    topic = "3. Get data from other flat files"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Reading tax_data')
    tax_data = pd.read_csv(csv_file)
    
    print('---------------------------------------------Explore tax_data')
    print("Shape: ", tax_data.shape)
    print(tax_data.head())
    print(tax_data.info())
    print(tax_data.dtypes)
    
    print('---------------------------------------------Plot the total of tax returns')
    # Plot the total number of tax returns by income group
    fig, ax = plt.subplots()
    counts = tax_data.groupby("agi_stub").N1.sum()
    counts.plot.bar(ax=ax)
    ax.set_ylabel('Dollars')
    ax.yaxis.set_major_formatter(StrMethodFormatter('$ {x:,.0f}'))
    ax.set_title('Plot the total number of tax returns by income group', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.15, bottom=None, right=.9, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()

    
    
def Modifying_flat_file_imports():
    print("****************************************************")
    topic = "4. Modifying flat file imports"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Limiting Columns')
    col_names = ['STATEFIPS', 'STATE', 'zipcode', 'agi_stub', 'N1']
    # Choose columns to load by name
    tax_data_v1 = pd.read_csv(csv_file, usecols=col_names)
    print("tax_data_v1 shape: ",tax_data_v1.shape)
    
    print('---------------------------------------------Getting equality')
    # Choose columns to load by number
    col_nums = [0, 1, 2, 3, 4]
    tax_data_v2 = pd.read_csv(csv_file, usecols=col_nums)
    print("tax_data1 == tx_data2 : ", tax_data_v1.equals(tax_data_v2))
    
    print('---------------------------------------------Limiting rows')
    tax_data_first1000 = pd.read_csv(csv_file, nrows=1000)
    print("tax_data_first1000 shape: ",tax_data_first1000.shape)
    
    col_names = list(tax_data_first1000)
    tax_data_next500 = pd.read_csv(csv_file,
                                   nrows=500,
                                   skiprows=1000,
                                   header=None,
                                   names=col_names)
    print("tax_data_next500 shape: ",tax_data_next500.shape)
    print(f"Head of tax_data_next500: \n{tax_data_next500.head()}")
    
    
        
def Import_a_subset_of_columns():
    print("****************************************************")
    topic = "5. Import a subset of columns"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Limiting Columns')
    # Create list of columns to use
    cols = ['zipcode', 'agi_stub', 'mars1', 'MARS2', 'NUMDEP']
    # Create data frame from csv using only selected columns
    data = pd.read_csv(csv_file, usecols=cols)
    
    print('---------------------------------------------View counts of dependents and tax returns')
    # View counts of dependents and tax returns by income level
    print(data.groupby("agi_stub").sum())
    
    
    
def Import_a_file_in_chunks():
    print("****************************************************")
    topic = "6. Import a file in chunks"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Reading first 500 rows')
    vt_data_first500 = pd.read_csv(csv_file, nrows=500)
    print(vt_data_first500.head())
    
    print('---------------------------------------------Reading next 500 rows')
    # Create data frame of next 500 rows with labeled columns
    vt_data_next500 = pd.read_csv(csv_file, 
                                  nrows=500,
                                  skiprows=500,
                                  header=None,
                                  names=list(vt_data_first500))
    # View the Vermont data frames to confirm they're different
    print(vt_data_next500.head())
    
    
def Handling_errors_and_missing_data():
    print("****************************************************")
    topic = "7. Handling errors and missing data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Specifying Data Types')
    tax_data = pd.read_csv(csv_file, dtype={"zipcode": str})
    print(tax_data.dtypes)
    print(tax_data.head())
    
    print('---------------------------------------------Customizing Missing Data Values')
    tax_data = pd.read_csv(csv_file,
                           na_values={"zipcode" : 0},
                           dtype={"zipcode": str})
    print(tax_data[tax_data.zipcode.isna()])
    
    print('---------------------------------------------Lines with Errors')
    tax_data = pd.read_csv(csv_file_corrupt,
                           error_bad_lines=False,
                           warn_bad_lines=True) # Show the msg: b'Skipping line 4: expected 147 fields, saw 151\n'
    print(tax_data.head())
    
    
    
def Specify_data_types():
    print("****************************************************")
    topic = "8. Specify data types"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Specifying Data Types')
    # Create dict specifying data types for agi_stub and zipcode
    data_types = {'agi_stub': 'category',
                  'zipcode' : str}
    # Load csv using dtype to set correct data types
    data = pd.read_csv(csv_file, dtype=data_types)
    # Print data types of resulting frame
    print(data.dtypes.head())
        
        
def Set_custom_NA_values():
    print("****************************************************")
    topic = "9. Set custom NA values"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Customizing Missing Data Values')
    # Create dict specifying data types for agi_stub and zipcode
    data_types = {'agi_stub': 'category', 'zipcode' : str}
    # Create dict specifying that 0s in zipcode are NA values
    null_values = {'zipcode': 0}    
    # Load csv using na_values keyword argument
    data = pd.read_csv(csv_file, dtype=data_types, 
                       na_values=null_values)    
    # View rows with NA ZIP codes
    print(data[data.zipcode.isna()])
    
    
def Skip_bad_data():
    print("****************************************************")
    topic = "10. Skip bad data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Lines with Errors')
    try:
        # Set warn_bad_lines to issue warnings about bad records
        data = pd.read_csv(csv_file_corrupt)  
        # View first 5 records
        print(data.head())
    except :
        print("Your data contained rows that could not be parsed. Try it again!")
        try:
            # Set warn_bad_lines to issue warnings about bad records
            data = pd.read_csv("vt_tax_data_2016_corrupt.csv", 
                                   error_bad_lines=False, warn_bad_lines=True)  
            # View first 5 records
            print(data.head())
            print("Did it well, this time!!!")
        except:
            print("Your data contained rows that could not be parsed (second attempt).")
    
    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_flat_files()
    Modifying_flat_file_imports()
    Import_a_subset_of_columns()
    Import_a_file_in_chunks()
    Handling_errors_and_missing_data()
    Specify_data_types()
    Set_custom_NA_values()
    Skip_bad_data()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')