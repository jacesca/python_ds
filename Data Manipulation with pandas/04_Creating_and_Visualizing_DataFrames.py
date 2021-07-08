# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:29:35 2020

@author: jacesca@gmail.com
Chapter 4: Creating and Visualizing DataFrames
    Learn to visualize the contents of your DataFrames, handle missing data values, 
    and import data from and export data to CSV files.
Source: https://learn.datacamp.com/courses/data-manipulation-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import MaxNLocator


###############################################################################
## Preparing the environment
###############################################################################
    
# Global variables
#SEED = 42
#np.random.seed(SEED) 

#Global configuration
#np.set_printoptions(formatter={'float': '{:,.3f}'.format})
#pd.options.display.float_format = '{:,.3f}'.format 
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 8, 'font.size': 8})
suptitle_param   = dict(color='darkblue', fontsize=10)
title_param      = {'color': 'darkred', 'fontsize': 12}


# Read the data
dog_pack = pd.read_pickle("dogs_shelter.pkl.bz2", compression='bz2')
avocados = pd.read_pickle("avoplotto.pkl")
avocados['date'] = pd.to_datetime(avocados['date'])
avocados_2016 = pd.read_csv("avocados_2016.csv", sep=';')

dogs = pd.DataFrame({'Name'         : ['Bella', 'Charlie', 'Lucy', 'Cooper', 'Max', 'Stella', 'Bernie'],
                     'Breed'        : ['Labrador', 'Poodle', 'Chow Chow', 'Schnauzer', 'Labrador', 'Chihuahua', 'St. Bernard'],
                     'Color'        : ['Brown', 'Black', 'Brown', 'Gray', 'Black', 'Tan', 'White'],
                     'Height (cm)'  : [56, 43, 46, 49, 59, 18, 77],
                     'Weight (kg)'  : [np.nan, 23, 22, np.nan, 29, 2, 74],
                     'Date of birth': ['2013-07-01', '2016-09-16', '2014-08-25', '2011-12-11', '2017-01-20', '2015-04-20', '2018-02-27']})
    
            
###############################################################################
## Main part of the code
###############################################################################
def Visualizing_your_data():
    print("****************************************************")
    topic = "1. Visualizing your data"; print("** %s" % topic)
    print("****************************************************")
    
    dog_pack["Updated_age"] = pd.Timestamp.now().year - dog_pack['DateTime'].dt.year
    
    print('------------------------------------------HISTOGRAMS')
    fig, ax = plt.subplots()
    dog_pack["Age upon Intake"].hist(ax=ax, rwidth=.95)
    ax.set_xlabel('Age upon Intake')
    ax.set_ylabel('Frequency')
    ax.set_title('Age upon Intake Distribution\n(Histogram Plot)', **title_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('-------------------------------------------------BAR')
    dog_avg_age_upon_intake = dog_pack.groupby('Breed')["Age upon Intake"].mean()
    print(dog_avg_age_upon_intake)
    
    fig, ax = plt.subplots()
    dog_avg_age_upon_intake.head(10).plot(ax=ax, kind="bar", width=.8, rot=70)
    ax.set_xlabel('Breed')
    ax.set_ylabel('Mean Age upon Intake')
    ax.set_title('Mean Age upon Intake per Breed\n(Bar Plot)', **title_param)
    plt.subplots_adjust(left=None, bottom=.3, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('-------------------------------------------LINE PLOT')
    dog_intake_per_month = dog_pack.groupby('MonthYear')['Animal Type'].count().reset_index()
    dog_intake_per_month['MonthYear'] = pd.to_datetime(dog_intake_per_month['MonthYear'])
    print(dog_intake_per_month.head())
    
    fig, ax = plt.subplots()
    dog_intake_per_month.plot(ax=ax, x='MonthYear', y='Animal Type', kind="line", lw=5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Dogs intake')
    ax.set_title('Dogs intake per Month\n(Line Plot)', **title_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------SCATTER')
    dog_intake_data = dog_pack.groupby('MonthYear').agg({'Animal Type': 'count', 
                                                         'Age upon Intake': 'mean'}).reset_index()
    dog_intake_data['MonthYear'] = pd.to_datetime(dog_intake_data['MonthYear'])
    print(dog_intake_data.head())
    
    fig, ax = plt.subplots()
    dog_intake_data.plot(ax=ax, x='Animal Type', y='Age upon Intake', kind="scatter")
    ax.set_xlabel('Dogs Intake')
    ax.set_ylabel('Mean Age upon Intake')
    ax.set_title("Relationship between Number Intakes and Dog's 'Age\n(Scatter Plot)", **title_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('--------------------------------------LAYERING PLOTS')
    fig, ax = plt.subplots()
    dog_pack[dog_pack["sex"]=="Female"]["Age upon Intake"].hist(ax=ax, alpha=.8, rwidth=.9)
    dog_pack[dog_pack["sex"]=="Male"]["Age upon Intake"].hist(ax=ax, alpha=.8, rwidth=.6)
    ax.set_xlabel('Age upon Intake')
    ax.set_ylabel('Frequency')
    ax.set_title('Age upon Intake between Female and Male\n(Histogram Plot)', **title_param)
    plt.legend(["Female", "Male"])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Which_avocado_size_is_most_popular():
    print("****************************************************")
    topic = "2. Which avocado size is most popular?"; print("** %s" % topic)
    print("****************************************************")
    
    # Look at the first few rows of data
    print(avocados.head())
    # Get the total number of avocados sold of each size
    nb_sold_by_size = avocados.groupby('size').nb_sold.sum()/1e+6
    print(nb_sold_by_size)
    
    # Create a bar plot of the number of avocados sold by size
    fig, ax = plt.subplots()
    nb_sold_by_size.plot(ax=ax, kind='bar', width=.8)
    ax.axhline(nb_sold_by_size.max(), ls='--', lw=.5, color='black')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) #Delete the scientist notation
    ax.set_xlabel('Avocado Size')
    ax.set_ylabel('Avocados sold (in million)')
    ax.set_title('Number of avocados sold per Size\n(Bar Plot)', **title_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Changes_in_sales_over_time():
    print("****************************************************")
    topic = "3. Changes in sales over time"; print("** %s" % topic)
    print("****************************************************")
    
    # Get the total number of avocados sold on each date
    nb_sold_by_date = avocados.groupby('date').nb_sold.sum()/1e+6
    
    # Create a bar plot of the number of avocados sold by size
    fig, ax = plt.subplots()
    #All valid ways to do it
    nb_sold_by_date.plot(ax=ax, kind='line', rot=70)
    #nb_sold_by_date.plot(ax=ax, x='date', y='nb_sold', kind='line', rot=70)
    #nb_sold_by_date.plot(ax=ax, x='index', y='nb_sold', kind='line', rot=70)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) #Delete the scientist notation
    ax.set_xlabel('Dates')
    ax.set_ylabel('Number of avocados sold (in million)')
    ax.set_title('Number of avocados sold per day\n(Line Plot)', **title_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Avocado_supply_and_demand():
    print("****************************************************")
    topic = "4. Avocado supply and demand"; print("** %s" % topic)
    print("****************************************************")
    
    # Scatter plot of nb_sold vs avg_price with title
    fig, ax = plt.subplots()
    avocados.plot(ax=ax, x='nb_sold', y='avg_price', kind='scatter')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) #Delete the scientist notation
    ax.yaxis.set_major_formatter(StrMethodFormatter('$ {x:,.2f}')) #Delete the scientist notation
    ax.set_xlabel('Avocados sold')
    ax.set_ylabel('Average price (in dollars)')
    ax.set_title('Number of avocados sold vs. average price\n(Scatter Plot)', **title_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Price_of_conventional_vs_organic_avocados():
    print("****************************************************")
    topic = "5. Price of conventional vs. organic avocados"; print("** %s" % topic)
    print("****************************************************")
    
    fig, ax = plt.subplots()
    bins = np.linspace(.6, 2.2, 17)
    # Histogram of conventional avg_price 
    avocados[avocados["type"] == "conventional"]["avg_price"].hist(ax=ax, alpha=0.7, bins=bins, rwidth=.9)
    # Histogram of organic avg_price
    avocados[avocados["type"] == "organic"]["avg_price"].hist(ax=ax, alpha=0.7, bins=bins, rwidth=.6)
    ax.xaxis.set_major_formatter(StrMethodFormatter('$ {x:,.2f}')) #Delete the scientist notation
    ax.set_xlabel('Average Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Average Price between different Avocados Kind\n(Histogram Plot)', **title_param)
    plt.legend(['conventional','organic'])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Missing_values():
    print("****************************************************")
    topic = "6. Missing values"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    print(dogs)
    
    print('----------------------------DETECTING MISSING VALUES')
    print(dogs.isna())
    
    print('------------------------DETECTING ANY MISSING VALUES')
    print(dogs.isna().any())
    
    print('-----------------------------COUNTING MISSING VALUES')
    print(dogs.isna().sum())
    
    print('-----------------------------PLOTTING MISSING VALUES')
    fig, ax = plt.subplots()
    dogs.isna().sum().plot(ax=ax, kind='bar', width=.9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) #Integer format
    ax.set_xlabel('Variables')
    ax.set_ylabel('Number of missing Values')
    ax.set_title('Counting Missing Values\n(Bar Plot)', **title_param)
    plt.subplots_adjust(left=None, bottom=.25, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('-----------------------------REMOVING MISSING VALUES')
    print(dogs.dropna())
    
    print('----------------------------REPLACING MISSING VALUES')
    print(dogs.fillna(0))
    
    
    
def Finding_missing_values():
    print("****************************************************")
    topic = "7. Finding missing values"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    print(avocados_2016)
    
    print('--------------CHECKING FOR INDIVIDUAL MISSING VALUES')
    # Check individual values for missing values
    print(avocados_2016.isna())
    
    print('-----------------CHECKING FOR MISSING IN EACH COLUMN')
    # Check each column for missing values
    print(avocados_2016.isna().any())
    
    print('-----------------------------COUNTING MISSING VALUES')
    print(avocados_2016.isna().sum())
    
    print('-------------------------PLOTTING THE MISSING VALUES')
    # Bar plot of missing values by variable
    fig, ax = plt.subplots()
    avocados_2016.isna().sum().plot(ax=ax, kind='bar', width=.9)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Number of missing Values')
    ax.set_title('Counting Missing Values\n(Bar Plot)', **title_param)
    plt.subplots_adjust(left=None, bottom=.25, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Removing_missing_values():
    print("****************************************************")
    topic = "8. Removing missing values"; print("** %s" % topic)
    print("****************************************************")
    
    # Remove rows with missing values
    avocados_complete = avocados_2016.dropna()
    # Check if any columns contain missing values
    print(avocados_complete.isna().any())
    
    
    
def Replacing_missing_values():
    print("****************************************************")
    topic = "9. Replacing missing values"; print("** %s" % topic)
    print("****************************************************")
    
    
    print('---------------------------------------DOGS EXAMPLES')
    # Plot histograms for multiple variables at a time 
    fig, axis = plt.subplots(1,2, figsize=(10, 4))
    dogs_columns = ["Height (cm)", "Weight (kg)"]
    dogs[dogs_columns].hist(ax=axis, rwidth=.9)
    
    for i, ax in enumerate(axis):
        ax.set_xlabel(dogs_columns[i])
        ax.set_ylabel('Frequency')
        ax.set_title(f'{dogs_columns[i]} Distribution\n(Histogram Plot)', **title_param)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)) #Integer format
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---AVOCADOS_2016 EXAMPLES BEFORE FILLING NULL VALUES')
    # List the columns with missing values
    cols_with_missing = ["small_sold", "large_sold", "xl_sold"]
    row_axis = {'before': 0, 'after': 1}
    # Create histograms showing the distributions cols_with_missing
    fig, axis = plt.subplots(2,3, figsize=(12.1, 5.5))
    avocados_2016[cols_with_missing].hist(ax=axis[row_axis['before'],:], rwidth=.9, xrot=45)
    
    print('----AVOCADOS_2016 EXAMPLES AFTER FILLING NULL VALUES')
    # Fill in missing values with 0
    avocados_filled = avocados_2016.fillna(0)
    
    # Create histograms of the filled columns
    avocados_filled[cols_with_missing].hist(ax=axis[row_axis['after'],:], rwidth=.9, xrot=45)
    
    for label, r_ax in row_axis.items():
        for i, ax in enumerate(axis[r_ax,:]):
            ax.set_xlabel(cols_with_missing[i])
            ax.set_ylabel('Frequency')
            ax.set_title(f'{cols_with_missing[i]} Distribution\n({label} filling Null Values)', **title_param)
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) #Delete the scientist notation
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.85, wspace=None, hspace=1.4); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
def Creating_DataFrames():
    print("****************************************************")
    topic = "10. Creating DataFrames"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------------FROM A LIST OF DICTIONARIES')
    list_of_dicts = [{"name": "Ginger", "breed": "Dachshund", "height_cm": 22, "weight_kg": 10, "date_of_birth": "2019-03-14"},
                     {"name": "Scout",  "breed": "Dalmatian", "height_cm": 59, "weight_kg": 25, "date_of_birth": "2019-05-09"}]
    new_dogs = pd.DataFrame(list_of_dicts)
    print(new_dogs)
    
    print('--------------------------FROM A DICTIONARY OF LISTS')
    dict_of_lists = {"name": ["Ginger", "Scout"],
                     "breed": ["Dachshund", "Dalmatian"],
                     "height_cm": [22, 59],
                     "weight_kg": [10, 25],
                     "date_of_birth": ["2019-03-14","2019-05-09"]}
    new_dogs = pd.DataFrame(dict_of_lists)
    print(new_dogs)
    
    
    
def List_of_dictionaries():
    print("****************************************************")
    topic = "11. List of dictionaries"; print("** %s" % topic)
    print("****************************************************")
    
    # Create a list of dictionaries with new data
    avocados_list = [{'date': '2019-11-03', 'small_sold': 10376832, 'large_sold': 7835071},
                     {'date': '2019-11-10', 'small_sold': 10717154, 'large_sold': 8561348}]
    # Convert list into DataFrame
    avocados_2019 = pd.DataFrame(avocados_list)
    # Print the new DataFrame
    print(avocados_2019)
    
    
    
def Dictionary_of_lists():
    print("****************************************************")
    topic = "12. Dictionary of lists"; print("** %s" % topic)
    print("****************************************************")
    
    # Create a dictionary of lists with new data
    avocados_dict = {"date": ['2019-11-17', '2019-12-01'],
                     "small_sold": [10859987, 9291631],
                     "large_sold": [7674135, 6238096]}
    # Convert dictionary into DataFrame
    avocados_2019 = pd.DataFrame(avocados_dict)
    # Print the new DataFrame
    print(avocados_2019)
    
    
    
def CSV_to_DataFrame():
    print("****************************************************")
    topic = "14. CSV to DataFrame"; print("** %s" % topic)
    print("****************************************************")
    
    # Read CSV as DataFrame called airline_bumping
    airline_bumping = pd.read_csv('airline_bumping.csv', sep=';')
    # Take a look at the DataFrame
    print(airline_bumping.head())
    # For each airline, select nb_bumped and total_passengers and sum
    airline_totals = airline_bumping.groupby('airline')[['nb_bumped','total_passengers']].sum()
    # Create new col, bumps_per_10k: no. of bumps per 10k passengers for each airline
    airline_totals["bumps_per_10k"] = airline_totals.nb_bumped / airline_totals.total_passengers * 10000
    # Print airline_totals
    print(airline_totals)
    return airline_totals
    
    
    
def DataFrame_to_CSV(airline_totals):
    print("****************************************************")
    topic = "15. DataFrame to CSV"; print("** %s" % topic)
    print("****************************************************")
    
    # Create airline_totals_sorted
    airline_totals_sorted = airline_totals.sort_values(by='bumps_per_10k', ascending=False)
    # Print airline_totals_sorted
    print(airline_totals_sorted)
    # Save as airline_totals_sorted.csv
    airline_totals_sorted.to_csv('airline_totals_sorted.csv')
    
    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Visualizing_your_data()
    Which_avocado_size_is_most_popular()
    Changes_in_sales_over_time()
    Avocado_supply_and_demand()
    Price_of_conventional_vs_organic_avocados()
    Missing_values()
    Finding_missing_values()
    Removing_missing_values()
    Replacing_missing_values()
    Creating_DataFrames()
    List_of_dictionaries()
    Dictionary_of_lists()
    airline_totals = CSV_to_DataFrame()
    DataFrame_to_CSV(airline_totals)
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()
    #np.set_printoptions(formatter = {'float': None}) #Return to default
    #pd.options.display.float_format = None
