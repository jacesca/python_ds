# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 1: Pandas
    In this chapter, you will review pandas basics including importing datasets, exploratory analysis, and basic plotting.
Source: https://learn.datacamp.com/courses/analyzing-marketing-campaigns-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
## Preparing the environment
###############################################################################
# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 8})
suptitle_param = dict(color='darkblue', fontsize=9)
title_param    = {'color': 'darkred', 'fontsize': 10}
       
###############################################################################
## Reading the data
###############################################################################
marketing = pd.read_csv('marketing_dataset_1.csv')
#marketing2 = pd.read_csv('marketing_dataset_2.csv')

###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_pandas_for_marketing():
    print("****************************************************")
    topic = "1. Introduction to pandas for marketing"; print("** %s" % topic)
    print("****************************************************")
    topic = "2. Importing the dataset"; print("** %s" % topic)
    print("****************************************************")
    topic = "3. Examining the data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Inspecting data')
    print(marketing.head())
    
    print('---------------------------------------------Summary statistics')
    print(marketing.describe())
    
    print('---------------------------------------------Missing values & data types')
    print(marketing.info())
    
    
    
def Data_types_and_data_merging():
    print("****************************************************")
    topic = "4. Data types and data merging"; print("** %s" % topic)
    print("****************************************************")
    topic = "5. Updating the data type of a column"; print("** %s" % topic)
    print("****************************************************")
    topic = "6. Adding new columns"; print("** %s" % topic)
    print("****************************************************")
    topic = "7. Date columns"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Data type of a column')
    # Print a data type of a single column
    print('converted column type: ', marketing['converted'].dtypes)
    
    print('---------------------------------------------Data type of all column')
    # Print data type of all columns
    print(marketing.dtypes)
    
    print('---------------------------------------------Read Date columns in pandas')
    # Read date columns using parse_dates
    marketing2 = pd.read_csv('marketing_dataset_1.csv',
                             parse_dates=['date_served', 'date_subscribed', 'date_canceled'])
    # Print data type of all columns
    print(marketing2.dtypes)
    
    print('---------------------------------------------Converting Date columns')
    # Or
    # Convert already existing column to datetime column
    marketing['date_served'] = pd.to_datetime(marketing['date_served'])
    marketing['date_subscribed'] = pd.to_datetime(marketing['date_served'])
    marketing['date_canceled'] = pd.to_datetime(marketing['date_canceled'])

    print('---------------------------------------------Changing the data type of a column')
    # Change the data type of a column
    marketing['converted'] = marketing['converted'].astype(bool)
    marketing['is_retained'] = marketing['is_retained'].astype(bool)
    print(marketing[['converted', 'is_retained']].dtypes)
    
    print('---------------------------------------------Creating new boolean column')
    marketing['is_house_ads'] = np.where(marketing['marketing_channel'] == 'House Ads', True, False)
    print(marketing[['marketing_channel', 'is_house_ads']].head(3))

    print('---------------------------------------------Creating Yes/No column')
    # Add the new column is_correct_lang
    marketing['is_correct_lang'] = np.where(marketing.language_displayed==marketing.language_preferred, 'Yes', 'No')
    print(marketing[['language_displayed', 'language_preferred', 'is_correct_lang']].head(3))
    
    print('---------------------------------------------Mapping values to existing columns')
    print("Values in marketing_channel column: ", marketing.marketing_channel.unique())
    channel_dict = {"House Ads": 1, 
                    "Instagram": 2, 
                    "Facebook": 3, 
                    "Email": 4, 
                    "Push": 5}
    marketing['channel_code'] = marketing['marketing_channel'].map(channel_dict)
    print(marketing[['marketing_channel', 'channel_code']].head(3))
        
    print('---------------------------------------------Day of week')
    marketing['day_served'] = marketing['date_served'].dt.dayofweek
    print('Unique values in day_served: ', marketing.day_served.unique())
    
    print('---------------------------------------------Explore')
    # Print data type of all columns
    print(marketing.dtypes)
    
    
    
def Initial_exploratory_analysis():
    print("****************************************************")
    topic = "8. Initial exploratory analysis"; print("** %s" % topic)
    print("****************************************************")
    topic = "9. Daily marketing reach by channel"; print("** %s" % topic)
    print("****************************************************")
    topic = "10. Visualizing daily marketing reach"; print("** %s" % topic)
    print("****************************************************")
    
    ask_to_respond ='How many users see marketing assets?'
    print(f'---------------------------------------------{ask_to_respond}')
    # Aggregate unique users that see ads by date
    daily_users = marketing.groupby(['date_served'])['user_id'].nunique()
    print(daily_users)
    
    # Visualizing results
    fig, ax = plt.subplots()
    daily_users.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of users')
    ax.set_title(ask_to_respond, **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    print('---------------------------------------------Explore')
    
    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_pandas_for_marketing()
    Data_types_and_data_merging()
    Initial_exploratory_analysis()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    plt.style.use('default')