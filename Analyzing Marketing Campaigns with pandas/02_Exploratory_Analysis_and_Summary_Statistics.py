# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 2: Exploratory Analysis & Summary Statistics
    In this chapter, you will learn about common marketing metrics and how to 
    calculate them using pandas. You will also visualize your results and practice 
    user segmentation.
Source: https://learn.datacamp.com/courses/streamlined-data-ingestion-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
import pandas as pd
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
marketing = pd.read_csv('marketing_dataset_3.csv', 
                        parse_dates=['date_served', 'date_subscribed', 'date_canceled'],
                        dtype={'converted': bool,
                               'is_retained': bool})
#true_values = pd.read_csv('test.csv', dtype={'Value':'bool'})
#marketing['is_retained'] = true_values.Value
#marketing.to_csv('marketing_dataset_2.csv')
###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_common_marketing_metrics():
    print("****************************************************")
    topic = "1. Introduction to common marketing metrics"; print("** %s" % topic)
    print("****************************************************")
    topic = "2. Calculating conversion rate"; print("** %s" % topic)
    print("****************************************************")
    topic = "3. Calculating retention rate"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------General Conversion rate')
    # Conversion rate = Number of people who convert / Total number of people we marketed to
    
    subscribers = marketing[marketing['converted'] == True]['user_id'].nunique()
    total = marketing['user_id'].nunique()
    
    conv_rate = subscribers/total
    
    print(round(conv_rate*100, 2), '%')
    
    print('---------------------------------------------General Retention rate')
    #Retention rate = Number of people who remain subscribed / Total number of people who converted
    
    retained = marketing[marketing['is_retained'] == True]['user_id'].nunique() 
    subscribers = marketing[marketing['converted'] == True]['user_id'].nunique()
    retention = retained / subscribers
    
    print(round(retention*100, 2), '%')
    
    
    
def Customer_segmentation():
    print("****************************************************")
    topic = "4. Customer segmentation"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Retantion rate for House Ads')
    # Subset to include only House Ads
    house_ads = marketing[marketing['subscribing_channel'] == 'House Ads']
    
    retained = house_ads[house_ads['is_retained'] == True]['user_id'].nunique()
    subscribers = house_ads[house_ads['converted'] == True]['user_id'].nunique()
    
    retention_rate = retained/subscribers
    print(round(retention_rate*100,2), '%')

    print('---------------------------------------------Chanel retention rate')
    # Group by subscribing_channel and calculate retention
    retained = marketing[marketing.is_retained == True].groupby(['subscribing_channel'])['user_id'].nunique()
    
    # Group by subscribing_channel and calculate subscribers
    subscribers = marketing[marketing['converted'] == True].groupby(['subscribing_channel'])['user_id'].nunique()
    
    # Calculate the retention rate across the DataFrame
    channel_retention_rate = (retained/subscribers)*100
    print(channel_retention_rate)
    
    
    
def Comparing_language_conversion_rate_I():
    print("****************************************************")
    topic = "5. Comparing language conversion rate (I)"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Conversion Rate')
    # Isolate english speakers
    english_speakers = marketing[marketing['language_displayed'] == 'English']
    
    # Calculate the total number of English speaking users
    total = english_speakers.user_id.nunique()
    
    # Calculate the number of English speakers who converted
    subscribers = english_speakers[english_speakers.converted == True].user_id.nunique()
    
    # Calculate conversion rate
    conversion_rate = subscribers/total
    print('English speaker conversion rate:', round(conversion_rate*100,2), '%')
    
    
    
def Comparing_language_conversion_rate_II():
    print("****************************************************")
    topic = "6. Comparing language conversion rate (II)"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Conversion rate')
    # Group by language_displayed and count unique users
    total = marketing.groupby('language_displayed').user_id.nunique()
    
    # Group by language_displayed and count unique conversions
    subscribers = marketing[marketing.converted==True].groupby('language_displayed').user_id.nunique()
    
    # Calculate the conversion rate for all languages
    language_conversion_rate = subscribers/total
    print(language_conversion_rate)
    return language_conversion_rate
    
    
def Aggregating_by_date():
    print("****************************************************")
    topic = "7. Aggregating by date"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Conversion rate')
    # Group by date_served and count unique users
    total = marketing.groupby('date_served').user_id.nunique()
    
    # Group by date_served and count unique converted users
    subscribers = marketing[marketing.converted == True].groupby('date_served').user_id.nunique()
    
    # Calculate the conversion rate per day
    daily_conversion_rate = subscribers/total
    print(daily_conversion_rate)
    
    return daily_conversion_rate
    
    
    
def Plotting_campaign_results_I(language_conversion_rate, daily_conversion_rate):
    print("****************************************************")
    topic = "8. Plotting campaign results (I)"; print("** %s" % topic)
    print("****************************************************")
    topic = "9. Visualize conversion rate by language"; print("** %s" % topic)
    print("****************************************************")
    topic = "10. Creating daily conversion rate DataFrame"; print("** %s" % topic)
    print("****************************************************")
    topic = "11. Setting up our data to visualize daily conversion"; print("** %s" % topic)
    print("****************************************************")
    topic = "12. Visualize daily conversion rate"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Language conversion rate')
    # Create a bar chart using channel retention DataFrame
    fig, ax = plt.subplots()
    language_conversion_rate.plot(kind = 'bar', ax=ax)
    ax.set_xlabel('Language')
    ax.set_ylabel('Conversion rate (%)')
    ax.set_title('Conversion rate by language', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    print('---------------------------------------------Date conversion rate')
    #Preparing data to be plotted over time
    # Reset index to turn the Series into a DataFrame
    #daily_retention_rate = pd.DataFrame(daily_conversion_rate.reset_index())
    daily_retention_rate = daily_conversion_rate.reset_index()
    # Rename columns
    daily_retention_rate.columns = ['date_subscribed', 'retention_rate']
    
    # Create a line chart using the daily_retention DataFrame
    fig, ax = plt.subplots()
    daily_retention_rate.plot('date_subscribed', 'retention_rate', ax=ax)
    ax.set_ylim(0)
    ax.set_xlabel('Date')
    ax.set_ylabel('1-month retention rate (%)')
    ax.set_title('Daily subscribed quality', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Plotting_campaign_results_II():
    print("****************************************************")
    topic = "13. Plotting campaign results (II)"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Preferred language over time')
    # (1) Grouping by multiple columns
    language = marketing.groupby(['date_served', 'language_preferred'])['user_id'].count()
    print(language.head())
    
    # (2) Unstacking after groupby
    language = language.unstack(level='language_preferred')
    print(language.head())
    
    # (3) Plotting preferred language over time
    fig, ax = plt.subplots()
    language.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Users')
    ax.legend(labels = language.columns.values)
    ax.set_title('Daily language preferences', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    print('---------------------------------------------Language preferences by age group')
    # (1) Grouping by multiple columns
    language_age = marketing.groupby(['language_preferred', 'age_group'])['user_id'].count()
    print(language_age.head())
    
    # (2) Unstacking after groupby
    language_age = language_age.unstack(level='age_group')
    print(language_age.head())
    
    # (3) Plotting preferred language over time
    fig, ax = plt.subplots()
    language_age.plot(kind='bar', ax=ax)
    ax.set_xlabel('Language')
    ax.set_ylabel('Users')
    ax.legend(labels = language_age.columns.values)
    ax.set_title('Language preferences by age group', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Marketing_channels_across_age_groups():
    print("****************************************************")
    topic = "14. Marketing channels across age groups"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------(1) Grouping by multiple columns')
    channel_age_df = marketing.groupby(['marketing_channel', 'age_group'])['user_id'].count()
    print(channel_age_df.head())
    
    print('---------------------------------------------(2) Unstacking after groupby')
    # Unstack channel_age and transform it into a DataFrame
    channel_age_df = channel_age_df.unstack(level = 'age_group')
    print(channel_age_df.head())
    
    print('---------------------------------------------(3) Plotting preferred language over time')
    # Plot channel_age
    fig, ax = plt.subplots()
    channel_age_df.plot(kind='bar', ax=ax)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Users')
    ax.legend(labels = channel_age_df.columns.values)
    ax.set_title('Marketing channels by age group', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Grouping_and_counting_by_multiple_columns():
    print("****************************************************")
    topic = "15. Grouping and counting by multiple columns"; print("** %s" % topic)
    print("****************************************************")
    
    # Which channel had the highest retention rate.
    print('---------------------------------------------Count subscriptor by channel over time')
    # Count the subs by subscribing channel and day
    retention_total = marketing.groupby(['date_subscribed', 'subscribing_channel']).user_id.nunique()
    # Print results
    print(retention_total.head())
    
    print('---------------------------------------------Counting retainers by channel over time')
    # Sum the retained subs by subscribing channel and date subscribed
    retention_subs = marketing[marketing.is_retained == True].groupby(['date_subscribed', 'subscribing_channel']).user_id.nunique()
    # Print results
    print(retention_subs.head())
    
    print('---------------------------------------------Retained rate')
    # Divide retained subscribers by total subscribers
    retention_rate = retention_subs / retention_total
    print(retention_rate.head())
    
    print('---------------------------------------------Unstacking after groupby')
    retention_rate_df = retention_rate.unstack(level='subscribing_channel')
    print(retention_rate_df.head())
    
    print('---------------------------------------------Plotting preferred language over time')
    # Plot channel_age
    fig, ax = plt.subplots()
    retention_rate_df.plot(ax=ax)
    ax.set_xlabel('Date Subscribed')
    ax.set_ylabel('Retention Rate (%)')
    ax.legend(labels = retention_rate_df.columns.values)
    ax.set_title('Retention Rate by Subscribing Channel', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_common_marketing_metrics()
    
    Customer_segmentation()
    Comparing_language_conversion_rate_I()
    language_conversion_rate = Comparing_language_conversion_rate_II()
    daily_conversion_rate = Aggregating_by_date()
    
    Plotting_campaign_results_I(language_conversion_rate, daily_conversion_rate)
    Plotting_campaign_results_II()
    Marketing_channels_across_age_groups()
    Grouping_and_counting_by_multiple_columns()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    plt.style.use('default')