# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: Conversion Attribution
    In this chapter, you will build functions to automate common marketing analysis 
    and determine why certain marketing channels saw lower than usual conversion 
    rates during late January.
Source: https://learn.datacamp.com/courses/streamlined-data-ingestion-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

###############################################################################
## Preparing the environment
###############################################################################
# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 8})
suptitle_param = dict(color='darkblue', fontsize=9)
title_param    = {'color': 'darkred', 'fontsize': 10, 'weight': 'bold'}
figsize        = (12.1, 5.9)

# Global Funtion
def retention_rate(df, col_groupby,  
                   col_user='user_id', col_retention='is_retained', col_convertion='converted'):
    """
    Return the retention rate of a set in agroup by structure.
    Parameters
    ----------
    df            : Dataframe with the data to analyze.
    col_groupby   : List with the columns_name to make the group by.
    col_user      : String with the name of column to identify users.
    col_retention : String with the name of ccolumn to identify retentions. The default is 'is_retained'.
    col_convertion: String with the name of column to identify conversions. The default is 'converted'.
    Returns
    -------
    retention_rate : Dataframe with the retention rates.
    """
    # Group by column_names and calculate retention
    retained = df[df[col_retention] == True].groupby(col_groupby)[col_user].nunique()
    # Group by column_names and calculate conversion
    converted = df[df[col_convertion] == True].groupby(col_groupby)[col_user].nunique()
    
    retention_rate = retained/converted
    retention_rate = retention_rate.fillna(0)
    return retention_rate

def conversion_rate(df, col_groupby,  
                   col_user='user_id', col_conversion='converted'):
    """
    Return the retention rate of a set in agroup by structure.
    Parameters
    ----------
    df            : Dataframe with the data to analyze.
    col_groupby   : List with the columns_name to make the group by.
    col_user      : String with the name of column to identify users.
    col_convertion: String with the name of column to identify conversions. The default is 'converted'.
    Returns
    -------
    retention_rate : Dataframe with the coversion rates.
    """
    # Group by column_names and calculate retention
    total_user = df.groupby(col_groupby)[col_user].nunique()
    # Group by column_names and calculate conversion
    conversed = df[df[col_conversion] == True].groupby(col_groupby)[col_user].nunique()
    conversion_rate = conversed/total_user
    conversion_rate = conversion_rate.fillna(0)
    return conversion_rate

def plotting_all_in_one(df, y_label, topic, 
                        kind='line', figsize=figsize,
                        title_param=title_param, suptitle_param=suptitle_param):
    """
    Make a plot of the analysis (retention or conversion). All in one plot.
    Parameters
    ----------
    df            : Dataframe with the data. For example:
                        age_group           0-18 years  19-24 years  ...  45-55 years  55+ years
                        language_preferred                           ...                        
                        Arabic                      19           26  ...           21         22
                        English                   1421         1560  ...         1240       1101
                        German                      31           29  ...           25         12
                        Spanish                     68           67  ...           67         52
    y_label       : String with the label of y-axis in the plot.
    topic         : String with the suptitle of the plot. Default is 2.
    kind          : Type of plot: line, bar, etc. Default 'line'. See Pandas.DataFrame.plot.
    figsize       : Size of the graph. The default is (12.1, 5.9).
    title_param   : Params to set the title.
    suptitle_param: PArams to set the suptitle.
    Returns
    -------
    None.
    """
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind=kind, ax=ax)
    #ax.set_ylim(-0.05, df.max().max()*1.05)
    ax.set_xlabel(df.index.name)
    ax.set_ylabel(y_label)
    ax.tick_params(labelsize=6)
    ax.legend(labels = df.columns.values)
    ax.set_title(f'{y_label} by {df.columns.name}', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
def plotting_one_by_one(df, y_label, topic, 
                        kind='line', ncols=2, figsize=figsize,
                        title_param=title_param, suptitle_param=suptitle_param,
                        left=None, bottom=.2, right=None, top=None, 
                        wspace=None, hspace=.7):
    """
    Make a plot of the analysis (retention or conversion). Each column in a different plot.
    Parameters
    ----------
    df            : Dataframe with the data. For example:
                        age_group           0-18 years  19-24 years  ...  45-55 years  55+ years
                        language_preferred                           ...                        
                        Arabic                      19           26  ...           21         22
                        English                   1421         1560  ...         1240       1101
                        German                      31           29  ...           25         12
                        Spanish                     68           67  ...           67         52
    y_label       : String with the label of y-axis in the plot.
    topic         : String with the suptitle of the plot. Default is 2.
    kind          : Type of plot: line, bar, etc. Default 'line'. See Pandas.DataFrame.plot.
    ncols         : Number of cols to make the plot.
    figsize       : Size of the graph. The default is (12.1, 5.9).
    title_param   : Params to set the title.
    suptitle_param: PArams to set the suptitle.
    Returns
    -------
    None.
    """
    nrows = math.ceil(df.shape[1]/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    #y_max_lim = df.max().max()
    for i, (ax, column_name) in enumerate(zip(axes, df)):
        df[column_name].plot(kind=kind, ax=ax)
        ax.set_xlabel(df.index.name)
        ax.set_ylabel(y_label)
        #ax.set_ylim(-0.05, y_max_lim*1.05)
        ax.set_title(f'{df.columns.name}: {column_name}', **title_param)    
    
    if i+1 < nrows*ncols: 
        for k in range(i+1, nrows*ncols): axes[k].axis('off')
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, 
                        wspace=wspace, hspace=hspace); #To set the margins 
    plt.show()
    
###############################################################################
## Reading the data
###############################################################################
marketing = pd.read_csv('marketing_dataset_2.csv', index_col=0,
                        parse_dates=['date_served', 'date_subscribed', 'date_canceled'],
                        dtype={'converted': bool,
                               'is_retained': bool})

###############################################################################
## Main part of the code
###############################################################################
def Building_functions_to_automate_analysis():
    print("****************************************************")
    topic = "1. Building functions to automate analysis"; print("** %s" % topic)
    print("****************************************************")
    topic = "2. Building a conversion function"; print("** %s" % topic)
    
    print('---------------------------------------------Convertion rate by channel')
    col_groupby = ['date_served', 'subscribing_channel']
    daily_conversion = conversion_rate(df=marketing, col_groupby=col_groupby)
    daily_conversion = daily_conversion.unstack(level=col_groupby[1])
    print(daily_conversion.head())
    
    y_label = 'Conversion rate (%)'
    #Plotting daily retention by channel, all in one
    plotting_all_in_one(daily_conversion, y_label=y_label, topic=topic)
    
    #Plotting daily retention by channel, one by one
    plotting_one_by_one(daily_conversion, y_label=y_label, topic=topic)
    
    print('---------------------------------------------Retention rate by channel')
    col_groupby = ['date_served', 'subscribing_channel']
    daily_retention = retention_rate(df=marketing, col_groupby=col_groupby)
    daily_retention = daily_retention.unstack(level=col_groupby[1])
    print(daily_retention.head())
    
    y_label = 'Retention rate (%)'
    #Plotting all in one
    plotting_all_in_one(daily_retention, y_label=y_label, topic=topic)
    
    #Plotting one by one
    plotting_one_by_one(daily_retention, y_label=y_label, topic=topic)
    
    
    
def Test_and_visualize_conversion_function():
    print("****************************************************")
    topic = "3. Test and visualize conversion function"; print("** %s" % topic)
    print("****************************************************")
    topic = "4. Plotting function"; print("** %s" % topic)
    print("****************************************************")
    topic = "5. Putting it all together"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Conversion rate by age_group')
    col_groupby = ['date_served', 'age_group']
    daily_conversion = conversion_rate(df=marketing, col_groupby=col_groupby)
    daily_conversion = daily_conversion.unstack(level=col_groupby[1])
    print(daily_conversion.head())
    
    y_label = 'Conversion rate (%)'
    #Plotting all in one
    plotting_all_in_one(daily_conversion, y_label=y_label, topic=topic)
    
    #Plotting one by one
    plotting_one_by_one(daily_conversion, y_label=y_label, topic=topic, 
                        ncols=3, wspace=.4, hspace=1.4,)
    
    
    
    
    
def Identifying_inconsistencies():
    print("****************************************************")
    topic = "6. Identifying inconsistencies"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Retention rate by Day of the week')
    col_groupby = ['DoW']
    daily_retention = retention_rate(df=marketing, col_groupby=col_groupby)
    daily_retention = pd.DataFrame(daily_retention)
    print(daily_retention.head())
    
    y_label = 'Retention rate (%)'
    #Plotting all in one
    plotting_all_in_one(daily_retention, y_label=y_label, topic=topic)    
    
    
    
def House_ads_conversion_rate():
    print("****************************************************")
    topic = "7. House ads conversion rate"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Conversion rate by Marketing channel')
    col_groupby = ['date_served', 'marketing_channel']
    # Calculate conversion rate by date served and channel
    daily_conv_channel = conversion_rate(marketing, col_groupby)
    # Unstack daily_conv_channel and convert it to a DataFrame
    daily_conv_channel = pd.DataFrame(daily_conv_channel.unstack(level = col_groupby[1]))
    print(daily_conv_channel.head())
    
    y_label = 'Conversion rate (%)'
    #Plotting all in one
    plotting_all_in_one(daily_conv_channel, y_label=y_label, topic=topic)
    
    #Plotting one by one
    plotting_one_by_one(daily_conv_channel, y_label=y_label, topic=topic, 
                        ncols=3, wspace=.4, hspace=1.4)
    
    
def Analyzing_House_ads_conversion_rate():
    print("****************************************************")
    topic = "8. Analyzing House ads conversion rate"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Conversion rate by DoW served and marketing channel')
    # Add day of week column to marketing
    marketing['DoW_served'] = marketing.date_served.dt.dayofweek
    
    col_groupby = ['DoW_served', 'marketing_channel']
    # Calculate conversion rate by day of week
    DoW_conversion = conversion_rate(marketing, col_groupby)
    
    # Unstack channels
    DoW_df = DoW_conversion.unstack(level = col_groupby[1])

    # Plot conversion rate by day of week
    y_label = 'Conversion rate (%)'
    plotting_all_in_one(DoW_df, y_label=y_label, topic=topic)
    
    
    
def House_ads_conversion_by_language():
    print("****************************************************")
    topic = "9. House ads conversion by language"; print("** %s" % topic)
    print("****************************************************")
    
    #We are going to analize the Email Channel, different to the lesson.
    print('---------------------------------------------Conversion rate in email channel')
    # Isolate the rows where marketing channel is House Ads
    house_ads = marketing[marketing.marketing_channel=='House Ads'].copy()
    print(f"house_ads [shape: {house_ads.shape}] - Head: \n{house_ads.head()}")
    
    col_groupby = ['date_served', 'language_displayed']
    # Calculate conversion rate by date served and channel
    daily_conv_channel = conversion_rate(house_ads, col_groupby)
    print(f"\n\ndaily_conv_channel [shape: {daily_conv_channel.shape}] - Head: \n{daily_conv_channel.head()}")
    # Unstack daily_conv_channel and convert it to a DataFrame
    daily_conv_channel = pd.DataFrame(daily_conv_channel.unstack(level = col_groupby[1]))
    print(f"\n\nunstacked daily_conv_channel [shape: {daily_conv_channel.shape}] - Head: \n{daily_conv_channel.head()}")
    
    y_label = 'Conversion rate (%)'
    #Plotting all in one
    plotting_all_in_one(daily_conv_channel, y_label=y_label, topic=topic)
    
    #Plotting one by one
    plotting_one_by_one(daily_conv_channel, y_label=y_label, topic=topic)
    
 
    
    topic = "10. Creating a DataFrame for house ads"; print("\n\n** %s" % topic)
    print('---------------------------------------------Is correct language')
    # Add the new column is_correct_lang
    #house_ads['is_correct_lang'] = np.where(house_ads['language_displayed'] == house_ads['language_preferred'], 'Yes', 'No')
    
    col_groupby = ['date_served', 'is_correct_lang']
    # Groupby date_served and correct_language
    language_check = house_ads.groupby(col_groupby).user_id.count()
    
    # Unstack language_check and fill missing values with 0's
    language_check_df = language_check.unstack(level=col_groupby[1]).fillna(0)
    print(f"language_check_df [shape: {language_check_df.shape}] - Head: \n{language_check_df.head()}")
    
    
    
    topic = "11. Confirming house ads error"; print("\n\n** %s" % topic)
    print('---------------------------------------------PCT of correct language')
    # Divide the count where language is correct by the row sum
    language_check_df['pct'] = language_check_df.Yes / language_check_df.sum(axis='columns')
    print(f"language_check_df [shape: {language_check_df.shape}] - Head: \n{language_check_df.head()}")
    
    # Plot and show your results
    fig, ax = plt.subplots(figsize=figsize)
    language_check_df['pct'].plot(ax=ax)
    #ax.plot(language_check_df.index, language_check_df.pct)
    ax.set_xlabel('Date')
    ax.set_ylabel('PCT of Correct Language')
    ax.set_title('Correct Language Served in House Ads', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    


    topic = "12. Resolving inconsistencies"; print("\n\n** %s" % topic)
    topic = "13. Setting up conversion indexes"; print("** %s" % topic)
    
    print('---------------------------------------------Assessing impact')
    # Calculate pre-error conversion rate
    # Bug arose sometime around '2018-01-11'
    house_ads_no_bug = house_ads[house_ads['date_served'] < '2018-01-11']
    lang_conv = conversion_rate(house_ads_no_bug, ['language_displayed'])
    print(f'Conversion rate prior 2018-01-11: \n{lang_conv}')
    
    print('---------------------------------------------Index other lang.conver.rate against English')
    spanish_index = lang_conv['Spanish']/lang_conv['English']
    arabic_index = lang_conv['Arabic']/lang_conv['English']
    german_index = lang_conv['German']/lang_conv['English']
    
    print("Spanish index:", spanish_index, '(times English rate)')
    print("Arabic index:", arabic_index, '(times English rate)')
    print("German index:", german_index, '(times English rate)')
    
    
    topic = "14. Analyzing user preferences"; print("\n\n** %s" % topic)
    print('---------------------------------------------Daily conversion')
    # Create actual conversion DataFrame
    language_conversion = house_ads.groupby(['date_served', 'language_preferred']).agg({'user_id':'nunique','converted':'sum'})
    print(f"language_conversion [shape: {language_conversion.shape}] - Head: \n{language_conversion.head()}")
    expected_conversion = language_conversion.unstack(level='language_preferred')
    print(f"\n\nexpected_conversion [shape: {expected_conversion.shape}] - Head: \n{expected_conversion.head()}")
    
    
    topic = "15. Creating a DataFrame based on indexes"; print("\n\n** %s" % topic)
    print('---------------------------------------------Create English conversion rate column')
    # Create English conversion rate column for affected period
    expected_conversion['english_conv_rate'] = expected_conversion.loc['2018-01-11':'2018-01-31'][('converted','English')]
    print(f"expected_conversion [shape: {expected_conversion.shape}] - A segment: \n{expected_conversion.loc['2018-01-09':'2018-01-13']}")
    
    print('---------------------------------------------Calculating daily expected conversion rate')
    # Create expected conversion rates for each language
    expected_conversion['expected_spanish_rate'] = expected_conversion['english_conv_rate']*spanish_index
    expected_conversion['expected_arabic_rate'] = expected_conversion['english_conv_rate']*arabic_index
    expected_conversion['expected_german_rate'] = expected_conversion['english_conv_rate']*german_index
    print(f"expected_conversion [shape: {expected_conversion.shape}] - A segment: \n{expected_conversion.loc['2018-01-09':'2018-01-13']}")
    
    print('---------------------------------------------Calculating daily expected conversions')
    # Multiply number of users by the expected conversion rate
    expected_conversion['expected_spanish_conv'] = expected_conversion['expected_spanish_rate']*expected_conversion[('user_id', 'Spanish')]/100
    expected_conversion['expected_arabic_conv']  = expected_conversion['expected_arabic_rate'] *expected_conversion[('user_id', 'Arabic')]/100
    expected_conversion['expected_german_conv']  = expected_conversion['expected_german_rate'] *expected_conversion[('user_id', 'German')]/100
    print(f"expected_conversion [shape: {expected_conversion.shape}] - A segment: \n{expected_conversion.loc['2018-01-09':'2018-01-13']}")
    
    
    topic = "16. Assessing bug impact"; print("\n\n** %s" % topic)
    
    print('---------------------------------------------Isolate the dates with problems')
    # Use .loc to slice only the relevant dates
    expected_conversion = expected_conversion.loc['2018-01-11':'2018-01-31']

    print('---------------------------------------------How many subscribers have we lost')
    # Sum expected subscribers for each language
    expected_subs = expected_conversion.expected_spanish_conv.sum() + expected_conversion.expected_arabic_conv.sum() + expected_conversion.expected_german_conv.sum()
    
    # Calculate how many subscribers we actually got
    actual_subs = expected_conversion[('converted','Spanish')].sum() + expected_conversion[('converted','Arabic')].sum() + expected_conversion[('converted','German')].sum()

    # Subtract how many subscribers we got despite the bug
    lost_subs = expected_subs - actual_subs
    print(lost_subs)
    
        
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Building_functions_to_automate_analysis()
    Test_and_visualize_conversion_function()
    
    Identifying_inconsistencies()
    House_ads_conversion_rate()
    Analyzing_House_ads_conversion_rate()
    
    House_ads_conversion_by_language()
        
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    plt.style.use('default')