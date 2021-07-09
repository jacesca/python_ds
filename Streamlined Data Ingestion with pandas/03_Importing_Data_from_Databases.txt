# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: Importing Data from Databases
    Combine pandas with the powers of SQL to find out just how many problems 
    New Yorkers have with their housing. This chapter features introductory 
    SQL topics like WHERE clauses, aggregate functions, and basic joins.
Source: https://learn.datacamp.com/courses/streamlined-data-ingestion-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
# Load pandas and sqlalchemy's create_engine
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine



###############################################################################
## Preparing the environment
###############################################################################
# Global parameters
string_URL = 'sqlite:///NYC_weather.db'

# Create database engine to manage connections
engine = create_engine(string_URL)

# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 8})
suptitle_param = dict(color='darkblue', fontsize=9)
title_param    = {'color': 'darkred', 'fontsize': 10}



###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_databases():
    print("****************************************************")
    topic = "1. Introduction to databases"; print("** %s" % topic)
    print("****************************************************")
    topic = "2. Connect to a database"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Exploring what is in the database')
    tables = engine.table_names()
    print(tables)
    
    print('---------------------------------------------Exploring tables')
    for table_data in tables:
        # Load entire table by table name
        df = pd.read_sql(table_data, engine)
        print(f"Head of {table_data}:\n{df.head()}")
    
    print('---------------------------------------------Using SQL sentences')
    # Load entire weather table with SQL
    weather = pd.read_sql("SELECT * FROM weather", engine)
    print(weather.head())
    
    
    
def Load_entire_tables():
    print("****************************************************")
    topic = "3. Load entire tables"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Reading hpd311calls')
    # Load hpd311calls without any SQL
    hpd_calls = pd.read_sql('hpd311calls', engine)
    # View the first few rows of data
    print(hpd_calls.head())
    
    
    print('---------------------------------------------Reading weather')
    # Create a SQL query to load the entire weather table
    query = """
    SELECT * 
    FROM   weather;
    """
    # Load weather with the SQL query
    weather = pd.read_sql(query, engine)
    # View the first few rows of data
    print(weather.head())

    
    
def Refining_imports_with_SQL_queries():
    print("****************************************************")
    topic = "4. Refining imports with SQL queries"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------SELECTing Columns')
    # Create a SQL query
    query = """
    SELECT date, tmax, tmin
    FROM   weather;
    """
    # Load weather with the SQL query
    df = pd.read_sql(query, engine)
    # View the first few rows of data
    print(df.head())
    
    print('---------------------------------------------Filtering by Numbers')
     # Create a SQL query
    query = """
    SELECT *
    FROM   weather
    WHERE  tmax > 32;
    """
    # Load weather with the SQL query
    df = pd.read_sql(query, engine)
    # View the first few rows of data
    print(df.head())
    
    print('---------------------------------------------Filtering Text')
     # Create a SQL query
    query = """
    /* Get records about incidents in Brooklyn */
    SELECT *
    FROM   hpd311calls
    WHERE  borough = 'BROOKLYN';
    """
    # Load weather with the SQL query
    df = pd.read_sql(query, engine)
    # View the first few rows of data
    print(df.head())
    print("Distinct values in borough column: ", df.borough.unique())
    
    print('---------------------------------------------Combining Conditions: AND')
     # Create a SQL query
    query = """SELECT *
               FROM   hpd311calls
               WHERE  borough = 'BRONX'
                      AND complaint_type = 'PLUMBING';"""
    # Load weather with the SQL query
    df = pd.read_sql(query, engine)
    # Check record count
    print("Shape: ", df.shape)
    
    print('---------------------------------------------Combining Conditions: OR')
     # Create a SQL query
    query = """SELECT *
               FROM   hpd311calls
               WHERE  complaint_type = 'WATER LEAK'
                      OR complaint_type = 'PLUMBING';"""
    # Load weather with the SQL query
    df = pd.read_sql(query, engine)
    # Check record count
    print("Shape: ", df.shape)
    
    
    
def Selecting_columns_with_SQL():
    print("****************************************************")
    topic = "5. Selecting columns with SQL"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Passing a query')
    # Write query to get date, tmax, and tmin from weather
    query = """
    SELECT date, 
           tmax, 
           tmin
    FROM   weather;
    """
    # Make a data frame by passing query and engine to read_sql()
    temperatures = pd.read_sql(query, engine)
    # View the resulting data frame
    print(temperatures)
    
    
    
def Selecting_rows():
    print("****************************************************")
    topic = "6. Selecting rows"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Extracting the data')
    # Create query to get hpd311calls records about safety
    query = """
    SELECT *
    FROM   hpd311calls
    WHERE  complaint_type='SAFETY';
    """
    # Query the database and assign result to safety_calls
    safety_calls = pd.read_sql(query, engine)
    
    print('---------------------------------------------Summarizing the data')
    call_counts = safety_calls.groupby('borough').unique_key.count()
    print(call_counts)
    
    print('---------------------------------------------Plotting the information')
    # Graph the number of safety calls by borough
    fig, ax = plt.subplots()
    call_counts.plot.barh(ax=ax)
    ax.set_xlabel('Frequency')
    ax.set_title('Complaints by City', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.2, bottom=None, right=.95, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Filtering_on_multiple_conditions():
    print("****************************************************")
    topic = "7. Filtering on multiple conditions"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Selecting base on conditions')
    # Create query for records with max temps <= 32 or snow >= 1
    query = """
    SELECT *
    FROM   weather
    WHERE  tmax<=32
           OR snow>=1;
    """
    # Query database and assign result to wintry_days
    wintry_days = pd.read_sql(query, engine)
    # View summary stats about the temperatures
    print(wintry_days.describe())
        
        
def More_complex_SQL_queries():
    print("****************************************************")
    topic = "8. More complex SQL queries"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------GROUP BY clause')
    # Write query to get plumbing call counts by borough
    query = """SELECT   borough, COUNT(*)
               FROM     hpd311calls
               WHERE    complaint_type = 'PLUMBING'
               GROUP BY borough;"""
    # Query databse and create data frame
    plumbing_call_counts = pd.read_sql(query, engine)
    print(plumbing_call_counts)
    
    
    
def Getting_distinct_values():
    print("****************************************************")
    topic = "9. Getting distinct values"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------DISTINCT clause')
    # Create query for unique combinations of borough and complaint_type
    query = """
    SELECT DISTINCT borough, 
           complaint_type
    FROM   hpd311calls;
    """
    # Load results of query to a data frame
    issues_and_boros = pd.read_sql(query, engine)
    # Check assumption about issues and boroughs
    print(issues_and_boros)
    
    
    
def Counting_in_groups():
    print("****************************************************")
    topic = "10. Counting in groups"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Extracting and summarizing the data')
    # Create query to get call counts by complaint_type
    query = """
    SELECT   complaint_type, 
             COUNT(*) as issues_counts
    FROM     hpd311calls
    GROUP BY complaint_type;
    """
    # Create data frame of call counts by issue
    calls_by_issue = pd.read_sql(query, engine)

    print('---------------------------------------------Plotting the information')
    # Graph the number of calls for each housing issue
    fig, ax = plt.subplots()
    calls_by_issue.plot.barh(x="complaint_type", ax=ax)
    ax.set_xlabel('Frequency')
    ax.set_title('Complaints by Issue', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.3, bottom=None, right=.95, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
def Working_with_aggregate_functions():
    print("****************************************************")
    topic = "11. Working with aggregate functions"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------GROUP BY clause')
    # Create query to get temperature and precipitation by month
    query = """
    SELECT   month, 
             MAX(tmax), MIN(tmin), SUM(prcp)
    FROM     weather 
    GROUP BY month;
    """
    # Get data frame of monthly weather stats
    weather_by_month = pd.read_sql(query, engine)
    # View weather stats by month
    print(weather_by_month)
    
    
    
def Loading_multiple_tables_with_joins():
    print("****************************************************")
    topic = "12. Loading multiple tables with joins"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Joining Tables')
    query = """
    SELECT *
    FROM   hpd311calls
    JOIN   weather
           ON hpd311calls.created_date = weather.date;
    """
    # Get data frame of monthly weather stats
    df = pd.read_sql(query, engine)
    # View shape of the data
    print('shape: ', df.shape)
    
    print('---------------------------------------------Joining and Filtering')
    query = """
    /* Get only heat/hot water calls and join in weather data */
    SELECT *
    FROM   hpd311calls
    JOIN   weather
           ON hpd311calls.created_date = weather.date
    WHERE  hpd311calls.complaint_type = 'HEAT/HOT WATER';
    """
    # Get data frame of monthly weather stats
    df = pd.read_sql(query, engine)
    # View shape of the data
    print('shape: ', df.shape)
    
    print('---------------------------------------------(1) Joining and Aggregating')
    query = """
    /* Get call counts by borough */
    SELECT   hpd311calls.borough, COUNT(*)
    FROM     hpd311calls
    GROUP BY hpd311calls.borough;
    """
    # Get data frame of monthly weather stats
    df = pd.read_sql(query, engine)
    # View shape of the data
    print(df)
    
    print('---------------------------------------------(2) Joining and Aggregating')
    query = """
    /* Get call counts by borough and join in population and housing counts */
    SELECT   hpd311calls.borough, 
             COUNT(*), 
             boro_census.total_population, boro_census.housing_units
    FROM     hpd311calls
    JOIN     boro_census
             ON hpd311calls.borough = boro_census.borough
    GROUP BY hpd311calls.borough
    """
    # Get data frame of monthly weather stats
    df = pd.read_sql(query, engine)
    # View shape of the data
    print(df)
    
    
def Joining_tables():
    print("****************************************************")
    topic = "13. Joining tables"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------JOIN clause')
    # Query to join weather to call records by date columns
    query = """
    SELECT * 
    FROM   hpd311calls
    JOIN   weather 
           ON hpd311calls.created_date = weather.date;
    """
    # Create data frame of joined tables
    calls_with_weather = pd.read_sql(query, engine)
    # View the data frame to make sure all columns were joined
    print(calls_with_weather.head())
    
    
def Joining_and_filtering():
    print("****************************************************")
    topic = "14. Joining and filtering"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Joining')
    # Query to get hpd311calls and precipitation values
    query = """
    SELECT hpd311calls.*, 
           weather.prcp 
       FROM hpd311calls
      JOIN weather
      ON hpd311calls.created_date = weather.date;"""
    # Load query results into the leak_calls data frame
    leak_calls = pd.read_sql(query, engine)  
    # View the data frame
    print(leak_calls.head())
    
    print('---------------------------------------------Filtering')
    # Query to get water leak calls and daily precipitation
    query = """
    SELECT  hpd311calls.*, weather.prcp
    FROM    hpd311calls
    JOIN    weather
            ON hpd311calls.created_date = weather.date
    WHERE   hpd311calls.complaint_type = 'WATER LEAK';"""
    # Load query results into the leak_calls data frame
    leak_calls = pd.read_sql(query, engine)  
    # View the data frame
    print(leak_calls.head())
    
    
    
def Joining_filtering_and_aggregating():
    print("****************************************************")
    topic = "15. Joining, filtering, and aggregating"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------GROUP BY clause')
    # Query to get heat/hot water call counts by created_date
    query = """
    SELECT   hpd311calls.created_date, 
             COUNT(*)
    FROM     hpd311calls 
    WHERE    hpd311calls.complaint_type = 'HEAT/HOT WATER'
    GROUP BY hpd311calls.created_date;
    """
    # Query database and save results as df
    df = pd.read_sql(query, engine)
    # View first 5 records
    print(df.head())
    
    print('---------------------------------------------JOIN clause')
    # Modify query to join tmax and tmin from weather by date
    query = """
    SELECT   hpd311calls.created_date, 
             COUNT(*), 
             weather.tmax, weather.tmin
    FROM     hpd311calls 
    JOIN     weather
             ON hpd311calls.created_date = weather.date
    WHERE    hpd311calls.complaint_type = 'HEAT/HOT WATER' 
    GROUP BY hpd311calls.created_date;
    """
    # Query database and save results as df
    df = pd.read_sql(query, engine)
    # View first 5 records
    print(df.head())

    
        
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_databases()
    Load_entire_tables()
    Refining_imports_with_SQL_queries()
    Selecting_columns_with_SQL()
    Selecting_rows()
    Filtering_on_multiple_conditions()
    More_complex_SQL_queries()
    Getting_distinct_values()
    Counting_in_groups()
    Working_with_aggregate_functions()
    Loading_multiple_tables_with_joins()
    Joining_tables()
    Joining_and_filtering()
    Joining_filtering_and_aggregating()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()