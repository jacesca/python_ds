# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Importing JSON Data and Working with APIs
    Learn how to work with JSON data and web APIs by exploring a public dataset 
    and getting cafe recommendations from Yelp. End by learning some techniques 
    to combine datasets once they have been loaded into data frames.
Source: https://learn.datacamp.com/courses/streamlined-data-ingestion-with-pandas
Data source:
    https://data.cityofnewyork.us/Social-Services/DHS-Daily-Report/k46n-sa2m
    https://data.cityofnewyork.us/resource/k46n-sa2m.json
    https://datahub.io/JohnSnowLabs/new-york-city-leading-causes-of-death
YEP APPI:
    https://www.yelp.com/developers/documentation/v3/authentication
    https://www.yelp.com/developers/documentation/v3/get_started
    https://www.yelp.com/developers/documentation/v3/business_search
"""
###############################################################################
## Importing libraries
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import requests
import pprint

from sqlalchemy import create_engine



###############################################################################
## Preparing the environment
###############################################################################
# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 8})
suptitle_param = dict(color='darkblue', fontsize=9)
title_param    = {'color': 'darkred', 'fontsize': 10}

# Global variables
string_URL = 'sqlite:///NYC_weather.db'

# Create database engine to manage connections
engine = create_engine(string_URL)

# Access to yep appi
#with open('ClientID.txt','r') as f: yelp_ClientId = f.read()
with open('APIKey.txt','r') as f: yelp_APIKey = f.read()
yelp_api_url = "https://api.yelp.com/v3/businesses/search"


###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_JSON():
    print("****************************************************")
    topic = "1. Introduction to JSON"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Reading json file')
    death_causes = pd.read_json("new-york-city-leading-causes-of-death-csv_json.json",
                                orient="records")
    print('Shape: ', death_causes.shape)
    print(death_causes.head())
    print("Columns: ", death_causes.columns)
    
    print('---------------------------------------------Saving to json file')
    death_causes.to_json('nyc_death_causes.json', orient='split')
    
    print('---------------------------------------------Specifying Orientation')
    death_causes = pd.read_json("nyc_death_causes.json", 
                                orient="split")
    print(death_causes.head())
    
    
    
def Load_JSON_data():
    print("****************************************************")
    topic = "2. Load JSON data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Reading from file')
    # Load the daily report to a data frame
    pop_in_shelters = pd.read_json('dhs_daily_report.json')
    #https://data.cityofnewyork.us/resource/k46n-sa2m.json
    # View summary stats about pop_in_shelters
    print(pop_in_shelters.describe())
    
    print('---------------------------------------------Saving  with different orientation')
    pop_in_shelters.to_json('dhs_report_reformatted.json', orient='split')
    
    print('---------------------------------------------Reading from URL')
    # Load the daily report to a data frame
    pop_in_shelters = pd.read_json('https://data.cityofnewyork.us/resource/k46n-sa2m.json')
    # View summary stats about pop_in_shelters
    print(pop_in_shelters.describe())
    
    
    
def Work_with_JSON_orientations():
    print("****************************************************")
    topic = "3. Work with JSON orientations"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Importing split json without orientation')
    try:
        # Load the JSON without keyword arguments
        df = pd.read_json('dhs_report_reformatted.json')
        print(df.head())
    except ValueError:
        print("pandas could not parse the JSON file without orientation key.")
    
    print('---------------------------------------------Trying again with orientation')
    try:
        # Load the JSON with orient specified
        df = pd.read_json("dhs_report_reformatted.json", orient='split')
        print(df.head())
    except ValueError:
        print("pandas could not parse the JSON (with orientation key this time!).")
    
    print('---------------------------------------------Plotting the information')
    # Plot total population in shelters over time
    df["date_of_census"] = pd.to_datetime(df["date_of_census"])
    
    fig, ax = plt.subplots()
    df.plot(x="date_of_census", y="total_individuals_in_shelter", ax=ax)
    ax.set_ylabel('Number of individuals')
    ax.set_title('Total individuals in Shelters per Day', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.2, bottom=None, right=.95, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    print('---------------------------------------------Loading with transposed column')
    # Load the JSON with transposed column and index names.
    df = pd.read_json("dhs_report_reformatted.json", orient='index')
    print(df.head())
    
    
    
def Introduction_to_APIs():
    print("****************************************************")
    topic = "4. Introduction to APIs"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Setting the configuration')
    pd.set_option("display.max_columns",20)
    
    print('---------------------------------------------Making Requests')
    api_url = yelp_api_url
    api_key = yelp_APIKey
    
    # Set up parameter dictionary according to documentation
    params = {"term": "bookstore",
              "location": "San Francisco"}
    
    # Set up header dictionary w/ API key according to documentation
    headers = {"Authorization": "Bearer {}".format(api_key)}
    
    # Call the API
    response = requests.get(api_url, params=params, headers=headers)
    
    print('---------------------------------------------Parsing Responses')
    # Isolate the JSON data from the response object
    data = response.json()
    keys_to_explore = data.keys()
    print(f"In the YELP API, we get a {type(data)} of size {len(data)} with keys: {list(keys_to_explore)}.\n")
    for key_name in keys_to_explore:
        if type(data[key_name]) == list:
            print(f"data['{key_name}'] is a {type(data[key_name])} of size {len(data[key_name])}\.n")
            print(f"First element of data['{key_name}']: \n{data[key_name][0]}.\n")
        else:
            print(f"data['{key_name}'] = {data[key_name]}.")
    
    print('---------------------------------------------Loading into DataFrame')
    # Load businesses data to a data frame
    bookstores = pd.DataFrame(data["businesses"])
    print(bookstores.head())
    
    print('---------------------------------------------Returning to default configuration')
    pd.reset_option("display.max_columns")
    
    
def Get_data_from_an_API():
    print("****************************************************")
    topic = "5. Get data from an API"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Setting the header and params')
    api_url = yelp_api_url
    api_key = yelp_APIKey
    
    # Set up parameter dictionary according to documentation
    params = {'location': 'NYC', 'term': 'cafe'}
    
    # Set up header dictionary w/ API key according to documentation
    headers = {"Authorization": "Bearer {}".format(api_key)}
    
    print('---------------------------------------------Parsing Responses')
    # Get data about NYC cafes from the Yelp API
    response = requests.get(api_url, headers=headers, params=params)
    
    # Extract JSON data from the response
    data = response.json()
    
    print('---------------------------------------------Exploring the DataFrame')
    # Load data to a data frame
    cafes = pd.DataFrame(data["businesses"])
    
    # View the data's dtypes
    print('Shape: ', cafes.shape)
    print(cafes.dtypes)
    
    
    
def Set_API_parameters():
    print("****************************************************")
    topic = "6. Set API parameters"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Setting the configuration')
    pd.set_option("display.max_columns",20)
    
    print('---------------------------------------------Setting the header and params')
    api_url = yelp_api_url
    api_key = yelp_APIKey
    
    # Set up parameter dictionary according to documentation
    parameters = {'term':'cafe', 'location':'NYC'}
    
    # Set up header dictionary w/ API key according to documentation
    headers = {"Authorization": "Bearer {}".format(api_key)}
    
    print('---------------------------------------------Parsing Responses')
    # Query the Yelp API with headers and params set
    response = requests.get(api_url, headers=headers, params=parameters)
    
    # Extract JSON data from response
    data = response.json()
    
    print('---------------------------------------------Reviewing the DataFrame')
    # Load "businesses" values to a data frame and print head
    cafes = pd.DataFrame(data['businesses'])
    print('Shape: ', cafes.shape)
    print(cafes.head(2))
    
    print('---------------------------------------------Returning to default configuration')
    pd.reset_option("display.max_columns")
    
    
    
def Set_request_headers():
    print("****************************************************")
    topic = "7. Set request headers"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Setting the header and params')
    api_url = yelp_api_url
    api_key = yelp_APIKey
    
    # Set up parameter dictionary according to documentation
    params = {'location': 'NYC', 'sort_by': 'rating', 'term': 'cafe'}
    
    # Set up header dictionary w/ API key according to documentation
    headers = {"Authorization": "Bearer {}".format(api_key)}
    
    print('---------------------------------------------Parsing Responses')
    # Query the Yelp API with headers and params set
    response = requests.get(api_url, headers=headers, params=params)
    # Extract JSON data from response
    data = response.json()
    
    print('---------------------------------------------NYC\'s cafe order by rates')
    # Load "businesses" values to a data frame and print names
    cafes = pd.DataFrame(data['businesses'])
    print(cafes.name)

    
    
def Working_with_nested_JSONs():
    print("****************************************************")
    topic = "8. Working with nested JSONs"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Setting the configuration')
    pd.set_option("display.max_columns",20)
    
    print('---------------------------------------------Setting the header and params')
    api_url = yelp_api_url
    api_key = yelp_APIKey
    # Set up header dictionary w/ API key according to documentation
    headers = {"Authorization": "Bearer {}".format(api_key)}
    # Set up parameter dictionary according to documentation
    params = {"term": "bookstore", "location": "San Francisco"} 
    
    print('---------------------------------------------Parsing Responses')
    # Query the Yelp API with headers and params set
    response = requests.get(api_url, headers=headers, params=params)
    # Extract JSON data from response
    data = response.json()
    
    print('---------------------------------------------Default json pandas importing')
    # Load to data frame
    bookstores = pd.DataFrame(data['businesses'])
    # Review the dataframe
    print('Shape: ', bookstores.shape)
    print(f'All columns: \n{bookstores.head(2)}\n\n')
    columns_to_analize = ['categories', 'coordinates', 'location']
    print(f'In detail: \n{bookstores[columns_to_analize].head()}')
    
    print('---------------------------------------------First approach')
    print("*** Loading Nested JSON Data ***")
    # Flatten data and load to data frame, with _ separators
    bookstores = pd.json_normalize(data["businesses"], sep="_")
    pprint.pprint(list(bookstores.columns))
    print('Shape: ', bookstores.shape)
    print(f'\n\nAll columns: \n{bookstores.head(2)}\n\n')
    print(f"But not changes in: \n{bookstores[['categories']].head()}")
    
    print('---------------------------------------------Second approach')
    print("*** Deeply Nested Data ***")
    # Flatten categories data, bring in business details
    bookstores = pd.json_normalize(data["businesses"],
                                   sep="_",
                                   record_path="categories",
                                   meta=["name", "alias", "rating",
                                         ["coordinates", "latitude"],
                                         ["coordinates", "longitude"]],
                                   meta_prefix="biz_")
    # Review the dataframe
    print('Shape: ', bookstores.shape)
    print(f'Head: \n{bookstores.head(2)}\n\n')
    
    print('---------------------------------------------Loading all')
    print("*** Deeply Nested Data ***")
    # Flatten categories data, bring in business details
    bookstores = pd.json_normalize(data["businesses"],
                                   sep="_",
                                   record_path=["categories"],
                                   meta=['id', 'alias', 'name', 'image_url', 
                                         'is_closed', 'url', 'review_count', 'rating', 
                                         'price', 'phone', 'display_phone', 'distance', 
                                         ['coordinates', 'latitude'],
                                         ['coordinates', 'longitude'],
                                         ['location', 'address1'],
                                         ['location', 'address2'],
                                         ['location', 'address3'],
                                         ['location', 'city'],
                                         ['location', 'zip_code'],
                                         ['location', 'country'],
                                         ['location', 'state'],
                                         ['location', 'display_address']],
                                   meta_prefix="biz_")
    # Review the dataframe
    print('Shape: ', bookstores.shape)
    pprint.pprint(list(bookstores.columns))
    print(f'Upload everything: \n{bookstores.head(2)}\n\n')
    
    print('---------------------------------------------Returning to default configuration')
    pd.reset_option("display.max_columns")
    
    
    
def Flatten_nested_JSONs():
    print("****************************************************")
    topic = "9. Flatten nested JSONs"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Setting the configuration')
    pd.set_option("display.max_columns",20)
    
    print('---------------------------------------------Isolate the JSON data from the API')
    api_url = yelp_api_url; api_key = yelp_APIKey;
    
    parameters = {'term':'cafe', 'location':'NYC'} # Set up parameter dictionary according to documentation
    headers = {"Authorization": "Bearer {}".format(api_key)} # Set up header dictionary w/ API key according to documentation
    response = requests.get(api_url, headers=headers, params=parameters) # Query the Yelp API with headers and params set
    data = response.json() # Extract JSON data from response
    
    print('---------------------------------------------Getting the dataframe')
    # Flatten business data into a data frame, replace separator
    cafes = pd.json_normalize(data["businesses"], sep='_')
    # View data
    print(cafes.head())
    
    print('---------------------------------------------Returning to default configuration')
    pd.reset_option("display.max_columns")
    
    print('---------------------------------------------Returning to default configuration')
    pd.reset_option("display.max_columns")
    
    
    
def Handle_deeply_nested_data():
    print("****************************************************")
    topic = "10. Handle deeply nested data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Setting the configuration')
    pd.set_option("display.max_columns",20)
    
    print('---------------------------------------------Isolate the JSON data from the API')
    api_url = yelp_api_url; api_key = yelp_APIKey;
    
    parameters = {'term':'cafe', 'location':'NYC'} # Set up parameter dictionary according to documentation
    headers = {"Authorization": "Bearer {}".format(api_key)} # Set up header dictionary w/ API key according to documentation
    response = requests.get(api_url, headers=headers, params=parameters) # Query the Yelp API with headers and params set
    data = response.json() # Extract JSON data from response
    
    print('---------------------------------------------Getting the dataframe')
    # Load other business attributes and set meta prefix
    flat_cafes = pd.json_normalize(data["businesses"],
                                   sep="_",
                                   record_path="categories",
                                   meta=['name', 
                                         'alias',  
                                         'rating',
                                         ['coordinates', 'latitude'], 
                                         ['coordinates', 'longitude']],
                                   meta_prefix='biz_')
    # View the data
    print(flat_cafes.head())
    
    print('---------------------------------------------Returning to default configuration')
    pd.reset_option("display.max_columns")
    
    
    
def Combining_multiple_datasets():
    print("****************************************************")
    topic = "11. Combining multiple datasets"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Append: Working with json and api\'s')
    # Set up header and parameters
    api_url = yelp_api_url; api_key = yelp_APIKey;
    headers = {"Authorization": "Bearer {}".format(api_key)}
    params = {"term": "bookstore", "location": "San Francisco"} 
    
    # Get first 20 bookstore results
    first_results = requests.get(api_url, headers=headers, params=params).json()
    first_20_bookstores = pd.json_normalize(first_results["businesses"], sep="_")
    print("First dataset : ", first_20_bookstores.shape)
    
    # Get the next 20 bookstores
    params["offset"] = 20
    next_results = requests.get(api_url, headers=headers, params=params).json()
    next_20_bookstores = pd.json_normalize(next_results["businesses"], sep="_")
    print("Second dataset: ", next_20_bookstores.shape)
    
    # Put bookstore datasets together, renumber rows
    bookstores = first_20_bookstores.append(next_20_bookstores, ignore_index=True)
    print("Total dataset : ", bookstores.shape)
    
    print('---------------------------------------------Merge: Working with SQL')
    # Load entire table by table name
    weather = pd.read_sql('weather', engine)
    print("Weather dataset    : ", weather.shape)
        
    query = """
    SELECT   created_date, 
             COUNT(*) as issues_counts
    FROM     hpd311calls
    GROUP BY created_date;
    """
    call_counts = pd.read_sql(query, engine)
    print("Call_counts dataset: ", call_counts.shape)
    
    # Merge weather into call counts on date columns
    merged = call_counts.merge(weather, left_on="created_date", right_on="date")
    print("Merged dataset     : ", merged.shape)
    
    
    
def Append_data_frames():
    print("****************************************************")
    topic = "12. Append data frames"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Isolate the JSON data from the API')
    api_url = yelp_api_url; api_key = yelp_APIKey;
    
    params = {'term':'cafe', 'location':'NYC', "sort_by": "rating", "limit": 50} # Set up parameter dictionary according to documentation
    headers = {"Authorization": "Bearer {}".format(api_key)} # Set up header dictionary w/ API key according to documentation
    response = requests.get(api_url, headers=headers, params=params) # Query the Yelp API with headers and params set
    top_50_cafes = pd.json_normalize(response.json()["businesses"], sep='_') # Extract JSON data from response
    
    print('---------------------------------------------Json & Append')
    # Add an offset parameter to get cafes 51-100
    params = {"term": "cafe", "location": "NYC", "sort_by": "rating", "limit": 50, 'offset': 50}
    result = requests.get(api_url, headers=headers, params=params)
    next_50_cafes = pd.json_normalize(result.json()["businesses"], sep='_')
    
    # Append the results, setting ignore_index to renumber rows
    cafes = top_50_cafes.append(next_50_cafes, ignore_index=True)
    
    # Print shape of cafes
    print(cafes.shape)
    return(cafes)
    
    
    
def Merge_data_frames(cafes):
    print("****************************************************")
    topic = "13. Merge data frames"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Read csv files')
    crosswalk = pd.read_csv('US_PUMA.csv', sep=';')
    crosswalk['zipcode'] = crosswalk.zipcode.astype(str)
    pop_data = pd.read_csv('US_census.csv', sep=';')
    
    print('---------------------------------------------Explore datasets')
    print("cafes columns: ", list(cafes))
    print("shape: ", cafes.shape)
    print(cafes.head(2), '\n\n')
    
    print("crosswalk columns: ", list(crosswalk))
    print("shape: ", crosswalk.shape)
    print(crosswalk.head(2), '\n\n')
    
    print("cafes columns: ", list(pop_data))
    print("shape: ", pop_data.shape)
    print(pop_data.head(2), '\n\n')
    
    print('---------------------------------------------Merging cafees & crosswalk')
    # Merge crosswalk into cafes on their zip code fields
    cafes_with_pumas = cafes.merge(crosswalk, left_on='location_zip_code', right_on='zipcode')
    print("shape: ", cafes_with_pumas.shape, '\n')
    
    print('---------------------------------------------Merging with pop_data')
    # Merge pop_data into cafes_with_pumas on puma field
    cafes_with_pop = cafes_with_pumas.merge(pop_data, on='puma')
    print("shape: ", cafes_with_pop.shape, '\n')
    
    print('---------------------------------------------Exploring the result')
    # View the data
    print(cafes_with_pop.head())
    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_JSON()
    Load_JSON_data()
    Work_with_JSON_orientations()
    
    Introduction_to_APIs()
    Get_data_from_an_API()
    Set_API_parameters()
    Set_request_headers()
    
    Working_with_nested_JSONs()
    Flatten_nested_JSONs()
    Handle_deeply_nested_data()
    Combining_multiple_datasets()
    cafes_top100 = Append_data_frames()
    Merge_data_frames(cafes_top100)
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    plt.style.use('default')