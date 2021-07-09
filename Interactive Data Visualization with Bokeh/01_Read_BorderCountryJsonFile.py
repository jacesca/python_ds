# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:52:26 2019

@author: jacqueline.cortez

Open a json file
Source of data: https://datahub.io/core/geo-countries#resource-geo-countries_zip  

This program read the coord of a country and display its shape graph.
"""
# Import packages
import pandas as pd
import numpy as np
#import tabula 
#import math
#import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
#import random
import json

#from math import radians
#from functools import reduce#import pandas as pd
#from pandas.api.types import CategoricalDtype #For categorical data
#from glob import glob
from bokeh.io import output_file, show
from bokeh.plotting import figure

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Reading the file...\n")

# Load JSON: json_data
file = 'countries.json'
with open(file,'r') as json_file:
    country_df = json.load(json_file)

country_df = country_df["features"]
country_df = pd.DataFrame(country_df)

# Getting data into columns from 'properties'
country_df['country'] = country_df["properties"].apply(lambda x: x["ADMIN"])
country_df['iso'] = country_df["properties"].apply(lambda x: x["ISO_A3"])

# Getting data into columns from 'geometry'
country_df['type'] = country_df["geometry"].apply(lambda x: x["type"])
country_df['coord'] = country_df["geometry"].apply(lambda x: x["coordinates"])
country_df['shape'] = country_df['coord'].apply(lambda x: np.array(x).shape)

# Setting the country index
#print("Unique country index: {}.\n".format(country_df["country"].value_counts()[0]==1)) #Validate the uniqueness of the country index
country_df.set_index("country", inplace=True)
country_df.sort_index(inplace=True)

# Drop unnecessary columns
country_df.drop(['geometry','properties'], axis='columns', inplace=True)

# Showing the database
while True:
    print("Countries in this database: \n{}.\n".format(country_df.index.values))

    print("*****   'exit' to quit.   *****")
    print("Enter the name of the country:")
    country = str(input()).strip()
    
    # Inicializa las variables
    #country = "El Salvador"
    lng_x = []
    lat_y =[]
    
    if country == 'exit':
        break
    elif country in country_df.index:
        #print("The country data:\n{}\n".format(country_df.loc[country]))
        
        if country_df.loc[country,'type'] == 'MultiPolygon':
            #coord = country_df.loc[country, 'coord'][2][0]
            coord = country_df.loc[country, 'coord']
            for location in coord:
                lng, lat = zip(*location[0])
                lng_x.append(lng)
                lat_y.append(lat)
        else:
            coord = country_df.loc[country, 'coord'][0]
            lng, lat = zip(*coord)
            lng_x.append(lng)
            lat_y.append(lat)
               
        #print ("Its coordinates are: \nLongitud : \n{}\n\nLatitud: \n{}\n".format(lng,lat))
    
        """ For one shape only
        if country_df.loc[country,'type'] == 'MultiPolygon':
            coord = country_df.loc[country, 'coord'][2][0]
        else:
            coord = country_df.loc[country, 'coord'][0]
               
        lng, lat = zip(*coord)
        #print ("Its coordinates are: \nLongitud : \n{}\n\nLatitud: \n{}\n".format(lng,lat))
        """
        
        # Begining the graph
        #"""
        # Create a figure 
        p = figure(plot_width=750, plot_height=550)
        
        # Create a list of az_lons, co_lons, nm_lons and ut_lons: x
        x_coor = lng_x
        
        # Create a list of az_lats, co_lats, nm_lats and ut_lats: y
        y_coor = lat_y
        
        # Add patches to figure p with line_color=white for x and y
        p.patches(x_coor, y_coor, line_color='white')
        
        # Specify the name of the output file and show the result
        output_file('01_09_Country_shape.html')
        show(p)
        #"""
        
    else:
        print("The country '{}' doesn't exist in this database.\n".format(country))

print("****************************************************")
print("** END                                            **")
print("****************************************************")

"""
https://bokeh.pydata.org/en/latest/docs/user_guide/geo.html#userguide-geo

from bokeh.io import output_file, show
from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure
from bokeh.sampledata.sample_geojson import geojson

output_file("geojson.html")

geo_source = GeoJSONDataSource(geojson=geojson)

p = figure(background_fill_color="lightgrey")
p.circle(x='x', y='y', size=15, alpha=0.7, source=geo_source)

show(p)
"""