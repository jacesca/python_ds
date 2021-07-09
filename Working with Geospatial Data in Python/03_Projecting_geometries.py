# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:33:24 2019

@author: jacqueline.cortez
Chapter 3: Projecting and transforming geometries
    In this chapter, we will take a deeper look into how the coordinates of the geometries are expressed based on their 
    Coordinate Reference System (CRS). You will learn the importance of those reference systems and how to handle it in 
    practice with GeoPandas. Further, you will also learn how to create new geometries based on the spatial relationships, 
    which will allow you to overlay spatial datasets. And you will further practice this all with Paris datasets!
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot as plt                                               #For creating charts
import pandas            as pd                                                #For loading tabular data
import contextily                                                             #To add a background web map to our plot
import geopandas         as gpd                                               #For working with geospatial data

from shapely.geometry                import LineString                        #(Geospatial) To create a Linestring geometry column 
from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 
from shapely.geometry                import Polygon                           #(Geospatial) To create a point geometry column 

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

pd.set_option("display.max_columns",20)

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

print("****************************************************")
topic = "Reading geospatialdata"; print("** %s\n" % topic)

#Read countries
df_countries_4326_geo = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
df_countries_2163_geo = df_countries_4326_geo.to_crs(epsg = '2163')
print("Columns of df_countries_geo:\n{}\n".format(df_countries_4326_geo.columns))

# Read the Paris districts dataset
filename = "paris_districts_utm.geojson"
df_paris_districts_32631_geo = gpd.read_file(filename)
df_paris_districts_32631_geo.crs = {'init': 'epsg:32631'}
df_paris_districts_3857_geo = df_paris_districts_32631_geo.to_crs(epsg = '3857')
df_paris_districts_3857_geo['area'] = df_paris_districts_3857_geo.geometry.area
df_paris_districts_3857_geo['population_density'] = df_paris_districts_3857_geo.population / df_paris_districts_3857_geo.area * (10**6) # Add a population density column
df_paris_districts_4326_geo = df_paris_districts_32631_geo.to_crs(epsg = '4326')
df_paris_districts_2154_geo = df_paris_districts_32631_geo.to_crs(epsg = 2154) # Convert the districts to the RGF93 reference system
print("Columns of df_paris_districts_3857_geo:\n{}\n".format(df_paris_districts_3857_geo.columns))

#Read the land use paris
filename = "paris_land_use\\Shapefiles\\FR001L1_PARIS_UA2012.shp"
df_paris_land_use_geo = gpd.read_file(filename)
df_paris_land_use_geo.drop(columns=['COUNTRY', 'CITIES', 'FUA_OR_CIT', 'CODE2012', 'PROD_DATE', 'IDENT', 'Shape_Leng', 'Shape_Area'], inplace=True)
df_paris_land_use_geo.columns=['land_use', 'pop2012', 'geometry']
print("Columns of df_paris_land_use_geo:\n{}\n".format(df_paris_land_use_geo.columns))

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "paris_restaurants.csv"
df_restaurants = pd.read_csv(filename)
df_restaurants_3857_geo = gpd.GeoDataFrame(df_restaurants, crs={'init': 'epsg:3857'}, geometry=gpd.points_from_xy(df_restaurants.x, df_restaurants.y))
df_restaurants_4326_geo = df_restaurants_3857_geo.to_crs(epsg = '4326')
print("Columns of df_restaurants_3857_geo:\n{}\n".format(df_restaurants_3857_geo.columns))


print("****************************************************")
topic = "2. Geographic vs projected coordinates"; print("** %s\n" % topic)

print("CRS of df_paris_districts_4326_geo is: {}\n".format(df_paris_districts_4326_geo.crs)) # Print the CRS information
print("Head of df_paris_districts_4326_geo:\n{}\n".format(df_paris_districts_4326_geo.head())) # Print the first rows of the GeoDataFrame


print("****************************************************")
topic = "3. Working with coordinate systems in GeoPandas"; print("** %s\n" % topic)

print("CRS of df_countries_4326_geo is: {}\n".format(df_countries_4326_geo.crs)) # Print the CRS information

df_North_America_4326 = df_countries_4326_geo.loc[df_countries_4326_geo.continent == 'North America']
df_North_America_2163 = df_countries_2163_geo.loc[df_countries_2163_geo.continent == 'North America']

# Set up figure and subplots
fig, axes = plt.subplots(ncols=2, figsize=(11,4))

# Plot Equal Area / EPSG:2163
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=2, markerscale=0.7, title='Countries', fontsize=6, title_fontsize=7)
df_North_America_2163.plot(column='name', legend=True, legend_kwds = legend_kwds, ax=axes[0])
axes[0].set_title('Nort America EPSG:2163')
axes[0].set_anchor('N'); 
axes[0].set_axis_off()

# Plot WGS84 / EPSG:4326
df_North_America_4326.plot(column='name', legend=True, legend_kwds = legend_kwds, ax=axes[1])
axes[1].set_title('Nort America EPSG:4326')
axes[1].set_anchor('N'); 
axes[1].set_axis_off()

# Display maps
plt.suptitle(topic);  
plt.subplots_adjust(left=0.05, bottom=None, right=0.75, top=0.8, wspace=1, hspace=None);
plt.show()


print("****************************************************")
topic = "4. Projecting a GeoDataFrame"; print("** %s\n" % topic)

# Set up figure and subplots
fig, axes = plt.subplots(ncols=2, figsize=(11,4))

df_paris_districts_4326_geo.plot(ax=axes[0]) # Plot the districts dataset
axes[0].set_title("District's Paris ({})".format(df_paris_districts_4326_geo.crs))
#axes[0].set_anchor('N'); 
axes[0].tick_params(labelsize=7); 
axes[0].set_xlabel('Longitude', fontsize=7); axes[0].set_ylabel('Latitude', fontsize=7); # Labeling the axis.


df_paris_districts_2154_geo.plot(ax=axes[1]) # Plot the districts dataset again
axes[1].set_title("District's Paris ({})".format(df_paris_districts_2154_geo.crs))
#axes[1].set_anchor('N'); 
axes[1].tick_params(labelsize=7); 
axes[1].set_xlabel('Longitude', fontsize=7); axes[1].set_ylabel('Latitude', fontsize=7); # Labeling the axis.


plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None);
plt.show()


print("****************************************************")
topic = "5. Projecting a Point"; print("** %s\n" % topic)

eiffel_tower = Point(2.2945, 48.8584)
s_eiffel_tower_4326 = gpd.GeoSeries([eiffel_tower], crs={'init': 'epsg:4326'}) # Put the point in a GeoSeries with the correct CRS
print("Eiffel Tower reference in epsg=4326: \n{}\n\n".format(s_eiffel_tower_4326))

s_eiffel_tower_2154 = s_eiffel_tower_4326.to_crs(epsg=2154) # Convert to other CRS
print("Eiffel Tower reference in epsg=2154: \n{}\n".format(s_eiffel_tower_2154))


print("****************************************************")
topic = "6. Calculating distance in a projected CRS"; print("** %s\n" % topic)

eiffel_tower = s_eiffel_tower_2154[0] # Extract the single Point
df_restaurants_2154_geo = df_restaurants_4326_geo.to_crs(s_eiffel_tower_2154.crs) # Ensure the restaurants use the same CRS

dist_eiffel = df_restaurants_2154_geo.geometry.distance(eiffel_tower) # The distance from each restaurant to the Eiffel Tower
print("The distance to the closest restaurant: {}\n".format(dist_eiffel.min())) # The distance to the closest restaurant


print("****************************************************")
topic = "7. Projecting to Web Mercator for using web tiles"; print("** %s\n" % topic)

# Plot the restaurants with a background map
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, title_fontsize=8, markerscale=0.7, fontsize=7, title='Type of restaurant')
ax = df_restaurants_3857_geo.plot(column='type', markersize=10, legend=True, legend_kwds = legend_kwds, figsize=(11,4))
contextily.add_basemap(ax)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Restaurants near Eiffel Tower'); plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "8. Spatial operations: creating new geometries"; print("** %s\n" % topic)

df_africa_4326_geo = df_countries_4326_geo[df_countries_4326_geo.continent=='Africa'].copy()
box = Polygon([[60, 10], [60,-10], [-20, -10], [-20, 10], [60, 10]])

df_africa_intersected = df_africa_4326_geo.copy()
df_africa_intersected['intersected'] = df_africa_intersected.intersection(box)
df_africa_intersected  = df_africa_intersected.set_geometry('intersected')
#df_africa_intersected.drop(df_africa_intersected[df_africa_intersected.is_empty].index, axis=0, inplace=True)
print("Head of Africa intersected by a box:\n{}\n".format(df_africa_intersected.head()))

# Set up figure and subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 5.5))

ax = axes[0, 0]
df_africa_4326_geo.plot(ax=ax, column='name') # Plot the districts dataset
ax.set_title("Africa", color='red', fontsize=10)
#axes[0].set_anchor('N'); 
ax.set_xlim(-25, 65); ax.set_ylim(-40, 40);
ax.tick_params(labelsize=7); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.

ax = axes[0, 1]
gpd.GeoSeries([box]).plot(ax=ax, color='red', alpha=0.5) # Plot the districts dataset
ax.set_title("Box", color='red', fontsize=10)
ax.set_xlim(-25, 65); ax.set_ylim(-40, 40);
#axes[0].set_anchor('N'); 
ax.tick_params(labelsize=7); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.

ax = axes[1, 0]
df_africa_4326_geo.plot(ax=ax, column='name') # Plot the districts dataset
gpd.GeoSeries([box]).plot(ax=ax, color='red', alpha=0.5) # Plot the districts dataset
ax.set_title("Intersecting...", color='red', fontsize=10)
ax.set_xlim(-25, 65); ax.set_ylim(-40, 40);
#axes[0].set_anchor('N'); 
ax.tick_params(labelsize=7); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.

ax = axes[1, 1]
df_africa_4326_geo.intersection(box).plot(ax=ax, cmap='tab20') # Plot the districts dataset
ax.set_title("The result", color='red', fontsize=10)
ax.set_xlim(-25, 65); ax.set_ylim(-40, 40);
#axes[0].set_anchor('N'); 
ax.tick_params(labelsize=7); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.

plt.suptitle(topic, color='darkblue', fontsize=11);  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5);
plt.show()

print("****************************************************")
topic = "9. Exploring a Land Use dataset"; print("** %s\n" % topic)

print("Uses Land Describe: \n{}\n".format(df_paris_land_use_geo.land_use.unique()))
df_paris_land_use_geo['area'] = df_paris_land_use_geo.geometry.area # Add the area as a new column
total_area = df_paris_land_use_geo.groupby('land_use')['area'].sum() / 1000**2 # Calculate the total area for each land use class
print("Total area for each land use: \n{}\n".format(total_area))


df_paris_districts_landuse_geo = df_paris_districts_3857_geo.to_crs(df_paris_land_use_geo.crs) #Change the crs in district df to use the same crs from land use
combined = gpd.overlay(df_paris_land_use_geo, df_paris_districts_landuse_geo, how='intersection') ###Needed in ex. 13 and 14

district_muette = df_paris_districts_landuse_geo.loc[df_paris_districts_landuse_geo.district_name=='Muette', 'geometry'].squeeze()
land_use_muette = df_paris_land_use_geo.intersection(district_muette) # Calculate the intersection of the land use polygons with Notre Dame

#df_paris_land_use_geo = df_paris_land_use_geo[(df_paris_land_use_geo.land_use=='Green urban areas')].copy()
df_paris_land_use_geo = df_paris_land_use_geo[(df_paris_land_use_geo.intersection(district_muette).is_empty == False)].copy()


# Make a plot of the land use with 'class' as the color
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, title_fontsize=7, fontsize=6, title='Land Use')
df_paris_land_use_geo.plot(column='land_use', legend=True, legend_kwds = legend_kwds, cmap='tab20', figsize=(10, 4))
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Paris Land Use (Muette district)'); plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "10. Intersection of two polygons"; print("** %s\n" % topic)

park_boulogne = df_paris_land_use_geo[(df_paris_land_use_geo.land_use=='Green urban areas')].dissolve(by='land_use')
park_boulogne = park_boulogne.geometry.squeeze()

intersection = park_boulogne.intersection(district_muette) # Calculate the intersection of both polygons
print("Proportion of district area that occupied park: {0:,.2f}\n".format(intersection.area / district_muette.area * 100)) # Print proportion of district area that occupied park



# Set up figure and subplots
fig, axes = plt.subplots(ncols=2, figsize=(11, 5.5))

# Plot the two polygons
ax = axes[0]
gpd.GeoSeries([park_boulogne, district_muette]).plot(ax=ax, alpha=0.5, color=['green', 'red'])
ax.set_title("Location of interest", color='red', fontsize=10)
#axes[0].set_anchor('N'); 
ax.tick_params(labelsize=7); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.


# Plot the intersection
ax = axes[1]
gpd.GeoSeries([intersection]).plot(ax=ax)
ax.set_title("Intersection result", color='red', fontsize=10)
#axes[0].set_anchor('N'); 
ax.tick_params(labelsize=7); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.

plt.suptitle(topic, color='darkblue', fontsize=11);  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None);
plt.show()


print("****************************************************")
topic = "11. Intersecting a GeoDataFrame with a Polygon"; print("** %s\n" % topic)

print("Type of district_muette: {}\n".format(type(district_muette)))
print("Head of :\n{}\n".format(land_use_muette.head()))

# Plot the intersection of the land use polygons with Notre Dame, calculated in ex.9
land_use_muette.plot(edgecolor='black', figsize=(10, 4), cmap='tab20')
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Mutte District Land Use'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "13. Overlay of two polygon layers"; print("** %s\n" % topic)

# Print the first five rows of both datasets
print("Head of Paris districts: \n{}\n\n".format(df_paris_districts_landuse_geo.head()))
print("Head of Land Use: \n{}\n\n".format(df_paris_land_use_geo.head())) #Only to know the columns and structure

# Overlay both datasets based on the intersection
### Made in ex. 9### #combined = gpd.overlay(df_paris_mini_land_use, df_paris_districts_landuse_geo, how='intersection')

# Print the first five rows of the result
print("Head of the overlay: \n{}\n".format(combined.head()))

print("****************************************************")
topic = "14. Inspecting the overlay result"; print("** %s\n" % topic)

combined['area'] = combined.geometry.area # Add the area as a column
land_use_muette = combined[combined.district_name == 'Muette'] # Take a subset for the Muette district
print("Total area for each land use class in Muette District: \n{}\n".format(land_use_muette.groupby('land_use')['area'].sum() / 1000**2)) # Calculate the total area for each land use class

# Visualize the land use of the Muette district
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, title_fontsize=7, fontsize=6, title='Land Use')
land_use_muette.plot(column='land_use', legend=True, legend_kwds = legend_kwds, cmap='tab20', figsize=(10, 4))
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Paris Land Use (Muette district)'); plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import contextily                                                             #To add a background web map to our plot
#import inspect                                                                #Used to get the code inside a function
#import folium                                                                 #To create map street folium.__version__'0.10.0'
#import geopandas         as gpd                                               #For working with geospatial data 
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt                                               #For creating charts
#import numpy             as np                                                #For making operations in lists
#import pandas            as pd                                                #For loading tabular data
#import seaborn           as sns                                               #For visualizing data
#import pprint                                                                 #Import pprint to format disctionary output
#import missingno         as msno                                              #Missing data visualization module for Python

#import os                                                                     #To raise an html page in python command
#import tempfile                                                               #To raise an html page in python command
#import webbrowser                                                             #To raise an html page in python command  

#import calendar                                                               #For accesing to a vary of calendar operations
#import math                                                                   #For accesing to a complex math operations
#import nltk                                                                   #For working with text data
#import random                                                                 #For generating random numbers
#import re                                                                     #For regular expressions
#import tabula                                                                 #For extracting tables from pdf
#import timeit                                                                 #For Measure execution time of small code snippets
#import time                                                                   #To measure the elapsed wall-clock time between two points
#import scykit-learn                                                           #For performing machine learning  
#import warnings
#import wikipedia

#from collections                     import defaultdict                       #Returns a new dictionary-like object
#from datetime                        import date                              #For obteining today function
#from datetime                        import datetime                          #For obteining today function
#from functools                       import reduce                            #For accessing to a high order functions (functions or operators that return functions)
#from glob                            import glob                              #For using with pathnames matching
#from itertools                       import combinations                      #For iterations
#from itertools                       import cycle                             #Used in the function plot_labeled_decision_regions()
#from math                            import ceil                              #Used in the function plot_labeled_decision_regions()
#from math                            import floor                             #Used in the function plot_labeled_decision_regions()
#from math                            import radian                            #For accessing a specific math operations
#from mpl_toolkits.mplot3d            import Axes3D
#from pandas.api.types                import CategoricalDtype                  #For categorical data
#from pandas.plotting                 import parallel_coordinates              #For Parallel Coordinates
#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
#from shapely.geometry                import LineString                        #(Geospatial) To create a Linestring geometry column 
#from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 
#from shapely.geometry                import Polygon                           #(Geospatial) To create a point geometry column 
#from string                          import Template                          #For working with string, regular expressions


#from bokeh.io                        import curdoc                            #For interacting visualizations
#from bokeh.io                        import output_file                       #For interacting visualizations
#from bokeh.io                        import show                              #For interacting visualizations
#from bokeh.plotting                  import ColumnDataSource                  #For interacting visualizations
#from bokeh.plotting                  import figure                            #For interacting visualizations
#from bokeh.layouts                   import column                            #For interacting visualizations
#from bokeh.layouts                   import gridplot                          #For interacting visualizations
#from bokeh.layouts                   import row                               #For interacting visualizations
#from bokeh.layouts                   import widgetbox                         #For interacting visualizations
#from bokeh.models                    import Button                            #For interacting visualizations
#from bokeh.models                    import CategoricalColorMapper            #For interacting visualizations
#from bokeh.models                    import CheckboxGroup                     #For interacting visualizations
#from bokeh.models                    import ColumnDataSource                  #For interacting visualizations
#from bokeh.models                    import HoverTool                         #For interacting visualizations
#from bokeh.models                    import RadioGroup                        #For interacting visualizations
#from bokeh.models                    import Select                            #For interacting visualizations
#from bokeh.models                    import Slider                            #For interacting visualizations
#from bokeh.models                    import Toggle                            #For interacting visualizations
#from bokeh.models.widgets            import Panel                             #For interacting visualizations
#from bokeh.models.widgets            import Tabs                              #For interacting visualizations
#from bokeh.palettes                  import Spectral6                         #For interacting visualizations


#import keras                                                                  #For DeapLearning
#import keras.backend as k                                                     #For DeapLearning
#from keras.applications.resnet50     import decode_predictions                #For DeapLearning
#from keras.applications.resnet50     import preprocess_input                  #For DeapLearning
#from keras.applications.resnet50     import ResNet50                          #For DeapLearning
#from keras.callbacks                 import EarlyStopping                     #For DeapLearning
#from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
#from keras.datasets                  import fashion_mnist                     #For DeapLearning
#from keras.datasets                  import mnist                             #For DeapLearning
#from keras.layers                    import BatchNormalization                #For DeapLearning
#from keras.layers                    import Concatenate                       #For DeapLearning
#from keras.layers                    import Conv2D                            #For DeapLearning
#from keras.layers                    import Dense                             #For DeapLearning
#from keras.layers                    import Dropout                           #For DeapLearning
#from keras.layers                    import Embedding                         #For DeapLearning
#from keras.layers                    import Flatten                           #For DeapLearning
#from keras.layers                    import GlobalMaxPooling1D                #For DeapLearning
#from keras.layers                    import Input                             #For DeapLearning
#from keras.layers                    import LSTM                              #For DeapLearning
#from keras.layers                    import MaxPool2D                         #For DeapLearning
#from keras.layers                    import SpatialDropout1D                  #For DeapLearning
#from keras.layers                    import Subtract                          #For DeapLearning
#from keras.models                    import load_model                        #For DeapLearning
#from keras.models                    import Model                             #For DeapLearning
#from keras.models                    import Sequential                        #For DeapLearning
#from keras.optimizers                import Adam                              #For DeapLearning
#from keras.optimizers                import SGD                               #For DeapLearning
#from keras.preprocessing             import image                             #For DeapLearning
#from keras.preprocessing.text        import Tokenizer                         #For DeapLearning
#from keras.preprocessing.sequence    import pad_sequences                     #For DeapLearning
#from keras.utils                     import plot_model                        #For DeapLearning
#from keras.utils                     import to_categorical                    #For DeapLearning
#from keras.wrappers.scikit_learn     import KerasClassifier                   #For DeapLearning


#import networkx          as nx                                                #For Network Analysis in Python
#import nxviz             as nv                                                #For Network Analysis in Python
#from nxviz                           import ArcPlot                           #For Network Analysis in Python
#from nxviz                           import CircosPlot                        #For Network Analysis in Python 
#from nxviz                           import MatrixPlot                        #For Network Analysis in Python 


#import scipy.stats as stats                                                   #For accesign to a vary of statistics functiosn
#from scipy.cluster.hierarchy         import dendrogram                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import fcluster                          #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import linkage                           #For learning machine - unsurpervised
#from scipy.ndimage                   import gaussian_filter                   #For working with images
#from scipy.ndimage                   import median_filter                     #For working with images
#from scipy.signal                    import convolve2d                        #For learning machine - deep learning
#from scipy.sparse                    import csr_matrix                        #For learning machine 
#from scipy.special                   import expit as sigmoid                  #For learning machine 
#from scipy.stats                     import pearsonr                          #For learning machine 
#from scipy.stats                     import randint                           #For learning machine 
       

#from skimage                         import exposure                          #For working with images
#from skimage                         import measure                           #For working with images
#from skimage.filters.thresholding    import threshold_otsu                    #For working with images
#from skimage.filters.thresholding    import threshold_local                   #For working with images 


#from sklearn                         import datasets                          #For learning machine
#from sklearn.cluster                 import KMeans                            #For learning machine - unsurpervised
#from sklearn.decomposition           import NMF                               #For learning machine - unsurpervised
#from sklearn.decomposition           import PCA                               #For learning machine - unsurpervised
#from sklearn.decomposition           import TruncatedSVD                      #For learning machine - unsurpervised
#from sklearn.ensemble                import AdaBoostClassifier                #For learning machine - surpervised
#from sklearn.ensemble                import BaggingClassifier                 #For learning machine - surpervised
#from sklearn.ensemble                import GradientBoostingRegressor         #For learning machine - surpervised
#from sklearn.ensemble                import RandomForestClassifier            #For learning machine
#from sklearn.ensemble                import RandomForestRegressor             #For learning machine - unsurpervised
#from sklearn.ensemble                import VotingClassifier                  #For learning machine - unsurpervised
#from sklearn.feature_selection       import chi2                              #For learning machine
#from sklearn.feature_selection       import SelectKBest                       #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                   #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                 #For learning machine
#from sklearn.feature_extraction.text import TfidfVectorizer                   #For learning machine - unsurpervised
#from sklearn.impute                  import SimpleImputer                     #For learning machine
#from sklearn.linear_model            import ElasticNet                        #For learning machine
#from sklearn.linear_model            import Lasso                             #For learning machine
#from sklearn.linear_model            import LinearRegression                  #For learning machine
#from sklearn.linear_model            import LogisticRegression                #For learning machine
#from sklearn.linear_model            import Ridge                             #For learning machine
#from sklearn.manifold                import TSNE                              #For learning machine - unsurpervised
#from sklearn.metrics                 import accuracy_score                    #For learning machine
#from sklearn.metrics                 import classification_report             #For learning machine
#from sklearn.metrics                 import confusion_matrix                  #For learning machine
#from sklearn.metrics                 import mean_squared_error as MSE         #For learning machine
#from sklearn.metrics                 import roc_auc_score                     #For learning machine
#from sklearn.metrics                 import roc_curve                         #For learning machine
#from sklearn.model_selection         import cross_val_score                   #For learning machine
#from sklearn.model_selection         import GridSearchCV                      #For learning machine
#from sklearn.model_selection         import KFold                             #For learning machine
#from sklearn.model_selection         import RandomizedSearchCV                #For learning machine
#from sklearn.model_selection         import train_test_split                  #For learning machine
#from sklearn.multiclass              import OneVsRestClassifier               #For learning machine
#from sklearn.neighbors               import KNeighborsClassifier as KNN       #For learning machine
#from sklearn.pipeline                import FeatureUnion                      #For learning machine
#from sklearn.pipeline                import make_pipeline                     #For learning machine - unsurpervised
#from sklearn.pipeline                import Pipeline                          #For learning machine
#from sklearn.preprocessing           import FunctionTransformer               #For learning machine
#from sklearn.preprocessing           import Imputer                           #For learning machine
#from sklearn.preprocessing           import MaxAbsScaler                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing           import Normalizer                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing           import normalize                         #For learning machine - unsurpervised
#from sklearn.preprocessing           import scale                             #For learning machine
#from sklearn.preprocessing           import StandardScaler                    #For learning machine
#from sklearn.svm                     import SVC                               #For learning machine
#from sklearn.tree                    import DecisionTreeClassifier            #For learning machine - supervised
#from sklearn.tree                    import DecisionTreeRegressor             #For learning machine - supervised


#import statsmodels             as sm                                          #For stimations in differents statistical models
#import statsmodels.api         as sm                                          #Make a prediction model
#import statsmodels.formula.api as smf                                         #Make a prediction model    

#import tensorflow              as tf                                          #For DeapLearning



# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#pd.set_option('display.max_rows', -1)                                         #Shows all rows

#register_matplotlib_converters()                                              #Require to explicitly register matplotlib converters.

#Setting images params
#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09
#plt.rcParams.update({'figure.max_open_warning': 0})                           #To solve the max images open
#plt.rcParams["axes.labelsize"] = 8                                            #Font
#plt.rc('xtick',labelsize=8)
#plt.rc('ytick',labelsize=6)
#plt.rcParams['figure.max_open_warning'] = 60                                  #params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 5), 'axes.labelsize': 'x-large', 'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
#plt.rcParams["legend.fontsize"] = 8
#plt.style.use('dark_background')
#plt.style.use('default')
#plt.xticks(fontsize=7); plt.yticks(fontsize=8);
#plt.xticks(rotation=45)

#from matplotlib.axes._axes import _log as matplotlib_axes_logger              #To avoid warnings
#matplotlib_axes_logger.setLevel('ERROR')
#matplotlib_axes_logger.setLevel(0)                                            #To restore default

#ax.tick_params(labelsize=6)                                                   #axis : {'x', 'y', 'both'}
#ax.tick_params(axis='x', rotation=45)                                         #Set rotation atributte

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.
#np.random.seed(SEED)

#tf.compat.v1.set_random_seed(SEED)                                            #Instead of tf.set_random_seed, because it is deprecated.

#sns.set(font_scale=0.8)                                                       #Font
#sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot
#sns.set(color_codes=True)                                                     #Habilita el uso de los codigos de color
#sns.set()                                                                     #Seaborn defult style
#sns.set_style(this_style)                                                     #['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
#sns.despine(left=True)                                                        #Remove the spines (all borders)
#sns.palettes.SEABORN_PALETTES                                                 #Despliega todas las paletas disponibles 
#sns.palplot(sns.color_palette())                                              #Display a palette
#sns.color_palette()                                                           #The current palette
#sns.set(style=”whitegrid”, palette=”pastel”, color_codes=True)
#sns.mpl.rc(“figure”, figsize=(10,6))

#warnings.filterwarnings('ignore', 'Objective did not converge*')              #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
#warnings.filterwarnings('default', 'Objective did not converge*')             #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


#Create categorical type data to use
#cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  ordered=True)
# Change the data type of 'rating' to category
#weather['rating'] = weather.rating.astype(cats)


#print("The area of your rectangle is {}cm\u00b2".format(area))                 #Print the superscript 2

### Show a basic html page
#tmp=tempfile.NamedTemporaryFile()
#path=tmp.name+'.html'
#f=open(path, 'w')
#f.write("<html><body><h1>Test</h1></body></html>")
#f.close()
#webbrowser.open('file://' + path)
