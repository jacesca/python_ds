# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:33:24 2019

@author: jacqueline.cortez
Chapter 2: Spatial relationships
    One of the key aspects of geospatial data is how they relate to each other in space. 
    In this chapter, you will learn the different spatial relationships, and how to use them in Python to query 
    the data or to perform spatial joins. Finally, you will also learn in more detail about choropleth 
    visualizations.
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

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

print("****************************************************")
topic = "Reading geospatialdata"; print("** %s\n" % topic)

df_countries_geo = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
print("Columns of df_countries_geo:\n{}\n".format(df_countries_geo.columns))

df_cities_geo = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
print("Columns of df_cities_geo:\n{}\n".format(df_cities_geo.columns))

# Read the Paris districts dataset
filename = "paris_districts_utm.geojson"
df_paris_districts_geo = gpd.read_file(filename)
df_paris_districts_geo.crs = {'init': 'epsg:32631'}
df_paris_districts_geo = df_paris_districts_geo.to_crs(epsg = '3857')
df_paris_districts_geo['area'] = df_paris_districts_geo.geometry.area
df_paris_districts_geo['population_density'] = df_paris_districts_geo.population / df_paris_districts_geo.area * (10**6) # Add a population density column
print("Columns of {}:\n{}\n".format(filename, df_paris_districts_geo.columns))

# Read the rivers
filename = "shapefile_path\\ne_50m_rivers_lake_centerlines.shp"
df_rivers_geo = gpd.read_file(filename)
print("Columns of df_rivers_geo:\n{}\n".format(df_rivers_geo.columns))

# Read the bike stations
filename = "paris_sharing_bike_stations_utm.geojson"
df_bike_stations_geo = gpd.read_file(filename)
df_bike_stations_geo.crs = {'init': 'epsg:32631'}
df_bike_stations_geo = df_bike_stations_geo.to_crs(epsg = '3857')
print("Columns of {}:\n{}\n".format(filename, df_bike_stations_geo.columns))


# Read the trees in paris
filename = "paris_trees_small.gpkg"
df_trees_in_paris_geo = gpd.read_file(filename)
df_trees_in_paris_geo.crs = {'init': 'epsg:32631'}
df_trees_in_paris_geo = df_trees_in_paris_geo.to_crs(epsg = '3857')
print("Columns of {}:\n{}\n".format(filename, df_trees_in_paris_geo.columns))

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "paris_restaurants.csv"
df_restaurants = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_restaurants.columns))
df_restaurants_geo = gpd.GeoDataFrame(df_restaurants, geometry=gpd.points_from_xy(df_restaurants.x, df_restaurants.y))
print("Columns of df_restaurants_geo:\n{}\n".format(df_restaurants_geo.columns))

# Take a subset of the African restaurants
df_african_restaurants_geo = df_restaurants_geo[df_restaurants_geo['type']=='African restaurant']

print("****************************************************")
topic = "1. Shapely geometries and spatial relationships"; print("** %s\n" % topic)

geo_Brussels =df_cities_geo.loc[df_cities_geo.name == 'Brussels', 'geometry'].squeeze()
geo_Paris =df_cities_geo.loc[df_cities_geo.name == 'Paris','geometry'].squeeze()
geo_Belgium = df_countries_geo.loc[df_countries_geo.name == 'Belgium', 'geometry'].squeeze() #Squeeze 1 dimensional axis objects into scalars.
geo_France = list(df_countries_geo.loc[df_countries_geo.name == 'France', 'geometry'].squeeze())[1] #Getting the shape (one of multipolygon) we are interested in.
geo_UK = df_countries_geo.loc[df_countries_geo.name == 'United Kingdom', 'geometry'].squeeze()

geo_path = LineString([geo_Brussels, geo_Paris])

print("** Spatial methods:")
print("   Area of Belgium: {}".format(geo_Belgium.area))
print("   Distance between Brussels and Paris: {}".format(geo_Brussels.distance(geo_Paris)))
print("   Does Belgium contain Brussels?: {}".format(geo_Belgium.contains(geo_Brussels)))
print("   Does France contain Brussels?: {}".format(geo_France.contains(geo_Brussels)))
print("   Is Brussels in Belgium?: {}".format(geo_Brussels.within(geo_Belgium)))
print("   Are Belgium and France neighbors?: {}".format(geo_Belgium.touches(geo_France)))
print("   Does the selected path intersect France?: {}".format(geo_path.intersects(geo_France)))
print("   Does the selected path intersect United Kingdom?: {}\n".format(geo_path.intersects(geo_UK)))

gpd.GeoSeries([geo_Belgium, geo_France, geo_UK, geo_Paris, geo_Brussels, geo_path]).plot(cmap='tab20')
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Belgium, France, United Kingdom, Paris and Brussels'); plt.suptitle(topic);  # Setting the titles.
plt.show()

print("****************************************************")
topic = "2. Creating a Point geometry"; print("** %s\n" % topic)

geo_eiffel_tower = Point(255422.6, 6250868.9) # Construct a point object for the Eiffel Tower
print("Eiffel Tower: {}\n".format(geo_eiffel_tower))

print("****************************************************")
topic = "3.Shapely's spatial methods"; print("** %s\n" % topic)

geo_district_montparnasse = df_paris_districts_geo.loc[df_paris_districts_geo.district_name=='Montparnasse', 'geometry'].squeeze() # Accessing the Montparnasse geometry (Polygon) and restaurant
geo_restaurant = df_restaurants_geo.loc[956, 'geometry']

print('Is the Eiffel Tower located within the Montparnasse district? {}'.format(geo_eiffel_tower.within(geo_district_montparnasse))) # Is the Eiffel Tower located within the Montparnasse district?
print('Does the Montparnasse district contains the restaurant? {}'.format(geo_district_montparnasse.contains(geo_restaurant))) # Does the Montparnasse district contains the restaurant?)
print('The distance between the Eiffel Tower and the restaurant? {}\n'.format(geo_eiffel_tower.distance(geo_restaurant))) # The distance between the Eiffel Tower and the restaurant?

gpd.GeoSeries([geo_district_montparnasse, geo_eiffel_tower, geo_restaurant]).plot(cmap='tab20')
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Montparnasse district'); plt.suptitle(topic);  # Setting the titles.
plt.show()

print("****************************************************")
topic = "4. Spatial relationships with GeoPandas"; print("** %s\n" % topic)

print("Which cities are in France?\n{}\n".format(df_cities_geo[df_cities_geo.within(geo_France)].name))

geo_Amazonas = df_rivers_geo.loc[df_rivers_geo.name=='Amazonas', 'geometry'].squeeze()
print("Which countries does the Amazon River intersects? \n{}\n".format(df_countries_geo[df_countries_geo.intersects(geo_Amazonas)].name))

print("****************************************************")
topic = "5. In which district in the Eiffel Tower located?"; print("** %s\n" % topic)

mask = df_paris_districts_geo.contains(geo_eiffel_tower) # Create a boolean Series
print("{}\n".format(df_paris_districts_geo[mask].district_name)) # Filter the districts with the boolean mask

print("****************************************************")
topic = "6. How far is the closest restaurant?"; print("** %s\n" % topic)

dist_eiffel = df_restaurants_geo.geometry.distance(geo_eiffel_tower) # The distance from each restaurant to the Eiffel Tower
print("The distance to the closest restaurant (from Eiffel Tower): {}mts.".format(dist_eiffel.min())) # The distance to the closest restaurant

geo_restaurants_eiffel = df_restaurants_geo[dist_eiffel<=1000] # Filter the restaurants for closer than 1 km

# Make a plot of the close-by restaurants
ax = geo_restaurants_eiffel.plot(color='darkgreen')
gpd.GeoSeries([geo_eiffel_tower]).plot(ax=ax, color='red')
contextily.add_basemap(ax)
ax.set_axis_off()
plt.title('Restaurants near Eiffel Tower'); plt.suptitle(topic);  # Setting the titles.
plt.show()

print("****************************************************")
topic = "7. The spatial join operation"; print("** %s\n" % topic)

df_cities_within_countries_geo = gpd.sjoin(df_cities_geo, df_countries_geo[['name', 'geometry']], op='within')
print("The result of sjoin from cities within countries (first 5 rows):\n{}\n".format(df_cities_within_countries_geo.head()))

print("****************************************************")
topic = "8. Paris: spatial join of districts and bike stations"; print("** %s\n" % topic)

df_bikes_in_paris_geo = gpd.sjoin(df_bike_stations_geo, df_paris_districts_geo[['district_name','geometry']], op='within') # Join the districts and stations datasets
print(df_bikes_in_paris_geo.head()) # Inspect the first five rows of the result

print("****************************************************")
topic = "9. Map of tree density by district (1)"; print("** %s\n" % topic)

df_trees_by_districts_geo = gpd.sjoin(df_trees_in_paris_geo, df_paris_districts_geo[['district_name', 'geometry']], op='within') # Spatial join of the trees and districts datasets

trees_by_district = df_trees_by_districts_geo.groupby('district_name').size() # Calculate the number of trees in each district
trees_by_district = trees_by_district.to_frame(name='n_trees') # Convert the series to a DataFrame and specify column name
print(trees_by_district.head()) # Inspect the result

print("****************************************************")
topic = "10. Map of tree density by district (2)"; print("** %s\n" % topic)

trees_by_district = trees_by_district.reset_index()
df_districts_trees_geo = pd.merge(df_paris_districts_geo, trees_by_district, on='district_name') # Merge the 'districts' and 'trees_by_district' dataframes
df_districts_trees_geo['n_trees_per_area'] = df_districts_trees_geo['n_trees'] / df_districts_trees_geo.geometry.area * (10**6) # Add a column with the tree density

# Make of map of the districts colored by 'n_trees_per_area'
fig, ax = plt.subplots(figsize=(10,4))
base = df_districts_trees_geo.plot(ax = ax, column='n_trees_per_area', legend=True)
#ax.set_anchor('N'); 
#Changing the fontsize of the color bar: (Source: https://gist.github.com/sebbacon/60ceec549f1d461af4543ccc5024c095?short_path=839cef5)
#     The Figure has two Axes: one for the map, and one for the Colorbar. The one we care about is the second one. 
#     cb_ax = fig.axes[1] #Axes for the colorbar.
#     The legend is actually a Colorbar object. To change the legend's font size, we have to get hold of the Colorbar's 
#     Axes object, and call .tick_params() on that. 
fig.axes[1].tick_params(labelsize=7) #Changing the fontsize of the colorbar in second axes of the graph.
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Tree Density in Paris'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "12. Equal interval choropleth"; print("** %s\n" % topic)

# Make a choropleth of the number of trees 
fig, ax = plt.subplots(figsize=(10,4))
df_districts_trees_geo.plot(ax = ax, column='n_trees', legend=True)
fig.axes[1].tick_params(labelsize=7) #Changing the fontsize of the colorbar in second axes of the graph.
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Tree Density in Paris (n_trees)'); plt.suptitle(topic);  # Setting the titles.
plt.show()


# Make a choropleth of the number of trees per area
fig, ax = plt.subplots(figsize=(10,4))
df_districts_trees_geo.plot(ax = ax, column='n_trees_per_area', legend=True)
fig.axes[1].tick_params(labelsize=7) #Changing the fontsize of the colorbar in second axes of the graph.
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Tree Density in Paris (n_trees_per_area)'); plt.suptitle(topic);  # Setting the titles.
plt.show()


# Make a choropleth of the number of trees 
df_districts_trees_geo.plot(column='n_trees_per_area', scheme='equal_interval', legend=True, legend_kwds=dict(fontsize=7), figsize=(10,4))
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Tree Density in Paris (n_trees_per_area)'); plt.suptitle(topic);  # Setting the titles.
plt.show()

print("****************************************************")
topic = "13. Quantiles choropleth"; print("** %s\n" % topic)

# Generate the choropleth and store the axis
df_districts_trees_geo.plot(column='n_trees_per_area', scheme='quantiles', k=7, cmap='YlGn', legend=True, legend_kwds=dict(fontsize=7), figsize=(10,4))
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Tree Density in Paris (n_trees_per_area)'); plt.suptitle(topic);  # Setting the titles.
plt.show()

print("****************************************************")
topic = "14. Compare classification algorithms"; print("** %s\n" % topic)

# Set up figure and subplots
fig, axes = plt.subplots(ncols=2, figsize=(11,4))

# Plot equal interval map
df_districts_trees_geo.plot(column='n_trees_per_area', scheme='equal_interval', k=5, legend=True, legend_kwds=dict(fontsize=7), ax=axes[0])
axes[0].set_title('Equal Interval')
axes[0].set_axis_off()

# Plot quantiles map
df_districts_trees_geo.plot(column='n_trees_per_area', scheme='quantiles', k=5, legend=True, legend_kwds=dict(fontsize=7), ax=axes[1])
axes[1].set_title('Quantiles')
axes[1].set_axis_off()

# Display maps
plt.suptitle(topic);  
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
