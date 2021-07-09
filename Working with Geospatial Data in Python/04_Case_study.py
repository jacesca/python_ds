# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:26:34 2019

@author: jacqueline.cortez
Chapter 4: Putting it all together - Artisanal mining sites case study
    In this final chapter, we leave the Paris data behind us, and apply everything we have learnt up to now 
    on a brand new dataset about artisanal mining sites in Eastern Congo. Further, you will still learn some 
    new spatial operations, how to apply custom spatial operations, and you will get a sneak preview into 
    raster data.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot as plt                                               #For creating charts
import pandas            as pd                                                #For loading tabular data
import geopandas         as gpd                                               #For working with geospatial data

from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 
from matplotlib                      import colors                            #To create custom cmap
from matplotlib.ticker               import StrMethodFormatter                #Import the necessary library to delete the scientist notation

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

pd.set_option("display.max_columns",20)
pd.options.display.float_format = '{:,.2f}'.format

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

print("****************************************************")
topic = "Reading geospatialdata"; print("** %s\n" % topic)

df_countries_4326_geo = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
print("Columns of df_countries_geo:\n{}".format(df_countries_4326_geo.columns))
print("CRS of df_countries_geo:\n{}\n".format(df_countries_4326_geo.crs))

# Read the national parks of Congo
filename = "Conservation\\RDC_aire_protegee_2013.shp"
df_congopark_projected_geo = gpd.read_file(filename)
df_congopark_4326_geo = df_congopark_projected_geo.to_crs({'init': 'epsg:4326'})
print("Columns of df_congopark_projected_geo:\n{}".format(df_congopark_projected_geo.columns))
print("CRS of df_congopark_projected_geo:\n{}\n".format(df_congopark_projected_geo.crs))

# Read the mines place from Congo
filename = "ipis_cod_mines.geojson"
df_congomines_crs84_geo = gpd.read_file(filename)
print("Columns of df_congomines_crs84_geo:\n{}".format(df_congomines_crs84_geo.columns))
print("CRS of df_congomines_crs84_geo:\n{}\n".format(df_congomines_crs84_geo.crs))

filename = "Rivers\\populated_places.shp"
df_places_4326_geo = gpd.read_file(filename)
print("Columns of df_places_4326_geo:\n{}".format(df_places_4326_geo.columns))
print("CRS of df_places_4326_geo:\n{}\n".format(df_places_4326_geo.crs))
df_places_3857_geo = df_places_4326_geo.to_crs(epsg = '3857')

filename = "Rivers\\rivers_lake_centerlines.shp"
df_rivers_4326_geo = gpd.read_file(filename)
print("Columns of df_rivers_4326_geo:\n{}".format(df_rivers_4326_geo.columns))
print("CRS of df_rivers_4326_geo:\n{}\n".format(df_rivers_4326_geo.crs))
df_rivers_3857_geo = df_rivers_4326_geo[df_rivers_4326_geo.geometry.notnull()].to_crs(epsg = '3857')

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

print("****************************************************")
topic = "2. Import and explore the data"; print("** %s\n" % topic)

crs_projected = {'no_defs': True, 'lat_ts': 5, 'x_0': 0, 'y_0': 0, 'units': 'm', 'datum': 'WGS84', 'lon_0': 0, 'proj': 'merc'}
congo_df = df_countries_4326_geo[df_countries_4326_geo.name=='Congo']

# Set up figure and subplots
fig, axes = plt.subplots(ncols=2, figsize=(11,4))

ax = axes[0]
congo_df.plot(ax=ax, alpha=0.5, color='brown', figsize=(11,4))
df_congopark_4326_geo.plot(ax=ax, color='green', alpha=0.75)
df_congomines_crs84_geo.plot(ax=ax, column='mineral')
#axes[0].set_anchor('N'); 
ax.tick_params(labelsize=7); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.
ax.set_title('Congo Africa')

df_congopark_crs84_geo = df_congopark_4326_geo.to_crs(crs_projected)
ax = axes[1]
df_congopark_crs84_geo.plot(ax=ax, color='green', alpha=0.75)
ax.tick_params(labelsize=5); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.
ax.set_title('Congo Africa - Green areas'); 

plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()


print("****************************************************")
topic = "3. Convert to common CRS and save to a file"; print("** %s\n" % topic)

# Convert both datasets to UTM projection
mining_sites_utm = df_congomines_crs84_geo.to_crs(epsg=32735)
national_parks_utm = df_congopark_crs84_geo.to_crs(epsg=32735)

# Plot the converted data again
ax = national_parks_utm.plot(color='green', alpha=0.75)
mining_sites_utm.plot(ax=ax, color='red')
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Congo'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

# Write converted data to a file
mining_sites_utm.to_file("my_data\\my_data_mines.gpkg", driver='GPKG')
national_parks_utm.to_file("my_data\\my_data_parks.shp", driver='ESRI Shapefile')

print("****************************************************")
topic = "4. Styling a multi-layered plot"; print("** %s\n" % topic)

# Read the national parks of Congo
filename = "my_data\\my_data_parks.shp"
df_national_parks = gpd.read_file(filename)
print("Columns of df_national_parks :\n{}".format(df_national_parks.columns))
print("CRS of df_national_parks :\n{}\n".format(df_national_parks.crs))

# Read the mines place from Congo
filename = "my_data\\my_data_mines.gpkg"
df_mining_sites = gpd.read_file(filename)
print("Columns of df_mining_sites:\n{}".format(df_mining_sites.columns))
print("CRS of df_mining_sites:\n{}\n".format(df_mining_sites.crs))

# Plot of the parks and mining sites
legend_kwds = dict(loc='best', markerscale=0.7, title='Mines Type', fontsize=7, title_fontsize=7)
cmap = colors.ListedColormap(['darkblue','darkgray','brown','gray','gold','blue'])

ax = df_national_parks.plot(color='green')
df_mining_sites.plot(ax=ax, column='mineral', markersize=5, legend=True, legend_kwds = legend_kwds, cmap=cmap)
ax.set_axis_off()
plt.title('Congo'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "6. Buffer around a point"; print("** %s\n" % topic)

city_goma = Point([746989.5594829298, 9816380.942287602])
city_goma_buffer = city_goma.buffer(50000)

# Check how many sites are located within the buffer
mask = df_mining_sites.within(city_goma_buffer)
print("There are {} mines sites in the city of Goma.".format(mask.sum()))

# Calculate the area of national park within the buffer
print("Total area of national parks within city of Goma: {0:,.2f} km\u00b2.".format(df_national_parks.intersection(city_goma_buffer).area.sum() / (1000**2)))

# Plot all sites
legend_kwds = dict(loc='best', markerscale=0.7, title='Mines Type', fontsize=7, title_fontsize=7)
cmap = colors.ListedColormap(['darkblue','darkgray','brown','gray','gold','blue'])

ax = gpd.GeoSeries([city_goma_buffer]).plot(color='red')
df_national_parks.plot(ax=ax, color='green', alpha=0.5)
df_mining_sites.plot(ax=ax, column='mineral', markersize=5, legend=True, legend_kwds = legend_kwds, cmap=cmap, alpha=0.85)
ax.annotate('Goma City', xy=(city_goma.x-35000, city_goma.y-35000), xytext=(-207965, 9270440), arrowprops=dict(color='darkred', alpha=0.5), color='darkred')
ax.set_axis_off()
plt.title('Congo'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "7. Mining sites within national parks"; print("** %s\n" % topic)

# Extract the single polygon for the Kahuzi-Biega National park
kahuzi = df_national_parks[df_national_parks['NOM'] == "Parc National du Kahuzi-Biega"].geometry.squeeze()

# Take a subset of the mining sites located within Kahuzi
sites_kahuzi = df_mining_sites[df_mining_sites.within(kahuzi)]
print("Mining sites located within Kahuzi-Biega National park:\n{}\n\n\n".format(sites_kahuzi))

# Plot all sites
legend_kwds = dict(loc='best', markerscale=0.7, title='Mines Type', fontsize=7, title_fontsize=7)
cmap = colors.ListedColormap(['darkblue', 'gold'])

#ax = gpd.GeoSeries([city_goma_buffer]).plot(color='red')
#ax.annotate('Goma City', xy=(city_goma.x-35000, city_goma.y-35000), xytext=(585733, 9839000), arrowprops=dict(color='darkred', alpha=0.5), color='darkred')
#gpd.GeoSeries([kahuzi]).plot(ax=ax, color='green')
ax = gpd.GeoSeries([kahuzi]).plot(color='green')
sites_kahuzi.plot(ax=ax, column='mineral', markersize=5, legend=True, legend_kwds = legend_kwds, cmap=cmap)
ax.set_axis_off()
plt.title('Kahuzi-Biega National park'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()




# Determine in which national park a mining site is located
sites_within_park = gpd.sjoin(df_mining_sites, df_national_parks, op='within', how='inner')
print("In which national park is the mining site located? (First 5 rows)\n{}\n\n\n".format(sites_within_park.head()))

# The number of mining sites in each national park
print("Number of mining sites in each national park: \n{}\n\n".format(sites_within_park['NOM'].value_counts()))


# Plot all sites
legend_kwds = dict(loc='best', markerscale=0.7, title='Mines Type', fontsize=7, title_fontsize=7)
cmap = colors.ListedColormap(['darkblue','darkgray','brown','gray','gold','blue'])

ax = df_national_parks.plot(column='NOM', cmap='YlGn')
df_mining_sites.plot(ax=ax, column='mineral', markersize=5, legend=True, legend_kwds = legend_kwds, cmap=cmap, alpha=0.85)
ax.set_axis_off()
plt.title('Mining sites in each Congo National Park' ); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()


print("****************************************************")
topic = "8. Applying custom spatial operations"; print("** %s\n" % topic)

distance_of_interest = 100000 #Area of interest, radio of the buffer

#Plot all the places
ax = df_rivers_3857_geo.plot(cmap='ocean')
df_places_3857_geo.geometry.buffer(distance_of_interest).plot(ax=ax, color='lightcoral', alpha=0.75)
df_places_3857_geo.geometry.plot(ax=ax, markersize=1, color='red')
plt.title('Rivers vs Populated Places' ); plt.suptitle(topic);  # Setting the titles.
ax.set_axis_off()
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

def rivers_length(geom, rivers, distance):
    area = geom.buffer(distance)
    rivers_within_area = rivers.intersection(area)
    return rivers_within_area.length.sum()/1000

df_places_3857_geo['river_length'] = df_places_3857_geo.geometry.apply(rivers_length, rivers=df_rivers_3857_geo, distance=distance_of_interest)
print('What is the total river length within 100km of each city? (First 5 rows)\n{}\n'.format(df_places_3857_geo.head()))

print("****************************************************")
topic = "9. Finding the name of the closest National Park"; print("** %s\n" % topic)

# Get the geometry of the first row
single_mine = df_mining_sites.loc[0, 'geometry']

# Calculate the distance from each national park to this mine
dist = df_national_parks.geometry.distance(single_mine)

# The index of the minimal distance
idx = dist.idxmin()

# Access the name of the corresponding national park
closest_park = df_national_parks.loc[idx, 'NOM']
print('The clossest park to the "{}" mine is: {}\n'.format(df_mining_sites.loc[0, 'name'], closest_park))

#Plot all the places
ax = df_national_parks.plot(color='green', alpha=0.75)
gpd.GeoSeries([single_mine]).plot(ax=ax, color='gold')
ax.annotate(closest_park, xy=(df_national_parks.loc[idx,'geometry'].centroid.x, df_national_parks.loc[idx,'geometry'].centroid.y), xytext=(-1035920, 10374400), arrowprops=dict(color='darkgoldenrod', width=0.5, headwidth=3), color='darkgoldenrod', fontsize=7)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.tick_params(labelsize=6); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('The closest park to "{}" mine'.format(df_mining_sites.loc[0, 'name'])); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "10. Applying a custom operation to each geometry"; print("** %s\n" % topic)

# Define a function that returns the closest national park
def closest_national_park(geom, national_parks):
    dist = national_parks.geometry.distance(geom)
    idx = dist.idxmin()
    closest_park = national_parks.loc[idx, 'NOM']
    return closest_park

# Call the function on single_mine
closest_park = closest_national_park(single_mine, df_national_parks)
print('The clossest park to the "{}" mine is: {}'.format(df_mining_sites.loc[0, 'name'], closest_park))

# Apply the function to all mining sites
df_mining_sites['closest_park'] = df_mining_sites.geometry.apply(closest_national_park, national_parks=df_national_parks)
print("The closest park to each mine:\n{}\n...\n{}\n\n".format(df_mining_sites.head(), df_mining_sites.tail()))
print("How many parks are near of mine:\n{}\nThey are:\n{}\n".format(df_mining_sites.closest_park.unique().shape[0], df_mining_sites.closest_park.unique()))

#Plotting all sites
cmap = colors.ListedColormap(['darkblue','darkgray','brown','gray','darkgoldenrod','blue'])
ax = df_national_parks.plot(column='NOM', cmap='YlGn', figsize=(10,5))
df_mining_sites.plot(ax=ax, column='mineral', cmap=cmap, alpha=0.5)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.tick_params(labelsize=6); 
ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Mining sites in each Congo National Park' ); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
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
#from matplotlib                      import colors                            #To create custom cmap
#from matplotlib.ticker               import StrMethodFormatter                #Import the necessary library to delete the scientist notation
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

#To supress the scientist notation in plt
#from matplotlib.ticker import StrMethodFormatter                              #Import the necessary library to delete the scientist notation
#ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation

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
