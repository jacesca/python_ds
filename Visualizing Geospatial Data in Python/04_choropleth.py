# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:26:57 2019

@author: jacqueline.cortez
Chapter 4: Creating a choropleth building permit density in Nashville
    In this chapter, you will learn about a special map called a choropleth. Then you will learn and practice 
    building choropleths using two different packages: geopandas and folium.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import folium                                                                 #To create map street
import pandas            as pd                                                #For loading tabular data
import geopandas         as gpd                                               #For working with geospatial data
import matplotlib.pyplot as plt                                               #For creating charts
import pprint                                                                 #Import pprint to format disctionary output

import os
import webbrowser 

from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

pd.set_option("display.max_columns",60)
pd.options.display.float_format = '{:,.4f}'.format

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

def displaying_folium_map(location, filename, zoom_start, child=None, main_marker=True):
    folium_map = folium.Map(location=location, zoom_start=zoom_start)
    if main_marker:
        folium.Marker(location, popup="Here!!!").add_to(folium_map)
    if child != None:
        for item in child:
            item.add_to(folium_map)
    folium_map.save(filename)

    cwd = os.getcwd()
    path=cwd+'\\'+filename
    webbrowser.open('file://' + path)



def displaying_html_folium_map(folium_map, filename):
    folium_map.save(filename)

    cwd = os.getcwd()
    path=cwd+'\\'+filename
    webbrowser.open('file://' + path)


    
print("****************************************************")
topic = "Reading geospatialdata"; print("** %s\n" % topic)

filename = "school_districts.geojson"
df_school_districts = gpd.read_file(filename)
print("Columns of {}:\n{}\n".format(filename, df_school_districts.columns))

filename = "council_districts.geojson"
df_council_districts = gpd.read_file(filename)
print("Columns of {}:\n{}\n".format(filename, df_council_districts.columns))

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "schools.csv"
df_schools = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_schools.columns))
df_schools['geometry'] = df_schools.apply(lambda x: Point((x.Longitude, x.Latitude)), axis=1)          # Calculating the geometry column.
df_schools_geo = gpd.GeoDataFrame(df_schools, crs={'init': 'epsg:4326'}, geometry=df_schools.geometry)   # Transforming to new GeoDataFrame.
print("Columns of df_schools_geo:\n{}\n".format(df_schools.columns))

filename = "building_permits_2017.csv"
df_building_permits = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_building_permits.columns))
df_building_permits['geometry'] = df_building_permits.apply(lambda x: Point((x.lng, x.lat)), axis=1)          # Calculating the geometry column.
df_building_permits_geo = gpd.GeoDataFrame(df_building_permits, crs=df_council_districts.crs, geometry=df_building_permits.geometry)   # Transforming to new GeoDataFrame.
print("Columns of df_building_permits_geo:\n{}\n".format(df_building_permits_geo.columns))

print("****************************************************")
topic = "1. What is a choropleth?"; print("** %s\n" % topic)

############################
## Exploring the data
############################
df_schools_in_district = gpd.sjoin(df_schools_geo, df_school_districts, op='within') # Finding schools in district one.

#Plot the schools in District
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, markerscale=0.7, fontsize=8, title='School Districts') 
ax = df_school_districts.plot(column = 'district', 
                              legend = True, legend_kwds = legend_kwds, cmap='Set3',
                              linewidth=0.3, edgecolor='black') # Plotting the district.
df_schools_in_district.plot(  ax=ax, 
                              cmap='Dark2', edgecolor='white', alpha=0.8) # Plotting the schools in district.
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Schools in Nashville by district'); plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=0.12, bottom=None, right=0.75, top=None, wspace=None, hspace=None);
plt.show()  # Show the plot

############################
## Preparing the data
############################
df_school_counts = df_schools_in_district.groupby(['district']).size().to_frame().reset_index()
df_school_counts.columns = ['district', 'school_count']

df_school_districts_with_counts = pd.merge(df_school_districts, df_school_counts, on='district')
df_school_districts_with_counts['area'] = df_school_districts_with_counts.geometry.area
df_school_districts_with_counts['school_density'] = df_school_districts_with_counts.apply(lambda x: x.school_count/x.area, axis=1)
print('df_school_districts_with_counts:\n{}\n'.format(df_school_districts_with_counts))

print("****************************************************")
topic = "2. Finding counts from a spatial join"; print("** %s\n" % topic)

############################
## Exploring the data
############################
df_permits_by_district = gpd.sjoin(df_building_permits_geo, df_council_districts, op = 'within') # Spatial join of permits_geo and council_districts

#Plot the schools in District
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=3, markerscale=0.7, fontsize=8, title='Council Districts') 
ax = df_council_districts.plot(column = 'district', 
                              legend = True, legend_kwds = legend_kwds, cmap='Set3',
                              linewidth=0.3, edgecolor='black') # Plotting the district.
df_permits_by_district.plot(  ax=ax, 
                              cmap='Dark2', edgecolor='white', alpha=0.8) # Plotting the schools in district.
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Building Permits by district in Nashville'); plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=0.12, bottom=None, right=0.65, top=None, wspace=None, hspace=None);
plt.show()  # Show the plot

############################
## Preparing the data
############################
df_permit_counts = df_permits_by_district.groupby(['district']).size() # Create permit_counts
print('df_permit_counts:\n{}\n'.format(df_permit_counts))

print("****************************************************")
topic = "3. Council district areas and permit counts"; print("** %s\n" % topic)

df_council_districts['area'] = df_council_districts.geometry.area # Create an area column in council_districts
df_permit_counts = df_permit_counts.to_frame() # Convert permit_counts to a DataFrame
df_permit_counts.reset_index(inplace=True) # Reset index and column names
df_permit_counts.columns = ['district', 'bldg_permits']

df_districts_permits = pd.merge(df_council_districts, df_permit_counts, on='district')

print("****************************************************")
topic = "4. Calculating a normalized metric"; print("** %s\n" % topic)

df_districts_permits['permit_density'] = df_districts_permits.apply(lambda x: x.bldg_permits/x.area, axis=1)
print('df_districts_permits:\n{}\n'.format(df_districts_permits))

print("****************************************************")
topic = "5. Choropleths with geopandas"; print("** %s\n" % topic)

############################
## Preparing the data
############################
df_schools_in_district = gpd.sjoin(df_school_districts, df_schools_geo, op='contains') # Finding schools in district one.
df_school_counts = df_schools_in_district.groupby(['district']).size().to_frame().reset_index()
df_school_counts.columns = ['district', 'school_count']
df_school_districts_with_counts = pd.merge(df_school_districts, df_school_counts, on='district')

#Changing to meters coord
gpd.GeoDataFrame.crs = {'init': 'epsg:4326'}
df_school_districts_with_counts.geometry = df_school_districts_with_counts.geometry.to_crs(epsg=3857)
df_school_districts_with_counts['area'] = df_school_districts_with_counts.geometry.area / 10**6
df_school_districts_with_counts['school_density'] = df_school_districts_with_counts.apply(lambda x: x.school_count/x.area, axis=1)

#Getting back to degrees coord
gpd.GeoDataFrame.crs = {'init': 'epsg:3857'}
df_school_districts_with_counts.geometry = df_school_districts_with_counts.geometry.to_crs(epsg=4326)
print('df_school_districts_with_counts:\n{}\n'.format(df_school_districts_with_counts))


############################
## Ploting the choropleth
############################
#With default color
df_school_districts_with_counts.plot(column = 'school_density', edgecolor='black', legend=True) #linewidth=0.3,  Plotting the district.
#plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Schools per km\u00b2.'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.12, bottom=None, right=0.75, top=None, wspace=None, hspace=None);
plt.show()  # Show the plot

#with cmap='BuGn'
df_school_districts_with_counts.plot(column = 'school_density', cmap='BuGn', edgecolor='black', legend=True) #linewidth=0.3,  Plotting the district.
#plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Schools per km\u00b2.'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.12, bottom=None, right=0.75, top=None, wspace=None, hspace=None);
plt.show()  # Show the plot

print("****************************************************")
topic = "6. Geopandas choropleths"; print("** %s\n" % topic)

############################
## Keeping the right measures
############################
#Changing to meters coord
gpd.GeoDataFrame.crs = {'init': 'epsg:4326'}
df_districts_permits.geometry = df_districts_permits.geometry.to_crs(epsg=3857)
df_districts_permits['area'] = df_districts_permits.geometry.area / 10**6
df_districts_permits['permit_density'] = df_districts_permits.apply(lambda x: x.bldg_permits/x.area, axis=1)

#Getting back to degrees coord
gpd.GeoDataFrame.crs = {'init': 'epsg:3857'}
df_districts_permits.geometry = df_districts_permits.geometry.to_crs(epsg=4326)

############################
## Ploting the choropleth
############################
df_districts_permits.plot(column = 'permit_density', cmap='BuGn', edgecolor='black', legend=True) #linewidth=0.3,  Plotting the district.
#plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xticks(rotation = 'vertical')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('2017 Building Project Density by Council District'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.12, bottom=None, right=0.75, top=None, wspace=None, hspace=None);
plt.show()  # Show the plot

print("****************************************************")
topic = "7. Area in km squared, geometry in decimal degrees"; print("** %s\n" % topic)

# Change council_districts crs to epsg 3857
gpd.GeoDataFrame.crs = {'init': 'epsg:4326'}
df_council_districts = df_council_districts.to_crs(epsg = 3857)

# Create area in square km
sqm_to_sqkm = 10**6
df_council_districts['area'] = df_council_districts.geometry.area / sqm_to_sqkm

# Change council_districts crs back to epsg 4326
gpd.GeoDataFrame.crs = {'init': 'epsg:3857'}
df_council_districts = df_council_districts.to_crs(epsg = 4326)

print("****************************************************")
topic = "8. Spatially joining and getting counts"; print("** %s\n" % topic)

df_building_permits_geo = gpd.GeoDataFrame(df_building_permits, crs=df_council_districts.crs, geometry=df_building_permits.geometry)   # Transforming to new GeoDataFrame.
df_permits_by_district = gpd.sjoin(df_building_permits_geo, df_council_districts, op = 'within') # Spatial join of permits_geo and council_districts
df_permit_counts = df_permits_by_district.groupby(['district']).size().to_frame().reset_index() # Create permit_counts
df_permit_counts.columns = ['district', 'bldg_permits']

print("****************************************************")
topic = "9. Building a polished Geopandas choropleth"; print("** %s\n" % topic)

df_districts_permits = pd.merge(df_council_districts, df_permit_counts, on='district') # Merge permits_by_district and counts_df
df_districts_permits['permit_density'] = df_districts_permits.apply(lambda row: row.bldg_permits / row.area, axis = 1)


############################
## Ploting the choropleth
############################
df_districts_permits.plot(column = 'permit_density', cmap='OrRd', edgecolor='black', legend=True) #linewidth=0.3,  Plotting the district.
#plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xticks(rotation = 'vertical')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('2017 Building Project Density by Council District'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.12, bottom=None, right=0.75, top=None, wspace=None, hspace=None);
plt.show()  # Show the plot

print("****************************************************")
topic = "10. Choropleths with folium"; print("** %s\n" % topic)
"""
folium_map = folium.Map(location=[36.1636, -86.7823], zoom_start=10)
folium.Choropleth(geo_data     = df_school_districts_with_counts,
                  name         = 'geometry',
                  data         = df_school_districts_with_counts,
                  columns      = ['district', 'school_density'],
                  key_on       = 'feature.properties.district',
                  fill_color   = 'YlGn',
                  fill_opacity = 0.75,
                  line_opacity = 0.5,
                  legend_name  = 'Schools per km squared by School District').add_to(folium_map)
folium.LayerControl().add_to(folium_map)
displaying_html_folium_map(folium_map, '04_10_cloropleths.html')
"""
#Displaying the folium map of the school density by district
child = [folium.Choropleth(geo_data     = df_school_districts_with_counts,
                           name         = 'geometry',
                           data         = df_school_districts_with_counts,
                           columns      = ['district', 'school_density'],
                           key_on       = 'feature.properties.district',
                           fill_color   = 'YlGn',
                           fill_opacity = 0.75,
                           line_opacity = 0.5,
                           legend_name  = 'Schools per km squared by School District'),
         folium.LayerControl()]
displaying_folium_map([36.1636, -86.7823], '04_10_cloropleths.html', zoom_start=10, child=child, main_marker=False)

print("****************************************************")
topic = "11. Folium choropleth"; print("** %s\n" % topic)

#Displaying the folium map of the building permits by district
my_cloropleth = folium.Choropleth(geo_data     = df_districts_permits,
                                  name         = 'geometry',
                                  data         = df_districts_permits,
                                  columns      = ['district', 'permit_density'],
                                  key_on       = 'feature.properties.district',
                                  fill_color   = 'Reds',
                                  fill_opacity = 0.5,
                                  line_opacity = 1.0,
                                  legend_name  = '2017 Permitted Building Projects per km squared')

child = [my_cloropleth, folium.LayerControl()]
pprint.pprint(child) #Only to check.

displaying_folium_map([36.1636,-86.7823], '04_11_cloropleths.html', zoom_start=10, child=child, main_marker=False)

print("****************************************************")
topic = "12. Folium choropleth with markers and popups"; print("** %s\n" % topic)

#Displaying the folium map of the building permits by district
my_cloropleth = folium.Choropleth(geo_data     = df_districts_permits,
                                  name         = 'geometry',
                                  data         = df_districts_permits,
                                  columns      = ['district', 'permit_density'],
                                  key_on       = 'feature.properties.district',
                                  fill_color   = 'Reds',
                                  fill_opacity = 0.5,
                                  line_opacity = 1.0,
                                  legend_name  = '2017 Permitted Building Projects per km squared')


df_districts_permits['center'] = df_districts_permits['geometry'].centroid
child = [folium.Marker(location = [row['center'].y, row['center'].x], 
                       popup = ('<strong>District:</strong><span style="color:red">' + str(row['district']) + '</span>;  ' + '<strong>Permits issued:</strong><span style="color:red">' + str(row['bldg_permits']) + '</span>')) for index, row in df_districts_permits.iterrows()]
child.insert(0, folium.LayerControl())
child.insert(0, my_cloropleth)
pprint.pprint(child) #Only to check.

displaying_folium_map([36.1636,-86.7823], '04_12_cloropleths.html', zoom_start=10, child=child, main_marker=False)

print("****************************************************")
print("** END                                            **")
print("****************************************************")

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
