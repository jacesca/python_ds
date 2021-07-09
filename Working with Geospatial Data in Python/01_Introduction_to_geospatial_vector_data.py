# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:25:18 2019

@author: jacqueline.cortez
Chapter 1: Introduction to geospatial vector data
    In this chapter, you will be introduced to the concepts of geospatial data, and more specifically of vector data. 
    You will then learn how to represent such data in Python using the GeoPandas library, and the basics to read, 
    explore and visualize such data. And you will exercise all this with some datasets about the city of Paris.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot as plt                                               #For creating charts
import pandas            as pd                                                #For loading tabular data
import contextily                                                             #To add a background web map to our plot
import geopandas         as gpd                                               #For working with geospatial data

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
print("Columns of {}:\n{}\n".format(filename, df_paris_districts_geo.columns))

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "paris_restaurants.csv"
df_restaurants = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_restaurants.columns))

print("****************************************************")
topic = "2. Restaurants in Paris"; print("** %s\n" % topic)

print("Head of paris_restaurants:\n{}\n".format(df_restaurants.head()))

# Make a plot of all points
fig, ax = plt.subplots()
ax.plot(df_restaurants.x, df_restaurants.y, marker='o', alpha=0.5, linestyle='None')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Restaurants in Paris'); plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "3. Adding a background map"; print("** %s\n" % topic)

# A figure of all restaurants with background
fig, ax = plt.subplots()
ax.plot(df_restaurants.x, df_restaurants.y, marker='o', alpha=0.5, linestyle='None', markersize=1)
contextily.add_basemap(ax)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Restaurants in Paris'); plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "4. Introduction to GeoPandas"; print("** %s\n" % topic)

df_countries_geo.plot()
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('World Map'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

df_countries_geo['area'] = df_countries_geo.geometry.area
print("Head of df_countries:\n{}\n".format(df_countries_geo.head()))

print("****************************************************")
topic = "5. Explore the Paris districts (I)"; print("** %s\n" % topic)

# Inspect the first rows
print("Head of df_paris_districts:\n{}\n".format(df_paris_districts_geo.head()))

# Make a quick visualization of the districts
df_paris_districts_geo.plot()
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Paris Districts'); plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "6. Explore the Paris districts (II)"; print("** %s\n" % topic)

df_paris_districts_geo['area'] = df_paris_districts_geo.geometry.area
print("Head of df_paris_districts:\n{}\n".format(df_paris_districts_geo.head()))

print("****************************************************")
topic = "7. The Paris restaurants as a GeoDataFrame"; print("** %s\n" % topic)

# Convert it to a GeoDataFrame
df_restaurants_geo = gpd.GeoDataFrame(df_restaurants, geometry=gpd.points_from_xy(df_restaurants.x, df_restaurants.y))
print("Head of df_restaurants_geo:\n{}\n".format(df_restaurants_geo.head()))

# Make a plot of the restaurants
ax = df_restaurants_geo.plot(markersize=1)
contextily.add_basemap(ax)
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Restaurants in Paris'); plt.suptitle(topic);  # Setting the titles.
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "8. Exploring and visualizing spatial data"; print("** %s\n" % topic)

#Filtering data
df_africa_geo = df_countries_geo[df_countries_geo.continent == 'Africa']
df_africa_geo.plot(color='red', edgecolor='black')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Africa Continent'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()


#Coloring the world map
df_countries_geo.plot(column='continent', edgecolor='black')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('World Map (Coloring by continent)'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()


#Coloring the world map
df_countries_geo.plot(column='gdp_md_est', edgecolor='black')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('World Map (Coloring by gdp_md_est)'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()


#Multilayered plot
fig, ax = plt.subplots(figsize=(10,4))
df_countries_geo.plot(ax = ax)
df_cities_geo.plot(ax=ax, color='red', markersize=10)
ax.set_axis_off()
plt.xlabel('Longitude'); plt.ylabel('Latitude'); # Labeling the axis.
plt.title('Main cities in the world'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "9. Visualizing the population density"; print("** %s\n" % topic)

df_paris_districts_geo['population_density'] = df_paris_districts_geo.population / df_paris_districts_geo.area * (10**6) # Add a population density column

# Make a plot of the districts colored by the population density
fig, ax = plt.subplots(figsize=(10,4))
base = df_paris_districts_geo.plot(ax = ax, column='population_density', legend=True)
#ax.set_anchor('N'); 
#Changing the fontsize of the color bar: (Source: https://gist.github.com/sebbacon/60ceec549f1d461af4543ccc5024c095?short_path=839cef5)
#     The Figure has two Axes: one for the map, and one for the Colorbar. The one we care about is the second one. 
#     cb_ax = fig.axes[1] #Axes for the colorbar.
#     The legend is actually a Colorbar object. To change the legend's font size, we have to get hold of the Colorbar's 
#     Axes object, and call .tick_params() on that. 
fig.axes[1].tick_params(labelsize=7) #Changing the fontsize of the colorbar in second axes of the graph.
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Density population in Paris'); plt.suptitle(topic);  # Setting the titles.
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "10. Using pandas functionality: groupby"; print("** %s\n" % topic) 

restaurants_type_counts = df_restaurants_geo.groupby('type').size() # Calculate the number of restaurants of each type

print("Restaurants by type (Using groupb)y:\n{}\n".format(restaurants_type_counts)) # Print the result
print("Restaurants by type (Using value_counts method):\n{}\n".format(df_restaurants_geo['type'].value_counts())) # Print the result

print("****************************************************")
topic = "11. Plotting multiple layers"; print("** %s\n" % topic)

# Take a subset of the African restaurants
df_african_restaurants_geo = df_restaurants_geo[df_restaurants_geo['type']=='African restaurant']

# Make a multi-layered plot
fig, ax = plt.subplots(figsize=(10, 5))
df_restaurants_geo.plot(ax=ax, color='darkgreen', markersize=1)
df_african_restaurants_geo.plot(ax=ax, color='red', markersize=3)
ax.set_axis_off()
contextily.add_basemap(ax)
plt.title('African Restaurants in Paris'); plt.suptitle(topic);  # Setting the titles.
plt.show()


#Making one more example:
#Source: http://datos.mop.gob.sv/?q=search/field_topic/proyectos-de-inversi%C3%B3n-6
filename = "SV_proyectos_MOP.csv"
df_mop_sv = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_mop_sv.columns))
df_mop_sv_geo = gpd.GeoDataFrame(df_mop_sv, geometry=gpd.points_from_xy(df_mop_sv.LONGITUD, df_mop_sv.LATITUD), crs={'init': 'epsg:4326'})
print("Columns of df_mop_sv_geo:\n{}\n".format(df_mop_sv_geo.columns))
gpd.GeoDataFrame.crs = {'init': 'epsg:4326'}
df_mop_sv_geo = df_mop_sv_geo.to_crs(epsg = 3857)

#print(dir(contextily.tile_providers)) # To use in url=contextily.tile_providers.OSM_A
#print(dir(contextily.providers))      # To use in url=contextily.providers.Stamen.TonerLite

# Make a multi-layered plot
ax = df_mop_sv_geo.plot(color='red', figsize=(10, 5))
contextily.add_basemap(ax) #, url=contextily.providers.Stamen.TonerLite)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude', fontsize=7); plt.ylabel('Latitude', fontsize=7); # Labeling the axis.
plt.title('Proyectos MOP en Ejecución (El Salvador 2018)'); plt.suptitle(topic);  # Setting the titles.
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
