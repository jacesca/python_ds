# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:40:47 2019

@author: jacqueline.cortez
Sourde: 
    https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import cartopy                                                                #For working with geospatial data
import matplotlib.pyplot as plt                                               #For creating charts

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

plt.rcParams['figure.max_open_warning'] = 60

print("****************************************************")
topic = "1. PlateCarree Projection"; print("** %s\n" % topic)

nplots = 2
fig = plt.figure(figsize=(10, 5))

for i in range(0, nplots):
    central_longitude = 0 if i == 0 else 180
    ax = fig.add_subplot(1, nplots, i+1, projection=cartopy.crs.PlateCarree(central_longitude=central_longitude))
    ax.coastlines(resolution='110m')
    ax.gridlines()
    plt.title('central_longitude = {}'.format(central_longitude))
plt.suptitle('{}\nThe world Map (2 PlateCarree projections)'.format(topic)); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.show()


print("****************************************************")
topic = "2. Map Projection"; print("** %s\n" % topic)

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(2, 3, 1, projection=cartopy.crs.AlbersEqualArea())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('AlbersEqualArea'); 

ax = fig.add_subplot(2, 3, 2, projection=cartopy.crs.AzimuthalEquidistant(central_latitude=90))
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('AzimuthalEquidistant'); 

ax = fig.add_subplot(2, 3, 3, projection=cartopy.crs.EquidistantConic())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('EquidistantConic'); 

ax = fig.add_subplot(2, 3, 4, projection=cartopy.crs.LambertConformal())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('LambertConformal'); 

ax = fig.add_subplot(2, 3, 5, projection=cartopy.crs.LambertCylindrical())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('LambertCylindrical'); 

ax = fig.add_subplot(2, 3, 6, projection=cartopy.crs.Mercator())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('Mercator'); 

plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "3. Map Projection"; print("** %s\n" % topic)

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(2, 3, 1, projection=cartopy.crs.Miller())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('(Miller'); 

ax = fig.add_subplot(2, 3, 2, projection=cartopy.crs.Mollweide())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('Mollweide'); 

ax = fig.add_subplot(2, 3, 3, projection=cartopy.crs.Orthographic())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('Orthographic'); 

ax = fig.add_subplot(2, 3, 4, projection=cartopy.crs.Robinson())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('Robinson'); 

ax = fig.add_subplot(2, 3, 5, projection=cartopy.crs.Sinusoidal())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('Sinusoidal'); 

ax = fig.add_subplot(2, 3, 6, projection=cartopy.crs.Stereographic())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('Stereographic'); 

plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "4. Map Projection"; print("** %s\n" % topic)

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(2, 3, 1, projection=cartopy.crs.TransverseMercator())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('TransverseMercator'); 

ax = fig.add_subplot(2, 3, 2, projection=cartopy.crs.InterruptedGoodeHomolosine())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('InterruptedGoodeHomolosine'); 

ax = fig.add_subplot(2, 3, 3, projection=cartopy.crs.RotatedPole(pole_latitude=37.5, pole_longitude=177.5))
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('RotatedPole'); 

ax = fig.add_subplot(2, 3, 4, projection=cartopy.crs.OSGB())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('OSGB'); 

ax = fig.add_subplot(2, 3, 5, projection=cartopy.crs.EuroPP())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('EuroPP'); 

ax = fig.add_subplot(2, 3, 6, projection=cartopy.crs.Geostationary())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('Geostationary'); 

plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "5. UTM Projection"; print("** %s\n" % topic)

nplots = 60
fig = plt.figure(figsize=(10, 5))

for i in range(0, nplots):
    ax = fig.add_subplot(1, nplots, i+1, projection=cartopy.crs.UTM(zone=i+1, southern_hemisphere=True))
    ax.coastlines(resolution='110m')
    ax.gridlines()
plt.title('The world Map (UTM projection)'); plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "6. Map Projection"; print("** %s\n" % topic)

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(2, 3, 1, projection=cartopy.crs.EckertI())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('EckertI'); 

ax = fig.add_subplot(2, 3, 2, projection=cartopy.crs.EckertII())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('EckertII'); 

ax = fig.add_subplot(2, 3, 3, projection=cartopy.crs.EckertIII())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('EckertIII'); 

ax = fig.add_subplot(2, 3, 4, projection=cartopy.crs.EckertIV())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('EckertIV'); 

ax = fig.add_subplot(2, 3, 5, projection=cartopy.crs.EckertV())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('EckertV'); 

ax = fig.add_subplot(2, 3, 6, projection=cartopy.crs.EckertVI())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('EckertVI'); 

plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "7. Map Projection"; print("** %s\n" % topic)

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(2, 3, 1, projection=cartopy.crs.NearsidePerspective(central_latitude=50.72, central_longitude=-3.53, satellite_height=10000000.0))
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('NearsidePerspective'); 

ax = fig.add_subplot(2, 3, 2, projection=cartopy.crs.EqualEarth())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('EqualEarth'); 

ax = fig.add_subplot(2, 3, 3, projection=cartopy.crs.Gnomonic())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('Gnomonic'); 

ax = fig.add_subplot(2, 3, 4, projection=cartopy.crs.LambertAzimuthalEqualArea())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('LambertAzimuthalEqualArea'); 

ax = fig.add_subplot(2, 3, 5, projection=cartopy.crs.NorthPolarStereo())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('NorthPolarStereo'); 

ax = fig.add_subplot(2, 3, 6, projection=cartopy.crs.SouthPolarStereo())
ax.coastlines(resolution='110m'); ax.gridlines();
ax.set_anchor('N'); plt.title('SouthPolarStereo'); 

plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "8. OSNI Projection"; print("** %s\n" % topic)

plt.figure(figsize=(10, 5))
ax = plt.axes(projection=cartopy.crs.OSNI())
ax.coastlines(resolution='10m')
ax.gridlines()
plt.title('The world Map (OSNI projection)'); plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import cartopy                                                                #For working with geospatial data
#import inspect                                                                #Used to get the code inside a function
#import folium                                                                 #To create map street folium.__version__'0.10.0'
#import geopandas         as gpd                                               #For working with geospatial data 
#import geoplot                                                                #For working with geospatial data
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt                                               #For creating charts
#import numpy             as np                                                #For making operations in lists
#import pandas            as pd                                                #For loading tabular data
#import seaborn           as sns                                               #For visualizing data
#import pprint                                                                 #Import pprint to format disctionary output

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
