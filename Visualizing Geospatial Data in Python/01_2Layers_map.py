# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:04:25 2019

@author: jacqueline.cortez
Chapter 1: Building 2-layer maps : combining polygons and scatterplots
    In this chapter, you will learn how to create a two-layer map by first plotting regions from a shapefile and 
    then plotting location points as a scatterplot.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import pandas            as pd                                                #For loading tabular data
import geopandas         as gpd                                               #For working with geospatial data
import matplotlib.pyplot as plt                                               #For creating charts

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

pd.set_option("display.max_columns",20)

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "father_son_height.csv"
father_son = pd.read_csv(filename, sep=';')
print("Columns of {}:\n{}\n".format(filename, father_son.columns))

filename = "public_art.csv"
df_art = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_art.columns))

df_mini_art = pd.DataFrame({'StopID':[4431,  588,  590,  541],
                            'StopName':['MUSIC CITY CENTRAL 5TH - BAY 11', 'CHARLOTTE AVE & 7TH AVE N WB',
                                        'CHARLOTTE AVE & 8TH AVE N WB', '11TH AVE / N GULCH STATION OUTBOUND'],
                            'Location':[(36.16659, -86.781996), (36.165, -86.78406), 
                                        (36.164393, -86.785451), (36.162249, -86.790464)]})
filename = 'Hen_Permits.csv'
df_chickens = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_chickens.columns))

filename = 'schools.csv'
df_schools = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_schools.columns))


print("****************************************************")
topic = "Reading geospatialdata"; print("** %s\n" % topic)

filename = "shapefile_path\\nashville.shp"
df_service_district = gpd.read_file(filename)
print("Columns of df_service_district:\n{}\n".format(df_service_district.columns))

filename = "school_districts.geojson"
df_school_districts = gpd.read_file(filename)
print("Columns of {}:\n{}\n".format(filename, df_school_districts.columns))
 
filename = "shapefile_path\\SLV_deptos.shp"
df_slv_deptos = gpd.read_file(filename, encoding = 'utf-8')
print("Columns of df_slv_deptos:\n{}\n".format(df_slv_deptos.columns))

print("****************************************************")
topic = "3. Styling a scatterplot"; print("** %s\n" % topic)

# Scatterplot 1 - father heights vs son heights with darkred square markers
plt.scatter(father_son.fheight, father_son.sheight, color = 'darkred', marker = 's', alpha=0.75)
plt.grid(False)
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot


# Scatterplot 2 - yellow markers with darkblue borders
plt.figure()
plt.scatter(father_son.fheight, father_son.sheight, c = 'yellow', edgecolor = 'darkblue')
plt.grid(False)
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot


# Scatterplot 3
plt.figure()
plt.scatter(father_son.fheight, father_son.sheight,  c = 'yellow', edgecolor = 'darkblue')
plt.grid(True)
plt.xlabel('father height (inches)')
plt.ylabel('son height (inches)')
plt.title('Son Height as a Function of Father Height')
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "4. Extracting longitude and latitude"; print("** %s\n" % topic)

print('Head of df_mini_art before extraction:\n{}'.format(df_mini_art.head())) # print the first few rows of df 
df_mini_art['lat'] = [loc[0] for loc in df_mini_art.Location] # extract latitude to a new column: lat
df_mini_art['lng'] = [loc[1] for loc in df_mini_art.Location] # extract longitude to a new column: lng
print('\nHead of df_mini_art after extraction:\n{}'.format(df_mini_art.head())) # print the first few rows of df again

print("****************************************************")
topic = "5. Plotting chicken locations"; print("** %s\n" % topic)

print('Head of chickens dataset:\n{}'.format(df_chickens.head())) # Look at the first few rows of the chickens DataFrame

# Plot the locations of all Nashville chicken permits
plt.scatter(x = df_chickens.lng, y = df_chickens.lat,  c = 'yellow', edgecolor = 'darkblue')
plt.grid(True)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('Location of the Nashville chickens')
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "7. Creating a GeoDataFrame & examining the geometry"; print("** %s\n" % topic)

print('Shape of df_service_district: {}'.format(df_service_district)) 
print('\ndf_service_district without geometry:\n{}'.format(df_service_district[['area_sq_mi', 'name', 'objectid']]))  

print("****************************************************")
topic = "8. Plotting shapefile polygons"; print("** %s\n" % topic)

# Plot the Service Districts without any additional arguments
df_service_district.plot()
plt.grid(True)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('Nashville')
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

# Plot the Service Districts, color them according to name, and show a legend
df_service_district.plot(column = 'name', legend = True)
plt.grid(True)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('Nashville')
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "9. Scatterplots over polygons"; print("** %s\n" % topic)

df_school_districts.plot(column='district', legend=True, cmap='Set2')
plt.scatter(df_schools.Longitude, df_schools.Latitude, marker='p', c='darkgreen')
plt.grid(True)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('Nashville Schools and School Districts')
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot


#Plotting EL SALVADOR
df_slv_deptos.query('SDP != "Desconocido"').plot(column='NA2', 
                    legend=True, legend_kwds=dict(fontsize=5, ncol=2, markerscale=0.5),
                    cmap='tab20')
plt.grid(True)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('El Salvador')
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "11. Plotting points over polygons - part 1"; print("** %s\n" % topic)

df_service_district.plot(column='name') # Plot the service district shapefile
plt.scatter(x=df_chickens.lng, y=df_chickens.lat, c = 'black', alpha=0.5) # Add the chicken locations
plt.grid(True)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('Chicken locations in Nashville')
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "12. Plotting points over polygons - part 2"; print("** %s\n" % topic)

df_service_district.plot(column='name') # Plot the service district shapefile
plt.scatter(x=df_chickens.lng, y=df_chickens.lat, c = 'black', alpha=0.5, edgecolor = 'white') # Add the chicken locations
plt.title('Nashville Chicken Permits') # Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True) # Add grid lines and show the plot
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import inspect                                                                #Used to get the code inside a function
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt                                               #For creating charts
#import numpy             as np                                                #For making operations in lists
#import pandas            as pd                                                #For loading tabular data
#import seaborn           as sns                                               #For visualizing data
#import geopandas         as gpd                                               #For working with geospatial data 

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

#warnings.filterwarnings('ignore', 'Objective did not converge*')              #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
#warnings.filterwarnings('default', 'Objective did not converge*')             #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


#Create categorical type data to use
#cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  ordered=True)
# Change the data type of 'rating' to category
#weather['rating'] = weather.rating.astype(cats)