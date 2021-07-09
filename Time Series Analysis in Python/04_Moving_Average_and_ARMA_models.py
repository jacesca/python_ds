# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:03:16 2019

@author: jacqueline.cortez
Chapter 4: Moving Average (MA) and ARMA Models
    In this chapter you'll learn about another kind of model, the moving average, or MA, model. 
    You will also see how to combine AR and MA models into a powerful ARMA model.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

#import datetime                                                               #For accesing datetime functions
import matplotlib.pyplot              as plt                                  #For creating charts
import numpy                          as np                                   #For making operations in lists
import pandas                         as pd                                   #For loading tabular data

from statsmodels.graphics.tsaplots    import plot_acf                         #For autocorrelation function
from statsmodels.tsa.arima_model      import ARMA                             #To estimate parameters from data simulated (AR model)
from statsmodels.tsa.arima_process    import ArmaProcess                      #For Simulate Autoregressive (AR) Time Series 

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 123
np.random.seed(SEED)

print("****************************************************")
topic = "1. Describe Model"; print("** %s\n" % topic)

ar = np.array([1])
ma = np.array([1, 0.5])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=1000)

plt.plot(simulated_data) # Plot HRB data
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title('Simulated data', color='red'); 
plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "2. Simulate MA(1) Time Series"; print("** %s\n" % topic)

plt.figure()

# Plot 1: MA parameter = -0.9
plt.subplot(2,1,1)
ar1 = np.array([1])
ma1 = np.array([1, -0.9])
MA_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = MA_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title('Simulated data 1 - MA parameter = -0.9', color='red'); 

# Plot 2: MA parameter = +0.9
plt.subplot(2,1,2)
ar2 = np.array([1])
ma2 = np.array([1, 0.9])
MA_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = MA_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title('Simulated data 2 - MA parameter = +0.9', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "3. Compute the ACF for Several MA Time Series"; print("** %s\n" % topic)

# simulated_data_3: MA parameter = -0.3
ar3 = np.array([1])
ma3 = np.array([1, -0.3])
MA_object3 = ArmaProcess(ar3, ma3)
simulated_data_3 = MA_object3.generate_sample(nsample=1000)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

# Plot 1: MA parameter = -0.9
plot_acf(simulated_data_1, lags=20, ax=ax1)
ax1.tick_params(labelsize=7)
ax1.set_title('Simulated data 1 - MA parameter = -0.9', color='red'); 

# Plot 2: MA parameter = 0.9
plot_acf(simulated_data_2, lags=20, ax=ax2)
ax2.tick_params(labelsize=7)
ax2.set_title('Simulated data 2 - MA parameter = +0.9', color='red'); 

# Plot 3: MA parameter = -0.3
plot_acf(simulated_data_3, lags=20, ax=ax3)
ax3.tick_params(labelsize=7)
ax3.set_title('Simulated data 3 - MA parameter = -0.3', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.6);
plt.show()

print("****************************************************")
topic = "6. Estimating an MA Model"; print("** %s\n" % topic)

# Simulating the data
ar1 = np.array([1])
ma1 = np.array([1, -0.9])
MA_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = MA_object1.generate_sample(nsample=1000)

## Import the ARMA module from statsmodels
#from statsmodels.tsa.arima_model import ARMA

# Fit an MA(1) model to the first simulated data
mod = ARMA(simulated_data_1, order=(0,1))
res = mod.fit()

# Print out summary information on the fit
print(res.summary())

# Print out the estimate for the constant and for theta
print("\n\nWhen the true theta=-0.9, the estimate of theta (and the constant) are:")
print(res.params)

print("****************************************************")
topic = "7. Forecasting with MA Model"; print("** %s\n" % topic)

res.plot_predict(start=990, end=1010)

plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title("An example with simulated data", color='red'); 
plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "9. High Frequency Stock Prices"; print("** %s\n" % topic)

file = "Sprint_Intraday.csv" 
intraday = pd.read_csv(file, header=None, usecols=[0,1])
print("INTRADAY DATA (First 5 rows): \n{}".format(intraday.head()))

# Change the first date to zero
intraday.iloc[0,0] = 0

# Change the column headers to 'DATE' and 'CLOSE'
intraday.columns = ['DATE', 'CLOSE']

# Examine the data types for each column
print(intraday.dtypes)

# Convert DATE column to numeric
intraday['DATE'] = pd.to_numeric(intraday.DATE)

# Make the `DATE` column the new index
intraday = intraday.set_index('DATE')

# Examine the data types for each column after changes
print("\nAfter changes...")
print(intraday.dtypes)
print(intraday.info())

print("****************************************************")
topic = "10. More Data Cleaning: Missing Data"; print("** %s\n" % topic)

# Notice that some rows are missing
print("If there were no missing rows, there would be 391 rows of minute data.")
print("The actual length of the DataFrame is:", len(intraday), ".")


set_everything = set(range(391)) # Everything
set_intraday = set(intraday.index) # The intraday index as a set
set_missing = set_everything - set_intraday # Calculate the difference


# Print the difference
print("Missing rows: ", set_missing)


intraday = intraday.reindex(range(391), method='ffill') # Fill in the missing rows
intraday.index = pd.date_range(start='2017-09-01 9:30', end='2017-09-01 16:00', freq='1min') # Change the index to the intraday times


# Plot the intraday time series
intraday.plot(grid=True)

plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title("One day's prices (on September 1, 2017) for Sprint stock ", color='red'); 
plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "11. Applying an MA Model"; print("** %s\n" % topic)

# Compute returns from prices and drop the NaN
returns = intraday.pct_change()
returns = returns.dropna()

# Plot ACF of returns with lags up to 60 minutes
plot_acf(returns, lags=60)
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title('ACF function with lags up to 60 minutes for Intraday', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.show()

# Fit the data to an MA(1) model
mod = ARMA(returns, order=(0,1))
res = mod.fit()
print(res.params)

print("****************************************************")
topic = "12. Equivalence of AR(1) and MA(infinity)"; print("** %s\n" % topic)


# Build a list MA parameters
ma = [0.8**i for i in range(30)]

# Simulate the MA(30) model
ar = np.array([1])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=5000)

# Plot the ACF
plot_acf(simulated_data, lags=30)
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title('Simulated data', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
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
#import datetime                                                               #For accesing datetime functions
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
#from math                            import sqrt
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


#import statsmodels                    as sm                                   #For stimations in differents statistical models
#import statsmodels.api                as sm                                   #Make a prediction model
#import statsmodels.formula.api        as smf                                  #Make a prediction model    
#from statsmodels.graphics.tsaplots    import plot_acf                         #For autocorrelation function
#from statsmodels.graphics.tsaplots    import plot_pacf                        #For simulating data and for plotting the PACF
#from statsmodels.tsa.arima_model      import ARMA                             #To estimate parameters from data simulated (AR model)
#from statsmodels.tsa.arima_process    import ArmaProcess                      #For simulating data and for plotting the PACF, For Simulate Autoregressive (AR) Time Series 
#from statsmodels.tsa.stattools        import acf                              #For autocorrelation function
#from statsmodels.tsa.stattools        import adfuller                         #Augmented Dickey-Fuller Test for Random Walk


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
