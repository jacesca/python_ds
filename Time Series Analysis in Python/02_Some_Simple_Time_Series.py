# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:03:16 2019

@author: jacqueline.cortez
Chapter 2: Some Simple Time Series
    In this chapter you'll learn about some simple time series models. These include white noise and a random walk.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot              as plt                                  #For creating charts
import pandas                         as pd                                   #For loading tabular data
import numpy                          as np                                   #For making operations in lists
import pprint                                                                 #Import pprint to format disctionary output

from math                             import sqrt
from matplotlib.ticker                import StrMethodFormatter               #Import the necessary library to delete the scientist notation
from statsmodels.graphics.tsaplots    import plot_acf                         #For autocorrelation function
from statsmodels.tsa.stattools        import acf                              #For autocorrelation function
from statsmodels.tsa.stattools        import adfuller                         #Augmented Dickey-Fuller Test for Random Walk

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 123
np.random.seed(SEED)

print("****************************************************")
topic = "2. Taxing Exercise: Compute the ACF"; print("** %s\n" % topic)

file = "HRB.data" 
HRB = pd.read_fwf(file, index_col = 'Quarter', parse_dates=True, infer_datetime_format=True)

# Compute the acf array of HRB
acf_array = acf(HRB, fft=False)
print("ACF:\n{}\n".format(acf_array))

# Plot the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

HRB.plot(grid=True, ax=ax1) # Plot HRB data
ax1.tick_params(labelsize=8)
ax1.set_xlabel('Quarter'); ax1.set_ylabel('Earnings'); # Labeling the axis.
ax1.set_title('Evolution of Earings', color='red'); 

plot_acf(HRB, fft=False, lags=40, alpha=1, ax=ax2) # Plot the acf function
ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
ax2.tick_params(labelsize=8)
ax2.set_title('Autocorrelation', color='red'); 


plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "3. Are We Confident This Stock is Mean Reverting?"; print("** %s\n" % topic)

file = "MSFT.data" 
MSFT = pd.read_fwf(file, index_col = 'Date', parse_dates=True, infer_datetime_format=True)

MSFT = MSFT.resample(rule='W').last() # Convert the daily data to weekly data
returns = MSFT.pct_change().dropna() # Compute the percentage change of prices
autocorrelation = returns.Adj_Close.autocorr() # Compute and print the autocorrelation of returns
print("The autocorrelation of weekly returns is %4.2f \n" %(autocorrelation)) #This is negative so is mean reverting.




# Find the number of observations by taking the length of the returns DataFrame
nobs = len(returns)

# Compute the approximate confidence interval
conf = 1.96/sqrt(nobs)
print("The approximate confidence interval is +/- %4.2f \n" %(conf))



# Plot the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

returns.plot(grid=True, ax=ax1) # Plot the entire time series diet and show gridlines
ax1.tick_params(labelsize=8)
ax1.set_xlabel('Period Time'); ax1.set_ylabel('MSFT Returns'); # Labeling the axis.
ax1.set_title('Correlation detected: {0:.4f}'.format(autocorrelation), color='red'); 

plot_acf(returns, fft=False, alpha=0.05, lags=20, ax=ax2) # Plot the acf function
ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
ax2.tick_params(labelsize=8)
ax2.set_title('Autocorrelation', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "4. White Noise"; print("** %s\n" % topic)

noise = np.random.normal(loc=0, scale=1, size=500)

#Plot all data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

ax1.plot(noise) # Plot the entire time series diet and show gridlines
ax1.grid()
ax1.tick_params(labelsize=8)
#ax1.set_xlabel('Period Time'); ax1.set_ylabel('MSFT Returns'); # Labeling the axis.
ax1.set_title('Noise data example', color='red'); 

plot_acf(noise, fft=False, alpha=0.05, lags=50, ax=ax2) # Plot the acf function
#ax2.grid()
ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
ax2.tick_params(labelsize=8)
ax2.set_title('Autocorrelation', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "5. Can't Forecast White Noise"; print("** %s\n" % topic)

# Simulate white noise returns
returns = np.random.normal(loc=0.02, scale=0.05, size=1000)

# Print out the mean and standard deviation of returns
mean = np.mean(returns)
std = np.std(returns)
print("The mean is %5.3f and the standard deviation is %5.3f\. \n" %(mean,std))


#Plot all data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
ax1.plot(returns) 
ax1.grid()
ax1.tick_params(labelsize=8)
#ax1.set_xlabel('Period Time'); ax1.set_ylabel('MSFT Returns'); # Labeling the axis.
ax1.set_title('Noise data example', color='red'); 

plot_acf(noise, fft=False, alpha=0.05, lags=20, ax=ax2) # Plot the acf function
#ax2.grid()
ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
ax2.tick_params(labelsize=8)
ax2.set_title('Autocorrelation', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "6. Random Walk"; print("** %s\n" % topic)

file = "stocks.data" 
stocks = pd.read_fwf(file, index_col = 'observation_date', parse_dates=True, infer_datetime_format=True)

returns = stocks.pct_change().dropna() # Compute the percentage change of prices
autocorrelation = returns.SP500.autocorr() # Compute and print the autocorrelation of returns
print("The autocorrelation of weekly returns is %4.2f \n" %(autocorrelation)) #This is negative so is mean reverting.

results = adfuller(stocks.SP500)
print("p-value: {0:,.4f}".format(results[1]))
print("ADF test result for Random Walk:")
pprint.pprint(results)
print("")

#Plot all data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.5))
stocks.plot(ax=ax1) 
ax1.grid()
ax1.tick_params(labelsize=8)
ax1.set_xlabel('Period Time'); ax1.set_ylabel('SP500'); # Labeling the axis.
ax1.set_title('SP500 data (Random Walk)', color='red'); 

returns.plot(ax=ax2) 
ax2.grid()
ax2.tick_params(labelsize=8)
ax2.set_xlabel('Period Time'); ax1.set_ylabel('SP500 Returns'); # Labeling the axis.
ax2.set_title('SP500 data (Returns)', color='red'); 

plot_acf(returns, fft=False, alpha=0.05, lags=20, ax=ax3) # Plot the acf function
#ax2.grid()
ax3.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
ax3.tick_params(labelsize=8)
ax3.set_title('Autocorrelation', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "7. Generate a Random Walk"; print("** %s\n" % topic)

# Generate 500 random steps with mean=0 and standard deviation=1
steps = np.random.normal(loc=0, scale=1, size=500)

# Set first element to 0 so that the first price will be the starting stock price
steps[0]=0

# Simulate stock prices, P with a starting price of 100
P = 100 + np.cumsum(steps)

# Plot the simulated stock prices
plt.figure()
plt.plot(P)
#plt.xlabel('Stock prices'); plt.ylabel('Bond Prices'); # Labeling the axis.
plt.title("Simulated Random Walk", color='red')
plt.suptitle(topic, color='navy', fontweight="bold");  # Setting the titles.
#plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "8. Get the Drift"; print("** %s\n" % topic)

# Generate 500 random steps
steps = np.random.normal(loc=0.001, scale=0.01, size=500) + 1

# Set first element to 1
steps[0]=1

# Simulate the stock price, P, by taking the cumulative product
P = 100 * np.cumprod(steps)

# Plot the simulated stock prices
plt.figure()
plt.plot(P)
#plt.xlabel('Stock prices'); plt.ylabel('Bond Prices'); # Labeling the axis.
plt.title("Simulated Random Walk with Drift", color='red')
plt.suptitle(topic, color='navy', fontweight="bold");  # Setting the titles.
#plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "9. Are Stock Prices a Random Walk?"; print("** %s\n" % topic)

file = "AMZN.csv" 
AMZN = pd.read_csv(file, index_col = 'Date', parse_dates=True, infer_datetime_format=True)

# Run the ADF test on the price series and print out the results
results = adfuller(AMZN['Adj Close'])
print("ADF test result for Amazon stock prices (Random Walk):")
pprint.pprint(results)
print("")

# Just print out the p-value
print('The p-value of the test on prices is: ' + str(results[1]))

returns = AMZN.pct_change().dropna()

#Plot all data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.5))
AMZN.plot(ax=ax1) 
ax1.grid()
ax1.tick_params(labelsize=8)
ax1.set_xlabel('Period Time'); ax1.set_ylabel('Amazon stock prices'); # Labeling the axis.
ax1.set_title('Amazon stock prices (Random Walk)', color='red'); 

returns.plot(ax=ax2) 
ax2.grid()
ax2.tick_params(labelsize=8)
ax2.set_xlabel('Period Time'); ax1.set_ylabel('Returns'); # Labeling the axis.
ax2.set_title('Amazon stock Returns', color='red'); 

plot_acf(returns, fft=False, alpha=0.05, lags=20, ax=ax3) # Plot the acf function
#ax2.grid()
ax3.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
ax3.tick_params(labelsize=8)
ax3.set_title('Autocorrelation', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "10. How About Stock Returns?"; print("** %s\n" % topic)

# Run the ADF test on the price series and print out the results
results = adfuller(returns['Adj Close'])
print("ADF test result for Amazon stock returns (Not a Random Walk):")
pprint.pprint(results)
print("")

# Just print out the p-value
print('The p-value of the test on prices is: ' + str(results[1]))

print("****************************************************")
topic = "13. Seasonal Adjustment During Tax Season"; print("** %s\n" % topic)

file = "HRB.data" 
HRB = pd.read_fwf(file, index_col = 'Quarter', parse_dates=True, infer_datetime_format=True)

HRBsa = HRB.diff(4) # Seasonally adjust quarterly earnings
print("First 10 rows of the seasonally adjusted series:\n{}".format(HRBsa.head(10))) # Print the first 10 rows of the seasonally adjusted series

HRBsa = HRBsa.dropna() # Drop the NaN data in the first four rows


#Plot all data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(HRB, fft=False, lags=40, alpha=0.05, ax=ax1) # Plot the acf function
ax1.tick_params(labelsize=8)
ax1.set_xlabel('Period'); ax1.set_ylabel('H&R earnings'); # Labeling the axis.
ax1.set_title("ACF of H&R Block's quarterly earnings", color='red', fontsize=10); 

# Plot the autocorrelation function of the seasonally adjusted series
plot_acf(HRBsa, fft=False, lags=38, alpha=0.05, ax=ax2) # Plot the acf function
plt.xticks(fontsize=8);
ax2.set_xlabel('Period'); ax2.set_ylabel('H&R earnings'); # Labeling the axis.
ax2.set_title("ACF of H&R Block's seasonal adjustment earnings", color='red', fontsize=10); 

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=0.5, hspace=None);
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
#from statsmodels.tsa.stattools        import acf                              #For autocorrelation function
#from statsmodels.graphics.tsaplots    import plot_acf                         #For autocorrelation function
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
