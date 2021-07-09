# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:03:16 2019

@author: jacqueline.cortez
Chapter 3: Autoregressive (AR) Models
    In this chapter you'll learn about autoregressive, or AR, models for time series. 
    These models use past values of the series to predict the current value.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot              as plt                                  #For creating charts
import numpy                          as np                                   #For making operations in lists
import pandas                         as pd                                   #For loading tabular data

from statsmodels.graphics.tsaplots    import plot_acf                         #For autocorrelation function
from statsmodels.graphics.tsaplots    import plot_pacf                        #Import the modules for simulating data and for plotting the PACF
from statsmodels.tsa.arima_model      import ARMA                             #To estimate parameters from data simulated (AR model)
from statsmodels.tsa.arima_process    import ArmaProcess                      #For Simulate Autoregressive (AR) Time Series 

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 123
np.random.seed(SEED)

print("****************************************************")
topic = "1. Describe AR Model"; print("** %s\n" % topic)

# AR parameter = +0.9
ar = np.array([1, -0.9])
ma = np.array([1])

AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=1000)

plt.plot(simulated_data) # Plot HRB data
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title('Simulated data', color='red'); 
plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "2. Simulate AR(1) Time Series"; print("** %s\n" % topic)

plt.figure(figsize=[10,4])
# Plot 1: AR parameter = +0.9
plt.subplot(2,2,1)
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title('Phi=0.9', color='red'); 

# Plot 2: AR parameter = -0.9
plt.subplot(2,2,2)
ar2 = np.array([1, 0.9])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = AR_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title('Phi=-0.9', color='red'); 

# Plot 1: AR parameter = +0.5
plt.subplot(2,2,3)
ar4 = np.array([1, -0.5])
ma4 = np.array([1])
AR_object4 = ArmaProcess(ar4, ma4)
simulated_data_4 = AR_object4.generate_sample(nsample=1000)
plt.plot(simulated_data_4)
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title('Phi=0.5', color='red'); 

# Plot 2: AR parameter = -0.5
plt.subplot(2,2,4)
ar5 = np.array([1, 0.5])
ma5 = np.array([1])
AR_object5 = ArmaProcess(ar5, ma5)
simulated_data_5 = AR_object5.generate_sample(nsample=1000)
plt.plot(simulated_data_5)
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.title('Phi=-0.5', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "3. Compare the ACF for Several AR Time Series"; print("** %s\n" % topic)

# Plot 3: AR parameter = +0.3
ar3 = np.array([1, -0.3])
ma3 = np.array([1])
AR_object3 = ArmaProcess(ar3, ma3)
simulated_data_3 = AR_object3.generate_sample(nsample=1000)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11,3))

# Plot 1: AR parameter = +0.9
plot_acf(simulated_data_1, alpha=1, lags=20, ax=ax1)
ax1.tick_params(labelsize=7)
ax1.set_title('Phi=+0.9', color='red'); 

# Plot 2: AR parameter = -0.90
plot_acf(simulated_data_2, alpha=1, lags=20, ax=ax2)
ax2.tick_params(labelsize=7)
ax2.set_title('Phi=-0.9', color='red'); 

# Plot 3: AR parameter = +0.3
plot_acf(simulated_data_3, alpha=1, lags=20, ax=ax3)
ax3.tick_params(labelsize=7)
ax3.set_title('Phi=+0.3', color='red'); 

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "5. Estimating and Forecasting AR Model"; print("** %s\n" % topic)

mod = ARMA(simulated_data, order=(1,0))
result = mod.fit()

print("Estimating an AR model: \n{}\n\n".format(result.summary()))
print("Printing only the parameters: {}\n".format(result.params))

#Forecastin the AR model (simulated data)
result.plot_predict(start=990, end=1010)
plt.title('Phi=+0.9', color='red'); 
plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "6. Estimating an AR Model"; print("** %s\n" % topic)

# Fit an AR(1) model to the first simulated data
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()

# Print out summary information on the fit
print(res.summary())

# Print out the estimate for the constant and for phi
print("\n\nWhen the true phi=0.9, the estimate of phi (and the constant) are:\n{}\n".format(res.params))

print("****************************************************")
topic = "7. Forecasting with an AR Model"; print("** %s\n" % topic)

# Forecast the first AR(1) model
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()
res.plot_predict(start=990, end=1010)
plt.title('Phi=+0.9', color='red'); 
plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "8. Let's Forecast Interest Rates"; print("** %s\n" % topic)

file = "daily_rates.data" 
daily_rates = pd.read_fwf(file, index_col = 'DATE', parse_dates=True, infer_datetime_format=True)

#This frequency is necesary because the next warning:
#     C:\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:219: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#     ' ignored when e.g. forecasting.', ValueWarning)
interest_rate_data = daily_rates.resample('A').ffill().US10Y
#daily_rates.reindex(pd.date_range(daily_rates.index[0], daily_rates.index[-1], freq='A-DEC')) --> Get null values becouse not all final year day is with data

# Forecast interest rates using an AR(1) model
mod = ARMA(interest_rate_data, order=(1,0))
res = mod.fit()

# Plot the original series and the forecasted series
res.plot_predict(start=0, end='2022-12-31')
plt.legend(fontsize=8)
plt.title('Interest Rates per Year', color='red'); 
plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "9. Compare AR Model with Random Walk"; print("** %s\n" % topic)

#Using the data from the exercise
#simulated_data = np.array([5.        , 4.77522278, 5.60354317, 5.96406402, 5.97965372, 6.02771876, 5.5470751 , 5.19867084, 5.01867859, 5.50452928,
#                           5.89293842, 4.6220103 , 5.06137835, 5.33377592, 5.09333293, 5.37389022, 4.9657092 , 5.57339283, 5.48431854, 4.68588587,
#                           5.25218625, 4.34800798, 4.34544412, 4.72362568, 4.12582912, 3.54622069, 3.43999885, 3.77116252, 3.81727011, 4.35256176,
#                           4.13664247, 3.8745768 , 4.01630403, 3.71276593, 3.55672457, 3.07062647, 3.45264414, 3.28123729, 3.39193866, 3.02947806,
#                           3.88707349, 4.28776889, 3.47360734, 3.33260631, 3.09729579, 2.94652178, 3.50079273, 3.61020341, 4.23021143, 3.94289347,
#                           3.58422345, 3.18253962, 3.26132564, 3.19777388, 3.43527681, 3.37204482])

# Plot the interest rate series and the simulated random walk series side-by-side
fig, axes = plt.subplots(2,1)

# Plot the autocorrelation of the interest rate series in the top plot
fig = plot_acf(interest_rate_data, alpha=1, lags=12, ax=axes[0])

# Plot the autocorrelation of the simulated random walk series in the bottom plot
fig = plot_acf(simulated_data, alpha=1, lags=12, ax=axes[1])

# Label axes
axes[0].set_title("Interest Rate Data")
axes[1].set_title("Simulated Random Walk Data")
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "11. Estimate Order of Model: PACF"; print("** %s\n" % topic)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,3))

# Simulate AR(1) with phi=+0.6
ma = np.array([1])
ar = np.array([1, -0.6])
AR_object = ArmaProcess(ar, ma)
simulated_data_1 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(1)
plot_pacf(simulated_data_1, lags=20, ax=ax1)
ax1.tick_params(labelsize=7)
ax1.set_title('Simulated data AR(1)', color='red'); 

# Simulate AR(2) with phi1=+0.6, phi2=+0.3
ma = np.array([1])
ar = np.array([1, -0.6, -0.3])
AR_object = ArmaProcess(ar, ma)
simulated_data_2 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(2)
plot_pacf(simulated_data_2, lags=20, ax=ax2)
ax2.tick_params(labelsize=7)
ax2.set_title('Simulated data AR(2)', color='red'); 


plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
plt.show()


print("****************************************************")
topic = "12. Estimate Order of Model: Information Criteria"; print("** %s\n" % topic)

# Fit the data to an AR(p) for p = 0,...,6 , and save the BIC
BIC = np.zeros(7)
for p in range(7):
    mod = ARMA(simulated_data_2, order=(p,0))
    res = mod.fit()
    # Save BIC for AR(p)    
    BIC[p] = res.bic
    
# Plot the BIC as a function of p
plt.figure()
plt.plot(range(1,7), BIC[1:7], marker='o')
plt.xlabel('Order of AR Model')
plt.ylabel('Bayesian Information Criterion')
plt.title('Simulated data AR(2)', color='red'); 
plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
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
