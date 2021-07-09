# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:54:55 2020

@author: jacqueline.cortez
Chapter 5: Putting It All Together
    This chapter will show you how to model two series jointly using cointegration 
    models. Then you'll wrap up with a case study where you look at a time series 
    of temperature data from New York City.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot              as plt                                  #For creating charts
import numpy                          as np                                   #For making operations in lists
import pandas                         as pd                                   #For loading tabular data
import statsmodels.api                as sm                                   #Make a prediction model

#from matplotlib.ticker                import StrMethodFormatter               #Import the necessary library to delete the scientist notation
from statsmodels.graphics.tsaplots    import plot_acf                         #For autocorrelation function
from statsmodels.graphics.tsaplots    import plot_pacf                        #For simulating data and for plotting the PACF. Partial Autocorrelation Function measures the incremental benefit of adding another lag.
from statsmodels.tsa.arima_model      import ARIMA                            #Similar to use ARMA but on original data (before differencing)
from statsmodels.tsa.arima_model      import ARMA                             #To estimate parameters from data simulated (AR model)
from statsmodels.tsa.stattools        import adfuller                         #Augmented Dickey-Fuller Test for Random Walk
from statsmodels.tsa.stattools        import coint                            #Test for cointegration

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 123
np.random.seed(SEED)

print("****************************************************")
topic = "2. A Dog on a Leash? (Part 1)"; print("** %s\n" % topic)

file = "CME_HO1.csv" 
HO = pd.read_csv(file, index_col="Date", parse_dates=True).sort_index()

file = "CME_NG1.csv" 
NG = pd.read_csv(file, index_col="Date", parse_dates=True).sort_index()

plt.figure()
# Plot the prices separately
plt.subplot(2,1,1)
plt.plot(7.25*HO.Close, label='Heating Oil')
plt.plot(NG, label='Natural Gas')
plt.legend(loc='best', fontsize='small')
#plt.xticks(rotation=90)
plt.xlabel('Year', fontsize=7); plt.ylabel('S/millionBTU', fontsize=7); # Labeling the axis.
plt.title("The Heating Oil and Natural Gas prices", color='red')

# Plot the spread
plt.subplot(2,1,2)
plt.plot(7.25*HO-NG, label='Spread')
plt.legend(loc='best', fontsize='small')
plt.axhline(y=0, linestyle='--', color='k')
#plt.xticks(rotation=90)
plt.xlabel('Year', fontsize=7); plt.ylabel('S/millionBTU', fontsize=7); # Labeling the axis.
plt.title("Difference between both prices", color='red')

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5);
plt.show()

print("****************************************************")
topic = "3. A Dog on a Leash? (Part 2)"; print("** %s\n" % topic)

# Compute the ADF for HO and NG
result_HO = adfuller(HO.Close, autolag='t-stat')
print("The p-value for the ADF test on HO is ", result_HO[1])
result_NG = adfuller(NG['Close'])
print("The p-value for the ADF test on NG is ", result_NG[1])

# Compute the ADF of the spread
result_spread = adfuller(7.25 * HO.Close - NG.Close)
print("The p-value for the ADF test on the spread is ", result_spread[1])

#Trying the coint test.
result_cointegration = coint(HO.Close, NG.Close)
print("The Cointegration test result:\n", result_cointegration)

print("****************************************************")
topic = "4. Are Bitcoin and Ethereum Cointegrated?"; print("** %s\n" % topic)

fwidths = [10,18]
file = "BTC.data"
BTC = pd.read_fwf(file, widths=fwidths, index_col="Date", parse_dates=True).sort_index()

file = "ETH.data"
ETH = pd.read_fwf(file, widths=fwidths, index_col="Date", parse_dates=True).sort_index()

# Regress BTC on ETH
ETH = sm.add_constant(ETH)
result = sm.OLS(BTC,ETH).fit()

# Compute ADF
b = result.params[1]
adf_stats = adfuller(BTC['Price'] - b*ETH['Price'])
print("The p-value for the ADF test is ", adf_stats[1])

print("****************************************************")
topic = "6. Is Temperature a Random Walk (with Drift)?"; print("** %s\n" % topic)

file = "temp_NY.data"
temp_NY = pd.read_fwf(file, index_col="DATE").sort_index()

# Convert the index to a datetime object
temp_NY.index = pd.to_datetime(temp_NY.index, format='%Y')

# Compute and print ADF p-value
result = adfuller(temp_NY['TAVG'])
footnote = "The p-value for the ADF test is {0:.4f}.".format(result[1])
print(footnote)

# Plot average temperatures
temp_NY.plot()
plt.ylabel('Average Temperature'); # Labeling the axis.
plt.title("Average Temperature per Year in New York City", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.figtext(0.5, 0.05, "Footnote: " + footnote, horizontalalignment='center') 
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
topic = "7. Getting \"Warmed\" Up: Look at Autocorrelations"; print("** %s\n" % topic)

# Take first difference of the temperature Series
chg_temp = temp_NY.diff()
chg_temp = chg_temp.dropna()

# Plot the ACF and PACF on the same page
fig, axes = plt.subplots(2,1)

# Plot the ACF
ax=axes[0]
plot_acf(chg_temp, lags=20, ax=ax)
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
#ax.tick_params(labelsize=8)
ax.set_title('Autocorrelation Function\n(Any significant non zero autocorrelations implies\nthat series can be forecast from the past)', color='red'); 

# Plot the PACF
ax=axes[1]
plot_pacf(chg_temp, lags=20, ax=ax)
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
#ax.tick_params(labelsize=8)
ax.set_title('Partial Autocorrelation Function\n(Benefit of Adding another Lag)', color='red'); 

footnote = "Footnote: Average Temperature per Year in New York City. There is no clear pattern in the\nACF and PACF except the negative lag-1 autocorrelation in the ACF."
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.80, wspace=None, hspace=0.6);
plt.figtext(0.01, 0.05, footnote, horizontalalignment='left') 
plt.show()

print("****************************************************")
topic = "8. Which ARMA Model is Best?"; print("** %s\n" % topic)

# Fit the data to an AR(1) model and print AIC:
mod_ar1 = ARMA(chg_temp, order=(1, 0), freq="AS-JAN")
res_ar1 = mod_ar1.fit()
print("The AIC for an AR(1) is: ", res_ar1.aic)

# Fit the data to an AR(2) model and print AIC:
mod_ar2 = ARMA(chg_temp, order=(2, 0), freq="AS-JAN")
res_ar2 = mod_ar2.fit()
print("The AIC for an AR(2) is: ", res_ar2.aic)

# Fit the data to an ARMA(1,1) model and print AIC:
mod_arma11 = ARMA(chg_temp, order=(1,1), freq="AS-JAN")
res_arma11 = mod_arma11.fit()
print("The AIC for an ARMA(1,1) is: ", res_arma11.aic)

print("\nThe ARMA(1,1) has the lowest AIC values among the three models.")

print("****************************************************")
topic = "9. Don't Throw Out That Winter Coat Yet"; print("** %s\n" % topic)

# Forecast temperatures using an ARIMA(1,1,1) model
# The d in order(p,d,q) is one, since we first differenced once.
mod = ARIMA(temp_NY, order=(1,1,1), freq="AS-JAN") 
res = mod.fit()

footnote = "Footnote: According to the model, the temperature is expected to be about 0.6 degrees\nhigher in 30 years (almost entirely due to the trend), but the 95% confidence\ninterval around that is over 5 degrees."

# Plot the original series and the forecasted series
res.plot_predict(start='1872-01-01', end='2046-01-01')
plt.xlabel('Year'); plt.ylabel('Average Temperature (°F)'); # Labeling the axis.
plt.title("Forecasting the temperature over the next 30 years\nin Central Park NY", color='red')

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.8, wspace=None, hspace=None);
plt.figtext(0.01, 0.05, footnote, horizontalalignment='left') 
plt.show()
 
print("****************************************************")
topic = "99. Applying to El Salvador"; print("** %s\n" % topic)
"""
file = "TAVG_1901_2016_SLV.csv" 
SLV = pd.read_csv(file, parse_dates=[["Year","Month","Day"]], index_col="Year_Month_Day", usecols=["Year","Month", "Day", "Temperature - (Celsius)"]).sort_index().rename(columns={"Temperature - (Celsius)":"TAVG"})
SLV.index.rename('Date', inplace = True)

SLV_yearly = SLV.resample('1A').mean()
"""

file = "TAVG_1901_2016_SLV.csv" 
SLV = pd.read_csv(file, usecols=["Year", "Temperature - (Celsius)"]).sort_index().rename(columns={"Temperature - (Celsius)":"TAVG"})

SLV_yearly = SLV.groupby("Year").mean()
SLV_yearly.index = pd.to_datetime(SLV_yearly.index, format='%Y')

###############################################################
# Is Temperature a Random Walk (with Drift)?
###############################################################
# Compute and print ADF p-value
result = adfuller(SLV_yearly['TAVG'])
footnote = "The p-value for the ADF test is {0:.4f}.".format(result[1])
print(footnote)
# Plot average temperatures
SLV_yearly.plot()
plt.ylabel('Average Temperature'); # Labeling the axis.
plt.title("Average Temperature per Year in El Salvador (1901-2016)", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.figtext(0.5, 0.05, "Footnote: " + footnote + " The data follows a random walk with drift.", horizontalalignment='center') 
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None);
plt.show()

###############################################################
# Getting "Warmed" Up: Look at Autocorrelations
###############################################################
# Take first difference of the temperature Series
SLV_chg_temp = SLV_yearly.diff()
SLV_chg_temp = SLV_chg_temp.dropna()
# Plot the ACF and PACF on the same page
fig, axes = plt.subplots(2,1)
# Plot the ACF
ax=axes[0]
plot_acf(SLV_chg_temp, lags=20, ax=ax)
ax.set_title('Autocorrelation Function\n(Any significant non zero autocorrelations implies\nthat series can be forecast from the past)', color='red'); 
# Plot the PACF
ax=axes[1]
plot_pacf(SLV_chg_temp, lags=20, ax=ax)
ax.set_title('Partial Autocorrelation Function\n(Benefit of Adding another Lag)', color='red'); 
#Preparing the last part of the graph
footnote = "Footnote: Average Temperature per Year in El Salvador (1901-2016)."
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.80, wspace=None, hspace=0.6);
plt.figtext(0.01, 0.05, footnote, horizontalalignment='left') 
plt.show()

###############################################################
# Which ARMA Model is Best?
###############################################################
# Fit the data to an AR(1) model and print AIC:
mod_ar1 = ARMA(SLV_chg_temp, order=(1, 0), freq="AS-JAN")
res_ar1 = mod_ar1.fit()
print("\nThe AIC for an AR(1) is: ", res_ar1.aic)
# Fit the data to an AR(2) model and print AIC:
mod_ar2 = ARMA(SLV_chg_temp, order=(2, 0), freq="AS-JAN")
res_ar2 = mod_ar2.fit()
print("The AIC for an AR(2) is: ", res_ar2.aic)
# Fit the data to an ARMA(1,1) model and print AIC:
mod_arma11 = ARMA(SLV_chg_temp, order=(1,1), freq="AS-JAN")
res_arma11 = mod_arma11.fit()
print("The AIC for an ARMA(1,1) is: ", res_arma11.aic)
print("\nThe ARMA(1,1) has the lowest AIC values among the three models.")

###############################################################
# Don't Throw Out That Winter Coat Yet"; print("** %s\n" % topic)
###############################################################
# Forecast temperatures using an ARIMA(1,1,1) model
# The d in order(p,d,q) is one, since we first differenced once.
mod = ARIMA(SLV_yearly, order=(1,1,1), freq="AS-JAN") 
res = mod.fit()
# Plot the original series and the forecasted series
res.plot_predict(start='1904-01-01', end='2026-01-01')
plt.xlabel('Year'); plt.ylabel('Average Temperature (°C)'); # Labeling the axis.
plt.title("Forecasting the temperature over the next 10 years\nin El Salvador", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.8, wspace=None, hspace=None);
plt.axhline(y=24.75, linestyle='--', linewidth=1, color='k')
plt.axhline(y=25.50, linestyle='--', linewidth=1, color='k')
plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import contextily                                                             #To add a background web map to our plot
#import inspect                                                                #Used to get the code inside a function
#import folium                                                                 #To create map street folium.__version__'0.10.0'
#import geopandas         as gpd                                               #For working with geospatial data 
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.dates  as mdates                                            #For providing sophisticated date plotting capabilities
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
#from statsmodels.graphics.tsaplots    import plot_pacf                        #For simulating data and for plotting the PACF. Partial Autocorrelation Function measures the incremental benefit of adding another lag.
#from statsmodels.tsa.arima_model      import ARIMA                            #Similar to use ARMA but on original data (before differencing)
#from statsmodels.tsa.arima_model      import ARMA                             #To estimate parameters from data simulated (AR model)
#from statsmodels.tsa.arima_process    import ArmaProcess                      #For simulating data and for plotting the PACF, For Simulate Autoregressive (AR) Time Series 
#from statsmodels.tsa.stattools        import acf                              #For autocorrelation function
#from statsmodels.tsa.stattools        import adfuller                         #Augmented Dickey-Fuller Test for Random Walk
#from statsmodels.tsa.stattools        import coint                            #Test for cointegration


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
#ax.xaxis.set_major_locator(plt.MaxNLocator(3))                                #Define the number of ticks to show on x axis.
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())                          #Other way to Define the number of ticks to show on x axis.
#ax.xaxis.set_major_formatter(mdates.AutoDateFormatter())                      #Other way to Define the number of ticks to show on x axis.


#plt.xticks(rotation=45)                                                       #rotate x-axis labels by 45 degrees
#plt.yticks(rotation=90)                                                       #rotate y-axis labels by 90 degrees
#plt.savefig("sample.jpg")                                                     #save image of `plt`

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
