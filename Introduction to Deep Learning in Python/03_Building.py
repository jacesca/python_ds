# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:12:08 2019

@author: jacqueline.cortez

Chapter 3. Building deep learning models with keras
Introduction:
    In this chapter, you'll use the Keras library to build deep learning models for both regression and classification. 
    You'll learn about the Specify-Compile-Fit workflow that you can use to make predictions, and by the end of the 
    chapter, you'll have all the tools necessary to build deep neural networks.
"""

# Import packages
import pandas as pd                                                                 #For loading tabular data
#import numpy as np                                                                  #For making operations in lists
#import matplotlib as mpl                                                            #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt                                                     #For creating charts
#import seaborn as sns                                                               #For visualizing data
#import scipy.stats as stats                                                         #For accesign to a vary of statistics functiosn
#import statsmodels as sm                                                            #For stimations in differents statistical models
#import scykit-learn                                                                 #For performing machine learning  
#import tabula                                                                       #For extracting tables from pdf
#import nltk                                                                         #For working with text data
#import math                                                                         #For accesing to a complex math operations
#import random                                                                       #For generating random numbers
#import calendar                                                                     #For accesing to a vary of calendar operations
#import re                                                                           #For regular expressions
#import timeit                                                                       #For Measure execution time of small code snippets
#import time                                                                         #To measure the elapsed wall-clock time between two points
#import warnings
#import wikipedia

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions
#from itertools import cycle                                                         #Used in the function plot_labeled_decision_regions()
#from math import floor                                                              #Used in the function plot_labeled_decision_regions()
#from math import ceil                                                               #Used in the function plot_labeled_decision_regions()

#from scipy.cluster.hierarchy import fcluster                                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import dendrogram                                      #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import linkage                                         #For learning machine - unsurpervised
#from scipy.sparse import csr_matrix                                                 #For learning machine 
#from scipy.stats import pearsonr                                                    #For learning machine 
#from scipy.stats import randint                                                     #For learning machine 

#from sklearn.cluster import KMeans                                                  #For learning machine - unsurpervised
#from sklearn.decomposition import NMF                                               #For learning machine - unsurpervised
#from sklearn.decomposition import PCA                                               #For learning machine - unsurpervised
#from sklearn.decomposition import TruncatedSVD                                      #For learning machine - unsurpervised

#from sklearn.ensemble import AdaBoostClassifier                                     #For learning machine - surpervised
#from sklearn.ensemble import BaggingClassifier                                      #For learning machine - surpervised
#from sklearn.ensemble import GradientBoostingRegressor                              #For learning machine - surpervised
#from sklearn.ensemble import RandomForestClassifier                                 #For learning machine
#from sklearn.ensemble import RandomForestRegressor                                  #For learning machine - unsurpervised
#from sklearn.ensemble import VotingClassifier                                       #For learning machine - unsurpervised
#from sklearn.feature_extraction.text import TfidfVectorizer                         #For learning machine - unsurpervised
#from sklearn.feature_selection import chi2                                          #For learning machine
#from sklearn.feature_selection import SelectKBest                                   #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                         #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                       #For learning machine
#from sklearn import datasets                                                        #For learning machine
#from sklearn.impute import SimpleImputer                                            #For learning machine
#from sklearn.linear_model import ElasticNet                                         #For learning machine
#from sklearn.linear_model import Lasso                                              #For learning machine
#from sklearn.linear_model import LinearRegression                                   #For learning machine
#from sklearn.linear_model import LogisticRegression                                 #For learning machine
#from sklearn.linear_model import Ridge                                              #For learning machine
#from sklearn.manifold import TSNE                                                   #For learning machine - unsurpervised
#from sklearn.metrics import accuracy_score                                          #For learning machine
#from sklearn.metrics import classification_report                                   #For learning machine
#from sklearn.metrics import confusion_matrix                                        #For learning machine
#from sklearn.metrics import mean_squared_error as MSE                               #For learning machine
from sklearn.metrics import roc_auc_score                                           #For learning machine
#from sklearn.metrics import roc_curve                                               #For learning machine
#from sklearn.model_selection import cross_val_score                                 #For learning machine
#from sklearn.model_selection import GridSearchCV                                    #For learning machine
#from sklearn.model_selection import RandomizedSearchCV                              #For learning machine
from sklearn.model_selection import train_test_split                                #For learning machine
#from sklearn.multiclass import OneVsRestClassifier                                  #For learning machine
#from sklearn.neighbors import KNeighborsClassifier as KNN                           #For learning machine
#from sklearn.pipeline import FeatureUnion                                           #For learning machine
#from sklearn.pipeline import make_pipeline                                          #For learning machine - unsurpervised
#from sklearn.pipeline import Pipeline                                               #For learning machine
#from sklearn.preprocessing import FunctionTransformer                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
#from sklearn.preprocessing import MaxAbsScaler                                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing import Normalizer                                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing import normalize                                         #For learning machine - unsurpervised
#from sklearn.preprocessing import scale                                             #For learning machine
#from sklearn.preprocessing import StandardScaler                                    #For learning machine
#from sklearn.svm import SVC                                                         #For learning machine
#from sklearn.tree import DecisionTreeClassifier                                     #For learning machine - supervised
#from sklearn.tree import DecisionTreeRegressor                                      #For learning machine - supervised

#import keras                                                                        #For DeapLearning
from keras.layers import Dense                                                      #For DeapLearning
from keras.models import Sequential                                                 #For DeapLearning
from keras.models import load_model                                                 #For DeapLearning
from keras.utils import to_categorical                                              #For DeapLearning

#from bokeh.io import curdoc, output_file, show                                      #For interacting visualizations
#from bokeh.plotting import figure, ColumnDataSource                                 #For interacting visualizations
#from bokeh.layouts import row, widgetbox, column, gridplot                          #For interacting visualizations
#from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper        #For interacting visualizations
#from bokeh.models import Slider, Select, Button, CheckboxGroup, RadioGroup, Toggle  #For interacting visualizations
#from bokeh.models.widgets import Tabs, Panel                                        #For interacting visualizations
#from bokeh.palettes import Spectral6                                                #For interacting visualizations


# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters() #Require to explicitly register matplotlib converters.

#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09

#Setting the numpy options
#np.set_printoptions(precision=3) #precision set the precision of the output:
#np.set_printoptions(suppress=True) #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf) #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8) #Return to default value.

#Setting images params
#plt.rcParams.update({'figure.max_open_warning': 0}) #To solve the max images open

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** User defined functions \n")

print("****************************************************")
print("** Getting the data for this program\n")

file = 'hourly_wages.csv'
wages_df = pd.read_csv(file)
wages_predictors = wages_df.drop(['wage_per_hour'], axis=1)
wages_target = wages_df.wage_per_hour.values

file = 'titanic_all_numeric.csv'
titanic_df = pd.read_csv(file)
titanic_predictors = titanic_df.drop(['survived'], axis=1).values
#titanic_target = titanic_df.survived.values

SEED = 42

print("****************************************************")
tema = "3. Specifying a model"; print("** %s\n" % tema)

n_cols = wages_predictors.shape[1] # Save the number of columns in predictors: n_cols

wages_model = Sequential() # Set up the model: model
wages_model.add(Dense(50, activation='relu', input_shape=(n_cols,))) # Add the first layer
wages_model.add(Dense(32, activation='relu')) # Add the second layer
wages_model.add(Dense(1)) # Add the output layer

print("****************************************************")
tema = "5. Compiling the model"; print("** %s\n" % tema)

wages_model.compile(optimizer='adam', loss='mean_squared_error') # Compile the model
print("Loss function: " + wages_model.loss) # Verify that model contains information from compiling

print("****************************************************")
tema = "6. Fitting the model"; print("** %s\n" % tema)

wages_model.fit(wages_predictors, wages_target, epochs=10) # Fit the model

print("****************************************************")
tema = "9. Last steps in classification models"; print("** %s\n" % tema)

titanic_target = to_categorical(titanic_df.survived) # Convert the target to categorical: target
n_cols = titanic_predictors.shape[1] # Save the number of columns in predictors: n_cols
titanic_X_train, titanic_X_test, titanic_y_train, titanic_y_test = train_test_split(titanic_predictors, titanic_target, test_size=0.1, random_state=SEED)

titanic_model = Sequential() # Set up the model
titanic_model.add(Dense(32, activation='relu', input_shape=(n_cols,))) # Add the first layer
titanic_model.add(Dense(2, activation='softmax')) # Add the output layer
titanic_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model
titanic_model.fit(titanic_X_train, titanic_y_train, epochs=10) # Fit the model

print("****************************************************")
tema = "10. Using models"; print("** %s\n" % tema)

file_out = 'titanic_model.h5' #Save the model
titanic_model.save(file_out)

titanic_model = load_model(file_out) #Get bacj the model
print(titanic_model.summary())

print("****************************************************")
tema = "11. Making predictions"; print("** %s\n" % tema)

predictions = titanic_model.predict(titanic_X_test) # Calculate predictions: predictions
predicted_prob_true = predictions[:,1] # Calculate predicted probability of survival: predicted_prob_true
print("Predicted values:\n", predicted_prob_true) # print predicted_prob_true
print("Actual values:\n", titanic_y_test[:,1]) # print predicted_prob_true

test_roc_auc = roc_auc_score(titanic_y_test[:,1], predicted_prob_true) # Compute test_roc_auc
print('Test set ROC AUC score: {:.6f}'.format(test_roc_auc)) # Print test_roc_auc

print("****************************************************")
print("** END                                            **")
print("****************************************************")