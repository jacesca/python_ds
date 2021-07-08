# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:09:42 2019

@author: jacqueline.cortez

Chapter 1. The Keras Functional API
Introduction:
    In this chapter, you'll become familiar with the basics of the Keras functional API. 
    You'll build a simple functional network using functional building blocks, fit it to data, 
    and make predictions.
"""

# Import packages
import pandas as pd                                                                 #For loading tabular data
import numpy as np                                                                  #For making operations in lists
#import matplotlib as mpl                                                            #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt                                                     #For creating charts
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
#from datetime import date                                                           #For obteining today function
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions
#from itertools import cycle                                                         #Used in the function plot_labeled_decision_regions()
#from math import floor                                                              #Used in the function plot_labeled_decision_regions()
#from math import ceil                                                               #Used in the function plot_labeled_decision_regions()
#from itertools import combinations                                                  #For iterations
#from collections import defaultdict                                                 #Returns a new dictionary-like object

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
#from sklearn.metrics import roc_auc_score                                           #For learning machine
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

#import statsmodels.api as sm                                                        #Make a prediction model
#import statsmodels.formula.api as smf                                               #Make a prediction model    

#import keras                                                                        #For DeapLearning
#from keras.callbacks import EarlyStopping                                           #For DeapLearning
from keras.layers import Input                                                      #For DeapLearning
from keras.layers import Dense                                                      #For DeapLearning
from keras.models import Model                                                      #For DeapLearning
#from keras.models import load_model                                                 #For DeapLearning
#from keras.models import Sequential                                                 #For DeapLearning
#from keras.optimizers import SGD                                                    #For DeapLearning
from keras.utils import plot_model                                                  #For DeapLearning
#from keras.utils import to_categorical                                              #For DeapLearning

#import networkx as nx                                                               #For Network Analysis in Python
#import nxviz as nv                                                                  #For Network Analysis in Python
#from nxviz import ArcPlot                                                           #For Network Analysis in Python
#from nxviz import CircosPlot                                                        #For Network Analysis in Python 
#from nxviz import MatrixPlot                                                        #For Network Analysis in Python 

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
print("** User defined variables \n")

SEED=1

#print("****************************************************")
print("** User Functions\n")

#print("****************************************************")
print("** Getting the data for this program\n")

file = "games_tourney.csv"
games_tourney_train = pd.read_csv(file)
print(games_tourney_train.head())
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(games_tourney_train.drop(['score_diff'], axis=1), games_tourney_train['score_diff'], test_size = 0.19, random_state=SEED)

print("****************************************************")
tema = '2. Input layers'; print("** %s\n" % tema)

# Create an input layer of shape 1
input_tensor = Input(shape=(1,))
print(input_tensor)


print("****************************************************")
tema = "3. Dense layers"; print("** %s\n" % tema)

input_tensor = Input(shape=(1,)) # Input layer
output_layer = Dense(1) # Dense layer
output_tensor = output_layer(input_tensor) # Connect the dense layer to the input_tensor

print(output_layer)
print(output_tensor)

print("****************************************************")
tema = "4. Output layers"; print("** %s\n" % tema)

# Create a dense layer and connect the dense layer to the input_tensor in one step
# Note that we did this in 2 steps in the previous exercise, but are doing it in one step now
output_tensor = Dense(1)(input_tensor)

print("****************************************************")
tema = "6. Build a model"; print("** %s\n" % tema)

model = Model(input_tensor, output_tensor)

print("****************************************************")
tema = "7. Compile a model"; print("** %s\n" % tema)

model.compile(optimizer='adam', loss='mean_absolute_error') # Compile the model

print("****************************************************")
tema = "8. Visualize a model"; print("** %s\n" % tema)

input_tensor = Input(shape=(1,), name='Input-Data') # Input layer
output_layer = Dense(1, name='Predicted-Score-Diff') # Dense layer
output_tensor = output_layer(input_tensor) # Connect the dense layer to the input_tensor
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error') # Compile the model

print(model.summary()) # Summarize the model
plot_model(model, to_file='01_08_model.png') # Plot the model

data = plt.imread('01_08_model.png') # Display the image
plt.imshow(data)
plt.title('My first Keras Model')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

print("****************************************************")
tema = "10. Fit the model to the tournament basketball data"; print("** %s\n" % tema)

# Building the model
input_tensor  = Input(shape=(1,), name='Input-Data') # Input layer
middle_tensor = Dense(1, name='Middle-layer')(input_tensor) #Middle layer
output_tensor = Dense(1, name='Predicted-Score-Diff')(middle_tensor) # Connect the dense layer to the input_tensor
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error') # Compile the model

# Summarazing the model
print(model.summary()) # Summarize the model
plot_model(model, to_file='01_10_model.png') # Plot the model

# Plotting the model
plt.figure()
data = plt.imread('01_10_model.png') # Display the image
plt.imshow(data)
plt.title('The Keras Model - One Input (Seed Diff)')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

# Fitting the model
np.random.seed(SEED) # Seed random number generator
model.fit(X_train['seed_diff'], y_train, # Now fit the model
          epochs=1, batch_size=128, validation_split=.10, verbose=True)

print("****************************************************")
tema = "11. Evaluate the model on a test set"; print("** %s\n" % tema)

print(model.evaluate(X_test['seed_diff'], y_test, verbose=False)) # Evaluate the model on the test data

print("****************************************************")
print("** END                                            **")
print("****************************************************")
