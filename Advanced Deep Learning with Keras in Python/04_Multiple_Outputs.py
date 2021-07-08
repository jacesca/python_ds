# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:09:06 2019

@author: jacqueline.cortez

Chapter 4. Multiple Outputs
Introduction:
    In this chapter, you will build neural networks with multiple outputs, which can be used to solve regression problems with 
    multiple targets. You will also build a model that solves a regression problem and a classification problem simultaneously.
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
from scipy.special import expit as sigmoid                                          #For learning machine 
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
#from sklearn.model_selection import train_test_split                                #For learning machine
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
import tensorflow as tf                                                              #For DeapLearning

#import keras                                                                        #For DeapLearning
#from keras.callbacks import EarlyStopping                                           #For DeapLearning
#from keras.layers import BatchNormalization                                         #For DeapLearning
#from keras.layers import Concatenate                                                 #For DeapLearning
from keras.layers import Dense                                                      #For DeapLearning
#from keras.layers import Embedding                                                  #For DeapLearning
#from keras.layers import Flatten                                                    #For DeapLearning
from keras.layers import Input                                                      #For DeapLearning
#from keras.layers import Subtract                                                   #For DeapLearning
#from keras.models import load_model                                                 #For DeapLearning
from keras.models import Model                                                      #For DeapLearning
#from keras.models import Sequential                                                 #For DeapLearning
from keras.optimizers import Adam                                                   #For DeapLearning
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
np.random.seed(SEED)
tf.set_random_seed(SEED)

#print("****************************************************")
print("** User Functions\n")

#print("****************************************************")
print("** Getting the data for this program\n")

file = "games_season_enriched.csv"
games_tourney = pd.read_csv(file)
print(games_tourney.head())

games_tourney_train = games_tourney.query("season < 2010")
games_tourney_test = games_tourney.query("season >= 2010")

print("****************************************************")
tema = "2. Simple two-output model"; print("** %s\n" % tema)

input_tensor = Input(shape=(2,), name='Input') # Define the input
output_tensor = Dense(2, name='Out')(input_tensor) # Define the output
model = Model(input_tensor, output_tensor) # Create a model
model.compile(optimizer='adam', loss='mean_absolute_error') # Compile the model

# Summarazing the model
print(model.summary()) # Summarize the model
plot_model(model, to_file='04_02_model.png', show_shapes=True, show_layer_names=True, rankdir='TB') # rankdir='TB' makes vertical plot and rankdir='LR' creates a horizontal plot

# Plotting the model
plt.figure()
data = plt.imread('04_02_model.png') # Display the image
plt.imshow(data)
plt.title('Multiple Out Model')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

print("****************************************************")
print("3. Fit a model with two outputs")

# Fit the model
model.fit(games_tourney_train[['seed_diff', 'pred']], games_tourney_train[['score_1', 'score_2']],
  		  verbose=False, epochs=100, batch_size=2048)

print("****************************************************")
print("4. Inspect the model (I)")

print(model.get_weights()) # Print the model's weights
print(games_tourney_train.mean()) # Print the column means of the training data

print("****************************************************")
print("5. Evaluate the model")

print(model.evaluate(games_tourney_test[['seed_diff', 'pred']], games_tourney_test[['score_1', 'score_2']], verbose=False)) # Evaluate the model on the tournament test data

print("****************************************************")
print("7. Classification and regression in one model")

input_tensor = Input(shape=(2,), name='Input') # Create an input layer with 2 columns
output_tensor_1 = Dense(1, activation='linear', use_bias=False, name='Reg_out')(input_tensor) # Create the first output
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False, name='Clas_out')(output_tensor_1) # Create the second output (use the first output as input here)
model = Model(input_tensor, [output_tensor_1, output_tensor_2]) # Create a model with 2 outputs

# Summarazing the model
print(model.summary()) # Summarize the model
plot_model(model, to_file='04_07_model.png', show_shapes=True, show_layer_names=True, rankdir='TB') # rankdir='TB' makes vertical plot and rankdir='LR' creates a horizontal plot

# Plotting the model
plt.figure()
data = plt.imread('04_07_model.png') # Display the image
plt.imshow(data)
plt.title('Multiple Out Model')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

print("****************************************************")
print("8. Compile and fit the model")

model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(lr=0.01)) # Compile the model with 2 losses and the Adam optimzer with a higher learning rate
model.fit(games_tourney_train[['seed_diff', 'pred']], # Fit the model to the tournament training data, with 2 inputs and 2 outputs
          [games_tourney_train[['score_diff']], games_tourney_train[['won']]],
          epochs=100, verbose=False, batch_size=16384)

print("****************************************************")
print("9. Inspect the model (II)")

print(model.get_weights()) # Print the model weights
print(games_tourney_train.mean()) # Print the training data means

weight = np.array(model.get_weights())[1][0][0] # Weight from the model
print(sigmoid(1 * weight)) # Print the approximate win probability predicted close game
print(sigmoid(10 * weight)) # Print the approximate win probability predicted blowout game

print("****************************************************")
print("10. Evaluate on new data with two metrics")

# Evaluate the model on new data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']],
                     [games_tourney_test[['score_diff']], games_tourney_test[['won']]], 
                     verbose=False))

print("****************************************************")
print("** END                                            **")
print("****************************************************")