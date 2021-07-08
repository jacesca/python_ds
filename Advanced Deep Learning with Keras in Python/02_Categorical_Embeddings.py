# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:53:05 2019

@author: jacqueline.cortez

Chapter 2. Two Input Networks Using Categorical Embeddings, Shared Layers, and Merge Layers
Introduction:
    In this chapter, you will build two-input networks that use categorical embeddings to represent 
    high-cardinality data, shared layers to specify re-usable building blocks, and merge layers to 
    join multiple inputs to a single output. By the end of this chapter, you will have the 
    foundational building blocks for designing neural networks with complex data flows.
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

#import keras                                                                        #For DeapLearning
#from keras.callbacks import EarlyStopping                                           #For DeapLearning
#from keras.layers import Dense                                                      #For DeapLearning
from keras.layers import Embedding                                                  #For DeapLearning
from keras.layers import Flatten                                                    #For DeapLearning
from keras.layers import Input                                                      #For DeapLearning
from keras.layers import Subtract                                                   #For DeapLearning
#from keras.models import load_model                                                 #For DeapLearning
from keras.models import Model                                                      #For DeapLearning
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

file = "games_season.csv"
games_season = pd.read_csv(file)
print(games_season.head())
# Split into training and test set
#X_train, X_test, y_train, y_test = train_test_split(games_season.drop(['score_diff'], axis=1), games_season['score_diff'], test_size = 0.19, random_state=SEED)

file = "games_tourney.csv"
games_tourney = pd.read_csv(file)
print(games_tourney.head())

print("****************************************************")
tema = "2. Define team lookup"; print("** %s\n" % tema)

n_teams = np.unique(games_season['team_1']).shape[0] # Count the unique number of teams
team_lookup = Embedding(input_dim=n_teams, output_dim=1, input_length=1, name='team_lookup') # Create an embedding layer

print("****************************************************")
tema = "3. Define team model"; print("** %s\n" % tema)

teamid_in = Input(shape=(1,), name='teamid_in') # Create an input layer for the team ID
strength_lookup = team_lookup(teamid_in) # Lookup the input in the team strength embedding layer
strength_lookup_flat = Flatten(name='strength_lookup_flat')(strength_lookup) # Flatten the output

team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model') # Combine the operations into a single, re-usable model

# Summarazing the model
print(team_strength_model.summary()) # Summarize the model
plot_model(team_strength_model, to_file='02_03_model.png') # Plot the model

# Plotting the model
plt.figure()
data = plt.imread('02_03_model.png') # Display the image
plt.imshow(data)
plt.title('"team_strength_model" (Embedding Team Number)')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

print("****************************************************")
tema = "5. Defining two inputs"; print("** %s\n" % tema)

team_in_1 = Input(shape=(1,), name="Team-1-In") # Input layer for team 1
team_in_2 = Input(shape=(1,), name="Team-2-In") # Separate input layer for team 2

print("****************************************************")
tema = "6. Lookup both inputs in the same model"; print("** %s\n" % tema)

team_1_strength = team_strength_model(team_in_1) # Lookup team 1 in the team strength model
team_2_strength = team_strength_model(team_in_2) # Lookup team 2 in the team strength model

print("****************************************************")
tema = "8. Output layer using shared layer"; print("** %s\n" % tema)

score_diff = Subtract()([team_1_strength, team_2_strength]) # Create a subtract layer using the inputs from the previous exercise

print("****************************************************")
tema = "9. Model using two inputs and one output"; print("** %s\n" % tema)

model = Model([team_in_1, team_in_2], score_diff) # Create the model
model.compile(optimizer='adam', loss='mean_absolute_error') # Compile the model

# Summarazing the model
print(model.summary()) # Summarize the model
plot_model(model, to_file='02_09_model.png') # Plot the model

# Plotting the model
plt.figure()
data = plt.imread('02_09_model.png') # Display the image
plt.imshow(data)
plt.title('"Shared Layers Model" (Using "team_strength_model")')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

print("****************************************************")
tema = "10. Fit the model to the regular season training data"; print("** %s\n" % tema)

input_1 = games_season['team_1'] # Get the team_1 column from the regular season data
input_2 = games_season['team_2'] # Get the team_2 column from the regular season data

model.fit([input_1, input_2], games_season['score_diff'], 
          epochs=1, batch_size=2048, validation_split=.10, verbose=True) # Fit the model to input 1 and 2, using score diff as a target

print("****************************************************")
tema = "11. Evaluate the model on the tournament test data"; print("** %s\n" % tema)

input_1 = games_tourney['team_1'] # Get team_1 from the tournament data
input_2 = games_tourney['team_2'] # Get team_2 from the tournament data

print(model.evaluate([input_1, input_2], games_tourney['score_diff'], verbose=False)) # Evaluate the model using these inputs

print("****************************************************")
print("** END                                            **")
print("****************************************************")