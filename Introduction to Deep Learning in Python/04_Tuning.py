# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:32:04 2019

@author: jacqueline.cortez

Chapter 4. Fine-tuning keras models
Introduction:
    In this chapter, you'll use the Keras library to build deep learning models for both regression and classification. 
    You'll learn about the Specify-Compile-Fit workflow that you can use to make predictions, and by the end of the 
    chapter, you'll have all the tools necessary to build deep neural networks.
"""

# Import packages
import pandas as pd                                                                 #For loading tabular data
import numpy as np                                                                  #For making operations in lists
#import matplotlib as mpl                                                            #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt                                                     #For creating charts
import seaborn as sns                                                               #For visualizing data
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

#import keras                                                                        #For DeapLearning
from keras.callbacks import EarlyStopping                                           #For DeapLearning
from keras.layers import Dense                                                      #For DeapLearning
from keras.models import Sequential                                                 #For DeapLearning
#from keras.models import load_model                                                 #For DeapLearning
from keras.optimizers import SGD                                                    #For DeapLearning
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

SEED = 42

print("****************************************************")
print("** Getting the data for this program\n")

file = 'titanic_all_numeric.csv'
titanic_df = pd.read_csv(file)
titanic_predictors = titanic_df.drop(['survived'], axis=1).values
#titanic_target = titanic_df.survived.values
titanic_target = to_categorical(titanic_df.survived) # Convert the target to categorical: target
n_cols = titanic_predictors.shape[1] # Save the number of columns in predictors: n_cols
input_shape = (n_cols,)

file = 'mnist.csv'
mnist_df = pd.read_csv(file, header=None)
mnist_predictors = mnist_df.drop([0], axis=1).values
mnist_target = to_categorical(mnist_df[0]) # Convert the target to categorical: target
mnist_n_cols = mnist_predictors.shape[1] # Save the number of columns in predictors: n_cols
mnist_input_shape = (mnist_n_cols,)


print("****************************************************")
tema = "3. Changing optimization parameters"; print("** %s" % tema)

lr_to_test = [0.000001, 0.01, 1] # Create list of learning rates: lr_to_test
for lr in lr_to_test: # Loop over learning rates
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    #model = get_new_model() # Build new model to test, unaffected by previous models
    titanic_model = Sequential() # Set up the model
    titanic_model.add(Dense(100, activation='relu', input_shape=input_shape)) # Add the first layer
    titanic_model.add(Dense(100, activation='relu')) # Add the first layer
    titanic_model.add(Dense(2, activation='softmax')) # Add the output layer
    
    my_optimizer = SGD(lr=lr) # Create SGD optimizer with specified learning rate: my_optimizer
    
    titanic_model.compile(optimizer=my_optimizer, loss='categorical_crossentropy') # Compile the model
    titanic_model.fit(titanic_predictors, titanic_target, epochs=10) # Fit the model
    
print("****************************************************")
tema = "5. Evaluating model accuracy on validation dataset"; print("** %s\n" % tema)

titanic_model = Sequential() # Specify the model
titanic_model.add(Dense(100, activation='relu', input_shape = input_shape))
titanic_model.add(Dense(100, activation='relu'))
titanic_model.add(Dense(2, activation='softmax'))

titanic_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model
hist = titanic_model.fit(titanic_predictors, titanic_target, epochs=10, validation_split=0.3) # Fit the model

print("****************************************************")
tema = "6. Early stopping: Optimizing the optimization"; print("** %s\n" % tema)

early_stopping_monitor = EarlyStopping(patience=2) # Define early_stopping_monitor

model = Sequential() # Specify the model
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model
model.fit(titanic_predictors, titanic_target, validation_split=0.3, epochs=30, callbacks=[early_stopping_monitor]) # Fit the model

print("****************************************************")
tema = "7. Experimenting with wider networks"; print("** %s\n" % tema)

np.random.seed(SEED)

early_stopping_monitor = EarlyStopping(patience=2) # Define early_stopping_monitor

model_1 = Sequential() # Create the new model: model_2
model_1.add(Dense(10, activation='relu', input_shape=input_shape)) # Add the first and second layers
model_1.add(Dense(10, activation='relu'))
model_1.add(Dense(2, activation='softmax')) # Add the output layer
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile model_2

model_2 = Sequential() # Create the new model: model_2
model_2.add(Dense(100, activation='relu', input_shape=input_shape)) # Add the first and second layers
model_2.add(Dense(100, activation='relu'))
model_2.add(Dense(2, activation='softmax')) # Add the output layer
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile model_2

model_1_training = model_1.fit(titanic_predictors, titanic_target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False) # Fit model_1
model_2_training = model_2.fit(titanic_predictors, titanic_target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False) # Fit model_2

sns.set() # Set default Seaborn style
#plt.figure()
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b') # Create the plot
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.title('Features Importances')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.35, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
tema = "8. Adding layers to a network"; print("** %s\n" % tema)

input_shape = (n_cols,) # The input shape to use in the first hidden layer

model_1 = Sequential() # Create the new model: model_2
model_1.add(Dense(50, activation='relu', input_shape=input_shape)) # Add the first, second, and third hidden layers
model_1.add(Dense(2, activation='softmax')) # Add the output layer

model_2 = Sequential() # Create the new model: model_2
model_2.add(Dense(50, activation='relu', input_shape=input_shape)) # Add the first, second, and third hidden layers
model_2.add(Dense(50, activation='relu'))
model_2.add(Dense(50, activation='relu'))
model_2.add(Dense(2, activation='softmax')) # Add the output layer

model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile model_1
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile model_2

model_1_training = model_1.fit(titanic_predictors, titanic_target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False) # Fit model 1
model_2_training = model_2.fit(titanic_predictors, titanic_target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False) # Fit model 2

sns.set() # Set default Seaborn style
plt.figure()
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b') # Create the plot
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.title('Features Importances')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.35, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
tema = "12. Building your own digit recognition model"; print("** %s\n" % tema)

mnist_model = Sequential() # Create the model: model
mnist_model.add(Dense(50, activation='relu', input_shape=mnist_input_shape)) # Add the first hidden layer
mnist_model.add(Dense(50, activation='relu')) # Add the second hidden layer
mnist_model.add(Dense(10, activation='softmax')) # Add the output layer
mnist_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model
mnist_model.fit(mnist_predictors, mnist_target, validation_split=0.3, epochs=10) # Fit the model
print("Loss function: ", mnist_model.loss) # Verify that model contains information from compiling

print("****************************************************")
print("** END                                            **")
print("****************************************************")