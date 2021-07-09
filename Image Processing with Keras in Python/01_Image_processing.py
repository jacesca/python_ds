# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:05:17 2019

@author: jacqueline.cortez

Chapter 1. Image Processing With Neural Networks
Introduction:
    Convolutional neural networks use the data that is represented in images to learn. In this chapter, we will probe data in images, 
    and we will learn how to use Keras to train a neural network to classify objects that appear in images.
"""

# Import packages
#import pandas as pd                                                                 #For loading tabular data
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
#from scipy.special import expit as sigmoid                                          #For learning machine 
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
import tensorflow as tf                                                             #For DeapLearning

#import keras                                                                        #For DeapLearning
from keras.callbacks import EarlyStopping                                           #For DeapLearning
#from keras.layers import BatchNormalization                                         #For DeapLearning
#from keras.layers import Concatenate                                                #For DeapLearning
from keras.layers import Dense                                                      #For DeapLearning
#from keras.layers import Embedding                                                  #For DeapLearning
#from keras.layers import Flatten                                                    #For DeapLearning
#from keras.layers import Input                                                      #For DeapLearning
#from keras.layers import Subtract                                                   #For DeapLearning
#from keras.models import load_model                                                 #For DeapLearning
#from keras.models import Model                                                      #For DeapLearning
from keras.models import Sequential                                                 #For DeapLearning
from keras.optimizers import Adam                                                   #For DeapLearning
#from keras.optimizers import SGD                                                    #For DeapLearning
from keras.utils import plot_model                                                  #For DeapLearning
from keras.utils import to_categorical                                              #For DeapLearning

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

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("****************************************************")
tema = "2. Images as data: visualizations"; print("** %s\n" % tema)

data = plt.imread('bricks.png') # Load the image
plt.imshow(data) # Display the image
plt.title('bricks.png')
plt.suptitle(tema)
plt.show()

print("Shape: ", data.shape)

print("****************************************************")
tema = "3. Images as data: changing images"; print("** %s\n" % tema)

plt.figure()
data[ 0: 40, 0:40, 0] = 1 # Set the red channel in this part of the image to 1
data[ 0: 40, 0:40, 1] = 0 # Set the green channel in this part of the image to 0
data[ 0: 40, 0:40, 2] = 0 # Set the blue channel in this part of the image to 0
data[40: 80, 0:40, 0] = 0 # Make color cyan
data[40: 80, 0:40, 1] = 1 
data[40: 80, 0:40, 2] = 1 
data[80:120, 0:40, 0] = 1 # Make color yellow
data[80:120, 0:40, 1] = 1 
data[80:120, 0:40, 2] = 0 
plt.imshow(data) # Visualize the result
plt.title('bricks.png')
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "5. Using one-hot encoding to represent images"; print("** %s\n" % tema)

labels       = ['shoe', 'shirt', 'shoe', 'shirt', 'dress', 'dress', 'dress']
n_categories = 3 # The number of image categories
categories   = np.array(["shirt", "dress", "shoe"]) # The unique values of categories in the data
ohe_labels   = np.zeros((len(labels), n_categories)) # Initialize ohe_labels as all zeros

for ii in range(len(labels)): # Loop over the labels
    jj = np.where(categories == labels[ii]) # Find the location of this label in the categories variable
    ohe_labels[ii, jj] = 1 # Set the corresponding zero to one

print(ohe_labels)
print(ohe_labels.shape)
    
print("****************************************************")
tema = "6. Evaluating a classifier"; print("** %s\n" % tema)

test_labels = np.array([[0., 0., 1.], [0., 1., 0.], [0., 0., 1.], [0., 1., 0.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 1., 0.]])
predictions = np.array([[0., 0., 1.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [0., 0., 1.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])

number_correct = (test_labels*predictions).sum() # Calculate the number of correct predictions
print("Number of correct predictions: ", number_correct)

proportion_correct = number_correct/len(predictions) # Calculate the proportion of correct predictions
print("Proportion of correct predictions: ", proportion_correct)

print("****************************************************")
tema = "8. Build a neural network"; print("** %s\n" % tema)

model = Sequential(name='hhh') # Initializes a sequential model
model.add(Dense(200, activation='relu', input_shape=(784,), name='Dense')) # First layer
model.add(Dense(200, activation='relu', name='Hidden-layer')) # Second layer
model.add(Dense(10, activation='softmax', name='Output')) # Output layer

# Summarazing the model
print(model.summary()) # Summarize the model
plot_model(model, to_file='01_08_model.png', show_shapes=False, show_layer_names=True, rankdir='TB') # rankdir='TB' makes vertical plot and rankdir='LR' creates a horizontal plot

# Plotting the model
plt.figure()
data = plt.imread('01_08_model.png') # Display the image
plt.imshow(data)
plt.title('A MNIST case in a Neural Red')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

print("****************************************************")
tema = "9. Compile a neural network"; print("** %s\n" % tema)

model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model

print("****************************************************")
tema = "10. Fitting a neural network model to clothing data"; print("** %s\n" % tema)

train_data   = x_train.reshape(len(x_train), 784) # Reshape the data to two-dimensional array
train_labels = to_categorical(y_train, len(class_names))

model.fit(train_data, train_labels,  
          epochs=150, verbose=True, validation_split=.20, batch_size=2048, callbacks=[EarlyStopping(patience=2)]) # Fit the model

print("****************************************************")
tema = "11. Cross-validation for neural network evaluation"; print("** %s\n" % tema)

test_data = x_test.reshape(len(x_test), 784) # Reshape test data
test_labels = to_categorical(y_test, len(class_names))

print(model.evaluate(test_data, test_labels)) # Evaluate the model

print("****************************************************")
print("** END                                            **")
print("****************************************************")