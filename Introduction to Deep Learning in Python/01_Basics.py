# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 08:43:01 2019

@author: jacqueline.cortez

Chapter 1. Basics of deep learning and neural networks
Introduction:
    In this chapter, you'll become familiar with the fundamental concepts and terminology used in deep learning, 
    and understand why deep learning techniques are so powerful today. You'll build simple neural networks and 
    generate predictions with them.
"""

# Import packages
#import pandas as pd                                                                 #For loading tabular data
import numpy as np                                                                  #For making operations in lists
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
#from sklearn.metrics import roc_auc_score                                           #For learning machine
#from sklearn.metrics import roc_curve                                               #For learning machine
#from sklearn.model_selection import cross_val_score                                 #For learning machine
#from sklearn.model_selection import GridSearchCV                                    #For learning machine
#from sklearn.model_selection import RandomizedSearchCV                              #For learning machine
#from sklearn.model_selection import train_test_split                                #For learning machine
#from sklearn.multiclass import OneVsRestClassifier                                   #For learning machine
#from sklearn.neighbors import KNeighborsClassifier as KNN                            #For learning machine
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

def relu(input):
    '''Define your relu activation function here'''
    output = max(0, input) # Calculate the value for the output of the relu function: output
    return(output) # Return the value just calculated


def onelayer_predict_with_network(input_data_row, weights): # Define predict_with_network()
    """Function to make prediction with one hidden layer"""
    node_0_input = (input_data_row * weights['node_0']).sum() # Calculate node 0 value
    node_0_output = relu(node_0_input)
    
    node_1_input = (input_data_row * weights['node_1']).sum() # Calculate node 1 value
    node_1_output = relu(node_1_input)
    
    hidden_layer_outputs = np.array([node_0_output, node_1_output]) # Put node values into array: hidden_layer_outputs
    
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum() # Calculate model output
    model_output = relu(input_to_final_layer)
    return(model_output)# Return model output
    

def twolayer_predict_with_network(input_data, weights):
    """Function to make prediction with two hidden layer"""
    node_0_0_input = (input_data * weights['node_0_0']).sum() # Calculate node 0 in the first hidden layer
    node_0_0_output = relu(node_0_0_input)
    node_0_1_input = (input_data * weights['node_0_1']).sum() # Calculate node 1 in the first hidden layer
    node_0_1_output = relu(node_0_1_input)
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output]) # Put node values into array: hidden_0_outputs
    
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum() # Calculate node 0 in the second hidden layer
    node_1_0_output = relu(node_1_0_input)
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum() # Calculate node 1 in the second hidden layer
    node_1_1_output = relu(node_1_1_input)
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output]) # Put node values into array: hidden_1_outputs    
    
    model_output = (hidden_1_outputs * weights['output']).sum() # Calculate model output: model_output
    
    return(model_output) # Return model_output


print("****************************************************")
print("** Getting the data for this program\n")

print("****************************************************")
tema = "4. Coding the forward propagation algorithm"; print("** %s\n" % tema)

weights = {'node_0': np.array([2, 4]), 'node_1': np.array([ 4, -5]), 
           'output': np.array([2, 7])}
input_data = np.array([3, 5])

node_0_value = (input_data * weights['node_0']).sum() # Calculate node 0 value: node_0_value
node_1_value = (input_data * weights['node_1']).sum() # Calculate node 1 value: node_1_value

hidden_layer_outputs = np.array([node_0_value, node_1_value]) # Put node values into array: hidden_layer_outputs

output = (hidden_layer_outputs * weights['output']).sum() # Calculate output: output

print('inout data: ', input_data)
print('weights: \n', weights)
print('hidden layers: ', hidden_layer_outputs)
print(output) # Print output

print("****************************************************")
tema = "5. The Rectified Linear Activation Function"; print("** %s\n" % tema)

weights = {'node_0': np.array([2, 4]), 'node_1': np.array([ 4, -5]), 
           'output': np.array([2, 7])}
input_data = np.array([3, 5])

node_0_input = (input_data * weights['node_0']).sum() # Calculate node 0 value: node_0_output
node_0_output = relu(node_0_input)

node_1_input = (input_data * weights['node_1']).sum() # Calculate node 1 value: node_1_output
node_1_output = relu(node_1_input)

hidden_layer_outputs = np.array([node_0_output, node_1_output]) # Put node values into array: hidden_layer_outputs
model_output = (hidden_layer_outputs * weights['output']).sum() # Calculate model output (do not apply relu)

print('inout data: ', input_data)
print('weights: \n', weights)
print('Node inputs: [{}, {}]'.format(node_0_input, node_1_input))
print('hidden layers output: ', hidden_layer_outputs)
print(model_output) # Print model output

print("****************************************************")
tema = "7. Applying the network to many observations/rows of data"; print("** %s\n" % tema)

weights = {'node_0': np.array([2, 4]), 'node_1': np.array([ 4, -5]), 
           'output': np.array([2, 7])}
input_data = [np.array([3, 5]), np.array([ 1, -1]), np.array([0, 0]), np.array([8, 4])]
print('inout data: ', input_data)
print('weights: \n', weights)

results = [] # Create empty list to store prediction results
for input_data_row in input_data:
    results.append(onelayer_predict_with_network(input_data_row, weights)) # Append prediction to results
    print('inout data: ', input_data_row)
print(results) # Print results

print("****************************************************")
tema = "10.Multi-layer neural networks"; print("** %s\n" % tema)

weights = {'node_0_0': np.array([2, 4]), 'node_0_1': np.array([ 4, -5]),
           'node_1_0': np.array([-1,  2]), 'node_1_1': np.array([1, 2]), 
           'output': np.array([2, 7])}
input_data = np.array([3, 5])
print('inout data: ', input_data)
print('weights: \n', weights)

output = twolayer_predict_with_network(input_data, weights)
print(output)

print("****************************************************")
print("** END                                            **")
print("****************************************************")