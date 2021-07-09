# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 23:09:49 2019

@author: jacqueline.cortez

Capítulo 4. Boosting
Introduction:
    Boosting refers to an ensemble method in which several models are trained sequentially with each model learning 
    from the errors of its predecessors. In this chapter, you'll be introduced to the two boosting methods of AdaBoost 
    and Gradient Boosting.
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

from sklearn.ensemble import AdaBoostClassifier                                     #For learning machine - surpervised
#from sklearn.ensemble import BaggingClassifier                                      #For learning machine - surpervised
from sklearn.ensemble import GradientBoostingRegressor                              #For learning machine - surpervised
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
from sklearn.metrics import mean_squared_error as MSE                               #For learning machine
from sklearn.metrics import roc_auc_score                                           #For learning machine
#from sklearn.metrics import roc_curve                                               #For learning machine
#from sklearn.model_selection import cross_val_score                                 #For learning machine
#from sklearn.model_selection import GridSearchCV                                    #For learning machine
#from sklearn.model_selection import RandomizedSearchCV                              #For learning machine
from sklearn.model_selection import train_test_split                                #For learning machine
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
from sklearn.tree import DecisionTreeClassifier                                     #For learning machine - supervised
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

SEED = 1

print("****************************************************")
print("** Getting the data for this program\n")

file = 'indian_liver_patient.csv'
liver_df = pd.read_csv(file)
liver_df.dropna(inplace=True)
liver_df['Dataset'] = liver_df.Dataset.map({1: 1, 2: 0})
liver_df['Is_male'] = liver_df.Gender.map({'Female':0,'Male':1})
liver_X = liver_df.drop(['Dataset', 'Gender'], axis=1)
liver_y = liver_df.Dataset

file = 'bikes.csv'
bikes_df = pd.read_csv(file)
bikes_X = bikes_df.drop(['cnt'], axis=1)
bikes_y = bikes_df.cnt

print("****************************************************")
tema = "2. Define the bagging classifier"; print("** %s\n" % tema)

dt = DecisionTreeClassifier(max_depth=2, random_state=SEED) # Instantiate dt
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=SEED) # Instantiate ada

print("****************************************************")
tema = "3. Train the AdaBoost classifier"; print("** %s\n" % tema)

liver_X_train, liver_X_test, liver_y_train, liver_y_test = train_test_split(liver_X, liver_y, stratify=liver_y, test_size=0.2, random_state=SEED)
ada.fit(liver_X_train, liver_y_train) # Fit ada to the training set
liver_y_pred_proba = ada.predict_proba(liver_X_test)[:,1] # Compute the probabilities of obtaining the positive class

print(liver_y_pred_proba)

print("****************************************************")
tema = "4. Evaluate the AdaBoost classifier"; print("** %s\n" % tema)

ada_roc_auc = roc_auc_score(liver_y_test, liver_y_pred_proba) # Evaluate test-set roc_auc_score
print('ROC AUC score: {:.6f}'.format(ada_roc_auc)) # Print roc_auc_score

print("****************************************************")
tema = "6. Define the GB regressor"; print("** %s\n" % tema)

SEED=2

gb = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=SEED) # Instantiate gb
print(gb)

print("****************************************************")
tema = "7. Train the GB regressor"; print("** %s\n" % tema)

bikes_X_train, bikes_X_test, bikes_y_train, bikes_y_test = train_test_split(bikes_X, bikes_y, test_size=0.2, random_state=SEED)

gb.fit(bikes_X_train, bikes_y_train) # Fit gb to the training set
bikes_y_pred = gb.predict(bikes_X_test) # Predict test set labels

print("****************************************************")
tema = "8. Evaluate the GB regressor"; print("** %s\n" % tema)

mse_test = MSE(bikes_y_test, bikes_y_pred) # Compute MSE
rmse_test = mse_test ** (1/2) # Compute RMSE
print('Test set RMSE of gb: {:.6f}'.format(rmse_test)) # Print RMSE

print("****************************************************")
tema = "10. Regression with SGB"; print("** %s\n" % tema)

sgbr = GradientBoostingRegressor(max_depth=4, subsample=0.9, max_features=0.75, n_estimators=200, random_state=2) # Instantiate sgbr

print("****************************************************")
tema = "11. Train the SGB regressor"; print("** %s\n" % tema)

sgbr.fit(bikes_X_train, bikes_y_train) # Fit sgbr to the training set
bikes_y_pred = sgbr.predict(bikes_X_test) # Predict test set labels

print("****************************************************")
tema = "12. Evaluate the SGB regressor"; print("** %s\n" % tema)

mse_test = MSE(bikes_y_test, bikes_y_pred) # Compute test set MSE
rmse_test = mse_test ** (1/2) # Compute test set RMSE
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test)) # Print rmse_test

print("****************************************************")
print("** END                                            **")
print("****************************************************")