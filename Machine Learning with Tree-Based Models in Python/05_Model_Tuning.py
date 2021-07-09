# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:55:20 2019

@author: jacqueline.cortez

Capítulo 5. Model Tuning
Introduction:
    The hyperparameters of a machine learning model are parameters that are not learned from data. 
    They should be set prior to fitting the model to the training set. In this chapter, you'll learn how 
    to tune the hyperparameters of a tree-based model using grid search cross validation.
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
from sklearn.ensemble import RandomForestRegressor                                  #For learning machine - unsurpervised
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
from sklearn.model_selection import GridSearchCV                                    #For learning machine
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
from sklearn.preprocessing import StandardScaler                                    #For learning machine
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
#liver_df['Gender'] = liver_df.Gender.map({'Female':0,'Male':1})
liver_df['Dataset'] = liver_df.Dataset.map({1: 1, 2: 0})
scaler = StandardScaler()
liver_df_standarized = liver_df.drop(['Gender','Dataset'], axis=1)
liver_df_standarized = pd.DataFrame(data=scaler.fit_transform(liver_df_standarized), columns=liver_df_standarized.columns)
liver_df_standarized['Is_Male'] = liver_df.Gender.map({'Female':0,'Male':1}).values
liver_df_standarized['Liver_Disease'] = liver_df['Dataset'].values
liver_X = liver_df_standarized.drop(['Liver_Disease'], axis=1)
liver_y = liver_df_standarized.Liver_Disease

file = 'bikes.csv'
bikes_df = pd.read_csv(file)
bikes_X = bikes_df.drop(['cnt'], axis=1)
bikes_y = bikes_df.cnt

print("****************************************************")
tema = "3. Set the tree's hyperparameter grid"; print("** %s\n" % tema)

# Define params_dt
params_dt = {'max_depth': [2,3,4], 
             'min_samples_leaf':[0.12, 0.14, 0.16, 0.18]}

print("****************************************************")
tema = "4. Search for the optimal tree"; print("** %s\n" % tema)

dt = DecisionTreeClassifier(random_state=SEED) # Instantiate dt
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='roc_auc', cv=5, n_jobs=-1, iid=True) # Instantiate grid_dt

print("****************************************************")
tema = "5. Evaluate the optimal tree"; print("** %s\n" % tema)

liver_X_train, liver_X_test, liver_y_train, liver_y_test = train_test_split(liver_X, liver_y, stratify=liver_y, test_size=0.2, random_state=SEED)
grid_dt.fit(liver_X_train, liver_y_train)

print("Best hyperparameters: \n", grid_dt.best_params_)
print("Best ROC score: ", grid_dt.best_score_)

best_model = grid_dt.best_estimator_ # Extract the best estimator
liver_y_pred_proba = grid_dt.predict_proba(liver_X_test)[:,1] # Predict the test set probabilities of the positive class
test_roc_auc = roc_auc_score(liver_y_test, liver_y_pred_proba) # Compute test_roc_auc
print('Test set ROC AUC score: {:.6f}'.format(test_roc_auc)) # Print test_roc_auc

print("****************************************************")
tema = "8. Evaluate the optimal tree"; print("** %s\n" % tema)

SEED = 2

rf = RandomForestRegressor(random_state=SEED)
params_rf = {'n_estimators':[100, 350, 500], 'max_features':['log2', 'auto', 'sqrt'], 'min_samples_leaf':[2,10,30]} # Define the dictionary 'params_rf'

print("****************************************************")
tema = "9. Search for the optimal forest"; print("** %s\n" % tema)

bikes_X_train, bikes_X_test, bikes_y_train, bikes_y_test = train_test_split(bikes_X, bikes_y, test_size=0.2, random_state=SEED)

grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1, iid=True) # Instantiate grid_rf
grid_rf.fit(bikes_X_train, bikes_y_train)

print("Best hyperparameters: \n", grid_dt.best_params_)
print("Best score: ", grid_dt.best_score_)

print("****************************************************")
tema = "10. Evaluate the optimal forest"; print("** %s\n" % tema)

best_model = grid_rf.best_estimator_ # Extract the best estimator
bikes_y_pred = best_model.predict(bikes_X_test) # Predict test set labels
rmse_test = MSE(bikes_y_test, bikes_y_pred)**(1/2) # Compute rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test)) # Print rmse_test

print("****************************************************")
print("** END                                            **")
print("****************************************************")