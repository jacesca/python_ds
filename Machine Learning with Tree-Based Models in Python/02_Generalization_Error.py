# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:30:01 2019

@author: jacqueline.cortez

Capítulo 2. The Bias-Variance Tradeoff
Introduction:
    The bias-variance tradeoff is one of the fundamental concepts in supervised machine learning. 
    In this chapter, you'll understand how to diagnose the problems of overfitting and underfitting. 
    You'll also be introduced to the concept of ensembling where the predictions of several models are 
    aggregated to produce predictions that are more robust.
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
#from sklearn.ensemble import RandomForestClassifier                                 #For learning machine
from sklearn.ensemble import VotingClassifier                                       #For learning machine - unsurpervised
#from sklearn.feature_extraction.text import TfidfVectorizer                         #For learning machine - unsurpervised
#from sklearn.feature_selection import chi2                                          #For learning machine
#from sklearn.feature_selection import SelectKBest                                   #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                         #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                       #For learning machine
#from sklearn import datasets                                                        #For learning machine
#from sklearn.impute import SimpleImputer                                            #For learning machine
#from sklearn.linear_model import ElasticNet                                         #For learning machine
#from sklearn.linear_model import Lasso                                              #For learning machine
from sklearn.linear_model import LinearRegression                                   #For learning machine
from sklearn.linear_model import LogisticRegression                                 #For learning machine
#from sklearn.linear_model import Ridge                                              #For learning machine
#from sklearn.manifold import TSNE                                                   #For learning machine - unsurpervised
from sklearn.metrics import accuracy_score                                          #For learning machine
#from sklearn.metrics import classification_report                                   #For learning machine
#from sklearn.metrics import confusion_matrix                                        #For learning machine
from sklearn.metrics import mean_squared_error as MSE                               #For learning machine
#from sklearn.metrics import roc_auc_score                                           #For learning machine
#from sklearn.metrics import roc_curve                                               #For learning machine
from sklearn.model_selection import cross_val_score                                 #For learning machine
#from sklearn.model_selection import GridSearchCV                                    #For learning machine
#from sklearn.model_selection import RandomizedSearchCV                              #For learning machine
from sklearn.model_selection import train_test_split                                #For learning machine
#from sklearn.multiclass import OneVsRestClassifier                                   #For learning machine
from sklearn.neighbors import KNeighborsClassifier as KNN                            #For learning machine
#from sklearn.pipeline import FeatureUnion                                           #For learning machine
from sklearn.pipeline import make_pipeline                                          #For learning machine - unsurpervised
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
from sklearn.tree import DecisionTreeRegressor                                      #For learning machine - supervised

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

file = 'auto.csv'
auto_df = pd.read_csv(file)
auto_X = auto_df.drop(['mpg'], axis=1)
auto_X = pd.get_dummies(auto_X, prefix_sep='_')#, drop_first=True)
autoX_displ = auto_df.displ
auto_y = auto_df.mpg


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

print("****************************************************")
tema = "5. Instantiate the model"; print("** %s\n" % tema)

auto_X_train, auto_X_test, auto_y_train, auto_y_test = train_test_split(auto_X, auto_y, test_size=0.3, random_state=SEED) # Split the data into 70% train and 30% test
auto_dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED) # Instantiate a DecisionTreeRegressor dt



print("****************************************************")
tema = "6. Evaluate the 10-fold CV error"; print("** %s\n" % tema)

auto_MSE_CV_scores = - cross_val_score(auto_dt, auto_X_train, auto_y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1) # Compute the array containing the 10-folds CV MSEs
RMSE_CV = (auto_MSE_CV_scores.mean())**(1/2) # Compute the 10-folds CV RMSE

print('CV RMSE: {}'.format(RMSE_CV)) # Print RMSE_CV


print("****************************************************")
tema = "7. Evaluate the training error"; print("** %s\n" % tema)

auto_dt.fit(auto_X_train, auto_y_train) # Fit dt to the training set
auto_y_pred_train = auto_dt.predict(auto_X_train) # Predict the labels of the training set
auto_RMSE_train = (MSE(auto_y_train, auto_y_pred_train))**(1/2) # Evaluate the training set RMSE of dt

print('Train RMSE: {}'.format(auto_RMSE_train)) # Print RMSE_train



print("****************************************************")
tema = "8. High bias or high variance?"; print("** %s\n" % tema)

SEED = 3

auto_X_train, auto_X_test, auto_y_train, auto_y_test = train_test_split(auto_X, auto_y, test_size=0.2, random_state=SEED)
scalar = StandardScaler()

lr = LinearRegression()
pipeline = make_pipeline(scalar, lr)
pipeline.fit(auto_X_train, auto_y_train) # Fit dt to the training set
auto_y_pred = pipeline.predict(auto_X_test) # Compute y_pred
auto_mse_lr = MSE(auto_y_test, auto_y_pred) # Compute mse_dt
auto_rmse_lr = auto_mse_lr**(1/2) # Compute rmse_dt

print('Baseline RMSE: {}'.format(auto_rmse_lr)) # Print RMSE_train
print("This model suffers from high bias because RMSE_CV ≈ RMSE_train and both scores are greater than baseline_RMSE. \nIt's necessary to increase the complexity of the model")


print("****************************************************")
tema = "10. Define the ensemble"; print("** %s\n" % tema)

SEED=1 # Set seed for reproducibility

lr = LogisticRegression(solver='liblinear', random_state=SEED) # Instantiate lr
knn = KNN(n_neighbors=27) # Instantiate knn
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED) # Instantiate dt
#lr = LogisticRegression(n_jobs=1, multi_class='ovr', solver='liblinear', random_state=SEED) # Instantiate lr
#knn = KNN(n_jobs=1, n_neighbors=27) # Instantiate knn
#dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED) # Instantiate dt

classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)] # Define the list classifiers
print(lr.get_params)
print(knn.get_params)
print(dt.get_params)


print("****************************************************")
tema = "11. Evaluate individual classifiers"; print("** %s\n" % tema)

liver_X_train, liver_X_test, liver_y_train, liver_y_test = train_test_split(liver_X, liver_y, test_size=0.3, random_state=SEED)

for clf_name, clf in classifiers:                         # Iterate over the pre-defined list of classifiers
    clf.fit(liver_X_train, liver_y_train)                 # Fit clf to the training set
    liver_y_pred = clf.predict(liver_X_test)              # Predict y_pred
    accuracy = accuracy_score(liver_y_test, liver_y_pred) # Calculate accuracy
    print('{:s}: {:.6f}. Patients with a prediction of liver disease: {}.'.format(clf_name, accuracy, liver_y_pred.sum()))     # Evaluate clf's accuracy on the test set
    
    


print("****************************************************")
tema = "12. Better performance with a Voting Classifier"; print("** %s\n" % tema)

vc = VotingClassifier(estimators=classifiers)             # Instantiate a VotingClassifier vc
vc.fit(liver_X_train, liver_y_train)                      # Fit vc to the training set
liver_y_pred = vc.predict(liver_X_test)                   # Evaluate the test set predictions
accuracy = accuracy_score(liver_y_test, liver_y_pred)     # Calculate accuracy score

print('Voting Classifier: {:.6f}'.format(accuracy))

    
print("****************************************************")
print("** END                                            **")
print("****************************************************")