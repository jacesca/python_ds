# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:07:20 2019

@author: jacqueline.cortez

Capítulo 2. Visualization with hierarchical clustering and t-SNE
Introduction:
    In this chapter, you'll learn about two unsupervised learning techniques for data visualization, hierarchical clustering 
    and t-SNE. Hierarchical clustering merges the data samples into ever-coarser clusters, yielding a tree visualization of 
    the resulting cluster hierarchy. t-SNE maps the data samples into 2d space so that the proximity of the samples to one 
    another can be visualized.
"""

# Import packages
import pandas as pd                                                                 #For loading tabular data
#import numpy as np                                                                  #For making operations in lists
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

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions

from scipy.cluster.hierarchy import fcluster                                        #For learning machine - unsurpervised
from scipy.cluster.hierarchy import dendrogram                                      #For learning machine - unsurpervised
from scipy.cluster.hierarchy import linkage                                         #For learning machine - unsurpervised
#from scipy.stats import randint                                                     #For learning machine 

#from sklearn.cluster import KMeans                                                  #For learning machine - unsurpervised
#from sklearn.ensemble import RandomForestClassifier                                 #For learning machine
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
from sklearn.manifold import TSNE                                                   #For learning machine - unsurpervised
#from sklearb.metrics import accuracy_score                                          #For learning machine
#from sklearn.metrics import classification_report                                   #For learning machine
#from sklearn.metrics import confusion_matrix                                        #For learning machine
#from sklearn.metrics import mean_squared_error                                      #For learning machine
#from sklearn.metrics import roc_auc_score                                           #For learning machine
#from sklearn.metrics import roc_curve                                               #For learning machine
#from sklearn.model_selection import cross_val_score                                 #For learning machine
#from sklearn.model_selection import GridSearchCV                                    #For learning machine
#from sklearn.model_selection import RandomizedSearchCV                              #For learning machine
from sklearn.model_selection import train_test_split                                #For learning machine
#from sklearn.multiclass import OneVsRestClassifier                                   #For learning machine
#from sklearn.neighbors import KNeighborsClassifier                                 #For learning machine
#from sklearn.pipeline import FeatureUnion                                           #For learning machine
#from sklearn.pipeline import make_pipeline                                          #For learning machine - unsurpervised
#from sklearn.pipeline import Pipeline                                               #For learning machine
#from sklearn.preprocessing import FunctionTransformer                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
#from sklearn.preprocessing import MaxAbsScaler                                      #For learning machine
#from sklearn.preprocessing import Normalizer                                        #For learning machine - unsurpervised
from sklearn.preprocessing import normalize                                         #For learning machine - unsurpervised
#from sklearn.preprocessing import scale                                             #For learning machine
#from sklearn.preprocessing import StandardScaler                                    #For learning machine
#from sklearn.svm import SVC                                                         #For learning machine
#from sklearn.tree import DecisionTreeClassifier                                     #For learning machine
#import multilabel                                                                   #For multivariable target, function created by Datacamp
#import multi_log_loss                                                               #Datacamp logloss for multiple targets score
#from SparseInteractions import SparseInteractions                                   #Implement interaction modeling like PolynomialFeatures

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


print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

file = 'seeds.csv'
seed = pd.read_csv(file, header=None, names=['area', 'perimeter', 'compactness', 'length', 'width', 'coefficient', 'groove', 'varieties'])
grain_samples = seed.drop('varieties', axis=1).values
grain_varieties = seed.varieties.values
seed_varieties = {1: 'Kama wheat', 2: 'Rosa wheat', 3: 'Canadian wheat'}
seed['variety_name'] = seed.varieties.map(seed_varieties)
grain_labels = seed['variety_name'].values

file = 'company-stock-movements-2010-2015-incl.csv'
movements = pd.read_csv(file)
movement_samples = movements.drop('Unnamed: 0', axis=1).values
movement_companies = movements['Unnamed: 0'].values


file = 'eurovision-2016.csv'
eurovision = pd.read_csv(file, usecols=['From country', 'To country', 'Televote Points'])
eurovision = eurovision.pivot(index='From country', columns='To country', values='Televote Points')
eurovision.fillna(0, inplace=True)
eurovision_samples = eurovision.values
eurovision_labels = eurovision.index.values


print("****************************************************")
tema = "3. Hierarchical clustering of the grain data"; print("** %s\n" % tema)

samples_train, samples_test, label_train, label_test = train_test_split(grain_samples, seed.variety_name.values, random_state=42, train_size=0.25)
# Calculate the linkage: mergings
mergings = linkage(samples_train, method='complete')

# Plot the dendrogram, using varieties as labels
sns.set() # Set default Seaborn style
dendrogram(mergings, labels=label_train, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Seed Samples')
#plt.ylabel('Levels of clustering')
plt.title('A Dendrogram of Seed Samples')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "4. Hierarchies of stocks"; print("** %s\n" % tema)

normalized_movements = normalize(movement_samples) # Normalize the movements: normalized_movements
mergings = linkage(normalized_movements, method='complete') # Calculate the linkage: mergings

# Plot the dendrogram
sns.set() # Set default Seaborn style
plt.figure()
dendrogram(mergings, labels=movement_companies, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Companies names')
#plt.ylabel('Levels of clustering')
plt.title('A Dendrogram for Companies Movements Samples')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.45, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "7. Different linkage, different hierarchical clustering!"; print("** %s\n" % tema)

#############################################################
###### METHOD = COMPLETE
#############################################################
mergings = linkage(eurovision_samples, method='complete') # Calculate the linkage: mergings

# Plot the dendrogram
sns.set() # Set default Seaborn style
plt.figure()
dendrogram(mergings, labels=eurovision_labels, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Countries')
plt.ylabel('Cluster distance')
plt.title('Dendrogram with COMPLETE method in Linkage')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#############################################################
###### METHOD = SINGLE
#############################################################
mergings = linkage(eurovision_samples, method='single') # Calculate the linkage: mergings

# Plot the dendrogram
sns.set() # Set default Seaborn style
plt.figure()
dendrogram(mergings, labels=eurovision_labels, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Countries')
plt.ylabel('Cluster distance')
plt.title('Dendrogram with SINGLE method in Linkage')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#############################################################
###### METHOD = AVERAGE
#############################################################
mergings = linkage(eurovision_samples, method='average') # Calculate the linkage: mergings

# Plot the dendrogram
sns.set() # Set default Seaborn style
plt.figure()
dendrogram(mergings, labels=eurovision_labels, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Countries')
plt.ylabel('Cluster distance')
plt.title('Dendrogram with AVERAGE method in Linkage')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#############################################################
###### METHOD = WEIGHTED
#############################################################
mergings = linkage(eurovision_samples, method='weighted') # Calculate the linkage: mergings

# Plot the dendrogram
sns.set() # Set default Seaborn style
plt.figure()
dendrogram(mergings, labels=eurovision_labels, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Countries')
plt.ylabel('Cluster distance')
plt.title('Dendrogram with WEIGHTED method in Linkage')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#############################################################
###### METHOD = CENTROID
#############################################################
mergings = linkage(eurovision_samples, method='centroid') # Calculate the linkage: mergings

# Plot the dendrogram
sns.set() # Set default Seaborn style
plt.figure()
dendrogram(mergings, labels=eurovision_labels, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Countries')
plt.ylabel('Cluster distance')
plt.title('Dendrogram with CENTROID method in Linkage')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#############################################################
###### METHOD = MEDIAN
#############################################################
mergings = linkage(eurovision_samples, method='median') # Calculate the linkage: mergings

# Plot the dendrogram
sns.set() # Set default Seaborn style
plt.figure()
dendrogram(mergings, labels=eurovision_labels, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Countries')
plt.ylabel('Cluster distance')
plt.title('Dendrogram with MEDIAN method in Linkage')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#############################################################
###### METHOD = WARD
#############################################################
mergings = linkage(eurovision_samples, method='ward') # Calculate the linkage: mergings

# Plot the dendrogram
sns.set() # Set default Seaborn style
plt.figure()
dendrogram(mergings, labels=eurovision_labels, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Countries')
plt.ylabel('Cluster distance')
plt.title('Dendrogram with WARD method in Linkage')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = "9. Extracting the cluster labels"; print("** %s\n" % tema)

mergings = linkage(grain_samples, method='complete') # Calculate the linkage: mergings
labels = fcluster(mergings, 8, criterion='distance') # Use fcluster to extract labels: labels



# Plot the dendrogram, using varieties as labels
sns.set() # Set default Seaborn style
plt.figure()
dendrogram(mergings, labels=grain_labels, leaf_rotation=90, leaf_font_size=6)
plt.xlabel('Seed Samples')
#plt.ylabel('Levels of clustering')
plt.title('A Dendrogram of Seed Samples')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



df = pd.DataFrame({'labels': labels, 'varieties': grain_labels}) # Create a DataFrame with labels and varieties as columns: df
ct = pd.crosstab(df['labels'], df['varieties']) # Create crosstab: ct
print(ct) # Display ct



print("****************************************************")
tema = "11. t-SNE visualization of grain dataset"; print("** %s\n" % tema)

model = TSNE(learning_rate=200) # Create a TSNE instance: model
tsne_features = model.fit_transform(grain_samples) # Apply fit_transform to samples: tsne_features

xs = tsne_features[:,0] # Select the 0th feature: xs
ys = tsne_features[:,1] # Select the 1st feature: ys

# Scatter plot, coloring by variety_numbers
#sns.set() # Set default Seaborn style
plt.figure()
plt.scatter(xs, ys, c=grain_varieties)
#plt.xlabel('Seed Samples')
#plt.ylabel('Levels of clustering')
plt.title('Seed Samples')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
plt.show()
#plt.style.use('default')



print("****************************************************")
tema = "12. A t-SNE map of the stock market"; print("** %s\n" % tema)

model = TSNE(learning_rate=50) # Create a TSNE instance: model
tsne_features = model.fit_transform(normalized_movements) # Apply fit_transform to normalized_movements: tsne_features
xs = tsne_features[:,0] # Select the 0th feature: xs
ys = tsne_features[:,1] # Select the 1th feature: ys

#sns.set() # Set default Seaborn style
plt.figure()
plt.scatter(xs, ys, alpha=0.5) # Scatter plot
for x, y, company in zip(xs, ys, movement_companies): # Annotate the points
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
#plt.xlabel('Seed Samples')
#plt.ylabel('Levels of clustering')
plt.title('Companies Samples')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
plt.show()
#plt.style.use('default')

print("****************************************************")
print("** END                                            **")
print("****************************************************")