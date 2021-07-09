# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:25:38 2019

@author: jacqueline.cortez

Capítulo 3. Decorrelating your data and dimension reduction
Introduction:
    Dimension reduction summarizes a dataset using its common occuring patterns. In this chapter, you'll learn about the most 
    fundamental of dimension reduction techniques, "Principal Component Analysis" ("PCA"). PCA is often used before supervised 
    learning to improve model performance and generalization. It can also be useful for unsupervised learning. For example, 
    you'll employ a variant of PCA will allow you to cluster Wikipedia articles by their content!
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

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions

#from scipy.cluster.hierarchy import fcluster                                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import dendrogram                                      #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import linkage                                         #For learning machine - unsurpervised
#from scipy.sparse import csr_matrix                                                 #For learning machine 
from scipy.stats import pearsonr                                                    #For learning machine 
#from scipy.stats import randint                                                     #For learning machine 

from sklearn.cluster import KMeans                                                  #For learning machine - unsurpervised
from sklearn.decomposition import PCA                                               #For learning machine - unsurpervised
from sklearn.decomposition import TruncatedSVD                                      #For learning machine - unsurpervised
#from sklearn.ensemble import RandomForestClassifier                                 #For learning machine
from sklearn.feature_extraction.text import TfidfVectorizer                         #For learning machine - unsurpervised
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
#from sklearb.metrics import accuracy_score                                          #For learning machine
#from sklearn.metrics import classification_report                                   #For learning machine
#from sklearn.metrics import confusion_matrix                                        #For learning machine
#from sklearn.metrics import mean_squared_error                                      #For learning machine
#from sklearn.metrics import roc_auc_score                                           #For learning machine
#from sklearn.metrics import roc_curve                                               #For learning machine
#from sklearn.model_selection import cross_val_score                                 #For learning machine
#from sklearn.model_selection import GridSearchCV                                    #For learning machine
#from sklearn.model_selection import RandomizedSearchCV                              #For learning machine
#from sklearn.model_selection import train_test_split                                #For learning machine
#from sklearn.multiclass import OneVsRestClassifier                                   #For learning machine
#from sklearn.neighbors import KNeighborsClassifier                                 #For learning machine
#from sklearn.pipeline import FeatureUnion                                           #For learning machine
from sklearn.pipeline import make_pipeline                                          #For learning machine - unsurpervised
#from sklearn.pipeline import Pipeline                                               #For learning machine
#from sklearn.preprocessing import FunctionTransformer                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
#from sklearn.preprocessing import MaxAbsScaler                                      #For learning machine
#from sklearn.preprocessing import Normalizer                                        #For learning machine - unsurpervised
#from sklearn.preprocessing import normalize                                         #For learning machine - unsurpervised
#from sklearn.preprocessing import scale                                             #For learning machine
from sklearn.preprocessing import StandardScaler                                    #For learning machine
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


file = 'seeds-width-vs-length.csv'
grain_wl = pd.read_csv(file, header=None, names=['width', 'length'])
grain_width = grain_wl.length.values # Assign the 0th column of grains: width
grain_length = grain_wl.width.values # Assign the 1st column of grains: length
grain_wl_samples = grain_wl.values


file = 'fish.csv'
fish = pd.read_csv(file, header=None, names=['fish_class','f1','f2','f3','f4','f5','f6'])
fish_samples = fish.drop('fish_class', axis=1).values
fish_labels = fish['fish_class'].values


phrases = ['cats say meow', 'dogs say woof', 'dogs chase cats']

wikipedia = pd.read_csv('wikipedia-vectors.csv', index_col=0)
titles = list(wikipedia.columns)
#articles = csr_matrix(wikipedia.transpose())


print("****************************************************")
tema = "2. Correlated data in nature"; print("** %s\n" % tema)

# Scatter plot width vs length
sns.set() # Set default Seaborn style
#plt.figure()
plt.scatter(grain_width, grain_length)
plt.axis('square')
#plt.xlabel('Width')
#plt.ylabel('Length')
plt.title('Correlation between width and length in grains samples')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


correlation, pvalue = pearsonr(grain_width, grain_length) # Calculate the Pearson correlation
print(correlation) # Display the correlation



print("****************************************************")
tema = "3. Decorrelating the grain measurements with PCA"; print("** %s\n" % tema)

model = PCA() # Create PCA instance: model
pca_features = model.fit_transform(grain_wl_samples) # Apply the fit_transform method of model to grains: pca_features

xs = pca_features[:,0] # Assign 0th column of pca_features: xs
ys = pca_features[:,1] # Assign 1st column of pca_features: ys


# Scatter plot xs vs ys
sns.set() # Set default Seaborn style
plt.figure()
plt.scatter(xs, ys)
plt.axis('square')
#plt.xlabel('Width')
#plt.ylabel('Length')
plt.title('Correlation between width and length in grains samples')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

correlation, pvalue = pearsonr(xs, ys) # Calculate the Pearson correlation of xs and ys
print(correlation) # Display the correlation



print("****************************************************")
tema = "6. The first principal component"; print("** %s\n" % tema)

# Scatter plot xs vs ys
sns.set() # Set default Seaborn style
plt.figure()
plt.scatter(grain_wl_samples[:,0], grain_wl_samples[:,1]) # Make a scatter plot of the untransformed points

model = PCA() # Create a PCA instance: model
model.fit(grain_wl_samples) # Fit model to points

mean = model.mean_ # Get the mean of the grain samples: mean
first_pc = model.components_[0,:] # Get the first principal component: first_pc
second_pc = model.components_[1,:]

plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01) # Plot first_pc as an arrow, starting at mean
plt.text(4, 6.2,"Align data.", color='red')
plt.arrow(mean[0], mean[1], second_pc[0], second_pc[1], color='green', width=0.01) # Plot first_pc as an arrow, starting at mean
plt.text(2.25, 5.85,"No align data.", color='green')
plt.xlim(2, 4.5)
plt.ylim(4.5, 7)
#plt.axis('square')
#plt.xlabel('Width')
#plt.ylabel('Length')
plt.title('Represantations of grains samples')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "7. Variance of the PCA features"; print("** %s\n" % tema)

scaler = StandardScaler() # Create scaler: scaler
pca = PCA() # Create a PCA instance: pca
pipeline = make_pipeline(scaler, pca) # Create pipeline: pipeline
pipeline.fit(fish_samples) # Fit the pipeline to 'samples'

features = range(pca.n_components_)

# Plot the explained variances
sns.set() # Set default Seaborn style
plt.figure()
plt.bar(features, pca.explained_variance_)
#plt.xlim(2, 4.5); plt.ylim(4.5, 7);
#plt.axis('square')
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.title('Features of fish samples')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "10. Dimension reduction of the fish measurements"; print("** %s\n" % tema)

scaled_samples = scaler.transform(fish_samples)

pca = PCA(n_components=2) # Create a PCA model with 2 components: pca
pca.fit(scaled_samples) # Fit the PCA instance to the scaled samples
pca_features = pca.transform(scaled_samples) # Transform the scaled samples: pca_features

print(pca_features.shape) # Print the shape of pca_features



print("****************************************************")
tema = "11. A tf-idf word-frequency array"; print("** %s\n" % tema)

tfidf = TfidfVectorizer() # Create a TfidfVectorizer: tfidf
csr_mat = tfidf.fit_transform(phrases) # Apply fit_transform to document: csr_mat
print(csr_mat.toarray()) # Print result of toarray() method

words = tfidf.get_feature_names() # Get the words: words
print(words) # Print words



print("****************************************************")
tema = "12. Clustering Wikipedia part I"; print("** %s\n" % tema)

np.random.seed(123) # Seed random number generator

tfidf = TfidfVectorizer() # Create a TfidfVectorizer: tfidf
articles = tfidf.fit_transform(titles) # Apply fit_transform to document: csr_mat
print(articles.toarray()) # Print result of toarray() method
words = tfidf.get_feature_names() # Get the words: words
print(words[:15]) # Print words



svd = TruncatedSVD(n_components=50) # Create a TruncatedSVD instance: svd
kmeans = KMeans(n_clusters=6) # Create a KMeans instance: kmeans
pipeline = make_pipeline(svd, kmeans) # Create a pipeline: pipeline


print("****************************************************")
tema = "13. Clustering Wikipedia part II"; print("** %s\n" % tema)

pipeline.fit(articles) # Fit the pipeline to articles
labels = pipeline.predict(articles) # Calculate the cluster labels: labels

df = pd.DataFrame({'label': labels, 'article': titles}) # Create a DataFrame aligning labels and titles: df
print(df.sort_values('label')) # Display df sorted by cluster label

print("****************************************************")
print("** END                                            **")
print("****************************************************")