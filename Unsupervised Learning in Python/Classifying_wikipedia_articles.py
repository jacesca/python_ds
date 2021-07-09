# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:24:05 2019

@author: jacqueline.cortez
"""

# Import packages
import pandas as pd                                                                 #For loading tabular data
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

#from scipy.cluster.hierarchy import fcluster                                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import dendrogram                                      #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import linkage                                         #For learning machine - unsurpervised
#from scipy.sparse import csr_matrix                                                 #For learning machine 
#from scipy.stats import pearsonr                                                    #For learning machine 
#from scipy.stats import randint                                                     #For learning machine 

from sklearn.cluster import KMeans                                                  #For learning machine - unsurpervised
#from sklearn.decomposition import PCA                                               #For learning machine - unsurpervised
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
print("** Initializing the script\n")

""" Read the articles from wikipedia
import time
import wikipedia

wikipedia_df = pd.read_csv('wikipedia-vectors.csv', index_col=0)
titles = list(wikipedia_df.columns)

articles = []
for i, title in enumerate(titles, 1):
    print("{} - {}".format(i, title))
    article = wikipedia.page(title)
    print(article.url)
    articles.append([title, article.content])
    time.sleep(5)
    
wikipedia_articles = np.array(articles)
wikipedia_df = pd.DataFrame({'title':wikipedia_articles[:,0], 'content':wikipedia_articles[:,1]})
wikipedia_df.to_csv('wikipedia_articles_df.csv', sep=';', quotechar='"', index=False)
"""
#TOKENS_ALPHANUMERIC = '[A-Za-z]+(?=\\s+)' # Create the token pattern: TOKENS_ALPHANUMERIC
#TOKENS_ALPHANUMERIC = '[A-Za-z]{3,}(?=\\s+)' # Create the token pattern: TOKENS_ALPHANUMERIC, pattern con 3 chars forward
#TOKENS_ALPHANUMERIC = '[A-Za-z]{3,}' # Create the token pattern: Palabras mayores de 3 letras alfanuméricas

file = 'wikipedia_articles_df.csv'
wikipedia_df = pd.read_csv(file, sep=';', quotechar='"', index_col='title')

file = 'wikipedia-vocabulary-utf8.txt'
vocabulary_df = pd.read_csv(file, names=['word'])
word_vocabulary = vocabulary_df.word.values

print("****************************************************")
print("** Look and read the articles\n")

titles = list(wikipedia_df.index)
content = wikipedia_df.content.values

print("****************************************************")
print("** Counting words in the articles\n")

np.random.seed(123) # Seed random number generator

#tfidf = TfidfVectorizer(analyzer='word', token_pattern=TOKENS_ALPHANUMERIC, max_features=13125) # Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer(vocabulary=word_vocabulary) # Create a TfidfVectorizer: tfidf
content_vectorizer = tfidf.fit_transform(content) # Apply fit_transform to document: csr_mat
print(content_vectorizer.toarray()) # Print result of toarray() method
print(content_vectorizer.toarray().shape) # Print result of toarray() method
words = tfidf.get_feature_names() # Get the words: words
print(words[:15]) # Print words

print("****************************************************")
print("** Applying reduction techniques\n")

svd = TruncatedSVD(n_components=50) # Create a TruncatedSVD instance: svd
kmeans = KMeans(n_clusters=6) # Create a KMeans instance: kmeans
pipeline = make_pipeline(svd, kmeans) # Create a pipeline: pipeline

print("****************************************************")
tema = " Clustering the articles from Wikipedia"; print("** %s\n" % tema)

pipeline.fit(content_vectorizer) # Fit the pipeline to articles
labels = pipeline.predict(content_vectorizer) # Calculate the cluster labels: labels

df = pd.DataFrame({'article': titles, 'label': labels}) # Create a DataFrame aligning labels and titles: df
print(df.sort_values('label')) # Display df sorted by cluster label

print("****************************************************")
print("** END                                            **")
print("****************************************************")