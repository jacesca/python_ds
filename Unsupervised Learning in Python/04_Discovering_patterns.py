# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 20:04:12 2019

@author: jacqueline.cortez

Capítulo 4. Discovering interpretable features
Introduction:
    In this chapter, you'll learn about a dimension reduction technique called "Non-negative matrix factorization" ("NMF") 
    that expresses samples as combinations of interpretable parts. For example, it expresses documents as combinations of 
    topics, and images in terms of commonly occurring visual patterns. You'll also learn to use NMF to build recommender 
    systems that can find you similar articles to read, or musical artists that match your listening history!
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
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions

#from scipy.cluster.hierarchy import fcluster                                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import dendrogram                                      #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import linkage                                         #For learning machine - unsurpervised
from scipy.sparse import csr_matrix                                                 #For learning machine 
#from scipy.stats import pearsonr                                                    #For learning machine 
#from scipy.stats import randint                                                     #For learning machine 

#from sklearn.cluster import KMeans                                                  #For learning machine - unsurpervised
from sklearn.decomposition import NMF                                               #For learning machine - unsurpervised
from sklearn.decomposition import PCA                                               #For learning machine - unsurpervised
#from sklearn.decomposition import TruncatedSVD                                      #For learning machine - unsurpervised
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
from sklearn.preprocessing import MaxAbsScaler                                      #For learning machine (transforms the data so that all users have the same influence on the model)
from sklearn.preprocessing import Normalizer                                        #For learning machine - unsurpervised (for pipeline)
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

#Setting images params
#plt.rcParams.update({'figure.max_open_warning': 0}) #To solve the max images open

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** User defined functions \n")

def show_as_image(sample, title, suptitle):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.suptitle(suptitle)
    plt.show()
    
def show_as_image_inrd(sample, title, suptitle):
    """
    Given a 1d vector representing an image, display that image in 
    black and white.  If there are negative values, then use red for 
    that pixel. 
    (Displaying negative pixel values in red allows e.g. visual contrasting
    of PCA and NMF components)
    """
    bitmap = sample.copy().reshape((13, 8))  # make a square array
    bitmap /= np.abs(sample).max()  # normalise (a copy!)
    bitmap = bitmap[:,:,np.newaxis]
    rgb_layers = [np.abs(bitmap)] + [bitmap.clip(0)] * 2
    rgb_bitmap = np.concatenate(rgb_layers, axis=-1)
    plt.figure()
    plt.imshow(rgb_bitmap, interpolation='nearest')
    plt.title(title)
    plt.suptitle(suptitle)
    plt.show()
    
print("****************************************************")
print("** Getting the data for this program\n")

file = 'wikipedia-vocabulary-utf8.txt'
vocabulary_df = pd.read_csv(file, names=['word'])
word_vocabulary = vocabulary_df.word.values


file = 'wikipedia_articles_df.csv'
wikipedia_df = pd.read_csv(file, sep=';', quotechar='"', index_col='title')
titles = list(wikipedia_df.index)
content = wikipedia_df.content.values
np.random.seed(123) # Seed random number generator
tfidf = TfidfVectorizer(vocabulary=word_vocabulary) # Create a TfidfVectorizer: tfidf
content_vectorizer = tfidf.fit_transform(content) # Apply fit_transform to document: csr_mat
#words = tfidf.get_feature_names() # Get the words: words


file = 'lcd-digits.csv'
digits_df = pd.read_csv(file, header=None)
digits_samples = digits_df.values



file = 'artists.csv'
artists_df = pd.read_csv(file, header=None, names=['name'])
artist_names = list(artists_df.name.values)
file = 'artists-scrobbler-small-sample.csv'
artists_users_df = pd.read_csv(file)
artists_users = csr_matrix(artists_users_df.pivot(index='artist_offset', columns='user_offset', values='playcount').fillna(0))


print("****************************************************")
tema = "3. NMF applied to Wikipedia articles"; print("** %s\n" % tema)

model = NMF(n_components=6) # Create an NMF instance: model

model.fit(content_vectorizer) # Fit the model to articles
nmf_features = model.transform(content_vectorizer) # Transform the articles: nmf_features

print("nmf_features: \n{}".format(nmf_features)) # Print the NMF features
print("nmf_features shape: \n{}".format(nmf_features.shape)) # Print the NMF features


print("****************************************************")
tema = "4. NMF features of the Wikipedia articles"; print("** %s\n" % tema)

df_wikipedia_articles = pd.DataFrame(nmf_features, index=titles) # Create a pandas DataFrame: df

print(df_wikipedia_articles.loc['Anne Hathaway']) # Print the row for 'Anne Hathaway'
print(df_wikipedia_articles.loc['Denzel Washington']) # Print the row for 'Denzel Washington'



print("****************************************************")
tema = "7. NMF learns topics of documents"; print("** %s\n" % tema)

components_df = pd.DataFrame(model.components_, columns=word_vocabulary) # Create a DataFrame: components_df
print("components_df: \n{}".format(components_df))
print("components_df shape: {}".format(components_df.shape)) # Print the shape of the DataFrame

component = components_df.iloc[3] # Select row 3: component
print(component.nlargest()) # Print result of nlargest


print("****************************************************")
tema = "8. Explore the LED digits dataset"; print("** %s\n" % tema)

print("Size of digit samples: ", digits_samples.shape)
digit = digits_samples[0,:] # Select the 0th row: digit
print(digit) # Print digit

bitmap = digit.reshape(13, 8) # Reshape digit to a 13x8 array:0 bitmap
print(bitmap) # Print bitmap


# Use plt.imshow to display bitmap
#sns.set() # Set default Seaborn style
#plt.figure()
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
#plt.axis('square')
#plt.xlabel('Width')
#plt.ylabel('Length')
plt.title('One digit sample')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
plt.show()
#plt.style.use('default')


print("****************************************************")
tema = "9. NMF learns the parts of images"; print("** %s\n" % tema)

nmf_model = NMF(n_components=7) # Create an NMF model: model
digit_features = nmf_model.fit_transform(digits_samples) # Apply fit_transform to samples: features

# Call show_as_image on each component
for i, digit_component in enumerate(nmf_model.components_, 1):
    show_as_image(digit_component, "Component No.{}".format(i), tema)

digit_features_selected = digit_features[0,:] # Assign the 0th row of features: digit_features
print(digit_features_selected) # Print digit_features



print("****************************************************")
tema = "10. PCA doesn't learn parts"; print("** %s\n" % tema)

pca_model = PCA(n_components=7) # Create a PCA instance: model
digit_features = pca_model.fit_transform(digits_samples) # Apply fit_transform to samples: features

# Call show_as_image on each component
for i, digit_component in enumerate(pca_model.components_, 1):
    show_as_image_inrd(digit_component, "Component No.{}".format(i), tema)


print("****************************************************")
tema = "12. Which articles are similar to 'Cristiano Ronaldo'?"; print("** %s\n" % tema)

norm_features = normalize(nmf_features) # Normalize the NMF features: norm_features
df = pd.DataFrame(norm_features, index=titles) # Create a DataFrame: df
article = df.loc['Cristiano Ronaldo'] # Select the row corresponding to 'Cristiano Ronaldo': article
similarities = df.dot(article) # Compute the dot products: similarities
print(similarities.nlargest()) # Display those with the largest cosine similarity


print("****************************************************")
tema = "13. Recommend musical artists part I"; print("** %s\n" % tema)

scaler = MaxAbsScaler() # Create a MaxAbsScaler: scaler
nmf = NMF(n_components=20) # Create an NMF model: nmf
normalizer = Normalizer() # Create a Normalizer: normalizer
pipeline = make_pipeline(scaler, nmf, normalizer) # Create a pipeline: pipeline
norm_features = pipeline.fit_transform(artists_users) # Apply fit_transform to artists: norm_features



print("****************************************************")
tema = "14. Recommend musical artists part II"; print("** %s\n" % tema)

df = pd.DataFrame(norm_features, index=artist_names) # Create a DataFrame: df
artist = df.loc['Bruce Springsteen'] # Select row of 'Bruce Springsteen': artist
similarities = df.dot(artist) # Compute cosine similarities: similarities
print(similarities.nlargest()) # Display those with highest cosine similarity


print("****************************************************")
print("** END                                            **")
print("****************************************************")