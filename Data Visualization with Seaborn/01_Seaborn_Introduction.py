# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:50:25 2019

@author: jacqueline.cortez
Chapter 1: Seaborn Introduction
    Introduction to the Seaborn library and where it fits in the Python visualization landscape.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import pandas            as pd                                                #For loading tabular data
import matplotlib.pyplot as plt                                               #For creating charts
import seaborn           as sns                                               #For visualizing data

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "white-wine.csv"
df_wine = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_wine.columns))
df_wine['type'] = 'good'
df_wine.loc[df_wine.quality<5, 'type'] = 'bad'

filename = "2010_US_School_Grants.csv"
df_school_grants = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_school_grants.columns))

filename = 'Automobile_Insurance_Premiums.csv'
df_automobile = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_automobile.columns))

print("****************************************************")
topic = "1. Introduction to Seaborn"; print("** %s\n" % topic)

plt.figure(figsize=(10,4)),
#Using Matplotlib
plt.subplot(1,3,1)
plt.hist(df_wine.alcohol)
plt.title('Using Matplotlib')

#Using Pandas plot integrating functions
plt.subplot(1,3,2)
df_wine.alcohol.plot.hist()
plt.title('Using Pandas Plot integrations functions')

#Using Seaborn
plt.subplot(1,3,3)
sns.distplot(df_wine.alcohol)
plt.title('Using Seaborn')

# Display the plot
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "4. Comparing a histogram and distplot"; print("** %s\n" % topic)

plt.figure(figsize=(10,4)),
#Using Pandas plot integrating functions
plt.subplot(1,2,1)
df_school_grants.Award_Amount.plot.hist()
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title('Using Pandas Plot integrations functions')

#Using Seaborn
plt.subplot(1,2,2)
#sns.set(font_scale=0.6)
sns.distplot(df_school_grants.Award_Amount)
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title('Using Seaborn')

# Display the plot
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "5. Using the distribution plot"; print("** %s\n" % topic)

plt.figure(figsize=(10,4)),
#Basic histogram
plt.subplot(1,3,1)
sns.distplot(df_wine.alcohol, kde=False, bins=10)
plt.title('A simple histogram')

#Making some customization
plt.subplot(1,3,2)
sns.distplot(df_wine.alcohol, hist=False, rug=True)
plt.title('Alternative data distribution')

#More customizations
plt.subplot(1,3,3)
sns.distplot(df_wine.alcohol, hist=False, rug=True, kde_kws={'shade':True})
plt.title('Further customizations')

# Display the plot
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "6. Plot a histogram"; print("** %s\n" % topic)
topic = "7. Rug plot and kde shading"; print("** %s\n" % topic)

plt.figure(figsize=(10,4)),
#Basic histogram
plt.subplot(1,2,1)
sns.distplot(df_school_grants.Award_Amount, kde=False, bins=20)
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title('Just a Histogram')

#A KDE with some customizations
plt.subplot(1,2,2)
#sns.set(font_scale=0.6)
sns.distplot(df_school_grants.Award_Amount, hist=False, rug=True, kde_kws={'shade':True})
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title('Using Seaborn')

# Display the plot
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "9. Regression Plots in Seaborn"; print("** %s\n" % topic)

#Introduction to regplot
plt.figure()
sns.regplot(x='alcohol', y='pH', data=df_wine)
plt.title('A regplot')
plt.suptitle(topic)
plt.show()

#Comparing regplot vs lmplot
plt.figure()
sns.regplot(x='alcohol', y='quality', data=df_wine)
plt.title('A regplot')
plt.suptitle(topic)
plt.show()

sns.lmplot(x='alcohol', y='quality', data=df_wine)
plt.title('A lmplot')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

#lmplot faceting
sns.lmplot(x='quality', y='alcohol', data=df_wine, hue='type')
plt.title('lmplot Faceting with hue')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

sns.lmplot(x='quality', y='alcohol', data=df_wine, col='type')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "10. Create a regression plot"; print("** %s\n" % topic)

#Regplot
plt.figure()
sns.regplot(data=df_automobile, x="insurance_losses", y="premiums")
plt.title('Using regplot')
plt.suptitle(topic)
plt.show()

#lmplot
sns.lmplot(data=df_automobile, x="insurance_losses", y="premiums")
plt.title('Using lmplot')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "11. Plotting multiple variables"; print("** %s\n" % topic)

# Create a regression plot using hue
sns.lmplot(data=df_automobile, x="insurance_losses", y="premiums", hue="Region")
plt.title('Relation between insurance losses and premiums, by Region')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=0.5)
plt.show()

print("****************************************************")
topic = "12. Facetting multiple regressions"; print("** %s\n" % topic)

# Create a regression plot with multiple rows
g = sns.lmplot(data=df_automobile, x="insurance_losses", y="premiums", row="Region", hue="Region")
g.fig.set_figheight(5) #Height and width of sns plot
g.fig.set_figwidth(6)
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.5)
plt.show()

# Create a regression plot with multiple rows
g = sns.lmplot(data=df_automobile,  x="insurance_losses", y="premiums", col="Region", hue="Region")
plt.suptitle(topic)
g.fig.set_figheight(5) #Height and width of sns plot
g.fig.set_figwidth(10)
plt.subplots_adjust(left=0.08, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.5)
plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import inspect                                                                #Used to get the code inside a function
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt                                               #For creating charts
#import numpy             as np                                                #For making operations in lists
#import pandas            as pd                                                #For loading tabular data
#import seaborn           as sns                                               #For visualizing data


#import calendar                                                               #For accesing to a vary of calendar operations
#import math                                                                   #For accesing to a complex math operations
#import nltk                                                                   #For working with text data
#import random                                                                 #For generating random numbers
#import re                                                                     #For regular expressions
#import tabula                                                                 #For extracting tables from pdf
#import timeit                                                                 #For Measure execution time of small code snippets
#import time                                                                   #To measure the elapsed wall-clock time between two points
#import scykit-learn                                                           #For performing machine learning  
#import warnings
#import wikipedia


#from collections                     import defaultdict                       #Returns a new dictionary-like object
#from datetime                        import date                              #For obteining today function
#from datetime                        import datetime                          #For obteining today function
#from functools                       import reduce                            #For accessing to a high order functions (functions or operators that return functions)
#from glob                            import glob                              #For using with pathnames matching
#from itertools                       import combinations                      #For iterations
#from itertools                       import cycle                             #Used in the function plot_labeled_decision_regions()
#from math                            import ceil                              #Used in the function plot_labeled_decision_regions()
#from math                            import floor                             #Used in the function plot_labeled_decision_regions()
#from math                            import radian                            #For accessing a specific math operations
#from mpl_toolkits.mplot3d            import Axes3D
#from pandas.api.types                import CategoricalDtype                  #For categorical data
#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
#from string                          import Template                          #For working with string, regular expressions


#from bokeh.io                        import curdoc                            #For interacting visualizations
#from bokeh.io                        import output_file                       #For interacting visualizations
#from bokeh.io                        import show                              #For interacting visualizations
#from bokeh.plotting                  import ColumnDataSource                  #For interacting visualizations
#from bokeh.plotting                  import figure                            #For interacting visualizations
#from bokeh.layouts                   import row                               #For interacting visualizations
#from bokeh.layouts                   import widgetbox                         #For interacting visualizations
#from bokeh.layouts                   import column                            #For interacting visualizations
#from bokeh.layouts                   import gridplot                          #For interacting visualizations
#from bokeh.models                    import HoverTool                         #For interacting visualizations
#from bokeh.models                    import ColumnDataSource                  #For interacting visualizations
#from bokeh.models                    import CategoricalColorMapper            #For interacting visualizations
#from bokeh.models                    import Slider                            #For interacting visualizations
#from bokeh.models                    import Select                            #For interacting visualizations
#from bokeh.models                    import Button                            #For interacting visualizations
#from bokeh.models                    import CheckboxGroup                     #For interacting visualizations
#from bokeh.models                    import RadioGroup                        #For interacting visualizations
#from bokeh.models                    import Toggle                            #For interacting visualizations
#from bokeh.models.widgets            import Tabs                              #For interacting visualizations
#from bokeh.models.widgets            import Panel                             #For interacting visualizations
#from bokeh.palettes                  import Spectral6                         #For interacting visualizations


#import keras                                                                  #For DeapLearning
#import keras.backend as k                                                     #For DeapLearning
#from keras.applications.resnet50     import decode_predictions                #For DeapLearning
#from keras.applications.resnet50     import preprocess_input                  #For DeapLearning
#from keras.applications.resnet50     import ResNet50                          #For DeapLearning
#from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
#from keras.callbacks                 import EarlyStopping                     #For DeapLearning
#from keras.datasets                  import fashion_mnist                     #For DeapLearning
#from keras.datasets                  import mnist                             #For DeapLearning
#from keras.layers                    import BatchNormalization                #For DeapLearning
#from keras.layers                    import Concatenate                       #For DeapLearning
#from keras.layers                    import Conv2D                            #For DeapLearning
#from keras.layers                    import Dense                             #For DeapLearning
#from keras.layers                    import Dropout                           #For DeapLearning
#from keras.layers                    import Embedding                         #For DeapLearning
#from keras.layers                    import Flatten                           #For DeapLearning
#from keras.layers                    import GlobalMaxPooling1D                #For DeapLearning
#from keras.layers                    import Input                             #For DeapLearning
#from keras.layers                    import LSTM                              #For DeapLearning
#from keras.layers                    import MaxPool2D                         #For DeapLearning
#from keras.layers                    import SpatialDropout1D                  #For DeapLearning
#from keras.layers                    import Subtract                          #For DeapLearning
#from keras.models                    import load_model                        #For DeapLearning
#from keras.models                    import Model                             #For DeapLearning
#from keras.models                    import Sequential                        #For DeapLearning
#from keras.optimizers                import Adam                              #For DeapLearning
#from keras.optimizers                import SGD                               #For DeapLearning
#from keras.preprocessing             import image                             #For DeapLearning
#from keras.preprocessing.text        import Tokenizer                         #For DeapLearning
#from keras.preprocessing.sequence    import pad_sequences                     #For DeapLearning
#from keras.utils                     import plot_model                        #For DeapLearning
#from keras.utils                     import to_categorical                    #For DeapLearning
#from keras.wrappers.scikit_learn     import KerasClassifier                   #For DeapLearning


#import networkx          as nx                                                #For Network Analysis in Python
#import nxviz             as nv                                                #For Network Analysis in Python
#from nxviz                           import ArcPlot                           #For Network Analysis in Python
#from nxviz                           import CircosPlot                        #For Network Analysis in Python 
#from nxviz                           import MatrixPlot                        #For Network Analysis in Python 

#import scipy.stats as stats                                                   #For accesign to a vary of statistics functiosn
#from scipy.cluster.hierarchy         import fcluster                          #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import dendrogram                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import linkage                           #For learning machine - unsurpervised
#from scipy.ndimage                   import gaussian_filter                   #For working with images
#from scipy.ndimage                   import median_filter                     #For working with images
#from scipy.signal                    import convolve2d                        #For learning machine - deep learning
#from scipy.sparse                    import csr_matrix                        #For learning machine 
#from scipy.special                   import expit as sigmoid                  #For learning machine 
#from scipy.stats                     import pearsonr                          #For learning machine 
#from scipy.stats                     import randint                           #For learning machine 
       
#from skimage                         import exposure                          #For working with images
#from skimage                         import measure                           #For working with images
#from skimage.filters.thresholding    import threshold_otsu                    #For working with images
#from skimage.filters.thresholding    import threshold_local                   #For working with images 

#from sklearn                         import datasets                          #For learning machine
#from sklearn.cluster                 import KMeans                            #For learning machine - unsurpervised
#from sklearn.decomposition           import NMF                               #For learning machine - unsurpervised
#from sklearn.decomposition           import PCA                               #For learning machine - unsurpervised
#from sklearn.decomposition           import TruncatedSVD                      #For learning machine - unsurpervised
#from sklearn.ensemble                import AdaBoostClassifier                #For learning machine - surpervised
#from sklearn.ensemble                import BaggingClassifier                 #For learning machine - surpervised
#from sklearn.ensemble                import GradientBoostingRegressor         #For learning machine - surpervised
#from sklearn.ensemble                import RandomForestClassifier            #For learning machine
#from sklearn.ensemble                import RandomForestRegressor             #For learning machine - unsurpervised
#from sklearn.ensemble                import VotingClassifier                  #For learning machine - unsurpervised
#from sklearn.feature_extraction.text import TfidfVectorizer                   #For learning machine - unsurpervised
#from sklearn.feature_selection       import chi2                              #For learning machine
#from sklearn.feature_selection       import SelectKBest                       #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                   #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                 #For learning machine
#from sklearn.impute                  import SimpleImputer                     #For learning machine
#from sklearn.linear_model            import ElasticNet                        #For learning machine
#from sklearn.linear_model            import Lasso                             #For learning machine
#from sklearn.linear_model            import LinearRegression                  #For learning machine
#from sklearn.linear_model            import LogisticRegression                #For learning machine
#from sklearn.linear_model            import Ridge                             #For learning machine
#from sklearn.manifold                import TSNE                              #For learning machine - unsurpervised
#from sklearn.metrics                 import accuracy_score                    #For learning machine
#from sklearn.metrics                 import classification_report             #For learning machine
#from sklearn.metrics                 import confusion_matrix                  #For learning machine
#from sklearn.metrics                 import mean_squared_error as MSE         #For learning machine
#from sklearn.metrics                 import roc_auc_score                     #For learning machine
#from sklearn.metrics                 import roc_curve                         #For learning machine
#from sklearn.model_selection         import cross_val_score                   #For learning machine
#from sklearn.model_selection         import KFold                             #For learning machine
#from sklearn.model_selection         import GridSearchCV                      #For learning machine
#from sklearn.model_selection         import RandomizedSearchCV                #For learning machine
#from sklearn.model_selection         import train_test_split                  #For learning machine
#from sklearn.multiclass              import OneVsRestClassifier               #For learning machine
#from sklearn.neighbors               import KNeighborsClassifier as KNN       #For learning machine
#from sklearn.pipeline                import FeatureUnion                      #For learning machine
#from sklearn.pipeline                import make_pipeline                     #For learning machine - unsurpervised
#from sklearn.pipeline                import Pipeline                          #For learning machine
#from sklearn.preprocessing           import FunctionTransformer               #For learning machine
#from sklearn.preprocessing           import Imputer                           #For learning machine
#from sklearn.preprocessing           import MaxAbsScaler                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing           import Normalizer                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing           import normalize                         #For learning machine - unsurpervised
#from sklearn.preprocessing           import scale                             #For learning machine
#from sklearn.preprocessing           import StandardScaler                    #For learning machine
#from sklearn.svm                     import SVC                               #For learning machine
#from sklearn.tree                    import DecisionTreeClassifier            #For learning machine - supervised
#from sklearn.tree                    import DecisionTreeRegressor             #For learning machine - supervised

#import statsmodels             as sm                                          #For stimations in differents statistical models
#import statsmodels.api         as sm                                          #Make a prediction model
#import statsmodels.formula.api as smf                                         #Make a prediction model    

#import tensorflow              as tf                                          #For DeapLearning




# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#pd.set_option('display.max_rows', -1)                                         #Shows all rows

#register_matplotlib_converters()                                              #Require to explicitly register matplotlib converters.

#Setting images params
#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09
#plt.rcParams.update({'figure.max_open_warning': 0})                           #To solve the max images open
#plt.rcParams["axes.labelsize"] = 8                                            #Font
#plt.rcParams['figure.max_open_warning'] = 60
#plt.style.use('dark_background')
#plt.style.use('default')

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.
#np.random.seed(SEED)

#tf.compat.v1.set_random_seed(SEED)                                            #Instead of tf.set_random_seed, because it is deprecated.

#sns.set(font_scale=0.8)                                                       #Font
#sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot
