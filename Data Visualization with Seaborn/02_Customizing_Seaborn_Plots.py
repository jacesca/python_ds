# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:27:30 2019

@author: jacqueline.cortez
Chapter 2: Customizing Seaborn Plots
    Overview of functions for customizing the display of Seaborn plots.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import pandas            as pd                                                #For loading tabular data
import matplotlib.pyplot as plt                                               #For creating charts
import seaborn           as sns                                               #For visualizing data

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

plt.rcParams['figure.max_open_warning'] = 60

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "2018_College_Scorecard_Tuition.csv"
df_college = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_college.columns))

filename = "US_Market_Rent.csv"
df_rent = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_rent.columns))

print("****************************************************")
topic = "1. Introduction to Seaborn"; print("** %s\n" % topic)

#Default Style
df_college.Tuition.plot.hist()
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.xlabel('Tuition')
plt.title('Pyplot Default Style')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()

#Seaborn Style
sns.set()
plt.figure()
df_college.Tuition.plot.hist()
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.xlabel('Tuition')
plt.title('Seaborn Style')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

#More Seaborn Styles
for this_style in ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
    plt.figure()
    sns.set_style(this_style)
    sns.distplot(df_college.loc[df_college.Tuition.notnull(), 'Tuition'])
    plt.title('{} Style'.format(this_style.capitalize()))
    plt.suptitle(topic)
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
    plt.show()
    plt.style.use('default')

#Seaborn Style
plt.figure()
sns.set_style('white')
sns.distplot(df_college.loc[df_college.Tuition.notnull(), 'Tuition'])
sns.despine(left=True)
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title('Left Despine Style')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
topic = "2. Setting the default style"; print("** %s\n" % topic)

# Plot the pandas histogram
plt.figure()
df_rent.fmr_2.plot.hist()
plt.xlabel('Fair market rent for a 2-bedroom apartment')
plt.title('Left Despine Style')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()

# Set the default seaborn style
plt.figure()
sns.set()
df_rent.fmr_2.plot.hist()
plt.title('Left Despine Style')
plt.xlabel('Fair market rent for a 2-bedroom apartment')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
topic = "3. Comparing styles"; print("** %s\n" % topic)

# Plot with a dark style 
plt.figure()
sns.set_style('dark')
sns.distplot(df_rent.fmr_2)
plt.title('Dark Style')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

# Plot with a dark style 
plt.figure()
sns.set_style('whitegrid')
sns.distplot(df_rent.fmr_2)
plt.title('Whitegrid Style')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
topic = "4. Removing spines"; print("** %s\n" % topic)

# Set the style to white
sns.set_style('white')
sns.lmplot(data=df_rent, x='pop2010', y='fmr_2') # Create a regression plo
sns.despine(left=True) # Remove the spines
plt.title('Relation between population and\nFair market rent for a 2-bedroom apartment')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
topic = "5. Colors in Seaborn"; print("** %s\n" % topic)

#Defining a color for a plot
plt.figure()
sns.set(color_codes=True)
sns.distplot(df_college.loc[df_college.Tuition.notnull(), 'Tuition'], color='g')
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.title('Defining a color for a plot')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

#Palettes
for p in sns.palettes.SEABORN_PALETTES:
    plt.figure()
    sns.set()
    sns.set_palette(p)
    sns.distplot(df_college.loc[df_college.Tuition.notnull(), 'Tuition'])
    plt.xticks(fontsize=7); plt.yticks(fontsize=8);
    plt.title('{} palette'.format(p))
    plt.suptitle(topic)
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
    plt.show()
    plt.style.use('default')
    
#Return the current palette
print('The current palette is: ', sns.color_palette())

#Display the palettes availables
for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.palplot(sns.color_palette())
    plt.title('{} palette'.format(p))
    plt.suptitle(topic)
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.5, wspace=None, hspace=None)
    plt.show()
plt.style.use('default')

#Define customs palettes
sns.palplot(sns.color_palette('Paired', 12))
plt.title('Circular colors')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.5, wspace=None, hspace=None)
plt.show()

sns.palplot(sns.color_palette('Blues', 12))
plt.title('Sequential colors')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.5, wspace=None, hspace=None)
plt.show()

sns.palplot(sns.color_palette('BrBG', 12))
plt.title('Diverging colors')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.5, wspace=None, hspace=None)
plt.show()

plt.style.use('default')

print("****************************************************")
topic = "6. Matplotlib color codes"; print("** %s\n" % topic)

# Set style, enable color code, and create a magenta distplot
plt.figure()
sns.set(color_codes=True)
sns.distplot(df_rent['fmr_3'], color='m')
plt.title('Fair market rent for a 3-bedroom apartment')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
topic = "7. Using default palettes"; print("** %s\n" % topic)

# Loop through differences between bright and colorblind palettes
plt.figure()
for i, p in enumerate(['bright', 'colorblind']):
    plt.subplot(1,2,i+1)
    sns.set_palette(p)
    sns.distplot(df_rent['fmr_3'])
    plt.title('{} palette'.format(p))
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()
plt.style.use('default')
    
print("****************************************************")
topic = "9. Creating Custom Palettes"; print("** %s\n" % topic)

# Create the coolwarm palette
sns.palplot(sns.color_palette("Purples", 8))
plt.title('Purples colors')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.5, wspace=None, hspace=None)
plt.show()

# Create the coolwarm palette
sns.palplot(sns.color_palette("husl", 10))
plt.title('Husl colors')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.5, wspace=None, hspace=None)
plt.show()

# Create the coolwarm palette
sns.palplot(sns.color_palette("coolwarm", 6))
plt.title('Coolwarm colors')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.5, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
topic = "10. Customizing with matplotlib"; print("** %s\n" % topic)

#Introducing axes
#plt.figure()
fig, ax = plt.subplots()
sns.distplot(df_college.loc[df_college.Tuition.notnull(), 'Tuition'], ax=ax)
ax.set(xlabel='Tuition 2018')
plt.title('Introducing Axes')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()

#More configurations on axes
#plt.figure()
fig, ax = plt.subplots()
sns.distplot(df_college.loc[df_college.Tuition.notnull(), 'Tuition'], ax=ax)
ax.set(xlabel = 'Tuition 2018',
       ylabel = 'Distribution',
       xlim   = (0, 50000),
       title  = 'More configurations on Axes')
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()

#Combining plots
#plt.figure()
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,4))
sns.distplot(df_college.loc[df_college.Tuition.notnull(), 'Tuition'], ax=ax0)
sns.distplot(df_college.query("Tuition.notnull() & Regions=='South East'", engine='python')['Tuition'], ax=ax1)
ax0.set(title  = 'All Region')
ax1.set(xlabel = 'Tuition (South East)',
        ylabel = 'Distribution',
        xlim   = (0, 70000),
        title  = 'Only South East Region')
ax1.axvline(x=20000, label='My budget', linestyle='--')
ax1.legend()
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "11. Using matplotlib axes"; print("** %s\n" % topic)

sns.set()
fig, ax = plt.subplots() # Create a figure and axes
sns.distplot(df_rent['fmr_3'], ax=ax) # Plot the distribution of data
ax.set(xlabel="3 Bedroom Fair Market Rent",
       title='US Market Rent Dataset') # Create a more descriptive x axis label
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show() # Show the plot
plt.style.use('default')

print("****************************************************")
topic = "12. Additional plot customizations"; print("** %s\n" % topic)

sns.set()
fig, ax = plt.subplots() # Create a figure and axes
sns.distplot(df_rent['fmr_1'], ax=ax) # Plot the distribution of 1 bedroom rents
ax.set(xlabel="1 Bedroom Fair Market Rent", # Modify the properties of the plot
       xlim=(100,1500),
       title="US Rent")
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show() # Show the plot
plt.style.use('default')

print("****************************************************")
topic = "13. Adding annotations"; print("** %s\n" % topic)

sns.set()
sns.set_palette('bright')
median = df_rent['fmr_1'].median()
mean = df_rent['fmr_1'].mean()

fig, ax = plt.subplots() # Create a figure and axes. Then plot the data
sns.distplot(df_rent['fmr_1'], ax=ax) # Plot the distribution of 1 bedroom rents
ax.set(xlabel = "1 Bedroom Fair Market Rent", # Customize the labels and limits
       xlim   = (100,1500), 
       title  = "US Rent")
ax.axvline(x=median, color='m', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
ax.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)
ax.legend() # Show the legend and plot the data
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show() # Show the plot
plt.style.use('default')

print("****************************************************")
topic = "14. Multiple plots"; print("** %s\n" % topic)

sns.set()
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,4)) # Create a plot with 1 row and 2 columns that share the y axis label

sns.distplot(df_rent['fmr_1'], ax=ax0) # Plot the distribution of 1 bedroom apartments on ax0
ax0.set(xlabel = "1 Bedroom Fair Market Rent", 
        xlim   = (100,1500),
        title  = 'US Rent for 1 Bedroom Fair Market')

sns.distplot(df_rent['fmr_2'], ax=ax1) # Plot the distribution of 2 bedroom apartments on ax1
ax1.set(xlabel = "2 Bedroom Fair Market Rent", 
        xlim   = (100,1500),
        title  = 'US Rent for 2 Bedroom Fair Market')
#ax1.tick_params(labelsize =8) #x_ticks, rotation=0, 

plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show() # Show the plot
plt.style.use('default')

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import inspect                                                                #Used to get the code inside a function
#import pandas            as pd                                                #For loading tabular data
#import numpy             as np                                                #For making operations in lists
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt                                               #For creating charts
#import seaborn           as sns                                               #For visualizing data
#import statsmodels       as sm                                                #For stimations in differents statistical models
#import networkx          as nx                                                #For Network Analysis in Python
#import nxviz             as nv                                                #For Network Analysis in Python


#import scykit-learn                                                           #For performing machine learning  
#import tabula                                                                 #For extracting tables from pdf
#import nltk                                                                   #For working with text data
#import math                                                                   #For accesing to a complex math operations
#import random                                                                 #For generating random numbers
#import calendar                                                               #For accesing to a vary of calendar operations
#import re                                                                     #For regular expressions
#import timeit                                                                 #For Measure execution time of small code snippets
#import time                                                                   #To measure the elapsed wall-clock time between two points
#import warnings
#import wikipedia


#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
#from math                            import radian                            #For accessing a specific math operations
#from functools                       import reduce                            #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types                import CategoricalDtype                  #For categorical data
#from glob                            import glob                              #For using with pathnames matching
#from datetime                        import date                              #For obteining today function
#from datetime                        import datetime                          #For obteining today function
#from string                          import Template                          #For working with string, regular expressions
#from itertools                       import cycle                             #Used in the function plot_labeled_decision_regions()
#from math                            import floor                             #Used in the function plot_labeled_decision_regions()
#from math                            import ceil                              #Used in the function plot_labeled_decision_regions()
#from itertools                       import combinations                      #For iterations
#from collections                     import defaultdict                       #Returns a new dictionary-like object
#from nxviz import ArcPlot                                                     #For Network Analysis in Python
#from nxviz import CircosPlot                                                  #For Network Analysis in Python 
#from nxviz import MatrixPlot                                                  #For Network Analysis in Python 


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

#import statsmodels.api as sm                                                  #Make a prediction model
#import statsmodels.formula.api as smf                                         #Make a prediction model    
#import tensorflow as tf                                                       #For DeapLearning

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
#plt.xticks(fontsize=7); plt.yticks(fontsize=8);

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.
#np.random.seed(SEED)

#tf.compat.v1.set_random_seed(SEED)                                            #Instead of tf.set_random_seed, because it is deprecated.

#sns.set(font_scale=0.8)                                                       #Font
#sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot
#sns.set(color_codes=True)                                                     #Habilita el uso de los codigos de color
#sns.set()                                                                     #Seaborn defult style
#sns.set_style(this_style)                                                     #['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
#sns.despine(left=True)                                                        #Remove the spines (all borders)
#sns.palettes.SEABORN_PALETTES                                                 #Despliega todas las paletas disponibles 
#sns.palplot(sns.color_palette())                                              #Display a palette
#sns.color_palette()                                                           #The current palette