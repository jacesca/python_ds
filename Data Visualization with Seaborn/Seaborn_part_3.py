# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:26:27 2019

@author: jacqueline.cortez
source:
    https://jovianlin.io/data-visualization-seaborn-part-3/
    https://github.com/dipanjanS/practical-machine-learning-with-python/tree/master/bonus%20content/effective%20data%20visualization
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates

print("****************************************************")
topic = "1. Global user variables"; print("** %s\n" % topic)

SEED=42

print("****************************************************")
topic = "2. Preparing the environment"; print("** %s\n" % topic)

np.random.seed(SEED)
pd.set_option("display.max_columns",20)

print("****************************************************")
topic = "3. Reading data"; print("** %s\n" % topic)

file = 'winequality-red.csv'
df_red_wine   = pd.read_csv(file,   sep=';')
#print('Head of {}:\n{}'.format(file, df_red_wine.head()))

file = 'winequality-white.csv'
df_white_wine = pd.read_csv(file,   sep=';')
#print('Head of {}:\n{}'.format(file, df_red_wine.head()))

print("****************************************************")
topic = "4. Feature engineering"; print("** %s\n" % topic)

df_red_wine['wine_type'] = 'red'   
df_white_wine['wine_type'] = 'white'

print('Red_wine\'s list of "quality":\t', sorted(df_red_wine['quality'].unique()))
print('White_wine\'s list of "quality":\t', sorted(df_white_wine['quality'].unique()))

df_red_wine['quality_label'] = df_red_wine['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
df_red_wine['quality_label'] = pd.Categorical(df_red_wine['quality_label'], categories=['low', 'medium', 'high'])

df_white_wine['quality_label'] = df_white_wine['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
df_white_wine['quality_label'] = pd.Categorical(df_white_wine['quality_label'], categories=['low', 'medium', 'high'])

#print('Head of Red_wine:\n{}'.format(df_red_wine.head()))
#print('Head of White_wine:\n{}'.format(df_red_wine.head()))

print("****************************************************")
topic = "5. Merging datasets"; print("** %s\n" % topic)

df_wines = pd.concat([df_red_wine, df_white_wine], axis=0)

# Re-shuffle records just to randomize data points.
# `drop=True`: this resets the index to the default integer index.
df_wines = df_wines.sample(frac=1.0, random_state=42).reset_index(drop=True)
print('Head of Wine dataset:\n{}'.format(df_wines.head()))

print("****************************************************")
topic = "6. Descriptive statistics"; print("** %s\n" % topic)

#First view
subset_attributes = ['residual sugar',        #1
                     'total sulfur dioxide',  #2
                     'sulphates',             #3
                     'alcohol',               #4
                     'volatile acidity',      #5
                     'quality']               #6

rs = round(df_red_wine[subset_attributes].describe(), 2)
#print('Red wine:\n{}'.format(rs))

ws = round(df_white_wine[subset_attributes].describe(), 2)
#print('\nWhite wine:\n{}'.format(ws))

print('First view...')
print(pd.concat([rs, ws], axis=1, 
                keys=['🔴 Red Wine Statistics', 
                      '⚪️ White Wine Statistics']))


#Second view
subset_attributes = ['alcohol', 'volatile acidity', 'pH', 'quality']

ls = round(df_wines[df_wines['quality_label'] == 'low'][subset_attributes].describe(), 2)
ms = round(df_wines[df_wines['quality_label'] == 'medium'][subset_attributes].describe(), 2)
hs = round(df_wines[df_wines['quality_label'] == 'high'][subset_attributes].describe(), 2)

print('\nSecond view...')
print(pd.concat([ls, ms, hs], axis=1, 
                 keys=['👎 Low Quality Wine', 
                       '👌 Medium Quality Wine', 
                       '👍 High Quality Wine']))

print("****************************************************")
topic = "9.1 3D: Visualizing Data in Three Dimensions"; print("** %s\n" % topic)

#For the following plot, we'll use color (i.e. hue) as the third dimension to represent wine_type.
# Attributes of interest
cols = ['density', 
        'residual sugar', 
        'total sulfur dioxide', 
        'fixed acidity', 
        'wine_type']

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

pp = sns.pairplot(data=df_wines[cols], hue='wine_type', # <== 😀 Look here!
                  height=1.4, aspect=1.2, 
                  palette={"red": "#FF9999", "white": "#FFE888"},
                  plot_kws=dict(edgecolor="black", linewidth=0.5))
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
fig.suptitle('{}\nWine Attributes Pairwise Plots'.format(topic), fontsize=10)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show()

plt.style.use('default')

print("****************************************************")
topic = "9.2 3D: Three Continuous Numeric Attributes"; print("** %s\n" % topic)

#The traditional way — using matplotlib
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')

xs = df_wines['residual sugar']
ys = df_wines['fixed acidity']
zs = df_wines['alcohol']
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('Residual Sugar')
ax.set_ylabel('Fixed Acidity')
ax.set_zlabel('Alcohol')

fig.suptitle('{}\nUsing matplotlib'.format(topic), fontsize=10)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show()


#The better alternative — using Seaborn + toggle the size via the s parameter
fig = plt.figure(figsize=(6, 4))
plt.scatter(x = df_wines['fixed acidity'], y = df_wines['alcohol'], s = df_wines['residual sugar']*25, # <== 😀 Look here!
            alpha=0.4, edgecolors='w')
plt.xlabel('Fixed Acidity')
plt.ylabel('Alcohol')
plt.title('Wine Alcohol Content - Fixed Acidity - Residual Sugar')
fig.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()

print("****************************************************")
topic = "9.3 3D: Three Discrete Categorical Attributes"; print("** %s\n" % topic)

#Using factorplot()
#The attribute quality is represented via the x-axis.
#The attribute wine_type is represented by the color.
#The attribute quality_label is split into 3 columns — low, medium, and high.
fc = sns.catplot(data=df_wines, x="quality", hue="wine_type", col="quality_label", # <== 😀 Look here!
                 height=4, aspect=0.7,
                 kind="count", palette={"red": "#FF9999", "white": "#FFE888"})
plt.suptitle('{}\nUsing factorplot()'.format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()
                                           
print("****************************************************")
topic = "9.4 3D Mixed Attributes (Numeric & Categorical)"; print("** %s\n" % topic)

#Using sns.pairplot():
#- The attribute sulphates is represented via the x-axis.
#- The attribute alcohol is represented via the y-axis.
#- The attribute wine_type is represented by the color.
# Plot pairwise relationships in a dataset.
jp = sns.pairplot(data=df_wines, x_vars=["sulphates"], y_vars=["alcohol"], hue="wine_type", # <== 😀 Look here!
                  height=4.5, aspect=1.2,
                  palette={"red": "#FF9999", "white": "#FFE888"},
                  plot_kws=dict(edgecolor="k", linewidth=0.5))
plt.suptitle('{}\nUsing pairplot()'.format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()


#Using sns.lmplot() to fit linear regression models to the scatter plots:
# Plot data and regression model fits across a FacetGrid.
lp = sns.lmplot(data=df_wines, x='sulphates', y='alcohol', hue='wine_type', # <== 😀 Look here!
                height=5, aspect=1.5,
                palette={"red": "#FF9999", "white": "#FFE888"},
                fit_reg=True, # <== 😀 Look here!
                legend=True,
                scatter_kws=dict(edgecolor="k", linewidth=0.5))
plt.suptitle('{}\nUsing lmplot()'.format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()


#Using sns.kdeplot():
fig = plt.figure(figsize=(6, 4))
ax = sns.kdeplot(df_white_wine['sulphates'], df_white_wine['alcohol'],   # <== 😀 Look here!
                 cmap="YlOrBr", shade=True, shade_lowest=False)
ax = sns.kdeplot(df_red_wine['sulphates'], df_red_wine['alcohol'],   # <== 😀 Look here!
                 cmap="Reds", shade=True, shade_lowest=False)
plt.suptitle('{}\nUsing kdeplot()'.format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show()


#For box plots [📦] we can split them based on wine_type:
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.suptitle('{}\nWine Type - Quality - Acidity'.format(topic), fontsize=14)

# Left Plot #
sns.violinplot(data=df_wines, x="quality", y="volatile acidity",
               inner="quart", linewidth=1.3, ax=ax1)
ax1.set_xlabel("Wine Quality",size=12,alpha=0.8)
ax1.set_ylabel("Wine Volatile Acidity",size=12,alpha=0.8)

# Right Plot #
sns.violinplot(data=df_wines, x="quality", y="volatile acidity", hue="wine_type", # <== 😀 Look here!
               split=True,      # <== 😀 Look here!
               palette={"red": "#FF9999", "white": "white"}, # <== 😀 Look here!
               inner="quart", linewidth=1.3, ax=ax2)
ax2.set_xlabel("Wine Quality",size=12,alpha=0.8)
ax2.set_ylabel("Wine Volatile Acidity",size=12,alpha=0.8)
plt.legend(loc='upper right', title='Wine Type')

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()



#For violin plots [🎻], we can split them based on wine_type:
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
f.suptitle('{}\nWine Type - Quality - Alcohol Content'.format(topic), fontsize=14)

# Left Plot #
sns.boxplot(data=df_wines, x="quality", y="alcohol", hue="wine_type", # <== 😀 Look here!
            palette={"red": "#FF9999", "white": "white"}, # <== 😀 Look here!
            ax=ax1)
ax1.set_xlabel("Wine Quality",size=12,alpha=0.8)
ax1.set_ylabel("Wine Alcohol %",size=12,alpha=0.8)

# Right Plot #
sns.boxplot(data=df_wines, x="quality_label", y="alcohol", hue="wine_type", # <== 😀 Look here!
            palette={"red": "#FF9999", "white": "white"}, # <== 😀 Look here!
            ax=ax2)
ax2.set_xlabel("Wine Quality Class",size=12,alpha=0.8)
ax2.set_ylabel("Wine Alcohol %",size=12,alpha=0.8)
plt.legend(loc='best', title='Wine Type')

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "9.5 4D: Visualizing Data in Four Dimensions"; print("** %s\n" % topic)

#Factors: X-axis, Y-axis, Size, Color
size = df_wines['residual sugar']*25
fill_colors = ['#FF9999' if wt=='red' else '#FFE888' for wt in list(df_wines['wine_type'])]
edge_colors = ['red' if wt=='red' else 'orange' for wt in list(df_wines['wine_type'])]

plt.figure()
plt.scatter(df_wines['fixed acidity'], # <== 😀 1st DIMENSION
            df_wines['alcohol'],       # <== 😀 2nd DIMENSION
            s=size,                 # <== 😀 3rd DIMENSION
            color=fill_colors,      # <== 😀 4th DIMENSION             
            edgecolors=edge_colors,
            alpha=0.4)
plt.xlabel('Fixed Acidity')
plt.ylabel('Alcohol')
plt.title('Wine Alcohol Content - Fixed Acidity - Residual Sugar - Type',y=1.05)
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()


#Factors: X-axis, Y-axis, Color, n-Columns
g = sns.FacetGrid(df_wines, 
                  col="wine_type",            # 😀 TWO COLUMNS coz there're TWO "wine types"
                  col_order=['red', 'white'], # -> Specify the labels
                  hue='quality_label',        # ADD COLOR
                  hue_order=['low', 'medium', 'high'],
                  aspect=1.2, height=3.5, 
                  palette=sns.light_palette('navy', 4)[1:])
g.map(plt.scatter,  "volatile acidity", "alcohol",          # <== y-axis
      alpha=0.9, edgecolor='white', linewidth=0.5, s=100)

fig = g.fig 
fig.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('Wine Type - Alcohol - Quality - Acidity', fontsize=14)
g.add_legend(title='Wine Quality Class')
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()



#Factors (same as the plot before): X-axis, Y-axis, Color, n-Columns
g = sns.FacetGrid(df_wines, 
                  col="wine_type",            # 😀 TWO COLUMNS coz there're TWO "wine types"
                  col_order=['red', 'white'], # -> Specify the labels
                  hue='quality_label',        # ADD COLOR
                  hue_order=['low', 'medium', 'high'],
                  aspect=1.2, height=3.5, 
                  palette=sns.light_palette('green', 4)[1:])
g.map(plt.scatter, 
      "volatile acidity",     # <== x-axis
      "total sulfur dioxide", # <== y-axis
      alpha=0.9, 
      edgecolor='white', linewidth=0.5, s=100)

fig = g.fig 
fig.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('{}\nWine Type - Sulfur Dioxide - Acidity - Quality'.format(topic), fontsize=14)
g.add_legend(title='Wine Quality Class')
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "9.6 5D: Visualizing Data in Five Dimensions"; print("** %s\n" % topic)

#Factors: X-axis, Y-axis, Color, n-Columns, Size
g = sns.FacetGrid(df_wines, 
                  col="wine_type",            # TWO COLUMNS coz there're TWO "wine types"
                  col_order=['red', 'white'], # -> Specify the labels
                  hue='quality_label',        # ADD COLOR
                  hue_order=['low', 'medium', 'high'],
                  aspect=1.2, height=3.5)
g.map(plt.scatter, 
      "residual sugar", # <== x-axis
      "alcohol",        # <== y-axis
      alpha=0.5, 
      edgecolor='white', 
      linewidth=0.5, 
      s=df_wines['total sulfur dioxide']*1.1) # <== 😀 Adjust the size

fig = g.fig 
fig.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('{}\nWine Type - Sulfur Dioxide - Residual Sugar - Alcohol - Quality'.format(topic), fontsize=10)
g.add_legend(title='Wine Quality Class')
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

print("****************************************************")
topic = "9.7 6D: Visualizing Data in Six Dimensions"; print("** %s\n" % topic)

#Factors: X-axis, Y-axis, Color, n-Columns, Size, m-Rows
plt.rc('xtick',labelsize=6)
plt.rc('ytick',labelsize=6)
plt.rcParams["axes.labelsize"] = 6
plt.rcParams["legend.fontsize"] = 6

g = sns.FacetGrid(df_wines, 
                  row='wine_type',     # <== 1) 😀 ROW
                  col="quality",       # <== 2) 😀 COLUMN
                  hue='quality_label', # <== 3) 😀 COLOR
                  #aspect=1.2, height=3.5
                  )
g.map(plt.scatter,  
      "residual sugar", # <== 4) 😀 x-axis
      "alcohol",        # <== 5) 😀 y-axis
      alpha=0.5, edgecolor='k', linewidth=0.5, 
      s=df_wines['total sulfur dioxide']*0.25) # <== 6) 😀 Size

fig = g.fig 
fig.set_size_inches(10, 4)
fig.subplots_adjust(top=0.8, wspace=0.3, hspace=0.3)
fig.suptitle('{}\nWine Type - Sulfur Dioxide - Residual Sugar - Alcohol - Quality Class - Quality Rating'.format(topic), fontsize=8)
g.add_legend(title='Wine Quality Class')
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

plt.style.use('default')

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
#from pandas.plotting                 import parallel_coordinates              #For Parallel Coordinates
#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
#from string                          import Template                          #For working with string, regular expressions


#from bokeh.io                        import curdoc                            #For interacting visualizations
#from bokeh.io                        import output_file                       #For interacting visualizations
#from bokeh.io                        import show                              #For interacting visualizations
#from bokeh.plotting                  import ColumnDataSource                  #For interacting visualizations
#from bokeh.plotting                  import figure                            #For interacting visualizations
#from bokeh.layouts                   import column                            #For interacting visualizations
#from bokeh.layouts                   import gridplot                          #For interacting visualizations
#from bokeh.layouts                   import row                               #For interacting visualizations
#from bokeh.layouts                   import widgetbox                         #For interacting visualizations
#from bokeh.models                    import Button                            #For interacting visualizations
#from bokeh.models                    import CategoricalColorMapper            #For interacting visualizations
#from bokeh.models                    import CheckboxGroup                     #For interacting visualizations
#from bokeh.models                    import ColumnDataSource                  #For interacting visualizations
#from bokeh.models                    import HoverTool                         #For interacting visualizations
#from bokeh.models                    import RadioGroup                        #For interacting visualizations
#from bokeh.models                    import Select                            #For interacting visualizations
#from bokeh.models                    import Slider                            #For interacting visualizations
#from bokeh.models                    import Toggle                            #For interacting visualizations
#from bokeh.models.widgets            import Panel                             #For interacting visualizations
#from bokeh.models.widgets            import Tabs                              #For interacting visualizations
#from bokeh.palettes                  import Spectral6                         #For interacting visualizations


#import keras                                                                  #For DeapLearning
#import keras.backend as k                                                     #For DeapLearning
#from keras.applications.resnet50     import decode_predictions                #For DeapLearning
#from keras.applications.resnet50     import preprocess_input                  #For DeapLearning
#from keras.applications.resnet50     import ResNet50                          #For DeapLearning
#from keras.callbacks                 import EarlyStopping                     #For DeapLearning
#from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
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
#from scipy.cluster.hierarchy         import dendrogram                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import fcluster                          #For learning machine - unsurpervised
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
#from sklearn.feature_selection       import chi2                              #For learning machine
#from sklearn.feature_selection       import SelectKBest                       #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                   #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                 #For learning machine
#from sklearn.feature_extraction.text import TfidfVectorizer                   #For learning machine - unsurpervised
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
#from sklearn.model_selection         import GridSearchCV                      #For learning machine
#from sklearn.model_selection         import KFold                             #For learning machine
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
#plt.rc('xtick',labelsize=8)
#plt.rc('ytick',labelsize=6)
#plt.rcParams['figure.max_open_warning'] = 60                                  #params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 5), 'axes.labelsize': 'x-large', 'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
#plt.rcParams["legend.fontsize"] = 8
#plt.style.use('dark_background')
#plt.style.use('default')
#plt.xticks(fontsize=7); plt.yticks(fontsize=8);

#from matplotlib.axes._axes import _log as matplotlib_axes_logger              #To avoid warnings
#matplotlib_axes_logger.setLevel('ERROR')
#matplotlib_axes_logger.setLevel(0)                                            #To restore default

#ax.tick_params(labelsize=6)                                                   #axis : {'x', 'y', 'both'}
#ax.tick_params(axis='x', rotation=45)                                         #Set rotation atributte

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

#warnings.filterwarnings('ignore', 'Objective did not converge*')              #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
#warnings.filterwarnings('default', 'Objective did not converge*')             #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


#Create categorical type data to use
#cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  ordered=True)
# Change the data type of 'rating' to category
#weather['rating'] = weather.rating.astype(cats)