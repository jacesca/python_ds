# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:26:27 2019

@author: jacqueline.cortez
source:
    https://jovianlin.io/data-visualization-seaborn-part-2/
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
topic = "8.1 2D: Heatmap on Correlation Matrix"; print("** %s\n" % topic)

# Compute pairwise correlation of Dataframe's attributes
corr = df_wines.corr()
print(corr)

fig, (ax) = plt.subplots(1, 1, figsize=(8,5))
hm = sns.heatmap(corr, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 annot_kws={"size": 6},
                 linewidths=.05)
fig.subplots_adjust(top=0.93)
plt.title('Wine Attributes Correlation Heatmap', fontsize=10, fontweight='bold')
plt.suptitle(topic)
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=1.2)
plt.show() # Show the plot


print("****************************************************")
topic = "8.2 2D: Pair-Wise Scatter Plots"; print("** %s\n" % topic)

plt.rc('xtick',labelsize=6)
plt.rc('ytick',labelsize=6)
plt.rcParams["axes.labelsize"] = 6

# Attributes of interest
cols = ['density', 
        'residual sugar', 
        'total sulfur dioxide', 
        'free sulfur dioxide', 
        'fixed acidity']

pp = sns.pairplot(df_wines[cols], height=1.15, aspect=1.2,
                  plot_kws=dict(edgecolor="navy", linewidth=0.5, s=25, alpha=0.5),
                  diag_kws=dict(shade=True), # "diag" adjusts/tunes the diagonal plots
                  diag_kind="kde") # use "kde" for diagonal plots
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
fig.suptitle('{}\nWine Attributes Pairwise Plots'.format(topic), 
              fontsize=8, fontweight='bold')
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=1.2)
plt.show() # Show the plot

pp = sns.pairplot(df_wines[cols], height=1.15, aspect=1.2,
                  plot_kws=dict(scatter_kws=dict(s=25, alpha=0.5)),
                  #diag_kws=dict(shade=True), # "diag" adjusts/tunes the diagonal plots
                  #diag_kind="kde") # use "kde" for diagonal plots
                  kind="reg") # <== 😀 linear regression to the scatter plots
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
fig.suptitle('{}\nWine Attributes Pairwise Plots'.format(topic), 
              fontsize=8, fontweight='bold')
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=1.2)
plt.show() # Show the plot

plt.style.use('default')

print("****************************************************")
topic = "8.3 2D: Parallel Coordinates"; print("** %s\n" % topic)

# Attributes of interest
cols = ['density', 
        'residual sugar', 
        'total sulfur dioxide', 
        'free sulfur dioxide', 
        'fixed acidity']

subset_df = df_wines[cols]

ss = StandardScaler()
scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)
final_df = pd.concat([scaled_df, df_wines['wine_type']], axis=1)
print(final_df.head())

fig = plt.figure(figsize=(10, 5))
plt.title("Parallel Coordinates (Scaled features)", fontsize=18)
title = fig.suptitle(topic)
fig.subplots_adjust(top=0.8, wspace=0)

pc = parallel_coordinates(final_df, 'wine_type', 
                          color=('skyblue', 'firebrick'))
plt.show() # Show the plot


# If you don't perform scaling beforehand, this is what you'll get:
fig = plt.figure(figsize=(10, 5))
plt.title("Parallel Coordinates (Without scaling)", fontsize=18)
title = fig.suptitle(topic)
fig.subplots_adjust(top=0.8, wspace=0)

new_cols = ['density', 'residual sugar', 'total sulfur dioxide', 'free sulfur dioxide', 'fixed acidity', 'wine_type']
pc = parallel_coordinates(df_wines[new_cols], 'wine_type', 
                          color=('skyblue', 'firebrick'))
plt.show() # Show the plot

print("****************************************************")
topic = "8.4 2D: Two Continuous Numeric Attributes"; print("** %s\n" % topic)

#The traditional way — using matplotlib:
plt.figure()
plt.scatter(df_wines['sulphates'], 
            df_wines['alcohol'],
            alpha=0.4, edgecolors='w')

plt.xlabel('Sulphates')
plt.ylabel('Alcohol')
plt.title('Wine Sulphates - Alcohol Content (Traditional way with matplotlib)', y=1.05)
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot


#The better alternative — using Seaborn's jointplot():
jp = sns.jointplot(data=df_wines, x='sulphates', y='alcohol', 
                   scatter_kws=dict(s=25, alpha=0.5),
                   kind='reg', # <== 😀 Add regression and kernel density fits
                   space=0, height=5, ratio=4)
plt.suptitle('{}\nWine Sulphates - Alcohol Content (Using Seaborn)'.format(topic), fontsize=10)
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot


#Replace the scatterplot with a joint histogram using hexagonal bins
jp = sns.jointplot(data=df_wines, x='sulphates',  y='alcohol', 
                   kind='hex', # <== 😀 Replace the scatterplot with a joint histogram using hexagonal bins
                   space=0, height=5, ratio=4)
plt.suptitle('{}\nWine Sulphates - Alcohol Content (Jointplot with histogram)'.format(topic), fontsize=10)
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot


#With KDE
jp = sns.jointplot(data=df_wines, x='sulphates', y='alcohol', 
                   kind='kde', # <== 😀 KDE
                   space=0, height=5, ratio=4)
plt.suptitle('{}\nWine Sulphates - Alcohol Content (Jointplot with KDE)'.format(topic), fontsize=10)
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "8.5 2D: Two Discrete Categorical Attributes"; print("** %s\n" % topic)

#The traditional way — using matplotlib:
fig = plt.figure(figsize=(10,4))
title = fig.suptitle("{}\nWine Type - Quality (With matplotlib)".format(topic), fontsize=14)
fig.subplots_adjust(top=0.8, wspace=0.3)

ax1 = fig.add_subplot(1,2,1)
ax1.set_title("Red Wine")
ax1.set_xlabel("Quality")
ax1.set_ylabel("Frequency") 
rw_q = df_red_wine['quality'].value_counts()
rw_q = (list(rw_q.index), list(rw_q.values))
ax1.set_ylim([0,2500])
ax1.tick_params(axis='both', which='major', labelsize=8.5)
bar1 = ax1.bar(rw_q[0], rw_q[1], 
               color='red', edgecolor='black', linewidth=1)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title("White Wine")
ax2.set_xlabel("Quality")
ax2.set_ylabel("Frequency") 
ww_q = df_white_wine['quality'].value_counts()
ww_q = (list(ww_q.index), list(ww_q.values))
ax2.set_ylim([0,2500])
ax2.tick_params(axis='both', which='major', labelsize=8.5)
bar2 = ax2.bar(ww_q[0], ww_q[1], 
               color='white', edgecolor='black', linewidth=1)

#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot


#The better alternative — using Seaborn's countplot()
fig = plt.figure(figsize=(10, 4))
title = fig.suptitle("{}\nWine Type - Quality (With Seaborn)".format(topic), fontsize=14)
cp = sns.countplot(data=df_wines, x="quality", hue="wine_type", 
                   palette={"red": "#FF9999", "white": "#FFE888"})
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot
                           
print("****************************************************")
topic = "8.6 2D: Mixed Attributes"; print("** %s\n" % topic)

#let's first look at the traditional way — using matplotlib (histograms)
fig = plt.figure(figsize=(10,4))
title = fig.suptitle("{}\nSulphates Content in Wine".format(topic), fontsize=14)
fig.subplots_adjust(top=0.80, wspace=0.3)

ax1 = fig.add_subplot(1,2,1)
ax1.set_title("Red Wine")
ax1.set_xlabel("Sulphates")
ax1.set_ylabel("Frequency") 
ax1.set_ylim([0, 1200])
ax1.text(1.2, 800, r'$\mu$='+str(round(df_red_wine['sulphates'].mean(),2)), 
         fontsize=12)
r_freq, r_bins, r_patches = ax1.hist(df_red_wine['sulphates'], color='red', bins=15,
                                     edgecolor='black', linewidth=1)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title("White Wine")
ax2.set_xlabel("Sulphates")
ax2.set_ylabel("Frequency")
ax2.set_ylim([0, 1200])
ax2.text(0.8, 800, r'$\mu$='+str(round(df_white_wine['sulphates'].mean(),2)), 
         fontsize=12)
w_freq, w_bins, w_patches = ax2.hist(df_white_wine['sulphates'], color='white', bins=15,
                                     edgecolor='black', linewidth=1)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot


#Using matplotlib (density plots)
fig = plt.figure(figsize=(10,4))
title = fig.suptitle("{}\nSulphates Content in Wine (Density matplotlib plots)".format(topic), fontsize=14)
fig.subplots_adjust(top=0.8, wspace=0.3)

ax1 = fig.add_subplot(1,2,1)
ax1.set_title("Red Wine")
ax1.set_xlabel("Sulphates")
ax1.set_ylabel("Density") 
sns.kdeplot(df_red_wine['sulphates'], ax=ax1, shade=True, color='r')

ax2 = fig.add_subplot(1,2,2)
ax2.set_title("White Wine")
ax2.set_xlabel("Sulphates")
ax2.set_ylabel("Density") 
sns.kdeplot(df_white_wine['sulphates'], ax=ax2, shade=True, color='y')

#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot

#The better alternative — using Seaborn's FacetGrid()
fig = plt.figure(figsize=(10,4))
title = fig.suptitle("{}\nSulphates Content in Wine (Using Facetgrid)".format(topic), fontsize=14)
fig.subplots_adjust(top=0.8, wspace=0.3)

ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Sulphates")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(data=df_wines, hue='wine_type', 
                  palette={"red": "r", "white": "y"})
g.map(sns.distplot, 'sulphates', 
      kde=True, bins=15, ax=ax)
ax.legend(title='Wine Type')
plt.close()

#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
#plt.show() # Show the plot

print("****************************************************")
topic = "8.7 2D: Box and Violin Plots"; print("** %s\n" % topic)

#[📦] Box plots are another way of effectively depicting groups of numeric data based on the different values in the categorical attribute.
f, (ax) = plt.subplots(1, 1, figsize=(10, 4))
f.suptitle('{}\nWine Quality - Alcohol Content'.format(topic), fontsize=14)

sns.boxplot(data=df_wines, x="quality", y="alcohol", ax=ax)
ax.set_xlabel("Wine Quality",size=12,alpha=0.8)
ax.set_ylabel("Wine Alcohol %",size=12,alpha=0.8)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot


#[🎻] Another similar visualization is violin plots, which is also an effective way to visualize grouped numeric data using kernel density plots — depicting the probability density of the data at different values.
f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('{}\nWine Quality - Sulphates Content'.format(topic), fontsize=14)

sns.violinplot(data=df_wines, x="quality", y="sulphates", ax=ax)
ax.set_xlabel("Wine Quality",size=12,alpha=0.8)
ax.set_ylabel("Wine Sulphates",size=12,alpha=0.8)
#plt.subplots_adjust(left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot

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