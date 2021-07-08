# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:53:09 2019

@author: jacqueline.cortez
Chapter 3: Additional Plot Types
    Overview of more complex plot types included in Seaborn.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot as plt                                               #For creating charts
import numpy             as np                                                #For making operations in lists
import pandas            as pd                                                #For loading tabular data
import seaborn           as sns                                               #For visualizing data

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "Medicare_Provider_Charge_Inpatient_DRG100_FY2011.csv"
df_medicare = pd.read_csv(filename)
df_medicare = df_medicare[(df_medicare['DRG Definition'].isin(['682 - RENAL FAILURE W MCC', '683 - RENAL FAILURE W CC', '684 - RENAL FAILURE W/O CC/MCC'])) &
                          (df_medicare['Provider State']=='CA')].copy()
df_medicare['Type Discharges'] = 'Low'
df_medicare.loc[df_medicare['Total Discharges'] > 100, 'Type Discharges'] = 'High'
print("Columns of {}:\n{}".format(filename, df_medicare.columns))

filename = "2010_US_School_Grants.csv"
df_grants = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_grants.columns))

filename = "Washington_Bike_Share.csv"
df_bike = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_bike.columns))

filename = "2018_College_Scorecard_Tuition.csv"
df_college = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_college.columns))

filename = 'Daily_Show_Guest.csv'
df_show = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_show.columns))

print("****************************************************")
topic = "1. Categorical Plot Types"; print("** %s\n" % topic)

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
#To solve the msg:
#'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping 
#will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row 
#if you really want to specify the same RGB or RGBA value for all points.

plt.rc('xtick',labelsize=7)
plt.rc('ytick',labelsize=7)

#The use of jitter in stripplot()
fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
sns.stripplot(data=df_medicare, x='Average Covered Charges', y='DRG Definition', jitter=False, ax=ax0)
ax0.set(title = 'Without jitter')
sns.stripplot(data=df_medicare, x='Average Covered Charges', y='DRG Definition', jitter=True, ax=ax1)
ax1.set(title = 'With jitter')
plt.suptitle('{}\nThe use of jitter in stripplot()'.format(topic))
plt.subplots_adjust(left=0.25, bottom=0.2, right=None, top=0.8, wspace=1.2, hspace=None)
plt.show() # Show the plot

#First group of categorical plots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
sns.stripplot(data=df_medicare, x='Average Covered Charges', y='DRG Definition', jitter=True, ax=ax[0])
ax[0].set(title = 'stripplot()')
sns.swarmplot(data=df_medicare, x='Average Covered Charges', y='DRG Definition', ax=ax[1])
ax[1].set(title = 'swarmplot()')
plt.suptitle('{}\nFirst group - Observations plots'.format(topic))
plt.subplots_adjust(left=0.25, bottom=0.2, right=None, top=0.8, wspace=1.2, hspace=None)
plt.show() # Show the plot

#Second group of categorical plots
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11,4))
sns.boxplot(data=df_medicare, x='Average Covered Charges', y='DRG Definition', ax=ax[0])
ax[0].set(title = 'boxplot()')
sns.violinplot(data=df_medicare, x='Average Covered Charges', y='DRG Definition', ax=ax[1])
ax[1].set(title = 'violinplot()')
sns.boxenplot(data=df_medicare, x='Average Covered Charges', y='DRG Definition', ax=ax[2])
ax[2].set(title = 'lvplot() or boxenplot()')
plt.suptitle('{}\nSecond group - Abstract representation'.format(topic))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.97, top=0.8, wspace=1.5, hspace=None)
plt.show() # Show the plot


#Third group of categorical plots
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11,4))
sns.barplot(data=df_medicare, x='Average Covered Charges', y='DRG Definition', hue='Type Discharges', ax=ax[0])
ax[0].set(title = 'barplot()')
ax[0].legend(loc='best')
sns.pointplot(data=df_medicare, x='Average Covered Charges', y='DRG Definition', hue='Type Discharges', ax=ax[1], alpha=0.5)
ax[1].set(title = 'pointplot()')
ax[1].legend(loc='best')
sns.countplot(data=df_medicare, y='DRG Definition', hue='Type Discharges', ax=ax[2])
ax[2].set(title = 'countplot()')
ax[2].legend(loc='lower right')
plt.suptitle('{}\nThird group - Statistical Estimates'.format(topic))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.99, top=0.8, wspace=1.4, hspace=None)
plt.show() # Show the plot

plt.style.use('default')

matplotlib_axes_logger.setLevel(0) #Se reestablecen las opciones.

print("****************************************************")
topic = "2. stripplot() and swarmplot()"; print("** %s\n" % topic)

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
sns.stripplot(data=df_grants, x='Award_Amount', y='Model Selected', jitter=True, ax=ax0) # Create the stripplot
ax0.set(title = 'stripplot()')
sns.swarmplot(data=df_grants, x='Award_Amount', y='Model Selected', hue='Region', ax=ax1) # Create and display a swarmplot with hue set to the Region
ax1.set(title = 'swarmplot()')
ax1.legend(loc='lower right', fontsize=7)
plt.suptitle(topic)
plt.subplots_adjust(left=0.2, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show() # Show the plot

plt.style.use('default')

print("****************************************************")
topic = "3. boxplots, violinplots and lvplots"; print("** %s\n" % topic)

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
#To solve the msg:
#'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping 
#will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row 
#if you really want to specify the same RGB or RGBA value for all points.

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams["axes.labelsize"] = 8

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
sns.boxplot(data=df_grants, x='Award_Amount', y='Model Selected', ax=ax0) # Create a boxplot
ax0.set(title = 'boxplot()')

sns.violinplot(data=df_grants, x='Award_Amount', y='Model Selected', palette='husl', ax=ax1) # Create a violinplot with the husl palette
ax1.set(title = 'violinplot()')

#UserWarning: The `lvplot` function has been renamed to `boxenplot`. The original name will be removed in a future release. Please update your code.
sns.boxenplot(data=df_grants, x='Award_Amount', y='Model Selected', palette='Paired', hue='Region', ax=ax2) # Create a lvplot with the Paired palette and the Region column as the hue
ax2.set(title = 'lvplot() / boxenplot()') 
ax2.legend(loc='best', fontsize=6)

plt.suptitle(topic)
plt.subplots_adjust(left=0.2, bottom=0.2, right=None, top=0.8, wspace=1.2, hspace=None)
plt.show() # Show the plot

plt.style.use('default')

matplotlib_axes_logger.setLevel(0) #Se reestablecen las opciones.

print("****************************************************")
topic = "4. barplots, pointplots and countplots"; print("** %s\n" % topic)

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams["axes.labelsize"] = 8

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
sns.countplot(data=df_grants, y="Model Selected", hue="Region", ax=ax0) # Show a countplot with the number of models used with each region a different color
ax0.set(title = 'countplot()')
ax0.legend(loc='lower right', fontsize=6)

sns.pointplot(data=df_grants, y='Award_Amount', x='Model Selected', capsize=.1, ax=ax1)# Create a pointplot and include the capsize in order to show bars on the confidence interval
ax1.set(title = 'pointplot()')
ax1.tick_params(axis='x', rotation=45)

sns.barplot(data=df_grants, y='Award_Amount', x='Model Selected', hue='Region', ax=ax2) # Create a barplot with each Region shown as a different color
ax2.set(title = 'barplot()')
ax2.legend(loc='best', fontsize=6)
ax2.tick_params(axis='x', rotation=45)

plt.suptitle(topic)
plt.subplots_adjust(left=0.2, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show() # Show the plot

plt.style.use('default')

print("****************************************************")
topic = "5. Regression Plots"; print("** %s\n" % topic)

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams["axes.labelsize"] = 8
plt.rcParams['axes.titlesize'] = 9

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,5))
sns.regplot(data=df_bike, x='temp', y='total_rentals', marker='+', 
            scatter_kws={'s': 1.5}, line_kws={'lw': 1, 'color': '#000000'}, ax=ax[0,0])
ax[0,0].set(title = 'regplot()')

sns.residplot(data=df_bike, x='temp', y='total_rentals', 
              scatter_kws={'s': 1.5}, ax=ax[0,1])
ax[0,1].set(title = 'residplot()')

ax[0,2].axis('off')

sns.regplot(data=df_bike, x='temp', y='total_rentals', order=2, 
            scatter_kws={'s': 1.5}, line_kws={'lw': 1, 'color': '#000000'}, ax=ax[1,0])
ax[1,0].set(title = 'Polynomial regression')

sns.residplot(data=df_bike, x='temp', y='total_rentals', order=2, 
              scatter_kws={'s': 1.5}, ax=ax[1,1])
ax[1,1].set(title = 'Polynomial Residplot')

ax[1,2].axis('off')

sns.regplot(data=df_bike, x='mnth', y='total_rentals', x_jitter=0.1, order=2,
            scatter_kws={'s': 1.5}, line_kws={'lw': 1, 'color': '#000000'}, ax=ax[2,0])
ax[2,0].set(title = 'Categorical Values')

sns.regplot(data=df_bike, x='mnth', y='total_rentals', x_estimator=np.mean, order=2,
            scatter_kws={'s': 1.5}, line_kws={'lw': 1, 'color': '#000000'}, ax=ax[2,1])
ax[2,1].set(title = 'Estimators')

sns.regplot(data=df_bike, x='temp', y='total_rentals', x_bins=4,
            scatter_kws={'s': 1.5}, line_kws={'lw': 1, 'color': '#000000'}, ax=ax[2,2])
ax[2,2].set(title = 'Binning the data',
            xlim  = (0, 1),
            ylim  = (0, 8000))

plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.9, wspace=1, hspace=1)
plt.show() # Show the plot

plt.style.use('default')

print("****************************************************")
topic = "6. Regression and residual plots"; print("** %s\n" % topic)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
# Display a regression plot for Tuition
sns.regplot(data=df_college, y='Tuition', x="SAT_AVG_ALL", marker='^', color='g',
            scatter_kws={'s': 1.5}, line_kws={'color': '#000000'}, ax=ax0)
ax0.set(title = 'regplot()')

# Display the residual plot
sns.residplot(data=df_college, y='Tuition', x="SAT_AVG_ALL", color='g',
              scatter_kws={'s': 1.5}, ax=ax1)
ax1.set(title = 'residplot()')

plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.5, hspace=None)
plt.show() # Show the plot

plt.style.use('default')

print("****************************************************")
topic = "7. Regression plot parameters"; print("** %s\n" % topic)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
# Plot a regression plot of Tuition and the Percentage of Pell Grants
sns.regplot(data=df_college, y='Tuition', x="PCTPELL",
            scatter_kws={'s': 1.5}, line_kws={'color': '#000000'}, ax=ax0)
ax0.set(title = 'A simple regression plot')

# Create another plot that estimates the tuition by PCTPELL
sns.regplot(data=df_college, y='Tuition', x="PCTPELL", x_bins=5,
            scatter_kws={'s': 1.5}, line_kws={'color': '#000000'}, ax=ax1)
ax1.set(title = 'Binning the data')

# The final plot should include a line using a 2nd order polynomial
sns.regplot(data=df_college, y='Tuition', x="PCTPELL", x_bins=5, order=2,
            scatter_kws={'s': 1.5}, line_kws={'color': '#000000'}, ax=ax2)
ax2.set(title = 'A Polynomial Regression Plot')

plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show() # Show the plot

plt.style.use('default')

print("****************************************************")
topic = "8. Binning data"; print("** %s\n" % topic)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
# Create a scatter plot by disabling the regression line
sns.regplot(data=df_college, y='Tuition', x="UG", fit_reg=False,
            scatter_kws={'s': 1.5}, ax=ax0)
ax0.set(title = 'Disabling the regression line')

# Create a scatter plot and bin the data into 5 bins
sns.regplot(data=df_college, y='Tuition', x="UG", x_bins=5,
            scatter_kws={'s': 1.5}, ax=ax1)
ax1.set(title = 'Binning the data (5 groups)')

# Create a regplot and bin the data into 8 bins
sns.regplot(data=df_college, y='Tuition', x="UG", x_bins=8,
            scatter_kws={'s': 1.5}, ax=ax2)
ax2.set(title = 'Binning the data (8 groups)')

plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show() # Show the plot

plt.style.use('default')

print("****************************************************")
topic = "9. Matrix plots"; print("** %s\n" % topic)

df_crosstab = pd.crosstab(df_bike.mnth, df_bike.weekday, values=df_bike.total_rentals, aggfunc='mean').round(0).applymap(np.int64)
df_crosstab.columns = ['Domingo', 'Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado']
df_crosstab.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=6)
plt.rcParams["axes.labelsize"] = 10
plt.rcParams['axes.titlesize'] = 10

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,5))
sns.heatmap(df_crosstab, ax=ax[0,0])
ax[0,0].set(title = 'A simple heatmap')
ax[0,0].tick_params(axis='x', rotation=90)

sns.heatmap(df_crosstab, annot=True, fmt='d', cmap='YlGnBu', cbar=False, linewidths=.5,
            annot_kws={"size":5}, ax=ax[0,1])
ax[0,1].set(title = 'Customizing a heatmap')
ax[0,1].tick_params(axis='x', rotation=90)

sns.heatmap(df_crosstab, annot=True, fmt='d', cmap='YlGnBu', cbar=True, center=df_crosstab.iloc[9,6],
            annot_kws={"size":5}, ax=ax[1,0])
ax[1,0].set(title = 'Centering a heatmap')
ax[1,0].tick_params(axis='x', rotation=90)

sns.heatmap(df_bike.corr(), ax=ax[1,1])
ax[1,1].set(title = 'Correlation Matrix')
ax[1,1].tick_params(axis='x', rotation=90)

plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.9, wspace=1.2, hspace=1.2)
plt.show() # Show the plot

plt.style.use('default')

print("****************************************************")
topic = "10. Creating heatmaps"; print("** %s\n" % topic)

# Create a crosstab table of the data
pd_crosstab = pd.crosstab(df_show["Group"], df_show["YEAR"])
print(pd_crosstab)

plt.figure()
sns.heatmap(pd_crosstab) # Plot a heatmap of the table
plt.yticks(rotation=0) # Rotate tick marks for visibility
plt.xticks(rotation=90)
plt.title('Daily_Show_Guest')
plt.suptitle(topic)
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.97, top=0.9, wspace=None, hspace=None)
plt.show()

print("****************************************************")
topic = "11. Customizing heatmaps"; print("** %s\n" % topic)

plt.figure()
sns.heatmap(pd_crosstab, cbar=False, cmap="BuGn", linewidths=0.3) # Plot a heatmap of the table with no color bar and using the BuGn palette
plt.yticks(rotation=0) # Rotate tick marks for visibility
plt.xticks(rotation=90)
plt.title('Daily_Show_Guest')
plt.suptitle(topic)
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.97, top=0.9, wspace=None, hspace=None)
plt.show()

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
