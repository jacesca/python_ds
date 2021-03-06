# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:18:02 2019

@author: jacqueline.cortez

Capítulo 3. Showing uncertainty
Introduction:
    Uncertainty occurs everywhere in data science, but it's frequently left out of visualizations where it should 
    be included. Here, we review what a confidence interval is and how to visualize them for both single estimates 
    and continuous functions. Additionally, we discuss the bootstrap resampling technique for assessing uncertainty 
    and how to visualize it properly.
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
#import wikipedia

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import date                                                           #For obteining today function
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions
#from itertools import cycle                                                         #Used in the function plot_labeled_decision_regions()
#from math import floor                                                              #Used in the function plot_labeled_decision_regions()
#from math import ceil                                                               #Used in the function plot_labeled_decision_regions()
#from itertools import combinations                                                  #For iterations
#from collections import defaultdict                                                 #Returns a new dictionary-like object

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

#from sklearn.ensemble import AdaBoostClassifier                                     #For learning machine - surpervised
#from sklearn.ensemble import BaggingClassifier                                      #For learning machine - surpervised
#from sklearn.ensemble import GradientBoostingRegressor                              #For learning machine - surpervised
#from sklearn.ensemble import RandomForestClassifier                                 #For learning machine
#from sklearn.ensemble import RandomForestRegressor                                  #For learning machine - unsurpervised
#from sklearn.ensemble import VotingClassifier                                       #For learning machine - unsurpervised
#from sklearn.feature_extraction.text import TfidfVectorizer                         #For learning machine - unsurpervised
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
#from sklearn.metrics import accuracy_score                                          #For learning machine
#from sklearn.metrics import classification_report                                   #For learning machine
#from sklearn.metrics import confusion_matrix                                        #For learning machine
#from sklearn.metrics import mean_squared_error as MSE                               #For learning machine
#from sklearn.metrics import roc_auc_score                                           #For learning machine
#from sklearn.metrics import roc_curve                                               #For learning machine
#from sklearn.model_selection import cross_val_score                                 #For learning machine
#from sklearn.model_selection import GridSearchCV                                    #For learning machine
#from sklearn.model_selection import RandomizedSearchCV                              #For learning machine
#from sklearn.model_selection import train_test_split                                #For learning machine
#from sklearn.multiclass import OneVsRestClassifier                                  #For learning machine
#from sklearn.neighbors import KNeighborsClassifier as KNN                           #For learning machine
#from sklearn.pipeline import FeatureUnion                                           #For learning machine
#from sklearn.pipeline import make_pipeline                                          #For learning machine - unsurpervised
#from sklearn.pipeline import Pipeline                                               #For learning machine
#from sklearn.preprocessing import FunctionTransformer                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
#from sklearn.preprocessing import MaxAbsScaler                                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing import Normalizer                                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing import normalize                                         #For learning machine - unsurpervised
#from sklearn.preprocessing import scale                                             #For learning machine
#from sklearn.preprocessing import StandardScaler                                    #For learning machine
#from sklearn.svm import SVC                                                         #For learning machine
#from sklearn.tree import DecisionTreeClassifier                                     #For learning machine - supervised
#from sklearn.tree import DecisionTreeRegressor                                      #For learning machine - supervised

import statsmodels.api as sm                                                        #Make a prediction model
#import statsmodels.formula.api as smf                                               #Make a prediction model    

#import keras                                                                        #For DeapLearning
#from keras.callbacks import EarlyStopping                                           #For DeapLearning
#from keras.layers import Dense                                                      #For DeapLearning
#from keras.models import Sequential                                                 #For DeapLearning
#from keras.models import load_model                                                 #For DeapLearning
#from keras.optimizers import SGD                                                    #For DeapLearning
#from keras.utils import to_categorical                                              #For DeapLearning

#import networkx as nx                                                               #For Network Analysis in Python
#import nxviz as nv                                                                  #For Network Analysis in Python
#from nxviz import ArcPlot                                                           #For Network Analysis in Python
#from nxviz import CircosPlot                                                        #For Network Analysis in Python 
#from nxviz import MatrixPlot                                                        #For Network Analysis in Python 

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
print("** User defined variables \n")

SEED=42

#print("****************************************************")
print("** Getting the data for this program\n")

def bootstrap(data, n_boots, function, seed):
    """Perform bootstrapped function on a vector."""
    np.random.seed(seed)
    return [function(np.random.choice(data, len(data))) for _ in range(n_boots)]



#print("****************************************************")
print("** Getting the data for this program\n")

file = "pollution_wide.csv" 
pollution = pd.read_csv(file)
pollution['Date'] = pd.to_datetime((pollution.year*1000+pollution.day).apply(str), format='%Y%j')
pollution['day_of_month'] = pollution['Date'].dt.day



print("****************************************************")
tema = '2. Basic confidence intervals'; print("** %s\n" % tema)

#Des Moines, Iowa Quality air
Des_Moines = pollution[(pollution.city=='Des Moines') & (pollution.Date=='2015-07-04')][['CO','NO2','O3','SO2']]
jul_cities = pollution[(pollution.month==7) & (pollution.year==2015)][['CO','NO2','O3','SO2']]
average_ests = pd.DataFrame({'pollutant': ['CO', 'NO2', 'O3', 'SO2'], 
                             'mean'     : jul_cities.mean().values,
                             'std_err'  : jul_cities.sem().values,
                             'y'        : ['95% Interval','95% Interval','95% Interval','95% Interval'],
                             'seen'     : Des_Moines.mean().values})
average_ests['lower'] = average_ests['mean'] - 1.96*average_ests['std_err'] #95% intervalo de confianza -->Valor de z=1.96
average_ests['upper'] = average_ests['mean'] + 1.96*average_ests['std_err'] #95% intervalo de confianza -->Valor de z=1.96
print(average_ests) # Construct CI bounds for averages


sns.set() # Set default Seaborn style
sns.set(font_scale=0.8)
g = sns.FacetGrid(average_ests, row = 'pollutant', sharex = False) # Setup a grid of plots, with non-shared x axes limits
g.map(plt.hlines, 'y', 'lower', 'upper') # Plot CI for average estimate
g.map(plt.scatter, 'seen', 'y', color = 'orangered').set_ylabels('').set_xlabels('Average') # Plot observed values for comparison and remove axes labels
g.fig.set_size_inches(6, 5)
plt.suptitle(tema)
plt.subplots_adjust(left=0.20, bottom=0.10, right=None, top=0.90, wspace=None, hspace=0.8)
plt.show()
plt.style.use('default')






print("****************************************************")
tema = '3. Annotating confidence intervals'; print("** %s\n" % tema)

#Cincinnati and Indiantapolis Quality Air
Ind_vs_Cin = pollution[(pollution.year>2012) & (pollution.city.isin(['Indianapolis','Cincinnati']))][['Date','city','SO2']].pivot(index='Date', columns='city', values='SO2').dropna().reset_index()#.fillna(0)
Ind_vs_Cin['mean_diff'] = Ind_vs_Cin.Cincinnati - Ind_vs_Cin.Indianapolis
Ind_vs_Cin['year'] = Ind_vs_Cin['Date'].dt.year
diffs_by_year = pd.DataFrame({'year'    : [2013, 2014, 2015],
                              'mean'    : Ind_vs_Cin.groupby('year')['mean_diff'].mean().values.reshape(-1),
                              'std_err' : Ind_vs_Cin.groupby('year')['mean_diff'].sem().values.reshape(-1)})
diffs_by_year['lower'] = diffs_by_year['mean'] - 1.96*diffs_by_year['std_err'] #95% intervalo de confianza -->Valor de z=1.96
diffs_by_year['upper'] = diffs_by_year['mean'] + 1.96*diffs_by_year['std_err'] #95% intervalo de confianza -->Valor de z=1.96
print(diffs_by_year) # Annotating confidence intervals


sns.set() # Set default Seaborn style
plt.figure()
plt.hlines(y = 'year', xmin = 'lower', xmax = 'upper', linewidth = 5, color = 'steelblue', alpha = 0.7, data = diffs_by_year) # Set start and ends according to intervals 
plt.plot('mean', 'year', 'k|', data = diffs_by_year) # Point estimates
plt.axvline(x = 0, color = 'orangered', linestyle = '--') # Add a 'null' reference line at 0 and color orangered
plt.xlabel('95% CI') # Set descriptive axis labels and title
plt.ylabel('Year') # Set descriptive axis labels and title
plt.yticks([2013, 2014, 2015])
plt.title('Avg SO2 differences between Cincinnati and Indianapolis')
plt.suptitle(tema)
plt.subplots_adjust(left=0.20, bottom=0.10, right=None, top=0.88, wspace=None, hspace=0.8)
plt.show()
plt.style.use('default')






print("****************************************************")
tema = '5. Making a confidence band'; print("** %s\n" % tema)

#Vandenberg Air Force Base Quality Air
vandenberg_NO2 = pollution[(pollution.city=='Vandenberg Air Force Base') & (pollution.year==2012)].dropna().set_index('Date')
vandenberg_NO2 = pd.DataFrame({'day'    : vandenberg_NO2.day,
                               'mean'   : vandenberg_NO2.rolling(window=25)['NO2'].mean(),
                               'std_err': vandenberg_NO2.rolling(window=25)['NO2'].std(),
                               'sem'    : vandenberg_NO2.rolling(window=25)['NO2'].std()/np.sqrt(25),
                               'lower'  : vandenberg_NO2.rolling(window=25)['NO2'].mean() - 2.58*vandenberg_NO2.rolling(window=25)['NO2'].std()/np.sqrt(25),
                               'upper'  : vandenberg_NO2.rolling(window=25)['NO2'].mean() + 2.58*vandenberg_NO2.rolling(window=25)['NO2'].std()/np.sqrt(25)})
vandenberg_NO2.dropna(inplace=True)
print(vandenberg_NO2.head()) # Confidence band


sns.set() # Set default Seaborn style
plt.figure()
plt.plot('day', 'mean', data = vandenberg_NO2, color = 'white', alpha = 0.4) # Plot mean estimate as a white semi-transparent line
plt.fill_between(x = 'day', y1 = 'lower', y2 = 'upper', data = vandenberg_NO2) # Fill between the upper and lower confidence band values
plt.xlabel('Time') # Set descriptive axis labels and title
plt.ylabel('NO2 level') # Set descriptive axis labels and title
plt.title('Avg NO2 Quality Air')
plt.suptitle(tema)
plt.subplots_adjust(left=0.20, bottom=0.10, right=None, top=0.88, wspace=None, hspace=0.8)
plt.show()
plt.style.use('default')






print("****************************************************")
tema = '6. Separating a lot of bands'; print("** %s\n" % tema)

#Quality Air in four cities: 'Cincinnati', 'Des Moines', 'Houston', 'Indianapolis'
eastern_SO2    = pollution[pollution.city.isin(['Cincinnati', 'Des Moines', 'Houston', 'Indianapolis'])].set_index('Date')
eastern_SO2    = pd.DataFrame({'city'   : eastern_SO2.city,
                               'day'    : eastern_SO2.day,
                               'mean'   : eastern_SO2.groupby(['city']).rolling(window=25)['SO2'].mean().values,
                               'std_'   : eastern_SO2.groupby(['city']).rolling(window=25)['SO2'].std().values,
                               'sem'    : eastern_SO2.groupby(['city']).rolling(window=25)['SO2'].std().values/np.sqrt(25),
                               'lower'  : eastern_SO2.groupby(['city']).rolling(window=25)['SO2'].mean().values - 1.96*eastern_SO2.groupby(['city']).rolling(window=25)['SO2'].std().values/np.sqrt(25),
                               'upper'  : eastern_SO2.groupby(['city']).rolling(window=25)['SO2'].mean().values + 1.96*eastern_SO2.groupby(['city']).rolling(window=25)['SO2'].std().values/np.sqrt(25)})
#eastern_SO2   = eastern_SO2.drop(eastern_SO2[:'2013-12-31'].index, axis=0)
#eastern_SO2   = eastern_SO2.drop(eastern_SO2['2015-01-01':].index, axis=0)
eastern_SO2    = eastern_SO2['2015'].copy()
print(eastern_SO2.head()) # Confidence band


sns.set() # Set default Seaborn style
#plt.figure()
g = sns.FacetGrid(eastern_SO2, col = 'city', col_wrap = 2) # Setup a grid of plots with columns divided by location
g.map(plt.fill_between, 'day', 'lower', 'upper', color = 'coral') # Map interval plots to each cities data with corol colored ribbons
g.map(plt.plot, 'day', 'mean', color = 'white') # Map overlaid mean plots with white line
plt.suptitle(tema+"\nAvg SO2 Quality Air")
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()






print("****************************************************")
tema = '7. Cleaning up bands for overlaps'; print("** %s\n" % tema)

#Quality Air in two cities: 'Denver', 'Des Moines', 'Houston', 'Indianapolis'
SO2_compare    = pollution[pollution.city.isin(['Denver', 'Long Beach'])].set_index('Date')
SO2_compare    = pd.DataFrame({'city'   : SO2_compare.city,
                               'day'    : SO2_compare.day,
                               'mean'   : SO2_compare.groupby(['city']).rolling(window=25)['SO2'].mean().values,
                               'std_'   : SO2_compare.groupby(['city']).rolling(window=25)['SO2'].std().values,
                               'sem'    : SO2_compare.groupby(['city']).rolling(window=25)['SO2'].std().values/np.sqrt(25),
                               'lower'  : SO2_compare.groupby(['city']).rolling(window=25)['SO2'].mean().values - 1.96*SO2_compare.groupby(['city']).rolling(window=25)['SO2'].std().values/np.sqrt(25),
                               'upper'  : SO2_compare.groupby(['city']).rolling(window=25)['SO2'].mean().values + 1.96*SO2_compare.groupby(['city']).rolling(window=25)['SO2'].std().values/np.sqrt(25)})
SO2_compare    = SO2_compare['2015']
print(SO2_compare.head()) # SO2 levesl in 'Denver', 'Long Beach'


sns.set() # Set default Seaborn style
plt.figure()
for city, color in [('Denver',"#66c2a5"), ('Long Beach', "#fc8d62")]:
    city_data = SO2_compare[SO2_compare.city  ==  city] # Filter data to desired city
    plt.fill_between(x = 'day', y1 = 'lower', y2 = 'upper', data = city_data, color = color, alpha = 0.4) # Set city interval color to desired and lower opacity
    plt.plot('day','mean', data = city_data, label = city, color = color, alpha = 0.25) # Draw a faint mean line for reference and give a label for legend   
plt.legend()
plt.xlabel('Day') # Set descriptive axis labels and title
plt.ylabel('SO2 level') # Set descriptive axis labels and title
plt.title('Avg SO2 Quality Air')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')
plt.show()





print("****************************************************")
tema = '8. Beyond 95%'; print("** %s\n" % tema)

print(diffs_by_year.head()) #Avg SO2 differences between Cincinnati and Indianapolis
print(vandenberg_NO2.head()) #Avg NO2 Quality Air in Vanderberg


sns.set() # Set default Seaborn style


#Interval size setup
sizes   = ['99%', '95%', '90%']
Z_score = [2.58, 1.96, 1.67]
colors  = ['#41b6c4', '#2c7fb8', '#253494']
          
plt.figure()
for size, Z, color in zip(sizes, Z_score, colors):
    plt.hlines(y    = diffs_by_year.year, 
               xmin = diffs_by_year['mean'] - Z*diffs_by_year['std_err'], 
               xmax = diffs_by_year['mean'] + Z*diffs_by_year['std_err'],
               linewidth = 12, color = color, alpha = 0.7, label = size) # Set start and ends according to intervals 
plt.plot('mean', 'year', 'ko', data = diffs_by_year, label = 'Point Estimate') # Point estimates
plt.axvline(x = 0, color = 'orangered', linestyle = '--') # Add a 'null' reference line at 0 and color orangered
plt.xlabel('SO2') # Set descriptive axis labels and title
plt.ylabel('Year') # Set descriptive axis labels and title
plt.yticks([2013, 2014, 2015])
plt.legend(loc='best')
plt.title('Avg SO2 differences between Cincinnati and Indianapolis')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()


#2 bands overlapping
widths  = ['99%', '90%']
Z_score = [2.58, 1.67]
colors  = ['#99d8c9', '#41ae76']

plt.figure()
for percent, Z, color in zip(widths, Z_score, colors):
    plt.fill_between(x  = vandenberg_NO2.day, 
                     y1 = vandenberg_NO2['mean'] - Z*vandenberg_NO2['sem'], 
                     y2 = vandenberg_NO2['mean'] + Z*vandenberg_NO2['sem'], 
                     color = color, alpha = 0.5, label = percent) # Fill between the upper and lower confidence band values
plt.xlabel('Time') # Set descriptive axis labels and title
plt.ylabel('NO2 level') # Set descriptive axis labels and title
plt.legend(loc='best')
plt.title('Avg NO2 Quality Air')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()


#Width lines setup when we don't have access to color
sizes   = ['99% CI', '95%', '90%']
Z_score = [2.58, 1.96, 1.67]
widths  = [5, 15, 25]
          
plt.figure()
for size, Z, width in zip(sizes, Z_score, widths):
    plt.hlines(y    = diffs_by_year.year, 
               xmin = diffs_by_year['mean'] - Z*diffs_by_year['std_err'], 
               xmax = diffs_by_year['mean'] + Z*diffs_by_year['std_err'],
               linewidth = width, color = 'grey', label = size) # Set start and ends according to intervals 
plt.plot('mean', 'year', 'ko', data = diffs_by_year, label = 'Point Estimate') # Point estimates
plt.axvline(x = 0, color = 'black', linestyle = '--') # Add a 'null' reference line at 0 and color orangered
plt.xlabel('SO2') # Set descriptive axis labels and title
plt.ylabel('Year') # Set descriptive axis labels and title
plt.yticks([2013, 2014, 2015])
plt.legend(loc='best', framealpha=0.5, handleheight=1.8) #, labelspacing=1.5, handleheight=2.5 
plt.margins(0.25)
plt.title('Avg SO2 differences between Cincinnati and Indianapolis')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()


plt.style.use('default')





print("****************************************************")
tema = '9. 90, 95, and 99% intervals'; print("** %s\n" % tema)

SO2_Fairbanks  = pollution[(pollution.city=='Fairbanks') & (pollution.year==2015)].set_index('Date').sort_index()
print(SO2_Fairbanks.head()) #Polutants in Fairbanks

#Make a prediction model
#import statsmodels.api as sm
#import statsmodels.formula.api as smf

#pollution_model = smf.ols(formula="SO2 ~ CO + NO2 + O3 + day", data=SO2_Fairbanks.drop(["city", "year", "month", "day_of_month"], axis=1).dropna()).fit()
#print(pollution_model.params)
#print(pollution_model.summary())
X = SO2_Fairbanks[["day", "CO", "NO2", "O3"]]
y = SO2_Fairbanks['SO2']
X = sm.add_constant(X, prepend=True)

pollution_model = sm.OLS(y, X).fit()
print("Type of model: ", type(pollution_model))
print("Params from the model:\n", pollution_model.params)
#print(pollution_model.summary())
print("99% CI:\n", pollution_model.conf_int(0.01))


# Add interval percent widths
alphas = [     0.01,  0.05,   0.1] 
widths = [ '99% CI', '95%', '90%']
colors = ['#fee08b','#fc8d59','#d53e4f']

sns.set() # Set default Seaborn style
plt.figure()
for alpha, color, width in zip(alphas, colors, widths):
    conf_ints = pollution_model.conf_int(alpha) # Grab confidence interval
    plt.hlines(y = conf_ints.index, 
               xmin = conf_ints[0], xmax = conf_ints[1], 
               colors = color, label = width, linewidth = 10) # Pass current interval color and legend label to plot
plt.plot(pollution_model.params, pollution_model.params.index, 'ko', label = 'Point Estimate') # Draw point estimates
plt.axvline(x = 0, color = 'orangered', linestyle = '--') # Add a 'null' reference line at 0 and color orangered
plt.xlabel('Estimations') # Set descriptive axis labels and title
plt.ylabel('Variables') # Set descriptive axis labels and title
plt.legend(loc='best')
plt.title('Pollutants estimation in Fairbanks')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')





print("****************************************************")
tema = '10. 90 and 95% bands'; print("** %s\n" % tema)

cinci_13_no2   = pollution[(pollution.city=='Cincinnati') & (pollution.year==2013)].set_index('Date')
cinci_13_no2   = pd.DataFrame({'day'    : cinci_13_no2.day,
                               'mean'   : cinci_13_no2.rolling(window=40)['NO2'].mean().values,
                               'std_'   : cinci_13_no2.rolling(window=40)['NO2'].std().values,
                               'sem'    : cinci_13_no2.rolling(window=40)['NO2'].std().values/np.sqrt(40)}).dropna()
print(cinci_13_no2.head()) #Polutants in Cincinnati


int_widths = ['90%', '99%']
z_scores = [1.67, 2.58]
colors = ['#fc8d59', '#fee08b']

sns.set() # Set default Seaborn style
plt.figure()
for percent, Z, color in zip(int_widths, z_scores, colors):
    plt.fill_between(x = cinci_13_no2.day, 
                     y1 = cinci_13_no2['mean'] - Z*cinci_13_no2['sem'], 
                     y2 = cinci_13_no2['mean'] + Z*cinci_13_no2['sem'],
                     alpha = 0.4, color = color, label = percent)# Pass lower and upper confidence bounds and lower opacity
plt.xlabel('Day of year') # Set descriptive axis labels and title
plt.ylabel('NO2 level') # Set descriptive axis labels and title
plt.legend(loc='best')
plt.title('Avg NO2 level in Cincinnati')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')





print("****************************************************")
tema = '11. Using band thickness instead of coloring'; print("** %s\n" % tema)

rocket_model   = pollution[(pollution.city=='Vandenberg Air Force Base') & (pollution.year==2015)].set_index('Date')[['SO2', 'NO2', 'CO', 'O3']].agg(['mean','sem']).transpose().reset_index()
rocket_model.columns = ['pollutant', 'est', 'sem']
print(rocket_model.head()) #Polutants in Vandenberg Air Force Base


# Decrase interval thickness as interval widens
sizes      = [    15,  10,  5]
int_widths = ['90% CI', '95%', '99%']
z_scores   = [    1.67,  1.96,  2.58]

sns.set() # Set default Seaborn style
plt.figure()
for percent, Z, size in zip(int_widths, z_scores, sizes):
    plt.hlines(y = rocket_model.pollutant, 
               xmin = rocket_model['est'] - Z*rocket_model['sem'],
               xmax = rocket_model['est'] + Z*rocket_model['sem'],
               label = percent, linewidth = size, color = 'gray') 
plt.plot('est', 'pollutant', 'wo', data = rocket_model, label = 'Point Estimate') # Add point estimate
plt.xlabel('Estimations') # Set descriptive axis labels and title
plt.ylabel('Pollutants') # Set descriptive axis labels and title
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.title('Quality Air in Vandenberg Air Force Base')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=0.7, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')





print("****************************************************")
tema = '12. Visualizing the bootstrap'; print("** %s\n" % tema)

denver_aug = pollution.query("city=='Denver' & month==8"); print("Denver in August: \n", denver_aug.head());
pollution_aug = pollution.query('month==8'); print("Pollution in August: \n", pollution_aug.head());

#Visualizing the boostrap
boots_denver = bootstrap(denver_aug.NO2, 1000, np.mean, SEED) #Generate 1,000 bootstrap samples.
lower, upper = np.percentile(boots_denver, [2.5, 97.5]) #Get lower and upper 95% interval bounds

sns.set() # Set default Seaborn style
plt.figure()
plt.axvspan(lower, upper, color='grey', alpha=0.2)
sns.distplot(boots_denver, bins=100, kde=False)
plt.xlabel('NO2 Bootstrap Samples') # Set descriptive axis labels and title
plt.ylabel('NO2 Mean Sample') # Set descriptive axis labels and title
plt.title('Denver in August')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#Make cataframe of boostrapped data
denver_aug_boot = pd.concat([denver_aug.sample(n=len(denver_aug), replace=True).assign(sample=i) for i in range(100)])

sns.set() # Set default Seaborn style
#plt.figure()
sns.lmplot('CO', 'O3', data=denver_aug_boot, scatter=False, hue='sample', line_kws={'color':'coral','alpha':0.2}, ci=None, legend=False)
plt.xlabel('CO') # Set descriptive axis labels and title
plt.ylabel('O3') # Set descriptive axis labels and title
plt.title('Denver in August')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#Visualize data by city
city_boots = pd.DataFrame()

for city in ['Denver', 'Long Beach', 'Houston', 'Indianapolis']:
    city_NO2 = pollution_aug[pollution_aug.city == city].NO2 #Filter to city's NO2
    cur_boot = pd.DataFrame({'NO2_avg': bootstrap(city_NO2, 100, np.mean, SEED),
                             'city'   : city })
    city_boots = pd.concat([city_boots, cur_boot])

sns.set() # Set default Seaborn style
plt.figure()
sns.swarmplot(y='city', x='NO2_avg', data=city_boots, color='coral')
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
plt.title('NO2 levels in August')
plt.suptitle(tema)
plt.subplots_adjust(left=0.22, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '13. The bootstrap histogram'; print("** %s\n" % tema)

cinci_may_NO2 = pollution.query("city  ==  'Cincinnati' & month  ==  5").NO2
print("NO2 level in May in Cincinnati: \n", cinci_may_NO2.head())


# Generate bootstrap samples
boot_means = bootstrap(cinci_may_NO2, 1000, np.mean, SEED)
lower, upper = np.percentile(boot_means, [2.5, 97.5]) # Get lower and upper 95% interval bounds

sns.set() # Set default Seaborn style
plt.figure()
plt.axvspan(lower, upper, color = 'gray', alpha = 0.2) # Plot shaded area for interval
sns.distplot(boot_means, bins = 100, kde = False) # Draw histogram of bootstrap samples
plt.xlabel('NO2 Bootstrap Samples') # Set descriptive axis labels and title
plt.ylabel('NO2 Mean Sample') # Set descriptive axis labels and title
plt.title('Cincinaty in May')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '14. Bootstrapped regressions'; print("** %s\n" % tema)

no2_so2 = pollution.query("city=='Long Beach' & month==5")[["NO2","SO2"]]; 
print("Pollution in Long Beach: \n", no2_so2.head());
no2_so2_boot = pd.concat([no2_so2.sample(n=no2_so2.shape[0], replace=True).assign(sample=i) for i in range(70)])
print("Boostrapped sampled from Long Beach: \n", no2_so2_boot.head());

sns.set() # Set default Seaborn style
#plt.figure()
sns.lmplot('NO2', 'SO2', data = no2_so2_boot, hue = 'sample', # Tell seaborn to a regression line for each sample 
           line_kws = {'color': 'steelblue', 'alpha': 0.2}, # Make lines blue and transparent
           ci = False, legend = False, scatter = False) # Disable built-in confidence intervals
plt.scatter('NO2', 'SO2', data = no2_so2) # Draw scatter of all points
#plt.xlabel('NO2 Bootstrap Samples') # Set descriptive axis labels and title
#plt.ylabel('NO2 Mean Sample') # Set descriptive axis labels and title
plt.title('Long Beach in May')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


plt.show()

print("****************************************************")
tema = '15. Lots of bootstraps with beeswarms'; print("** %s\n" % tema)

pollution_may = pollution.query("month==5"); print("Pollution in May: \n", pollution_may.head())

city_boots = pd.DataFrame() # Initialize a holder DataFrame for bootstrap results
for city in ['Cincinnati', 'Des Moines', 'Indianapolis', 'Houston']:
    city_NO2 = pollution_may[pollution_may.city  ==  city].NO2 # Filter to city
    cur_boot = pd.DataFrame({'NO2_avg': bootstrap(city_NO2, 100, np.mean, SEED), 'city': city}) # Bootstrap city data & put in DataFrame
    city_boots = pd.concat([city_boots,cur_boot]) # Append to other city's bootstraps
    
sns.set() # Set default Seaborn style
plt.figure()
sns.swarmplot(y = "city", x = "NO2_avg", data = city_boots, color = 'coral') # Beeswarm plot of averages with citys on y axis
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
plt.title('NO2 levels in May')
plt.suptitle(tema)
plt.subplots_adjust(left=0.22, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
print("** END                                            **")
print("****************************************************")