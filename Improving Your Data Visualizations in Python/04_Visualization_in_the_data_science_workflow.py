# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 20:02:09 2019

@author: jacqueline.cortez

Chapter 4. Visualization in the data science workflow
Introduction:
    Often visualization is taught in isolation, with best practices only discussed in a general way. 
    In reality, you will need to bend the rules for different scenarios. From messy exploratory 
    visualizations to polishing the font sizes of your final product; in this chapter, we dive into 
    how to optimize your visualizations at each step of a data science workflow.
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
import calendar                                                                     #For accesing to a vary of calendar operations
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

#import statsmodels.api as sm                                                        #Make a prediction model
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
pd.set_option("display.max_columns",20)
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

#SEED=42

#print("****************************************************")
print("** Getting the data for this program\n")

file = "pollution_wide.csv" 
pollution = pd.read_csv(file)
pollution['Date'] = pd.to_datetime((pollution.year*1000+pollution.day).apply(str), format='%Y%j')
pollution['day_of_month'] = pollution['Date'].dt.day

file = "markets_cleaned.csv" 
markets = pd.read_csv(file)
markets['num_items_sold'] = markets[['Bakedgoods', 'Beans', 'Cheese', 'Coffee', 'Crafts', 'Eggs', 'Flowers', 'Fruits', 'Grains', 'Herbs', 'Honey', 'Jams', 'Juices', 'Maple', 'Meat', 'Mushrooms', 'Nursery', 'Nuts', 'PetFood', 'Plants', 'Poultry', 'Prepared', 'Seafood', 'Soap', 'Tofu', 'Trees', 'Vegetables', 'WildHarvested', 'Wine']].sum(axis=1)

file = "census-state-populations.csv" 
city_populations = pd.read_csv(file)
markets = pd.merge(markets, city_populations, how='inner', on='state')
markets['log_pop'] = np.log(markets['pop_est_2014']) # Create a new logged population column 


#5. Exploring the patterns
long_beach_data = pollution.query("city=='Long Beach'")
long_beach_avgs = long_beach_data.groupby('month')[['month', 'CO', 'NO2', 'O3', 'SO2']].mean()
long_beach_avgs['month'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#7. What state is the most market-friendly?
markets_and_pop = (markets.groupby('state', as_index = False)
                          .agg({'name': lambda d: np.log(len(d)), 'pop_est_2014': lambda d: np.log(d.iloc[0]) })
                          .rename(columns = {'name': 'log_markets', 'pop_est_2014': 'log_pop' }))

#8. Popularity of goods sold by state
goods_by_state = pd.melt(markets, id_vars=['state'], 
                         value_vars=['Bakedgoods', 'Beans', 'Cheese', 'Coffee', 'Crafts', 'Eggs', 'Flowers', 
                                     'Fruits', 'Grains', 'Herbs', 'Honey', 'Jams', 'Juices', 'Maple', 'Meat', 
                                     'Mushrooms', 'Nursery', 'Nuts', 'PetFood', 'Plants', 'Poultry', 'Prepared',
                                     'Seafood', 'Soap', 'Tofu', 'Trees', 'Vegetables','WildHarvested', 'Wine'],
                         value_name='sold').rename(columns = {'variable': 'good'})
goods_by_state = pd.merge(goods_by_state.groupby(['state', 'good'])['sold'].sum().reset_index(),
                          goods_by_state.groupby(['state'])['sold'].sum().reset_index().rename(columns = {'sold': 'total'}),
                          how='left', on='state')
goods_by_state['prop_selling'] = goods_by_state.sold / goods_by_state.total
to_plot = ['Cheese','Maple','Fruits','Grains','Seafood','Plants','Vegetables'] # Subset goods to interesting ones
goods_by_state_small = goods_by_state.query("good in "+str(to_plot))


#9. Making your visualizations efficient
long_beach_data = pollution.query("city=='Long Beach'")
#pol_by_month = long_beach_data.groupby(['year', 'month'])[['CO', 'NO2', 'O3', 'SO2']].mean().reset_index()
#pol_by_month['month'] = long_beach_avgs['month'].apply(lambda x: calendar.month_abbr[x])
#obs_by_year = long_beach_data.groupby(['year'])[['NO2']].count().reset_index().rename(columns = {'NO2': 'count'})
pol_by_month = long_beach_data.groupby(['year', 'month'])[['NO2']].agg(['mean', 'count']).set_axis(['mean','count'], axis=1, inplace=False).reset_index()
pol_by_month['month'] = pol_by_month['month'].apply(lambda x: calendar.month_abbr[x])
obs_by_year = pol_by_month.groupby(['year'])[['count']].sum().reset_index()


#11. Using a plot as a legend
markets_by_state =(markets.groupby(['state'])[['name', 'pop_est_2014']].agg({'name':'count', 'pop_est_2014':'last'})
                          .rename(columns = {'name': 'num_markets', 'pop_est_2014': 'population'}).reset_index())
markets_by_state['people_per_market'] = markets_by_state['population']/markets_by_state['num_markets']
markets_by_state['log_pop']           = np.log(markets_by_state['population'])
markets_by_state['log_markets']       = np.log(markets_by_state['num_markets'])
markets_by_state['is_selected']       = markets_by_state['state'].apply(lambda x: x if x in ['Maryland', 'Texas', 'Vermont'] else 'other')
markets_by_state.sort_values(by='people_per_market', inplace=True)


#13. Cleaning up the background
selected_goods_by_state = goods_by_state.query("good in ['Cheese', 'Eggs', 'Fruits', 'Maple', 'Poultry', 'Wine']")
highlighted = selected_goods_by_state.query("state in ['New Mexico','North Dakota','Vermont']") # Draw lines across goods for highlighted states
last_rows = highlighted.groupby('state', as_index = False).agg('last')


#14. Remixing a plot
markets_by_month = markets.groupby(['state', 'months_open'], as_index=False)[['num_items_sold']].sum().pivot(index='state', columns='months_open', values='num_items_sold').fillna(0).apply(lambda x: x/sum(x)).set_axis(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], axis=1, inplace=False)
state_by_lon = markets.groupby(['state'])[['lon']].mean().sort_values(by='lon', ascending=False)



print("****************************************************")
tema = '1. First explorations'; print("** %s\n" % tema)

print("Pollution dataset: \n", pollution.head())
print("\n", pollution.describe(percentiles=[0.5], include='all'))

sns.set() # Set default Seaborn style
#plt.figure()
pd.plotting.scatter_matrix(pollution[['month', 'day', 'CO', 'NO2', 'O3', 'SO2']], alpha=0.2)
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
#plt.title('A Dataset little view')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')




print("****************************************************")
tema = '2. Looking at the farmers market data'; print("** %s\n" % tema)

first_rows = markets.head(3).transpose() # Print first three rows of data and transpose
print(first_rows)

col_descriptions = markets.describe(include = 'all', percentiles = [0.5]).transpose() # Get descriptions of every column
print(col_descriptions)




print("****************************************************")
tema = '3. Scatter matrix of numeric columns'; print("** %s\n" % tema)

numeric_columns = ['lat', 'lon', 'months_open', 'num_items_sold', 'pop_est_2014'] # Select just the numeric columns (exluding individual goods)
#sns.set() # Set default Seaborn style
#plt.figure()
sm = pd.plotting.scatter_matrix(markets[numeric_columns], figsize = (10, 5), alpha = 0.5) # Make a scatter matrix of numeric columns
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
#plt.title('A Dataset little view')
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)] #Change label rotation
[s.get_yaxis().set_label_coords(-0.4, 0.5) for s in sm.reshape(-1)] #May need to offset label when rotating to prevent overlap of figure
#[s.get_yaxis().set_label_position('right') for s in sm.reshape(-1)] #Apply alignment to y label
#plt.xticks(fontsize = 2)
#plt.yticks(fontsize = 2)
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '4. Digging in with basic transforms'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
sns.scatterplot(x = 'log_pop', y = 'num_items_sold', alpha = 0.25, data = markets) # Draw a scatterplot of log-population to # of items sold
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
plt.title('Relation between Population and Items sold')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '5. Exploring the patterns'; print("** %s\n" % tema)

print(long_beach_data.head())
print(long_beach_avgs)

#Explore relationship between NO2 and CO.
sns.set() # Set default Seaborn style
plt.figure()
sns.regplot('NO2', 'CO', ci=False, data=long_beach_data, scatter_kws={'alpha':0.2, 'color':'grey'})
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
plt.annotate('Reduce point opacity\nshows overlap in\ndese areas.', xy=(30,1), xytext=(0,3.5), color='red', arrowprops={'color':'red'})
plt.title('Relation between NO2 and CO')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#Set label in the scatter plot
sns.set() # Set default Seaborn style
plt.figure()
g = sns.scatterplot('SO2', 'CO', data=long_beach_avgs)
for _, row in long_beach_avgs.iterrows(): #Iterate over the rows of our data
    month, CO, NO2, O3, SO2 = row #Unpack columns from row
    g.annotate(month, (SO2, CO))
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
plt.title('Long Beach avg SO2 by CO')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '6. Is longitude related to months open?'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
sns.regplot(x = 'lon', y = 'months_open', scatter_kws = {'alpha':0.1, 'color':'gray'}, ci = False, data = markets)
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
plt.title('Relation between lon and months_open')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '7. What state is the most market-friendly?'; print("** %s\n" % tema)

print(markets_and_pop.head())
sns.set() # Set default Seaborn style
plt.figure()
g = sns.regplot("log_markets", "log_pop", ci = False, scatter_kws = {'s':2}, data = markets_and_pop)
for _, row in markets_and_pop.iterrows(): # Iterate over the rows of the data
    state, log_markets, log_pop = row
    g.annotate(state, (log_markets, log_pop), size = 6) # Place annotation and reduce size for clarity
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
plt.title('Relation between states and market-friendly')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')




print("****************************************************")
tema = '8. Popularity of goods sold by state'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()
g = sns.scatterplot('good','prop_selling', data = goods_by_state_small, s = 0)
for _,row in goods_by_state_small.iterrows():
    g.annotate(row['state'], (row['good'], row['prop_selling']), ha = 'center', size = 10)
#plt.xlabel('CO') # Set descriptive axis labels and title
#plt.ylabel('O3') # Set descriptive axis labels and title
plt.title('Are there any interesting relation here?')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')




print("****************************************************")
tema = '9. Making your visualizations efficient'; print("** %s\n" % tema)

print(long_beach_avgs.head())

sns.set() # Set default Seaborn style
#plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.tick_params(axis='x', labelsize=6); ax1.tick_params(axis='y', labelsize=7); #, rotation=90
ax2.tick_params(axis='x', labelsize=6); ax2.tick_params(axis='y', labelsize=7); #, rotation=90
sns.lineplot('month', 'mean', 'year', ax=ax1, data=pol_by_month, palette='RdBu'); 
sns.barplot('year', 'count', ax=ax2, data=obs_by_year, palette='RdBu')
ax1.legend_.remove(); #ax1.legend(fontsize=7);
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=0.25, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '10. Stacking to find trends'; print("** %s\n" % tema)

_, (ax1, ax2) = plt.subplots(1, 2) # Setup two stacked plots
sns.scatterplot("lat", "lon", 'months_open', data = markets, 
                palette = sns.light_palette("orangered",n_colors = 12), legend = False, ax = ax1) # Draw location scatter plot on first plot
sns.regplot('lat', 'months_open', data = markets, 
            scatter_kws = {'alpha': 0.2, 'color': 'gray', 'marker': 'o'}, lowess = True, marker = 'o', ax = ax2) # Plot a regression plot on second plot
#ax1.axis('square'); ax2.axis('square');
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=0.25, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '11. Using a plot as a legend'; print("** %s\n" % tema)

print(markets_by_state.head())

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5)) # Set up two side-by-side plots
sns.barplot('people_per_market', 'state', hue = 'is_selected', data = markets_by_state, 
            dodge = False, ax = ax1) # Map the column for selected states to the bar color
sns.scatterplot('log_pop', 'log_markets', hue = 'is_selected', data = markets_by_state, 
                ax = ax2, s = 100) # Map selected states to point color
ax1.tick_params(axis='x', labelsize=7); ax1.tick_params(axis='y', labelsize=7); ax1.legend_.remove();
ax2.tick_params(axis='x', labelsize=7); ax2.tick_params(axis='y', labelsize=7); ax2.legend_.remove();
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=0.25, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '13. Cleaning up the background'; print("** %s\n" % tema)

sns.set_style('whitegrid') # Set background to white with grid
plt.figure()
plt.scatter('good','prop_selling', marker = '_', alpha = 0.7, data = selected_goods_by_state)
sns.lineplot('good','prop_selling', 'state', data = highlighted, legend = False)
for _,row in last_rows.iterrows(): # Draw state name at end of lines
    plt.annotate("  {}".format(row['state']), (row['good'], row['prop_selling']), ha = 'left') #, xytext = (5,0), textcoords = 'offset pixels')
sns.despine(bottom = True, left = True) # Remove all borders
plt.title('Goods being sold by the proportion of markets in a state')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=0.85, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '14. Remixing a plot'; print("** %s\n" % tema)

sns.set_style('whitegrid') # Set background to white with grid
plt.figure()
#sns.set(font_scale = 1) # Decrease font size so state names are less crowded
blue_pal = sns.light_palette("steelblue", as_cmap = True) # Switch to an appropriate color palette
g = sns.heatmap(markets_by_month.reindex(state_by_lon.index), linewidths = 0.1, cmap = blue_pal, cbar = False, yticklabels = True) # Order states by longitude
g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize=6) # Rotate y-axis ticks 
plt.xlabel('Month open') # Set descriptive axis labels and title
plt.ylabel('By Longitud in descendent order') # Set descriptive axis labels and title
plt.title('Distribution of months open for farmers markets by longitude')
plt.suptitle(tema)
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')




print("****************************************************")
tema = '15. Enhancing legibility'; print("** %s\n" % tema)

markets_by_state['state_colors'] = markets_by_state.state.map({'Vermont':'steelblue', 'Texas':'orangered'}).fillna('gray')
tx_message = 'Texas has a large population\nand relatively few farmers\nmarkets.'

#sns.set() # Set default Seaborn style
f, (ax1, ax2) = plt.subplots(1, 2) # Set up two side-by-side plots
sns.barplot(x='people_per_market', y='state', data = markets_by_state, 
            palette = markets_by_state['state_colors'].tolist(), ax = ax1) # Draw barplot w/ colors mapped to state_colors vector
p = sns.scatterplot(x='population', y='num_markets', hue='state_colors', data = markets_by_state, 
                    ax = ax2, s = 60, palette=dict(steelblue='steelblue', orangered='orangered', gray='gray'))#, 
ax1.tick_params(axis='y', labelsize=6); ax2.legend_.remove();
ax2.set(xscale = "log", yscale = 'log') # Log the x and y scales of our scatter plot so it's easier to read
ax2.annotate(tx_message, xy = (26956958,230), xytext = (26956958, 450), ha = 'right', color='orangered',
             size = 7, backgroundcolor = 'white', arrowprops = {'color':'orangered', 'width': 3}) # Increase annotation text size for legibility
sns.set_style('whitegrid')
plt.suptitle('{}\nDistribution of months open for farmers markets by longitude'.format(tema))
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()
plt.style.use('default')




print("****************************************************")
print("** END                                            **")
print("****************************************************")
