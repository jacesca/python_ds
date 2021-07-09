# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:04:53 2019

@author: jacqueline.cortez

Capítulo 1. Exploring the raw data
Introduction:
    In this chapter, you'll be introduced to the problem you'll be solving in this course. How do you accurately classify line-items 
    in a school budget based on what that money is being used for? You will explore the raw text and numeric values in the dataset, 
    both quantitatively and visually. And you'll learn how to measure success when trying to predict class labels for each row of the 
    dataset.
"""

# Import packages
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
#from scipy.stats import randint                                                     #For learning machine 

#from sklearn import datasets                                                        #For learning machine
#from sklearn.impute import SimpleImputer                                            #For learning machine
#from sklearn.linear_model import ElasticNet                                         #For learning machine
#from sklearn.linear_model import Lasso                                              #For learning machine
#from sklearn.linear_model import LinearRegression                                   #For learning machine
#from sklearn.linear_model import LogisticRegression                                 #For learning machine
#from sklearn.linear_model import Ridge                                              #For learning machine
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
#from sklearn.neighbors import KNeighborsClassifier                                  #For learning machine
#from sklearn.pipeline import Pipeline                                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
#from sklearn.preprocessing import scale                                             #For learning machine
#from sklearn.preprocessing import StandardScaler                                    #For learning machine
#from sklearn.svm import SVC                                                         #For learning machine
#from sklearn.tree import DecisionTreeClassifier                                     #For learning machine
#import multilabel                                                                   #For multivariable target, function created by Datacamp

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



print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

def compute_log_loss(predicted, actual, eps=1e-14):
    """ Computes the logarithmic loss between predicted and actual when these are 1D arrays.
    
        :param predicted: The predicted probabilities as floats between 0-1.
        :param actual: The actual binary labels. Either 0 or 1.
        :param eps (optional): log(0) is inf, so we need to offset our predicted values slightly by eps from 0 or 1.
    """
    predicted = np.clip(predicted, eps, 1-eps)
    loss = -1*np.mean(actual*np.log(predicted) + (1-actual)*np.log(1-predicted))
    return loss

file = "TrainingSetSample.csv" 
budget_df = pd.read_csv(file, index_col=0)
LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']


print("****************************************************")
tema = '6. Summarizing the data'; print("** %s\n" % tema)

print("Head of data:\n{}".format(budget_df.head()))
print("\nTail of data:\n{}".format(budget_df.tail()))
print("\nInfo related to the columns: "); print(budget_df.info())
print("\nSize of data:\n{}".format(budget_df.shape))
print("\nSummarize statistic information of the data:\n{}".format(budget_df.describe()))

mu = budget_df['FTE'].dropna().mean()
#sigma = budget_df['FTE'].dropna().std()

# Create the histogram
sns.set() # Set default Seaborn style
#plt.figure()
plt.hist(budget_df['FTE'].dropna())
#n, bins, patches = plt.hist(budget_df['FTE'].dropna()) # Plot the histogram of the replicates
#y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
#plt.plot(bins, y, '--', color='black')
plt.axvline(mu, color='red', linestyle='dashed', linewidth=2)
plt.text(mu+0.01, 40,"mean", color='red', fontsize=9)
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#Improving the graph
sns.set() # Set default Seaborn style
plt.figure()
sns.kdeplot(budget_df['FTE'].dropna(), shade = True, cut=0) # cut=0-->Limit the density curve within the range of the data
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('KDE (Kernel Density Estimator)')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


#Another graph
sns.set() # Set default Seaborn style
plt.figure()
sns.distplot(budget_df['FTE'].dropna()) # cut=0-->Limit the density curve within the range of the data
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('\nKDE (Kernel Density Estimator)')
plt.ylim((0, 1.5))
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '8. Exploring datatypes in pandas'; print("** %s\n" % tema)

print("Types in data:\n{}".format(budget_df.dtypes))
print("\nSummarizing:\n{}".format(budget_df.dtypes.value_counts()))


print("****************************************************")
tema = '9. Encode the labels as categorical variables'; print("** %s\n" % tema)

categorize_label = lambda x: x.astype('category') # Define the lambda function: categorize_label
budget_df[LABELS] = budget_df[LABELS].apply(categorize_label, axis=0) # Convert df[LABELS] to a categorical type

print(budget_df[LABELS].dtypes) # Print the converted dtypes


print("****************************************************")
tema = '10. Counting unique labels'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
plt.figure()

num_unique_labels = budget_df[LABELS].apply(pd.Series.nunique, axis=0) # nunique is for count distinct values. Calculate number of unique values for each label: num_unique_labels
num_unique_labels.plot(kind='bar') # Plot number of unique values for each label

plt.xlabel('Labels')
plt.ylabel('Number of unique values')
plt.title('Labels found in each category')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.38, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("Unique labels:\n{}".format(num_unique_labels))
print("Total unique labels: {}".format(num_unique_labels.sum()))
print("\nCategories found in Pre_k: {}".format(budget_df.Pre_K.nunique()))
print("Categories index of Pre_k: {}".format(budget_df.Pre_K.cat.categories))
print("Numerical values of Pre_k: {}".format(budget_df.Pre_K.cat.codes.unique()))

print("****************************************************")
tema = '13. Computing log loss with NumPy'; print("** %s\n" % tema)

actual_labels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
correct_confident = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05])
correct_not_confident = np.array([0.65, 0.65, 0.65, 0.65, 0.65, 0.35, 0.35, 0.35, 0.35, 0.35])
wrong_confident = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95])
wrong_not_confident = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.65, 0.65, 0.65, 0.65, 0.65])

correct_confident_loss = compute_log_loss(correct_confident, actual_labels) # Compute and print log loss for 1st case
correct_not_confident_loss = compute_log_loss(correct_not_confident, actual_labels) # Compute log loss for 2nd case
wrong_not_confident_loss = compute_log_loss(wrong_not_confident, actual_labels) # Compute and print log loss for 3rd case
wrong_confident_loss = compute_log_loss(wrong_confident, actual_labels) # Compute and print log loss for 4th case
actual_labels_loss = compute_log_loss(actual_labels, actual_labels) # Compute and print log loss for actual labels

print("Log loss, correct and confident: {}".format(correct_confident_loss)) 
print("Log loss, correct and not confident: {}".format(correct_not_confident_loss)) 
print("Log loss, wrong and not confident: {}".format(wrong_not_confident_loss)) 
print("Log loss, wrong and confident: {}".format(wrong_confident_loss)) 
print("Log loss, actual labels: {}".format(actual_labels_loss))

print("****************************************************")
print("** END                                            **")
print("****************************************************")