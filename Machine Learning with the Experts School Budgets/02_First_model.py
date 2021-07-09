# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:04:01 2019

@author: jacqueline.cortez

Capítulo 2. Creating a simple first model
Introduction:
    In this chapter, you'll build a first-pass model. You'll use numeric data only to train the model. 
    Spoiler alert - throwing out all of the text data is bad for performance! But you'll learn how to format 
    your predictions. Then, you'll be introduced to natural language processing (NLP) in order to start working 
    with the large amounts of text in the data.
"""

# Import packages
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
import warnings

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions
#from scipy.stats import randint                                                     #For learning machine 

from sklearn.feature_extraction.text import CountVectorizer                         #For learning machine
#from sklearn import datasets                                                        #For learning machine
#from sklearn.impute import SimpleImputer                                            #For learning machine
#from sklearn.linear_model import ElasticNet                                         #For learning machine
#from sklearn.linear_model import Lasso                                              #For learning machine
#from sklearn.linear_model import LinearRegression                                   #For learning machine
from sklearn.linear_model import LogisticRegression                                 #For learning machine
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
from sklearn.multiclass import OneVsRestClassifier                                   #For learning machine
#from sklearn.neighbors import KNeighborsClassifier                                 #For learning machine
#from sklearn.pipeline import Pipeline                                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
#from sklearn.preprocessing import scale                                             #For learning machine
#from sklearn.preprocessing import StandardScaler                                    #For learning machine
#from sklearn.svm import SVC                                                         #For learning machine
#from sklearn.tree import DecisionTreeClassifier                                     #For learning machine
import multilabel                                                                   #For multivariable target, function created by Datacamp
import multi_log_loss                                                               #Datacamp logloss for multiple targets score



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


#def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
def combine_text_columns(data_frame, to_drop): # Define combine_text_columns()
    """ converts all text in each row of data_frame to single vector """ 
    
    to_drop = set(to_drop) & set(data_frame.columns.tolist()) # Drop non-text columns that are in the df
    text_data = data_frame.drop(to_drop, axis=1)
    text_data.fillna('', inplace=True) # Replace nans with blanks
        
    return text_data.apply(lambda x: " ".join(x), axis=1) # Join all text items in a row that have a space in between
    


LABELS                   = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']
NUMERIC_COLUMNS          = ['FTE', 'Total']
BOX_PLOTS_COLUMN_INDICES = [range(37), range(37,48), range(48,51), range(51,76), range(76,79), range(79,82), range(82,87), range(87,96), range(96,104)]
PATH_TO_PREDICTIONS      = 'predictions.csv'
PATH_TO_HOLDOUT_LABELS   = 'TestSetLabelsSample.csv'
TOKENS_ALPHANUMERIC      = '[A-Za-z0-9]+(?=\\s+)' # Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_BASIC             = '\\S+(?=\\s+)' # Create the basic token pattern


file = "TrainingSetSample.csv" 
budget_df = pd.read_csv(file, index_col=0)

categorize_label = lambda x: x.astype('category') # Define the lambda function: categorize_label
budget_df[LABELS] = budget_df[LABELS].apply(categorize_label, axis=0) # Convert df[LABELS] to a categorical type


# Load the holdout data: holdout
file = "TestSetSample.csv"
holdout = pd.read_csv(file, index_col=0)



print("****************************************************")
tema = '2. Setting up a train-test split in scikit-learn'; print("** %s\n" % tema)

numeric_data_only = budget_df[NUMERIC_COLUMNS].fillna(-1000) # Create the new DataFrame: numeric_data_only
label_dummies = pd.get_dummies(budget_df[LABELS], prefix_sep='__') # Get labels and convert to dummy variables: label_dummies

warnings.filterwarnings('ignore', 'Size less than number of columns*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
X_train, X_test, y_train, y_test = multilabel.multilabel_train_test_split(numeric_data_only, label_dummies,  size=0.2, seed=123) # Create training and test sets
warnings.filterwarnings('default', 'Size less than number of columns*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


# Print the info
print("X_train info:"); print(X_train.info())
print("\nX_test info:"); print(X_test.info())
print("\ny_train info:"); print(y_train.info())
print("\ny_test info:"); print(y_test.info())


print("****************************************************")
tema = '3. Training a model'; print("** %s\n" % tema)

clf = OneVsRestClassifier(LogisticRegression(solver='liblinear')) # Instantiate the classifier: clf
clf.fit(X_train, y_train) # Fit the classifier to the training data

print("Accuracy: {}".format(clf.score(X_test, y_test))) # Print the accuracy



print("****************************************************")
tema = '5. Use your model to predict values on holdout data'; print("** %s\n" % tema)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))



print("****************************************************")
tema = '6. Writing out your results to a csv for submission'; print("** %s\n" % tema)

prediction_df = pd.DataFrame(columns=pd.get_dummies(budget_df[LABELS], prefix_sep='__').columns, index=holdout.index, data=predictions) # Format predictions in DataFrame: prediction_df
prediction_df.to_csv(PATH_TO_PREDICTIONS) # Save prediction_df to csv
score = multi_log_loss.score_submission(pred_path=PATH_TO_PREDICTIONS, holdout_path=PATH_TO_HOLDOUT_LABELS, column_indices=BOX_PLOTS_COLUMN_INDICES) # Submit the predictions for scoring: score

# Print score
print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))



print("****************************************************")
tema = '11. Creating a bag-of-words in scikit-learn'; print("** %s\n" % tema)

budget_df.Position_Extra.fillna('', inplace=True) # Fill missing values in df.Position_Extra
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC) # Instantiate the CountVectorizer: vec_alphanumeric
vec_alphanumeric.fit(budget_df.Position_Extra) # Fit to the data

# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names())



print("****************************************************")
tema = "13. What's in a token?"; print("** %s\n" % tema)


vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC) # Instantiate basic CountVectorizer: vec_basic
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC) # Instantiate alphanumeric CountVectorizer: vec_alphanumeric

text_vector = combine_text_columns(budget_df, to_drop=NUMERIC_COLUMNS+LABELS) # Create the text vector

vec_basic.fit_transform(text_vector) # Fit and transform vec_basic
# Print number of tokens of vec_basic
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))

vec_alphanumeric.fit_transform(text_vector) # Fit and transform vec_alphanumeric
# Print number of tokens of vec_alphanumeric
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))



print("****************************************************")
print("** END                                            **")
print("****************************************************")