# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:23:06 2019

@author: jacqueline.cortez

Capítulo 4. Learning from the experts
Introduction:
    In this chapter, you will learn the tricks used by the competition winner, and implement them yourself using scikit-learn. Enjoy!
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

#from sklearn.ensemble import RandomForestClassifier                                 #For learning machine
from sklearn.feature_selection import chi2                                          #For learning machine
from sklearn.feature_selection import SelectKBest                                   #For learning machine
from sklearn.feature_extraction.text import CountVectorizer                         #For learning machine
from sklearn.feature_extraction.text import HashingVectorizer                       #For learning machine
#from sklearn import datasets                                                        #For learning machine
from sklearn.impute import SimpleImputer                                            #For learning machine
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
from sklearn.pipeline import FeatureUnion                                           #For learning machine
from sklearn.pipeline import Pipeline                                               #For learning machine
from sklearn.preprocessing import FunctionTransformer                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
from sklearn.preprocessing import MaxAbsScaler                                      #For learning machine
#from sklearn.preprocessing import scale                                             #For learning machine
#from sklearn.preprocessing import StandardScaler                                    #For learning machine
#from sklearn.svm import SVC                                                         #For learning machine
#from sklearn.tree import DecisionTreeClassifier                                     #For learning machine
import multilabel                                                                   #For multivariable target, function created by Datacamp
import multi_log_loss                                                               #Datacamp logloss for multiple targets score
from SparseInteractions import SparseInteractions                                   #Implement interaction modeling like PolynomialFeatures

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

LABELS                   = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']
NUMERIC_COLUMNS          = ['FTE', 'Total']
#def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS+LABELS): # Define combine_text_columns()
    """ converts all text in each row of data_frame to single vector """ 
    
    to_drop = set(to_drop) & set(data_frame.columns.tolist()) # Drop non-text columns that are in the df
    text_data = data_frame.drop(to_drop, axis=1)
    text_data.fillna('', inplace=True) # Replace nans with blanks
        
    return text_data.apply(lambda x: " ".join(x), axis=1) # Join all text items in a row that have a space in between
    

# School Budget Database
BOX_PLOTS_COLUMN_INDICES = [range(37), range(37,48), range(48,51), range(51,76), range(76,79), range(79,82), range(82,87), range(87,96), range(96,104)]
PATH_TO_PREDICTIONS      = 'predictions.csv'
PATH_TO_HOLDOUT_LABELS   = 'TestSetLabelsSample.csv'
TOKENS_ALPHANUMERIC      = '[A-Za-z0-9]+(?=\\s+)' # Create the token pattern: TOKENS_ALPHANUMERIC

file                     = "TrainingSetSample.csv" 
budget_df                = pd.read_csv(file, index_col=0)
categorize_label         = lambda x: x.astype('category') # Define the lambda function: categorize_label
budget_df[LABELS]        = budget_df[LABELS].apply(categorize_label, axis=0) # Convert df[LABELS] to a categorical type

NON_LABELS               = [c for c in budget_df.columns if c not in LABELS] # Get the columns that are features in the original df

# Load the holdout data: holdout
file               = "TestSetSample.csv"
holdout            = pd.read_csv(file, index_col=0)



print("****************************************************")
tema = "3. Deciding what's a word"; print("** %s\n" % tema)

dummy_labels = pd.get_dummies(budget_df[LABELS], prefix_sep='__') # Get labels and convert to dummy variables: label_dummies

warnings.filterwarnings('ignore', 'Size less than number of columns*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
X_train, X_test, y_train, y_test = multilabel.multilabel_train_test_split(budget_df[NON_LABELS], dummy_labels,  size=0.2, seed=42) # Create training and test sets
warnings.filterwarnings('default', 'Size less than number of columns*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


text_vector = combine_text_columns(X_train) # Create the text vector
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC) # Instantiate the CountVectorizer: text_features
text_features.fit(text_vector) # Fit text_features to the text vector


msg = "There are {} alpha-numeric tokens in the dataset"
print(msg.format(len(text_features.get_feature_names())))
print("First 10 tokens: {}".format(text_features.get_feature_names()[:10]))
print("Last 10 tokens: {}\n".format(text_features.get_feature_names()[-10:]))




print("****************************************************")
tema = "4. N-gram range in scikit-learn"; print("** %s\n" % tema)

chi_k = 300 # Select 300 best features
np.random.seed(42) # Seed random number generator

# Perform preprocessing
get_text_data    = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Instantiate pipeline: pl
pl = Pipeline([('union', FeatureUnion([('numeric_features', Pipeline([('selector', get_numeric_data),
                                                                      ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))])),
                                       ('text_features',    Pipeline([('selector', get_text_data),
                                                                      ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1,2))),
                                                                      ('dim_red', SelectKBest(chi2, chi_k))]))])),
               ('scale', MaxAbsScaler()),
               ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))])
pl.fit(X_train, y_train) # Fit to the training data

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("Accuracy on budget dataset: ", accuracy)


predictions = pl.predict_proba(holdout)
prediction_df = pd.DataFrame(columns=dummy_labels.columns, index=holdout.index, data=predictions) # Format predictions in DataFrame: prediction_df
prediction_df.to_csv(PATH_TO_PREDICTIONS) # Save prediction_df to csv
score = multi_log_loss.score_submission(pred_path=PATH_TO_PREDICTIONS, holdout_path=PATH_TO_HOLDOUT_LABELS, column_indices=BOX_PLOTS_COLUMN_INDICES) # Submit the predictions for scoring: score
# Print score
print('Your model yields logloss score: {}'.format(score))

   
 
print("****************************************************")
tema = "7. Implement interaction modeling in scikit-learn"; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

# Instantiate pipeline: pl
pl = Pipeline([('union', FeatureUnion([('numeric_features', Pipeline([('selector', get_numeric_data),
                                                                      ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))])),
                                       ('text_features',    Pipeline([('selector', get_text_data),
                                                                      ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1, 2))),  
                                                                      ('dim_red', SelectKBest(chi2, chi_k))]))])),
               ('int', SparseInteractions(degree=2, feature_name_separator="__")),
               ('scale', MaxAbsScaler()),
               ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))])
pl.fit(X_train, y_train) # Fit to the training data

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("Accuracy on budget dataset: ", accuracy)


predictions = pl.predict_proba(holdout)
prediction_df = pd.DataFrame(columns=dummy_labels.columns, index=holdout.index, data=predictions) # Format predictions in DataFrame: prediction_df
prediction_df.to_csv(PATH_TO_PREDICTIONS) # Save prediction_df to csv
score = multi_log_loss.score_submission(pred_path=PATH_TO_PREDICTIONS, holdout_path=PATH_TO_HOLDOUT_LABELS, column_indices=BOX_PLOTS_COLUMN_INDICES) # Submit the predictions for scoring: score
# Print score
print('Your model yields logloss score: {}'.format(score))


print("****************************************************")
tema = "10. Implementing the hashing trick in scikit-learn"; print("** %s\n" % tema)

text_data = combine_text_columns(X_train) # Get text data: text_data
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC) # Instantiate the HashingVectorizer: hashing_vec
hashed_text = hashing_vec.fit_transform(text_data) # Fit and transform the Hashing Vectorizer
hashed_df = pd.DataFrame(hashed_text.data) # Create DataFrame and print the head

print(hashed_df.head())
print(hashed_df.shape)



print("****************************************************")
tema = "11. Build the winning model"; print("** %s\n" % tema)
np.random.seed(42) # Seed random number generator

# Instantiate the winning model pipeline: pl
pl = Pipeline([('union', FeatureUnion([('numeric_features', Pipeline([('selector', get_numeric_data),
                                                                      ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))])),
                                       ('text_features',    Pipeline([('selector', get_text_data),
                                                                      ('vectorizer', HashingVectorizer(alternate_sign = False, token_pattern=TOKENS_ALPHANUMERIC, norm=None, binary=False, ngram_range=(1, 2))), #the option non_negative=True has been deprecated in 0.19 and will be removed in version 0.21.
                                                                      ('dim_red', SelectKBest(chi2, chi_k))]))])),
               ('int', SparseInteractions(degree=2, feature_name_separator="__")),
               ('scale', MaxAbsScaler()),
               ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))])
pl.fit(X_train, y_train) # Fit to the training data

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("Accuracy on budget dataset: ", accuracy)


predictions = pl.predict_proba(holdout)
prediction_df = pd.DataFrame(columns=dummy_labels.columns, index=holdout.index, data=predictions) # Format predictions in DataFrame: prediction_df
prediction_df.to_csv(PATH_TO_PREDICTIONS) # Save prediction_df to csv
score = multi_log_loss.score_submission(pred_path=PATH_TO_PREDICTIONS, holdout_path=PATH_TO_HOLDOUT_LABELS, column_indices=BOX_PLOTS_COLUMN_INDICES) # Submit the predictions for scoring: score
# Print score
print('Your model yields logloss score: {}'.format(score))

print("****************************************************")
print("** END                                            **")
print("****************************************************")