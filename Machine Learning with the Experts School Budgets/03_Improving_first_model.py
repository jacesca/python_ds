# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:02:48 2019

@author: jacqueline.cortez

Capítulo 3. Improving your model
Introduction:
    Here, you'll improve on your benchmark model using pipelines. Because the budget consists of both text and numeric data, you'll learn to 
    how build pipielines that process multiple types of data. You'll also explore how the flexibility of the pipeline workflow makes testing 
    different approaches efficient, even in complicated problems like this one!
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

from sklearn.ensemble import RandomForestClassifier                                 #For learning machine
from sklearn.feature_extraction.text import CountVectorizer                         #For learning machine
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
from sklearn.model_selection import train_test_split                                #For learning machine
from sklearn.multiclass import OneVsRestClassifier                                   #For learning machine
from sklearn.neighbors import KNeighborsClassifier                                 #For learning machine
from sklearn.pipeline import FeatureUnion                                           #For learning machine
from sklearn.pipeline import Pipeline                                               #For learning machine
from sklearn.preprocessing import FunctionTransformer                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
#from sklearn.preprocessing import scale                                             #For learning machine
#from sklearn.preprocessing import StandardScaler                                    #For learning machine
#from sklearn.svm import SVC                                                         #For learning machine
#from sklearn.tree import DecisionTreeClassifier                                     #For learning machine
import multilabel                                                                   #For multivariable target, function created by Datacamp
#import multi_log_loss                                                               #Datacamp logloss for multiple targets score


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
TOKENS_BASIC             = '\\S+(?=\\s+)' # Create the basic token pattern

file                     = "TrainingSetSample.csv" 
budget_df                = pd.read_csv(file, index_col=0)
categorize_label         = lambda x: x.astype('category') # Define the lambda function: categorize_label
budget_df[LABELS]        = budget_df[LABELS].apply(categorize_label, axis=0) # Convert df[LABELS] to a categorical type

NON_LABELS               = [c for c in budget_df.columns if c not in LABELS] # Get the columns that are features in the original df

# Load the holdout data: holdout
file               = "TestSetSample.csv"
holdout            = pd.read_csv(file, index_col=0)



# Creating the sample dataframe
rng                = np.random.RandomState(123)
SIZE               = 1000
sample_data        = {'numeric': rng.normal(0, 10, size=SIZE),
                      'text': rng.choice(['', 'foo', 'bar', 'foo bar', 'bar foo'], size=SIZE),
                      'with_missing': rng.normal(loc=3, size=SIZE)}
sample_df          = pd.DataFrame(sample_data)
sample_df.loc[rng.choice(sample_df.index, size=np.floor_divide(sample_df.shape[0], 5)), 'with_missing'] = np.nan
foo_values         = sample_df.text.str.contains('foo') * 10
bar_values         = sample_df.text.str.contains('bar') * -25
no_text            = ((foo_values + bar_values) == 0) * 1
val                = 2 * sample_df.numeric + -2 * (foo_values + bar_values + no_text) + 4 * sample_df.with_missing.fillna(3)
val               += rng.normal(0, 8, size=SIZE)
sample_df['label'] = np.where(val > np.median(val), 'a', 'b')


print("****************************************************")
tema = '2. Instantiate pipeline'; print("** %s\n" % tema)

print("Head of data:\n{}".format(sample_df.head()))
print("\nInfo related to the columns: "); print(sample_df.info())
print("\nSize of data:\n{}".format(sample_df.shape))


# Split and select numeric data only, no nans 
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']], pd.get_dummies(sample_df['label']), random_state=22)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
    ])

pl.fit(X_train, y_train) # Fit the pipeline to the training data
accuracy = pl.score(X_test, y_test) # Compute and print accuracy

print("\nAccuracy on sample data - numeric, no nans: {}\n".format(accuracy))



print("****************************************************")
tema = '3. Preprocessing numeric features'; print("** %s\n" % tema)

# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']], pd.get_dummies(sample_df['label']), random_state=456)

# Insantiate Pipeline object: pl
pl = Pipeline([
        ('imp', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
    ])

pl.fit(X_train, y_train) # Fit the pipeline to the training data
accuracy = pl.score(X_test, y_test) # Compute and print accuracy

print("Accuracy on sample data - all numeric, incl nans: {}\n".format(accuracy))


print("****************************************************")
tema = "5. Preprocessing text features"; print("** %s\n" % tema)

# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df.text, pd.get_dummies(sample_df['label']), random_state=456)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
    ])

pl.fit(X_train, y_train) # Fit to the training data
accuracy = pl.score(X_test, y_test) # Compute and print accuracy

print("\nAccuracy on sample data - just text data: ", accuracy)


print("****************************************************")
tema = "6. Multiple types of processing: FunctionTransformer"; print("** %s\n" % tema)

get_text_data = FunctionTransformer(lambda x: x['text'], validate=False) # Obtain the text data: get_text_data
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False) # Obtain the numeric data: get_numeric_data

just_text_data = get_text_data.fit_transform(sample_df) # Fit and transform the text data: just_text_data
just_numeric_data = get_numeric_data.fit_transform(sample_df) # Fit and transform the numeric data: just_numeric_data


# Print head to check results
print('Text Data'); print(just_text_data.head());
print('\nNumeric Data'); print(just_numeric_data.head());



print("****************************************************")
tema = "7. Multiple types of processing: FunctionTransformer"; print("** %s\n" % tema)


# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']], pd.get_dummies(sample_df['label']), random_state=22)

# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(transformer_list = [('numeric_features', Pipeline([('selector', get_numeric_data), 
                                                                                            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))])),
                                                             ('text_features',    Pipeline([('selector', get_text_data),
                                                                                            ('vectorizer', CountVectorizer())]))])
# Instantiate nested pipeline: pl
pl = Pipeline([('union', process_and_join_features),
               ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))])

# Fit pl to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("Accuracy on sample data - all data: {}\n".format(accuracy))



print("****************************************************")
tema = "9. Using FunctionTransformer on the main dataset"; print("** %s\n" % tema)

dummy_labels = pd.get_dummies(budget_df[LABELS], prefix_sep='__') # Get labels and convert to dummy variables: label_dummies

warnings.filterwarnings('ignore', 'Size less than number of columns*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
X_train, X_test, y_train, y_test = multilabel.multilabel_train_test_split(budget_df[NON_LABELS], dummy_labels,  size=0.2, seed=123) # Create training and test sets
warnings.filterwarnings('default', 'Size less than number of columns*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394

#combined_text_columns_with_params = combine_text_columns(to_drop=NUMERIC_COLUMNS+LABELS)

get_text_data = FunctionTransformer(combine_text_columns, validate=False) # Preprocess the text data: get_text_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False) # Preprocess the numeric data: get_numeric_data



print("****************************************************")
tema = "10. Add a model to the pipeline"; print("** %s\n" % tema)

# OneVsRestClassifier
pl = Pipeline([('union', FeatureUnion(transformer_list = [('numeric_features', Pipeline([('selector', get_numeric_data),
                                                                                         ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))])),
                                                          ('text_features',    Pipeline([('selectortxt', get_text_data),
                                                                                         ('vectorizer', CountVectorizer())]))])),
               ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))])

pl.fit(X_train, y_train) # Fit to the training data
accuracy = pl.score(X_test, y_test) # Compute and print accuracy
print("Accuracy on budget dataset for OneVsRestClassifier: ", accuracy)



# RandomForestClassifier
pl = Pipeline([('union', FeatureUnion(transformer_list = [('numeric_features', Pipeline([('selector', get_numeric_data),
                                                                                         ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))])),
                                                          ('text_features',    Pipeline([('selectortxt', get_text_data),
                                                                                         ('vectorizer', CountVectorizer())]))])),
               ('clf', RandomForestClassifier(n_estimators=10, random_state=123))])

pl.fit(X_train, y_train) # Fit to the training data
accuracy = pl.score(X_test, y_test) # Compute and print accuracy
print("Accuracy on budget dataset for RandomForestClassifier: ", accuracy)



# KNeighborsClassifier
pl = Pipeline([('union', FeatureUnion(transformer_list = [('numeric_features', Pipeline([('selector', get_numeric_data),
                                                                                         ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))])),
                                                          ('text_features',    Pipeline([('selectortxt', get_text_data),
                                                                                         ('vectorizer', CountVectorizer())]))])),
               ('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7)))])

pl.fit(X_train, y_train) # Fit to the training data
accuracy = pl.score(X_test, y_test) # Compute and print accuracy
print("Accuracy on budget dataset for KNeighborsClassifier: ", accuracy)



print("****************************************************")
tema = "12. Can you adjust the model or parameters to improve accuracy?"; print("** %s\n" % tema)

# Add model step to pipeline: pl
pl = Pipeline([('union', FeatureUnion(transformer_list = [('numeric_features', Pipeline([('selector', get_numeric_data),
                                                                                         ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))])),
                                                          ('text_features',    Pipeline([('selectortxt', get_text_data),
                                                                                         ('vectorizer', CountVectorizer())]))])),
               ('clf', RandomForestClassifier(n_estimators=15, random_state=123))])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("Accuracy on budget dataset: ", accuracy)

#predictions = pl.predict_proba(holdout[NON_LABELS])
#prediction_df = pd.DataFrame(columns=dummy_labels.columns, index=holdout.index, data=predictions) # Format predictions in DataFrame: prediction_df
#prediction_df.to_csv(PATH_TO_PREDICTIONS) # Save prediction_df to csv
#score = multi_log_loss.score_submission(pred_path=PATH_TO_PREDICTIONS, holdout_path=PATH_TO_HOLDOUT_LABELS, column_indices=BOX_PLOTS_COLUMN_INDICES) # Submit the predictions for scoring: score

# Print score
#print('\nYour model, trained with numeric data only, yields logloss score: {}'.format(score))

print("****************************************************")
print("** END                                            **")
print("****************************************************")