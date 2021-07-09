# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 01:48:21 2019

@author: jacqueline.cortez

Capítulo 4. Preprocessing and pipelines
Introduction:
    This chapter will introduce the notion of pipelines and how scikit-learn allows for transformers and estimators to be 
    chained together and used as a single unit. Pre-processing techniques will be then be introduced as a way to enhance model 
    performance and pipelines will be the glue that ties together concepts in the prior chapters.
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
import warnings

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions
#from scipy.stats import randint                                                     #For learning machine 

#from sklearn import datasets                                                        #For learning machine
from sklearn.impute import SimpleImputer                                            #For learning machine
from sklearn.linear_model import ElasticNet                                         #For learning machine
#from sklearn.linear_model import Lasso                                              #For learning machine
#from sklearn.linear_model import LinearRegression                                   #For learning machine
#from sklearn.linear_model import LogisticRegression                                 #For learning machine
from sklearn.linear_model import Ridge                                              #For learning machine
#from sklearb.metrics import accuracy_score                                          #For learning machine
from sklearn.metrics import classification_report                                   #For learning machine
#from sklearn.metrics import confusion_matrix                                        #For learning machine
#from sklearn.metrics import mean_squared_error                                      #For learning machine
#from sklearn.metrics import roc_auc_score                                           #For learning machine
#from sklearn.metrics import roc_curve                                               #For learning machine
from sklearn.model_selection import cross_val_score                                 #For learning machine
from sklearn.model_selection import GridSearchCV                                    #For learning machine
#from sklearn.model_selection import RandomizedSearchCV                              #For learning machine
from sklearn.model_selection import train_test_split                                #For learning machine
from sklearn.neighbors import KNeighborsClassifier                                  #For learning machine
from sklearn.pipeline import Pipeline                                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
from sklearn.preprocessing import scale                                             #For learning machine
from sklearn.preprocessing import StandardScaler                                    #For learning machine
from sklearn.svm import SVC                                                         #For learning machine
#from sklearn.tree import DecisionTreeClassifier                                     #For learning machine


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

def display_plot(alpha_space, cv_scores, cv_scores_std, best_alpha):
    sns.set() # Set default Seaborn style
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha\nBest parameter: {} (in red)'.format(best_alpha))
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.axvline(best_alpha, color='red', linestyle='solid', linewidth=1)
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.title('Cross Validation Score for different alphas in Rigid Regression')
    plt.suptitle(tema)
    plt.subplots_adjust(left=0.15, bottom=0.3, right=None, top=0.85, wspace=None, hspace=None)
    plt.show()
    plt.style.use('default')



file = "gm_2008_region.csv"
gapminder = pd.read_csv(file)

file = "house-votes-84.csv" 
vote_df = pd.read_csv(file, header = None,
                      names = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
                                 'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
                                 'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])
vote_df.replace('n',0, inplace=True)
vote_df.replace('y',1, inplace=True)

file = "white-wine.csv"
white_wine = pd.read_csv(file)

file = "gm_2008_region.csv"
gapminder = pd.read_csv(file)

print("****************************************************")
tema = '2. Exploring categorical features'; print("** %s\n" % tema)

sns.set() # Set default Seaborn style
sns.set(font_scale=0.8)
gapminder.boxplot('life', 'Region', rot=60) # Create a boxplot of life expectancy per region
plt.xlabel('Life Expectancy')
plt.ylabel('Region')
plt.title('Boxplot grouped by Region')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.40, right=None, top=0.87, wspace=None, hspace=None)
plt.show() # Show the plot
plt.style.use('default')



print("****************************************************")
tema = '3. Creating dummy variables'; print("** %s\n" % tema)

# Create dummy variables: df_region
gapminder_region = pd.get_dummies(gapminder)

# Print the columns of df_region
print(gapminder_region.columns)

# Create dummy variables with drop_first=True: df_region
gapminder_region = pd.get_dummies(gapminder, drop_first=True)

# Print the new columns of df_region
print(gapminder_region.columns)



print("****************************************************")
tema = '4. Regression with categorical features'; print("** %s\n" % tema)

#############################################################
# Getting the target y and the features X
#############################################################
y = gapminder_region.life.values # Create arrays for features and target variable
X = gapminder_region.drop(['life'], axis=1).values #X = df.drop('life', axis=1).values


# Analyzing the behiver with different alphas.
alpha_space = np.logspace(-4, 0, 50)
print('alpha_space: ', alpha_space)
ridge_scores = []
ridge_scores_std = []

ridge = Ridge(normalize=True) # Create a ridge regressor: ridge
for alpha in alpha_space: # Compute scores over range of alphas
    ridge.alpha = alpha # Specify the alpha value to use: ridge.alpha
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10) # Perform 10-fold CV: ridge_cv_scores
    ridge_scores.append(np.mean(ridge_cv_scores)) # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores_std.append(np.std(ridge_cv_scores)) # Append the std of ridge_cv_scores to ridge_scores_std

#############################################################
# Getting the best parameter
#############################################################
param_grid = {'alpha': alpha_space}
ridge = Ridge(normalize=True)
ridge_cv = GridSearchCV(ridge, param_grid, cv=5, iid=True) # Instantiate the GridSearchCV object: logreg_cv
ridge_cv.fit(X, y) # Fit it to the data

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(ridge_cv.best_params_)) 
print("Best score is {}".format(ridge_cv.best_score_))
best_alpha = ridge_cv.best_params_['alpha']

# Display the plot
display_plot(alpha_space, ridge_scores, ridge_scores_std, best_alpha)




#############################################################
# Back to the exercise
#############################################################
# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print("Scores: {}".format(ridge_cv))
print(ridge_cv.mean())


#############################################################
# Trying again with the best parameter
#############################################################
# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=best_alpha, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print("Scores: {}".format(ridge_cv))
print(ridge_cv.mean())



print("****************************************************")
tema = "6. Dropping missing data"; print("** %s\n" % tema)

# Convert '?' to NaN
vote_df[vote_df  == '?'] = np.nan

# Print the number of NaNs
print(vote_df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(vote_df.shape))

# Drop missing values and print shape of new DataFrame
vote_df_dropped  = vote_df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(vote_df_dropped.shape))



print("****************************************************")
tema = "7. Imputing missing data in a ML Pipeline I"; print("** %s\n" % tema)

y = vote_df['party'].values # Create arrays for the features and the response variable
X = vote_df.drop('party', axis=1).values

# Setup the Imputation transformer: imp
#imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0) #other strategy: mean, median, most_frequent
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') #other strategy: mean, median, most_frequent, constant

# Instantiate the SVC classifier: clf
clf = SVC(gamma='auto')

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]


print("****************************************************")
tema = "8. Imputing missing data in a ML Pipeline II"; print("** %s\n" % tema)

# Setup the pipeline steps: steps
#steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
#        ('SVM', SVC())]

pipeline = Pipeline(steps) # Create the pipeline: pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Create training and test sets
pipeline.fit(X_train, y_train) # Fit the pipeline to the train set
y_pred = pipeline.predict(X_test) # Predict the labels of the test set

print(classification_report(y_test, y_pred)) # Compute metrics



print("****************************************************")
tema = "10. Centering and scaling your data"; print("** %s\n" % tema)

y = white_wine.quality.values <= 5 # Create arrays for the features and the response variable
X = white_wine.drop('quality', axis=1).values

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))


print("****************************************************")
tema = "11. Centering and scaling in a pipeline"; print("** %s\n" % tema)

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
pipeline = Pipeline(steps) # Create the pipeline: pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Create train and test sets

knn_scaled = pipeline.fit(X_train, y_train) # Fit the pipeline to the training set: knn_scaled
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train) # Instantiate and fit a k-NN classifier to the unscaled data

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))


print("****************************************************")
tema = "12. Bringing it all together I: Pipeline for classification"; print("** %s\n" % tema)

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


print("****************************************************")
tema = "13. Bringing it all together II: Pipeline for regression"; print("** %s\n" % tema)

y = gapminder.life.values # Create arrays for features and target variable
X = gapminder.drop(['life', 'Region'], axis=1).values #X = df.drop('life', axis=1).values

# Setup the pipeline steps: steps
steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='mean')),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

pipeline = Pipeline(steps) # Create the pipeline: pipeline 

parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)} # Specify the hyperparameter space

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) # Create train and test sets

gm_cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, iid=True) # Create the GridSearchCV object: gm_cv
warnings.filterwarnings('ignore', 'Objective did not converge*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
gm_cv.fit(X_train, y_train) # Fit to the training set
warnings.filterwarnings('default', 'Objective did not converge*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394

r2 = gm_cv.score(X_test, y_test) # Compute and print the metrics

print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))



print("****************************************************")
print("** END                                            **")
print("****************************************************")