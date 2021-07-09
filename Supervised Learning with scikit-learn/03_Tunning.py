# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:07:35 2019

@author: jacqueline.cortez

Capítulo 3. Fine-tuning your model
Introduction:
    Having trained your model, your next task is to evaluate its performance. What metrics can you use to gauge how good your model is? 
    So far, you have used accuracy for classification and R-squared for regression. In this chapter, you will learn about some of the 
    other metrics available in scikit-learn that will allow you to assess your model's performance in a more nuanced manner. You will 
    then learn to optimize both your classification as well as regression models using hyperparameter tuning.
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
#from sklearn import datasets                                                        #For learning machine
from sklearn.neighbors import KNeighborsClassifier                                  #For learning machine
from sklearn.model_selection import train_test_split, cross_val_score               #For learning machine
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV                #For learning machine
#from sklearn.linear_model import LinearRegression, Lasso                            #For learning machine
from sklearn.linear_model import Ridge                                              #For learning machine
from sklearn.linear_model import LogisticRegression, ElasticNet                     #For learning machine
from sklearn.metrics import mean_squared_error                                      #For learning machine
from sklearn.metrics import confusion_matrix, classification_report                 #For learning machine
from sklearn.metrics import roc_curve, roc_auc_score                                #For learning machine
from scipy.stats import randint                                                     #For learning machine 
from sklearn.tree import DecisionTreeClassifier                                     #For learning machine

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


print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

file = "diabetes.csv"
diabetes = pd.read_csv(file)
insulin_avg = diabetes[diabetes.insulin!=0].insulin.mean()
bmi_avg = diabetes[diabetes.bmi!=0].bmi.mean()
triceps_avg = diabetes[diabetes.triceps!=0].triceps.mean()
diabetes.insulin.replace(0, insulin_avg, inplace=True) 
diabetes.bmi.replace(0, bmi_avg, inplace=True)            
diabetes.triceps.replace(0, triceps_avg, inplace=True)            

file = "gm_2008_region.csv"
gapminder = pd.read_csv(file)

print("****************************************************")
tema = '2. Metrics for classification'; print("** %s\n" % tema)

#np.random.seed(42)

y = diabetes.diabetes # Create arrays for features and target variable
X = diabetes.drop(['diabetes'], axis=1) #X = df.drop('life', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) # Create training and test set
knn = KNeighborsClassifier(n_neighbors=6) # Instantiate a k-NN classifier: knn
knn.fit(X_train, y_train) # Fit the classifier to the training data
y_pred = knn.predict(X_test) # Predict the labels of the test data: y_pred

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



print("****************************************************")
tema = '4. Building a logistic regression model'; print("** %s\n" % tema)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42) # Create training and test sets
logreg = LogisticRegression(solver='liblinear') # Create the classifier: logreg
logreg.fit(X_train, y_train) # Fit the classifier to the training data
y_pred = logreg.predict(X_test) # Predict the labels of the test set: y_pred

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



print("****************************************************")
tema = "5. Plotting an ROC curve"; print("** %s\n" % tema)

y_pred_prob = logreg.predict_proba(X_test)[:,1] # Compute predicted probabilities: y_pred_prob
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) # Generate ROC curve values: fpr, tpr, thresholds

# Plot ROC curve
sns.set() # Set default Seaborn style
#plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = "8. AUC computation"; print("** %s\n" % tema)

np.random.seed(42)

print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob))) # Compute and print AUC score
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc') # Compute cross-validated AUC scores: cv_auc

print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc)) # Print list of AUC scores
print("AUC mean scores from 5-fold cross-validation: {}".format(cv_auc.mean()))



print("****************************************************")
tema = "10. Hyperparameter tuning with GridSearchCV"; print("** %s\n" % tema)

c_space = np.logspace(-5, 8, 15) # Setup the hyperparameter grid
param_grid = {'C': c_space}
logreg = LogisticRegression(solver='liblinear') # Instantiate a logistic regression classifier: logreg
#np.random.seed(42)
logreg_cv = GridSearchCV(logreg, param_grid, cv=5) # Instantiate the GridSearchCV object: logreg_cv
logreg_cv.fit(X, y) # Fit it to the data

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))



print("****************************************************")
tema = "11. Hyperparameter tuning with GridSearchCV"; print("** %s\n" % tema)


param_dist = {"max_depth": [3, None], "max_features": randint(1, 9), "min_samples_leaf": randint(1, 9), "criterion": ["gini", "entropy"]} # Setup the parameters and distributions to sample from: param_dist
tree = DecisionTreeClassifier() # Instantiate a Decision Tree classifier: tree
#np.random.seed(42)
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5) # Instantiate the RandomizedSearchCV object: tree_cv
tree_cv.fit(X, y) # Fit it to the data

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

ridge = Ridge(alpha=0.5, normalize=True)


print("****************************************************")
tema = "14. Hold-out set in practice I: Classification"; print("** %s\n" % tema)

c_space = np.logspace(-5, 8, 15) # Create the hyperparameter grid
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
logreg = LogisticRegression(solver='liblinear') # Instantiate the logistic regression classifier: logreg
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) # Create train and test sets
logreg_cv = GridSearchCV(logreg, param_grid, cv=5, iid=True) # Instantiate the GridSearchCV object: logreg_cv
logreg_cv.fit(X_train, y_train) # Fit it to the training data

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))




print("****************************************************")
tema = "15. Hold-out set in practice II: Regression"; print("** %s\n" % tema)

warnings.filterwarnings('ignore', 'Objective did not converge*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394

y = gapminder.life.values # Create arrays for features and target variable
X = gapminder.drop(['life', 'Region'], axis=1).values #X = df.drop('life', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) # Create train and test sets
l1_space = np.linspace(0, 1, 30) # Create the hyperparameter grid
param_grid = {'l1_ratio': l1_space}

elastic_net = ElasticNet(tol=0.0001, max_iter=1000) # Instantiate the ElasticNet regressor: elastic_net

gm_cv = GridSearchCV(elastic_net, param_grid, cv=5, iid=True) # Setup the GridSearchCV object: gm_cv

gm_cv.fit(X_train, y_train) # Fit it to the training data

y_pred = gm_cv.predict(X_test) # Predict on the test set and compute metrics
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)

print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))

warnings.filterwarnings('default', 'Objective did not converge*') #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


print("****************************************************")
print("** END                                            **")
print("****************************************************")