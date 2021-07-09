# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 22:51:50 2019

@author: jacqueline.cortez

Capítulo 2. Regression
Introduction:
    In the previous chapter, you made use of image and political datasets to predict binary as well as multiclass outcomes. 
    But what if your problem requires a continuous outcome? Regression, which is the focus of this chapter, is best suited to 
    solving such problems. You will learn about fundamental concepts in regression and apply them to predict the life expectancy 
    in a given country using Gapminder data.
"""

# Import packages
import pandas as pd                   #For loading tabular data
import numpy as np                    #For making operations in lists
#import matplotlib as mpl              #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt       #For creating charts
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
import timeit                                                                       #For Measure execution time of small code snippets
import time                                                                         #To measure the elapsed wall-clock time between two points

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions
#from sklearn import datasets                                                        #For learning machine
#from sklearn.neighbors import KNeighborsClassifier                                  #For learning machine
from sklearn.linear_model import LinearRegression, Lasso, Ridge                     #For learning machine
from sklearn.metrics import mean_squared_error                                      #For learning machine
from sklearn.model_selection import train_test_split, cross_val_score               #For learning machine

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

def display_plot(alpha_space, cv_scores, cv_scores_std):
    sns.set() # Set default Seaborn style
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.title('Cross Validation Score for different alphas in Rigid Regression')
    plt.suptitle(tema)
    plt.subplots_adjust(left=0.15, bottom=0.3, right=None, top=0.85, wspace=None, hspace=None)
    plt.show()
    plt.style.use('default')

    
file = "gm_2008_region.csv"
gapminder = pd.read_csv(file)



print("****************************************************")
tema = '3. Importing data for supervised learning'; print("** %s\n" % tema)


y = gapminder.life.values # Create arrays for features and target variable
X = gapminder.drop(['life', 'Region'], axis=1).values #X = df.drop('life', axis=1).values
X_fertility = gapminder.fertility.values #X = df.drop('life', axis=1).values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X_fertility.shape))

# Reshape X and y
y = y.reshape(-1, 1)
X_fertility = X_fertility.reshape(-1, 1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X_fertility.shape))



print("****************************************************")
tema = '4. Importing data for supervised learning'; print("** %s\n" % tema)

sns.heatmap(gapminder.corr(), square=True, cmap='RdYlGn')
plt.xlabel('Features')
plt.ylabel('Features')
plt.title("Gapminder's Heatmap")
plt.suptitle(tema)
plt.subplots_adjust(left=0.10, bottom=0.30, right=None, top=0.85, wspace=None, hspace=None)
plt.show()


print("****************************************************")
tema = '6. Fit & predict for regression'; print("** %s\n" % tema)

np.random.seed(42)

reg = LinearRegression() # Create the regressor: reg
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1) # Create the prediction space
reg.fit(X_fertility, y) # Fit the model to the data
y_pred = reg.predict(prediction_space) # Compute predictions over the prediction space: y_pred
print(reg.score(X_fertility, y)) # Print R^2 


# Plot regression line
sns.set() # Set default Seaborn style
plt.figure()
plt.scatter(X_fertility, y, color='blue')
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.xlabel('Fertility')
plt.ylabel('Life Expectancy')
plt.title("Life's Expectancy Prediccion Model")
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = '7. Train/test split for regression'; print("** %s\n" % tema)

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) # Create training and test sets
reg_all = LinearRegression() # Create the regressor: reg_all
reg_all.fit(X_train, y_train) # Fit the regressor to the training data
y_pred = reg_all.predict(X_test) # Predict on the test data: y_pred

print("R^2: {}".format(reg_all.score(X_test, y_test))) # Compute and print R^2 and RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


print("****************************************************")
tema = '9. 5-fold cross-validation'; print("** %s\n" % tema)

np.random.seed(42)

reg = LinearRegression() # Create a linear regression object: reg
cv_scores = cross_val_score(reg, X, y, cv=5) # Compute 5-fold cross-validation scores: cv_scores

print(cv_scores) # Print the 5-fold cross-validation scores
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))



print("****************************************************")
tema = '10. K-Fold CV comparison'; print("** %s\n" % tema)

np.random.seed(42)
reg = LinearRegression() # Create a linear regression object: reg
start_CPU, start_time = timeit.timeit(), time.time()
cvscores_3 = cross_val_score(reg, X, y, cv=3) # Perform 3-fold CV
end_CPU, end_time = timeit.timeit(), time.time()
print('Score: ', np.mean(cvscores_3))
print('CPU process consumed: ', start_CPU - end_CPU)
print('Elapsed time: ', end_time - start_time)
#%timeit cross_val_score(reg, X, y, cv = 3)

#Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
end_CPU, end_time = timeit.timeit(), time.time()
end = timeit.timeit()
print('Score: ', np.mean(cvscores_10))
print('CPU process consumed: ', start_CPU - end_CPU)
print('Elapsed time: ', end_time - start_time)
#%timeit cross_val_score(reg, X, y, cv = 10)


print("****************************************************")
tema = '12. Regularization I: Lasso'; print("** %s\n" % tema)

np.random.seed(42)
gapminder_columns = gapminder.drop(['life', 'Region'], axis=1).columns 
lasso = Lasso(alpha=0.4, normalize=True) # Instantiate a lasso regressor: lasso
lasso.fit(X, y) # Fit the regressor to the data
lasso_coef = lasso.fit(X, y).coef_ # Compute and print the coefficients
print("Lasso coef: ", lasso_coef)

# Plot the coefficients
sns.set() # Set default Seaborn style
plt.figure()
plt.plot(range(len(gapminder_columns)), lasso_coef)
plt.xticks(range(len(gapminder_columns)), gapminder_columns.values, rotation=60)
plt.margins(0.02)
plt.xlabel('Feature')
plt.ylabel('Lasso Regression Coef')
plt.title("Features evaluated for Life's Expectancy")
plt.suptitle(tema)
plt.subplots_adjust(left=0.15, bottom=0.3, right=None, top=0.85, wspace=None, hspace=None)
plt.show()
plt.style.use('default')



print("****************************************************")
tema = '13. Regularization II: Ridge'; print("** %s\n" % tema)

"""
numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)[source]
Return numbers spaced evenly on a log scale.
Example:
    >>> np.logspace(2.0, 3.0, num=4)
    array([  100.        ,   215.443469  ,   464.15888336,  1000.        ])
"""

# Setup the array of alphas and lists to store scores
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

# Display the plot
display_plot(alpha_space, ridge_scores, ridge_scores_std)



print("****************************************************")
print("** END                                            **")
print("****************************************************")