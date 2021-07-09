# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 2: Loss functions
    In this chapter you will discover the conceptual framework behind logistic 
    regression and SVMs (Support Vector Machines). This will let you delve 
    deeper into the inner workings of these models.
Source: https://learn.datacamp.com/courses/linear-classifiers-in-python
More documentation: http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions #A function for plotting decision regions of classifiers in 1 or 2 dimensions.
from mlxtend.plotting import plot_learning_curves #A function to plot learning curves for classifiers. Learning curves are extremely useful to analyze if a model is suffering from over- or under-fitting (high variance or high bias).

import seaborn as sns

import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression

from scipy.optimize import minimize 

###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param = dict(color='darkblue', fontsize=10, weight='bold')
title_param    = {'color': 'darkred', 'fontsize': 12}
plot_param     = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                  'legend.fontsize': 8, 'font.size': 8}
figsize        = (12.1, 5.9)
SEED           = 42

# Global configuration
sns.set()
pd.set_option("display.max_columns",24)
plt.rcParams.update(**plot_param)
np.random.seed(SEED)

###############################################################################
## Reading the data
###############################################################################



###############################################################################
## Main part of the code
###############################################################################
def Linear_classifiers_the_coefficients(seed=SEED):
    print("****************************************************")
    topic = "1. Linear classifiers: the coefficients"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('------------------------------------------Operator @')
    x = np.arange(3)
    y = np.arange(3, 6)
    print('np.sum(x*y) = ', np.sum(x*y))
    print('x@y         = ', x@y)
    
    print('----------------Read breast cancer wisconsin dataset')
    cancer = sklearn.datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_scaled = preprocessing.scale(X) #To avoid warning "...scale the data..."
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)
    
    print('----------------------------------Making predictions')
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    print(f"10th element prediction: {y_pred[10]}, (real: {y_test[10]})")
    print(f"50th element prediction: {y_pred[50]}, (real: {y_test[50]})")
    
    print('----------------------------------How does lr do it?')
    print('Raw model [10]: ', lr.coef_ @ X_test[10] + lr.intercept_) # raw model output)
    print('Raw model [50]: ', lr.coef_ @ X_test[50] + lr.intercept_) # raw model output)
    
    print('-------------------------------Plotting the lr model')
    #Select best features to plot
    print(f"Features: \n{cancer.feature_names}")
    features = np.array(cancer.feature_names)
    selector = SelectKBest(k=2)
    selector.fit(X_scaled, y)
    features_index = selector.get_support(indices=True)
    x_label, y_label = features[features_index[0]], features[features_index[1]]
    X_new = selector.transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=.25)
    print(f"Features selected: [{x_label}, {y_label}]\n")
    
    # Fit the classifiers
    lr.fit(X_train, y_train)
        
    #Linear classifiers with only 2 features
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    ax = axes[0, 0]
    score = lr.score(X_test, y_test)
    plot_decision_regions(X_train, y_train, lr, legend=2, ax=ax)
    msg = f'score: {score:.4f}'
    t = ax.text(0.02, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'Original Model:\nY = {lr.coef_}·X + {lr.intercept_}', **title_param)
    
    ax = axes[0,1]
    lr.intercept_ = 3
    score = lr.score(X_test, y_test)
    plot_decision_regions(X_train, y_train, lr, legend=2, ax=ax)
    msg = f'score: {score:.4f}'
    t = ax.text(0.02, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'Modiffying the intercept:\nY = {lr.coef_}·X + {lr.intercept_}', **title_param)
    
    ax = axes[1,0]
    lr.coef_ = np.array([[-9, 0]])
    score = lr.score(X_test, y_test)
    plot_decision_regions(X_train, y_train, lr, legend=2, ax=ax)
    msg = f'score: {score:.4f}'
    t = ax.text(0.02, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'Modiffying the coefficient:\nY = {lr.coef_}·X + {lr.intercept_}', **title_param)
    
    ax = axes[1,1]
    ax.axis('off') 
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5); #To set the margins 
    plt.show()
    
    
    
def Changing_the_model_coefficients(seed=SEED):
    print("****************************************************")
    topic = "3. Changing the model coefficients"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('---------------------------------------Read the data')
    y = np.array([-1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1, -1])
    X = np.array([[ 1.78862847,  0.43650985], [ 0.09649747, -1.8634927 ],
                  [-0.2773882 , -0.35475898], [-3.08274148,  2.37299932],
                  [-3.04381817,  2.52278197], [-1.31386475,  0.88462238],
                  [-2.11868196,  4.70957306], [-2.94996636,  2.59532259],
                  [-3.54535995,  1.45352268], [ 0.98236743, -1.10106763],
                  [-1.18504653, -0.2056499 ], [-1.51385164,  3.23671627],
                  [-4.02378514,  2.2870068 ], [ 0.62524497, -0.16051336],
                  [-3.76883635,  2.76996928], [ 0.74505627,  1.97611078],
                  [-1.24412333, -0.62641691], [-0.80376609, -2.41908317],
                  [-0.92379202, -1.02387576], [ 1.12397796, -0.13191423]])
    
    print('----------------------------------LogisticRegression')
    lr = LogisticRegression()
    lr.fit(X, y)
    print("Coeficient: ", lr.coef_)
    print("Intercept: ", lr.intercept_)
    
    print('-------------------------------Plotting the lr model')
    coef = [np.array([[0,1]]), np.array([[1,0]]), np.array([[1,1]]), np.array([[-1,1]])]
    intercept = [0, 0, 0, -3]
    
    #Linear classifiers with only 2 features
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    for i, ax in enumerate(axes.flatten()):
        lr.coef_ = coef[i]
        lr.intercept_ = intercept[i]
        
        y_pred  = lr.predict(X)
        score   = lr.score(X, y)
        num_err = np.sum(y != y_pred)
        msg = f'score: {score:.2%}.\n' +\
              f'Number of errors: {num_err}.'
        
        plot_decision_regions(X, y, lr, legend=2, ax=ax)
        t = ax.text(0.05, 0.2, msg, transform=ax.transAxes, color='black', ha='left', va='top')
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
        
        ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2');
        ax.set_title(f'Y = {lr.coef_}·X + {lr.intercept_}', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5); #To set the margins 
    plt.show()
    
    
    
def What_is_a_loss_function(seed=SEED):
    print("****************************************************")
    topic = "4. What is a loss function?"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Minimizing a loss')
    x = np.linspace(-2, 2, 101)
    y = np.square(x)
    
    results = minimize(np.square, .001)#, tol=0.00001)
    print(f'X value found to minimize np.square(X) \n{results}')
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, color='red')
    ax.set_xlabel('x'); ax.set_ylabel('f(x)')
    ax.set_title(f'f(x) = x² \nMinimun value x={results.x}', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.1, right=None, top=.85, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    
    
    
def Minimizing_a_loss_function(seed=SEED):
    print("****************************************************")
    topic = "6. Minimizing a loss function"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------Boston housing price data set')
    boston = sklearn.datasets.load_boston()
    X, y = boston.data, boston.target
    
    print('-----------------------------------Minimizing a loss')
    # The squared error, summed over training examples
    def my_loss(w, X, y):
        """Return the loss value like sum((valor real - valor predict)²)"""
        return np.sum([(y[i] - w@X[i])**2 for i in range(y.size)])
    
    # Returns the w that makes my_loss(w) smallest
    w_fit = minimize(my_loss, X[0], args=(X, y))
    print(f'Found in "my_loss" function: \n{w_fit.x}')
    
    # Compare with scikit-learn's LinearRegression coefficients
    lr = LinearRegression(fit_intercept=False).fit(X,y)
    print(f'Found in the Linear Regression model: \n{lr.coef_}')
        
        
    
def Loss_function_diagrams(seed=SEED):
    print("****************************************************")
    topic = "7. Loss function diagrams"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('----------------------------Plotting Learning Curves')
    cancer = sklearn.datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_scaled = preprocessing.scale(X) #To avoid warning "...scale the data..."
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)
    
    lr = LogisticRegression()
    
    fig = plt.figure()
    plot_learning_curves(X_train, y_train, X_test, y_test, lr, style='fast')
    plt.title('Learning Curves\nLogisticRegression(Breast Cancer Dataset)', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.1, right=None, top=.85, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    
    
def Comparing_the_logistic_and_hinge_losses(seed=SEED):
    print("****************************************************")
    topic = "9. Comparing the logistic and hinge losses"; print("** %s" % topic)
    print("****************************************************")
    
    print('-----------------------------Plotting Loss Functions')
    # Mathematical functions for logistic and hinge losses
    def log_loss(raw_model_output):
        return np.log(1+np.exp(-raw_model_output))
    def hinge_loss(raw_model_output):
        return np.maximum(0,1-raw_model_output)
    
    # Create a grid of values and plot
    grid = np.linspace(-2,2,1000)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(grid, log_loss(grid), label='logistic')
    ax.plot(grid, hinge_loss(grid), label='hinge')
    ax.legend()
    ax.set_xlabel('Raw Model Output'); ax.set_ylabel('Loss')
    ax.set_title('Loss Functions', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.1, right=None, top=.85, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    
    
def Implementing_logistic_regression(seed=SEED):
    print("****************************************************")
    topic = "10. Implementing logistic regression"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------------------------Log loss')
    # Mathematical functions for logistic and hinge losses
    def log_loss(raw_model_output):
        return np.log(1+np.exp(-raw_model_output))
    
    print('----------------Read breast cancer wisconsin dataset')
    cancer = sklearn.datasets.load_breast_cancer()
    X, y = cancer.data[:,:10], cancer.target
    y = np.where(y>0, 1, -1)
    X = preprocessing.scale(X) #To avoid warning "...scale the data..."
    
    print('--------------------------------Finding coefficients')
    # The logistic loss, summed over training examples
    def my_loss(w, X, y):
        """Return the loss value like raw_model_output * y_real."""
        return np.sum([log_loss((w@X[i]) * y[i]) for i in range(y.size)])
    
    # Returns the w that makes my_loss(w) smallest
    w_fit = minimize(my_loss, X[0], args=(X, y)).x
    print(w_fit)
    
    # Compare with scikit-learn's LogisticRegression
    lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X,y)
    print(lr.coef_)
    
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Linear_classifiers_the_coefficients()
    Changing_the_model_coefficients()
    What_is_a_loss_function()
    Minimizing_a_loss_function()
    Loss_function_diagrams()
    Comparing_the_logistic_and_hinge_losses()
    Implementing_logistic_regression()

    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})