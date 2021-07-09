# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Support Vector Machines
    In this chapter you will learn all about the details of support vector 
    machines. You'll learn about tuning hyperparameters for these models and 
    using kernels to fit non-linear decision boundaries.
Source: https://learn.datacamp.com/courses/linear-classifiers-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from plot_classifier_cbar import plot_classifier

import sklearn.datasets
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                  'legend.fontsize': 8, 'font.size': 8}
figsize        = (12.1, 5.9)
SEED           = 42
SIZE           = 10000

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
def Support_vectors(seed=SEED):
    print("****************************************************")
    topic = "1. Support vectors"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('---------------------------------------Read the data')
    X, y = sklearn.datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_classes=2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 3.5))
    print('------------------------------------------Linear SVM')
    svm = SVC().fit(X, y)
    ax = axes[0]
    plot_classifier(X, y, svm, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('SVC', **title_param)
    
    print('------------------------------------------Linear SVC')
    svm = LinearSVC().fit(X, y)
    ax = axes[1]
    plot_classifier(X, y, svm, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('LinearSVC', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def Effect_of_removing_examples(seed=SEED):
    print("****************************************************")
    topic = "3. Effect of removing examples"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('------------------------------------Read data: wines')
    wine = sklearn.datasets.load_wine()
    X, y = wine.data[:,:2], wine.target
    xlabel, ylabel = wine.feature_names[:2]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 3.5))
    print('--------------------------------SVC(kernel="linear")')
    # Train a linear SVM
    svm = SVC(kernel="linear")
    svm.fit(X, y)
    # Make a new data set keeping only the support vectors
    print("Number of original examples", len(X))
    msg = 'Score: {}'.format(svm.score(X_test, y_test))
    
    ax = axes[0]
    plot_decision_regions(X_train, y_train, svm, ax=ax)
    t = ax.text(0.1, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlim(11,15); ax.set_ylim( 0, 6);
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title('SVC(kernel="linear")', **title_param)
        
    
    
    print('----SVC(kernel="linear") - with only support vectors')
    print("Number of support vectors", len(svm.support_))
    X_small = X[svm.support_]
    y_small = y[svm.support_]
    
    # Train a new SVM using only the support vectors
    svm_small = SVC(kernel="linear")
    svm_small.fit(X_small, y_small)
    msg = 'Score: {}'.format(svm.score(X_test, y_test))
    
    ax = axes[1]
    plot_decision_regions(X_small, y_small, svm_small, ax=ax)
    t = ax.text(0.1, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlim(11,15); ax.set_ylim( 0, 6);
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title('with only support vectors', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def Kernel_SVMs(seed=SEED):
    print("****************************************************")
    topic = "4. Kernel SVMs"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('---------------------------------------Read the data')
    #X, y = sklearn.datasets.make_gaussian_quantiles(n_samples=40, n_features=2, n_classes=3)
    X, y = sklearn.datasets.make_circles(n_samples=40, noise=0.25, factor=0.25, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, stratify=y)
    
    print('-----------------------------------------SVC (gamma)')
    gamma = [1, 0.01, 10]
    
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 3.5))
    for ax, g in zip(axes.flat, gamma):
        svm = SVC(gamma=g).fit(X_train, y_train)
        
        plot_classifier(X, y, svm, ax)
        msg = 'Score: {}'.format(svm.score(X_test, y_test))
        
        t = ax.text(0.1, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='top')
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    
        ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
        ax.set_title(f'SVC (gamma={g})', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def GridSearchCV_warmup(seed=SEED):
    print("****************************************************")
    topic = "5. GridSearchCV warm-up"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Read data: digits')
    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, stratify=y)
    
    print('-------------------------------------------------SVC')
    # Instantiate an RBF SVM
    svm = SVC()
    
    # Instantiate the GridSearchCV object and run the search
    parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]}
    searcher = GridSearchCV(svm, param_grid=parameters)
    searcher.fit(X_train, y_train)
    
    # Report the best parameters
    print("Best CV params", searcher.best_params_)
    
    print('------------------------------------Reviewing scores')
    for gamma in parameters['gamma']:
        svm = SVC(gamma=gamma).fit(X_train, y_train)
        print('Score for gamma={}: {}'.format(gamma, svm.score(X_test, y_test)))
    
    
    
def Jointly_tuning_gamma_and_C_with_GridSearchCV(seed=SEED):
    print("****************************************************")
    topic = "6. Jointly tuning gamma and C with GridSearchCV"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Read data: digits')
    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, stratify=y)
    
    print('-------------------------------------------------SVC')
    # Instantiate an RBF SVM
    svm = SVC()
    
    # Instantiate the GridSearchCV object and run the search
    parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
    searcher = GridSearchCV(svm, param_grid=parameters)
    searcher.fit(X_train, y_train)
    
    # Report the best parameters and the corresponding score
    print("Best CV params", searcher.best_params_)
    print("Best CV accuracy", searcher.best_score_)
    
    # Report the test accuracy using these best parameters
    print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
    
    
    
def Using_SGDClassifier(seed=SEED):
    print("****************************************************")
    topic = "10. Using SGDClassifier"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Read data: digits')
    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y)
    
    print('---------------------------------------SGDClassifier')
    # We set random_state=0 for reproducibility 
    sgdc = SGDClassifier(random_state=0, max_iter=2500)
    
    # Instantiate the GridSearchCV object and run the search
    parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
                  'loss':['hinge','log'], 'penalty':['l1','l2']}
    searcher = GridSearchCV(sgdc, parameters, cv=10)
    searcher.fit(X_train, y_train)
    
    # Report the best parameters and the corresponding score
    print("Best CV params", searcher.best_params_)
    print("Best CV accuracy", searcher.best_score_)
    print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
    
    
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Support_vectors()
    Effect_of_removing_examples()
    Kernel_SVMs()
    GridSearchCV_warmup()
    Jointly_tuning_gamma_and_C_with_GridSearchCV()
    Using_SGDClassifier()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})