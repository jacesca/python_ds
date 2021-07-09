# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: Logistic regression
    In this chapter you will delve into the details of logistic regression. 
    You'll learn all about regularization and how to interpret model output.
Source: https://learn.datacamp.com/courses/linear-classifiers-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#from mlxtend.plotting import plot_decision_regions
from plot_classifier_cbar import plot_classifier

import sklearn.datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                  'legend.fontsize': 8, 'font.size': 8}
cbar_param     = {'fontsize':12, 'labelpad':20, 'color':'maroon'}
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
def Logistic_regression_and_regularization(seed=SEED):
    print("****************************************************")
    topic = "1. Logistic regression and regularization"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------Read breast cancer wisconsin dataset')
    print('--------------------LogisticRegression, C=1.0, C=0.1')
    # Set the seed for random    
    np.random.seed(seed)
    
    cancer = sklearn.datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_scaled = preprocessing.scale(X) #To avoid warning "...scale the data..."
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)
    
    # Compare with scikit-learn's LogisticRegression
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4))
    
    ax =axes[0]
    lr = LogisticRegression().fit(X_train, y_train)
    ax.plot(cancer.feature_names, lr.coef_.reshape(-1))
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('Features'); ax.set_ylabel('Coefficient')
    ax.set_title('Default regularization', **title_param)
    
    ax =axes[1]
    lr = {'LogisticRegression(C=1)' : LogisticRegression(C=1.0),
          'LogisticRegression(C=.1)': LogisticRegression(C=0.1)}
    for model in lr:
        lr[model].fit(X_train, y_train)
        ax.plot(cancer.feature_names, lr[model].coef_.reshape(-1), label=model)
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('Features'); ax.set_ylabel('Coefficient')
    ax.legend()
    ax.set_title('Custom regularization', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.45, right=None, top=.85, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    
    print('-----------------------------------Movie review data')
    print('-------------------LogisticRegression, C=100, C=0.01')
    # Set the seed for random    
    np.random.seed(seed)
    
    X, y = sklearn.datasets.load_svmlight_file('aclImdb/train/labeledBow.feat')
    y = np.where(y<=4, -1, np.where(y>=7, 1, 0))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    lr_weak_reg = LogisticRegression(C=100, max_iter=2500)
    lr_strong_reg = LogisticRegression(C=0.01, max_iter=2500)
    
    lr_weak_reg.fit(X_train, y_train)
    lr_strong_reg.fit(X_train, y_train)
    
    print('Regularization Training Score:')
    print('C=100,  score:', lr_weak_reg.score(X_train, y_train))
    print('C=0.01, score:', lr_strong_reg.score(X_train, y_train))
    
    print('\nRegularization Test Score:')
    print('C=100,  score:', lr_weak_reg.score(X_test, y_test))
    print('C=0.01, score:', lr_strong_reg.score(X_test, y_test))
    
    # Compare with scikit-learn's LogisticRegression
    fig, ax = plt.subplots()
    ax.plot(lr_weak_reg.coef_.flatten(), label='C=100')
    ax.plot(lr_strong_reg.coef_.flatten(), label='C=0.01')
    ax.legend()
    ax.set_xlabel('Features index'); ax.set_ylabel('Coefficient Value')
    ax.set_title('Custom regularization', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.9, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    print('----------------Read breast cancer wisconsin dataset')
    print('----------LogisticRegression, penalty=l1, penalty=l2')
    # Set the seed for random    
    np.random.seed(seed)
    
    X, y = cancer.data, cancer.target
    X_scaled = preprocessing.scale(X) #To avoid warning "...scale the data..."
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)
    
    lr_L1 = LogisticRegression(penalty='l1', solver='liblinear')
    lr_L2 = LogisticRegression(penalty='l2', solver='liblinear') # penalty='l2' by default
    
    lr_L1.fit(X_train, y_train)
    lr_L2.fit(X_train, y_train)
    
    # Compare with scikit-learn's LogisticRegression
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4))
    model = {'penalty l1': lr_L1, 'penalty l2': lr_L2}
    for lr, ax in zip(model, axes.flat):
        ax.plot(model[lr].coef_.flatten())
        ax.axhline(0, ls='--', lw=.5, color='black')
        ax.set_xlabel('Features index'); ax.set_ylabel('Coefficient Value')
        ax.set_title(lr, **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    
    
def Regularized_logistic_regression(seed=SEED):
    print("****************************************************")
    topic = "2. Regularized logistic regression"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Read data: digits')
    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    print('----------------------------------LogisticRegression')
    # Train and validaton errors initialized as empty list
    train_errs = list()
    valid_errs = list()
    
    # Loop over values of C_value
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for C_value in C_values:
        # Create LogisticRegression object and fit
        lr = LogisticRegression(C=C_value, solver='newton-cg')
        lr.fit(X_train, y_train)
        
        # Evaluate error rates and append to lists
        train_errs.append( 1.0 - lr.score(X_train, y_train) )
        valid_errs.append( 1.0 - lr.score(X_test, y_test) )
    
    # Compare with scikit-learn's LogisticRegression
    fig, ax = plt.subplots()
    ax.semilogx(C_values, train_errs, C_values, valid_errs)
    ax.legend(("train", "validation"))
    ax.set_xlabel('C (inverse regularization strength)'); ax.set_ylabel('Classification Error')
    ax.set_title('Custom regularization', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    
    
def Logistic_regression_and_feature_selection(seed=SEED):
    print("****************************************************")
    topic = "3. Logistic regression and feature selection"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Movie review data')
    X, y = sklearn.datasets.load_svmlight_file('aclImdb/train/labeledBow.feat')
    y = np.where(y<=4, -1, np.where(y>=7, 1, 0))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    print('----------------------------------LogisticRegression')
    # Specify L1 regularization
    lr = LogisticRegression(penalty='l1', solver='liblinear')
    
    # Instantiate the GridSearchCV object and run the search
    searcher = GridSearchCV(lr, {'C':[0.001, 0.01, 0.1, 1, 10]})
    searcher.fit(X_train, y_train)
    
    # Report the best parameters
    print("Best CV params", searcher.best_params_)
    
    # Find the number of nonzero coefficients (selected features)
    best_lr = searcher.best_estimator_
    coefs = best_lr.coef_
    print("Total number of features:", coefs.size)
    print("Number of selected features:", np.count_nonzero(coefs))
    
    
    
def Identifying_the_most_positive_and_negative_words(seed=SEED):
    print("****************************************************")
    topic = "4. Identifying the most positive and negative words"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Movie review data')
    vocab = np.loadtxt("aclImdb/imdb.vocab", dtype=str, delimiter="\r", encoding='utf-8')
    X, y = sklearn.datasets.load_svmlight_file('aclImdb/train/labeledBow.feat')
    y = np.where(y<=4, -1, np.where(y>=7, 1, 0))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    print('----------------------------------LogisticRegression')
    # Specify L1 regularization
    lr = LogisticRegression(penalty='l2', solver='liblinear')
    lr.fit(X_train, y_train)
    
    print('--------------------Most positive and negative words')
    # Get the indices of the sorted cofficients
    inds_ascending = np.argsort(lr.coef_.flatten()) 
    inds_descending = inds_ascending[::-1]
    
    # Print the most positive and negative words
    print("Most positive words: ", vocab[inds_descending[:5]])
    print("Most negative words: ", vocab[inds_ascending[:5]])
    
    
    
def Logistic_regression_and_probabilities(seed=SEED):
    print("****************************************************")
    topic = "5. Logistic regression and probabilities"; print("** %s" % topic)
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
    lr = LogisticRegression(C=10e+8)
    lr.fit(X, y)
    
    print('---------------------------------------------Explore')
    fig, ax = plt.subplots()
    plot_classifier(X, y, lr, ax, proba=True, params_cbar=cbar_param)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('Without Regularization (C=10e+8)', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    
    
def Regularization_and_probabilities(seed=SEED):
    print("****************************************************")
    topic = "7. Regularization and probabilities"; print("** %s" % topic)
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
    
    print('----------------LogisticRegression C=0.1, 1.0, 1e+10')
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 3.5))
    
    # Set the regularization strength and fit the model
    C = [0.1, 1.0, 1e+10]
    for ax, c in zip(axes.flat, C):
        lr = LogisticRegression(C=c)
        lr.fit(X,y)
    
        # Plot the model
        plot_classifier(X, y, lr, ax, proba=True, params_cbar=cbar_param)
        ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
        ax.set_title(f'Without Regularization\n(C={c})', **title_param)
        
        # Predict probabilities on training points
        prob = lr.predict_proba(X)
        msg = f"Max pred. prob.: {np.max(prob):.4f}"
        t = ax.text(0.05, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='top')
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
        print(f"C={c} --> {msg}")
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.75, wspace=.5, hspace=None) #To set the margins 
    plt.show()
    
    
    
def Visualizing_easy_and_difficult_examples(seed=SEED):
    print("****************************************************")
    topic = "8. Visualizing easy and difficult examples"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Read data: digits')
    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target
    
    def show_digit(i, lr, ax):
        ax.imshow(X[i].reshape((8, 8)), cmap=plt.cm.gray)
        ax.axis('off')
        ax.set_title(f"label:{y[i]}, prediction:{lr.predict([X[i]])[0]}\nproba:{lr.predict_proba([X[i]]).max():.2%}", **title_param)
    
    print('----------------------------------LogisticRegression')
    lr = LogisticRegression(solver='newton-cg')
    lr.fit(X, y)
    
    # Get predicted probabilities
    proba = lr.predict_proba(X)
    # Sort the example indices by their maximum probability
    proba_inds = np.argsort(np.max(proba,axis=1))
    
    print('--------------------Plot the most and less confident')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # Show the most confident (least ambiguous) digit
    ax = axes[0]
    show_digit(proba_inds[-1], lr, ax)
    
    # Show the least confident (most ambiguous) digit
    ax = axes[1]
    show_digit(proba_inds[0], lr, ax)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.75, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
      
    
def Multiclass_logistic_regression(seed=SEED):
    print("****************************************************")
    topic = "9. Multi-class logistic regression"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('------------------------------------Read data: wines')
    wine = sklearn.datasets.load_wine()
    X, y = wine.data, wine.target
    X = preprocessing.scale(X)
    
    print('---------------------LogisticRegression: one-vs-rest')
    print('FIRST APRROXIMATION: ')
    # Combining binary classifiers with one-vs-rest
    lr0, lr1, lr2 = LogisticRegression(), LogisticRegression(), LogisticRegression()
    lr0.fit(X, y==0)
    lr1.fit(X, y==1)
    lr2.fit(X, y==2)
    
    # get raw model output
    y_decision = np.zeros((len(y),3))
    y_decision[:,0] = lr0.decision_function(X)
    y_decision[:,1] = lr1.decision_function(X)
    y_decision[:,2] = lr2.decision_function(X)
    y_pred = np.argsort(y_decision,axis=1)[:,2]
    
    print(f'Prediction [10 first elements]: {y_pred[:10]} \n'+\
          f'Real       [10 first elements]: {y[:10]}')
    print(f'Score       : {np.mean(y_pred==y):.2%}')
    print(f'Coefficients: {lr0.coef_.shape}, {lr1.coef_.shape}, {lr2.coef_.shape}\n'+\
          f'Intercepts  : {lr0.intercept_.shape}, {lr1.intercept_.shape}, {lr2.intercept_.shape}\n\n')
    
    print('SECOND APPROXIMATION: ')
    lr = LogisticRegression().fit(X, y)
    y_pred = lr.predict(X)
    print(f'Prediction [10 first elements]: {y_pred[:10]} \n'+\
          f'Real       [10 first elements]: {y[:10]}')
    print(f'Score       : {lr.score(X, y):.2%}')
    print(f'Coefficients: {lr.coef_.shape}\n'+\
          f'Intercepts  : {lr.intercept_.shape}\n\n')
    
    print('----------LogisticRegression: multinomial or softmax')
    lr_mn = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    lr_mn.fit(X,y)
    
    y_pred = lr_mn.predict(X)
    print(f'Prediction [10 first elements]: {y_pred[:10]} \n'+\
          f'Real       [10 first elements]: {y[:10]}')
    print(f'Score       : {lr_mn.score(X, y):.2%}')
    print(f'Coefficients: {lr_mn.coef_.shape}\n'+\
          f'Intercepts  : {lr_mn.intercept_.shape}\n\n')
    
    
    
def Fitting_multiclass_logistic_regression(seed=SEED):
    print("****************************************************")
    topic = "11. Fitting multi-class logistic regression"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Read data: digits')
    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target
    
    X_scaled = preprocessing.scale(X) #To avoid warning "...scale the data..."
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)
    
    print('---------------------LogisticRegression: one-vs-rest')
    # Fit one-vs-rest logistic regression classifier
    lr_ovr = LogisticRegression().fit(X_train, y_train)
    
    print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
    print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))
    
    print('----------LogisticRegression: multinomial or softmax')
    # Fit softmax classifier
    lr_mn = LogisticRegression(multi_class="multinomial", solver="lbfgs").fit(X_train, y_train)
    
    print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
    print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))
    
    """
    #Without scaling
    print('-----------------------------------Read data: digits')
    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    print('---------------------LogisticRegression: one-vs-rest')
    # Fit one-vs-rest logistic regression classifier
    lr_ovr = LogisticRegression(max_iter=2500).fit(X_train, y_train)
    
    print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
    print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))
    
    print('----------LogisticRegression: multinomial or softmax')
    # Fit softmax classifier
    lr_mn = LogisticRegression(max_iter=2500, multi_class="multinomial", solver="lbfgs").fit(X_train, y_train)
    
    print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
    print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))
    """
    
    
    
def Visualizing_multiclass_logistic_regression(seed=SEED):
    print("****************************************************")
    topic = "12. Visualizing multi-class logistic regression"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('---------------------------------------Read the data')
    X, y = sklearn.datasets.load_svmlight_file('multiclass_dataset_cap3_ex12.feat')
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y.astype(np.integer), test_size=.5, shuffle=True, stratify=y)
    
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 4.5))
    print('---------------------LogisticRegression: one-vs-rest')
    ax=axes[0]
    # Fit one-vs-rest logistic regression classifier
    lr_ovr = LogisticRegression(C=100, multi_class='ovr', solver='liblinear').fit(X_train, y_train)
    
    print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
    print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))
    
    # Plot the binary classifier (class 1 vs. rest)
    #plot_decision_regions(X_train, y_train, lr_ovr, legend=2, ax=ax)
    plot_classifier(X_train, y_train, lr_ovr, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('LogisticRegression:\none-vs-rest', **title_param)
    
    print('----------LogisticRegression: multinomial or softmax')
    ax=axes[1]
    # Fit softmax classifier
    lr_mn = LogisticRegression(C=100, multi_class="multinomial", solver="lbfgs").fit(X_train, y_train)
    
    print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
    print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))
    
    # Plot the binary classifier (class 1 vs. rest)
    #plot_decision_regions(X_train, y_train, lr_mn, legend=2, ax=ax)
    plot_classifier(X_train, y_train, lr_mn, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('LogisticRegression:\nmultinomial or softmax', **title_param)
    
    print('-----Create the binary classifier (class 1 vs. rest)')
    ax=axes[2]
    # Create the binary classifier (class 1 vs. rest)
    lr_class_1 = LogisticRegression(C=100, multi_class="multinomial", solver="lbfgs")
    lr_class_1.fit(X_train, y_train==1)

    # Plot the binary classifier (class 1 vs. rest)
    #plot_decision_regions(X_train, y_train, lr_class_1, legend=2, ax=ax)
    plot_classifier(X_train, y_train==1, lr_class_1, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('Binary Classifier\n(class 1 vs. rest)', **title_param)
        
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.75, wspace=.5, hspace=None) #To set the margins 
    plt.show()
    
    
def Onevsrest_SVM(seed=SEED):
    print("****************************************************")
    topic = "13. One-vs-rest SVM"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------Read the data')
    X, y = sklearn.datasets.load_svmlight_file('multiclass_dataset_cap3_ex12.feat')
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y.astype(np.integer), test_size=.5, shuffle=True, stratify=y)
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    print('---------------------------OneVsRestClassifier - SVC')
    svm_ovr = OneVsRestClassifier(LinearSVC()).fit(X_train, y_train)
    ax = axes[0, 0]
    plot_classifier(X_train, y_train, svm_ovr, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('OneVsRestClassifier\nLinearSVC', **title_param)
    
    print('-------------------------------------------LinearSVC')
    svm = LinearSVC().fit(X_train, y_train)
    ax = axes[0, 1]
    plot_classifier(X_train, y_train, svm, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('LinearSVC\n', **title_param)
    
    print('---------------------------------LinearSVC One Class')
    # Create/plot the binary classifier (class 1 vs. rest)
    svm_class_1 = LinearSVC().fit(X_train, y_train==1)
    ax = axes[0, 2]
    plot_classifier(X_train, y_train==1, svm_class_1, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('LinearSVC\nOne Class', **title_param)
    
    print('---------------------OneVsRestClassifier - LinearSVC')
    svm_ovr = OneVsRestClassifier(SVC()).fit(X_train, y_train)
    ax = axes[1,0]
    plot_classifier(X_train, y_train, svm_ovr, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('OneVsRestClassifier\nSVC', **title_param)
    
    print('-------------------------------------------------SVC')
    svm = SVC().fit(X_train, y_train)
    ax = axes[1, 1]
    plot_classifier(X_train, y_train, svm, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('SVC\n', **title_param)
    
    print('---------------------------------------SVC One Class')
    # Create/plot the binary classifier (class 1 vs. rest)
    svm_class_1 = SVC().fit(X_train, y_train==1)
    ax = axes[1, 2]
    plot_classifier(X_train, y_train==1, svm_class_1, ax)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('SVC\nOne Class', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.05, bottom=None, right=.95, top=.85, wspace=.3, hspace=.5) #To set the margins 
    plt.show()
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Logistic_regression_and_regularization()
    Regularized_logistic_regression()
    Logistic_regression_and_feature_selection()
    Identifying_the_most_positive_and_negative_words()
    
    Logistic_regression_and_probabilities()
    Regularization_and_probabilities()
    Visualizing_easy_and_difficult_examples()
    
    Multiclass_logistic_regression()
    Fitting_multiclass_logistic_regression()
    Visualizing_multiclass_logistic_regression()
    Onevsrest_SVM()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})