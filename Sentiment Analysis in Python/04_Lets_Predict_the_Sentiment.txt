# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Let's Predict the Sentiment
    We employ machine learning to predict the sentiment of a review based on 
    the words used in the review. We use logistic regression and evaluate its 
    performance in a few different ways. These are some solid first models!
Source: https://learn.datacamp.com/courses/sentiment-analysis-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer #Score tfidf result like CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #Using accuracy score
from sklearn.metrics import classification_report #Precision, recall, f1-score and support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 9, 'axes.labelweight': 'bold', 'xtick.labelsize': 9, 'ytick.labelsize': 9, 
                  'legend.fontsize': 8, 'font.size': 8}
cbar_param     = {'fontsize':8, 'labelpad':20, 'color':'maroon'}
figsize        = (12.1, 5.9)
SEED           = 42
SIZE           = 10000

# Global configuration
sns.set()
pd.set_option("display.max_columns",24) 
plt.rcParams.update(**plot_param)
np.random.seed(SEED)

# Global functions
def modelize_text_data(X_train, y_train, X_test, y_test, model, ax, Tfidf=False,
                       my_pattern = r'\b[a-zA-Z][a-zA-Z]+\b', my_stop_words = ENGLISH_STOP_WORDS, 
                       verbose=True, labels_plot=None, title_plot='', title_param=title_param):
    # Log the start time
    if verbose: start_time = time.time()
    
    # Build the text_model 
    if Tfidf:
        pipe = make_pipeline(TfidfVectorizer(token_pattern=my_pattern, stop_words=my_stop_words),
                             model)
    else:
        pipe = make_pipeline(CountVectorizer(token_pattern=my_pattern, stop_words=my_stop_words),
                             Normalizer(),
                             model)
    pipe.fit(X_train, y_train)
    if verbose: print("Model Trained...")
    
    # Accuracy score with train split
    y_pred = pipe.predict(X_train)
    print('Accuracy of model in train data: ', accuracy_score(y_train, y_pred))
    
    # Make the predictions
    y_pred = pipe.predict(X_test)
    print('Accuracy of model in test data: ', accuracy_score(y_test, y_pred))
    
    # Confusion matrix
    plot_confusion_matrix(pipe, X_test, y_test, display_labels=labels_plot, 
                          cmap=plt.cm.Blues, normalize='true', values_format='.1%', 
                          ax=ax)
    ax.set_title(title_plot, **title_param)
    ax.grid(False)
        
    # Precision, recall, f1-score and support
    if verbose: print(f'Complete model evaluation: \n{classification_report(y_test, y_pred, zero_division=1)}')
    
    # Log the end time
    if verbose: print(f'\nUsed time: {time.time()-start_time} seg. ({len(X_train)+len(X_test)} processed rows)')
    return pipe

###############################################################################
## Reading the data
###############################################################################
amazon = pd.read_csv('amazon_reviews_sample.csv', index_col=0) #label: 1-->Positive, 0-->Negative
tweets = pd.read_csv('tweets.csv') #airline_sentiment: negative tweets, neutral, positive ones.
movies = pd.read_csv('IMDB_sample.csv', index_col=0) #label: 1-->Positive, 0-->Negative

###############################################################################
## Main part of the code
###############################################################################
def Logistic_regression_using_Amazon_data(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Let's predict the sentiment!"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    X, y = amazon.review, amazon.score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
    
    # Define the stop words
    my_stop_words = ENGLISH_STOP_WORDS.union({'score'})
    
    # Define the model to use
    model = LogisticRegression(solver='newton-cg', random_state=seed)
    
    print('---------------------------------------------Explore')
    print(amazon.score.value_counts())
    
    # For Confusion matrix plot. 
    fig, axes = plt.subplots(1, 2, figsize=figsize, clear=True)
    
    print('-------------------------------------------------BOW')
    final_model = modelize_text_data(X_train, y_train, X_test, y_test, model, axes[0], Tfidf=False, 
                                     my_stop_words=my_stop_words, verbose=True, 
                                     labels_plot=['Negative', 'Positive'], 
                                     title_plot="Amazon CountVectorizer's\nConfusion Matrix")
    
    print('-----------------------------------------------Tfidf')
    final_model = modelize_text_data(X_train, y_train, X_test, y_test, model, axes[1], Tfidf=True, 
                                     my_stop_words=my_stop_words, verbose=True, 
                                     labels_plot=['Negative', 'Positive'], 
                                     title_plot="Amazon TfidfVectorizer's\nConfusion Matrix")
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=None)
    fig.suptitle(topic + '\nRESULTS ON TEST DATA', **suptitle_param)
    plt.show()
    
    
    
def Logistic_regression_using_Movie_reviews(size=SIZE, seed=321):
    print("****************************************************")
    topic = "2. Logistic regression of movie reviews"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    X, y = movies.review, movies.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
    
    # Define the stop words
    my_stop_words = ENGLISH_STOP_WORDS.union({'movies', 'film', 'label'})
    
    # Define the model to use
    model = LogisticRegression(solver='newton-cg', random_state=seed)
    
    print('---------------------------------------------Explore')
    print(movies.label.value_counts())
    
    # For Confusion matrix plot. 
    fig, axes = plt.subplots(1, 2, figsize=figsize, clear=True)
    
    print('-------------------------------------------------BOW')
    final_model = modelize_text_data(X_train, y_train, X_test, y_test, model, axes[0], Tfidf=False, 
                                     my_stop_words=my_stop_words, verbose=True, 
                                     labels_plot=['Negative', 'Positive'], 
                                     title_plot="IMDB CountVectorizer's\nConfusion Matrix")
    
    # Predict the probability of the 0 class
    prob_0 = final_model.predict_proba(X_test)[:, 0]
    
    # Predict the probability of the 1 class
    prob_1 = final_model.predict_proba(X_test)[:, 1]
    
    print("First 10 predicted probabilities of class 0: ", prob_0[:10])
    print("First 10 predicted probabilities of class 1: ", prob_1[:10])
    
    print('-----------------------------------------------Tfidf')
    final_model = modelize_text_data(X_train, y_train, X_test, y_test, model, axes[1], Tfidf=True, 
                                     my_stop_words=my_stop_words, verbose=True, 
                                     labels_plot=['Negative', 'Positive'], 
                                     title_plot="IMDB TfidfVectorizer's\nConfusion Matrix")
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=None)
    fig.suptitle(topic + '\nRESULTS ON TEST DATA', **suptitle_param)
    plt.show()
    
    
def Logistic_regression_using_Twitter_data(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Logistic regression using Twitter data"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    X, y = tweets.text, tweets.airline_sentiment.map({'negative':0, 'neutral':1, 'positive':2})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
    
    # Define the stop words
    my_stop_words = ENGLISH_STOP_WORDS.union({'airline', 'airlines', 'am', 'pm'})
    
    # Define the model to use
    #model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1500, random_state=seed)
    #model = LogisticRegression(max_iter=1500, random_state=seed)
    model = SVC(kernel="linear", random_state=seed)
    
    print('---------------------------------------------Explore')
    print(tweets.airline_sentiment.value_counts())
    
    # For Confusion matrix plot. 
    fig, axes = plt.subplots(1, 2, figsize=figsize, clear=True)
    
    print('-------------------------------------------------BOW')
    final_model = modelize_text_data(X_train, y_train, X_test, y_test, model, axes[0], Tfidf=False, 
                                     my_stop_words=my_stop_words, verbose=True, 
                                     labels_plot=['Negative', 'Neutral', 'Positive'], 
                                     title_plot="Twitter CountVectorizer's\nConfusion Matrix")
    
    print('-----------------------------------------------Tfidf')
    final_model = modelize_text_data(X_train, y_train, X_test, y_test, model, axes[1], Tfidf=True, 
                                     my_stop_words=my_stop_words, verbose=True, 
                                     labels_plot=['Negative', 'Neutral', 'Positive'], 
                                     title_plot="Twitter TfidfVectorizer's\nConfusion Matrix")
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=None)
    fig.suptitle(topic + '\nRESULTS ON TEST DATA', **suptitle_param)
    plt.show()
    
    
    
def Product_reviews_with_regularization(size=SIZE, seed=123):
    print("****************************************************")
    topic = "10. Product reviews with regularization"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    X, y = amazon.review, amazon.score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
    
    # Define the stop words
    my_stop_words = ENGLISH_STOP_WORDS.union({'score'})
    
    # Define the model to use
    model_lowC  = LogisticRegression(C=0.001, max_iter=500, random_state=seed)
    model_highC = LogisticRegression(C=1000,  max_iter=500, random_state=seed)
    
    print('---------------------------------------------Explore')
    print(amazon.score.value_counts())
    
    # For Confusion matrix plot. 
    fig, axes = plt.subplots(2, 2, figsize=figsize, clear=True)
    
    print('-----------------------------------Testing the models')
    for i, ax in enumerate(axes.flat):
        title = "Amazon {}'s - {}".format(str(np.where(i>1, 'TfidfVectorizer', 'CountVectorizer')), 
                                          str(np.where((i%2) == 0, "Lower C = 0.001", "Higher C = 1000")))
        print(title)
        if (i%2 == 0):
            final_model = modelize_text_data(X_train, y_train, X_test, y_test, model_lowC, ax,
                                             Tfidf         = (i>1), 
                                             my_stop_words = my_stop_words, 
                                             verbose       = True,
                                             labels_plot   = ['Negative', 'Positive'], 
                                             title_plot    = title,
                                             title_param   = {'color': 'darkred', 'fontsize': 10, 'weight': 'bold'})
        else:
            final_model = modelize_text_data(X_train, y_train, X_test, y_test, model_highC, ax,
                                             Tfidf         = (i>1), 
                                             my_stop_words = my_stop_words, 
                                             verbose       = True, 
                                             labels_plot   = ['Negative', 'Positive'], 
                                             title_plot    = title,
                                             title_param   = {'color': 'darkred', 'fontsize': 10, 'weight': 'bold'})
        print('-----------------------------------------------------')
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=.3, hspace=.5)
    fig.suptitle(topic + '\nRESULTS ON TEST DATA', **suptitle_param)
    plt.show()
    
    
    
def Regularizing_models_with_Twitter_data(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "11. Regularizing models with Twitter data"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    X, y = tweets.text, tweets.airline_sentiment.map({'negative':0, 'neutral':1, 'positive':2})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
    
    # Define the stop words
    my_stop_words = ENGLISH_STOP_WORDS.union({'score'})
    
    # Define the model to use
    model_lowC  = LogisticRegression(C=0.1, max_iter=1500, random_state=seed)
    model_highC = LogisticRegression(C=100,  max_iter=1500, random_state=seed)
    
    print('---------------------------------------------Explore')
    print(amazon.score.value_counts())
    
    # For Confusion matrix plot. 
    fig, axes = plt.subplots(2, 2, figsize=figsize, clear=True)
    
    print('-----------------------------------Testing the models')
    for i, ax in enumerate(axes.flat):
        title = "Twitter {}'s - {}".format(str(np.where(i>1, 'TfidfVectorizer', 'CountVectorizer')), 
                                          str(np.where((i%2) == 0, "Lower C = 0.1", "Higher C = 100")))
        print(title)
        if (i%2 == 0):
            final_model = modelize_text_data(X_train, y_train, X_test, y_test, model_lowC, ax,
                                             Tfidf         = (i>1), 
                                             my_stop_words = my_stop_words, 
                                             verbose       = False, 
                                             labels_plot   = ['Negative', 'Positive'], 
                                             title_plot    = title,
                                             title_param   = {'color': 'darkred', 'fontsize': 10, 'weight': 'bold'})
        else:
            final_model = modelize_text_data(X_train, y_train, X_test, y_test, model_highC, ax,
                                             Tfidf         = (i>1), 
                                             my_stop_words = my_stop_words, 
                                             verbose       = False, 
                                             labels_plot   = ['Negative', 'Positive'], 
                                             title_plot    = title,
                                             title_param   = {'color': 'darkred', 'fontsize': 10, 'weight': 'bold'})
            
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=.3, hspace=.5)
    fig.suptitle(topic + '\nRESULTS ON TEST DATA', **suptitle_param)
    plt.show()
    
    
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Logistic_regression_using_Amazon_data()
    Logistic_regression_using_Movie_reviews()
    Logistic_regression_using_Twitter_data()
    
    Product_reviews_with_regularization()
    
    Regularizing_models_with_Twitter_data()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})