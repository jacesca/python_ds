# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Building a "fake news" classifier
    You'll apply the basics of what you've learned along with some supervised 
    machine learning to build a "fake news" detector. You'll begin by learning 
    the basics of supervised machine learning, and then move forward by choosing 
    a few important features and testing ideas to identify and classify fake 
    news articles.
Source: https://learn.datacamp.com/courses/introduction-to-natural-language-processing-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import plot_confusion_matrix

from polyglot.text import Text


###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 10, 'weight': 'bold'}
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
def CountVectorizer_for_text_classification(size=SIZE, seed=53):
    print("****************************************************")
    topic = "5. CountVectorizer for text classification"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    df = pd.read_csv('fake_or_real_news.csv', encoding='utf_8')
    
    #using .isprintable() to avoid error "input contains invalid UTF-8" get in Text().language.name
    df['lang'] = [Text(''.join(c for c in sinopsis if c.isprintable())).language.code for sinopsis in df.text]
    df = df[df.lang == 'en']
    
    print('-------------------------------------------Exploring')
    # Print the head of df
    print(df.head())
    
    print('------------------------------BOWs - CountVectorizer')
    # Create a series to store the features and the labels: y
    X = df.text
    y = df.label
    
    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, stratify=y, random_state=seed)
    
    # Initialize a CountVectorizer object: count_vectorizer
    count_vectorizer = CountVectorizer(stop_words='english')
    
    # Transform the training data using only the 'text' column values: count_train 
    count_train = count_vectorizer.fit_transform(X_train)
    print(f'Train data ({count_train.shape[0]} regs.): \n{count_train.toarray()}\n')
    
    # Transform the test data using only the 'text' column values: count_test 
    count_test = count_vectorizer.transform(X_test)
    print(f'Test data ({count_test.shape[0]} regs.): \n{count_test.toarray()}\n')
    
    # Print the first 10 features of the count_vectorizer
    print(f'First 10 features extracted: \n{count_vectorizer.get_feature_names()[:10]}\n')
    
    
    
    print("****************************************************")
    topic = "6. TfidfVectorizer for text classification"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------BOWs - TfidfVectorizer')
    # Initialize a TfidfVectorizer object: tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    
    # Transform the training data: tfidf_train 
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    
    # Transform the test data: tfidf_test 
    tfidf_test = tfidf_vectorizer.transform(X_test)
    
    # Print the first 10 features
    print(tfidf_vectorizer.get_feature_names()[:10])
    
    # Print the first 5 vectors of the tfidf training data
    print(tfidf_train.A[:5])

    
    
    print("****************************************************")
    topic = "7. Inspecting the vectors"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------------Transforming to dataframes')
    # Create the CountVectorizer DataFrame: count_df
    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
    
    # Create the TfidfVectorizer DataFrame: tfidf_df
    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
    
    print('---------------------------------------------Explore')
    # Print the head of count_df
    print(f'Head of count_df: \n{count_df.head()}\n')
    
    # Print the head of tfidf_df
    print(f'Head of tfidf_df: \n{tfidf_df.head()}\n')
    
    print('--------------------------------Difference in colums')
    # Calculate the difference in columns: difference
    difference = set(count_df.columns) - set(tfidf_df.columns)
    print(difference)
    
    print('------------------------------Equals both dataframes')
    # Check whether the DataFrames are equal
    print(count_df.equals(tfidf_df))
    
    
    
    print("****************************************************")
    topic = '10. Training and testing the "fake news" model with CountVectorizer'; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------MultinomialNB')
    # Instantiate a Multinomial Naive Bayes classifier: nb_classifier
    nb_count_classifier = MultinomialNB()
    
    # Fit the classifier to the training data
    nb_count_classifier.fit(count_train, y_train)
    
    print('----------------------------------Making predictions')
    # Create the predicted tags: pred
    pred = nb_count_classifier.predict(count_test)
    
    print('--------------------------------Evaluating the model')
    # Calculate the accuracy score: score
    score = metrics.accuracy_score(y_test, pred)
    print(f'Accuracy: {score}\n')
    
    # Calculate the confusion matrix: cm
    model_labels = ['FAKE', 'REAL']
    cm = metrics.confusion_matrix(y_test, pred, labels=model_labels)
    print(f'Confusion Matrix: \n{cm}\n')
    
    # Calculate the confusion matrix: normalized cm
    model_labels = ['FAKE', 'REAL']
    cm = metrics.confusion_matrix(y_test, pred, labels=model_labels, normalize='true')
    print(f'Confusion Matrix: \n{cm}\n')
    
    # Plotting Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    plot_confusion_matrix(nb_count_classifier, count_test, y_test, display_labels=model_labels, 
                          cmap=plt.cm.Blues, normalize='true', values_format='.1%', 
                          ax=ax)
    ax.set_title(f'MultinolialNB model with CountVectorizer\nAccuracy score: {score:.1%}', **title_param)
    ax.grid(False)
    
    print("****************************************************")
    topic = '11. Training and testing the "fake news" model with TfidfVectorizer'; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------MultinomialNB')
    # Create a Multinomial Naive Bayes classifier: nb_classifier
    nb_tfidf_classifier = MultinomialNB()
    
    # Fit the classifier to the training data
    nb_tfidf_classifier.fit(tfidf_train, y_train)
    
    print('----------------------------------Making predictions')
    # Create the predicted tags: pred
    pred = nb_tfidf_classifier.predict(tfidf_test)
    
    print('--------------------------------Evaluating the model')
    # Calculate the accuracy score: score
    score = metrics.accuracy_score(y_test, pred)
    print(f'Accuracy: {score}\n')
    
    # Calculate the confusion matrix: cm
    model_labels = ['FAKE', 'REAL']
    cm = metrics.confusion_matrix(y_test, pred, labels=model_labels)
    print(f'Confusion Matrix: \n{cm}\n')
    
    # Calculate the confusion matrix: normalized cm
    model_labels = ['FAKE', 'REAL']
    cm = metrics.confusion_matrix(y_test, pred, labels=model_labels, normalize='true')
    print(f'Confusion Matrix: \n{cm}\n')
    
    # Plotting Confusion matrix
    ax = axes[1]
    plot_confusion_matrix(nb_tfidf_classifier, count_test, y_test, display_labels=model_labels, 
                          cmap=plt.cm.Blues, normalize='true', values_format='.1%', 
                          ax=ax)
    ax.set_title(f'MultinolialNB model with TfidfVectorizer\nAccuracy score: {score:.1%}', **title_param)
    ax.grid(False)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=.3, hspace=None)
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
    print("****************************************************")
    topic = "14. Improving your model"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-Setting alph of MultinomialNB using TfidfVectorizer')
    # Create the list of alphas: alphas
    alphas = np.arange(0.1, 1.1, step=.1)
    
    # Define train_and_predict()
    def train_and_predict(alpha):
        # Instantiate the classifier: nb_classifier
        nb_classifier = MultinomialNB(alpha=alpha)
        
        # Fit to the training data
        nb_classifier.fit(tfidf_train, y_train)
        
        # Predict the labels: pred
        pred = nb_classifier.predict(tfidf_test)
        
        # Compute accuracy: score
        score = metrics.accuracy_score(y_test, pred)
        
        return score
    
    # Iterate over the alphas and print the corresponding score
    scores = {alpha: train_and_predict(alpha) for alpha in alphas}
    for k, v in scores.items():
        print(f'Alpha: {k:.2f}, Score: {v:.2%}')
    
    best_alpha = max(scores, key=scores.get)
    print(f'\nBest alpha: {best_alpha}\n')
    
    
    
    print("****************************************************")
    topic = "15. Inspecting your model"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------Preparing the right model')
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=best_alpha)
    
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    
    print('---Inspecting 20 most important words for each class')
    # Get the class labels: class_labels
    class_labels = nb_classifier.classes_
    
    # Extract the features: feature_names
    feature_names = tfidf_vectorizer.get_feature_names()
    
    # Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
    feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))
    
    # Print the first class label and the top 20 feat_with_weights entries
    print(f'For "class_labels[0]" label: \n{dict(feat_with_weights[:20]).values()}')
    
    # Print the second class label and the bottom 20 feat_with_weights entries
    print(f'For "class_labels[1]" label: \n{dict(feat_with_weights[-20:]).values()}')
    
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    CountVectorizer_for_text_classification()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})