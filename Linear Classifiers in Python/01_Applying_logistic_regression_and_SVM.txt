# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 1: Applying_logistic_regression_and_SVM
    In this chapter you will learn the basics of applying logistic regression 
    and support vector machines (SVMs) to classification problems. You'll use 
    the scikit-learn library to fit classification models to real data.
Source: https://learn.datacamp.com/courses/linear-classifiers-in-python
More documentation: http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

import sklearn.datasets
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest

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

# Global configuration
np.set_printoptions(suppress=True)

###############################################################################
## Reading the data
###############################################################################
# Global configuration
plt.rcParams.update(**plot_param)
np.random.seed(SEED)

###############################################################################
## Main part of the code
###############################################################################
def scikit_learn_refresher(seed=SEED):
    print("****************************************************")
    topic = "1. scikit-learn refresher"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-------------------------------Read data: newsgroups')
    newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()
    X, y = newsgroups.data, newsgroups.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    print('------------------------------------Explore the data')
    print(f"""
           X shape: {X.shape}
           y shape: {y.shape}
           
           Head of X: 
           {X[:1,:15000]}
           Head of y:
           {y[:5]}""")
    
    print('--------------------------------KNeighborsClassifier')
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    print(f"""
           Head of Prediction: {y_pred[:5]}
           Head of Test Y: {y_test[:5]}
           
           Score: {score}""")
    
    
    
def KNN_classification(seed=SEED):
    print("****************************************************")
    topic = "2. KNN classification"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('----------------------Read data: sentimental reviews')
    X, y = sklearn.datasets.load_svmlight_file('aclImdb/train/labeledBow.feat')
    y = np.where(y<=4, -1, np.where(y>=7, 1, 0))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    print('--------------------------------KNeighborsClassifier')
    # Create and fit the model
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    
    # Predict on the test features, print the results
    y_pred = knn.predict(X_test)
    print("Prediction for the 10 first elements:", y_pred[:10])
    print("Values for the 10 first elements    :", y_test[:10])
    
    score = knn.score(X_test, y_test)
    print("Score:", score)
    
    
    
def Comparing_models(seed=SEED):
    print("****************************************************")
    topic = "3. Comparing models"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Read data: digits')
    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    print('--------------------------------KNeighborsClassifier')
    # Create and fit the model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Predict on the test features, print the results
    y_pred = knn.predict(X_test)
    print("Prediction for the 10 first elements:", y_pred[:10])
    print("Values for the 10 first elements    :", y_test[:10])
    
    score = knn.score(X_test, y_test)
    print("Score:", score)
    
    
    
def Applying_logistic_regression_and_SVM(seed=SEED):
    print("****************************************************")
    topic = "5. Applying logistic regression and SVM"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('------------------------------------Read data: wines')
    wine = sklearn.datasets.load_wine()
    X, y = wine.data, wine.target
    X_scaled = preprocessing.scale(X) #To avoid warning "...scale the data..."
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)
    
    print('----------------------------------LogisticRegression')
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    print(f"""
           Head of Prediction: {y_pred[:5]}
           Head of Test Y: {y_test[:5]}
           
           Score: {score}
           """)
    
    print("Prediction for the first element: \n{}".format(lr.predict_proba(X_test[:1])))
    
    print('-------------------------------------------LinearSVC')
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    score = svm.score(X_test, y_test)
    print(f"""
           Head of Prediction: {y_pred[:5]}
           Head of Test Y: {y_test[:5]}
           
           Score: {score}
           """)
    
    print('-------------------------------------------------SVC')
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    svm = SVC()
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    score = svm.score(X_test, y_test)
    print(f"""
           Head of Prediction: {y_pred[:5]}
           Head of Test Y: {y_test[:5]}
           
           Score: {score}
           """)
    
    
    
def Running_LogisticRegression_and_SVC(seed=SEED):
    print("****************************************************")
    topic = "6. Running LogisticRegression and SVC"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('-----------------------------------Read data: digits')
    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target
    X_scaled = preprocessing.scale(X) #To avoid warning "...scale the data..."
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)
    
    print('----------------------------------LogisticRegression')
    # Apply logistic regression and print scores
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    print(lr.score(X_train, y_train))
    print(lr.score(X_test, y_test))
    
    print('-------------------------------------------------SVC')
    # Apply SVM and print scores
    svm = SVC()
    svm.fit(X_train, y_train)
    print(svm.score(X_train, y_train))
    print(svm.score(X_test, y_test))
    
    
    
def Sentiment_analysis_for_movie_reviews(seed=SEED):
    print("****************************************************")
    topic = "7. Sentiment analysis for movie reviews"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    lines = np.loadtxt("aclImdb/imdb.vocab", dtype=str, delimiter="\r", encoding='utf-8')
    vocab = dict(zip(lines, range(len(lines))))
    
    def get_features(review, vocabulary=vocab):
        vectorizer = CountVectorizer(vocabulary=vocabulary)
        X = vectorizer.fit_transform([review])
        return X
    
    print('----------------------Read data: sentimental reviews')
    X, y = sklearn.datasets.load_svmlight_file('aclImdb/train/labeledBow.feat')
    y = np.where(y<=4, -1, np.where(y>=7, 1, 0))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    print('----------------------------------LogisticRegression')
    # Instantiate logistic regression and train
    lr = LogisticRegression(solver='sag', max_iter=2500)
    lr.fit(X_train, y_train)
    
    # Predict sentiment for a glowing review
    review1 = "LOVED IT! This movie was amazing. Top 10 this year."
    review1_features = get_features(review1)
    print("Review:", review1)
    print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])
    print("Probability of negative review:", lr.predict_proba(review1_features)[0,0])
    
    # Predict sentiment for a poor review
    review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
    review2_features = get_features(review2)
    print("\n\nReview:", review2)
    print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])
    print("Probability of negative review:", lr.predict_proba(review2_features)[0,0])
    
    
def Visualizing_decision_boundaries(seed=SEED):
    print("****************************************************")
    topic = "10. Visualizing decision boundaries"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('------------------------------------Read data: wines')
    wine = sklearn.datasets.load_wine()
    X, y = wine.data, wine.target
    #X_scaled = preprocessing.scale(X[:,:2]) #To avoid warning "...scale the data..."
    X_scaled = preprocessing.scale(X)
    
    print('------------------------Select best features to plot')
    print(f"Features: \n{wine.feature_names}")
    features = np.array(wine.feature_names)
    selector = SelectKBest(k=2)
    selector.fit(X_scaled, y)
    features_index = selector.get_support(indices=True)
    x_label = features[features_index[0]]
    y_label = features[features_index[1]]
    X_new = selector.transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=.25)
    print(f"Features selected: [{x_label}, {y_label}]\n")
    
    print('-------------Linear classifiers with only 2 features')
    # Define the classifiers
    classifiers = {'LogisticRegression': LogisticRegression(), 
                   'LinearSVC': LinearSVC(), 
                   'SVC': SVC(), 
                   'KNeighborsClassifier': KNeighborsClassifier()}
    
    # Fit the classifiers
    fig, axes = plt.subplots(2,2, figsize=figsize)
    for ax, c in zip(axes.flatten(), classifiers):
        classifiers[c].fit(X_train, y_train)
        msg = f'score: {classifiers[c].score(X_test, y_test):.4f}'
        print(f'{c} {msg}')
        plot_decision_regions(X_train, y_train, classifiers[c], legend=2, ax=ax)
        t = ax.text(0.02, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='top')
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(c, **title_param)
        
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.4); #To set the margins 
    plt.show()
    
    print('----------------Linear classifiers with all features')
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)
    for c in classifiers:
        classifiers[c].fit(X_train, y_train)
        msg = f'score: {classifiers[c].score(X_test, y_test):.4f}'
        print(f'{c} {msg}')
    
    
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    scikit_learn_refresher()
    KNN_classification()
    Comparing_models()
    
    Applying_logistic_regression_and_SVM()
    Running_LogisticRegression_and_SVC()
    Sentiment_analysis_for_movie_reviews()
    
    Visualizing_decision_boundaries()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})
    np.set_printoptions(suppress=False)