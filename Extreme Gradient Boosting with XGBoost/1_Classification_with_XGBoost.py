# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:04:30 2021

@author: jacesca@gmail.com
"""
# Import libraries
import xgboost as xgb
import pandas as pd
import numpy as np

import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Define global variables
SEED = 123

# Read data
churn_data = pd.read_csv("churn_data.csv", index_col='Id')
#print(churn_data.head())
#print(churn_data.info())

cancer = sklearn.datasets.load_breast_cancer()
#print(type(cancer))
#print(cancer.keys())

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 1. Classification with XGBoost')
print('*********************************************************')
print('** 1.1 Welcome to the course!')
print('*********************************************************')
print('** 1.2 Which of these is a classification problem?')
print('*********************************************************')
print('** 1.3 Which of these is a binary classification problem?')
print('*********************************************************')
print('** 1.4 Introducing XGBoost')
print('*********************************************************')
print('** 1.5 XGBoost: Fit/Predict')
print('*********************************************************')
# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(use_label_encoder=False, eval_metric='error', n_estimators=10, objective='binary:logistic', seed=SEED)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

print('*********************************************************')
print('** 1.6 What is a decision tree?')
print('*********************************************************')
print('** 1.7 Decision trees')
print('*********************************************************')
# Create arrays for the features and the target: X, y
X, y = cancer.data, cancer.target

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)

print('*********************************************************')
print('** 1.8 What is Boosting?')
print('*********************************************************')
print('** 1.9 Measuring accuracy')
print('*********************************************************')
# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=4, num_boost_round=10, 
                  metrics="error", as_pandas=True, seed=SEED)

# Print cv_results
print(cv_results)

# Print the accuracy
print("\nAccuracy: %f" %((1-cv_results["test-error-mean"]).iloc[-1]))

print('*********************************************************')
print('** 1.10 Measuring AUC')
print('*********************************************************')
# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=4, num_boost_round=10, 
                  metrics="auc", as_pandas=True, seed=SEED)

# Print cv_results
print(cv_results)

# Print the AUC
print("\nAUC:", (cv_results["test-auc-mean"]).iloc[-1])

print('*********************************************************')
print('** 1.11 When should I use XGBoost?')
print('*********************************************************')
print('** 1.12 Using XGBoost')
print('*********************************************************')
print('END')
print('*********************************************************')