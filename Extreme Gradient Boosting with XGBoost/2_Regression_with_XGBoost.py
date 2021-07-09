# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:34:24 2021

@author: jacesca@gmail.com
"""
# Import libraries
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error# Import libraries

# Global configuration
sns.set()
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/' #To avoid msg "ExecutableNotFound: failed to execute ['dot', '-Kdot', '-Tpng'], make sure the Graphviz executables are on your systems' PATH"
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 8, 'font.size': 8})

# Define global variables
SEED = 123
suptitle_param = dict(color='darkblue', fontsize=11)
title_param = {'color': 'darkred', 'fontsize': 12}


# Read data
boston_data = pd.read_csv("boston.csv")
print(boston_data.head())
#print(boston_data.info())

iowa_df = pd.read_csv("ames_housing_trimmed_processed.csv")
print(iowa_df.head())
#print(iowa_df.info())

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 2. Regression with XGBoost')
print('*********************************************************')
print('** 2.1 Regression review')
print('*********************************************************')
print('** 2.2 Which of these is a regression problem?')
print('*********************************************************')
print('** 2.3 Objective (loss) functions and base learners')
print('*********************************************************')
# Trees as base learners example: Scikit-learn API
X, y = boston_data.iloc[:,:-1],boston_data.iloc[:,-1]
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=SEED)

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=SEED)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))
print("Tree as base learnesrs - RMSE: %f" % (rmse))

# Linear base learners example: learning API only
DM_train = xgb.DMatrix(data=X_train,label=y_train)
DM_test = xgb.DMatrix(data=X_test,label=y_test)

params = {"booster":"gblinear","objective":"reg:squarederror", 'seed':SEED}

xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=10)

preds = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))
print("Linear as base learners - RMSE: %f" % (rmse))


print('*********************************************************')
print('** 2.4 Decision trees as base learners')
print('*********************************************************')
# Create arrays for the features and the target: X, y
X, y = iowa_df.iloc[:,:-1],iowa_df.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(booster="gbtree", # Default value, it is not necessary to specify. 
                          objective='reg:squarederror', n_estimators=10, seed=SEED)

# Fit the regressor to the training set
xg_reg.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


print('*********************************************************')
print('** 2.5 Linear base learners')
print('*********************************************************')
# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test =  xgb.DMatrix(data = X_test, label = y_test)

# Create the parameter dictionary: params
params = {"booster":"gblinear", "objective":"reg:squarederror", 'seed':SEED}

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain = DM_train, num_boost_round = 10)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))

print('*********************************************************')
print('** 2.6 Evaluating model quality')
print('*********************************************************')
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, 
                    params=params, nfold=4, num_boost_round=5, 
                    metrics='rmse', as_pandas=True, 
                    seed=SEED)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print('RMSE: %f' % (cv_results["test-rmse-mean"]).tail(1))

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, 
                    params=params, nfold=4, num_boost_round=5, 
                    metrics='mae', as_pandas=True, 
                    seed=SEED)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print('MAE: %f' % (cv_results["test-mae-mean"]).tail(1))

print('*********************************************************')
print('** 2.7 Regularization and base learners in XGBoost')
print('*********************************************************')
# L1 regularization in XGBoost example

X,y = boston_data.iloc[:,:-1],boston_data.iloc[:,-1]

boston_dmatrix = xgb.DMatrix(data=X,label=y)

params    = {"objective":"reg:squarederror","max_depth":4}
l1_params = [0.1, 0.5, 0.9, 0.95, 1, 1.05, 3, 5, 10,100]
rmses_l1  = []

for reg in l1_params:
    params["alpha"] = reg
    
    cv_results = xgb.cv(dtrain          = boston_dmatrix, 
                        params          = params,
                        nfold           = 4,
                        num_boost_round = 10,
                        metrics         = "rmse",
                        as_pandas       = True,
                        seed            = SEED)
    
    rmses_l1.append(cv_results["test-rmse-mean"].tail(1).values[0])

print("Best rmse as a function of l1:")
print(pd.DataFrame(list(zip(l1_params,rmses_l1)), columns=["l1","rmse"]))
     
print('*********************************************************')
print('** 2.8 Using regularization in XGBoost')
print('*********************************************************')
# Create arrays for the features and the target: X, y
X, y = iowa_df.iloc[:,:-1],iowa_df.iloc[:,-1]

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

reg_params = [0.1, 0.5, 0.9, 0.95, 1, 1.05, 3, 5, 10,100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective":"reg:squarederror","max_depth":4}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:

    # Update l2 strength
    params["lambda"] = reg
    
    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, 
                             params=params, 
                             nfold=4, 
                             num_boost_round=10, 
                             metrics="rmse", 
                             as_pandas=True, 
                             seed=SEED)
    
    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))

print('*********************************************************')
topic = '2.9 Visualizing individual XGBoost trees'; print(f'** {topic}')
print('*********************************************************')
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"booster":"gbtree", # Default value, it is not necessary to specify.
          "objective":"reg:squarederror", "max_depth":2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

fig, axes = plt.subplots(2, 2, figsize=(11.75, 5.9))
fig.suptitle(topic, **suptitle_param)

for ax, num, title in zip(axes.flatten(), [0, 4, 9], ['First Tree', 'Fith Tree', 'Last Tree']):
    # Plot the tree
    xgb.plot_tree(xg_reg, num_trees=0, ax=ax)
    ax.set_title(title, **title_param)

axes.flatten()[-1].axis('off') 
plt.show()
print('*********************************************************')
topic = '2.10 Visualizing feature importances'; print(f'** {topic}')
print('*********************************************************')
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the feature importances
fig, ax = plt.subplots()
fig.suptitle(topic, **suptitle_param)

xgb.plot_importance(xg_reg, ax=ax)
plt.title('Feature importance in IOWA Housing', **title_param)
plt.subplots_adjust(left=.3, right=None, bottom=None, top=None, hspace=None, wspace=None);
plt.show()

print('*********************************************************')
print('END')
print('*********************************************************')
plt.style.use('default')