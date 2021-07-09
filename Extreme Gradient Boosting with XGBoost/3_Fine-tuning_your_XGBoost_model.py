# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:09:02 2021

@author: jacesca@gmail.com
"""
# Import libraries
import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Define global variables
SEED = 123

# Read data
housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
#print(housing_data.head())
#print(housing_data.info())

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 3 Fine-tuning your XGBoost model')
print('*********************************************************')
print('** 3.1 Why tune your model?')
print('*********************************************************')
# Untuned model example
#X, y = housing_data[housing_data.columns.tolist()[:-1]],
#       housing_data[housing_data.columns.tolist()[-1]]
X, y = housing_data.iloc[:,:-1],housing_data.iloc[:,-1]


housing_dmatrix = xgb.DMatrix(data=X,label=y)

untuned_params={"objective":"reg:squarederror"}
untuned_cv_results_rmse = xgb.cv(dtrain=housing_dmatrix,
                                 params=untuned_params,
                                 nfold=4,
                                 metrics="rmse",
                                 as_pandas=True,
                                 seed=SEED)
print("Untuned rmse: %f" %((untuned_cv_results_rmse["test-rmse-mean"]).tail(1)))

# Tuned model example
tuned_params = {"objective":"reg:squarederror",
                'colsample_bytree': 0.3,
                'learning_rate': 0.1, 
                'max_depth': 5}

tuned_cv_results_rmse = xgb.cv(dtrain=housing_dmatrix,
                               params=tuned_params, 
                               nfold=4, 
                               num_boost_round=200, 
                               metrics="rmse",
                               as_pandas=True, 
                               seed=SEED)
print("Tuned rmse  : %f" %((tuned_cv_results_rmse["test-rmse-mean"]).tail(1)))
reduction = (untuned_cv_results_rmse.iloc[-1]['test-rmse-mean'] - tuned_cv_results_rmse.iloc[-1]['test-rmse-mean']) / untuned_cv_results_rmse.iloc[-1]['test-rmse-mean']
print(f'Means {reduction:.2%} reduction')


print('*********************************************************')
print('** 3.2 When is tuning your model a bad idea?')
print('*********************************************************')
print('** 3.3 Tuning the number of boosting rounds')
print('*********************************************************')
# Create arrays for the features and the target: X, y
X, y = housing_data.iloc[:,:-1],housing_data.iloc[:,-1]

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params 
params = {"objective":"reg:squarederror", "max_depth":3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15, 100, 150, 175, 180, 200, 300]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)
    
    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))

print('*********************************************************')
print('** 3.4 Automated boosting round selection using early_stopping')
print('*********************************************************')
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain = housing_dmatrix,
                    params = params,
                    nfold = 3,
                    metrics = 'rmse',
                    early_stopping_rounds = 10,
                    num_boost_round = 50,
                    seed = SEED,
                    as_pandas = True)

# Print cv_results
print(cv_results)

print('*********************************************************')
print("** 3.5 Overview of XGBoost's hyperparameters")
print('*********************************************************')
print('** 3.6 Tuning eta')
print('*********************************************************')
# Create arrays for the features and the target: X, y
X, y = housing_data.iloc[:,:-1],housing_data.iloc[:,-1]

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective":"reg:squarederror", "max_depth":3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta 
for curr_val in eta_vals:

    params["eta"] = curr_val
    
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain = housing_dmatrix,
                        params = params,
                        nfold = 3,
                        early_stopping_rounds = 5,
                        num_boost_round = 10,
                        metrics = 'rmse',
                        seed = SEED,
                        as_pandas = True)    
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))

print('*********************************************************')
print('** 3.7 Tuning max_depth')
print('*********************************************************')
# Create the parameter dictionary
params = {"objective":"reg:squarederror"}

# Create list of max_depth values
max_depths = [2, 5, 10, 20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depth"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain = housing_dmatrix,
                        params = params,
                        nfold = 2,
                        early_stopping_rounds = 5,
                        num_boost_round = 10,
                        metrics = 'rmse',
                        seed = SEED,
                        as_pandas = True)
    
    
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))

print('*********************************************************')
print('** 3.8 Tuning colsample_bytree')
print('*********************************************************')
# Create the parameter dictionary
params={"objective":"reg:squarederror","max_depth":3}

# Create list of hyperparameter values: colsample_bytree_vals
colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
best_rmse = []

# Systematically vary the hyperparameter value 
for curr_val in colsample_bytree_vals:

    params['colsample_bytree'] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, 
                        params=params, 
                        nfold=2,
                        num_boost_round=10, 
                        early_stopping_rounds=5,
                        metrics="rmse", 
                        as_pandas=True, 
                        seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree","best_rmse"]))

print('*********************************************************')
print('** 3.9 Review of grid search and random search')
print('*********************************************************')
# Grid search: example
print('** Using "Grid search":')

X, y = housing_data.iloc[:,:-1],housing_data.iloc[:,-1]
housing_dmatrix = xgb.DMatrix(data=X,label=y)

gbm_param_grid = {'learning_rate': [0.01,0.1,0.5,0.9],
                  'n_estimators': [200],
                  'subsample': [0.3, 0.5, 0.9]}

gbm = xgb.XGBRegressor()

grid_mse = GridSearchCV(estimator = gbm,
                        param_grid = gbm_param_grid,
                        scoring = 'neg_mean_squared_error', 
                        cv = 4, 
                        verbose = 1)
grid_mse.fit(X, y)

print("Best parameters found: ",grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))   

# Random search: example
print('\n** Using "Random search":')

gbm_param_grid = {'learning_rate': np.arange(0.05,1.05,.05),
                  'n_estimators': [200],
                  'subsample': np.arange(0.05,1.05,.05)}

gbm = xgb.XGBRegressor()

randomized_mse = RandomizedSearchCV(estimator = gbm, 
                                    param_distributions = gbm_param_grid,
                                    n_iter = 25, 
                                    scoring = 'neg_mean_squared_error', 
                                    cv = 4, 
                                    random_state = SEED,
                                    verbose = 1)

randomized_mse.fit(X, y)

print("Best parameters found: ",randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

print('*********************************************************')
print('** 3.10 Grid search with XGBoost')
print('*********************************************************')
# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator = gbm,
                        param_grid = gbm_param_grid,
                        scoring = 'neg_mean_squared_error',
                        cv = 4,
                        verbose = 1)


# Fit grid_mse to the data
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

print('*********************************************************')
print('** 3.11 Random search with XGBoost')
print('*********************************************************')
# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator = gbm,
                                    param_distributions = gbm_param_grid,
                                    n_iter = 5,
                                    scoring = 'neg_mean_squared_error',
                                    cv = 4,
                                    random_state = SEED,
                                    verbose = 1)

# Fit randomized_mse to the data
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

print('*********************************************************')
print('** 3.12 Limits of grid search and random search')
print('*********************************************************')
print('** 3.13 When should you use grid search and random search?')
print('*********************************************************')
print('END')
print('*********************************************************')
