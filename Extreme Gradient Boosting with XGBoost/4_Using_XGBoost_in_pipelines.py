# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 00:07:42 2021

@author: jacesca@gmail.com
"""

# Import libraries
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

# Define global variables
SEED = 123

# Define global configuration
np.random.seed(SEED) 

# Read data
names = ["crime","zone","industry","charles","no","rooms", "age", "distance","radial","tax","pupil","aam","lower","med_price"]
boston_data = pd.read_csv("boston.csv", names=names, skiprows=1)
#print(boston_data.head())
#print(boston_data.info())

housing_unproc = pd.read_csv("ames_unprocessed_data.csv")
#print(housing_unproc.head())
#print(housing_unproc.info())

columns_name = ['age', 'bp'   , 'sg' , 'al' , 'su'   , 'rbc', 'pc', 'pcc', 'ba' , 'bgr', 
                'bu' , 'sc'   , 'sod', 'pot', 'hemo' , 'pcv', 'wc' , 'rc', 'htn', 'dm' ,
                'cad', 'appet', 'pe' , 'ane', 'class'] 
kideney = pd.read_csv("chronic_kidney_disease.csv", names=columns_name, na_values='?')
#print(kideney.head())
#print(kideney.info())

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 4. Using XGBoost in pipelines')
print('*********************************************************')
print('** 4.1 Review of pipelines using sklearn')
print('*********************************************************')
# Scikit-learn pipeline example
#X, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]
X, y = boston_data.drop('med_price', axis=1), boston_data.med_price

rf_pipeline = Pipeline([("st_scaler", StandardScaler()), 
                        ("rf_model", RandomForestRegressor(random_state=SEED))])

scores = cross_val_score(rf_pipeline,
                         X, y, 
                         scoring="neg_mean_squared_error", 
                         cv=10)

final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))
print("Final RMSE:", final_avg_rmse)

print('*********************************************************')
print('** 4.2 Exploratory data analysis')
print('*********************************************************')
#print(housing_unproc.head())
#print(housing_unproc.info())
print(housing_unproc.describe())

print('*********************************************************')
print('** 4.3 Encoding categorical columns I: LabelEncoder')
print('*********************************************************')
# Fill missing values with 0
housing_unproc.LotFrontage = housing_unproc.LotFrontage.fillna(0)

# Copy the dataframe to work with
df_sk = housing_unproc.copy(deep = True)

# Create a boolean mask for categorical columns
categorical_mask = (df_sk.dtypes == object)

# Get list of categorical column names
categorical_columns = df_sk.columns[categorical_mask].tolist()

# Print count distinct values in each categorical columns
print(df_sk[categorical_columns].nunique(axis=0),'\n')

# Print the head of the categorical columns
#print(df_sk[categorical_columns].head())

# Print the head of unlabeled categorical columns
print(df_sk[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df_sk[categorical_columns] = df_sk[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df_sk[categorical_columns].head())

# Exploring a little bit more
#print('\nData in Neighborhood:')
#print('Before encoding: ', housing_unproc['Neighborhood'].unique())
#print('After  encoding: ', df_sk['Neighborhood'].unique())

print('*********************************************************')
print('** 4.4 Encoding categorical columns II: OneHotEncoder')
print('*********************************************************')
print('** Using OneHotEncoder on labeled data to encode')
print('---------------------------------------------------------')
# Print the list of categorical column names
#print(df_sk.head(1))
#print(categorical_mask)
#print(df_sk.columns[categorical_mask].tolist(),'\n')

############################################################
## Without drop='first' and after labeling
############################################################
# Create OneHotEncoder: ohe
ohe = ColumnTransformer([("OneHotEncoder", OneHotEncoder(), categorical_mask)], 
                        remainder = 'passthrough',
                        sparse_threshold = 0,
                        verbose = True)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
housing_encoded = ohe.fit_transform(df_sk)

# Print the shape of the original DataFrame
print("Before OneHotEncoder:", df_sk.shape)

# Print the shape of the transformed array
print("After OneHotEncoder:", housing_encoded.shape)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print('\nFirst row of the resulting dataset:')
print(housing_encoded[:1, :], '\n')

# Transforming to df
df_data = pd.DataFrame(data=housing_encoded, columns=ohe.get_feature_names())
print(df_data.head())
print(ohe.get_feature_names())

print('---------------------------------------------------------')
print('** Using OneHotEncoder(drop="first") on unlabeled data to encode')
print('---------------------------------------------------------')
############################################################
## With drop='first' and not labeling
############################################################
# Create OneHotEncoder: ohe
ohe = ColumnTransformer([("OneHotEncoder", OneHotEncoder(drop='first'), categorical_mask)], 
                        remainder = 'passthrough',
                        sparse_threshold = 0,
                        verbose = True)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
housing_encoded_drop = ohe.fit_transform(housing_unproc)

# Print the shape of the original DataFrame
print("Before OneHotEncoder:", df_sk.shape)

# Print the shape of the transformed array
print("After OneHotEncoder:", housing_encoded_drop.shape)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print('\nFirst row of the resulting dataset:')
print(housing_encoded_drop[:1, :])

print('---------------------------------------------------------')
print('** Using pd.get_dummies to encode')
print('---------------------------------------------------------')
#print(housing_unproc.head(1))
#print(housing_unproc.shape)

# Using pandas to encode categorical columns
print('Encoded with Pandas Whithout drop parameters')
df_pd = housing_unproc.copy(deep = True)
df_pd = pd.get_dummies(df_pd)
print(df_pd.head())
print('Shape:', df_pd.shape)

# Using pandas to encode categorical columns
print('\n\nEncoded with Pandas Whit drop parameters')
df_pd = housing_unproc.copy(deep = True)
df_pd = pd.get_dummies(df_pd, drop_first=True)
print(df_pd.head())
print('Shape:', df_pd.shape)

print("""
---------------------------------------------------------
      OneHotEncoder vs get_dummies:
          For machine learning, you almost definitely 
          want to use sklearn.OneHotEncoder. 
          For other tasks like simple analyses, you 
          might be able to use pd.get_dummies, which 
          is a bit more convenient.
---------------------------------------------------------
      """)

print('*********************************************************')
print('** 4.5 Encoding categorical columns III: DictVectorizer')
print('*********************************************************')
# Convert df into a dictionary: df_dict
df_dict = housing_unproc.to_dict('records')

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse = False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first two rows
print('First row of the resulting dataset:')
print(df_encoded[:1,:], '\n')

# Print the vocabulary
print('Vocabulary:')
print(dv.vocabulary_)

# Transforming to df
print('\nTransforming to dataframe:')
df_data = pd.DataFrame(data=df_encoded, columns=dv.feature_names_)
print(df_data.head())

#cols = dict(sorted(dv.vocabulary_.items(), key=lambda w: w[1]))
#print(cols)

print('*********************************************************')
print('** 4.6 Preprocessing within a pipeline')
print('*********************************************************')
# Create arrays for the features and the target: X, y
#X, y = housing_unproc.iloc[:,:-1], housing_unproc.iloc[:,-1]
X, y = housing_unproc.drop('SalePrice', axis=1), housing_unproc.SalePrice

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse = False)),
         ("xgb_model", xgb.XGBRegressor(seed = SEED))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X_train.to_dict('records'), y_train)

# Predict the labels of the test set: preds
preds = xgb_pipeline.predict(X_test.to_dict('records'))

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

print('*********************************************************')
print('** 4.7 Incorporating XGBoost into pipelines')
print('*********************************************************')
# Scikit-learn pipeline example with XGBoost
#X, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]
X, y = boston_data.drop('med_price', axis=1), boston_data.med_price

# if you use Pipeline[] instead of Pipeline(), you get the error
# TypeError: 'ABCMeta' object is not subscriptable
xgb_pipeline = Pipeline([("st_scaler", StandardScaler()),
                         ("xgb_model",xgb.XGBRegressor(seed=SEED))])

scores = cross_val_score(xgb_pipeline, 
                         X, y,
                         scoring="neg_mean_squared_error",
                         cv = 10)

final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))

print("Final XGB RMSE:", final_avg_rmse)

print('*********************************************************')
print('** 4.8 Cross-validating your XGBoost model')
print('*********************************************************')
# Create arrays for the features and the target: X, y
#X, y = housing_unproc.iloc[:,:-1], housing_unproc.iloc[:,-1]
X, y = housing_unproc.drop('SalePrice', axis=1), housing_unproc.SalePrice

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:squarederror", seed=SEED))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline,
                                   X.to_dict('records'), y,
                                   cv = 10,
                                   scoring = "neg_mean_squared_error")

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))

print('*********************************************************')
print('** 4.9 Kidney disease case study I: Categorical Imputer')
print('*********************************************************')
# Exploring the data
print('Exploring the data:')
print(kideney.info())

# Create arrays for the features and the target: X, y
df = kideney.copy(deep = True)
X, y = df.drop('class', axis=1), df['class']

# Apply LabelEncoder to target columns
le = LabelEncoder()
y = le.fit_transform(y)

# Check number of nulls in each feature column
print('\nNulls per column in kideney dataset: ')
print(X.isnull().sum())

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
    [([numeric_feature], SimpleImputer(strategy="median")) for numeric_feature in non_categorical_columns],
    input_df=True, df_out=True
)

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
    [(category_feature, SimpleImputer(strategy='most_frequent')) for category_feature in categorical_columns],
    input_df=True, df_out=True
)

print('*********************************************************')
print('** 4.10 Kidney disease case study II: Feature Union')
print('*********************************************************')
# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
    ("num_mapper", numeric_imputation_mapper),
    ("cat_mapper", categorical_imputation_mapper)
])

print('*********************************************************')
print('** 4.11 Kidney disease case study III: Full pipeline')
print('*********************************************************')
# Custom transformer to convert Pandas DataFrame into Dict (needed for DictVectorizer)
class Dictifier(BaseEstimator, TransformerMixin):   
    """
    Encapsulates converting a DataFrame using .to_dict("records") without you having to do it explicitly 
    (and so that it works in a pipeline). 
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.to_dict('records')

# Initialize an empty list for Simple Imputer details
transformers = []

# Apply numeric imputer
transformers.extend(
    [([numeric_feature], [SimpleImputer(strategy="median"), StandardScaler()]) for numeric_feature in non_categorical_columns]
)

# Apply categorical imputer
transformers.extend(
    [([category_feature], [SimpleImputer(strategy='most_frequent')]) for category_feature in categorical_columns]
)
# Combine the numeric and categorical transformations
numeric_categorical_union = DataFrameMapper(transformers, input_df=True, df_out=True)

# Create full pipeline
pipeline = Pipeline([("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, 
                                               use_label_encoder=False, eval_metric='error', 
                                               max_depth=3, seed=SEED))])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X, y, cv=3, scoring='roc_auc')

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))

print('*********************************************************')
print('** 4.12 Tuning XGBoost hyperparameters')
print('*********************************************************')
# Scikit-learn pipeline example
#X, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]
X, y = boston_data.drop('med_price', axis=1), boston_data.med_price

# Tuning XGBoost hyperparameters in a pipeline
xgb_pipeline = Pipeline([("st_scaler", StandardScaler()), 
                         ("xgb_model",xgb.XGBRegressor(seed=SEED))])
gbm_param_grid = {
    'xgb_model__subsample': np.arange(.05, 1, .05),
    'xgb_model__max_depth': np.arange(3,20,1),
    'xgb_model__colsample_bytree': np.arange(.1,1.05,.05) 
}

randomized_neg_mse = RandomizedSearchCV(
    estimator=xgb_pipeline, 
    param_distributions=gbm_param_grid, 
    n_iter=10,
    scoring='neg_mean_squared_error', 
    cv=4,
    random_state=SEED
)
randomized_neg_mse.fit(X, y)

print("Best rmse: ", np.sqrt(np.abs(randomized_neg_mse.best_score_)))
print("Best model: ", randomized_neg_mse.best_estimator_)

print('*********************************************************')
print('** 4.13 Bringing it all together')
print('*********************************************************')
###########################################################
## Step 1, ex.4.9
###########################################################
# Create arrays for the features and the target: X, y
df = kideney.copy(deep = True)
X, y = df.drop('class', axis=1), df['class']

# Apply LabelEncoder to target columns
le = LabelEncoder()
y = le.fit_transform(y)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

###########################################################
## Step 2, ex.4.11
###########################################################
# Initialize an empty list for Simple Imputer details
transformers = []

# Apply numeric imputer
transformers.extend(
    [([numeric_feature], [SimpleImputer(strategy="median"), StandardScaler()]) for numeric_feature in non_categorical_columns]
)

# Apply categorical imputer
transformers.extend(
    [([category_feature], [SimpleImputer(strategy='most_frequent')]) for category_feature in categorical_columns]
)

# Combine the numeric and categorical transformations
numeric_categorical_union = DataFrameMapper(transformers, input_df=True, df_out=True)

# Create full pipeline
pipeline = Pipeline([("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, 
                                               use_label_encoder=False, eval_metric='error', 
                                               max_depth=3, seed=SEED))])

###########################################################
## Step 3, ex.4.13
###########################################################
# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth'    : np.arange(3, 10, 1),
    'clf__n_estimators' : np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=gbm_param_grid,
    n_iter=2,
    cv=2,
    scoring='roc_auc',
    verbose=1,
    random_state=SEED
)

# Fit the estimator
randomized_roc_auc.fit(X, y)

# Compute metrics
print(randomized_roc_auc.best_score_)
print(randomized_roc_auc.best_estimator_)

print('*********************************************************')
print('** 4.14 Final Thoughts')
print('*********************************************************')
print('END')
print('*********************************************************')

# Return to default values
pd.set_option("display.max_columns", 0)