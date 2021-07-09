# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 22:01:19 2019

@author: Datacamp
https://github.com/datacamp/course-resources-ml-with-experts-budgets/blob/master/notebooks/1.0-full-model.ipynb
"""
from __future__ import division
from __future__ import print_function

print("*************************************************************************")
print("BEGIN")
print("*************************************************************************")
print("1. From Raw Data to Predictions")
print("*************************************************************************\n")
#This notebook is designed as a follow-up to the Machine Learning with the Experts: School Budgets course on Datacamp. 
#We won't explain all the tools and techniques we use here. If you're curious about any of the tools, code, or methods 
#used here, make sure to check out the course!

# ignore deprecation warnings in sklearn
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

#import os
#import sys

# add the 'src' directory as one where we can import modules
#src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
#sys.path.append(src_dir)

#Librerias creadas durante el curso
#from multilabel import multilabel_sample_dataframe, multilabel_train_test_split
#from SparseInteractions import SparseInteractions
#from multi_log_loss import score_submission

#Librerías del ganador
from full_model_multilabel import multilabel_sample_dataframe, multilabel_train_test_split
from full_model_sparseInteractions import SparseInteractions
from full_model_metrics import multi_multi_log_loss




print("*************************************************************************")
print("2. Load Data")
print("*************************************************************************\n")
#First, we'll load the entire training data set available from DrivenData. In order to make this notebook run, you will need to:
#Sign up for an account on DrivenData
#Join the Box-plots for education competition
#Download the competition data to the data folder in this repository. Files should be named TrainingSet.csv and TestSet.csv.
#Enjoy!
#path_to_training_data = os.path.join(os.pardir, 'data', 'TrainingSet.csv')
path_to_training_data= 'TrainingSet.csv'

df = pd.read_csv(path_to_training_data, index_col=0)
print(df.shape)


print("*************************************************************************")
print("3. Resample Data")
print("*************************************************************************\n")
#400,277 rows is too many to work with locally while we develop our approach. We'll sample down to 10,000 rows so that 
#it is easy and quick to run our analysis.
#We'll also create dummy variables for our labels and split our sampled dataset into a training set and a test set.

LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']

NON_LABELS = [c for c in df.columns if c not in LABELS]

SAMPLE_SIZE = 40000

sampling = multilabel_sample_dataframe(df, pd.get_dummies(df[LABELS]), size=SAMPLE_SIZE, min_count=25, seed=43)

dummy_labels = pd.get_dummies(sampling[LABELS])

X_train, X_test, y_train, y_test = multilabel_train_test_split(sampling[NON_LABELS], dummy_labels, 0.2, min_count=3, seed=43)


print("*************************************************************************")
print("4. Create preprocessing tools")
print("*************************************************************************\n")
#We need tools to preprocess our text and numeric data. We'll create those tools here. The combine_text_columns function will 
#take a DataFrame of text columns and return a single series where all of the text in the columns has been joined together.
#We'll then create FunctionTransformer objects that select our text and numeric data from the dataframe.
#Finally, we create a custom scoring method that uses the multi_multi_log_loss function that is the evaluation metric for the competition.

NUMERIC_COLUMNS = ['FTE', "Total"]

def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ Takes the dataset as read in, drops the non-feature, non-text columns and
        then combines all of the text columns into a single vector that has all of
        the text for a row.
        
        :param data_frame: The data as read in with read_csv (no preprocessing necessary)
        :param to_drop (optional): Removes the numeric and label columns by default.
    """
    # drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
    
    # replace nans with blanks
    text_data.fillna("", inplace=True)
    
    # joins all of the text items in a row (axis=1)
    # with a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

from sklearn.preprocessing import FunctionTransformer

get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

print(get_text_data.fit_transform(sampling.head(5)))
print(get_numeric_data.fit_transform(sampling.head(5)))

from sklearn.metrics.scorer import make_scorer

log_loss_scorer = make_scorer(multi_multi_log_loss)


print("*************************************************************************")
print("5. Train model pipeline")
print("*************************************************************************\n")
#Now we'll train the final pipeline from the course that takes text and numeric data, does the necessary preprocessing, 
#and trains the classifier.
from sklearn.feature_selection import chi2, SelectKBest

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MaxAbsScaler

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# set a reasonable number of features before adding interactions
chi_k = 300

# create the pipeline object
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     alternate_sign=False, norm=None, binary=False,
                                                     ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# fit the pipeline to our training data
pl.fit(X_train, y_train.values)

# print the score of our trained pipeline on our test set
print("Logloss score of trained pipeline: ", log_loss_scorer(pl, X_test, y_test.values))


print("*************************************************************************")
print("6. Predict holdout set and write submission")
print("*************************************************************************\n")
#Finally, we want to use our trained pipeline to predict the holdout dataset. We will write our predictions to a file, 
#predictions.csv, that we can submit on DrivenData!
#path_to_holdout_data = os.path.join(os.pardir,'data','TestSet.csv')
path_to_holdout_data = 'TestSet.csv'

# Load holdout data
holdout = pd.read_csv(path_to_holdout_data, index_col=0)

# Make predictions
predictions = pl.predict_proba(holdout)

# Format correctly in new DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns, index=holdout.index, data=predictions)

# Save prediction_df to csv called "predictions.csv"
prediction_df.to_csv("predictions.csv")
warnings.filterwarnings("default")

print("*************************************************************************")
print("END")
print("*************************************************************************")
