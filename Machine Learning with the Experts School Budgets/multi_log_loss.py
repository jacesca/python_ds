# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:44:23 2019

@author: Datacamp
file: multi_log_loss.py
https://blog.csdn.net/u011292816/article/details/97865973
"""

import numpy as np
import pandas as pd


#def _multi_multi_log_loss(predicted, actual, class_column_indices=BOX_PLOTS_COLUMN_INDICES, eps=1e-15):
def _multi_multi_log_loss(predicted, actual, class_column_indices, eps=1e-15):
    """ Multi class version of Logarithmic Loss metric as implemented on
    DrivenData.org
    """
    class_scores = np.ones(len(class_column_indices), dtype=np.float64)

    # calculate log loss for each set of columns that belong to a class:
    for k, this_class_indices in enumerate(class_column_indices):
    # get just the columns for this class
        preds_k = predicted[:, this_class_indices].astype(np.float64)

        # normalize so probabilities sum to one (unless sum is zero, then we clip)
        preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.inf)

        actual_k = actual[:, this_class_indices]

        # shrink predictions so
        y_hats = np.clip(preds_k, eps, 1 - eps)
        sum_logs = np.sum(actual_k * np.log(y_hats))
        class_scores[k] = (-1.0 / actual.shape[0]) * sum_logs

    return np.average(class_scores)


#def score_submission(pred_path=PATH_TO_PREDICTIONS, holdout_path=PATH_TO_HOLDOUT_LABELS):
def score_submission(pred_path, holdout_path, column_indices):
    # this happens on the backend to get the score
    holdout_labels = pd.get_dummies(pd.read_csv(holdout_path, index_col=0).apply(lambda x: x.astype('category'), axis=0), prefix_sep='__')

    preds = pd.read_csv(pred_path, index_col=0)

    # make sure that format is correct
    #print(preds.columns)
    #print(holdout_labels.columns)
    assert (preds.columns == holdout_labels.columns).all()
    assert (preds.index == holdout_labels.index).all()

    return _multi_multi_log_loss(preds.values, holdout_labels.values, class_column_indices=column_indices)