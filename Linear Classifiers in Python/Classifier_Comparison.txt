# -*- coding: utf-8 -*-
"""
Created locally for test on Mon Jun 15 11:23:33 2020

ORiginal code source: Gaël Varoquaux, Andreas Müller
Original documentation by Jaques Grobler
Modify by Jacesca
License: BSD 3 clause
Source: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
"""

###############################################################################
# Importing the necessary libraries                                           #
###############################################################################
import numpy as np

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

###############################################################################
# Defining the models to apply                                                #
###############################################################################
names       = ["Nearest Neighbors", 
               "Linear SVM", 
               "RBF SVM", 
               "Gaussian Process",
               "Decision Tree", 
               "Random Forest", 
               "Neural Net", 
               "AdaBoost",
               "Naive Bayes", 
               "QDA"]

classifiers = [KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025),
               SVC(gamma=2, C=1),
               GaussianProcessClassifier(1.0 * RBF(1.0)),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               MLPClassifier(alpha=1, max_iter=1000),
               AdaBoostClassifier(),
               GaussianNB(),
               QuadraticDiscriminantAnalysis()]
SEED         = 42

###############################################################################
# Preparing the data                                                          #
###############################################################################
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]


###############################################################################
# Setting the plot                                                            #
###############################################################################
suptitle_param = dict(color='darkblue', fontsize=10)
title_param    = {'color': 'darkred', 'fontsize': 8, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 
                  'legend.fontsize': 6, 'font.size': 5}
figsize        = (12.1, 5.9)
plt.rcParams.update(**plot_param)

fig            = plt.figure(figsize=figsize)

###############################################################################
# Begin the process                                                           #
###############################################################################
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    
    # print(len(datasets), len(classifiers))
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers), i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        plot_decision_regions(X_train, y_train, clf, 
                              scatter_kwargs={'s':20},
                              legend=0, ax=ax)
        t = ax.text(0.1, 0.1, score, transform=ax.transAxes, color='black', ha='left', va='top')
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
        if i<11: ax.set_title(name, **title_param)
        i += 1

fig.suptitle('Classifier Comparison', **suptitle_param)
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=.9, wspace=None, hspace=.4); #To set the margins 
plt.show()
#plt.savefig("classifier_comparison.jpg")