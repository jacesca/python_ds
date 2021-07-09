# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:18:02 2019

@author: jacqueline.cortez

Capítulo 1. Classification
Introduction:
    In this chapter, you will be introduced to classification problems and learn how to solve them using supervised learning techniques. 
    Classification problems are prevalent in a variety of domains, ranging from finance to healthcare. Here, you will have the chance to 
    apply what you are learning to a political dataset, where you classify the party affiliation of United States Congressmen based on their 
    voting records.
"""

# Import packages
import pandas as pd                   #For loading tabular data
import numpy as np                    #For making operations in lists
#import matplotlib as mpl              #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt       #For creating charts
import seaborn as sns                 #For visualizing data
#import scipy.stats as stats          #For accesign to a vary of statistics functiosn
#import statsmodels as sm             #For stimations in differents statistical models
#import scykit-learn                  #For performing machine learning  
#import tabula                        #For extracting tables from pdf
#import nltk                          #For working with text data
#import math                          #For accesing to a complex math operations
#import random                        #For generating random numbers
#import calendar                      #For accesing to a vary of calendar operations
#import re                             #For regular expressions

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                        #For obteining today function
#from string import Template                                                          #For working with string, regular expressions
from sklearn import datasets                                                          #For learning machine
from sklearn.neighbors import KNeighborsClassifier                                    # Import KNeighborsClassifier from sklearn.neighbors
from sklearn.model_selection import train_test_split

#from bokeh.io import curdoc, output_file, show                                      #For interacting visualizations
#from bokeh.plotting import figure, ColumnDataSource                                 #For interacting visualizations
#from bokeh.layouts import row, widgetbox, column, gridplot                          #For interacting visualizations
#from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper        #For interacting visualizations
#from bokeh.models import Slider, Select, Button, CheckboxGroup, RadioGroup, Toggle  #For interacting visualizations
#from bokeh.models.widgets import Tabs, Panel                                        #For interacting visualizations
#from bokeh.palettes import Spectral6                                                #For interacting visualizations

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters() #Require to explicitly register matplotlib converters.

#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09

#Setting the numpy options
#np.set_printoptions(precision=3) #precision set the precision of the output:
#np.set_printoptions(suppress=True) #suppress suppresses the use of scientific notation for small numbers

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

file = "house-votes-84.csv" 
vote_df = pd.read_csv(file, header = None, na_values='?',
                      names = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
                                 'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
                                 'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])
vote_df.fillna('n', inplace=True)
vote_df.replace('n',0, inplace=True)
vote_df.replace('y',1, inplace=True)

print("****************************************************")
tema = '3. Exploratory data analysis'; print("** %s\n" % tema)

iris = datasets.load_iris()
print(type(iris))
print(iris.keys())
print(type(iris.data), type(iris.target))
print(iris.data.shape)
print(iris.target_names)

x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
plt.style.use('ggplot')

print(df.head())

pd.plotting.scatter_matrix(df, c=y, figsize=[8,8], s=150, marker='D')
plt.style.use('default')


print("****************************************************")
tema = '5. Visual EDA'; print("** %s\n" % tema)

plt.figure()
plt.style.use('ggplot')
sns.set(font_scale=0.8)

plt.subplot(2,2,1)
sns.countplot(x='education', hue='party', data=vote_df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.title("Education Votation")

plt.subplot(2,2,2)
sns.countplot(x='satellite', hue='party', data=vote_df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.title("Satellite Votation")

plt.subplot(2,2,3)
sns.countplot(x='missile', hue='party', data=vote_df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.title("Satellite Votation")

plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.10, right=None, top=0.90, wspace=0.5, hspace=0.7)
plt.show()
plt.style.use('default')


print("****************************************************")
tema = "7. k-Nearest Neighbors: Fit"; print("** %s\n" % tema)

y = vote_df['party'].values # Create arrays for the features and the response variable
X = vote_df.drop('party', axis=1).values

knn = KNeighborsClassifier(n_neighbors=6) # Create a k-NN classifier with 6 neighbors
knn.fit(X, y) # Fit the classifier to the data



print("****************************************************")
tema = "8. k-Nearest Neighbors: Predict"; print("** %s\n" % tema)

np.random.seed(42)
#X_new = np.random.random(size=16)
X_new = np.random.randint(2, size=16)

y_pred = knn.predict(X) # Predict the labels for the training data X
new_prediction = knn.predict([X_new]) # Predict and print the label for the new data point X_new
print("X_new: {}".format(X_new))
print("Prediction: {}".format(new_prediction))
aciertos = sum([a==b for a, b in zip(y, y_pred)])
print("X Data Aciertos: {0} de {1} ({2:0.2f}%).".format(aciertos, len(y), aciertos/len(y)*100))



print("****************************************************")
tema = "10. The digits recognition dataset"; print("** %s\n" % tema)

digits = datasets.load_digits() # Load the digits dataset: digits
print(digits.keys()) # Print the keys and DESCR of the dataset
#print(digits.DESCR)
print(digits.images.shape) # Print the shape of the images and data keys
print(digits.data.shape)
print(digits.target_names)

plt.figure()
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest') # Display digit 1010
plt.show()


print("****************************************************")
tema = "11. Train/Test Split + Fit/Predict/Accuracy"; print("** %s\n" % tema)


# Create feature and target arrays
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42, stratify=y) # Split into training and test set
knn = KNeighborsClassifier(n_neighbors=7) # Create a k-NN classifier with 7 neighbors: knn
knn.fit(X_train, y_train) # Fit the classifier to the training data

# Print the accuracy
print(knn.score(X_test, y_test))



print("****************************************************")
tema = '12. Overfitting and underfitting'; print("** %s\n" % tema)

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
sns.set() # Set default Seaborn style
plt.figure()
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('k-NN: Varying Number of Neighbors')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.32, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


print("****************************************************")
print("** END                                            **")
print("****************************************************")