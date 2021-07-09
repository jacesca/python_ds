# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 23:28:19 2019

@author: jacqueline.cortez

Chapter 3. Going Deeper
Introduction:
    Convolutional neural networks gain a lot of power when they are constructed with multiple layers (deep networks). 
    In this chapter, you will learn how to stack multiple convolutional layers into a deep network. You will also learn 
    how to keep track of the number of parameters, as the network grows, and how to control this number.
"""

# Import packages                          
#import pandas            as pd                                                #For loading tabular data
import numpy             as np                                                #For making operations in lists
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt                                               #For creating charts
#import seaborn           as sns                                               #For visualizing data
#import statsmodels       as sm                                                #For stimations in differents statistical models
#import networkx          as nx                                                #For Network Analysis in Python
#import nxviz             as nv                                                #For Network Analysis in Python


#import scykit-learn                                                           #For performing machine learning  
#import tabula                                                                 #For extracting tables from pdf
#import nltk                                                                   #For working with text data
#import math                                                                   #For accesing to a complex math operations
#import random                                                                 #For generating random numbers
#import calendar                                                               #For accesing to a vary of calendar operations
#import re                                                                     #For regular expressions
#import timeit                                                                 #For Measure execution time of small code snippets
#import time                                                                   #To measure the elapsed wall-clock time between two points
#import warnings
#import wikipedia


#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
#from math                            import radian                            #For accessing a specific math operations
#from functools                       import reduce                            #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types                import CategoricalDtype                  #For categorical data
#from glob                            import glob                              #For using with pathnames matching
#from datetime                        import date                              #For obteining today function
#from datetime                        import datetime                          #For obteining today function
#from string                          import Template                          #For working with string, regular expressions
#from itertools                       import cycle                             #Used in the function plot_labeled_decision_regions()
#from math                            import floor                             #Used in the function plot_labeled_decision_regions()
#from math                            import ceil                              #Used in the function plot_labeled_decision_regions()
#from itertools                       import combinations                      #For iterations
#from collections                     import defaultdict                       #Returns a new dictionary-like object
#from nxviz import ArcPlot                                                     #For Network Analysis in Python
#from nxviz import CircosPlot                                                  #For Network Analysis in Python 
#from nxviz import MatrixPlot                                                  #For Network Analysis in Python 


#import scipy.stats as stats                                                   #For accesign to a vary of statistics functiosn
#from scipy.cluster.hierarchy         import fcluster                          #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import dendrogram                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import linkage                           #For learning machine - unsurpervised
#from scipy.sparse                    import csr_matrix                        #For learning machine 
#from scipy.special                   import expit as sigmoid                  #For learning machine 
#from scipy.stats                     import pearsonr                          #For learning machine 
#from scipy.stats                     import randint                           #For learning machine 
       

#from sklearn                         import datasets                          #For learning machine
#from sklearn.cluster                 import KMeans                            #For learning machine - unsurpervised
#from sklearn.decomposition           import NMF                               #For learning machine - unsurpervised
#from sklearn.decomposition           import PCA                               #For learning machine - unsurpervised
#from sklearn.decomposition           import TruncatedSVD                      #For learning machine - unsurpervised
#from sklearn.ensemble                import AdaBoostClassifier                #For learning machine - surpervised
#from sklearn.ensemble                import BaggingClassifier                 #For learning machine - surpervised
#from sklearn.ensemble                import GradientBoostingRegressor         #For learning machine - surpervised
#from sklearn.ensemble                import RandomForestClassifier            #For learning machine
#from sklearn.ensemble                import RandomForestRegressor             #For learning machine - unsurpervised
#from sklearn.ensemble                import VotingClassifier                  #For learning machine - unsurpervised
#from sklearn.feature_extraction.text import TfidfVectorizer                   #For learning machine - unsurpervised
#from sklearn.feature_selection       import chi2                              #For learning machine
#from sklearn.feature_selection       import SelectKBest                       #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                   #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                 #For learning machine
#from sklearn.impute                  import SimpleImputer                     #For learning machine
#from sklearn.linear_model            import ElasticNet                        #For learning machine
#from sklearn.linear_model            import Lasso                             #For learning machine
#from sklearn.linear_model            import LinearRegression                  #For learning machine
#from sklearn.linear_model            import LogisticRegression                #For learning machine
#from sklearn.linear_model            import Ridge                             #For learning machine
#from sklearn.manifold                import TSNE                              #For learning machine - unsurpervised
#from sklearn.metrics                 import accuracy_score                    #For learning machine
#from sklearn.metrics                 import classification_report             #For learning machine
#from sklearn.metrics                 import confusion_matrix                  #For learning machine
#from sklearn.metrics                 import mean_squared_error as MSE         #For learning machine
#from sklearn.metrics                 import roc_auc_score                     #For learning machine
#from sklearn.metrics                 import roc_curve                         #For learning machine
#from sklearn.model_selection         import cross_val_score                   #For learning machine
#from sklearn.model_selection         import GridSearchCV                      #For learning machine
#from sklearn.model_selection         import RandomizedSearchCV                #For learning machine
#from sklearn.model_selection         import train_test_split                  #For learning machine
#from sklearn.multiclass              import OneVsRestClassifier               #For learning machine
#from sklearn.neighbors               import KNeighborsClassifier as KNN       #For learning machine
#from sklearn.pipeline                import FeatureUnion                      #For learning machine
#from sklearn.pipeline                import make_pipeline                     #For learning machine - unsurpervised
#from sklearn.pipeline                import Pipeline                          #For learning machine
#from sklearn.preprocessing           import FunctionTransformer               #For learning machine
#from sklearn.preprocessing           import Imputer                           #For learning machine
#from sklearn.preprocessing           import MaxAbsScaler                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing           import Normalizer                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing           import normalize                         #For learning machine - unsurpervised
#from sklearn.preprocessing           import scale                             #For learning machine
#from sklearn.preprocessing           import StandardScaler                    #For learning machine
#from sklearn.svm                     import SVC                               #For learning machine
#from sklearn.tree                    import DecisionTreeClassifier            #For learning machine - supervised
#from sklearn.tree                    import DecisionTreeRegressor             #For learning machine - supervised

#import statsmodels.api as sm                                                  #Make a prediction model
#import statsmodels.formula.api as smf                                         #Make a prediction model    
import tensorflow as tf                                                       #For DeapLearning

#import keras                                                                 #For DeapLearning
from keras.callbacks                 import EarlyStopping                     #For DeapLearning
#from keras.layers                    import BatchNormalization                #For DeapLearning
#from keras.layers                    import Concatenate                       #For DeapLearning
from keras.layers                    import Conv2D                            #For DeapLearning
from keras.layers                    import Dense                             #For DeapLearning
#from keras.layers                    import Embedding                         #For DeapLearning
from keras.layers                    import Flatten                           #For DeapLearning
#from keras.layers                    import Input                             #For DeapLearning
from keras.layers                    import MaxPool2D                         #For DeapLearning
#from keras.layers                    import Subtract                          #For DeapLearning
#from keras.models                    import load_model                        #For DeapLearning
#from keras.models                    import Model                             #For DeapLearning
from keras.models                    import Sequential                        #For DeapLearning
from keras.optimizers                import Adam                              #For DeapLearning
#from keras.optimizers                import SGD                               #For DeapLearning
from keras.utils                     import plot_model                        #For DeapLearning
from keras.utils                     import to_categorical                    #For DeapLearning


#from bokeh.io                        import curdoc                            #For interacting visualizations
#from bokeh.io                        import output_file                       #For interacting visualizations
#from bokeh.io                        import show                              #For interacting visualizations
#from bokeh.plotting                  import ColumnDataSource                  #For interacting visualizations
#from bokeh.plotting                  import figure                            #For interacting visualizations
#from bokeh.layouts                   import row                               #For interacting visualizations
#from bokeh.layouts                   import widgetbox                         #For interacting visualizations
#from bokeh.layouts                   import column                            #For interacting visualizations
#from bokeh.layouts                   import gridplot                          #For interacting visualizations
#from bokeh.models                    import HoverTool                         #For interacting visualizations
#from bokeh.models                    import ColumnDataSource                  #For interacting visualizations
#from bokeh.models                    import CategoricalColorMapper            #For interacting visualizations
#from bokeh.models                    import Slider                            #For interacting visualizations
#from bokeh.models                    import Select                            #For interacting visualizations
#from bokeh.models                    import Button                            #For interacting visualizations
#from bokeh.models                    import CheckboxGroup                     #For interacting visualizations
#from bokeh.models                    import RadioGroup                        #For interacting visualizations
#from bokeh.models                    import Toggle                            #For interacting visualizations
#from bokeh.models.widgets            import Tabs                              #For interacting visualizations
#from bokeh.models.widgets            import Panel                             #For interacting visualizations
#from bokeh.palettes                  import Spectral6                         #For interacting visualizations


# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters()                                              #Require to explicitly register matplotlib converters.

#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.

#Setting images params
#plt.rcParams.update({'figure.max_open_warning': 0})                           #To solve the max images open


print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** User defined variables \n")

SEED=1
np.random.seed(SEED)
tf.set_random_seed(SEED)

#print("****************************************************")
print("** User Functions\n")

def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3], [0.2989, 0.5870, 0.1140])

#print("****************************************************")
print("** Getting the data for this program\n")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
img_rows, img_cols = 28, 28
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #Data normalization

plt.pcolor(x_train[10],  cmap='gray') # Visualize the result
plt.gca().invert_yaxis()
plt.title('Figure 1')
plt.suptitle("0. Fashion MNIST Database")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.figure(figsize=(6,6))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
plt.suptitle("0. Fashion MNIST Database")
plt.show()

print("****************************************************")
tema = "1. Going deeper"; print("** %s\n" % tema)

train_data   = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
train_labels = to_categorical(y_train, len(class_names))
test_data    = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
test_labels  = to_categorical(y_test, len(class_names))

model = Sequential() # Initialize the model
model.add(Conv2D(64, kernel_size=2, activation='relu', input_shape=(img_rows, img_cols, 1), padding='same', name='Input')) # Add the convolutional layer
model.add(Conv2D(64, kernel_size=2, activation='relu', name='Conv2D')) # Add the convolutional layer
model.add(Flatten(name='Flatten')) # Feed into output layer
model.add(Dense(10, activation='softmax', name='Output'))

print(model.summary())
plot_model(model, to_file='03_01_model.png', show_shapes=False, show_layer_names=True, rankdir='TB') # rankdir='TB' makes vertical plot and rankdir='LR' creates a horizontal plot

# Plotting the model
plt.figure()
data = plt.imread('03_01_model.png') # Display the image
plt.imshow(data)
plt.title('A MNIST case in a Neural Red')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model
model.fit(train_data, train_labels,  
          epochs=10, verbose=True, validation_split=.20, batch_size=64, callbacks=[EarlyStopping(patience=2)]) # Fit the model
print(model.evaluate(test_data, test_labels, batch_size=10)) # Evaluate the model on separate test data

print("****************************************************")
tema = "2. Creating a deep learning network"; print("** %s\n" % tema)

model = Sequential()
model.add(Conv2D(64, kernel_size=2, activation='relu', input_shape=(img_rows, img_cols, 1), name='Input')) # Add a convolutional layer (15 units)
model.add(Conv2D(20, kernel_size=2, activation='relu', name='Conv2D')) # Add another convolutional layer (5 units)
model.add(Flatten(name='Flatten')) # Flatten and feed to output layer
model.add(Dense(10, activation='softmax', name='Output'))

print(model.summary())

print("****************************************************")
tema = "3. Train a deep CNN to classify clothing images"; print("** %s\n" % tema)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile model
model.fit(train_data, train_labels, validation_split=0.2, epochs=10, batch_size=64) # Fit the model to training data 
print(model.evaluate(test_data, test_labels, batch_size=64)) # Evaluate the model on test data

print("****************************************************")
tema = "7. How many parameters in a deep CNN?"; print("** %s\n" % tema)

# CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=2, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Summarize the model 
print(model.summary())

print("****************************************************")
tema = "9. Write your own pooling operation"; print("** %s\n" % tema)

data = plt.imread('bricks.png') # Load the image
im   = rgb2gray(data)

plt.figure()
plt.pcolor(im,  cmap='gray') # Visualize the result
plt.title('bricks.png (Original image)')
plt.suptitle(tema)
plt.show()

# Result placeholder
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2, jj*2:jj*2+2])
        
plt.figure()
plt.pcolor(result,  cmap='gray') # Visualize the result
plt.title('bricks.png (After pooling)')
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "10. Keras pooling layers"; print("** %s\n" % tema)


model = Sequential()
model.add(Conv2D(64, kernel_size=2, activation='relu', input_shape=(img_rows, img_cols, 1), name='Input')) # Add a convolutional layer (15 units)
model.add(MaxPool2D(2, name='MaxPool2D-1')) # Add a pooling operation
model.add(Conv2D(20, kernel_size=2, activation='relu', name='Conv2D')) # Add another convolutional layer (5 units)
model.add(Flatten(name='Flatten')) # Flatten and feed to output layer
model.add(Dense(10, activation='softmax', name='Output'))

print(model.summary())

print("****************************************************")
tema = "11. Train a deep CNN with pooling to classify images"; print("** %s\n" % tema)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile model
model.fit(train_data, train_labels, validation_split=0.2, epochs=10, batch_size=64) # Fit the model to training data 
print(model.evaluate(test_data, test_labels, batch_size=64)) # Evaluate the model on test data

print("****************************************************")
print("** END                                            **")
print("****************************************************")