# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:09:21 2019

@author: jacqueline.cortez

Chapter 1. Introducing Keras
Introduction:
    In this first chapter, you will get introduced to neural networks, understand what kind of 
    problems they can solve, and when to use them. You will also build several networks and save 
    the earth by training a regression model that approximates the orbit of a meteor that is 
    approaching us!
"""
print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Importing libraries \n")

import matplotlib.pyplot as plt                                               #For creating charts
import numpy             as np                                                #For making operations in lists
import pandas            as pd                                                #For loading tabular data
import tensorflow as tf                                                       #For DeapLearning

from keras.layers                    import Dense                             #For DeapLearning
from keras.models                    import Sequential                        #For DeapLearning
from keras.optimizers                import Adam                              #For DeapLearning
from keras.utils                     import plot_model                        #For DeapLearning

print("****************************************************")
print("** Preparing the environment \n")

SEED=42
np.random.seed(SEED)
tf.set_random_seed(SEED)

print("****************************************************")
print("** User functions \n")

def plot_orbit(model_preds, name, tema):
    plt.figure()
    axeslim = int(len(model_preds)/2)
    plt.plot(np.arange(-axeslim, axeslim + 1),
             np.arange(-axeslim, axeslim + 1)**2, 
             color="mediumslateblue", alpha=0.5)
    plt.plot(np.arange(-axeslim, axeslim + 1), model_preds, color="orange", alpha=0.5)
    plt.xlabel('Time step')
    plt.ylabel('Coordinate')
    plt.axis([-40, 41, -5, 550])
    plt.legend(["Scientist's Orbit", 'Your orbit'],loc="lower left")
    plt.title("{} Model orbit vs Scientist's Orbit".format(name))
    plt.suptitle(tema)
    plt.show()
  
print("****************************************************")
tema = "Reading the data"; print("** %s\n" % tema)

file = 'meteor_orbit_train.csv'
orbit_data_train = pd.read_csv(file, sep=';')
X_feature = orbit_data_train.Time_step.values
y_coordinate = orbit_data_train.Coordinate.values 

plt.plot(X_feature, y_coordinate)
plt.xlabel('Time step')
plt.ylabel('Coordinate')
plt.title('The data register by the scientist')
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "9. Specifying a model"; print("** %s\n" % tema)

model = Sequential() # Instantiate a Sequential model
model.add(Dense(50, input_shape=(1,), activation='relu', name="Dense")) # Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, activation='relu', name="Dense_2")) # Add two Dense layers with 50 neurons and relu activation
model.add(Dense(50, activation='relu', name="Dense_3")) 
model.add(Dense(1, name='Output')) # End your model with a Dense layer and no activation

model.summary() # Summarize your model
plot_model(model, to_file='01_09_model.png', show_shapes=False, show_layer_names=True, rankdir='TB') # rankdir='TB' makes vertical plot and rankdir='LR' creates a horizontal plot

# Plotting the model
plt.figure()
data = plt.imread('01_09_model.png') # Display the image
plt.imshow(data)
plt.axis('off');
plt.title('A simple regression model to predict the meteor impact')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

print("****************************************************")
tema = "10. Training"; print("** %s\n" % tema)

model.compile(optimizer=Adam(lr=0.01), loss='mse') # Compile your model
print("Training started..., this can take a while:")

training = model.fit(X_feature, y_coordinate, epochs=30) # Fit your model on your data for 30 epochs
print("Final lost value:",model.evaluate(X_feature, y_coordinate)) # Evaluate your model 

plt.figure()
plt.plot(training.history['loss']) # Plot the training loss 
plt.ylabel('loss function: mse')
plt.xlabel('epochs')
plt.title('Evaluation results in each epoch')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

print("****************************************************")
tema = "11. Predicting the orbit!"; print("** %s\n" % tema)

ten_min_orbit = model.predict(np.arange(-10, 11)) # Predict the twenty minutes orbit
plot_orbit(ten_min_orbit, name='10 min', tema=tema) # Plot the twenty minute orbit 

twenty_min_orbit = model.predict(np.arange(-20, 21)) # Predict the twenty minutes orbit
plot_orbit(twenty_min_orbit, name='20 min', tema=tema) # Plot the twenty minute orbit 

eighty_min_orbit = model.predict(np.arange(-40, 41)) # Predict the eighty minute orbit
plot_orbit(eighty_min_orbit, '40 min', tema=tema) # Plot the eighty minute orbit 

plt.figure()
plt.plot(X_feature, y_coordinate, 'b--', alpha=0.5, label='Data register by scientist')
plt.plot(np.arange(-10, 11), ten_min_orbit, color='red', alpha=0.5, label='Predictions')
plt.xlabel('Time step')
plt.ylabel('Coordinate')
plt.legend()
plt.title('Showing predictions')
plt.suptitle(tema)
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")



"""
print("****************************************************")
tema = "5. Hello nets!"; print("** %s\n" % tema)

model = Sequential() # Create a Sequential model
model.add(Dense(10, input_shape=(2,), activation="relu")) # Add an input layer and a hidden layer with 10 neurons
model.add(Dense(1)) # Add a 1-neuron output layer
model.summary() # Summarise your model

"""
"""
print("****************************************************")
tema = "6. Counting parameters"; print("** %s\n" % tema)

model = Sequential() # Instantiate a new Sequential model
model.add(Dense(5, input_shape=(3,), activation="relu")) # Add a Dense layer with five neurons and three inputs
model.add(Dense(1)) # Add a final Dense layer with one neuron and no activation
model.summary() # Summarize your model
"""
"""
print("****************************************************")
tema = "7. Build as shown!"; print("** %s\n" % tema)

model = Sequential() # Instantiate a Sequential model
model.add(Dense(3, input_shape=(2,), activation='relu')) # Build the input and hidden layer
model.add(Dense(1)) # Add the ouput layer
model.summary() # Summarize your model
"""
#import pandas            as pd                                                #For loading tabular data
#import numpy             as np                                                #For making operations in lists
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt                                               #For creating charts
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
#from scipy.ndimage                   import gaussian_filter                   #For working with images
#from scipy.ndimage                   import median_filter                     #For working with images
#from scipy.signal                    import convolve2d                        #For learning machine - deep learning
#from scipy.sparse                    import csr_matrix                        #For learning machine 
#from scipy.special                   import expit as sigmoid                  #For learning machine 
#from scipy.stats                     import pearsonr                          #For learning machine 
#from scipy.stats                     import randint                           #For learning machine 
       
#from skimage                         import exposure                          #For working with images
#from skimage                         import measure                           #For working with images
#from skimage.filters.thresholding    import threshold_otsu                    #For working with images
#from skimage.filters.thresholding    import threshold_local                   #For working with images 

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
#import tensorflow as tf                                                       #For DeapLearning

#import keras                                                                  #For DeapLearning
#from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
#from keras.callbacks                 import EarlyStopping                     #For DeapLearning
#from keras.layers                    import BatchNormalization                #For DeapLearning
#from keras.layers                    import Concatenate                       #For DeapLearning
#from keras.layers                    import Conv2D                            #For DeapLearning
#from keras.layers                    import Dense                             #For DeapLearning
#from keras.layers                    import Dropout                           #For DeapLearning
#from keras.layers                    import Embedding                         #For DeapLearning
#from keras.layers                    import Flatten                           #For DeapLearning
#from keras.layers                    import Input                             #For DeapLearning
#from keras.layers                    import MaxPool2D                         #For DeapLearning
#from keras.layers                    import Subtract                          #For DeapLearning
#from keras.models                    import load_model                        #For DeapLearning
#from keras.models                    import Model                             #For DeapLearning
#from keras.models                    import Sequential                        #For DeapLearning
#from keras.optimizers                import Adam                              #For DeapLearning
#from keras.optimizers                import SGD                               #For DeapLearning
#from keras.utils                     import plot_model                        #For DeapLearning
#from keras.utils                     import to_categorical                    #For DeapLearning


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
#pd.set_option('display.max_rows', -1)                                         #Shows all rows
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