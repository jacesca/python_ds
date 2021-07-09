# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:42:57 2019

@author: jacqueline.cortez

Chapter 2. Going Deeper
Introduction:
    By the end of this chapter, you will know how to solve binary, multi-class, and multi-label problems with neural networks. 
    All of this by solving problems like detecting fake dollar bills, deciding who threw which dart at a board, and building an 
    intelligent system to water your farm. You will also be able to plot model training metrics and to stop training and save your 
    models when they no longer improve.
    MULTICLASS CLASSIFICATION EXAMPLE CODE
"""
print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Importing libraries \n")

import numpy             as np                                                #For making operations in lists
import pandas            as pd                                                #For loading tabular data
import matplotlib.pyplot as plt                                               #For creating charts
import seaborn           as sns                                               #For visualizing data
import tensorflow as tf                                                       #For DeapLearning

#from pandas.api.types                import CategoricalDtype                  #For categorical data

from keras.layers                    import Dense                             #For DeapLearning
from keras.models                    import Sequential                        #For DeapLearning
from keras.utils                     import plot_model                        #For DeapLearning
from keras.utils                     import to_categorical                    #For DeapLearning

from sklearn.model_selection         import train_test_split                  #For learning machine

print("****************************************************")
print("** Preparing the environment \n")

SEED=42
np.random.seed(SEED)
np.set_printoptions(suppress=True)
tf.set_random_seed(SEED)

print("****************************************************")
print("** User functions \n")

def model_display(model, sup_title, file_name):
    print(model.summary()) # Summarize your model
    
    plot_model(model, to_file=file_name, show_shapes=False, show_layer_names=True, rankdir='TB') # rankdir='TB' makes vertical plot and rankdir='LR' creates a horizontal plot
    
    # Plotting the model
    plt.figure()
    data = plt.imread(file_name) # Display the image
    plt.imshow(data)
    plt.axis('off');
    plt.title('Defined Model')
    plt.suptitle(sup_title)
    #plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
    plt.show()

def learning_curve_display(training, loss_name):
    plt.figure()
    plt.plot(training.history['loss']) # Plot the training loss 
    plt.ylabel(loss_name)
    plt.xlabel('epochs')
    plt.title('Evaluation results in each epoch')
    plt.suptitle(tema)
    #plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
    plt.show()

print("****************************************************")
tema = "Reading the data"; print("** %s\n" % tema)

file = 'darts.csv'
darts_df = pd.read_csv(file)
#cats = CategoricalDtype(categories=darts_df.competitor.unique().tolist()) #,  ordered=True #Create categorical type data to use
#darts_df['competitor'] = darts_df['competitor'].astype(cats) # Change the data type of 'rating' to category

print('Dataset stats: \n', darts_df.describe()) # Describe the data
print('\nObservations per class: \n{}'.format(darts_df['competitor'].value_counts())) # Count the number of observations of each class

g = sns.relplot(x='xCoord', y='yCoord', data=darts_df, hue='competitor',
                kind='scatter', alpha=0.4) # Use pairplot and set the hue to be our class
plt.xticks(fontsize=8) #Fontsize in sns plot
plt.yticks(fontsize=8)
plt.title('Dataset: {}'.format(file))
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.88, wspace=None, hspace=None)
plt.show() # Show the plot
plt.style.use('default')


print("****************************************************")
tema = "6. A multi-class model"; print("** %s\n" % tema)

model = Sequential() # Instantiate a sequential model
model.add(Dense(128, input_shape=(2,), activation='relu', name='Dense')) # Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(64, activation='relu', name='Dense_2'))
model.add(Dense(32, activation='relu', name='Dense_3'))
model.add(Dense(4, activation='softmax', name='Output')) # Add a dense layer with as many neurons as competitors
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile your model using categorical_crossentropy loss

model_display(model, tema, file_name='02_06_model.png')

print("****************************************************")
tema = "7. Prepare your dataset"; print("** %s\n" % tema)

darts_df.competitor = pd.Categorical(darts_df.competitor) # Transform into a categorical variable
darts_df.competitor = darts_df.competitor.cat.codes # Assign a number to each category (label encoding)
print('Label encoded competitors: \n{}'.format(darts_df.competitor.head())) # Print the label encoded competitors

coordinates = darts_df.drop(['competitor'], axis=1) # Use to_categorical on your labels
competitors = to_categorical(darts_df.competitor)
print('\nOne-hot encoded competitors: \n{}'.format(competitors)) # Now print the to_categorical() result

print("****************************************************")
tema = "8. Training on dart throwers"; print("** %s\n" % tema)

coord_train, coord_test, competitors_train, competitors_test = train_test_split(coordinates, competitors, # Create training and test sets
                                                                                stratify=competitors,
                                                                                test_size=0.2, random_state=SEED)

training = model.fit(coord_train, competitors_train, epochs=200) # Train your model on the training data for 200 epochs
accuracy = model.evaluate(coord_test, competitors_test)[1] # Evaluate your model accuracy on the test data

print('Accuracy:', accuracy) # Print accuracy
learning_curve_display(training, loss_name='categorical_crossentropy')

print("****************************************************")
tema = "9. Softmax predictions"; print("** %s\n" % tema)

preds = model.predict(coord_test) # Predict on X_small_test
print("{:45} | {}".format('Raw Model Predictions','True labels')) # Print preds vs true values
for i, pred in enumerate(preds):
    print("{} | {}".format(pred, competitors_test[i]))

preds = [np.argmax(pred) for pred in preds] # Extract the indexes of the highest probable predictions
print("{:10} | {}".format('Rounded Model Predictions','True labels')) # Print preds vs true values
for i,pred in enumerate(preds):
    print("{:25} | {}".format(pred, competitors_test[i]))

print("Classes predicted: ", model.predict_classes(coord_test))  # Predict labels

print("****************************************************")
print("** END                                            **")
print("****************************************************")


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

#Setting images params
#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09
#plt.rcParams.update({'figure.max_open_warning': 0})                           #To solve the max images open
#plt.rcParams["axes.labelsize"] = 8                                            #Font

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.

#sns.set(font_scale=0.8)                                                       #Font
#sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot