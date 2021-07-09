# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:01:51 2019

@author: jacqueline.cortez

Chapter 3. Improving Your Model Performance
Introduction:
    In the previous chapters, you've trained a lot of models! You will now learn how to interpret 
    learning curves to understand your models as they train. You will also visualize the effects of 
    activation functions, batch-sizes, and batch-normalization. 
    Finally, you will learn how to perform automatic hyperparameter optimization to your Keras models 
    using sklearn.
"""
print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Importing libraries \n")

import matplotlib.pyplot as plt                                               #For creating charts
import numpy             as np                                                #For making operations in lists
import pandas            as pd                                                #For loading tabular data
import tensorflow as tf                                                       #For DeapLearning

from keras.callbacks                 import EarlyStopping                     #For DeapLearning
from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
from keras.layers                    import Dense                             #For DeapLearning
from keras.models                    import Sequential                        #For DeapLearning
from keras.optimizers                import Adam                              #For DeapLearning
from keras.utils                     import plot_model                        #For DeapLearning
from keras.wrappers.scikit_learn     import KerasClassifier                   #For DeapLearning

from sklearn.model_selection         import cross_val_score                   #For learning machine
from sklearn.model_selection         import KFold                             #For learning machine
from sklearn.model_selection         import GridSearchCV                      #For learning machine
from sklearn.model_selection         import RandomizedSearchCV                #For learning machine
from sklearn.model_selection         import train_test_split                  #For learning machine

print("****************************************************")
print("** Preparing the environment \n")

SEED=42
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED) #Instead of tf.set_random_seed, because it is deprecated.

print("****************************************************")
print("** User functions \n")

def model_display(model, sup_title, file_name):
    """
    model_display function make a plot of the defined model. This function need the followin parameters:
        model: the model to plot.
        sup_title: the text that is going to be printed as suptitle in the plot.
        file_name: the file where to save the image.
    """
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

def learning_curve_compare(train, validation, metrics, sup_title):
    """
    learning_curve_compare function show the curve of the learning performance. 
    This function need the followin parameters:
        - train: the metrics result in the train data per epochs.
        - validation: the metrics result in the validation  data per epochs.
        - metrics: the metrics that is tracked and is going to show in the plot.
        - sup_title: the text that is going to be printed as suptitle in the plot.
    """
    plt.figure()
    plt.plot(train)
    plt.plot(validation)
    plt.title('Model {}'.format(metrics))
    plt.ylabel(metrics)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.suptitle(sup_title)
    plt.show()

def plot_results(train_accs, test_accs, train_sizes, sup_title):
    """
    plot_results function shows the evolution of accuracy in the learning process through different size samples.
    This function needs the following parameters:
        - train_accs: the accuracy tracked with the train data.
        - test_accs: the accuracy tracked with the validation data.
        - train_sizes: the different sizes of samples using to train and test.
        - sup_title: the text that is going to be printed as suptitle in the plot.
    """
    plt.figure()
    plt.plot(train_sizes, train_accs, 'o-', label="Training Accuracy")
    plt.plot(train_sizes, test_accs, 'o-', label="Test Accuracy")
    plt.xticks(train_sizes); 
    plt.title('Accuracy vs Number of training samples')
    plt.xlabel('Training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.suptitle(sup_title)
    plt.show()
    
def compare_histories_acc(h1,h2,metric='acc',ylabel='Accuracy',sup_title=''):
    """
    compare_histories_acc funtion shows the comparative learning curve, between Batchnormalized and standard model.
    This function needs the following parameters:
         - h1: the metrics saved in the normal standard model.
         - h2: the matrics saved in the normalized model.
         - metrics: metrics tracked in the models.
         - ylabel: the label of y axis.
         - sup_title: the text that is going to be printed as suptitle in the plot.
    """
    plt.figure()
    plt.plot(h1.history[metric])
    plt.plot(h1.history['val_{}'.format(metric)])
    plt.plot(h2.history[metric])
    plt.plot(h2.history['val_{}'.format(metric)])
    plt.title("Batch Normalization Effects")
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(['Train', 'Test', 'Train with Batch Normalization', 'Test with Batch Normalization'], loc='best')
    plt.suptitle(sup_title)
    plt.show()


def create_model(learning_rate=0.01, activation='relu', input_shape=(30,)):
    """
    create_model function creates a model given an activation and learning rate.
    """
    opt = Adam(lr=learning_rate) # Create an Adam optimizer with the given learning rate
    model = Sequential() # Create your binary classification model  
    model.add(Dense(128, input_shape=input_shape, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']) # Compile your model with your optimizer, loss, and metrics
    return model


def create_model_big(learning_rate=0.01, activation='relu', nl=1, nn=256, input_shape=(30,)):
    """
    create_model_big function creates a model given an activation, learning rate, number of layers (nl)
    and number of neurons (nn) inside each layer.
    """
    opt = Adam(lr=learning_rate) # Create an Adam optimizer with the given learning rate
    model = Sequential() # Create your binary classification model  
    model.add(Dense(128, input_shape=input_shape, activation=activation))
    for i in range(nl): # Add as many hidden layers as specified in nl.
        model.add(Dense(nn, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']) # Compile your model with your optimizer, loss, and metrics
    return model


print("****************************************************")
tema = "Reading the data"; print("** %s\n" % tema)

file = 'wbc.csv'
wbc_df = pd.read_csv(file, index_col='id')
WBC_X = wbc_df.drop(['diagnosis', 'Unnamed: 32'], axis=1)
WBC_y = wbc_df.diagnosis.map({'B':0, 'M':1})

X_train, X_test, y_train, y_test = train_test_split(WBC_X, WBC_y, test_size=0.2, stratify=WBC_y, random_state=SEED)


print("****************************************************")
tema = "15. Tuning the model parameters"; print("** %s\n" % tema)

model = KerasClassifier(build_fn=create_model) # Create a KerasClassifier
params = {'activation'   :['relu', 'tanh'], 
          'batch_size'   :[32, 128, 256], 
          'epochs'       :[50, 100, 200], 
          'learning_rate':[0.1, 0.01, 0.001]} # Define the parameters to try out

random_search = RandomizedSearchCV(model, param_distributions=params, cv=KFold(3), iid=False) # Create a randomize search cv object and fit it on the data to obtain the results
#best_params = {'learning_rate': 0.001, 'epochs': 200, 'batch_size': 128, 'activation': 'tanh'}

#random_search = GridSearchCV(model, param_grid=params, cv=KFold(3), iid=False) # Create a randomize search cv object and fit it on the data to obtain the results
#best_params = {'activation': 'tanh', 'batch_size': 256, 'epochs': 200, 'learning_rate': 0.001}
random_search_result = random_search.fit(X_train, y_train, verbose=0)

print("Best result: {:,.4f} using: {}".format(random_search_result.best_score_, random_search_result.best_params_))
best_params = random_search_result.best_params_

model = create_model(learning_rate=best_params['learning_rate'], activation=best_params['activation'])
monitor_val_loss = EarlyStopping(monitor='loss', patience=5) # Define a callback to monitor val_acc
modelCheckpoint = ModelCheckpoint('03_15_model_wbc.hdf5', save_best_only=True) # Save the best model as best_banknote_model.hdf5
training = model.fit(X_train, y_train, validation_data=(X_test, y_test),  
                     epochs=best_params['epochs'], batch_size=best_params['batch_size'],
                     verbose=0, callbacks=[monitor_val_loss, modelCheckpoint]) # Train your model for 60 epochs, using X_test and y_test as validation data

learning_curve_compare(training.history['loss'], training.history['val_loss'], metrics='Loss', sup_title=tema) # Plot train vs test loss during training

print("****************************************************")
tema = "16. Training with cross-validation"; print("** %s\n" % tema)

def create_special_model():
	opt = Adam(lr=best_params['learning_rate'])
	model = Sequential()
	model.add(Dense(128,input_shape=(30,),activation=best_params['activation']))
	model.add(Dense(256,activation=best_params['activation']))
	model.add(Dense(1,activation='sigmoid'))  	
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

model = KerasClassifier(build_fn=create_special_model, epochs=best_params['epochs'], 
                        batch_size=best_params['batch_size'], verbose=0)

kfolds = cross_val_score(model, X_train, y_train, cv=3) # Calculate the accuracy score for each fold
print('The mean accuracy was:', kfolds.mean()) # Print the mean accuracy
print('With a standard deviation of:', kfolds.std()) # Print the accuracy standard deviation

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import inspect                                                                #Used to get the code inside a function
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
#from sklearn.model_selection         import KFold                             #For learning machine
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
#from keras.datasets                  import fashion_mnist                     #For DeapLearning
#from keras.datasets                  import mnist                             #For DeapLearning
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
#from keras.wrappers.scikit_learn     import KerasClassifier                   #For DeapLearning


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