# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:00:20 2019

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

import numpy             as np                                                #For making operations in lists
import matplotlib.pyplot as plt                                               #For creating charts
import tensorflow as tf                                                       #For DeapLearning

from keras.callbacks                 import EarlyStopping                     #For DeapLearning
from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
#from keras.datasets                  import mnist                             #For DeapLearning
from keras.layers                    import Dense                             #For DeapLearning
from keras.models                    import Sequential                        #For DeapLearning
from keras.utils                     import plot_model                        #For DeapLearning
from keras.utils                     import to_categorical                    #For DeapLearning

from sklearn                         import datasets                          #For learning machine
from sklearn.model_selection         import train_test_split                  #For learning machine

print("****************************************************")
print("** Preparing the environment \n")

SEED=42
np.random.seed(SEED)
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

def learning_curve_compare(train, validation, metrics):
    plt.figure()
    plt.plot(train)
    plt.plot(validation)
    plt.title('Model {}'.format(metrics))
    plt.ylabel(metrics)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()

def plot_results(train_accs, test_accs, train_sizes):
    plt.figure()
    plt.plot(train_sizes, train_accs, 'o-', label="Training Accuracy")
    plt.plot(train_sizes, test_accs, 'o-', label="Test Accuracy")
    plt.xticks(train_sizes); 
    plt.title('Accuracy vs Number of training samples')
    plt.xlabel('Training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.show()

def get_model(act_function):
    if act_function not in ['relu', 'linear', 'sigmoid', 'tanh']:
        raise ValueError('Make sure your activation functions are named correctly!')  
    else:
        print("Finishing with", act_function, "...")  
        model = Sequential() # Instantiate a Sequential model
        model.add(Dense(64, input_shape=(20,), activation=act_function, name='Dense_1')) # Add a hidden layer of 64 neurons and a 20 neuron's input
        model.add(Dense(3, activation='sigmoid', name='Output')) # Add an output layer of 3 neurons with sigmoid activation
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile your model with adam and binary crossentropy loss
        return model
    
print("****************************************************")
tema = "Reading the data"; print("** %s\n" % tema)

digits = datasets.load_digits()
X_features = digits.images
y_target   = digits.target

x_train, x_test, y_train, y_test = train_test_split(X_features, y_target, stratify=y_target,
                                                    test_size=0.2, random_state=SEED)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0 #Data normalization

#plt.pcolor(x_train[10],  cmap='gray') # Visualize the result
#plt.imshow(x_train[10], cmap = plt.get_cmap(name = 'gray'))
#plt.gca().invert_yaxis()
#plt.title('Number: {}'.format(y_train[10]))
#plt.suptitle(tema)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()

plt.figure(figsize=(10, 4))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    #plt.imshow(x_train[i], cmap=plt.get_cmap(name='gray'))
    plt.xlabel('Number: {}'.format(y_train[i]))
plt.suptitle("Number MNIST Database")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()

X_train = x_train.reshape(len(x_train), 64)
X_test  =  x_test.reshape(len(x_test ), 64)
y_target_train = to_categorical(y_train)
y_target_test = to_categorical(y_test)

#X_train = x_train.reshape(len(x_train), 784)
#X_test  =  x_test.reshape(len(x_test ), 784)

print("****************************************************")
tema = "2. Learning the digits"; print("** %s\n" % tema)

model = Sequential() # Instantiate a Sequential model
model.add(Dense(16, input_shape=(64,), activation='relu', name='Dense_1')) # Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(Dense(10, activation='softmax', name='Output')) # Output layer with 10 neurons (one per digit) and softmax
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile your model

initial_weights = model.get_weights() #Get initial weight for the recently defined model

model_display(model, tema, file_name='03_02_model.png')

#preds = model.predict(X_train) # Test if your model works and can process input data
#print('Predictions without training: \n{}'.format(preds)) # Print preds
#
#preds_rounded = np.round(preds)
#print('\nRounded Predictions: \n{}'.format(preds_rounded)) # Print rounded preds
#
#print("\nThe 5 first predictions\n{} | {} | {}\n".format('Raw Model Predictions','True labels', 'Target')) # Print preds vs true values
#for i, pred in enumerate(preds_rounded[:5]):
#    print("{:>21} | {} | {}".format(str(pred), y_target_train[i], y_train[i]))

print("****************************************************")
tema = "3. Is the model overfitting?"; print("** %s\n" % tema)

monitor_val_loss = EarlyStopping(monitor='loss', patience=5) # Define a callback to monitor val_acc
modelCheckpoint = ModelCheckpoint('03_03_model_mnist.hdf5', save_best_only=True) # Save the best model as best_banknote_model.hdf5

training = model.fit(X_train, y_target_train, validation_data=(X_test, y_target_test),  
                     epochs=60, verbose=0, callbacks=[monitor_val_loss, modelCheckpoint]) # Train your model for 60 epochs, using X_test and y_test as validation data
#training = model.fit(X_train, y_target_train, epochs=60, validation_split=0.2, verbose=0, 
#                     callbacks=[monitor_val_loss, modelCheckpoint]) # Trai2 your model for 60 epochs, using X_test and y_test as validation data

learning_curve_compare(training.history['loss'], training.history['val_loss'], metrics='Loss') # Plot train vs test loss during training

print("****************************************************")
tema = "4. Do we need more data?"; print("** %s\n" % tema)

training_sizes = np.array([ 125,  502,  879, 1255, 1435])
#training_sizes = np.array([ 110, 220, 330, 440, 550, 660, 770, 880, 990, 1100, 1210, 1320, 1430])
#training_sizes = np.array([ 110, 330, 550, 770, 990, 1210, 1430])

train_accs = []
test_accs = []

for size in training_sizes:
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_target_train, train_size=size) # Get a fraction of training data    
    model.set_weights(initial_weights) # Set the model weights to the initial weights and fit the model
    model.fit(X_train_frac, y_train_frac, epochs=50, callbacks=[monitor_val_loss], verbose=0)

    # Evaluate and store results for the training fraction and the test set
    train_accs.append(model.evaluate(X_train_frac, y_train_frac, verbose=0)[1])
    test_accs.append(model.evaluate(X_test, y_target_test, verbose=0)[1])

plot_results(train_accs, test_accs, training_sizes) # Plot train vs test accuracies

print("****************************************************")
tema = "Batch normalizing a familiar model"; print("** %s\n" % tema)

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