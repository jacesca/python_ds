# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:12:55 2019

@author: jacqueline.cortez

Chapter 4. Advanced Model Architectures
Introduction:
    It's time to get introduced to more advanced architectures! You will create an autoencoder 
    to reconstruct noisy images, visualize convolutional neural network activations, use deep 
    pre-trained models to classify images and learn more about recurrent neural networks and 
    working with text as you build a network that predicts the next word in a sentence.
"""
print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Importing libraries \n")

#import keras.backend     as k                                                 #For DeapLearning
import numpy             as np                                                #For making operations in lists
import matplotlib.pyplot as plt                                               #For creating charts
import tensorflow as tf                                                       #For DeapLearning

from keras.datasets                  import mnist                             #For DeapLearning
from keras.layers                    import Dense                             #For DeapLearning
from keras.models                    import Sequential                        #For DeapLearning
from keras.utils                     import plot_model                        #For DeapLearning
from keras.utils                     import to_categorical                    #For DeapLearning

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

def show_encodings(X_noise, y_target, encoded_imgs, number=4, sup_title=""):
    n = 5  # how many digits we will display
    original = X_noise
    original = original[np.where(y_target == number)]
    y_target = y_target[np.where(y_target == number)]
    encoded_imgs = encoded_imgs[np.where(y_target==number)]
    plt.figure(figsize=(10, 4))
    #plt.title('Original '+str(number)+' vs Encoded representation')
    for i in range(min(n,len(original))):
        # display original imgs
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Number: {}".format(y_target[i]))
        
        # display encoded imgs
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.tile(encoded_imgs[i],(32,1)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Number: {}".format(y_target[i]))
    plt.suptitle("{}/nEncoding images".format(sup_title))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=0.5)
    plt.show()

def compare_plot(original, decoded_imgs, y_target, sup_title="", number=4):
    original = original[np.where(y_target == number)]
    decoded_imgs = decoded_imgs[np.where(y_target == number)]
    y_target = y_target[np.where(y_target==number)]
    
    n = 4  # How many digits we will display
    plt.figure(figsize=(10, 4))
    for i in range(min(n,len(original))):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Number: {}".format(y_target[i]))
        
        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Number: {}".format(y_target[i]))
    plt.suptitle("{}/nNoisy vs Decoded images".format(sup_title))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=0.5)
    plt.show()


print("****************************************************")
tema = "Reading the data"; print("** %s\n" % tema)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #Data normalization

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

plt.figure(figsize=(10, 1.5))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    #plt.pcolor(x_train[i],  cmap='gray') # Visualize the result
    #plt.gca().invert_yaxis()
    plt.imshow(x_train[i], cmap=plt.get_cmap(name='gray'))
    plt.xlabel('Number: {}'.format(y_train[i]))
plt.suptitle("Number MNIST Database")
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

y_target_train = to_categorical(y_train)
y_target_test = to_categorical(y_test)

X_train = x_train.reshape(len(x_train), 784)
X_test  = x_test.reshape(len(x_test ), 784)

print("****************************************************")
tema = "Creating the noise matrix"; print("** %s\n" % tema)

#n_rows = x_test.shape[1]
#n_cols = x_test.shape[2]
#mean = 0.5; stddev = 0.3;
#noise = np.random.normal(mean, stddev, (n_rows, n_cols))
#x_test_noisy = x_test + noise # creating the noisy test data by adding X_test with noise

# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noise = x_train + noise
x_train_noise = np.clip(x_train_noise, 0., 1.)
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noise = x_test + noise
x_test_noise = np.clip(x_test_noise, 0., 1.)

plt.figure(figsize=(10, 1.5))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    #plt.pcolor(x_train_noisy[i],  cmap='gray') # Visualize the result
    #plt.gca().invert_yaxis()
    plt.imshow(x_train_noise[i], cmap=plt.get_cmap(name='gray'))
    plt.xlabel('Number: {}'.format(y_train[i]))
plt.suptitle("Noise MNIST Database")
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=0.5, hspace=None)
plt.show()

X_train_noise = x_train_noise.reshape(len(x_train_noise), 784)
X_test_noise  = x_test_noise.reshape(len(x_test_noise), 784)

print("****************************************************")
tema = "4. Building an autoencoder"; print("** %s\n" % tema)

autoencoder = Sequential() # Start with a sequential model
autoencoder.add(Dense(32, input_shape=(784, ), activation="relu", name='Dense')) # Add a dense layer with the original image as input
autoencoder.add(Dense(784, activation="sigmoid", name='Output')) # Add an output layer with as many nodes as the image
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # Compile your model
autoencoder.name = 'Autoencoder'

model_display(autoencoder, tema, file_name='04_04_model.png')

autoencoder.fit(X_train_noise, X_train, validation_split=0.2, epochs=30, batch_size=128) # Train the autoencoder

print("****************************************************")
tema = "5. De-noising like an autoencoder"; print("** %s\n" % tema)

encoder = Sequential() # Build your encoder
encoder.add(autoencoder.layers[0])
encoder.name = 'Encoder'

model_display(encoder, tema, file_name='04_05_model.png')

preds = encoder.predict(X_test_noise) # Encode the images and show the encodings
show_encodings(X_test_noise, y_test, preds, number=7, sup_title=tema)

decoded_imgs = autoencoder.predict(X_test_noise) # Predict on the noisy images with your autoencoder
compare_plot(X_test_noise, decoded_imgs, y_test, sup_title=tema, number=7) # Plot noisy vs decoded images

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
#import keras.backend as k                                                     #For DeapLearning
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
