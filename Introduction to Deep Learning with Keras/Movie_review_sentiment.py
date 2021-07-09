# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 09:38:46 2019

@author: jacqueline.cortez
Source: 
    https://towardsdatascience.com/machine-learning-recurrent-neural-networks-and-long-short-term-memory-lstm-python-keras-example-86001ceaaebc
    https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data
    
train.tsv 
    contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so 
    that you can track which phrases belong to a single sentence.
test.tsv 
    contains just phrases. You must assign a sentiment label to each phrase.
The sentiment labels are:
    0 - negative
    1 - somewhat negative
    2 - neutral
    3 - somewhat positive
    4 - positive
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
tema = "1. Importing libraries"; print("** %s\n" % tema)

import matplotlib.pyplot as plt                                               #For creating charts
import numpy             as np                                                #For making operations in lists
import pandas            as pd                                                #For loading tabular data
import tensorflow as tf                                                       #For DeapLearning

from keras.callbacks                 import EarlyStopping                     #For DeapLearning
from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
from keras.layers                    import Dense                             #For DeapLearning
from keras.layers                    import Dropout                           #For DeapLearning
from keras.layers                    import Embedding                         #For DeapLearning
from keras.layers                    import LSTM                              #For DeapLearning
from keras.layers                    import SpatialDropout1D                  #For DeapLearning
from keras.models                    import Sequential                        #For DeapLearning
from keras.preprocessing.sequence    import pad_sequences                     #For DeapLearning
from keras.preprocessing.text        import Tokenizer                         #For DeapLearning
from keras.utils                     import plot_model                        #For DeapLearning
from keras.utils                     import to_categorical                    #For DeapLearning

print("****************************************************")
tema = "2. Preparing the environment"; print("** %s\n" % tema)

SEED=42
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED) #Instead of tf.set_random_seed, because it is deprecated.

pd.set_option("display.max_columns",20)

print("****************************************************")
tema = "3. User functions"; print("** %s\n" % tema)

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

def clean_text(text):
    replace_list = {   r"i'm": 'i am',
                       r"'re": ' are',
                     r"let’s": 'let us',
                        r"'s":  ' is',
                       r"'ve": ' have',
                     r"can't": 'can not',
                    r"cannot": 'can not',
                    r"shan’t": 'shall not',
                       r"n't": ' not',
                        r"'d": ' would',
                       r"'ll": ' will',
                    r"'scuse": 'excuse',
                          ',': ' ,',
                          '.': ' .',
                          '!': ' !',
                          '?': ' ?',
                        '\s+': ' '}
    
    text = text.lower()
    for s in replace_list:
        text = text.replace(s, replace_list[s])
    text = ' '.join(text.split())
    return text

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
    
def prepare_data_to_predict(df_test, max_phrase_len, max_words):
    X = df_test['Phrase'].apply(lambda p: clean_text(p))
    
    tokenizer = Tokenizer(num_words = max_words, filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~') 
    tokenizer.fit_on_texts(X)
    
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen = max_phrase_len)
    return X

print("****************************************************")
tema = "4. Reading the data"; print("** %s\n" % tema)

filename = "movie_review_train.tsv"
df_movie_review = pd.read_csv(filename, sep='\t')
print('Train set: {0}'.format(df_movie_review.shape))
print('Columns: ', df_movie_review.columns)
print('First two rows:')
print(df_movie_review.head(2))

sentiment_label = {0: 'negative', 1: 'somewhat negative', 2: 'neutral', 3: 'somewhat positive', 4: 'positive'}

filename = "movie_review_test.tsv"
df_movie_test = pd.read_csv(filename, sep='\t')
print('Test set: {0}'.format(df_movie_test.shape))
print('Columns: ', df_movie_test.columns)
print('First two rows:')
print(df_movie_test.head(2))

print("****************************************************")
tema = "5. Cleaning the data"; print("** %s\n" % tema)

# ASCII characters are ultimately interpreted by the computer as hexadecimal. In consequence, to a computer, 
# ‘A’ is not the same as ‘a’. Therefore, we’ll want to change all characters to lowercase. Since we’re going 
# to be splitting the sentences up into individual words based off of white spaces, a word with a period right 
# after it is not equivalent to one without a period following it (happy. != happy). In addition, contractions 
# are going to be interpreted differently than the original which will have repercussions for the model 
# (I’m != I am). Thus, we replace all occurrences using the proceeding function.
X = df_movie_review['Phrase'].apply(lambda p: clean_text(p))

# Obtaining the target
y = df_movie_review['Sentiment']

print("****************************************************")
tema = "6. Exploring the data"; print("** %s\n" % tema)

phrase_len = X.apply(lambda p: len(p.split(' ')))
max_phrase_len = phrase_len.max()
print('max phrase len: {0}'.format(max_phrase_len))

plt.figure(figsize = (6, 4))
plt.hist(phrase_len, alpha = 0.2, density = True)
plt.xlabel('Phrase len')
plt.ylabel('Probability')
#plt.yscale('log')
plt.title('Individual length of each phrases in the corpus')
plt.suptitle(tema)
plt.grid(alpha = 0.25)
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None)
plt.show()

print("****************************************************")
tema = "7. Preparing the data for the Machine Learning"; print("** %s\n" % tema)

max_words = 8192 # Only the most common num_words-1 words will be kept.
# By default, all punctuation is removed, turning the text into a space separated sequence of words. 
# The tokens are then vectorized. By vectorized we mean that they are mapped to integers. 0 is a reserved 
# index that won't be assigned to any word.
tokenizer = Tokenizer(num_words = max_words, filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~') 
tokenizer.fit_on_texts(X)

X_token = tokenizer.texts_to_sequences(X)
# pad_sequence is used to ensure that all the phrase are the same length. Sequences that are shorter than 
# maxlen are padded with value (0 by default) at the end.
X_train = pad_sequences(X_token, maxlen = max_phrase_len)
y_train = to_categorical(y)

print("****************************************************")
tema = "9. Contruct the model"; print("** %s\n" % tema)

model_lstm = Sequential(name='Sentiment_Predictor')
model_lstm.add(Embedding(input_dim = max_words, output_dim = 256, input_length = max_phrase_len, name='Embedding'))
model_lstm.add(SpatialDropout1D(0.3, name='Correlated'))
model_lstm.add(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3, name='LSTM'))
model_lstm.add(Dense(256, activation = 'relu', name='Dense'))
model_lstm.add(Dropout(0.3, name='Dropout')) #We use dropout to prevent overfitting.
model_lstm.add(Dense(5, activation = 'softmax', name='Output'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model_display(model_lstm, sup_title=tema, file_name='Movie_prediction_09_Model.png')

print("****************************************************")
tema = "10. Train the model"; print("** %s\n" % tema)

monitor_val_loss = EarlyStopping(monitor='loss', patience=5) # Define a callback to monitor val_acc
modelCheckpoint = ModelCheckpoint('Movie_prediction_10_weights.hdf5', save_best_only=True) # Save the best model as best_banknote_model.hdf5

training = model_lstm.fit(X_train, y_train, validation_split = 0.2, 
                          epochs = 8, callbacks=[monitor_val_loss, modelCheckpoint])

learning_curve_compare(training.history['loss'], training.history['val_loss'], metrics='Loss', sup_title=tema) # Plot train vs test loss during training
learning_curve_compare(training.history['acc'], training.history['val_acc'], metrics='Accuracy', sup_title=tema) # Plot train vs test accuracy during training

print("****************************************************")
tema = "11. Making predictions"; print("** %s\n" % tema)

df_test = df_movie_test.copy()
X_test = prepare_data_to_predict(df_test, max_phrase_len, max_words)

#pred = model_lstm.predict(X_test)
pred = model_lstm.predict_classes(X_test)
print(pred)

df_test['Sentiment'] = pred
df_test['Sentiment_interpreter'] = df_test['Sentiment'].replace(sentiment_label)
print(df_test)

out_file = "Movie_Prediction_11_Result.csv"
df_test.to_csv(out_file, index=False)

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
#from keras.applications.resnet50     import decode_predictions                #For DeapLearning
#from keras.applications.resnet50     import preprocess_input                  #For DeapLearning
#from keras.applications.resnet50     import ResNet50                          #For DeapLearning
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
#from keras.layers                    import GlobalMaxPooling1D                #For DeapLearning
#from keras.layers                    import Input                             #For DeapLearning
#from keras.layers                    import LSTM                              #For DeapLearning
#from keras.layers                    import MaxPool2D                         #For DeapLearning
#from keras.layers                    import SpatialDropout1D                  #For DeapLearning
#from keras.layers                    import Subtract                          #For DeapLearning
#from keras.models                    import load_model                        #For DeapLearning
#from keras.models                    import Model                             #For DeapLearning
#from keras.models                    import Sequential                        #For DeapLearning
#from keras.optimizers                import Adam                              #For DeapLearning
#from keras.optimizers                import SGD                               #For DeapLearning
#from keras.preprocessing             import image                             #For DeapLearning
#from keras.preprocessing.text        import Tokenizer                         #For DeapLearning
#from keras.preprocessing.sequence    import pad_sequences                     #For DeapLearning
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
#plt.rcParams['figure.max_open_warning'] = 60
#plt.style.use('dark_background')
#plt.style.use('default')

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.

#sns.set(font_scale=0.8)                                                       #Font
#sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot
