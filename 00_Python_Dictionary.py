# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:48:22 2020

@author: jacesca@gmail.com
Help:
    https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py
"""

import astropy.time

from bokeh.io import curdoc #For interacting visualizations
from bokeh.io import output_file #For interacting visualizations
from bokeh.io import show #For interacting visualizations
from bokeh.plotting import ColumnDataSource #For interacting visualizations
from bokeh.plotting import figure #For interacting visualizations
from bokeh.layouts import column #For interacting visualizations
from bokeh.layouts import gridplot #For interacting visualizations
from bokeh.layouts import row #For interacting visualizations
from bokeh.layouts import widgetbox #For interacting visualizations
from bokeh.models import Button #For interacting visualizations
from bokeh.models import CategoricalColorMapper #For interacting visualizations
from bokeh.models import CheckboxGroup #For interacting visualizations
from bokeh.models import ColumnDataSource #For interacting visualizations
from bokeh.models import HoverTool #For interacting visualizations
from bokeh.models import RadioGroup #For interacting visualizations
from bokeh.models import Select #For interacting visualizations
from bokeh.models import Slider #For interacting visualizations
from bokeh.models import Toggle #For interacting visualizations
from bokeh.models.widgets import Panel #For interacting visualizations
from bokeh.models.widgets import Tabs #For interacting visualizations
from bokeh.palettes import Spectral6 #For interacting visualizations

from bs4 import BeautifulSoup #Clean html tag fromt text. Documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
from bson.regex import Regex

from cairosvg import svg2png #Transform svg file into png file

import calendar  #For accesing to a vary of calendar operations

import cartopy.crs as ccrs #For cartopy maps
import cartopy.feature as cfeature #For cartopy maps
import cartopy.io.shapereader as shpreader #For cartopy maps 
from cartopy.feature.nightshade import Nightshade
from cartopy.io.img_tiles import Stamen

import collections
from collections import Counter #To calculate frequencies. counts elements in some iterable and returns a dictionary-like structure containing the count of each element.
from collections import defaultdict #to handle the case of a missing key in a dict. To initialize a dictionary that will assign a default value to non-existent keys. Returns a new dictionary-like object
from collections import deque # A deque is a like a queue or a stack except it works both ways.
from collections import namedtuple #The namedtuple generates a class which is similar to a tuple, but has named entries.
from collections import OrderedDict #An OrderedDict is exactly like a dict, but it remembers the insertion order of the keys.
from collections.abc import Iterable

import contextily  #To add a background web map to our plot

import csv

import datetime  #For accesing datetime functions
import datetime as dt #For accesing datetime functions
from datetime import date #For obteining today function
from datetime import datetime #For obteining today function
from datetime import timedelta

import dateutil.parser
from dateutil.relativedelta import relativedelta

import dc_stat_think as dcst #Statistic library for statistic hack

import dill #To retrieve source code, similar to inspect

import expectexception #For Jupiter Notebook--> Ej, %%expect_exception pd.errors.MergeError, %%expect_exception TypeError

from fancyimpute import KNN
from fancyimpute import IterativeImputer

import ffmpeg #work with pydub, for file formats differents from .wav, like .mp3

from flask import Flask
from flask import request

from fractions import Fraction #Work with fractions

from functools import reduce #For accessing to a high order functions (functions or operators that return functions)

from fuzzywuzzy import fuzz #Minimal distance to transform a string A into B
from fuzzywuzzy import process #Minimal distance to transform a string A into B

import inspect  #Used to get the code inside a function

import itertools  #Allows us to iterate through a set of sequences as if they were one continuous sequence. 
from itertools import cycle #Used in the function plot_labeled_decision_regions()
from itertools import chain #To flat a list

import folium  #To create map street folium.__version__'0.10.0'

from gensim.corpora.dictionary import Dictionary #To build corpora and dictionaries using simple classes and functions. Documentation: https://radimrehurek.com/gensim/auto_examples/index.html
from gensim.models.tfidfmodel import TfidfModel #To calculate the Term frequency - inverse document frequency

import geopandas as gpd #For working with geospatial data 

from glob import glob #For using with pathnames matching and find files

import imageio  #To create animation using png files

import inspect #To retrieve source code

from IPython.display import HTML

from itertools import combinations #For iterations
from itertools import cycle #Used in the function plot_labeled_decision_regions()
from itertools import groupby

from joblib import dump
from joblib import load

import json

import keras  #For DeapLearning
import keras.backend as k #For DeapLearning
from keras.applications.resnet50 import decode_predictions #For DeapLearning
from keras.applications.resnet50 import preprocess_input #For DeapLearning
from keras.applications.resnet50 import ResNet50 #For DeapLearning
from keras.callbacks import EarlyStopping #For DeapLearning
from keras.callbacks import ModelCheckpoint #For DeapLearning
from keras.datasets import fashion_mnist #For DeapLearning
from keras.datasets import mnist #For DeapLearning
from keras.layers import BatchNormalization #For DeapLearning
from keras.layers import Concatenate #For DeapLearning
from keras.layers import Conv2D #For DeapLearning
from keras.layers import Dense #For DeapLearning
from keras.layers import Dropout #For DeapLearning
from keras.layers import Embedding #For DeapLearning
from keras.layers import Flatten #For DeapLearning
from keras.layers import GlobalMaxPooling1D #For DeapLearning
from keras.layers import Input #For DeapLearning
from keras.layers import LSTM #For DeapLearning
from keras.layers import MaxPool2D #For DeapLearning
from keras.layers import SpatialDropout1D #For DeapLearning
from keras.layers import Subtract #For DeapLearning
from keras.models import load_model #For DeapLearning
from keras.models import Model #For DeapLearning
from keras.models import Sequential #For DeapLearning
from keras.optimizers import Adam #For DeapLearning
from keras.optimizers import SGD #For DeapLearning
from keras.preprocessing import image #For DeapLearning
from keras.preprocessing.text import Tokenizer #For DeapLearning
from keras.preprocessing.sequence import pad_sequences #For DeapLearning
from keras.utils import plot_model #For DeapLearning
from keras.utils import to_categorical #For DeapLearning
from keras.wrappers.scikit_learn import KerasClassifier #For DeapLearning

from langdetect import detect_langs #Documentation: https://github.com/Mimino666/langdetect
from langdetect import DetectorFactory

from legibilidad import legibilidad

import logging  #Tracking an error

import math  #For accesing to a complex math operations. https://docs.python.org/3/library/math.html
from math import ceil #Used in the function plot_labeled_decision_regions()
from math import factorial
from math import floor #Used in the function plot_labeled_decision_regions()
from math import pi
from math import radian #For accessing a specific math operations
from math import sqrt 

import matplotlib as mpl #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
from matplotlib import cm
from matplotlib import colors
from matplotlib import dates as mdates
import matplotlib.animation as animation #Make animations in python
from matplotlib.animation import FuncAnimation #The animation function package
from matplotlib.animation import PillowWriter #The writr to ave the animation as gif
from matplotlib.axes._axes import _log as matplotlib_axes_logger #To avoid warnings
import matplotlib.cbook
import matplotlib.cm as cm #Working with colors
import matplotlib.dates as mdates #For providing sophisticated date plotting capabilities
from matplotlib.dates import DateFormatter #To work with dates on axes. To format axes
from matplotlib.lines import Line2D as Line #For cartopy maps
from matplotlib.offsetbox import AnchoredText #Text in a cartopy maps
import matplotlib.patches as mpatches #For cartopy maps
import matplotlib.path as mpath #For cartopy maps
from matplotlib.patheffects import Stroke #For cartopy maps
from matplotlib.pylab import *
import matplotlib.pyplot as plt #For creating charts
from matplotlib.pyplot import stem # To plot PDF
from matplotlib.sankey import Sankey #Sankey diagrams
import matplotlib.ticker as ticker
import matplotlib.ticker as mtick #To work with percentage axis
from matplotlib.ticker import MaxNLocator #Integer format
from matplotlib.ticker import PercentFormatter #To work with percentage axis
from matplotlib.ticker import StrMethodFormatter #Import the necessary library to delete the scientist notation

from matplotlib_venn import venn2 #For Venn Diagrams
from matplotlib_venn import venn2_unweighted #For Venn Diagrams
from matplotlib_venn import venn2_circles #For Venn Diagrams
from matplotlib_venn import venn3 #For Venn Diagrams
from matplotlib_venn import venn3_unweighted #For Venn Diagrams
from matplotlib_venn import venn3_circles #For Venn Diagrams

import missingno as msno #Missing data visualization module for Python. Package for graphical analysis of missing values. Help on missingno library --> https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py

import mlxtend #Used in the function plot_labeled_decision_regions
#from mlxtend import plot_decision_regions #More documentation: http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
from mlxtend.plotting import plot_decision_regions #More documentation: http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
from mlxtend.plotting import plot_learning_curves #A function to plot learning curves for classifiers. Learning curves are extremely useful to analyze if a model is suffering from over- or under-fitting (high variance or high bias).

import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.basemap import Basemap #Working with maps
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx #For Network Analysis in Python

import nltk  #For working with text data
from nltk import ne_chunk
from nltk import pos_tag
from nltk import sent_tokenize
from nltk import word_tokenize #Documentation: https://www.nltk.org/api/nltk.tokenize.html
from nltk.chunk import conlltags2tree #Trees transformation
from nltk.chunk import ChunkParserI
from nltk.chunk import tree2conlltags #Trees transformation
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer #Stemming of strings - English             |
from nltk.stem import WordNetLemmatizer #Lemmatization of a string
from nltk.stem.snowball import SnowballStemmer #To use foreign language stemmers: Danish, Dutch, English, Finnish, French, German, Hungarian,Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish
from nltk.tag import ClassifierBasedTagger
from nltk.tokenize import blankline_tokenize
from nltk.tokenize import sent_tokenize #tokenize a document into sentences
from nltk.tokenize import regexp_tokenize #tokenize a string or document based on a regular expression pattern
from nltk.tokenize import TweetTokenizer #special class just for tweet tokenization, allowing you to separate hashtags, mentions and lots of exclamation points!!!
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize

import numpy as np #For making operations in lists
from numpy.random import randint #numpy.random.randint(low, high=None, size=None, dtype='l')-->Return random integers from low (inclusive) to high (exclusive). 

import nxviz as nv #For Network Analysis in Python
from nxviz import ArcPlot #For Network Analysis in Python
from nxviz import CircosPlot #For Network Analysis in Python 
from nxviz import MatrixPlot #For Network Analysis in Python 

from operator import itemgetter # Used in a sort options

import os  #To raise an html page in python command

import pandas as pd #For loading tabular data
from pandas.api.types import CategoricalDtype #For categorical data
from pandas.core.common import flatten
from pandas.plotting import parallel_coordinates #For Parallel Coordinates
from pandas.plotting import register_matplotlib_converters #For conversion as datetime index in x-axis

from pathlib import Path # Work with directories and urls

import pendulum

import pickle #pickle.format_version

from PIL import Image #To make a Mask from a image. pillow library to import the image -->conda install -c anaconda pillow

import plotly
import plotly.graph_objs as go

from polyglot.downloader import downloader #To download and review existing modules
from polyglot.text import Text #To Named entity recognition. Documentation: https://polyglot.readthedocs.io/en/latest/Installation.html , https://polyglot.readthedocs.io/en/latest/Download.html
from polyglot.transliteration import Transliterator #To make traductions

import pprint  #Import pprint to format disctionary output
from pprint import pprint

from punctuator import Punctuator

import pyaudio #Work with pydub. To play audios -->COULD NOT BE INSTALLED IN WINDOWS, USE SIMPLEAUDIO INSTEAD.

from pydub import AudioSegment #Provides a gold mine of tools for manipulating audio files
from pydub.effects import normalize
from pydub.playback import play #Needs simpleaudio module installed
from pydub.silence import split_on_silence

from pymongo import MongoClient

import pytest # TDD

from pytz import timezone

import random  #For generating random numbers
from random import randint

##rasa_nlu.__version__==0.11.3
from rasa_nlu.components import ComponentBuilder #If multiple models are created, it is reasonable to share components between the different models. E.g. the 'nlp_spacy' component, which is used by every pipeline that wants to have access to the spacy word vectors, can be cached to avoid storing the large word vectors more than once in main memory. To use the caching, a ComponentBuilder should be passed when loading and training models.
from rasa_nlu.config import RasaNLUConfig #rasa_nlu.__version__==0.11.3 #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/
from rasa_nlu.converters import load_data #rasa_nlu.__version__==0.11.3 #To generate the json file: https://rodrigopivi.github.io/Chatito/ #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Metadata #to load the metadata of your model
from rasa_nlu.model import Trainer #rasa_nlu.__version__==0.11.3 #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/

import re  #For regular expressions
from re import search

import recordlinkage  #Link dataframes based on similarities

import requests #To read URL html pages

import seaborn as sns #For visualizing data

import scikit_learn #For performing machine learning 

import scipy.stats
import scipy.stats as stats #For accesign to a vary of statistics functiosn
from scipy import interpolate
from scipy import optimize #Provides several commonly used optimization algorithms
from scipy import stats
from scipy.cluster import hierarchy #Making dendrogram graph
from scipy.cluster.hierarchy import dendrogram #For learning machine - unsurpervised
from scipy.cluster.hierarchy import fcluster #For learning machine - unsurpervised
from scipy.cluster.hierarchy import linkage #For learning machine - unsurpervised
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter #For working with images
from scipy.ndimage import median_filter #For working with images
from scipy.optimize import minimize #Find the value that minimize a function that works with arrays, ex minimize(x**2, 2(<--this is a initial guess)) = aprox.0, because 0 gives us the minimun possible value in x**2
from scipy.optimize import minimize_scalar #Find the value that minimize a function that only work with scalar
from scipy.signal import convolve2d #For learning machine - deep learning
from scipy.sparse import csr_matrix #For learning machine 
from scipy.special import expit as sigmoid #For learning machine 
from scipy.special import expit #Inverse of the logistic function.
from scipy.stats import anderson #For Anderson-Darling Normality Test. Tests whether a data sample has a Gaussian distribution.
from scipy.stats import bernoulli #Generate bernoulli data
from scipy.stats import binom #Generate binomial data
from scipy.stats import chi2_contingency #For Chi-Squared Test. Tests whether two categorical variables are related or independent
from scipy.stats import describe #To get the arithmetic statistics.
from scipy.stats import f_oneway #For Analysis of Variance Test. Tests whether the means of two or more independent samples are significantly different.
from scipy.stats import find_repeats #To find repeated data in a sample. Statistical terms
from scipy.stats import friedmanchisquare #For Friedman Test. Tests whether the distributions of two or more paired samples are equal or not.
from scipy.stats import geom #Generate gometric distribution
from scipy.stats import kendalltau #For Kendall's Rank Correlation Test. To check if two samples are related.
from scipy.stats import kruskal #For Kruskal-Wallis H Test. Tests whether the distributions of two or more independent samples are equal or not.
from scipy.stats import linregress #Fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
from scipy.stats import mannwhitneyu #For Mann-Whitney U Test. Tests whether the distributions of two independent samples are equal or not.
from scipy.stats import norm #Generate normal data
from scipy.stats import normaltest #For D'Agostino's K^2 Normality Test. Tests whether a data sample has a Gaussian distribution.
from scipy.stats import pearsonr #For learning machine. For Pearson's Correlation test. To check if two samples are related.
from scipy.stats import poisson #To generate poisson distribution.
from scipy.stats import randint #For learning machine 
from scipy.stats import relfreq #To calculate the relative frequency of each outcome
from scipy.stats import sem #For statistic thinking 
from scipy.stats import shapiro #For Shapiro-Wilk Normality Test. Tests whether a data sample has a Gaussian distribution.
from scipy.stats import spearmanr #For Spearman's Rank Correlation Test. To check if two samples are related.
from scipy.stats import t #For statistic thinking 
from scipy.stats import ttest_ind #For Student's t-test. Tests whether the means of two independent samples are significantly different.
from scipy.stats import ttest_rel #For Paired Student's t-test. Tests whether the means of two paired samples are significantly different.
from scipy.stats import wilcoxon #For Wilcoxon Signed-Rank Test. Tests whether the distributions of two paired samples are equal or not.

import shapely.geometry as sgeom #For cartopy maps 
from shapely.geometry import LineString #(Geospatial) To create a Linestring geometry column 
from shapely.geometry import Point #(Geospatial) To create a point geometry column 
from shapely.geometry import Polygon #(Geospatial) To create a point geometry column 
from shapely.ops import transform as geom_transform #For cartopy maps

import shutil #Work with directories -->To create dir, ej. shutil.rmtree(chunk_folder)

import simpleaudio #Work with pydub. To play audios

from skimage import exposure #For working with images
from skimage import measure #For working with images
from skimage.filters.thresholding import threshold_otsu #For working with images
from skimage.filters.thresholding import threshold_local #For working with images 

import sklearn.datasets
from sklearn import datasets #For learning machine
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.cluster import KMeans #For learning machine - unsurpervised
from sklearn.datasets import load_digits #Import handwritten digtis dataset example
from sklearn.datasets import load_wine #Import wine database
from sklearn.decomposition import NMF #For learning machine - unsurpervised
from sklearn.decomposition import PCA #For learning machine - unsurpervised
from sklearn.decomposition import TruncatedSVD #For learning machine - unsurpervised
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier #For learning machine - surpervised
from sklearn.ensemble import BaggingClassifier #For learning machine - surpervised
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor #For learning machine - surpervised
from sklearn.ensemble import RandomForestClassifier #For learning machine
from sklearn.ensemble import RandomForestRegressor #For learning machine - unsurpervised
from sklearn.ensemble import VotingClassifier #For learning machine - unsurpervised
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer #Documentation: https://scikit-learn.org/stable/index.html
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import HashingVectorizer #For learning machine
from sklearn.feature_extraction.text import TfidfVectorizer #Score tfidf result like CountVectorizer
from sklearn.feature_selection import chi2 #Select the best features. For learning machine
from sklearn.feature_selection import SelectKBest #Select the best features. For learning machine
from sklearn.impute import SimpleImputer #For learning machine
from sklearn.linear_model import ElasticNet #For learning machine
from sklearn.linear_model import Lasso #For learning machine
from sklearn.linear_model import LinearRegression #For learning machine. Calculate a linear least-squares regression for two sets of measurements. To get the parameters (slope and intercept) from a model
from sklearn.linear_model import LogisticRegression #For learning machine. Logistic Regression (aka logit, MaxEnt) classifier.
from sklearn.linear_model import Ridge #For learning machine
from sklearn.linear_model import SGDClassifier #Used for classifiers problems, recommended for big datasets
from sklearn.manifold import TSNE #For learning machine - unsurpervised
from sklearn import metrics
from sklearn.metrics import accuracy_score #Using accuracy score
from sklearn.metrics import classification_report #Precision, recall, f1-score and support
from sklearn.metrics import confusion_matrix #For learning machine
from sklearn.metrics import mean_absolute_error as MAE #For learning machine
from sklearn.metrics import mean_squared_error as MSE #For learning machine
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score #Compute the precision of the model. Precision is the number of true positives over the number of true positives plus false positives and is linked to the rate of the type 1 error.
from sklearn.metrics import recall_score #Compute the recall of the model. Recall is the number of true positives over the number of true positives plus false negatives and is linked to the rate of type 2 error.
from sklearn.metrics import roc_auc_score #For learning machine
from sklearn.metrics import roc_curve #For learning machine
from sklearn.metrics.pairwise import cosine_similarity #To evaluate the NLP model
from sklearn.model_selection import cross_val_score #For learning machine
from sklearn.model_selection import GridSearchCV #Select best params. For learning machine
from sklearn.model_selection import KFold #For learning machine
from sklearn.model_selection import RandomizedSearchCV #For learning machine
from sklearn.model_selection import train_test_split #For learning machine
from sklearn.multiclass import OneVsRestClassifier #For learning machine
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier as KNN #For learning machine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion #For learning machine
from sklearn.pipeline import make_pipeline #For learning machine - unsurpervised
from sklearn.pipeline import Pipeline #For learning machine
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer #For learning machine
from sklearn.preprocessing import Imputer #For learning machine
from sklearn.preprocessing import LabelEncoder #Create the encoder and print our encoded new_vals
from sklearn.preprocessing import MaxAbsScaler #For learning machine (transforms the data so that all users have the same influence on the model)
from sklearn.preprocessing import MinMaxScaler #Used for normalize data in a dataframe
from sklearn.preprocessing import Normalizer #For dataframe. For learning machine - unsurpervised (for pipeline)
from sklearn.preprocessing import normalize #For arrays. For learning machine - unsurpervised
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import scale #For learning machine
from sklearn.preprocessing import StandardScaler #For learning machine
from sklearn.svm import LinearSVC #Linear regression
from sklearn.svm import SVC #For learning machine
from sklearn.tree import DecisionTreeClassifier #For learning machine - supervised
from sklearn.tree import DecisionTreeRegressor #For learning machine - supervised
from sklearn.utils.validation import check_is_fitted

from sklearn_crfsuite import CRF #conditional random fields

from sklearn_pandas import DataFrameMapper

import spacy #To use lematize function in other language. Documentation: https://spacy.io/usage/models  https://spacy.io/api/annotation#pos-tagging
import en_core_web_lg #English word vector to work with Spacy. Documentation: https://spacy.io/usage/models#languages
import es_core_news_lg #Spanish word vector to work with Spacy. Documentation: https://spacy.io/usage/models#languages
#en_core_web_lg #See Spacy
#es_core_news_lg #See Spacy
from spacy import displacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.matcher import Matcher #https://spacy.io/usage/rule-based-matching#adding-patterns-attributes
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.tokens import Token

import speech_recognition as sr #To interact with many speech-to-text APIs. To convert the spoken language in your audio files to text. --> https://realpython.com/python-speech-recognition/

from sqlalchemy import and_
from sqlalchemy import Boolean
from sqlalchemy import case
from sqlalchemy import cast
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import delete
from sqlalchemy import desc
from sqlalchemy import Float
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import insert
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table

import sqlite3

from statistics import variance

import statsmodels as sm #For stimations in differents statistical models
import statsmodels.api as sm #Make a prediction model
import statsmodels.formula.api as smf #Make a prediction model 
from statsmodels.formula.api import ols #Create a Model from a formula and dataframe.
from statsmodels.graphics.tsaplots import plot_acf #For autocorrelation function
from statsmodels.graphics.tsaplots import plot_pacf #For simulating data and for plotting the PACF. Partial Autocorrelation Function measures the incremental benefit of adding another lag.
from statsmodels.sandbox.stats.multicomp import multipletests #To adjust the p-value when you run multiple tests.
from statsmodels.stats.power import TTestIndPower #Explain how the effect, power and significance level affect the sample size. Create results object for t-test analysis
from statsmodels.stats.power import zt_ind_solve_power #To determinate sample size. Assign and print the needed sample size
from statsmodels.stats.proportion import proportion_confint #Fon confidence interval-->proportion_confint(number of successes, number of trials, alpha value represented by 1 minus our confidence level)
from statsmodels.stats.proportion import proportion_effectsize #To determinate sample size. Standardize the effect size
from statsmodels.stats.proportion import proportions_ztest #To run the Z-score test, when you know the population standard deviation
from statsmodels.tsa.arima_model import ARIMA #Similar to use ARMA but on original data (before differencing)
from statsmodels.tsa.arima_model import ARMA #To estimate parameters from data simulated (AR model)
from statsmodels.tsa.arima_process import ArmaProcess #For simulating data and for plotting the PACF, For Simulate Autoregressive (AR) Time Series 
from statsmodels.tsa.stattools import acf #For autocorrelation function
from statsmodels.tsa.stattools import adfuller #For Augmented Dickey-Fuller unit root test. Test for Random Walk. Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive. Tests that you can use to check if a time series is stationary or not.
from statsmodels.tsa.stattools import coint #Test for cointegration
from statsmodels.tsa.stattools import kpss #For Kwiatkowski-Phillips-Schmidt-Shin test. Tests whether a time series is trend stationary or not.

from stop_words import get_stop_words #Documentation: https://pypi.org/project/stop-words/

import string
from string import Template #For working with string, regular expressions

import sympy.stats

import sys

import tabula  #For extracting tables from pdf

import tempfile  #To raise an html page in python command

import tensorflow as tf #For DeapLearning

from textblob import TextBlob #For processing textual data. Documentation https://textblob.readthedocs.io/en/dev/

from textatistic import Textatistic

import time  #To measure the elapsed wall-clock time between two points
from time import time

import timeit  #For Measure execution time of small code snippets

import webbrowser  #To raise an html page in python command 

import warnings

import wave #To see and manipulate audio files 

import wikipedia

from wordcloud import WordCloud #Documentation: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
from wordcloud import STOPWORDS

import xgboost as xgb

from zipfile import ZipFile 


###############################################################################
################################################################ AVOID WARNINGS
###############################################################################
warnings.filterwarnings('ignore', 'Objective did not converge*')                 #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
warnings.filterwarnings('default', 'Objective did not converge*')                #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
############################# To avoid matplotlib deprecated functions warnings
## import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)       #Ignore warning
warnings.filterwarnings("default",category=matplotlib.cbook.mplDeprecation)      #Restore to default
###############################################################################
## Avoid 'Reloaded modules: <module_name>' message in Python
## Go in spider to Tools > Preferences > Python interpreter
###############################################################################


###############################################################################
############################################################# TRACKING AN ERROR
###############################################################################
#import logging
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.INFO)
###############################################################################


###############################################################################
############################################# Hack to fix missing PROJ4 env var
###############################################################################
## import os
os.environ["PROJ_LIB"] = "C:\\Anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share" #Look for epsg file and set the path to environment variable PROJ_LIB
###############################################################################


###############################################################################
################################################ When to use z-score or t-tests
###############################################################################
## Usually you use a t-test when you do not know the population standard 
## deviation σ, and you use the standard error instead. You usually use the 
## z-test when you do know the population standard deviation. Although it is 
## true that the central limit theorem kicks in at around n=30. I think that 
## formally, the convergence in distribution of a sequence of t′s to a normal 
## is pretty good when n>30.
## Source: http://lecture.riazulislam.com/uploads/3/9/8/5/3985970/python_practice_3_2019_1.pdf
###############################################################################
######################################### Finding z-crtical values 95% one side    
######################################### 100*(1-alpha)% Confidence Level
###############################################################################
## import scipy.stats as stats
alpha=0.05                                                                    
stats.norm.ppf(1-alpha, loc=0, scale=1) # One-sided; ppf=percent point function, Out --> 1.6448536269514722
###############################################################################
######################################## Finding z-crtical values 95% two sides
############################################### 100*(1-alpha)% Confidence Level
###############################################################################
## import scipy.stats as stats
alpha=0.05
stats.norm.ppf(1-alpha/2, loc=0, scale=1) # Two-sided, Out --> 1.959963984540054
###############################################################################
######################################### Finding t-crtical values 95% one side
############################################### 100*(1-alpha)% Confidence Level
###############################################################################
## import scipy.stats as stats
alpha=0.05
stats.t.ppf(1-alpha, df=4) # One-sided; df=degrees of fredom, Out --> 2.13184678133629
###############################################################################
######################################## Finding t-crtical values 95% two sides
############################################### 100*(1-alpha)% Confidence Level
###############################################################################
#import scipy.stats as stats
alpha=0.05
stats.t.ppf(1-alpha/2, df=4) # Two-sided; df=degrees of fredom, Out --> 2.7764451051977987
###############################################################################


###############################################################################
#################################################### Setting the pandas options
###############################################################################
pd.options.display.float_format = '{:,.4f}'.format 
pd.reset_option("all")
pd.set_option("display.max_columns",20)
pd.set_option('display.max_rows', -1) 
pd.set_option('display.max_colwidth', -1) #display full (non-truncated) dataframe information

## dataframe.index.values.tolist() #Retrieve the index of a Dataframe as a list
## df.index.name = None #Delete index name 
register_matplotlib_converters() #Require to explicitly register matplotlib converters.

########################################################################## Save
df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
                   'mask': ['red', 'purple'],
                   'weapon': ['sai', 'bo staff']})
df.to_csv("filename.csv", index=False)

#Create ‘out.zip’ containing ‘out.csv’
compression_opts = dict(method='zip',
                        archive_name='out.csv')  
df.to_csv('out.zip', index=False,
          compression=compression_opts)

#####################################Reindex passing from 31/1/2020 to 1/1/2020
"""######################################################### Working with dates

df = pd.DataFrame({'Date': ['2013-01-31', '2013-02-28', '2013-03-31']})
df['Date'] = pd.to_datetime(df.Date, format="%Y/%m/%d")
print(df.Date.values)

df['Date'] = df.Date + pd.offsets.MonthEnd(-1) + pd.offsets.Day(df.Date[0].day-10)
print(df.Date.values)

df['Date'] = df.Date + pd.offsets.MonthEnd(-1) + pd.offsets.Day(2)
print(df.Date.values)

df['Date'] = df.Date + pd.offsets.MonthBegin(-1) 
print(df.Date.values)

df['Date'] = df.Date + pd.offsets.MonthBegin(1) 
print(df.Date.values)

Result:
>>>>>    ['2013-01-31' '2013-02-28', '2013-03-31']
>>>>>    ['2013-01-21' '2013-02-21', '2013-03-21']
>>>>>    ['2013-01-02' '2013-02-02', '2013-03-02']
>>>>>    ['2013-01-01' '2013-02-01', '2013-03-01']
>>>>>    ['2013-02-01' '2013-03-01', '2013-04-01']
"""


###############################################################################
## NORMALIZED DATA
## airquality_scalar = pd.DataFrame(StandardScaler().fit_transform(airquality_fillna), columns=airquality_fillna.columns, index=airquality_fillna.index)
########################################### Create categorical type data to use
cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  ordered=True)
################################## Change the data type of 'rating' to category
## weather['rating'] = weather.rating.astype(cats)
###############################################################################


###############################################################################
################################################## Working with dates in x axes
###############################################################################
## plt.gca().xaxis_date()
## plt.xlim(x.min(), x.max())
## plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%b-%d'))
###############################################################################
## FORCE TO INTEGER TICKS
###############################################################################
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
###############################################################################
###############################################################################


fig, axis = plt.subplots(1, 2, figsize=(11.5, 4))
#fig = plt.figure(figsize=figsize)
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
    
#fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax.set_xlim(0, 1); ax.set_ylim(0, 1);
###############################################################################
############################## Force matplotlib to not use any Xwindows backend
###############################################################################
target = mpl.get_backend() #Save the default configuration
mpl.use('Agg') #With this it avoid show the graph
mpl.use(target) #Return to the default
###############################################################################
#ax.text(1, 0.4, year, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800) #Use general metrics in plot
    

###############################################################################
######################################################### Setting images params
###############################################################################
## from matplotlib.axes._axes import _log as matplotlib_axes_logger           #To avoid warnings
matplotlib_axes_logger.setLevel('ERROR')                                      #To avoid warnings
matplotlib_axes_logger.setLevel(0)                                            #To restore default

################################################################ Ticks fontsize
#mpl.rc('xtick', labelsize=6) #global
#mpl.rc('ytick', labelsize=6) #global
#ax.tick_params(labelsize=6)
#plt.xticks(fontsize= )
#ax.set_xticklabels(xlabels, fontsize= )
#plt.setp(ax.get_xticklabels(), fontsize=)
#plt.setp(ax.get_label(), fontsize=)                              ## axis label 
#plt.setp(ax.get_ticklabels(), fontsize=)                         ## axis label 
#ax.tick_params(axis='x', labelsize= )


#ax.set(xlabel='area (square feet)', ylabel='price (dollars)', title=title)
################################################################ Label fontsize
#mpl.rc('axes', labelsize=6) #Global
#ax.xaxis.label.set_size(6)
#ax.yaxis.label.set_size(6)
#for label in ax1.xaxis.get_ticklabels():
#    label.set_color('red')
#    label.set_rotation(45)
#    label.set_fontsize(16)
    
#for ax in axis.reshape(-1):
#    ax.tick_params(labelsize=6) # ticks fontsize
#    ax.xaxis.label.set_size(6)
#    ax.yaxis.label.set_size(6)
     

ax.set_anchor('N');                                                           #Aling to the top
## fig.legend(handles, labels)                                                #To make a single legend for many subplots with matplotlib
## handles, labels = ax.get_legend_handles_labels()                           #To make a single legend for many subplots with matplotlib
fig = plt.gcf()                                                               #Get the current figure.
ax = plt.gca()                                                                #To get the current active axes: x_ax = ax.coords[0]; y_ax = ax.coords[1]
ax.legend().set_visible(False)
ax.xaxis.set_major_locator(plt.MaxNLocator(3))                                #Define the number of ticks to show on x axis.
ax.xaxis.set_major_locator(mdates.AutoDateLocator())                          #Other way to Define the number of ticks to show on x axis.
ax.xaxis.set_major_formatter(mdates.AutoDateFormatter())                      #Other way to Define the number of ticks to show on x axis.
ax = plt.gca()                                                                #To get the current active axes: x_ax = ax.coords[0]; y_ax = ax.coords[1]
ax.grid(True); 
ax.axis('off')                                                                #Turn off the axis in a graph, subplots.
###################################### To supress the scientist notation in plt
## from matplotlib.ticker import StrMethodFormatter                           #Import the necessary library to delete the scientist notation
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
ax.yaxis.set_major_locator(MaxNLocator(integer=True))                         #Format integer 
ax.tick_params(labelsize=6)                                                   #axis : {'x', 'y', 'both'}
ax.tick_params(axis='x', rotation=45)                                         #Set rotation atributte
ax.ticklabel_format(axis='x', style='plain')                                  #{'sci' (or 'scientific'), 'plain'} plain turns off scientific notation, https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html#matplotlib.axes.Axes.ticklabel_format
ax.ticklabel_format(style='plain')

font = {'size'   : 9}
mpl.rc('font', **font)

## plt.close(fig)
plt.style.use('dark_background')
plt.axis('off')                                                               #Turn off the axis in a graph, subplots.
ax.get_xaxis().set_visible(False)
ax.set_xticks([])
plt.xticks(fontsize=7); plt.yticks(fontsize=8);
plt.xticks(rotation=45)
plt.yticks(rotation=90)                                                       #rotate y-axis labels by 90 degrees
plt.xlabel('Average getted', fontsize=8); 
plt.ylabel('Number of times', fontsize=8, rotation=90); # Labeling the axis.
plt.gca().set_yticklabels(['{:,.2f}'.format(x) for x in plt.gca().get_yticks()]) #To format y axis
plt.legend().set_visible(False)
plt.tight_layout()                                                            #To fix overlapping the supplots elements.
plt.savefig("sample.jpg")                                                     #save image of `plt`

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=6)

params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 5), 'axes.labelsize': 'x-large', 'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
plt.rcParams.update(**params)
plt.rcParams.update({'axes.labelsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6})
plt.rcParams["legend.fontsize"] = 8
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.constrained_layout.h_pad'] = 0.09
plt.rcParams["axes.labelsize"] = 8                                            #Font of xlabel and ylabel
plt.rcParams['figure.max_open_warning'] = 60                                  
plt.rcParams['xtick.labelsize']=6
plt.rcParams['ytick.labelsize']=6
    
plt.rcParams.update({'figure.max_open_warning': 0})                           #To solve the max images open
plt.rcParams.update({'axes.labelsize': 7})                                    #Font of xlabel and ylabel
plt.subplots_adjust(left=None, right=None, bottom=.15, top=None, hspace=.7, wspace=.4);
    
###############################################################################
## Visualiize PMF
###############################################################################
## # Pmf of two equal throw dice
## values, counts = np.unique(wins, return_counts=True)
## ax.stem(values, counts, use_line_collection=True);
    
###############################################################################
#                                  Other properties to us in rcParams:
#                                        axes.titlesize : 24
#                                        axes.labelsize : 20
#                                        lines.linewidth : 3
#                                        lines.markersize : 10
#                                        xtick.labelsize : 16
#                                        ytick.labelsize : 16
## Complete list of properties: https://matplotlib.org/tutorials/introductory/customizing.html
###############################################################################
plt.rcParams = plt.rcParamsDefault
plt.style.use('default')
mpl.style.use('default')
###############################################################################


###############################################################################
###################################################### Working with Tensor Flow
###############################################################################
## tf.compat.v1.set_random_seed(SEED)                                            #Instead of tf.set_random_seed, because it is deprecated.
###############################################################################


###############################################################################
##################################################### Setting the numpy options
###############################################################################
np.set_printoptions(precision=3)                                              #precision set the precision of the output:
np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
np.set_printoptions(threshold=8)                                              #Return to default value.
## np.random.seed(SEED)
np.set_printoptions(suppress=True) #suppress suppresses the use of scientific notation for small numbers
np.set_printoptions(formatter={'float': '{:,.3f}'.format}) #float_kind instead of float
np.set_printoptions(formatter={'float': None}) #Return to default

###############################################################################
## TO AVOID CIENTIFIC NOTATION
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(np.logspace(1, 10, 10)) #base=10, default
np.set_printoptions(formatter = {'float': None}) #RETURN TO NORMALITY

###############################################################################
####################################################### Setting seaborn options
###############################################################################
sns.set(font_scale=0.8)                                                       #Font
sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot
sns.set(color_codes=True)                                                     #Habilita el uso de los codigos de color
sns.set()                                                                     #Seaborn defult style
## sns.set_style(this_style)                                                     #['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
sns.despine(left=True)                                                        #Remove the spines (all borders)
sns.palettes.SEABORN_PALETTES                                                 #Despliega todas las paletas disponibles 
sns.palplot(sns.color_palette())                                              #Display a palette
sns.color_palette()                                                           #The current palette
sns.set(style='whitegrid', palette='pastel', color_codes=True)
# Applying temporary comfiguration in plot
#with sns.plotting_context(rc={'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 
#                              'legend.fontsize': 10, 'font.size': 10}):

sns.mpl.rc('figure', figsize=(10,6))
## sns.distplot(data, bins=10, hist_kws={"density": True})                       #Example of hist_kws parameter 
## sns.distplot(data, hist=False, hist_kws=dict(edgecolor='k', linewidth=1))     #Example of hist_kws parameter 
###############################################################################


###############################################################################
######################################################## Show a basic html page
###############################################################################
tmp = tempfile.NamedTemporaryFile()
path = tmp.name + '.html'
f = open(path, 'w')
f.write("<html><body><h1>Test</h1></body></html>")
f.close()
webbrowser.open('file://' + path)


###############################################################################
########################################### Format y printing spetial caracters
###############################################################################
print("The area of your rectangle is {}cm\u00b2".format(32))                 #Print the superscript 2
"""
>>> SUB = str.maketrans("0123456789  Ⴖꓴ ⊄ α", "µπ∫»«≈ᵢ₋₀₁₂₃₄₅₆₇₈₉")
>>> SUP = str.maketrans("0123456789", "˙ᵐᵘʲⁿⁱ⁰¹²³⁴⁵⁶⁷⁸⁹⁻ˣᶰᶻᵏᵐᵃᵇˣʸ")
# https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts
# https://es.wikipedia.org/wiki/Flecha_(s%C3%ADmbolo)
>>> "H2SO4".translate(SUB)
'H₂SO₄'
"""
#plt.xlabel(r'$\tau$ (games)')
###############################################################################


###############################################################################
## Preparing the environment
###############################################################################
    
# Global variables
#suptitle_param = dict(color='darkblue', fontsize=11)
#title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
#plot_param     = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
#                  'legend.fontsize': 8, 'font.size': 8}
#figsize        = (12.1, 5.9)
#SEED = 42
#np.random.seed(SEED) 

#Global configuration
#np.set_printoptions(formatter={'float': '{:,.3f}'.format})
#pd.options.display.float_format = '{:,.3f}'.format 
#plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
#                     'legend.fontsize': 8, 'font.size': 8})
#suptitle_param   = dict(color='darkblue', fontsize=10)
#title_param      = {'color': 'darkred', 'fontsize': 12}


#Return to default
#np.set_printoptions(formatter = {'float': None}) #Return to default
#pd.options.display.float_format = None
#pd.set_option("display.max_columns", 0)

#plt.style.use('default')

# Adding %-formatting to the y-axis
#ax.yaxis.set_major_formatter(mtick.PercentFormatter())

###############################################################################
## Functions applying to read data into pandas
###############################################################################
## Course 48 Data Manipulations with Pandas >> 02...py
animal_shelter = pd.read_excel('austin animal intakes oct 2013 nov 2016.xlsx', 
                                sheet_name  =  'Austin_Animal_Center_Intakes',
                                parse_dates = ['DateTime', ],
                                na_values   = ['Unknown', ''],# ['Unknown', '', 'NULL'], --> NULL is not necesary, because is already a default identifier.
                                converters  = {'MonthYear'       : lambda x: pd.to_datetime(x).strftime('%Y-%m'),
                                               'Age upon Intake' : lambda x: pd.to_numeric(re.sub(r' year\w*', '', x), errors='coerce')},
                                dtype       = {#'Breed'           : 'category',
                                               'Color'           : 'category',
                                               'Animal Type'     : 'category',
                                               'Intake Condition': 'category',
                                               'Intake Type'     : 'category',
                                               'Sex upon Intake' : 'category'})

###############################################################################
## Importing bool columns
###############################################################################
from io import StringIO
import numpy as np
import pandas as pd

pd.read_csv(StringIO("""var1, var2
0,   0
0,   1
1,   3
-9,  0
0,   2
1,   7"""), converters = {'var1': lambda x: bool(int(x)) if x != '-9' else np.nan})


###############################################################################
## REad a text file
###############################################################################
# file-input.py
f = open('dhs_report_reformatted.json','r')
message = f.read()
print(message)
f.close()

# Open read and close in one line
with open('dhs_report_reformatted.json','r') as f: yep_ClientId = f.read()

#with open(filename, 'r', encoding='utf-8') as f:
with open('grail.txt','r') as f: 
    scene_one = f.readline()  # Read one line
    scene_one = f.readlines() # Read oll lines and put them in a list
    scene_one = f.read()      # Read the complete file at once


###############################################################################
## Read a zip file
###############################################################################
# import io
# import zipfile
# 
# with zipfile.ZipFile("files.zip") as zf:
#     with io.TextIOWrapper(zf.open("text1.txt"), encoding="utf-8") as f:
#         ...
        
        
###############################################################################
## FORMAT NUMPY AS FRACTION - NumPy: convert decimals to fractions
#import fractions
#np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})
#print(A_inv)
###############################################################################
## MAX DICT

d = {'a': 10,
    'b': 20,
    'c': 30,
    'd': 25,
    'e': 15,
    'f': 5}

max_dict = max(d, key=d.get)
print(max_dict)


###############################################################################
## RETRIEVE SOURCE CODE
###############################################################################
#import inspect
#inspect.getsource(entity_type) #>>> 'def entity_type(word):\n    _type = None\n    if word.text in colors:\n        _type = "color"\n    elif word.text in items:\n        _type = "item"\n    return _type\n'

###############################################################################
## WHAT IS THE ENVIRONMENT???
###############################################################################
import sys
def get_env():
    sp = sys.path[1].split("\\")
    if "envs" in sp:
        return sp[sp.index("envs") + 1]
    else:
        return ""

print(get_env())

###############################################################################
## TO SEE EVERYTHING, TO SEE ALL PROPERTIES IN A VARIABLE
###############################################################################
#$ dir(variable)
















