# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:55:22 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 2: Exploratory Data Analysis
    In this chapter, you will prepare for statistical concepts related to exploratory data analysis. 
    The topics include descriptive statistics, dealing with categorical variables, and relationships 
    between variables. The exercises will prepare you for an analytical assessment or stats-based 
    coding question.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import math
import matplotlib.pyplot             as plt                                   #For creating charts
import numpy                         as np                                    #For making operations in lists
import pandas                        as pd                                    #For loading tabular data
import seaborn                       as sns                                   #For visualizing data

from sklearn                         import preprocessing
from scipy.stats                     import pearsonr                          #For learning machine. For Pearson's Correlation test. To check if two samples are related.

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 123
np.random.seed(SEED)

pd.options.display.float_format = '{:,.4f}'.format

def print_pdf_data(data, title, x_label, bins=-1):
    """Print the PDF of the data."""
    mu = data.mean()
    sigma = data.std()
    median = np.median(data)
    #theorical = np.random.normal(mu,sigma,100000)
    
    title = "{}{}".format(title, ("\nSkewed Left" if (mu < median) else "\nSkewed Right" if (mu > median) else ""))
    bins = (round((data.max() - data.min())*(len(data)**(1/3))/(3.49*sigma)) if bins==-1 else bins)
    sns.set_style('darkgrid')
    plt.figure()
    sns.distplot(data, kde=False, norm_hist=True, bins=bins)
    #sns.distplot(theorical, color='black', hist=False, label='Theorical', hist_kws=dict(edgecolor='k', linewidth=1))
    plt.axvline(x=mu, color='b', label='Mean', linestyle='-', linewidth=2)
    plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean
    plt.xlabel(x_label, fontsize=8); plt.ylabel('Probability (PDF)', fontsize=8); # Labeling the axis.
    plt.xticks(fontsize=8); plt.yticks(fontsize=8);
    plt.legend(loc='best', fontsize='small')
    plt.title(title, color='red')
    plt.suptitle(topic, color='navy');  # Setting the titles.
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=0.5);
    plt.show()
    plt.style.use('default')
    
    return mu, sigma, median


def print_pearson_coef(serie1, serie2):
    # Compute and print the Pearson correlation
    r = serie1.corr(serie2)
    print("Pearson correlation:", r)
    
    data = pd.concat([serie1, serie2], axis=1).dropna()
    stat, p = pearsonr(data[serie1.name], data[serie2.name])
    print('stat=%.8f, p=%.8f' % (stat, p))

    if p > 0.05:
        print('Probably independent (p > 0.05).')
    else:
        print('Probably dependent (p <= 0.05).')

    # Calculate the r-squared value and print the result
    r2 = r**2
    print("{} explains around {:,.0f}% of the variability in the {} feature.\n".format(serie1.name, r2*100, serie2.name))


    
print("****************************************************")
topic = "2. Mean or median"; print("** %s\n" % topic)

file = "weather-ex-australia.data"
weather = pd.read_fwf(file).sort_index()

columns = ["Temp3pm", "Temp9am"]
for column in columns:
    data = weather[column]
    mean, sigma, median = print_pdf_data(data, title = "Weather in Australia", x_label = column, bins=10)
    
    print("Column: ", column)
    print('  Mean:', mean) # Assign the mean to the variable and print the result
    print('  Median:', median, "\n") # Assign the median to the variable and print the result

    


print("****************************************************")
topic = "3. Standard deviation by hand"; print("** %s\n" % topic)

# Create a sample list
nums = [1, 2, 3, 4, 5]

# Compute the mean of the list
mean = sum(nums) / len(nums)

# Compute the variance and print the std of the list
variance = sum(pow(x - mean, 2) for x in nums) / len(nums)
std = math.sqrt(variance)
print("Standard deviation manually computed   : ", std)

# Compute and print the actual result from numpy
real_std = np.array(nums).std()
print("Standard deviation usign python fuction: ", real_std)




print("****************************************************")
topic = "5. Encoding techniques"; print("** %s\n" % topic)

file = "laptops-prices.data"
laptops = pd.read_fwf(file, index_col="Id").sort_index()

print("Method fo encoding categorival data: LABEL ENCODING")
# Create the encoder and print our encoded new_vals
encoder = preprocessing.LabelEncoder()
new_vals = encoder.fit_transform(laptops.Company)
print(new_vals)

print("Method fo encoding categorival data: ONE HOT ENCODING")
# One-hot encode Company for laptops2
laptops2 = pd.get_dummies(data=laptops, columns=["Company"])
print(laptops2.head())




print("****************************************************")
topic = "6. Exploring laptop prices"; print("** %s\n" % topic)

file = "laptops-prices2.data"
laptops2 = pd.read_fwf(file, index_col="Id").sort_index()

# Get some initial info about the data
print(laptops2.info())

# Produce a countplot of companies
plt.figure()
ax = sns.countplot(laptops2.Company)
plt.title("How many observations are from each brand?", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.show()

# Visualize the relationship with price
laptops2.boxplot("Price", "Company", rot=0, figsize=(10,5), vert=False)
plt.gca().set_xticklabels(['{:,.2f}'.format(x) for x in plt.gca().get_xticks()])
plt.xticks(fontsize=8); plt.xlabel('Price in $'); plt.ylabel('Company', rotation=90);
plt.title("Relationship between the Price and Company", color='red')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=0.15, bottom=None, right=None, top=0.85, wspace=None, hspace=None);
plt.show()




print("****************************************************")
topic = "8. Types of relationships"; print("** %s\n" % topic)

#file = "weather-dataset-australia.csv" 
#weather_AUS = pd.read_csv(file, index_col="Date", parse_dates=True).sort_index()
file = "weather-ex-australia_4.data"
weather_AUS = pd.read_fwf(file).sort_index()

plt.figure(figsize=(10,5))
# Display a scatter plot and examine the relationship
plt.subplot(2,2,1)
plt.scatter(weather_AUS.MinTemp, weather_AUS.MaxTemp)
plt.xticks(fontsize=8); plt.yticks(fontsize=8); 
plt.xlabel('MinTemp', fontsize=8); plt.ylabel('MaxTemp', fontsize=8);
plt.title("Relationship between MinTemp and MaxTemp\n(Positive Relationship)", color='red', fontsize=9)

# Display a scatter plot and examine the relationship
plt.subplot(2,2,2)
plt.scatter(weather_AUS.MaxTemp, weather_AUS.Humidity9am)
plt.xticks(fontsize=8); plt.yticks(fontsize=8); 
plt.xlabel('MaxTemp', fontsize=8); plt.ylabel('Humidity9am', fontsize=8);
plt.title("Relationship between MaxTemp and Humidity9am\n(Negative Relationship)", color='red', fontsize=9)


# Display a scatter plot and examine the relationship
plt.subplot(2,2,3)
plt.scatter(weather_AUS.MinTemp, weather_AUS.Humidity3pm)
plt.xticks(fontsize=8); plt.yticks(fontsize=8); 
plt.xlabel('MinTemp', fontsize=8); plt.ylabel('Humidity3pm', fontsize=8);
plt.title("Relationship between MinTemp and Humidity3pm\n(No Apparent Relationship)", color='red', fontsize=9)

plt.subplot(2,2,4)
plt.axis('off')

plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.6);
plt.show()




print("****************************************************")
topic = "9. Pearson correlation"; print("** %s\n" % topic)

plt.rc('xtick',labelsize=7)
plt.rc('ytick',labelsize=7)
plt.rcParams["axes.labelsize"] = 7

# Generate the pair plot for the weather dataset
sns.pairplot(weather_AUS, height=1.4, aspect=2, 
             plot_kws=dict(edgecolor="navy", linewidth=0.5, s=25, alpha=0.5),
             diag_kws=dict(color='steelblue', edgecolor='black', linewidth=0.5)) # "diag" adjusts/tunes the diagonal plots
plt.suptitle(topic, color='navy', fontweight='bold');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0.3, hspace=0.3);
plt.show()

# Generate the pair plot for the weather dataset
pp = sns.pairplot(weather_AUS, height=1.4, aspect=2, 
                  plot_kws=dict(edgecolor="navy", linewidth=0.5, s=25, alpha=0.5),
                  diag_kws=dict(shade=True, linewidth=0.5), # "diag" adjusts/tunes the diagonal plots
                  diag_kind="kde")
fig = pp.fig 
fig.subplots_adjust(top=0.9, wspace=0.3, hspace=0.3)
fig.suptitle('{} (kde)'.format(topic), fontweight='bold')
plt.show()

plt.style.use('default')


# Look at the scatter plot for the humidity variables
plt.figure()
plt.scatter(weather_AUS.Humidity9am, weather_AUS.Humidity3pm)
plt.xticks(); plt.yticks(); 
plt.xlabel('Humidity9am'); plt.ylabel('Humidity3pm');
plt.title("Relationship between Humidity9am and Humidity3pm", color='red')
plt.suptitle(topic, color='navy', fontweight='bold');  # Setting the titles.
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.6);
plt.show()

print("Pearson correlation in Australian Weather Dataset")
print_pearson_coef(weather_AUS.Humidity9am, weather_AUS.Humidity3pm)

"""
# Compute and print the Pearson correlation
r = weather_AUS['Humidity9am'].corr(weather_AUS.Humidity3pm)
print("Pearson correlation:", r)

weather_AUS2 = weather_AUS.dropna()
stat, p = pearsonr(weather_AUS2.Humidity9am, weather_AUS2.Humidity3pm)
print('stat=%.8f, p=%.8f' % (stat, p))

if p > 0.05:
	print('Probably independent (p > 0.05).')
else:
	print('Probably dependent (p <= 0.05).')

# Calculate the r-squared value and print the result
r2 = r**2
print("Humidity9am explains around {:,.0f}% of the variability in the Humidity3pm variable.".format(r2*100))
"""



print("****************************************************")
topic = "10. Sensitivity to outliers"; print("** %s\n" % topic)

df = pd.DataFrame({
        "X": [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
        "Y": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    })

# Display the scatter plot of X and Y
plt.figure()
plt.scatter(df.X, df.Y)
plt.xticks(); plt.yticks(); 
plt.xlabel('X Feature'); plt.ylabel('Y Feature');
plt.title("Anscombe's quartet", color='red')
plt.suptitle(topic, color='navy', fontweight='bold');  # Setting the titles.
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.6);
plt.show()

# Compute and print the correlation once more
print("Pearson correlation in Anscombe's quarter with outliers")
print_pearson_coef(df.X, df.Y)

# Drop the outlier from the dataset
df = df.drop(index=2)

# Compute and print the correlation once more
print("Pearson correlation in Anscombe's quarter without outliers")
print_pearson_coef(df.X, df.Y)



print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import contextily                                                             #To add a background web map to our plot
#import inspect                                                                #Used to get the code inside a function
#import folium                                                                 #To create map street folium.__version__'0.10.0'
#import geopandas         as gpd                                               #For working with geospatial data 
#import math                                                                   #https://docs.python.org/3/library/math.html
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.dates  as mdates                                            #For providing sophisticated date plotting capabilities
#import matplotlib.pyplot as plt                                               #For creating charts
#import numpy             as np                                                #For making operations in lists
#import pandas            as pd                                                #For loading tabular data
#import seaborn           as sns                                               #For visualizing data
#import pprint                                                                 #Import pprint to format disctionary output
#import missingno         as msno                                              #Missing data visualization module for Python

#import os                                                                     #To raise an html page in python command
#import tempfile                                                               #To raise an html page in python command
#import webbrowser                                                             #To raise an html page in python command  

#import calendar                                                               #For accesing to a vary of calendar operations
#import datetime                                                               #For accesing datetime functions
#import math                                                                   #For accesing to a complex math operations
#import nltk                                                                   #For working with text data
#import random                                                                 #For generating random numbers
#import re                                                                     #For regular expressions
#import tabula                                                                 #For extracting tables from pdf
#import timeit                                                                 #For Measure execution time of small code snippets
#import time                                                                   #To measure the elapsed wall-clock time between two points
#import scykit-learn                                                           #For performing machine learning  
#import warnings
#import wikipedia

#from collections                     import defaultdict                       #Returns a new dictionary-like object
#from datetime                        import date                              #For obteining today function
#from datetime                        import datetime                          #For obteining today function
#from functools                       import reduce                            #For accessing to a high order functions (functions or operators that return functions)
#from glob                            import glob                              #For using with pathnames matching
#from itertools                       import combinations                      #For iterations
#from itertools                       import cycle                             #Used in the function plot_labeled_decision_regions()
#from math                            import ceil                              #Used in the function plot_labeled_decision_regions()
#from math                            import floor                             #Used in the function plot_labeled_decision_regions()
#from math                            import radian                            #For accessing a specific math operations
#from math                            import sqrt
#from matplotlib                      import colors                            #To create custom cmap
#from matplotlib.ticker               import StrMethodFormatter                #Import the necessary library to delete the scientist notation
#from mpl_toolkits.mplot3d            import Axes3D
#from numpy.random                    import randint                           #numpy.random.randint(low, high=None, size=None, dtype='l')-->Return random integers from low (inclusive) to high (exclusive).  
#from pandas.api.types                import CategoricalDtype                  #For categorical data
#from pandas.plotting                 import parallel_coordinates              #For Parallel Coordinates
#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
#from shapely.geometry                import LineString                        #(Geospatial) To create a Linestring geometry column 
#from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 
#from shapely.geometry                import Polygon                           #(Geospatial) To create a point geometry column 
#from string                          import Template                          #For working with string, regular expressions


#from bokeh.io                        import curdoc                            #For interacting visualizations
#from bokeh.io                        import output_file                       #For interacting visualizations
#from bokeh.io                        import show                              #For interacting visualizations
#from bokeh.plotting                  import ColumnDataSource                  #For interacting visualizations
#from bokeh.plotting                  import figure                            #For interacting visualizations
#from bokeh.layouts                   import column                            #For interacting visualizations
#from bokeh.layouts                   import gridplot                          #For interacting visualizations
#from bokeh.layouts                   import row                               #For interacting visualizations
#from bokeh.layouts                   import widgetbox                         #For interacting visualizations
#from bokeh.models                    import Button                            #For interacting visualizations
#from bokeh.models                    import CategoricalColorMapper            #For interacting visualizations
#from bokeh.models                    import CheckboxGroup                     #For interacting visualizations
#from bokeh.models                    import ColumnDataSource                  #For interacting visualizations
#from bokeh.models                    import HoverTool                         #For interacting visualizations
#from bokeh.models                    import RadioGroup                        #For interacting visualizations
#from bokeh.models                    import Select                            #For interacting visualizations
#from bokeh.models                    import Slider                            #For interacting visualizations
#from bokeh.models                    import Toggle                            #For interacting visualizations
#from bokeh.models.widgets            import Panel                             #For interacting visualizations
#from bokeh.models.widgets            import Tabs                              #For interacting visualizations
#from bokeh.palettes                  import Spectral6                         #For interacting visualizations


#import keras                                                                  #For DeapLearning
#import keras.backend as k                                                     #For DeapLearning
#from keras.applications.resnet50     import decode_predictions                #For DeapLearning
#from keras.applications.resnet50     import preprocess_input                  #For DeapLearning
#from keras.applications.resnet50     import ResNet50                          #For DeapLearning
#from keras.callbacks                 import EarlyStopping                     #For DeapLearning
#from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
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


#import networkx          as nx                                                #For Network Analysis in Python
#import nxviz             as nv                                                #For Network Analysis in Python
#from nxviz                           import ArcPlot                           #For Network Analysis in Python
#from nxviz                           import CircosPlot                        #For Network Analysis in Python 
#from nxviz                           import MatrixPlot                        #For Network Analysis in Python 


#import scipy.stats as stats                                                   #For accesign to a vary of statistics functiosn
#from scipy.cluster.hierarchy         import dendrogram                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import fcluster                          #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import linkage                           #For learning machine - unsurpervised
#from scipy.ndimage                   import gaussian_filter                   #For working with images
#from scipy.ndimage                   import median_filter                     #For working with images
#from scipy.signal                    import convolve2d                        #For learning machine - deep learning
#from scipy.sparse                    import csr_matrix                        #For learning machine 
#from scipy.special                   import expit as sigmoid                  #For learning machine 
#from scipy.stats                     import anderson                          #For Anderson-Darling Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import bernoulli                         #Generate bernoulli data
#from scipy.stats                     import binom                             #Generate binomial data
#from scipy.stats                     import chi2_contingency                  #For Chi-Squared Test. Tests whether two categorical variables are related or independent
#from scipy.stats                     import f_oneway                          #For Analysis of Variance Test. Tests whether the means of two or more independent samples are significantly different.
#from scipy.stats                     import friedmanchisquare                 #For Friedman Test. Tests whether the distributions of two or more paired samples are equal or not.
#from scipy.stats                     import kendalltau                        #For Kendall's Rank Correlation Test. To check if two samples are related.
#from scipy.stats                     import kruskal                           #For Kruskal-Wallis H Test. Tests whether the distributions of two or more independent samples are equal or not.
#from scipy.stats                     import mannwhitneyu                      #For Mann-Whitney U Test. Tests whether the distributions of two independent samples are equal or not.
#from scipy.stats                     import norm                              #Generate normal data
#from scipy.stats                     import normaltest                        #For D'Agostino's K^2 Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import pearsonr                          #For learning machine. For Pearson's Correlation test. To check if two samples are related.
#from scipy.stats                     import randint                           #For learning machine 
#from scipy.stats                     import shapiro                           #For Shapiro-Wilk Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import spearmanr                         #For Spearman's Rank Correlation Test.  To check if two samples are related.
#from scipy.stats                     import ttest_ind                         #For Student's t-test. Tests whether the means of two independent samples are significantly different.
#from scipy.stats                     import ttest_rel                         #For Paired Student's t-test. Tests whether the means of two paired samples are significantly different.
#from scipy.stats                     import wilcoxon                          #For Wilcoxon Signed-Rank Test. Tests whether the distributions of two paired samples are equal or not.


#from skimage                         import exposure                          #For working with images
#from skimage                         import measure                           #For working with images
#from skimage.filters.thresholding    import threshold_otsu                    #For working with images
#from skimage.filters.thresholding    import threshold_local                   #For working with images 


#from sklearn                         import datasets                          #For learning machine
#from sklearn                         import preprocessing
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
#from sklearn.feature_selection       import chi2                              #For learning machine
#from sklearn.feature_selection       import SelectKBest                       #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                   #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                 #For learning machine
#from sklearn.feature_extraction.text import TfidfVectorizer                   #For learning machine - unsurpervised
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
#from sklearn.model_selection         import KFold                             #For learning machine
#from sklearn.model_selection         import RandomizedSearchCV                #For learning machine
#from sklearn.model_selection         import train_test_split                  #For learning machine
#from sklearn.multiclass              import OneVsRestClassifier               #For learning machine
#from sklearn.neighbors               import KNeighborsClassifier as KNN       #For learning machine
#from sklearn.pipeline                import FeatureUnion                      #For learning machine
#from sklearn.pipeline                import make_pipeline                     #For learning machine - unsurpervised
#from sklearn.pipeline                import Pipeline                          #For learning machine
#from sklearn.preprocessing           import FunctionTransformer               #For learning machine
#from sklearn.preprocessing           import Imputer                           #For learning machine
#from sklearn.preprocessing           import LabelEncoder                      #Create the encoder and print our encoded new_vals
#from sklearn.preprocessing           import MaxAbsScaler                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing           import Normalizer                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing           import normalize                         #For learning machine - unsurpervised
#from sklearn.preprocessing           import scale                             #For learning machine
#from sklearn.preprocessing           import StandardScaler                    #For learning machine
#from sklearn.svm                     import SVC                               #For learning machine
#from sklearn.tree                    import DecisionTreeClassifier            #For learning machine - supervised
#from sklearn.tree                    import DecisionTreeRegressor             #For learning machine - supervised


#import statsmodels                   as sm                                    #For stimations in differents statistical models
#import statsmodels.api               as sm                                    #Make a prediction model
#import statsmodels.formula.api       as smf                                   #Make a prediction model    
#from statsmodels.graphics.tsaplots   import plot_acf                          #For autocorrelation function
#from statsmodels.graphics.tsaplots   import plot_pacf                         #For simulating data and for plotting the PACF. Partial Autocorrelation Function measures the incremental benefit of adding another lag.
#from statsmodels.tsa.arima_model     import ARIMA                             #Similar to use ARMA but on original data (before differencing)
#from statsmodels.tsa.arima_model     import ARMA                              #To estimate parameters from data simulated (AR model)
#from statsmodels.tsa.arima_process   import ArmaProcess                       #For simulating data and for plotting the PACF, For Simulate Autoregressive (AR) Time Series 
#from statsmodels.tsa.stattools       import acf                               #For autocorrelation function
#from statsmodels.tsa.stattools       import adfuller                          #For Augmented Dickey-Fuller unit root test. Test for Random Walk. Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive. Tests that you can use to check if a time series is stationary or not.
#from statsmodels.tsa.stattools       import coint                             #Test for cointegration
#from statsmodels.tsa.stattools       import kpss                              #For Kwiatkowski-Phillips-Schmidt-Shin test. Tests whether a time series is trend stationary or not.


#import tensorflow              as tf                                          #For DeapLearning



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
#plt.rc('xtick',labelsize=8)
#plt.rc('ytick',labelsize=6)
#plt.rcParams['figure.max_open_warning'] = 60                                  #params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 5), 'axes.labelsize': 'x-large', 'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
#plt.rcParams["legend.fontsize"] = 8
#plt.style.use('dark_background')
#plt.style.use('default')
#plt.xticks(fontsize=7); plt.yticks(fontsize=8);
#plt.xticks(rotation=45)
#plt.xlabel('Average getted', fontsize=8); 
#plt.ylabel('Number of times', fontsize=8, rotation=90); # Labeling the axis.
#plt.gca().set_yticklabels(['{:,.2f}'.format(x) for x in plt.gca().get_yticks()]) #To format y axis
#ax.xaxis.set_major_locator(plt.MaxNLocator(3))                                #Define the number of ticks to show on x axis.
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())                          #Other way to Define the number of ticks to show on x axis.
#ax.xaxis.set_major_formatter(mdates.AutoDateFormatter())                      #Other way to Define the number of ticks to show on x axis.
#ax = plt.gca()                                                                #To get the current active axes: x_ax = ax.coords[0]; y_ax = ax.coords[1]

#plt.axis('off')                                                               #Turn off the axis in a graph, subplots.
#plt.xticks(rotation=45)                                                       #rotate x-axis labels by 45 degrees
#plt.yticks(rotation=90)                                                       #rotate y-axis labels by 90 degrees
#plt.savefig("sample.jpg")                                                     #save image of `plt`

#To supress the scientist notation in plt
#from matplotlib.ticker import StrMethodFormatter                              #Import the necessary library to delete the scientist notation
#ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation

#from matplotlib.axes._axes import _log as matplotlib_axes_logger              #To avoid warnings
#matplotlib_axes_logger.setLevel('ERROR')
#matplotlib_axes_logger.setLevel(0)                                            #To restore default

#ax.tick_params(labelsize=6)                                                   #axis : {'x', 'y', 'both'}
#ax.tick_params(axis='x', rotation=45)                                         #Set rotation atributte

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.
#np.random.seed(SEED)

#tf.compat.v1.set_random_seed(SEED)                                            #Instead of tf.set_random_seed, because it is deprecated.

#sns.set(font_scale=0.8)                                                       #Font
#sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot
#sns.set(color_codes=True)                                                     #Habilita el uso de los codigos de color
#sns.set()                                                                     #Seaborn defult style
#sns.set_style(this_style)                                                     #['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
#sns.despine(left=True)                                                        #Remove the spines (all borders)
#sns.palettes.SEABORN_PALETTES                                                 #Despliega todas las paletas disponibles 
#sns.palplot(sns.color_palette())                                              #Display a palette
#sns.color_palette()                                                           #The current palette
#sns.set(style=”whitegrid”, palette=”pastel”, color_codes=True)
#sns.mpl.rc(“figure”, figsize=(10,6))
#sns.distplot(data, bins=10, hist_kws={"density": True})                       #Example of hist_kws parameter 
#sns.distplot(data, hist=False, hist_kws=dict(edgecolor='k', linewidth=1))     #Example of hist_kws parameter 


#warnings.filterwarnings('ignore', 'Objective did not converge*')              #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
#warnings.filterwarnings('default', 'Objective did not converge*')             #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


#Create categorical type data to use
#cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  ordered=True)
# Change the data type of 'rating' to category
#weather['rating'] = weather.rating.astype(cats)


#print("The area of your rectangle is {}cm\u00b2".format(area))                 #Print the superscript 2

### Show a basic html page
#tmp=tempfile.NamedTemporaryFile()
#path=tmp.name+'.html'
#f=open(path, 'w')
#f.write("<html><body><h1>Test</h1></body></html>")
#f.close()
#webbrowser.open('file://' + path)
