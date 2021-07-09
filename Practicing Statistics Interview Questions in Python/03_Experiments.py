# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:47:15 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 3: Statistical Experiments and Significance Testing
    Prepare to dive deeper into crucial concepts regarding experiments and testing by reviewing 
    confidence intervals, hypothesis testing, multiple tests, and the role that power and sample 
    size play. We'll also discuss types of errors, and what they mean in practice.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot             as plt                                   #For creating charts
import numpy                         as np                                    #For making operations in lists
import pandas                        as pd                                    #For loading tabular data

from scipy.stats                     import binom                             #Generate binomial data
from scipy.stats                     import sem                               #For statistic thinking 
from scipy.stats                     import t                                 #For statistic thinking 
from scipy.stats                     import ttest_ind                         #For Student's t-test. Tests whether the means of two independent samples are significantly different.
from statsmodels.sandbox.stats.multicomp import multipletests                 #To adjust the p-value when you run multiple tests.
from statsmodels.stats.power         import TTestIndPower                     #Explain how the effect, power and significance level affect the sample size. Create results object for t-test analysis
from statsmodels.stats.power         import zt_ind_solve_power                #To determinate sample size. Assign and print the needed sample size
from statsmodels.stats.proportion    import proportion_confint                #Fon confidence interval-->proportion_conf(number of successes, number of trials, alpha value represented by 1 minus our confidence level)
from statsmodels.stats.proportion    import proportion_effectsize             #To determinate sample size. Standardize the effect size
from statsmodels.stats.proportion    import proportions_ztest                 #To run the Z-score test, when you know the population standard deviation


print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 123
np.random.seed(SEED)
    
print("****************************************************")
topic = "2. Confidence interval by hand"; print("** %s\n" % topic)

#z_score     = 2.7764451051977987
#sample_mean = 3.0
data         = [1, 2, 3, 4, 5]
confidence   = 0.95

sample_mean  = np.mean(data)
alpha        = 1-confidence
z_score      = t.ppf(1-alpha/2, df=4) # Two-sided; df=degrees of fredom


print("EXAMPLE: ", data, "with mean=", sample_mean)

print("\nMANUALLY COMPUTED: ")
# Compute the standard error and margin of error
std_err      = sem(data)
margin_error = std_err * z_score
lower        = sample_mean - margin_error # Compute and print the lower threshold
upper        = sample_mean + margin_error # Compute and print the upper threshold
print("Lower threshold in this example:", lower)
print("Upper threshold in this example: ", upper)
    
print("\nAUTOMATIC COMPUTED: ")
threshold    = t.interval(confidence, len(data)-1, loc=sample_mean, scale=std_err)
print("Threshold in this example:", threshold)

print("****************************************************")
topic = "3. Applying confidence intervals"; print("** %s\n" % topic)

#heads = binom.rvs(1, 0.5, size=50).sum() #How many heads i get in 50 coin flips (one simple flip each)
heads = binom.rvs(50, 0.5, size=1) #How many heads i get in 50 coin flips (one simple flip each)

# Compute and print the 99% confidence interval -> alpha = 1-confidence
confidence_99 = proportion_confint(heads, 50, 0.01) #proportion_conf(number of successes, number of trials, alpha value represented by 1 minus our confidence level)
# Compute and print the 90% confidence interval
confidence_90 = proportion_confint(heads, 50, 0.1)

print("Example: NUMBER OF HEADS IN 50 FAIR COIN FLIPS --> Got ", heads, "Heads.")
print("99% confidence interval for 50 trials: ", confidence_99)
print("90% confidence interval for 50 trials: ", confidence_90)


# Repeat this process 10 times 
print("\nExample: REPEAT THE SAME PROCESS 10 TIMES")
heads = binom.rvs(50, 0.5, size=10)
for val in heads:
    confidence_interval = proportion_confint(val, 50, .10)
    print("90% confidence interval for 50 trials (got {} heads): ".format(val), confidence_interval)


# Repeat this process 10 times 
print("\nExample: REPEAT THE SAME PROCESS 10 TIMES")
heads = binom.rvs(50, 0.5, size=10)
for val in heads:
    confidence_interval = proportion_confint(val, 50, .01)
    print("99% confidence interval for 50 trials (got {} heads): ".format(val), confidence_interval)


print("****************************************************")
topic = "5. One tailed z-test (Data from the course)"; print("** %s\n" % topic)

file = "ab_data_sample.data"
sample = pd.read_fwf(file, index_col="id")

# Assign and print the conversion rate for each group
conv_rates = sample.groupby('group').mean()
print("Conversion rate for each group: \n{}".format(conv_rates))

# Assign the number of control conversions and trials
num_control = sample[sample.group == 'control']['converted'].sum()
total_control = len(sample[sample.group == 'control'])

# Assign the number of conversions and total trials
num_treat = sample[sample.group == 'treatment']['converted'].sum()
total_treat = len(sample[sample.group == 'treatment'])

count = np.array([num_treat, num_control]) 
nobs = np.array([total_treat, total_control])

##################################################################
##EXPLANATION OF PARAMENTER "alternative" IN "proportions_ztest" #
##"alternative" can be [‘two-sided’, ‘smaller’, ‘larger’]        #
##The alternative hypothesis can be either two-sided or one of   #
##the one- sided tests, smaller means that the alternative       #
##hypothesis is prop < value and larger means prop > value.      #
##                                                               #
##In the two sample test, smaller means that the alternative     #
##hypothesis is p1 < p2 and larger means p1 > p2 where p1 is the #
#proportion of the first sample and p2 of the second one.        #
##################################################################

# Run the z-test and print the result 
# alternative="larger" --> Conversion of treatment > control

#H0 = The treatment not effecting the outcome in any way.
#H1 = The treatment does have a conclusive effect on the outcome.

stat, pval = proportions_ztest(count, nobs, alternative="larger")
print('\nZ-score: {0:0.3f}'.format(pval))

if pval > 0.05:
	print('The treatment does not affect the outcome in any way (pval > 0.05).')
else:
	print('The treatment does have a conclusive effect on the outcome (pval <= 0.05).')


print("****************************************************")
topic = "5. One tailed z-test (Data from the source)"; print("** %s\n" % topic)
#Source: https://www.kaggle.com/zhangluyuan/a-b-testing#Table-of-Contents
##############################################################
##Preparing the data
##############################################################
##5.1 Read the data and store it.
file = "ab_data.csv" 
ab_data = pd.read_csv(file, parse_dates=["timestamp"])
print("(1). Reading the data...\n{}\n".format(ab_data.head()))

##############################################################
##5.2 Find the number of rows in the dataset.
print("(2). Finding the shape (rows, columns) of the dataset: {}.\n".format(ab_data.shape))

##############################################################
##5.3 The number of unique users in the dataset.
print("(3). Unique users in the dataset: {:,.0f} users.\n".format(ab_data.user_id.nunique()))

##############################################################
##5.4 The proportion of users converted.
print("(4). The proportion of users converted: {:.0%}.\n".format((ab_data.converted==1).mean()))

##############################################################
##5.5 The number of times the new_page and treatment don't line up.
wrong_rows = ((ab_data.group=='treatment') & (ab_data.landing_page=='old_page')).sum()+ ((ab_data.group=='control') & (ab_data.landing_page=='new_page')).sum()
print("(5). The number of times that landing_page and group don't line up: {:,.0f} rows.\n".format(wrong_rows))

##############################################################
##5.6 Find the missing values in the dataset.
print("(6). Finding the missing values in the dataset:")
print(ab_data.info(),"\n")

##############################################################
##5.7 Create a new dataset with misaligned rows dropped.
print("(7). Dropping misaligned rows...")
ab_data['misaligned'] = ((ab_data.group=='treatment') & (ab_data.landing_page=='old_page')) | ((ab_data.group=='control') & (ab_data.landing_page=='new_page'))
sample = ab_data.query('misaligned==False')
print("Shape after dropping: {} --> {:,.0f} deleted rows.".format(sample.shape, ab_data.misaligned.sum()))
wrong_rows = ((sample.group=='treatment') & (sample.landing_page=='old_page')).sum() + ((sample.group=='control') & (sample.landing_page=='new_page')).sum()
print("The number of times that landing_page and group don't line up: {:,.0f} rows.".format(wrong_rows))
print("Unique users in the dataset: {:,.0f} users.\n".format(sample.user_id.nunique()))

##############################################################
##5.8 Finding duplicated users.
duplicated_user = sample.user_id.value_counts().sort_values(ascending=False) #Finding howmany times an user appears
duplicated_user = duplicated_user[duplicated_user>1] #Making the filter
duplicated_user = sample[sample.user_id.isin(duplicated_user.index.values)] #Retrieving only duplicated
print("(8). Finding duplicated users...")
print(duplicated_user)
#duplicated_user = duplicated_user.reset_index().groupby("user_id")['index'].last()
sample = sample.drop_duplicates(subset=["user_id"], keep='first')
print("\nShape after dropping: {} --> {:,.0f} deleted rows.".format(sample.shape, len(duplicated_user)/2))
print("Unique users in the dataset: {:,.0f} users.\n".format(ab_data.user_id.nunique()))

##############################################################
##5.9 What is the probability of an individual converting regardless of the page they receive?
print("(9). What is the probability of an individual converting regardless of the page they receive?")
print("{:,.4%}\n".format((sample.converted==1).mean()))

##############################################################
##5.10 Given that an individual was in the control group, what is the probability they converted?
print("(10). Given that an individual was in the control group, what is the probability they converted?")
print("{:,.4%}\n".format((sample.query("group == 'control'")["converted"]==1).mean()))

##############################################################
##5.11 Given that an individual was in the treatment group, what is the probability they converted?
print("(11). Given that an individual was in the treatment group, what is the probability they converted?")
print("{:,.4%}\n".format((sample.query("group == 'treatment'")["converted"]==1).mean()))

##############################################################
##5.12 A/B Test
sample = sample[["group", "converted"]]

# Assign and print the conversion rate for each group
conv_rates = sample.groupby('group').mean()
print("Conversion rate for each group: \n{}".format(conv_rates))

# Assign the number of control conversions and trials
num_control = sample[sample.group == 'control']['converted'].sum()
total_control = len(sample[sample.group == 'control'])

# Assign the number of conversions and total trials
num_treat = sample[sample.group == 'treatment']['converted'].sum()
total_treat = len(sample[sample.group == 'treatment'])

count = np.array([num_treat, num_control]) 
nobs = np.array([total_treat, total_control])

#H0 = The treatment not effecting the outcome in any way.
#H1 = The treatment does have a conclusive effect on the outcome.
stat, pval = proportions_ztest(count, nobs, alternative="larger")
print('\nZ-score: {0:0.3f}'.format(pval))

if pval > 0.05:
	print('The treatment does not affect the outcome in any way (pval > 0.05).')
else:
	print('The treatment does have a conclusive effect on the outcome (pval <= 0.05).')

print("****************************************************")
topic = "6. Two tailed t-test"; print("** %s\n" % topic)

file = "laptops-prices2.data"
laptops2 = pd.read_fwf(file, index_col="Id").sort_index()
#laptops3 = laptops2.drop(laptops2[laptops2.Company.isin(['Acer'])].index, axis=0)
laptops3 = laptops2.drop(laptops2.query("Company in ['Acer']").index, axis=0)

pd.options.display.float_format = '{:,.2f}'.format

# Display the mean price for each group
prices = laptops3.groupby('Company').mean()
print("The mean price for each group: \n{}".format(prices))

# Assign the prices of each group
asus = laptops3[laptops3['Company'] == 'Asus']['Price']
toshiba = laptops3[laptops3['Company'] == 'Toshiba']['Price']

# Run the t-test
tstat, pval = ttest_ind(asus, toshiba)
print('\nt-Test: {0:0.3f}'.format(pval))

if pval > 0.05:
    print('Probably the same distribution (p > 0.05).')
    print("There's not enough evidence here to conclude that there are differences in prices between Toshiba and Asus laptops.")
else:
    print('Probably different distributions (p <= 0.05).')
    print('Toshiba laptops are significantly more expensive than Asus.')

print("****************************************************")
topic = "9. Calculating sample size"; print("** %s\n" % topic)

std_effect = proportion_effectsize(.20, .25)
print("std_effect=",std_effect)

sample_size = zt_ind_solve_power(effect_size=std_effect, nobs1=None, alpha=.05, power=.95)
print("Sample_size=",sample_size)

print("****************************************************")
topic = "10. Visualizing the relationship"; print("** %s\n" % topic)

sample_sizes = np.array(range(5, 100))
effect_sizes = np.array([0.2, 0.5, 0.8])
alpha_sizes = np.array([.01, .05, .1])
# Create results object for t-test analysis
results = TTestIndPower()

# Plot the power analysis with the nobs on x-axis
results.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
plt.ylabel('Power tto conclude with high probability'); # Labeling the axis.
plt.title("Power of test", color='red') #Not neccesary, use the same title that the function use with a different color
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.show()


# Plot the power analysis with effect on x-axis
plt.figure(figsize=(11,5.7))
ax = plt.subplot(1,1,1)
results.plot_power(dep_var='effect_size', nobs=sample_sizes, effect_size=effect_sizes, ax=ax)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=3, fontsize=7, fancybox=True, title='Number of Observation')
plt.ylabel('Power tto conclude with high probability'); # Labeling the axis.
plt.title("Power of test", color='red') #Not neccesary, use the same title that the function use with a different color
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=0.1, bottom=None, right=0.75, top=None, wspace=None, hspace=None)
plt.show()

# Plot the power analysis with the nobs on x-axis for differents confidence levels (alpha)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rcParams["axes.labelsize"] = 8
plt.figure(figsize=(10,4))
for i, alpha in enumerate(alpha_sizes, start=1):
    ax = plt.subplot(1,3,i); 
    alpha=alpha_sizes[i-1]
    results.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes, alpha=alpha, ax=ax)
    plt.legend(loc='best', fontsize=8)
    plt.ylabel('Power tto conclude with high probability'); # Labeling the axis.
    plt.title("Power of test with alpha={:,.2f}".format(alpha), color='red', fontsize=9) #Not neccesary, use the same title that the function use with a different color
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.show()
plt.style.use('default')

# Plot the power analysis with effect on x-axis for differents confidence levels (alpha)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rcParams["axes.labelsize"] = 8
fig = plt.figure(figsize=(11,5.7))
for i, alpha in enumerate(alpha_sizes, start=1):
    ax = plt.subplot(1,3,i); 
    alpha=alpha_sizes[i-1]
    results.plot_power(dep_var='effect_size', nobs=sample_sizes, effect_size=effect_sizes, alpha=alpha, ax=ax)
    ax.legend().set_visible(False)
    plt.ylabel('Power tto conclude with high probability'); # Labeling the axis.
    plt.title("Power of test with alpha={:,.2f}".format(alpha), color='red', fontsize=9) #Not neccesary, use the same title that the function use with a different color
plt.suptitle(topic, color='navy');  # Setting the titles.
handles, labels = ax.get_legend_handles_labels()
#labels = [x[0:4] for x in labels]
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.75, 0.95), ncol=3, fontsize=7, title='Number of Observation')
plt.subplots_adjust(left=0.05, bottom=None, right=0.75, top=None, wspace=None, hspace=None)
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.show()
plt.style.use('default')


print("****************************************************")
topic = "12. Calculating error rates"; print("** %s\n" % topic)

# Print error rate for 60 tests with 5% significance
error_rate = 1 - (1-0.05)**60
print("Error rate for 60 tests with 5% significance:", error_rate)

# Print error rate for 30 tests with 5% significance
error_rate = 1 - (.95**(30))
print("Error rate for 30 tests with 5% significance:", error_rate)

# Print error rate for 10 tests with 5% significance
error_rate = 1 - (.95**(10))
print("Error rate for 1 0 tests with 5% significance:", error_rate)


print("****************************************************")
topic = "13. Bonferroni correction"; print("** %s\n" % topic)

pvals = [.01, .05, .10, .50, .99]
print("p-values:", pvals)
# Create a list of the adjusted p-values
p_adjusted = multipletests(pvals, alpha=.05, method='bonferroni')

# Print the resulting conclusions
print("Resulting conclusions:", p_adjusted[0])

# Print the adjusted p-values themselves 
print("p-values adjusted with Bonferroni Correction:",p_adjusted[1])


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
#from scipy.stats                     import sem                               #For statistic thinking 
#from scipy.stats                     import shapiro                           #For Shapiro-Wilk Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import spearmanr                         #For Spearman's Rank Correlation Test.  To check if two samples are related.
#from scipy.stats                     import t                                 #For statistic thinking 
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
#from sklearn.metrics                 import mean_absolute_error as MAE        #For learning machine
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
#from sklearn.preprocessing           import MinMaxScaler                      #Used for normalize data in a dataframe
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
#from statsmodels.sandbox.stats.multicomp import multipletests                 #To adjust the p-value when you run multiple tests.
#from statsmodels.stats.power         import TTestIndPower                     #Explain how the effect, power and significance level affect the sample size. Create results object for t-test analysis
#from statsmodels.stats.power         import zt_ind_solve_power                #To determinate sample size. Assign and print the needed sample size
#from statsmodels.stats.proportion    import proportion_confint                #Fon confidence interval-->proportion_confint(number of successes, number of trials, alpha value represented by 1 minus our confidence level)
#from statsmodels.stats.proportion    import proportion_effectsize             #To determinate sample size. Standardize the effect size
#from statsmodels.stats.proportion    import proportions_ztest                 #To run the Z-score test, when you know the population standard deviation
#from statsmodels.tsa.arima_model     import ARIMA                             #Similar to use ARMA but on original data (before differencing)
#from statsmodels.tsa.arima_model     import ARMA                              #To estimate parameters from data simulated (AR model)
#from statsmodels.tsa.arima_process   import ArmaProcess                       #For simulating data and for plotting the PACF, For Simulate Autoregressive (AR) Time Series 
#from statsmodels.tsa.stattools       import acf                               #For autocorrelation function
#from statsmodels.tsa.stattools       import adfuller                          #For Augmented Dickey-Fuller unit root test. Test for Random Walk. Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive. Tests that you can use to check if a time series is stationary or not.
#from statsmodels.tsa.stattools       import coint                             #Test for cointegration
#from statsmodels.tsa.stattools       import kpss                              #For Kwiatkowski-Phillips-Schmidt-Shin test. Tests whether a time series is trend stationary or not.

#import tensorflow              as tf                                          #For DeapLearning


###When to use z-score or t-tests --> Usually you use a t-test when you do not know the population standard 
###                                   deviation σ, and you use the standard error instead. You usually use the 
###                                   z-test when you do know the population standard deviation. Although it is 
###                                   true that the central limit theorem kicks in at around n=30. I think that 
###                                   formally, the convergence in distribution of a sequence of t′s to a normal 
###                                   is pretty good when n>30.
#############################################
##Finding z-crtical values 95% one side    ##
##100*(1-alpha)% Confidence Level          ##
#############################################
#import scipy.stats as sp
#alpha=0.05                                                                    
#sp.norm.ppf(1-alpha, loc=0, scale=1)                                          # One-sided; ppf=percent point function
##Out --> 1.6448536269514722
#############################################
##Finding z-crtical values 95% two sides   ##
## 100*(1-alpha)% Confidence Level         ##
#############################################
#import scipy.stats as sp
#alpha=0.05
#sp.norm.ppf(1-alpha/2, loc=0, scale=1)                                        # Two-sided
##Out --> 1.959963984540054
#############################################
##Finding t-crtical values 95% one side    ##
# 100*(1-alpha)% Confidence Level          ##
#############################################
#import scipy.stats as sp
#alpha=0.05
#sp.t.ppf(1-alpha, df=4)                                                       # One-sided; df=degrees of fredom
##Out --> 2.13184678133629
#############################################
##Finding t-crtical values 95% two sides   ##
# 100*(1-alpha)% Confidence Level          ##
#############################################
#import scipy.stats as sp
#alpha=0.05
#sp.t.ppf(1-alpha/2, df=4) # Two-sided; df=degrees of fredom
##Out --> 2.7764451051977987
##Source --> http://lecture.riazulislam.com/uploads/3/9/8/5/3985970/python_practice_3_2019_1.pdf



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

#handles, labels = ax.get_legend_handles_labels()                              #To make a single legend for many subplots with matplotlib
#fig.legend(handles, labels)                                                   #To make a single legend for many subplots with matplotlib

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
