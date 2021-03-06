# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:35:20 2019

@author: jacqueline.cortez

Chapter 1. Introduction to networks
Introduction:
    In this chapter, you'll be introduced to fundamental concepts in network analytics while exploring a real-world Twitter network dataset. 
    You'll also learn about NetworkX, a library that allows you to manipulate, analyze, and model graph data. You'll learn about the different 
    types of graphs and how to rationally visualize them.
"""

# Import packages
#import pandas as pd                                                                 #For loading tabular data
#import numpy as np                                                                  #For making operations in lists
#import matplotlib as mpl                                                            #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt                                                     #For creating charts
#import seaborn as sns                                                               #For visualizing data
#import scipy.stats as stats                                                         #For accesign to a vary of statistics functiosn
#import statsmodels as sm                                                            #For stimations in differents statistical models
#import scykit-learn                                                                 #For performing machine learning  
#import tabula                                                                       #For extracting tables from pdf
#import nltk                                                                         #For working with text data
#import math                                                                         #For accesing to a complex math operations
#import random                                                                       #For generating random numbers
#import calendar                                                                     #For accesing to a vary of calendar operations
#import re                                                                           #For regular expressions
#import timeit                                                                       #For Measure execution time of small code snippets
#import time                                                                         #To measure the elapsed wall-clock time between two points
#import warnings
#import wikipedia

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
from datetime import date                                                           #For obteining today function
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions
#from itertools import cycle                                                         #Used in the function plot_labeled_decision_regions()
#from math import floor                                                              #Used in the function plot_labeled_decision_regions()
#from math import ceil                                                               #Used in the function plot_labeled_decision_regions()

#from scipy.cluster.hierarchy import fcluster                                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import dendrogram                                      #For learning machine - unsurpervised
#from scipy.cluster.hierarchy import linkage                                         #For learning machine - unsurpervised
#from scipy.sparse import csr_matrix                                                 #For learning machine 
#from scipy.stats import pearsonr                                                    #For learning machine 
#from scipy.stats import randint                                                     #For learning machine 

#from sklearn.cluster import KMeans                                                  #For learning machine - unsurpervised
#from sklearn.decomposition import NMF                                               #For learning machine - unsurpervised
#from sklearn.decomposition import PCA                                               #For learning machine - unsurpervised
#from sklearn.decomposition import TruncatedSVD                                      #For learning machine - unsurpervised

#from sklearn.ensemble import AdaBoostClassifier                                     #For learning machine - surpervised
#from sklearn.ensemble import BaggingClassifier                                      #For learning machine - surpervised
#from sklearn.ensemble import GradientBoostingRegressor                              #For learning machine - surpervised
#from sklearn.ensemble import RandomForestClassifier                                 #For learning machine
#from sklearn.ensemble import RandomForestRegressor                                  #For learning machine - unsurpervised
#from sklearn.ensemble import VotingClassifier                                       #For learning machine - unsurpervised
#from sklearn.feature_extraction.text import TfidfVectorizer                         #For learning machine - unsurpervised
#from sklearn.feature_selection import chi2                                          #For learning machine
#from sklearn.feature_selection import SelectKBest                                   #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                         #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                       #For learning machine
#from sklearn import datasets                                                        #For learning machine
#from sklearn.impute import SimpleImputer                                            #For learning machine
#from sklearn.linear_model import ElasticNet                                         #For learning machine
#from sklearn.linear_model import Lasso                                              #For learning machine
#from sklearn.linear_model import LinearRegression                                   #For learning machine
#from sklearn.linear_model import LogisticRegression                                 #For learning machine
#from sklearn.linear_model import Ridge                                              #For learning machine
#from sklearn.manifold import TSNE                                                   #For learning machine - unsurpervised
#from sklearn.metrics import accuracy_score                                          #For learning machine
#from sklearn.metrics import classification_report                                   #For learning machine
#from sklearn.metrics import confusion_matrix                                        #For learning machine
#from sklearn.metrics import mean_squared_error as MSE                               #For learning machine
#from sklearn.metrics import roc_auc_score                                           #For learning machine
#from sklearn.metrics import roc_curve                                               #For learning machine
#from sklearn.model_selection import cross_val_score                                 #For learning machine
#from sklearn.model_selection import GridSearchCV                                    #For learning machine
#from sklearn.model_selection import RandomizedSearchCV                              #For learning machine
#from sklearn.model_selection import train_test_split                                #For learning machine
#from sklearn.multiclass import OneVsRestClassifier                                  #For learning machine
#from sklearn.neighbors import KNeighborsClassifier as KNN                           #For learning machine
#from sklearn.pipeline import FeatureUnion                                           #For learning machine
#from sklearn.pipeline import make_pipeline                                          #For learning machine - unsurpervised
#from sklearn.pipeline import Pipeline                                               #For learning machine
#from sklearn.preprocessing import FunctionTransformer                               #For learning machine
#from sklearn.preprocessing import Imputer                                           #For learning machine
#from sklearn.preprocessing import MaxAbsScaler                                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing import Normalizer                                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing import normalize                                         #For learning machine - unsurpervised
#from sklearn.preprocessing import scale                                             #For learning machine
#from sklearn.preprocessing import StandardScaler                                    #For learning machine
#from sklearn.svm import SVC                                                         #For learning machine
#from sklearn.tree import DecisionTreeClassifier                                     #For learning machine - supervised
#from sklearn.tree import DecisionTreeRegressor                                      #For learning machine - supervised

#import keras                                                                        #For DeapLearning
#from keras.callbacks import EarlyStopping                                           #For DeapLearning
#from keras.layers import Dense                                                      #For DeapLearning
#from keras.models import Sequential                                                 #For DeapLearning
#from keras.models import load_model                                                 #For DeapLearning
#from keras.optimizers import SGD                                                    #For DeapLearning
#from keras.utils import to_categorical                                              #For DeapLearning

import networkx as nx                                                               #For Network Analysis in Python
import nxviz as nv                                                                  #For Network Analysis in Python
from nxviz import CircosPlot                                                        #For Network Analysis in Python 
from nxviz import ArcPlot                                                           #For Network Analysis in Python

#from bokeh.io import curdoc, output_file, show                                      #For interacting visualizations
#from bokeh.plotting import figure, ColumnDataSource                                 #For interacting visualizations
#from bokeh.layouts import row, widgetbox, column, gridplot                          #For interacting visualizations
#from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper        #For interacting visualizations
#from bokeh.models import Slider, Select, Button, CheckboxGroup, RadioGroup, Toggle  #For interacting visualizations
#from bokeh.models.widgets import Tabs, Panel                                        #For interacting visualizations
#from bokeh.palettes import Spectral6                                                #For interacting visualizations


# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters() #Require to explicitly register matplotlib converters.

#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09

#Setting the numpy options
#np.set_printoptions(precision=3) #precision set the precision of the output:
#np.set_printoptions(suppress=True) #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf) #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8) #Return to default value.

#Setting images params
#plt.rcParams.update({'figure.max_open_warning': 0}) #To solve the max images open

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** User defined functions \n")

def find_selfloop_nodes(G): # Define find_selfloop_nodes()
    """Finds all nodes that have self-loops in the graph G."""
    nodes_in_selfloops = []
    for u, v in G.edges(): # Iterate over all the edges of G
        if u == v: # Check if node u and node v are the same
            nodes_in_selfloops.append(u) # Append node u to nodes_in_selfloops
    return nodes_in_selfloops


print("****************************************************")
print("** Getting the data for this program\n")

file = 'ego-twitter.p'
T = nx.read_gpickle(file)

print("****************************************************")
tema = "1. Introduction to networks"; print("** %s\n" % tema)

G = nx.Graph()
G.add_nodes_from([1, 2, 3])
G.add_edge(1, 2)
G.node[1]['label'] = 'blue'

#plt.figure()
nx.draw(G)
plt.suptitle(tema)
plt.show()

print("Nodes: ", G.nodes())
print("Edges: ", G.edges())
print("Nodes' structure: ", G.nodes(data=True))

print("****************************************************")
tema = "4. Basic drawing of a network using NetworkX"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
T_sub.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 42), (1, 43), (1, 44), (1, 45), (1, 46), (1, 47), (1, 48), (1, 49), (16, 18), (16, 35), (16, 36), (16, 48), (18, 16), (18, 24), (18, 35), (18, 36), (19, 5), (19, 8), (19, 11), (19, 13), (19, 15), (19, 17), (19, 20), (19, 21), (19, 24), (19, 30), (19, 31), (19, 35), (19, 36), (19, 37), (19, 48), (28, 1), (28, 5), (28, 7), (28, 8), (28, 11), (28, 14), (28, 15), (28, 17), (28, 20), (28, 21), (28, 24), (28, 25), (28, 27), (28, 29), (28, 30), (28, 31), (28, 35), (28, 36), (28, 37), (28, 44), (28, 48), (28, 49), (36, 5), (36, 24), (36, 35), (36, 37), (37, 24), (37, 35), (37, 36), (39, 1), (39, 24), (39, 33), (39, 35), (39, 36), (39, 38), (39, 40), (39, 41), (39, 45), (42, 1), (43, 24), (43, 29), (43, 35), (43, 36), (43, 37), (43, 47), (43, 48), (45, 1), (45, 39), (45, 41)])

plt.figure()
nx.draw(T_sub) # Draw the graph to screen
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "5. Queries on a graph"; print("** %s\n" % tema)

noi = [n for n, d in list(T.nodes(data=True)) if d['occupation'] == 'scientist'] # Use a list comprehension to get the nodes of interest: noi
eoi = [(u, v) for u, v, d in list(T.edges(data=True)) if (d['date'] < date(2010, 1, 1))] # Use a list comprehension to get the edges of interest: eoi

print(noi[:5])
print(eoi[:5])

print("****************************************************")
tema = "6. Types of graphs"; print("** %s\n" % tema)

D = nx.DiGraph()
M = nx.MultiGraph()
MD= nx.MultiDiGraph()

print("Type of G: ", type(G))
print("Type of D: ", type(D))
print("Type of M: ", type(M))
print("Type of MD: ", type(MD))

G.edges[1, 2]['weight'] = 2
print(G.edges[1, 2])

print("Nodes: ", G.nodes(data=True))
print("Edges: ", G.edges(data=True))

print("****************************************************")
tema = "8. Specifying a weight on edges"; print("** %s\n" % tema)

T.edges[1,10]['weight'] = 2 # Set the weight of the edge
print(T.edges[1, 10])

for u, v, d in list(T.edges(data=True)): # Iterate over all the edges (with metadata)
    if 293 in [u,v]: # Check if node 293 is involved
        T.edges[u,v]['weight'] = 1.1 # Set the weight to 1.1

e293 = [(u, v, d) for u, v, d in list(T.edges(data=True)) if 293 in [u, v]] 
print(e293)

print("****************************************************")
tema = "9. Checking whether there are self-loops in the graph"; print("** %s\n" % tema)

print("Self nodes in T: ", T.number_of_selfloops())

self_nodes = len(find_selfloop_nodes(T))
print("Result from find_selfloop_nodes()", self_nodes)

assert T.number_of_selfloops() == self_nodes # Check whether number of self loops equals the number of nodes in self loops

print("****************************************************")
tema = "11. Visualizing using Matrix plots"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([17, 19, 28, 36, 42, 94, 104, 393, 538, 613, 897, 927, 931, 969, 983, 1078, 1090, 1134, 1318, 1368, 1373, 1696, 1706, 1804, 1820, 1971, 2025, 2048, 2503, 2562, 2563, 2646, 2658, 2755, 3159, 3177, 3199, 3201, 3230, 3269, 3277, 3289, 3356, 3531, 3550, 3571, 3580, 3683, 3792, 3960, 4211, 4228, 4242, 4732, 4767, 4966, 5039, 5040, 5077, 5082, 5105, 5161, 5165, 5424, 5546, 5651, 5656, 5864, 6073, 6083, 6112, 6115, 6168, 6204, 6260, 6337, 6502, 6504, 6511, 6514, 6567, 6593, 6595, 6624, 6681, 6770, 6880, 7609, 7612, 7701, 7716, 7719, 7746, 8242, 8294, 8367, 8392, 8416, 8598, 8614, 8829, 9553, 9696, 9703, 9977, 9985, 10004, 10100, 10334, 10366, 10373, 10619, 10658, 10730, 10752, 10758, 10790, 10833, 10966, 11281, 11589, 11601, 11824, 11853, 11912, 12018, 12034, 12162, 12166, 12276, 12299, 12638, 12679, 12701, 12925, 12929, 13036, 13168, 13185, 13200, 14037, 14066, 14278, 14313, 14390, 14484, 14485, 14545, 14954, 15188, 15244, 15624, 16080, 16319, 16455, 16472, 16624, 16637, 16823, 16830, 16900, 16984, 16986, 17184, 17822, 17997, 18132, 18185, 18222, 18421, 18478, 18917, 18980, 19106, 19111, 19152, 19426, 19428, 19613, 19696, 19706, 19745, 20045, 20400, 20423, 20429, 21401, 21404, 21478, 21606, 21919, 22158, 22293])
T_sub.add_edges_from([(19, 17), (19, 36), (28, 8829), (28, 17), (28, 5424), (28, 36), (42, 20400), (94, 104), (897, 3683), (927, 931), (927, 969), (969, 20045), (983, 1078), (1134, 1090), (1134, 538), (1134, 897), (1368, 22293), (1373, 2048), (1373, 16319), (1696, 8367), (1696, 8416), (1971, 22158), (1971, 5546), (2025, 10334), (2503, 17822), (2503, 11824), (2562, 2563), (2646, 2658), (2646, 2755), (2658, 2755), (2658, 897), (3159, 3177), (3159, 3199), (3159, 3201), (3177, 3159), (3177, 3201), (3177, 897), (3230, 6337), (3269, 3277), (3289, 613), (3289, 3356), (3531, 3550), (3571, 3580), (3792, 1804), (3960, 7701), (4211, 4228), (4211, 4242), (4732, 4767), (4966, 393), (4966, 897), (5040, 5039), (5077, 613), (5077, 5082), (5077, 5105), (5161, 5165), (5424, 17), (5424, 1318), (5424, 36), (5651, 5656), (5864, 17184), (5864, 6204), (6073, 6083), (6112, 6115), (6260, 19152), (6502, 6504), (6511, 6514), (6593, 6595), (6624, 6681), (6770, 19613), (6770, 2658), (6880, 19696), (7609, 7612), (7609, 13200), (7716, 7719), (7719, 7716), (7746, 1090), (7746, 538), (7746, 5546), (8242, 8294), (8367, 8392), (8367, 8416), (8367, 1696), (8598, 8614), (9696, 9703), (9977, 9985), (10004, 1706), (10004, 10100), (10373, 10366), (10619, 4732), (10619, 10658), (10730, 1318), (10730, 36), (10752, 10758), (10790, 10833), (11281, 5546), (11281, 897), (11589, 11601), (11824, 11853), (11824, 11912), (11824, 2503), (12018, 9553), (12018, 12034), (12162, 7719), (12162, 10966), (12276, 3230), (12276, 12299), (12638, 6112), (12638, 12925), (12638, 16080), (12638, 36), (12679, 12701), (12925, 12929), (13036, 15188), (13168, 13185), (13200, 6168), (13200, 7609), (13200, 36), (14037, 14066), (14278, 14313), (14390, 16900), (14484, 14485), (14545, 12166), (14954, 18132), (15244, 6567), (16455, 16472), (16624, 16637), (16823, 16830), (16984, 16986), (17997, 1706), (18185, 18222), (18421, 1820), (18421, 15624), (18421, 18421), (18478, 9553), (18917, 18980), (19106, 19111), (19426, 19428), (19706, 19745), (20423, 20429), (21401, 21404), (21478, 11281), (21478, 21606), (21919, 3683)])

#plt.figure()
m = nv.MatrixPlot(T_sub) # Create the MatrixPlot object: m
m.draw() # Draw m to the screen
plt.suptitle(tema)
plt.show() # Display the plot

A = nx.to_numpy_matrix(T_sub) # Convert T to a matrix format: A
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph()) # Convert A back to the NetworkX form as a directed graph: T_conv

for n, d in T_conv.nodes(data=True): # Check that the `category` metadata field is lost from each node
    assert 'category' not in d.keys()
    
print("****************************************************")
tema = "12. Visualizing using Circos plots"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([20, 48, 50, 53, 232, 234, 289, 382, 383, 774, 812, 819, 843, 844, 852, 864, 873, 890, 897, 1074, 1243, 1270, 1271, 1294, 1368, 1696, 1875, 2141, 2164, 2241, 2311, 2350, 2491, 2501, 2564, 2598, 2621, 2639, 2658, 2739, 2789, 2871, 2913, 3016, 3086, 3100, 3104, 3200, 3265, 3385, 3414, 3531, 3563, 3826, 4124, 4131, 4210, 4317, 4362, 4393, 4603, 4811, 4966, 5368, 5380, 5384, 5405, 5565, 5568, 5717, 5755, 6021, 6064, 6099, 6140, 6275, 6306, 6355, 6624, 6733, 6735, 6885, 6912, 6968, 7102, 7163, 7354, 7494, 7534, 7609, 7746, 8477, 8825, 8829, 8847, 9568, 9591, 9600, 9632, 9689, 9694, 9798, 9804, 9813, 9896, 9977, 9997, 10117, 10375, 10877, 12025, 12117, 12176, 12679, 12717, 12831, 12833, 12925, 13471, 13561, 13605, 13616, 13626, 13702, 14245, 14416, 14544, 14833, 14864, 15049, 15310, 15341, 15430, 15452, 15518, 15867, 15889, 15978, 16000, 16239, 16396, 16405, 16690, 16692, 17676, 17865, 17891, 18185, 18198, 18320, 18545, 18550, 18917, 19009, 19012, 19316, 19332, 19401, 19787, 19809, 19946, 19976, 20423, 20428, 20521, 20603, 20947, 20978, 20984, 21110, 21123, 21197, 21445, 21449, 22025, 22046, 22113, 22128, 22672, 22701, 22790, 22825])
T_sub.add_edges_from([(50, 53), (232, 234), (232, 289), (382, 383), (812, 819), (812, 843), (812, 1270), (843, 812), (843, 844), (844, 812), (844, 819), (844, 3016), (844, 843), (844, 864), (864, 819), (864, 843), (864, 844), (873, 890), (873, 897), (1271, 1243), (1271, 1294), (1294, 1243), (1294, 1271), (1368, 50), (1368, 1875), (1368, 9813), (1368, 4603), (1696, 20603), (2141, 2164), (2141, 2241), (2311, 2350), (2501, 2491), (2501, 10375), (2598, 2621), (2598, 2639), (2598, 2658), (2598, 2739), (2598, 2789), (2639, 7494), (2639, 7534), (2658, 2789), (2658, 897), (2871, 17676), (3016, 819), (3016, 844), (3016, 852), (3086, 3100), (3086, 3104), (3086, 897), (3086, 1074), (3265, 6355), (3265, 4811), (3265, 9798), (3385, 3414), (3531, 3563), (3826, 3100), (3826, 897), (4124, 4131), (4124, 1270), (4210, 14416), (4317, 4362), (4317, 4393), (4811, 6355), (4811, 9798), (4811, 9804), (4811, 3265), (4966, 383), (4966, 3104), (4966, 3563), (4966, 897), (5368, 6735), (5368, 15867), (5368, 6064), (5368, 15889), (5380, 5384), (5380, 5405), (5380, 20), (5380, 48), (5565, 5384), (5565, 5568), (5717, 5755), (6021, 3100), (6021, 5368), (6021, 6064), (6021, 897), (6099, 12176), (6275, 6306), (6624, 2241), (6733, 6735), (6885, 6912), (6968, 14245), (7102, 7163), (7354, 48), (7609, 10877), (7609, 12117), (7746, 2913), (7746, 3265), (8477, 3200), (8477, 8477), (8825, 8829), (8825, 20), (8825, 8847), (8825, 48), (8847, 8825), (8847, 20), (8847, 48), (9568, 9591), (9568, 9600), (9689, 9694), (9813, 4603), (9813, 9896), (9813, 1368), (9813, 9632), (9977, 9997), (9997, 9977), (10117, 20521), (10375, 1243), (10375, 2501), (10877, 5384), (10877, 5568), (10877, 6140), (10877, 7609), (12025, 2564), (12025, 15049), (12679, 12717), (12831, 12833), (12925, 289), (13471, 15978), (13471, 16000), (13561, 13605), (13616, 13626), (13616, 6968), (13702, 5384), (13702, 5568), (13702, 6140), (13702, 7609), (14544, 15518), (14833, 9568), (14833, 14864), (15310, 15341), (15430, 15452), (16396, 16405), (16690, 16692), (17865, 9632), (17891, 16239), (18185, 18198), (18185, 18320), (18545, 18550), (18917, 19009), (18917, 19012), (19316, 19332), (19316, 2501), (19316, 19401), (19787, 19809), (19946, 19976), (20423, 20428), (20947, 3100), (20947, 1074), (20978, 20984), (20978, 9568), (21110, 21123), (21197, 2871), (21445, 21449), (22046, 22025), (22113, 774), (22113, 22128), (22672, 22701), (22790, 22825)])

#plt.figure()
c = CircosPlot(T_sub) # Create the CircosPlot object: c
c.draw() # Draw c to the screen
plt.suptitle(tema)
plt.show() # Display the plot

print("****************************************************")
tema = "13. Visualizing using Arc plots"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([(9, {'category': 'D', 'occupation': 'scientist'}), (45, {'category': 'D', 'occupation': 'scientist'}), (70, {'category': 'D', 'occupation': 'scientist'}), (148, {'category': 'I', 'occupation': 'celebrity'}), (401, {'category': 'P', 'occupation': 'celebrity'}), (425, {'category': 'P', 'occupation': 'politician'}), (486, {'category': 'P', 'occupation': 'scientist'}), (537, {'category': 'I', 'occupation': 'politician'}), (538, {'category': 'I', 'occupation': 'celebrity'}), (799, {'category': 'D', 'occupation': 'celebrity'}), (843, {'category': 'D', 'occupation': 'celebrity'}), (873, {'category': 'P', 'occupation': 'scientist'}), (891, {'category': 'D', 'occupation': 'politician'}), (905, {'category': 'D', 'occupation': 'politician'}), (983, {'category': 'P', 'occupation': 'scientist'}), (1022, {'category': 'D', 'occupation': 'celebrity'}), (1043, {'category': 'D', 'occupation': 'scientist'}), (1086, {'category': 'P', 'occupation': 'celebrity'}), (1093, {'category': 'P', 'occupation': 'celebrity'}), (1271, {'category': 'P', 'occupation': 'politician'}), (1417, {'category': 'P', 'occupation': 'scientist'}), (1423, {'category': 'D', 'occupation': 'politician'}), (1444, {'category': 'D', 'occupation': 'politician'}), (1631, {'category': 'P', 'occupation': 'politician'}), (1639, {'category': 'P', 'occupation': 'politician'}), (1683, {'category': 'D', 'occupation': 'politician'}), (1696, {'category': 'P', 'occupation': 'celebrity'}), (1731, {'category': 'D', 'occupation': 'scientist'}), (1732, {'category': 'P', 'occupation': 'scientist'}), (1799, {'category': 'P', 'occupation': 'politician'}), (1804, {'category': 'P', 'occupation': 'politician'}), (1815, {'category': 'P', 'occupation': 'celebrity'}), (1895, {'category': 'I', 'occupation': 'politician'}), (2141, {'category': 'P', 'occupation': 'scientist'}), (2195, {'category': 'P', 'occupation': 'scientist'}), (2479, {'category': 'I', 'occupation': 'celebrity'}), (2514, {'category': 'D', 'occupation': 'scientist'}), (2679, {'category': 'P', 'occupation': 'politician'}), (2716, {'category': 'D', 'occupation': 'politician'}), (2717, {'category': 'D', 'occupation': 'politician'}), (2934, {'category': 'P', 'occupation': 'scientist'}), (2993, {'category': 'I', 'occupation': 'celebrity'}), (3213, {'category': 'D', 'occupation': 'politician'}), (3289, {'category': 'D', 'occupation': 'scientist'}), (3330, {'category': 'I', 'occupation': 'celebrity'}), (3420, {'category': 'P', 'occupation': 'politician'}), (3443, {'category': 'P', 'occupation': 'scientist'}), (3769, {'category': 'D', 'occupation': 'politician'}), (4269, {'category': 'P', 'occupation': 'celebrity'}), (4481, {'category': 'P', 'occupation': 'scientist'}), (4517, {'category': 'D', 'occupation': 'celebrity'}), (4538, {'category': 'I', 'occupation': 'celebrity'}), (5077, {'category': 'D', 'occupation': 'scientist'}), (5084, {'category': 'D', 'occupation': 'celebrity'}), (5276, {'category': 'I', 'occupation': 'scientist'}), (5281, {'category': 'D', 'occupation': 'celebrity'}), (5290, {'category': 'I', 'occupation': 'politician'}), (5398, {'category': 'D', 'occupation': 'celebrity'}), (5529, {'category': 'P', 'occupation': 'politician'}), (5531, {'category': 'D', 'occupation': 'celebrity'}), (6050, {'category': 'D', 'occupation': 'scientist'}), (6182, {'category': 'P', 'occupation': 'politician'}), (6275, {'category': 'P', 'occupation': 'politician'}), (6318, {'category': 'P', 'occupation': 'politician'}), (6325, {'category': 'D', 'occupation': 'celebrity'}), (6430, {'category': 'P', 'occupation': 'celebrity'}), (6442, {'category': 'I', 'occupation': 'celebrity'}), (6615, {'category': 'P', 'occupation': 'politician'}), (6620, {'category': 'D', 'occupation': 'celebrity'}), (6863, {'category': 'I', 'occupation': 'scientist'}), (6885, {'category': 'I', 'occupation': 'politician'}), (6910, {'category': 'I', 'occupation': 'politician'}), (6931, {'category': 'P', 'occupation': 'scientist'}), (6945, {'category': 'D', 'occupation': 'politician'}), (6998, {'category': 'D', 'occupation': 'scientist'}), (7002, {'category': 'I', 'occupation': 'scientist'}), (7008, {'category': 'I', 'occupation': 'scientist'}), (7102, {'category': 'D', 'occupation': 'celebrity'}), (7144, {'category': 'D', 'occupation': 'scientist'}), (7376, {'category': 'I', 'occupation': 'politician'}), (7383, {'category': 'D', 'occupation': 'politician'}), (7638, {'category': 'P', 'occupation': 'scientist'}), (7669, {'category': 'P', 'occupation': 'politician'}), (7723, {'category': 'I', 'occupation': 'celebrity'}), (7980, {'category': 'P', 'occupation': 'politician'}), (8367, {'category': 'I', 'occupation': 'politician'}), (8415, {'category': 'P', 'occupation': 'politician'}), (8477, {'category': 'I', 'occupation': 'scientist'}), (8522, {'category': 'D', 'occupation': 'politician'}), (8532, {'category': 'P', 'occupation': 'scientist'}), (8543, {'category': 'I', 'occupation': 'celebrity'}), (8847, {'category': 'D', 'occupation': 'politician'}), (9083, {'category': 'P', 'occupation': 'scientist'}), (9091, {'category': 'P', 'occupation': 'celebrity'}), (9110, {'category': 'D', 'occupation': 'celebrity'}), (9219, {'category': 'D', 'occupation': 'celebrity'}), (9262, {'category': 'I', 'occupation': 'scientist'}), (9290, {'category': 'P', 'occupation': 'scientist'}), (9305, {'category': 'P', 'occupation': 'scientist'}), (9568, {'category': 'D', 'occupation': 'politician'}), (9813, {'category': 'P', 'occupation': 'scientist'}), (9854, {'category': 'D', 'occupation': 'politician'}), (10966, {'category': 'D', 'occupation': 'politician'}), (10967, {'category': 'I', 'occupation': 'politician'}), (11020, {'category': 'D', 'occupation': 'politician'}), (11039, {'category': 'I', 'occupation': 'celebrity'}), (11525, {'category': 'I', 'occupation': 'politician'}), (11540, {'category': 'I', 'occupation': 'celebrity'}), (11544, {'category': 'D', 'occupation': 'scientist'}), (11728, {'category': 'I', 'occupation': 'celebrity'}), (11734, {'category': 'I', 'occupation': 'politician'}), (11824, {'category': 'P', 'occupation': 'celebrity'}), (11966, {'category': 'D', 'occupation': 'celebrity'}), (12276, {'category': 'P', 'occupation': 'politician'}), (12288, {'category': 'P', 'occupation': 'politician'}), (12340, {'category': 'P', 'occupation': 'celebrity'}), (12354, {'category': 'P', 'occupation': 'scientist'}), (12501, {'category': 'D', 'occupation': 'politician'}), (12530, {'category': 'P', 'occupation': 'politician'}), (12936, {'category': 'I', 'occupation': 'scientist'}), (13036, {'category': 'D', 'occupation': 'politician'}), (13417, {'category': 'D', 'occupation': 'scientist'}), (13616, {'category': 'I', 'occupation': 'politician'}), (13650, {'category': 'I', 'occupation': 'politician'}), (13932, {'category': 'D', 'occupation': 'celebrity'}), (14613, {'category': 'I', 'occupation': 'celebrity'}), (14741, {'category': 'P', 'occupation': 'scientist'}), (14762, {'category': 'D', 'occupation': 'politician'}), (14957, {'category': 'D', 'occupation': 'politician'}), (14984, {'category': 'I', 'occupation': 'celebrity'}), (15169, {'category': 'I', 'occupation': 'scientist'}), (15430, {'category': 'D', 'occupation': 'celebrity'}), (15436, {'category': 'P', 'occupation': 'celebrity'}), (15636, {'category': 'D', 'occupation': 'politician'}), (15761, {'category': 'P', 'occupation': 'politician'}), (15777, {'category': 'D', 'occupation': 'politician'}), (15891, {'category': 'D', 'occupation': 'celebrity'}), (16945, {'category': 'D', 'occupation': 'scientist'}), (16959, {'category': 'D', 'occupation': 'politician'}), (17113, {'category': 'I', 'occupation': 'celebrity'}), (18185, {'category': 'I', 'occupation': 'scientist'}), (18288, {'category': 'P', 'occupation': 'celebrity'}), (18421, {'category': 'I', 'occupation': 'scientist'}), (18478, {'category': 'I', 'occupation': 'scientist'}), (18486, {'category': 'P', 'occupation': 'scientist'}), (18532, {'category': 'I', 'occupation': 'celebrity'}), (18657, {'category': 'I', 'occupation': 'politician'}), (18658, {'category': 'I', 'occupation': 'scientist'}), (18667, {'category': 'D', 'occupation': 'celebrity'}), (18745, {'category': 'D', 'occupation': 'scientist'}), (18809, {'category': 'I', 'occupation': 'scientist'}), (19034, {'category': 'P', 'occupation': 'politician'}), (19071, {'category': 'D', 'occupation': 'celebrity'}), (19091, {'category': 'D', 'occupation': 'celebrity'}), (19104, {'category': 'P', 'occupation': 'politician'}), (19185, {'category': 'I', 'occupation': 'scientist'}), (19186, {'category': 'P', 'occupation': 'scientist'}), (19787, {'category': 'D', 'occupation': 'politician'}), (19791, {'category': 'P', 'occupation': 'scientist'}), (20012, {'category': 'I', 'occupation': 'celebrity'}), (20018, {'category': 'P', 'occupation': 'celebrity'}), (20035, {'category': 'D', 'occupation': 'scientist'}), (20041, {'category': 'D', 'occupation': 'scientist'}), (20089, {'category': 'D', 'occupation': 'scientist'}), (20309, {'category': 'I', 'occupation': 'celebrity'}), (20315, {'category': 'P', 'occupation': 'politician'}), (20485, {'category': 'I', 'occupation': 'politician'}), (20488, {'category': 'P', 'occupation': 'celebrity'}), (20499, {'category': 'I', 'occupation': 'politician'}), (20526, {'category': 'D', 'occupation': 'scientist'}), (20615, {'category': 'I', 'occupation': 'politician'}), (20978, {'category': 'D', 'occupation': 'politician'}), (21148, {'category': 'I', 'occupation': 'politician'}), (21279, {'category': 'P', 'occupation': 'politician'}), (21303, {'category': 'P', 'occupation': 'politician'}), (21325, {'category': 'P', 'occupation': 'scientist'}), (21393, {'category': 'P', 'occupation': 'celebrity'}), (21432, {'category': 'P', 'occupation': 'scientist'}), (21478, {'category': 'P', 'occupation': 'politician'}), (21560, {'category': 'D', 'occupation': 'politician'}), (22184, {'category': 'P', 'occupation': 'scientist'}), (22196, {'category': 'I', 'occupation': 'scientist'}), (22319, {'category': 'D', 'occupation': 'politician'}), (22595, {'category': 'I', 'occupation': 'politician'}), (22790, {'category': 'I', 'occupation': 'scientist'}), (23024, {'category': 'P', 'occupation': 'politician'}), (23026, {'category': 'D', 'occupation': 'politician'}), (23233, {'category': 'D', 'occupation': 'scientist'}), (23239, {'category': 'D', 'occupation': 'politician'})]) 
T_sub.add_edges_from([(45, 21432), (70, 9083), (70, 9091), (401, 425), (401, 486), (537, 1086), (537, 1093), (537, 538), (843, 799), (873, 891), (983, 1022), (983, 1043), (1093, 537), (1271, 10967), (1271, 7723), (1423, 1444), (1639, 1631), (1683, 6931), (1683, 6945), (1696, 1683), (1696, 8367), (1696, 20615), (1731, 1732), (1799, 537), (1799, 1804), (1799, 1093), (1799, 3769), (1895, 425), (2141, 2195), (2679, 11728), (2679, 2195), (2679, 2717), (2717, 2679), (2717, 2716), (2934, 2993), (3289, 3330), (3420, 3443), (4269, 2514), (4481, 4517), (4481, 4538), (4517, 6430), (5077, 5084), (5276, 5281), (5276, 5290), (5529, 5531), (6050, 17113), (6182, 7376), (6275, 537), (6275, 6318), (6275, 6325), (6318, 21148), (6442, 11544), (6615, 537), (6615, 538), (6615, 6620), (6620, 537), (6620, 1086), (6620, 1093), (6620, 538), (6620, 6615), (6863, 14984), (6885, 6910), (6998, 7002), (6998, 7008), (7008, 7980), (7102, 7144), (7383, 905), (7638, 7669), (7723, 15891), (7723, 10967), (8367, 8415), (8367, 1696), (8477, 8522), (8477, 8477), (8532, 8543), (8847, 5398), (9110, 4517), (9110, 148), (9219, 2479), (9262, 9290), (9262, 9305), (9813, 9854), (11020, 11039), (11525, 11540), (11728, 11734), (11824, 70), (11824, 11966), (12276, 12288), (12340, 12354), (12501, 2195), (12501, 2514), (12530, 20089), (12936, 10966), (13036, 15169), (13417, 1022), (13616, 13650), (13932, 537), (14613, 14741), (14613, 14762), (14957, 3213), (15430, 15436), (15636, 9083), (15761, 15777), (16945, 22595), (16959, 9), (18185, 18288), (18421, 1815), (18421, 18421), (18478, 18486), (18478, 18532), (18657, 18658), (18657, 1417), (18657, 18667), (18745, 18809), (19034, 19071), (19091, 19104), (19185, 19186), (19787, 19791), (19787, 3420), (20012, 20018), (20035, 20041), (20309, 20315), (20485, 20488), (20499, 20526), (20978, 9568), (21279, 21303), (21325, 21393), (21478, 21560), (22184, 22196), (22319, 537), (22319, 1086), (22319, 538), (22790, 18658), (23024, 23026), (23233, 23239)])

a = ArcPlot(T_sub) # Create the un-customized ArcPlot object: a
a.draw() # Draw a to the screen
plt.show() # Display the plot

a2 = ArcPlot(T_sub, node_order='category', node_color='category') # Create the customized ArcPlot object: a2
a2.draw() # Draw a2 to the screen
plt.show() # Display the plot

print("****************************************************")
print("** END                                            **")
print("****************************************************")