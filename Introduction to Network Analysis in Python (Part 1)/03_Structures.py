# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:42:30 2019

@author: jacqueline.cortez

Chapter 3. Structures
Introduction:
    This chapter is all about finding interesting structures within network data. You'll learn about essential concepts such as cliques, 
    communities, and subgraphs, which will leverage all of the skills you acquired in Chapter 2. By the end of this chapter, you'll be 
    ready to apply the concepts you've learned to a real-world case study.
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
#from datetime import date                                                           #For obteining today function
#from datetime import datetime                                                       #For obteining today function
#from string import Template                                                         #For working with string, regular expressions
#from itertools import cycle                                                         #Used in the function plot_labeled_decision_regions()
#from math import floor                                                              #Used in the function plot_labeled_decision_regions()
#from math import ceil                                                               #Used in the function plot_labeled_decision_regions()
from itertools import combinations                                                  #For iterations

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
#import nxviz as nv                                                                  #For Network Analysis in Python
#from nxviz import CircosPlot                                                        #For Network Analysis in Python 
#from nxviz import ArcPlot                                                           #For Network Analysis in Python

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

def is_in_triangle(G, n):# Define is_in_triangle()
    """Checks whether a node `n` in graph `G` is in a triangle relationship or not. Returns a boolean."""
    in_triangle = False
    for n1, n2 in combinations(G.neighbors(n), 2): # Iterate over all possible triangle relationship combinations
        if G.has_edge(n1, n2): # Check if an edge exists between n1 and n2
            in_triangle = True
            break
    return in_triangle


def nodes_in_triangle(G, n): # Write a function that identifies all nodes in a triangle relationship with a given node.
    """Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`. """
    triangle_nodes = set([n])
    for n1, n2 in combinations(G.neighbors(n),2): # Iterate over all possible triangle relationship combinations
        if G.has_edge(n1, n2): # Check if n1 and n2 have an edge between them
            triangle_nodes.add(n1) # Add n1 to triangle_nodes
            triangle_nodes.add(n2) # Add n2 to triangle_nodes0
    return triangle_nodes


def node_in_open_triangle(G, n): # Define node_in_open_triangle()
    """Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`."""
    in_open_triangle = False
    for n1, n2 in combinations(G.neighbors(n), 2): # Iterate over all possible triangle relationship combinations
        if not G.has_edge(n1, n2): # Check if n1 and n2 do NOT have an edge between them
            in_open_triangle = True
            break
    return in_open_triangle


def maximal_cliques(G, size): # Define maximal_cliques()
    """Finds all maximal cliques in graph `G` that are of size `size`."""
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs


def get_nodes_and_nbrs(G, nodes_of_interest): # Define get_nodes_and_nbrs()
    """Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors."""
    nodes_to_draw = []
    for n in nodes_of_interest: # Iterate over the nodes of interest
        nodes_to_draw.append(n) # Append the nodes of interest to nodes_to_draw
        for nbr in G.neighbors(n): # Iterate over all the neighbors of node n
            nodes_to_draw.append(nbr) # Append the neighbors of n to nodes_to_draw
    return G.subgraph(nodes_to_draw)


print("****************************************************")
print("** Getting the data for this program\n")

print("****************************************************")
tema = "2. Identifying triangle relationships"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
T_sub.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 42), (1, 43), (1, 44), (1, 45), (1, 46), (1, 47), (1, 48), (1, 49), (5, 19), (5, 28), (5, 36), (7, 28), (8, 19), (8, 28), (11, 19), (11, 28), (13, 19), (14, 28), (15, 19), (15, 28), (16, 18), (16, 35), (16, 36), (16, 48), (17, 19), (17, 28), (18, 24), (18, 35), (18, 36), (19, 20), (19, 21), (19, 24), (19, 30), (19, 31), (19, 35), (19, 36), (19, 37), (19, 48), (20, 28), (21, 28), (24, 28), (24, 36), (24, 37), (24, 39), (24, 43), (25, 28), (27, 28), (28, 29), (28, 30), (28, 31), (28, 35), (28, 36), (28, 37), (28, 44), (28, 48), (28, 49), (29, 43), (33, 39), (35, 36), (35, 37), (35, 39), (35, 43), (36, 37), (36, 39), (36, 43), (37, 43), (38, 39), (39, 40), (39, 41), (39, 45), (41, 45), (43, 47), (43, 48)])

node=3; print("Is Node {} in a triangle? {}".format(node, is_in_triangle(T_sub, node)));
node=5; print("Is Node {} in a triangle? {}".format(node, is_in_triangle(T_sub, node)));

print("****************************************************")
tema = "3. Finding nodes involved in triangles"; print("** %s\n" % tema)

node=1; print("Node {} is in {} triangles.".format(node, len(nodes_in_triangle(T_sub, node))))

print("****************************************************")
tema = "4. Finding open triangles"; print("** %s\n" % tema)

num_open_triangles = 0 # Compute the number of open triangles in T
for n in T_sub.nodes(): # Iterate over all the nodes in T
    if node_in_open_triangle(T_sub, n): # Check if the current node is in an open triangle
        num_open_triangles += 1 # Increment num_open_triangles

print("There is {} open triangles in T_sub graph.".format(num_open_triangles))

print("****************************************************")
tema = "5. Maximal cliques"; print("** %s\n" % tema)

H = nx.barbell_graph(m1=5, m2=1)
print("Maximal cliques found in H: \n", list(nx.find_cliques(H)))

#plt.figure()
nx.draw(H, with_labels=True, node_color=range(len(H.nodes())), cmap=plt.cm.Blues)
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "6. Finding all maximal cliques of size 'n'"; print("** %s\n" % tema)

cliques = list(nx.find_cliques(T_sub))
print("There are {} cliques found in T_sub and they are: \n{} ".format(len(cliques), cliques))
size=3; cliques=maximal_cliques(T_sub, size) 
print("\nThere are {} cliques of size {} in T_sub and they are: \n{}".format(len(cliques), size, cliques))

print("****************************************************")
tema = "7. Subgraphs"; print("** %s\n" % tema)

G = nx.erdos_renyi_graph(n=20, p=0.2)
print("Nodes of G: ", G.nodes())
print("Edges of G: ", G.edges())

plt.figure()
nx.draw(G, with_labels=True, node_color=range(20), cmap=plt.cm.Reds)
plt.suptitle("{} - Graph G".format(tema))
plt.show()

item = 8
nodes = list(G.neighbors(item))
nodes.append(item)
G_eight = G.subgraph(nodes)
print("Nodes of G_eight: ", G_eight.nodes())
print("Edges of G_eight: ", G_eight.edges())

plt.figure()
nx.draw(G_eight, with_labels=True, node_color='red')
plt.suptitle("{} - Graph G_eight".format(tema))
plt.show()

print("****************************************************")
tema = "8. Subgraphs I"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
T_sub.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 42), (1, 43), (1, 44), (1, 45), (1, 46), (1, 47), (1, 48), (1, 49), (5, 19), (5, 28), (5, 36), (7, 28), (8, 19), (8, 28), (11, 19), (11, 28), (13, 19), (14, 28), (15, 19), (15, 28), (16, 18), (16, 35), (16, 36), (16, 48), (17, 19), (17, 28), (18, 24), (18, 35), (18, 36), (19, 20), (19, 21), (19, 24), (19, 30), (19, 31), (19, 35), (19, 36), (19, 37), (19, 48), (20, 28), (21, 28), (24, 28), (24, 36), (24, 37), (24, 39), (24, 43), (25, 28), (27, 28), (28, 29), (28, 30), (28, 31), (28, 35), (28, 36), (28, 37), (28, 44), (28, 48), (28, 49), (29, 43), (33, 39), (35, 36), (35, 37), (35, 39), (35, 43), (36, 37), (36, 39), (36, 43), (37, 43), (38, 39), (39, 40), (39, 41), (39, 45), (41, 45), (43, 47), (43, 48)])

nodes_of_interest = [29, 38, 42]

T_draw = get_nodes_and_nbrs(T_sub, nodes_of_interest) # Extract the subgraph with the nodes of interest: T_draw

plt.figure()
nx.draw(T_draw, with_labels=True, node_color=range(len(T_draw.nodes())), cmap=plt.cm.Greens) # Draw the subgraph to the screen
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "9. Subgraphs II"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([(1, {'category': 'I', 'occupation': 'politician'}), (3, {'category': 'D', 'occupation': 'celebrity'}), (4, {'category': 'I', 'occupation': 'politician'}), (5, {'category': 'I', 'occupation': 'scientist'}), (6, {'category': 'D', 'occupation': 'politician'}), (7, {'category': 'I', 'occupation': 'politician'}), (8, {'category': 'I', 'occupation': 'celebrity'}), (9, {'category': 'D', 'occupation': 'scientist'}), (10, {'category': 'D', 'occupation': 'celebrity'}), (11, {'category': 'I', 'occupation': 'celebrity'}), (12, {'category': 'I', 'occupation': 'celebrity'}), (13, {'category': 'P', 'occupation': 'scientist'}), (14, {'category': 'D', 'occupation': 'celebrity'}), (15, {'category': 'P', 'occupation': 'scientist'}), (16, {'category': 'P', 'occupation': 'politician'}), (17, {'category': 'I', 'occupation': 'scientist'}), (18, {'category': 'I', 'occupation': 'celebrity'}), (19, {'category': 'I', 'occupation': 'scientist'}), (20, {'category': 'P', 'occupation': 'scientist'}), (21, {'category': 'I', 'occupation': 'celebrity'}), (22, {'category': 'D', 'occupation': 'scientist'}), (23, {'category': 'D', 'occupation': 'scientist'}), (24, {'category': 'P', 'occupation': 'politician'}), (25, {'category': 'I', 'occupation': 'celebrity'}), (26, {'category': 'P', 'occupation': 'celebrity'}), (27, {'category': 'D', 'occupation': 'scientist'}), (28, {'category': 'P', 'occupation': 'celebrity'}), (29, {'category': 'I', 'occupation': 'celebrity'}), (30, {'category': 'P', 'occupation': 'scientist'}), (31, {'category': 'D', 'occupation': 'scientist'}), (32, {'category': 'P', 'occupation': 'politician'}), (33, {'category': 'I', 'occupation': 'politician'}), (34, {'category': 'D', 'occupation': 'celebrity'}), (35, {'category': 'P', 'occupation': 'scientist'}), (36, {'category': 'D', 'occupation': 'scientist'}), (37, {'category': 'I', 'occupation': 'scientist'}), (38, {'category': 'P', 'occupation': 'celebrity'}), (39, {'category': 'D', 'occupation': 'celebrity'}), (40, {'category': 'I', 'occupation': 'celebrity'}), (41, {'category': 'I', 'occupation': 'celebrity'}), (42, {'category': 'P', 'occupation': 'scientist'}), (43, {'category': 'I', 'occupation': 'celebrity'}), (44, {'category': 'I', 'occupation': 'politician'}), (45, {'category': 'D', 'occupation': 'scientist'}), (46, {'category': 'I', 'occupation': 'politician'}), (47, {'category': 'I', 'occupation': 'celebrity'}), (48, {'category': 'P', 'occupation': 'celebrity'}), (49, {'category': 'P', 'occupation': 'politician'})]) 
T_sub.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 42), (1, 43), (1, 44), (1, 45), (1, 46), (1, 47), (1, 48), (1, 49), (5, 19), (5, 28), (5, 36), (7, 28), (8, 19), (8, 28), (11, 19), (11, 28), (13, 19), (14, 28), (15, 19), (15, 28), (16, 18), (16, 35), (16, 36), (16, 48), (17, 19), (17, 28), (18, 24), (18, 35), (18, 36), (19, 20), (19, 21), (19, 24), (19, 30), (19, 31), (19, 35), (19, 36), (19, 37), (19, 48), (20, 28), (21, 28), (24, 28), (24, 36), (24, 37), (24, 39), (24, 43), (25, 28), (27, 28), (28, 29), (28, 30), (28, 31), (28, 35), (28, 36), (28, 37), (28, 44), (28, 48), (28, 49), (29, 43), (33, 39), (35, 36), (35, 37), (35, 39), (35, 43), (36, 37), (36, 39), (36, 43), (37, 43), (38, 39), (39, 40), (39, 41), (39, 45), (41, 45), (43, 47), (43, 48)])

nodes = [n for n, d in list(T_sub.nodes(data=True)) if d['occupation'] == 'celebrity'] # Extract the nodes of interest: nodes
nodeset = set(nodes) # Create the set of nodes: nodeset

for n in nodes: # Iterate over nodes
    nbrs = T_sub.neighbors(n) # Compute the neighbors of n: nbrs
    nodeset = nodeset.union(nbrs) # Compute the union of nodeset and nbrs: nodeset
    
T_draw = T_sub.subgraph(nodeset) # Compute the subgraph using nodeset: T_sub

plt.figure()
nx.draw(T_draw, with_labels=True, node_color=range(len(T_draw.nodes())), cmap=plt.cm.Purples) # Draw the subgraph to the screen
plt.suptitle(tema)
plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")