# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:41:59 2019

@author: jacqueline.cortez

Chapter 2. Important nodes
Introduction:
    You'll learn about ways to identify nodes that are important in a network. In doing so, you'll be introduced to more advanced 
    concepts in network analysis as well as the basics of path-finding algorithms. The chapter concludes with a deep dive into the 
    Twitter network dataset which will reinforce the concepts you've learned, such as degree centrality and betweenness centrality.
"""

# Import packages
#import pandas as pd                                                                 #For loading tabular data
#import numpy as np                                                                  #For making operations in lists
#import matplotlib as mpl                                                            #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt                                                     #For creating charts
import seaborn as sns                                                               #For visualizing data
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

def nodes_with_m_nbrs(G, m): # Define nodes_with_m_nbrs()
    """Returns all nodes in graph G that have m neighbors."""
    nodes = set()
    for n in G.nodes(): # Iterate over all nodes in G
        if len(list(G.neighbors(n))) == m: # Check if the number of neighbors of n matches m
            nodes.add(n) # Add the node n to the set
    return nodes # Return the nodes with m neighbors

# Define path_exists()
def path_exists(G, node1, node2):
    """ This function checks whether a path exists between two nodes (node1, node2) in graph G."""
    visited_nodes = set()
    queue = [node1] # Initialize the queue of nodes to visit with the first node: queue
    for node in queue: # Iterate over the nodes in the queue
        neighbors = list(G.neighbors(node)) # Get neighbors of the node
        if node2 in neighbors: # Check to see if the destination node is in the set of neighbors
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
        else:
            visited_nodes.add(node) # Add current node to visited nodes
            queue.extend([n for n in neighbors if n not in visited_nodes]) # Add neighbors of current node that have not yet been visited
        if node == queue[-1]: # Check to see if the final element of the queue has been reached
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))
            return False # Place the appropriate return statement
            
        
def find_nodes_with_highest_deg_cent(G): # Define find_nodes_with_highest_deg_cent()
    """This function returns the node(s) with the highest degree centrality"""
    deg_cent = nx.degree_centrality(G) # Compute the degree centrality of G: deg_cent
    max_dc = max(list(deg_cent.values())) # Compute the maximum degree centrality: max_dc
    nodes = set()    
    for k, v in deg_cent.items(): # Iterate over the degree centrality dictionary
        if v == max_dc: # Check if the current value has the maximum degree centrality
            nodes.add(k) # Add the current node to the set of nodes
    return nodes


def find_node_with_highest_bet_cent(G, k=None): # Define find_node_with_highest_bet_cent()
    """This function returns the node(s) with the highest betweenness centrality."""
    bet_cent = nx.betweenness_centrality(G, k) # Compute betweenness centrality: bet_cent
    max_bc = max(list(bet_cent.values())) # Compute maximum betweenness centrality: max_bc
    nodes = [k for k, v in bet_cent.items() if v == max_bc]
    return nodes


print("****************************************************")
print("** Getting the data for this program\n")

file = 'ego-twitter.p'
T = nx.read_gpickle(file)

print("****************************************************")
tema = "1. Degree centrality"; print("** %s\n" % tema)

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
G.add_edges_from([(1, 2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9)])

#plt.figure()
nx.draw(G)
plt.suptitle(tema)
plt.show()

print("Nodes: ", G.nodes())
print("Edges: ", G.edges())
print("Neighbors of 1: ", list(G.neighbors(1)))
print("Neighbors of 8: ", list(G.neighbors(8)))

print("Centrality degrees (sel-loops are not considered): \n", nx.degree_centrality(G))

print("****************************************************")
tema = "2. Compute number of neighbors for each node"; print("** %s\n" % tema)

six_nbrs = nodes_with_m_nbrs(T, 6) # Compute and print all nodes in T that have 6 neighbors
print(len(six_nbrs), "nodes with 6 neighbors: ", six_nbrs)

print("****************************************************")
tema = "3. Compute degree distribution"; print("** %s\n" % tema)

degrees = {n: len(list(T.neighbors(n))) for n in T.nodes() if len(list(T.neighbors(n)))>0} # Compute the degree of every node: degrees
print("Nodes with beighbors: \n", [*degrees.keys()])

print("****************************************************")
tema = "4. Degree centrality distribution"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
T_sub.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 42), (1, 43), (1, 44), (1, 45), (1, 46), (1, 47), (1, 48), (1, 49), (5, 19), (5, 28), (5, 36), (7, 28), (8, 19), (8, 28), (11, 19), (11, 28), (13, 19), (14, 28), (15, 19), (15, 28), (16, 18), (16, 35), (16, 36), (16, 48), (17, 19), (17, 28), (18, 24), (18, 35), (18, 36), (19, 20), (19, 21), (19, 24), (19, 30), (19, 31), (19, 35), (19, 36), (19, 37), (19, 48), (20, 28), (21, 28), (24, 28), (24, 36), (24, 37), (24, 39), (24, 43), (25, 28), (27, 28), (28, 29), (28, 30), (28, 31), (28, 35), (28, 36), (28, 37), (28, 44), (28, 48), (28, 49), (29, 43), (33, 39), (35, 36), (35, 37), (35, 39), (35, 43), (36, 37), (36, 39), (36, 43), (37, 43), (38, 39), (39, 40), (39, 41), (39, 45), (41, 45), (43, 47), (43, 48)])

degrees = [len(list(T_sub.neighbors(n))) for n in T_sub.nodes()]
deg_cent = nx.degree_centrality(T_sub) # Compute the degree centrality of the Twitter network: deg_cent

# Plot a histogram of the degree centrality distribution of the graph.
sns.set() # Set default Seaborn style
plt.figure()
plt.hist(list(deg_cent.values()), bins=10)
plt.xlabel('degree_centrality')
#plt.ylabel('')
plt.title('Degree Centrality Histogram')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


# Plot a histogram of the degree distribution of the graph
sns.set() # Set default Seaborn style
plt.figure()
plt.hist(degrees, bins=10)
plt.xlabel('degree distribution')
#plt.ylabel('')
plt.title('Degree Distribution Histogram')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')


# Plot a scatter plot of the centrality distribution and the degree distribution
sns.set() # Set default Seaborn style
plt.figure()
plt.scatter(degrees, list(deg_cent.values()))
plt.xlabel('degree_centrality')
plt.ylabel('degree distribution')
plt.ylim(0)
plt.title('Relation between degree centrality and degree distribution')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
tema = "8. Shortest Path III"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
T_sub.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 42), (1, 43), (1, 44), (1, 45), (1, 46), (1, 47), (1, 48), (1, 49), (5, 19), (5, 28), (5, 36), (8, 19), (8, 28), (11, 19), (11, 28), (13, 19), (14, 28), (15, 19), (15, 28), (16, 18), (16, 35), (16, 36), (16, 48), (17, 19), (17, 28), (18, 24), (18, 35), (18, 36), (19, 20), (19, 21), (19, 24), (19, 30), (19, 31), (19, 35), (19, 36), (19, 37), (19, 48), (20, 28), (21, 28), (24, 28), (24, 36), (24, 37), (24, 39), (24, 43), (25, 28), (27, 28), (28, 29), (28, 30), (28, 31), (28, 35), (28, 36), (28, 37), (28, 44), (28, 48), (28, 49), (29, 43), (33, 39), (35, 36), (35, 37), (35, 39), (35, 43), (36, 37), (36, 39), (36, 43), (37, 43), (38, 39), (39, 40), (39, 41), (39, 45), (41, 45), (43, 47), (43, 48)])

node1=1; node2=19; print(node1, node2, path_exists(T_sub, node1, node2));
node1=1; node2=7; print(node1, node2, path_exists(T_sub, node1, node2));

print("****************************************************")
tema = "9. Betweenness centrality"; print("** %s\n" % tema)

H = nx.barbell_graph(m1=5, m2=1)
print(nx.betweenness_centrality(H))

plt.figure()
nx.draw(H)
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "10. NetworkX betweenness centrality on a social network"; print("** %s\n" % tema)

T_sub = nx.Graph()
T_sub.add_nodes_from([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
T_sub.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 42), (1, 43), (1, 44), (1, 45), (1, 46), (1, 47), (1, 48), (1, 49), (16, 18), (16, 35), (16, 36), (16, 48), (18, 16), (18, 24), (18, 35), (18, 36), (19, 5), (19, 8), (19, 11), (19, 13), (19, 15), (19, 17), (19, 20), (19, 21), (19, 24), (19, 30), (19, 31), (19, 35), (19, 36), (19, 37), (19, 48), (28, 1), (28, 5), (28, 7), (28, 8), (28, 11), (28, 14), (28, 15), (28, 17), (28, 20), (28, 21), (28, 24), (28, 25), (28, 27), (28, 29), (28, 30), (28, 31), (28, 35), (28, 36), (28, 37), (28, 44), (28, 48), (28, 49), (36, 5), (36, 24), (36, 35), (36, 37), (37, 24), (37, 35), (37, 36), (39, 1), (39, 24), (39, 33), (39, 35), (39, 36), (39, 38), (39, 40), (39, 41), (39, 45), (42, 1), (43, 24), (43, 29), (43, 35), (43, 36), (43, 37), (43, 47), (43, 48), (45, 1), (45, 39), (45, 41)])

bet_cen = nx.betweenness_centrality(T_sub) # Compute the betweenness centrality of T: bet_cen
deg_cen = nx.degree_centrality(T_sub) # Compute the degree centrality of T: deg_cen

sns.set() # Set default Seaborn style
plt.figure()
plt.scatter(list(bet_cen.values()), list(deg_cen.values())) # Create a scatter plot of betweenness centrality and degree centrality
plt.xlabel('betweenness centrality')
plt.ylabel('degree centrality')
plt.title('Relation between betweenness centrality and degree centrality')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
tema = "11. Deep dive - Twitter network"; print("** %s\n" % tema)

top_dc = find_nodes_with_highest_deg_cent(T) # Find the node(s) that has the highest degree centrality in T: top_dc
print("The node with the highest degree centrality is: ", top_dc)

for node in top_dc: # Write the assertion statement
    assert nx.degree_centrality(T)[node] == max(nx.degree_centrality(T).values())
    
print("****************************************************")
tema = "12. Deep dive - Twitter network part II"; print("** %s\n" % tema)

top_bc = find_node_with_highest_bet_cent(T_sub) # Use that function to find the node(s) that has the highest betweenness centrality in the network: top_bc
print(top_bc)
for node in top_bc: # Write an assertion statement that checks that the node(s) is/are correctly identified.
    assert nx.betweenness_centrality(T_sub)[node] == max(nx.betweenness_centrality(T_sub).values())
        
print("****************************************************")
print("** END                                            **")
print("****************************************************")