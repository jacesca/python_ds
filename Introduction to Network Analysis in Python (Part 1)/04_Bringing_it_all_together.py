# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:02:17 2019

@author: jacqueline.cortez

Chapter 4. Bringing it all together
Introduction:
    In this final chapter of the course, you'll consolidate everything you've learned through an in-depth case study of GitHub 
    collaborator network data. This is a great example of real-world social network data, and your newly acquired skills will be 
    fully tested. By the end of this chapter, you'll have developed your very own recommendation system to connect GitHub users who 
    should collaborate together.
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
from itertools import combinations                                                  #For iterations
from collections import defaultdict                                                 #Returns a new dictionary-like object

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
from nxviz import ArcPlot                                                           #For Network Analysis in Python
from nxviz import CircosPlot                                                        #For Network Analysis in Python 
from nxviz import MatrixPlot                                                        #For Network Analysis in Python 

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

SEED=42

print("****************************************************")
print("** Getting the data for this program\n")

file = 'github_users.p'
G = nx.read_gpickle(file)
G_components = nx.connected_component_subgraphs(G)
G_comp_dict = {idx: comp.nodes() for idx, comp in enumerate(G_components)}
G_attr = {n: comp_id for comp_id, nodes in G_comp_dict.items() for n in nodes}
nx.set_node_attributes(G, name="grouping", values=G_attr)

print("****************************************************")
tema = "1. Case study!"; print("** %s\n" % tema)

H = nx.erdos_renyi_graph(n=20, p=0.2, seed=SEED)

print("Nodes of G: ", H.nodes())
print("Edges of G: ", H.edges())
print("Type of G: ", type(H))

print("Degree centrality: \n", nx.degree_centrality(H))
print("Betweeness centrality: \n", nx.betweenness_centrality(H))

#plt.figure()
nx.draw(H, with_labels=True, node_color=range(len(H.nodes())), cmap=plt.cm.Blues)
plt.suptitle("{} - Graph G_eight".format(tema))
plt.show()

print("****************************************************")
tema = "2. Characterizing the network (I)"; print("** %s\n" % tema)

print("Number of nodes in G: ", len(G.nodes()))
print("Number of edges in G: ", len(G.edges()))

print("****************************************************")
tema = "3. Characterizing the network (II)"; print("** %s\n" % tema)

# Plot the degree distribution of the GitHub collaboration network
sns.set() # Set default Seaborn style
plt.figure()
plt.hist(list(nx.degree_centrality(G).values()))
plt.xlabel('degree_centrality')
#plt.ylabel('')
plt.title('Degree Centrality Histogram of G')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
tema = "4. Characterizing the network (III)"; print("** %s\n" % tema)

G_sub = nx.Graph()
G_sub.add_nodes_from(['u41', 'u69', 'u96', 'u156', 'u297', 'u298', 'u315', 'u322', 'u435', 'u440', 'u640', 'u655', 'u698', 'u821', 'u863', 'u901', 'u914', 'u1254', 'u1407', 'u1468', 'u1908', 'u2022', 'u2066', 'u2137', 'u2289', 'u2482', 'u2552', 'u2643', 'u2737', 'u2906', 'u3083', 'u3174', 'u3231', 'u3243', 'u3271', 'u3658', 'u3974', 'u3979', 'u4159', 'u4199', 'u4329', 'u4412', 'u4513', 'u4710', 'u4761', 'u4953', 'u5082', 'u5337', 'u5693', 'u5993', 'u6081', 'u7418', 'u7623', 'u7963', 'u8135', 'u9866', 'u9869', 'u9997', 'u10090', 'u10340', 'u10500', 'u10603', 'u14964'])
G_sub.add_edges_from([('u41', 'u2022'), ('u41', 'u69'), ('u41', 'u5082'), ('u41', 'u298'), ('u41', 'u901'), ('u69', 'u315'), ('u69', 'u4513'), ('u69', 'u5082'), ('u69', 'u901'), ('u69', 'u298'), ('u69', 'u2022'), ('u96', 'u315'), ('u96', 'u2482'), ('u96', 'u10500'), ('u96', 'u2022'), ('u96', 'u863'), ('u96', 'u9997'), ('u96', 'u297'), ('u96', 'u698'), ('u96', 'u2066'), ('u96', 'u7963'), ('u96', 'u156'), ('u96', 'u2906'), ('u96', 'u2552'), ('u156', 'u315'), ('u156', 'u2482'), ('u156', 'u10500'), ('u156', 'u863'), ('u156', 'u2022'), ('u156', 'u297'), ('u156', 'u9997'), ('u156', 'u698'), ('u156', 'u2066'), ('u156', 'u7963'), ('u156', 'u2906'), ('u156', 'u2552'), ('u297', 'u315'), ('u297', 'u2482'), ('u297', 'u863'), ('u297', 'u2022'), ('u297', 'u9997'), ('u297', 'u698'), ('u297', 'u10500'), ('u297', 'u2066'), ('u297', 'u7963'), ('u297', 'u2906'), ('u297', 'u2552'), ('u298', 'u5082'), ('u298', 'u901'), ('u298', 'u2022'), ('u315', 'u2482'), ('u315', 'u10500'), ('u315', 'u2022'), ('u315', 'u863'), ('u315', 'u9997'), ('u315', 'u698'), ('u315', 'u2066'), ('u315', 'u7963'), ('u315', 'u2906'), ('u315', 'u4513'), ('u315', 'u2552'), ('u322', 'u3174'), ('u322', 'u3974'), ('u322', 'u7623'), ('u322', 'u8135'), ('u322', 'u2022'), ('u322', 'u4953'), ('u322', 'u640'), ('u322', 'u4412'), ('u322', 'u10340'), ('u322', 'u4159'), ('u322', 'u1908'), ('u322', 'u3979'), ('u322', 'u435'), ('u322', 'u821'), ('u322', 'u9869'), ('u322', 'u9866'), ('u322', 'u10603'), ('u322', 'u3083'), ('u322', 'u4710'), ('u435', 'u7623'), ('u435', 'u8135'), ('u435', 'u2022'), ('u435', 'u10603'), ('u435', 'u10340'), ('u435', 'u9866'), ('u440', 'u5693'), ('u440', 'u2643'), ('u440', 'u10090'), ('u440', 'u3271'), ('u640', 'u1908'), ('u640', 'u3174'), ('u640', 'u3979'), ('u640', 'u3974'), ('u640', 'u821'), ('u640', 'u4953'), ('u640', 'u4412'), ('u640', 'u9869'), ('u640', 'u4159'), ('u640', 'u4710'), ('u640', 'u3083'), ('u655', 'u2643'), ('u655', 'u2906'), ('u655', 'u2137'), ('u655', 'u914'), ('u655', 'u4513'), ('u698', 'u10500'), ('u698', 'u2066'), ('u698', 'u2906'), ('u698', 'u2552'), ('u698', 'u9997'), ('u698', 'u7963'), ('u698', 'u2482'), ('u698', 'u863'), ('u698', 'u2022'), ('u821', 'u3174'), ('u821', 'u3974'), ('u821', 'u4953'), ('u821', 'u4412'), ('u821', 'u4159'), ('u821', 'u1908'), ('u821', 'u3979'), ('u821', 'u9869'), ('u821', 'u4710'), ('u821', 'u3083'), ('u863', 'u10500'), ('u863', 'u2022'), ('u863', 'u7963'), ('u863', 'u2906'), ('u863', 'u2482'), ('u863', 'u9997'), ('u863', 'u2552'), ('u863', 'u2066'), ('u901', 'u2022'), ('u901', 'u5082'), ('u914', 'u2022'), ('u914', 'u3231'), ('u1254', 'u2737'), ('u1254', 'u2289'), ('u1254', 'u2643'), ('u1254', 'u4329'), ('u1254', 'u4761'), ('u1407', 'u2643'), ('u1407', 'u6081'), ('u1407', 'u3658'), ('u1468', 'u5993'), ('u1468', 'u2643'), ('u1468', 'u2022'), ('u1468', 'u7418'), ('u1468', 'u5337'), ('u1468', 'u9869'), ('u1908', 'u3174'), ('u1908', 'u3979'), ('u1908', 'u3974'), ('u1908', 'u4953'), ('u1908', 'u4412'), ('u1908', 'u9869'), ('u1908', 'u4159'), ('u1908', 'u4710'), ('u1908', 'u3083'), ('u2022', 'u2482'), ('u2022', 'u5993'), ('u2022', 'u7623'), ('u2022', 'u8135'), ('u2022', 'u10500'), ('u2022', 'u10340'), ('u2022', 'u9997'), ('u2022', 'u3231'), ('u2022', 'u2643'), ('u2022', 'u2906'), ('u2022', 'u5082'), ('u2022', 'u4199'), ('u2022', 'u9869'), ('u2022', 'u2066'), ('u2022', 'u9866'), ('u2022', 'u7418'), ('u2022', 'u7963'), ('u2022', 'u5337'), ('u2022', 'u10603'), ('u2022', 'u2552'), ('u2066', 'u10500'), ('u2066', 'u7963'), ('u2066', 'u2906'), ('u2066', 'u2482'), ('u2066', 'u9997'), ('u2066', 'u2552'), ('u2137', 'u2643'), ('u2137', 'u4513'), ('u2289', 'u4329'), ('u2289', 'u4761'), ('u2289', 'u2643'), ('u2289', 'u2737'), ('u2482', 'u10500'), ('u2482', 'u7963'), ('u2482', 'u2906'), ('u2482', 'u9997'), ('u2482', 'u2552'), ('u2552', 'u10500'), ('u2552', 'u9997'), ('u2552', 'u2906'), ('u2552', 'u7963'), ('u2643', 'u10090'), ('u2643', 'u5993'), ('u2643', 'u5693'), ('u2643', 'u4329'), ('u2643', 'u4761'), ('u2643', 'u9869'), ('u2643', 'u6081'), ('u2643', 'u2737'), ('u2643', 'u3658'), ('u2643', 'u3243'), ('u2643', 'u7418'), ('u2643', 'u5337'), ('u2643', 'u4513'), ('u2643', 'u3271'), ('u2737', 'u4329'), ('u2737', 'u4761'), ('u2906', 'u10500'), ('u2906', 'u9997'), ('u2906', 'u7963'), ('u3083', 'u3174'), ('u3083', 'u3974'), ('u3083', 'u4953'), ('u3083', 'u4412'), ('u3083', 'u4159'), ('u3083', 'u3979'), ('u3083', 'u9869'), ('u3083', 'u4710'), ('u3174', 'u3974'), ('u3174', 'u4953'), ('u3174', 'u4412'), ('u3174', 'u4159'), ('u3174', 'u3979'), ('u3174', 'u9869'), ('u3174', 'u4710'), ('u3231', 'u4159'), ('u3243', 'u3271'), ('u3271', 'u10090'), ('u3271', 'u5693'), ('u3658', 'u6081'), ('u3974', 'u4953'), ('u3974', 'u4412'), ('u3974', 'u4159'), ('u3974', 'u3979'), ('u3974', 'u9869'), ('u3974', 'u4710'), ('u3979', 'u4953'), ('u3979', 'u4412'), ('u3979', 'u4159'), ('u3979', 'u9869'), ('u3979', 'u4710'), ('u4159', 'u4412'), ('u4159', 'u9869'), ('u4159', 'u4710'), ('u4159', 'u4953'), ('u4329', 'u4761'), ('u4412', 'u4953'), ('u4412', 'u9869'), ('u4412', 'u4710'), ('u4710', 'u4953'), ('u4710', 'u9869'), ('u4953', 'u9869'), ('u5337', 'u5993'), ('u5337', 'u7418'), ('u5337', 'u9869'), ('u5693', 'u10090'), ('u5993', 'u7418'), ('u5993', 'u9869'), ('u7418', 'u9869'), ('u7623', 'u8135'), ('u7623', 'u10603'), ('u7623', 'u10340'), ('u7623', 'u9866'), ('u7963', 'u10500'), ('u7963', 'u9997'), ('u8135', 'u10603'), ('u8135', 'u10340'), ('u8135', 'u9866'), ('u9866', 'u10603'), ('u9866', 'u10340'), ('u9997', 'u10500'), ('u10340', 'u10603')])

# Plot the degree distribution of the GitHub collaboration network
sns.set() # Set default Seaborn style
plt.figure()
plt.hist(list(nx.betweenness_centrality(G_sub).values()))
plt.xlabel('betweenness_centrality')
#plt.ylabel('')
plt.title('Betweeness Centrality Histogram of G')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=None)
plt.show()
plt.style.use('default')

print("****************************************************")
tema = "5. Case study part II: Visualization"; print("** %s\n" % tema)

H = nx.erdos_renyi_graph(n=20, p=0.3, seed=SEED)
H_attr = {idx: int(idx/4) for idx, comp in enumerate(H.nodes())}
nx.set_node_attributes(H, name="key", values=H_attr)

circ = nv.CircosPlot(H, node_color='key', node_group='key')
circ.draw()
plt.suptitle("{} - First Generated Graph".format(tema))
plt.show()


H = nx.erdos_renyi_graph(n=60, p=0.07, seed=SEED)
H_components = nx.connected_component_subgraphs(H)
H_comp_dict = {idx: comp.nodes() for idx, comp in enumerate(H_components)}
H_attr = {n: comp_id for comp_id, nodes in H_comp_dict.items() for n in nodes}
nx.set_node_attributes(H, name="grouping", values=H_attr)

circ = nv.CircosPlot(H, node_color='grouping', node_group='grouping')
circ.draw()
plt.suptitle("{} - Second Generated Graph".format(tema))
plt.show()

print(list(nx.connected_component_subgraphs(H)))
for i, h in enumerate(list(nx.connected_component_subgraphs(H)), 1):
    print("Nodes in subgraph {}: {}".format(i, len(h.nodes())))
    print(h.nodes(data=True))

plt.figure()
nx.draw(H, node_color=range(len(H.nodes())), cmap=plt.cm.Blues)
plt.suptitle("{} - Second Generated Graph".format(tema))
plt.show()

print("****************************************************")
tema = "6. MatrixPlot"; print("** %s\n" % tema)

nodes = ['u41', 'u69', 'u96', 'u156', 'u297', 'u298', 'u315', 'u322', 'u435', 'u440', 'u640', 'u655', 'u698', 'u821', 'u863', 'u901', 'u914', 'u1254', 'u1407', 'u1468', 'u1908', 'u2022', 'u2066', 'u2137', 'u2289', 'u2482', 'u2552', 'u2643', 'u2737', 'u2906', 'u3083', 'u3174', 'u3231', 'u3243', 'u3271', 'u3658', 'u3974', 'u3979', 'u4159', 'u4199', 'u4329', 'u4412', 'u4513', 'u4710', 'u4761', 'u4953', 'u5082', 'u5337', 'u5693', 'u5993', 'u6081', 'u7418', 'u7623', 'u7963', 'u8135', 'u9866', 'u9869', 'u9997', 'u10090', 'u10340', 'u10500', 'u10603', 'u14964']
G_sub = G.subgraph(nodes)

# Calculate the largest connected component subgraph: largest_ccs
largest_ccs = sorted(nx.connected_component_subgraphs(G_sub), key=lambda x: len(x))[-1]

# Create the customized MatrixPlot object: h
#plt.figure()
h = MatrixPlot(graph=largest_ccs, node_grouping='grouping')
h.draw() # Draw the MatrixPlot to the screen
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "7. ArcPlot"; print("** %s\n" % tema)

for n, d in G_sub.nodes(data=True): # Iterate over all the nodes in G, including the metadata
    G_sub.node[n]['degree'] = nx.degree(G_sub, n) # Calculate the degree of each node: G.node[n]['degree']
#G_sub_attr = {n: nx.degree(G_sub, n) for n, d in G_sub.nodes(data=True)}
#nx.set_node_attributes(G_sub, name="degree", values=G_sub_attr)
    
a = ArcPlot(G_sub, node_order='degree') # Create the ArcPlot object: a
a.draw() # Draw the ArcPlot to the screen
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "8. CircosPlot"; print("** %s\n" % tema)

c = CircosPlot(G_sub, node_order='degree', node_grouping='grouping', node_color='grouping') # Create the CircosPlot object: c
c.draw() # Draw the CircosPlot object to the screen
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "10. Finding cliques (I)"; print("** %s\n" % tema)

cliques = nx.find_cliques(G_sub) # Calculate the maximal cliques in G: cliques
print("Number of maximal cliques: ", len(list(cliques))) # Count and print the number of maximal cliques in G

print("****************************************************")
tema = "11. Finding cliques (II)"; print("** %s\n" % tema)

largest_clique = sorted(nx.find_cliques(G_sub), key=lambda x:len(x))[-1] # Find the author(s) that are part of the largest maximal clique: largest_clique
G_lc = G.subgraph(largest_clique) # Create the subgraph of the largest_clique: G_lc

# Draw the CircosPlot to the screen
c = CircosPlot(G_lc) # Create the CircosPlot object: c
c.draw()
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "13. Finding important collaborators"; print("** %s\n" % tema)

deg_cent = nx.degree_centrality(G) # Compute the degree centralities of G: deg_cent
max_dc = max(deg_cent.values()) # Compute the maximum degree centrality: max_dc
prolific_collaborators = [n for n, dc in deg_cent.items() if dc == max_dc] # Find the user(s) that have collaborated the most: prolific_collaborators

print("{} is the most prolific collaborator.".format(prolific_collaborators)) # Print the most prolific collaborator(s)

print("****************************************************")
tema = "14. Characterizing editing communities"; print("** %s\n" % tema)

largest_max_clique = set(sorted(nx.find_cliques(G_sub), key=lambda x: len(x))[-1]) # Identify the largest maximal clique: largest_max_clique
G_lmc = G_sub.subgraph(largest_max_clique).copy() # Create a subgraph from the largest_max_clique: G_lmc

for node in list(G_lmc.nodes()): # Go out 1 degree of separation
    ngb = G_sub.neighbors(node)
    G_lmc.add_nodes_from(ngb)
    G_lmc.add_edges_from(zip([node]*len(list(ngb)), ngb))

for n in G_lmc.nodes(): # Record each node's degree centrality score
    G_lmc.node[n]['degree centrality'] = nx.degree_centrality(G_lmc)[n]

# Create the ArcPlot object: a
a = ArcPlot(G_lmc, node_color='degree centrality', node_order='degree centrality')
a.draw()
plt.suptitle(tema)
plt.show()

print("****************************************************")
tema = "15. Recommending co-editors who have yet to edit together"; print("** %s\n" % tema)

recommended = defaultdict(int) # Initialize the defaultdict: recommended
for n, d in G_sub.nodes(data=True): # Iterate over all the nodes in G
    for n1, n2 in combinations(G_sub.neighbors(n), 2): # Iterate over all possible triangle relationship combinations
        if not G_sub.has_edge(n1, n2): # Check whether n1 and n2 do not have an edge
            recommended[(n1, n2)] += 1 # Increment recommended
            
all_counts = sorted(recommended.values()) # Identify the top 10 pairs of users
top10_pairs = [pair for pair, count in recommended.items() if count > all_counts[-10]]
print("Recomending relations: ", top10_pairs)

print("****************************************************")
print("** END                                            **")
print("****************************************************")