# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:47:46 2020

@author: jacesca@gmail.com
Source:
        https://python-graph-gallery.com/network-chart/
        https://python-graph-gallery.com/320-basic-network-from-pandas-data-frame/
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


print("****************************************************")
topic = "Initial configuration"; print("** %s\n" % topic)

SEED = 13
np.random.seed(SEED) 
plt.rcParams['figure.max_open_warning'] = 60


###############################################################################
print("****************************************************")
topic = "Graph No.320 Basic Network from pandas data frame"; print("** %s\n" % topic)
###############################################################################
## This example is probably the most basic network chart you can realise.
## A network chart is constituted by nodes. These nodes are interconnected by 
## edges. So a basic format is a data frame where each line describes a 
## connection.
## Here we construct a data frame with 4 lines, describing the 4 connections 
## of this plot! So if you have a csv file with your connections, load it and 
## you are ready to visualise it!
## Next step: customise the chart parameters!
###############################################################################

# Build a dataframe with 4 connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# Plot it
nx.draw(G, with_labels=True)

#Show the graph
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No320.png', bbox_inches='tight')


###############################################################################
print("****************************************************")
topic = "Graph No.321 Custom NetworkX graph appearance"; print("** %s\n" % topic)
###############################################################################
## The chart #320 explain how to realise a basic network chart. Now, let’s have 
## a look to the arguments that allows to custom the appearance of the chart. 
## The customisations are separated in 3 main categories: nodes, node labels 
## and edges:
###############################################################################
## ---------------------------------------------------------------------> NODES
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# Graph with Custom nodes:
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, node_color="skyblue", node_shape="s", alpha=0.5, linewidths=40)

#Show the graph
plt.title('Nodes')
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No321_nodes.png', bbox_inches='tight')


## --------------------------------------------------------------------> LABELS
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# Custom the edges:
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, font_size=25, font_color="yellow", font_weight="bold")
plt.title('Labels')

#Show the graph
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No321_labels.png', bbox_inches='tight')


## ---------------------------------------------------------------------> EDGES
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# Chart with Custom edges:
plt.figure()
nx.draw_networkx(G, with_labels=True, width=5, edge_color="skyblue", style="solid")

#Show the graph
plt.title('Edges')
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No321_edges.png', bbox_inches='tight')


## -----------------------------------------------------------------------> ALL
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# All together we can do something fancy
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, node_color="skyblue", 
                 node_shape="o", alpha=0.5, linewidths=4, font_size=25, font_color="grey", 
                 font_weight="bold", width=2, edge_color="grey")

#Show the graph
plt.title('All')
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No321_all.png', bbox_inches='tight')


###############################################################################
print("****************************************************")
topic = "Graph No.322 Network layout possibilities"; print("** %s\n" % topic)
###############################################################################
## Chart #320 and #321 explain how to realise a basic network chart and custom 
## its appearance. The next step is to control the layout of your network. 
## There is actually an algorithm that calculate the most optimal position of 
## each node. Several algorithm have been developed and are proposed by 
## NetworkX. This page illustrate this concept by taking the same small dataset 
## and applying different layout algorithm on it. If you have no idea which one 
## is the best for you, just let it by default! (It will be the fruchterman 
## Reingold solution). Read more about it with help(nx.layout).Chart #320 and 
## #321 explain how to realise a basic network chart and custom its appearance. 
## The next step is to control the layout of your network. There is actually an 
## algorithm that calculate the most optimal position of each node. Several 
## algorithm have been developed and are proposed by NetworkX. This page 
## illustrate this concept by taking the same small dataset and applying 
## different layout algorithm on it. If you have no idea which one is the best 
## for you, just let it by default! (It will be the fruchterman Reingold 
## solution). Read more about it with help(nx.layout).
###############################################################################
## ------------------------------------------------------> Fruchterman Reingold
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A','E','F','E','G','G','D','F'], 
                    'to'  :['D', 'A', 'E','C','A','F','G','D','B','G','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# Fruchterman Reingold
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.fruchterman_reingold_layout(G))

#Show the graph
plt.title("fruchterman_reingold")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No322_Fruchterman_Reingold.png', bbox_inches='tight')



## ------------------------------------------------------------------> Circular
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A','E','F','E','G','G','D','F'], 
                    'to'  :['D', 'A', 'E','C','A','F','G','D','B','G','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# Circular
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.circular_layout(G))

#Show the graph
plt.suptitle(topic, fontsize=12, color='darkred')
plt.title("circular")
plt.show()

# Save as png
plt.savefig('networkx/No322_Circular.png', bbox_inches='tight')



## --------------------------------------------------------------------> Random
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A','E','F','E','G','G','D','F'], 
                    'to'  :['D', 'A', 'E','C','A','F','G','D','B','G','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# Random
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.random_layout(G))

#Show the graph
plt.title("random")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No322_Random.png', bbox_inches='tight')



## ------------------------------------------------------------------> Spectral
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A','E','F','E','G','G','D','F'], 
                    'to'  :['D', 'A', 'E','C','A','F','G','D','B','G','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# Spectral
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.spectral_layout(G))

#Show the graph
plt.title("spectral")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No322_Spectral.png', bbox_inches='tight')



## --------------------------------------------------------------------> Spring
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A','E','F','E','G','G','D','F'], 
                    'to'  :['D', 'A', 'E','C','A','F','G','D','B','G','C']})
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to')
 
# Spring
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.spring_layout(G))

#Show the graph
plt.title("spring")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No322_Spring.png', bbox_inches='tight')



###############################################################################
print("****************************************************")
topic = "Graph No.323 Directed or Undirected network"; print("** %s\n" % topic)
###############################################################################
## Network charts can be split in 2 main categories: directed and undirected 
## networks.
## 
## If it is directed, there is a notion of flow between the 2 nodes, thus 
## leaving a place to go somewhere else. Like money goes from company A to 
## company B. That’s why you can see (kind of) arrows on the left chart, 
## it gives the direction. The flow goes from B to A for example.
## 
## If it is undirected, there is just a link between the 2 nodes, like mister 
## A and mister B are friend.
## 
## When you build your graph, you have to use the function that suits your 
## need: Graph() is used for undirected (default), DiGraph is used for 
## directed graph.
###############################################################################
## ------------------------------------------------------------------> DIRECTED
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['D', 'A', 'B', 'C','A'], 'to':['A', 'D', 'A', 'E','C']})
 
# Build your graph. Note that we use the DiGraph function to create the graph!
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph() )
 
# Make the graph
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)

#Show the graph
plt.title("Direct relationship")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No323_directed.png', bbox_inches='tight')



## ----------------------------------------------------------------> UNDIRECTED
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['D', 'A', 'B', 'C','A'], 'to':['A', 'D', 'A', 'E','C']})
 
# Build your graph. Note that we use the Graph function to create the graph!
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
 
# Make the graph
plt.figure()
nx.draw_networkx(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)

#Show the graph
plt.title("Undirect relationship")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No323_undirected.png', bbox_inches='tight')



###############################################################################
print("****************************************************")
topic = "Graph No.324 Map a color to network nodes"; print("** %s\n" % topic)
###############################################################################
## A common task is to color each node of your network chart following a 
## feature of your node (we call it mapping a color). It allows to display more 
## information in your chart. There are 2 possibilities:
## 
## (1) The feature you want to map is a numerical value. Then we will use a 
##     continuous color scale. On the left graph, A is darker than C that is 
##     darker than B…
## (2) The feature is categorical. On the right graph, A and B belongs to the 
##     same group, D and E are grouped together and C is alone in his group. 
##     We used a categorical color scale.
## 
## Usually we work with 2 tables. The first one provides the links between 
## nodes. The second one provides the features of each node. You can link these 
## 2 files as follows.
###############################################################################
## ----------------------------------------------------> Continuous color scale
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
 
# And a data frame with characteristics for your nodes
carac = pd.DataFrame({ 'ID':['A', 'B', 'C','D','E'], 'myvalue':[123, 25, 76, 12, 34] })
 
# Build your graph
plt.figure()
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph())

# The order of the node for networkX is the following order:
# Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!
print("Nodes to order: ",G.nodes())
 
# Here is the tricky part: I need to reorder carac, to assign the good color to each node
carac = carac.set_index('ID')
carac = carac.reindex(G.nodes())
 
# Plot it, providing a continuous color scale with cmap:
nx.draw_networkx(G, with_labels=True, node_color=carac['myvalue'], cmap=plt.cm.Blues, node_size=1500)

#Show the graph
plt.title("Continuous color scale")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No324_Continuous_color_scale.png', bbox_inches='tight')



## ---------------------------------------------------> Categorical color scale
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
 
# And a data frame with characteristics for your nodes
carac = pd.DataFrame({ 'ID':['A', 'B', 'C','D','E'], 'myvalue':['group1','group1','group2','group3','group3'] })
 
# Build your graph
plt.figure()
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
 
# The order of the node for networkX is the following order:
# Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!
G.nodes()
print("Nodes to order: ",G.nodes())
 
# Here is the tricky part: I need to reorder carac to assign the good color to each node
carac = carac.set_index('ID')
carac = carac.reindex(G.nodes())
 
# And I need to transform my categorical column in a numerical value: group1->1, group2->2...
carac['myvalue'] = pd.Categorical(carac['myvalue'])
print("Nodes to order: ", carac['myvalue'].cat.codes)
 
# Custom the nodes:
nx.draw_networkx(G, with_labels=True, node_color=carac['myvalue'].cat.codes, cmap=plt.cm.Set1, node_size=1500)

#Show the graph
plt.title("Categorical color scale")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No324_Categorical_color_scale.png', bbox_inches='tight')



###############################################################################
print("****************************************************")
topic = "Graph No.325 Map colour to the edges of a Network"; print("** %s\n" % topic)
###############################################################################
## This chart follows the chart #324 where we learned how to map a color to 
## each nodes of a network. This time, we suppose that we have a feature for 
## each edge of our network. For example, this feature can be the amount of 
## money that this links represents (numerical value), or on which continent 
## it happened (categorical value). We want the edge to be different according 
## to this variable, and here is how to do it:
###############################################################################
## -----------------------------------------------------------------> numerical
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C'], 'value':[1, 10, 5, 5]})
 
# Build your graph
plt.figure()
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
 
# Custom the nodes:
nx.draw_networkx(G, with_labels=True, node_color='skyblue', node_size=1500, 
                                      edge_color=df['value'], width=10.0, edge_cmap=plt.cm.Blues)

#Show the graph
plt.title("numerical")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No325_numerical.png', bbox_inches='tight')



## ---------------------------------------------------------------> categorical
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C'], 'value':['typeA', 'typeA', 'typeB', 'typeB']})

# And I need to transform my categorical column in a numerical value typeA->1, typeB->2...
df['value'] = pd.Categorical(df['value'])
print("Nodes to order: ", carac['myvalue'].cat.codes)
 
# Build your graph
plt.figure()
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
 
# Custom the nodes:
nx.draw_networkx(G, with_labels=True, node_color='skyblue', node_size=1500, 
                                      edge_color=df['value'].cat.codes, width=10.0, edge_cmap=plt.cm.Set2)

#Show the graph
plt.title("categorical")
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No325_categorical.png', bbox_inches='tight')



###############################################################################
print("****************************************************")
topic = "Graph No.326 Background colour of network chart"; print("** %s\n" % topic)
###############################################################################
## You can change the background colour of your network chart with 
## fig.set_facecolor(). Note that you need to add fig.get_facecolor if you want 
## to keep your background colour for your png.
###############################################################################
## -----------------------------------------------------------------> with draw
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C'] })
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
 
# Custom the nodes:
fig = plt.figure()
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='white')
fig.set_facecolor("#00000F")

#Show the graph
plt.suptitle(topic, fontsize=12, color='peachpuff', weight='bold')
plt.show()

# Save as png
plt.savefig('networkx/No326_with_draw.png', bbox_inches='tight', facecolor=fig.get_facecolor())



## --------------------------------------------------------> with draw_networkx
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C'] })
 
# Build your graph
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
 
# Custom the nodes:
fig = plt.figure()
nx.draw_networkx(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='white')
fig.set_facecolor("#00000F")
#plt.gca().set_facecolor('deepskyblue')
plt.gca().axis('off')

#Show the graph
plt.title("with draw_networkx", color='white')
plt.suptitle(topic, fontsize=12, color='peachpuff', weight='bold')
plt.show()

# Save as png
plt.savefig('networkx/No326_with_draw_networkx.png', bbox_inches='tight', facecolor=fig.get_facecolor())



###############################################################################
print("****************************************************")
topic = "Graph No.327 Network from correlation matrix"; print("** %s\n" % topic)
###############################################################################
## This page explains how to draw a correlation network: a network build on a 
## correlation matrix.
## 
## Suppose that you have 10 individuals, and know how close they are related 
## to each other. It is possible to represent these relationships in a network. 
## Each individual will be a node. If 2 individuals are close enough 
## (we set a threshold), then they are linked by a edge. That will show the 
## structure of the population!
## 
## In this example, we see that our population is clearly split in 2 groups!
###############################################################################
## -----------------------------------------------------------------> with draw
# I build a data set: 10 individuals and 5 variables for each
ind1 = [5,10,3,4,8,10,12,1,9,4]
ind5 = [1,1,13,4,18,5,2,11,3,8]
df = pd.DataFrame({ 'A': ind1, 
                    'B': ind1 + np.random.randint(10, size=(10)) , 
                    'C': ind1 + np.random.randint(10, size=(10)) , 
                    'D': ind1 + np.random.randint(5, size=(10)) , 
                    'E': ind1 + np.random.randint(5, size=(10)), 
                    'F': ind5, 
                    'G': ind5 + np.random.randint(5, size=(10)) , 
                    'H': ind5 + np.random.randint(5, size=(10)), 
                    'I': ind5 + np.random.randint(5, size=(10)), 
                    'J': ind5 + np.random.randint(5, size=(10))
                })
print("Dataframe: \n{}\n".format(df))

# Calculate the correlation between individuals. We have to transpose first, because the corr function calculate the pairwise correlations between columns.
corr = df.corr()
print("Correlation: \n{}\n".format(corr))
 
# Transform it in a links data frame (3 columns only):
links = corr.stack().reset_index()
links.columns = ['var1', 'var2','value']
print("Links: \n{}\n".format(links))

# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
links_filtered=links.loc[ (links['value'] > 0.8) & (links['var1'] != links['var2']) ]
print("Filtered links: \n{}\n".format(links_filtered))


# Build your graph
G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')#, create_using=nx.Graph() )
 
# Custom the nodes:
fig = plt.figure()
nx.draw_networkx(G, with_labels=True, node_color='orange', 
                    node_size=400, edge_color='black', linewidths=1, font_size=15, pos=nx.circular_layout(G))

#Show the graph
plt.title("Network from correlation matrix", color='white')
plt.suptitle(topic, fontsize=12, color='darkred')
plt.show()

# Save as png
plt.savefig('networkx/No327_correlation_network.png', bbox_inches='tight')



print("****************************************************")
topic = "Restore default configuration"; print("** %s\n" % topic)

plt.style.use('default')



print("****************************************************")
print("** END                                            **")
print("****************************************************")

