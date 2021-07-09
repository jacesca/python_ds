# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:14:41 2020

@author: jacesca@gmail.com
"""

###############################################################################
##                                                            L I B R A R I E S
###############################################################################
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import seaborn as sns
import pprint

from scipy.cluster import hierarchy 
from scipy.cluster.hierarchy import dendrogram 
from scipy.cluster.hierarchy import linkage 

###############################################################################
##                                    I N I T I A L   C O N F I G U R A T I O N
###############################################################################
plt.rcParams['figure.max_open_warning'] = 60 

###############################################################################
##                                              R E A D I N G   T H E   D A T A
###############################################################################
# Import the mtcars dataset from the web + keep only numeric variables
#url = 'https://python-graph-gallery.com/wp-content/uploads/mtcars.csv'
url = "dendrogram/mtcars.csv"
df = pd.read_csv(url)
df = df.set_index('model')
df.index.name = None
print("Data:\n{}\n\n".format(df))


###############################################################################
##          P R E P E A R I N G   T H E   D A T A   F O R   D E N D R O G R A M
###############################################################################
##  This page aims to describe how to realise a basic dendrogram with Python. 
##  To realise such a dendrogram, you first need to have a numeric matrix. 
##  Each line represent an entity (here a car). Each column is a variable that 
##  describes the cars. The objective is to cluster the entities to know who 
##  share similarities with who.
##  At the end, entities that are highly similar are close in the Tree. Let’s 
##  start by loading a dataset and the requested libraries:
###############################################################################
# All right, now that we have our numeric matrix, we can calculate the distance 
# between each car, and realise the hierarchical clustering. This is done 
# through the linkage function. I do not enter in the details now, but I 
# strongly advise to visit the graph #401 for more details concerning this 
# crucial step.

# Calculate the distance between each sample
# You have to think about the metric you use (how to measure similarity) + 
# about the method of clusterization you use (How to group cars)
Z = linkage(df, 'ward')



###############################################################################
##                                                            *** 400 GRAPH ***
###############################################################################
# You can easily plot this object as a dendrogram using the dendrogram 
# function. See graph #401 for possible customisation.
###############################################################################
# Prepare the plot
plt.suptitle('Hierarchical Clustering Dendrogram', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
plt.subplots_adjust(left=None, bottom=.35, right=None, top=None, wspace=1, hspace=.1);

# Make the dendrogram
dendrogram(Z, labels=df.index, leaf_rotation=90)
plt.show()
print('Graph No.400 created...')

###############################################################################
##                                                            *** 401 GRAPH ***
###############################################################################
## The chart #400 gives the basic steps to realise a dendrogram from a numeric 
## matrix. Here, let’s describe a few customisation that you can easily apply 
## to your dendrogram.
###############################################################################
## First Customisation---------------------------------------------->LEAF LABEL
# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram - Leaf label', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
plt.subplots_adjust(left=None, bottom=.35, right=None, top=None, wspace=1, hspace=.1);

# Plot with Custom leaves
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=6, labels=df.index)
plt.show()



## Second Customisation-------------------------------------------># OF CLUSTER
# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram - # of cluster', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
plt.subplots_adjust(left=None, bottom=.35, right=None, top=None, wspace=1, hspace=.1);

# Control number of clusters in the plot + add horizontal line.
hierarchy.dendrogram(Z, color_threshold=200, labels=df.index, leaf_rotation=90)
plt.axhline(y=240, c='grey', lw=1, linestyle='dashed')
plt.show()



## Third Customisation--------------------------------------------------->COLOR
# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram - Color', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
plt.subplots_adjust(left=None, bottom=.35, right=None, top=None, wspace=1, hspace=.1);

# Set the colour of the cluster here:
#hierarchy.set_link_color_palette(['green','red', 'orange','navy'])
hierarchy.set_link_color_palette(list(sns.color_palette("colorblind").as_hex()))
 
# Make the dendrogram and give the colour above threshold
hierarchy.dendrogram(Z, color_threshold=200, above_threshold_color='grey', labels=df.index, leaf_rotation=90)
 
# Add horizontal line.
plt.axhline(y=200, c='grey', lw=1, linestyle='dashed')
plt.show()



## Fourth Customisation----------------------------------------------->TRUNCATE
# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram - Truncate (method 1)', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
#plt.subplots_adjust(left=None, bottom=.35, right=None, top=None, wspace=1, hspace=.1);
# method 1: lastp
hierarchy.dendrogram(Z, truncate_mode = 'lastp', p=4) # -> you will have 4 leaf at the bottom of the plot
plt.show()

 

# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram - Truncate (method 2)', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
#plt.subplots_adjust(left=None, bottom=.35, right=None, top=None, wspace=1, hspace=.1);
# method 2: level
#There is a bug in the dendrogram when the p especified generate a set with 1 or 0 elements. In this example, this occurs with p>1
tree = hierarchy.dendrogram(Z, truncate_mode = 'level', p=1) # -> No more than ``p`` levels of the dendrogram tree are displayed.
plt.show()
pprint.pprint(tree)



## Fifth Customisation--------------------------------------------->ORIENTATION
# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram - Orientation Right', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
plt.subplots_adjust(left=.25, bottom=None, right=None, top=None, wspace=1, hspace=.1);
# Orientation of the dendrogram
hierarchy.dendrogram(Z, orientation="right", labels=df.index)
plt.show()



# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram - Orientation Left', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
plt.subplots_adjust(left=None, bottom=None, right=.8, top=None, wspace=1, hspace=.1);
# or
hierarchy.dendrogram(Z, orientation="left", labels=df.index)
plt.show()



# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram - Orientation Bottom', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
plt.subplots_adjust(left=None, bottom=None, right=None, top=.65, wspace=1, hspace=.1);
# or
hierarchy.dendrogram(Z, orientation="bottom", labels=df.index, leaf_rotation=90)
plt.show()



# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram - Orientation Top (default)', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
plt.subplots_adjust(left=None, bottom=None, right=.8, top=None, wspace=1, hspace=.1);
# or
hierarchy.dendrogram(Z, orientation="top", labels=df.index, leaf_rotation=90)
plt.show()

print('9 graphs of seccion No.401 created...')



###############################################################################
##                                                            *** 402 GRAPH ***
###############################################################################
## Using graph #400 and #401 you should be able to build a dendrogram and 
## customise most of its features. But now, you probably want to compare the 
## structure you get with your expectations.
## In this example we will consider the mtcars dataset. It is a numeric matrix 
## that gives the feature of several cars. We can cluster these cars, represent 
## their structure in a group, and color the car names following their cylinder 
## (the ‘cyl’ column). Thus, we will know if the cylinder is responsible of 
## this structure!
###############################################################################
# Prepare the plot
plt.figure()
plt.suptitle('Hierarchical Clustering Dendrogram', color='maroon')
plt.title('Car clasification', weight='bold', color='darkblue')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
plt.subplots_adjust(left=None, bottom=None, right=.8, top=None, wspace=1, hspace=.1);

# Make the dendro
dendrogram(Z, labels=df.index, orientation="left", color_threshold=240, above_threshold_color='grey')

# Create a color palette with 3 color for the 3 cyl possibilities
my_palette = plt.cm.get_cmap("Accent", 3)
 
# Transforme the 'cyl' column in a categorical variable. It will allow to put one color on each level.
#The next 2 lines are comment to preserv the original DataFrame
#df['cyl']=pd.Categorical(df['cyl']) #df['cyl'].unique()-->[6, 4, 8]
#my_color=df['cyl'].cat.codes
#To preserve the original dataframe
df_cyl = df[['cyl']].copy()
df_cyl['cyl'] = pd.Categorical(df_cyl['cyl']) #df['cyl'].unique()-->[6, 4, 8]
my_color = df_cyl['cyl'].cat.codes
 
# Apply the right color to each label
axes = plt.gca()
ylbls = axes.get_ymajorticklabels()
num=-1
for lbl in ylbls:
    marca = str(lbl)[12:len(str(lbl))-2] #lbl like "Text(1, 0, 'Honda Civic')"
    lbl.set_color(my_palette(my_color[marca]))
plt.show()

print('Graph No.402 created...')



###############################################################################
##                                                            *** 404 GRAPH ***
###############################################################################
## When you use a dendrogram to display the result of a cluster analysis, it 
## is a good practice to add the corresponding heatmap. It allows you to 
## visualise the structure of your entities (dendrogram), and to understand 
## if this structure is logical (heatmap).  This is easy work thanks to the 
## seaborn library that provides an awesome ‘cluster map’ function. This page 
## aims to describe how it works, and note that once more the seaborn 
## documentation is awesome.print('Graph No.404 created...')
###############################################################################
"""
# Data set
url = "dendrogram/mtcars.csv"
df = pd.read_csv(url)
df = df.set_index('model')
df.index.name = None
"""
## First Customisation-------------------------------------------->DEFAULT PLOT
# Default plot
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(1) Default plot', color='maroon', ha='left', x=.20)
#g.ax_heatmap.set_title('Default clustermap', weight='bold', color='darkblue')
#plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
#plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90) # For x axis
#plt.subplots_adjust(left=None, bottom=None, right=None, top=.95, wspace=None, hspace=None);
plt.show()


## Second Customisation---------------------------------------------->NORMALIZE
# Standardize or Normalize every column in the figure
# Standardize:
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, standard_scale=1, figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(2) Standarize', color='maroon', ha='left', x=.20)
plt.show()


# Or
# Normalize
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, z_score=1, figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(3) Normalize', color='maroon', ha='left', x=.20)
plt.show()


## Third Customisation----------------------------------------->DISTANCE METHOD
# OK now we can compare our individuals. But how do you determine the similarity 
# between 2 cars?
# Several way to calculate that. the 2 most common ways are:

# Correlation method
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, metric="correlation", standard_scale=1, figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(4) Correlation distance', color='maroon', ha='left', x=.20)
plt.show()

# and Euclidean method
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, metric="euclidean", standard_scale=1, figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(5) Eucledean distance', color='maroon', ha='left', x=.20)
plt.show()


## Fourth Customisation----------------------------------------->CLUSTER METHOD
# OK now we determined the distance between 2 individuals. But how to do the 
# clusterisation? Several methods exist.
# If you have no idea, ward is probably a good start.
# Single clustering
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, metric="euclidean", standard_scale=1, method="single", figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(6) Single cluster', color='maroon', ha='left', x=.20)
plt.show()

# and Ward clustering
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, metric="euclidean", standard_scale=1, method="ward", figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(7) Ward cluster', color='maroon', ha='left', x=.20)
plt.show()


## Fifth Customisation--------------------------------------------------->COLOR
# Change color palette - mako
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, metric="euclidean", standard_scale=1, method="ward", cmap="mako", figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(8) mako cmap', color='maroon', ha='left', x=.20)
plt.show()

# Change color palette - viridis
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, metric="euclidean", standard_scale=1, method="ward", cmap="viridis", figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(9) viridis cmap', color='maroon', ha='left', x=.20)
plt.show()

# Change color palette - Blues
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, metric="euclidean", standard_scale=1, method="ward", cmap="Blues", figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(10) Blues cmap', color='maroon', ha='left', x=.20)
plt.show()


## Sixth Customisation------------------------------------------------>OUTLIERS
# Ignore outliers
# Let's create an outlier in the dataset:
original_value = df.at['Lincoln Continental', 'drat']
df.at['Lincoln Continental', 'drat'] = 1000
#df.loc['Lincoln Continental','drat']=1000

# use the outlier detection
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, robust=True, figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(11) Outliers detection', color='maroon', ha='left', x=.20)
plt.show()
 
# do not use it
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, robust=False, figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\n(12) Without outliers detection', color='maroon', ha='left', x=.20)
plt.show()


print('12 graphs of seccion No.404 created...')
#Return to its original value.
df.at['Lincoln Continental', 'drat'] = original_value 


###############################################################################
##                                                            *** 405 GRAPH ***
###############################################################################
## The chart No.404 describes in detail how to do a dendrogram with heatmap 
## using seaborn. I strongly advise to read it before doing this chart. Once 
## you understood how to study the structure of your population, you probably 
## want to compare it with your expectation.
## Here I use the mtcars dataset that gives the features of several cars 
## through a few numerical variables. I represent how these cars are clustered. 
## Then, I add a color sheme on the left part of the plot. The 3 colours 
## represent the 3 possible values of the ‘cyl’ column. Now, you know if this 
## column explain the structure of our car population!
###############################################################################
"""
# Data set
url = "dendrogram/mtcars.csv"
df = pd.read_csv(url)
df = df.set_index('model')
df.index.name = None
"""
# Prepare a vector of color mapped to the 'cyl' column
my_palette = dict(zip(df.cyl.unique(), ["orange","yellow","brown"]))
row_colors = df.cyl.map(my_palette)
 
# plot
plt.rc('ytick',labelsize=6)
g = sns.clustermap(df, metric="correlation", method="single", cmap="Blues", standard_scale=1, row_colors=row_colors, figsize=(11,5.75))
g.fig.suptitle('Hierarchical Clustering Dendrogram\nExpectation', color='maroon', ha='left', x=.20)
plt.show()
print('Graph No.405 created...')


print('END.')
plt.style.use('default')
