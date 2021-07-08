# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:32:02 2019

@author: jacqueline.cortez
Chapter 4: Creating Plots on Data Aware Grids
    Using Seaborn to draw multiple plots in a single figure.
    
Source: (legend help)
https://stackoverflow.com/questions/47325845/setting-legend-only-for-one-of-the-marginal-plots-in-seaborn
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot as plt                                               #For creating charts
import pandas            as pd                                                #For loading tabular data
import seaborn           as sns                                               #For visualizing data
import scipy.stats       as stats                                             #For accesign to a vary of statistics functiosn

from pandas.api.types                import CategoricalDtype                  #For categorical data

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

plt.rcParams['figure.max_open_warning'] = 60

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "2018_College_Scorecard_Tuition.csv"
df_college = pd.read_csv(filename)
#Create categorical type data to use
cats = CategoricalDtype(categories=[1,2,3,4],  ordered=True) #order only works for categorical types
# Change the data type of 'rating' to category
df_college['HIGHDEG'] = df_college.HIGHDEG.astype(cats)
print("Columns of {}:\n{}".format(filename, df_college.columns))

filename = "US_Market_Rent.csv"
df_rent = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_rent.columns))

filename = "Automobile_Insurance_Premiums.csv"
df_auto = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_auto.columns))

filename = "Washington_Bike_Share.csv"
df_bike = pd.read_csv(filename)
print("Columns of {}:\n{}".format(filename, df_bike.columns))

print("****************************************************")
topic = "1. Using FacetGrid, factorplot and lmplot"; print("** %s\n" % topic)

#Facetgrid categorical example with boxplot
g = sns.FacetGrid(df_college, col='HIGHDEG', height=4, aspect=0.7)
g.map(sns.boxplot, 'Tuition', order=[1,2,3,4])
plt.suptitle("{}\nFacetgrid categorical example with boxplot".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

#Factorplot
#Warning: C:\Anaconda3\lib\site-packages\seaborn\categorical.py:3666: UserWarning: 
#The `factorplot` function has been renamed to `catplot`. 
#The original name will be removed in a future release. Please update your code. 
#Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` 
#in `catplot`.
sns.catplot(data=df_college, x='Tuition', col='HIGHDEG', kind='box', height=4, aspect=0.7)
plt.suptitle("{}\nfactorplot() / catplot()".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot


#Facegrid example with scatterplot
g = sns.FacetGrid(df_college, hue='HIGHDEG', col='HIGHDEG', height=4, aspect=0.7)
g.map(plt.scatter, 'Tuition', 'SAT_AVG_ALL', s=10, alpha=0.5)
plt.suptitle("{}\nFacegrid example with scatterplot".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

#lmplot
sns.lmplot(data=df_college, x='Tuition', y='SAT_AVG_ALL', 
           scatter_kws={'s': 15, 'alpha': 0.5},
           hue='HIGHDEG', col='HIGHDEG', fit_reg=False, height=4, aspect=0.7)
plt.suptitle("{}\nlmplot()".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "2. Building a FacetGrid"; print("** %s\n" % topic)

# Create FacetGrid with Degree_Type and specify the order of the rows using row_order
g2 = sns.FacetGrid(df_college, row="Degree_Type", height=1.4, aspect=6,
                   row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])

# Map a pointplot of SAT_AVG_ALL onto the grid
g2.map(sns.pointplot, 'SAT_AVG_ALL', order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])

plt.suptitle("{}\nFacetGrid with pointplot".format(topic))
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

print("****************************************************")
topic = "3. Using a factorplot"; print("** %s\n" % topic)

#C:\Anaconda3\lib\site-packages\seaborn\categorical.py:3666: UserWarning: The `factorplot` function has been renamed to 
#`catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in 
#`factorplot` (`'point'`) has changed `'strip'` in `catplot`.}

# Create a factor plot that contains boxplots of Tuition values
sns.catplot(data=df_college, x='Tuition', row='Degree_Type',
               #row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'], 
               kind='box', height=1.1, aspect=6)
plt.suptitle("{}\nfactorplot = catplot".format(topic))
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


# Create a facetted pointplot of Average SAT_AVG_ALL scores facetted by Degree Type 
sns.catplot(data=df_college, x='SAT_AVG_ALL', row='Degree_Type',
               row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'],
               kind='point', height=1.4, aspect=6)
plt.suptitle("{}\nCatplot with pointplot".format(topic))
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

print("****************************************************")
topic = "4. Using a lmplot"; print("** %s\n" % topic)

degree_ord = ['Graduate', 'Bachelors', 'Associates']
inst_ord   = ['Public', 'Private non-profit']

# Create a FacetGrid varying by column and columns ordered with the degree_order variable
g = sns.FacetGrid(df_college, col="Degree_Type", col_order=degree_ord,
                  height=4, aspect=1)
g.map(plt.scatter, 'UG', 'PCTPELL') # Map a scatter plot of Undergrad Population compared to PCTPELL
plt.suptitle("{}\nTrying first with FacetGrid()".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


# Re-create the plot above as an lmplot
sns.lmplot(data=df_college, x='UG', y='PCTPELL', 
           col="Degree_Type", col_order=degree_ord,
           height=4, aspect=1)
plt.suptitle(topic)
plt.suptitle("{}\nImproving using lmplot()".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


# Create an lmplot that has a column for Ownership, a row for Degree_Type and hue based on the WOMENONLY column
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

sns.lmplot(data=df_college, x='SAT_AVG_ALL', y='Tuition', 
           col="Ownership", col_order=inst_ord,
           row='Degree_Type', row_order=['Graduate', 'Bachelors'], 
           hue='WOMENONLY',
           scatter_kws=dict(alpha=0.5, s=25),
           line_kws={'lw': 1},
           height=2.5, aspect=2)
plt.suptitle("{}\nMore lmplot()".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

plt.style.use('default')

print("****************************************************")
topic = "5. Using PairGrid and pairplot"; print("** %s\n" % topic)

#PairGrid
g = sns.PairGrid(df_rent, vars=['fmr_0', 'acs_2017_2'], 
                 height=2.5, aspect=1.25)
g.map(plt.scatter)
plt.suptitle("{}\nPairGrid()".format(topic))
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

#PairGrid Custumized
g = sns.PairGrid(df_rent, vars=['fmr_0', 'acs_2017_2'], 
                 height=2.5, aspect=1.25)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter, alpha=0.5, s=35)
plt.suptitle("{}\nPairGrid()".format(topic))
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


#PairPlot
g = sns.pairplot(df_rent, vars=['fmr_0', 'acs_2017_2'], 
                 kind='reg', diag_kind='hist',
                 height=2.5, aspect=1.25)
plt.suptitle("{}\npairplot()".format(topic))
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


#PairPlot Customized
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

g = sns.pairplot(df_rent.dropna(), vars=['fmr_0', 'acs_2017_2', 'pop2010'], 
                 hue='fmr_type', 
                 diag_kind='kde', 
                 palette='husl',
                 plot_kws=dict(s=25, alpha=0.5),
                 height=1.8, aspect=1.25)
plt.suptitle("{}\npairplot()".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

plt.style.use('default')


print("****************************************************")
topic = "6. Building a PairGrid"; print("** %s\n" % topic)

# Create a PairGrid with a scatter plot for fatal_collisions and premiums
g = sns.PairGrid(df_auto, vars=["fatal_collisions", "premiums"],
                 height=2.5, aspect=1.25)
g2 = g.map(plt.scatter)
plt.suptitle("{}\nPairGrid with scatter plot".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot


# Create the same PairGrid but map a histogram on the diag
g = sns.PairGrid(df_auto, vars=["fatal_collisions", "premiums"],
                 height=2.5, aspect=1.25)
g = g.map_diag(plt.hist, bins=10, alpha=0.5)
g = g.map_offdiag(plt.scatter, alpha=0.5, s=35, color='red')
plt.suptitle("{}\nPairGrid with histogram and scatter plots".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "7. Using a pairplot"; print("** %s\n" % topic)

# Create a pairwise plot of the variables using a scatter plot
g = sns.pairplot(data=df_auto, vars=["fatal_collisions", "premiums"],
                 kind='scatter', plot_kws=dict(s=25, alpha=0.5),
                 height=2.5, aspect=1.25)
plt.suptitle("{}\nPairplot with scatter plot".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


# Plot the same data but use a different color palette and color code by Region
g = sns.pairplot(data=df_auto, vars=["fatal_collisions", "premiums"],
                 hue='Region', palette='RdBu',
                 kind='scatter', plot_kws=dict(s=25, alpha=0.5),
                 diag_kind='hist', diag_kws={'alpha':.5},
                 height=2.5, aspect=1.25)
plt.suptitle("{}\nPairplot with scatter plot".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

print("****************************************************")
topic = "8. Additional pairplots"; print("** %s\n" % topic)

# Build a pairplot with different x and y variables
sns.pairplot(data=df_auto,
        x_vars=["fatal_collisions_speeding", "fatal_collisions_alc"],
        y_vars=['premiums', 'insurance_losses'],
        hue='Region', palette='BrBG',
        kind='scatter',
        height=2.5, aspect=1.25)
plt.suptitle("{}\nPairplot with different x and y".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


# plot relationships between insurance_losses and premiums
sns.pairplot(data=df_auto, vars=["insurance_losses", "premiums"],
             hue='Region', palette='husl', #markers=["o", "s", "D", '+'],
             kind='reg', plot_kws=dict(scatter_kws=dict(alpha=0.95, s=35), line_kws=dict(linewidth=1)),
             diag_kind = 'kde', diag_kws={'alpha':.5},
             height=2.5, aspect=1.25)
plt.suptitle("{}\nMore pairplot".format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

print("****************************************************")
topic = "9. Using JointGrid and jointplot"; print("** %s\n" % topic)

#Basic JointGrid
g = sns.JointGrid(data=df_college.dropna(), x='Tuition', y='ADM_RATE_ALL',
                  height=5, ratio=4)
g = g.plot(sns.regplot, sns.distplot)
plt.suptitle('{}\nBasic JoinGrid'.format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


#Advanced JointGrid
legend_properties = {'weight':'bold','size':8}

g = sns.JointGrid(data=df_college.dropna(), x='Tuition', y='ADM_RATE_ALL',
                  height=5, ratio=4) #, space=0.01
g = g.plot_joint(sns.kdeplot, #label='Relation betwwen Tuition and Admission Rate',
                 cmap="Reds_d") #If where plt.scatter--> color="g", s=40, edgecolor="white"
g = g.plot_marginals(sns.kdeplot, shade=True, label='KDE')
g = g.annotate(stats.pearsonr)
#legendMain = g.ax_joint.legend(prop=legend_properties, loc='lower right')
legendSide = g.ax_marg_x.legend(prop=legend_properties,loc='upper right')
legendSide = g.ax_marg_y.legend(prop=legend_properties,loc='lower left')
#rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2
#g = g.annotate(rsquare, template="{stat}: {val:.2f}", stat="$R^2$", loc="upper left", fontsize=12)
plt.suptitle('{}\nAdvanced JoinGrid'.format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


#Advanced JointGrid With only one legend in the marginals
legend_properties = {'weight':'bold','size':8}

g = sns.JointGrid(data=df_college.dropna(), x='Tuition', y='ADM_RATE_ALL',
                  height=5, ratio=4) #, space=0.01
g = g.plot_joint(sns.kdeplot, #label='Relation betwwen Tuition and Admission Rate',
                 cmap="Reds_d") #If where plt.scatter--> color="g", s=40, edgecolor="white"
g = g.plot_marginals(sns.kdeplot, shade=True)
g = g.annotate(stats.pearsonr)
#legendMain = g.ax_joint.legend(prop=legend_properties, loc='lower right')
legendSide = g.ax_marg_x.legend(labels=['x'], prop=legend_properties,loc='upper right')
#legendSide = g.ax_marg_y.legend(prop=legend_properties,loc='lower left')
#rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2
#g = g.annotate(rsquare, template="{stat}: {val:.2f}", stat="$R^2$", loc="upper left", fontsize=12)
plt.suptitle('{}\nAdvanced JoinGrid'.format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


#Basic jointplot()
g = sns.jointplot(data=df_college.dropna(), x='Tuition', y='ADM_RATE_ALL', 
                  kind='hex',
                  height=5, ratio=4)
plt.suptitle('{}\nBasic jointplot'.format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

#Advanced jointplot()
g = (sns.jointplot(data=df_college.query('UG < 2500 & Ownership == "Public"'), x='Tuition', y='ADM_RATE_ALL', 
                   #kind='scatter', #scatter | reg | resid | kde | hex
                   #kind= 'reg',
                   #xlim=(0,25000), 
                   marginal_kws=dict(bins=15, hist_kws=dict(edgecolor='gray')),
                   #marginal_kws=dict(bins=15, rug=True, hist_kws=dict(edgecolor='red')),
                   joint_kws = dict(alpha=0.5, s=15),
                   #joint_kws = dict(scatter_kws=dict(alpha=0.5, s=35), line_kws=dict(linewidth=1)), #For regression kind.
                   height=5, ratio=4).plot_joint(sns.kdeplot, 
                                                 #shade=True, shade_lowest=False,
                                                 #cmap="Reds", #color="r",
                                                 zorder=0, n_levels=6)) #.set_axis_labels("x", "y") #inside parentesis
plt.suptitle('{}\nAdvanced jointplot'.format(topic))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

print("****************************************************")
topic = "10. Building a JointGrid and jointplot"; print("** %s\n" % topic)

# Build a JointGrid comparing humidity and total_rentals
sns.set_style("whitegrid")
g = sns.JointGrid(data=df_bike, x="hum", y="total_rentals",
                  #xlim=(0.1, 1.0),
                  height=5, ratio=4) 
g.plot(sns.regplot, sns.distplot)
plt.suptitle('{}\nJointGrid'.format(topic))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

# Create a jointplot similar to the JointGrid 
sns.jointplot(data=df_bike, x="hum",  y="total_rentals",
              kind='reg', #joint_kws = dict(scatter_kws=dict(alpha=0.4, s=35, edgecolor='white'), line_kws=dict(linewidth=1)),
              height=5, ratio=4)
plt.suptitle('{}\njointplot'.format(topic))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

# Create a jointplot similar to the JointGrid 
sns.jointplot(data=df_bike, x="hum",  y="total_rentals",
              kind='resid', #joint_kws = dict(scatter_kws=dict(alpha=0.4, s=35, edgecolor='white'), line_kws=dict(linewidth=1)),
              height=5, ratio=4)
plt.suptitle('{}\njointplot (Residual)'.format(topic))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

print("****************************************************")
topic = "11. Jointplots and regression"; print("** %s\n" % topic)

# Plot temp vs. total_rentals as a regression plot
sns.jointplot(data=df_bike, x="temp", y="total_rentals",
              kind='reg', order=2, xlim=(0, 1), joint_kws = dict(scatter_kws=dict(alpha=0.4, s=35, edgecolor='white'), line_kws=dict(linewidth=1)),
              height=5, ratio=4).annotate(stats.pearsonr)
plt.suptitle('{}\nSecond Order Regression Plot'.format(topic))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot



# Plot a jointplot showing the residuals
sns.jointplot(data=df_bike, x="temp", y="total_rentals",
              kind='resid', order=2, xlim=(0, 1), 
              joint_kws = dict(scatter_kws=dict(alpha=0.4, s=35, edgecolor='white'), line_kws=dict(linewidth=1)),
              annot_kws = dict(fontsize=8),
              height=5, ratio=4).annotate(stats.pearsonr)
plt.suptitle('{}\nResidual Plot'.format(topic))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot

print("****************************************************")
topic = "12. Complex jointplots"; print("** %s\n" % topic)

# Create a jointplot of temp vs. casual riders
# Include a kdeplot over the scatter plot
g = (sns.jointplot(data=df_bike, x="temp", y="casual",
                   kind='scatter',
                   joint_kws = dict(alpha=0.4, s=35, edgecolor='white'),
                   marginal_kws=dict(bins=10, rug=True),
                   height=5, ratio=4).plot_joint(sns.kdeplot))
plt.suptitle('{}\nCasual riders'.format(topic))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=None, hspace=0.6)
plt.show() # Show the plot


# Replicate the above plot but only for registered riders
g = (sns.jointplot(data=df_bike, x="temp", y="registered",
                   kind='scatter',
                   joint_kws = dict(alpha=0.4, s=35, edgecolor='white', color='green'),
                   marginal_kws=dict(bins=10, rug=True),
                   height=5, ratio=4).plot_joint(sns.kdeplot, linewidths=0.5))
plt.suptitle('{}\nRegistered riders'.format(topic))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=None, hspace=1)
plt.show() # Show the plot

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import inspect                                                                #Used to get the code inside a function
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt                                               #For creating charts
#import numpy             as np                                                #For making operations in lists
#import pandas            as pd                                                #For loading tabular data
#import seaborn           as sns                                               #For visualizing data


#import calendar                                                               #For accesing to a vary of calendar operations
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
#from mpl_toolkits.mplot3d            import Axes3D
#from pandas.api.types                import CategoricalDtype                  #For categorical data
#from pandas.plotting                 import parallel_coordinates              #For Parallel Coordinates
#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
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
#from sklearn.preprocessing           import MaxAbsScaler                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing           import Normalizer                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing           import normalize                         #For learning machine - unsurpervised
#from sklearn.preprocessing           import scale                             #For learning machine
#from sklearn.preprocessing           import StandardScaler                    #For learning machine
#from sklearn.svm                     import SVC                               #For learning machine
#from sklearn.tree                    import DecisionTreeClassifier            #For learning machine - supervised
#from sklearn.tree                    import DecisionTreeRegressor             #For learning machine - supervised


#import statsmodels             as sm                                          #For stimations in differents statistical models
#import statsmodels.api         as sm                                          #Make a prediction model
#import statsmodels.formula.api as smf                                         #Make a prediction model    

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

#warnings.filterwarnings('ignore', 'Objective did not converge*')              #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
#warnings.filterwarnings('default', 'Objective did not converge*')             #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


#Create categorical type data to use
#cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  ordered=True)
# Change the data type of 'rating' to category
#weather['rating'] = weather.rating.astype(cats)