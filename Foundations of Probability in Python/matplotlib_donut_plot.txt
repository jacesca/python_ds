# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:38:29 2020

@author: jacesca@gmail.com
Source:
    https://python-graph-gallery.com/donut-plot/
    https://python-graph-gallery.com/pie-plot/
"""

# library
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
##                                                  *** D O N U T   P L O T ***
###############################################################################
# Data
names='groupA', 'groupB', 'groupC', 'groupD',
size=[12,11,3,30]
 
# create a figure and set different background
fig = plt.figure()
#fig.patch.set_facecolor('black')
 
# Change color of text
plt.rcParams['text.color'] = 'darkblue'
 
# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')
 
# Pieplot + circle on it
plt.pie(size, labels=names)
plt.legend()
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
plt.style.use('default')



###############################################################################
##                                                      *** P I E   P L O T ***
###############################################################################
 
# --- dataset 1: just 4 values for 4 groups:
df = pd.DataFrame([8,8,1,2], index=['a', 'b', 'c', 'd'], columns=['x'])
 
# make the plot
df.plot(kind='pie', subplots=True, figsize=(6, 5))


# --- dataset 2: 3 columns and rownames
df = pd.DataFrame({'var1':[8,3,4,2], 'var2':[1,3,4,1]}, index=['a', 'b', 'c', 'd'] )
 
# make the multiple plot
df.plot(kind='pie', subplots=True, figsize=(10,5))
plt.style.use('default')
