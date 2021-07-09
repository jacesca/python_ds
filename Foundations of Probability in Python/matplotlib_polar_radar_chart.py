# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:04:44 2020

@author: jaces
"""

###############################################################################
##  Libraries
###############################################################################
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 
###############################################################################
##  Set data 
###############################################################################
df = pd.DataFrame({'group': ['A','B','C','D'],
                   'var1': [38, 1.5, 30, 4],
                   'var2': [29, 10, 9, 34],
                   'var3': [8, 39, 23, 24],
                   'var4': [7, 31, 33, 14],
                   'var5': [28, 15, 32, 14]})
 
 
 
###############################################################################
##                                            *** F I R S T   E X A M P L E ***
###############################################################################
## ------- PART 1: Create background
###############################################################################
 # number of variable
categories=list(df)[1:] #-->['var1', 'var2', 'var3', 'var4', 'var5']
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
plt.ylim(0,40)
 
 
###############################################################################
# ------- PART 2: Add plots
###############################################################################
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group A")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group B")
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()




###############################################################################
##                                          *** S E C O N D   E X A M P L E ***
###############################################################################
# ------- PART 1: Define a function that do a plot for one line of the dataset!
###############################################################################
def make_spider( row, title, color):
    # number of variable
    categories=list(df)[1:]
    N = len(categories)
 
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    # Initialise the spider plot
    ax = plt.subplot(2,2,row+1, polar=True, )
 
    # If you want the first axis to be on top
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    plt.ylim(0,40)
 
    # Ind1
    values=df.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
 
    # Add a title
    plt.title(title, size=11, color=color, y=1.1)
 


###############################################################################
# ------- PART 2: Apply to all individuals
###############################################################################
# initialize the figure
my_dpi=96
#plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
plt.figure(figsize=(11,6), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(df.index))
 
# Loop to plot
for row in range(0, len(df.index)):
    make_spider( row=row, title='group '+df['group'][row], color=my_palette(row))

#plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=.5, hspace=None);
plt.tight_layout()
plt.show()
plt.style.use('default')