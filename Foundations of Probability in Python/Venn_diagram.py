# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 09:01:47 2020

@author: jacesca@gmail.com
Source:https://python-graph-gallery.com/venn-diagram/
https://pypi.org/project/matplotlib-venn/
"""

###############################################################################
##  Libraries
###############################################################################
# Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib_venn import venn2
from matplotlib_venn import venn2_unweighted 
from matplotlib_venn import venn2_circles 
from matplotlib_venn import venn3
#from matplotlib_venn import venn3_unweighted 
from matplotlib_venn import venn3_circles 

###############################################################################
##                                            *** F I R S T   E X A M P L E ***
###############################################################################
# Basic Venn
plt.figure(figsize=(12,5.75))
plt.subplot(3, 4, 1)
# First way to call the 2 group Venn diagram:
venn2(subsets = (10, 5, 2), set_labels = ('Group A', 'Group B'), alpha = 0.2)
 
plt.subplot(3, 4, 2)
# Second way
plt.title("Using set instead of weights",color='red')
#S1 = ['A', 'B', 'C', 'D']; S2 = ['D', 'E', 'F'])]
#Venn diagram --> S1 ꓴ S2
venn2([set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']), 
       set(['K', 'L', 'M', 'N', 'O', 'P', 'Q'])])

plt.subplot(3, 4, 3)
v = venn2( (10, 5, 2), alpha = 1 )
# Change Backgroud
plt.gca().set_facecolor('skyblue')
plt.gca().set_axis_on()

plt.subplot(3, 4, 4)
plt.title("Same size (unweightened)",color='red')
venn2_unweighted(subsets = (10, 5, 2), set_labels = ('Group A', 'Group B'))

plt.subplot(3, 4, 5)
plt.title("Only lines",color='red')
#c=venn2_circles(subsets = (10, 5, 2), linestyle='solid', linewidth=1, color="red")
c=venn2_circles(subsets = (1, 1, 1), linestyle='solid', linewidth=1, color="red")

plt.subplot(3, 4, 6)
v = venn2(subsets={'10': 10, '01': 5, '11': 2}, set_labels = ('A', 'B'))
c = venn2_circles(subsets=(10, 5, 2), linestyle='dashed')
v.get_patch_by_id('10').set_alpha(1.0)
v.get_patch_by_id('10').set_color('white')
v.get_label_by_id('10').set_text('Unknown')
v.get_label_by_id('A').set_text('Set A')

plt.subplot(3, 4, 7)
v = venn2(subsets={'10': 10, '01': 5, '11': 2}, set_labels = ('A', 'B'), alpha=0)
c = venn2_circles(subsets=(10, 5, 2), linestyle='solid', linewidth=0.5, color='green')
c[0].set_lw(1)
c[0].set_ls('dashed')
#c[0].set_color('red')
c[0].set_edgecolor('red')

plt.subplot(3, 4, 8)
set_a = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'])
set_b = set(['K', 'L', 'M', 'N', 'O', 'P', 'Q'])
total = len(set_a.union(set_b))
venn2([set_a, set_b], set_labels = ('Group A', 'Group B'), subset_label_formatter=lambda x: f"{(x/total):1.0%}")
plt.title("Percentage",color='red')

plt.subplot(3, 4, 9)
total = 10+5+2
venn2_unweighted(subsets = (10, 5, 2), set_labels = ('Group A', 'Group B'), subset_label_formatter=lambda x: f"{(x/total):1.0%}")
plt.title("Percentage (second method)",color='red')

plt.subplot(3, 4, 10)
v = venn2_unweighted(subsets = (10, 5, 2), set_labels = ('Group A', 'Group B'))
for text in v.set_labels: text.set_fontsize(10) #Labels
for text in v.subset_labels: text.set_fontsize(6) #values
plt.title("Fontsize",color='red')

plt.subplot(3, 4, 11)
v = venn2_unweighted(subsets = (10, 5, 2), set_labels = ('Group A', 'Group B'))
v.get_label_by_id('10').set_text('')
v.get_label_by_id('01').set_text('')
v.get_label_by_id('11').set_text('')
plt.title("Without numbers",color='red')

# Show it
plt.suptitle("Venn Diagrams Example", color='darkblue', weight='bold', fontsize=17)
plt.subplots_adjust(left=.05, bottom=None, right=.95, top=None, wspace=1, hspace=.1);
#plt.tight_layout()
plt.show()

###############################################################################
##                                          *** S E C O N D   E X A M P L E ***
###############################################################################
# Make the diagram
plt.figure(figsize=(12,5.5))

plt.subplot(2, 2, 1)
venn3(subsets = (10, 8, 22, 6,9,4,2))


plt.subplot(2, 2, 2)
# Custom text labels: change the label of group A
v=venn3(subsets = (10, 8, 22, 6,9,4,2), set_labels = ('Group A', 'Group B', 'Group C'))
v.get_label_by_id('A').set_text('My Favourite group!')
 

plt.subplot(2, 2, 3)
# Line style: can be 'dashed' or 'dotted' for example
v=venn3(subsets = (10, 8, 22, 6,9,4,2), set_labels = ('Group A', 'Group B', 'Group C'))
c=venn3_circles(subsets = (10, 8, 22, 6,9,4,2), linestyle='dashed', linewidth=1, color="grey")
 

plt.subplot(2, 2, 4)
# Change one group only
v=venn3(subsets = (10, 8, 22, 6,9,4,2), set_labels = ('Group A', 'Group B', 'Group C'))
c=venn3_circles(subsets = (10, 8, 22, 6,9,4,2), linestyle='dashed', linewidth=1, color="grey")
c[0].set_lw(8.0)
c[0].set_ls('dotted')
c[0].set_color('skyblue')


# Show it
plt.suptitle("Venn Diagrams Example", color='darkblue', weight='bold', fontsize=17)
plt.show()
 


###############################################################################
##                                          *** T H I R D   E X A M P L E ***
###############################################################################
plt.figure()
 
# Make a Basic Venn
v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
 
# Custom it
v.get_patch_by_id('010').set_alpha(1.0)
v.get_patch_by_id('010').set_color('white')
v.get_label_by_id('010').set_text('Unknown')
v.get_label_by_id('B').set_text('Set "B"')

c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
c[0].set_lw(1.0)
c[0].set_ls('dotted')
 
# Add title and annotation
plt.title("Sample Venn diagram")
plt.annotate('The familiar x', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
             ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
 
# Add title and annotation
plt.title("Sample Venn diagram")
plt.annotate('Unknown set', xy=v.get_label_by_id('010').get_position() + np.array([0, -0.15]), xytext=(70,-100),
             ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
# Show it
plt.show()




###############################################################################
##                                          *** F O U R T H   E X A M P L E ***
###############################################################################
plt.figure()
 
v3 = venn3(subsets = {'100':30, '010':30, '110':17,
                      '001':30, '101':17, '011':17, '111':5},
           set_labels = ('', '', ''))

v3.get_patch_by_id('100').set_color('red')
v3.get_patch_by_id('010').set_color('yellow')
v3.get_patch_by_id('001').set_color('blue')
v3.get_patch_by_id('110').set_color('orange')
v3.get_patch_by_id('101').set_color('purple')
v3.get_patch_by_id('011').set_color('green')
v3.get_patch_by_id('111').set_color('grey')

v3.get_label_by_id('100').set_text('Mathematique')
v3.get_label_by_id('010').set_text('Computer\nscience')
v3.get_label_by_id('001').set_text('Domain\nexpertise')
v3.get_label_by_id('110').set_text('Machine\nlearning')
v3.get_label_by_id('101').set_text('Statistical\nresearch')
v3.get_label_by_id('011').set_text('Data\nprocessing')
v3.get_label_by_id('111').set_text('Data\nscience')

for text in v3.subset_labels: text.set_fontsize(8)

plt.show()



###############################################################################
##                                            *** F I F T H   E X A M P L E ***
###############################################################################
plt.figure()

# First way
df = pd.DataFrame({'Product': ['Only cheese', 'Only red wine', 'Both'],
                   'NbClient': [900, 1200, 400]})

v = venn2(subsets = {'10': df[df.Product == 'Only cheese']['NbClient'].sum(), #df.loc[0, 'NbClient'],
                     '01': df[df.Product == 'Only red wine']['NbClient'].sum(), #df.loc[1, 'NbClient'],
                     '11': df[df.Product == 'Both']['NbClient'].sum()}, #df.loc[2, 'NbClient']},
          set_labels=('', ''))
                      
v.get_patch_by_id('10').set_color('yellow')
v.get_patch_by_id('01').set_color('red')
v.get_patch_by_id('11').set_color('orange')

v.get_patch_by_id('10').set_edgecolor('none')
v.get_patch_by_id('01').set_edgecolor('none')
v.get_patch_by_id('11').set_edgecolor('none')

for item, tag in zip(['10','01','11'], df.Product.values.tolist()):
    v.get_label_by_id(item).set_text('{}\n{:,.0f}\n{:,.2%}'.format(tag,
                                                                   df[df.Product == tag]['NbClient'].sum(),
                                                                   df[df.Product == tag]['NbClient'].sum()/df.NbClient.sum()*100))
    #v.get_label_by_id(item).set_text('%s\n%d\n(%.0f%%)' % (tag,
    #                                                       df[df.Product == tag]['NbClient'].sum(),
    #                                                       df[df.Product == tag]['NbClient'].sum()/df.NbClient.sum()*100))
"""
v.get_label_by_id('10').set_text('%s\n%d\n(%.0f%%)' % (df.loc[0, 'Product'], 
                                                       df.loc[0, 'NbClient'],
                                                       np.divide(df.loc[0, 'NbClient'],
                                                                 df.NbClient.sum())*100))

v.get_label_by_id('01').set_text('%s\n%d\n(%.0f%%)' % (df.loc[1, 'Product'],
                                                       df.loc[1, 'NbClient'],
                                                       np.divide(df.loc[1, 'NbClient'],
                                                                 df.NbClient.sum())*100))

v.get_label_by_id('11').set_text('%s\n%d\n(%.0f%%)' % (df.loc[2, 'Product'],
                                                       df.loc[2, 'NbClient'],
                                                       np.divide(df.loc[2, 'NbClient'],
                                                                 df.NbClient.sum())*100))
"""
for text in v.subset_labels: text.set_fontsize(10); text.set_weight('bold')

plt.show()
plt.style.use('default')