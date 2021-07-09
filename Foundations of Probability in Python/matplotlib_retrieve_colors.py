# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:41:12 2020

@author: jaces
To retrieve a list of color
"""
from matplotlib import cm, colors

cmap = cm.get_cmap('Accent', 5)    # PiYG

for i in range(cmap.N):
    rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
    print(colors.rgb2hex(rgb))
    
    
print([colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)])    


import seaborn as sns
colors = sns.color_palette("Accent")
print (colors.as_hex())