# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:15:35 2020

@author: jacesca@gmail.com
"""


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

Dict = {"Alice": 0, "Bob": 1, "Carol": 2}
cmap = plt.cm.Accent
"""
norm = mpl.colors.Normalize(vmin=0, vmax=2)
m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

for key, value in Dict.items():
    print(key, value, m.to_rgba(value))
plt.hlines(1, 1, 5, m.to_rgba(Dict["Alice"]), linewidth=50)
plt.hlines(2, 1, 5, m.to_rgba(Dict["Bob"]),   linewidth=50)
plt.hlines(3, 1, 5, m.to_rgba(Dict["Carol"]), linewidth=50)
"""
G = nx.Graph()
G.add_nodes_from(Dict.keys())

nodelist,node_color = zip(*Dict.items())
nx.draw_networkx(G, nodelist=nodelist, node_size=1000, node_color=node_color,vmin=0.0,vmax=2.0, cmap=cmap)
plt.show()