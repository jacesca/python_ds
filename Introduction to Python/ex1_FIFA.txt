# -*- coding: utf-8 -*-
"""
Created on Tue May  7 05:40:28 2019

@author: jacqueline.cortez
"""

import pandas as pd
import numpy as np

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
# Read 'police.csv' into a DataFrame named ri
file = "data_fifa_player_list.csv"
fifa = pd.read_csv(file, parse_dates=True, 
                 index_col=0)

print(fifa.head())

print("****************************************************")
# heights and positions are available as lists
heights = np.loadtxt(file, skiprows=1, usecols=8, delimiter=",", dtype=np.int32, encoding="utf8")
positions = np.loadtxt(file, skiprows=1, usecols=4, delimiter=",", dtype=str, encoding="utf8")

# Define the position 
position_i = "GK"

# Heights of the interested position: gk_heights
gk_heights=heights[positions==position_i]

# Heights of the other players: other_heights
other_heights=heights[positions!=position_i]

# Print out the median height of the interested position. 
print("Median height of {}: {}".format(position_i,str(np.median(gk_heights))))

# Print out the median height of other players. 
print("Median height of other players: " + str(np.median(other_heights)))

print("****************************************************")
print("** END                                            **")
print("****************************************************")

