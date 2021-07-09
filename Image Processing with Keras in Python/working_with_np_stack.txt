# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:25:14 2019

@author: jacqueline.cortez
"""

import matplotlib.pyplot as plt                  #For creating charts
import numpy             as np                   #For making operations in lists
from scipy.signal        import convolve2d       #For learning machine - deep learning

file='test.jpg'
pic = plt.imread(file)
print(pic)
kernel = [[0,0,0],[0,1,0],[0,0,0]]

print("--------------")
print("--------------")
conv_bucket = []
for d in range(pic.ndim):
        conv_channel = convolve2d(pic[:,:,d], kernel,  mode="same", boundary="symm")
        print(conv_channel)
        conv_bucket.append(conv_channel)
        print(conv_bucket)
        print("--------------")
print("--------------")
print(np.stack(conv_bucket, axis=2).astype("uint8"))
