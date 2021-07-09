# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:57:54 2020

@author: jacesca@gmail.com

how to get the index of an array in a random.choice???
"""


import random # plain random module, not numpy's

random.seed(SEED)
a = np.array([10,40,10,30,30,20,10,40])
b = random.sample(list(enumerate(a)),3)
b = np.array(b).reshape(3,2)

print("Sourc values: ", a)
print("Random selected: \n{}\n".format(b))
print("indices: ", b[:,0])
print("Values: ", b[:,1])