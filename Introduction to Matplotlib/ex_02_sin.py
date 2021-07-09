# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:26:54 2019

@author: jacqueline.cortez
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi

#x = np.linspace(-1*4*pi, 4*pi, 1000)
x = np.linspace(-pi/2, pi, 1000)
y = np.sin(x)

fig, ax = plt.subplots()
plt.plot(x,y)

z=np.repeat(0,1000)
plt.plot(x,z)
plt.plot(z,y)

w=np.repeat(pi/2,1000)
plt.plot(w,y)

plt.text(-1.5,0.1,"Y=0")
plt.text(0.1,-1,"X=0")
plt.text(pi/2+0.1,-1,"X=pi/2")
plt.xlabel("X")
plt.ylabel("Y=g(x)")

#plt.plot([1,2],[4,6])
plt.show()