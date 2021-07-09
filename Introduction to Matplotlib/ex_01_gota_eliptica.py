# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:28:41 2019

@author: jacqueline.cortez
"""

import matplotlib.pyplot as plt
import numpy as np

from math import pi
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import Axes3d

#########################################################
## GOTA ELIPTICA
#########################################################
#primer gráfico
teta = np.array([np.linspace(0, 2*pi, 35).tolist()])
phi  = np.array([np.linspace(0, 2*pi, 35).tolist()])
teta, phi = np.meshgrid(teta, phi)
a, b, d = 4, 4, 4

x = d + (a*np.cos(teta))
y = ((b * np.sin(teta)) + (d/2 * np.sin(2*teta))) * np.cos(phi)
z = ((b * np.sin(teta)) + (d/2 * np.sin(2*teta))) * np.sin(phi)
#z = np.array([np.linspace(0, 0, 35).tolist()])

fig = plt.figure(figsize=plt.figaspect(1)) #set up a figure twice as it is tall
#fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, linewidth=0.3)
ax.set_title('[a={}, b={}, d={}]'.format(a,b,d))
#ax.view_init(60,35) #Define the viewing angle
plt.show()


#segundo gráfico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = fig.gca(projection='3d')
#ax = plt.axes(projection='3d')
teta = np.linspace(0, 2*pi, 35)
phi  = np.linspace(0, 2*pi, 35)

teta, phi = np.meshgrid(teta, phi)

a = 4
b = 4
d = 4

x = d + (a*np.cos(teta))
y = ((b * np.sin(teta)) + (d/2 * np.sin(2*teta))) * np.cos(phi)
z = ((b * np.sin(teta)) + (d/2 * np.sin(2*teta))) * np.sin(phi)

#ax.plot_surface(x, y, z, alpha=0.3, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.contour3D(x, y, z, 20, cmap='binary')
plt.draw()
plt.show()

#########################################################
## FUNCIÓN SEN
#########################################################
x = np.linspace(-1*4*pi, 4*pi, 1000)
y = np.sin(x)

fig, ax = plt.subplots()
plt.plot(x,y)
#plt.plot([1,2],[4,6])
plt.show()