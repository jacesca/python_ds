# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:48:12 2020

@author: jacesca@gmail.com
"""


import matplotlib.pyplot as plt
from matplotlib import animation

b = 6 #num bars

def barlist(n): 
    return [n*k for k in range(1,b+1)] #[1/float(n*k) for k in range(1,6)]

fig=plt.figure()

n = 100 #Number of frames
x = range(1,b+1)

#barcollection = plt.bar(x,barlist(n))
barcollection = plt.barh(x,barlist(n))

def animate(i):
    y=barlist(i+1)
    for i, b in enumerate(barcollection):
        #b.set_height(y[i])
        b.set_width(y[i])
    #for i in range(len(barcollection)):
    #    barcollection[i].set_height(y[i])

anim=animation.FuncAnimation(fig,animate, repeat=True, blit=False, frames=n, interval=100)

#anim.save('mymovie.mp4',writer=animation.FFMpegWriter(fps=10))
plt.show()

