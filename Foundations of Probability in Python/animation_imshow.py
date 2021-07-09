# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:57:41 2020

@author: jacesca@gmail.com
"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation

res_x, res_y = [1,2,3], [1,2,3]

fig = plt.figure()
ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2,2), (1,1))

rects = ax3.bar(res_x, res_y, color='b')
im1 = ax.imshow([[1,2],[2,3]], vmin=0)
im2 = ax2.imshow([[1,2],[2,3]], vmin=0)

def animate(i):

    im1.set_data([[1,2],[(i/100.),3]])
    im2.set_data([[(i/100.),2],[2.4,3]])

    for rect, yi in zip(rects, range(len(res_x))):
        rect.set_height((i/100.)*(yi+0.2))
    return [rect for rect in rects]+[im1, im2]

anim = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=True)
plt.show()