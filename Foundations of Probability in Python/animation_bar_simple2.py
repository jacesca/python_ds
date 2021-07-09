import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

fig = plt.figure()

position = np.arange(6) + .5 

plt.tick_params(axis = 'x', colors = '#072b57')
plt.tick_params(axis = 'y', colors = '#072b57')

speeds = [.01, .02, .03, .04, .01, .02]
heights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

rects = plt.bar(position, np.zeros_like(heights), align = 'center', 
                color=['red', 'orange', 'blue', 'pink', 'green','purple']) 
plt.xticks(position, ('Anger', 'Sadness', 'Disgust', 'Fear', 'Happy', 'Surprise'))

plt.xlabel('Emotion', color = '#072b57')
plt.ylabel('Probabilities', color = '#072b57')
plt.title('Emotion - Ally', color = '#072b57')

plt.ylim((0,1))
plt.xlim((0,6))

plt.grid(True)


frames = 200
min_speed = np.min(speeds)

def init():
    return rects

def animate(i):
    for h,r,s in zip(heights,rects, speeds):
        new_height = i / (frames-1) * h * s / min_speed
        new_height= min(new_height, h)
        r.set_height(new_height)
    return rects

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=frames, interval=20, blit=True, repeat=True)

plt.show()