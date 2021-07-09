# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:45:55 2020

@author: jaces
Source:
    https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
    https://www.swharden.com/wp/2008-11-17-linear-data-smoothing-in-python/
"""

###############################################################################
##  Lets go ahead and import all dependencies.
###############################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

###############################################################################
##  Now to get ready for animating the data we only need to load it and 
##  put it into a Pandas DataFrame.
###############################################################################
overdoses = pd.read_excel('overdose_data_1999-2015.xls', sheet_name='Online', skiprows =6)
def get_data(table,rownum,title):
    data = pd.DataFrame(table.loc[rownum][2:]).astype(float)
    data.columns = {title}
    return data

###############################################################################
##  In my case I now retrieve the heroin overdose numbers from the table 
##  using the get_data function and pack it into a Pandas DataFrame with 
##  two columns. One for the year and the other for the count of overdoses.
###############################################################################
title = 'Heroin Overdoses'
d = get_data(overdoses, 18, title) #Get data from 18th row and from 2nd column to the end
x = np.array(d.index)
y = np.array(d['Heroin Overdoses'])
overdose = pd.DataFrame(data=y, index=x)
#XN,YN = augment(x,y,10)
#augmented = pd.DataFrame(YN,XN)
overdose.columns = {title}

###############################################################################
##  Now lets create a figure with some labels. Make sure to set the limits 
##  for the x and y axis so your animation doesn’t jump around with the 
##  range of the data currently displayed.
###############################################################################
fig = plt.figure(figsize=(10,5))
plt.xlim(1999, 2016)
plt.ylim(np.min(overdose)[0], np.max(overdose)[0])
plt.xlabel('Year',fontsize=20)
plt.ylabel(title,fontsize=20)
plt.title('Heroin Overdoses per Year',fontsize=20)

###############################################################################
##  To avoid the jumpiness of it we need some more data points in between 
##  the ones we already have. For this we can use another function which I 
##  call here augment .
###############################################################################

def augment(xold,yold,numsteps):
    xnew = []
    ynew = []
    for i in range(len(xold)-1):
        difX = xold[i+1]-xold[i]
        stepsX = difX/numsteps
        difY = yold[i+1]-yold[i]
        stepsY = difY/numsteps
        for s in range(numsteps):
            xnew = np.append(xnew,xold[i]+s*stepsX)
            ynew = np.append(ynew,yold[i]+s*stepsY)
    return xnew,ynew

x1, y1 = augment(x, y, 10)
x = np.append(np.append(x[0],x1),x[-1])
y = np.append(np.append(x[0],y1),y[-1])
overdose = pd.DataFrame(data=y, index=x)
overdose.columns = {title}
overdose = overdose.loc[~overdose.index.duplicated(keep='last')] #Delete duplicated index

###############################################################################
##  To get rid of these we can implement a smoothing function as described 
##  here: https://www.swharden.com/wp/2008-11-17-linear-data-smoothing-in-python/
###############################################################################

def smoothListGaussian(listin,strippedXs=False,degree=5):  
    window=degree*2-1  
    weight=np.array([1.0]*window)  
    weightGauss=[]  
    for i in range(window):  
        i=i-degree+1  
        frac=i/float(window)  
        gauss=1/(np.exp((4*(frac))**2))  
        weightGauss.append(gauss)
    weight=np.array(weightGauss)*weight  
    smoothed=[0.0]*(len(listin)-window)  
    for i in range(len(smoothed)):        smoothed[i]=sum(np.array(listin[i:i+window])*weight)/sum(weight)  
    return smoothed
x1 = smoothListGaussian(x, degree=5); y1 = smoothListGaussian(y)
x = np.append(np.append(x[0],x1),x[-1])
y = np.append(np.append(x[0],y1),y[-1])
overdose = pd.DataFrame(data=y, index=x)
overdose.columns = {title}


###############################################################################
##  The heart piece of your animation is your animation function in which 
##  you define what happens in each frame of your video. Here i represents 
##  the index of the frame in the animation. With this index you can select 
##  the data range which should be visible in this frame. After doing that 
##  I use a seaborn lineplot to plot this data selection. The last two lines 
##  are just to make the plot look a bit more pleasing.
###############################################################################
def animate(i):
    data = overdose.iloc[:int(i+1)] #select data range --> rowa
    p = sns.lineplot(x=data.index, y=data[title], data=data, color="r") 
    p.tick_params(labelsize=17) #fontsize of labels.
    plt.setp(p.lines, linewidth=7) #Set a property on an artist object. 

###############################################################################
##  To start the animation use matplotlib.animation.FuncAnimation in which 
##  you link the animation function and define how many frames your animation 
##  should contain. frames therefore defines how often animate(i) 
##  is being called.
###############################################################################
#ani = matplotlib.animation.FuncAnimation(fig, animate, frames=17, repeat=True, interval=800, repeat_delay=800) #just animate function.
#ani = matplotlib.animation.FuncAnimation(fig, animate, frames=162, repeat=True, interval=10, repeat_delay=400) #animate and augmented function.
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=155, repeat=True, interval=10, repeat_delay=100) #include smoothListGaussian function. Frames=overdose.shape[0]
plt.subplots_adjust(left=.15, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
plt.show()

###############################################################################
##  Next we initialize a writer which uses ffmpeg and records at 20 fps with 
##  a bitrate of 1800. You can of course pick these values yourself.
###############################################################################
#Writer = animation.writers['ffmpeg'] #for .mp4
#writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800) #for .mp4
writer = PillowWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800) #For .gif. 

###############################################################################
##  To save this animation as an mp4 you can simply call ani.save() . 
##  If you just want to take a look at it before you save it call 
##  plt.show() instead.
###############################################################################
#ani.save('HeroinOverdosesJumpy.mp4', writer=writer)
ani.save('animation_HeroinOverdosesJumpy.gif', writer=writer)
plt.style.use('default')

