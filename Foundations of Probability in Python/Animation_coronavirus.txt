# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:02:17 2020

@author: jaces
Source:
    https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
    https://www.swharden.com/wp/2008-11-17-linear-data-smoothing-in-python/
"""
###############################################################################
##  Importing libraries.
###############################################################################
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.animation as animation

from matplotlib.animation import PillowWriter
from matplotlib.dates import DateFormatter

###############################################################################
##  Reading data.
###############################################################################
#LATAM = ['ARG','BOL','BRA','CHL','COL','CRI','CUB','DOM','ECU','GTM','HND','HTI','MEX','NIC','PAN','PER','PRY','SLV','URY','VEN']
#CA = ['CRI','GTM','HND','NIC','PAN','SLV']
#SLV = ['SLV']
#suptitle = 'Coronavirus Disease (COVID-19) – LATAM Statistics'
suptitle = 'Coronavirus Disease (COVID-19) – World Statistics'
    
file = "coronavirus.csv"
coronavirus = pd.read_csv(file, parse_dates=True, dayfirst=True, index_col= "dateRep").sort_index()
#coronavirus = coronavirus[coronavirus.countryterritoryCode.isin(LATAM)] #latam
#coronavirus = coronavirus[coronavirus.countryterritoryCode.isin(SLV)] #El Salvador
coronavirus = coronavirus.groupby('dateRep')[['cases','deaths']].sum().cumsum()
coronavirus = coronavirus[(coronavirus != 0).all(1)]

x = coronavirus.index.values
y = coronavirus.cases

title = y.name.upper()
y_max_lim = 5 if title=='DEATHS' else 400000 #All the world
y_max_lim = 35500 if title=='DEATHS' else 750000 #All the world withou log scale
#y_max_lim = 100 #if title=='DEATHS' else 16500 #LATAM

frames = len(y)+1
corona = pd.DataFrame(data=y, index=x)
corona.columns = {title}

###############################################################################
##  Preparing the figure.
###############################################################################
sns.set()
fig = plt.figure(figsize=(10,5))
plt.gca().xaxis_date()
plt.xlim(x.min(), x.max())
#plt.xlim(x.min().strftime("%Y-%m-%d"), x.max().strftime("%Y-%m-%d"))
#plt.xlim(['2019-12-31', '2020-04-01'])
plt.ylim([y.min(), y.max()])

"""#Log scale for y
plt.yscale('log')
exp = int(math.log10(max(y.values))) #To define the log scale, including the max value of y
majors = [10**i for i in range(exp+1)]
majors = [i for i in majors if i>y.min()]
plt.yticks([y.min()] + majors + [y.max()])

"""#Not log scale
majors, _ = plt.yticks()
majors = [i for i in majors if np.logical_and(i>y.min(),i<y.max())]
#majors = [i for i in majors if np.logical_and(i>y.min(),i<500)]
plt.yticks([y.min()] + majors + [y.max()])

plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in plt.gca().get_yticks()]) #To format y axis
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%b-%d'))
#plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.xlabel('DATE',fontsize=20)
plt.ylabel(title,fontsize=20)
plt.suptitle(suptitle, fontsize=18, color='darkred')
plt.title("Source: https://ourworldindata.org/coronavirus", color='darkblue')

###############################################################################
##  The animation function in which we define what happens in each frame of 
##  your video. 
###############################################################################
def animate(i):
    data = corona.iloc[:int(i+1)] #select data range --> rowa
    p = sns.lineplot(x=data.index, y=data[title], data=data, color="rosybrown") 
    p.tick_params(axis='y', labelsize=14) #fontsize of labels.
    plt.setp(p.lines, linewidth=7) #Set a property on an artist object.
    ###### First option: print values each 7 days.
    y_data = data.iloc[(i-1),0]
    x_date = data.index[-1]
    imprime = ((int(x_date.strftime("%d")) % 7) == 0)
    plt.text(x_date, y_data, ("{:,.0f}".format(y_data) if imprime else ''), weight='bold', fontsize=12)  
    ###### Second option: print values in the middle
    x_date = corona.index.mean()
    y_value = y_max_lim
    #y_data = data.iloc[(i-1),0]
    plt.text(x_date, y_value, "     {:,.0f}     ".format(y_data), ha='center', weight='bold', fontsize=17, backgroundcolor='rosybrown')

###############################################################################
##  To start the animation use matplotlib.animation.FuncAnimation in which 
##  you link the animation function and define how many frames your animation 
##  should contain. 
###############################################################################
#ani = mpl.animation.FuncAnimation(fig, animate, frames=frames, repeat=True, interval=100, repeat_delay=1000) 
ani = mpl.animation.FuncAnimation(fig, animate, frames=frames, repeat=False, interval=100) 
#ani = mpl.animation.FuncAnimation(fig, animate, frames=frames, repeat=False, interval=2000) #LATAM
plt.subplots_adjust(left=.15, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
plt.show()

###############################################################################
##  Next we initialize a writer to save the image.
###############################################################################
#writer = PillowWriter(fps=20, metadata=dict(artist='jacesca@gmail.com'), bitrate=1800) #For .gif. 
#ani.save('coronavirus.gif', writer=writer)
plt.style.use('default')