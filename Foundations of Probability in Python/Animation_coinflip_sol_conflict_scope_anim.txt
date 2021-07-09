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

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.dates import DateFormatter
from scipy.stats                     import binom

global anim2

def Make_Animation(n, SEED = 42):
    global anim2
    
    ###############################################################################
    ##  Reading data.
    ###############################################################################
    suptitle = '1. From sample mean to population mean'
    
    #n = 250  
    #SEED = 42
      
    x = [ii for ii in range(1,n+1)]
    y = [binom.rvs(n=1, p=0.5, size=ii, random_state=SEED).mean() for ii in x]
    
    flipcoin_sample = pd.DataFrame({'Size of sample': x, 'Sample mean': y})
        
    frames = len(y)+1
    
    ###############################################################################
    ##  Preparing the figure.
    ###############################################################################
    sns.set()
    fig = plt.figure(figsize=(10,5))
    #plt.xlim([np.min(x), np.max(x)])
    plt.ylim([np.min(y), np.max(y)])
    
    #plt.xlabel('Size of sample', fontsize=20)
    #plt.ylabel('Sample mean', fontsize=20)
    plt.suptitle(suptitle, fontsize=18, color='darkred')
    
    ###############################################################################
    ##  Construct the animation. 
    ###############################################################################
    def init():
        plt.gca().clear()
        #sns.set()
        plt.grid(True, color='white')
        
    def animate(i):
        data = flipcoin_sample.iloc[:int(i+1)] #select data range --> rowa
        p = sns.lineplot(x=data['Size of sample'], y=data['Sample mean'], data=data, color="darkblue") 
        p.tick_params(labelsize=14) #fontsize of labels.
        plt.setp(p.lines, linewidth=2) #Set a property on an artist object.
        plt.ylim([0, 1])
        plt.axhline(y=0.5, lw=2, color='red')
        plt.legend(['Sample mean', 'Population mean'], loc='upper right')
        plt.title('Coin Flip', fontsize='20', color='red') 
         
        plt.text(1, 0.9, " {:.5f} ".format(data.iloc[(i-1),1]), weight='bold', fontsize=17, color='whitesmoke', backgroundcolor='rosybrown')
    
    ###############################################################################
    ##  Start the animation. 
    ###############################################################################
    plt.subplots_adjust(left=.15, bottom=.15, right=None, top=.85, wspace=None, hspace=None);
    anim2 = FuncAnimation(fig, animate, init_func=init, frames=frames, repeat=True, interval=100) 
    plt.show()
    plt.style.use('default')
    
def main():
    Make_Animation(n=250)
        
if __name__ == '__main__':
    main()
    
    