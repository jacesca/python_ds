# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:17:01 2020

@author: jaces

Objetcive of the program:
    The idea is to change the camera view and then use every resulting image 
    to create an animation. 
Source:
    https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
    https://python-graph-gallery.com/342-animation-on-3d-plot/
    https://python-graph-gallery.com/
Related:
    https://makersportal.com/blog/2018/7/27/how-to-make-a-gif-using-python-an-application-with-the-united-states-wind-turbine-database
"""

###############################################################################
##  Importing libraries.
###############################################################################
#import imageio
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from glob import glob
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
##  Creating the function to read the data
###############################################################################
def Read_volcano_data():
    # Get the data (csv file is hosted on the web)
    #url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
    url = 'volcano/volcano.csv'
    data = pd.read_csv(url)
    print("Data readed (head): \n{}\n".format(data.head())) #Comment this
    print("Data info: \n{}\n\n".format(data.info())) #Comment this
    
    # Transform it to a long format
    df=data.unstack().reset_index()
    print("Unstack data with reset_index (head): \n{}\n".format(df.head())) #Comment this
    print("Data info: \n{}\n\n".format(df.info())) #Comment this
    df.columns=["X","Y","Z"]

    # And transform the old column name in something numeric
    df['X']=pd.Categorical(df['X'])
    print("Categorical values: \n{}\n".format(df['X'].cat.codes.unique())) #Comment this
    print("Data info: \n{}\n\n".format(df.info())) #Comment this
    df['X']=df['X'].cat.codes
    print("Categorical codes data (head): \n{}\n".format(df.head())) #Comment this
    print("Data info: \n{}\n\n".format(df.info())) #Comment this
    print("Data readed. Task ended.")
    return df

###############################################################################
##  Defining the fucntion used to create the images.
###############################################################################
def Create_individual_images(df):
    # Force matplotlib to not use any Xwindows backend.
    target = mpl.get_backend()
    mpl.use('Agg')
    n_img = 0

    # We are going to do 20 plots, for 20 different angles
    image_to_create = len(range(70,210,2))
    plt.rcParams['figure.max_open_warning'] = image_to_create

    for angle in range(70,210,2):
        # Make the plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2) #To plot in 3 dimensions
        
        ax.view_init(30,angle)
        
        filename='volcano/volcano_step'+str(angle)+'.png'
        plt.savefig(filename, dpi=96, transparent=True)#, bbox_inches='tight'
        plt.gca()
        n_img+=1
        print("Saving image \"{}\"...".format(filename))

    print("Creation of individual images ended.\n\n\n\n")
    plt.style.use('default')
    mpl.use(target) 


###############################################################################
##  Call the functions to read the data and create the images.
###############################################################################

df = Read_volcano_data() #To read the data
Create_individual_images(df) #To create the images

###############################################################################
##  Now, it's time to create the animation.
###############################################################################
#Get the file names of the images to add
filenames = glob("volcano/volcano_step*.png")
file_animation = "3DPlot_volcano.gif"

images = []
frame_length = 0.2 # seconds between frames
end_pause = 4 # seconds to stay on last frame

# loop through files, join them to image array, and write to GIF called 'test.gif'
for file_image in filenames:       
    images.append(imageio.imread(file_image))
    print("Reading image \"{}\"...".format(file_image))
    
    #if file_image == filenames[-1]:
    #    for jj in range(0, int(end_pause/frame_length)):
    #        images.append(imageio.imread(file_image))
    #else:
    #    images.append(imageio.imread(file_image))

# the duration is the time spent on each image (1/duration is frame rate)
imageio.mimsave(file_animation, images,'GIF',duration=frame_length)
print("\nAnimation file \"{}\" created. \nEnd\n".format(file_animation))
plt.style.use('default')