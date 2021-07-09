# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:31:39 2020

@author: jaces
"""

###############################################################################
##  Importing libraries.
###############################################################################
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from glob import glob

###############################################################################
##  Creating the function to read the data
###############################################################################
def read_data_file():
    # Get the data (csv file is hosted on the web)
    #url = 'https://python-graph-gallery.com/wp-content/uploads/gapminderData.csv'
    url = 'gapminderData.csv'
    data = pd.read_csv(url)
    
    # And I need to transform my categorical column (continent) in a numerical value group1->1, group2->2...
    data['continent']=pd.Categorical(data['continent'])
    print("Head of the file:\n{}\m".format(data.head()))
    print("Data readed. Task ended.\n\n")
    return data

###############################################################################
##  Defining the fucntion used to create the images.
###############################################################################
def create_individual_images(data, my_dpi=96):
    target = mpl.get_backend()
    mpl.use('Agg')
    sns.set_style("white")

    # For each year
    for i in data.year.unique():
        
        # initialize a figure
        plt.figure(figsize=(680/my_dpi, 480/my_dpi), dpi=my_dpi)
        
        # Change color with c and alpha. I map the color to the X axis value.
        tmp=data[ data.year == i ]
        plt.scatter(tmp['lifeExp'], tmp['gdpPercap'] , s=tmp['pop']/200000 , c=tmp['continent'].cat.codes, cmap="Accent", alpha=0.6, edgecolors="white", linewidth=2)
 
        # Add titles (main and on axis)
        plt.yscale('log')
        plt.xlabel("Life Expectancy")
        plt.ylabel("GDP per Capita")
        plt.title("Year: "+str(i) )
        plt.ylim(1,100000)
        plt.xlim(30, 90)
 
        # Save it
        filename='gapminder/Gapminder_step'+str(i)+'.png'
        plt.savefig(filename, dpi=96, transparent=True)
        plt.gca()
        print("Saving image \"{}\"...".format(filename))
        
        
    print("Creation of individual images ended.\n\n")
    plt.style.use('default')
    mpl.use(target) 

###############################################################################
##  Call the functions to read the data and create the images.
###############################################################################
df = read_data_file() #To read the data
create_individual_images(df) #To create the images

###############################################################################
##  Now, it's time to create the animation.
###############################################################################
#Get the file names of the images to add
filenames = glob("gapminder/Gapminder_step*.png")
file_animation = "Gapminder.gif"

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

