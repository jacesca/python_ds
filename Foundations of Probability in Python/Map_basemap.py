# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:37:41 2020

@author: jacesca@gmail.com
Source:
    https://python-graph-gallery.com/map/
    https://python-graph-gallery.com/281-basic-map-with-basemap/
    https://python-graph-gallery.com/connection-map/
    https://python-graph-gallery.com/300-draw-a-connection-line/
    https://python-graph-gallery.com/bubble-map/
    https://python-graph-gallery.com/315-a-world-map-of-surf-tweets/
Error Fix:
    To avoido message:
        MatplotlibDeprecationWarning: 
        The dedent function was deprecated in Matplotlib 3.1 and will be removed in 3.3. Use inspect.cleandoc instead.
        m=Basemap()
    Steps:
        1. Go to file "C:\Anaconda3\pkgs\basemap-1.2.0-py37h4e5d7af_0\Lib\site-packages\mpl_toolkits\basemap\__init__.py" 
        2. Look for code line:
            26: from matplotlib.cbook import dedent
        3. Replace it with:
            from inspect import cleandoc #as dedent
        4. Replace all the incidence of dedent replace for cleandoc
        5. Save
        6. Go to file "C:\Anaconda3\pkgs\basemap-1.2.0-py37h4e5d7af_0\Lib\site-packages\mpl_toolkits\basemap\proj.py"
        7. Look for code line:
            6: from matplotlib.cbook import dedent
        8. Replace it with:
            from inspect import cleandoc as dedent
        9. Replace all the incidence of dedent replace for cleandoc
       10. Save
"""
###############################################################################
##                                    I N I T I A L   C O N F I G U R A T I O N
###############################################################################
# Hack to fix missing PROJ4 env var
import os
os.environ["PROJ_LIB"] = "C:\\Anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share" #Look for epsg file and set the path to environment variable PROJ_LIB

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 60

###############################################################################
##                                                            L I B R A R I E S
###############################################################################
import pandas as pd
from mpl_toolkits.basemap import Basemap


###############################################################################
##                                *** GRAPH No.281 - BASIC MAP WITH BASEMAP ***
###############################################################################
## Here is the most basic map you can do with the basemap library of python. 
## It allows to understand the basic use of this library. Always start by 
## initialising the map with the Basemap() function. Then, add the elements 
## your needs, like continents, coastlines… Since this basemap is not really 
## gorgeous, I propose you to custom it easily.
###############################################################################
# Always start witht the basemap function to initialize a map
m = Basemap()

# Then add element: draw coast line, map boundary, and fill continents:
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents()
 
# You can add rivers as well
#m.drawrivers(color='#0000ff')

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Basic map with basemap", color='darkblue')
plt.suptitle('GRAPH No.281', fontsize=18, color='darkred')
plt.show()
print("Graph No.281 - Basic map with basemap PRINTED...")


###############################################################################
##                                         *** GRAPH No.282 - CUSTOM COLORS ***
###############################################################################
## The graph #281 shows how to draw a basic map with basemap. Here, I show how 
## to improve its appearance. Each element that you draw on the map (boundary, 
## continent, coast lines…) can be customised. Moreover, note that it is 
## interesting to control the limits of the map, but we will learn more about 
## this here.
###############################################################################
# initialise the map
plt.figure()
m=Basemap(llcrnrlon=-180, llcrnrlat=-60,urcrnrlon=180,urcrnrlat=70)
 
# Control the background color
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
 
# Fill the continent
m.fillcontinents(color='grey', alpha=0.7, lake_color='blue')
 
# Draw the coastline
m.drawcoastlines(linewidth=2.1, color="white")
 
# to save if needed
plt.savefig('map_basemap/282_Custom_Basemap.png', dpi=110, bbox_inches='tight')
 
# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Customizing colors", color='darkblue')
plt.suptitle('GRAPH No.282', fontsize=18, color='darkred')
plt.show()
print("Graph No.282 - Customizing colors PRINTED...")


###############################################################################
##                           *** GRAPH No.283 - SET BOUNDING BOX IN BASEMAP ***
###############################################################################
## Charts #281 and #282 show how to make a world map with desired colors. But 
## what if you want to look at a specific region of the globe?
## When you initialise the map, you can determine a specific region with 4 
## arguments. You have to provide the coordinates of 2 opposite corners of 
## the zone.
###############################################################################
plt.figure()
# Control the position of the square. Give the coordinate of 2 corners
m=Basemap(llcrnrlon=-100, llcrnrlat=-58,urcrnrlon=-30,urcrnrlat=15)
# looking for El Salvador
#m=Basemap(llcrnrlon=(-100), llcrnrlat=(0),urcrnrlon=(-30),urcrnrlat=(15))
 
# Draw the components of the map
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='brown', alpha=0.6, lake_color='grey')
m.drawcoastlines(linewidth=0.1, color="white")
plt.show()

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Set bounding box in basemap", color='darkblue')
plt.suptitle('GRAPH No.283', fontsize=18, color='darkred')
plt.show()
print("Graph No.283 - Customizing colors PRINTED...")


###############################################################################
##                       *** GRAPH No.284 DIFFERENT TYPES OF MAP PROJECTION ***
###############################################################################
## The earth being a sphere, it has always been tricky to represent it in 2 
## dimensions. Several projections exist, all with pro and con, and a few of 
## them are implemented in matplotlib. Pick the one you need!
###############################################################################
## ----------------------------------------------------------> ORTHO PROJECTION
# ortho
plt.figure()
m=Basemap(lat_0=0, lon_0=0, projection='ortho' )
m.drawmapboundary(fill_color='#A6CAE0')
m.fillcontinents(color='darkgreen', alpha=0.3)

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Different types of map projection - [ortho]", color='darkblue')
plt.suptitle('GRAPH No.284', fontsize=18, color='darkred')
plt.show()
print("Graph No.284 - Different types of map projection - [ortho] PRINTED...")


## ----------------------------------------------------------> ORTHO PROJECTION
plt.figure()
m=Basemap(llcrnrlon=-180, llcrnrlat=-60,urcrnrlon=180,urcrnrlat=80, projection='merc')
m.drawmapboundary(fill_color='#A6CAE0')
m.fillcontinents(color='darkgreen', alpha=0.3)

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Different types of map projection - [merc]", color='darkblue')
plt.suptitle('GRAPH No.284', fontsize=18, color='darkred')
plt.show()
print("Graph No.284 - Different types of map projection - [merc] PRINTED...")


## ----------------------------------------------------------> ROBIN PROJECTION
plt.figure()
m=Basemap(lat_0=0, lon_0=0, projection='robin' )
m.drawmapboundary(fill_color='#A6CAE0')
m.fillcontinents(color='darkgreen', alpha=0.3)

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Different types of map projection - [robin]", color='darkblue')
plt.suptitle('GRAPH No.284', fontsize=18, color='darkred')
plt.show()
print("Graph No.284 - Different types of map projection - [robin] PRINTED...")


## ----------------------------------------------------------> AEQD PROJECTION
plt.figure()
#aeqd --> you HAVE to provide lon_0 and lat_0
m=Basemap(lat_0=30, lon_0=30, projection='aeqd' )
m.drawmapboundary(fill_color='#A6CAE0')
m.fillcontinents(color='grey', alpha=0.3)

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Different types of map projection - [aeqd]", color='darkblue')
plt.suptitle('GRAPH No.284', fontsize=18, color='darkred')
plt.show()
print("Graph No.284 - Different types of map projection - [aeqd] PRINTED...")


## ----------------------------------------------------------> NSPER PROJECTION
plt.figure()
m=Basemap(lat_0=0, lon_0=0, projection='nsper' )
m.drawmapboundary(fill_color='#A6CAE0')
m.fillcontinents(color='grey', alpha=0.3)

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Different types of map projection - [nsper]", color='darkblue')
plt.suptitle('GRAPH No.284', fontsize=18, color='darkred')
plt.show()
print("Graph No.284 - Different types of map projection - [nsper] PRINTED...")


## ------------------------------------------------------------> CYL PROJECTION
plt.figure()
m=Basemap(llcrnrlon=-180, llcrnrlat=-60,urcrnrlon=180,urcrnrlat=80, projection='cyl' )
m.drawmapboundary(fill_color='#A6CAE0')
m.fillcontinents(color='grey', alpha=0.3)

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Different types of map projection - [cyl]", color='darkblue')
plt.suptitle('GRAPH No.284', fontsize=18, color='darkred')
plt.show()
print("Graph No.284 - Different types of map projection - [cyl] PRINTED...")


###############################################################################
##                                  *** GRAPH No.285 USE A BACKGROUND LAYER ***
###############################################################################
## Instead of colouring your map with uniform colors, you can use layer for 
## your background. Here are a few examples showing the layers that are 
## available using python and basemap.
###############################################################################
## -----------------------------------------------------> BLUEMARBEL BACKGROUND
plt.figure()
m = Basemap(llcrnrlon=-10.5,llcrnrlat=33,urcrnrlon=10.,urcrnrlat=46., resolution='i', projection='cass', lat_0 = 39.5, lon_0 = 0.)
m.bluemarble()

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Use a background layer - [Bluemarbel]", color='darkblue')
plt.suptitle('GRAPH No.285', fontsize=18, color='darkred')
plt.show()
print("Graph No.285 - Use a background layer - [Bluemarbel] PRINTED...")


## ---------------------------------------------------> SHADEDRELIEF BACKGROUND
plt.figure()
m = Basemap(llcrnrlon=-10.5,llcrnrlat=33,urcrnrlon=10.,urcrnrlat=46., resolution='i', projection='cass', lat_0 = 39.5, lon_0 = 0.)
m.shadedrelief()

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Use a background layer - [Shadedrelief]", color='darkblue')
plt.suptitle('GRAPH No.285', fontsize=18, color='darkred')
plt.show()
print("Graph No.285 - Use a background layer - [Shadedrelief] PRINTED...")


## ----------------------------------------------------------> ETOPO BACKGROUND
plt.figure()
m = Basemap(llcrnrlon=-10.5,llcrnrlat=33,urcrnrlon=10.,urcrnrlat=46., resolution='i')
m.etopo()

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Use a background layer - [Etopo]", color='darkblue')
plt.suptitle('GRAPH No.285', fontsize=18, color='darkred')
plt.show()
print("Graph No.285 - Use a background layer - [Etopo] PRINTED...")


## -----------------------------------> ARCGIS BACKGROUND - WORLD SHADED RELIEF
plt.figure()
m = Basemap(projection='mill',llcrnrlon=-123. ,llcrnrlat=37,urcrnrlon=-121 ,urcrnrlat=39, resolution = 'l', epsg = 4326)
m.arcgisimage(service='World_Shaded_Relief', xpixels = 1500, verbose= False)

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Use a background layer - [Arcgis World Shaded Relief]", color='darkblue')
plt.suptitle('GRAPH No.285', fontsize=18, color='darkred')
plt.show()
print("Graph No.285 - Use a background layer - [Arcgis World Shaded Relief] PRINTED...")


## -----------------------------------------> ARCGIS BACKGROUND - OCEAN BASEMAP
plt.figure()
# Ocean Basemap
m = Basemap(projection='mill',llcrnrlon=-123. ,llcrnrlat=37,urcrnrlon=-121 ,urcrnrlat=39, resolution = 'l', epsg = 4326)
m.arcgisimage(service='Ocean_Basemap', xpixels = 1500, verbose= False)

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Use a background layer - [Arcgis Ocean Basemap]", color='darkblue')
plt.suptitle('GRAPH No.285', fontsize=18, color='darkred')
plt.show()
print("Graph No.285 - Use a background layer - [Arcgis Ocean Basemap] PRINTED...")


###############################################################################
##                          *** GRAPH No.286 BOUNDARIES PROVIDED IN BASEMAP ***
###############################################################################
## The basemap library (closely linked with matplotlib) contains a database 
## with several boundaries. Thus, it is easy to represent the limits of 
## countries, states and counties, without having to load a shape file. Here 
## is how to show these 3 types of boundaries: countries, states and counties.
## Counties doesn't work. give a msg-->"UnicodeDecodeError: 'utf-8' codec can't 
## decode byte 0xf1 in position 2: invalid continuation byte
##          map.drawcounties()
###############################################################################
## -----------------------------------------------------> COUNTRYIES BOUNDARIES
plt.figure()
# Initialize the map
map = Basemap(llcrnrlon=-160, llcrnrlat=-60,urcrnrlon=160,urcrnrlat=70)
 
# Continent and countries!
map.drawmapboundary(fill_color='#A6CAE0')
map.fillcontinents(color='#e6b800',lake_color='#e6b800')
map.drawcountries(color="white")


# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Boundaries provided in Basemap - [Countries Boundaries]", color='darkblue')
plt.suptitle('GRAPH No.286', fontsize=18, color='darkred')
plt.show()
print("Graph No.286 - Boundaries provided in Basemap - [Countries Boundaries] PRINTED...")


## ---------------------------------------------------------> STATES BOUNDARIES
plt.figure()
# initialise
map = Basemap(llcrnrlon=-130, llcrnrlat=25, urcrnrlon=-65.,urcrnrlat=52.,resolution='i', lat_0 = 40., lon_0 = -80)
 
# map states
map.drawmapboundary(fill_color='#A6CAE0')
map.fillcontinents(color='#e6b800',lake_color='#A6CAE0')
map.drawstates()
map.drawcountries()


# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Boundaries provided in Basemap - [States Boundaries]", color='darkblue')
plt.suptitle('GRAPH No.286', fontsize=18, color='darkred')
plt.show()
print("Graph No.286 - Boundaries provided in Basemap - [States Boundaries] PRINTED...")


###############################################################################
##                *** GRAPH No.300 DRAW A CONNECTION LINE WITH GREAT CIRCLE ***
###############################################################################
## This page describe how to add a connection line between 2 places on a map 
## with python and the basemap library. Here we represent the connection 
## between New York and London. Note that the line is not straight: it is 
## indeed the shortest route between these 2 cities, taking into account that 
## the earth is a sphere. We call it a great circle, and it can be drawn with 
## the drawgreatcircle function:
###############################################################################
plt.figure()
# A basic map
m=Basemap(llcrnrlon=-100, llcrnrlat=20,urcrnrlon=30,urcrnrlat=70)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')
m.drawcoastlines(linewidth=0.1, color="white")
 
# Add a connection between new york and London
NYlat = 40.78; NYlon = -73.98
Londonlat = 51.53; Londonlon = 0.08
m.drawgreatcircle(NYlon,NYlat,Londonlon,Londonlat, linewidth=2, color='orange')

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Draw a connection line with great circle", color='darkblue')
plt.suptitle('GRAPH No.300', fontsize=18, color='darkred')
plt.show()
print("Graph No.300 - Draw a connection line with great circle...")


###############################################################################
##                                  *** GRAPH No.310 BASIC MAP WITH MARKERS ***
###############################################################################
## During 300 days, I harvested every tweet containing the hashtags #surf, 
## #kitesurf and #windsurf. Here is a map showing the localisation of these 
## tweets. This projects is explained more in detail in the blog of the R graph 
## gallery. This map is done using the Basemap library of python. See more 
## example in the dedicated section of the Python Graph Gallery. 
###############################################################################
# Make a data frame with the GPS of a few cities:
data = pd.DataFrame({
    'lat':[-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],
    'lon':[-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],
    'name':['Buenos Aires', 'Paris', 'melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador']
})
 
# A basic map
plt.figure()
m = Basemap(llcrnrlon=-160, llcrnrlat=-75,urcrnrlon=160,urcrnrlat=80)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')
m.drawcoastlines(linewidth=0.1, color="white")
 
# Add a marker per city of the data frame!
m.plot(data['lat'], data['lon'], linestyle='none', marker="o", markersize=16, alpha=0.6, c="orange", markeredgecolor="black", markeredgewidth=1)

# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("Basic map with markers", color='darkblue')
plt.suptitle('GRAPH No.310', fontsize=18, color='darkred')
plt.show()
print("Graph No.310 - Basic map with markers...")


###############################################################################
##                             *** GRAPH No.315 A WORLD MAP OF #SURF TWEETS ***
###############################################################################
## During 300 days, I harvested every tweet containing the hashtags #surf, 
## #kitesurf and #windsurf. Here is a map showing the localisation of these 
## tweets. This projects is explained more in detail in the blog of the R graph 
## gallery. This map is done using the Basemap library of python. See more 
## example in the dedicated section of the Python Graph Gallery. 
###############################################################################
# Set the dimension of the figure
my_dpi=96
plt.figure(figsize=(11, 5.5), dpi=my_dpi)
 
# read the data (on the web)
#data = pd.read_csv('http://python-graph-gallery.com/wp-content/uploads/TweetSurfData.csv', sep=";")
data = pd.read_csv('map_basemap/TweetSurfData.csv', sep=";")
 
# Make the background map
m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color="white")
 
# prepare a color for each point depending on the continent.
data['labels_enc'] = pd.factorize(data['homecontinent'])[0]
 
# Add a point per position
m.scatter(data['homelon'], data['homelat'], s=data['n']/6, alpha=0.4, c=data['labels_enc'], cmap="Set1")
 
# copyright and source data info
plt.text( -170, -58,'Where people talk about #Surf\n\nData collected on twitter by @R_Graph_Gallery during 300 days\nPlot realized with Python and the Basemap library', ha='left', va='bottom', size=9, color='#555555' )
 
# Save as png
plt.savefig('map_basemap/315_Tweet_Surf_Bubble_map1.png', bbox_inches='tight')


# Show
plt.xlabel('Longitud',fontsize=8)
plt.ylabel('Latitude',fontsize=8)
plt.title("A world map of #surf tweets", color='darkblue')
plt.suptitle('GRAPH No.315', fontsize=18, color='darkred')
plt.show()
print("Graph No.315 - A world map of #surf tweets...")


###############################################################################
##                                          R E S T O R E   T O   D E F A U L T
###############################################################################
plt.style.use('default')
warnings.filterwarnings('default', category=matplotlib.cbook.mplDeprecation)
print("END.")