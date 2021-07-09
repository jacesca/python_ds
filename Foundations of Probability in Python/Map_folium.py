# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:39:31 2020

@author: jaces
Maps availables:
    https://leaflet-extras.github.io/leaflet-providers/preview/
    https://python-graph-gallery.com/288-map-background-with-folium/
    https://python-visualization.github.io/folium/modules.html
    https://python-graph-gallery.com/choropleth-map/
    https://python-graph-gallery.com/292-choropleth-map-with-folium/
Documentation:    
    https://python-visualization.github.io/folium/modules.html
"""

# import the library
import folium
import pandas as pd
import os
 
###############################################################################
##                              *** GRAPH No.288 MAP BACKGROUND WITH FOLIUM ***
###############################################################################
## Folium is a python library allowing to call the Leaflet.js Javascript 
## library. It allows you to manipulate your data with python and map them 
## using the power of leaflet! It is really easy to call a map using this 
## library. Your first need to install it using
##                        conda install -c conda-forge folium
## Then, use the folium.Map function to initialise a map. Several tiles are 
## available: OpenStreetMap, Stamen Terrain, Stamen Toner, Mapbox Bright, 
## and Mapbox Control Room. Note that the tile you need probably depends on 
## the zoom you want to apply to your map. Here are a few example, with 2 
## different degree of zoom.
###############################################################################
Coord_Paris = [48.85, 2.35]
Coord_ElSalvador = [13.601749, -89.286807]
Coord_init = Coord_ElSalvador

# Make an empty map
#m = folium.Map(location=[20, 0], zoom_start=3.5)
 
# Other tiles:
# OpenStreetMap, Stamen Terrain, Stamen Toner, Mapbox Bright, and Mapbox Control Room
m = folium.Map(location=Coord_init, tiles="Stamen Toner", zoom_start=2)
m.save('map_folium/No288_folium_stamen_tower.html')

m = folium.Map(location=Coord_init, tiles="OpenStreetMap", zoom_start=2)
m.save('map_folium/No288_folium_open_streetmap.html')
 

# Same but with a zoom
m = folium.Map(location=Coord_init, tiles="Stamen Toner", zoom_start=10)
m.save('map_folium/No288_folium_zoom_stamen_tower.html')

m = folium.Map(location=Coord_init, tiles="Stamen Terrain", zoom_start=10)
m.save('map_folium/No288_folium_zoom_stamen_terrain.html')

m = folium.Map(location=Coord_init, tiles="OpenStreetMap", zoom_start=10)
m.save('map_folium/No288_folium_zoom_open_streetmap.html')

m = folium.Map(location=Coord_init, tiles="Stamen Watercolor", zoom_start=10)
m.save('map_folium/No288_folium_zoom_stamen_watercolor.html')

m = folium.Map(location=Coord_init, tiles="CartoDB positron ", zoom_start=10)
m.save('map_folium/No288_folium_zoom_CartoDB_positron.html')

m = folium.Map(location=Coord_init, tiles="CartoDB dark_matter", zoom_start=10)
m.save('map_folium/No288_folium_zoom_CartoDB_dark_matter.html')

print("8 html documents saved. Graph No.288 - Map background with folium.")


###############################################################################
##                              *** GRAPH No.292 CHOROPLETH MAP WITH FOLIUM ***
###############################################################################
## Here is an example of a choropleth map made using the Folium library. 
## This example comes directly from the (awesome) documentation of this 
## library. Note that you need 2 elements to build a chloropleth map. 
## i - A shape file in the geojson format: it gives the boundaries of every zone 
##     that you want to represent. 
## ii- A data frame that gives the values of each zone. You can file the 2 
##     files used to build this example: 
##     https://github.com/python-visualization/folium/tree/master/examples/data
##############################################################################
# Load the shape of the zone (US states)
# You have to download this file and set the directory where you saved it
state_geo = os.path.join('map_folium/', 'us-states.json')
 
# Load the unemployment value of each state
state_unemployment = os.path.join('map_folium/', 'US_Unemployment_Oct2012.csv')
state_data = pd.read_csv(state_unemployment)
 
# Initialize the map:
m = folium.Map(location=[37, -102], zoom_start=5)
 
# Add the color for the chloropleth:
folium.Choropleth(geo_data=state_geo, 
                  name='choropleth',
                  data=state_data,
                  columns=['State', 'Unemployment'],
                  key_on='feature.id', #From the json file.
                  fill_color='YlGn',
                  fill_opacity=0.7,
                  line_opacity=0.2,
                  legend_name='Unemployment Rate (%)').add_to(m)
folium.LayerControl().add_to(m)
 
# Save to html
m.save('map_folium/No292_folium_chloropleth_USA1.html')

print("1 html documents saved. Graph No.292 - Choropleth map with folium.")


###############################################################################
##                               *** GRAPH No.312 ADD MARKERS ON FOLIUM MAP ***
###############################################################################
## This page describes how to add markers to your folium map. It is done using 
## the folium.Marker function. Note that you can custom the popup window of 
## each marker with any html code! 
###############################################################################
# Make a data frame with dots to show on the map
data = pd.DataFrame({
'lat':[-89.286807, -58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],
'lon':[13.601749, -34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],
'name':['San Salvador, CA', 'Buenos Aires', 'Paris', 'Melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador, Br']
})

# Make an empty map
m = folium.Map(location=[20, 0], zoom_start=2)
 
# I can add marker one by one on the map
folium.LayerControl().add_to(m)
for i in range(0,len(data)):
    folium.Marker(location = [data.iloc[i]['lon'], data.iloc[i]['lat']], 
                  popup=data.iloc[i]['name']).add_to(m)

# Save it as html
m.save('map_folium/No312_markers_on_folium_map1.html')

print("1 html documents saved. Graph No.312 - Add markers on folium map.")


###############################################################################
##                                  *** GRAPH No.313 BUBBLE MAP WITH FOLIUM ***
###############################################################################
## This page describes how to add bubbles to your folium map. It is done using 
## the folium.Circle function. Each bubble has a size related to a specific 
## value. Note that if you zoom on the map, the circle will get bigger. If you 
## want the circle to stay the same size whatever the zoom, you have to use the 
## folium.CircleMarker (exactly the same arguments!). 
###############################################################################
# Make a data frame with dots to show on the map
data = pd.DataFrame({
   'lat'  :[-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],
   'lon'  :[-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],
   'name' :['Buenos Aires', 'Paris', 'melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador'],
   'value':[10., 12, 40, 70, 23, 43, 100, 43]
})


## -------------------------------------------------------------> FOLIUM.CIRCLE
# Make an empty map
m = folium.Map(location=[20,0], zoom_start=2)
 
# I can add marker one by one on the map
folium.LayerControl().add_to(m)
for i in range(0, len(data)):
    folium.Circle(
        location=[data.iloc[i]['lon'], data.iloc[i]['lat']],
        popup=data.iloc[i]['name'],
        radius=data.iloc[i]['value']*10000,
        color='crimson',
        fill=True,
        fill_color='crimson'
    ).add_to(m)
 
# Save it as html
m.save('map_folium/No313_Circle.html')


## -------------------------------------------------------> FOLIUM.CIRCLEMARKER
# Make an empty map
m = folium.Map(location=[20,0], zoom_start=2)
 
# I can add marker one by one on the map
folium.LayerControl().add_to(m)
for i in range(0, len(data)):
    folium.CircleMarker(
        location=[data.iloc[i]['lon'], data.iloc[i]['lat']],
        popup=data.iloc[i]['name'],
        radius=data.iloc[i]['value'],
        #color='crimson',
        fill=True,
        #fill_color='crimson'
    ).add_to(m)
 
# Save it as html
m.save('map_folium/No313_Circle_Marker.html')


print("2 html documents saved. Graph No.313 - Bubble map with Folium.")


###############################################################################
##                                                                  *** END ***
###############################################################################
print("END.")
