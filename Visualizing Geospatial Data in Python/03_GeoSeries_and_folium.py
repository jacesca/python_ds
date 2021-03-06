# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:26:44 2019

@author: jacqueline.cortez
Chapter 3: GeoSeries and folium
    First you will learn to get information about the geometries in your data with three different GeoSeries attributes 
    and methods. Then you will learn to create a street map layer using folium.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import folium                                                                 #To create map street
import pandas            as pd                                                #For loading tabular data
import geopandas         as gpd                                               #For working with geospatial data
import matplotlib.pyplot as plt                                               #For creating charts
import pprint                                                                 #Import pprint to format disctionary output

import os
#import tempfile 
import webbrowser 

from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

pd.set_option("display.max_columns",60)
pd.options.display.float_format = '{:,.4f}'.format

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

def displaying_folium_map(location, filename, zoom_start, child=None, main_marker=True):
    folium_map = folium.Map(location=location, zoom_start=zoom_start)
    if main_marker:
        folium.Marker(location, popup="Here!!!").add_to(folium_map)
    if child != None:
        for item in child:
            item.add_to(folium_map)
    folium_map.save(filename)

    cwd = os.getcwd()
    path=cwd+'\\'+filename
    webbrowser.open('file://' + path)

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = 'schools.csv'
df_schools = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_schools.columns))

filename = "public_art.csv"
df_public_art = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_public_art.columns))

print("****************************************************")
topic = "Reading geospatialdata"; print("** %s\n" % topic)

filename = "school_districts.geojson"
df_school_districts = gpd.read_file(filename)
print("Columns of {}:\n{}\n".format(filename, df_school_districts.columns))

filename = "shapefile_path\\nashville.shp"
df_service_district = gpd.read_file(filename)
print("Columns of df_service_district:\n{}\n".format(df_service_district.columns))

filename = "neighborhoods.geojson"
df_neighborhoods = gpd.read_file(filename)
print("Columns of {}:\n{}\n".format(filename, df_neighborhoods.columns))

print("****************************************************")
topic = "1. GeoSeries attributes and methods I"; print("** %s\n" % topic)

print("Type of geometry column in df_school_districts: {}".format(type(df_school_districts.geometry))) #The geometry column is a GeoSeries
print("Area of first polygon in df_school_districts: {}\n\n".format(df_school_districts.geometry[0].area)) #The area of the firs polygon in df_school_districts

#Plot the first polygon
first_polygon = df_school_districts.head(1).copy()
first_polygon.plot()
plt.xlabel('Longitude'); plt.ylabel('Latitude'); 
plt.title('Distrinct No.{0}\nTotal area: {1:,.4f} decimal degrees squared'.format(first_polygon.district[0], df_school_districts.geometry[0].area)); plt.suptitle(topic);
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot

#calculate the area of each school district
df_school_districts_area = df_school_districts.copy()
df_school_districts_area['area'] = df_school_districts_area.geometry.area
print("Coordinate reference system: {}".format(df_school_districts_area.crs))
print("df_school_districts_area: \n{}\n\n".format(df_school_districts_area[['last_name', 'area', 'geometry']].sort_values(by='area', ascending=False)))

#Finding the area in kilometers squared
df_school_districts_3857 = df_school_districts.copy()
gpd.GeoDataFrame.crs = {'init': 'epsg:4326'}
df_school_districts_3857.geometry = df_school_districts_3857.geometry.to_crs(epsg=3857)

sqm_to_sqkm = 10**6
df_school_districts_3857['area'] = df_school_districts_3857.geometry.area / sqm_to_sqkm
print("Coordinate reference system: {}".format(df_school_districts_3857.crs))
print("df_school_districts_3857: \n{}\n\n".format(df_school_districts_3857[['last_name', 'area', 'geometry']].sort_values(by='area', ascending=False)))

print("****************************************************")
topic = "2. Find the area of the Urban Residents neighborhood"; print("** %s\n" % topic)

df_urban_polygon = df_service_district.query('name == "Urban Services District"').copy()
print("Rows in df_urban_polygon:\n{}\n\n".format(df_urban_polygon)) # Print the head of the urban polygon 

# Create a copy of the urban_polygon using EPSG:3857 and print the head
gpd.GeoDataFrame.crs = {'init': 'epsg:4326'}
df_urban_poly_3857 = df_urban_polygon.to_crs(epsg = 3857)
print("Rows in df_urban_poly_3857:\n{}\n\n".format(df_urban_poly_3857)) # Print the head of the urban polygon 

# Print the area of urban_poly_3857 in kilometers squared
area = df_urban_poly_3857.geometry.area / 10**6
print('The area of the Urban Residents neighborhood is {0:,.4f} km squared.\n'.format(area[0]))

#Plot the Urban Residents Neighborhood
df_urban_poly_3857.plot(color='green')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); 
plt.title('Urban Services District\nTotal area: {0:,.4f} km\u00b2'.format(area[0])); plt.suptitle(topic);
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "3. GeoSeries attributes and methods II"; print("** %s\n" % topic)

################################################
### Finding the distances between two points
################################################
first_polygon = df_school_districts.head(1).copy()                             # Getting the first polygon.
first_polygon_centroid = first_polygon.geometry.centroid                       # Calculating the center.

x = -86.8; y = 36.35; black_point = gpd.GeoSeries(Point(x, y));                # Defining the second point.
black_point.crs = {'init': 'epsg:4326', 'no_defs': True};                      # Defining the coord system of the second point.

gpd.GeoDataFrame.crs        = {'init': 'epsg:4326'}                            # Making the conversion.
first_polygon_3857          = first_polygon.to_crs(epsg = 3857)                
first_polygon_centroid_3857 = first_polygon_centroid.to_crs(epsg = 3857)
black_point_3857            = black_point.to_crs(epsg = 3857)

area = first_polygon_3857.geometry.area[0] / 10**6                             # Getting the area in kilometers squared.
distances = black_point_3857.distance(other =  first_polygon_centroid_3857)[0]    #Getting the distance in kilometers.

#Plotting the respective graph 
title_of_graph = 'Distrinct No.{0} \nTotal area: {1:,.0f} km\u00b2. \nDistances between two points: {2:,.0f} km.'.format(first_polygon.district[0], area, distances)
ax = first_polygon.plot(alpha=0.4, cmap='Pastel1')
first_polygon_centroid.plot(ax=ax, color='Red', label='Center')  
black_point.plot(ax=ax, color='darkgreen', label='Reference')
ax.annotate('From here', xy=(x-0.01, y), xytext=(-86.99, 36.37), arrowprops=dict(color='darkgreen'), color='darkgreen')
plt.legend(loc='lower right', markerscale=0.9, fontsize=9)
plt.xlabel('Longitude'); plt.ylabel('Latitude'); 
plt.title(title_of_graph); plt.suptitle(topic);
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot


#####################################################
###Finding the distances between schools and center
#####################################################
df_district_one = df_school_districts.loc[df_school_districts.district == '1']                         # Getting the school district 1.
df_schools['geometry'] = df_schools.apply(lambda x: Point((x.Longitude, x.Latitude)), axis=1)          # Calculating the geometry column.
df_schools_geo = gpd.GeoDataFrame(df_schools, crs=df_district_one.crs, geometry=df_schools.geometry)   # Transforming to new GeoDataFrame.
df_schools_geo = gpd.sjoin(df_schools_geo, df_district_one, op='within')                               # Finding schools in district one.

#Plot the schools in District 1 
ax = df_district_one.plot(alpha=0.4, cmap='Pastel1')                           # Plotting the district 1.
df_schools_geo.plot(ax=ax, cmap='Dark2')                                       # Plotting the schools in district 1.
plt.xlabel('Longitude'); plt.ylabel('Latitude');                               # Labeling the axis.
plt.title('Schools in District No.1'); plt.suptitle(topic);                    # Setting the titles.
#plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.8, wspace=None, hspace=None)
plt.show()                                                                     # Show the plot

gpd.GeoDataFrame.crs = {'init': 'epsg:4326'}
df_district_one_3857 = df_district_one.to_crs(epsg = 3857)
df_schools_geo_3857 = df_schools_geo[['School Name', 'geometry']].to_crs(epsg = 3857)

center = df_district_one_3857.geometry[0].centroid
distances = dict((row['School Name'].ljust(50, ' '), round(row.geometry.distance(other=center))) for index, row in df_schools_geo_3857.iterrows())   #Padding the name of the school
#distances = sorted(distances.items(), key=lambda x:x[1], reverse=True)        #Tuplas order by values.
pprint.pprint(distances)

print("****************************************************")
topic = "4. The center of the Urban Residents neighborhood"; print("** %s\n" % topic)

downtown_center = df_urban_poly_3857.geometry.centroid # Create downtown_center from urban_poly_3857
print(type(downtown_center)) # Print the type of downtown_center 

# Plot the urban_poly_3857 as ax and add the center point
ax = df_urban_poly_3857.plot(color = 'lightgreen')
downtown_center.plot(ax = ax, color = 'black')
plt.xticks(rotation = 45); plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Longitude'); plt.ylabel('Latitude');
plt.title('Urban Residents neighborhood'); plt.suptitle(topic);
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "5. Prepare to calculate distances"; print("** %s\n" % topic)

# Getting back the urban residents art
df_urban_art_polygon = df_neighborhoods.loc[df_neighborhoods.name == "Urban Residents"] # Get just the Urban Residents neighborhood polygon and save it as urban_polygon
gpd.GeoDataFrame.crs = {'init': 'epsg:4326'}
df_urban_art_polygon_3857 = df_urban_art_polygon.to_crs(epsg = 3857)

# Creating the geoDataFrame from art.
df_public_art['geometry'] = df_public_art.apply(lambda x: Point(float(x.Longitude), float(x.Latitude)), axis=1) # Create a geometry column from lng & lat
df_public_art_geo = gpd.GeoDataFrame(df_public_art, geometry = df_public_art.geometry, crs = {'init': 'epsg:4326'}) # Create df_public_art using df_public_art and the geometry from df_public_art
print('Head of df_public_art_geo ({}):\n{}\n'.format(df_public_art_geo.crs, df_public_art_geo.head(2)))

# Set the crs of art_dist_meters to use EPSG:3857
gpd.GeoDataFrame.crs = {'init': 'epsg:4326'}
df_art_dist_meters = df_public_art_geo.copy()
df_art_dist_meters.geometry = df_art_dist_meters.geometry.to_crs(epsg = 3857)
print('Head of df_art_dist_meters ({}):\n{}\n'.format(df_art_dist_meters.crs, df_art_dist_meters.head(2)))

# Add a column to art_meters, center
downtown_center = df_urban_art_polygon_3857.geometry.centroid.values[0]
df_art_dist_meters['center'] = downtown_center

print("****************************************************")
topic = "6. Art distances from neighborhood center"; print("** %s\n" % topic)

# Build a dictionary of titles and distances for Urban Residents art
art_distances = dict((vals['Title'].ljust(60, ' '), round(vals.geometry.distance(other=vals.center))) for index, vals in df_art_dist_meters.iterrows())   #Padding the name of the school
pprint.pprint(art_distances) # Pretty print the art_distances

#Plotting all art location inside and outside from Urban Residents area
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, title_fontsize=7, fontsize=7, title='Art Type')
ax = df_urban_art_polygon.plot(color = 'darkgreen') # Plot the urban_polygon as ax 
#ax.scatter(x=df_public_art_geo.Longitude, y=df_public_art_geo.Latitude, alpha=0.15)
df_public_art_geo.plot(ax = ax, column = 'Type', legend = True, legend_kwds = legend_kwds, alpha=0.25); # Add a plot of the urban_art and show it
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Art locations vs Urban Residents'); plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.subplots_adjust(left=0.15, bottom=None, right=0.75, top=None, wspace=None, hspace=None)
plt.show()

#Plotting only art locations in the urban area
df_urban_neighborhoods = gpd.sjoin(df_public_art_geo, df_urban_art_polygon, op = 'within')
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, markerscale=0.7, title_fontsize=7, fontsize=7, title='Art Type')
ax = df_urban_art_polygon.plot(color = 'lightgreen') # Plot the urban_polygon as ax 
df_urban_neighborhoods.plot(ax = ax, column = 'Type', legend = True, legend_kwds = legend_kwds); # Add a plot of the urban_art and show it
plt.xticks(fontsize=7); plt.yticks(fontsize=7); 
plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Art locations near Urban Residents'); plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.subplots_adjust(left=0.15, bottom=None, right=0.75, top=None, wspace=None, hspace=None)
plt.show()

print("****************************************************")
topic = "7. Street maps with folium"; print("** %s\n" % topic)

#Displaying the Eiffel Tower in Paris
displaying_folium_map([48.8583736, 2.2922926], '03_07_eiffel_tower.html', zoom_start=25)

#Displaying the London location
displaying_folium_map([51.507351, -0.127758], '03_07_london.html', zoom_start=15)

#Displaying El Salvador
displaying_folium_map([13.601749, -89.286807], '03_07_el_salvador.html', zoom_start=15)

#Displaying Taj Majal
displaying_folium_map([27.173891, 78.042068], '03_07_tajmahal.html', zoom_start=15)

#Displaying Europe
child = [folium.PolyLine([[43,7],[43,13],[47,13],[47,7],[43,7]], color='red').add_child(folium.Popup("outline Popup on Polyline")),
         folium.GeoJson({"type": "Polygon", "coordinates": [[[27,43],[33,43],[33,47],[27,47]]]}).add_child(folium.Popup("outline Popup on GeoJSON"))]
displaying_folium_map([45,0], '03_07_europe.html', zoom_start=5, child=child)

###The same example but without using the custom function to display the map
#europe = folium.Map([45,0], zoom_start=4)
#folium.Marker([45,0], popup="Here!!!").add_to(europe)
#ls = folium.PolyLine([[43,7],[43,13],[47,13],[47,7],[43,7]], color='red')
#ls.add_child(folium.Popup("outline Popup on Polyline"))
#ls.add_to(europe)
#gj = folium.GeoJson({"type": "Polygon", "coordinates": [[[27,43],[33,43],[33,47],[27,47]]]})
#gj.add_child(folium.Popup("outline Popup on GeoJSON"))
#gj.add_to(europe)
#filename='03_07_europe.html'; europe.save(filename);
#cwd = os.getcwd(); path=cwd+'\\'+filename; webbrowser.open('file://' + path)


#Giving context to the district one graph
center_point_district1 = df_district_one.geometry.centroid[0]
district1_center = [center_point_district1.y, center_point_district1.x] #Getting the latitude and the longitude
print(center_point_district1); print(district1_center);

displaying_folium_map(district1_center, '03_07_district1_map.html', zoom_start=10, child=[folium.GeoJson(df_district_one.geometry)])

print("****************************************************")
topic = "8. Create a folium location from the urban centroid"; print("** %s\n" % topic)

urban_center = df_urban_art_polygon.geometry.centroid.values[0] # Create urban_center from the urban_polygon center
urban_location = [urban_center.y, urban_center.x]               # Create array for folium called urban_location
print("Urban location: {}\n".format(urban_location))            # Print urban_location

print("****************************************************")
topic = "9. Create a folium map of downtown Nashville"; print("** %s\n" % topic)

displaying_folium_map(urban_location, '03_09_urban_nashville.html', zoom_start=15)

print("****************************************************")
topic = "10. Folium street map of the downtown neighborhood"; print("** %s\n" % topic)

child = [folium.GeoJson(df_urban_art_polygon.geometry)]
displaying_folium_map(urban_location, '03_10_urban_nashville.html', zoom_start=15, child=child)

print("****************************************************")
topic = "11. Creating markers and popups in folium"; print("** %s\n" % topic)

df_district_one         = df_school_districts.loc[df_school_districts.district == '1']                                    # Getting the school district 1.
df_schools['geometry']  = df_schools.apply(lambda x: Point((x.Longitude, x.Latitude)), axis=1)                            # Calculating the geometry column.
df_schools_geo          = gpd.GeoDataFrame(df_schools, crs=df_district_one.crs, geometry=df_schools.geometry)             # Transforming to new GeoDataFrame.
df_schools_in_district1 = gpd.sjoin(df_schools_geo, df_district_one, op='within')                                         # Finding schools in district one.
district1_center        = [df_district_one.geometry.centroid.values[0].y, df_district_one.geometry.centroid.values[0].x]  # Getting the coord to center the map.

child = [folium.GeoJson(df_district_one.geometry)]
#Create a marker for each school
for index, row in df_schools_in_district1.iterrows():
    location = [row['Latitude'], row['Longitude']]
    popup = '<strong>' + row['School Name'] +'</strong>'
    child.append(folium.Marker(location=location, popup=popup))
pprint.pprint(child)

displaying_folium_map(district1_center, '03_11_district_one.html', zoom_start=11, child=child, main_marker=False)

print("****************************************************")
topic = "12. Adding markers for the public art"; print("** %s\n" % topic)

child = [folium.GeoJson(df_urban_art_polygon.geometry)]
# Create a location and marker with each iteration for the downtown_map
for index, row in df_urban_neighborhoods.iterrows():
    location = [row['Latitude'], row['Longitude']]
    child.append(folium.Marker(location=location))
pprint.pprint(child)

displaying_folium_map(urban_location, '03_12_urban_nashville.html', zoom_start=15, child=child, main_marker=False)

print("****************************************************")
topic = "13. Troubleshooting data issues"; print("** %s\n" % topic)

print("Titles of art locations: \n{}\n".format(df_urban_neighborhoods.Title)) # Print the urban_art titles
print("Description of art locations: \n{}\n".format(df_urban_neighborhoods.Description)) # Print the urban_art descriptions

# Replace Nan and ' values in description
df_urban_neighborhoods.Description.fillna('', inplace = True)
df_urban_neighborhoods.Description = df_urban_neighborhoods.Description.str.replace("'", "`")
df_urban_neighborhoods.Description = df_urban_neighborhoods.Description.str.replace('"','')

print("Description of art locations (cleaned): \n{}\n".format(df_urban_neighborhoods.Description)) # Print the urban_art descriptions

print("****************************************************")
topic = "14. A map of downtown art"; print("** %s\n" % topic)

# Create a location and marker with each iteration for the downtown_map
child = [folium.Marker(location=[row['Latitude'], row['Longitude']], 
                       popup='<strong style="color:darkblue">' + row['Title'] +'</strong>') for index, row in df_urban_neighborhoods.iterrows()]
child.append(folium.GeoJson(df_urban_art_polygon.geometry))
pprint.pprint(child)
    
#child = [folium.GeoJson(df_urban_art_polygon.geometry)]
#for index, row in df_urban_neighborhoods.iterrows():
#    location = [row['Latitude'], row['Longitude']]
#    popup = '<strong style="color:darkblue">' + row['Title'] +'</strong>'
#    child.append(folium.Marker(location=location, popup=popup))
displaying_folium_map(urban_location, '03_14_urban_nashville.html', zoom_start=15, child=child, main_marker=False)

print("****************************************************")
print("** END                                            **")
print("****************************************************")
