# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:33:06 2019

@author: jacqueline.cortez
Chapter 2: Creating and joining GeoDataFrames
    You'll work with GeoJSON to create polygonal plots, learn about projections and coordinate reference systems, 
    and get practice spatially joining data in this chapter.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import pandas            as pd                                                #For loading tabular data
import geopandas         as gpd                                               #For working with geospatial data
import matplotlib.pyplot as plt                                               #For creating charts

from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

pd.set_option("display.max_columns",20)

print("****************************************************")
topic = "User Variables"; print("** %s\n" % topic)

print("****************************************************")
topic = "Defined functions"; print("** %s\n" % topic)

print("****************************************************")
topic = "Reading data"; print("** %s\n" % topic)

filename = "schools.csv"
df_schools = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_schools.columns))

filename = "public_art.csv"
df_public_art = pd.read_csv(filename)
print("Columns of {}:\n{}\n".format(filename, df_public_art.columns))

print("****************************************************")
topic = "Reading geospatialdata"; print("** %s\n" % topic)

filename = "council_districts.geojson"
df_council_districts = gpd.read_file(filename)
print("Columns of {}:\n{}\n".format(filename, df_council_districts.columns))

filename = "school_districts.geojson"
df_school_districts = gpd.read_file(filename)
print("Columns of {}:\n{}\n".format(filename, df_school_districts.columns))

filename = "neighborhoods.geojson"
df_neighborhoods = gpd.read_file(filename)
print("Columns of {}:\n{}\n".format(filename, df_neighborhoods.columns))

print("****************************************************")
topic = "1. GeoJSON and plotting with geopandas"; print("** %s\n" % topic)

print('Head of df_council_districts dataset without geometry:\n{}'.format(df_council_districts[['first_name', 'email', 'res_phone', 'bus_phone', 'last_name', 'position', 'district']].head())) # Look at the first few rows of the chickens DataFrame

#Without legend
df_council_districts.plot(cmap='Set2')
plt.grid(True)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Council District (Without Legend)')
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot


#Without legend customization
df_council_districts.plot(column='district', legend=True)
plt.grid(True)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Council District (Without legend customization)')
plt.suptitle(topic)
#plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=None, hspace=None)
plt.show() # Show the plot


#With legend customization
legend_kwds = dict(title='number of observation',
                   loc='upper left',
                   fontsize=7,
                   markerscale=0.8,
                   bbox_to_anchor=(1, 1.03),
                   ncol=3)
df_council_districts.plot(column='district', cmap='Set3', legend=True, legend_kwds=legend_kwds)
plt.grid(True)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Council District (With legend customization)')
plt.suptitle(topic)
plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=None, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "3. Colormaps"; print("** %s\n" % topic)

# Set legend style
lgnd_kwds = {'title': 'School Districts', 'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 1}


# Plot the school districts using the tab20 colormap (qualitative)
df_school_districts.plot(column = 'district', cmap = 'tab20', legend = True, legend_kwds = lgnd_kwds)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Nashville School Districts (qualitative)')
plt.suptitle(topic)
plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=None, wspace=None, hspace=None)
plt.show() # Show the plot


# Plot the school districts using the summer colormap (sequential)
df_school_districts.plot(column = 'district', cmap = 'summer', legend = True, legend_kwds = lgnd_kwds)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Nashville School Districts (sequential)')
plt.suptitle(topic)
plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=None, wspace=None, hspace=None)
plt.show() # Show the plot


# Plot the school districts using Set3 colormap without the column argument
df_school_districts.plot(cmap = 'Set3', legend = True, legend_kwds = lgnd_kwds)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Nashville School Districts (without legend)')
plt.suptitle(topic)
#plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=None, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "4. Map Nashville neighborhoods"; print("** %s\n" % topic)

# Print the first few rows of neighborhoods
print(df_neighborhoods.head())

# Plot the neighborhoods, color according to name and use the Dark2 colormap
df_neighborhoods.plot(column = 'name', cmap = 'Dark2')
plt.title('Nashville neighborhoods')
plt.suptitle(topic)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
#plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=None, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "5. Projections and coordinate reference systems"; print("** %s\n" % topic)

df_schools['geometry'] = df_schools.apply(lambda x: Point((x.Longitude, x.Latitude)), axis=1)
print('Head of df_schools:\n{}'.format(df_schools.head())) # print the first few rows of df 

schools_crs = {'ini': 'epsg:4326'}

df_schools_geo = gpd.GeoDataFrame(df_schools, crs=schools_crs, geometry=df_schools.geometry)
print('Head of df_schools_geo (epsg:4326):\n{}\n'.format(df_schools_geo.head())) # print the first few rows of df 

"""
RuntimeError: b'no arguments in initialization list'
First step:
    Change the file C:\Anaconda3\Lib\site-packages\pyproj
    Rewrite the next line:
        import os; pyproj_datadir = os.environ.get('PROJ_LIB', '')
        to:
            import os; pyproj_datadir = os.environ.get('PROJ_LIB', 'C:/Anaconda3/Library/share')
    Be sure to use "/" instead of "\"
Second step:
    Initialize the GeoDataFrame:
        df_schools_geo.crs = {'init': 'epsg:3857'}
    and after make the change:
        df_schools_geo.geometry = df_schools_geo.geometry.to_crs(epsg=3857)
"""
df_schools_geo.crs = {'init': 'epsg:4326'}
df_schools_geo.geometry = df_schools_geo.geometry.to_crs(epsg=3857)
#df_schools_geo.geometry = df_schools_geo.geometry.to_crs({'init': 'epsg:3857'})
print('Head of df_schools_geo (epsg:3857):\n{}'.format(df_schools_geo.head())) # print the first few rows of df 

# Plot the neighborhoods, color according to name and use the Dark2 colormap
df_schools_geo.plot(column = 'City', cmap = 'Dark2')
plt.title('Schools in Nashville')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.suptitle(topic)
#plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=None, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "6. Changing coordinate reference systems"; print("** %s\n" % topic)

filename = "school_districts.geojson"
df_school_districts = gpd.read_file(filename)
print("Columns of {}:\n{}\n".format(filename, df_school_districts.columns))

# Print the first row of school districts GeoDataFrame and the crs
print("First row of school districts (CRS={}):\n{}\n\n\n".format(df_school_districts.crs['init'], df_school_districts.head(1)))

# Convert the crs to epsg:3857
df_school_districts.geometry = df_school_districts.geometry.to_crs(epsg = 3857)
                     
# Print the first row of school districts GeoDataFrame and the crs again
print("First row of school districts (CRS={}):\n{}\n".format(df_school_districts.crs['init'], df_school_districts.head(1)))

# Plot the school districts using the summer colormap (sequential)
df_school_districts.plot(column = 'district', cmap = 'summer', legend = True, legend_kwds = lgnd_kwds)
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.xticks(rotation=90)
plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Nashville School Districts (crs={})'.format(df_school_districts.crs['init']))
plt.suptitle(topic)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show the plot

print("****************************************************")
topic = "7. Construct a GeoDataFrame from a DataFrame"; print("** %s\n" % topic)

# Print the first few rows of the art DataFrame
print("Head of df_public_art:\n{}\n".format(df_public_art.head()))

# Create a geometry column from lng & lat
df_public_art['geometry'] = df_public_art.apply(lambda x: Point(float(x.Longitude), float(x.Latitude)), axis=1)

# Create a GeoDataFrame from art and verify the type
df_public_art_geo = gpd.GeoDataFrame(df_public_art, crs = df_neighborhoods.crs, geometry = df_public_art.geometry)
print("Type of df_public_art_geo: {}\n".format(type(df_public_art_geo)))

print("****************************************************")
topic = "8. Spatial joins"; print("** %s\n" % topic)

filename = "school_districts.geojson"
df_school_districts = gpd.read_file(filename)

# Plot df_council_districts
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=3, fontsize=8, title='Council Districts')
df_council_districts.plot(column = 'district', cmap = 'summer', legend = True, legend_kwds = legend_kwds)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Council districts'); plt.suptitle(topic); plt.subplots_adjust(left=0.1, bottom=None, right=0.65, top=None, wspace=None, hspace=None);
plt.show() # Show the plot

# Plot df_school_districts
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, fontsize=8, title='School Districts')
df_school_districts.plot(column = 'district', legend = True, legend_kwds = legend_kwds)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('School districts'); plt.suptitle(topic); plt.subplots_adjust(left=0.12, bottom=None, right=0.75, top=None, wspace=None, hspace=None);
plt.show() # Show the plot

print('Columns of df_council_districts: \n{}\n'.format(df_council_districts.columns))
print('Columns of df_school_districts: \n{}\n'.format(df_school_districts.columns))

print("CRS of df_council_districts:{}".format(df_council_districts.crs['init']))
print("CRS of df_school_districts:{}".format(df_school_districts.crs['init']))

###############################################################################
## Find council districts within school districts                            ##
###############################################################################
print('**The sjoin.() op argument - within\n')
df_within_gdf = gpd.sjoin(df_council_districts, df_school_districts, op='within')
print('Council districts within school districts: ', df_within_gdf.shape[0])

print('Head of df_within_gdf: {}'.format(df_within_gdf.head(5)))
print('Shape of df_within_gdf: {}'.format(df_within_gdf.shape))
print('Columns of df_within_gdf: \n{}'.format(df_within_gdf.columns))
print("CRS of df_within_gdf:{}\n".format(df_within_gdf.crs['init']))

# Plot df_within_gdf
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, fontsize=8, title='Council Districts Contained')
df_within_gdf.plot(column = 'district_left', cmap = 'summer', legend = True, legend_kwds = legend_kwds)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Finding council districts within school districts'); plt.suptitle(topic); plt.subplots_adjust(left=0.1, bottom=None, right=0.65, top=None, wspace=None, hspace=None);
plt.show() # Show the plot

# Aggregate council districts by school district to see how many council districts are within each school district.
# First rename district_left and district_right
df_within_gdf.rename(columns={'district_left':'council_district', 'district_right':'school_district'}, inplace=True)
df_council_within = df_within_gdf[['council_district','school_district']].groupby('school_district').agg('count').sort_values('council_district', ascending=False)
print("How many council districts are entirely within a school district?\n{}\n".format(df_council_within))

###############################################################################
## Find school districts that contains council districts                     ##
###############################################################################
print('**The sjoin.() op argument - contains\n')
df_contains_gdf = gpd.sjoin(df_school_districts, df_council_districts, op='contains')
print('School districts contains council districts: ', df_contains_gdf.shape[0])

print('Head of df_contains_gdf: {}'.format(df_contains_gdf.head(5)))
print('Shape of df_contains_gdf: {}'.format(df_contains_gdf.shape))
print('Columns of df_contains_gdf: \n{}'.format(df_contains_gdf.columns))
print("CRS of df_contains_gdf:{}\n".format(df_contains_gdf.crs['init']))

# Plot df_within_gdf
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, fontsize=8, title='School Districts Contained')
df_contains_gdf.plot(column = 'district_left', cmap = 'summer', legend = True, legend_kwds = legend_kwds)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Finding school districts that contains council districts'); plt.suptitle(topic); plt.subplots_adjust(left=0.1, bottom=None, right=0.65, top=None, wspace=None, hspace=None);
plt.show() # Show the plot

###############################################################################
## Find council districts that intersect with school districts               ##
###############################################################################
print('**The sjoin.() op argument - intersects\n')
df_intersects_gdf = gpd.sjoin(df_council_districts, df_school_districts, op='intersects')
print('Council districts intersects school districts: ', df_intersects_gdf.shape[0])

print('Head of df_within_gdf: {}'.format(df_intersects_gdf.head(5)))
print('Shape of df_within_gdf: {}'.format(df_intersects_gdf.shape))
print('Columns of df_within_gdf: \n{}'.format(df_intersects_gdf.columns))
print("CRS of df_within_gdf:{}\n".format(df_intersects_gdf.crs['init']))

# Plot df_within_gdf
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=3, fontsize=8, title='Council Districts Intersected')
df_intersects_gdf.plot(column = 'district_left', cmap = 'summer', legend = True, legend_kwds = legend_kwds)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Finding council districts that intersect school districts'); plt.suptitle(topic); plt.subplots_adjust(left=0.1, bottom=None, right=0.65, top=None, wspace=None, hspace=None);
plt.show() # Show the plot


print("****************************************************")
topic = "9. Spatial join practice"; print("** %s\n" % topic)

filename = "public_art.csv"; df_public_art = pd.read_csv(filename);
df_public_art['geometry'] = df_public_art.apply(lambda x: Point(float(x.Longitude), float(x.Latitude)), axis=1) # Create a geometry column from lng & lat
df_public_art_geo = gpd.GeoDataFrame(df_public_art, crs = df_neighborhoods.crs, geometry = df_public_art.geometry) # Create a GeoDataFrame from art and verify the type

print("CRS of df_public_art:{}".format(df_public_art_geo.crs['init']))
print("CRS of df_neighborhoods:{}\n".format(df_neighborhoods.crs['init']))

# Plot the art geo using Tab20 colormap
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, title_fontsize=9, fontsize=8, title='Art Type')
df_public_art_geo.plot(column = 'Type', cmap = 'tab20', legend = True, legend_kwds = legend_kwds)
plt.title('Nashville Art Locations'); plt.suptitle(topic); 
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude'); 
plt.subplots_adjust(left=0.12, bottom=None, right=0.75, top=None, wspace=None, hspace=None)
plt.show() # Show the plot

# Plot the neighborhoods, color according to name and use the Dark2 colormap
df_neighborhoods.plot(column = 'name', cmap = 'Dark2')
plt.title('Nashville neighborhoods'); plt.suptitle(topic);
plt.xticks(fontsize=8); plt.yticks(fontsize=8); plt.xlabel('Latitude'); plt.ylabel('Longitude'); 
plt.xlabel('Latitude'); plt.ylabel('Longitude'); #plt.subplots_adjust(left=0.1, bottom=None, right=0.5, top=None, wspace=None, hspace=None)
plt.show() # Show the plot


###############################################################################
## Spatially join art_geo and neighborhoods (op = 'intersects')              ##
###############################################################################
df_art_intersect_neighborhoods = gpd.sjoin(df_public_art_geo, df_neighborhoods, op = 'intersects')
print('***df_art_intersect_neighborhoods***')
print('Shape: {}'.format(df_art_intersect_neighborhoods.shape))
print('Columns: \n{}'.format(df_art_intersect_neighborhoods.columns))
print("CRS:{}".format(df_art_intersect_neighborhoods.crs['init']))
print('Head:\n{}\n'.format(df_art_intersect_neighborhoods.head(2)))
"""
# Plot df_within_gdf
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, title_fontsize=9, fontsize=8, title='Art Type intersected')
df_art_intersect_neighborhoods.plot(column = 'name', cmap = 'tab20', legend = True, legend_kwds = legend_kwds)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Finding art locations that intersects neighborhoods'); plt.suptitle(topic); plt.subplots_adjust(left=0.1, bottom=None, right=0.65, top=None, wspace=None, hspace=None);
plt.show() # Show the plot
"""
# Aggregate council districts by school district to see how many council districts are within each school district.
df_art_intersect_neighborhoods.rename(columns={'Type':'art_type', 'name':'neighborhoods'}, inplace=True)
df_intersects = df_art_intersect_neighborhoods[['art_type','neighborhoods']].groupby('neighborhoods').agg('count').sort_values('art_type', ascending=False)
print("How many art locations intersects neighborhoods? {} files found\n{}\n".format(df_intersects.shape[0], df_intersects))

# Plot df_within_gdf
legend_kwds = dict(loc='best', ncol=1, title_fontsize=8, markerscale=0.7, fontsize=7, title='Art Type intersected')
ax = df_neighborhoods[df_neighborhoods.name.isin(df_art_intersect_neighborhoods.neighborhoods)].plot(column = 'name', cmap = 'Set3', alpha=0.8)
df_art_intersect_neighborhoods.plot(ax = ax, column = 'art_type', legend = True, legend_kwds = legend_kwds, edgecolor='white'); # Add a plot of the urban_art and show it
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Finding art locations that intersects neighborhoods'); plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.65, top=None, wspace=None, hspace=None);
plt.show() # Show the plot


###############################################################################
# Create art_within_neighborhoods by spatially joining art_geo and           ##
# neighborhoods                                                              ##
###############################################################################
df_art_within_neighborhoods = gpd.sjoin(df_public_art_geo, df_neighborhoods, op = 'within')
print('***df_art_within_neighborhoods***')
print('Shape: {}'.format(df_art_within_neighborhoods.shape))
print('Columns: \n{}'.format(df_art_within_neighborhoods.columns))
print("CRS:{}\n".format(df_art_within_neighborhoods.crs['init']))
"""
# Plot df_within_gdf
legend_kwds = dict(loc='upper left', bbox_to_anchor=(1, 1.03), ncol=1, title_fontsize=9, fontsize=8, title='Art Locations within\nneighborhoods')
df_art_within_neighborhoods.plot(column = 'name', cmap = 'tab20', legend = True, legend_kwds = legend_kwds)
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Finding art locations within neighborhoods'); plt.suptitle(topic); plt.subplots_adjust(left=0.1, bottom=None, right=0.65, top=None, wspace=None, hspace=None);
plt.show() # Show the plot
"""
# Aggregate council districts by school district to see how many council districts are within each school district.
df_art_within_neighborhoods.rename(columns={'Type':'art_type', 'name':'neighborhoods'}, inplace=True)
df_within = df_art_within_neighborhoods[['art_type','neighborhoods']].groupby('neighborhoods').agg('count').sort_values('art_type', ascending=False)
print("How many art locations are within neighborhoods? {} files found\n{}\n".format(df_within.shape[0], df_within))

# Plot df_within_gdf
legend_kwds = dict(loc='best', ncol=1, title_fontsize=8, markerscale=0.7, fontsize=7, title='Art Type intersected')
ax = df_neighborhoods[df_neighborhoods.name.isin(df_art_within_neighborhoods.neighborhoods)].plot(column = 'name', cmap = 'Set3', alpha=0.8)
df_art_within_neighborhoods.plot(ax = ax, column = 'art_type', legend = True, legend_kwds = legend_kwds, edgecolor='white'); # Add a plot of the urban_art and show it
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Finding art locations within neighborhoods'); plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.65, top=None, wspace=None, hspace=None);
plt.show() # Show the plot


###############################################################################
# Spatially join art_geo and neighborhoods and using the contains op         ##
###############################################################################
df_art_containing_neighborhoods = gpd.sjoin(df_public_art_geo, df_neighborhoods, op = 'contains')
print('***df_art_containing_neighborhoods***')
print('Shape: {}'.format(df_art_containing_neighborhoods.shape))
print('Columns: \n{}'.format(df_art_containing_neighborhoods.columns))
print("CRS:{}\n".format(df_art_containing_neighborhoods.crs['init']))

# Aggregate council districts by school district to see how many council districts are within each school district.
df_art_containing_neighborhoods.rename(columns={'Type':'art_type', 'Mapped Location':'neighborhoods'}, inplace=True)
df_contains = df_art_containing_neighborhoods[['art_type','neighborhoods']].groupby('neighborhoods').agg('count').sort_values('art_type', ascending=False)
print("How many art locations contain neighborhoods? {} files found\n{}\n".format(df_contains.shape[0], df_contains))


print("****************************************************")
topic = "10. Finding the neighborhood with the most public art"; print("** %s\n" % topic)

# Print the first few rows
print('First 5 rows from df_art_within_neighborhoods:\n{}\n'.format(df_art_within_neighborhoods.head()))

print("****************************************************")
topic = "11. Aggregating points within polygons"; print("** %s\n" % topic)

df_neighborhood_art_grouped = df_art_within_neighborhoods[['neighborhoods', 'Title']].groupby('neighborhoods') # Get name and title from neighborhood_art and group by name
print(df_neighborhood_art_grouped.agg('count').sort_values(by = 'Title', ascending = False)) # Aggregate the grouped data and count the artworks within each polygon

print("****************************************************")
topic = "12. Plotting the Urban Residents neighborhood and art"; print("** %s\n" % topic)

df_urban_art = df_art_within_neighborhoods.loc[df_art_within_neighborhoods.neighborhoods == 'Urban Residents'] # Create urban_art from neighborhood_art where the neighborhood name is Urban Residents
df_urban_polygon = df_neighborhoods.loc[df_neighborhoods.name == "Urban Residents"] # Get just the Urban Residents neighborhood polygon and save it as urban_polygon

legend_kwds = dict(loc='best', ncol=1, markerscale=0.8, title_fontsize=9, fontsize=8, title='Art Type')
ax = df_urban_polygon.plot(color = 'lightgreen') # Plot the urban_polygon as ax 
df_urban_art.plot( ax = ax, column = 'art_type', legend = True, legend_kwds = legend_kwds); # Add a plot of the urban_art and show it
plt.xticks(fontsize=7); plt.yticks(fontsize=7); plt.xlabel('Latitude'); plt.ylabel('Longitude');
plt.title('Finding art locations within neighborhoods'); plt.suptitle(topic); #plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None);
plt.show()

print("****************************************************")
print("** END                                            **")
print("****************************************************")

#import inspect                                                                #Used to get the code inside a function
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt                                               #For creating charts
#import numpy             as np                                                #For making operations in lists
#import pandas            as pd                                                #For loading tabular data
#import seaborn           as sns                                               #For visualizing data
#import geopandas         as gpd                                               #For working with geospatial data 

#import calendar                                                               #For accesing to a vary of calendar operations
#import math                                                                   #For accesing to a complex math operations
#import nltk                                                                   #For working with text data
#import random                                                                 #For generating random numbers
#import re                                                                     #For regular expressions
#import tabula                                                                 #For extracting tables from pdf
#import timeit                                                                 #For Measure execution time of small code snippets
#import time                                                                   #To measure the elapsed wall-clock time between two points
#import scykit-learn                                                           #For performing machine learning  
#import warnings
#import wikipedia

#from collections                     import defaultdict                       #Returns a new dictionary-like object
#from datetime                        import date                              #For obteining today function
#from datetime                        import datetime                          #For obteining today function
#from functools                       import reduce                            #For accessing to a high order functions (functions or operators that return functions)
#from glob                            import glob                              #For using with pathnames matching
#from itertools                       import combinations                      #For iterations
#from itertools                       import cycle                             #Used in the function plot_labeled_decision_regions()
#from math                            import ceil                              #Used in the function plot_labeled_decision_regions()
#from math                            import floor                             #Used in the function plot_labeled_decision_regions()
#from math                            import radian                            #For accessing a specific math operations
#from mpl_toolkits.mplot3d            import Axes3D
#from pandas.api.types                import CategoricalDtype                  #For categorical data
#from pandas.plotting                 import parallel_coordinates              #For Parallel Coordinates
#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
#from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 
#from string                          import Template                          #For working with string, regular expressions


#from bokeh.io                        import curdoc                            #For interacting visualizations
#from bokeh.io                        import output_file                       #For interacting visualizations
#from bokeh.io                        import show                              #For interacting visualizations
#from bokeh.plotting                  import ColumnDataSource                  #For interacting visualizations
#from bokeh.plotting                  import figure                            #For interacting visualizations
#from bokeh.layouts                   import column                            #For interacting visualizations
#from bokeh.layouts                   import gridplot                          #For interacting visualizations
#from bokeh.layouts                   import row                               #For interacting visualizations
#from bokeh.layouts                   import widgetbox                         #For interacting visualizations
#from bokeh.models                    import Button                            #For interacting visualizations
#from bokeh.models                    import CategoricalColorMapper            #For interacting visualizations
#from bokeh.models                    import CheckboxGroup                     #For interacting visualizations
#from bokeh.models                    import ColumnDataSource                  #For interacting visualizations
#from bokeh.models                    import HoverTool                         #For interacting visualizations
#from bokeh.models                    import RadioGroup                        #For interacting visualizations
#from bokeh.models                    import Select                            #For interacting visualizations
#from bokeh.models                    import Slider                            #For interacting visualizations
#from bokeh.models                    import Toggle                            #For interacting visualizations
#from bokeh.models.widgets            import Panel                             #For interacting visualizations
#from bokeh.models.widgets            import Tabs                              #For interacting visualizations
#from bokeh.palettes                  import Spectral6                         #For interacting visualizations


#import keras                                                                  #For DeapLearning
#import keras.backend as k                                                     #For DeapLearning
#from keras.applications.resnet50     import decode_predictions                #For DeapLearning
#from keras.applications.resnet50     import preprocess_input                  #For DeapLearning
#from keras.applications.resnet50     import ResNet50                          #For DeapLearning
#from keras.callbacks                 import EarlyStopping                     #For DeapLearning
#from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
#from keras.datasets                  import fashion_mnist                     #For DeapLearning
#from keras.datasets                  import mnist                             #For DeapLearning
#from keras.layers                    import BatchNormalization                #For DeapLearning
#from keras.layers                    import Concatenate                       #For DeapLearning
#from keras.layers                    import Conv2D                            #For DeapLearning
#from keras.layers                    import Dense                             #For DeapLearning
#from keras.layers                    import Dropout                           #For DeapLearning
#from keras.layers                    import Embedding                         #For DeapLearning
#from keras.layers                    import Flatten                           #For DeapLearning
#from keras.layers                    import GlobalMaxPooling1D                #For DeapLearning
#from keras.layers                    import Input                             #For DeapLearning
#from keras.layers                    import LSTM                              #For DeapLearning
#from keras.layers                    import MaxPool2D                         #For DeapLearning
#from keras.layers                    import SpatialDropout1D                  #For DeapLearning
#from keras.layers                    import Subtract                          #For DeapLearning
#from keras.models                    import load_model                        #For DeapLearning
#from keras.models                    import Model                             #For DeapLearning
#from keras.models                    import Sequential                        #For DeapLearning
#from keras.optimizers                import Adam                              #For DeapLearning
#from keras.optimizers                import SGD                               #For DeapLearning
#from keras.preprocessing             import image                             #For DeapLearning
#from keras.preprocessing.text        import Tokenizer                         #For DeapLearning
#from keras.preprocessing.sequence    import pad_sequences                     #For DeapLearning
#from keras.utils                     import plot_model                        #For DeapLearning
#from keras.utils                     import to_categorical                    #For DeapLearning
#from keras.wrappers.scikit_learn     import KerasClassifier                   #For DeapLearning


#import networkx          as nx                                                #For Network Analysis in Python
#import nxviz             as nv                                                #For Network Analysis in Python
#from nxviz                           import ArcPlot                           #For Network Analysis in Python
#from nxviz                           import CircosPlot                        #For Network Analysis in Python 
#from nxviz                           import MatrixPlot                        #For Network Analysis in Python 


#import scipy.stats as stats                                                   #For accesign to a vary of statistics functiosn
#from scipy.cluster.hierarchy         import dendrogram                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import fcluster                          #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import linkage                           #For learning machine - unsurpervised
#from scipy.ndimage                   import gaussian_filter                   #For working with images
#from scipy.ndimage                   import median_filter                     #For working with images
#from scipy.signal                    import convolve2d                        #For learning machine - deep learning
#from scipy.sparse                    import csr_matrix                        #For learning machine 
#from scipy.special                   import expit as sigmoid                  #For learning machine 
#from scipy.stats                     import pearsonr                          #For learning machine 
#from scipy.stats                     import randint                           #For learning machine 
       

#from skimage                         import exposure                          #For working with images
#from skimage                         import measure                           #For working with images
#from skimage.filters.thresholding    import threshold_otsu                    #For working with images
#from skimage.filters.thresholding    import threshold_local                   #For working with images 


#from sklearn                         import datasets                          #For learning machine
#from sklearn.cluster                 import KMeans                            #For learning machine - unsurpervised
#from sklearn.decomposition           import NMF                               #For learning machine - unsurpervised
#from sklearn.decomposition           import PCA                               #For learning machine - unsurpervised
#from sklearn.decomposition           import TruncatedSVD                      #For learning machine - unsurpervised
#from sklearn.ensemble                import AdaBoostClassifier                #For learning machine - surpervised
#from sklearn.ensemble                import BaggingClassifier                 #For learning machine - surpervised
#from sklearn.ensemble                import GradientBoostingRegressor         #For learning machine - surpervised
#from sklearn.ensemble                import RandomForestClassifier            #For learning machine
#from sklearn.ensemble                import RandomForestRegressor             #For learning machine - unsurpervised
#from sklearn.ensemble                import VotingClassifier                  #For learning machine - unsurpervised
#from sklearn.feature_selection       import chi2                              #For learning machine
#from sklearn.feature_selection       import SelectKBest                       #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                   #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                 #For learning machine
#from sklearn.feature_extraction.text import TfidfVectorizer                   #For learning machine - unsurpervised
#from sklearn.impute                  import SimpleImputer                     #For learning machine
#from sklearn.linear_model            import ElasticNet                        #For learning machine
#from sklearn.linear_model            import Lasso                             #For learning machine
#from sklearn.linear_model            import LinearRegression                  #For learning machine
#from sklearn.linear_model            import LogisticRegression                #For learning machine
#from sklearn.linear_model            import Ridge                             #For learning machine
#from sklearn.manifold                import TSNE                              #For learning machine - unsurpervised
#from sklearn.metrics                 import accuracy_score                    #For learning machine
#from sklearn.metrics                 import classification_report             #For learning machine
#from sklearn.metrics                 import confusion_matrix                  #For learning machine
#from sklearn.metrics                 import mean_squared_error as MSE         #For learning machine
#from sklearn.metrics                 import roc_auc_score                     #For learning machine
#from sklearn.metrics                 import roc_curve                         #For learning machine
#from sklearn.model_selection         import cross_val_score                   #For learning machine
#from sklearn.model_selection         import GridSearchCV                      #For learning machine
#from sklearn.model_selection         import KFold                             #For learning machine
#from sklearn.model_selection         import RandomizedSearchCV                #For learning machine
#from sklearn.model_selection         import train_test_split                  #For learning machine
#from sklearn.multiclass              import OneVsRestClassifier               #For learning machine
#from sklearn.neighbors               import KNeighborsClassifier as KNN       #For learning machine
#from sklearn.pipeline                import FeatureUnion                      #For learning machine
#from sklearn.pipeline                import make_pipeline                     #For learning machine - unsurpervised
#from sklearn.pipeline                import Pipeline                          #For learning machine
#from sklearn.preprocessing           import FunctionTransformer               #For learning machine
#from sklearn.preprocessing           import Imputer                           #For learning machine
#from sklearn.preprocessing           import MaxAbsScaler                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing           import Normalizer                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing           import normalize                         #For learning machine - unsurpervised
#from sklearn.preprocessing           import scale                             #For learning machine
#from sklearn.preprocessing           import StandardScaler                    #For learning machine
#from sklearn.svm                     import SVC                               #For learning machine
#from sklearn.tree                    import DecisionTreeClassifier            #For learning machine - supervised
#from sklearn.tree                    import DecisionTreeRegressor             #For learning machine - supervised


#import statsmodels             as sm                                          #For stimations in differents statistical models
#import statsmodels.api         as sm                                          #Make a prediction model
#import statsmodels.formula.api as smf                                         #Make a prediction model    

#import tensorflow              as tf                                          #For DeapLearning



# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#pd.set_option('display.max_rows', -1)                                         #Shows all rows

#register_matplotlib_converters()                                              #Require to explicitly register matplotlib converters.

#Setting images params
#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09
#plt.rcParams.update({'figure.max_open_warning': 0})                           #To solve the max images open
#plt.rcParams["axes.labelsize"] = 8                                            #Font
#plt.rc('xtick',labelsize=8)
#plt.rc('ytick',labelsize=6)
#plt.rcParams['figure.max_open_warning'] = 60                                  #params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 5), 'axes.labelsize': 'x-large', 'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
#plt.rcParams["legend.fontsize"] = 8
#plt.style.use('dark_background')
#plt.style.use('default')
#plt.xticks(fontsize=7); plt.yticks(fontsize=8);
#plt.xticks(rotation=45)

#from matplotlib.axes._axes import _log as matplotlib_axes_logger              #To avoid warnings
#matplotlib_axes_logger.setLevel('ERROR')
#matplotlib_axes_logger.setLevel(0)                                            #To restore default

#ax.tick_params(labelsize=6)                                                   #axis : {'x', 'y', 'both'}
#ax.tick_params(axis='x', rotation=45)                                         #Set rotation atributte

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.
#np.random.seed(SEED)

#tf.compat.v1.set_random_seed(SEED)                                            #Instead of tf.set_random_seed, because it is deprecated.

#sns.set(font_scale=0.8)                                                       #Font
#sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot
#sns.set(color_codes=True)                                                     #Habilita el uso de los codigos de color
#sns.set()                                                                     #Seaborn defult style
#sns.set_style(this_style)                                                     #['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
#sns.despine(left=True)                                                        #Remove the spines (all borders)
#sns.palettes.SEABORN_PALETTES                                                 #Despliega todas las paletas disponibles 
#sns.palplot(sns.color_palette())                                              #Display a palette
#sns.color_palette()                                                           #The current palette
#sns.set(style=”whitegrid”, palette=”pastel”, color_codes=True)
#sns.mpl.rc(“figure”, figsize=(10,6))

#warnings.filterwarnings('ignore', 'Objective did not converge*')              #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
#warnings.filterwarnings('default', 'Objective did not converge*')             #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


#Create categorical type data to use
#cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  ordered=True)
# Change the data type of 'rating' to category
#weather['rating'] = weather.rating.astype(cats)