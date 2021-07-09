# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:32:02 2019

@author: jacqueline.cortez
"""

import geopandas         as gpd                                               #For working with geospatial data
"""
In [1]: gpd.show_versions()

SYSTEM INFO
-----------
python     : 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
executable : C:\Anaconda3\pythonw.exe
machine    : Windows-10-10.0.15063-SP0

GEOS, GDAL, PROJ INFO
---------------------
GEOS       : None
GEOS lib   : None
GDAL       : 2.3.3
GDAL data dir: None
PROJ       : 0.5.20
PROJ data dir: C:/Anaconda3\share\proj

PYTHON DEPENDENCIES
-------------------
geopandas  : 0.6.0
pandas     : 0.25.1
fiona      : 1.8.4
numpy      : 1.16.5
shapely    : 1.6.4.post1
rtree      : 0.8.3
pyproj     : 1.9.6
matplotlib : 3.1.1
mapclassify: None
pysal      : None
geopy      : 1.20.0
psycopg2   : None
"""


filename = "paris_districts_utm.geojson"
df_paris_districts_geo = gpd.read_file(filename)
print(df_paris_districts_geo.crs)

#gpd.GeoDataFrame.crs = {'init': 'epsg:32631'}
df_paris_districts_geo.crs = {'init': 'epsg:32631'}
print(df_paris_districts_geo.crs)
#df_paris_districts_geo.crs = {'init': 'epsg:32631'}
#df_paris_districts_geo.crs = {'init': 'epsg:32631'}
#gpd.GeoDataFrame.crs = {'init': 'epsg:32631'}
df_paris_districts_geo = df_paris_districts_geo.to_crs(epsg = '3857')
