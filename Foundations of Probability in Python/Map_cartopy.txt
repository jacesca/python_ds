# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:11:11 2020

@author: jacesca@gmail.com

Source:
    https://scitools.org.uk/cartopy/docs/latest/gallery/index.html
    http://www.naturalearthdata.com/downloads/110m-physical-vectors/
Data for cfeature:
    https://www.naturalearthdata.com/?s=coastline
    https://scitools.org.uk/cartopy/docs/v0.16/matplotlib/feature_interface.html
"""
###############################################################################
##                                                            L I B R A R I E S
###############################################################################
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import datetime
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeom

from cartopy.feature.nightshade import Nightshade
from cartopy.io.img_tiles import Stamen
from matplotlib.lines import Line2D as Line
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import Stroke
from shapely.ops import transform as geom_transform



###############################################################################
##                                            *** 1 - CUSTOM BOUNDARY SHAPE ***
###############################################################################
## This example demonstrates how a custom shape geometry may be used instead 
## of the projection’s default boundary.
## 
## In this instance, we define the boundary as a circle in axes coordinates. 
## This means that no matter the extent of the map itself, the boundary will 
## always be a circle.
###############################################################################
def Custom_Boundary_Shape():
    fig = plt.figure(figsize=[10, 5])
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.SouthPolarStereo())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo(),
                          sharex=ax1, sharey=ax1)
    fig.subplots_adjust(bottom=0.05, top=0.95,
                        left=0.04, right=0.95, wspace=0.02)

    # Limit the map to -60 degrees latitude and below.
    ax1.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.OCEAN)

    ax1.gridlines()
    ax2.gridlines()

    ax2.add_feature(cfeature.LAND)
    ax2.add_feature(cfeature.OCEAN)

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax2.set_boundary(circle, transform=ax2.transAxes)

    ax1.set_title("Squared Map")
    ax2.set_title("Circle Map")
    plt.suptitle("01. Custom Boundary Shape", fontsize=12, color='darkred')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=.5, hspace=None)
    plt.show()
    
    # Save as png
    plt.savefig('map_cartopy/01_Custom_Boundary_Shape.png', bbox_inches='tight')
    print("01. Custom Boundary Shape... Done!")
    

    
###############################################################################
##                                                 *** 2 - Feature creation ***
###############################################################################
## This example manually instantiates a cartopy.feature.NaturalEarthFeature to 
## access administrative boundaries (states and provinces).
## 
## Note that this example is intended to illustrate the ability to construct 
## Natural Earth features that cartopy does not necessarily know about a 
## priori. In this instance however, it would be possible to make use of the 
## pre-defined cartopy.feature.STATES constant.
###############################################################################
def Feature_Creation():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([80, 170, -45, 30], crs=ccrs.PlateCarree())

    # Put a background image on for nice sea rendering.
    ax.stock_img()

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lines',
                                                    scale='50m',
                                                    facecolor='none')

    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(states_provinces, edgecolor='gray')

    # Add a text annotation for the license information to the
    # the bottom right corner.
    text = AnchoredText('© {}; license: {}'.format(SOURCE, LICENSE),
                        loc=4, prop={'size': 12}, frameon=True)
    ax.add_artist(text)
    plt.suptitle("02. Feature Creation", fontsize=12, color='darkred')
    plt.show()
    
    # Save as png
    plt.savefig('map_cartopy/02_Feature_Creation.png', bbox_inches='tight')
    print("02. Feature Creation... Done!")



###############################################################################
##                                                          *** 3 - Feature ***
###############################################################################
## A demonstration of some of the built-in Natural Earth features found in 
## cartopy.
###############################################################################
def Feature():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-20, 60, -40, 45], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    plt.suptitle("03. Feature", fontsize=12, color='darkred')
    plt.show()
    
    # Save as png
    plt.savefig('map_cartopy/03_Feature.png', bbox_inches='tight')
    print("03. Feature... Done!")



###############################################################################
##                                                       *** 4 - Global Map ***
###############################################################################
## An example of a simple map that compares Geodetic and Plate Carree lines 
## between two locations.
###############################################################################
def Global_Map():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()

    ax.stock_img()
    ax.coastlines()

    ax.plot(-0.08, 51.53, 'o', transform=ccrs.PlateCarree())
    ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.PlateCarree(), label='PlateCarree')
    ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.Geodetic(), label='Geodetic')

    plt.legend()
    plt.suptitle("04. Global Map", fontsize=12, color='darkred')
    plt.show()
    
    # Save as png
    plt.savefig('map_cartopy/04_Global_Map.png', bbox_inches='tight')
    print("04. Global Map... Done!")



###############################################################################
##                                                *** 5 - Hurricane Katrina ***
###############################################################################
## This example uses the power of Shapely to illustrate states that are likely 
## to have been significantly impacted by Hurricane Katrina.
###############################################################################
def sample_data():
    """
    Return a list of latitudes and a list of longitudes (lons, lats)
    for Hurricane Katrina (2005).

    The data was originally sourced from the HURDAT2 dataset from AOML/NOAA:
    http://www.aoml.noaa.gov/hrd/hurdat/newhurdat-all.html on 14th Dec 2012.

    """
    lons = [-75.1, -75.7, -76.2, -76.5, -76.9, -77.7, -78.4, -79.0,
            -79.6, -80.1, -80.3, -81.3, -82.0, -82.6, -83.3, -84.0,
            -84.7, -85.3, -85.9, -86.7, -87.7, -88.6, -89.2, -89.6,
            -89.6, -89.6, -89.6, -89.6, -89.1, -88.6, -88.0, -87.0,
            -85.3, -82.9]

    lats = [23.1, 23.4, 23.8, 24.5, 25.4, 26.0, 26.1, 26.2, 26.2, 26.0,
            25.9, 25.4, 25.1, 24.9, 24.6, 24.4, 24.4, 24.5, 24.8, 25.2,
            25.7, 26.3, 27.2, 28.2, 29.3, 29.5, 30.2, 31.1, 32.6, 34.1,
            35.6, 37.0, 38.6, 40.1]

    return lons, lats


def Hurricane_Katrina():
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal())

    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())

    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='110m',
                                         category='cultural', name=shapename)

    lons, lats = sample_data()

    # to get the effect of having just the states without a map "background"
    # turn off the outline and background patches
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)

    ax.set_title('US States which intersect the track of '
                 'Hurricane Katrina (2005)')

    # turn the lons and lats into a shapely LineString
    track = sgeom.LineString(zip(lons, lats))

    # buffer the linestring by two degrees (note: this is a non-physical
    # distance)
    track_buffer = track.buffer(2)

    def colorize_state(geometry):
        facecolor = (0.9375, 0.9375, 0.859375)
        if geometry.intersects(track):
            facecolor = 'red'
        elif geometry.intersects(track_buffer):
            facecolor = '#FF7E00'
        return {'facecolor': facecolor, 'edgecolor': 'black'}

    ax.add_geometries(
        shpreader.Reader(states_shp).geometries(),
        ccrs.PlateCarree(),
        styler=colorize_state)

    ax.add_geometries([track_buffer], ccrs.PlateCarree(),
                      facecolor='#C8A2C8', alpha=0.5)
    ax.add_geometries([track], ccrs.PlateCarree(),
                      facecolor='none', edgecolor='k')

    # make two proxy artists to add to a legend
    direct_hit = mpatches.Rectangle((0, 0), 1, 1, facecolor="red")
    within_2_deg = mpatches.Rectangle((0, 0), 1, 1, facecolor="#FF7E00")
    labels = ['State directly intersects\nwith track',
              'State is within \n2 degrees of track']
    ax.legend([direct_hit, within_2_deg], labels,
              loc='lower left', bbox_to_anchor=(0.025, -0.1), fancybox=True)

    plt.suptitle("05. Hurricane Katrina", fontsize=12, color='darkred')
    plt.show()
    
    # Save as png
    plt.savefig('map_cartopy/05_Hurricane_Katrina.png', bbox_inches='tight')    
    print("05. Hurricane Katrina... Done!")
    


###############################################################################
##                                               *** 6 - Nightshade feature ***
###############################################################################
## Draws a polygon where there is no sunlight for the given datetime.
###############################################################################
def Nightshade_Feature():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    date = datetime.datetime(1999, 12, 31, 12)
    #date = datetime.datetime(2020, 4, 8, 12)
    
    ax.set_title('Night time shading for {}'.format(date))
    ax.stock_img()
    ax.add_feature(Nightshade(date, alpha=0.2))
    plt.suptitle("06. Nightshade Feature", fontsize=12, color='darkred')
    plt.show()
    
    # Save as png
    plt.savefig('map_cartopy/06_Nightshade_Feature.png', bbox_inches='tight')
    print("06. Nightshade Feature... Done!")



###############################################################################
##                                               *** 7 - Rotated pole boxes ***
###############################################################################
## A demonstration of the way a box is warped when it is defined in a rotated 
## pole coordinate system.
## Try changing the box_top to 44, 46 and 75 to see the effect that including 
## the pole in the polygon has.
###############################################################################
def Rotated_Pole_Boxes():
    rotated_pole = ccrs.RotatedPole(pole_latitude=45, pole_longitude=180)

    box_top = 45
    x, y = [-44, -44, 45, 45, -44], [-45, box_top, box_top, -45, -45]

    fig = plt.figure(figsize=(11, 5.5))

    ax = fig.add_subplot(2, 1, 1, projection=rotated_pole)
    ax.stock_img()
    ax.coastlines()
    ax.plot(x, y, marker='o', transform=rotated_pole)
    ax.fill(x, y, color='coral', transform=rotated_pole, alpha=0.4)
    ax.set_title("rotated_pole")
    ax.gridlines()

    ax = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.plot(x, y, marker='o', transform=rotated_pole)
    ax.fill(x, y, transform=rotated_pole, color='coral', alpha=0.4)
    ax.set_title("PlateCarree")
    ax.gridlines()
    
    plt.suptitle("07. Rotated pole boxes", fontsize=12, color='darkred')
    plt.show()
    
    # Save as png
    plt.savefig('map_cartopy/07_Rotated_Pole_Boxes.png', bbox_inches='tight')
    print("07. Rotated pole boxes... Done!")



###############################################################################
##                       *** 8 - The effect of badly referencing an ellipse ***
###############################################################################
## This example demonstrates the effect of referencing your data to an 
## incorrect ellipse.
## First we define two coordinate systems - one using the World Geodetic System 
## established in 1984 and the other using a spherical globe. Next we extract 
## data from the Natural Earth land dataset and convert the Geodetic 
## coordinates (referenced in WGS84) into the respective coordinate systems 
## that we have defined. Finally, we plot these datasets onto a map assuming 
## that they are both referenced to the WGS84 ellipse and compare how the 
## coastlines are shifted as a result of referencing the incorrect ellipse.
###############################################################################
def transform_fn_factory(target_crs, source_crs):
    """
    Return a function which can be used by ``shapely.op.transform``
    to transform the coordinate points of a geometry.

    The function explicitly *does not* do any interpolation or clever
    transformation of the coordinate points, so there is no guarantee
    that the resulting geometry would make any sense.

    """
    def transform_fn(x, y, z=None):
        new_coords = target_crs.transform_points(source_crs,
                                                 np.asanyarray(x),
                                                 np.asanyarray(y))
        return new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]

    return transform_fn


def Badly_Referencing():
    # Define the two coordinate systems with different ellipses.
    wgs84 = ccrs.PlateCarree(globe=ccrs.Globe(datum='WGS84',
                                              ellipse='WGS84'))
    sphere = ccrs.PlateCarree(globe=ccrs.Globe(datum='WGS84',
                                               ellipse='sphere'))

    # Define the coordinate system of the data we have from Natural Earth and
    # acquire the 1:10m physical coastline shapefile.
    # Source: https://www.naturalearthdata.com/?s=coastline
    geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
    dataset = cfeature.NaturalEarthFeature(category='physical',
                                           name='Coastline',
                                           scale='10m')

    # Create a Stamen map tiler instance, and use its CRS for the GeoAxes.
    tiler = Stamen('terrain-background')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)
    ax.set_title('The effect of incorrectly referencing the Solomon Islands')

    # Pick the area of interest. In our case, roughly the Solomon Islands, and
    # get hold of the coastlines for that area.
    extent = [155, 163, -11.5, -6]
    ax.set_extent(extent, geodetic)
    geoms = list(dataset.intersecting_geometries(extent))

    # Add the Stamen aerial imagery at zoom level 7.
    ax.add_image(tiler, 7)

    # Transform the geodetic coordinates of the coastlines into the two
    # projections of differing ellipses.
    wgs84_geoms = [geom_transform(transform_fn_factory(wgs84, geodetic),
                                  geom) for geom in geoms]
    sphere_geoms = [geom_transform(transform_fn_factory(sphere, geodetic),
                                   geom) for geom in geoms]

    # Using these differently referenced geometries, assume that they are
    # both referenced to WGS84.
    ax.add_geometries(wgs84_geoms, wgs84, edgecolor='white', facecolor='none')
    ax.add_geometries(sphere_geoms, wgs84, edgecolor='gray', facecolor='none')

    # Create a legend for the coastlines.
    legend_artists = [Line([0], [0], color=color, linewidth=3)
                      for color in ('white', 'gray')]
    legend_texts = ['Correct ellipse\n(WGS84)', 'Incorrect ellipse\n(sphere)']
    legend = ax.legend(legend_artists, legend_texts, fancybox=True,
                       loc='lower left', framealpha=0.75)
    legend.legendPatch.set_facecolor('wheat')

    # Create an inset GeoAxes showing the location of the Solomon Islands.
    sub_ax = fig.add_axes([0.7, 0.625, 0.2, 0.2],
                          projection=ccrs.PlateCarree())
    sub_ax.set_extent([110, 180, -50, 10], geodetic)

    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)
    sub_ax.outline_patch.set_path_effects([effect])

    # Add the land, coastlines and the extent of the Solomon Islands.
    sub_ax.add_feature(cfeature.LAND)
    sub_ax.coastlines()
    extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='none',
                          edgecolor='blue', linewidth=2)

    plt.suptitle("08. The effect of badly referencing an ellipse", fontsize=12, color='darkred')
    plt.show()
    
    # Save as png
    plt.savefig('map_cartopy/08_Badly_Referencing.png', bbox_inches='tight')
    print("08. The effect of badly referencing an ellipse... Done!")



###############################################################################
##                                            *** M A I N   F U N C T I O N ***
###############################################################################
def main():
    Custom_Boundary_Shape()
    Feature_Creation()
    Feature()
    Global_Map()
    Hurricane_Katrina()
    Nightshade_Feature()
    Rotated_Pole_Boxes()
    Badly_Referencing()
    plt.style.use('default')



if __name__ == '__main__':
    main()


