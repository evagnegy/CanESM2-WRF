#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:45:41 2023

@author: evagnegy
"""
import matplotlib.pyplot as plt 
import matplotlib as mpl
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import rgb2hex 
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import  ScalarMappable
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import datetime
import matplotlib.cbook
from cartopy.io.shapereader import Reader
from scipy import stats
import cartopy.feature as cf #part of cartopy that adds features, like borders, to the map
import shapefile as shp
from pyproj import Proj
from shapely.geometry import shape, Point
import WRFDomainLib


shapefile_filepath = '/Users/evagnegy/Desktop/GLISA/shapefiles/'

topo10km_file = '/Users/evagnegy/Desktop/CanESM2_WRF/topography/10km_example'
topo25km_file = '/Users/evagnegy/Desktop/CanESM2_WRF/topography/CanRCM4.nc'
topo1deg_file = '/Users/evagnegy/Desktop/CanESM2_WRF/topography/MIROC5_1deg.nc'
topo28deg_file = '/Users/evagnegy/Desktop/CanESM2_WRF/topography/CanESM2.nc'
#topod03_file = '/Users/evagnegy/Desktop/CanESM2_WRF/topography/geo_em.d03.nc'


topo_nc_10km = Dataset(topo10km_file, mode='r')
topo_nc_25km = Dataset(topo25km_file, mode='r')
topo_nc_1deg = Dataset(topo1deg_file, mode='r')
topo_nc_28deg = Dataset(topo28deg_file, mode='r')
#topo_nc_d03 = Dataset(topod03_file, mode='r')

lat_10km = topo_nc_10km.variables['lat'][:]
lat_25km = topo_nc_25km.variables['lat'][:]
lat_1deg = topo_nc_1deg.variables['lat'][:]
lat_28deg = topo_nc_28deg.variables['lat'][:]
#lat_d03 =  np.squeeze(topo_nc_d03.variables['XLAT_C'][:])

lon_10km = topo_nc_10km.variables['lon'][:]
lon_25km = topo_nc_25km.variables['lon'][:]
lon_1deg = topo_nc_1deg.variables['lon'][:]
lon_28deg = topo_nc_28deg.variables['lon'][:]
#lon_d03 =  np.squeeze(topo_nc_d03.variables['XLONG_C'][:])

lon_10km, lat_10km = np.meshgrid(lon_10km,lat_10km)
lon_1deg, lat_1deg = np.meshgrid(lon_1deg,lat_1deg)
lon_28deg, lat_28deg = np.meshgrid(lon_28deg,lat_28deg)


topo_10km = np.squeeze(topo_nc_10km.variables['orog'][:])
topo_25km = np.squeeze(topo_nc_25km.variables['orog'][:])
topo_1deg = np.squeeze(topo_nc_1deg.variables['orog'][:])
topo_28deg = np.squeeze(topo_nc_28deg.variables['orog'][:])
#topo_d03 = np.squeeze(topo_nc_d03.variables['HGT_M'][:])


topod01_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d01.nc'
topo_nc_d01 = Dataset(topod01_file, mode='r')
lat_d01 =  np.squeeze(topo_nc_d01.variables['XLAT_C'][:])
lon_d01 =  np.squeeze(topo_nc_d01.variables['XLONG_C'][:])
topo_d01 = np.squeeze(topo_nc_d01.variables['HGT_M'][:])
landmask_d01 = np.squeeze(topo_nc_d01.variables['LANDMASK'][:])

topod02_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d02.nc'
topo_nc_d02 = Dataset(topod02_file, mode='r')
lat_d02 =  np.squeeze(topo_nc_d02.variables['XLAT_C'][:])
lon_d02 =  np.squeeze(topo_nc_d02.variables['XLONG_C'][:])
topo_d02 = np.squeeze(topo_nc_d02.variables['HGT_M'][:])
landmask_d02 = np.squeeze(topo_nc_d02.variables['LANDMASK'][:])

topod03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
topo_nc_d03 = Dataset(topod03_file, mode='r')
lat_d03 =  np.squeeze(topo_nc_d03.variables['XLAT_C'][:])
lon_d03 =  np.squeeze(topo_nc_d03.variables['XLONG_C'][:])
topo_d03 = np.squeeze(topo_nc_d03.variables['HGT_M'][:])
landmask_d03 = np.squeeze(topo_nc_d03.variables['LANDMASK'][:])

#%%

WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

cmap = matplotlib.cm.terrain
vmin = 0
vmax = 3000

def topo_plot(lons,lats,orog,filename):
    fig = plt.figure(figsize=(10, 6),dpi=200) 
    #ax = fig.add_subplot(1, 1, 1, projection=wpsproj) 
    #ax = fig.add_subplot(1, 1, 1, projection=ccrs.RotatedPole(pole_latitude=42.5, pole_longitude=83.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.RotatedPole(pole_latitude=80, pole_longitude=83.0))

    
    
    plt.pcolormesh(lons, lats, orog, cmap='terrain', vmin=0,vmax=2500, transform=ccrs.PlateCarree())
    
    ax.add_geometries(Reader(shapefile_filepath + 'st99_d00/st99_d00.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)
    ax.add_geometries(Reader(shapefile_filepath + 'province/province.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)               
    
    ax.set_extent([-126, -122, 47, 51], crs=ccrs.PlateCarree())
    #ax.set_extent([-132, -118, 44.7, 53.2], crs=ccrs.PlateCarree())
    #ax.set_extent([-130, -120, 46, 52.2], crs=ccrs.PlateCarree())

    # d03 box

    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/13
    random_x_factor = corner_x3[0]/65

    ax.add_patch(matplotlib.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],
                                        fill=None, lw=3, edgecolor='red', zorder=3))
    #ax.text(corner_x3[0]-length_x[2]*0.6, corner_y3[0]+length_y[2]*1.4, 'D03', va='top', ha='left',
    #         fontweight='bold', size=20, color='red', zorder=2)



    cb = plt.colorbar(orientation='vertical',extend='max',fraction=0.045,pad=0.03)
    cb.ax.tick_params(labelsize=15)
    cb.ax.set_ylabel('Elevation (m)',fontsize=18)  
           
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF/topography/' + filename + '.png',bbox_inches='tight')

#topo_plot(lon_10km,lat_10km,topo_10km,'10km')
#topo_plot(lon_25km,lat_25km,topo_25km,'25km')
#topo_plot(lon_1deg,lat_1deg,topo_1deg,'1deg')
topo_plot(lon_28deg,lat_28deg,topo_28deg,'28deg')
topo_plot(lon_d03,lat_d03,topo_d03,'d03')