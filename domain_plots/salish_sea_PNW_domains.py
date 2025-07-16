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

shapefile_filepath = '/Users/evagnegy/Desktop/GLISA/shapefiles/'
salish_shapefilepath = '/Users/evagnegy/Desktop/CanESM2_WRF/'

topo_file = '/Users/evagnegy/Downloads/elev.americas.5-min.nc'
topo_nc = Dataset(topo_file, mode='r')

lat = topo_nc.variables['lat'][:]
lon = topo_nc.variables['lon'][:]
topo = np.squeeze(topo_nc.variables['data'][:])

lons, lats = np.meshgrid(lon,lat)

#%%

# desired lat/lon boundaries
lat_min = 40
lat_max = 60
lon_min = -135
lon_max = -110

lons = lons-360

row_indices=[]
col_indices=[]
for row in range(len(lats[:,0])): #go through each row of lats/lons 
    for col in range(len(lons[0,:])): #go through each column of lats/lons
        if (lat_min<lats[row,col]<lat_max):
            if (lon_min<lons[row,col]<lon_max):
                row_indices.append(row)
                col_indices.append(col)

topo_trim = topo[min(row_indices):max(row_indices),min(col_indices):max(col_indices)]

lat2=lats[min(row_indices):max(row_indices),min(col_indices):max(col_indices)]
lon2=lons[min(row_indices):max(row_indices),min(col_indices):max(col_indices)]  
    

#%%
def trim_basins(lat, lon, var_all):
    
    var_fill = var_all.copy()
    var_fill[:,:]= None
    
    sf=shp.Reader(salish_shapefilepath + 'Salish_Sea_basin_boundary/Salish_Sea_basin_boundary')
    
    # obtain shapes and its records
    shapes = sf.shapes()    
    
    myProj = Proj("EPSG:3857")
    
    UTM_lon, UTM_lat = myProj(lon,lat)
    
    for row in range(len(lat[:,0])-1): #go through each row of lats/lons    
        for col in range(len(lon[0,:])-1): #go through each column of lats/lons
        
            polygon_ = shape(shapes[0])
                
            if polygon_.contains(Point(UTM_lon[row,col], UTM_lat[row,col]))==True:
                #var[:,row,col] = None 
                var_fill[row,col] = var_all[row,col]
  
    return(var_fill)          
    

topo2 = trim_basins(lat2, lon2, topo_trim)

#%%


fig = plt.figure(figsize=(10, 6),dpi=200) 
#ax = fig.add_subplot(1, 1,1, projection=ccrs.PlateCarree(central_longitude=180))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.RotatedPole(pole_latitude=42.5, pole_longitude=83.0))


cax = plt.pcolormesh(lon2, lat2, topo_trim, cmap='YlOrBr', vmin=0,vmax=7000, transform=ccrs.PlateCarree())
cax = plt.pcolormesh(lon2, lat2, topo2, cmap='Greens', vmin=-1000,vmax=6000, transform=ccrs.PlateCarree())



ax.add_feature(cf.OCEAN,linewidth=0,zorder=1)
ax.add_geometries(Reader(shapefile_filepath + 'st99_d00/st99_d00.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)
ax.add_geometries(Reader(shapefile_filepath + 'province/province.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)               

ax.add_geometries(Reader(salish_shapefilepath + 'Salish_Sea_basin_boundary/Salish_Sea_basin_boundary.shp').geometries(),ccrs.epsg(3857),edgecolor='k',linewidth=2.5,facecolor='none',zorder=2)



ax.set_extent([-129, -118, 45, 54], crs=ccrs.PlateCarree())
#cb = plt.colorbar(orientation='horizontal',extend='both',fraction=0.045,pad=0.03)
#cb.ax.set_xlabel('Elevation [m]')  
       
#plt.savefig('/Users/evagnegy/Desktop/Conferences/PNWWW 2023/topo.png',bbox_inches='tight')

#%%
fig = plt.figure(figsize=(10, 6),dpi=100) 
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-80,40))


ax.add_feature(cf.BORDERS,linewidth=0.2)
ax.add_feature(cf.OCEAN,linewidth=1,color='#D1D1D1')
ax.add_feature(cf.LAND,linewidth=1,color='#8A8A8A')
plt.scatter(-123.1, 49.3,s=500,marker=(4, 0, 30),linewidths=2,facecolors='none', edgecolors='r',transform=ccrs.PlateCarree())

ax.set_global()
plt.axis('off')
#plt.savefig('/Users/evagnegy/Desktop/Conferences/PNWWW 2023/globe.png',bbox_inches='tight',transparent=True)


#%%

#WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
#wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

topod03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
topo_nc_d03 = Dataset(topod03_file, mode='r')
lat_d03 =  np.squeeze(topo_nc_d03.variables['XLAT_C'][:])
lon_d03 =  np.squeeze(topo_nc_d03.variables['XLONG_C'][:])
topo_d03 = np.squeeze(topo_nc_d03.variables['HGT_M'][:])



fig = plt.figure(figsize=(10, 6),dpi=200) 
#ax = fig.add_subplot(1, 1,1, projection=ccrs.PlateCarree(central_longitude=180))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.RotatedPole(pole_latitude=42.5, pole_longitude=83.0))


cax = plt.pcolormesh(lon2, lat2, topo_trim, cmap='YlOrBr', vmin=0,vmax=7000, transform=ccrs.PlateCarree())
cax = plt.pcolormesh(lon_d03, lat_d03, topo_d03, cmap='Greens', vmin=-1000,vmax=6000, transform=ccrs.PlateCarree())



ax.add_feature(cf.OCEAN,linewidth=0,zorder=1)
ax.add_geometries(Reader(shapefile_filepath + 'st99_d00/st99_d00.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)
ax.add_geometries(Reader(shapefile_filepath + 'province/province.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)               




ax.set_extent([-129, -118, 45, 54], crs=ccrs.PlateCarree())
#cb = plt.colorbar(orientation='horizontal',extend='both',fraction=0.045,pad=0.03)
#cb.ax.set_xlabel('Elevation [m]')  
       
#plt.savefig('/Users/evagnegy/Desktop/Conferences/PNWWW 2023/topo.png',bbox_inches='tight')
