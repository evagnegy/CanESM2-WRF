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
import xarray as xr
from cartopy.util import add_cyclic_point
import WRFDomainLib

#%%

# =============================================================================
# DEMFile = 'ETOPO1.0_1degree.nc'
# DEMDs = xr.open_dataset(DEMFile)
# dem = DEMDs['DEM'].values
# dem_lat = DEMDs['lat'].values
# dem_lon = DEMDs['lon'].values
# 
# dem_lons, dem_lats = np.meshgrid(dem_lon, dem_lat)
# 
# for i in np.arange(dem.shape[0]):
#     for j in np.arange(dem.shape[1]):
#         if dem[i,j]<0:
#             dem[i,j]=0
# =============================================================================
  #%%          
            
WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

cmap = matplotlib.cm.terrain
vmin = 0
vmax = 3000

#%%
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

#%% plot all 3 domains



fig1 = plt.figure(figsize=(10, 10),dpi=200)
ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

# =============================================================================
# topo_d03[landmask_d03==False] = np.nan
# topo_d02[landmask_d02==False] = np.nan
# topo_d01[landmask_d01==False] = np.nan
# =============================================================================

#ax1.pcolormesh(dem_lons, dem_lats, dem, cmap=cmap, vmin=vmin, vmax=vmax, alpha=1, transform=ccrs.PlateCarree(), zorder=0)
ax1.pcolormesh(lon_d01, lat_d01, topo_d01, cmap=cmap, vmin=vmin,vmax=vmax, alpha=1, transform=ccrs.PlateCarree(),zorder=0)


# d01
corner_x1, corner_y1 = WRFDomainLib.reproject_corners(corner_lon_full[0,:], corner_lat_full[0,:], wpsproj, latlonproj)

ax1.set_xlim([corner_x1[0]-length_x[0]/15, corner_x1[3]+length_x[0]/15])
ax1.set_ylim([corner_y1[0]-length_y[0]/15, corner_y1[3]+length_y[0]/15])

# d01 box
ax1.add_patch(matplotlib.patches.Rectangle((corner_x1[0], corner_y1[0]+20000),  length_x[0], length_y[0]-50000, 
                                    fill=None, lw=5, edgecolor='blue', zorder=2))
ax1.text(corner_x1[0]+length_x[0]*0.03, corner_y1[0]+length_y[0]*0.92, 'D01',
         fontweight='bold', size=28, color='blue', zorder=4)


# d02 box
ax1.pcolormesh(lon_d02, lat_d02, topo_d02, cmap=cmap, vmin=vmin,vmax=vmax, alpha=1, transform=ccrs.PlateCarree(),zorder=1)

corner_x2, corner_y2 = WRFDomainLib.reproject_corners(corner_lon_full[1,:], corner_lat_full[1,:], wpsproj, latlonproj)
random_y_factor2 = corner_y2[0]/15
random_x_factor2 = corner_x2[0]/100

ax1.add_patch(matplotlib.patches.Rectangle((corner_x2[0]+random_x_factor2, corner_y2[0]+random_y_factor2),  length_x[1], length_y[1], 
                                    fill=None, lw=5, edgecolor='black', zorder=2))
ax1.text(corner_x2[0]-length_x[1]*0.25, corner_y2[0]+length_y[1]*1.05, 'D02',
         fontweight='bold', size=28, color='black', zorder=4)


# d03 box
ax1.pcolormesh(lon_d03, lat_d03, topo_d03, cmap=cmap, vmin=vmin,vmax=vmax, alpha=1, transform=ccrs.PlateCarree(),zorder=1)


corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
random_y_factor = -corner_y3[0]/13
random_x_factor = corner_x3[0]/65

ax1.add_patch(matplotlib.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],
                                    fill=None, lw=5, edgecolor='red', zorder=2))
ax1.text(corner_x3[0]-length_x[2]*0.9, corner_y3[0]+length_y[2]*1.5, 'D03', va='top', ha='left',
         fontweight='bold', size=28, color='red', zorder=4)


ax1.text(corner_x1[0]-length_x[0]*0.2, corner_y1[0]+length_y[0]*1.03, '(a)',
         fontweight='bold', size=26, color='k', zorder=2)




# decorations
ax1.coastlines('10m', linewidth=0.8)
ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
ax1.add_feature(cf.LAKES, edgecolor='face', facecolor='lightblue', zorder=1)

shapefile_filepath = '/Users/evagnegy/Desktop/GLISA/shapefiles/'
ax1.add_geometries(Reader(shapefile_filepath + 'st99_d00/st99_d00.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)
ax1.add_geometries(Reader(shapefile_filepath + 'province/province.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)               


gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', alpha=1,lw=1.5)
gl.top_labels = False
gl.bottom_labels = False
gl.left_labels = False
gl.right_labels = True

    
gl.xlocator = matplotlib.ticker.FixedLocator(np.arange(-180,-49,20))
gl.ylocator = matplotlib.ticker.FixedLocator(np.arange(0,81,20))

gl.xlabel_style = {'size': 0}
gl.ylabel_style = {'size':20}

ax1.text(corner_x1[0]-length_x[0]*0.1, corner_y1[0]+length_y[0]*0.83, '180$\degree$W',
          size=20, color='k', zorder=10)

ax1.text(corner_x1[0]+length_x[0]*0.005, corner_y1[0]+length_y[0]*0.55, '160$\degree$W',
          size=20, color='k', zorder=10)

ax1.text(corner_x1[0]+length_x[0]*0.1, corner_y1[0]+length_y[0]*0.3, '140$\degree$W',
          size=20, color='k', zorder=10)

ax1.text(corner_x1[0]+length_x[0]*0.3, corner_y1[0]+length_y[0]*0.1, '120$\degree$W',
          size=20, color='k', zorder=10)

#ax1.set_title('WRF nested domain setup', size=16)

#cbar_ax = fig1.add_axes([0.175, 0.08, 0.68, 0.03])
#fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
#              cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal',extend='max')

#cbar_ax.tick_params(labelsize=20)

#cbar_ax.set_xlabel('Elevation (m)',size=25)  

fig1.savefig('/Users/evagnegy/Desktop/paper_figures/domain/WRF_domains.png', dpi=600,bbox_inches='tight')

#plt.close()
#del(fig1)



#%% plot only D03


fig1 = plt.figure(figsize=(10, 10),dpi=200)
ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

#topo_d03[landmask_d03==False] = np.nan
#topo_d02[landmask_d02==False] = np.nan

topo_d02[topo_d02<2] = np.nan
topo_d03[topo_d03<2] = np.nan

ax1.pcolormesh(lon_d02, lat_d02, topo_d02, cmap=cmap, vmin=vmin,vmax=vmax, alpha=1, transform=ccrs.PlateCarree(),zorder=0)
ax1.pcolormesh(lon_d03, lat_d03, topo_d03, cmap=cmap, vmin=vmin,vmax=vmax, alpha=1, transform=ccrs.PlateCarree(),zorder=1)

random_y_factor = -corner_y3[0]/13
random_x_factor = corner_x3[0]/65
# d03 box
corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
ax1.set_xlim([corner_x3[0]+random_x_factor-length_x[0]/30, corner_x3[3]+random_x_factor+length_x[0]/30])
ax1.set_ylim([corner_y3[0]+random_y_factor-length_y[0]/35, corner_y3[3]+random_y_factor+length_y[0]/35])

ax1.add_patch(matplotlib.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],
                                    fill=None, lw=5, edgecolor='red', zorder=2))
ax1.text(corner_x3[0]+length_x[2]*0, corner_y3[0]+length_y[2]*0.9, 'D03', va='top', ha='left',
         fontweight='bold', size=28, color='red', zorder=2)


ax1.text(corner_x3[0]-length_x[2]*0.53, corner_y3[0]+length_y[2]*1.1, '(b)',
         fontweight='bold', size=26, color='k', zorder=2)

# decorations

ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=0)
#ax1.add_feature(cf.LAKES, edgecolor='k', facecolor='lightblue', zorder=1)

shapefile_filepath = '/Users/evagnegy/Desktop/GLISA/shapefiles/'
ax1.add_geometries(Reader(shapefile_filepath + 'st99_d00/st99_d00.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)
ax1.add_geometries(Reader(shapefile_filepath + 'province/province.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)               


gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', alpha=1,linewidth=1.5,color='grey')
gl.top_labels = False
gl.bottom_labels = False
gl.left_labels = False
gl.right_labels = True
gl.xlocator = matplotlib.ticker.FixedLocator(np.arange(-180,-49,4))
gl.ylocator = matplotlib.ticker.FixedLocator(np.arange(0,81,4))

gl.xlabel_style = {'size': 0}
gl.ylabel_style = {'size':20}


#ax1.set_title('WRF nested domain setup', size=16)


ax1.text(corner_x3[0]-length_x[2]*0.3, corner_y3[0]+length_y[2]*1, '136$\degree$W',
         size=20, color='k', zorder=10)

ax1.text(corner_x3[0]-length_x[2]*0.3, corner_y3[0]+length_y[2]*0.5, '132$\degree$W',
         size=20, color='k', zorder=10)

ax1.text(corner_x3[0]-length_x[2]*0.3, corner_y3[0]-length_y[2]*0.05, '128$\degree$W',
         size=20, color='k', zorder=10)

ax1.text(corner_x3[0]-length_x[2]*0.02, corner_y3[0]-length_y[2]*0.25, '124$\degree$W',
         size=20, color='k', zorder=10)

ax1.text(corner_x3[0]+length_x[2]*0.4, corner_y3[0]-length_y[2]*0.25, '120$\degree$W',
         size=20, color='k', zorder=10)

ax1.text(corner_x3[0]+length_x[2]*0.8, corner_y3[0]-length_y[2]*0.25, '118$\degree$W',
         size=20, color='k', zorder=10)

cbar_ax = fig1.add_axes([0.175, 0.08, 0.68, 0.03])
fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
              cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal',extend='max')

cbar_ax.tick_params(labelsize=20)
cbar_ax.set_xlabel('Elevation (m)',size=25) 
 
fig1.savefig('/Users/evagnegy/Desktop/paper_figures/domain/WRF_D3.png', dpi=600,bbox_inches='tight')



