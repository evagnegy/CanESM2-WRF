#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 21:06:59 2022

@author: evagnegy
"""

from netCDF4 import Dataset
import matplotlib.patches as mpatches
import numpy as np
import matplotlib 
from cartopy.io.shapereader import Reader
import matplotlib.pyplot as plt
import cftime
import matplotlib.units as munits
from matplotlib.dates import ConciseDateConverter
munits.registry[cftime.DatetimeGregorian] = ConciseDateConverter()
import cartopy.crs as ccrs
shapefile_filepath = '/Users/evagnegy/Desktop/GLISA/shapefiles/'
import WRFDomainLib
from matplotlib.colors import ListedColormap

YlGnBu_cmap = plt.cm.get_cmap('YlGnBu')
colors = YlGnBu_cmap(np.linspace(0, 1, 256))[45:200]  # Modify the colors as needed
modified_cmap = ListedColormap(colors)
new_colors = modified_cmap(np.linspace(0, 1, 1000))
new_colormap = ListedColormap(new_colors)

bath_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/new_bath.nc'
canesm2_bath_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/canesm2_bath.nc'

nc_bath = Dataset(bath_file, mode='r')
lats_bath = nc_bath.variables['lat'][:]
lons_bath = nc_bath.variables['lon'][:]
bath = nc_bath.variables['elevation'][:].astype(float)

bath = bath*-1
bath[bath< 0] = np.nan


nc_canesm2 = Dataset(canesm2_bath_file, mode='r')
lats_canesm2 = nc_canesm2.variables['lat'][:]
lons_canesm2 = nc_canesm2.variables['lon'][:]
bath_canesm2 = nc_canesm2.variables['deptho'][:].astype(float)
#%%
WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)


#%%

fig1 = plt.figure(figsize=(10, 10),dpi=200)
ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

vmin = 0
vmax=5000

#delete some points that go over land over the nep36 area
bath_canesm2[145:160,160:170] = np.nan

ax1.pcolormesh(lons_canesm2,lats_canesm2,bath_canesm2, vmin=vmin,vmax=vmax,cmap=new_colormap,zorder=0,transform=ccrs.PlateCarree())
ax1.pcolormesh(lons_bath,lats_bath,bath, vmin=vmin,vmax=vmax,cmap=new_colormap,zorder=1,transform=ccrs.PlateCarree())


ax1.add_patch(mpatches.Rectangle(xy=[-138, 45.333], width=0.01, height=13.4,
                                facecolor='none', edgecolor='white',linewidth=5,linestyle='--',zorder=10,
                                transform=ccrs.PlateCarree()))


# this patch adds the bottom line of the NEP36 domain
radius = 800  # Radius of the arc in degrees
center_latitude = 45.2+radius  # Center latitude of the arc
center_longitude = -137+13  # Center longitude of the arc

start_angle = 269  # Start angle of the arc in degrees
end_angle = 270 

arc_patch = mpatches.Arc(
    xy=(center_longitude, center_latitude),  # Center of the arc
    width=2 * radius,  # Width of the arc
    height=2 * radius,  # Height of the arc
    angle=0,  # Rotation angle (0 for a circular arc)
    theta1=start_angle,  # Start angle of the arc
    theta2=end_angle,  # End angle of the arc
    edgecolor='white',  # Edge color of the arc
    linewidth=5,  # Line width
    linestyle='--',
    zorder=4,
    transform=ccrs.PlateCarree(),
)


ax1.add_patch(arc_patch)

ax1.text(corner_x2[0]+length_x[1]*0.2, corner_y2[0]+length_y[1]*0.7, 'NEP36-CanOE',
         fontweight='bold', size=22, rotation=45,color='white', zorder=5)



corner_x2, corner_y2 = WRFDomainLib.reproject_corners(corner_lon_full[1,:], corner_lat_full[1,:], wpsproj, latlonproj)
random_y_factor2 = corner_y2[0]/15
random_x_factor2 = corner_x2[0]/100

ax1.add_patch(matplotlib.patches.Rectangle((corner_x2[0]+random_x_factor2, corner_y2[0]+random_y_factor2),  length_x[1], length_y[1], 
                                    fill=None, lw=5, edgecolor='black', zorder=3))
ax1.text(corner_x2[0]+length_x[1]*0.02, corner_y2[0]+length_y[1]*0.91, 'D02',
         fontweight='bold', size=30, color='black', zorder=5)



corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)

random_y_factor = -corner_y3[0]/13
random_x_factor = corner_x3[0]/65
# d03 box
corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
ax1.set_xlim([corner_x2[0]+random_x_factor-length_x[0]/30, corner_x2[3]+random_x_factor+length_x[0]/30])
ax1.set_ylim([corner_y2[0]+random_y_factor-length_y[0]/35, corner_y2[3]+random_y_factor+length_y[0]/35])

ax1.add_patch(matplotlib.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],
                                    fill=None, lw=5, edgecolor='red', zorder=4))
ax1.text(corner_x3[0]-length_x[2]*0.23, corner_y3[0]+length_y[2]*1.17, 'D03', va='top', ha='left',
         fontweight='bold', size=30, color='red', zorder=4)



ax1.add_geometries(Reader(shapefile_filepath + 'st99_d00/st99_d00.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=1)
ax1.add_geometries(Reader(shapefile_filepath + 'province/province.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=1)               


gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', alpha=1,linewidth=1.2,zorder=1)
gl.top_labels = True
gl.bottom_labels = False
gl.left_labels = False
gl.right_labels = True
gl.xlocator = matplotlib.ticker.FixedLocator(np.arange(-180,-49,6))
gl.ylocator = matplotlib.ticker.FixedLocator(np.arange(0,81,6))

gl.xlabel_style = {'size': 0}
gl.ylabel_style = {'size':20}



cbar_ax = fig1.add_axes([0.175, 0.08, 0.68, 0.03])
fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=new_colormap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
              cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal',extend='max')

ax1.text(corner_x2[0]-length_x[1]*0.04, corner_y2[0]+length_y[1]*1.07, '156$\degree$W',
         rotation=40, size=20, color='k', zorder=2)

ax1.text(corner_x2[0]+length_x[1]*0.23, corner_y2[0]+length_y[1]*1.07, '150$\degree$W',
         rotation=40, size=20, color='k', zorder=2)

ax1.text(corner_x2[0]+length_x[1]*0.44, corner_y2[0]+length_y[1]*1.07, '144$\degree$W',
         rotation=40, size=20, color='k', zorder=2)

ax1.text(corner_x2[0]+length_x[1]*0.62, corner_y2[0]+length_y[1]*1.07, '138$\degree$W',
         rotation=40, size=20, color='k', zorder=2)

ax1.text(corner_x2[0]+length_x[1]*0.77, corner_y2[0]+length_y[1]*1.07, '132$\degree$W',
         rotation=40, size=20, color='k', zorder=2)

ax1.text(corner_x2[0]+length_x[1]*0.9, corner_y2[0]+length_y[1]*1.07, '126$\degree$W',
         rotation=40, size=20, color='k', zorder=2)

ax1.text(corner_x2[0]+length_x[1]*1.01, corner_y2[0]+length_y[1]*1.07, '120$\degree$W',
         rotation=40, size=20, color='k', zorder=2)

cbar_ax.tick_params(labelsize=20)
cbar_ax.set_xlabel('Bathymetry (m)',size=25) 
cbar_ax.set(xticks=[0,1000,2000,3000,4000,5000]) 
fig1.savefig('/Users/evagnegy/Desktop/paper_figures/domain/NEP36.png', dpi=600,bbox_inches='tight')







