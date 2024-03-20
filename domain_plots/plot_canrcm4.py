
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
import shapely
import shapefile

shapefile_filepath = '/Users/evagnegy/Desktop/GLISA/shapefiles/'
#file = '/Users/evagnegy/Desktop/CanESM2_WRF/topography/CanRCM4.nc'
#file2 = '/Users/evagnegy/Downloads/sftlf_NAM-22_CCCma-CanESM2_historical_r1i1p1_CCCma-CanRCM4_r2_fx.nc'
#file = '/Users/evagnegy/Downloads/orog.hist.CanESM2.CanRCM4.fixed.NAM-22i.raw.nc'
file2 = '/Users/evagnegy/Downloads/ds_ice.nc'

#topo_nc_25km = Dataset(file, mode='r')
nc = Dataset(file2, mode='r')

lats = nc.variables['lat'][:]
lons = nc.variables['lon'][:]
lons,lats=np.meshgrid(lons,lats)

#topo_25km = np.squeeze(topo_nc_25km.variables['orog'][:])
#land_frac = np.squeeze(landfrac.variables['sftlf'][:])

var = np.squeeze(nc.variables['__xarray_dataarray_variable__'][:])
#topo_25km[land_frac!=100] = np.nan



#%%

import cartopy.io.shapereader as shpreader


min_lon = -82
min_lat = 42
max_lon = -77.5
max_lat = 45

# Create a bounding box geometry
bbox = shapely.geometry.box(min_lon, min_lat, max_lon, max_lat)

# Read the shapefile
shapefile_path = '/Users/evagnegy/Downloads/lcsd000a23a_e/lcsd000a23a_e.shp'
reader = shpreader.Reader(shapefile_path,ccrs.epsg(3347))

# Filter geometries within the bounding box
filtered_geometries = []
for geometry in reader.geometries():
    print(geometry)
    if geometry.within(bbox):
        filtered_geometries.append(geometry)



        #%%
fig = plt.figure(figsize=(10, 6),dpi=200) 
ax = fig.add_subplot(1, 1, 1, projection=ccrs.RotatedPole(pole_latitude=42.5, pole_longitude=83.0))


plt.pcolormesh(lons, lats, var,vmin=19.5,vmax=31, transform=ccrs.PlateCarree())

ax.add_geometries(Reader(shapefile_filepath + 'st99_d00/st99_d00.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)
ax.add_geometries(Reader(shapefile_filepath + 'province/province.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)               
ax.add_geometries(Reader('/Users/evagnegy/Downloads/lcsd000a23a_e/lcsd000a23a_e.shp').geometries(),ccrs.epsg(3347),facecolor='none',edgecolor='k',linewidth=1)               


ax.set_extent([-82, -75, 42, 45], crs=ccrs.PlateCarree())
#plt.title('0.22\N{degree sign}i',fontsize=24)

cbar_ax = fig.add_axes([0.2, 0.08, 0.62, 0.03])
fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=19.5, vmax=31)),cax=cbar_ax, orientation='horizontal',extend='both')
cbar_ax.tick_params(labelsize=20)
#cbar_ax.set_xlabel('Land Fraction',size=20) 

