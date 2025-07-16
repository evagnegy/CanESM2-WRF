#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:42:39 2025

@author: evagnegy
"""

from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys
import numpy as np
import datetime
import matplotlib.ticker as ticker
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/functions/')
#from canesm2_eval_funcs import *
import WRFDomainLib
import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib import cm
import xarray as xr
import scipy
import matplotlib as mpl
import matplotlib.colors as pltcol

#%%

variable = 'wind'
period = 'rcp85'
domain = 'd03'



if variable == "t":
    var = 'T2'
    filename = 't_'+domain+'_tas_daily'
elif variable == "pr":
    var = 'pr'
    filename = 'pr_'+domain+'_daily'
elif variable == "wind":
    var = 'wspd'
    if domain=="d03":
        filename = 'wspd_'+domain+'_mon'
    else:
        filename = "wind_"+domain+'_daily'
    
geo_em_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.'+domain+'.nc'

#gridded_data_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/'
gridded_data_path = '/Volumes/EVA/gridded_model_data/'


#%%
ds_mask = np.squeeze(xr.open_dataset(geo_em_file)['LANDMASK'])
ds_mask = ds_mask.rename({'south_north': 'lons', 'west_east': 'lats'})


var_hist_daily = xr.open_dataset(gridded_data_path + filename + '_hist.nc')[var]
var_fut_daily = xr.open_dataset(gridded_data_path + filename + '_' + period + '.nc')[var]


seas_hist = var_hist_daily.sel(time=var_hist_daily['time.month'].isin([12,1,2]))
seas_fut = var_fut_daily.sel(time=var_fut_daily['time.month'].isin([12,1,2]))


var_hist = seas_hist.mean(dim='time')
var_fut = seas_fut.mean(dim='time')

ds_delta = (var_fut - var_hist)/var_hist

#%%
ds_masked = ds_delta.where(ds_mask==0)

#%%

area_weights = np.cos(np.deg2rad(ds_delta.lat))
ds_weighted = ds_masked.weighted(area_weights).mean(dim=("lats","lons"))


print(ds_weighted)
#%%


WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

fig1 = plt.figure(figsize=(10, 10),dpi=200)
ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

ax1.pcolormesh(ds_masked.lon.values, ds_masked.lat.values, ds_masked, transform=ccrs.PlateCarree())

#ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
ax1.add_feature(cf.BORDERS,linewidth=0.5)
ax1.add_feature(cf.STATES,linewidth=0.5)

ax1.set_extent([-131+1.4, -119-1.15, 46+0.4, 52-0.3], crs=ccrs.PlateCarree())






































