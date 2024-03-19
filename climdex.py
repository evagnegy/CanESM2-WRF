#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:18:21 2024

@author: evagnegy
"""

import xarray as xr
import xclim
from netCDF4 import Dataset
import datetime
import numpy as np
import time

import WRFDomainLib
import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

#%%

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'

geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])
#%%

wrf_d03_tx_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/t_d03_tmax_daily_hist.nc'
wrf_d03_tn_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/t_d03_tmin_daily_hist.nc'
wrf_d03_pr_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/pr_d03_daily_hist.nc'

#tx_hist = xr.open_dataset(wrf_d03_tx_hist_file)['T2'].chunk({'time': -1, 'lat': 20, 'lon': 20} )
#tn_hist = xr.open_dataset(wrf_d03_tn_hist_file)['T2'].chunk({'time': -1, 'lat': 20, 'lon': 20} )
pr_hist = xr.open_dataset(wrf_d03_pr_hist_file)['pr'].chunk({'time': -1, 'lat': 20, 'lon': 20} )

#tx_hist.attrs['units'] = 'K'
#tn_hist.attrs['units'] = 'K'
#pr_hist.attrs['units'] = 'mm/day'

#tas_hist = xclim.indicators.atmos.tg(tasmin=tn_hist, tasmax=tx_hist).chunk({'time': -1, 'lat': 20, 'lon': 20} )

lons = pr_hist.coords['lon'].values
lats = pr_hist.coords['lat'].values

#%%
tx10p_hist = xclim.core.calendar.percentile_doy(tx_hist, window=5, per=10.0, alpha=1/3, beta=1/3, copy=True)
tx90p_hist = xclim.core.calendar.percentile_doy(tx_hist, window=5, per=90.0, alpha=1/3, beta=1/3, copy=True)

tn10p_hist = xclim.core.calendar.percentile_doy(tn_hist, window=5, per=10.0, alpha=1/3, beta=1/3, copy=True)
tn90p_hist = xclim.core.calendar.percentile_doy(tn_hist, window=5, per=90.0, alpha=1/3, beta=1/3, copy=True)

pr75p_hist = xclim.core.calendar.percentile_doy(pr_hist, window=29, per=75.0, alpha=1/3, beta=1/3, copy=True)


#%%

csdi = xclim.indicators.atmos.cold_spell_duration_index(tasmin=tn_hist, tasmin_per=tn10p_hist, window=6)
cdd = xclim.indicators.atmos.cooling_degree_days(tas=tas_hist, thresh='18.0 degC', freq='YS')
dlyfrzthw = xclim.indicators.atmos.daily_freezethaw_cycles(tasmin=tn_hist, tasmax=tx_hist, thresh_tasmin='0 degC', thresh_tasmax='0 degC', op_tasmin='<=', op_tasmax='>', freq='YS')
#freshet_start = xclim.indicators.atmos.freshet_start(tas=tas_hist, thresh='0 degC', op='>', after_date='01-01', window=5, freq='YS')
sdii = xclim.indicators.atmos.daily_pr_intensity(pr=pr_hist, thresh='1 mm/day', freq='YS', op='>=')

#%%

#csdi_peryear = np.squeeze(np.mean(csdi,axis=0))
csdi_peryear = np.mean(csdi)

toplot = csdi_peryear.load()

#%%

def plot_climdex(gridded_data,title,vmin,vmax):
    if vmin==0:
        cmap='viridis'
        xlabel='Count'
    else:
        cmap='bwr'
        xlabel='Diff. Count'
        
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
    
    #gridded_data[land_d03==0]=np.nan
    
    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap,vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    ax1.text(-3680871, 700000, 'D03', va='top', ha='left',fontweight='bold', size=15, color='red', zorder=2)
    
    plt.title(title,fontsize=20)
    
    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(0, vmax+1, 0.5))
    cbar_ax.tick_params(labelsize=20)
    cbar_ax.set_xlabel(xlabel + " (avg. per year)",size=20) 
    
    
csdi = Dataset('/Users/evagnegy/Downloads/csdi_hist.nc').variables['csdi_6'][:]
lons = Dataset('/Users/evagnegy/Downloads/csdi_hist.nc').variables['lon'][:]
lats = Dataset('/Users/evagnegy/Downloads/csdi_hist.nc').variables['lat'][:]
#%%
csdi_peryear = np.squeeze(np.mean(csdi,axis=0))

#%%
plot_climdex(csdi_peryear,"CSDI",0,10)
