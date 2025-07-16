

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

variable = 'wind' #t, tmin, tmax, wind, pr
period = 'rcp45'
domain = "d03"
minusmed = 'no' #yes or not

if variable == "tmin":
    perc = 5
else:
    perc = 95



if variable == "t":
    var = 'T2'
    filename = 't_'+domain+'_tas_daily'
elif variable == "pr":
    var = 'pr'
    filename = 'pr_'+domain+'_daily'
elif variable == "wind":
    var = 'wspd'
    if domain != "d03":
        filename = 'wind_'+domain+'_daily'
    else:
        filename = 'wind_'+domain+'_daily_wspd'
elif variable == "tmax":
    var = 'T2'
    filename = 't_'+domain+'_tmax_daily'
elif variable == "tmin":
    var = 'T2'
    filename = 't_'+domain+'_tmin_daily'
    
geo_em_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.'+domain+'.nc'
gridded_data_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/'
#gridded_data_path = '/Volumes/EVA/gridded_model_data/'


#%%
ds_mask = np.squeeze(xr.open_dataset(geo_em_file)['LANDMASK'])
ds_mask = ds_mask.rename({'south_north': 'lats', 'west_east': 'lons'})


directory='/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/ncfiles/'

if variable.startswith("t"):
    varkey = "T2"
elif variable == "wind":
    varkey = "wspd"
    
lons = Dataset(gridded_data_path + '/t_'+domain+'_tas_daily_hist.nc','r').variables['lon'][:]
lats = Dataset(gridded_data_path + '/t_'+domain+'_tas_daily_hist.nc','r').variables['lat'][:]

lons[lons > 0] += -360
    

ds_lat = xr.DataArray(
    data=lats,
    coords={
        'lat': (['lats', 'lats'], lats),
        'lon': (['lons', 'lons'], lons)
    },
    dims=['lats', 'lons'],
    name='delta'
)
    #%%
perc_MAM_hist = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_mam_hist.nc')[varkey].values
perc_JJA_hist = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_jja_hist.nc')[varkey].values
perc_SON_hist = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_son_hist.nc')[varkey].values
perc_DJF_hist = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_djf_hist.nc')[varkey].values

perc_MAM_fut = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_mam_{period}.nc')[varkey].values
perc_JJA_fut = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_jja_{period}.nc')[varkey].values
perc_SON_fut = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_son_{period}.nc')[varkey].values
perc_DJF_fut = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_djf_{period}.nc')[varkey].values


#%%

perc_MAM_delta_perc = ((perc_MAM_fut-perc_MAM_hist)/perc_MAM_hist)*100
perc_JJA_delta_perc = ((perc_JJA_fut-perc_JJA_hist)/perc_JJA_hist)*100
perc_SON_delta_perc = ((perc_SON_fut-perc_SON_hist)/perc_SON_hist)*100
perc_DJF_delta_perc = ((perc_DJF_fut-perc_DJF_hist)/perc_DJF_hist)*100

    #%%
perc_MAM_delta = perc_MAM_fut-perc_MAM_hist
perc_JJA_delta = perc_JJA_fut-perc_JJA_hist
perc_SON_delta = perc_SON_fut-perc_SON_hist
perc_DJF_delta = perc_DJF_fut-perc_DJF_hist

#%%
directory = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/ncfiles/'

perc_MAM_hist_minusmed = xr.open_dataset(f"{directory}{variable}_{perc}p_mam_hist_minusmed.nc")['delta_minusmed'].values
perc_JJA_hist_minusmed = xr.open_dataset(f"{directory}{variable}_{perc}p_jja_hist_minusmed.nc")['delta_minusmed'].values
perc_SON_hist_minusmed = xr.open_dataset(f"{directory}{variable}_{perc}p_son_hist_minusmed.nc")['delta_minusmed'].values
perc_DJF_hist_minusmed = xr.open_dataset(f"{directory}{variable}_{perc}p_djf_hist_minusmed.nc")['delta_minusmed'].values

perc_MAM_fut_minusmed = xr.open_dataset(f"{directory}{variable}_{perc}p_mam_{period}_minusmed.nc")['delta_minusmed'].values
perc_JJA_fut_minusmed = xr.open_dataset(f"{directory}{variable}_{perc}p_jja_{period}_minusmed.nc")['delta_minusmed'].values
perc_SON_fut_minusmed = xr.open_dataset(f"{directory}{variable}_{perc}p_son_{period}_minusmed.nc")['delta_minusmed'].values
perc_DJF_fut_minusmed = xr.open_dataset(f"{directory}{variable}_{perc}p_djf_{period}_minusmed.nc")['delta_minusmed'].values

perc_MAM_delta_minusmed = perc_MAM_fut_minusmed-perc_MAM_hist_minusmed
perc_JJA_delta_minusmed = perc_JJA_fut_minusmed-perc_JJA_hist_minusmed
perc_SON_delta_minusmed = perc_SON_fut_minusmed-perc_SON_hist_minusmed
perc_DJF_delta_minusmed = perc_DJF_fut_minusmed-perc_DJF_hist_minusmed

#%%

ds = xr.DataArray(
    data=perc_JJA_delta_perc,
    coords={
        'lat': (['lats', 'lats'], lats),
        'lon': (['lons', 'lons'], lons)
    },
    dims=['lats', 'lons'],
    name='perc'
)


ds_masked = ds.where(ds_mask==1)


area_weights = np.cos(np.deg2rad(ds_lat))
ds_weighted = ds_masked.weighted(area_weights).mean(dim=("lats","lons"))


print(ds_weighted)
#


WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

fig1 = plt.figure(figsize=(10, 10),dpi=200)
ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

ax1.pcolormesh(ds_masked.lon.values, ds_masked.lat.values, ds_masked, transform=ccrs.PlateCarree(),vmin=-5,vmax=5,cmap='bwr')

#ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
ax1.add_feature(cf.BORDERS,linewidth=0.5)
ax1.add_feature(cf.STATES,linewidth=0.5)

ax1.set_extent([-131+1.4, -119-1.15, 46+0.4, 52-0.3], crs=ccrs.PlateCarree())






































