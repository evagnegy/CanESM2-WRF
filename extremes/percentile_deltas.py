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
import WRFDomainLib
import cartopy.feature as cf
import matplotlib.colors as pltcol
import scipy
import time
import xarray as xr

variable = 'pr' #t, tmin, tmax, wind, pr
period = 'rcp45'
domain = "d03"
minusmed = 'no' #yes or not

if variable == "tmin":
    perc = 5
else:
    perc = 95


if domain == "d03":
    gridded_data_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/'
else:
    gridded_data_path = '/Volumes/EVA/gridded_model_data/'

geo_em_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.'+domain+'.nc'
geo_em_nc = Dataset(geo_em_file, mode='r')
land = np.squeeze(geo_em_nc.variables['LANDMASK'][:])

lons = Dataset(gridded_data_path + '/t_'+domain+'_tas_daily_hist.nc','r').variables['lon'][:]
lats = Dataset(gridded_data_path + '/t_'+domain+'_tas_daily_hist.nc','r').variables['lat'][:]

lons[lons > 0] += -360
#%%

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
    
    
wrf_var_hist = xr.open_dataset(f"{gridded_data_path}{filename}_hist.nc")
wrf_var_fut = xr.open_dataset(f"{gridded_data_path}{filename}_{period}.nc")
    
#%%

def get_seasonal_data(ds):
    da = ds[var]
    
    if variable == "pr":
        da = da.where(da >= 1, np.nan)

    djf = da.sel(time=da['time'].dt.month.isin([12, 1, 2]))
    mam = da.sel(time=da['time'].dt.month.isin([3, 4, 5]))
    jja = da.sel(time=da['time'].dt.month.isin([6, 7, 8]))
    son = da.sel(time=da['time'].dt.month.isin([9, 10, 11]))

    return djf,mam,jja,son


def get_perc(seas,p):
    return seas.quantile(p/100, dim='time')

wrf_DJF_hist, wrf_MAM_hist, wrf_JJA_hist, wrf_SON_hist = get_seasonal_data(wrf_var_hist)
wrf_DJF_fut, wrf_MAM_fut, wrf_JJA_fut, wrf_SON_fut = get_seasonal_data(wrf_var_fut)
#%%

perc_MAM_hist = get_perc(wrf_MAM_hist,perc)
perc_JJA_hist = get_perc(wrf_JJA_hist,perc)
perc_SON_hist = get_perc(wrf_SON_hist,perc)
perc_DJF_hist = get_perc(wrf_DJF_hist,perc)

perc_MAM_fut = get_perc(wrf_MAM_fut,perc)
perc_JJA_fut = get_perc(wrf_JJA_fut,perc)
perc_SON_fut = get_perc(wrf_SON_fut,perc)
perc_DJF_fut = get_perc(wrf_DJF_fut,perc)

#%%

med_MAM_hist = get_perc(wrf_MAM_hist,50)
med_JJA_hist = get_perc(wrf_JJA_hist,50)
med_SON_hist = get_perc(wrf_SON_hist,50)
med_DJF_hist = get_perc(wrf_DJF_hist,50)

med_MAM_fut = get_perc(wrf_MAM_fut,50)
med_JJA_fut = get_perc(wrf_JJA_fut,50)
med_SON_fut = get_perc(wrf_SON_fut,50)
med_DJF_fut = get_perc(wrf_DJF_fut,50)

#%%
directory='/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/ncfiles/'
perc_MAM_hist.to_netcdf(f'{directory}{variable}_{str(perc)}p_mam_hist.nc')
perc_JJA_hist.to_netcdf(f'{directory}{variable}_{str(perc)}p_jja_hist.nc')
perc_SON_hist.to_netcdf(f'{directory}{variable}_{str(perc)}p_son_hist.nc')
perc_DJF_hist.to_netcdf(f'{directory}{variable}_{str(perc)}p_djf_hist.nc')

perc_MAM_fut.to_netcdf(f'{directory}{variable}_{str(perc)}p_mam_{period}.nc')
perc_JJA_fut.to_netcdf(f'{directory}{variable}_{str(perc)}p_jja_{period}.nc')
perc_SON_fut.to_netcdf(f'{directory}{variable}_{str(perc)}p_son_{period}.nc')
perc_DJF_fut.to_netcdf(f'{directory}{variable}_{str(perc)}p_djf_{period}.nc')

#%%
directory='/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/ncfiles/'
med_MAM_hist.to_netcdf(f'{directory}{variable}_50p_mam_hist.nc')
med_JJA_hist.to_netcdf(f'{directory}{variable}_50p_jja_hist.nc')
med_SON_hist.to_netcdf(f'{directory}{variable}_50p_son_hist.nc')
med_DJF_hist.to_netcdf(f'{directory}{variable}_50p_djf_hist.nc')

med_MAM_fut.to_netcdf(f'{directory}{variable}_50p_mam_{period}.nc')
med_JJA_fut.to_netcdf(f'{directory}{variable}_50p_jja_{period}.nc')
med_SON_fut.to_netcdf(f'{directory}{variable}_50p_son_{period}.nc')
med_DJF_fut.to_netcdf(f'{directory}{variable}_50p_djf_{period}.nc')
#%%
directory='/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/ncfiles/'

if variable.startswith("t"):
    varkey = "T2"
elif variable == "wind":
    varkey = "wspd"
    
perc_MAM_hist = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_mam_hist.nc')[varkey].values
perc_JJA_hist = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_jja_hist.nc')[varkey].values
perc_SON_hist = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_son_hist.nc')[varkey].values
perc_DJF_hist = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_djf_hist.nc')[varkey].values

perc_MAM_fut = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_mam_{period}.nc')[varkey].values
perc_JJA_fut = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_jja_{period}.nc')[varkey].values
perc_SON_fut = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_son_{period}.nc')[varkey].values
perc_DJF_fut = xr.open_dataset(f'{directory}{variable}_{str(perc)}p_djf_{period}.nc')[varkey].values

#%%
irectory='/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/ncfiles/'

if variable.startswith("t"):
    varkey = "T2"
elif variable == "wind":
    varkey = "wspd"
    
med_MAM_hist = xr.open_dataset(f'{directory}{variable}_50p_mam_hist.nc')[varkey].values
med_JJA_hist = xr.open_dataset(f'{directory}{variable}_50p_jja_hist.nc')[varkey].values
med_SON_hist = xr.open_dataset(f'{directory}{variable}_50p_son_hist.nc')[varkey].values
med_DJF_hist = xr.open_dataset(f'{directory}{variable}_50p_djf_hist.nc')[varkey].values

med_MAM_fut = xr.open_dataset(f'{directory}{variable}_50p_mam_{period}.nc')[varkey].values
med_JJA_fut = xr.open_dataset(f'{directory}{variable}_50p_jja_{period}.nc')[varkey].values
med_SON_fut = xr.open_dataset(f'{directory}{variable}_50p_son_{period}.nc')[varkey].values
med_DJF_fut = xr.open_dataset(f'{directory}{variable}_50p_djf_{period}.nc')[varkey].values

#%%
perc_MAM_delta = perc_MAM_fut-perc_MAM_hist
perc_JJA_delta = perc_JJA_fut-perc_JJA_hist
perc_SON_delta = perc_SON_fut-perc_SON_hist
perc_DJF_delta = perc_DJF_fut-perc_DJF_hist
#%%
if variable in ['t','tmax','tmin'] and minusmed == "yes":

    if variable == 'tmin':
        perc_MAM_hist_minusmed = med_MAM_hist-perc_MAM_hist
        perc_JJA_hist_minusmed = med_JJA_hist-perc_JJA_hist
        perc_SON_hist_minusmed = med_SON_hist-perc_SON_hist
        perc_DJF_hist_minusmed = med_DJF_hist-perc_DJF_hist
    
        perc_MAM_fut_minusmed = med_MAM_fut-perc_MAM_fut
        perc_JJA_fut_minusmed = med_JJA_fut-perc_JJA_fut
        perc_SON_fut_minusmed = med_SON_fut-perc_SON_fut
        perc_DJF_fut_minusmed = med_DJF_fut-perc_DJF_fut
    
    else:
        perc_MAM_hist_minusmed = perc_MAM_hist-med_MAM_hist
        perc_JJA_hist_minusmed = perc_JJA_hist-med_JJA_hist
        perc_SON_hist_minusmed = perc_SON_hist-med_SON_hist
        perc_DJF_hist_minusmed = perc_DJF_hist-med_DJF_hist
    
        perc_MAM_fut_minusmed = perc_MAM_fut-med_MAM_fut
        perc_JJA_fut_minusmed = perc_JJA_fut-med_JJA_fut
        perc_SON_fut_minusmed = perc_SON_fut-med_SON_fut
        perc_DJF_fut_minusmed = perc_DJF_fut-med_DJF_fut
    
    perc_MAM_delta_minusmed = perc_MAM_fut_minusmed-perc_MAM_hist_minusmed
    perc_JJA_delta_minusmed = perc_JJA_fut_minusmed-perc_JJA_hist_minusmed
    perc_SON_delta_minusmed = perc_SON_fut_minusmed-perc_SON_hist_minusmed
    perc_DJF_delta_minusmed = perc_DJF_fut_minusmed-perc_DJF_hist_minusmed

if variable=="pr" or variable == "wind":

    perc_MAM_delta_perc = ((perc_MAM_fut-perc_MAM_hist)/perc_MAM_hist)*100
    perc_JJA_delta_perc = ((perc_JJA_fut-perc_JJA_hist)/perc_JJA_hist)*100
    perc_SON_delta_perc = ((perc_SON_fut-perc_SON_hist)/perc_SON_hist)*100
    perc_DJF_delta_perc = ((perc_DJF_fut-perc_DJF_hist)/perc_DJF_hist)*100


#%%

def save_minusmed(value,file):

    da = xr.DataArray(
        data=value,
        coords={
            'lat': (['lats', 'lats'], lats),
            'lon': (['lons', 'lons'], lons)
        },
        dims=['lats', 'lons'],
        name='delta_minusmed'
    )
        
    directory = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/percentile_deltas/rcp45/ncfiles/'+file
    da.to_netcdf(directory)
    

save_minusmed(perc_MAM_hist_minusmed,'{variable}_{perc}p_mam_hist_minusmed.nc')
save_minusmed(perc_JJA_hist_minusmed,'{variable}_{perc}p_jja_hist_minusmed.nc')
save_minusmed(perc_SON_hist_minusmed,'{variable}_{perc}p_son_hist_minusmed.nc')
save_minusmed(perc_DJF_hist_minusmed,'{variable}_{perc}p_djf_hist_minusmed.nc')

save_minusmed(perc_MAM_fut_minusmed,f'{variable}_{perc}p_mam_{period}_minusmed.nc')
save_minusmed(perc_JJA_fut_minusmed,f'{variable}_{perc}p_jja_{period}_minusmed.nc')
save_minusmed(perc_SON_fut_minusmed,f'{variable}_{perc}p_son_{period}_minusmed.nc')
save_minusmed(perc_DJF_fut_minusmed,f'{variable}_{perc}p_djf_{period}_minusmed.nc')

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
#%%

if minusmed == "no":
    def bootstrappin(hist,fut,iters):
    
        all_vals = np.concatenate((hist,fut),axis=0)
        
        def percentile_stat(data):
            return np.nanpercentile(data,perc,axis=0)
        
        rng = np.random.default_rng()
        
        percentile_diff = np.zeros((iters,np.shape(hist)[1],np.shape(hist)[2]))
        percentile_diff[:] = np.nan
        
        for k in range(iters):
            
            print(k)
            resampled_hist = rng.choice(all_vals,size=np.shape(all_vals)[0],replace=True,axis=0)
            resampled_fut = rng.choice(all_vals,size=np.shape(all_vals)[0],replace=True,axis=0)
        
            percentile_diff[k,:,:] = np.abs(percentile_stat(resampled_fut) - percentile_stat(resampled_hist))
        
        #the actual values
        perc_delta = np.abs(percentile_stat(fut)-percentile_stat(hist))
        
        p_value = np.sum(percentile_diff >= perc_delta,axis=0) / iters
       
        return(p_value)
    
    
    iters = 5#100
    
    p_value_MAM = bootstrappin(wrf_MAM_hist,wrf_MAM_fut,iters)
    p_value_JJA = bootstrappin(wrf_JJA_hist,wrf_JJA_fut,iters)
    p_value_SON = bootstrappin(wrf_SON_hist,wrf_SON_fut,iters)
    p_value_DJF = bootstrappin(wrf_DJF_hist,wrf_DJF_fut,iters)

#%%

if minusmed == "yes":
    def bootstrappin(hist_da,fut_da,iters):
    
        hist = hist_da.values
        fut = fut_da.values
        
        all_vals = np.concatenate((hist,fut),axis=0)
        
        def percentile_stat(data):
            if variable == "tmin":
                stat =  np.nanpercentile(data,50,axis=0) - np.nanpercentile(data,perc,axis=0)
            else:
                stat =  np.nanpercentile(data,perc,axis=0) -  np.nanpercentile(data,50,axis=0)
            return stat
        
        rng = np.random.default_rng()
        
        percentile_diff = np.zeros((iters,np.shape(hist)[1],np.shape(hist)[2]))
        percentile_diff[:] = np.nan
        
        for k in range(iters):
            
            print(k)
            resampled_hist = rng.choice(all_vals,size=np.shape(all_vals)[0],replace=True,axis=0)        
            resampled_fut = rng.choice(all_vals,size=np.shape(all_vals)[0],replace=True,axis=0)
            percentile_diff[k,:,:] = np.abs(percentile_stat(resampled_fut) - percentile_stat(resampled_hist))
        
        #the actual values
        perc_delta = np.abs(percentile_stat(fut)-percentile_stat(hist))
        
        p_value = np.sum(percentile_diff >= perc_delta,axis=0) / iters
       
        return(p_value)
    
    
    iters = 5
    
    p_value_MAM = bootstrappin(wrf_MAM_hist,wrf_MAM_fut,iters)
    p_value_JJA = bootstrappin(wrf_JJA_hist,wrf_JJA_fut,iters)
    p_value_SON = bootstrappin(wrf_SON_hist,wrf_SON_fut,iters)
    p_value_DJF = bootstrappin(wrf_DJF_hist,wrf_DJF_fut,iters)
    

#%%

def save_pvalue(pvalue,file):

    da = xr.DataArray(
        data=pvalue,
        coords={
            'lat': (['lats', 'lats'], lats),
            'lon': (['lons', 'lons'], lons)
        },
        dims=['lats', 'lons'],
        name='pvalue'
    )
        
    directory = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/ncfiles/pvalues/'+file
    da.to_netcdf(directory)
    
    #%%

save_pvalue(p_value_MAM,f'{variable}_{domain}_{perc}p_mam_minusmed_{period}.nc')
save_pvalue(p_value_JJA,f'{variable}_{domain}_{perc}p_jja_minusmed_{period}.nc')
save_pvalue(p_value_SON,f'{variable}_{domain}_{perc}p_son_minusmed_{period}.nc')
save_pvalue(p_value_DJF,f'{variable}_{domain}_{perc}p_djf_minusmed_{period}.nc')

#%%
#save_pvalue(p_value_MAM,f'{variable}_{perc}p_mam_{period}.nc')
#save_pvalue(p_value_JJA,f'{variable}_{perc}p_jja_{period}.nc')
#save_pvalue(p_value_SON,f'{variable}_{perc}p_son_{period}.nc')
#save_pvalue(p_value_DJF,f'{variable}_{perc}p_djf_{period}.nc')
#%%

directory = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/ncfiles/pvalues/'

p_value_MAM = xr.open_dataset(f"{directory}{variable}_{perc}p_mam_minusmed_{period}.nc")['pvalue'].values
p_value_JJA = xr.open_dataset(f"{directory}{variable}_{perc}p_jja_minusmed_{period}.nc")['pvalue'].values
p_value_SON = xr.open_dataset(f"{directory}{variable}_{perc}p_son_minusmed_{period}.nc")['pvalue'].values
p_value_DJF = xr.open_dataset(f"{directory}{variable}_{perc}p_djf_minusmed_{period}.nc")['pvalue'].values

#%%

directory = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/ncfiles/pvalues/'

p_value_MAM = xr.open_dataset(f"{directory}{variable}_{perc}p_mam_{period}.nc")['pvalue'].values
p_value_JJA = xr.open_dataset(f"{directory}{variable}_{perc}p_jja_{period}.nc")['pvalue'].values
p_value_SON = xr.open_dataset(f"{directory}{variable}_{perc}p_son_{period}.nc")['pvalue'].values
p_value_DJF = xr.open_dataset(f"{directory}{variable}_{perc}p_djf_{period}.nc")['pvalue'].values


#%%

def plot_map(gridded_data,p_value,seas,vmin,vmax,cmap):

    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
    
    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    
    masked_grid = p_value.copy()
    masked_grid[masked_grid>0.1] = np.nan
    ax1.pcolor(lons, lats, masked_grid, transform=ccrs.PlateCarree(), hatch='...', alpha=0,vmin=-5,vmax=5)
    mpl.rcParams['hatch.linewidth'] = 0.8
         

    #ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    #corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    #random_y_factor = -corner_y3[0]/12.5
    #random_x_factor = corner_x3[0]/65
    
       
            
    #ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    #ax1.text(-3700000, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
    
    #plt.title(make_title(seas),fontsize=20)
    
    
    #ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    ax1.set_extent([-131+1.4, -119-1.15, 46+0.4, 52-0.3], crs=ccrs.PlateCarree())
    
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', alpha=1,linewidth=1)
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,-49,4))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(0,81,4))

# =============================================================================
#     ax1.text(corner_x3[0]+length_x[2]*-0.175, corner_y3[0]+length_y[2]*0.13, '44$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-40,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*-0.175, corner_y3[0]+length_y[2]*0.78, '48$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-38,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.935, corner_y3[0]+length_y[2]*0.55, '52$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-30,alpha=0.8)
# 
#     ax1.text(corner_x3[0]+length_x[2]*0.1, corner_y3[0]+length_y[2]*1.01, '132$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.53, corner_y3[0]+length_y[2]*1.01, '128$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.875, corner_y3[0]+length_y[2]*1.01, '124$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.1, corner_y3[0]+length_y[2]*-0.08, '124$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.53, corner_y3[0]+length_y[2]*-0.08, '120$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.9, corner_y3[0]+length_y[2]*-0.08, '116$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
# 
# =============================================================================
    
    font=25
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(vmin, vmax+1, 2.5))
    cbar_ax.tick_params(labelsize=font)
    
    
    if variable == "tmax":
        if minusmed == "yes":
            cbar_ax.set_xlabel('$\Delta$ Tmax 95p-50p ($\degree$C)',size=25)
        elif minusmed == "no":
            cbar_ax.set_xlabel("Tmax $\Delta$ 95p ($\degree$C)",size=font) 
        #cbar_ax.set_xlabel('$\Delta$ Tmax 95p-50p ($\degree$C)',size=25)
        #cbar_ax.set_xlabel('Tmax ($\degree$C)',size=25)

    elif variable == "pr":
        #cbar_ax.set_xlabel("Precipitation $\Delta$ 95p (mm/day)",size=font)    
        cbar_ax.set_xlabel("Precipitation $\Delta$ 95p (%)",size=25)    

    elif variable == "wind":
        cbar_ax.set_xlabel("Wind Speed $\Delta$ 95p (%)",size=font)   
        #cbar_ax.set_xlabel("Wspd (m/s)",size=25)   

    elif variable == "t":
        cbar_ax.set_xlabel("Tas $\Delta$ 95p ($\degree$C)",size=25) 
        #cbar_ax.set_xlabel('Tas $\Delta$ 95p - $\Delta$ mean ($\degree$C)',size=font)
    elif variable == "tmin":
        if minusmed == "yes":
            cbar_ax.set_xlabel('$\Delta$ Tmin 50p-5p ($\degree$C)',size=font)
        elif minusmed == "no": 
            cbar_ax.set_xlabel("Tmin $\Delta$ 5p ($\degree$C)",size=25) 

    if minusmed == "yes":
        plt.savefig(f'/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/other/canesm2_wrf_{domain}_{variable}_{period}_{seas}_{str(perc)}_minusmed.png',bbox_inches='tight')
    elif minusmed == "no": 
        plt.savefig(f'/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/other/canesm2_wrf_{domain}_{variable}_{period}_{seas}_{str(perc)}.png',bbox_inches='tight')
    
    plt.close()



if variable in ["t", "tmax", "tmin"]:
# =============================================================================
#     vmin= -6
#     vmax = 6
#     #cmap = 'bwr'
#     t_colors = ['#142f60','#3465aa','#5392c1','#99c4dd','#d3e5f0','#f7f7f7',
#                         '#fadcc8','#eea785','#ce6451','#ab242f','#630921']
#     cmap = pltcol.LinearSegmentedColormap.from_list("custom", t_colors,N=26)
#     cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
#     cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=24)
#     cmap.set_over(t_colors[-1]) #add the max arrow color
#     cmap.set_under(t_colors[0]) #add the min arrow color
# =============================================================================
    
    if minusmed == "no": 
        vmin = 0
        vmax = 8
        #cmap = newcmp_t
        cmap = cm.get_cmap('YlOrRd', 16)

    elif minusmed == "yes": 
        vmin= -5
        vmax = 5
        cmap = plt.get_cmap('PuOr_r')
        colors = [cmap(i / (22 - 1)) for i in range(22)]
    
        cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors,N=22)
        cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
        cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=20)
        cmap.set_over(colors[-1]) #add the max arrow color
        cmap.set_under(colors[0]) #add the min arrow color


elif variable == "pr":
    colors_pr = ['#543005','#8c5109','#a4671b','#c7974a','#d4b775','#f5e7c6','#c7e7e2','#80cdc1','#35978f','#12766e','#01665e','#003c30']
    cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors_pr,N=18)
    cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
    cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=16)
    cmap.set_over(colors_pr[-1]) #add the max arrow color
    cmap.set_under(colors_pr[0]) #add the min arrow color
    
    vmin=-80
    vmax=80
    
elif variable == "wind":
    vmin= -20
    vmax = 20
# =============================================================================
#     #cmap = 'bwr'
#     t_colors = ['#142f60','#3465aa','#5392c1','#99c4dd','#d3e5f0','#f7f7f7',
#                         '#fadcc8','#eea785','#ce6451','#ab242f','#630921']
#     cmap = pltcol.LinearSegmentedColormap.from_list("custom", t_colors,N=14)
#     cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
#     cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=12)
#     cmap.set_over(t_colors[-1]) #add the max arrow color
#     cmap.set_under(t_colors[0]) #add the min arrow color
# =============================================================================

    colors_wspd_delta = ['#424c03','#41641a','#4b8c49','#79b17d','#aed0b2','#d7e3e0',
                         '#aec3d5','#7394b5','#3e6896','#294072','#2c194d'][::-1]
    
    cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors_wspd_delta,N=22)
    cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
    cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=20)
    cmap.set_over(colors_wspd_delta[-1]) #add the max arrow color
    cmap.set_under(colors_wspd_delta[0]) #add the min arrow color
#%%

if minusmed == "yes" and variable.startswith("t"):
    #plot_map(perc_ANN_delta_minusmed,p_value_ANN, "ANN", vmin,vmax,cmap)
    plot_map(perc_MAM_delta_minusmed,p_value_MAM, "MAM", vmin,vmax,cmap)
    plot_map(perc_JJA_delta_minusmed,p_value_JJA, "JJA", vmin,vmax,cmap)
    plot_map(perc_SON_delta_minusmed,p_value_SON, "SON", vmin,vmax,cmap)
    plot_map(perc_DJF_delta_minusmed,p_value_DJF, "DJF", vmin,vmax,cmap)

elif minusmed == "no" and variable.startswith("t"):
    plot_map(perc_MAM_delta,p_value_MAM, "MAM", vmin,vmax,cmap)
    plot_map(perc_JJA_delta,p_value_JJA, "JJA", vmin,vmax,cmap)
    plot_map(perc_SON_delta,p_value_SON, "SON", vmin,vmax,cmap)
    plot_map(perc_DJF_delta,p_value_DJF, "DJF", vmin,vmax,cmap)
    
elif variable in ['pr','wind']:
    plot_map(perc_MAM_delta_perc,p_value_MAM, "MAM", vmin,vmax,cmap)
    plot_map(perc_JJA_delta_perc,p_value_JJA, "JJA", vmin,vmax,cmap)
    plot_map(perc_SON_delta_perc,p_value_SON, "SON", vmin,vmax,cmap)
    plot_map(perc_DJF_delta_perc,p_value_DJF, "DJF", vmin,vmax,cmap)
