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

variable = 'wind'
period = 'rcp85'
perc = 95




gridded_data_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/'

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

lons = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['lon'][:]
lats = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['lat'][:]

def get_seas_values(values,time):
    vals_MAM = [vals for vals, date in zip(values, time) if date.month in [3,4,5]]
    vals_JJA = [vals for vals, date in zip(values, time) if date.month in [6,7,8]]
    vals_SON = [vals for vals, date in zip(values, time) if date.month in [9,10,11]]
    vals_DJF = [vals for vals, date in zip(values, time) if date.month in [1,2,12]]
    return(np.array(vals_MAM),np.array(vals_JJA),np.array(vals_SON),np.array(vals_DJF))

def get_seas_dates(time):
    dates_MAM = [date for date in time if date.month in [3,4,5]]
    dates_JJA = [date for date in time if date.month in [6,7,8]]
    dates_SON = [date for date in time if date.month in [9,10,11]]
    dates_DJF = [date for date in time if date.month in [1,2,12]]
    return(np.array(dates_MAM),np.array(dates_JJA),np.array(dates_SON),np.array(dates_DJF))
   
 #%%
    
if variable == "t":
    var = 'T2'
    filename = 't_d03_tas_daily'
elif variable == "pr":
    var = 'pr'
    filename = 'pr_d03_daily'
elif variable == "wind":
    var = 'wspd'
    filename = 'wind_d03_daily_wspd'
elif variable == "tmax":
    var = 'T2'
    filename = 't_d03_tmax_daily'
elif variable == "tmin":
    var = 'T2'
    filename = 't_d03_tmin_daily'
    
wrf_d03_var_hist = Dataset(gridded_data_path + filename + '_hist.nc','r').variables[var][:]
wrf_d03_var_fut = Dataset(gridded_data_path + filename + '_' + period + '.nc','r').variables[var][:]

wrf_d03_time_hist = Dataset(gridded_data_path + filename + '_hist.nc','r').variables['time'][:]         
wrf_d03_time_fut = Dataset(gridded_data_path + filename + '_rcp45.nc','r').variables['time'][:]

wrf_d03_time_hist = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_hist]
wrf_d03_time_fut = [datetime.datetime(2046, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_fut]

land_d03_time = np.tile(land_d03, (len(wrf_d03_time_hist), 1, 1))


wrf_d03_var_fut[land_d03_time==0] = np.nan
wrf_d03_var_hist[land_d03_time==0] = np.nan

#%%
wrf_d03_var_MAM_hist,wrf_d03_var_JJA_hist,wrf_d03_var_SON_hist,wrf_d03_var_DJF_hist = get_seas_values(wrf_d03_var_hist, wrf_d03_time_hist)
dates_MAM_hist,dates_JJA_hist,dates_SON_hist,dates_DJF_hist = get_seas_dates(wrf_d03_time_hist)

wrf_d03_var_MAM_fut,wrf_d03_var_JJA_fut,wrf_d03_var_SON_fut,wrf_d03_var_DJF_fut = get_seas_values(wrf_d03_var_fut, wrf_d03_time_fut)
dates_MAM_fut,dates_JJA_fut,dates_SON_fut,dates_DJF_fut = get_seas_dates(wrf_d03_time_fut)



if variable == "pr":
    wrf_d03_var_hist[wrf_d03_var_hist<1]=np.nan
    wrf_d03_var_MAM_hist[wrf_d03_var_MAM_hist<1]=np.nan
    wrf_d03_var_JJA_hist[wrf_d03_var_JJA_hist<1]=np.nan
    wrf_d03_var_SON_hist[wrf_d03_var_SON_hist<1]=np.nan
    wrf_d03_var_DJF_hist[wrf_d03_var_DJF_hist<1]=np.nan
    
    wrf_d03_var_fut[wrf_d03_var_fut<1]=np.nan
    wrf_d03_var_MAM_fut[wrf_d03_var_MAM_fut<1]=np.nan
    wrf_d03_var_JJA_fut[wrf_d03_var_JJA_fut<1]=np.nan
    wrf_d03_var_SON_fut[wrf_d03_var_SON_fut<1]=np.nan
    wrf_d03_var_DJF_fut[wrf_d03_var_DJF_fut<1]=np.nan
    

perc_ANN_hist = np.nanpercentile(wrf_d03_var_hist,perc,axis=0)
perc_MAM_hist = np.nanpercentile(wrf_d03_var_MAM_hist,perc,axis=0)
perc_JJA_hist = np.nanpercentile(wrf_d03_var_JJA_hist,perc,axis=0)
perc_SON_hist = np.nanpercentile(wrf_d03_var_SON_hist,perc,axis=0)
perc_DJF_hist = np.nanpercentile(wrf_d03_var_DJF_hist,perc,axis=0)

perc_ANN_fut = np.nanpercentile(wrf_d03_var_fut,perc,axis=0)
perc_MAM_fut = np.nanpercentile(wrf_d03_var_MAM_fut,perc,axis=0)
perc_JJA_fut = np.nanpercentile(wrf_d03_var_JJA_fut,perc,axis=0)
perc_SON_fut = np.nanpercentile(wrf_d03_var_SON_fut,perc,axis=0)
perc_DJF_fut = np.nanpercentile(wrf_d03_var_DJF_fut,perc,axis=0)

perc_ANN_delta = perc_ANN_fut-perc_ANN_hist
perc_MAM_delta = perc_MAM_fut-perc_MAM_hist
perc_JJA_delta = perc_JJA_fut-perc_JJA_hist
perc_SON_delta = perc_SON_fut-perc_SON_hist
perc_DJF_delta = perc_DJF_fut-perc_DJF_hist

if variable in ['t','tmax']:

    means_ANN_hist = np.nanmean(wrf_d03_var_hist,axis=0)
    means_MAM_hist = np.nanmean(wrf_d03_var_MAM_hist,axis=0)
    means_JJA_hist = np.nanmean(wrf_d03_var_JJA_hist,axis=0)
    means_SON_hist = np.nanmean(wrf_d03_var_SON_hist,axis=0)
    means_DJF_hist = np.nanmean(wrf_d03_var_DJF_hist,axis=0)
    
    means_ANN_fut = np.nanmean(wrf_d03_var_fut,axis=0)
    means_MAM_fut = np.nanmean(wrf_d03_var_MAM_fut,axis=0)
    means_JJA_fut = np.nanmean(wrf_d03_var_JJA_fut,axis=0)
    means_SON_fut = np.nanmean(wrf_d03_var_SON_fut,axis=0)
    means_DJF_fut = np.nanmean(wrf_d03_var_DJF_fut,axis=0)
    
    perc_ANN_hist_minusmean = perc_ANN_hist-means_ANN_hist
    perc_MAM_hist_minusmean = perc_MAM_hist-means_MAM_hist
    perc_JJA_hist_minusmean = perc_JJA_hist-means_JJA_hist
    perc_SON_hist_minusmean = perc_SON_hist-means_SON_hist
    perc_DJF_hist_minusmean = perc_DJF_hist-means_DJF_hist

    perc_ANN_fut_minusmean = perc_ANN_fut-means_ANN_fut
    perc_MAM_fut_minusmean = perc_MAM_fut-means_MAM_fut
    perc_JJA_fut_minusmean = perc_JJA_fut-means_JJA_fut
    perc_SON_fut_minusmean = perc_SON_fut-means_SON_fut
    perc_DJF_fut_minusmean = perc_DJF_fut-means_DJF_fut

    perc_ANN_delta_minusmean = perc_ANN_fut_minusmean-perc_ANN_hist_minusmean
    perc_MAM_delta_minusmean = perc_MAM_fut_minusmean-perc_MAM_hist_minusmean
    perc_JJA_delta_minusmean = perc_JJA_fut_minusmean-perc_JJA_hist_minusmean
    perc_SON_delta_minusmean = perc_SON_fut_minusmean-perc_SON_hist_minusmean
    perc_DJF_delta_minusmean = perc_DJF_fut_minusmean-perc_DJF_hist_minusmean


if variable=="pr":

    perc_ANN_delta_perc = ((perc_ANN_fut-perc_ANN_hist)/perc_ANN_hist)*100
    perc_MAM_delta_perc = ((perc_MAM_fut-perc_MAM_hist)/perc_MAM_hist)*100
    perc_JJA_delta_perc = ((perc_JJA_fut-perc_JJA_hist)/perc_JJA_hist)*100
    perc_SON_delta_perc = ((perc_SON_fut-perc_SON_hist)/perc_SON_hist)*100
    perc_DJF_delta_perc = ((perc_DJF_fut-perc_DJF_hist)/perc_DJF_hist)*100


#%%

def bootstrappin(hist,fut,iters):

    all_vals = np.concatenate((hist,fut),axis=0)
    
    def percentile_stat(data):
        return np.nanpercentile(data,95,axis=0)
    
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


iters = 100

p_value_MAM = bootstrappin(wrf_d03_var_MAM_hist,wrf_d03_var_MAM_fut,iters)
p_value_JJA = bootstrappin(wrf_d03_var_JJA_hist,wrf_d03_var_JJA_fut,iters)
p_value_SON = bootstrappin(wrf_d03_var_SON_hist,wrf_d03_var_SON_fut,iters)
p_value_DJF = bootstrappin(wrf_d03_var_DJF_hist,wrf_d03_var_DJF_fut,iters)
#p_value_ANN = bootstrappin(wrf_d03_var_hist,wrf_d03_var_fut,iters)

#%%

def bootstrappin(hist,fut,iters):

    all_vals = np.concatenate((hist,fut),axis=0)
    
    def percentile_stat(data):
        stat =  np.nanpercentile(data,95,axis=0) -  np.nanmean(data,axis=0)
        return stat
    
    rng = np.random.default_rng()
    
    percentile_diff = np.zeros((iters,np.shape(hist)[1],np.shape(hist)[2]))
    percentile_diff[:] = np.nan
    
    for k in range(iters):
        
        print(k)
        resampled_hist = rng.choice(all_vals,size=np.shape(all_vals)[0],replace=True,axis=0)        resampled_fut = rng.choice(all_vals,size=np.shape(all_vals)[0],replace=True,axis=0)
.    
        percentile_diff[k,:,:] = np.abs(percentile_stat(resampled_fut) - percentile_stat(resampled_hist))
    
    #the actual values
    perc_delta = np.abs(percentile_stat(fut)-percentile_stat(hist))
    
    p_value = np.sum(percentile_diff >= perc_delta,axis=0) / iters
   
    return(p_value)


iters = 200

p_value_MAM = bootstrappin(wrf_d03_var_MAM_hist,wrf_d03_var_MAM_fut,iters)
#p_value_JJA = bootstrappin(wrf_d03_var_JJA_hist,wrf_d03_var_JJA_fut,iters)
#p_value_SON = bootstrappin(wrf_d03_var_SON_hist,wrf_d03_var_SON_fut,iters)
#p_value_DJF = bootstrappin(wrf_d03_var_DJF_hist,wrf_d03_var_DJF_fut,iters)
#p_value_ANN = bootstrappin(wrf_d03_var_hist,wrf_d03_var_fut,iters)

#%%

def plot_map(gridded_data,p_value,seas,vmin,vmax,cmap):

    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
    
    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap, vmin=-vmax,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    
    masked_grid = p_value.copy()
    masked_grid[masked_grid>0.1] = np.nan
    ax1.pcolor(lons, lats, masked_grid, transform=ccrs.PlateCarree(), hatch='...', alpha=0,vmin=-5,vmax=5)
    mpl.rcParams['hatch.linewidth'] = 1.2
         
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
       
            
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    #ax1.text(-3700000, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
    
    #plt.title(make_title(seas),fontsize=20)
    
    
    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', alpha=1,linewidth=1)
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,-49,4))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(0,81,4))

    ax1.text(corner_x3[0]+length_x[2]*-0.175, corner_y3[0]+length_y[2]*0.13, '44$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-40,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*-0.175, corner_y3[0]+length_y[2]*0.78, '48$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-38,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.935, corner_y3[0]+length_y[2]*0.55, '52$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-30,alpha=0.8)

    ax1.text(corner_x3[0]+length_x[2]*0.1, corner_y3[0]+length_y[2]*1.01, '132$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.53, corner_y3[0]+length_y[2]*1.01, '128$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.875, corner_y3[0]+length_y[2]*1.01, '124$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.1, corner_y3[0]+length_y[2]*-0.08, '124$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.53, corner_y3[0]+length_y[2]*-0.08, '120$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.9, corner_y3[0]+length_y[2]*-0.08, '116$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)

    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(0, vmax+1, 0.5))
    cbar_ax.tick_params(labelsize=25)
    
    
    if variable == "tmax":
        #cbar_ax.set_xlabel("Tmax $\Delta$ 95p ($\degree$C)",size=25) 
        cbar_ax.set_xlabel('Tmax $\Delta$ 95p - $\Delta$ mean ($\degree$C)',size=25)
        #cbar_ax.set_xlabel('Tmax ($\degree$C)',size=25)

    elif variable == "pr":
        cbar_ax.set_xlabel("Precipitation $\Delta$ 95p (mm/day)",size=25)    
        #cbar_ax.set_xlabel("Precipitation $\Delta$ 95p (%)",size=25)    

    elif variable == "wind":
        cbar_ax.set_xlabel("Wspd $\Delta$ 95p (m/s)",size=25)   
        #cbar_ax.set_xlabel("Wspd (m/s)",size=25)   

    elif variable == "t":
        #cbar_ax.set_xlabel("Tas $\Delta$ 95p (deg C)",size=25) 
        cbar_ax.set_xlabel('Tas $\Delta$ 95p - $\Delta$ mean ($\degree$C)',size=25)
    elif variable == "tmin":
        #cbar_ax.set_xlabel("Tmin $\Delta$ 95p (deg C)",size=25) 
        cbar_ax.set_xlabel('Tmin $\Delta$ 95p - $\Delta$ mean ($\degree$C)',size=25)

    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/percentiles/' + period + '_' + variable + '_' + seas + '_' + str(perc) + 'p_pvalue.png',bbox_inches='tight')

colors_pr = ['#8c5109','#a4671b','#c7974a','#d4b775','#f5e7c6','#c7e7e2','#80cdc1','#35978f','#12766e','#01665e']
newcmp_pr = pltcol.LinearSegmentedColormap.from_list("custom", colors_pr,N=16) 
under = '#543005'
over = '#003c30'
newcmp_pr.set_over(over) #add the max arrow color
newcmp_pr.set_under(under) #add the min arrow color


vmin=-3
vmax=3
cmap='bwr'

#plot_map(perc_ANN_delta_minusmean,p_value_ANN, "ANN", vmin,vmax,cmap)
plot_map(perc_MAM_delta,p_value_MAM, "MAM", vmin,vmax,cmap)
#plot_map(perc_JJA_delta,p_value_JJA, "JJA", vmin,vmax,cmap)
#plot_map(perc_SON_delta,p_value_SON, "SON", vmin,vmax,cmap)
#plot_map(perc_DJF_delta,p_value_DJF, "DJF", vmin,vmax,cmap)


#%%

def make_title(seas):
    if period == "rcp45":
        title_f = "RCP4.5"
        #title_f = "Historical"

    elif period == "rcp85":
        #title_f = "Historical"
        title_f = "RCP8.5"

    return(seas + " " + title_f + " " + str(perc) + "p")


def plot_map(gridded_data,seas,vmin,vmax,cmap):
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)

    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    ax1.text(-3700000, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
    
    #plt.title(make_title(seas),fontsize=20)


    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(0, vmax+1, 0.5))
    cbar_ax.tick_params(labelsize=25)
    
    
    if variable == "tmax":
        #cbar_ax.set_xlabel("Tmax $\Delta$ 95p (deg C)",size=25) 
        #cbar_ax.set_xlabel('Tmax $\Delta$ 95p - $\Delta$ mean ($\degree$C)',size=25)
        cbar_ax.set_xlabel('Tmax ($\degree$C)',size=25)

    elif variable == "pr":
        #cbar_ax.set_xlabel("Precipitation $\Delta$ 95p (mm/day)",size=25)    
        #cbar_ax.set_xlabel("Precipitation $\Delta$ 95p (%)",size=25)    
        cbar_ax.set_xlabel("Precipitation 95p (mm/day)",size=25)    

    elif variable == "wind":
        cbar_ax.set_xlabel("Wspd $\Delta$ 95p (m/s)",size=25)   
        #cbar_ax.set_xlabel("Wspd (m/s)",size=25)   

    elif variable == "t":
        #cbar_ax.set_xlabel("Tas $\Delta$ 95p (deg C)",size=25) 
        cbar_ax.set_xlabel('Tas $\Delta$ 95p - $\Delta$ mean ($\degree$C)',size=25)
    elif variable == "tmin":
        #cbar_ax.set_xlabel("Tmin $\Delta$ 95p (deg C)",size=25) 
        cbar_ax.set_xlabel('Tmin $\Delta$ 95p - $\Delta$ mean ($\degree$C)',size=25)

    
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/percentiles/' + period + '_' + variable + '_' + seas + '_' + str(perc) + 'p_minusmean.png',bbox_inches='tight')
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/percentiles/' + period + '_' + variable + '_' + seas + '_' + str(perc) + 'p.png',bbox_inches='tight')
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/percentiles/' + period + '_' + variable + '_' + seas + '_' + str(perc) + 'p_abs.png',bbox_inches='tight')
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/percentiles/hist_' + variable + '_' + seas + '_' + str(perc) + 'p_abs.png',bbox_inches='tight')




colors_pr = ['#8c5109','#a4671b','#c7974a','#d4b775','#f5e7c6','#c7e7e2','#80cdc1','#35978f','#12766e','#01665e']
newcmp_pr = pltcol.LinearSegmentedColormap.from_list("custom", colors_pr,N=16) 
under = '#543005'
over = '#003c30'
newcmp_pr.set_over(over) #add the max arrow color
newcmp_pr.set_under(under) #add the min arrow color


if variable in ["t", "tmax", "tmin"]:
    vmin= -5
    vmax = 5
    cmap = 'bwr'
    #cmap = cm.get_cmap('YlOrRd', 24)


    vmin = 0 
    vmax = 45
    cmap='jet'
    
elif variable == "pr":
    vmin= 0
    vmax = 105
    cmap = 'gist_ncar'
    
elif variable == "wind":
    vmin= -3
    vmax = 3
    cmap = 'bwr'
    
    #vmin= 0
    #vmax = 15
    #cmap = 'jet'
    
#plot_map(means_ANN_hist, "ANN", vmin,vmax,cmap)
plot_map(perc_MAM_delta, "MAM", vmin,vmax,cmap)
plot_map(perc_JJA_delta, "JJA", vmin,vmax,cmap)
plot_map(perc_SON_delta, "SON", vmin,vmax,cmap)
plot_map(perc_DJF_delta, "DJF", vmin,vmax,cmap)


# =============================================================================
# plot_map(perc_ANN_delta_minusmean, "ANN", vmin,vmax,cmap)
# plot_map(perc_MAM_delta_minusmean, "MAM", vmin,vmax,cmap)
# plot_map(perc_JJA_delta_minusmean, "JJA", vmin,vmax,cmap)
# plot_map(perc_SON_delta_minusmean, "SON", vmin,vmax,cmap)
# plot_map(perc_DJF_delta_minusmean, "DJF", vmin,vmax,cmap)
# =============================================================================
# =============================================================================
# plot_map(perc_ANN_delta_perc, "ANN", vmin,vmax,cmap)
# plot_map(perc_MAM_delta_perc, "MAM", vmin,vmax,cmap)
# plot_map(perc_JJA_delta_perc, "JJA", vmin,vmax,cmap)
# plot_map(perc_SON_delta_perc, "SON", vmin,vmax,cmap)
# plot_map(perc_DJF_delta_perc, "DJF", vmin,vmax,cmap)
# =============================================================================
# =============================================================================
# plot_map(perc_ANN_delta, "ANN", vmin,vmax,cmap)
# plot_map(perc_MAM_delta, "MAM", vmin,vmax,cmap)
# plot_map(perc_JJA_delta, "JJA", vmin,vmax,cmap)
# plot_map(perc_SON_delta, "SON", vmin,vmax,cmap)
# plot_map(perc_DJF_delta, "DJF", vmin,vmax,cmap)
# =============================================================================
