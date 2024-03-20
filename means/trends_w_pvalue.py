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
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic
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

variable = 'pr'
period = 'hist'

#%%

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

def get_wetdry_values(values,time):
    vals_WET = [vals for vals, date in zip(values, time) if date.month in [10,11,12,1,2,3]]
    vals_DRY = [vals for vals, date in zip(values, time) if date.month in [4,5,6,7,8,9]]

    return(np.array(vals_WET),np.array(vals_DRY))


def get_seas_dates(time):
    dates_MAM = [date for date in time if date.month in [3,4,5]]
    dates_JJA = [date for date in time if date.month in [6,7,8]]
    dates_SON = [date for date in time if date.month in [9,10,11]]
    dates_DJF = [date for date in time if date.month in [1,2,12]]
    
    return(np.array(dates_MAM),np.array(dates_JJA),np.array(dates_SON),np.array(dates_DJF))

def get_wetdry_dates(time):
    dates_WET = [date for date in time if date.month in [10,11,12,1,2,3]]
    dates_DRY = [date for date in time if date.month in [4,5,6,7,8,9]]
    
    return(np.array(dates_WET),np.array(dates_DRY))


def get_yearly_sums(values,time):
    yearly_sums = {}
    for value, date in zip(values,time):
        yearly_sums[date.year] = yearly_sums.get(date.year,0) + value

    return(np.array(list(yearly_sums.values())))
    
def get_yearly_means(values,time):
    yearly_mean = {}
    days={}
    for value, date in zip(values,time):
        yearly_mean[date.year] = yearly_mean.get(date.year,0) + value
        days[date.year] = days.get(date.year,0)+1

    for year in yearly_mean:
        yearly_mean[year] /= days[year]

    return(np.array(list(yearly_mean.values())))      
 #%%
    
if variable == "t":
    var = 'T2'
    filename = 't_d03_tas_daily'
elif variable == "pr":
    var = 'pr'
    filename = 'pr_d03_daily'
elif variable == "wind":
    var = 'wspd'
    filename = 'wspd_d03_mon'
    
wrf_d03_var_hist = Dataset(gridded_data_path + filename + '_hist.nc','r').variables[var][:]
wrf_d03_var_fut = Dataset(gridded_data_path + filename + '_' + period + '.nc','r').variables[var][:]


wrf_d03_time_hist = Dataset(gridded_data_path + filename + '_hist.nc','r').variables['time'][:]         
wrf_d03_time_fut = Dataset(gridded_data_path + filename + '_rcp45.nc','r').variables['time'][:]

wrf_d03_time_hist = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_hist]
wrf_d03_time_fut = [datetime.datetime(2046, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_fut]

#%%

wrf_d03_var_MAM_hist,wrf_d03_var_JJA_hist,wrf_d03_var_SON_hist,wrf_d03_var_DJF_hist = get_seas_values(wrf_d03_var_hist, wrf_d03_time_hist)
dates_MAM_hist,dates_JJA_hist,dates_SON_hist,dates_DJF_hist = get_seas_dates(wrf_d03_time_hist)

wrf_d03_var_MAM_fut,wrf_d03_var_JJA_fut,wrf_d03_var_SON_fut,wrf_d03_var_DJF_fut = get_seas_values(wrf_d03_var_fut, wrf_d03_time_fut)
dates_MAM_fut,dates_JJA_fut,dates_SON_fut,dates_DJF_fut = get_seas_dates(wrf_d03_time_fut)

means_ANN_yearly_hist = get_yearly_means(wrf_d03_var_hist,wrf_d03_time_hist)
means_MAM_yearly_hist = get_yearly_means(wrf_d03_var_MAM_hist,dates_MAM_hist)
means_JJA_yearly_hist = get_yearly_means(wrf_d03_var_JJA_hist,dates_JJA_hist)
means_SON_yearly_hist = get_yearly_means(wrf_d03_var_SON_hist,dates_SON_hist)
means_DJF_yearly_hist = get_yearly_means(wrf_d03_var_DJF_hist,dates_DJF_hist)

means_ANN_yearly_fut = get_yearly_means(wrf_d03_var_fut,wrf_d03_time_fut)
means_MAM_yearly_fut = get_yearly_means(wrf_d03_var_MAM_fut,dates_MAM_fut)
means_JJA_yearly_fut = get_yearly_means(wrf_d03_var_JJA_fut,dates_JJA_fut)
means_SON_yearly_fut = get_yearly_means(wrf_d03_var_SON_fut,dates_SON_fut)
means_DJF_yearly_fut = get_yearly_means(wrf_d03_var_DJF_fut,dates_DJF_fut)

#%%

# =============================================================================
# wrf_d03_var_WET_hist,wrf_d03_var_DRY_hist = get_wetdry_values(wrf_d03_var_hist, wrf_d03_time_hist)
# dates_WET_hist,dates_DRY_hist = get_wetdry_dates(wrf_d03_time_hist)
# 
# wrf_d03_var_WET_fut,wrf_d03_var_DRY_fut = get_wetdry_values(wrf_d03_var_fut, wrf_d03_time_fut)
# dates_WET_fut,dates_DRY_fut = get_wetdry_dates(wrf_d03_time_fut)
# 
# means_WET_yearly_hist = get_yearly_means(wrf_d03_var_WET_hist,dates_WET_hist)
# means_DRY_yearly_hist = get_yearly_means(wrf_d03_var_DRY_hist,dates_DRY_hist)
# 
# means_WET_yearly_fut = get_yearly_means(wrf_d03_var_WET_fut,dates_WET_fut)
# means_DRY_yearly_fut = get_yearly_means(dates_DRY_fut,dates_DRY_fut)
# 
# =============================================================================

#%%

means_ANN_fut = np.mean(means_ANN_yearly_fut,axis=0)
means_MAM_fut = np.mean(means_MAM_yearly_fut,axis=0)
means_JJA_fut = np.mean(means_JJA_yearly_fut,axis=0)
means_SON_fut = np.mean(means_SON_yearly_fut,axis=0)
means_DJF_fut = np.mean(means_DJF_yearly_fut,axis=0)

means_ANN_hist = np.mean(means_ANN_yearly_hist,axis=0)
means_MAM_hist = np.mean(means_MAM_yearly_hist,axis=0)
means_JJA_hist = np.mean(means_JJA_yearly_hist,axis=0)
means_SON_hist = np.mean(means_SON_yearly_hist,axis=0)
means_DJF_hist = np.mean(means_DJF_yearly_hist,axis=0)

if variable in ['t','wind']:
    means_ANN_delta = means_ANN_fut-means_ANN_hist
    means_MAM_delta = means_MAM_fut-means_MAM_hist
    means_JJA_delta = means_JJA_fut-means_JJA_hist
    means_SON_delta = means_SON_fut-means_SON_hist
    means_DJF_delta = means_DJF_fut-means_DJF_hist
    
elif variable=="pr":
    
    means_ANN_delta = ((means_ANN_fut-means_ANN_hist)/means_ANN_hist)*100
    means_MAM_delta = ((means_MAM_fut-means_MAM_hist)/means_MAM_hist)*100
    means_JJA_delta = ((means_JJA_fut-means_JJA_hist)/means_JJA_hist)*100
    means_SON_delta = ((means_SON_fut-means_SON_hist)/means_SON_hist)*100
    means_DJF_delta = ((means_DJF_fut-means_DJF_hist)/means_DJF_hist)*100
    
ttest_ANN = scipy.stats.ttest_ind(np.squeeze(means_ANN_yearly_hist), np.squeeze(means_ANN_yearly_fut),axis=0)
ttest_MAM = scipy.stats.ttest_ind(np.squeeze(means_MAM_yearly_hist), np.squeeze(means_MAM_yearly_fut),axis=0)
ttest_JJA = scipy.stats.ttest_ind(np.squeeze(means_JJA_yearly_hist), np.squeeze(means_JJA_yearly_fut),axis=0)
ttest_SON = scipy.stats.ttest_ind(np.squeeze(means_SON_yearly_hist), np.squeeze(means_SON_yearly_fut),axis=0)
ttest_DJF = scipy.stats.ttest_ind(np.squeeze(means_DJF_yearly_hist), np.squeeze(means_DJF_yearly_fut),axis=0)


#%%

# =============================================================================
# means_WET_fut = np.mean(means_WET_yearly_fut,axis=0)
# means_DRY_fut = np.mean(means_DRY_yearly_fut,axis=0)
# 
# means_WET_hist = np.mean(means_WET_yearly_hist,axis=0)
# means_DRY_hist = np.mean(means_DRY_yearly_hist,axis=0)
# 
# if variable in ['t','wind']:
#     means_WET_delta = means_WET_fut-means_WET_hist
#     means_DRY_delta = means_DRY_fut-means_DRY_hist
# 
# elif variable=="pr":
#     
#     means_WET_delta = ((means_WET_fut-means_WET_hist)/means_WET_hist)*100
#     means_DRY_delta = ((means_DRY_fut-means_DRY_hist)/means_DRY_hist)*100
# 
# ttest_WET = scipy.stats.ttest_ind(np.squeeze(means_WET_yearly_hist), np.squeeze(means_WET_yearly_fut),axis=0)
# ttest_DRY = scipy.stats.ttest_ind(np.squeeze(means_DRY_yearly_hist), np.squeeze(means_DRY_yearly_fut),axis=0)
# 
# 
# =============================================================================
#%%
def make_title(seas):
    if period == "rcp45":
        title_f = "RCP4.5"
    elif period == "rcp85":
        title_f = "RCP8.5"

    return(seas + " " + title_f + " 2046-2065 mean relative to 1986-2005")


def plot_map(gridded_data,pvalue,seas,vmin,vmax,cmap):
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    
# =============================================================================
#     masked_grid = pvalue.copy()
#     masked_grid[masked_grid>0.1] = np.nan
#     ax1.pcolor(lons, lats, masked_grid, transform=ccrs.PlateCarree(), hatch='...', alpha=0,vmin=vmin,vmax=vmax)
#     mpl.rcParams['hatch.linewidth'] = 0.8
# 
# =============================================================================
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
    if variable == "t":
        cbar_ax.set_xlabel("$\Delta$ Temperature ($\degree$C)",size=25) 
    elif variable == "pr":
        cbar_ax.set_xlabel("$\Delta$ Precipitation (%)",size=25)      
    elif variable == "wind":
        cbar_ax.set_xlabel("$\Delta$ Wind Speed (m/s)",size=25)   
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/future_changes/' + period + '_' + variable + '_' + seas + '.png',bbox_inches='tight')

    
    
#%%

t_colors = ['#f2f8d4','#fdce62','#f28a2c','#e84c0f','#bd1717','#6f150c','#741744']
newcmp_t = pltcol.LinearSegmentedColormap.from_list("custom", t_colors[1:-1],N=20) 
newcmp_t.set_over(t_colors[-1]) #add the max arrow color
newcmp_t.set_under(t_colors[0]) #add the min arrow color



#colors_pr = ['#dec17d','#e8dbb8','#f4f4f4','#a7e0da','#80ccc0','#40a898','#409498','#418898','#005e71']
colors_pr = ['#8c5109','#a4671b','#c7974a','#d4b775','#f5e7c6','#c7e7e2','#80cdc1','#35978f','#12766e','#01665e']
newcmp_pr = pltcol.LinearSegmentedColormap.from_list("custom", colors_pr,N=16) 
under = '#543005'
over = '#003c30'
newcmp_pr.set_over(over) #add the max arrow color
newcmp_pr.set_under(under) #add the min arrow color

#%%
if variable == "t":
    vmin= 0
    vmax = 5
    #cmap = newcmp_t
    cmap = cm.get_cmap('YlOrRd', 24)


elif variable == "pr":
    vmin= -80
    vmax = 80
    cmap = newcmp_pr
    
elif variable == "wind":
    vmin= -1
    vmax = 1
    cmap = 'bwr'
    

plot_map(means_ANN_delta,ttest_ANN.pvalue, "ANN",vmin,vmax,cmap)
plot_map(means_MAM_delta,ttest_MAM.pvalue, "MAM",vmin,vmax,cmap)
plot_map(means_JJA_delta,ttest_JJA.pvalue, "JJA",vmin,vmax,cmap)
plot_map(means_SON_delta,ttest_SON.pvalue, "SON",vmin,vmax,cmap)
plot_map(means_DJF_delta,ttest_DJF.pvalue, "DJF",vmin,vmax,cmap)

#plot_map(means_WET_delta,ttest_WET.pvalue, "SON",vmin,vmax,cmap)
#plot_map(means_DRY_delta,ttest_DRY.pvalue, "DJF",vmin,vmax,cmap)

