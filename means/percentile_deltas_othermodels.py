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

variable = 'tmax'
period = 'rcp85'
minusmed = 'yes' #yes or not
model = 'canrcm4' #canesm2 or canrcm4


if variable == "tmin":
    perc = 5
else:
    perc = 95



#%%

if model == "canesm2":
    #canesm2_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/'
    canesm2_path = '/Volumes/EVA/gridded_model_data/CanESM2_raw/'
    
    
    canesm2_hist = Dataset(canesm2_path + variable + '_hist.nc','r')
    canesm2_fut = Dataset(canesm2_path + variable + '_' + period + '.nc','r')
    
    lats = canesm2_fut.variables['lat'][:]
    lons = canesm2_fut.variables['lon'][:]
    lons,lats = np.meshgrid(lons,lats)
        
elif model == "canrcm4":
    #canrcm4_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/'
    canrcm4_path = '/Volumes/EVA/gridded_model_data/CanRCM4/'

    canrcm4_hist = Dataset(canrcm4_path + variable + '_NAM22_hist.nc','r')
    canrcm4_fut = Dataset(canrcm4_path + variable + '_NAM22_' + period + '.nc','r')
    
    lats = canrcm4_fut.variables['lat'][:]
    lons = canrcm4_fut.variables['lon'][:]

#%% 365 day cal

def get_times_365cal(startyear,endyear):
    start_date = datetime.datetime(startyear, 1, 1)
    end_date = datetime.datetime(endyear, 12, 31)
    day = datetime.timedelta(days=1)
    
    time_array = []
    
    current_date = start_date
    while current_date <= end_date:
        if current_date.month != 2 or (current_date.month == 2 and current_date.day != 29):
            time_array.append(current_date)
        current_date += day
    
    return time_array

def get_seas_values(values,time):
    vals_MAM = [vals for vals, date in zip(values, time) if date.month in [3,4,5]]
    vals_JJA = [vals for vals, date in zip(values, time) if date.month in [6,7,8]]
    vals_SON = [vals for vals, date in zip(values, time) if date.month in [9,10,11]]
    vals_DJF = [vals for vals, date in zip(values, time) if date.month in [1,2,12]]

    return(np.array(vals_MAM),np.array(vals_JJA),np.array(vals_SON),np.array(vals_DJF))
    

def get_perc(seas,p):
    return np.percentile(seas,p,axis=0)
#%%
if model == "canrcm4":
    canrcm4_time_hist = get_times_365cal(1986,2005)
    canrcm4_time_fut = get_times_365cal(2046,2065)
    
    if variable=="tmax":
        canrcm4_var_hist = canrcm4_hist.variables["tasmax"][:]
        canrcm4_var_fut = canrcm4_fut.variables["tasmax"][:]
        
    elif variable=="tmin":
        canrcm4_var_hist = canrcm4_hist.variables["tasmin"][:]
        canrcm4_var_fut = canrcm4_fut.variables["tasmin"][:]
    
    elif variable=="wind":
        canrcm4_var_fut = canrcm4_fut.variables["sfcWind"][:]
        canrcm4_var_hist = canrcm4_hist.variables["sfcWind"][:]
    
    else:
        canrcm4_var_hist = canrcm4_hist.variables[variable][:]
        canrcm4_var_fut = canrcm4_fut.variables[variable][:]

    if variable == "pr":
        canrcm4_var_hist *= 86400
        canrcm4_var_fut *= 86400
    #elif variable.startswith("t"):
    #    canrcm4_var_hist += -273.15
    #    canrcm4_var_fut += -273.15
    
    canrcm4_var_MAM_hist,canrcm4_var_JJA_hist,canrcm4_var_SON_hist,canrcm4_var_DJF_hist = get_seas_values(canrcm4_var_hist, canrcm4_time_hist)
    canrcm4_var_MAM_fut,canrcm4_var_JJA_fut,canrcm4_var_SON_fut,canrcm4_var_DJF_fut = get_seas_values(canrcm4_var_fut, canrcm4_time_fut)

    perc_MAM_hist = get_perc(canrcm4_var_MAM_hist,perc)
    perc_JJA_hist = get_perc(canrcm4_var_JJA_hist,perc)
    perc_SON_hist = get_perc(canrcm4_var_SON_hist,perc)
    perc_DJF_hist = get_perc(canrcm4_var_DJF_hist,perc)

    perc_MAM_fut = get_perc(canrcm4_var_MAM_fut,perc)
    perc_JJA_fut = get_perc(canrcm4_var_JJA_fut,perc)
    perc_SON_fut = get_perc(canrcm4_var_SON_fut,perc)
    perc_DJF_fut = get_perc(canrcm4_var_DJF_fut,perc)
    
    med_MAM_hist = get_perc(canrcm4_var_MAM_hist,50)
    med_JJA_hist = get_perc(canrcm4_var_JJA_hist,50)
    med_SON_hist = get_perc(canrcm4_var_SON_hist,50)
    med_DJF_hist = get_perc(canrcm4_var_DJF_hist,50)
    
    med_MAM_fut = get_perc(canrcm4_var_MAM_fut,50)
    med_JJA_fut = get_perc(canrcm4_var_JJA_fut,50)
    med_SON_fut = get_perc(canrcm4_var_SON_fut,50)
    med_DJF_fut = get_perc(canrcm4_var_DJF_fut,50)
    
elif model == "canesm2":
    if variable.startswith("t"):
        canesm2_time_hist = get_times_365cal(1979,2005)
    else:
        canesm2_time_hist = get_times_365cal(1850,2005)

    canesm2_time_fut = get_times_365cal(2006,2100)
    
    index_start_hist = canesm2_time_hist.index(datetime.datetime(1986, 1, 1))
    index_end_hist = canesm2_time_hist.index(datetime.datetime(2005, 12, 31))
    index_start_fut = canesm2_time_fut.index(datetime.datetime(2046, 1, 1))
    index_end_fut = canesm2_time_fut.index(datetime.datetime(2065, 12, 31))
    
    if variable == "wind":
        canesm2_var_hist = canesm2_hist.variables['sfcWind'][index_start_hist:index_end_hist+1,:,:]
        canesm2_var_fut = canesm2_fut.variables['sfcWind'][index_start_fut:index_end_fut+1,:,:]
    elif variable == "tmin":
        canesm2_var_hist = canesm2_hist.variables['tasmin'][index_start_hist:index_end_hist+1,:,:]
        canesm2_var_fut = canesm2_fut.variables['tasmin'][index_start_fut:index_end_fut+1,:,:]
    elif variable == "tmax":
        canesm2_var_hist = canesm2_hist.variables['tasmax'][index_start_hist:index_end_hist+1,:,:]
        canesm2_var_fut = canesm2_fut.variables['tasmax'][index_start_fut:index_end_fut+1,:,:]
    else:
        canesm2_var_hist = canesm2_hist.variables[variable][index_start_hist:index_end_hist+1,:,:]
        canesm2_var_fut = canesm2_fut.variables[variable][index_start_fut:index_end_fut+1,:,:]
        
    canesm2_time_hist = get_times_365cal(1986,2005)
    canesm2_time_fut = get_times_365cal(2046,2065)

    canesm2_var_MAM_hist,canesm2_var_JJA_hist,canesm2_var_SON_hist,canesm2_var_DJF_hist = get_seas_values(canesm2_var_hist, canesm2_time_hist)
    canesm2_var_MAM_fut,canesm2_var_JJA_fut,canesm2_var_SON_fut,canesm2_var_DJF_fut = get_seas_values(canesm2_var_fut, canesm2_time_fut)

    perc_MAM_hist = get_perc(canesm2_var_MAM_hist,perc)
    perc_JJA_hist = get_perc(canesm2_var_JJA_hist,perc)
    perc_SON_hist = get_perc(canesm2_var_SON_hist,perc)
    perc_DJF_hist = get_perc(canesm2_var_DJF_hist,perc)
    
    perc_MAM_fut = get_perc(canesm2_var_MAM_fut,perc)
    perc_JJA_fut = get_perc(canesm2_var_JJA_fut,perc)
    perc_SON_fut = get_perc(canesm2_var_SON_fut,perc)
    perc_DJF_fut = get_perc(canesm2_var_DJF_fut,perc)

    med_MAM_hist = get_perc(canesm2_var_MAM_hist,50)
    med_JJA_hist = get_perc(canesm2_var_JJA_hist,50)
    med_SON_hist = get_perc(canesm2_var_SON_hist,50)
    med_DJF_hist = get_perc(canesm2_var_DJF_hist,50)
    
    med_MAM_fut = get_perc(canesm2_var_MAM_fut,50)
    med_JJA_fut = get_perc(canesm2_var_JJA_fut,50)
    med_SON_fut = get_perc(canesm2_var_SON_fut,50)
    med_DJF_fut = get_perc(canesm2_var_DJF_fut,50)

#%%
perc_MAM_delta = perc_MAM_fut-perc_MAM_hist
perc_JJA_delta = perc_JJA_fut-perc_JJA_hist
perc_SON_delta = perc_SON_fut-perc_SON_hist
perc_DJF_delta = perc_DJF_fut-perc_DJF_hist

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
    
    
    

if minusmed == "yes":
    def bootstrappin(hist,fut,iters):
        
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

iters = 5#100

if model == "canesm2":
    p_value_MAM = bootstrappin(canesm2_var_MAM_hist,canesm2_var_MAM_fut,iters)
    p_value_JJA = bootstrappin(canesm2_var_JJA_hist,canesm2_var_JJA_fut,iters)
    p_value_SON = bootstrappin(canesm2_var_SON_hist,canesm2_var_SON_fut,iters)
    p_value_DJF = bootstrappin(canesm2_var_DJF_hist,canesm2_var_DJF_fut,iters)


elif model == "canrcm4":
    p_value_MAM = bootstrappin(canrcm4_var_MAM_hist,canrcm4_var_MAM_fut,iters)
    p_value_JJA = bootstrappin(canrcm4_var_JJA_hist,canrcm4_var_JJA_fut,iters)
    p_value_SON = bootstrappin(canrcm4_var_SON_hist,canrcm4_var_SON_fut,iters)
    p_value_DJF = bootstrappin(canrcm4_var_DJF_hist,canrcm4_var_DJF_fut,iters)


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
        plt.savefig(f'/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/other/{model}_{variable}_{period}_{seas}_{str(perc)}_minusmed.png',bbox_inches='tight')
    elif minusmed == "no": 
        plt.savefig(f'/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends_extremes/other/{model}_{variable}_{period}_{seas}_{str(perc)}.png',bbox_inches='tight')
    
    
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
