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

stat = 'mean'

#%%

if stat == "mean" and variable != "wind":
    if variable=="pr":
        varkey = 'rel'
    else:
        varkey=''
        
    cmip5_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CMIP5_ensemble/'
    cordex_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/NA-CORDEX_ensemble/'
    
    cmip5_MAM = Dataset(cmip5_path + 'CMIP5_' + variable + '_change_' + period + '_MAM.nc','r')
    cmip5_JJA = Dataset(cmip5_path + 'CMIP5_' + variable + '_change_' + period + '_JJA.nc','r')
    cmip5_SON = Dataset(cmip5_path + 'CMIP5_' + variable + '_change_' + period + '_SON.nc','r')
    cmip5_DJF = Dataset(cmip5_path + 'CMIP5_' + variable + '_change_' + period + '_DJF.nc','r')
    
    cordex_MAM = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_MAM.nc','r')
    cordex_JJA = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_JJA.nc','r')
    cordex_SON = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_SON.nc','r')
    cordex_DJF = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_DJF.nc','r')
    
    cmip5_var_MAM = cmip5_MAM.variables[variable + '_' + varkey + 'anom'][:]
    cmip5_var_JJA = cmip5_JJA.variables[variable + '_' + varkey + 'anom'][:]
    cmip5_var_SON = cmip5_SON.variables[variable + '_' + varkey + 'anom'][:]
    cmip5_var_DJF = cmip5_DJF.variables[variable + '_' + varkey + 'anom'][:]
    
    cmip5_lats = cmip5_MAM.variables['lat'][:]
    cmip5_lons = cmip5_MAM.variables['lon'][:]
    
    cordex_var_MAM = cordex_MAM.variables[variable + '_' + varkey + 'anom'][:]
    cordex_var_JJA = cordex_JJA.variables[variable + '_' + varkey + 'anom'][:]
    cordex_var_SON = cordex_SON.variables[variable + '_' + varkey + 'anom'][:]
    cordex_var_DJF = cordex_DJF.variables[variable + '_' + varkey + 'anom'][:]
    
    cordex_lats = cordex_MAM.variables['lat'][:]
    cordex_lons = cordex_MAM.variables['lon'][:]



    canesm2_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/'
    
    canesm2_hist = Dataset(canesm2_path + variable + '_hist.nc','r')
    canesm2_fut = Dataset(canesm2_path + variable + '_' + period + '.nc','r')
    
    canesm2_lats = canesm2_fut.variables['lat'][:]
    canesm2_lons = canesm2_fut.variables['lon'][:]
    canesm2_lons,canesm2_lats = np.meshgrid(canesm2_lons,canesm2_lats)

#%%
canrcm4_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/'

if variable == "wind":

    canrcm4_u_hist = Dataset(canrcm4_path  + 'uas_NAM22_hist.nc','r')
    canrcm4_u_fut = Dataset(canrcm4_path  + 'uas_NAM22_' + period + '.nc','r')

    canrcm4_v_hist = Dataset(canrcm4_path  + 'vas_NAM22_hist.nc','r')
    canrcm4_v_fut = Dataset(canrcm4_path  + 'vas_NAM22_' + period + '.nc','r')

    canrcm4_lats = canrcm4_v_fut.variables['lat'][:]
    canrcm4_lons = canrcm4_v_fut.variables['lon'][:]

else:    
    canrcm4_hist = Dataset(canrcm4_path + variable + '_NAM22_hist.nc','r')
    canrcm4_fut = Dataset(canrcm4_path + variable + '_NAM22_' + period + '.nc','r')
    
    canrcm4_lats = canrcm4_fut.variables['lat'][:]
    canrcm4_lons = canrcm4_fut.variables['lon'][:]

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
    
#%%
canrcm4_time_hist = get_times_365cal(1986,2005)
canrcm4_time_fut = get_times_365cal(2046,2065)

if variable=="tmax":
    canrcm4_var_hist = canrcm4_hist.variables[variable][:]
    canrcm4_var_fut = canrcm4_fut.variables["tasmax"][:]

elif variable=="wind":
    canrcm4_var_u_fut = canrcm4_u_fut.variables["uas"][:]
    canrcm4_var_v_fut = canrcm4_v_fut.variables["vas"][:]
    canrcm4_var_u_hist = canrcm4_u_hist.variables["uas"][:]
    canrcm4_var_v_hist = canrcm4_v_hist.variables["vas"][:]

    canrcm4_var_fut = np.sqrt(canrcm4_var_v_fut**2 + canrcm4_var_u_fut**2)
    canrcm4_var_hist = np.sqrt(canrcm4_var_v_hist**2 + canrcm4_var_u_hist**2)
        
else:
    canrcm4_var_hist = canrcm4_hist.variables[variable][:]
    canrcm4_var_fut = canrcm4_fut.variables[variable][:]

if variable == "pr":
    canrcm4_var_hist *= 86400
    canrcm4_var_fut *= 86400
elif variable =="tmax":
    canrcm4_var_fut += -273.15
    
    #%%
if stat == "mean" and variable != "wind":
    canesm2_time_hist = get_times_365cal(1850,2005)
    canesm2_time_fut= get_times_365cal(2006,2100)
    
    index_start_hist = canesm2_time_hist.index(datetime.datetime(1986, 1, 1))
    index_end_hist = canesm2_time_hist.index(datetime.datetime(2005, 12, 31))
    index_start_fut = canesm2_time_fut.index(datetime.datetime(2046, 1, 1))
    index_end_fut = canesm2_time_fut.index(datetime.datetime(2065, 12, 31))
    
    canesm2_var_hist = canesm2_hist.variables[variable][index_start_hist:index_end_hist+1,:,:]
    canesm2_var_fut = canesm2_fut.variables[variable][index_start_fut:index_end_fut+1,:,:]
    
    canesm2_time_hist = get_times_365cal(1986,2005)
    canesm2_time_fut = get_times_365cal(2046,2065)

#%%
def get_seas_values(values,time):
    vals_MAM = [vals for vals, date in zip(values, time) if date.month in [3,4,5]]
    vals_JJA = [vals for vals, date in zip(values, time) if date.month in [6,7,8]]
    vals_SON = [vals for vals, date in zip(values, time) if date.month in [9,10,11]]
    vals_DJF = [vals for vals, date in zip(values, time) if date.month in [1,2,12]]

    return(np.array(vals_MAM),np.array(vals_JJA),np.array(vals_SON),np.array(vals_DJF))

canrcm4_var_MAM_hist,canrcm4_var_JJA_hist,canrcm4_var_SON_hist,canrcm4_var_DJF_hist = get_seas_values(canrcm4_var_hist, canrcm4_time_hist)
canrcm4_var_MAM_fut,canrcm4_var_JJA_fut,canrcm4_var_SON_fut,canrcm4_var_DJF_fut = get_seas_values(canrcm4_var_fut, canrcm4_time_fut)

#%%
if stat=="mean" and variable!="wind":
    canesm2_var_MAM_hist,canesm2_var_JJA_hist,canesm2_var_SON_hist,canesm2_var_DJF_hist = get_seas_values(canesm2_var_hist, canesm2_time_hist)
    canesm2_var_MAM_fut,canesm2_var_JJA_fut,canesm2_var_SON_fut,canesm2_var_DJF_fut = get_seas_values(canesm2_var_fut, canesm2_time_fut)

#%%

if stat == "mean":
    if variable != "pr":
        canrcm4_var_MAM = np.mean(canrcm4_var_MAM_fut,axis=0) - np.mean(canrcm4_var_MAM_hist,axis=0)
        canrcm4_var_JJA = np.mean(canrcm4_var_JJA_fut,axis=0) - np.mean(canrcm4_var_JJA_hist,axis=0)
        canrcm4_var_SON = np.mean(canrcm4_var_SON_fut,axis=0) - np.mean(canrcm4_var_SON_hist,axis=0)
        canrcm4_var_DJF = np.mean(canrcm4_var_DJF_fut,axis=0) - np.mean(canrcm4_var_DJF_hist,axis=0)
        
        #canesm2_var_MAM = np.mean(canesm2_var_MAM_fut,axis=0) - np.mean(canesm2_var_MAM_hist,axis=0)
        #canesm2_var_JJA = np.mean(canesm2_var_JJA_fut,axis=0) - np.mean(canesm2_var_JJA_hist,axis=0)
        #canesm2_var_SON = np.mean(canesm2_var_SON_fut,axis=0) - np.mean(canesm2_var_SON_hist,axis=0)
        #canesm2_var_DJF = np.mean(canesm2_var_DJF_fut,axis=0) - np.mean(canesm2_var_DJF_hist,axis=0)
    else:
        canrcm4_var_MAM = 100*(np.mean(canrcm4_var_MAM_fut,axis=0) - np.mean(canrcm4_var_MAM_hist,axis=0))/np.mean(canrcm4_var_MAM_hist,axis=0)
        canrcm4_var_JJA = 100*(np.mean(canrcm4_var_JJA_fut,axis=0) - np.mean(canrcm4_var_JJA_hist,axis=0))/np.mean(canrcm4_var_JJA_hist,axis=0)
        canrcm4_var_SON = 100*(np.mean(canrcm4_var_SON_fut,axis=0) - np.mean(canrcm4_var_SON_hist,axis=0))/np.mean(canrcm4_var_SON_hist,axis=0)
        canrcm4_var_DJF = 100*(np.mean(canrcm4_var_DJF_fut,axis=0) - np.mean(canrcm4_var_DJF_hist,axis=0))/np.mean(canrcm4_var_DJF_hist,axis=0)
    
        #canesm2_var_MAM = 100*(np.mean(canesm2_var_MAM_fut,axis=0) - np.mean(canesm2_var_MAM_hist,axis=0))/np.mean(canesm2_var_MAM_hist,axis=0)
        #canesm2_var_JJA = 100*(np.mean(canesm2_var_JJA_fut,axis=0) - np.mean(canesm2_var_JJA_hist,axis=0))/np.mean(canesm2_var_JJA_hist,axis=0)
        #canesm2_var_SON = 100*(np.mean(canesm2_var_SON_fut,axis=0) - np.mean(canesm2_var_SON_hist,axis=0))/np.mean(canesm2_var_SON_hist,axis=0)
        #canesm2_var_DJF = 100*(np.mean(canesm2_var_DJF_fut,axis=0) - np.mean(canesm2_var_DJF_hist,axis=0))/np.mean(canesm2_var_DJF_hist,axis=0)

elif stat == "95_perc":
        canrcm4_var_MAM = np.percentile(canrcm4_var_MAM_fut,95,axis=0) - np.percentile(canrcm4_var_MAM_hist,95,axis=0)
        canrcm4_var_JJA = np.percentile(canrcm4_var_JJA_fut,95,axis=0) - np.percentile(canrcm4_var_JJA_hist,95,axis=0)
        canrcm4_var_SON = np.percentile(canrcm4_var_SON_fut,95,axis=0) - np.percentile(canrcm4_var_SON_hist,95,axis=0)
        canrcm4_var_DJF = np.percentile(canrcm4_var_DJF_fut,95,axis=0) - np.percentile(canrcm4_var_DJF_hist,95,axis=0)
        
#%%

def plot_map(gridded_data,lons,lats,seas,vmin,vmax,cmap,folder):

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
    #ax1.text(-3700000, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
    
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
    

    if variable == "pr":
        cbar_ax.set_xlabel("Precipitation change (mm/day)",size=25)    

    elif variable == "wind":
        cbar_ax.set_xlabel("Wind speed change (m/s)",size=25)   

    elif variable == "tas":
        cbar_ax.set_xlabel('Temperature change ($\degree$C)',size=25)
    elif variable == "tmax":
        cbar_ax.set_xlabel('Tmax change ($\degree$C)',size=25)

    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/future_changes/' + folder + "/" + period + '_' + variable + '_' + seas + '_' + stat  + '_change.png',bbox_inches='tight')
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/percentiles/' + folder + "/" + period + '_' + variable + '_' + seas + '_' + stat  + '_change.png',bbox_inches='tight')



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

if variable == "tas" or variable=="tmax":
    
    if stat == "mean":
        vmin= 0
        vmax = 5
        #cmap = newcmp_t
        cmap = cm.get_cmap('YlOrRd', 24)
    elif stat == "95_perc":
        vmin=-5
        vmax=5
        cmap='bwr'

elif variable == "pr":
    if stat == "mean":
        vmin= -80
        vmax = 80
        cmap = newcmp_pr
    elif stat == "95_perc":
        vmin= -20
        vmax = 20
        cmap = newcmp_pr
    
    
elif variable == "wind":
    vmin= -1
    vmax = 1
    cmap = 'bwr'
    


# =============================================================================
#
# plot_map(cmip5_var_MAM,cmip5_lons,cmip5_lats, "MAM", vmin,vmax,cmap,'cmip5_ens','mean')
# plot_map(cmip5_var_JJA,cmip5_lons,cmip5_lats, "JJA", vmin,vmax,cmap,'cmip5_ens','mean')
# plot_map(cmip5_var_SON,cmip5_lons,cmip5_lats, "SON", vmin,vmax,cmap,'cmip5_ens','mean')
# plot_map(cmip5_var_DJF,cmip5_lons,cmip5_lats, "DJF", vmin,vmax,cmap,'cmip5_ens','mean')
# 
# plot_map(cordex_var_MAM,cordex_lons,cordex_lats, "MAM", vmin,vmax,cmap,'cordex_ens','mean')
# plot_map(cordex_var_JJA,cordex_lons,cordex_lats, "JJA", vmin,vmax,cmap,'cordex_ens','mean')
# plot_map(cordex_var_SON,cordex_lons,cordex_lats, "SON", vmin,vmax,cmap,'cordex_ens','mean')
# plot_map(cordex_var_DJF,cordex_lons,cordex_lats, "DJF", vmin,vmax,cmap,'cordex_ens','mean')
# 
# =============================================================================

plot_map(canrcm4_var_MAM,canrcm4_lons,canrcm4_lats, "MAM", vmin,vmax,cmap,'canrcm4')
plot_map(canrcm4_var_JJA,canrcm4_lons,canrcm4_lats, "JJA", vmin,vmax,cmap,'canrcm4')
plot_map(canrcm4_var_SON,canrcm4_lons,canrcm4_lats, "SON", vmin,vmax,cmap,'canrcm4')
plot_map(canrcm4_var_DJF,canrcm4_lons,canrcm4_lats, "DJF", vmin,vmax,cmap,'canrcm4')

# =============================================================================
# plot_map(canesm2_var_MAM,canesm2_lons,canesm2_lats, "MAM", vmin,vmax,cmap,'canesm2')
# plot_map(canesm2_var_JJA,canesm2_lons,canesm2_lats, "JJA", vmin,vmax,cmap,'canesm2')
# plot_map(canesm2_var_SON,canesm2_lons,canesm2_lats, "SON", vmin,vmax,cmap,'canesm2')
# plot_map(canesm2_var_DJF,canesm2_lons,canesm2_lats, "DJF", vmin,vmax,cmap,'canesm2')
# 
# =============================================================================
