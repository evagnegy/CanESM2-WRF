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

variable = 'pr'
period = 'rcp45'

stat = '95_perc'

#%%

if variable=="pr":
    varkey = variable+'_rel'
elif variable=="wind":
    varkey = 'sfcWind_rel'
else:
    varkey=variable + "_"
    
if variable=="wind":
    cmip = 'CMIP6'
else:
    cmip = 'CMIP5'
        
#%%
if stat == "mean":

        
    cmip5_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/'+cmip+'_ensemble/'
    cordex_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/NA-CORDEX_ensemble/'
    

    cmip5_MAM = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_MAM.nc','r')
    cmip5_JJA = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_JJA.nc','r')
    cmip5_SON = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_SON.nc','r')
    cmip5_DJF = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_DJF.nc','r')

    cmip5_var_MAM = cmip5_MAM.variables[varkey + 'anom'][:]
    cmip5_var_JJA = cmip5_JJA.variables[varkey + 'anom'][:]
    cmip5_var_SON = cmip5_SON.variables[varkey + 'anom'][:]
    cmip5_var_DJF = cmip5_DJF.variables[varkey + 'anom'][:]
    
    cmip5_lats = cmip5_MAM.variables['lat'][:]
    cmip5_lons = cmip5_MAM.variables['lon'][:]
        

        
    cordex_MAM = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_MAM.nc','r')
    cordex_JJA = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_JJA.nc','r')
    cordex_SON = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_SON.nc','r')
    cordex_DJF = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_DJF.nc','r')
    
    cordex_var_MAM = cordex_MAM.variables[varkey + 'anom'][:]
    cordex_var_JJA = cordex_JJA.variables[varkey + 'anom'][:]
    cordex_var_SON = cordex_SON.variables[varkey + 'anom'][:]
    cordex_var_DJF = cordex_DJF.variables[varkey + 'anom'][:]
    
    cordex_lats = cordex_MAM.variables['lat'][:]
    cordex_lons = cordex_MAM.variables['lon'][:]


#%%

canesm2_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/'

canesm2_hist = Dataset(canesm2_path + variable + '_hist.nc','r')
canesm2_fut = Dataset(canesm2_path + variable + '_' + period + '.nc','r')

canesm2_lats = canesm2_fut.variables['lat'][:]
canesm2_lons = canesm2_fut.variables['lon'][:]
canesm2_lons,canesm2_lats = np.meshgrid(canesm2_lons,canesm2_lats)
    
#%%
canrcm4_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/'

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
    canrcm4_var_fut = canrcm4_fut.variables["sfcWind"][:]
    canrcm4_var_hist = canrcm4_hist.variables["sfcWind"][:]

else:
    canrcm4_var_hist = canrcm4_hist.variables[variable][:]
    canrcm4_var_fut = canrcm4_fut.variables[variable][:]

if variable == "pr":
    canrcm4_var_hist *= 86400
    canrcm4_var_fut *= 86400
elif variable =="tmax":
    canrcm4_var_fut += -273.15
    
    #%%
if stat == "mean":
    canesm2_time_hist = get_times_365cal(1850,2005)
    canesm2_time_fut= get_times_365cal(2006,2100)
    
    index_start_hist = canesm2_time_hist.index(datetime.datetime(1986, 1, 1))
    index_end_hist = canesm2_time_hist.index(datetime.datetime(2005, 12, 31))
    index_start_fut = canesm2_time_fut.index(datetime.datetime(2046, 1, 1))
    index_end_fut = canesm2_time_fut.index(datetime.datetime(2065, 12, 31))
    
    if variable == "wind":
        canesm2_var_hist = canesm2_hist.variables['sfcWind'][index_start_hist:index_end_hist+1,:,:]
        canesm2_var_fut = canesm2_fut.variables['sfcWind'][index_start_fut:index_end_fut+1,:,:]
    else:
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
if stat=="mean":
    canesm2_var_MAM_hist,canesm2_var_JJA_hist,canesm2_var_SON_hist,canesm2_var_DJF_hist = get_seas_values(canesm2_var_hist, canesm2_time_hist)
    canesm2_var_MAM_fut,canesm2_var_JJA_fut,canesm2_var_SON_fut,canesm2_var_DJF_fut = get_seas_values(canesm2_var_fut, canesm2_time_fut)

#%%

if stat == "mean":
    if variable.startswith('t'):
        canrcm4_var_MAM = np.mean(canrcm4_var_MAM_fut,axis=0) - np.mean(canrcm4_var_MAM_hist,axis=0)
        canrcm4_var_JJA = np.mean(canrcm4_var_JJA_fut,axis=0) - np.mean(canrcm4_var_JJA_hist,axis=0)
        canrcm4_var_SON = np.mean(canrcm4_var_SON_fut,axis=0) - np.mean(canrcm4_var_SON_hist,axis=0)
        canrcm4_var_DJF = np.mean(canrcm4_var_DJF_fut,axis=0) - np.mean(canrcm4_var_DJF_hist,axis=0)
        
        canesm2_var_MAM = np.mean(canesm2_var_MAM_fut,axis=0) - np.mean(canesm2_var_MAM_hist,axis=0)
        canesm2_var_JJA = np.mean(canesm2_var_JJA_fut,axis=0) - np.mean(canesm2_var_JJA_hist,axis=0)
        canesm2_var_SON = np.mean(canesm2_var_SON_fut,axis=0) - np.mean(canesm2_var_SON_hist,axis=0)
        canesm2_var_DJF = np.mean(canesm2_var_DJF_fut,axis=0) - np.mean(canesm2_var_DJF_hist,axis=0)
    else: #wind and pr
        canrcm4_var_MAM = 100*(np.mean(canrcm4_var_MAM_fut,axis=0) - np.mean(canrcm4_var_MAM_hist,axis=0))/np.mean(canrcm4_var_MAM_hist,axis=0)
        canrcm4_var_JJA = 100*(np.mean(canrcm4_var_JJA_fut,axis=0) - np.mean(canrcm4_var_JJA_hist,axis=0))/np.mean(canrcm4_var_JJA_hist,axis=0)
        canrcm4_var_SON = 100*(np.mean(canrcm4_var_SON_fut,axis=0) - np.mean(canrcm4_var_SON_hist,axis=0))/np.mean(canrcm4_var_SON_hist,axis=0)
        canrcm4_var_DJF = 100*(np.mean(canrcm4_var_DJF_fut,axis=0) - np.mean(canrcm4_var_DJF_hist,axis=0))/np.mean(canrcm4_var_DJF_hist,axis=0)
    
        canesm2_var_MAM = 100*(np.mean(canesm2_var_MAM_fut,axis=0) - np.mean(canesm2_var_MAM_hist,axis=0))/np.mean(canesm2_var_MAM_hist,axis=0)
        canesm2_var_JJA = 100*(np.mean(canesm2_var_JJA_fut,axis=0) - np.mean(canesm2_var_JJA_hist,axis=0))/np.mean(canesm2_var_JJA_hist,axis=0)
        canesm2_var_SON = 100*(np.mean(canesm2_var_SON_fut,axis=0) - np.mean(canesm2_var_SON_hist,axis=0))/np.mean(canesm2_var_SON_hist,axis=0)
        canesm2_var_DJF = 100*(np.mean(canesm2_var_DJF_fut,axis=0) - np.mean(canesm2_var_DJF_hist,axis=0))/np.mean(canesm2_var_DJF_hist,axis=0)

elif stat == "95_perc":
        canrcm4_var_MAM = np.percentile(canrcm4_var_MAM_fut,95,axis=0) - np.percentile(canrcm4_var_MAM_hist,95,axis=0)
        canrcm4_var_JJA = np.percentile(canrcm4_var_JJA_fut,95,axis=0) - np.percentile(canrcm4_var_JJA_hist,95,axis=0)
        canrcm4_var_SON = np.percentile(canrcm4_var_SON_fut,95,axis=0) - np.percentile(canrcm4_var_SON_hist,95,axis=0)
        canrcm4_var_DJF = np.percentile(canrcm4_var_DJF_fut,95,axis=0) - np.percentile(canrcm4_var_DJF_hist,95,axis=0)
        
#%%

if variable=="wind":
    varkey='sfcWind'
else:
    varkey=variable
    
cmip5_MAM_robust = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_MAM_robustness.nc','r')
cmip5_JJA_robust = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_JJA_robustness.nc','r')
cmip5_SON_robust = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_SON_robustness.nc','r')
cmip5_DJF_robust = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_DJF_robustness.nc','r')

cmip5_var_MAM_robust = cmip5_MAM_robust.variables[varkey+'_anom_robustness'][:]
cmip5_var_JJA_robust = cmip5_JJA_robust.variables[varkey+'_anom_robustness'][:]
cmip5_var_SON_robust = cmip5_SON_robust.variables[varkey+'_anom_robustness'][:]
cmip5_var_DJF_robust = cmip5_DJF_robust.variables[varkey+'_anom_robustness'][:]

cmip5_MAM_consensus = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_MAM_consensus.nc','r')
cmip5_JJA_consensus = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_JJA_consensus.nc','r')
cmip5_SON_consensus = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_SON_consensus.nc','r')
cmip5_DJF_consensus = Dataset(cmip5_path + cmip +'_' + variable + '_change_' + period + '_DJF_consensus.nc','r')

cmip5_var_MAM_consensus = cmip5_MAM_consensus.variables[varkey+'_anom_consensus'][:]
cmip5_var_JJA_consensus = cmip5_JJA_consensus.variables[varkey+'_anom_consensus'][:]
cmip5_var_SON_consensus = cmip5_SON_consensus.variables[varkey+'_anom_consensus'][:]
cmip5_var_DJF_consensus = cmip5_DJF_consensus.variables[varkey+'_anom_consensus'][:]

#%%

if variable=="wind":
    varkey='sfcWind'
else:
    varkey=variable
    
cordex_MAM_robust = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_MAM_robustness.nc','r')
cordex_JJA_robust = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_JJA_robustness.nc','r')
cordex_SON_robust = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_SON_robustness.nc','r')
cordex_DJF_robust = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_DJF_robustness.nc','r')

cordex_var_MAM_robust = cordex_MAM_robust.variables[varkey+'_anom_robustness'][:]
cordex_var_JJA_robust = cordex_JJA_robust.variables[varkey+'_anom_robustness'][:]
cordex_var_SON_robust = cordex_SON_robust.variables[varkey+'_anom_robustness'][:]
cordex_var_DJF_robust = cordex_DJF_robust.variables[varkey+'_anom_robustness'][:]

cordex_MAM_consensus = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_MAM_consensus.nc','r')
cordex_JJA_consensus = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_JJA_consensus.nc','r')
cordex_SON_consensus = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_SON_consensus.nc','r')
cordex_DJF_consensus = Dataset(cordex_path + 'CORDEX_' + variable + '_change_' + period + '_DJF_consensus.nc','r')

cordex_var_MAM_consensus = cordex_MAM_consensus.variables[varkey+'_anom_consensus'][:]
cordex_var_JJA_consensus = cordex_JJA_consensus.variables[varkey+'_anom_consensus'][:]
cordex_var_SON_consensus = cordex_SON_consensus.variables[varkey+'_anom_consensus'][:]
cordex_var_DJF_consensus = cordex_DJF_consensus.variables[varkey+'_anom_consensus'][:]

#%%
canesm2_ttest_MAM = scipy.stats.ttest_ind(np.squeeze(canesm2_var_MAM_hist), np.squeeze(canesm2_var_MAM_fut),axis=0)
canesm2_ttest_JJA = scipy.stats.ttest_ind(np.squeeze(canesm2_var_JJA_hist), np.squeeze(canesm2_var_JJA_fut),axis=0)
canesm2_ttest_SON = scipy.stats.ttest_ind(np.squeeze(canesm2_var_SON_hist), np.squeeze(canesm2_var_SON_fut),axis=0)
canesm2_ttest_DJF = scipy.stats.ttest_ind(np.squeeze(canesm2_var_DJF_hist), np.squeeze(canesm2_var_DJF_fut),axis=0)

canrcm4_ttest_MAM = scipy.stats.ttest_ind(np.squeeze(canrcm4_var_MAM_hist), np.squeeze(canrcm4_var_MAM_fut),axis=0)
canrcm4_ttest_JJA = scipy.stats.ttest_ind(np.squeeze(canrcm4_var_JJA_hist), np.squeeze(canrcm4_var_JJA_fut),axis=0)
canrcm4_ttest_SON = scipy.stats.ttest_ind(np.squeeze(canrcm4_var_SON_hist), np.squeeze(canrcm4_var_SON_fut),axis=0)
canrcm4_ttest_DJF = scipy.stats.ttest_ind(np.squeeze(canrcm4_var_DJF_hist), np.squeeze(canrcm4_var_DJF_fut),axis=0)

#%%

def plot_map(gridded_data,pvalue,pvalue2,lons,lats,seas,vmin,vmax,cmap,folder):

    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
    
    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    

    masked_grid = pvalue.copy()
    if folder.startswith("can"):
        masked_grid[masked_grid>0.1] = np.nan
    else:
        masked_grid[(pvalue2<0.8) & (masked_grid<0.66)] = np.nan
    ax1.pcolor(lons, lats, masked_grid, transform=ccrs.PlateCarree(), hatch='...', alpha=0,vmin=vmin,vmax=vmax)
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
    
    #ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    ax1.set_extent([-131+1.4, -119-1.15, 46+0.4, 52-0.3], crs=ccrs.PlateCarree())
    
    
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

    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/trends/other/' + folder + "_" + period + '_' + variable + '_' + seas + '_' + stat  + '_change.png',bbox_inches='tight')

    plt.close()


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
    vmin= -20
    vmax = 20
    #cmap = cmap = cm.get_cmap('bwr', 24)
    #t_colors = ['#142f60','#3465aa','#5392c1','#99c4dd','#d3e5f0','#f7f7f7',
    #                    '#fadcc8','#eea785','#ce6451','#ab242f','#630921']
    
    colors_wspd_delta = ['#424c03','#41641a','#4b8c49','#79b17d','#aed0b2','#d7e3e0',
                         '#aec3d5','#7394b5','#3e6896','#294072','#2c194d'][::-1]
    
    cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors_wspd_delta,N=22)
    cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
    cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=20)
    cmap.set_over(colors_wspd_delta[-1]) #add the max arrow color
    cmap.set_under(colors_wspd_delta[0]) #add the min arrow color
    

#%%

if variable=="wind":
    folder='cmip6_ens'
else:
    folder='cmip5_ens'
    
plot_map(cmip5_var_MAM,cmip5_var_MAM_robust,cmip5_var_MAM_consensus,cmip5_lons,cmip5_lats, "MAM", vmin,vmax,cmap,folder)
plot_map(cmip5_var_JJA,cmip5_var_JJA_robust,cmip5_var_JJA_consensus,cmip5_lons,cmip5_lats, "JJA", vmin,vmax,cmap,folder)
# =============================================================================
# plot_map(cmip5_var_SON,cmip5_var_SON_robust,cmip5_var_SON_consensus,cmip5_lons,cmip5_lats, "SON", vmin,vmax,cmap,folder)
# plot_map(cmip5_var_DJF,cmip5_var_DJF_robust,cmip5_var_DJF_consensus,cmip5_lons,cmip5_lats, "DJF", vmin,vmax,cmap,folder)
# 
# plot_map(cordex_var_MAM,cordex_var_MAM_robust,cordex_var_MAM_consensus,cordex_lons,cordex_lats, "MAM", vmin,vmax,cmap,'cordex_ens')
# plot_map(cordex_var_JJA,cordex_var_JJA_robust,cordex_var_JJA_consensus,cordex_lons,cordex_lats, "JJA", vmin,vmax,cmap,'cordex_ens')
# plot_map(cordex_var_SON,cordex_var_SON_robust,cordex_var_SON_consensus,cordex_lons,cordex_lats, "SON", vmin,vmax,cmap,'cordex_ens')
# plot_map(cordex_var_DJF,cordex_var_DJF_robust,cordex_var_DJF_consensus,cordex_lons,cordex_lats, "DJF", vmin,vmax,cmap,'cordex_ens')
# =============================================================================
#%%
plot_map(canrcm4_var_MAM,canrcm4_ttest_MAM.pvalue,[],canrcm4_lons,canrcm4_lats, "MAM", vmin,vmax,cmap,'canrcm4')
plot_map(canrcm4_var_JJA,canrcm4_ttest_JJA.pvalue,[],canrcm4_lons,canrcm4_lats, "JJA", vmin,vmax,cmap,'canrcm4')
plot_map(canrcm4_var_SON,canrcm4_ttest_SON.pvalue,[],canrcm4_lons,canrcm4_lats, "SON", vmin,vmax,cmap,'canrcm4')
plot_map(canrcm4_var_DJF,canrcm4_ttest_DJF.pvalue,[],canrcm4_lons,canrcm4_lats, "DJF", vmin,vmax,cmap,'canrcm4')

plot_map(canesm2_var_MAM,canesm2_ttest_MAM.pvalue,[],canesm2_lons,canesm2_lats, "MAM", vmin,vmax,cmap,'canesm2')
plot_map(canesm2_var_JJA,canesm2_ttest_JJA.pvalue,[],canesm2_lons,canesm2_lats, "JJA", vmin,vmax,cmap,'canesm2')
plot_map(canesm2_var_SON,canesm2_ttest_SON.pvalue,[],canesm2_lons,canesm2_lats, "SON", vmin,vmax,cmap,'canesm2')
plot_map(canesm2_var_DJF,canesm2_ttest_DJF.pvalue,[],canesm2_lons,canesm2_lats, "DJF", vmin,vmax,cmap,'canesm2')

