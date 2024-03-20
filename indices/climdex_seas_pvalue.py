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

#%%

climdex = "wsdi"  #csdi, dsfreq, wsdi

period = 'rcp85' #rcp45, rcp85

base_period = "hist" #fut or hist



climdex_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/'

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

lons = Dataset(climdex_path + 'csdi_hist_mon.nc').variables['lon'][:]
lats = Dataset(climdex_path + 'csdi_hist_mon.nc').variables['lat'][:]


if climdex=="csdi":
    input_var = "csdi_6"
elif climdex=="dsfreq":
    input_var = "dry_spell_frequency"
elif climdex=="wsdi":
    input_var="warm_spell_duration_index"
    
def get_seas_values(values,time):
    vals_MAM = [vals for vals, date in zip(values, time) if date.month in [3,4,5]]
    vals_JJA = [vals for vals, date in zip(values, time) if date.month in [6,7,8]]
    vals_SON = [vals for vals, date in zip(values, time) if date.month in [9,10,11]]
    vals_DJF = [vals for vals, date in zip(values, time) if date.month in [1,2,12]]

    return(np.array(vals_MAM),np.array(vals_JJA),np.array(vals_SON),np.array(vals_DJF))
    
    
wrf_d03_time_hist = Dataset(climdex_path + '/csdi_hist_mon.nc','r').variables['time'][:]
wrf_d03_time_hist = [datetime.datetime(1986, 1, 1) + datetime.timedelta(days=int(days)) for days in wrf_d03_time_hist]


climdex_hist_mon = np.squeeze(Dataset(climdex_path + climdex + '_hist_mon.nc').variables[input_var][:])

climdex_MAM_hist,climdex_JJA_hist,climdex_SON_hist,climdex_DJF_hist = get_seas_values(climdex_hist_mon, wrf_d03_time_hist)

climdex_ANN_hist = np.sum(climdex_hist_mon.reshape(20, 12, 300, 300), axis=1)
climdex_MAM_hist = np.sum(climdex_MAM_hist.reshape(20, 3, 300, 300), axis=1)
climdex_JJA_hist = np.sum(climdex_JJA_hist.reshape(20, 3, 300, 300), axis=1)
climdex_SON_hist = np.sum(climdex_SON_hist.reshape(20, 3, 300, 300), axis=1)
climdex_DJF_hist = np.sum(climdex_DJF_hist.reshape(20, 3, 300, 300), axis=1)

# =============================================================================
# std_ANN_hist = np.std(climdex_ANN_hist,axis=0)
# std_MAM_hist = np.std(climdex_MAM_hist,axis=0)
# std_JJA_hist = np.std(climdex_JJA_hist,axis=0)
# std_SON_hist = np.std(climdex_SON_hist,axis=0)
# std_DJF_hist = np.std(climdex_DJF_hist,axis=0)
# =============================================================================

if period != "hist":
    wrf_d03_time_fut = Dataset(climdex_path + '/csdi_rcp45_mon_base_fut.nc','r').variables['time'][:]
    wrf_d03_time_fut = [datetime.datetime(1986, 1, 1) + datetime.timedelta(days=int(days)) for days in wrf_d03_time_fut]
    
    climdex_fut_mon = np.squeeze(Dataset(climdex_path + climdex + '_' + period + '_mon_base_' + base_period + '.nc').variables[input_var][:])
    
    climdex_MAM_fut,climdex_JJA_fut,climdex_SON_fut,climdex_DJF_fut = get_seas_values(climdex_fut_mon, wrf_d03_time_fut)
    
    climdex_ANN_fut = np.sum(climdex_fut_mon.reshape(20, 12, 300, 300), axis=1)
    climdex_MAM_fut = np.sum(climdex_MAM_fut.reshape(20, 3, 300, 300), axis=1)
    climdex_JJA_fut = np.sum(climdex_JJA_fut.reshape(20, 3, 300, 300), axis=1)
    climdex_SON_fut = np.sum(climdex_SON_fut.reshape(20, 3, 300, 300), axis=1)
    climdex_DJF_fut = np.sum(climdex_DJF_fut.reshape(20, 3, 300, 300), axis=1)
    
    climdex_ANN_delta = np.mean(climdex_ANN_fut,axis=0)-np.mean(climdex_ANN_hist,axis=0)
    climdex_MAM_delta = np.mean(climdex_MAM_fut,axis=0)-np.mean(climdex_MAM_hist,axis=0)
    climdex_JJA_delta = np.mean(climdex_JJA_fut,axis=0)-np.mean(climdex_JJA_hist,axis=0)
    climdex_SON_delta = np.mean(climdex_SON_fut,axis=0)-np.mean(climdex_SON_hist,axis=0)
    climdex_DJF_delta = np.mean(climdex_DJF_fut,axis=0)-np.mean(climdex_DJF_hist,axis=0)
    
# =============================================================================
#     std_ANN_fut = np.std(climdex_ANN_fut,axis=0)
#     std_MAM_fut = np.std(climdex_MAM_fut,axis=0)
#     std_JJA_fut = np.std(climdex_JJA_fut,axis=0)
#     std_SON_fut = np.std(climdex_SON_fut,axis=0)
#     std_DJF_fut = np.std(climdex_DJF_fut,axis=0)
# =============================================================================
    
    ttest_ANN = scipy.stats.ttest_ind(np.squeeze(climdex_ANN_hist), np.squeeze(climdex_ANN_fut),axis=0)
    ttest_MAM = scipy.stats.ttest_ind(np.squeeze(climdex_MAM_hist), np.squeeze(climdex_MAM_fut),axis=0)
    ttest_JJA = scipy.stats.ttest_ind(np.squeeze(climdex_JJA_hist), np.squeeze(climdex_JJA_fut),axis=0)
    ttest_SON = scipy.stats.ttest_ind(np.squeeze(climdex_SON_hist), np.squeeze(climdex_SON_fut),axis=0)
    ttest_DJF = scipy.stats.ttest_ind(np.squeeze(climdex_DJF_hist), np.squeeze(climdex_DJF_fut),axis=0)
#%%

def make_title(seas):
    if climdex=="csdi":
        climdex_f = "CSDI"
    elif climdex=="dsfreq":
        climdex_f = "DS Freq"
    elif climdex=="wsdi":
        climdex_f = "WSDI"
    
    if period == "hist":
        title_f = "Historical"
        years = "(1986-2005)"
        base_years = "1986-2005"

        return(climdex_f + " " + title_f + " " + seas + " " + years)

    else:
        if period == "rcp45":
            title_f = "RCP4.5"
        elif period == "rcp85":
            title_f = "RCP8.5"
        
        years = "(2046-2065)"
        if base_period == "fut":
            base_years = "2046-2065"
        elif base_period == "hist":
            base_years = "1986-2005"


        return(climdex_f + " " + title_f + " " + seas + " " + years + ", Base " + base_years)





def plot_climdex(gridded_data,pvalue,seas,vmin,vmax):
    if vmin==0:
        cmap='viridis'
        xlabel = 'Count (avg per year)'
    else:
        cmap='bwr'
        xlabel = 'Diff. Count (avg per year)'
        
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
    
    gridded_data[land_d03==0]=np.nan
    
    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap,vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    
    if period != "hist":
        masked_grid = pvalue.copy()
        masked_grid[masked_grid>0.1] = np.nan
        ax1.pcolor(lons, lats, masked_grid, transform=ccrs.PlateCarree(), hatch='...', alpha=0,vmin=vmin,vmax=vmax)
        mpl.rcParams['hatch.linewidth'] = 1.2
        
    
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', alpha=1,linewidth=1)
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,-49,4))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(0,81,4))


    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    ax1.text(-3700000, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
    
    plt.title(make_title(seas),fontsize=20)
    
    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(0, vmax+1, 0.5))
    cbar_ax.tick_params(labelsize=20)
    cbar_ax.set_xlabel(xlabel,size=20)         
  
    if period != "hist":
        plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/climdex/' + climdex + "_" + period + "_base_" + base_period + "_" + seas,bbox_inches='tight')
    else:
        plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/climdex/' + climdex + "_" + period + "_" + seas,bbox_inches='tight')

        

if period == "hist":
    vmin=0
    vmax=20
    
    plot_climdex(np.mean(climdex_ANN_hist,axis=0), None,"ANN", vmin*4,vmax*4)
    plot_climdex(np.mean(climdex_MAM_hist,axis=0), None,"MAM", vmin,vmax)
    plot_climdex(np.mean(climdex_JJA_hist,axis=0), None,"JJA", vmin,vmax)
    plot_climdex(np.mean(climdex_SON_hist,axis=0), None,"SON", vmin,vmax)
    plot_climdex(np.mean(climdex_DJF_hist,axis=0), None,"DJF", vmin,vmax)

else:
    vmin=-30
    vmax=30

    plot_climdex(climdex_ANN_delta, ttest_ANN.pvalue,"ANN", vmin*4,vmax*4)
    plot_climdex(climdex_MAM_delta, ttest_MAM.pvalue,"MAM", vmin,vmax)
    plot_climdex(climdex_JJA_delta, ttest_JJA.pvalue,"JJA", vmin,vmax)
    plot_climdex(climdex_SON_delta, ttest_SON.pvalue,"SON", vmin,vmax)
    plot_climdex(climdex_DJF_delta, ttest_DJF.pvalue,"DJF", vmin,vmax)


