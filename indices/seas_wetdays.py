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

#%%

period = 'rcp45' # hist, rcp45, rcp85

#ignored if period=hist
fut_type = 'delta' #delta, abs 

#%%

gridded_data_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/'

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

lons = Dataset(gridded_data_path + 'pr_d03_daily_hist.nc').variables['lon'][:]
lats = Dataset(gridded_data_path + 'pr_d03_daily_hist.nc').variables['lat'][:]

def calculate_seas_wetdays():

    def get_seas_values(values,time):
        vals_MAM = [vals for vals, date in zip(values, time) if date.month in [3,4,5]]
        vals_JJA = [vals for vals, date in zip(values, time) if date.month in [6,7,8]]
        vals_SON = [vals for vals, date in zip(values, time) if date.month in [9,10,11]]
        vals_DJF = [vals for vals, date in zip(values, time) if date.month in [1,2,12]]
    
        return(np.array(vals_MAM),np.array(vals_JJA),np.array(vals_SON),np.array(vals_DJF))
                   
    if period =='hist' or fut_type == "delta":
        
        wrf_d03_time_hist = Dataset(gridded_data_path + '/pr_d03_daily_hist.nc','r').variables['time'][:]
        wrf_d03_time_hist = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_hist]
    
        daily_pr_hist = Dataset(gridded_data_path + '/pr_d03_daily_hist.nc').variables['pr'][:]
        
        pr_MAM_hist,pr_JJA_hist,pr_SON_hist,pr_DJF_hist = get_seas_values(daily_pr_hist, wrf_d03_time_hist)
        
        #wet_days_ANN_hist = np.sum(daily_pr_hist>1,axis=0)/20
        #wet_days_MAM_hist = np.sum(pr_MAM_hist>1,axis=0)/20
        #wet_days_JJA_hist = np.sum(pr_JJA_hist>1,axis=0)/20
        #wet_days_SON_hist = np.sum(pr_SON_hist>1,axis=0)/20
        #wet_days_DJF_hist = np.sum(pr_DJF_hist>1,axis=0)/20
        
        daily_pr_hist[daily_pr_hist<1]=0
        pr_MAM_hist[pr_MAM_hist<1]=0
        pr_JJA_hist[pr_JJA_hist<1]=0
        pr_SON_hist[pr_SON_hist<1]=0
        pr_DJF_hist[pr_DJF_hist<1]=0
        wet_days_ANN_hist = np.mean(daily_pr_hist,axis=0)
        wet_days_MAM_hist = np.mean(pr_MAM_hist,axis=0)
        wet_days_JJA_hist = np.mean(pr_JJA_hist,axis=0)
        wet_days_SON_hist = np.mean(pr_SON_hist,axis=0)
        wet_days_DJF_hist = np.mean(pr_DJF_hist,axis=0)
        
        if period =='hist':
            return(wet_days_ANN_hist,wet_days_MAM_hist,wet_days_JJA_hist,wet_days_SON_hist,wet_days_DJF_hist)
   
    if period != "hist":
        
        wrf_d03_time_fut = Dataset(gridded_data_path + '/pr_d03_daily_rcp45.nc','r').variables['time'][:]
        wrf_d03_time_fut = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_fut]
    
        daily_pr_fut = Dataset(gridded_data_path + '/pr_d03_daily_' + period + '.nc').variables['pr'][:]

        pr_MAM_fut,pr_JJA_fut,pr_SON_fut,pr_DJF_fut = get_seas_values(daily_pr_fut, wrf_d03_time_fut)

        #wet_days_ANN_fut = np.sum(daily_pr_fut>1,axis=0)/20
        #wet_days_MAM_fut = np.sum(pr_MAM_fut>1,axis=0)/20
        #wet_days_JJA_fut = np.sum(pr_JJA_fut>1,axis=0)/20
        #wet_days_SON_fut = np.sum(pr_SON_fut>1,axis=0)/20
        #wet_days_DJF_fut = np.sum(pr_DJF_fut>1,axis=0)/20
    
        daily_pr_fut[daily_pr_fut<1]=0
        pr_MAM_fut[pr_MAM_fut<1]=0
        pr_JJA_fut[pr_JJA_fut<1]=0
        pr_SON_fut[pr_SON_fut<1]=0
        pr_DJF_fut[pr_DJF_fut<1]=0
        wet_days_ANN_fut = np.mean(daily_pr_fut,axis=0)
        wet_days_MAM_fut = np.mean(pr_MAM_fut,axis=0)
        wet_days_JJA_fut = np.mean(pr_JJA_fut,axis=0)
        wet_days_SON_fut = np.mean(pr_SON_fut,axis=0)
        wet_days_DJF_fut = np.mean(pr_DJF_fut,axis=0)
        
        if fut_type=="abs":
            return(wet_days_ANN_fut,wet_days_MAM_fut,wet_days_JJA_fut,wet_days_SON_fut,wet_days_DJF_fut)
        
        elif fut_type=="delta":
            wetdays_ANN_fut_delta = wet_days_ANN_fut-wet_days_ANN_hist
            wetdays_MAM_fut_delta = wet_days_MAM_fut-wet_days_MAM_hist
            wetdays_JJA_fut_delta = wet_days_JJA_fut-wet_days_JJA_hist
            wetdays_SON_fut_delta = wet_days_SON_fut-wet_days_SON_hist
            wetdays_DJF_fut_delta = wet_days_DJF_fut-wet_days_DJF_hist
            
            return(wetdays_ANN_fut_delta,wetdays_MAM_fut_delta,wetdays_JJA_fut_delta,wetdays_SON_fut_delta,wetdays_DJF_fut_delta)

    
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
    
    gridded_data[land_d03==0]=np.nan
    
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
    #cbar_ax.set_xlabel(xlabel + " (avg. per year)",size=20)   
    cbar_ax.set_xlabel("SDII diff mm/day",size=20)        
  
def make_title(seas):
    #climdex_f = "Wet Days"
    climdex_f = "SDII"
    
    if period == "hist":
        title_f = "Historical"

    else:
        if period == "rcp45":
            title_f = "RCP4.5"
        elif period == "rcp85":
            title_f = "RCP8.5"

    return(climdex_f + " " + seas + " " + title_f)


count_event_peryear_ANN,count_event_peryear_MAM,count_event_peryear_JJA,count_event_peryear_SON,count_event_peryear_DJF = calculate_seas_wetdays()
#%%
vmin=-5
vmax=5
plot_climdex(count_event_peryear_ANN, make_title("ANN"),vmin,vmax)
#plot_climdex(count_event_peryear_MAM, make_title("MAM"),vmin/4,vmax/4)
#plot_climdex(count_event_peryear_JJA, make_title("JJA"),vmin/4,vmax/4)
#plot_climdex(count_event_peryear_SON, make_title("SON"),vmin/4,vmax/4)
#plot_climdex(count_event_peryear_DJF, make_title("DJF"),vmin/4,vmax/4)

plot_climdex(count_event_peryear_MAM, make_title("MAM"),vmin,vmax)
plot_climdex(count_event_peryear_JJA, make_title("JJA"),vmin,vmax)
plot_climdex(count_event_peryear_SON, make_title("SON"),vmin,vmax)
plot_climdex(count_event_peryear_DJF, make_title("DJF"),vmin,vmax)