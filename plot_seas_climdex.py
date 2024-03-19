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

climdex = "wsdi"  #csdi, dsfreq, wsdi

period = 'rcp85' # hist, rcp45, rcp85

#ignored if period=hist
fut_type = 'delta' #delta, abs 

#%%

climdex_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/'

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

lons = Dataset(climdex_path + 'csdi_hist_mon.nc').variables['lon'][:]
lats = Dataset(climdex_path + 'csdi_hist_mon.nc').variables['lat'][:]

def calculate_seas_climdex():

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
        

    def get_seas_dates(values,time):
        dates_MAM = [date for date in time if date.month in [3,4,5]]
        dates_JJA = [date for date in time if date.month in [6,7,8]]
        dates_SON = [date for date in time if date.month in [9,10,11]]
        dates_DJF = [date for date in time if date.month in [1,2,12]]
        
        return(np.array(dates_MAM),np.array(dates_JJA),np.array(dates_SON),np.array(dates_DJF))

    if period =='hist' or fut_type == "delta":
        
        wrf_d03_time_hist = Dataset(climdex_path + '/csdi_hist_mon.nc','r').variables['time'][:]
        wrf_d03_time_hist = [datetime.datetime(1986, 1, 1) + datetime.timedelta(days=int(days)) for days in wrf_d03_time_hist]
    
        climdex_hist_mon = np.squeeze(Dataset(climdex_path + climdex + '_hist_mon.nc').variables[input_var][:])
        
        climdex_MAM_hist,climdex_JJA_hist,climdex_SON_hist,climdex_DJF_hist = get_seas_values(climdex_hist_mon, wrf_d03_time_hist)
        
        climdex_ANN_hist_avg = np.sum(climdex_hist_mon,axis=0)/20
        climdex_MAM_hist_avg = np.sum(climdex_MAM_hist,axis=0)/20
        climdex_JJA_hist_avg = np.sum(climdex_JJA_hist,axis=0)/20
        climdex_SON_hist_avg = np.sum(climdex_SON_hist,axis=0)/20
        climdex_DJF_hist_avg = np.sum(climdex_DJF_hist,axis=0)/20
    
        if period =='hist':
            return(climdex_ANN_hist_avg,climdex_MAM_hist_avg,climdex_JJA_hist_avg,climdex_SON_hist_avg,climdex_DJF_hist_avg)
   
    if period != "hist":
        
        wrf_d03_time_fut = Dataset(climdex_path + '/csdi_rcp45_mon.nc','r').variables['time'][:]
        wrf_d03_time_fut = [datetime.datetime(1986, 1, 1) + datetime.timedelta(days=int(days)) for days in wrf_d03_time_fut]
    
        climdex_fut_mon = np.squeeze(Dataset(climdex_path + climdex + '_' + period + '_mon.nc').variables[input_var][:])
        
        climdex_MAM_fut,climdex_JJA_fut,climdex_SON_fut,climdex_DJF_fut = get_seas_values(climdex_fut_mon, wrf_d03_time_fut)
        
        climdex_ANN_fut_avg = np.sum(climdex_fut_mon,axis=0)/20
        climdex_MAM_fut_avg = np.sum(climdex_MAM_fut,axis=0)/20
        climdex_JJA_fut_avg = np.sum(climdex_JJA_fut,axis=0)/20
        climdex_SON_fut_avg = np.sum(climdex_SON_fut,axis=0)/20
        climdex_DJF_fut_avg = np.sum(climdex_DJF_fut,axis=0)/20
    
        if fut_type=="abs":
            return(climdex_ANN_fut_avg,climdex_MAM_fut_avg,climdex_JJA_fut_avg,climdex_SON_fut_avg,climdex_DJF_fut_avg)
        elif fut_type=="delta":
            climdex_ANN_fut_delta = climdex_ANN_hist_avg-climdex_ANN_fut_avg
            climdex_MAM_fut_delta = climdex_MAM_hist_avg-climdex_MAM_fut_avg
            climdex_JJA_fut_delta = climdex_JJA_hist_avg-climdex_JJA_fut_avg
            climdex_SON_fut_delta = climdex_SON_hist_avg-climdex_SON_fut_avg
            climdex_DJF_fut_delta = climdex_DJF_hist_avg-climdex_DJF_fut_avg
            return(climdex_ANN_fut_delta,climdex_MAM_fut_delta,climdex_JJA_fut_delta,climdex_SON_fut_delta,climdex_DJF_fut_delta)

    
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
    cbar_ax.set_xlabel(xlabel + " (avg. per year)",size=20)         
  
def make_title(seas):
    if climdex=="csdi":
        climdex_f = "Cold Spell Duration Index"
    elif climdex=="dsfreq":
        climdex_f = "Dry Spell Frequency"
    elif climdex=="wsdi":
        climdex_f = "Warm Spell Duration Index"
    
    if period == "hist":
        title_f = "Historical"

    else:
        if period == "rcp45":
            title_f = "RCP4.5"
        elif period == "rcp85":
            title_f = "RCP8.5"

    return(climdex_f + " " + seas + " " + title_f)


count_event_peryear_ANN,count_event_peryear_MAM,count_event_peryear_JJA,count_event_peryear_SON,count_event_peryear_DJF = calculate_seas_climdex()
#%%
vmin=-12
vmax=12
plot_climdex(count_event_peryear_ANN, make_title("ANN"),vmin,vmax)
plot_climdex(count_event_peryear_MAM, make_title("MAM"),vmin/4,vmax/4)
plot_climdex(count_event_peryear_JJA, make_title("JJA"),vmin/4,vmax/4)
plot_climdex(count_event_peryear_SON, make_title("SON"),vmin/4,vmax/4)
plot_climdex(count_event_peryear_DJF, make_title("DJF"),vmin/4,vmax/4)

