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

t_type = "cold night" #warm day, warm night, cold day, cold night
pr_type = "dry" #wet, dry

period = 'rcp85' # hist, rcp45, rcp85

#ignored if period=hist
fut_type = 'delta' #delta, abs 
base_period = "fut" #fut or hist


#%%

gridded_data_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/'
percentiles_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/percentiles/'

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

lons = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['lon'][:]
lats = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['lat'][:]


def calculate_compound_event():
    daysinyear = 366
    
    def get_percentiles(t_perc,pr_perc,wrf_d03_time):
    
        t_perc_long_ANN,t_perc_long_MAM,t_perc_long_JJA,t_perc_long_SON,t_perc_long_DJF = [],[],[],[],[]
        pr_perc_long_ANN,pr_perc_long_MAM,pr_perc_long_JJA,pr_perc_long_SON,pr_perc_long_DJF = [],[],[],[],[]
    
        for date in wrf_d03_time:
            for i in range(1,daysinyear+1):
                if date.timetuple().tm_yday == i:
        
                    t_perc_long_ANN.append(t_perc[:,:,i-1])
                    pr_perc_long_ANN.append(pr_perc[:,:,i-1])
                    
                    if date.month in [3,4,5]:
                        t_perc_long_MAM.append(t_perc[:,:,i-1])
                        pr_perc_long_MAM.append(pr_perc[:,:,i-1])
                    elif date.month in [6,7,8]:
                        t_perc_long_JJA.append(t_perc[:,:,i-1])
                        pr_perc_long_JJA.append(pr_perc[:,:,i-1])
                    elif date.month in [9,10,11]:
                        t_perc_long_SON.append(t_perc[:,:,i-1])
                        pr_perc_long_SON.append(pr_perc[:,:,i-1])
                    elif date.month in [1,2,12]:
                        t_perc_long_DJF.append(t_perc[:,:,i-1])
                        pr_perc_long_DJF.append(pr_perc[:,:,i-1])

        return(t_perc_long_ANN,np.array(t_perc_long_MAM),np.array(t_perc_long_JJA),np.array(t_perc_long_SON),np.array(t_perc_long_DJF),
               pr_perc_long_ANN,np.array(pr_perc_long_MAM),np.array(pr_perc_long_JJA),np.array(pr_perc_long_SON),np.array(pr_perc_long_DJF))
   
    
    def get_seas_values(values,time):
        vals_MAM = [vals for vals, date in zip(values, time) if date.month in [3,4,5]]
        vals_JJA = [vals for vals, date in zip(values, time) if date.month in [6,7,8]]
        vals_SON = [vals for vals, date in zip(values, time) if date.month in [9,10,11]]
        vals_DJF = [vals for vals, date in zip(values, time) if date.month in [1,2,12]]

        return(np.array(vals_MAM),np.array(vals_JJA),np.array(vals_SON),np.array(vals_DJF))
    
    def count_events(t_perc,pr_perc,wrf_d03_time,wrf_d03_t,wrf_d03_pr):
        
        t_perc_long_ANN,t_perc_long_MAM,t_perc_long_JJA,t_perc_long_SON,t_perc_long_DJF,\
            pr_perc_long_ANN,pr_perc_long_MAM,pr_perc_long_JJA,pr_perc_long_SON,pr_perc_long_DJF = get_percentiles(t_perc,pr_perc,wrf_d03_time)
   
        wrf_d03_t_MAM,wrf_d03_t_JJA,wrf_d03_t_SON,wrf_d03_t_DJF = get_seas_values(wrf_d03_t, wrf_d03_time)
        wrf_d03_pr_MAM,wrf_d03_pr_JJA,wrf_d03_pr_SON,wrf_d03_pr_DJF = get_seas_values(wrf_d03_pr, wrf_d03_time)


        if "warm" in t_type and pr_type == "wet":
            event_bool_ANN = (wrf_d03_t > t_perc_long_ANN) & (wrf_d03_pr > pr_perc_long_ANN)
            event_bool_MAM = (wrf_d03_t_MAM > t_perc_long_MAM) & (wrf_d03_pr_MAM > pr_perc_long_MAM)
            event_bool_JJA = (wrf_d03_t_JJA > t_perc_long_JJA) & (wrf_d03_pr_JJA > pr_perc_long_JJA)
            event_bool_SON = (wrf_d03_t_SON > t_perc_long_SON) & (wrf_d03_pr_SON > pr_perc_long_SON)
            event_bool_DJF = (wrf_d03_t_DJF > t_perc_long_DJF) & (wrf_d03_pr_DJF > pr_perc_long_DJF)
        
        elif "cold" in t_type and pr_type == "wet":
            event_bool_ANN = (wrf_d03_t < t_perc_long_ANN) & (wrf_d03_pr > pr_perc_long_ANN)
            event_bool_MAM = (wrf_d03_t_MAM < t_perc_long_MAM) & (wrf_d03_pr_MAM > pr_perc_long_MAM)
            event_bool_JJA = (wrf_d03_t_JJA < t_perc_long_JJA) & (wrf_d03_pr_JJA > pr_perc_long_JJA)
            event_bool_SON = (wrf_d03_t_SON < t_perc_long_SON) & (wrf_d03_pr_SON > pr_perc_long_SON)
            event_bool_DJF = (wrf_d03_t_DJF < t_perc_long_DJF) & (wrf_d03_pr_DJF > pr_perc_long_DJF)
        
        elif "warm" in t_type and pr_type == "dry":
            event_bool_ANN = (wrf_d03_t > t_perc_long_ANN) & (wrf_d03_pr < 1)
            event_bool_MAM = (wrf_d03_t_MAM > t_perc_long_MAM) & (wrf_d03_pr_MAM < 1)
            event_bool_JJA = (wrf_d03_t_JJA > t_perc_long_JJA) & (wrf_d03_pr_JJA < 1)
            event_bool_SON = (wrf_d03_t_SON > t_perc_long_SON) & (wrf_d03_pr_SON < 1)
            event_bool_DJF = (wrf_d03_t_DJF > t_perc_long_DJF) & (wrf_d03_pr_DJF < 1)
        
        elif "cold" in t_type and pr_type == "dry":
            event_bool_ANN = (wrf_d03_t < t_perc_long_ANN) & (wrf_d03_pr < 1)
            event_bool_MAM = (wrf_d03_t_MAM < t_perc_long_MAM) & (wrf_d03_pr_MAM < 1)
            event_bool_JJA = (wrf_d03_t_JJA < t_perc_long_JJA) & (wrf_d03_pr_JJA < 1)
            event_bool_SON = (wrf_d03_t_SON < t_perc_long_SON) & (wrf_d03_pr_SON < 1)
            event_bool_DJF = (wrf_d03_t_DJF < t_perc_long_DJF) & (wrf_d03_pr_DJF < 1)
                      
        
        count_event_peryear_ANN = sum(event_bool_ANN)/20
        count_event_peryear_MAM = sum(event_bool_MAM)/20
        count_event_peryear_JJA = sum(event_bool_JJA)/20
        count_event_peryear_SON = sum(event_bool_SON)/20
        count_event_peryear_DJF = sum(event_bool_DJF)/20
        
        return(count_event_peryear_ANN,count_event_peryear_MAM,count_event_peryear_JJA,count_event_peryear_SON,count_event_peryear_DJF)
        
    # read in only necessary data
    if "day" in t_type:
        if period == "hist" or fut_type == "delta":
            wrf_d03_t_hist = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['T2'][:]
    
            if "warm" in t_type: 
                t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tx90p_hist.nc','r').variables['per'][:])
            elif "cold" in t_type:
                t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tx10p_hist.nc','r').variables['per'][:])

        if period != 'hist':
            wrf_d03_t_fut = Dataset(gridded_data_path + 't_d03_tmax_daily_' + period + '.nc','r').variables['T2'][:]
        
            if base_period == "fut":
                if "warm" in t_type: 
                    t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tx90p_' + period + '.nc','r').variables['per'][:])
                elif "cold" in t_type:
                    t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tx10p_' + period + '.nc','r').variables['per'][:])
            elif base_period == "hist":
                if "warm" in t_type: 
                    t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tx90p_hist.nc','r').variables['per'][:])
                elif "cold" in t_type:
                    t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tx10p_hist.nc','r').variables['per'][:])


    elif "night" in t_type:
        if period == "hist" or fut_type == "delta":
            wrf_d03_t_hist = Dataset(gridded_data_path + 't_d03_tmin_daily_hist.nc','r').variables['T2'][:]
        
            if "warm" in t_type: 
                t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tn90p_hist.nc','r').variables['per'][:])
            elif "cold" in t_type:
                t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tn10p_hist.nc','r').variables['per'][:])

        if period != 'hist':
            wrf_d03_t_fut = Dataset(gridded_data_path + 't_d03_tmin_daily_' + period + '.nc','r').variables['T2'][:]

            if base_period == "fut":
                if "warm" in t_type: 
                    t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tn90p_' + period + '.nc','r').variables['per'][:])
                elif "cold" in t_type:
                    t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tn10p_' + period + '.nc','r').variables['per'][:])
            elif base_period == "hist":
                if "warm" in t_type: 
                    t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tn90p_hist.nc','r').variables['per'][:])
                elif "cold" in t_type:
                    t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tn10p_hist.nc','r').variables['per'][:])

    if period == "hist" or fut_type == "delta":
        wrf_d03_pr_hist = Dataset(gridded_data_path + 'pr_d03_daily_hist.nc','r').variables['pr'][:]
        
        pr_perc_hist_raw = np.squeeze(Dataset(percentiles_path + 'pr75p_hist.nc','r').variables['per'][:])
        pr_perc_hist = np.zeros((300,300,daysinyear))
        pr_perc_hist[1:-1,1:-1,:] = pr_perc_hist_raw


        wrf_d03_time_hist = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['time'][:]
        wrf_d03_time_hist = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_hist]
        
        count_event_peryear_ANN_hist,count_event_peryear_MAM_hist,count_event_peryear_JJA_hist,count_event_peryear_SON_hist,count_event_peryear_DJF_hist = count_events(t_perc_hist,pr_perc_hist,wrf_d03_time_hist,wrf_d03_t_hist,wrf_d03_pr_hist)
    
    if period != "hist":
        wrf_d03_pr_fut = Dataset(gridded_data_path + 'pr_d03_daily_' + period + '.nc','r').variables['pr'][:]
        
        if base_period == "fut":
            pr_perc_fut_raw = np.squeeze(Dataset(percentiles_path + 'pr75p_' + period + '.nc','r').variables['per'][:])
        elif base_period == "hist":
            pr_perc_fut_raw = np.squeeze(Dataset(percentiles_path + 'pr75p_hist.nc','r').variables['per'][:])

        pr_perc_fut = np.zeros((300,300,daysinyear))
        pr_perc_fut[1:-1,1:-1,:] = pr_perc_fut_raw

        
        wrf_d03_time_fut = Dataset(gridded_data_path + 't_d03_tmax_daily_rcp45.nc','r').variables['time'][:]
        wrf_d03_time_fut = [datetime.datetime(2046, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_fut]
        
        count_event_peryear_ANN_fut,count_event_peryear_MAM_fut,count_event_peryear_JJA_fut,count_event_peryear_SON_fut,count_event_peryear_DJF_fut = count_events(t_perc_fut,pr_perc_fut,wrf_d03_time_fut,wrf_d03_t_fut,wrf_d03_pr_fut)

    if period == "hist":
        return(count_event_peryear_ANN_hist,count_event_peryear_MAM_hist,count_event_peryear_JJA_hist,count_event_peryear_SON_hist,count_event_peryear_DJF_hist)
    
    elif period != "hist":
        
        if fut_type == "abs":
            return(count_event_peryear_ANN_fut,count_event_peryear_MAM_fut,count_event_peryear_JJA_fut,count_event_peryear_SON_fut,count_event_peryear_DJF_fut)
        
        elif fut_type == "delta":
            
            count_event_peryear_ANN_delta = count_event_peryear_ANN_fut-count_event_peryear_ANN_hist
            count_event_peryear_MAM_delta = count_event_peryear_MAM_fut-count_event_peryear_MAM_hist
            count_event_peryear_JJA_delta = count_event_peryear_JJA_fut-count_event_peryear_JJA_hist
            count_event_peryear_SON_delta = count_event_peryear_SON_fut-count_event_peryear_SON_hist
            count_event_peryear_DJF_delta = count_event_peryear_DJF_fut-count_event_peryear_DJF_hist

            return(count_event_peryear_ANN_delta,count_event_peryear_MAM_delta,count_event_peryear_JJA_delta,count_event_peryear_SON_delta,count_event_peryear_DJF_delta)
  
    
def plot_compound_events(gridded_data,title,vmin,vmax):
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
    if period == "hist":
        title_f = "Historical"
        years = "(1986-2005)"
        base_years = "1986-2005"
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
            
    if pr_type == "wet":
        pr_type_f = "Wet"
        pr_stat = "pr > Q75"
    elif pr_type == "dry":
        pr_type_f = "Dry"
        pr_stat = "pr < 1mm"
        
    if t_type == "warm day":
        t_type_f = "Warm days"
        t_stat = "Tx > Q90"
    elif t_type == "cold day":
        t_type_f = "Cold days"
        t_stat = "Tx < Q10"
    elif t_type == "warm night":
        t_type_f = "Warm nights"
        t_stat = "Tn > Q90"
    elif t_type == "cold night":
        t_type_f = "Cold nights"
        t_stat = "Tn < Q10"

    return(title_f + " " + seas + " " + years + ", Base " + base_years + "\n" + pr_type_f + "-" + t_type_f + " (" + pr_stat + ", " + t_stat + ")")


count_event_peryear_ANN,count_event_peryear_MAM,count_event_peryear_JJA,count_event_peryear_SON,count_event_peryear_DJF = calculate_compound_event()

vmin=-2
vmax=2
plot_compound_events(count_event_peryear_ANN, make_title("ANN"),vmin,vmax)
plot_compound_events(count_event_peryear_MAM, make_title("MAM"),vmin,vmax)
plot_compound_events(count_event_peryear_JJA, make_title("JJA"),vmin,vmax)
plot_compound_events(count_event_peryear_SON, make_title("SON"),vmin,vmax)
plot_compound_events(count_event_peryear_DJF, make_title("DJF"),vmin,vmax)

