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

t_type = "warm day" #warm day, warm night, cold day, cold night
pr_type = "wet" #wet, dry

period = 'rcp45' #rcp45, rcp85


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
    
    def get_seas_dates(time):
        dates_MAM = [date for date in time if date.month in [3,4,5]]
        dates_JJA = [date for date in time if date.month in [6,7,8]]
        dates_SON = [date for date in time if date.month in [9,10,11]]
        dates_DJF = [date for date in time if date.month in [1,2,12]]
        
        return(np.array(dates_MAM),np.array(dates_JJA),np.array(dates_SON),np.array(dates_DJF))

    def get_yearly_sums(values,time):
        yearly_sums = {}
        for value, date in zip(values,time):
            yearly_sums[date.year] = yearly_sums.get(date.year,0) + value

        return(np.array(list(yearly_sums.values())))
    
    def count_events(t_perc,pr_perc,wrf_d03_time,wrf_d03_t,wrf_d03_pr):
        
        t_perc_long_ANN,t_perc_long_MAM,t_perc_long_JJA,t_perc_long_SON,t_perc_long_DJF,\
            pr_perc_long_ANN,pr_perc_long_MAM,pr_perc_long_JJA,pr_perc_long_SON,pr_perc_long_DJF = get_percentiles(t_perc,pr_perc,wrf_d03_time)
   
        wrf_d03_t_MAM,wrf_d03_t_JJA,wrf_d03_t_SON,wrf_d03_t_DJF = get_seas_values(wrf_d03_t, wrf_d03_time)
        wrf_d03_pr_MAM,wrf_d03_pr_JJA,wrf_d03_pr_SON,wrf_d03_pr_DJF = get_seas_values(wrf_d03_pr, wrf_d03_time)
        
        dates_MAM,dates_JJA,dates_SON,dates_DJF = get_seas_dates(wrf_d03_time)


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
              
        
        count_event_ANN_yearly = get_yearly_sums(event_bool_ANN,wrf_d03_time)
        count_event_MAM_yearly = get_yearly_sums(event_bool_MAM,dates_MAM)
        count_event_JJA_yearly = get_yearly_sums(event_bool_JJA,dates_JJA)
        count_event_SON_yearly = get_yearly_sums(event_bool_SON,dates_SON)
        count_event_DJF_yearly = get_yearly_sums(event_bool_DJF,dates_DJF)
           
        return(count_event_ANN_yearly,count_event_MAM_yearly,count_event_JJA_yearly,count_event_SON_yearly,count_event_DJF_yearly)
        
    
    # read in only necessary data
    if "day" in t_type:
        
        wrf_d03_t_hist = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['T2'][:]
    
        if "warm" in t_type: 
            t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tx90p_hist.nc','r').variables['per'][:])
        elif "cold" in t_type:
            t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tx10p_hist.nc','r').variables['per'][:])

        wrf_d03_t_fut = Dataset(gridded_data_path + 't_d03_tmax_daily_' + period + '.nc','r').variables['T2'][:]
        
        if "warm" in t_type: 
            t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tx90p_' + period + '.nc','r').variables['per'][:])
        elif "cold" in t_type:
            t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tx10p_' + period + '.nc','r').variables['per'][:])


    elif "night" in t_type:
        wrf_d03_t_hist = Dataset(gridded_data_path + 't_d03_tmin_daily_hist.nc','r').variables['T2'][:]
    
        if "warm" in t_type: 
            t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tn90p_hist.nc','r').variables['per'][:])
        elif "cold" in t_type:
            t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tn10p_hist.nc','r').variables['per'][:])

        wrf_d03_t_fut = Dataset(gridded_data_path + 't_d03_tmin_daily_' + period + '.nc','r').variables['T2'][:]
            
        if "warm" in t_type: 
            t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tn90p_' + period + '.nc','r').variables['per'][:])
        elif "cold" in t_type:
            t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tn10p_' + period + '.nc','r').variables['per'][:])

    wrf_d03_pr_hist = Dataset(gridded_data_path + 'pr_d03_daily_hist.nc','r').variables['pr'][:]
    
    pr_perc_hist_raw = np.squeeze(Dataset(percentiles_path + 'pr75p_hist.nc','r').variables['per'][:])
    pr_perc_hist = np.zeros((300,300,daysinyear))
    pr_perc_hist[1:-1,1:-1,:] = pr_perc_hist_raw

    wrf_d03_time_hist = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['time'][:]
    wrf_d03_time_hist = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_hist]
    
    count_event_peryear_ANN_hist,count_event_peryear_MAM_hist,count_event_peryear_JJA_hist,count_event_peryear_SON_hist,count_event_peryear_DJF_hist = count_events(t_perc_hist,pr_perc_hist,wrf_d03_time_hist,wrf_d03_t_hist,wrf_d03_pr_hist)
    
    print(np.shape(count_event_peryear_ANN_hist))
    print(np.shape(count_event_peryear_MAM_hist))

    wrf_d03_pr_fut = Dataset(gridded_data_path + 'pr_d03_daily_' + period + '.nc','r').variables['pr'][:]
    
    pr_perc_fut_raw = np.squeeze(Dataset(percentiles_path + 'pr75p_' + period + '.nc','r').variables['per'][:])

    pr_perc_fut = np.zeros((300,300,daysinyear))
    pr_perc_fut[1:-1,1:-1,:] = pr_perc_fut_raw

    
    wrf_d03_time_fut = Dataset(gridded_data_path + 't_d03_tmax_daily_rcp45.nc','r').variables['time'][:]
    wrf_d03_time_fut = [datetime.datetime(2046, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_fut]
    
    count_event_peryear_ANN_fut,count_event_peryear_MAM_fut,count_event_peryear_JJA_fut,count_event_peryear_SON_fut,count_event_peryear_DJF_fut = count_events(t_perc_fut,pr_perc_fut,wrf_d03_time_fut,wrf_d03_t_fut,wrf_d03_pr_fut)

    print(np.shape(count_event_peryear_ANN_fut))
    print(np.shape(count_event_peryear_MAM_fut))

    #ttest_ANN = scipy.stats.ttest_ind(np.squeeze(count_event_peryear_ANN_hist), np.squeeze(count_event_peryear_ANN_fut),axis=0)
    #ttest_MAM = scipy.stats.ttest_ind(np.squeeze(count_event_peryear_MAM_hist), np.squeeze(count_event_peryear_MAM_fut),axis=0)
    #ttest_JJA = scipy.stats.ttest_ind(np.squeeze(count_event_peryear_JJA_hist), np.squeeze(count_event_peryear_JJA_fut),axis=0)
    #ttest_SON = scipy.stats.ttest_ind(np.squeeze(count_event_peryear_SON_hist), np.squeeze(count_event_peryear_SON_fut),axis=0)
    #ttest_DJF = scipy.stats.ttest_ind(np.squeeze(count_event_peryear_DJF_hist), np.squeeze(count_event_peryear_DJF_fut),axis=0)

    #return(ttest_ANN.pvalue,ttest_MAM.pvalue,ttest_JJA.pvalue,ttest_SON.pvalue,ttest_DJF.pvalue)
    
    std_ANN_hist = np.std(count_event_peryear_ANN_hist,axis=0)
    std_MAM_hist = np.std(count_event_peryear_MAM_hist,axis=0)
    std_JJA_hist = np.std(count_event_peryear_JJA_hist,axis=0)
    std_SON_hist = np.std(count_event_peryear_SON_hist,axis=0)
    std_DJF_hist = np.std(count_event_peryear_DJF_hist,axis=0)
    
    std_ANN_fut = np.std(count_event_peryear_ANN_fut,axis=0)
    std_MAM_fut = np.std(count_event_peryear_MAM_fut,axis=0)
    std_JJA_fut = np.std(count_event_peryear_JJA_fut,axis=0)
    std_SON_fut = np.std(count_event_peryear_SON_fut,axis=0)
    std_DJF_fut = np.std(count_event_peryear_DJF_fut,axis=0)
    
    
    return(std_ANN_hist,std_MAM_hist,std_JJA_hist,std_SON_hist,std_DJF_hist,std_ANN_fut,std_MAM_fut,std_JJA_fut,std_SON_fut,std_DJF_fut)

def plot_compound_events(gridded_data,title,vmin,vmax):
    if vmin==0:
        cmap='viridis'
    
    else:
        cmap='bwr'
        
        
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
    #cbar_ax.set_xlabel("P-value",size=20)  
    cbar_ax.set_xlabel("Standard Deviation",size=20)  

  
def make_title(seas):

    if period == "rcp45":
        title_f = "RCP4.5"
    elif period == "rcp85":
        title_f = "RCP8.5"


    if pr_type == "wet":
        pr_type_f = "Wet"
    elif pr_type == "dry":
        pr_type_f = "Dry"
        
    if t_type == "warm day":
        t_type_f = "Warm days"
        t_type_f = "Cold days"
    elif t_type == "warm night":
        t_type_f = "Warm nights"
    elif t_type == "cold night":
        t_type_f = "Cold nights"


    return(title_f + " " + seas + "\n" + pr_type_f + "-" + t_type_f)

    return(seas)

#pvalue_ANN,pvalue_MAM,pvalue_JJA,pvalue_SON,pvalue_DJF = calculate_compound_event()

#std_ANN_hist,std_MAM_hist,std_JJA_hist,std_SON_hist,std_DJF_hist,std_ANN_fut,std_MAM_fut,std_JJA_fut,std_SON_fut,std_DJF_fut = calculate_compound_event()

#vmin=0
#vmax=0.05

#plot_compound_events(pvalue_ANN, make_title("ANN"),vmin,vmax)
#plot_compound_events(pvalue_MAM, make_title("MAM"),vmin,vmax)
#plot_compound_events(pvalue_JJA, make_title("JJA"),vmin,vmax)
#plot_compound_events(pvalue_SON, make_title("SON"),vmin,vmax)
#plot_compound_events(pvalue_DJF, make_title("DJF"),vmin,vmax)


vmin=0
vmax=8
#plot_compound_events(std_ANN_hist, make_title("ANN"),vmin,vmax)
#plot_compound_events(std_MAM_hist, make_title("MAM"),vmin,vmax)
#plot_compound_events(std_JJA_hist, make_title("JJA"),vmin,vmax)
#plot_compound_events(std_SON_hist, make_title("SON"),vmin,vmax)
#plot_compound_events(std_DJF_hist, make_title("DJF"),vmin,vmax)

plot_compound_events(std_ANN_fut, make_title("ANN"),vmin,vmax)
plot_compound_events(std_MAM_fut, make_title("MAM"),vmin,vmax)
plot_compound_events(std_JJA_fut, make_title("JJA"),vmin,vmax)
plot_compound_events(std_SON_fut, make_title("SON"),vmin,vmax)
plot_compound_events(std_DJF_fut, make_title("DJF"),vmin,vmax)
