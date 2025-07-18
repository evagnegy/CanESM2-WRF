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
from canesm2_eval_funcs import *
import WRFDomainLib
import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib import cm
import xarray as xr
import  scipy
import matplotlib.colors as pltcol
#%%

t_type = "warm" #warm, cold 
pr_type = "dry" #wet, dry

period = 'rcp85' # hist, rcp45, rcp85

#ignored if period=hist
fut_type = 'delta' #delta, abs 
base_period = "hist" #fut or hist

#%%


gridded_data_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/'
percentiles_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/percentiles/'

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

lons = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['lon'][:]
lats = Dataset(gridded_data_path + '/t_d03_tmax_daily_hist.nc','r').variables['lat'][:]


def calculate_compound_event(seas):
    daysinyear = 366
    
    if seas=="mam":
        months = [3,4,5]
    elif seas=="jja":
        months = [6,7,8]
    elif seas=="son":
        months = [9,10,11]
    elif seas=="djf":
        months = [1,2,12]
            
    def get_percentiles(t_perc,pr_perc,wrf_d03_time):
            
            
        t_perc_long_seas = []
        pr_perc_long_seas = []
    
        for date in wrf_d03_time:
            for i in range(1,daysinyear+1):
                if date.timetuple().tm_yday == i:
        
                    if date.month in months:
                        t_perc_long_seas.append(t_perc[:,:,i-1])
                        pr_perc_long_seas.append(pr_perc[:,:,i-1])


        return(np.array(t_perc_long_seas),np.array(pr_perc_long_seas))
   
    
    def get_seas_values(values,time):
        vals_seas = [vals for vals, date in zip(values, time) if date.month in months]

        return(np.array(vals_seas))
    
    def get_seas_dates(time):
        dates_seas = [date for date in time if date.month in months]
       
        return(np.array(dates_seas))

    def get_yearly_sums(values,time):
        yearly_sums = {}
        for value, date in zip(values,time):
            yearly_sums[date.year] = yearly_sums.get(date.year,0) + value

        return(np.array(list(yearly_sums.values())))
    
    def count_events(t_perc,pr_perc,wrf_d03_time,wrf_d03_t,wrf_d03_pr):
        
        t_perc_long_seas,pr_perc_long_seas = get_percentiles(t_perc,pr_perc,wrf_d03_time)
   
        wrf_d03_t_seas = get_seas_values(wrf_d03_t, wrf_d03_time)
        wrf_d03_pr_seas = get_seas_values(wrf_d03_pr, wrf_d03_time)

        dates_seas = get_seas_dates(wrf_d03_time)

        if "warm" in t_type and pr_type == "wet":
            event_bool_seas = (wrf_d03_t_seas > t_perc_long_seas) & (wrf_d03_pr_seas > pr_perc_long_seas)

        elif "cold" in t_type and pr_type == "wet":
            event_bool_seas = (wrf_d03_t_seas < t_perc_long_seas) & (wrf_d03_pr_seas > pr_perc_long_seas)
        
        elif "warm" in t_type and pr_type == "dry":
            event_bool_seas = (wrf_d03_t_seas > t_perc_long_seas) & (wrf_d03_pr_seas < 1)
        
        elif "cold" in t_type and pr_type == "dry":
            event_bool_seas = (wrf_d03_t_seas < t_perc_long_seas) & (wrf_d03_pr_seas < 1)
                      
        count_event_seas_yearly = get_yearly_sums(event_bool_seas,dates_seas)

        return(count_event_seas_yearly)
        
    # read in only necessary data
    if period == "hist" or fut_type == "delta":
        wrf_d03_t_hist = Dataset(gridded_data_path + '/t_d03_tas_daily_hist.nc','r').variables['T2'][:]

        if "warm" in t_type: 
            t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tas90p_hist.nc','r').variables['per'][:])
        elif "cold" in t_type:
            t_perc_hist = np.squeeze(Dataset(percentiles_path + 'tas10p_hist.nc','r').variables['per'][:])

    if period != 'hist':
        wrf_d03_t_fut = Dataset(gridded_data_path + 't_d03_tas_daily_' + period + '.nc','r').variables['T2'][:]
    
        if base_period == "fut":
            if "warm" in t_type: 
                t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tas90p_' + period + '.nc','r').variables['per'][:])
            elif "cold" in t_type:
                t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tas10p_' + period + '.nc','r').variables['per'][:])
        elif base_period == "hist":
            if "warm" in t_type: 
                t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tas90p_hist.nc','r').variables['per'][:])
            elif "cold" in t_type:
                t_perc_fut = np.squeeze(Dataset(percentiles_path + 'tas10p_hist.nc','r').variables['per'][:])



    if period == "hist" or fut_type == "delta":
        wrf_d03_pr_hist = Dataset(gridded_data_path + 'pr_d03_daily_hist.nc','r').variables['pr'][:]
        
        pr_perc_hist_raw = np.squeeze(Dataset(percentiles_path + 'pr75p_hist.nc','r').variables['per'][:])
        pr_perc_hist = np.zeros((300,300,daysinyear))
        pr_perc_hist[1:-1,1:-1,:] = pr_perc_hist_raw


        wrf_d03_time_hist = Dataset(gridded_data_path + '/t_d03_tas_daily_hist.nc','r').variables['time'][:]
        wrf_d03_time_hist = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_hist]
        
        count_event_peryear_seas_hist = count_events(t_perc_hist,pr_perc_hist,wrf_d03_time_hist,wrf_d03_t_hist,wrf_d03_pr_hist)

    if period != "hist":
        wrf_d03_pr_fut = Dataset(gridded_data_path + 'pr_d03_daily_' + period + '.nc','r').variables['pr'][:]
        
        if base_period == "fut":
            pr_perc_fut_raw = np.squeeze(Dataset(percentiles_path + 'pr75p_' + period + '.nc','r').variables['per'][:])
        elif base_period == "hist":
            pr_perc_fut_raw = np.squeeze(Dataset(percentiles_path + 'pr75p_hist.nc','r').variables['per'][:])

        pr_perc_fut = np.zeros((300,300,daysinyear))
        pr_perc_fut[1:-1,1:-1,:] = pr_perc_fut_raw

        
        wrf_d03_time_fut = Dataset(gridded_data_path + 't_d03_tas_daily_rcp45.nc','r').variables['time'][:]
        wrf_d03_time_fut = [datetime.datetime(2046, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_fut]
        
        count_event_peryear_seas_fut = count_events(t_perc_fut,pr_perc_fut,wrf_d03_time_fut,wrf_d03_t_fut,wrf_d03_pr_fut)

    if period == "hist":
    
        count_event_peryear_seas_hist = np.mean(count_event_peryear_seas_hist,axis=0)
        return(count_event_peryear_seas_hist)

    elif period != "hist":
        
        if fut_type == "abs":
            count_event_peryear_seas_fut = np.mean(count_event_peryear_seas_fut,axis=0)
            return(count_event_peryear_seas_fut)
        
        elif fut_type == "delta":
            count_event_peryear_seas_delta = np.mean(count_event_peryear_seas_fut,axis=0)-np.mean(count_event_peryear_seas_hist,axis=0)

            ttest_seas = scipy.stats.ttest_ind(np.squeeze(count_event_peryear_seas_hist), np.squeeze(count_event_peryear_seas_fut),axis=0)

            return(ttest_seas.pvalue,count_event_peryear_seas_delta)

#%%

def save_netcdf(value,file,name):

    da = xr.DataArray(
        data=value,
        coords={
            'lat': (['lats', 'lats'], lats),
            'lon': (['lons', 'lons'], lons)
        },
        dims=['lats', 'lons'],
        name=name
    )
        
    directory = f'/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/compound_events/ncfiles/{file}'
    da.to_netcdf(directory)
    
    
    
if period == "hist":
    print("mam")
    count_event_peryear_MAM = calculate_compound_event("mam")
    save_netcdf(count_event_peryear_MAM,f'{t_type}_{pr_type}_{period}_mam.nc','count')
    print("jja")
    count_event_peryear_JJA = calculate_compound_event("jja")
    save_netcdf(count_event_peryear_JJA,f'{t_type}_{pr_type}_{period}_jja.nc','count')
    print("son")
    count_event_peryear_SON = calculate_compound_event("son")
    save_netcdf(count_event_peryear_SON,f'{t_type}_{pr_type}_{period}_son.nc','count')
    print("djf")
    count_event_peryear_DJF = calculate_compound_event("djf")
    save_netcdf(count_event_peryear_DJF,f'{t_type}_{pr_type}_{period}_djf.nc','count')

else:
    print("mam")
    pvalue_MAM,count_event_peryear_MAM = calculate_compound_event("mam")
    save_netcdf(pvalue_MAM,f'{t_type}_{pr_type}_{period}_base_{base_period}_mam_pvalue.nc','pvalue')
    save_netcdf(count_event_peryear_MAM,f'{t_type}_{pr_type}_{period}_base_{base_period}_mam.nc','count')
    print("jja")
    pvalue_JJA,count_event_peryear_JJA = calculate_compound_event("jja")
    save_netcdf(pvalue_JJA,f'{t_type}_{pr_type}_{period}_base_{base_period}_jja_pvalue.nc','pvalue')
    save_netcdf(count_event_peryear_JJA,f'{t_type}_{pr_type}_{period}_base_{base_period}_jja.nc','count')
    print("son")
    pvalue_SON,count_event_peryear_SON = calculate_compound_event("son")
    save_netcdf(pvalue_SON,f'{t_type}_{pr_type}_{period}_base_{base_period}_son_pvalue.nc','pvalue')
    save_netcdf(count_event_peryear_SON,f'{t_type}_{pr_type}_{period}_base_{base_period}_son.nc','count')
    print("djf")
    pvalue_DJF,count_event_peryear_DJF = calculate_compound_event("djf")
    save_netcdf(pvalue_DJF,f'{t_type}_{pr_type}_{period}_base_{base_period}_djf_pvalue.nc','pvalue')
    save_netcdf(count_event_peryear_DJF,f'{t_type}_{pr_type}_{period}_base_{base_period}_djf.nc','count')

#%%

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
        
    if t_type == "warm":
        t_type_f = "Warm"
        t_stat = "Tas > Q90"
    elif t_type == "cold":
        t_type_f = "Cold"
        t_stat = "Tas < Q10"


    return(title_f + " " + seas + " " + years + ", Base " + base_years + "\n" + pr_type_f + "-" + t_type_f + " (" + pr_stat + ", " + t_stat + ")")


    
def plot_compound_events(gridded_data,pvalue,seas,vmin,vmax):
    if vmin==0:
        
        cmap = plt.get_cmap('Purples')
        colors = [cmap(i / (11 - 1)) for i in range(11)]

        cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors,N=11)
        cmap = cmap(np.linspace(0, 1, cmap.N))[:-1] 
        cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=12)
        cmap.set_over(colors[-1]) #add the max arrow color
        
        xlabel='Count/year'
    else:
# =============================================================================
#         cmap = plt.get_cmap('PiYG')
#         colors = [cmap(i / (22 - 1)) for i in range(22)]
# 
#         cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors,N=22)
#         cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
#         cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=20)
#         cmap.set_over(colors[-1]) #add the max arrow color
#         cmap.set_under(colors[0]) #add the min arrow color
# =============================================================================
        colors = ['#424c03','#41641a','#4b8c49','#79b17d','#aed0b2', '#cfe6d1','#eddaeb',
                             '#d6b4d2','#c98dc1','#ad49a0','#8c037a','#5c0250'][::-1]
        

        
        cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors,N=20)
        cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
        cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=18)
        cmap.set_over(colors[-1]) #add the max arrow color
        cmap.set_under(colors[0]) #add the min arrow color
            
        xlabel='$\Delta$ Count/year'
        
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
        
    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap,vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    
    if period != "hist":
        masked_grid = pvalue.copy()
        masked_grid[masked_grid>0.1] = np.nan
        ax1.pcolor(lons, lats, masked_grid, transform=ccrs.PlateCarree(), hatch='...', alpha=0,vmin=vmin,vmax=vmax)
        mpl.rcParams['hatch.linewidth'] = 0.8

    
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

    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    
    if period=="hist":
        fig1.colorbar(cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                      cax=cbar_ax, orientation='horizontal',extend='max')
    else:        
        fig1.colorbar(cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(0, vmax+1, 0.5))
    

    
    
    cbar_ax.tick_params(labelsize=20)
    cbar_ax.set_xlabel(xlabel,size=20)         
  
    if period != "hist":
        plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/compound_events/' + t_type + "_" + pr_type + "_" + period + "_base_" + base_period + "_" + seas,bbox_inches='tight')
    else:
        plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/compound_events/' + t_type + "_" + pr_type + "_" + period + "_" + seas,bbox_inches='tight')
        #plt.close()



#%


t_type = "warm" #warm, cold 
pr_type = "wet" #wet, dry

period = 'rcp45' # hist, rcp45, rcp85

#ignored if period=hist
fut_type = 'delta' #delta, abs 
base_period = "hist" #fut or hist


directory = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/compound_events/ncfiles/'

if period == "hist":
    count_event_peryear_MAM = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_mam.nc')['count'].values
    count_event_peryear_JJA = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_jja.nc')['count'].values
    count_event_peryear_SON = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_son.nc')['count'].values
    count_event_peryear_DJF = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_djf.nc')['count'].values

else:
    count_event_peryear_MAM = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_base_{base_period}_mam.nc')['count'].values
    count_event_peryear_JJA = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_base_{base_period}_jja.nc')['count'].values
    count_event_peryear_SON = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_base_{base_period}_son.nc')['count'].values
    count_event_peryear_DJF = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_base_{base_period}_djf.nc')['count'].values
    
    pvalue_MAM = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_base_{base_period}_mam_pvalue.nc')['pvalue'].values
    pvalue_JJA = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_base_{base_period}_jja_pvalue.nc')['pvalue'].values
    pvalue_SON = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_base_{base_period}_son_pvalue.nc')['pvalue'].values
    pvalue_DJF = xr.open_dataset(f'{directory}{t_type}_{pr_type}_{period}_base_{base_period}_djf_pvalue.nc')['pvalue'].values



if period == "hist":
    
    vmin=0
    vmax=12

    plot_compound_events(np.array(count_event_peryear_MAM),None, "MAM",vmin,vmax)
    plot_compound_events(np.array(count_event_peryear_JJA),None, "JJA",vmin,vmax)
    plot_compound_events(np.array(count_event_peryear_SON),None, "SON",vmin,vmax)
    plot_compound_events(np.array(count_event_peryear_DJF),None, "DJF",vmin,vmax)

else:
    vmin=-60
    vmax=60

    plot_compound_events(np.array(count_event_peryear_MAM),pvalue_MAM, "MAM",vmin,vmax)
    plot_compound_events(np.array(count_event_peryear_JJA),pvalue_JJA, "JJA",vmin,vmax)
    plot_compound_events(np.array(count_event_peryear_SON),pvalue_SON, "SON",vmin,vmax)
    plot_compound_events(np.array(count_event_peryear_DJF),pvalue_DJF, "DJF",vmin,vmax)


