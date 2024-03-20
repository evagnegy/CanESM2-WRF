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
pr_type = "sdii" #wd or sdii
period = 'rcp45' # hist, rcp45, rcp85


gridded_data_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily/'

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

lons = Dataset(gridded_data_path + 'pr_d03_daily_hist.nc').variables['lon'][:]
lats = Dataset(gridded_data_path + 'pr_d03_daily_hist.nc').variables['lat'][:]


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

def get_yearly_means(values,time):
    yearly_mean = {}
    days={}
    for value, date in zip(values,time):
        yearly_mean[date.year] = yearly_mean.get(date.year,0) + value
        days[date.year] = days.get(date.year,0)+1

    for year in yearly_mean:
        yearly_mean[year] /= days[year]

    return(np.array(list(yearly_mean.values())))      
 
def get_yearly_sums(values,time):
    yearly_sums = {}
    for value, date in zip(values,time):
        yearly_sums[date.year] = yearly_sums.get(date.year,0) + value

    return(np.array(list(yearly_sums.values())))
    
wrf_d03_time_hist = Dataset(gridded_data_path + '/pr_d03_daily_hist.nc','r').variables['time'][:]
wrf_d03_time_hist = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_hist]

daily_pr_hist = Dataset(gridded_data_path + '/pr_d03_daily_hist.nc').variables['pr'][:]

pr_MAM_hist,pr_JJA_hist,pr_SON_hist,pr_DJF_hist = get_seas_values(daily_pr_hist, wrf_d03_time_hist)
dates_MAM,dates_JJA,dates_SON,dates_DJF = get_seas_dates(wrf_d03_time_hist)

if pr_type == "wd":
    #wet_days_ANN_hist = np.sum(daily_pr_hist>1,axis=0)/20
    #wet_days_MAM_hist = np.sum(pr_MAM_hist>1,axis=0)/20
    #wet_days_JJA_hist = np.sum(pr_JJA_hist>1,axis=0)/20
    #wet_days_SON_hist = np.sum(pr_SON_hist>1,axis=0)/20
    #wet_days_DJF_hist = np.sum(pr_DJF_hist>1,axis=0)/20
    
    wet_days_ANN_hist = get_yearly_sums(daily_pr_hist>1,wrf_d03_time_hist)
    wet_days_MAM_hist = get_yearly_sums(pr_MAM_hist>1,dates_MAM)
    wet_days_JJA_hist = get_yearly_sums(pr_JJA_hist>1,dates_JJA)
    wet_days_SON_hist = get_yearly_sums(pr_SON_hist>1,dates_SON)
    wet_days_DJF_hist = get_yearly_sums(pr_DJF_hist>1,dates_DJF)

elif pr_type == "sdii":
    daily_pr_hist[daily_pr_hist<1]=0
    pr_MAM_hist[pr_MAM_hist<1]=0
    pr_JJA_hist[pr_JJA_hist<1]=0
    pr_SON_hist[pr_SON_hist<1]=0
    pr_DJF_hist[pr_DJF_hist<1]=0
    wet_days_ANN_hist = get_yearly_means(daily_pr_hist,wrf_d03_time_hist)
    wet_days_MAM_hist = get_yearly_means(pr_MAM_hist,dates_MAM)
    wet_days_JJA_hist = get_yearly_means(pr_JJA_hist,dates_JJA)
    wet_days_SON_hist = get_yearly_means(pr_SON_hist,dates_SON)
    wet_days_DJF_hist = get_yearly_means(pr_DJF_hist,dates_DJF)

if period != "hist":
    wrf_d03_time_fut = Dataset(gridded_data_path + '/pr_d03_daily_rcp45.nc','r').variables['time'][:]
    wrf_d03_time_fut = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_fut]
    
    daily_pr_fut = Dataset(gridded_data_path + '/pr_d03_daily_' + period + '.nc').variables['pr'][:]
    
    pr_MAM_fut,pr_JJA_fut,pr_SON_fut,pr_DJF_fut = get_seas_values(daily_pr_fut, wrf_d03_time_fut)
    
    dates_MAM,dates_JJA,dates_SON,dates_DJF = get_seas_dates(wrf_d03_time_fut)
    
    if pr_type == "wd":
        #wet_days_ANN_fut = np.sum(daily_pr_fut>1,axis=0)/20
        #wet_days_MAM_fut = np.sum(pr_MAM_fut>1,axis=0)/20
        #wet_days_JJA_fut = np.sum(pr_JJA_fut>1,axis=0)/20
        #wet_days_SON_fut = np.sum(pr_SON_fut>1,axis=0)/20
        #wet_days_DJF_fut = np.sum(pr_DJF_fut>1,axis=0)/20
    
        wet_days_ANN_fut = get_yearly_sums(daily_pr_fut>1,wrf_d03_time_fut)
        wet_days_MAM_fut = get_yearly_sums(pr_MAM_fut>1,dates_MAM)
        wet_days_JJA_fut = get_yearly_sums(pr_JJA_fut>1,dates_JJA)
        wet_days_SON_fut = get_yearly_sums(pr_SON_fut>1,dates_SON)
        wet_days_DJF_fut = get_yearly_sums(pr_DJF_fut>1,dates_DJF)
    
    elif pr_type == "sdii":
    
        daily_pr_fut[daily_pr_fut<1]=0
        pr_MAM_fut[pr_MAM_fut<1]=0
        pr_JJA_fut[pr_JJA_fut<1]=0
        pr_SON_fut[pr_SON_fut<1]=0
        pr_DJF_fut[pr_DJF_fut<1]=0
        wet_days_ANN_fut = get_yearly_means(daily_pr_fut,wrf_d03_time_fut)
        wet_days_MAM_fut = get_yearly_means(pr_MAM_fut,dates_MAM)
        wet_days_JJA_fut = get_yearly_means(pr_JJA_fut,dates_JJA)
        wet_days_SON_fut = get_yearly_means(pr_SON_fut,dates_SON)
        wet_days_DJF_fut = get_yearly_means(pr_DJF_fut,dates_DJF)
    
    wetdays_ANN_delta = wet_days_ANN_fut-wet_days_ANN_hist
    wetdays_MAM_delta = wet_days_MAM_fut-wet_days_MAM_hist
    wetdays_JJA_delta = wet_days_JJA_fut-wet_days_JJA_hist
    wetdays_SON_delta = wet_days_SON_fut-wet_days_SON_hist
    wetdays_DJF_delta = wet_days_DJF_fut-wet_days_DJF_hist
    
    ttest_ANN = scipy.stats.ttest_ind(np.squeeze(wet_days_ANN_hist), np.squeeze(wet_days_ANN_fut),axis=0)
    ttest_MAM = scipy.stats.ttest_ind(np.squeeze(wet_days_MAM_hist), np.squeeze(wet_days_MAM_fut),axis=0)
    ttest_JJA = scipy.stats.ttest_ind(np.squeeze(wet_days_JJA_hist), np.squeeze(wet_days_JJA_fut),axis=0)
    ttest_SON = scipy.stats.ttest_ind(np.squeeze(wet_days_SON_hist), np.squeeze(wet_days_SON_fut),axis=0)
    ttest_DJF = scipy.stats.ttest_ind(np.squeeze(wet_days_DJF_hist), np.squeeze(wet_days_DJF_fut),axis=0)

#%%

def make_title(seas):
    
    if pr_type == "wd":
        climdex_f = "Wet Days"
    elif pr_type == "sdii":
        climdex_f = "SDII"
    
    if period == "hist":
        title_f = "Historical"
        years = "(1986-2005)"

    else:
        years = "(2046-2065)"
        if period == "rcp45":
            title_f = "RCP4.5"
        elif period == "rcp85":
            title_f = "RCP8.5"

    return(climdex_f + " " + title_f + " " + seas + " " + years)



def plot_climdex(gridded_data,pvalue,seas,vmin,vmax):

    if vmin==0:
        cmap='viridis'
        if pr_type == "wd":
            xlabel = "Wet Days"
        elif pr_type == "sdii":
            xlabel = "SDII"
    else:
        cmap='bwr'
        if pr_type == "wd":
            xlabel = "Diff. in Wet Days (avg. per year)"
        elif pr_type == "sdii":
            xlabel = "Diff. in SDII (mm/day)"
        
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
    
    #gridded_data[land_d03==0]=np.nan
    
    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap,vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    
    if period != "hist":
        masked_grid = pvalue.copy()
        masked_grid[masked_grid>0.05] = np.nan
        ax1.pcolor(lons, lats, masked_grid, transform=ccrs.PlateCarree(), hatch='...', alpha=0,vmin=vmin,vmax=vmax)
        mpl.rcParams['hatch.linewidth'] = 0.8
    
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
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

    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/climdex/' + pr_type + "_" + period + "_" + seas + "_95.png", bbox_inches='tight')
    




if period == "hist":
    vmin=0
    vmax=25
    
    plot_climdex(np.mean(wet_days_ANN_hist,axis=0), None,"ANN", vmin,vmax)
    plot_climdex(np.mean(wet_days_MAM_hist,axis=0), None,"MAM", vmin,vmax)
    plot_climdex(np.mean(wet_days_JJA_hist,axis=0), None,"JJA", vmin,vmax)
    plot_climdex(np.mean(wet_days_SON_hist,axis=0), None,"SON", vmin,vmax)
    plot_climdex(np.mean(wet_days_DJF_hist,axis=0), None,"DJF", vmin,vmax)

else:
    vmin=-5
    vmax=5

    plot_climdex(np.mean(wetdays_ANN_delta,axis=0),ttest_ANN.pvalue, "ANN",vmin,vmax)
    plot_climdex(np.mean(wetdays_MAM_delta,axis=0),ttest_MAM.pvalue, "MAM",vmin,vmax)
    plot_climdex(np.mean(wetdays_JJA_delta,axis=0),ttest_JJA.pvalue, "JJA",vmin,vmax)
    plot_climdex(np.mean(wetdays_SON_delta,axis=0),ttest_SON.pvalue, "SON",vmin,vmax)
    plot_climdex(np.mean(wetdays_DJF_delta,axis=0),ttest_DJF.pvalue, "DJF",vmin,vmax)
    
