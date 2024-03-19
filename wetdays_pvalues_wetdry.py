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
    vals_WET = [vals for vals, date in zip(values, time) if date.month in [10,11,12,1,2,3]]
    vals_DRY = [vals for vals, date in zip(values, time) if date.month in [4,5,6,7,8,9]]

    return(np.array(vals_WET),np.array(vals_DRY))
        
def get_seas_dates(time):
    dates_WET = [date for date in time if date.month in [10,11,12,1,2,3]]
    dates_DRY = [date for date in time if date.month in [4,5,6,7,8,9]]

    return(np.array(dates_WET),np.array(dates_DRY))

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

pr_WET_hist,pr_DRY_hist = get_seas_values(daily_pr_hist, wrf_d03_time_hist)
dates_WET,dates_DRY = get_seas_dates(wrf_d03_time_hist)

if pr_type == "wd":

    wet_days_ANN_hist = get_yearly_sums(daily_pr_hist>1,wrf_d03_time_hist)
    wet_days_WET_hist = get_yearly_sums(pr_WET_hist>1,dates_WET)
    wet_days_DRY_hist = get_yearly_sums(pr_DRY_hist>1,dates_DRY)

elif pr_type == "sdii":
    daily_pr_hist[daily_pr_hist<1]=0
    pr_WET_hist[pr_WET_hist<1]=0
    pr_DRY_hist[pr_DRY_hist<1]=0

    wet_days_ANN_hist = get_yearly_means(daily_pr_hist,wrf_d03_time_hist)
    wet_days_WET_hist = get_yearly_means(pr_WET_hist,dates_WET)
    wet_days_DRY_hist = get_yearly_means(pr_DRY_hist,dates_DRY)

if period != "hist":
    wrf_d03_time_fut = Dataset(gridded_data_path + '/pr_d03_daily_rcp45.nc','r').variables['time'][:]
    wrf_d03_time_fut = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=hours) for hours in wrf_d03_time_fut]
    
    daily_pr_fut = Dataset(gridded_data_path + '/pr_d03_daily_' + period + '.nc').variables['pr'][:]
    
    pr_WET_fut,pr_DRY_fut = get_seas_values(daily_pr_fut, wrf_d03_time_fut)
    
    dates_WET,dates_DRY = get_seas_dates(wrf_d03_time_fut)
    
    if pr_type == "wd":

        wet_days_ANN_fut = get_yearly_sums(daily_pr_fut>1,wrf_d03_time_fut)
        wet_days_WET_fut = get_yearly_sums(pr_WET_fut>1,dates_WET)
        wet_days_DRY_fut = get_yearly_sums(pr_DRY_fut>1,dates_DRY)

    elif pr_type == "sdii":
    
        daily_pr_fut[daily_pr_fut<1]=0
        pr_WET_fut[pr_WET_fut<1]=0
        pr_DRY_fut[pr_DRY_fut<1]=0

        wet_days_ANN_fut = get_yearly_means(daily_pr_fut,wrf_d03_time_fut)
        wet_days_WET_fut = get_yearly_means(pr_WET_fut,dates_WET)
        wet_days_DRY_fut = get_yearly_means(pr_DRY_fut,dates_DRY)
   
    
    wetdays_ANN_delta = np.mean(wet_days_ANN_fut,axis=0)-np.mean(wet_days_ANN_hist,axis=0)
    wetdays_WET_delta = np.mean(wet_days_WET_fut,axis=0)-np.mean(wet_days_WET_hist,axis=0)
    wetdays_DRY_delta = np.mean(wet_days_DRY_fut,axis=0)-np.mean(wet_days_DRY_hist,axis=0)

    ttest_ANN = scipy.stats.ttest_ind(np.squeeze(wet_days_ANN_hist), np.squeeze(wet_days_ANN_fut),axis=0)
    ttest_WET = scipy.stats.ttest_ind(np.squeeze(wet_days_WET_hist), np.squeeze(wet_days_WET_fut),axis=0)
    ttest_DRY = scipy.stats.ttest_ind(np.squeeze(wet_days_DRY_hist), np.squeeze(wet_days_DRY_fut),axis=0)

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
        masked_grid[masked_grid>0.1] = np.nan
        ax1.pcolor(lons, lats, masked_grid, transform=ccrs.PlateCarree(), hatch='...', alpha=0,vmin=vmin,vmax=vmax)
        mpl.rcParams['hatch.linewidth'] = 1
    
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

    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/climdex/' + pr_type + "_" + period + "_" + seas + ".png", bbox_inches='tight')
    


if period == "hist":
    vmin=0
    vmax=25
    
    plot_climdex(np.mean(wet_days_ANN_hist,axis=0), None,"ANN", vmin,vmax)
    plot_climdex(np.mean(wet_days_WET_hist,axis=0), None,"WET (ONDJFM)", vmin,vmax)
    plot_climdex(np.mean(wet_days_DRY_hist,axis=0), None,"DRY (AMJJAS)", vmin,vmax)

else:
    vmin=-5
    vmax=5

    plot_climdex(wetdays_ANN_delta,ttest_ANN.pvalue, "ANN",vmin,vmax)
    plot_climdex(wetdays_WET_delta,ttest_WET.pvalue, "WET (ONDJFM)",vmin,vmax)
    plot_climdex(wetdays_DRY_delta,ttest_DRY.pvalue, "DRY (AMJJAS)",vmin,vmax)


