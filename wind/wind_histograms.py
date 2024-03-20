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
import cftime

hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/wind_avgs/wind_weighted_means_hist.nc'
rcp45_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/wind_avgs/wind_weighted_means_rcp45.nc'
rcp85_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/wind_avgs/wind_weighted_means_rcp85.nc'

nc_hist = Dataset(hist_file,'r')
wspd_all_hist = np.squeeze(nc_hist.variables['wspd_all'][:])
wdir_all_hist = np.squeeze(nc_hist.variables['wdir_all'][:])
wspd_land_hist = np.squeeze(nc_hist.variables['wspd_land'][:])
wdir_land_hist = np.squeeze(nc_hist.variables['wdir_land'][:])
wspd_ocean_hist = np.squeeze(nc_hist.variables['wspd_ocean'][:])
wdir_ocean_hist = np.squeeze(nc_hist.variables['wdir_ocean'][:])

nc_rcp45 = Dataset(rcp45_file,'r')
wspd_all_rcp45 = np.squeeze(nc_rcp45.variables['wspd_all'][:])
wdir_all_rcp45 = np.squeeze(nc_rcp45.variables['wdir_all'][:])
wspd_land_rcp45 = np.squeeze(nc_rcp45.variables['wspd_land'][:])
wdir_land_rcp45 = np.squeeze(nc_rcp45.variables['wdir_land'][:])
wspd_ocean_rcp45 = np.squeeze(nc_rcp45.variables['wspd_ocean'][:])
wdir_ocean_rcp45 = np.squeeze(nc_rcp45.variables['wdir_ocean'][:])

nc_rcp85 = Dataset(rcp85_file,'r')
wspd_all_rcp85 = np.squeeze(nc_rcp85.variables['wspd_all'][:])
wdir_all_rcp85 = np.squeeze(nc_rcp85.variables['wdir_all'][:])
wspd_land_rcp85 = np.squeeze(nc_rcp85.variables['wspd_land'][:])
wdir_land_rcp85 = np.squeeze(nc_rcp85.variables['wdir_land'][:])
wspd_ocean_rcp85 = np.squeeze(nc_rcp85.variables['wspd_ocean'][:])
wdir_ocean_rcp85 = np.squeeze(nc_rcp85.variables['wdir_ocean'][:])

#%%

time = nc_hist.variables['time'][:]
time_dates = cftime.num2date(time, nc_hist.variables['time'].units, calendar=nc_hist.variables['time'].calendar)

djf_indices = [i for i, dt in enumerate(time_dates) if dt.month in [12,1,2]]
mam_indices = [i for i, dt in enumerate(time_dates) if dt.month in [3,4,5]]
jja_indices = [i for i, dt in enumerate(time_dates) if dt.month in [6,7,8]]
son_indices = [i for i, dt in enumerate(time_dates) if dt.month in [9,10,11]]

#%%

def plot_hist_by_domain(hist,rcp45,rcp85,title):

    #wspd_all_hist_day_avg = np.mean(hist.reshape(-1, 24), axis=1)
    #wspd_all_rcp45_day_avg = np.mean(rcp45.reshape(-1, 24), axis=1)
    #wspd_all_rcp85_day_avg = np.mean(rcp85.reshape(-1, 24), axis=1)
    
    wspd_all_hist_day_max = np.max(hist.reshape(-1, 24), axis=1)
    wspd_all_rcp45_day_max = np.max(rcp45.reshape(-1, 24), axis=1)
    wspd_all_rcp85_day_max = np.max(rcp85.reshape(-1, 24), axis=1)
    

    wspd_all_hist_10 = np.percentile(hist, 10) 
    wspd_all_rcp45_10 = np.percentile(rcp45, 10) 
    wspd_all_rcp85_10 = np.percentile(rcp85, 10) 
    
    wspd_all_hist_50 = np.percentile(hist, 50) 
    wspd_all_rcp45_50 = np.percentile(rcp45, 50) 
    wspd_all_rcp85_50 = np.percentile(rcp85, 50) 
    
    wspd_all_hist_90 = np.percentile(hist, 90) 
    wspd_all_rcp45_90 = np.percentile(rcp45, 90) 
    wspd_all_rcp85_90 = np.percentile(rcp85, 90) 
    
    wspd_all_hist_95 = np.percentile(hist, 95) 
    wspd_all_rcp45_95 = np.percentile(rcp45, 95) 
    wspd_all_rcp85_95 = np.percentile(rcp85, 95) 
    

    
    wspd_all_hist_99 = np.percentile(hist, 99) 
    wspd_all_rcp45_99 = np.percentile(rcp45, 99) 
    wspd_all_rcp85_99 = np.percentile(rcp85, 99) 
    
    
    wspd_all_hist_10_day_max = np.percentile(wspd_all_hist_day_max, 10) 
    wspd_all_rcp45_10_day_max = np.percentile(wspd_all_rcp45_day_max, 10) 
    wspd_all_rcp85_10_day_max = np.percentile(wspd_all_rcp85_day_max, 10) 
    
    wspd_all_hist_50_day_max = np.percentile(wspd_all_hist_day_max, 50) 
    wspd_all_rcp45_50_day_max = np.percentile(wspd_all_rcp45_day_max, 50) 
    wspd_all_rcp85_50_day_max = np.percentile(wspd_all_rcp85_day_max, 50) 
    
    wspd_all_hist_90_day_max = np.percentile(wspd_all_hist_day_max, 90) 
    wspd_all_rcp45_90_day_max = np.percentile(wspd_all_rcp45_day_max, 90) 
    wspd_all_rcp85_90_day_max = np.percentile(wspd_all_rcp85_day_max, 90) 
    
    wspd_all_hist_95_day_max = np.percentile(wspd_all_hist_day_max, 95) 
    wspd_all_rcp45_95_day_max = np.percentile(wspd_all_rcp45_day_max, 95) 
    wspd_all_rcp85_95_day_max = np.percentile(wspd_all_rcp85_day_max, 95) 
    
    wspd_all_hist_99_day_max = np.percentile(wspd_all_hist_day_max, 99) 
    wspd_all_rcp45_99_day_max = np.percentile(wspd_all_rcp45_day_max, 99) 
    wspd_all_rcp85_99_day_max = np.percentile(wspd_all_rcp85_day_max, 99) 
    

    fig, ax = plt.subplots(figsize=(7, 4))
       
    bins=20
    
    plt.hist(hist,bins,label="historical",color="k",density=True,histtype='step')
    plt.hist(rcp45,bins,label="RCP4.5",color="C0",density=True,histtype='step')
    plt.hist(rcp85,bins,label="RCP8.5",color="C1",density=True,histtype='step')
    
    plt.axvline(x=wspd_all_hist_10, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_10, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_10, color='C1', linestyle='dashed', linewidth=1)
    
    plt.axvline(x=wspd_all_hist_50, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_50, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_50, color='C1', linestyle='dashed', linewidth=1)
    
    plt.axvline(x=wspd_all_hist_90, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_90, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_90, color='C1', linestyle='dashed', linewidth=1)
    
    plt.axvline(x=wspd_all_hist_95, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_95, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_95, color='C1', linestyle='dashed', linewidth=1)
    
    plt.axvline(x=wspd_all_hist_99, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_99, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_99, color='C1', linestyle='dashed', linewidth=1)
    
    plt.legend(loc='upper right')
    
    plt.ylabel('PDF',fontsize=12)
    plt.xlabel('Hourly wind speed [m/s]',fontsize=12)
    plt.title(title,fontsize=14)
    plt.xlim([0,14])

    
    # =============================================================================
    # fig, ax = plt.subplots(figsize=(7, 4))
    #    
    # bins=20
    # 
    # plt.hist(wspd_all_hist_day_avg,bins,label="historical",color="k",density=True,histtype='step')
    # plt.hist(wspd_all_rcp45_day_avg,bins,label="RCP4.5",color="C0",density=True,histtype='step')
    # plt.hist(wspd_all_rcp85_day_avg,bins,label="RCP8.5",color="C1",density=True,histtype='step')
    # 
    # plt.legend(loc='upper right')
    # 
    # plt.ylabel('PDF',fontsize=12)
    # plt.xlabel('Daily mean wind speed [m/s]',fontsize=12)
    # plt.title('Annual',fontsize=14)
    # =============================================================================
    

    
    fig, ax = plt.subplots(figsize=(7, 4))
       
    bins=20
    
    plt.hist(wspd_all_hist_day_max,bins,label="historical",color="k",density=True,histtype='step')
    plt.hist(wspd_all_rcp45_day_max,bins,label="RCP4.5",color="C0",density=True,histtype='step')
    plt.hist(wspd_all_rcp85_day_max,bins,label="RCP8.5",color="C1",density=True,histtype='step')
    
    plt.axvline(x=wspd_all_hist_10_day_max, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_10_day_max, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_10_day_max, color='C1', linestyle='dashed', linewidth=1)
    
    plt.axvline(x=wspd_all_hist_50_day_max, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_50_day_max, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_50_day_max, color='C1', linestyle='dashed', linewidth=1)
    
    plt.axvline(x=wspd_all_hist_90_day_max, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_90_day_max, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_90_day_max, color='C1', linestyle='dashed', linewidth=1)
    
    plt.axvline(x=wspd_all_hist_95_day_max, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_95_day_max, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_95_day_max, color='C1', linestyle='dashed', linewidth=1)
    
    plt.axvline(x=wspd_all_hist_99_day_max, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp45_99_day_max, color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(x=wspd_all_rcp85_99_day_max, color='C1', linestyle='dashed', linewidth=1)
    plt.legend(loc='upper right')
    
    plt.ylabel('PDF',fontsize=12)
    plt.xlabel('Daily max wind speed [m/s]',fontsize=12)
    plt.title(title,fontsize=14)
    plt.xlim([0,14])
    
#plot_hist_by_domain(wspd_all_hist,wspd_all_rcp45,wspd_all_rcp85,'Annual')
#plot_hist_by_domain(wspd_land_hist,wspd_land_rcp45,wspd_land_rcp85,'Annual (land)')
#plot_hist_by_domain(wspd_ocean_hist,wspd_ocean_rcp45,wspd_ocean_rcp85,'Annual (ocean)')

wspd_all_hist_djf = wspd_all_hist[djf_indices]
wspd_all_hist_mam = wspd_all_hist[mam_indices]
wspd_all_hist_jja = wspd_all_hist[jja_indices]
wspd_all_hist_son = wspd_all_hist[son_indices]

wspd_all_rcp45_djf = wspd_all_rcp45[djf_indices]
wspd_all_rcp45_mam = wspd_all_rcp45[mam_indices]
wspd_all_rcp45_jja = wspd_all_rcp45[jja_indices]
wspd_all_rcp45_son = wspd_all_rcp45[son_indices]

wspd_all_rcp85_djf = wspd_all_rcp85[djf_indices]
wspd_all_rcp85_mam = wspd_all_rcp85[mam_indices]
wspd_all_rcp85_jja = wspd_all_rcp85[jja_indices]
wspd_all_rcp85_son = wspd_all_rcp85[son_indices]

plot_hist_by_domain(wspd_all_hist_djf,wspd_all_rcp45_djf,wspd_all_rcp85_djf,'DJF (all domain)')
plot_hist_by_domain(wspd_all_hist_mam,wspd_all_rcp45_mam,wspd_all_rcp85_mam,'MAM (all domain)')
plot_hist_by_domain(wspd_all_hist_jja,wspd_all_rcp45_jja,wspd_all_rcp85_jja,'JJA (all domain)')
plot_hist_by_domain(wspd_all_hist_son,wspd_all_rcp45_son,wspd_all_rcp85_son,'SON (all domain)')
