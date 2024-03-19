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
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_noaa_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic,plot_all_d03,plot_zoomed_in

variable = 'pr'
d='d03'



if variable  == "t":
    diff_data_rcp45 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_rcp45_diff.nc'
    diff_data_rcp85 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_rcp85_diff.nc'
    
    diff_nc_rcp45 = Dataset(diff_data_rcp45, mode='r');
    diff_nc_rcp85 = Dataset(diff_data_rcp85, mode='r');
       
    lats = np.squeeze(diff_nc_rcp45.variables['lat'][:])
    lons = np.squeeze(diff_nc_rcp85.variables['lon'][:])
    lons[lons>0] = lons[lons>0]-360
    
    ann_var_change_rcp45 = np.squeeze(diff_nc_rcp45.variables['T2'][:,:,:])
    ann_var_change_rcp85 = np.squeeze(diff_nc_rcp85.variables['T2'][:,:,:])
    
    
elif variable == "pr":
    hist_data = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_mean_hist.nc'
    rcp45_data = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_mean_rcp45.nc'
    rcp85_data = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_mean_rcp85.nc'

    nc_hist = Dataset(hist_data, mode='r');
    nc_rcp45 = Dataset(rcp45_data, mode='r');
    nc_rcp85 = Dataset(rcp85_data, mode='r');
       
    lats = np.squeeze(nc_hist.variables['lat'][:])
    lons = np.squeeze(nc_hist.variables['lon'][:])
    lons[lons>0] = lons[lons>0]-360
    
    ann_var_hist = np.squeeze(nc_hist.variables['pr'][:,:,:])
    ann_var_rcp45 = np.squeeze(nc_rcp45.variables['pr'][:,:,:])
    ann_var_rcp85 = np.squeeze(nc_rcp85.variables['pr'][:,:,:])
    
    ann_var_change_rcp45 = ((ann_var_rcp45 - ann_var_hist) / ann_var_hist ) *100
    ann_var_change_rcp85 = ((ann_var_rcp85 - ann_var_hist) / ann_var_hist ) *100

#%%

if variable  == "t":
    seasdiff_data_rcp45 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_rcp45_seasdiff.nc'
    seasdiff_data_rcp85 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_rcp85_seasdiff.nc'
    
    seasdiff_nc_rcp45 = Dataset(seasdiff_data_rcp45, mode='r');
    seasdiff_nc_rcp85 = Dataset(seasdiff_data_rcp85, mode='r');
       
    lats = np.squeeze(seasdiff_nc_rcp45.variables['lat'][:])
    lons = np.squeeze(seasdiff_nc_rcp85.variables['lon'][:])
    lons[lons>0] = lons[lons>0]-360
    
    seas_var_change_rcp45 = np.squeeze(seasdiff_nc_rcp45.variables['T2'][:,:,:])
    seas_var_change_rcp85 = np.squeeze(seasdiff_nc_rcp85.variables['T2'][:,:,:])
    
    djf_var_change_rcp45 = seas_var_change_rcp45[0,:,:]
    mam_var_change_rcp45 = seas_var_change_rcp45[1,:,:]
    jja_var_change_rcp45 = seas_var_change_rcp45[2,:,:]
    son_var_change_rcp45 = seas_var_change_rcp45[3,:,:]
    
    djf_var_change_rcp85 = seas_var_change_rcp85[0,:,:]
    mam_var_change_rcp85 = seas_var_change_rcp85[1,:,:]
    jja_var_change_rcp85 = seas_var_change_rcp85[2,:,:]
    son_var_change_rcp85 = seas_var_change_rcp85[3,:,:]
    
elif variable == "pr":
    hist_data = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_seasmean_hist.nc'
    rcp45_data = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_seasmean_rcp45.nc'
    rcp85_data = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/' + variable + '_' + d + '_seasmean_rcp85.nc'

    nc_hist = Dataset(hist_data, mode='r');
    nc_rcp45 = Dataset(rcp45_data, mode='r');
    nc_rcp85 = Dataset(rcp85_data, mode='r');
       
    lats = np.squeeze(nc_hist.variables['lat'][:])
    lons = np.squeeze(nc_hist.variables['lon'][:])
    lons[lons>0] = lons[lons>0]-360
    
    seas_var_hist = np.squeeze(nc_hist.variables['pr'][:,:,:])
    seas_var_rcp45 = np.squeeze(nc_rcp45.variables['pr'][:,:,:])
    seas_var_rcp85 = np.squeeze(nc_rcp85.variables['pr'][:,:,:])
    
    djf_var_change_rcp45 = (seas_var_rcp45[0,:,:] - seas_var_hist[0,:,:])/seas_var_hist[0,:,:] * 100
    mam_var_change_rcp45 = (seas_var_rcp45[1,:,:] - seas_var_hist[1,:,:])/seas_var_hist[1,:,:] * 100
    jja_var_change_rcp45 = (seas_var_rcp45[2,:,:] - seas_var_hist[2,:,:])/seas_var_hist[2,:,:] * 100
    son_var_change_rcp45 = (seas_var_rcp45[3,:,:] - seas_var_hist[3,:,:])/seas_var_hist[3,:,:] * 100

    djf_var_change_rcp85 = (seas_var_rcp85[0,:,:] - seas_var_hist[0,:,:])/seas_var_hist[0,:,:] * 100
    mam_var_change_rcp85 = (seas_var_rcp85[1,:,:] - seas_var_hist[1,:,:])/seas_var_hist[1,:,:] * 100
    jja_var_change_rcp85 = (seas_var_rcp85[2,:,:] - seas_var_hist[2,:,:])/seas_var_hist[2,:,:] * 100
    son_var_change_rcp85 = (seas_var_rcp85[3,:,:] - seas_var_hist[3,:,:])/seas_var_hist[3,:,:] * 100

#%%

# =============================================================================
# output_freq = "yearly" #yearly monthly or daily
# 
# eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
# bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'
# noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations.csv'
# 
# df = pd.read_csv(eccc_daily_stations,header=None)
# eccc_station_IDs = list(df.iloc[:,4])
# eccc_station_names = list(df.iloc[:,1])
# 
# eccc_lats = df.iloc[:,7]
# eccc_lons = df.iloc[:,8]
# eccc_lats.index = eccc_station_IDs
# eccc_lons.index = eccc_station_IDs
# 
# df = pd.read_csv(bch_daily_stations)
# bch_station_IDs = list(df["STATION_NO"])
# bch_station_names = list(df["STATION_NA"])
# 
# bch_lats = df['Y']
# bch_lons = df['X']
# bch_lats.index = bch_station_IDs
# bch_lons.index = bch_station_IDs
# 
# df = pd.read_csv(noaa_daily_stations)
# 
# noaa_station_IDs = list(df.iloc[:,0])
# noaa_station_names = list(df.iloc[:,1])
# 
# noaa_lats = df.iloc[:,2]
# noaa_lons = df.iloc[:,3]
# noaa_lats.index = noaa_station_IDs
# noaa_lons.index = noaa_station_IDs
# =============================================================================
#%%

# =============================================================================
# stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/'
# 
# WRF_files_hist = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/historical/'
# WRF_files_rcp45 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/rcp45/'
# WRF_files_rcp85 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/rcp85/'
# 
# 
# #%%
# 
# wrf_d03_bch_hist = get_wrf("yearly", "BCH", bch_station_IDs, "d03", "historical", variable, WRF_files_hist,1986)
# wrf_d03_eccc_hist = get_wrf("yearly", "ECCC", eccc_station_IDs, "d03", "historical", variable, WRF_files_hist,1986)
# wrf_d03_noaa_hist = get_wrf("yearly", "NOAA", noaa_station_IDs, "d03", "historical", variable, WRF_files_hist,1986)
# 
# wrf_d03_bch_rcp45 = get_wrf("yearly", "BCH", bch_station_IDs, "d03", "historical", variable, WRF_files_rcp45,2046)
# wrf_d03_eccc_rcp45 = get_wrf("yearly", "ECCC", eccc_station_IDs, "d03", "historical", variable, WRF_files_rcp45,2046)
# wrf_d03_noaa_rcp45 = get_wrf("yearly", "NOAA", noaa_station_IDs, "d03", "historical", variable, WRF_files_rcp45,2046)
# 
# wrf_d03_bch_rcp85 = get_wrf("yearly", "BCH", bch_station_IDs, "d03", "historical", variable, WRF_files_rcp85,2046)
# wrf_d03_eccc_rcp85 = get_wrf("yearly", "ECCC", eccc_station_IDs, "d03", "historical", variable, WRF_files_rcp85,2046)
# wrf_d03_noaa_rcp85 = get_wrf("yearly", "NOAA", noaa_station_IDs, "d03", "historical", variable, WRF_files_rcp85,2046)
# 
# #%%
# 
# wrf_d03_bch_mon_hist = get_wrf("monthly", "BCH", bch_station_IDs, "d03", "historical", variable, WRF_files_hist,1986)
# wrf_d03_eccc_mon_hist = get_wrf("monthly", "ECCC", eccc_station_IDs, "d03", "historical", variable, WRF_files_hist,1986)
# wrf_d03_noaa_mon_hist = get_wrf("monthly", "NOAA", noaa_station_IDs, "d03", "historical", variable, WRF_files_hist,1986)
# 
# wrf_d03_bch_mon_rcp45 = get_wrf("monthly", "BCH", bch_station_IDs, "d03", "historical", variable, WRF_files_rcp45,2046)
# wrf_d03_eccc_mon_rcp45 = get_wrf("monthly", "ECCC", eccc_station_IDs, "d03", "historical", variable, WRF_files_rcp45,2046)
# wrf_d03_noaa_mon_rcp45 = get_wrf("monthly", "NOAA", noaa_station_IDs, "d03", "historical", variable, WRF_files_rcp45,2046)
# 
# wrf_d03_bch_mon_rcp85 = get_wrf("monthly", "BCH", bch_station_IDs, "d03", "historical", variable, WRF_files_rcp85,2046)
# wrf_d03_eccc_mon_rcp85 = get_wrf("monthly", "ECCC", eccc_station_IDs, "d03", "historical", variable, WRF_files_rcp85,2046)
# wrf_d03_noaa_mon_rcp85 = get_wrf("monthly", "NOAA", noaa_station_IDs, "d03", "historical", variable, WRF_files_rcp85,2046)
# 
# =============================================================================
#%%

# =============================================================================
# # remove stations not in the original list
# for station in noaa_station_IDs:
#     if station not in list(wrf_d03_bch_hist.columns):
#         wrf_d03_noaa_hist.drop(station, inplace=True, axis=1)
#         wrf_d03_noaa_rcp45.drop(station, inplace=True, axis=1)
#         wrf_d03_noaa_rcp85.drop(station, inplace=True, axis=1)
#         noaa_lats.drop(station,inplace=True)
#         noaa_lons.drop(station,inplace=True)
# =============================================================================

#%%


# =============================================================================
# wrf_d03_eccc_hist_mam = wrf_d03_eccc_mon_hist.copy()
# wrf_d03_bch_hist_mam = wrf_d03_bch_mon_hist.copy()
# wrf_d03_noaa_hist_mam = wrf_d03_noaa_mon_hist.copy()
# 
# wrf_d03_eccc_hist_djf = wrf_d03_eccc_mon_hist.copy()
# wrf_d03_bch_hist_djf = wrf_d03_bch_mon_hist.copy()
# wrf_d03_noaa_hist_djf = wrf_d03_noaa_mon_hist.copy()
# 
# wrf_d03_eccc_hist_jja = wrf_d03_eccc_mon_hist.copy()
# wrf_d03_bch_hist_jja = wrf_d03_bch_mon_hist.copy()
# wrf_d03_noaa_hist_jja = wrf_d03_noaa_mon_hist.copy()
# 
# wrf_d03_eccc_hist_son = wrf_d03_eccc_mon_hist.copy()
# wrf_d03_bch_hist_son = wrf_d03_bch_mon_hist.copy()
# wrf_d03_noaa_hist_son = wrf_d03_noaa_mon_hist.copy()
# 
# wrf_d03_eccc_rcp45_mam = wrf_d03_eccc_mon_rcp45.copy()
# wrf_d03_bch_rcp45_mam = wrf_d03_bch_mon_rcp45.copy()
# wrf_d03_noaa_rcp45_mam = wrf_d03_noaa_mon_rcp45.copy()
# 
# wrf_d03_eccc_rcp45_djf = wrf_d03_eccc_mon_rcp45.copy()
# wrf_d03_bch_rcp45_djf = wrf_d03_bch_mon_rcp45.copy()
# wrf_d03_noaa_rcp45_djf = wrf_d03_noaa_mon_rcp45.copy()
# 
# wrf_d03_eccc_rcp45_jja = wrf_d03_eccc_mon_rcp45.copy()
# wrf_d03_bch_rcp45_jja = wrf_d03_bch_mon_rcp45.copy()
# wrf_d03_noaa_rcp45_jja = wrf_d03_noaa_mon_rcp45.copy()
# 
# wrf_d03_eccc_rcp45_son = wrf_d03_eccc_mon_rcp45.copy()
# wrf_d03_bch_rcp45_son = wrf_d03_bch_mon_rcp45.copy()
# wrf_d03_noaa_rcp45_son = wrf_d03_noaa_mon_rcp45.copy()
# 
# wrf_d03_eccc_rcp85_mam = wrf_d03_eccc_mon_rcp85.copy()
# wrf_d03_bch_rcp85_mam = wrf_d03_bch_mon_rcp85.copy()
# wrf_d03_noaa_rcp85_mam = wrf_d03_noaa_mon_rcp85.copy()
# 
# wrf_d03_eccc_rcp85_djf = wrf_d03_eccc_mon_rcp85.copy()
# wrf_d03_bch_rcp85_djf = wrf_d03_bch_mon_rcp85.copy()
# wrf_d03_noaa_rcp85_djf = wrf_d03_noaa_mon_rcp85.copy()
# 
# wrf_d03_eccc_rcp85_jja = wrf_d03_eccc_mon_rcp85.copy()
# wrf_d03_bch_rcp85_jja = wrf_d03_bch_mon_rcp85.copy()
# wrf_d03_noaa_rcp85_jja = wrf_d03_noaa_mon_rcp85.copy()
# 
# wrf_d03_eccc_rcp85_son = wrf_d03_eccc_mon_rcp85.copy()
# wrf_d03_bch_rcp85_son = wrf_d03_bch_mon_rcp85.copy()
# wrf_d03_noaa_rcp85_son = wrf_d03_noaa_mon_rcp85.copy()
# 
# for i in [1,2,6,7,8,9,10,11,12]:
#     wrf_d03_bch_hist_mam = wrf_d03_bch_hist_mam[wrf_d03_bch_hist_mam.index.month != i]
#     wrf_d03_eccc_hist_mam = wrf_d03_eccc_hist_mam[wrf_d03_eccc_hist_mam.index.month != i]
#     wrf_d03_noaa_hist_mam = wrf_d03_noaa_hist_mam[wrf_d03_noaa_hist_mam.index.month != i]
# 
#     wrf_d03_bch_rcp45_mam = wrf_d03_bch_rcp45_mam[wrf_d03_bch_rcp45_mam.index.month != i]
#     wrf_d03_eccc_rcp45_mam = wrf_d03_eccc_rcp45_mam[wrf_d03_eccc_rcp45_mam.index.month != i]
#     wrf_d03_noaa_rcp45_mam = wrf_d03_noaa_rcp45_mam[wrf_d03_noaa_rcp45_mam.index.month != i]
# 
#     wrf_d03_bch_rcp85_mam = wrf_d03_bch_rcp85_mam[wrf_d03_bch_rcp85_mam.index.month != i]
#     wrf_d03_eccc_rcp85_mam = wrf_d03_eccc_rcp85_mam[wrf_d03_eccc_rcp85_mam.index.month != i]
#     wrf_d03_noaa_rcp85_mam = wrf_d03_noaa_rcp85_mam[wrf_d03_noaa_rcp85_mam.index.month != i]
# 
# for i in [1,2,3,4,5,9,10,11,12]:
#     wrf_d03_bch_hist_jja = wrf_d03_bch_hist_jja[wrf_d03_bch_hist_jja.index.month != i]
#     wrf_d03_eccc_hist_jja = wrf_d03_eccc_hist_jja[wrf_d03_eccc_hist_jja.index.month != i]
#     wrf_d03_noaa_hist_jja = wrf_d03_noaa_hist_jja[wrf_d03_noaa_hist_jja.index.month != i]
# 
#     wrf_d03_bch_rcp45_jja = wrf_d03_bch_rcp45_jja[wrf_d03_bch_rcp45_jja.index.month != i]
#     wrf_d03_eccc_rcp45_jja = wrf_d03_eccc_rcp45_jja[wrf_d03_eccc_rcp45_jja.index.month != i]
#     wrf_d03_noaa_rcp45_jja = wrf_d03_noaa_rcp45_jja[wrf_d03_noaa_rcp45_jja.index.month != i]
# 
#     wrf_d03_bch_rcp85_jja = wrf_d03_bch_rcp85_jja[wrf_d03_bch_rcp85_jja.index.month != i]
#     wrf_d03_eccc_rcp85_jja = wrf_d03_eccc_rcp85_jja[wrf_d03_eccc_rcp85_jja.index.month != i]
#     wrf_d03_noaa_rcp85_jja = wrf_d03_noaa_rcp85_jja[wrf_d03_noaa_rcp85_jja.index.month != i]
#     
# for i in [1,2,3,4,5,6,7,8,12]:
#     wrf_d03_bch_hist_son = wrf_d03_bch_hist_son[wrf_d03_bch_hist_son.index.month != i]
#     wrf_d03_eccc_hist_son = wrf_d03_eccc_hist_son[wrf_d03_eccc_hist_son.index.month != i]
#     wrf_d03_noaa_hist_son = wrf_d03_noaa_hist_son[wrf_d03_noaa_hist_son.index.month != i]
#     
#     wrf_d03_bch_rcp45_son = wrf_d03_bch_rcp45_son[wrf_d03_bch_rcp45_son.index.month != i]
#     wrf_d03_eccc_rcp45_son = wrf_d03_eccc_rcp45_son[wrf_d03_eccc_rcp45_son.index.month != i]
#     wrf_d03_noaa_rcp45_son = wrf_d03_noaa_rcp45_son[wrf_d03_noaa_rcp45_son.index.month != i]
# 
#     wrf_d03_bch_rcp85_son = wrf_d03_bch_rcp85_son[wrf_d03_bch_rcp85_son.index.month != i]
#     wrf_d03_eccc_rcp85_son = wrf_d03_eccc_rcp85_son[wrf_d03_eccc_rcp85_son.index.month != i]
#     wrf_d03_noaa_rcp85_son = wrf_d03_noaa_rcp85_son[wrf_d03_noaa_rcp85_son.index.month != i]
#     
# for i in [3,4,5,6,7,8,9,10,11]:
#     wrf_d03_bch_hist_djf = wrf_d03_bch_hist_djf[wrf_d03_bch_hist_djf.index.month != i]
#     wrf_d03_eccc_hist_djf = wrf_d03_eccc_hist_djf[wrf_d03_eccc_hist_djf.index.month != i]
#     wrf_d03_noaa_hist_djf = wrf_d03_noaa_hist_djf[wrf_d03_noaa_hist_djf.index.month != i]
#     
#     wrf_d03_bch_rcp45_djf = wrf_d03_bch_rcp45_djf[wrf_d03_bch_rcp45_djf.index.month != i]
#     wrf_d03_eccc_rcp45_djf = wrf_d03_eccc_rcp45_djf[wrf_d03_eccc_rcp45_djf.index.month != i]
#     wrf_d03_noaa_rcp45_djf = wrf_d03_noaa_rcp45_djf[wrf_d03_noaa_rcp45_djf.index.month != i]
# 
#     wrf_d03_bch_rcp85_djf = wrf_d03_bch_rcp85_djf[wrf_d03_bch_rcp85_djf.index.month != i]
#     wrf_d03_eccc_rcp85_djf = wrf_d03_eccc_rcp85_djf[wrf_d03_eccc_rcp85_djf.index.month != i]
#     wrf_d03_noaa_rcp85_djf = wrf_d03_noaa_rcp85_djf[wrf_d03_noaa_rcp85_djf.index.month != i]
#     
# if variable == "t":
#     wrf_d03_bch_change_rcp45_mam = wrf_d03_bch_rcp45_mam.mean() - wrf_d03_bch_hist_mam.mean()
#     wrf_d03_eccc_change_rcp45_mam = wrf_d03_eccc_rcp45_mam.mean() - wrf_d03_eccc_hist_mam.mean()
#     wrf_d03_noaa_change_rcp45_mam = wrf_d03_noaa_rcp45_mam.mean() - wrf_d03_noaa_hist_mam.mean()
#     wrf_d03_bch_change_rcp85_mam = wrf_d03_bch_rcp85_mam.mean() - wrf_d03_bch_hist_mam.mean()
#     wrf_d03_eccc_change_rcp85_mam = wrf_d03_eccc_rcp85_mam.mean() - wrf_d03_eccc_hist_mam.mean()
#     wrf_d03_noaa_change_rcp85_mam = wrf_d03_noaa_rcp85_mam.mean() - wrf_d03_noaa_hist_mam.mean()
#     
#     wrf_d03_bch_change_rcp45_djf = wrf_d03_bch_rcp45_djf.mean() - wrf_d03_bch_hist_djf.mean()
#     wrf_d03_eccc_change_rcp45_djf = wrf_d03_eccc_rcp45_djf.mean() - wrf_d03_eccc_hist_djf.mean()
#     wrf_d03_noaa_change_rcp45_djf = wrf_d03_noaa_rcp45_djf.mean() - wrf_d03_noaa_hist_djf.mean()
#     wrf_d03_bch_change_rcp85_djf = wrf_d03_bch_rcp85_djf.mean() - wrf_d03_bch_hist_djf.mean()
#     wrf_d03_eccc_change_rcp85_djf = wrf_d03_eccc_rcp85_djf.mean() - wrf_d03_eccc_hist_djf.mean()
#     wrf_d03_noaa_change_rcp85_djf = wrf_d03_noaa_rcp85_djf.mean() - wrf_d03_noaa_hist_djf.mean()
#     
#     wrf_d03_bch_change_rcp45_son = wrf_d03_bch_rcp45_son.mean() - wrf_d03_bch_hist_son.mean()
#     wrf_d03_eccc_change_rcp45_son = wrf_d03_eccc_rcp45_son.mean() - wrf_d03_eccc_hist_son.mean()
#     wrf_d03_noaa_change_rcp45_son = wrf_d03_noaa_rcp45_son.mean() - wrf_d03_noaa_hist_son.mean()
#     wrf_d03_bch_change_rcp85_son = wrf_d03_bch_rcp85_son.mean() - wrf_d03_bch_hist_son.mean()
#     wrf_d03_eccc_change_rcp85_son = wrf_d03_eccc_rcp85_son.mean() - wrf_d03_eccc_hist_son.mean()
#     wrf_d03_noaa_change_rcp85_son = wrf_d03_noaa_rcp85_son.mean() - wrf_d03_noaa_hist_son.mean()
#     
#     wrf_d03_bch_change_rcp45_jja = wrf_d03_bch_rcp45_jja.mean() - wrf_d03_bch_hist_jja.mean()
#     wrf_d03_eccc_change_rcp45_jja = wrf_d03_eccc_rcp45_jja.mean() - wrf_d03_eccc_hist_jja.mean()
#     wrf_d03_noaa_change_rcp45_jja = wrf_d03_noaa_rcp45_jja.mean() - wrf_d03_noaa_hist_jja.mean()
#     wrf_d03_bch_change_rcp85_jja = wrf_d03_bch_rcp85_jja.mean() - wrf_d03_bch_hist_jja.mean()
#     wrf_d03_eccc_change_rcp85_jja = wrf_d03_eccc_rcp85_jja.mean() - wrf_d03_eccc_hist_jja.mean()
#     wrf_d03_noaa_change_rcp85_jja = wrf_d03_noaa_rcp85_jja.mean() - wrf_d03_noaa_hist_jja.mean()
# 
# elif variable == "pr":
#     wrf_d03_bch_change_rcp45_mam = (wrf_d03_bch_rcp45_mam.mean() - wrf_d03_bch_hist_mam.mean())/wrf_d03_bch_hist_mam.mean() *100
#     wrf_d03_eccc_change_rcp45_mam = (wrf_d03_eccc_rcp45_mam.mean() - wrf_d03_eccc_hist_mam.mean())/wrf_d03_eccc_hist_mam.mean() *100
#     wrf_d03_noaa_change_rcp45_mam = (wrf_d03_noaa_rcp45_mam.mean() - wrf_d03_noaa_hist_mam.mean())/wrf_d03_noaa_hist_mam.mean() *100
#     wrf_d03_bch_change_rcp85_mam = (wrf_d03_bch_rcp85_mam.mean() - wrf_d03_bch_hist_mam.mean())/wrf_d03_bch_hist_mam.mean() *100
#     wrf_d03_eccc_change_rcp85_mam = (wrf_d03_eccc_rcp85_mam.mean() - wrf_d03_eccc_hist_mam.mean())/wrf_d03_eccc_hist_mam.mean() *100
#     wrf_d03_noaa_change_rcp85_mam = (wrf_d03_noaa_rcp85_mam.mean() - wrf_d03_noaa_hist_mam.mean())/wrf_d03_noaa_hist_mam.mean() *100
# 
#     wrf_d03_bch_change_rcp45_djf = (wrf_d03_bch_rcp45_djf.mean() - wrf_d03_bch_hist_djf.mean())/wrf_d03_bch_hist_djf.mean() *100
#     wrf_d03_eccc_change_rcp45_djf = (wrf_d03_eccc_rcp45_djf.mean() - wrf_d03_eccc_hist_djf.mean())/wrf_d03_eccc_hist_djf.mean() *100
#     wrf_d03_noaa_change_rcp45_djf = (wrf_d03_noaa_rcp45_djf.mean() - wrf_d03_noaa_hist_djf.mean())/wrf_d03_noaa_hist_djf.mean() *100
#     wrf_d03_bch_change_rcp85_djf = (wrf_d03_bch_rcp85_djf.mean() - wrf_d03_bch_hist_djf.mean())/wrf_d03_bch_hist_djf.mean() *100
#     wrf_d03_eccc_change_rcp85_djf = (wrf_d03_eccc_rcp85_djf.mean() - wrf_d03_eccc_hist_djf.mean())/wrf_d03_eccc_hist_djf.mean() *100
#     wrf_d03_noaa_change_rcp85_djf = (wrf_d03_noaa_rcp85_djf.mean() - wrf_d03_noaa_hist_djf.mean())/wrf_d03_noaa_hist_djf.mean() *100
# 
#     wrf_d03_bch_change_rcp45_son = (wrf_d03_bch_rcp45_son.mean() - wrf_d03_bch_hist_son.mean())/wrf_d03_bch_hist_son.mean() *100
#     wrf_d03_eccc_change_rcp45_son = (wrf_d03_eccc_rcp45_son.mean() - wrf_d03_eccc_hist_son.mean())/wrf_d03_eccc_hist_son.mean() *100
#     wrf_d03_noaa_change_rcp45_son = (wrf_d03_noaa_rcp45_son.mean() - wrf_d03_noaa_hist_son.mean())/wrf_d03_noaa_hist_son.mean() *100
#     wrf_d03_bch_change_rcp85_son = (wrf_d03_bch_rcp85_son.mean() - wrf_d03_bch_hist_son.mean())/wrf_d03_bch_hist_son.mean() *100
#     wrf_d03_eccc_change_rcp85_son = (wrf_d03_eccc_rcp85_son.mean() - wrf_d03_eccc_hist_son.mean())/wrf_d03_eccc_hist_son.mean() *100
#     wrf_d03_noaa_change_rcp85_son = (wrf_d03_noaa_rcp85_son.mean() - wrf_d03_noaa_hist_son.mean())/wrf_d03_noaa_hist_son.mean() *100
# 
#     wrf_d03_bch_change_rcp45_jja = (wrf_d03_bch_rcp45_jja.mean() - wrf_d03_bch_hist_jja.mean())/wrf_d03_bch_hist_jja.mean() *100
#     wrf_d03_eccc_change_rcp45_jja = (wrf_d03_eccc_rcp45_jja.mean() - wrf_d03_eccc_hist_jja.mean())/wrf_d03_eccc_hist_jja.mean() *100
#     wrf_d03_noaa_change_rcp45_jja = (wrf_d03_noaa_rcp45_jja.mean() - wrf_d03_noaa_hist_jja.mean())/wrf_d03_noaa_hist_jja.mean() *100
#     wrf_d03_bch_change_rcp85_jja = (wrf_d03_bch_rcp85_jja.mean() - wrf_d03_bch_hist_jja.mean())/wrf_d03_bch_hist_jja.mean() *100
#     wrf_d03_eccc_change_rcp85_jja = (wrf_d03_eccc_rcp85_jja.mean() - wrf_d03_eccc_hist_jja.mean())/wrf_d03_eccc_hist_jja.mean() *100
#     wrf_d03_noaa_change_rcp85_jja = (wrf_d03_noaa_rcp85_jja.mean() - wrf_d03_noaa_hist_jja.mean())/wrf_d03_noaa_hist_jja.mean() *100
# 
# #%%
# if variable == "t":
#     wrf_change_eccc_rcp45 = np.mean(wrf_d03_eccc_rcp45) - np.mean(wrf_d03_eccc_hist)
#     wrf_change_bch_rcp45 = np.mean(wrf_d03_bch_rcp45) - np.mean(wrf_d03_bch_hist)
#     wrf_change_noaa_rcp45 = np.mean(wrf_d03_noaa_rcp45) - np.mean(wrf_d03_noaa_hist)
# 
#     wrf_change_eccc_rcp85 = np.mean(wrf_d03_eccc_rcp85) - np.mean(wrf_d03_eccc_hist)
#     wrf_change_bch_rcp85 = np.mean(wrf_d03_bch_rcp85) - np.mean(wrf_d03_bch_hist)
#     wrf_change_noaa_rcp85 = np.mean(wrf_d03_noaa_rcp85) - np.mean(wrf_d03_noaa_hist)
# 
# elif variable == "pr":
#     wrf_change_eccc_rcp45 = (np.mean(wrf_d03_eccc_rcp45) - np.mean(wrf_d03_eccc_hist))/np.mean(wrf_d03_eccc_hist) * 100
#     wrf_change_bch_rcp45 = (np.mean(wrf_d03_bch_rcp45) - np.mean(wrf_d03_bch_hist))/np.mean(wrf_d03_bch_hist) * 100
#     wrf_change_noaa_rcp45 = (np.mean(wrf_d03_noaa_rcp45) - np.mean(wrf_d03_noaa_hist))/np.mean(wrf_d03_noaa_hist) * 100
# 
#     wrf_change_eccc_rcp85 = (np.mean(wrf_d03_eccc_rcp85) - np.mean(wrf_d03_eccc_hist))/np.mean(wrf_d03_eccc_hist) * 100
#     wrf_change_bch_rcp85 = (np.mean(wrf_d03_bch_rcp85) - np.mean(wrf_d03_bch_hist))/np.mean(wrf_d03_bch_hist) * 100
#     wrf_change_noaa_rcp85 = (np.mean(wrf_d03_noaa_rcp85) - np.mean(wrf_d03_noaa_hist))/np.mean(wrf_d03_noaa_hist) * 100
# 
# 
# =============================================================================

#%%



#t_colors = [[149/256, 187/256, 234/256, 1],[254/256, 240/256, 217/256, 1],[253/256, 219/256, 171/256, 1],[252/256, 195/256, 131/256, 1],
#            [252/256, 159/256, 103/256, 1],[244/256, 121/256, 78/256, 1],[230/256, 83/256, 56/256, 1],[206/256, 42/256, 29/256, 1],[179/256, 0/256, 0/256, 1]]

#t_colors = ['#f6fbfc','#f4f57f','#fffcae','#fbd28b','#fabb4a','#f57a3f','#f0593d','#ed3f24','#eb2623','#e61b26','#cb1b23','#86161a','#631316','#b2289a','#9f2697','#552378','#261229']
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



#def plot_change(gridded_data,vmin,vmax,eccc_change,bch_change,noaa_change,title,cmap,fig_name):
def plot_change(gridded_data,vmin,vmax,title,cmap,fig_name):

    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    
    #plt.scatter(eccc_lons, eccc_lats, c=eccc_change,s=300,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    #plt.scatter(bch_lons, bch_lats, c=bch_change,s=300,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    #plt.scatter(noaa_lons, noaa_lats, c=noaa_change,s=300,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)

    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    ax1.text(-3700000, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
    
    plt.title(title,fontsize=20)


    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(0, vmax+1, 0.5))
    cbar_ax.tick_params(labelsize=25)
    if variable == "t":
        cbar_ax.set_xlabel("Temperature Change (deg C)",size=25) 
    elif variable == "pr":
        cbar_ax.set_xlabel("Precipitation Change (%)",size=25)      

    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/future_changes/' + fig_name + '_' + variable + '.png',bbox_inches='tight')



if variable == "t":
    vmin= 0
    vmax = 5
    #cmap = newcmp_t
    cmap = cm.get_cmap('YlOrRd', 24)


elif variable == "pr":
    vmin= -80
    vmax = 80
    cmap = newcmp_pr
    
title = '2046-2065 mean relative to 1986-2005'

# =============================================================================
# plot_change(ann_var_change_rcp45,vmin,vmax,wrf_change_eccc_rcp45,wrf_change_bch_rcp45,wrf_change_noaa_rcp45,'Annual RCP4.5 ' + title,cmap,'annual_rcp45')
# plot_change(ann_var_change_rcp85,vmin,vmax,wrf_change_eccc_rcp85,wrf_change_bch_rcp85,wrf_change_noaa_rcp85,'Annual RCP8.5 ' + title,cmap,'annual_rcp85')
# 
# plot_change(djf_var_change_rcp45,vmin,vmax,wrf_d03_eccc_change_rcp45_djf,wrf_d03_bch_change_rcp45_djf,wrf_d03_noaa_change_rcp45_djf,'DJF RCP4.5 ' + title,cmap,'djf_rcp45')
# plot_change(djf_var_change_rcp85,vmin,vmax,wrf_d03_eccc_change_rcp85_djf,wrf_d03_bch_change_rcp85_djf,wrf_d03_noaa_change_rcp85_djf,'DJF RCP8.5 ' + title,cmap,'djf_rcp85')
# 
# plot_change(mam_var_change_rcp45,vmin,vmax,wrf_d03_eccc_change_rcp45_mam,wrf_d03_bch_change_rcp45_mam,wrf_d03_noaa_change_rcp45_mam,'MAM RCP4.5 ' + title,cmap,'mam_rcp45')
# plot_change(mam_var_change_rcp85,vmin,vmax,wrf_d03_eccc_change_rcp85_mam,wrf_d03_bch_change_rcp85_mam,wrf_d03_noaa_change_rcp85_mam,'MAM RCP8.5 ' + title,cmap,'mam_rcp85')
# 
# plot_change(jja_var_change_rcp45,vmin,vmax,wrf_d03_eccc_change_rcp45_jja,wrf_d03_bch_change_rcp45_jja,wrf_d03_noaa_change_rcp45_jja,'JJA RCP4.5 ' + title,cmap,'jja_rcp45')
# plot_change(jja_var_change_rcp85,vmin,vmax,wrf_d03_eccc_change_rcp85_jja,wrf_d03_bch_change_rcp85_jja,wrf_d03_noaa_change_rcp85_jja,'JJA RCP8.5 ' + title,cmap,'jja_rcp85')
# 
# plot_change(son_var_change_rcp45,vmin,vmax,wrf_d03_eccc_change_rcp45_son,wrf_d03_bch_change_rcp45_son,wrf_d03_noaa_change_rcp45_son,'SON RCP4.5 ' + title,cmap,'son_rcp45')
# plot_change(son_var_change_rcp85,vmin,vmax,wrf_d03_eccc_change_rcp85_son,wrf_d03_bch_change_rcp85_son,wrf_d03_noaa_change_rcp85_son,'SON RCP8.5 ' + title,cmap,'son_rcp85')
# 
# =============================================================================

plot_change(ann_var_change_rcp45,vmin,vmax,'Annual RCP4.5 ' + title,cmap,'annual_rcp45')
plot_change(ann_var_change_rcp85,vmin,vmax,'Annual RCP8.5 ' + title,cmap,'annual_rcp85')

plot_change(djf_var_change_rcp45,vmin,vmax,'DJF RCP4.5 ' + title,cmap,'djf_rcp45')
plot_change(djf_var_change_rcp85,vmin,vmax,'DJF RCP8.5 ' + title,cmap,'djf_rcp85')

plot_change(mam_var_change_rcp45,vmin,vmax,'MAM RCP4.5 ' + title,cmap,'mam_rcp45')
plot_change(mam_var_change_rcp85,vmin,vmax,'MAM RCP8.5 ' + title,cmap,'mam_rcp85')

plot_change(jja_var_change_rcp45,vmin,vmax,'JJA RCP4.5 ' + title,cmap,'jja_rcp45')
plot_change(jja_var_change_rcp85,vmin,vmax,'JJA RCP8.5 ' + title,cmap,'jja_rcp85')

plot_change(son_var_change_rcp45,vmin,vmax,'SON RCP4.5 ' + title,cmap,'son_rcp45')
plot_change(son_var_change_rcp85,vmin,vmax,'SON RCP8.5 ' + title,cmap,'son_rcp85')
