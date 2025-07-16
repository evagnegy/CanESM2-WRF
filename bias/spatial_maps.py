import pandas as pd
import warnings
import sys
import numpy as np
from netCDF4 import Dataset
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2-WRF-scripts/functions/')
from canesm2_eval_funcs import *
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib
from datetime import datetime, timedelta
import math
from make_colorbars import make_colorbar

variable = 'wind' #t or pr or wind
run = 'historical' #historical rcp45 or rcp85
output_freq = "yearly" #yearly monthly or daily
#%%
noaa_daily_stations_buoys = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_buoys.csv'
eccc_daily_stations_buoys = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_buoys.csv'


if variable == "wind":
    eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations_hourly.csv'
    noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations_wind.csv'

elif variable == "t":
    eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
    noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations_tas.csv'

elif variable == "pr":
    eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
    noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations.csv'

bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'

df = pd.read_csv(eccc_daily_stations,header=None)
eccc_station_IDs = list(df.iloc[:,4])
eccc_station_names = list(df.iloc[:,1])

eccc_lats = df.iloc[:,7]
eccc_lons = df.iloc[:,8]
eccc_lats.index = eccc_station_IDs
eccc_lons.index = eccc_station_IDs

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])

bch_lats = df['Y']
bch_lons = df['X']
bch_lats.index = bch_station_IDs
bch_lons.index = bch_station_IDs

df = pd.read_csv(noaa_daily_stations)

noaa_station_IDs = list(df.iloc[:,0])
noaa_station_names = list(df.iloc[:,1])

noaa_lats = df.iloc[:,2]
noaa_lons = df.iloc[:,3]
noaa_lats.index = noaa_station_IDs
noaa_lons.index = noaa_station_IDs


df = pd.read_csv(noaa_daily_stations_buoys)
noaa_buoy_station_IDs = list(df["STATION_ID"])

noaa_buoy_lats = df['Y']
noaa_buoy_lons = df['X']
noaa_buoy_heights = df['Z']
noaa_buoy_heights.index = noaa_buoy_station_IDs


df = pd.read_csv(eccc_daily_stations_buoys)
eccc_buoy_station_IDs = list(df["STATION_ID"])

eccc_buoy_lats = df['Y']
eccc_buoy_lons = df['X']

#%%

stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/'

WRF_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/' + run + '/'
raw_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/' + run + '/'
rcm_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/' + run + '/'
pcic_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_PCIC/' + run + '/'


if run == 'historical':
    start_year = 1986
    end_year = 2005
else:
    start_year = 2046
    end_year = 2065  

#%%

if run == 'historical': #no station obs for rcps
    eccc_obs = get_eccc_obs(output_freq,eccc_station_IDs,stations_dir,variable)
    noaa_obs = get_noaa_obs(output_freq,noaa_station_IDs,stations_dir,variable)

    if variable != "pr":
        noaa_buoy_obs = get_noaa_buoys(output_freq,noaa_buoy_station_IDs,noaa_buoy_heights,stations_dir,variable)
        eccc_buoy_obs = get_eccc_buoys(output_freq,eccc_buoy_station_IDs,stations_dir,variable)

    if variable != "wind":
        bch_obs = get_bch_obs(output_freq,bch_station_IDs,stations_dir,variable)
#%

if variable != "wind":
    wrf_d01_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d01", run, variable, WRF_files_dir,start_year)
    wrf_d02_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
    wrf_d03_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
    raw_bch = get_canesm2(output_freq, "BCH", bch_station_IDs, run, variable, raw_files_dir,start_year)
    rcm_bch = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, variable, rcm_files_dir)
    pcic_bch = get_pcic(output_freq, "BCH", bch_station_IDs, run, variable, pcic_files_dir)
    
    pcic_eccc = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, variable, pcic_files_dir)

if variable != "pr":
    wrf_d01_noaa_buoy = get_wrf(output_freq, "NOAA_buoy", noaa_buoy_station_IDs, "d01", run, variable, WRF_files_dir,start_year)
    wrf_d02_noaa_buoy = get_wrf(output_freq, "NOAA_buoy", noaa_buoy_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
    wrf_d03_noaa_buoy = get_wrf(output_freq, "NOAA_buoy", noaa_buoy_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
    raw_noaa_buoy = get_canesm2(output_freq, "NOAA_buoy", noaa_buoy_station_IDs, run, variable, raw_files_dir,start_year)
    rcm_noaa_buoy = get_canrcm4(output_freq, "NOAA_buoy", noaa_buoy_station_IDs, run, variable, rcm_files_dir)

    wrf_d01_eccc_buoy = get_wrf(output_freq, "ECCC_buoy", eccc_buoy_station_IDs, "d01", run, variable, WRF_files_dir,start_year)
    wrf_d02_eccc_buoy = get_wrf(output_freq, "ECCC_buoy", eccc_buoy_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
    wrf_d03_eccc_buoy = get_wrf(output_freq, "ECCC_buoy", eccc_buoy_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
    raw_eccc_buoy = get_canesm2(output_freq, "ECCC_buoy", eccc_buoy_station_IDs, run, variable, raw_files_dir,start_year)
    rcm_eccc_buoy = get_canrcm4(output_freq, "ECCC_buoy", eccc_buoy_station_IDs, run, variable, rcm_files_dir)

wrf_d01_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d01", run, variable, WRF_files_dir,start_year)
wrf_d02_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_eccc = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, variable, raw_files_dir,start_year)
rcm_eccc = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, variable, rcm_files_dir)

wrf_d01_noaa = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d01", run, variable, WRF_files_dir,start_year)
wrf_d02_noaa = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_noaa = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_noaa = get_canesm2(output_freq, "NOAA", noaa_station_IDs, run, variable, raw_files_dir,start_year)
rcm_noaa = get_canrcm4(output_freq, "NOAA", noaa_station_IDs, run, variable, rcm_files_dir)


#%%

# =============================================================================
# if variable == "pr":
#     for i in eccc_station_IDs:
#         if pcic_eccc[i].sum() == 0:
#             pcic_eccc.loc[:,i] = np.nan
# 
# if variable != "pr":
#     wrf_d03_noaa_buoy = wrf_d03_noaa_buoy.mask(noaa_buoy_obs.isnull())
#     wrf_d02_noaa_buoy = wrf_d02_noaa_buoy.mask(noaa_buoy_obs.isnull())
#     raw_noaa_buoy = raw_noaa_buoy.mask(noaa_buoy_obs.isnull())
#     rcm_noaa_buoy = rcm_noaa_buoy.mask(noaa_buoy_obs.isnull())
# 
#     wrf_d03_eccc_buoy = wrf_d03_eccc_buoy.mask(eccc_buoy_obs.isnull())
#     wrf_d02_eccc_buoy = wrf_d02_eccc_buoy.mask(eccc_buoy_obs.isnull())
#     raw_eccc_buoy = raw_eccc_buoy.mask(eccc_buoy_obs.isnull())
#     rcm_eccc_buoy = rcm_eccc_buoy.mask(eccc_buoy_obs.isnull())
#        
# wrf_d02_eccc = wrf_d02_eccc.mask(eccc_obs.isnull())
# wrf_d03_eccc = wrf_d03_eccc.mask(eccc_obs.isnull())
# raw_eccc = raw_eccc.mask(eccc_obs.isnull())
# rcm_eccc = rcm_eccc.mask(eccc_obs.isnull())
# 
# wrf_d02_noaa = wrf_d02_noaa.mask(noaa_obs.isnull())
# wrf_d03_noaa = wrf_d03_noaa.mask(noaa_obs.isnull())
# raw_noaa = raw_noaa.mask(noaa_obs.isnull())
# rcm_noaa = rcm_noaa.mask(noaa_obs.isnull())
# 
# if variable != "wind":
#     wrf_d02_bch = wrf_d02_bch.mask(bch_obs.isnull())
#     wrf_d03_bch = wrf_d03_bch.mask(bch_obs.isnull())
#     raw_bch = raw_bch.mask(bch_obs.isnull())
#     rcm_bch = rcm_bch.mask(bch_obs.isnull())
#     pcic_bch = pcic_bch.mask(bch_obs.isnull())
#     
#     pcic_eccc = pcic_eccc.mask(eccc_obs.isnull())
# =============================================================================


#%%
def get_seas_values(df):
    
    vals_MAM = df[(df.index.month >= 3) & (df.index.month <= 5)]
    vals_JJA = df[(df.index.month >= 6) & (df.index.month <= 8)]
    vals_SON = df[(df.index.month >= 9) & (df.index.month <= 11)]
    vals_DJF = df[(df.index.month == 1) | (df.index.month == 2) | (df.index.month == 12)]

    return(vals_MAM,vals_JJA,vals_SON,vals_DJF)
    
eccc_MAM,eccc_JJA,eccc_SON,eccc_DJF = get_seas_values(eccc_obs)
wrf_d02_eccc_MAM,wrf_d02_eccc_JJA,wrf_d02_eccc_SON,wrf_d02_eccc_DJF = get_seas_values(wrf_d02_eccc)
wrf_d03_eccc_MAM,wrf_d03_eccc_JJA,wrf_d03_eccc_SON,wrf_d03_eccc_DJF = get_seas_values(wrf_d03_eccc)
raw_eccc_MAM,raw_eccc_JJA,raw_eccc_SON,raw_eccc_DJF = get_seas_values(raw_eccc)
rcm_eccc_MAM,rcm_eccc_JJA,rcm_eccc_SON,rcm_eccc_DJF = get_seas_values(rcm_eccc)

if variable != "wind":
    pcic_eccc_MAM,pcic_eccc_JJA,pcic_eccc_SON,pcic_eccc_DJF = get_seas_values(pcic_eccc)
    
    bch_MAM,bch_JJA,bch_SON,bch_DJF = get_seas_values(bch_obs)
    wrf_d02_bch_MAM,wrf_d02_bch_JJA,wrf_d02_bch_SON,wrf_d02_bch_DJF = get_seas_values(wrf_d02_bch)
    wrf_d03_bch_MAM,wrf_d03_bch_JJA,wrf_d03_bch_SON,wrf_d03_bch_DJF = get_seas_values(wrf_d03_bch)
    raw_bch_MAM,raw_bch_JJA,raw_bch_SON,raw_bch_DJF = get_seas_values(raw_bch)
    rcm_bch_MAM,rcm_bch_JJA,rcm_bch_SON,rcm_bch_DJF = get_seas_values(rcm_bch)
    pcic_bch_MAM,pcic_bch_JJA,pcic_bch_SON,pcic_bch_DJF = get_seas_values(pcic_bch)

noaa_MAM,noaa_JJA,noaa_SON,noaa_DJF = get_seas_values(noaa_obs)
wrf_d02_noaa_MAM,wrf_d02_noaa_JJA,wrf_d02_noaa_SON,wrf_d02_noaa_DJF = get_seas_values(wrf_d02_noaa)
wrf_d03_noaa_MAM,wrf_d03_noaa_JJA,wrf_d03_noaa_SON,wrf_d03_noaa_DJF = get_seas_values(wrf_d03_noaa)
raw_noaa_MAM,raw_noaa_JJA,raw_noaa_SON,raw_noaa_DJF = get_seas_values(raw_noaa)
rcm_noaa_MAM,rcm_noaa_JJA,rcm_noaa_SON,rcm_noaa_DJF = get_seas_values(rcm_noaa)

#%% annual

if variable == "t": #or variable=="wind":
    
    wrf_bias_eccc_d01 = np.mean(wrf_d01_eccc) - np.mean(eccc_obs)
    wrf_bias_noaa_d01 = np.mean(wrf_d01_noaa) - np.mean(noaa_obs)
    wrf_bias_eccc_buoy_d01 = np.mean(wrf_d01_eccc_buoy) - np.mean(eccc_buoy_obs)
    wrf_bias_noaa_buoy_d01 = np.mean(wrf_d01_noaa_buoy) - np.mean(noaa_buoy_obs)
    
    
    wrf_bias_eccc_d02 = np.mean(wrf_d02_eccc) - np.mean(eccc_obs)
    wrf_bias_noaa_d02 = np.mean(wrf_d02_noaa) - np.mean(noaa_obs)
    wrf_bias_eccc_buoy_d02 = np.mean(wrf_d02_eccc_buoy) - np.mean(eccc_buoy_obs)
    wrf_bias_noaa_buoy_d02 = np.mean(wrf_d02_noaa_buoy) - np.mean(noaa_buoy_obs)
    
    wrf_bias_eccc = np.mean(wrf_d03_eccc) - np.mean(eccc_obs)
    wrf_bias_noaa = np.mean(wrf_d03_noaa) - np.mean(noaa_obs)
    wrf_bias_eccc_buoy = np.mean(wrf_d03_eccc_buoy) - np.mean(eccc_buoy_obs)
    wrf_bias_noaa_buoy = np.mean(wrf_d03_noaa_buoy) - np.mean(noaa_buoy_obs)
    
    raw_bias_eccc = np.mean(raw_eccc) - np.mean(eccc_obs)
    raw_bias_noaa = np.mean(raw_noaa) - np.mean(noaa_obs)
    raw_bias_eccc_buoy = np.mean(raw_eccc_buoy) - np.mean(eccc_buoy_obs)
    raw_bias_noaa_buoy = np.mean(raw_noaa_buoy) - np.mean(noaa_buoy_obs)
    
    rcm_bias_eccc = np.mean(rcm_eccc) - np.mean(eccc_obs)
    rcm_bias_noaa = np.mean(rcm_noaa) - np.mean(noaa_obs)
    rcm_bias_eccc_buoy = np.mean(rcm_eccc_buoy) - np.mean(eccc_buoy_obs)
    rcm_bias_noaa_buoy = np.mean(rcm_noaa_buoy) - np.mean(noaa_buoy_obs)
    
    
    wrf_bias_bch_d01 = np.mean(wrf_d01_bch) - np.mean(bch_obs)
    wrf_bias_bch_d02 = np.mean(wrf_d02_bch) - np.mean(bch_obs)
    wrf_bias_bch = np.mean(wrf_d03_bch) - np.mean(bch_obs)
    raw_bias_bch = np.mean(raw_bch) - np.mean(bch_obs)
    rcm_bias_bch = np.mean(rcm_bch) - np.mean(bch_obs)
    
    #pcic_bias_eccc = np.mean(pcic_eccc) - np.mean(eccc_obs)
    #pcic_bias_bch = np.mean(pcic_bch) - np.mean(bch_obs)
    
    #else:
    #    wrf_bias_bch_d02=[]
    #    wrf_bias_bch=[]
    #    raw_bias_bch=[]
    #    rcm_bias_bch=[]
        
elif variable=="pr" or variable=="wind":
    wrf_bias_eccc_d01 = 100*(np.mean(wrf_d01_eccc) - np.mean(eccc_obs))/np.mean(eccc_obs) 
    wrf_bias_noaa_d01 = 100*(np.mean(wrf_d01_noaa) - np.mean(noaa_obs))/np.mean(noaa_obs)
    
    wrf_bias_eccc_d02 = 100*(np.mean(wrf_d02_eccc) - np.mean(eccc_obs))/np.mean(eccc_obs) 
    wrf_bias_noaa_d02 = 100*(np.mean(wrf_d02_noaa) - np.mean(noaa_obs))/np.mean(noaa_obs)
    
    wrf_bias_eccc = 100*(np.mean(wrf_d03_eccc) - np.mean(eccc_obs))/np.mean(eccc_obs)
    wrf_bias_noaa = 100*(np.mean(wrf_d03_noaa) - np.mean(noaa_obs))/np.mean(noaa_obs)
    
    raw_bias_eccc = 100*(np.mean(raw_eccc) - np.mean(eccc_obs))/np.mean(eccc_obs)
    raw_bias_noaa = 100*(np.mean(raw_noaa) - np.mean(noaa_obs))/np.mean(noaa_obs)
    
    rcm_bias_eccc = 100*(np.mean(rcm_eccc) - np.mean(eccc_obs))/np.mean(eccc_obs)
    rcm_bias_noaa = 100*(np.mean(rcm_noaa) - np.mean(noaa_obs))/np.mean(noaa_obs)
    
    if variable == "pr":
        wrf_bias_bch_d01 = 100*(np.mean(wrf_d01_bch) - np.mean(bch_obs))/np.mean(bch_obs)
        wrf_bias_bch_d02 = 100*(np.mean(wrf_d02_bch) - np.mean(bch_obs))/np.mean(bch_obs)
        wrf_bias_bch = 100*(np.mean(wrf_d03_bch) - np.mean(bch_obs))/np.mean(bch_obs)
        raw_bias_bch = 100*(np.mean(raw_bch) - np.mean(bch_obs))/np.mean(bch_obs)
        rcm_bias_bch = 100*(np.mean(rcm_bch) - np.mean(bch_obs))/np.mean(bch_obs)
        
        pcic_bias_eccc = 100*(np.mean(pcic_eccc) - np.mean(eccc_obs))/np.mean(eccc_obs)
        pcic_bias_bch = 100*(np.mean(pcic_bch) - np.mean(bch_obs))/np.mean(bch_obs)
    
        wrf_bias_eccc_buoy=[]
        wrf_bias_eccc_buoy_d01=[]
        wrf_bias_eccc_buoy_d02=[]
        raw_bias_eccc_buoy=[]
        rcm_bias_eccc_buoy=[]
        
        wrf_bias_noaa_buoy=[]
        wrf_bias_noaa_buoy_d01=[]
        wrf_bias_noaa_buoy_d02=[]
        raw_bias_noaa_buoy=[]
        rcm_bias_noaa_buoy=[]
        
    else:
        wrf_bias_bch_d01=[]
        wrf_bias_bch_d02=[]
        wrf_bias_bch=[]
        raw_bias_bch=[]
        rcm_bias_bch=[]


        wrf_bias_eccc_buoy = 100*(np.mean(wrf_d03_eccc_buoy) - np.mean(eccc_buoy_obs))/np.mean(eccc_buoy_obs)
        wrf_bias_eccc_buoy_d01 = 100*(np.mean(wrf_d01_eccc_buoy) - np.mean(eccc_buoy_obs))/np.mean(eccc_buoy_obs)
        wrf_bias_eccc_buoy_d02 = 100*(np.mean(wrf_d02_eccc_buoy) - np.mean(eccc_buoy_obs))/np.mean(eccc_buoy_obs)
        raw_bias_eccc_buoy = 100*(np.mean(raw_eccc_buoy) - np.mean(eccc_buoy_obs))/np.mean(eccc_buoy_obs)
        rcm_bias_eccc_buoy = 100*(np.mean(rcm_eccc_buoy) - np.mean(eccc_buoy_obs))/np.mean(eccc_buoy_obs)

        wrf_bias_noaa_buoy = 100*(np.mean(wrf_d03_noaa_buoy) - np.mean(noaa_buoy_obs))/np.mean(noaa_buoy_obs)
        wrf_bias_noaa_buoy_d01 = 100*(np.mean(wrf_d01_noaa_buoy) - np.mean(noaa_buoy_obs))/np.mean(noaa_buoy_obs)
        wrf_bias_noaa_buoy_d02 = 100*(np.mean(wrf_d02_noaa_buoy) - np.mean(noaa_buoy_obs))/np.mean(noaa_buoy_obs)
        raw_bias_noaa_buoy = 100*(np.mean(raw_noaa_buoy) - np.mean(noaa_buoy_obs))/np.mean(noaa_buoy_obs)
        rcm_bias_noaa_buoy = 100*(np.mean(rcm_noaa_buoy) - np.mean(noaa_buoy_obs))/np.mean(noaa_buoy_obs)

    

#%% seasonal

if variable!="pr":
    wrf_bias_eccc_d02_MAM = np.mean(wrf_d02_eccc_MAM) - np.mean(eccc_MAM)
    wrf_bias_noaa_d02_MAM = np.mean(wrf_d02_noaa_MAM) - np.mean(noaa_MAM)
    
    wrf_bias_eccc_MAM = np.mean(wrf_d03_eccc_MAM) - np.mean(eccc_MAM)
    wrf_bias_noaa_MAM = np.mean(wrf_d03_noaa_MAM) - np.mean(noaa_MAM)
    
    raw_bias_eccc_MAM = np.mean(raw_eccc_MAM) - np.mean(eccc_MAM)
    raw_bias_noaa_MAM = np.mean(raw_noaa_MAM) - np.mean(noaa_MAM)
    
    rcm_bias_eccc_MAM = np.mean(rcm_eccc_MAM) - np.mean(eccc_MAM)
    rcm_bias_noaa_MAM = np.mean(rcm_noaa_MAM) - np.mean(noaa_MAM)
    
    if variable != "wind":
        wrf_bias_bch_d02_MAM = np.mean(wrf_d02_bch_MAM) - np.mean(bch_MAM)
        wrf_bias_bch_MAM = np.mean(wrf_d03_bch_MAM) - np.mean(bch_MAM)
        raw_bias_bch_MAM = np.mean(raw_bch_MAM) - np.mean(bch_MAM)
        rcm_bias_bch_MAM = np.mean(rcm_bch_MAM) - np.mean(bch_MAM)
        pcic_bias_eccc_MAM = np.mean(pcic_eccc_MAM) - np.mean(eccc_MAM)
        pcic_bias_bch_MAM = np.mean(pcic_bch_MAM) - np.mean(bch_MAM)
    
    else:
        wrf_bias_bch_d02_MAM,wrf_bias_bch_MAM,raw_bias_bch_MAM,rcm_bias_bch_MAM=[],[],[],[]
    
    
    wrf_bias_eccc_d02_JJA = np.mean(wrf_d02_eccc_JJA) - np.mean(eccc_JJA)
    wrf_bias_noaa_d02_JJA = np.mean(wrf_d02_noaa_JJA) - np.mean(noaa_JJA)
    
    wrf_bias_eccc_JJA = np.mean(wrf_d03_eccc_JJA) - np.mean(eccc_JJA)
    wrf_bias_noaa_JJA = np.mean(wrf_d03_noaa_JJA) - np.mean(noaa_JJA)
    
    raw_bias_eccc_JJA = np.mean(raw_eccc_JJA) - np.mean(eccc_JJA)
    raw_bias_noaa_JJA = np.mean(raw_noaa_JJA) - np.mean(noaa_JJA)
    
    rcm_bias_eccc_JJA = np.mean(rcm_eccc_JJA) - np.mean(eccc_JJA)
    rcm_bias_noaa_JJA = np.mean(rcm_noaa_JJA) - np.mean(noaa_JJA)
    
    if variable != "wind":
        wrf_bias_bch_d02_JJA = np.mean(wrf_d02_bch_JJA) - np.mean(bch_JJA)
        wrf_bias_bch_JJA = np.mean(wrf_d03_bch_JJA) - np.mean(bch_JJA)
        raw_bias_bch_JJA = np.mean(raw_bch_JJA) - np.mean(bch_JJA)
        rcm_bias_bch_JJA = np.mean(rcm_bch_JJA) - np.mean(bch_JJA)
        pcic_bias_eccc_JJA = np.mean(pcic_eccc_JJA) - np.mean(eccc_JJA)
        pcic_bias_bch_JJA = np.mean(pcic_bch_JJA) - np.mean(bch_JJA)
    
    else:
        wrf_bias_bch_d02_JJA,wrf_bias_bch_JJA,raw_bias_bch_JJA,rcm_bias_bch_JJA=[],[],[],[]
        
    
    
    wrf_bias_eccc_d02_SON = np.mean(wrf_d02_eccc_SON) - np.mean(eccc_SON)
    wrf_bias_noaa_d02_SON = np.mean(wrf_d02_noaa_SON) - np.mean(noaa_SON)
    
    wrf_bias_eccc_SON = np.mean(wrf_d03_eccc_SON) - np.mean(eccc_SON)
    wrf_bias_noaa_SON = np.mean(wrf_d03_noaa_SON) - np.mean(noaa_SON)
    
    raw_bias_eccc_SON = np.mean(raw_eccc_SON) - np.mean(eccc_SON)
    raw_bias_noaa_SON = np.mean(raw_noaa_SON) - np.mean(noaa_SON)
    
    rcm_bias_eccc_SON = np.mean(rcm_eccc_SON) - np.mean(eccc_SON)
    rcm_bias_noaa_SON = np.mean(rcm_noaa_SON) - np.mean(noaa_SON)
    
    if variable != "wind":
        wrf_bias_bch_d02_SON = np.mean(wrf_d02_bch_SON) - np.mean(bch_SON)
        wrf_bias_bch_SON = np.mean(wrf_d03_bch_SON) - np.mean(bch_SON)
        raw_bias_bch_SON = np.mean(raw_bch_SON) - np.mean(bch_SON)
        rcm_bias_bch_SON = np.mean(rcm_bch_SON) - np.mean(bch_SON)
        pcic_bias_eccc_SON = np.mean(pcic_eccc_SON) - np.mean(eccc_SON)
        pcic_bias_bch_SON = np.mean(pcic_bch_SON) - np.mean(bch_SON)
    
    else:
        wrf_bias_bch_d02_SON,wrf_bias_bch_SON,raw_bias_bch_SON,rcm_bias_bch_SON=[],[],[],[]
    
    
    
    wrf_bias_eccc_d02_DJF = np.mean(wrf_d02_eccc_DJF) - np.mean(eccc_DJF)
    wrf_bias_noaa_d02_DJF = np.mean(wrf_d02_noaa_DJF) - np.mean(noaa_DJF)
    
    wrf_bias_eccc_DJF = np.mean(wrf_d03_eccc_DJF) - np.mean(eccc_DJF)
    wrf_bias_noaa_DJF = np.mean(wrf_d03_noaa_DJF) - np.mean(noaa_DJF)
    
    raw_bias_eccc_DJF = np.mean(raw_eccc_DJF) - np.mean(eccc_DJF)
    raw_bias_noaa_DJF = np.mean(raw_noaa_DJF) - np.mean(noaa_DJF)
    
    rcm_bias_eccc_DJF = np.mean(rcm_eccc_DJF) - np.mean(eccc_DJF)
    rcm_bias_noaa_DJF = np.mean(rcm_noaa_DJF) - np.mean(noaa_DJF)
    
    if variable != "wind":
        wrf_bias_bch_d02_DJF = np.mean(wrf_d02_bch_DJF) - np.mean(bch_DJF)
        wrf_bias_bch_DJF = np.mean(wrf_d03_bch_DJF) - np.mean(bch_DJF)
        raw_bias_bch_DJF = np.mean(raw_bch_DJF) - np.mean(bch_DJF)
        rcm_bias_bch_DJF = np.mean(rcm_bch_DJF) - np.mean(bch_DJF)      
        pcic_bias_eccc_DJF = np.mean(pcic_eccc_DJF) - np.mean(eccc_DJF)
        pcic_bias_bch_DJF = np.mean(pcic_bch_DJF) - np.mean(bch_DJF)
    
    else:
        wrf_bias_bch_d02_DJF,wrf_bias_bch_DJF,raw_bias_bch_DJF,rcm_bias_bch_DJF=[],[],[],[]
    
elif variable=="pr":
    wrf_bias_eccc_d02_MAM = 100*(np.mean(wrf_d02_eccc_MAM) - np.mean(eccc_MAM))/np.mean(eccc_MAM)
    wrf_bias_noaa_d02_MAM = 100*(np.mean(wrf_d02_noaa_MAM) - np.mean(noaa_MAM))/np.mean(noaa_MAM)

    wrf_bias_eccc_MAM = 100*(np.mean(wrf_d03_eccc_MAM) - np.mean(eccc_MAM))/np.mean(eccc_MAM)
    wrf_bias_noaa_MAM = 100*(np.mean(wrf_d03_noaa_MAM) - np.mean(noaa_MAM))/np.mean(noaa_MAM)

    raw_bias_eccc_MAM = 100*(np.mean(raw_eccc_MAM) - np.mean(eccc_MAM))/np.mean(eccc_MAM)
    raw_bias_noaa_MAM = 100*(np.mean(raw_noaa_MAM) - np.mean(noaa_MAM))/np.mean(noaa_MAM)

    rcm_bias_eccc_MAM = 100*(np.mean(rcm_eccc_MAM) - np.mean(eccc_MAM))/np.mean(eccc_MAM)
    rcm_bias_noaa_MAM = 100*(np.mean(rcm_noaa_MAM) - np.mean(noaa_MAM))/np.mean(noaa_MAM)

    if variable != "wind":
        wrf_bias_bch_d02_MAM = 100*(np.mean(wrf_d02_bch_MAM) - np.mean(bch_MAM))/np.mean(bch_MAM)
        wrf_bias_bch_MAM = 100*(np.mean(wrf_d03_bch_MAM) - np.mean(bch_MAM))/np.mean(bch_MAM)
        raw_bias_bch_MAM = 100*(np.mean(raw_bch_MAM) - np.mean(bch_MAM))/np.mean(bch_MAM)
        rcm_bias_bch_MAM = 100*(np.mean(rcm_bch_MAM) - np.mean(bch_MAM))/np.mean(bch_MAM)
        pcic_bias_eccc_MAM = 100*(np.mean(pcic_eccc_MAM) - np.mean(eccc_MAM))/np.mean(eccc_MAM)
        pcic_bias_bch_MAM = 100*(np.mean(pcic_bch_MAM) - np.mean(bch_MAM))/np.mean(bch_MAM)

    else:
        wrf_bias_bch_d02_MAM,wrf_bias_bch_MAM,raw_bias_bch_MAM,rcm_bias_bch_MAM=[],[],[],[]


    wrf_bias_eccc_d02_JJA = 100*(np.mean(wrf_d02_eccc_JJA) - np.mean(eccc_JJA))/np.mean(eccc_JJA)
    wrf_bias_noaa_d02_JJA = 100*(np.mean(wrf_d02_noaa_JJA) - np.mean(noaa_JJA))/np.mean(noaa_JJA)

    wrf_bias_eccc_JJA = 100*(np.mean(wrf_d03_eccc_JJA) - np.mean(eccc_JJA))/np.mean(eccc_JJA)
    wrf_bias_noaa_JJA = 100*(np.mean(wrf_d03_noaa_JJA) - np.mean(noaa_JJA))/np.mean(noaa_JJA)

    raw_bias_eccc_JJA = 100*(np.mean(raw_eccc_JJA) - np.mean(eccc_JJA))/np.mean(eccc_JJA)
    raw_bias_noaa_JJA = 100*(np.mean(raw_noaa_JJA) - np.mean(noaa_JJA))/np.mean(noaa_JJA)

    rcm_bias_eccc_JJA = 100*(np.mean(rcm_eccc_JJA) - np.mean(eccc_JJA))/np.mean(eccc_JJA)
    rcm_bias_noaa_JJA = 100*(np.mean(rcm_noaa_JJA) - np.mean(noaa_JJA))/np.mean(noaa_JJA)

    if variable != "wind":
        wrf_bias_bch_d02_JJA = 100*(np.mean(wrf_d02_bch_JJA) - np.mean(bch_JJA))/np.mean(bch_JJA)
        wrf_bias_bch_JJA = 100*(np.mean(wrf_d03_bch_JJA) - np.mean(bch_JJA))/np.mean(bch_JJA)
        raw_bias_bch_JJA = 100*(np.mean(raw_bch_JJA) - np.mean(bch_JJA))/np.mean(bch_JJA)
        rcm_bias_bch_JJA = 100*(np.mean(rcm_bch_JJA) - np.mean(bch_JJA))/np.mean(bch_JJA)
        pcic_bias_eccc_JJA = 100*(np.mean(pcic_eccc_JJA) - np.mean(eccc_JJA))/np.mean(eccc_JJA)
        pcic_bias_bch_JJA = 100*(np.mean(pcic_bch_JJA) - np.mean(bch_JJA))/np.mean(bch_JJA)

    else:
        wrf_bias_bch_d02_JJA,wrf_bias_bch_JJA,raw_bias_bch_JJA,rcm_bias_bch_JJA=[],[],[],[]
        


    wrf_bias_eccc_d02_SON = 100*(np.mean(wrf_d02_eccc_SON) - np.mean(eccc_SON))/np.mean(eccc_SON)
    wrf_bias_noaa_d02_SON = 100*(np.mean(wrf_d02_noaa_SON) - np.mean(noaa_SON))/np.mean(noaa_SON)

    wrf_bias_eccc_SON = 100*(np.mean(wrf_d03_eccc_SON) - np.mean(eccc_SON))/np.mean(eccc_SON)
    wrf_bias_noaa_SON = 100*(np.mean(wrf_d03_noaa_SON) - np.mean(noaa_SON))/np.mean(noaa_SON)

    raw_bias_eccc_SON = 100*(np.mean(raw_eccc_SON) - np.mean(eccc_SON))/np.mean(eccc_SON)
    raw_bias_noaa_SON = 100*(np.mean(raw_noaa_SON) - np.mean(noaa_SON))/np.mean(noaa_SON)

    rcm_bias_eccc_SON = 100*(np.mean(rcm_eccc_SON) - np.mean(eccc_SON))/np.mean(eccc_SON)
    rcm_bias_noaa_SON = 100*(np.mean(rcm_noaa_SON) - np.mean(noaa_SON))/np.mean(noaa_SON)

    if variable != "wind":
        wrf_bias_bch_d02_SON = 100*(np.mean(wrf_d02_bch_SON) - np.mean(bch_SON))/np.mean(bch_SON)
        wrf_bias_bch_SON = 100*(np.mean(wrf_d03_bch_SON) - np.mean(bch_SON))/np.mean(bch_SON)
        raw_bias_bch_SON = 100*(np.mean(raw_bch_SON) - np.mean(bch_SON))/np.mean(bch_SON)
        rcm_bias_bch_SON = 100*(np.mean(rcm_bch_SON) - np.mean(bch_SON))/np.mean(bch_SON)
        pcic_bias_eccc_SON = 100*(np.mean(pcic_eccc_SON) - np.mean(eccc_SON))/np.mean(eccc_SON)
        pcic_bias_bch_SON = 100*(np.mean(pcic_bch_SON) - np.mean(bch_SON))/np.mean(bch_SON)

    else:
        wrf_bias_bch_d02_SON,wrf_bias_bch_SON,raw_bias_bch_SON,rcm_bias_bch_SON=[],[],[],[]



    wrf_bias_eccc_d02_DJF = 100*(np.mean(wrf_d02_eccc_DJF) - np.mean(eccc_DJF))/np.mean(eccc_DJF)
    wrf_bias_noaa_d02_DJF = 100*(np.mean(wrf_d02_noaa_DJF) - np.mean(noaa_DJF))/np.mean(noaa_DJF)

    wrf_bias_eccc_DJF = 100*(np.mean(wrf_d03_eccc_DJF) - np.mean(eccc_DJF))/np.mean(eccc_DJF)
    wrf_bias_noaa_DJF = 100*(np.mean(wrf_d03_noaa_DJF) - np.mean(noaa_DJF))/np.mean(noaa_DJF)

    raw_bias_eccc_DJF = 100*(np.mean(raw_eccc_DJF) - np.mean(eccc_DJF))/np.mean(eccc_DJF)
    raw_bias_noaa_DJF = 100*(np.mean(raw_noaa_DJF) - np.mean(noaa_DJF))/np.mean(noaa_DJF)

    rcm_bias_eccc_DJF = 100*(np.mean(rcm_eccc_DJF) - np.mean(eccc_DJF))/np.mean(eccc_DJF)
    rcm_bias_noaa_DJF = 100*(np.mean(rcm_noaa_DJF) - np.mean(noaa_DJF))/np.mean(noaa_DJF)

    if variable != "wind":
        wrf_bias_bch_d02_DJF = 100*(np.mean(wrf_d02_bch_DJF) - np.mean(bch_DJF))/np.mean(bch_DJF)
        wrf_bias_bch_DJF = 100*(np.mean(wrf_d03_bch_DJF) - np.mean(bch_DJF))/np.mean(bch_DJF)
        raw_bias_bch_DJF = 100*(np.mean(raw_bch_DJF) - np.mean(bch_DJF))/np.mean(bch_DJF)
        rcm_bias_bch_DJF = 100*(np.mean(rcm_bch_DJF) - np.mean(bch_DJF))/np.mean(bch_DJF)      
        pcic_bias_eccc_DJF = 100*(np.mean(pcic_eccc_DJF) - np.mean(eccc_DJF))/np.mean(eccc_DJF)
        pcic_bias_bch_DJF = 100*(np.mean(pcic_bch_DJF) - np.mean(bch_DJF))/np.mean(bch_DJF)

    else:
        wrf_bias_bch_d02_DJF,wrf_bias_bch_DJF,raw_bias_bch_DJF,rcm_bias_bch_DJF=[],[],[],[]

#%%
#temp
def print_absbias(eccc_bias,bch_bias,noaa_bias,noaa_buoy_bias,eccc_buoy_bias,title):

    df_allstations = pd.concat([abs(eccc_bias),abs(bch_bias),abs(noaa_bias),abs(noaa_buoy_bias),abs(eccc_buoy_bias)])
    
    print(df_allstations.max())
    
    avg_allstations = round(df_allstations.mean(),1)
    
    print(title)
    print(avg_allstations)
    
print_absbias(wrf_bias_eccc,wrf_bias_bch,wrf_bias_noaa,wrf_bias_noaa_buoy,wrf_bias_eccc_buoy,'CanESM2-WRF D03')
print_absbias(wrf_bias_eccc_d02,wrf_bias_bch_d02,wrf_bias_noaa_d02,wrf_bias_noaa_buoy_d02,wrf_bias_eccc_buoy_d02,'CanESM2-WRF D02')
print_absbias(raw_bias_eccc,raw_bias_bch,raw_bias_noaa,raw_bias_noaa_buoy,raw_bias_eccc_buoy,'CanESM2')
print_absbias(rcm_bias_eccc,rcm_bias_bch,rcm_bias_noaa,rcm_bias_noaa_buoy,rcm_bias_eccc_buoy,'CanRCM4')
print_absbias(wrf_bias_eccc_d01,wrf_bias_bch_d01,wrf_bias_noaa_d01,wrf_bias_noaa_buoy_d01,wrf_bias_eccc_buoy_d01,'CanESM2-WRF D01')

#%% pr
def print_absbias(eccc_bias,bch_bias,noaa_bias,title):

    df_allstations = pd.concat([abs(eccc_bias),abs(bch_bias),abs(noaa_bias)])
    
    
    avg_allstations = round((df_allstations.mean())/365.25,2)
    
    print(title)
    print(avg_allstations)
    
print_absbias(wrf_bias_eccc,wrf_bias_bch,wrf_bias_noaa,'CanESM2-WRF D03')
print_absbias(wrf_bias_eccc_d02,wrf_bias_bch_d02,wrf_bias_noaa_d02,'CanESM2-WRF D02')
print_absbias(raw_bias_eccc,raw_bias_bch,raw_bias_noaa,'CanESM2')
print_absbias(rcm_bias_eccc,rcm_bias_bch,rcm_bias_noaa,'CanRCM4')
print_absbias(wrf_bias_eccc_d01,wrf_bias_bch_d01,wrf_bias_noaa_d01,'CanESM2-WRF D01')

#%% wind
def print_absbias(eccc_bias,noaa_bias,noaa_buoy_bias,eccc_buoy_bias,title):

    df_allstations = pd.concat([abs(eccc_bias),abs(noaa_bias),abs(noaa_buoy_bias),abs(eccc_buoy_bias)])
        
    avg_allstations = round(df_allstations.mean(),1)
    
    print(title)
    print(avg_allstations)
    
print_absbias(wrf_bias_eccc,wrf_bias_noaa,wrf_bias_noaa_buoy,wrf_bias_eccc_buoy,'CanESM2-WRF D03')
print_absbias(wrf_bias_eccc_d02,wrf_bias_noaa_d02,wrf_bias_noaa_buoy_d02,wrf_bias_eccc_buoy_d02,'CanESM2-WRF D02')
print_absbias(raw_bias_eccc,raw_bias_noaa,raw_bias_noaa_buoy,raw_bias_eccc_buoy,'CanESM2')
print_absbias(rcm_bias_eccc,rcm_bias_noaa,rcm_bias_noaa_buoy,rcm_bias_eccc_buoy,'CanRCM4')
print_absbias(wrf_bias_eccc_d01,wrf_bias_noaa_d01,wrf_bias_noaa_buoy_d01,wrf_bias_eccc_buoy_d01,'CanESM2-WRF D01')


#%%

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
lat_d03 = np.squeeze(geo_em_d03_nc.variables['XLAT_C'][:])
lon_d03 = np.squeeze(geo_em_d03_nc.variables['XLONG_C'][:])
topo_d03 = np.squeeze(geo_em_d03_nc.variables['HGT_M'][:])

geo_em_d02_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d02.nc'
geo_em_d02_nc = Dataset(geo_em_d02_file, mode='r')
lat_d02 = np.squeeze(geo_em_d02_nc.variables['XLAT_C'][:])
lon_d02 = np.squeeze(geo_em_d02_nc.variables['XLONG_C'][:])
topo_d02 = np.squeeze(geo_em_d02_nc.variables['HGT_M'][:])

geo_em_d01_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d01.nc'
geo_em_d01_nc = Dataset(geo_em_d01_file, mode='r')
lat_d01 = np.squeeze(geo_em_d02_nc.variables['XLAT_C'][:])
lon_d01 = np.squeeze(geo_em_d02_nc.variables['XLONG_C'][:])
topo_d01 = np.squeeze(geo_em_d02_nc.variables['HGT_M'][:])

#%%


def plot_bias(eccc_bias,bch_bias,noaa_bias,noaa_buoy_bias,eccc_buoy_bias, fig_name,title,seas=None):
    #fig,ax = plot_all_d03_climo(title,variable)
    fig,ax = plot_all_d03(title)

    if variable == "t":
        divide = 1
        
        colors_tas_delta = ['#142f60','#3465aa','#5392c1','#99c4dd','#d3e5f0','#f7f7f7',
                            '#fadcc8','#eea785','#ce6451','#ab242f','#630921']
        lim = [-5,5] 
        cmap,_ = make_colorbar(colors_tas_delta,lim)

        #cmap = 'bwr'
        
        if seas==None:
            label = 'Annual Temperature Bias (\N{degree sign}C)'
        else:
            label = seas + ' Temperature Bias (\N{degree sign}C)'
        vmax = 5
    elif variable == "pr":
        
# =============================================================================
#         #if you want mm/day and not percentage (fix code above to represent that)
#         if seas == None:
#             divide = 365.25
#             label = "Daily Precipitation Bias (mm/day)"
#         elif seas=="MAM" or seas=="JJA":
#             divide = 92/3
#         elif seas=="SON":
#             divide = 91/3
#         elif seas=="DJF":
#             divide = 90.25/3
#         
#         if seas!=None:
#             label = "Daily " + seas + " Precipitation Bias (mm/day)"
#         cmap = 'bwr_r'
#         vmax = 5
# =============================================================================

        divide=1
        label = "Annual Precipitation Bias (%)"
        if seas==None:
            label = "Annual Precipitation Bias (%)"
        else:
            label = seas + " Precipitation Bias (%)"
            
        colors_pr_delta = ['#386158','#41847e','#67aba5','#a0d6cd','#d5edea','#f7f7f7','#f3e7c4',
                           '#dbc37f','#b88234','#865214','#503009'][::-1]

        lim = [-100,100] 
        cmap,_ = make_colorbar(colors_pr_delta,lim)
            
        #cmap = 'bwr_r'
        vmax = 100
        
    elif variable == "wind":
        divide = 1
        colors_wspd_delta = ['#424c03','#41641a','#4b8c49','#79b17d','#aed0b2','#d7e3e0',
                             '#aec3d5','#7394b5','#3e6896','#294072','#2c194d'][::-1]

        #lim = [-5,5] 
        lim = [-100,100]
        cmap,_ = make_colorbar(colors_wspd_delta,lim)
        
        if seas == None:
            label = "Daily Wind Speed Bias (m/s)"
        else:
            label = "Daily " + seas + " Wind Speed Bias (m/s)"
        vmax = 100
        
        
    plt.scatter(eccc_lons, eccc_lats, c=eccc_bias/divide,s=250,cmap=cmap,vmin=-vmax,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3,marker='o')
    if "PCIC" not in title:
        plt.scatter(noaa_lons, noaa_lats, c=noaa_bias/divide,s=250,cmap=cmap,vmin=-vmax,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3,marker='o')

    if variable != "wind":
        plt.scatter(bch_lons, bch_lats, c=bch_bias/divide,s=250,cmap=cmap,vmin=-vmax,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3,marker='o')
    
    if variable != "pr" and "PCIC" not in title:
        plt.scatter(noaa_buoy_lons, noaa_buoy_lats, c=noaa_buoy_bias/divide,s=250,cmap=cmap,vmin=-vmax,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3,marker='o')
        plt.scatter(eccc_buoy_lons, eccc_buoy_lats, c=eccc_buoy_bias/divide,s=250,cmap=cmap,vmin=-vmax,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3,marker='o')

    if variable == "wind": #no bch
        df_allstations = pd.concat([abs(eccc_bias),abs(noaa_bias),abs(noaa_buoy_bias),abs(eccc_buoy_bias)])
    if "PCIC" in title:
         df_allstations = pd.concat([abs(eccc_bias),abs(bch_bias)])
    
    else:
        if variable =="pr":
            df_allstations = pd.concat([abs(eccc_bias),abs(bch_bias),abs(noaa_bias)])
        elif variable == "wind":
            df_allstations = pd.concat([abs(eccc_bias),abs(noaa_bias),abs(noaa_buoy_bias),abs(eccc_buoy_bias)])

        else:
            df_allstations = pd.concat([abs(eccc_bias),abs(bch_bias),abs(noaa_bias),abs(noaa_buoy_bias),abs(eccc_buoy_bias)])


    avg_allstations = round(df_allstations.mean()/divide,2)
    
    #plt.title(title,fontsize=20)# + ', mean absolute bias: ' + str(avg_allstations),fontsize=20)


   # cbar_ax = fig.add_axes([0.15, 0.08, 0.73, 0.03])
   # fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)),cax=cbar_ax, orientation='horizontal',extend='both')
   # cbar_ax.tick_params(labelsize=24)
   # cbar_ax.set_xlabel(label,size=24) 


    
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/bias/' + fig_name + '_' + variable + '_bias_w_ocean.png',bbox_inches='tight')



#% annual




plot_bias(wrf_bias_eccc,wrf_bias_bch,wrf_bias_noaa,wrf_bias_noaa_buoy,wrf_bias_eccc_buoy,'canesm2_wrf_d03_annual', 'CanESM2-WRF D03')
plot_bias(wrf_bias_eccc_d02,wrf_bias_bch_d02,wrf_bias_noaa_d02,wrf_bias_noaa_buoy_d02,wrf_bias_eccc_buoy_d02,'canesm2_wrf_d02_annual', 'CanESM2-WRF D02')
plot_bias(raw_bias_eccc,raw_bias_bch,raw_bias_noaa,raw_bias_noaa_buoy,raw_bias_eccc_buoy,'canesm2_raw_annual', 'CanESM2')
plot_bias(rcm_bias_eccc,rcm_bias_bch,rcm_bias_noaa,rcm_bias_noaa_buoy,rcm_bias_eccc_buoy,'canrcm4_annual', 'CanRCM4')
plot_bias(wrf_bias_eccc_d01,wrf_bias_bch_d01,wrf_bias_noaa_d01,wrf_bias_noaa_buoy_d01,wrf_bias_eccc_buoy_d01,'canesm2_wrf_d01_annual', 'CanESM2-WRF D01')


# =============================================================================
# if variable != "wind":
#     plot_bias(pcic_bias_eccc,pcic_bias_bch,[],[],[],'pcic_annual', 'PCIC (CanESM2)')
# 
# =============================================================================

#%%
#df_d03 = pd.concat([np.nanmean(wrf_d03_eccc),np.nanmean(wrf_d03_noaa)])
#obs = pd.concat([np.nanmean(eccc_obs),np.nanmean(noaa_obs)])

#MSE = np.square(np.subtract(obs,df_d03)).mean() 
#RMSE = math.sqrt(MSE)
    
    
#%%
plot_bias(wrf_bias_eccc_MAM,wrf_bias_bch_MAM,wrf_bias_noaa_MAM,'canesm2_wrf_d03_MAM', 'CanESM2-WRF D03','MAM')
plot_bias(wrf_bias_eccc_d02_MAM,wrf_bias_bch_d02_MAM,wrf_bias_noaa_d02_MAM,'canesm2_wrf_d02_MAM', 'CanESM2-WRF D02','MAM')
plot_bias(raw_bias_eccc_MAM,raw_bias_bch_MAM,raw_bias_noaa_MAM,'canesm2_raw_MAM', 'CanESM2','MAM')
plot_bias(rcm_bias_eccc_MAM,rcm_bias_bch_MAM,rcm_bias_noaa_MAM,'canrcm4_MAM', 'CanRCM4','MAM')
if variable != "wind":
    plot_bias(pcic_bias_eccc_MAM,pcic_bias_bch_MAM,[],'pcic_MAM', 'PCIC (CanESM2)','MAM')

plot_bias(wrf_bias_eccc_JJA,wrf_bias_bch_JJA,wrf_bias_noaa_JJA,'canesm2_wrf_d03_JJA', 'CanESM2-WRF D03','JJA')
plot_bias(wrf_bias_eccc_d02_JJA,wrf_bias_bch_d02_JJA,wrf_bias_noaa_d02_JJA,'canesm2_wrf_d02_JJA', 'CanESM2-WRF D02','JJA')
plot_bias(raw_bias_eccc_JJA,raw_bias_bch_JJA,raw_bias_noaa_JJA,'canesm2_raw_JJA', 'CanESM2','JJA')
plot_bias(rcm_bias_eccc_JJA,rcm_bias_bch_JJA,rcm_bias_noaa_JJA,'canrcm4_JJA', 'CanRCM4','JJA')
if variable != "wind":
    plot_bias(pcic_bias_eccc_JJA,pcic_bias_bch_JJA,[],'pcic_JJA', 'PCIC (CanESM2)','JJA')

plot_bias(wrf_bias_eccc_SON,wrf_bias_bch_SON,wrf_bias_noaa_SON,'canesm2_wrf_d03_SON', 'CanESM2-WRF D03','SON')
plot_bias(wrf_bias_eccc_d02_SON,wrf_bias_bch_d02_SON,wrf_bias_noaa_d02_SON,'canesm2_wrf_d02_SON', 'CanESM2-WRF D02','SON')
plot_bias(raw_bias_eccc_SON,raw_bias_bch_SON,raw_bias_noaa_SON,'canesm2_raw_SON', 'CanESM2','SON')
plot_bias(rcm_bias_eccc_SON,rcm_bias_bch_SON,rcm_bias_noaa_SON,'canrcm4_SON', 'CanRCM4','SON')
if variable != "wind":
    plot_bias(pcic_bias_eccc_SON,pcic_bias_bch_SON,[],'pcic_SON', 'PCIC (CanESM2)','SON')

plot_bias(wrf_bias_eccc_DJF,wrf_bias_bch_DJF,wrf_bias_noaa_DJF,'canesm2_wrf_d03_DJF', 'CanESM2-WRF D03','DJF')
plot_bias(wrf_bias_eccc_d02_DJF,wrf_bias_bch_d02_DJF,wrf_bias_noaa_d02_DJF,'canesm2_wrf_d02_DJF', 'CanESM2-WRF D02','DJF')
plot_bias(raw_bias_eccc_DJF,raw_bias_bch_DJF,raw_bias_noaa_DJF,'canesm2_raw_DJF', 'CanESM2','DJF')
plot_bias(rcm_bias_eccc_DJF,rcm_bias_bch_DJF,rcm_bias_noaa_DJF,'canrcm4_DJF', 'CanRCM4','DJF')
if variable != "wind":
    plot_bias(pcic_bias_eccc_DJF,pcic_bias_bch_DJF,[],'pcic_DJF', 'PCIC (CanESM2)','DJF')