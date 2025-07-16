import pandas as pd
import warnings
import sys
import numpy as np
from netCDF4 import Dataset
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2-WRF-scripts/functions/')
from canesm2_eval_funcs import *
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib
from scipy.stats import linregress
import math
import matplotlib as mpl

variable = 'wind' #t or pr
run = 'historical' #historical rcp45 or rcp85
output_freq = "monthly" #yearly monthly or daily
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

eccc_elev = (df.iloc[:,11])
eccc_elev.index = eccc_station_IDs

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])

bch_elev = (df["ELEV"])
bch_elev.index = bch_station_IDs

df = pd.read_csv(noaa_daily_stations)
noaa_station_IDs = list(df["STATION"])
noaa_station_names = list(df["NAME"])

noaa_elev = (df["ELEVATION"])
noaa_elev.index = noaa_station_IDs

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

#%%
if variable != "wind":
    wrf_d01_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d01", run, variable, WRF_files_dir,start_year)
    wrf_d02_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
    wrf_d03_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
    raw_bch = get_canesm2(output_freq, "BCH", bch_station_IDs, run, variable, raw_files_dir,start_year)
    rcm_bch = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, variable, rcm_files_dir)
    #pcic_bch = get_pcic(output_freq, "BCH", bch_station_IDs, run, variable, pcic_files_dir)
    
    #pcic_eccc = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, variable, pcic_files_dir)

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
# # remove stations not in the original list
# for station in noaa_station_IDs:
#     if station not in list(noaa_obs.columns):
#         wrf_d02_noaa.drop(station, inplace=True, axis=1)
#         wrf_d03_noaa.drop(station, inplace=True, axis=1)
#         raw_noaa.drop(station, inplace=True, axis=1)
#         rcm_noaa.drop(station, inplace=True, axis=1)
#         noaa_elev.drop(station,inplace=True)
# 
# =============================================================================
        
#%%
raw_eccc.index = raw_eccc.index.to_timestamp()
rcm_eccc.index = rcm_eccc.index.to_timestamp()
raw_eccc = raw_eccc.reindex(eccc_obs.index)
rcm_eccc = rcm_eccc.reindex(eccc_obs.index)

raw_noaa.index = raw_noaa.index.to_timestamp()
rcm_noaa.index = rcm_noaa.index.to_timestamp()
raw_noaa = raw_noaa.reindex(noaa_obs.index)
rcm_noaa = rcm_noaa.reindex(noaa_obs.index)



eccc_buoy_obs = eccc_buoy_obs.reindex(wrf_d03_eccc_buoy.index)


#%%
#def get_var_ordered(szn,bch,eccc,noaa):
#def get_var_ordered(szn,bch,eccc,noaa,eccc_buoy,noaa_buoy):
def get_var_ordered(szn,eccc,noaa,eccc_buoy,noaa_buoy):

    #bch_var_szn = bch.copy()
    eccc_var_szn = eccc.copy()
    noaa_var_szn = noaa.copy()
    eccc_buoy_var_szn = eccc_buoy.copy()
    noaa_buoy_var_szn = noaa_buoy.copy()
    
    if szn == "mam":
        for i in [1,2,6,7,8,9,10,11,12]:
            #bch_var_szn = bch_var_szn[bch_var_szn.index.month != i]
            eccc_var_szn = eccc_var_szn[eccc_var_szn.index.month != i]
            noaa_var_szn = noaa_var_szn[noaa_var_szn.index.month != i]
            eccc_buoy_var_szn = eccc_buoy_var_szn[eccc_buoy_var_szn.index.month != i]
            noaa_buoy_var_szn = noaa_buoy_var_szn[noaa_buoy_var_szn.index.month != i]

    elif szn == "jja":
        for i in [1,2,3,4,5,9,10,11,12]:
            #bch_var_szn = bch_var_szn[bch_var_szn.index.month != i]
            eccc_var_szn = eccc_var_szn[eccc_var_szn.index.month != i]
            noaa_var_szn = noaa_var_szn[noaa_var_szn.index.month != i]
            eccc_buoy_var_szn = eccc_buoy_var_szn[eccc_buoy_var_szn.index.month != i]
            noaa_buoy_var_szn = noaa_buoy_var_szn[noaa_buoy_var_szn.index.month != i]

    elif szn == "son":
        for i in [1,2,3,4,5,6,7,8,12]:
            #bch_var_szn = bch_var_szn[bch_var_szn.index.month != i]
            eccc_var_szn = eccc_var_szn[eccc_var_szn.index.month != i]
            noaa_var_szn = noaa_var_szn[noaa_var_szn.index.month != i]
            eccc_buoy_var_szn = eccc_buoy_var_szn[eccc_buoy_var_szn.index.month != i]
            noaa_buoy_var_szn = noaa_buoy_var_szn[noaa_buoy_var_szn.index.month != i]

    elif szn == "djf":
        for i in [3,4,5,6,7,8,9,10,11]:
            #bch_var_szn = bch_var_szn[bch_var_szn.index.month != i]
            eccc_var_szn = eccc_var_szn[eccc_var_szn.index.month != i]
            noaa_var_szn = noaa_var_szn[noaa_var_szn.index.month != i]
            eccc_buoy_var_szn = eccc_buoy_var_szn[eccc_buoy_var_szn.index.month != i]
            noaa_buoy_var_szn = noaa_buoy_var_szn[noaa_buoy_var_szn.index.month != i]

# =============================================================================
#     elif szn == "wet":
#         for i in [4,5,6,7,8,9]: 
#             #bch_var_szn = bch_var_szn[bch_var_szn.index.month != i]
#             eccc_var_szn = eccc_var_szn[eccc_var_szn.index.month != i]
#             noaa_var_szn = noaa_var_szn[noaa_var_szn.index.month != i]
# 
#     elif szn == "dry":
#         for i in [1,2,3,10,11,12]:
#             #bch_var_szn = bch_var_szn[bch_var_szn.index.month != i]
#             eccc_var_szn = eccc_var_szn[eccc_var_szn.index.month != i]
#             noaa_var_szn = noaa_var_szn[noaa_var_szn.index.month != i]
# =============================================================================

    if variable == "pr":
        bch_var_szn_avg = bch_var_szn.groupby(bch_var_szn.index.year).sum().mean().sort_index()
        eccc_var_szn_avg = eccc_var_szn.groupby(eccc_var_szn.index.year).sum().mean().sort_index()
        noaa_var_szn_avg = noaa_var_szn.groupby(noaa_var_szn.index.year).sum().mean().sort_index()
        
        wrf_var_szn_avg = pd.concat([eccc_var_szn_avg,bch_var_szn_avg,noaa_var_szn_avg])
        
    elif variable == "t" or variable == "wind":
        #bch_var_szn_avg = bch_var_szn.mean().sort_index()
        eccc_var_szn_avg = eccc_var_szn.mean().sort_index()
        noaa_var_szn_avg = noaa_var_szn.mean().sort_index()
        eccc_buoy_var_szn_avg = eccc_buoy_var_szn.mean().sort_index()
        noaa_buoy_var_szn_avg = noaa_buoy_var_szn.mean().sort_index()

        #wrf_var_szn_avg = pd.concat([bch_var_szn_avg,eccc_var_szn_avg,noaa_var_szn_avg,eccc_buoy_var_szn_avg,noaa_buoy_var_szn_avg])  
        wrf_var_szn_avg = pd.concat([eccc_var_szn_avg,noaa_var_szn_avg,eccc_buoy_var_szn_avg,noaa_buoy_var_szn_avg])  
        
    wrf_var_szn_avg.index = wrf_var_szn_avg.index.astype(str)
    wrf_var_szn_avg = wrf_var_szn_avg.sort_index()
    
    return(wrf_var_szn_avg)
    
#%%
# =============================================================================
# obs_mam  = get_var_ordered("mam",bch_obs,eccc_obs,noaa_obs)
# wrf_d03_mam  = get_var_ordered("mam",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
# wrf_d02_mam  = get_var_ordered("mam",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
# raw_mam  = get_var_ordered("mam",raw_bch,raw_eccc,raw_noaa)
# rcm_mam  = get_var_ordered("mam",rcm_bch,rcm_eccc,rcm_noaa)
# 
# obs_jja  = get_var_ordered("jja",bch_obs,eccc_obs,noaa_obs)
# wrf_d03_jja  = get_var_ordered("jja",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
# wrf_d02_jja  = get_var_ordered("jja",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
# raw_jja  = get_var_ordered("jja",raw_bch,raw_eccc,raw_noaa)
# rcm_jja  = get_var_ordered("jja",rcm_bch,rcm_eccc,rcm_noaa)
# 
# obs_son  = get_var_ordered("son",bch_obs,eccc_obs,noaa_obs)
# wrf_d03_son  = get_var_ordered("son",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
# wrf_d02_son  = get_var_ordered("son",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
# raw_son  = get_var_ordered("son",raw_bch,raw_eccc,raw_noaa)
# rcm_son  = get_var_ordered("son",rcm_bch,rcm_eccc,rcm_noaa)
# 
# obs_djf  = get_var_ordered("djf",bch_obs,eccc_obs,noaa_obs)
# wrf_d03_djf  = get_var_ordered("djf",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
# wrf_d02_djf  = get_var_ordered("djf",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
# raw_djf  = get_var_ordered("djf",raw_bch,raw_eccc,raw_noaa)
# rcm_djf  = get_var_ordered("djf",rcm_bch,rcm_eccc,rcm_noaa)
# 
# obs_wet  = get_var_ordered("wet",bch_obs,eccc_obs,noaa_obs)
# wrf_d03_wet  = get_var_ordered("wet",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
# wrf_d02_wet  = get_var_ordered("wet",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
# raw_wet  = get_var_ordered("wet",raw_bch,raw_eccc,raw_noaa)
# rcm_wet  = get_var_ordered("wet",rcm_bch,rcm_eccc,rcm_noaa)
# 
# obs_dry  = get_var_ordered("dry",bch_obs,eccc_obs,noaa_obs)
# wrf_d03_dry  = get_var_ordered("dry",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
# wrf_d02_dry  = get_var_ordered("dry",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
# raw_dry  = get_var_ordered("dry",raw_bch,raw_eccc,raw_noaa)
# rcm_dry  = get_var_ordered("dry",rcm_bch,rcm_eccc,rcm_noaa)
# 
# elev = pd.concat([eccc_elev,bch_elev,noaa_elev])
# elev.index = elev.index.astype(str)
# elev = elev.sort_index()
# =============================================================================

if variable=="t":
    obs_mam  = get_var_ordered("mam",bch_obs,eccc_obs,noaa_obs,eccc_buoy_obs,noaa_buoy_obs)
    wrf_d03_mam  = get_var_ordered("mam",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa,wrf_d03_eccc_buoy,wrf_d03_noaa_buoy)
    wrf_d02_mam  = get_var_ordered("mam",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa,wrf_d02_eccc_buoy,wrf_d02_noaa_buoy)
    wrf_d01_mam  = get_var_ordered("mam",wrf_d01_bch,wrf_d01_eccc,wrf_d01_noaa,wrf_d01_eccc_buoy,wrf_d01_noaa_buoy)
    raw_mam  = get_var_ordered("mam",raw_bch,raw_eccc,raw_noaa,raw_eccc_buoy,raw_noaa_buoy)
    rcm_mam  = get_var_ordered("mam",rcm_bch,rcm_eccc,rcm_noaa,rcm_eccc_buoy,rcm_noaa_buoy)
    
    
    obs_jja  = get_var_ordered("jja",bch_obs,eccc_obs,noaa_obs,eccc_buoy_obs,noaa_buoy_obs)
    wrf_d03_jja  = get_var_ordered("jja",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa,wrf_d03_eccc_buoy,wrf_d03_noaa_buoy)
    wrf_d02_jja  = get_var_ordered("jja",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa,wrf_d02_eccc_buoy,wrf_d02_noaa_buoy)
    wrf_d01_jja  = get_var_ordered("jja",wrf_d01_bch,wrf_d01_eccc,wrf_d01_noaa,wrf_d01_eccc_buoy,wrf_d01_noaa_buoy)
    raw_jja  = get_var_ordered("jja",raw_bch,raw_eccc,raw_noaa,raw_eccc_buoy,raw_noaa_buoy)
    rcm_jja  = get_var_ordered("jja",rcm_bch,rcm_eccc,rcm_noaa,rcm_eccc_buoy,rcm_noaa_buoy)
    
    
    obs_son  = get_var_ordered("son",bch_obs,eccc_obs,noaa_obs,eccc_buoy_obs,noaa_buoy_obs)
    wrf_d03_son  = get_var_ordered("son",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa,wrf_d03_eccc_buoy,wrf_d03_noaa_buoy)
    wrf_d02_son  = get_var_ordered("son",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa,wrf_d02_eccc_buoy,wrf_d02_noaa_buoy)
    wrf_d01_son  = get_var_ordered("son",wrf_d01_bch,wrf_d01_eccc,wrf_d01_noaa,wrf_d01_eccc_buoy,wrf_d01_noaa_buoy)
    raw_son  = get_var_ordered("son",raw_bch,raw_eccc,raw_noaa,raw_eccc_buoy,raw_noaa_buoy)
    rcm_son  = get_var_ordered("son",rcm_bch,rcm_eccc,rcm_noaa,rcm_eccc_buoy,rcm_noaa_buoy)
    
    
    obs_djf  = get_var_ordered("djf",bch_obs,eccc_obs,noaa_obs,eccc_buoy_obs,noaa_buoy_obs)
    wrf_d03_djf  = get_var_ordered("djf",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa,wrf_d03_eccc_buoy,wrf_d03_noaa_buoy)
    wrf_d02_djf  = get_var_ordered("djf",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa,wrf_d02_eccc_buoy,wrf_d02_noaa_buoy)
    wrf_d01_djf  = get_var_ordered("djf",wrf_d01_bch,wrf_d01_eccc,wrf_d01_noaa,wrf_d01_eccc_buoy,wrf_d01_noaa_buoy)
    raw_djf  = get_var_ordered("djf",raw_bch,raw_eccc,raw_noaa,raw_eccc_buoy,raw_noaa_buoy)
    rcm_djf  = get_var_ordered("djf",rcm_bch,rcm_eccc,rcm_noaa,rcm_eccc_buoy,rcm_noaa_buoy)

elif variable=="pr":
    obs_mam  = get_var_ordered("mam",bch_obs,eccc_obs,noaa_obs)
    wrf_d03_mam  = get_var_ordered("mam",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
    wrf_d02_mam  = get_var_ordered("mam",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
    wrf_d01_mam  = get_var_ordered("mam",wrf_d01_bch,wrf_d01_eccc,wrf_d01_noaa)
    raw_mam  = get_var_ordered("mam",raw_bch,raw_eccc,raw_noaa)
    rcm_mam  = get_var_ordered("mam",rcm_bch,rcm_eccc,rcm_noaa)
    
    
    obs_jja  = get_var_ordered("jja",bch_obs,eccc_obs,noaa_obs)
    wrf_d03_jja  = get_var_ordered("jja",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
    wrf_d02_jja  = get_var_ordered("jja",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
    wrf_d01_jja  = get_var_ordered("jja",wrf_d01_bch,wrf_d01_eccc,wrf_d01_noaa)
    raw_jja  = get_var_ordered("jja",raw_bch,raw_eccc,raw_noaa)
    rcm_jja  = get_var_ordered("jja",rcm_bch,rcm_eccc,rcm_noaa)
    
    
    obs_son  = get_var_ordered("son",bch_obs,eccc_obs,noaa_obs)
    wrf_d03_son  = get_var_ordered("son",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
    wrf_d02_son  = get_var_ordered("son",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
    wrf_d01_son  = get_var_ordered("son",wrf_d01_bch,wrf_d01_eccc,wrf_d01_noaa)
    raw_son  = get_var_ordered("son",raw_bch,raw_eccc,raw_noaa)
    rcm_son  = get_var_ordered("son",rcm_bch,rcm_eccc,rcm_noaa)
    
    
    obs_djf  = get_var_ordered("djf",bch_obs,eccc_obs,noaa_obs)
    wrf_d03_djf  = get_var_ordered("djf",wrf_d03_bch,wrf_d03_eccc,wrf_d03_noaa)
    wrf_d02_djf  = get_var_ordered("djf",wrf_d02_bch,wrf_d02_eccc,wrf_d02_noaa)
    wrf_d01_djf  = get_var_ordered("djf",wrf_d01_bch,wrf_d01_eccc,wrf_d01_noaa)
    raw_djf  = get_var_ordered("djf",raw_bch,raw_eccc,raw_noaa)
    rcm_djf  = get_var_ordered("djf",rcm_bch,rcm_eccc,rcm_noaa)

elif variable=="wind":
    obs_mam  = get_var_ordered("mam",eccc_obs,noaa_obs,eccc_buoy_obs,noaa_buoy_obs)
    wrf_d03_mam  = get_var_ordered("mam",wrf_d03_eccc,wrf_d03_noaa,wrf_d03_eccc_buoy,wrf_d03_noaa_buoy)
    wrf_d02_mam  = get_var_ordered("mam",wrf_d02_eccc,wrf_d02_noaa,wrf_d02_eccc_buoy,wrf_d02_noaa_buoy)
    wrf_d01_mam  = get_var_ordered("mam",wrf_d01_eccc,wrf_d01_noaa,wrf_d01_eccc_buoy,wrf_d01_noaa_buoy)
    raw_mam  = get_var_ordered("mam",raw_eccc,raw_noaa,raw_eccc_buoy,raw_noaa_buoy)
    rcm_mam  = get_var_ordered("mam",rcm_eccc,rcm_noaa,rcm_eccc_buoy,rcm_noaa_buoy)
    
    
    obs_jja  = get_var_ordered("jja",eccc_obs,noaa_obs,eccc_buoy_obs,noaa_buoy_obs)
    wrf_d03_jja  = get_var_ordered("jja",wrf_d03_eccc,wrf_d03_noaa,wrf_d03_eccc_buoy,wrf_d03_noaa_buoy)
    wrf_d02_jja  = get_var_ordered("jja",wrf_d02_eccc,wrf_d02_noaa,wrf_d02_eccc_buoy,wrf_d02_noaa_buoy)
    wrf_d01_jja  = get_var_ordered("jja",wrf_d01_eccc,wrf_d01_noaa,wrf_d01_eccc_buoy,wrf_d01_noaa_buoy)
    raw_jja  = get_var_ordered("jja",raw_eccc,raw_noaa,raw_eccc_buoy,raw_noaa_buoy)
    rcm_jja  = get_var_ordered("jja",rcm_eccc,rcm_noaa,rcm_eccc_buoy,rcm_noaa_buoy)
    
    
    obs_son  = get_var_ordered("son",eccc_obs,noaa_obs,eccc_buoy_obs,noaa_buoy_obs)
    wrf_d03_son  = get_var_ordered("son",wrf_d03_eccc,wrf_d03_noaa,wrf_d03_eccc_buoy,wrf_d03_noaa_buoy)
    wrf_d02_son  = get_var_ordered("son",wrf_d02_eccc,wrf_d02_noaa,wrf_d02_eccc_buoy,wrf_d02_noaa_buoy)
    wrf_d01_son  = get_var_ordered("son",wrf_d01_eccc,wrf_d01_noaa,wrf_d01_eccc_buoy,wrf_d01_noaa_buoy)
    raw_son  = get_var_ordered("son",raw_eccc,raw_noaa,raw_eccc_buoy,raw_noaa_buoy)
    rcm_son  = get_var_ordered("son",rcm_eccc,rcm_noaa,rcm_eccc_buoy,rcm_noaa_buoy)
    
    
    obs_djf  = get_var_ordered("djf",eccc_obs,noaa_obs,eccc_buoy_obs,noaa_buoy_obs)
    wrf_d03_djf  = get_var_ordered("djf",wrf_d03_eccc,wrf_d03_noaa,wrf_d03_eccc_buoy,wrf_d03_noaa_buoy)
    wrf_d02_djf  = get_var_ordered("djf",wrf_d02_eccc,wrf_d02_noaa,wrf_d02_eccc_buoy,wrf_d02_noaa_buoy)
    wrf_d01_djf  = get_var_ordered("djf",wrf_d01_eccc,wrf_d01_noaa,wrf_d01_eccc_buoy,wrf_d01_noaa_buoy)
    raw_djf  = get_var_ordered("djf",raw_eccc,raw_noaa,raw_eccc_buoy,raw_noaa_buoy)
    rcm_djf  = get_var_ordered("djf",rcm_eccc,rcm_noaa,rcm_eccc_buoy,rcm_noaa_buoy)
# =============================================================================
# obs_wet  = get_var_ordered("wet",eccc_obs,noaa_obs)
# wrf_d03_wet  = get_var_ordered("wet",wrf_d03_eccc,wrf_d03_noaa)
# wrf_d02_wet  = get_var_ordered("wet",wrf_d02_eccc,wrf_d02_noaa)
# raw_wet  = get_var_ordered("wet",raw_eccc,raw_noaa)
# rcm_wet  = get_var_ordered("wet",rcm_eccc,rcm_noaa)
# 
# obs_dry  = get_var_ordered("dry",eccc_obs,noaa_obs)
# wrf_d03_dry  = get_var_ordered("dry",wrf_d03_eccc,wrf_d03_noaa)
# wrf_d02_dry  = get_var_ordered("dry",wrf_d02_eccc,wrf_d02_noaa)
# raw_dry  = get_var_ordered("dry",raw_eccc,raw_noaa)
# rcm_dry  = get_var_ordered("dry",rcm_eccc,rcm_noaa)
# 
# elev = pd.concat([eccc_elev,noaa_elev])
# elev.index = elev.index.astype(str)
# elev = elev.sort_index()
# 
# =============================================================================
#%%

def plot_scatter(obs,model,vmin,vmax,color,title,unit,figname):
    plt.figure(figsize=(10, 10),dpi=200)
    
    plt.scatter(obs,model,color=color,s=150,marker='o')
    
    plt.xlim([vmin,vmax])
    plt.ylim([vmin,vmax])
    
    plt.xlabel('Observed ' + unit,fontsize=18)
    plt.ylabel('Simulated ' + unit,fontsize=18)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    #print(min(model_avg))
    #print(min(obs))
    
    #print(max(model_avg))
    #print(max(obs))
    
    slope, intercept, r_value, p_value, std_err = linregress(obs, model)
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope * x + intercept
    plt.plot(x, line_of_best_fit,'--', color='grey')
    plt.plot(x, x, color='k')
    
    MSE = np.square(np.subtract(obs,model)).mean() 
    RMSE = math.sqrt(MSE)

    if variable == "pr":
        r=0
    else:
        r=1
    plt.scatter(-999,-999,color='white',s=0,label='$R^2$ = ' + str(round(r_value**2,2)))
    plt.scatter(-999,-999,color='white',s=0,label='$RMSE$ = ' + str(round(RMSE,r)))

    #plt.title(title, fontsize=22)
    plt.legend(fontsize=20)

    
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/r2/' + figname + '.png',bbox_inches='tight')

if variable == "t":
    title = 'Mean MAM air temperature: '
    vmin = -9
    vmax = 12
    unit = '(deg C)'
    plot_scatter(obs_mam,wrf_d03_mam,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_mam_tas_r2')
    
    plot_scatter(obs_mam,wrf_d01_mam,vmin,vmax,'C0',title + 'CanESM2-WRF D01', unit,'canesm2_wrf_d01_mam_tas_r2')
    plot_scatter(obs_mam,wrf_d02_mam,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_mam_tas_r2')
    plot_scatter(obs_mam,raw_mam,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_mam_tas_r2')
    plot_scatter(obs_mam,rcm_mam,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_mam_tas_r2')
    
# =============================================================================
#     title = 'Mean JJA air temperature: '
#     vmin = 0
#     vmax = 25
#     unit = '(deg C)'
#     plot_scatter(obs_jja,wrf_d03_jja,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_jja_tas_r2')
#     plot_scatter(obs_jja,wrf_d02_jja,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_jja_tas_r2')
#     plot_scatter(obs_jja,raw_jja,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_jja_tas_r2')
#     plot_scatter(obs_jja,rcm_jja,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_jja_tas_r2')
#     
#     title = 'Mean SON air temperature: '
#     vmin = -7
#     vmax = 13
#     unit = '(deg C)'
#     plot_scatter(obs_son,wrf_d03_son,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_son_tas_r2')
#     plot_scatter(obs_son,wrf_d02_son,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_son_tas_r2')
#     plot_scatter(obs_son,raw_son,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_son_tas_r2')
#     plot_scatter(obs_son,rcm_son,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_son_tas_r2')
#     
#     title = 'Mean DJF air temperature: '
#     vmin = -14
#     vmax = 7
#     unit = '(deg C)'
#     plot_scatter(obs_djf,wrf_d03_djf,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_djf_tas_r2')
#     plot_scatter(obs_djf,wrf_d02_djf,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_djf_tas_r2')
#     plot_scatter(obs_djf,raw_djf,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_djf_tas_r2')
#     plot_scatter(obs_djf,rcm_djf,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_djf_tas_r2')
# =============================================================================




elif variable == "pr":
    title = 'Mean MAM total precipitation: '
    vmin = 0
    vmax = 1300
    unit = '(mm/season)'
    plot_scatter(obs_mam,wrf_d03_mam,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_mam_pr_r2')
    plot_scatter(obs_mam,wrf_d02_mam,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_mam_pr_r2')
    plot_scatter(obs_mam,raw_mam,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_mam_pr_r2')
    plot_scatter(obs_mam,rcm_mam,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_mam_pr_r2')
    
    title = 'Mean JJA total precipitation: '
    vmin = 0
    vmax = 500
    unit = '(mm/season)'
    plot_scatter(obs_jja,wrf_d03_jja,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_jja_pr_r2')
    plot_scatter(obs_jja,wrf_d02_jja,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_jja_pr_r2')
    plot_scatter(obs_jja,raw_jja,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_jja_pr_r2')
    plot_scatter(obs_jja,rcm_jja,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_jja_pr_r2')
    
    title = 'Mean SON total precipitation: '
    vmin = 0
    vmax = 2000
    unit = '(mm/season)'
    plot_scatter(obs_son,wrf_d03_son,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_son_pr_r2')
    plot_scatter(obs_son,wrf_d02_son,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_son_pr_r2')
    plot_scatter(obs_son,raw_son,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_son_pr_r2')
    plot_scatter(obs_son,rcm_son,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_son_pr_r2')
    
    title = 'Mean DJF total precipitation: '
    vmin = 0
    vmax = 2300
    unit = '(mm/season)'
    plot_scatter(obs_djf,wrf_d03_djf,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_djf_pr_r2')
    plot_scatter(obs_djf,wrf_d02_djf,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_djf_pr_r2')
    plot_scatter(obs_djf,raw_djf,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_djf_pr_r2')
    plot_scatter(obs_djf,rcm_djf,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_djf_pr_r2')
    
    title = 'Mean WET total precipitation: '
    vmin = 0
    vmax = 4500
    unit = '(mm/6mo)'
    plot_scatter(obs_wet,wrf_d03_wet,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_wet_pr_r2')
    plot_scatter(obs_wet,wrf_d02_wet,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_wet_pr_r2')
    plot_scatter(obs_wet,raw_wet,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_wet_pr_r2')
    plot_scatter(obs_wet,rcm_wet,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_wet_pr_r2')
    
    title = 'Mean DRY total precipitation: '
    vmin = 0
    vmax = 1500
    unit = '(mm/6mo)'
    plot_scatter(obs_dry,wrf_d03_dry,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_dry_pr_r2')
    plot_scatter(obs_dry,wrf_d02_dry,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_dry_pr_r2')
    plot_scatter(obs_dry,raw_dry,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_dry_pr_r2')
    plot_scatter(obs_dry,rcm_dry,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_dry_pr_r2')
    
#%%

def plot_elev(obs,model,vmin,vmax,title,unit,figname):
    fig = plt.figure(figsize=(10, 10),dpi=200)
    
    plt.scatter(obs,model,c=elev,cmap='terrain',s=150,marker='o',edgecolor='k',vmin=0,vmax=3000)
    
    plt.xlim([vmin,vmax])
    plt.ylim([vmin,vmax])
    
    plt.xlabel('Observed ' + unit,fontsize=18)
    plt.ylabel('Simulated ' + unit,fontsize=18)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    #print(min(model_avg))
    #print(min(obs))
    
    #print(max(model_avg))
    #print(max(obs))
    
    slope, intercept, r_value, p_value, std_err = linregress(obs, model)
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope * x + intercept
    plt.plot(x, line_of_best_fit,'--', color='grey')
    plt.plot(x, x, color='k')
    
    MSE = np.square(np.subtract(obs,model)).mean() 
    RMSE = math.sqrt(MSE)

    if variable == "pr":
        r=0
    else:
        r=1
    plt.scatter(-999,-999,color='white',s=0,label='$R^2$ = ' + str(round(r_value**2,2)))
    plt.scatter(-999,-999,color='white',s=0,label='$RMSE$ = ' + str(round(RMSE,r)))

    plt.title(title, fontsize=22)
    plt.legend(fontsize=20)

    cbar_ax = fig.add_axes([0.93, 0.12, 0.03, 0.76])
    fig.colorbar(mpl.cm.ScalarMappable(cmap='terrain', norm=mpl.colors.Normalize(vmin=0, vmax=3000)),
                  cax=cbar_ax, ticks=np.arange(0, 3000+1, 500), orientation='vertical')
    cbar_ax.tick_params(labelsize=16)
    cbar_ax.set_ylabel('Elevation [m]',size=18) 

    
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/r2/' + figname + '_terrain.png',bbox_inches='tight')

if variable == "t":
    title = 'Mean MAM air temperature: '
    vmin = -9
    vmax = 12
    unit = '(deg C)'
    plot_elev(obs_mam,wrf_d03_mam,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_mam_tas_r2')
    plot_elev(obs_mam,wrf_d02_mam,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_mam_tas_r2')
    plot_elev(obs_mam,raw_mam,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_mam_tas_r2')
    plot_elev(obs_mam,rcm_mam,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_mam_tas_r2')
    
    title = 'Mean JJA air temperature: '
    vmin = 0
    vmax = 25
    unit = '(deg C)'
    plot_elev(obs_jja,wrf_d03_jja,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_jja_tas_r2')
    plot_elev(obs_jja,wrf_d02_jja,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_jja_tas_r2')
    plot_elev(obs_jja,raw_jja,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_jja_tas_r2')
    plot_elev(obs_jja,rcm_jja,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_jja_tas_r2')
    
    title = 'Mean SON air temperature: '
    vmin = -7
    vmax = 13
    unit = '(deg C)'
    plot_elev(obs_son,wrf_d03_son,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_son_tas_r2')
    plot_elev(obs_son,wrf_d02_son,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_son_tas_r2')
    plot_elev(obs_son,raw_son,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_son_tas_r2')
    plot_elev(obs_son,rcm_son,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_son_tas_r2')
    
    title = 'Mean DJF air temperature: '
    vmin = -14
    vmax = 7
    unit = '(deg C)'
    plot_elev(obs_djf,wrf_d03_djf,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_djf_tas_r2')
    plot_elev(obs_djf,wrf_d02_djf,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_djf_tas_r2')
    plot_elev(obs_djf,raw_djf,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_djf_tas_r2')
    plot_elev(obs_djf,rcm_djf,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_djf_tas_r2')

elif variable == "pr":
    title = 'Mean MAM total precipitation: '
    vmin = 0
    vmax = 1300
    unit = '(mm/season)'
    plot_elev(obs_mam,wrf_d03_mam,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_mam_pr_r2')
    plot_elev(obs_mam,wrf_d02_mam,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_mam_pr_r2')
    plot_elev(obs_mam,raw_mam,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_mam_pr_r2')
    plot_elev(obs_mam,rcm_mam,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_mam_pr_r2')
    
    title = 'Mean JJA total precipitation: '
    vmin = 0
    vmax = 500
    unit = '(mm/season)'
    plot_elev(obs_jja,wrf_d03_jja,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_jja_pr_r2')
    plot_elev(obs_jja,wrf_d02_jja,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_jja_pr_r2')
    plot_elev(obs_jja,raw_jja,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_jja_pr_r2')
    plot_elev(obs_jja,rcm_jja,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_jja_pr_r2')
    
    title = 'Mean SON total precipitation: '
    vmin = 0
    vmax = 2000
    unit = '(mm/season)'
    plot_elev(obs_son,wrf_d03_son,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_son_pr_r2')
    plot_elev(obs_son,wrf_d02_son,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_son_pr_r2')
    plot_elev(obs_son,raw_son,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_son_pr_r2')
    plot_elev(obs_son,rcm_son,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_son_pr_r2')
    
    title = 'Mean DJF total precipitation: '
    vmin = 0
    vmax = 2300
    unit = '(mm/season)'
    plot_elev(obs_djf,wrf_d03_djf,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_djf_pr_r2')
    plot_elev(obs_djf,wrf_d02_djf,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_djf_pr_r2')
    plot_elev(obs_djf,raw_djf,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_djf_pr_r2')
    plot_elev(obs_djf,rcm_djf,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_djf_pr_r2')
    
    title = 'Mean WET total precipitation: '
    vmin = 0
    vmax = 4500
    unit = '(mm/6mo)'
    plot_elev(obs_wet,wrf_d03_wet,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_wet_pr_r2')
    plot_elev(obs_wet,wrf_d02_wet,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_wet_pr_r2')
    plot_elev(obs_wet,raw_wet,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_wet_pr_r2')
    plot_elev(obs_wet,rcm_wet,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_wet_pr_r2')
    
    title = 'Mean DRY total precipitation: '
    vmin = 0
    vmax = 1500
    unit = '(mm/6mo)'
    plot_elev(obs_dry,wrf_d03_dry,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_dry_pr_r2')
    plot_elev(obs_dry,wrf_d02_dry,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_dry_pr_r2')
    plot_elev(obs_dry,raw_dry,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_dry_pr_r2')
    plot_elev(obs_dry,rcm_dry,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_dry_pr_r2')
    
    
    
#%%

# =============================================================================
# obs_mam.index = [index + '_mam' for index in obs_mam.index]
# obs_jja.index = [index + '_jja' for index in obs_jja.index]
# obs_son.index = [index + '_son' for index in obs_son.index]
# obs_djf.index = [index + '_djf' for index in obs_djf.index]
# 
# obs = pd.concat([obs_mam,obs_jja,obs_son,obs_djf])
# obs.index = obs.index.astype(str)
# obs = obs.sort_index()
# =============================================================================
    
    #%%


def plot_all_szn(model_mam_input,model_jja_input,model_son_input,model_djf_input,color,model_name,figname,obs_mam_input,obs_jja_input,obs_son_input,obs_djf_input): 
    
    obs_mam = obs_mam_input.copy()
    obs_jja = obs_jja_input.copy()
    obs_son = obs_son_input.copy()
    obs_djf = obs_djf_input.copy()

    model_mam = model_mam_input.copy()
    model_jja = model_jja_input.copy()
    model_son = model_son_input.copy()
    model_djf = model_djf_input.copy()
    
    plt.figure(figsize=(10, 10),dpi=200)
    

    if variable == "t":
        vmin = -16
        vmax = 29
        unit = '(\N{degree sign}C)'
        
    elif variable == "pr":
        vmin = 0
        vmax = 2300
        unit = '(mm/season)'
        
    elif variable == "wind":
        vmin = 0
        vmax = 15
        unit = '(m/s)'
        
    plt.xlim([vmin,vmax])
    plt.ylim([vmin,vmax])
    
    plt.xlabel('Observed ' + unit,fontsize=35)
    plt.ylabel('Modelled ' + unit,fontsize=35)
    
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
    
    obs_ann = (obs_mam + obs_jja + obs_son + obs_djf) / 4
    model_ann = (model_mam + model_jja + model_son + model_djf) / 4

    obs_ann = obs_ann[model_ann.dropna().index]
    model_ann = model_ann[model_ann.dropna().index]
    
    slope_ann, intercept_ann, r_annual, p_value, std_err = linregress(obs_ann, model_ann)


    model_mam.index = [index + '_mam' for index in model_mam.index]
    model_jja.index = [index + '_jja' for index in model_jja.index]
    model_son.index = [index + '_son' for index in model_son.index]
    model_djf.index = [index + '_djf' for index in model_djf.index]


    obs_mam.index = [index + '_mam' for index in obs_mam.index]
    obs_jja.index = [index + '_jja' for index in obs_jja.index]
    obs_son.index = [index + '_son' for index in obs_son.index]
    obs_djf.index = [index + '_djf' for index in obs_djf.index]


 
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope_ann * x + intercept_ann
    plt.plot(x, line_of_best_fit,'--', color='grey')
    plt.plot(x, x, color='k')
    
    MSE = np.square(np.subtract(obs_ann,model_ann)).mean() 

    RMSE = math.sqrt(MSE)

    if variable == "pr":
        r=1
    else:
        r=1
    
    
    obs_mam = obs_mam[model_mam.dropna().index]
    model_mam = model_mam[model_mam.dropna().index]
    obs_jja = obs_jja[model_jja.dropna().index]
    model_jja = model_jja[model_jja.dropna().index]
    obs_son = obs_son[model_son.dropna().index]
    model_son = model_son[model_son.dropna().index]
    obs_djf = obs_djf[model_djf.dropna().index]
    model_djf = model_djf[model_djf.dropna().index]
    
    slope, intercept, r_mam, p_value, std_err = linregress(obs_mam, model_mam)

    slope, intercept, r_jja, p_value, std_err = linregress(obs_jja, model_jja)
    slope, intercept, r_son, p_value, std_err = linregress(obs_son, model_son)
    slope, intercept, r_djf, p_value, std_err = linregress(obs_djf, model_djf)

    RMSE_mam = math.sqrt(np.square(np.subtract(obs_mam,model_mam)).mean())
    RMSE_jja = math.sqrt(np.square(np.subtract(obs_jja,model_jja)).mean())
    RMSE_son = math.sqrt(np.square(np.subtract(obs_son,model_son)).mean())
    RMSE_djf = math.sqrt(np.square(np.subtract(obs_djf,model_djf)).mean())

    plt.scatter(obs_mam,model_mam,facecolors='none',s=150,marker='o',edgecolor=color,linewidths=2,label='MAM ($R^2$: ' + str(round(r_mam**2,2)) + ", $RMSE$: "  + str(round(RMSE_mam,r)) + ")")
    plt.scatter(obs_jja,model_jja,facecolors='none',s=150,marker='s',edgecolor=color,linewidths=2,label='JJA ($R^2$: ' + str(round(r_jja**2,2)) + ", $RMSE$: "  + str(round(RMSE_jja,r)) + ")")
    plt.scatter(obs_son,model_son,color=color,s=150,marker='D',edgecolor='k',label='SON ($R^2$: ' + str(round(r_son**2,2)) + ", $RMSE$: "  + str(round(RMSE_son,r)) + ")")
    plt.scatter(obs_djf,model_djf,color=color,s=150,marker='^',edgecolor='k',label='DJF ($R^2$: ' + str(round(r_djf**2,2)) + ", $RMSE$: "  + str(round(RMSE_djf,r)) + ")")



    plt.scatter(-999,-999,color='white',s=0,label='Annual ($R^2$: ' + str(round(r_annual**2,2)) + ", $RMSE$: " + str(round(RMSE,r)) + ")")

# =============================================================================
#     if variable == "t":
#         plt.title('Mean air temperature: ' + model_name, fontsize=22)
#     elif variable == "pr":
#         plt.title('Mean total precipitation: ' + model_name, fontsize=22)
#     elif variable == "wind":
#         plt.title('Mean wind speed: ' + model_name, fontsize=22)
# =============================================================================
            
    plt.legend(fontsize=20)

    
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/r2/' + figname + '_' + variable + '.png',bbox_inches='tight')


    
plot_all_szn(wrf_d03_mam,wrf_d03_jja,wrf_d03_son,wrf_d03_djf,'C0','CanESM2-WRF D03','wrf_d03_allszn',obs_mam,obs_jja,obs_son,obs_djf)
plot_all_szn(wrf_d02_mam,wrf_d02_jja,wrf_d02_son,wrf_d02_djf,'C1','CanESM2-WRF D02','wrf_d02_allszn',obs_mam,obs_jja,obs_son,obs_djf)
plot_all_szn(wrf_d01_mam,wrf_d01_jja,wrf_d01_son,wrf_d01_djf,'C2','CanESM2-WRF D01','wrf_d01_allszn',obs_mam,obs_jja,obs_son,obs_djf)
plot_all_szn(raw_mam,raw_jja,raw_son,raw_djf,'C3','CanESM2','raw_allszn',obs_mam,obs_jja,obs_son,obs_djf)
plot_all_szn(rcm_mam,rcm_jja,rcm_son,rcm_djf,'C4','CanRCM4','rcm_allszn',obs_mam,obs_jja,obs_son,obs_djf)

#%%

def plot_all_szn_elev(model_mam_input,model_jja_input,model_son_input,model_djf_input,model_name,figname): 
    
    model_mam = model_mam_input.copy()
    model_jja = model_jja_input.copy()
    model_son = model_son_input.copy()
    model_djf = model_djf_input.copy()
    
    fig = plt.figure(figsize=(10, 10),dpi=200)
    

    if variable == "t":
        vmin = -15
        vmax = 27
        unit = '(deg C)'
        
    if variable == "pr":
        vmin = 0
        vmax = 2300
        unit = '(mm/season)'
        
    plt.xlim([vmin,vmax])
    plt.ylim([vmin,vmax])
    
    plt.xlabel('Observed ' + unit,fontsize=18)
    plt.ylabel('Simulated ' + unit,fontsize=18)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    model_mam.index = [index + '_mam' for index in model_mam.index]
    model_jja.index = [index + '_jja' for index in model_jja.index]
    model_son.index = [index + '_son' for index in model_son.index]
    model_djf.index = [index + '_djf' for index in model_djf.index]

    model = pd.concat([model_mam,model_jja,model_son,model_djf])
    model.index = model.index.astype(str)
    model = model.sort_index()
    
    slope, intercept, r_value, p_value, std_err = linregress(obs, model)
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope * x + intercept
    plt.plot(x, line_of_best_fit,'--', color='grey')
    plt.plot(x, x, color='k')
    
    MSE = np.square(np.subtract(obs,model)).mean() 
    RMSE = math.sqrt(MSE)
    
    if variable == "pr":
        r=0
    else:
        r=1

    slope, intercept, r_mam, p_value, std_err = linregress(obs_mam, model_mam)
    slope, intercept, r_jja, p_value, std_err = linregress(obs_jja, model_jja)
    slope, intercept, r_son, p_value, std_err = linregress(obs_son, model_son)
    slope, intercept, r_djf, p_value, std_err = linregress(obs_djf, model_djf)

    RMSE_mam = math.sqrt(np.square(np.subtract(obs_mam,model_mam)).mean())
    RMSE_jja = math.sqrt(np.square(np.subtract(obs_jja,model_jja)).mean())
    RMSE_son = math.sqrt(np.square(np.subtract(obs_son,model_son)).mean())
    RMSE_djf = math.sqrt(np.square(np.subtract(obs_djf,model_djf)).mean())

    plt.scatter(obs_mam,model_mam,c=elev,cmap='terrain',s=150,marker='o',edgecolor='k',vmin=0,vmax=3000,label='MAM ($R^2$: ' + str(round(r_mam**2,2)) + ", RMSE: "  + str(round(RMSE_mam,r)) + ")")
    plt.scatter(obs_jja,model_jja,c=elev,cmap='terrain',s=150,marker='s',edgecolor='k',vmin=0,vmax=3000,label='JJA ($R^2$: ' + str(round(r_jja**2,2)) + ", RMSE: "  + str(round(RMSE_jja,r)) + ")")
    plt.scatter(obs_son,model_son,c=elev,cmap='terrain',s=150,marker='D',edgecolor='k',vmin=0,vmax=3000,label='SON ($R^2$: ' + str(round(r_son**2,2)) + ", RMSE: "  + str(round(RMSE_son,r)) + ")")
    plt.scatter(obs_djf,model_djf,c=elev,cmap='terrain',s=150,marker='^',edgecolor='k',vmin=0,vmax=3000,label='DJF ($R^2$: ' + str(round(r_djf**2,2)) + ", RMSE: "  + str(round(RMSE_djf,r)) + ")")



    plt.scatter(-999,-999,color='white',s=0,label='$R^2$ = ' + str(round(r_value**2,2)))
    plt.scatter(-999,-999,color='white',s=0,label='$RMSE$ = ' + str(round(RMSE,r)))

    if variable == "t":
        plt.title('Mean air temperature: ' + model_name, fontsize=22)
    else:
        plt.title('Mean total precipitation: ' + model_name, fontsize=22)
        
    plt.legend(fontsize=16)

    cbar_ax = fig.add_axes([0.93, 0.12, 0.03, 0.76])
    fig.colorbar(mpl.cm.ScalarMappable(cmap='terrain', norm=mpl.colors.Normalize(vmin=0, vmax=3000)),
                  cax=cbar_ax, ticks=np.arange(0, 3000+1, 500), orientation='vertical')
    cbar_ax.tick_params(labelsize=16)
    cbar_ax.set_ylabel('Elevation [m]',size=18) 

    
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/r2/' + figname + '_' + variable + '_terrain.png',bbox_inches='tight')


    
plot_all_szn_elev(wrf_d03_mam,wrf_d03_jja,wrf_d03_son,wrf_d03_djf,'CanESM2-WRF D03','wrf_d03_allszn')
plot_all_szn_elev(wrf_d02_mam,wrf_d02_jja,wrf_d02_son,wrf_d02_djf,'CanESM2-WRF D02','wrf_d02_allszn')
plot_all_szn_elev(raw_mam,raw_jja,raw_son,raw_djf,'CanESM2','raw_allszn')
plot_all_szn_elev(rcm_mam,rcm_jja,rcm_son,rcm_djf,'CanRCM4','rcm_allszn')
       
   #%%

obs_wet.index = [index + '_wet' for index in obs_wet.index]
obs_dry.index = [index + '_dry' for index in obs_dry.index]

obs_wetdry = pd.concat([obs_wet,obs_dry])
obs_wetdry.index = obs_wetdry.index.astype(str)
obs_wetdry = obs_wetdry.sort_index()

#%%

def plot_wetdry(model_wet_input,model_dry_input,color,model_name,figname): 
    
    model_wet = model_wet_input.copy()
    model_dry = model_dry_input.copy()

    plt.figure(figsize=(10, 10),dpi=200)
    
    if variable == "t":
        vmin = -15
        vmax = 27
        unit = '(deg C)'
        
    if variable == "pr":
        vmin = 0
        vmax = 4500
        unit = '(mm/6mo)'
        
    plt.xlim([vmin,vmax])
    plt.ylim([vmin,vmax])
    
    plt.xlabel('Observed ' + unit,fontsize=18)
    plt.ylabel('Simulated ' + unit,fontsize=18)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    model_wet.index = [index + '_wet' for index in model_wet.index]
    model_dry.index = [index + '_dry' for index in model_dry.index]

    
    model = pd.concat([model_wet,model_dry])
    model.index = model.index.astype(str)
    model = model.sort_index()
    
    slope, intercept, r_value, p_value, std_err = linregress(obs_wetdry, model)
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope * x + intercept
    plt.plot(x, line_of_best_fit,'--', color='grey')
    plt.plot(x, x, color='k')
    
    MSE = np.square(np.subtract(obs_wetdry,model)).mean() 
    RMSE = math.sqrt(MSE)
    
    if variable == "pr":
        r=0
    else:
        r=1


    slope, intercept, r_wet, p_value, std_err = linregress(obs_wet, model_wet)
    slope, intercept, r_dry, p_value, std_err = linregress(obs_dry, model_dry)

    RMSE_wet = math.sqrt(np.square(np.subtract(obs_wet,model_wet)).mean())
    RMSE_dry = math.sqrt(np.square(np.subtract(obs_dry,model_dry)).mean())


    plt.scatter(obs_wet,model_wet,color=color,s=150,marker='D',edgecolor='k',label='WET ($R^2$: ' + str(round(r_wet**2,2)) + ", RMSE: "  + str(round(RMSE_wet,r)) + ")")
    plt.scatter(obs_dry,model_dry,facecolor='none',s=150,marker='o',linewidths=2,edgecolor=color,label='DRY ($R^2$: ' + str(round(r_dry**2,2)) + ", RMSE: "  + str(round(RMSE_dry,r)) + ")")


    plt.scatter(-999,-999,color='white',s=0,label='$R^2$ = ' + str(round(r_value**2,2)))
    plt.scatter(-999,-999,color='white',s=0,label='$RMSE$ = ' + str(round(RMSE,r)))

    if variable == "t":
        plt.title('Mean air temperature: ' + model_name, fontsize=22)
    else:
        plt.title('Mean total precipitation: ' + model_name, fontsize=22)
        
    plt.legend(fontsize=16)

    
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/r2/' + figname + '_' + variable + '.png',bbox_inches='tight')


    
plot_wetdry(wrf_d03_wet,wrf_d03_dry,'C0','CanESM2-WRF D03','wrf_d03_wetdry')
plot_wetdry(wrf_d02_wet,wrf_d02_dry,'C1','CanESM2-WRF D02','wrf_d02_wetdry')
plot_wetdry(raw_wet,raw_dry,'C2','CanESM2','raw_wetdry')
plot_wetdry(rcm_wet,rcm_dry,'C3','CanRCM4','rcm_wetdry')
     
#%%
def plot_wetdry_elev(model_wet_input,model_dry_input,model_name,figname): 
    
    model_wet = model_wet_input.copy()
    model_dry = model_dry_input.copy()

    fig = plt.figure(figsize=(10, 10),dpi=200)
    
    if variable == "t":
        vmin = -15
        vmax = 27
        unit = '(deg C)'
        
    if variable == "pr":
        vmin = 0
        vmax = 4500
        unit = '(mm/6mo)'
        
    plt.xlim([vmin,vmax])
    plt.ylim([vmin,vmax])
    
    plt.xlabel('Observed ' + unit,fontsize=18)
    plt.ylabel('Simulated ' + unit,fontsize=18)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    model_wet.index = [index + '_wet' for index in model_wet.index]
    model_dry.index = [index + '_dry' for index in model_dry.index]

    
    model = pd.concat([model_wet,model_dry])
    model.index = model.index.astype(str)
    model = model.sort_index()
    
    slope, intercept, r_value, p_value, std_err = linregress(obs_wetdry, model)
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope * x + intercept
    plt.plot(x, line_of_best_fit,'--', color='grey')
    plt.plot(x, x, color='k')
    
    MSE = np.square(np.subtract(obs_wetdry,model)).mean() 
    RMSE = math.sqrt(MSE)
    
    if variable == "pr":
        r=0
    else:
        r=1


    slope, intercept, r_wet, p_value, std_err = linregress(obs_wet, model_wet)
    slope, intercept, r_dry, p_value, std_err = linregress(obs_dry, model_dry)

    RMSE_wet = math.sqrt(np.square(np.subtract(obs_wet,model_wet)).mean())
    RMSE_dry = math.sqrt(np.square(np.subtract(obs_dry,model_dry)).mean())


    plt.scatter(obs_wet,model_wet,c=elev,cmap='terrain',s=150,marker='o',edgecolor='k',vmin=0,vmax=3000,label='WET ($R^2$: ' + str(round(r_wet**2,2)) + ", RMSE: "  + str(round(RMSE_wet,r)) + ")")
    plt.scatter(obs_dry,model_dry,c=elev,cmap='terrain',s=150,marker='s',edgecolor='k',vmin=0,vmax=3000,label='DRY ($R^2$: ' + str(round(r_dry**2,2)) + ", RMSE: "  + str(round(RMSE_dry,r)) + ")")


    plt.scatter(-999,-999,color='white',s=0,label='$R^2$ = ' + str(round(r_value**2,2)))
    plt.scatter(-999,-999,color='white',s=0,label='$RMSE$ = ' + str(round(RMSE,r)))

    if variable == "t":
        plt.title('Mean air temperature: ' + model_name, fontsize=22)
    else:
        plt.title('Mean total precipitation: ' + model_name, fontsize=22)
        
    plt.legend(fontsize=16)

    cbar_ax = fig.add_axes([0.93, 0.12, 0.03, 0.76])
    fig.colorbar(mpl.cm.ScalarMappable(cmap='terrain', norm=mpl.colors.Normalize(vmin=0, vmax=3000)),
                  cax=cbar_ax, ticks=np.arange(0, 3000+1, 500), orientation='vertical')
    cbar_ax.tick_params(labelsize=16)
    cbar_ax.set_ylabel('Elevation [m]',size=18) 
        
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/r2/' + figname + '_' + variable + '_terrain.png',bbox_inches='tight')


    
plot_wetdry_elev(wrf_d03_wet,wrf_d03_dry,'CanESM2-WRF D03','wrf_d03_wetdry')
plot_wetdry_elev(wrf_d02_wet,wrf_d02_dry,'CanESM2-WRF D02','wrf_d02_wetdry')
plot_wetdry_elev(raw_wet,raw_dry,'CanESM2','raw_wetdry')
plot_wetdry_elev(rcm_wet,rcm_dry,'CanRCM4','rcm_wetdry')
            
               
        