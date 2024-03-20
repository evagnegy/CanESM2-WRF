
import numpy as np
import matplotlib.pyplot as plt
import sys 

sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from taylordiagram import TaylorDiagram

import pandas as pd
import warnings
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_noaa_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic,plot_all_d03,plot_zoomed_in
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib
from scipy.stats import linregress

variable = 'wind' #t or pr or wind
run = 'historical' #historical rcp45 or rcp85

output_freq = "daily" #yearly monthly or daily

#%%

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

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])

df = pd.read_csv(noaa_daily_stations)
noaa_station_IDs = list(df["STATION"])
noaa_station_names = list(df["NAME"])

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

    if variable != "wind":
        bch_obs = get_bch_obs(output_freq,bch_station_IDs,stations_dir,variable)


if variable != "wind":
    wrf_d02_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
    wrf_d03_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
    raw_bch = get_canesm2(output_freq, "BCH", bch_station_IDs, run, variable, raw_files_dir,start_year)
    rcm_bch = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, variable, rcm_files_dir)
    #pcic_bch = get_pcic(output_freq, "BCH", bch_station_IDs, run, variable, pcic_files_dir)

wrf_d02_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_eccc = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, variable, raw_files_dir,start_year)
rcm_eccc = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, variable, rcm_files_dir)
#pcic_eccc = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, variable, pcic_files_dir)

wrf_d02_noaa = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_noaa = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_noaa = get_canesm2(output_freq, "NOAA", noaa_station_IDs, run, variable, raw_files_dir,start_year)
rcm_noaa = get_canrcm4(output_freq, "NOAA", noaa_station_IDs, run, variable, rcm_files_dir)

#%%
# remove stations not in the original list
for station in noaa_station_IDs:
    if station not in list(noaa_obs.columns):
        wrf_d02_noaa.drop(station, inplace=True, axis=1)
        wrf_d03_noaa.drop(station, inplace=True, axis=1)
        raw_noaa.drop(station, inplace=True, axis=1)
        rcm_noaa.drop(station, inplace=True, axis=1)


#%%

raw_eccc.index = raw_eccc.index.to_timestamp()
rcm_eccc.index = rcm_eccc.index.to_timestamp()
raw_eccc = raw_eccc.reindex(eccc_obs.index)
rcm_eccc = rcm_eccc.reindex(eccc_obs.index)

raw_noaa.index = raw_noaa.index.to_timestamp()
rcm_noaa.index = rcm_noaa.index.to_timestamp()
raw_noaa = raw_noaa.reindex(noaa_obs.index)
rcm_noaa = rcm_noaa.reindex(noaa_obs.index)


wrf_d02_eccc[np.isnan(eccc_obs.values)] = np.nan
wrf_d03_eccc[np.isnan(eccc_obs.values)] = np.nan
raw_eccc[np.isnan(eccc_obs.values)] = np.nan
rcm_eccc[np.isnan(eccc_obs.values)] = np.nan

wrf_d02_noaa[np.isnan(noaa_obs.values)] = np.nan
wrf_d03_noaa[np.isnan(noaa_obs.values)] = np.nan
raw_noaa[np.isnan(noaa_obs.values)] = np.nan
rcm_noaa[np.isnan(noaa_obs.values)] = np.nan

#%%
# =============================================================================
# eccc_obs_avg = eccc_obs.mean().sort_index()
# bch_obs_avg = bch_obs.mean().sort_index()
# noaa_obs_avg = noaa_obs.mean().sort_index()
# obs_avg = pd.concat([eccc_obs_avg,bch_obs_avg,noaa_obs_avg])
# obs_avg.index = obs_avg.index.astype(str)
# obs_avg = obs_avg.sort_index()
# 
# wrf_eccc_d02_avg = wrf_d02_eccc.mean().sort_index()
# wrf_bch_d02_avg = wrf_d02_bch.mean().sort_index()
# wrf_noaa_d02_avg = wrf_d02_noaa.mean().sort_index()
# wrf_d02_avg = pd.concat([wrf_eccc_d02_avg,wrf_bch_d02_avg,wrf_noaa_d02_avg])
# wrf_d02_avg.index = wrf_d02_avg.index.astype(str)
# wrf_d02_avg = wrf_d02_avg.sort_index()
# 
# wrf_eccc_d03_avg = wrf_d03_eccc.mean().sort_index()
# wrf_bch_d03_avg = wrf_d03_bch.mean().sort_index()
# wrf_noaa_d03_avg = wrf_d03_noaa.mean().sort_index()
# wrf_d03_avg = pd.concat([wrf_eccc_d03_avg,wrf_bch_d03_avg,wrf_noaa_d03_avg])
# wrf_d03_avg.index = wrf_d03_avg.index.astype(str)
# wrf_d03_avg = wrf_d03_avg.sort_index()
# 
# raw_eccc_avg = raw_eccc.mean().sort_index()
# raw_bch_avg = raw_bch.mean().sort_index()
# raw_noaa_avg = raw_noaa.mean().sort_index()
# raw_avg = pd.concat([raw_eccc_avg,raw_bch_avg,raw_noaa_avg])
# raw_avg.index = raw_avg.index.astype(str)
# raw_avg = raw_avg.sort_index()
# 
# rcm_eccc_avg = rcm_eccc.mean().sort_index()
# rcm_bch_avg = rcm_bch.mean().sort_index()
# rcm_noaa_avg = rcm_noaa.mean().sort_index()
# rcm_avg = pd.concat([rcm_eccc_avg,rcm_bch_avg,rcm_noaa_avg])
# rcm_avg.index = rcm_avg.index.astype(str)
# rcm_avg = rcm_avg.sort_index()
# 
# =============================================================================

eccc_obs_avg = eccc_obs.mean().sort_index()
noaa_obs_avg = noaa_obs.mean().sort_index()
obs_avg = pd.concat([eccc_obs_avg,noaa_obs_avg])
obs_avg.index = obs_avg.index.astype(str)
obs_avg = obs_avg.sort_index()

wrf_eccc_d02_avg = wrf_d02_eccc.mean().sort_index()
wrf_noaa_d02_avg = wrf_d02_noaa.mean().sort_index()
wrf_d02_avg = pd.concat([wrf_eccc_d02_avg,wrf_noaa_d02_avg])
wrf_d02_avg.index = wrf_d02_avg.index.astype(str)
wrf_d02_avg = wrf_d02_avg.sort_index()

wrf_eccc_d03_avg = wrf_d03_eccc.mean().sort_index()
wrf_noaa_d03_avg = wrf_d03_noaa.mean().sort_index()
wrf_d03_avg = pd.concat([wrf_eccc_d03_avg,wrf_noaa_d03_avg])
wrf_d03_avg.index = wrf_d03_avg.index.astype(str)
wrf_d03_avg = wrf_d03_avg.sort_index()

raw_eccc_avg = raw_eccc.mean().sort_index()
raw_noaa_avg = raw_noaa.mean().sort_index()
raw_avg = pd.concat([raw_eccc_avg,raw_noaa_avg])
raw_avg.index = raw_avg.index.astype(str)
raw_avg = raw_avg.sort_index()

rcm_eccc_avg = rcm_eccc.mean().sort_index()
rcm_noaa_avg = rcm_noaa.mean().sort_index()
rcm_avg = pd.concat([rcm_eccc_avg,rcm_noaa_avg])
rcm_avg.index = rcm_avg.index.astype(str)
rcm_avg = rcm_avg.sort_index()

#%%

output_freq = "monthly" 

WRF_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/' + run + '/'
raw_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/' + run + '/'
rcm_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/' + run + '/'
pcic_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_PCIC/' + run + '/'

if run == 'historical': #no station obs for rcps
    eccc_obs_m = get_eccc_obs(output_freq,eccc_station_IDs,stations_dir,variable)
    #bch_obs_m = get_bch_obs(output_freq,bch_station_IDs,stations_dir,variable)
    noaa_obs_m = get_noaa_obs(output_freq,noaa_station_IDs,stations_dir,variable)

#wrf_d02_bch_m = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
#wrf_d03_bch_m = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
#raw_bch_m = get_canesm2(output_freq, "BCH", bch_station_IDs, run, variable, raw_files_dir,start_year)
#rcm_bch_m = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, variable, rcm_files_dir)
#pcic_bch = get_pcic(output_freq, "BCH", bch_station_IDs, run, variable, pcic_files_dir)

wrf_d02_eccc_m = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_eccc_m = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_eccc_m = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, variable, raw_files_dir,start_year)
rcm_eccc_m = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, variable, rcm_files_dir)
#pcic_eccc = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, variable, pcic_files_dir)

wrf_d02_noaa_m = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_noaa_m = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_noaa_m = get_canesm2(output_freq, "NOAA", noaa_station_IDs, run, variable, raw_files_dir,start_year)
rcm_noaa_m = get_canrcm4(output_freq, "NOAA", noaa_station_IDs, run, variable, rcm_files_dir)

#%%

# remove stations not in the original list
for station in noaa_station_IDs:
    if station not in list(noaa_obs_m.columns):
        wrf_d02_noaa_m.drop(station, inplace=True, axis=1)
        wrf_d03_noaa_m.drop(station, inplace=True, axis=1)
        raw_noaa_m.drop(station, inplace=True, axis=1)
        rcm_noaa_m.drop(station, inplace=True, axis=1)


#%% obs 

obs_bch_mam = bch_obs_m.copy()
obs_eccc_mam = eccc_obs_m.copy()
obs_bch_jja = bch_obs_m.copy()
obs_eccc_jja = eccc_obs_m.copy()
obs_bch_son = bch_obs_m.copy()
obs_eccc_son = eccc_obs_m.copy()
obs_bch_djf = bch_obs_m.copy()
obs_eccc_djf = eccc_obs_m.copy()
obs_bch_wet = bch_obs_m.copy()
obs_eccc_wet = eccc_obs_m.copy()
obs_bch_dry = bch_obs_m.copy()
obs_eccc_dry = eccc_obs_m.copy()

obs_noaa_mam = noaa_obs_m.copy()
obs_noaa_jja = noaa_obs_m.copy()
obs_noaa_son = noaa_obs_m.copy()
obs_noaa_djf = noaa_obs_m.copy()
obs_noaa_wet = noaa_obs_m.copy()
obs_noaa_dry = noaa_obs_m.copy()



for i in [1,2,6,7,8,9,10,11,12]:
    obs_bch_mam = obs_bch_mam[obs_bch_mam.index.month != i]
    obs_eccc_mam = obs_eccc_mam[obs_eccc_mam.index.month != i]
    obs_noaa_mam = obs_noaa_mam[obs_noaa_mam.index.month != i]


for i in [1,2,3,4,5,9,10,11,12]:
    obs_bch_jja = obs_bch_jja[obs_bch_jja.index.month != i]
    obs_eccc_jja = obs_eccc_jja[obs_eccc_jja.index.month != i]
    obs_noaa_jja = obs_noaa_jja[obs_noaa_jja.index.month != i]


for i in [1,2,3,4,5,6,7,8,12]:
    obs_bch_son = obs_bch_son[obs_bch_son.index.month != i]
    obs_eccc_son = obs_eccc_son[obs_eccc_son.index.month != i]
    obs_noaa_son = obs_noaa_son[obs_noaa_son.index.month != i]

    
for i in [3,4,5,6,7,8,9,10,11]:
    obs_bch_djf = obs_bch_djf[obs_bch_djf.index.month != i]
    obs_eccc_djf = obs_eccc_djf[obs_eccc_djf.index.month != i]
    obs_noaa_djf = obs_noaa_djf[obs_noaa_djf.index.month != i]


for i in [4,5,6,7,8,9]:
    obs_bch_wet = obs_bch_wet[obs_bch_wet.index.month != i]
    obs_eccc_wet = obs_eccc_wet[obs_eccc_wet.index.month != i]
    obs_noaa_wet = obs_noaa_wet[obs_noaa_wet.index.month != i]

    
for i in [1,2,3,10,11,12]:
    obs_bch_dry = obs_bch_dry[obs_bch_dry.index.month != i]
    obs_eccc_dry = obs_eccc_dry[obs_eccc_dry.index.month != i]
    obs_noaa_dry = obs_noaa_dry[obs_noaa_dry.index.month != i]

    
if variable == "pr":
    eccc_obs_mam_avg = obs_eccc_mam.groupby(obs_eccc_mam.index.year).sum().mean().sort_index()
    bch_obs_mam_avg = obs_bch_mam.groupby(obs_bch_mam.index.year).sum().mean().sort_index()
    noaa_obs_mam_avg = obs_noaa_mam.groupby(obs_noaa_mam.index.year).sum().mean().sort_index()
    obs_mam_avg = pd.concat([eccc_obs_mam_avg,bch_obs_mam_avg,noaa_obs_mam_avg])
    obs_mam_avg.index = obs_mam_avg.index.astype(str)
    obs_mam_avg = obs_mam_avg.sort_index()
    
    eccc_obs_jja_avg = obs_eccc_jja.groupby(obs_eccc_jja.index.year).sum().mean().sort_index()
    bch_obs_jja_avg = obs_bch_jja.groupby(obs_bch_jja.index.year).sum().mean().sort_index()
    noaa_obs_jja_avg = obs_noaa_jja.groupby(obs_noaa_jja.index.year).sum().mean().sort_index()
    obs_jja_avg = pd.concat([eccc_obs_jja_avg,bch_obs_jja_avg,noaa_obs_jja_avg])
    obs_jja_avg.index = obs_jja_avg.index.astype(str)
    obs_jja_avg = obs_jja_avg.sort_index()
    
    eccc_obs_son_avg = obs_eccc_son.groupby(obs_eccc_son.index.year).sum().mean().sort_index()
    bch_obs_son_avg = obs_bch_son.groupby(obs_bch_son.index.year).sum().mean().sort_index()
    noaa_obs_son_avg = obs_noaa_son.groupby(obs_noaa_son.index.year).sum().mean().sort_index()
    obs_son_avg = pd.concat([eccc_obs_son_avg,bch_obs_son_avg,noaa_obs_son_avg])
    obs_son_avg.index = obs_son_avg.index.astype(str)
    obs_son_avg = obs_son_avg.sort_index()
    
    eccc_obs_djf_avg = obs_eccc_djf.groupby(obs_eccc_djf.index.year).sum().mean().sort_index()
    bch_obs_djf_avg = obs_bch_djf.groupby(obs_bch_djf.index.year).sum().mean().sort_index()
    noaa_obs_djf_avg = obs_noaa_djf.groupby(obs_noaa_djf.index.year).sum().mean().sort_index()
    obs_djf_avg = pd.concat([eccc_obs_djf_avg,bch_obs_djf_avg,noaa_obs_djf_avg])
    obs_djf_avg.index = obs_djf_avg.index.astype(str)
    obs_djf_avg = obs_djf_avg.sort_index()
    
    eccc_obs_wet_avg = obs_eccc_wet.groupby(obs_eccc_wet.index.year).sum().mean().sort_index()
    bch_obs_wet_avg = obs_bch_wet.groupby(obs_bch_wet.index.year).sum().mean().sort_index()
    noaa_obs_wet_avg = obs_noaa_wet.groupby(obs_noaa_wet.index.year).sum().mean().sort_index()
    obs_wet_avg = pd.concat([eccc_obs_wet_avg,bch_obs_wet_avg,noaa_obs_wet_avg])
    obs_wet_avg.index = obs_wet_avg.index.astype(str)
    obs_wet_avg = obs_wet_avg.sort_index()
    
    eccc_obs_dry_avg = obs_eccc_dry.groupby(obs_eccc_dry.index.year).sum().mean().sort_index()
    bch_obs_dry_avg = obs_bch_dry.groupby(obs_bch_dry.index.year).sum().mean().sort_index()
    noaa_obs_dry_avg = obs_noaa_dry.groupby(obs_noaa_dry.index.year).sum().mean().sort_index()
    obs_dry_avg = pd.concat([eccc_obs_dry_avg,bch_obs_dry_avg,noaa_obs_dry_avg])
    obs_dry_avg.index = obs_dry_avg.index.astype(str)
    obs_dry_avg = obs_dry_avg.sort_index()
 
    
wrf_d02_bch_mam = wrf_d02_bch_m.copy()
wrf_d02_eccc_mam = wrf_d02_eccc_m.copy()
wrf_d03_bch_mam = wrf_d03_bch_m.copy()
wrf_d03_eccc_mam = wrf_d03_eccc_m.copy()
raw_bch_mam = raw_bch_m.copy()
raw_eccc_mam = raw_eccc_m.copy()
rcm_bch_mam = rcm_bch_m.copy()
rcm_eccc_mam = rcm_eccc_m.copy()
wrf_d02_noaa_mam = wrf_d02_noaa_m.copy()
wrf_d03_noaa_mam = wrf_d03_noaa_m.copy()
raw_noaa_mam = raw_noaa_m.copy()
rcm_noaa_mam = rcm_noaa_m.copy()

for i in [1,2,6,7,8,9,10,11,12]:
    wrf_d02_bch_mam = wrf_d02_bch_mam[wrf_d02_bch_mam.index.month != i]
    wrf_d02_eccc_mam = wrf_d02_eccc_mam[wrf_d02_eccc_mam.index.month != i]
    wrf_d03_bch_mam = wrf_d03_bch_mam[wrf_d03_bch_mam.index.month != i]
    wrf_d03_eccc_mam = wrf_d03_eccc_mam[wrf_d03_eccc_mam.index.month != i]
    raw_bch_mam = raw_bch_mam[raw_bch_mam.index.month != i]
    raw_eccc_mam = raw_eccc_mam[raw_eccc_mam.index.month != i]
    rcm_bch_mam = rcm_bch_mam[rcm_bch_mam.index.month != i]
    rcm_eccc_mam = rcm_eccc_mam[rcm_eccc_mam.index.month != i]
    wrf_d02_noaa_mam = wrf_d02_noaa_mam[wrf_d02_noaa_mam.index.month != i]
    wrf_d03_noaa_mam = wrf_d03_noaa_mam[wrf_d03_noaa_mam.index.month != i]
    raw_noaa_mam = raw_noaa_mam[raw_noaa_mam.index.month != i]
    rcm_noaa_mam = rcm_noaa_mam[rcm_noaa_mam.index.month != i]
    
wrf_d02_bch_jja = wrf_d02_bch_m.copy()
wrf_d02_eccc_jja = wrf_d02_eccc_m.copy()
wrf_d03_bch_jja = wrf_d03_bch_m.copy()
wrf_d03_eccc_jja = wrf_d03_eccc_m.copy()
raw_bch_jja = raw_bch_m.copy()
raw_eccc_jja = raw_eccc_m.copy()
rcm_bch_jja = rcm_bch_m.copy()
rcm_eccc_jja = rcm_eccc_m.copy()
wrf_d02_noaa_jja = wrf_d02_noaa_m.copy()
wrf_d03_noaa_jja = wrf_d03_noaa_m.copy()
raw_noaa_jja = raw_noaa_m.copy()
rcm_noaa_jja = rcm_noaa_m.copy()

for i in [1,2,3,4,5,9,10,11,12]:
    wrf_d02_bch_jja = wrf_d02_bch_jja[wrf_d02_bch_jja.index.month != i]
    wrf_d02_eccc_jja = wrf_d02_eccc_jja[wrf_d02_eccc_jja.index.month != i]
    wrf_d03_bch_jja = wrf_d03_bch_jja[wrf_d03_bch_jja.index.month != i]
    wrf_d03_eccc_jja = wrf_d03_eccc_jja[wrf_d03_eccc_jja.index.month != i]
    raw_bch_jja = raw_bch_jja[raw_bch_jja.index.month != i]
    raw_eccc_jja = raw_eccc_jja[raw_eccc_jja.index.month != i]   
    rcm_bch_jja = rcm_bch_jja[rcm_bch_jja.index.month != i]
    rcm_eccc_jja = rcm_eccc_jja[rcm_eccc_jja.index.month != i]
    wrf_d02_noaa_jja = wrf_d02_noaa_jja[wrf_d02_noaa_jja.index.month != i]
    wrf_d03_noaa_jja = wrf_d03_noaa_jja[wrf_d03_noaa_jja.index.month != i]
    raw_noaa_jja = raw_noaa_jja[raw_noaa_jja.index.month != i]   
    rcm_noaa_jja = rcm_noaa_jja[rcm_noaa_jja.index.month != i]
    
wrf_d02_bch_son = wrf_d02_bch_m.copy()
wrf_d02_eccc_son = wrf_d02_eccc_m.copy()
wrf_d03_bch_son = wrf_d03_bch_m.copy()
wrf_d03_eccc_son = wrf_d03_eccc_m.copy()
raw_bch_son = raw_bch_m.copy()
raw_eccc_son = raw_eccc_m.copy()
rcm_bch_son = rcm_bch_m.copy()
rcm_eccc_son = rcm_eccc_m.copy()
wrf_d02_noaa_son = wrf_d02_noaa_m.copy()
wrf_d03_noaa_son = wrf_d03_noaa_m.copy()
raw_noaa_son = raw_noaa_m.copy()
rcm_noaa_son = rcm_noaa_m.copy()

for i in [1,2,3,4,5,6,7,8,12]:
    wrf_d02_bch_son = wrf_d02_bch_son[wrf_d02_bch_son.index.month != i]
    wrf_d02_eccc_son = wrf_d02_eccc_son[wrf_d02_eccc_son.index.month != i]
    wrf_d03_bch_son = wrf_d03_bch_son[wrf_d03_bch_son.index.month != i]
    wrf_d03_eccc_son = wrf_d03_eccc_son[wrf_d03_eccc_son.index.month != i]
    raw_bch_son = raw_bch_son[raw_bch_son.index.month != i]
    raw_eccc_son = raw_eccc_son[raw_eccc_son.index.month != i]
    rcm_bch_son = rcm_bch_son[rcm_bch_son.index.month != i]
    rcm_eccc_son = rcm_eccc_son[rcm_eccc_son.index.month != i]
    wrf_d02_noaa_son = wrf_d02_noaa_son[wrf_d02_noaa_son.index.month != i]
    wrf_d03_noaa_son = wrf_d03_noaa_son[wrf_d03_noaa_son.index.month != i]
    raw_noaa_son = raw_noaa_son[raw_noaa_son.index.month != i]
    rcm_noaa_son = rcm_noaa_son[rcm_noaa_son.index.month != i]
 
    
wrf_d02_bch_djf = wrf_d02_bch_m.copy()
wrf_d02_eccc_djf = wrf_d02_eccc_m.copy()
wrf_d03_bch_djf = wrf_d03_bch_m.copy()
wrf_d03_eccc_djf = wrf_d03_eccc_m.copy()
raw_bch_djf = raw_bch_m.copy()
raw_eccc_djf = raw_eccc_m.copy()
rcm_bch_djf = rcm_bch_m.copy()
rcm_eccc_djf = rcm_eccc_m.copy()
wrf_d02_noaa_djf = wrf_d02_noaa_m.copy()
wrf_d03_noaa_djf = wrf_d03_noaa_m.copy()
raw_noaa_djf = raw_noaa_m.copy()
rcm_noaa_djf = rcm_noaa_m.copy()

for i in [3,4,5,6,7,8,9,10,11]:
    wrf_d02_bch_djf = wrf_d02_bch_djf[wrf_d02_bch_djf.index.month != i]
    wrf_d02_eccc_djf = wrf_d02_eccc_djf[wrf_d02_eccc_djf.index.month != i]
    wrf_d03_bch_djf = wrf_d03_bch_djf[wrf_d03_bch_djf.index.month != i]
    wrf_d03_eccc_djf = wrf_d03_eccc_djf[wrf_d03_eccc_djf.index.month != i]
    raw_bch_djf = raw_bch_djf[raw_bch_djf.index.month != i]
    raw_eccc_djf = raw_eccc_djf[raw_eccc_djf.index.month != i]  
    rcm_bch_djf = rcm_bch_djf[rcm_bch_djf.index.month != i]
    rcm_eccc_djf = rcm_eccc_djf[rcm_eccc_djf.index.month != i]
    wrf_d02_noaa_djf = wrf_d02_noaa_djf[wrf_d02_noaa_djf.index.month != i]
    wrf_d03_noaa_djf = wrf_d03_noaa_djf[wrf_d03_noaa_djf.index.month != i]
    raw_noaa_djf = raw_noaa_djf[raw_noaa_djf.index.month != i]  
    rcm_noaa_djf = rcm_noaa_djf[rcm_noaa_djf.index.month != i]


wrf_d02_bch_wet = wrf_d02_bch_m.copy()
wrf_d02_eccc_wet = wrf_d02_eccc_m.copy()
wrf_d03_bch_wet = wrf_d03_bch_m.copy()
wrf_d03_eccc_wet = wrf_d03_eccc_m.copy()
raw_bch_wet = raw_bch_m.copy()
raw_eccc_wet = raw_eccc_m.copy()
rcm_bch_wet = rcm_bch_m.copy()
rcm_eccc_wet = rcm_eccc_m.copy()
wrf_d02_noaa_wet = wrf_d02_noaa_m.copy()
wrf_d03_noaa_wet = wrf_d03_noaa_m.copy()
raw_noaa_wet = raw_noaa_m.copy()
rcm_noaa_wet = rcm_noaa_m.copy()

for i in [4,5,6,7,8,9]:
    wrf_d02_bch_wet = wrf_d02_bch_wet[wrf_d02_bch_wet.index.month != i]
    wrf_d02_eccc_wet = wrf_d02_eccc_wet[wrf_d02_eccc_wet.index.month != i]
    wrf_d03_bch_wet = wrf_d03_bch_wet[wrf_d03_bch_wet.index.month != i]
    wrf_d03_eccc_wet = wrf_d03_eccc_wet[wrf_d03_eccc_wet.index.month != i]
    raw_bch_wet = raw_bch_wet[raw_bch_wet.index.month != i]
    raw_eccc_wet = raw_eccc_wet[raw_eccc_wet.index.month != i]  
    rcm_bch_wet = rcm_bch_wet[rcm_bch_wet.index.month != i]
    rcm_eccc_wet = rcm_eccc_wet[rcm_eccc_wet.index.month != i]
    wrf_d02_noaa_wet = wrf_d02_noaa_wet[wrf_d02_noaa_wet.index.month != i]
    wrf_d03_noaa_wet = wrf_d03_noaa_wet[wrf_d03_noaa_wet.index.month != i]
    raw_noaa_wet = raw_noaa_wet[raw_noaa_wet.index.month != i]  
    rcm_noaa_wet = rcm_noaa_wet[rcm_noaa_wet.index.month != i]

wrf_d02_bch_dry = wrf_d02_bch_m.copy()
wrf_d02_eccc_dry = wrf_d02_eccc_m.copy()
wrf_d03_bch_dry = wrf_d03_bch_m.copy()
wrf_d03_eccc_dry = wrf_d03_eccc_m.copy()
raw_bch_dry = raw_bch_m.copy()
raw_eccc_dry = raw_eccc_m.copy()
rcm_bch_dry = rcm_bch_m.copy()
rcm_eccc_dry = rcm_eccc_m.copy()
wrf_d02_noaa_dry = wrf_d02_noaa_m.copy()
wrf_d03_noaa_dry = wrf_d03_noaa_m.copy()
raw_noaa_dry = raw_noaa_m.copy()
rcm_noaa_dry = rcm_noaa_m.copy()

for i in [1,2,3,10,11,12]:
    wrf_d02_bch_dry = wrf_d02_bch_dry[wrf_d02_bch_dry.index.month != i]
    wrf_d02_eccc_dry = wrf_d02_eccc_dry[wrf_d02_eccc_dry.index.month != i]
    wrf_d03_bch_dry = wrf_d03_bch_dry[wrf_d03_bch_dry.index.month != i]
    wrf_d03_eccc_dry = wrf_d03_eccc_dry[wrf_d03_eccc_dry.index.month != i]
    raw_bch_dry = raw_bch_dry[raw_bch_dry.index.month != i]
    raw_eccc_dry = raw_eccc_dry[raw_eccc_dry.index.month != i]  
    rcm_bch_dry = rcm_bch_dry[rcm_bch_dry.index.month != i]
    rcm_eccc_dry = rcm_eccc_dry[rcm_eccc_dry.index.month != i]
    wrf_d02_noaa_dry = wrf_d02_noaa_dry[wrf_d02_noaa_dry.index.month != i]
    wrf_d03_noaa_dry = wrf_d03_noaa_dry[wrf_d03_noaa_dry.index.month != i]
    raw_noaa_dry = raw_noaa_dry[raw_noaa_dry.index.month != i]  
    rcm_noaa_dry = rcm_noaa_dry[rcm_noaa_dry.index.month != i]
 
if variable == "pr":

    wrf_eccc_d02_mam_avg = wrf_d02_eccc_mam.groupby(wrf_d02_eccc_mam.index.year).sum().mean().sort_index()
    wrf_bch_d02_mam_avg = wrf_d02_bch_mam.groupby(wrf_d02_bch_mam.index.year).sum().mean().sort_index()
    wrf_eccc_d02_jja_avg = wrf_d02_eccc_jja.groupby(wrf_d02_eccc_jja.index.year).sum().mean().sort_index()
    wrf_bch_d02_jja_avg = wrf_d02_bch_jja.groupby(wrf_d02_bch_jja.index.year).sum().mean().sort_index()
    wrf_eccc_d02_son_avg = wrf_d02_eccc_son.groupby(wrf_d02_eccc_son.index.year).sum().mean().sort_index()
    wrf_bch_d02_son_avg = wrf_d02_bch_son.groupby(wrf_d02_bch_son.index.year).sum().mean().sort_index()
    wrf_eccc_d02_djf_avg = wrf_d02_eccc_djf.groupby(wrf_d02_eccc_djf.index.year).sum().mean().sort_index()
    wrf_bch_d02_djf_avg = wrf_d02_bch_djf.groupby(wrf_d02_bch_djf.index.year).sum().mean().sort_index()
    wrf_eccc_d02_wet_avg = wrf_d02_eccc_wet.groupby(wrf_d02_eccc_wet.index.year).sum().mean().sort_index()
    wrf_bch_d02_wet_avg = wrf_d02_bch_wet.groupby(wrf_d02_bch_wet.index.year).sum().mean().sort_index()
    wrf_eccc_d02_dry_avg = wrf_d02_eccc_dry.groupby(wrf_d02_eccc_dry.index.year).sum().mean().sort_index()
    wrf_bch_d02_dry_avg = wrf_d02_bch_dry.groupby(wrf_d02_bch_dry.index.year).sum().mean().sort_index()
    
    wrf_eccc_d03_mam_avg = wrf_d03_eccc_mam.groupby(wrf_d03_eccc_mam.index.year).sum().mean().sort_index()
    wrf_bch_d03_mam_avg = wrf_d03_bch_mam.groupby(wrf_d03_bch_mam.index.year).sum().mean().sort_index()
    wrf_eccc_d03_jja_avg = wrf_d03_eccc_jja.groupby(wrf_d03_eccc_jja.index.year).sum().mean().sort_index()
    wrf_bch_d03_jja_avg = wrf_d03_bch_jja.groupby(wrf_d03_bch_jja.index.year).sum().mean().sort_index()
    wrf_eccc_d03_son_avg = wrf_d03_eccc_son.groupby(wrf_d03_eccc_son.index.year).sum().mean().sort_index()
    wrf_bch_d03_son_avg = wrf_d03_bch_son.groupby(wrf_d03_bch_son.index.year).sum().mean().sort_index()
    wrf_eccc_d03_djf_avg = wrf_d03_eccc_djf.groupby(wrf_d03_eccc_djf.index.year).sum().mean().sort_index()
    wrf_bch_d03_djf_avg = wrf_d03_bch_djf.groupby(wrf_d03_bch_djf.index.year).sum().mean().sort_index()
    wrf_eccc_d03_wet_avg = wrf_d03_eccc_wet.groupby(wrf_d03_eccc_wet.index.year).sum().mean().sort_index()
    wrf_bch_d03_wet_avg = wrf_d03_bch_wet.groupby(wrf_d03_bch_wet.index.year).sum().mean().sort_index()
    wrf_eccc_d03_dry_avg = wrf_d03_eccc_dry.groupby(wrf_d03_eccc_dry.index.year).sum().mean().sort_index()
    wrf_bch_d03_dry_avg = wrf_d03_bch_dry.groupby(wrf_d03_bch_dry.index.year).sum().mean().sort_index()
    
    raw_eccc_mam_avg = raw_eccc_mam.groupby(raw_eccc_mam.index.year).sum().mean().sort_index()
    raw_bch_mam_avg = raw_bch_mam.groupby(raw_bch_mam.index.year).sum().mean().sort_index()
    raw_eccc_jja_avg = raw_eccc_jja.groupby(raw_eccc_jja.index.year).sum().mean().sort_index()
    raw_bch_jja_avg = raw_bch_jja.groupby(raw_bch_jja.index.year).sum().mean().sort_index()
    raw_eccc_son_avg = raw_eccc_son.groupby(raw_eccc_son.index.year).sum().mean().sort_index()
    raw_bch_son_avg = raw_bch_son.groupby(raw_bch_son.index.year).sum().mean().sort_index()
    raw_eccc_djf_avg = raw_eccc_djf.groupby(raw_eccc_djf.index.year).sum().mean().sort_index()
    raw_bch_djf_avg = raw_bch_djf.groupby(raw_bch_djf.index.year).sum().mean().sort_index()
    raw_eccc_wet_avg = raw_eccc_wet.groupby(raw_eccc_wet.index.year).sum().mean().sort_index()
    raw_bch_wet_avg = raw_bch_wet.groupby(raw_bch_wet.index.year).sum().mean().sort_index()
    raw_eccc_dry_avg = raw_eccc_dry.groupby(raw_eccc_dry.index.year).sum().mean().sort_index()
    raw_bch_dry_avg = raw_bch_dry.groupby(raw_bch_dry.index.year).sum().mean().sort_index()
    
    rcm_eccc_mam_avg = rcm_eccc_mam.groupby(rcm_eccc_mam.index.year).sum().mean().sort_index()
    rcm_bch_mam_avg =  rcm_bch_mam.groupby(rcm_bch_mam.index.year).sum().mean().sort_index()
    rcm_eccc_jja_avg = rcm_eccc_jja.groupby(rcm_eccc_jja.index.year).sum().mean().sort_index()
    rcm_bch_jja_avg = rcm_bch_jja.groupby(rcm_bch_jja.index.year).sum().mean().sort_index()
    rcm_eccc_son_avg = rcm_eccc_son.groupby(rcm_eccc_son.index.year).sum().mean().sort_index()
    rcm_bch_son_avg = rcm_bch_son.groupby(rcm_bch_son.index.year).sum().mean().sort_index()
    rcm_eccc_djf_avg = rcm_eccc_djf.groupby(rcm_eccc_djf.index.year).sum().mean().sort_index()
    rcm_bch_djf_avg = rcm_bch_djf.groupby(rcm_bch_djf.index.year).sum().mean().sort_index()
    rcm_eccc_wet_avg = rcm_eccc_wet.groupby(rcm_eccc_wet.index.year).sum().mean().sort_index()
    rcm_bch_wet_avg = rcm_bch_wet.groupby(rcm_bch_wet.index.year).sum().mean().sort_index()
    rcm_eccc_dry_avg = rcm_eccc_dry.groupby(rcm_eccc_dry.index.year).sum().mean().sort_index()
    rcm_bch_dry_avg = rcm_bch_dry.groupby(rcm_bch_dry.index.year).sum().mean().sort_index()

    wrf_noaa_d02_mam_avg = wrf_d02_noaa_mam.groupby(wrf_d02_noaa_mam.index.year).sum().mean().sort_index()
    wrf_noaa_d02_jja_avg = wrf_d02_noaa_jja.groupby(wrf_d02_noaa_jja.index.year).sum().mean().sort_index()
    wrf_noaa_d02_son_avg = wrf_d02_noaa_son.groupby(wrf_d02_noaa_son.index.year).sum().mean().sort_index()
    wrf_noaa_d02_djf_avg = wrf_d02_noaa_djf.groupby(wrf_d02_noaa_djf.index.year).sum().mean().sort_index()
    wrf_noaa_d02_wet_avg = wrf_d02_noaa_wet.groupby(wrf_d02_noaa_wet.index.year).sum().mean().sort_index()
    wrf_noaa_d02_dry_avg = wrf_d02_noaa_dry.groupby(wrf_d02_noaa_dry.index.year).sum().mean().sort_index()
    
    wrf_noaa_d03_mam_avg = wrf_d03_noaa_mam.groupby(wrf_d03_noaa_mam.index.year).sum().mean().sort_index()
    wrf_noaa_d03_jja_avg = wrf_d03_noaa_jja.groupby(wrf_d03_noaa_jja.index.year).sum().mean().sort_index()
    wrf_noaa_d03_son_avg = wrf_d03_noaa_son.groupby(wrf_d03_noaa_son.index.year).sum().mean().sort_index()
    wrf_noaa_d03_djf_avg = wrf_d03_noaa_djf.groupby(wrf_d03_noaa_djf.index.year).sum().mean().sort_index()
    wrf_noaa_d03_wet_avg = wrf_d03_noaa_wet.groupby(wrf_d03_noaa_wet.index.year).sum().mean().sort_index()
    wrf_noaa_d03_dry_avg = wrf_d03_noaa_dry.groupby(wrf_d03_noaa_dry.index.year).sum().mean().sort_index()
    
    raw_noaa_mam_avg = raw_noaa_mam.groupby(raw_noaa_mam.index.year).sum().mean().sort_index()
    raw_noaa_jja_avg = raw_noaa_jja.groupby(raw_noaa_jja.index.year).sum().mean().sort_index()
    raw_noaa_son_avg = raw_noaa_son.groupby(raw_noaa_son.index.year).sum().mean().sort_index()
    raw_noaa_djf_avg = raw_noaa_djf.groupby(raw_noaa_djf.index.year).sum().mean().sort_index()
    raw_noaa_wet_avg = raw_noaa_wet.groupby(raw_noaa_wet.index.year).sum().mean().sort_index()
    raw_noaa_dry_avg = raw_noaa_dry.groupby(raw_noaa_dry.index.year).sum().mean().sort_index()
    
    rcm_noaa_mam_avg = rcm_noaa_mam.groupby(rcm_noaa_mam.index.year).sum().mean().sort_index()
    rcm_noaa_jja_avg = rcm_noaa_jja.groupby(rcm_noaa_jja.index.year).sum().mean().sort_index()
    rcm_noaa_son_avg = rcm_noaa_son.groupby(rcm_noaa_son.index.year).sum().mean().sort_index()
    rcm_noaa_djf_avg = rcm_noaa_djf.groupby(rcm_noaa_djf.index.year).sum().mean().sort_index()
    rcm_noaa_wet_avg = rcm_noaa_wet.groupby(rcm_noaa_wet.index.year).sum().mean().sort_index()
    rcm_noaa_dry_avg = rcm_noaa_dry.groupby(rcm_noaa_dry.index.year).sum().mean().sort_index()

if variable == "t":
    eccc_obs_mam_avg = obs_eccc_mam.groupby(obs_eccc_mam.index.year).mean().mean().sort_index()
    bch_obs_mam_avg = obs_bch_mam.groupby(obs_bch_mam.index.year).mean().mean().sort_index()
    noaa_obs_mam_avg = obs_noaa_mam.groupby(obs_noaa_mam.index.year).mean().mean().sort_index()
    obs_mam_avg = pd.concat([eccc_obs_mam_avg,bch_obs_mam_avg,noaa_obs_mam_avg])
    obs_mam_avg.index = obs_mam_avg.index.astype(str)
    obs_mam_avg = obs_mam_avg.sort_index()
    
    eccc_obs_jja_avg = obs_eccc_jja.groupby(obs_eccc_jja.index.year).mean().mean().sort_index()
    bch_obs_jja_avg = obs_bch_jja.groupby(obs_bch_jja.index.year).mean().mean().sort_index()
    noaa_obs_jja_avg = obs_noaa_jja.groupby(obs_noaa_jja.index.year).mean().mean().sort_index()
    obs_jja_avg = pd.concat([eccc_obs_jja_avg,bch_obs_jja_avg,noaa_obs_jja_avg])
    obs_jja_avg.index = obs_jja_avg.index.astype(str)
    obs_jja_avg = obs_jja_avg.sort_index()
    
    eccc_obs_son_avg = obs_eccc_son.groupby(obs_eccc_son.index.year).mean().mean().sort_index()
    bch_obs_son_avg = obs_bch_son.groupby(obs_bch_son.index.year).mean().mean().sort_index()
    noaa_obs_son_avg = obs_noaa_son.groupby(obs_noaa_son.index.year).mean().mean().sort_index()
    obs_son_avg = pd.concat([eccc_obs_son_avg,bch_obs_son_avg,noaa_obs_son_avg])
    obs_son_avg.index = obs_son_avg.index.astype(str)
    obs_son_avg = obs_son_avg.sort_index()
    
    eccc_obs_djf_avg = obs_eccc_djf.groupby(obs_eccc_djf.index.year).mean().mean().sort_index()
    bch_obs_djf_avg = obs_bch_djf.groupby(obs_bch_djf.index.year).mean().mean().sort_index()
    noaa_obs_djf_avg = obs_noaa_djf.groupby(obs_noaa_djf.index.year).mean().mean().sort_index()
    obs_djf_avg = pd.concat([eccc_obs_djf_avg,bch_obs_djf_avg,noaa_obs_djf_avg])
    obs_djf_avg.index = obs_djf_avg.index.astype(str)
    obs_djf_avg = obs_djf_avg.sort_index()
    
    eccc_obs_wet_avg = obs_eccc_wet.groupby(obs_eccc_wet.index.year).mean().mean().sort_index()
    bch_obs_wet_avg = obs_bch_wet.groupby(obs_bch_wet.index.year).mean().mean().sort_index()
    noaa_obs_wet_avg = obs_noaa_wet.groupby(obs_noaa_wet.index.year).mean().mean().sort_index()
    obs_wet_avg = pd.concat([eccc_obs_wet_avg,bch_obs_wet_avg,noaa_obs_wet_avg])
    obs_wet_avg.index = obs_wet_avg.index.astype(str)
    obs_wet_avg = obs_wet_avg.sort_index()
    
    eccc_obs_dry_avg = obs_eccc_dry.groupby(obs_eccc_dry.index.year).mean().mean().sort_index()
    bch_obs_dry_avg = obs_bch_dry.groupby(obs_bch_dry.index.year).mean().mean().sort_index()
    noaa_obs_dry_avg = obs_noaa_dry.groupby(obs_noaa_dry.index.year).mean().mean().sort_index()
    obs_dry_avg = pd.concat([eccc_obs_dry_avg,bch_obs_dry_avg,noaa_obs_dry_avg])
    obs_dry_avg.index = obs_dry_avg.index.astype(str)
    obs_dry_avg = obs_dry_avg.sort_index()
    

    wrf_eccc_d02_mam_avg = wrf_d02_eccc_mam.groupby(wrf_d02_eccc_mam.index.year).mean().mean().sort_index()
    wrf_bch_d02_mam_avg = wrf_d02_bch_mam.groupby(wrf_d02_bch_mam.index.year).mean().mean().sort_index()
    wrf_eccc_d02_jja_avg = wrf_d02_eccc_jja.groupby(wrf_d02_eccc_jja.index.year).mean().mean().sort_index()
    wrf_bch_d02_jja_avg = wrf_d02_bch_jja.groupby(wrf_d02_bch_jja.index.year).mean().mean().sort_index()
    wrf_eccc_d02_son_avg = wrf_d02_eccc_son.groupby(wrf_d02_eccc_son.index.year).mean().mean().sort_index()
    wrf_bch_d02_son_avg = wrf_d02_bch_son.groupby(wrf_d02_bch_son.index.year).mean().mean().sort_index()
    wrf_eccc_d02_djf_avg = wrf_d02_eccc_djf.groupby(wrf_d02_eccc_djf.index.year).mean().mean().sort_index()
    wrf_bch_d02_djf_avg = wrf_d02_bch_djf.groupby(wrf_d02_bch_djf.index.year).mean().mean().sort_index()
    wrf_eccc_d02_wet_avg = wrf_d02_eccc_wet.groupby(wrf_d02_eccc_wet.index.year).mean().mean().sort_index()
    wrf_bch_d02_wet_avg = wrf_d02_bch_wet.groupby(wrf_d02_bch_wet.index.year).mean().mean().sort_index()
    wrf_eccc_d02_dry_avg = wrf_d02_eccc_dry.groupby(wrf_d02_eccc_dry.index.year).mean().mean().sort_index()
    wrf_bch_d02_dry_avg = wrf_d02_bch_dry.groupby(wrf_d02_bch_dry.index.year).mean().mean().sort_index()
    
    wrf_eccc_d03_mam_avg = wrf_d03_eccc_mam.groupby(wrf_d03_eccc_mam.index.year).mean().mean().sort_index()
    wrf_bch_d03_mam_avg = wrf_d03_bch_mam.groupby(wrf_d03_bch_mam.index.year).mean().mean().sort_index()
    wrf_eccc_d03_jja_avg = wrf_d03_eccc_jja.groupby(wrf_d03_eccc_jja.index.year).mean().mean().sort_index()
    wrf_bch_d03_jja_avg = wrf_d03_bch_jja.groupby(wrf_d03_bch_jja.index.year).mean().mean().sort_index()
    wrf_eccc_d03_son_avg = wrf_d03_eccc_son.groupby(wrf_d03_eccc_son.index.year).mean().mean().sort_index()
    wrf_bch_d03_son_avg = wrf_d03_bch_son.groupby(wrf_d03_bch_son.index.year).mean().mean().sort_index()
    wrf_eccc_d03_djf_avg = wrf_d03_eccc_djf.groupby(wrf_d03_eccc_djf.index.year).mean().mean().sort_index()
    wrf_bch_d03_djf_avg = wrf_d03_bch_djf.groupby(wrf_d03_bch_djf.index.year).mean().mean().sort_index()
    wrf_eccc_d03_wet_avg = wrf_d03_eccc_wet.groupby(wrf_d03_eccc_wet.index.year).mean().mean().sort_index()
    wrf_bch_d03_wet_avg = wrf_d03_bch_wet.groupby(wrf_d03_bch_wet.index.year).mean().mean().sort_index()
    wrf_eccc_d03_dry_avg = wrf_d03_eccc_dry.groupby(wrf_d03_eccc_dry.index.year).mean().mean().sort_index()
    wrf_bch_d03_dry_avg = wrf_d03_bch_dry.groupby(wrf_d03_bch_dry.index.year).mean().mean().sort_index()
    
    raw_eccc_mam_avg = raw_eccc_mam.groupby(raw_eccc_mam.index.year).mean().mean().sort_index()
    raw_bch_mam_avg = raw_bch_mam.groupby(raw_bch_mam.index.year).mean().mean().sort_index()
    raw_eccc_jja_avg = raw_eccc_jja.groupby(raw_eccc_jja.index.year).mean().mean().sort_index()
    raw_bch_jja_avg = raw_bch_jja.groupby(raw_bch_jja.index.year).mean().mean().sort_index()
    raw_eccc_son_avg = raw_eccc_son.groupby(raw_eccc_son.index.year).mean().mean().sort_index()
    raw_bch_son_avg = raw_bch_son.groupby(raw_bch_son.index.year).mean().mean().sort_index()
    raw_eccc_djf_avg = raw_eccc_djf.groupby(raw_eccc_djf.index.year).mean().mean().sort_index()
    raw_bch_djf_avg = raw_bch_djf.groupby(raw_bch_djf.index.year).mean().mean().sort_index()
    raw_eccc_wet_avg = raw_eccc_wet.groupby(raw_eccc_wet.index.year).mean().mean().sort_index()
    raw_bch_wet_avg = raw_bch_wet.groupby(raw_bch_wet.index.year).mean().mean().sort_index()
    raw_eccc_dry_avg = raw_eccc_dry.groupby(raw_eccc_dry.index.year).mean().mean().sort_index()
    raw_bch_dry_avg = raw_bch_dry.groupby(raw_bch_dry.index.year).mean().mean().sort_index()
    
    rcm_eccc_mam_avg = rcm_eccc_mam.groupby(rcm_eccc_mam.index.year).mean().mean().sort_index()
    rcm_bch_mam_avg =  rcm_bch_mam.groupby(rcm_bch_mam.index.year).mean().mean().sort_index()
    rcm_eccc_jja_avg = rcm_eccc_jja.groupby(rcm_eccc_jja.index.year).mean().mean().sort_index()
    rcm_bch_jja_avg = rcm_bch_jja.groupby(rcm_bch_jja.index.year).mean().mean().sort_index()
    rcm_eccc_son_avg = rcm_eccc_son.groupby(rcm_eccc_son.index.year).mean().mean().sort_index()
    rcm_bch_son_avg = rcm_bch_son.groupby(rcm_bch_son.index.year).mean().mean().sort_index()
    rcm_eccc_djf_avg = rcm_eccc_djf.groupby(rcm_eccc_djf.index.year).mean().mean().sort_index()
    rcm_bch_djf_avg = rcm_bch_djf.groupby(rcm_bch_djf.index.year).mean().mean().sort_index()
    rcm_eccc_wet_avg = rcm_eccc_wet.groupby(rcm_eccc_wet.index.year).mean().mean().sort_index()
    rcm_bch_wet_avg = rcm_bch_wet.groupby(rcm_bch_wet.index.year).mean().mean().sort_index()
    rcm_eccc_dry_avg = rcm_eccc_dry.groupby(rcm_eccc_dry.index.year).mean().mean().sort_index()
    rcm_bch_dry_avg = rcm_bch_dry.groupby(rcm_bch_dry.index.year).mean().mean().sort_index()


    wrf_noaa_d02_mam_avg = wrf_d02_noaa_mam.groupby(wrf_d02_noaa_mam.index.year).mean().mean().sort_index()
    wrf_noaa_d02_jja_avg = wrf_d02_noaa_jja.groupby(wrf_d02_noaa_jja.index.year).mean().mean().sort_index()
    wrf_noaa_d02_son_avg = wrf_d02_noaa_son.groupby(wrf_d02_noaa_son.index.year).mean().mean().sort_index()
    wrf_noaa_d02_djf_avg = wrf_d02_noaa_djf.groupby(wrf_d02_noaa_djf.index.year).mean().mean().sort_index()
    wrf_noaa_d02_wet_avg = wrf_d02_noaa_wet.groupby(wrf_d02_noaa_wet.index.year).mean().mean().sort_index()
    wrf_noaa_d02_dry_avg = wrf_d02_noaa_dry.groupby(wrf_d02_noaa_dry.index.year).mean().mean().sort_index()
    
    wrf_noaa_d03_mam_avg = wrf_d03_noaa_mam.groupby(wrf_d03_noaa_mam.index.year).mean().mean().sort_index()
    wrf_noaa_d03_jja_avg = wrf_d03_noaa_jja.groupby(wrf_d03_noaa_jja.index.year).mean().mean().sort_index()
    wrf_noaa_d03_son_avg = wrf_d03_noaa_son.groupby(wrf_d03_noaa_son.index.year).mean().mean().sort_index()
    wrf_noaa_d03_djf_avg = wrf_d03_noaa_djf.groupby(wrf_d03_noaa_djf.index.year).mean().mean().sort_index()
    wrf_noaa_d03_wet_avg = wrf_d03_noaa_wet.groupby(wrf_d03_noaa_wet.index.year).mean().mean().sort_index()
    wrf_noaa_d03_dry_avg = wrf_d03_noaa_dry.groupby(wrf_d03_noaa_dry.index.year).mean().mean().sort_index()
    
    raw_noaa_mam_avg = raw_noaa_mam.groupby(raw_noaa_mam.index.year).mean().mean().sort_index()
    raw_noaa_jja_avg = raw_noaa_jja.groupby(raw_noaa_jja.index.year).mean().mean().sort_index()
    raw_noaa_son_avg = raw_noaa_son.groupby(raw_noaa_son.index.year).mean().mean().sort_index()
    raw_noaa_djf_avg = raw_noaa_djf.groupby(raw_noaa_djf.index.year).mean().mean().sort_index()
    raw_noaa_wet_avg = raw_noaa_wet.groupby(raw_noaa_wet.index.year).mean().mean().sort_index()
    raw_noaa_dry_avg = raw_noaa_dry.groupby(raw_noaa_dry.index.year).mean().mean().sort_index()
    
    rcm_noaa_mam_avg = rcm_noaa_mam.groupby(rcm_noaa_mam.index.year).mean().mean().sort_index()
    rcm_noaa_jja_avg = rcm_noaa_jja.groupby(rcm_noaa_jja.index.year).mean().mean().sort_index()
    rcm_noaa_son_avg = rcm_noaa_son.groupby(rcm_noaa_son.index.year).mean().mean().sort_index()
    rcm_noaa_djf_avg = rcm_noaa_djf.groupby(rcm_noaa_djf.index.year).mean().mean().sort_index()
    rcm_noaa_wet_avg = rcm_noaa_wet.groupby(rcm_noaa_wet.index.year).mean().mean().sort_index()
    rcm_noaa_dry_avg = rcm_noaa_dry.groupby(rcm_noaa_dry.index.year).mean().mean().sort_index()



wrf_d02_mam_avg = pd.concat([wrf_eccc_d02_mam_avg,wrf_bch_d02_mam_avg,wrf_noaa_d02_mam_avg])
wrf_d02_mam_avg.index = wrf_d02_mam_avg.index.astype(str)
wrf_d02_mam_avg = wrf_d02_mam_avg.sort_index()

wrf_d02_jja_avg = pd.concat([wrf_eccc_d02_jja_avg,wrf_bch_d02_jja_avg,wrf_noaa_d02_jja_avg])
wrf_d02_jja_avg.index = wrf_d02_jja_avg.index.astype(str)
wrf_d02_jja_avg = wrf_d02_jja_avg.sort_index()

wrf_d02_son_avg = pd.concat([wrf_eccc_d02_son_avg,wrf_bch_d02_son_avg,wrf_noaa_d02_son_avg])
wrf_d02_son_avg.index = wrf_d02_son_avg.index.astype(str)
wrf_d02_son_avg = wrf_d02_son_avg.sort_index()

wrf_d02_djf_avg = pd.concat([wrf_eccc_d02_djf_avg,wrf_bch_d02_djf_avg,wrf_noaa_d02_djf_avg])
wrf_d02_djf_avg.index = wrf_d02_djf_avg.index.astype(str)
wrf_d02_djf_avg = wrf_d02_djf_avg.sort_index()

wrf_d03_mam_avg = pd.concat([wrf_eccc_d03_mam_avg,wrf_bch_d03_mam_avg,wrf_noaa_d03_mam_avg])
wrf_d03_mam_avg.index = wrf_d03_mam_avg.index.astype(str)
wrf_d03_mam_avg = wrf_d03_mam_avg.sort_index()

wrf_d03_jja_avg = pd.concat([wrf_eccc_d03_jja_avg,wrf_bch_d03_jja_avg,wrf_noaa_d03_jja_avg])
wrf_d03_jja_avg.index = wrf_d03_jja_avg.index.astype(str)
wrf_d03_jja_avg = wrf_d03_jja_avg.sort_index()

wrf_d03_son_avg = pd.concat([wrf_eccc_d03_son_avg,wrf_bch_d03_son_avg,wrf_noaa_d03_son_avg])
wrf_d03_son_avg.index = wrf_d03_son_avg.index.astype(str)
wrf_d03_son_avg = wrf_d03_son_avg.sort_index()

wrf_d03_djf_avg = pd.concat([wrf_eccc_d03_djf_avg,wrf_bch_d03_djf_avg,wrf_noaa_d03_djf_avg])
wrf_d03_djf_avg.index = wrf_d03_djf_avg.index.astype(str)
wrf_d03_djf_avg = wrf_d03_djf_avg.sort_index()

raw_mam_avg = pd.concat([raw_eccc_mam_avg,raw_bch_mam_avg,raw_noaa_mam_avg])
raw_mam_avg.index = raw_mam_avg.index.astype(str)
raw_mam_avg = raw_mam_avg.sort_index()

raw_jja_avg = pd.concat([raw_eccc_jja_avg,raw_bch_jja_avg,raw_noaa_jja_avg])
raw_jja_avg.index = raw_jja_avg.index.astype(str)
raw_jja_avg = raw_jja_avg.sort_index()

raw_son_avg = pd.concat([raw_eccc_son_avg,raw_bch_son_avg,raw_noaa_son_avg])
raw_son_avg.index = raw_son_avg.index.astype(str)
raw_son_avg = raw_son_avg.sort_index()

raw_djf_avg = pd.concat([raw_eccc_djf_avg,raw_bch_djf_avg,raw_noaa_djf_avg])
raw_djf_avg.index = raw_djf_avg.index.astype(str)
raw_djf_avg = raw_djf_avg.sort_index()

rcm_mam_avg = pd.concat([rcm_eccc_mam_avg,rcm_bch_mam_avg,rcm_noaa_mam_avg])
rcm_mam_avg.index = rcm_mam_avg.index.astype(str)
rcm_mam_avg = rcm_mam_avg.sort_index()

rcm_jja_avg = pd.concat([rcm_eccc_jja_avg,rcm_bch_jja_avg,rcm_noaa_jja_avg])
rcm_jja_avg.index = rcm_jja_avg.index.astype(str)
rcm_jja_avg = rcm_jja_avg.sort_index()

rcm_son_avg = pd.concat([rcm_eccc_son_avg,rcm_bch_son_avg,rcm_noaa_son_avg])
rcm_son_avg.index = rcm_son_avg.index.astype(str)
rcm_son_avg = rcm_son_avg.sort_index()

rcm_djf_avg = pd.concat([rcm_eccc_djf_avg,rcm_bch_djf_avg,rcm_noaa_djf_avg])
rcm_djf_avg.index = rcm_djf_avg.index.astype(str)
rcm_djf_avg = rcm_djf_avg.sort_index()


wrf_d02_wet_avg = pd.concat([wrf_eccc_d02_wet_avg,wrf_bch_d02_wet_avg,wrf_noaa_d02_wet_avg])
wrf_d02_wet_avg.index = wrf_d02_wet_avg.index.astype(str)
wrf_d02_wet_avg = wrf_d02_wet_avg.sort_index()

wrf_d03_wet_avg = pd.concat([wrf_eccc_d03_wet_avg,wrf_bch_d03_wet_avg,wrf_noaa_d03_wet_avg])
wrf_d03_wet_avg.index = wrf_d03_wet_avg.index.astype(str)
wrf_d03_wet_avg = wrf_d03_wet_avg.sort_index()

raw_wet_avg = pd.concat([raw_eccc_wet_avg,raw_bch_wet_avg,raw_noaa_wet_avg])
raw_wet_avg.index = raw_wet_avg.index.astype(str)
raw_wet_avg = raw_wet_avg.sort_index()

rcm_wet_avg = pd.concat([rcm_eccc_wet_avg,rcm_bch_wet_avg,rcm_noaa_wet_avg])
rcm_wet_avg.index = rcm_wet_avg.index.astype(str)
rcm_wet_avg = rcm_wet_avg.sort_index()

wrf_d02_dry_avg = pd.concat([wrf_eccc_d02_dry_avg,wrf_bch_d02_dry_avg,wrf_noaa_d02_dry_avg])
wrf_d02_dry_avg.index = wrf_d02_dry_avg.index.astype(str)
wrf_d02_dry_avg = wrf_d02_dry_avg.sort_index()

wrf_d03_dry_avg = pd.concat([wrf_eccc_d03_dry_avg,wrf_bch_d03_dry_avg,wrf_noaa_d03_dry_avg])
wrf_d03_dry_avg.index = wrf_d03_dry_avg.index.astype(str)
wrf_d03_dry_avg = wrf_d03_dry_avg.sort_index()

raw_dry_avg = pd.concat([raw_eccc_dry_avg,raw_bch_dry_avg,raw_noaa_dry_avg])
raw_dry_avg.index = raw_dry_avg.index.astype(str)
raw_dry_avg = raw_dry_avg.sort_index()

rcm_dry_avg = pd.concat([rcm_eccc_dry_avg,rcm_bch_dry_avg,rcm_noaa_dry_avg])
rcm_dry_avg.index = rcm_dry_avg.index.astype(str)
rcm_dry_avg = rcm_dry_avg.sort_index()

#%%
obs_eccc_mam = eccc_obs_m.copy()
obs_eccc_jja = eccc_obs_m.copy()
obs_eccc_son = eccc_obs_m.copy()
obs_eccc_djf = eccc_obs_m.copy()
obs_eccc_wet = eccc_obs_m.copy()
obs_eccc_dry = eccc_obs_m.copy()

obs_noaa_mam = noaa_obs_m.copy()
obs_noaa_jja = noaa_obs_m.copy()
obs_noaa_son = noaa_obs_m.copy()
obs_noaa_djf = noaa_obs_m.copy()
obs_noaa_wet = noaa_obs_m.copy()
obs_noaa_dry = noaa_obs_m.copy()



for i in [1,2,6,7,8,9,10,11,12]:
    obs_eccc_mam = obs_eccc_mam[obs_eccc_mam.index.month != i]
    obs_noaa_mam = obs_noaa_mam[obs_noaa_mam.index.month != i]


for i in [1,2,3,4,5,9,10,11,12]:
    obs_eccc_jja = obs_eccc_jja[obs_eccc_jja.index.month != i]
    obs_noaa_jja = obs_noaa_jja[obs_noaa_jja.index.month != i]


for i in [1,2,3,4,5,6,7,8,12]:
    obs_eccc_son = obs_eccc_son[obs_eccc_son.index.month != i]
    obs_noaa_son = obs_noaa_son[obs_noaa_son.index.month != i]

    
for i in [3,4,5,6,7,8,9,10,11]:
    obs_eccc_djf = obs_eccc_djf[obs_eccc_djf.index.month != i]
    obs_noaa_djf = obs_noaa_djf[obs_noaa_djf.index.month != i]


for i in [4,5,6,7,8,9]:
    obs_eccc_wet = obs_eccc_wet[obs_eccc_wet.index.month != i]
    obs_noaa_wet = obs_noaa_wet[obs_noaa_wet.index.month != i]

    
for i in [1,2,3,10,11,12]:
    obs_eccc_dry = obs_eccc_dry[obs_eccc_dry.index.month != i]
    obs_noaa_dry = obs_noaa_dry[obs_noaa_dry.index.month != i]

    
if variable == "wind":
    eccc_obs_mam_avg = obs_eccc_mam.groupby(obs_eccc_mam.index.year).mean().mean().sort_index()
    noaa_obs_mam_avg = obs_noaa_mam.groupby(obs_noaa_mam.index.year).mean().mean().sort_index()
    obs_mam_avg = pd.concat([eccc_obs_mam_avg,noaa_obs_mam_avg])
    obs_mam_avg.index = obs_mam_avg.index.astype(str)
    obs_mam_avg = obs_mam_avg.sort_index()
    
    eccc_obs_jja_avg = obs_eccc_jja.groupby(obs_eccc_jja.index.year).mean().mean().sort_index()
    noaa_obs_jja_avg = obs_noaa_jja.groupby(obs_noaa_jja.index.year).mean().mean().sort_index()
    obs_jja_avg = pd.concat([eccc_obs_jja_avg,noaa_obs_jja_avg])
    obs_jja_avg.index = obs_jja_avg.index.astype(str)
    obs_jja_avg = obs_jja_avg.sort_index()
    
    eccc_obs_son_avg = obs_eccc_son.groupby(obs_eccc_son.index.year).mean().mean().sort_index()
    noaa_obs_son_avg = obs_noaa_son.groupby(obs_noaa_son.index.year).mean().mean().sort_index()
    obs_son_avg = pd.concat([eccc_obs_son_avg,noaa_obs_son_avg])
    obs_son_avg.index = obs_son_avg.index.astype(str)
    obs_son_avg = obs_son_avg.sort_index()
    
    eccc_obs_djf_avg = obs_eccc_djf.groupby(obs_eccc_djf.index.year).mean().mean().sort_index()
    noaa_obs_djf_avg = obs_noaa_djf.groupby(obs_noaa_djf.index.year).mean().mean().sort_index()
    obs_djf_avg = pd.concat([eccc_obs_djf_avg,noaa_obs_djf_avg])
    obs_djf_avg.index = obs_djf_avg.index.astype(str)
    obs_djf_avg = obs_djf_avg.sort_index()
    
    eccc_obs_wet_avg = obs_eccc_wet.groupby(obs_eccc_wet.index.year).mean().mean().sort_index()
    noaa_obs_wet_avg = obs_noaa_wet.groupby(obs_noaa_wet.index.year).mean().mean().sort_index()
    obs_wet_avg = pd.concat([eccc_obs_wet_avg,noaa_obs_wet_avg])
    obs_wet_avg.index = obs_wet_avg.index.astype(str)
    obs_wet_avg = obs_wet_avg.sort_index()
    
    eccc_obs_dry_avg = obs_eccc_dry.groupby(obs_eccc_dry.index.year).mean().mean().sort_index()
    noaa_obs_dry_avg = obs_noaa_dry.groupby(obs_noaa_dry.index.year).mean().mean().sort_index()
    obs_dry_avg = pd.concat([eccc_obs_dry_avg,noaa_obs_dry_avg])
    obs_dry_avg.index = obs_dry_avg.index.astype(str)
    obs_dry_avg = obs_dry_avg.sort_index()
 
    
wrf_d02_eccc_mam = wrf_d02_eccc_m.copy()
wrf_d03_eccc_mam = wrf_d03_eccc_m.copy()
raw_eccc_mam = raw_eccc_m.copy()
rcm_eccc_mam = rcm_eccc_m.copy()
wrf_d02_noaa_mam = wrf_d02_noaa_m.copy()
wrf_d03_noaa_mam = wrf_d03_noaa_m.copy()
raw_noaa_mam = raw_noaa_m.copy()
rcm_noaa_mam = rcm_noaa_m.copy()

for i in [1,2,6,7,8,9,10,11,12]:
    wrf_d02_eccc_mam = wrf_d02_eccc_mam[wrf_d02_eccc_mam.index.month != i]
    wrf_d03_eccc_mam = wrf_d03_eccc_mam[wrf_d03_eccc_mam.index.month != i]
    raw_eccc_mam = raw_eccc_mam[raw_eccc_mam.index.month != i]
    rcm_eccc_mam = rcm_eccc_mam[rcm_eccc_mam.index.month != i]
    wrf_d02_noaa_mam = wrf_d02_noaa_mam[wrf_d02_noaa_mam.index.month != i]
    wrf_d03_noaa_mam = wrf_d03_noaa_mam[wrf_d03_noaa_mam.index.month != i]
    raw_noaa_mam = raw_noaa_mam[raw_noaa_mam.index.month != i]
    rcm_noaa_mam = rcm_noaa_mam[rcm_noaa_mam.index.month != i]
    
wrf_d02_eccc_jja = wrf_d02_eccc_m.copy()
wrf_d03_eccc_jja = wrf_d03_eccc_m.copy()
raw_eccc_jja = raw_eccc_m.copy()
rcm_eccc_jja = rcm_eccc_m.copy()
wrf_d02_noaa_jja = wrf_d02_noaa_m.copy()
wrf_d03_noaa_jja = wrf_d03_noaa_m.copy()
raw_noaa_jja = raw_noaa_m.copy()
rcm_noaa_jja = rcm_noaa_m.copy()

for i in [1,2,3,4,5,9,10,11,12]:
    wrf_d02_eccc_jja = wrf_d02_eccc_jja[wrf_d02_eccc_jja.index.month != i]
    wrf_d03_eccc_jja = wrf_d03_eccc_jja[wrf_d03_eccc_jja.index.month != i]
    raw_eccc_jja = raw_eccc_jja[raw_eccc_jja.index.month != i]   
    rcm_eccc_jja = rcm_eccc_jja[rcm_eccc_jja.index.month != i]
    wrf_d02_noaa_jja = wrf_d02_noaa_jja[wrf_d02_noaa_jja.index.month != i]
    wrf_d03_noaa_jja = wrf_d03_noaa_jja[wrf_d03_noaa_jja.index.month != i]
    raw_noaa_jja = raw_noaa_jja[raw_noaa_jja.index.month != i]   
    rcm_noaa_jja = rcm_noaa_jja[rcm_noaa_jja.index.month != i]
    
wrf_d02_eccc_son = wrf_d02_eccc_m.copy()
wrf_d03_eccc_son = wrf_d03_eccc_m.copy()
raw_eccc_son = raw_eccc_m.copy()
rcm_eccc_son = rcm_eccc_m.copy()
wrf_d02_noaa_son = wrf_d02_noaa_m.copy()
wrf_d03_noaa_son = wrf_d03_noaa_m.copy()
raw_noaa_son = raw_noaa_m.copy()
rcm_noaa_son = rcm_noaa_m.copy()

for i in [1,2,3,4,5,6,7,8,12]:
    wrf_d02_eccc_son = wrf_d02_eccc_son[wrf_d02_eccc_son.index.month != i]
    wrf_d03_eccc_son = wrf_d03_eccc_son[wrf_d03_eccc_son.index.month != i]
    raw_eccc_son = raw_eccc_son[raw_eccc_son.index.month != i]
    rcm_eccc_son = rcm_eccc_son[rcm_eccc_son.index.month != i]
    wrf_d02_noaa_son = wrf_d02_noaa_son[wrf_d02_noaa_son.index.month != i]
    wrf_d03_noaa_son = wrf_d03_noaa_son[wrf_d03_noaa_son.index.month != i]
    raw_noaa_son = raw_noaa_son[raw_noaa_son.index.month != i]
    rcm_noaa_son = rcm_noaa_son[rcm_noaa_son.index.month != i]
 
    
wrf_d02_eccc_djf = wrf_d02_eccc_m.copy()
wrf_d03_eccc_djf = wrf_d03_eccc_m.copy()
raw_eccc_djf = raw_eccc_m.copy()
rcm_eccc_djf = rcm_eccc_m.copy()
wrf_d02_noaa_djf = wrf_d02_noaa_m.copy()
wrf_d03_noaa_djf = wrf_d03_noaa_m.copy()
raw_noaa_djf = raw_noaa_m.copy()
rcm_noaa_djf = rcm_noaa_m.copy()

for i in [3,4,5,6,7,8,9,10,11]:
    wrf_d02_eccc_djf = wrf_d02_eccc_djf[wrf_d02_eccc_djf.index.month != i]
    wrf_d03_eccc_djf = wrf_d03_eccc_djf[wrf_d03_eccc_djf.index.month != i]
    raw_eccc_djf = raw_eccc_djf[raw_eccc_djf.index.month != i]  
    rcm_eccc_djf = rcm_eccc_djf[rcm_eccc_djf.index.month != i]
    wrf_d02_noaa_djf = wrf_d02_noaa_djf[wrf_d02_noaa_djf.index.month != i]
    wrf_d03_noaa_djf = wrf_d03_noaa_djf[wrf_d03_noaa_djf.index.month != i]
    raw_noaa_djf = raw_noaa_djf[raw_noaa_djf.index.month != i]  
    rcm_noaa_djf = rcm_noaa_djf[rcm_noaa_djf.index.month != i]


wrf_d02_eccc_wet = wrf_d02_eccc_m.copy()
wrf_d03_eccc_wet = wrf_d03_eccc_m.copy()
raw_eccc_wet = raw_eccc_m.copy()
rcm_eccc_wet = rcm_eccc_m.copy()
wrf_d02_noaa_wet = wrf_d02_noaa_m.copy()
wrf_d03_noaa_wet = wrf_d03_noaa_m.copy()
raw_noaa_wet = raw_noaa_m.copy()
rcm_noaa_wet = rcm_noaa_m.copy()

for i in [4,5,6,7,8,9]:
    wrf_d02_eccc_wet = wrf_d02_eccc_wet[wrf_d02_eccc_wet.index.month != i]
    wrf_d03_eccc_wet = wrf_d03_eccc_wet[wrf_d03_eccc_wet.index.month != i]
    raw_eccc_wet = raw_eccc_wet[raw_eccc_wet.index.month != i]  
    rcm_eccc_wet = rcm_eccc_wet[rcm_eccc_wet.index.month != i]
    wrf_d02_noaa_wet = wrf_d02_noaa_wet[wrf_d02_noaa_wet.index.month != i]
    wrf_d03_noaa_wet = wrf_d03_noaa_wet[wrf_d03_noaa_wet.index.month != i]
    raw_noaa_wet = raw_noaa_wet[raw_noaa_wet.index.month != i]  
    rcm_noaa_wet = rcm_noaa_wet[rcm_noaa_wet.index.month != i]

wrf_d02_eccc_dry = wrf_d02_eccc_m.copy()
wrf_d03_eccc_dry = wrf_d03_eccc_m.copy()
raw_eccc_dry = raw_eccc_m.copy()
rcm_eccc_dry = rcm_eccc_m.copy()
wrf_d02_noaa_dry = wrf_d02_noaa_m.copy()
wrf_d03_noaa_dry = wrf_d03_noaa_m.copy()
raw_noaa_dry = raw_noaa_m.copy()
rcm_noaa_dry = rcm_noaa_m.copy()

for i in [1,2,3,10,11,12]:
    wrf_d02_eccc_dry = wrf_d02_eccc_dry[wrf_d02_eccc_dry.index.month != i]
    wrf_d03_eccc_dry = wrf_d03_eccc_dry[wrf_d03_eccc_dry.index.month != i]
    raw_eccc_dry = raw_eccc_dry[raw_eccc_dry.index.month != i]  
    rcm_eccc_dry = rcm_eccc_dry[rcm_eccc_dry.index.month != i]
    wrf_d02_noaa_dry = wrf_d02_noaa_dry[wrf_d02_noaa_dry.index.month != i]
    wrf_d03_noaa_dry = wrf_d03_noaa_dry[wrf_d03_noaa_dry.index.month != i]
    raw_noaa_dry = raw_noaa_dry[raw_noaa_dry.index.month != i]  
    rcm_noaa_dry = rcm_noaa_dry[rcm_noaa_dry.index.month != i]
 
if variable == "wind":

    wrf_eccc_d02_mam_avg = wrf_d02_eccc_mam.groupby(wrf_d02_eccc_mam.index.year).mean().mean().sort_index()
    wrf_eccc_d02_jja_avg = wrf_d02_eccc_jja.groupby(wrf_d02_eccc_jja.index.year).mean().mean().sort_index()
    wrf_eccc_d02_son_avg = wrf_d02_eccc_son.groupby(wrf_d02_eccc_son.index.year).mean().mean().sort_index()
    wrf_eccc_d02_djf_avg = wrf_d02_eccc_djf.groupby(wrf_d02_eccc_djf.index.year).mean().mean().sort_index()
    wrf_eccc_d02_wet_avg = wrf_d02_eccc_wet.groupby(wrf_d02_eccc_wet.index.year).mean().mean().sort_index()
    wrf_eccc_d02_dry_avg = wrf_d02_eccc_dry.groupby(wrf_d02_eccc_dry.index.year).mean().mean().sort_index()
    
    wrf_eccc_d03_mam_avg = wrf_d03_eccc_mam.groupby(wrf_d03_eccc_mam.index.year).mean().mean().sort_index()
    wrf_eccc_d03_jja_avg = wrf_d03_eccc_jja.groupby(wrf_d03_eccc_jja.index.year).mean().mean().sort_index()
    wrf_eccc_d03_son_avg = wrf_d03_eccc_son.groupby(wrf_d03_eccc_son.index.year).mean().mean().sort_index()
    wrf_eccc_d03_djf_avg = wrf_d03_eccc_djf.groupby(wrf_d03_eccc_djf.index.year).mean().mean().sort_index()
    wrf_eccc_d03_wet_avg = wrf_d03_eccc_wet.groupby(wrf_d03_eccc_wet.index.year).mean().mean().sort_index()
    wrf_eccc_d03_dry_avg = wrf_d03_eccc_dry.groupby(wrf_d03_eccc_dry.index.year).mean().mean().sort_index()
    
    raw_eccc_mam_avg = raw_eccc_mam.groupby(raw_eccc_mam.index.year).mean().mean().sort_index()
    raw_eccc_jja_avg = raw_eccc_jja.groupby(raw_eccc_jja.index.year).mean().mean().sort_index()
    raw_eccc_son_avg = raw_eccc_son.groupby(raw_eccc_son.index.year).mean().mean().sort_index()
    raw_eccc_djf_avg = raw_eccc_djf.groupby(raw_eccc_djf.index.year).mean().mean().sort_index()
    raw_eccc_wet_avg = raw_eccc_wet.groupby(raw_eccc_wet.index.year).mean().mean().sort_index()
    raw_eccc_dry_avg = raw_eccc_dry.groupby(raw_eccc_dry.index.year).mean().mean().sort_index()
    
    rcm_eccc_mam_avg = rcm_eccc_mam.groupby(rcm_eccc_mam.index.year).mean().mean().sort_index()
    rcm_eccc_jja_avg = rcm_eccc_jja.groupby(rcm_eccc_jja.index.year).mean().mean().sort_index()
    rcm_eccc_son_avg = rcm_eccc_son.groupby(rcm_eccc_son.index.year).mean().mean().sort_index()
    rcm_eccc_djf_avg = rcm_eccc_djf.groupby(rcm_eccc_djf.index.year).mean().mean().sort_index()
    rcm_eccc_wet_avg = rcm_eccc_wet.groupby(rcm_eccc_wet.index.year).mean().mean().sort_index()
    rcm_eccc_dry_avg = rcm_eccc_dry.groupby(rcm_eccc_dry.index.year).mean().mean().sort_index()

    wrf_noaa_d02_mam_avg = wrf_d02_noaa_mam.groupby(wrf_d02_noaa_mam.index.year).mean().mean().sort_index()
    wrf_noaa_d02_jja_avg = wrf_d02_noaa_jja.groupby(wrf_d02_noaa_jja.index.year).mean().mean().sort_index()
    wrf_noaa_d02_son_avg = wrf_d02_noaa_son.groupby(wrf_d02_noaa_son.index.year).mean().mean().sort_index()
    wrf_noaa_d02_djf_avg = wrf_d02_noaa_djf.groupby(wrf_d02_noaa_djf.index.year).mean().mean().sort_index()
    wrf_noaa_d02_wet_avg = wrf_d02_noaa_wet.groupby(wrf_d02_noaa_wet.index.year).mean().mean().sort_index()
    wrf_noaa_d02_dry_avg = wrf_d02_noaa_dry.groupby(wrf_d02_noaa_dry.index.year).mean().mean().sort_index()
    
    wrf_noaa_d03_mam_avg = wrf_d03_noaa_mam.groupby(wrf_d03_noaa_mam.index.year).mean().mean().sort_index()
    wrf_noaa_d03_jja_avg = wrf_d03_noaa_jja.groupby(wrf_d03_noaa_jja.index.year).mean().mean().sort_index()
    wrf_noaa_d03_son_avg = wrf_d03_noaa_son.groupby(wrf_d03_noaa_son.index.year).mean().mean().sort_index()
    wrf_noaa_d03_djf_avg = wrf_d03_noaa_djf.groupby(wrf_d03_noaa_djf.index.year).mean().mean().sort_index()
    wrf_noaa_d03_wet_avg = wrf_d03_noaa_wet.groupby(wrf_d03_noaa_wet.index.year).mean().mean().sort_index()
    wrf_noaa_d03_dry_avg = wrf_d03_noaa_dry.groupby(wrf_d03_noaa_dry.index.year).mean().mean().sort_index()
    
    raw_noaa_mam_avg = raw_noaa_mam.groupby(raw_noaa_mam.index.year).mean().mean().sort_index()
    raw_noaa_jja_avg = raw_noaa_jja.groupby(raw_noaa_jja.index.year).mean().mean().sort_index()
    raw_noaa_son_avg = raw_noaa_son.groupby(raw_noaa_son.index.year).mean().mean().sort_index()
    raw_noaa_djf_avg = raw_noaa_djf.groupby(raw_noaa_djf.index.year).mean().mean().sort_index()
    raw_noaa_wet_avg = raw_noaa_wet.groupby(raw_noaa_wet.index.year).mean().mean().sort_index()
    raw_noaa_dry_avg = raw_noaa_dry.groupby(raw_noaa_dry.index.year).mean().mean().sort_index()
    
    rcm_noaa_mam_avg = rcm_noaa_mam.groupby(rcm_noaa_mam.index.year).mean().mean().sort_index()
    rcm_noaa_jja_avg = rcm_noaa_jja.groupby(rcm_noaa_jja.index.year).mean().mean().sort_index()
    rcm_noaa_son_avg = rcm_noaa_son.groupby(rcm_noaa_son.index.year).mean().mean().sort_index()
    rcm_noaa_djf_avg = rcm_noaa_djf.groupby(rcm_noaa_djf.index.year).mean().mean().sort_index()
    rcm_noaa_wet_avg = rcm_noaa_wet.groupby(rcm_noaa_wet.index.year).mean().mean().sort_index()
    rcm_noaa_dry_avg = rcm_noaa_dry.groupby(rcm_noaa_dry.index.year).mean().mean().sort_index()


wrf_d02_mam_avg = pd.concat([wrf_eccc_d02_mam_avg,wrf_noaa_d02_mam_avg])
wrf_d02_mam_avg.index = wrf_d02_mam_avg.index.astype(str)
wrf_d02_mam_avg = wrf_d02_mam_avg.sort_index()

wrf_d02_jja_avg = pd.concat([wrf_eccc_d02_jja_avg,wrf_noaa_d02_jja_avg])
wrf_d02_jja_avg.index = wrf_d02_jja_avg.index.astype(str)
wrf_d02_jja_avg = wrf_d02_jja_avg.sort_index()

wrf_d02_son_avg = pd.concat([wrf_eccc_d02_son_avg,wrf_noaa_d02_son_avg])
wrf_d02_son_avg.index = wrf_d02_son_avg.index.astype(str)
wrf_d02_son_avg = wrf_d02_son_avg.sort_index()

wrf_d02_djf_avg = pd.concat([wrf_eccc_d02_djf_avg,wrf_noaa_d02_djf_avg])
wrf_d02_djf_avg.index = wrf_d02_djf_avg.index.astype(str)
wrf_d02_djf_avg = wrf_d02_djf_avg.sort_index()

wrf_d03_mam_avg = pd.concat([wrf_eccc_d03_mam_avg,wrf_noaa_d03_mam_avg])
wrf_d03_mam_avg.index = wrf_d03_mam_avg.index.astype(str)
wrf_d03_mam_avg = wrf_d03_mam_avg.sort_index()

wrf_d03_jja_avg = pd.concat([wrf_eccc_d03_jja_avg,wrf_noaa_d03_jja_avg])
wrf_d03_jja_avg.index = wrf_d03_jja_avg.index.astype(str)
wrf_d03_jja_avg = wrf_d03_jja_avg.sort_index()

wrf_d03_son_avg = pd.concat([wrf_eccc_d03_son_avg,wrf_noaa_d03_son_avg])
wrf_d03_son_avg.index = wrf_d03_son_avg.index.astype(str)
wrf_d03_son_avg = wrf_d03_son_avg.sort_index()

wrf_d03_djf_avg = pd.concat([wrf_eccc_d03_djf_avg,wrf_noaa_d03_djf_avg])
wrf_d03_djf_avg.index = wrf_d03_djf_avg.index.astype(str)
wrf_d03_djf_avg = wrf_d03_djf_avg.sort_index()

raw_mam_avg = pd.concat([raw_eccc_mam_avg,raw_noaa_mam_avg])
raw_mam_avg.index = raw_mam_avg.index.astype(str)
raw_mam_avg = raw_mam_avg.sort_index()

raw_jja_avg = pd.concat([raw_eccc_jja_avg,raw_noaa_jja_avg])
raw_jja_avg.index = raw_jja_avg.index.astype(str)
raw_jja_avg = raw_jja_avg.sort_index()

raw_son_avg = pd.concat([raw_eccc_son_avg,raw_noaa_son_avg])
raw_son_avg.index = raw_son_avg.index.astype(str)
raw_son_avg = raw_son_avg.sort_index()

raw_djf_avg = pd.concat([raw_eccc_djf_avg,raw_noaa_djf_avg])
raw_djf_avg.index = raw_djf_avg.index.astype(str)
raw_djf_avg = raw_djf_avg.sort_index()

rcm_mam_avg = pd.concat([rcm_eccc_mam_avg,rcm_noaa_mam_avg])
rcm_mam_avg.index = rcm_mam_avg.index.astype(str)
rcm_mam_avg = rcm_mam_avg.sort_index()

rcm_jja_avg = pd.concat([rcm_eccc_jja_avg,rcm_noaa_jja_avg])
rcm_jja_avg.index = rcm_jja_avg.index.astype(str)
rcm_jja_avg = rcm_jja_avg.sort_index()

rcm_son_avg = pd.concat([rcm_eccc_son_avg,rcm_noaa_son_avg])
rcm_son_avg.index = rcm_son_avg.index.astype(str)
rcm_son_avg = rcm_son_avg.sort_index()

rcm_djf_avg = pd.concat([rcm_eccc_djf_avg,rcm_noaa_djf_avg])
rcm_djf_avg.index = rcm_djf_avg.index.astype(str)
rcm_djf_avg = rcm_djf_avg.sort_index()


wrf_d02_wet_avg = pd.concat([wrf_eccc_d02_wet_avg,wrf_noaa_d02_wet_avg])
wrf_d02_wet_avg.index = wrf_d02_wet_avg.index.astype(str)
wrf_d02_wet_avg = wrf_d02_wet_avg.sort_index()

wrf_d03_wet_avg = pd.concat([wrf_eccc_d03_wet_avg,wrf_noaa_d03_wet_avg])
wrf_d03_wet_avg.index = wrf_d03_wet_avg.index.astype(str)
wrf_d03_wet_avg = wrf_d03_wet_avg.sort_index()

raw_wet_avg = pd.concat([raw_eccc_wet_avg,raw_noaa_wet_avg])
raw_wet_avg.index = raw_wet_avg.index.astype(str)
raw_wet_avg = raw_wet_avg.sort_index()

rcm_wet_avg = pd.concat([rcm_eccc_wet_avg,rcm_noaa_wet_avg])
rcm_wet_avg.index = rcm_wet_avg.index.astype(str)
rcm_wet_avg = rcm_wet_avg.sort_index()

wrf_d02_dry_avg = pd.concat([wrf_eccc_d02_dry_avg,wrf_noaa_d02_dry_avg])
wrf_d02_dry_avg.index = wrf_d02_dry_avg.index.astype(str)
wrf_d02_dry_avg = wrf_d02_dry_avg.sort_index()

wrf_d03_dry_avg = pd.concat([wrf_eccc_d03_dry_avg,wrf_noaa_d03_dry_avg])
wrf_d03_dry_avg.index = wrf_d03_dry_avg.index.astype(str)
wrf_d03_dry_avg = wrf_d03_dry_avg.sort_index()

raw_dry_avg = pd.concat([raw_eccc_dry_avg,raw_noaa_dry_avg])
raw_dry_avg.index = raw_dry_avg.index.astype(str)
raw_dry_avg = raw_dry_avg.sort_index()

rcm_dry_avg = pd.concat([rcm_eccc_dry_avg,rcm_noaa_dry_avg])
rcm_dry_avg.index = rcm_dry_avg.index.astype(str)
rcm_dry_avg = rcm_dry_avg.sort_index()
#%%

def plot_taylordiagram(obs,wrf_d03,wrf_d02,raw,rcm,szn,figname,unit,max_std):
    slope, intercept, r_d03, p_value, std_err = linregress(obs, wrf_d03)
    slope, intercept, r_d02, p_value, std_err = linregress(obs, wrf_d02)
    slope, intercept, r_raw, p_value, std_err = linregress(obs, raw)
    slope, intercept, r_rcm, p_value, std_err = linregress(obs, rcm)
    
    # Reference std
    stdrefs = np.std(obs)
    
    
    # Sample std,rho: Be sure to check order and that correct numbers are placed!
    samples = [[np.std(wrf_d03), r_d03, "CanESM2-WRF D03"],
               [np.std(wrf_d02), r_d02, "CanESM2-WRF D02"],
               [np.std(raw), r_raw, "CanESM2"],
               [np.std(rcm), r_rcm, "CanRCM4"]]
    
    
    fig = plt.figure(figsize=(5, 5),dpi=200)
    fig.suptitle(szn, size='large')
    
    
    dia = TaylorDiagram(stdrefs, fig=fig,label='Observations',max_std=max_std,unit=unit)
    
    #dia.ax.plot(x95,y95,color='k')
    #dia.ax.plot(x99,y99,color='k')
    
    
    # # Add samples to Taylor diagram
    for i,(stddev,corrcoef,name) in enumerate(samples):
       dia.add_sample(stddev, corrcoef,label=name,
                     marker='.' , ms=15)
                     #mfc=colors[i], mec=colors[i], # Colors
    
     
    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
    dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
    
     
    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # # Can also use special options here:
    # # http://matplotlib.sourceforge.net/users/legend_guide.html
    # 
    fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size='small'), loc='upper right')
    
    fig.tight_layout()
    
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/taylor/' + figname + '.png',bbox_inches='tight')

if variable == 't':
    unit = 'deg C'
    plot_taylordiagram(obs_avg,wrf_d03_avg,wrf_d02_avg,raw_avg,rcm_avg,"Temperature Annual",'taylordiagram_t_ann',unit,6)
    plot_taylordiagram(obs_mam_avg,wrf_d03_mam_avg,wrf_d02_mam_avg,raw_mam_avg,rcm_mam_avg,"Temperature MAM",'taylordiagram_t_mam',unit,6)
    plot_taylordiagram(obs_jja_avg,wrf_d03_jja_avg,wrf_d02_jja_avg,raw_jja_avg,rcm_jja_avg,"Temperature JJA",'taylordiagram_t_jja',unit,6)
    plot_taylordiagram(obs_son_avg,wrf_d03_son_avg,wrf_d02_son_avg,raw_son_avg,rcm_son_avg,"Temperature SON",'taylordiagram_t_son',unit,6)
    plot_taylordiagram(obs_djf_avg,wrf_d03_djf_avg,wrf_d02_djf_avg,raw_djf_avg,rcm_djf_avg,"Temperature DJF",'taylordiagram_t_djf',unit,6)

elif variable == 'pr':   
    unit = 'mm/season'
    plot_taylordiagram(obs_avg,wrf_d03_avg,wrf_d02_avg,raw_avg,rcm_avg,"Precipitation Annual",'taylordiagram_pr_ann','mm/year',1800)
    plot_taylordiagram(obs_mam_avg,wrf_d03_mam_avg,wrf_d02_mam_avg,raw_mam_avg,rcm_mam_avg,"Precipitation MAM",'taylordiagram_pr_mam',unit,600)
    plot_taylordiagram(obs_jja_avg,wrf_d03_jja_avg,wrf_d02_jja_avg,raw_jja_avg,rcm_jja_avg,"Precipitation JJA",'taylordiagram_pr_jja',unit,600)
    plot_taylordiagram(obs_son_avg,wrf_d03_son_avg,wrf_d02_son_avg,raw_son_avg,rcm_son_avg,"Precipitation SON",'taylordiagram_pr_son',unit,600)
    plot_taylordiagram(obs_djf_avg,wrf_d03_djf_avg,wrf_d02_djf_avg,raw_djf_avg,rcm_djf_avg,"Precipitation DJF",'taylordiagram_pr_djf',unit,600)
    plot_taylordiagram(obs_wet_avg,wrf_d03_wet_avg,wrf_d02_wet_avg,raw_wet_avg,rcm_wet_avg,"Precipitation WET (ONDJFM)",'taylordiagram_pr_wet',unit,1500)
    plot_taylordiagram(obs_dry_avg,wrf_d03_dry_avg,wrf_d02_dry_avg,raw_dry_avg,rcm_dry_avg,"Precipitation DRY (AMJJAS)",'taylordiagram_pr_dry',unit,500)

elif variable == 'wind':
    unit = 'm/s'
    plot_taylordiagram(obs_avg,wrf_d03_avg,wrf_d02_avg,raw_avg,rcm_avg,"Wind Speed ANN",'taylordiagram_wind_ann',unit,2)
    plot_taylordiagram(obs_mam_avg,wrf_d03_mam_avg,wrf_d02_mam_avg,raw_mam_avg,rcm_mam_avg,"Wind Speed  MAM",'taylordiagram_wind_mam',unit,2)
    plot_taylordiagram(obs_jja_avg,wrf_d03_jja_avg,wrf_d02_jja_avg,raw_jja_avg,rcm_jja_avg,"Wind Speed  JJA",'taylordiagram_wind_jja',unit,2)
    plot_taylordiagram(obs_son_avg,wrf_d03_son_avg,wrf_d02_son_avg,raw_son_avg,rcm_son_avg,"Wind Speed  SON",'taylordiagram_wind_son',unit,2)
    plot_taylordiagram(obs_djf_avg,wrf_d03_djf_avg,wrf_d02_djf_avg,raw_djf_avg,rcm_djf_avg,"Wind Speed  DJF",'taylordiagram_wind_djf',unit,2)
