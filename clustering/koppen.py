import pandas as pd
import warnings
import sys
import numpy as np
from netCDF4 import Dataset
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_noaa_obs,plot_all_d03_flexdomain,get_wrf,get_canesm2,get_canrcm4,get_pcic,plot_all_d03,plot_zoomed_in
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib
import matplotlib as mpl


run = 'historical' #historical rcp45 or rcp85
output_freq = "monthly" #yearly monthly or daily
#%%

eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'
noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations.csv'

df = pd.read_csv(eccc_daily_stations,header=None)
eccc_station_IDs = list(df.iloc[:,4])
eccc_station_names = list(df.iloc[:,1])
eccc_lats = (df.iloc[:,7])
eccc_lons = (df.iloc[:,8])
eccc_elev = (df.iloc[:,11])
eccc_elev.index = eccc_station_IDs

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])
bch_lats = (df["Y"])
bch_lons = (df["X"])
bch_elev = (df["ELEV"])
bch_elev.index = bch_station_IDs

df = pd.read_csv(noaa_daily_stations)
noaa_station_IDs = list(df["STATION"])
noaa_station_names = list(df["NAME"])
noaa_lats = df.iloc[:,2]
noaa_lons = df.iloc[:,3]
noaa_elev = (df["ELEVATION"])
noaa_elev.index = noaa_station_IDs

#%%
eccc_lats.index = eccc_station_IDs
bch_lats.index = bch_station_IDs
noaa_lats.index = noaa_station_IDs

eccc_lats = eccc_lats.rename('lats')
bch_lats = bch_lats.rename('lats')
noaa_lats = noaa_lats.rename('lats')

obs_lats = pd.concat([eccc_lats,bch_lats,noaa_lats],axis=0)
obs_lats = pd.DataFrame(obs_lats)

eccc_lons.index = eccc_station_IDs
bch_lons.index = bch_station_IDs
noaa_lons.index = noaa_station_IDs

eccc_lons = eccc_lons.rename('lons')
bch_lons = bch_lons.rename('lons')
noaa_lons = noaa_lons.rename('lons')

obs_lons = pd.concat([eccc_lons,bch_lons,noaa_lons],axis=0)
obs_lons = pd.DataFrame(obs_lons)


#%%
stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/'

eccc_obs_pr = get_eccc_obs(output_freq,eccc_station_IDs,stations_dir,"pr")
bch_obs_pr = get_bch_obs(output_freq,bch_station_IDs,stations_dir,"pr")
noaa_obs_pr = get_noaa_obs(output_freq,noaa_station_IDs,stations_dir,"pr")

eccc_obs_pr_avg = eccc_obs_pr.groupby(eccc_obs_pr.index.month).mean().sort_index()
bch_obs_pr_avg = bch_obs_pr.groupby(bch_obs_pr.index.month).mean().sort_index()
noaa_obs_pr_avg = noaa_obs_pr.groupby(noaa_obs_pr.index.month).mean().sort_index()

eccc_obs_tmax = get_eccc_obs(output_freq,eccc_station_IDs,stations_dir,"tmax")
bch_obs_tmax = get_bch_obs(output_freq,bch_station_IDs,stations_dir,"tmax")
noaa_obs_tmax = get_noaa_obs(output_freq,noaa_station_IDs,stations_dir,"tmax")

eccc_obs_tmax_avg = eccc_obs_tmax.groupby(eccc_obs_tmax.index.month).mean().sort_index()
bch_obs_tmax_avg = bch_obs_tmax.groupby(bch_obs_tmax.index.month).mean().sort_index()
noaa_obs_tmax_avg = noaa_obs_tmax.groupby(noaa_obs_tmax.index.month).mean().sort_index()

eccc_obs_tmin = get_eccc_obs(output_freq,eccc_station_IDs,stations_dir,"tmin")
bch_obs_tmin = get_bch_obs(output_freq,bch_station_IDs,stations_dir,"tmin")
noaa_obs_tmin = get_noaa_obs(output_freq,noaa_station_IDs,stations_dir,"tmin")

eccc_obs_tmin_avg = eccc_obs_tmin.groupby(eccc_obs_tmin.index.month).mean().sort_index()
bch_obs_tmin_avg = bch_obs_tmin.groupby(bch_obs_tmin.index.month).mean().sort_index()
noaa_obs_tmin_avg = noaa_obs_tmin.groupby(noaa_obs_tmin.index.month).mean().sort_index()

obs_pr = pd.concat([eccc_obs_pr_avg,bch_obs_pr_avg,noaa_obs_pr_avg],axis=1)
obs_tmax = pd.concat([eccc_obs_tmax_avg,bch_obs_tmax_avg,noaa_obs_tmax_avg],axis=1)
obs_tmin = pd.concat([eccc_obs_tmin_avg,bch_obs_tmin_avg,noaa_obs_tmin_avg],axis=1)

columns_to_remove = set(obs_lats.index) - set(obs_tmax.columns)
obs_pr = obs_pr.drop(columns=columns_to_remove)
obs_lats = obs_lats.drop(index=columns_to_remove)
obs_lons = obs_lons.drop(index=columns_to_remove)

#%%

WRF_files = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/historical/'

eccc_wrf_d03_pr = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d03", 'historical',"pr",WRF_files,1986)
bch_wrf_d03_pr = get_wrf(output_freq,"BCH",bch_station_IDs,"d03", 'historical',"pr",WRF_files,1986)
#noaa_wrf_d03_pr = get_noaa_wrf_d03(output_freq,noaa_station_IDs,WRF_files,"pr")

eccc_wrf_d03_pr_avg = eccc_wrf_d03_pr.groupby(eccc_wrf_d03_pr.index.month).mean().sort_index()
bch_wrf_d03_pr_avg = bch_wrf_d03_pr.groupby(bch_wrf_d03_pr.index.month).mean().sort_index()
#noaa_wrf_d03_pr_avg = noaa_wrf_d03_pr.groupby(noaa_wrf_d03_pr.index.month).mean().sort_index()

eccc_wrf_d03_tmax = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d03", 'historical',"tmax",WRF_files,1986)
bch_wrf_d03_tmax = get_wrf(output_freq,"BCH",bch_station_IDs,"d03", 'historical',"tmax",WRF_files,1986)
#noaa_wrf_d03_tmax = get_noaa_wrf_d03(output_freq,noaa_station_IDs,WRF_files,"tmax")

eccc_wrf_d03_tmax_avg = eccc_wrf_d03_tmax.groupby(eccc_wrf_d03_tmax.index.month).mean().sort_index()
bch_wrf_d03_tmax_avg = bch_wrf_d03_tmax.groupby(bch_wrf_d03_tmax.index.month).mean().sort_index()
#noaa_wrf_d03_tmax_avg = noaa_wrf_d03_tmax.groupby(noaa_wrf_d03_tmax.index.month).mean().sort_index()

eccc_wrf_d03_tmin = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d03", 'historical',"tmin",WRF_files,1986)
bch_wrf_d03_tmin = get_wrf(output_freq,"BCH",bch_station_IDs,"d03", 'historical',"tmin",WRF_files,1986)
#noaa_wrf_d03_tmin = get_noaa_wrf_d03(output_freq,noaa_station_IDs,WRF_files,"tmin")

eccc_wrf_d03_tmin_avg = eccc_wrf_d03_tmin.groupby(eccc_wrf_d03_tmin.index.month).mean().sort_index()
bch_wrf_d03_tmin_avg = bch_wrf_d03_tmin.groupby(bch_wrf_d03_tmin.index.month).mean().sort_index()
#noaa_wrf_d03_tmin_avg = noaa_wrf_d03_tmin.groupby(noaa_wrf_d03_tmin.index.month).mean().sort_index()

wrf_d03_pr = pd.concat([eccc_wrf_d03_pr_avg,bch_wrf_d03_pr_avg],axis=1)
wrf_d03_tmax = pd.concat([eccc_wrf_d03_tmax_avg,bch_wrf_d03_tmax_avg],axis=1)
wrf_d03_tmin = pd.concat([eccc_wrf_d03_tmin_avg,bch_wrf_d03_tmin_avg],axis=1)

#%%

WRF_files_rcp45 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/rcp45/'

eccc_wrf_d03_pr_rcp45 = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d03", 'rcp45',"pr",WRF_files_rcp45,1986)
bch_wrf_d03_pr_rcp45 = get_wrf(output_freq,"BCH",bch_station_IDs,"d03", 'rcp45',"pr",WRF_files_rcp45,1986)
#noaa_wrf_d03_pr = get_noaa_wrf_d03(output_freq,noaa_station_IDs,WRF_files,"pr")

eccc_wrf_d03_pr_avg_rcp45 = eccc_wrf_d03_pr_rcp45.groupby(eccc_wrf_d03_pr_rcp45.index.month).mean().sort_index()
bch_wrf_d03_pr_avg_rcp45 = bch_wrf_d03_pr_rcp45.groupby(bch_wrf_d03_pr_rcp45.index.month).mean().sort_index()
#noaa_wrf_d03_pr_avg = noaa_wrf_d03_pr.groupby(noaa_wrf_d03_pr.index.month).mean().sort_index()

eccc_wrf_d03_tmax_rcp45 = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d03", 'rcp45',"tmax",WRF_files_rcp45,1986)
bch_wrf_d03_tmax_rcp45 = get_wrf(output_freq,"BCH",bch_station_IDs,"d03", 'rcp45',"tmax",WRF_files_rcp45,1986)
#noaa_wrf_d03_tmax = get_noaa_wrf_d03(output_freq,noaa_station_IDs,WRF_files,"tmax")

eccc_wrf_d03_tmax_avg_rcp45 = eccc_wrf_d03_tmax_rcp45.groupby(eccc_wrf_d03_tmax_rcp45.index.month).mean().sort_index()
bch_wrf_d03_tmax_avg_rcp45 = bch_wrf_d03_tmax_rcp45.groupby(bch_wrf_d03_tmax_rcp45.index.month).mean().sort_index()
#noaa_wrf_d03_tmax_avg = noaa_wrf_d03_tmax.groupby(noaa_wrf_d03_tmax.index.month).mean().sort_index()

eccc_wrf_d03_tmin_rcp45 = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d03", 'rcp45',"tmin",WRF_files_rcp45,1986)
bch_wrf_d03_tmin_rcp45 = get_wrf(output_freq,"BCH",bch_station_IDs,"d03", 'rcp45',"tmin",WRF_files_rcp45,1986)
#noaa_wrf_d03_tmin = get_noaa_wrf_d03(output_freq,noaa_station_IDs,WRF_files,"tmin")

eccc_wrf_d03_tmin_avg_rcp45 = eccc_wrf_d03_tmin_rcp45.groupby(eccc_wrf_d03_tmin_rcp45.index.month).mean().sort_index()
bch_wrf_d03_tmin_avg_rcp45 = bch_wrf_d03_tmin_rcp45.groupby(bch_wrf_d03_tmin_rcp45.index.month).mean().sort_index()
#noaa_wrf_d03_tmin_avg = noaa_wrf_d03_tmin.groupby(noaa_wrf_d03_tmin.index.month).mean().sort_index()

wrf_d03_pr_rcp45 = pd.concat([eccc_wrf_d03_pr_avg_rcp45,bch_wrf_d03_pr_avg_rcp45],axis=1)
wrf_d03_tmax_rcp45 = pd.concat([eccc_wrf_d03_tmax_avg_rcp45,bch_wrf_d03_tmax_avg_rcp45],axis=1)
wrf_d03_tmin_rcp45 = pd.concat([eccc_wrf_d03_tmin_avg_rcp45,bch_wrf_d03_tmin_avg_rcp45],axis=1)

#%%

WRF_files_rcp85 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/rcp85/'

eccc_wrf_d03_pr_rcp85 = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d03", 'rcp85',"pr",WRF_files_rcp85,1986)
bch_wrf_d03_pr_rcp85 = get_wrf(output_freq,"BCH",bch_station_IDs,"d03", 'rcp85',"pr",WRF_files_rcp85,1986)
#noaa_wrf_d03_pr = get_noaa_wrf_d03(output_freq,noaa_station_IDs,WRF_files,"pr")

eccc_wrf_d03_pr_avg_rcp85 = eccc_wrf_d03_pr_rcp85.groupby(eccc_wrf_d03_pr_rcp85.index.month).mean().sort_index()
bch_wrf_d03_pr_avg_rcp85 = bch_wrf_d03_pr_rcp85.groupby(bch_wrf_d03_pr_rcp85.index.month).mean().sort_index()
#noaa_wrf_d03_pr_avg = noaa_wrf_d03_pr.groupby(noaa_wrf_d03_pr.index.month).mean().sort_index()

eccc_wrf_d03_tmax_rcp85 = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d03", 'rcp85',"tmax",WRF_files_rcp85,1986)
bch_wrf_d03_tmax_rcp85 = get_wrf(output_freq,"BCH",bch_station_IDs,"d03", 'rcp85',"tmax",WRF_files_rcp85,1986)
#noaa_wrf_d03_tmax = get_noaa_wrf_d03(output_freq,noaa_station_IDs,WRF_files,"tmax")

eccc_wrf_d03_tmax_avg_rcp85 = eccc_wrf_d03_tmax_rcp85.groupby(eccc_wrf_d03_tmax_rcp85.index.month).mean().sort_index()
bch_wrf_d03_tmax_avg_rcp85 = bch_wrf_d03_tmax_rcp85.groupby(bch_wrf_d03_tmax_rcp85.index.month).mean().sort_index()
#noaa_wrf_d03_tmax_avg = noaa_wrf_d03_tmax.groupby(noaa_wrf_d03_tmax.index.month).mean().sort_index()

eccc_wrf_d03_tmin_rcp85 = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d03", 'rcp85',"tmin",WRF_files_rcp85,1986)
bch_wrf_d03_tmin_rcp85 = get_wrf(output_freq,"BCH",bch_station_IDs,"d03", 'rcp85',"tmin",WRF_files_rcp85,1986)
#noaa_wrf_d03_tmin = get_noaa_wrf_d03(output_freq,noaa_station_IDs,WRF_files,"tmin")

eccc_wrf_d03_tmin_avg_rcp85 = eccc_wrf_d03_tmin_rcp85.groupby(eccc_wrf_d03_tmin_rcp85.index.month).mean().sort_index()
bch_wrf_d03_tmin_avg_rcp85 = bch_wrf_d03_tmin_rcp85.groupby(bch_wrf_d03_tmin_rcp85.index.month).mean().sort_index()
#noaa_wrf_d03_tmin_avg = noaa_wrf_d03_tmin.groupby(noaa_wrf_d03_tmin.index.month).mean().sort_index()

wrf_d03_pr_rcp85 = pd.concat([eccc_wrf_d03_pr_avg_rcp85,bch_wrf_d03_pr_avg_rcp85],axis=1)
wrf_d03_tmax_rcp85 = pd.concat([eccc_wrf_d03_tmax_avg_rcp85,bch_wrf_d03_tmax_avg_rcp85],axis=1)
wrf_d03_tmin_rcp85 = pd.concat([eccc_wrf_d03_tmin_avg_rcp85,bch_wrf_d03_tmin_avg_rcp85],axis=1)
#%%

WRF_files = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/historical/'

eccc_wrf_d02_pr = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d02", 'historical',"pr",WRF_files,1986)
bch_wrf_d02_pr = get_wrf(output_freq,"BCH",bch_station_IDs,"d02", 'historical',"pr",WRF_files,1986)
#noaa_wrf_d02_pr = get_wrf(output_freq,noaa_station_IDs,WRF_files,"pr")

eccc_wrf_d02_pr_avg = eccc_wrf_d02_pr.groupby(eccc_wrf_d02_pr.index.month).mean().sort_index()
bch_wrf_d02_pr_avg = bch_wrf_d02_pr.groupby(bch_wrf_d02_pr.index.month).mean().sort_index()
#noaa_wrf_d02_pr_avg = noaa_wrf_d02_pr.groupby(noaa_wrf_d02_pr.index.month).mean().sort_index()

eccc_wrf_d02_tmax = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d02", 'historical',"tmax",WRF_files,1986)
bch_wrf_d02_tmax = get_wrf(output_freq,"BCH",bch_station_IDs,"d02", 'historical',"tmax",WRF_files,1986)
#noaa_wrf_d02_tmax = get_wrf(output_freq,noaa_station_IDs,WRF_files,"tmax")

eccc_wrf_d02_tmax_avg = eccc_wrf_d02_tmax.groupby(eccc_wrf_d02_tmax.index.month).mean().sort_index()
bch_wrf_d02_tmax_avg = bch_wrf_d02_tmax.groupby(bch_wrf_d02_tmax.index.month).mean().sort_index()
#noaa_wrf_d02_tmax_avg = noaa_wrf_d02_tmax.groupby(noaa_wrf_d02_tmax.index.month).mean().sort_index()

eccc_wrf_d02_tmin = get_wrf(output_freq,"ECCC",eccc_station_IDs,"d02", 'historical',"tmin",WRF_files,1986)
bch_wrf_d02_tmin = get_wrf(output_freq,"BCH",bch_station_IDs,"d02", 'historical',"tmin",WRF_files,1986)
#noaa_wrf_d02_tmin = get_wrf(output_freq,noaa_station_IDs,WRF_files,"tmin")

eccc_wrf_d02_tmin_avg = eccc_wrf_d02_tmin.groupby(eccc_wrf_d02_tmin.index.month).mean().sort_index()
bch_wrf_d02_tmin_avg = bch_wrf_d02_tmin.groupby(bch_wrf_d02_tmin.index.month).mean().sort_index()
#noaa_wrf_d02_tmin_avg = noaa_wrf_d02_tmin.groupby(noaa_wrf_d02_tmin.index.month).mean().sort_index()

wrf_d02_pr = pd.concat([eccc_wrf_d02_pr_avg,bch_wrf_d02_pr_avg],axis=1)
wrf_d02_tmax = pd.concat([eccc_wrf_d02_tmax_avg,bch_wrf_d02_tmax_avg],axis=1)
wrf_d02_tmin = pd.concat([eccc_wrf_d02_tmin_avg,bch_wrf_d02_tmin_avg],axis=1)

#%%

raw_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/' + run + '/'

eccc_raw_pr = get_canesm2(output_freq,"ECCC",eccc_station_IDs, 'historical',"pr",raw_files_dir,1986)
bch_raw_pr = get_canesm2(output_freq,"BCH",bch_station_IDs, 'historical',"pr",raw_files_dir,1986)
#noaa_raw_pr = get_canesm2(output_freq,noaa_station_IDs,WRF_files,"pr")

eccc_raw_pr_avg = eccc_raw_pr.groupby(eccc_raw_pr.index.month).mean().sort_index()
bch_raw_pr_avg = bch_raw_pr.groupby(bch_raw_pr.index.month).mean().sort_index()
#noaa_raw_pr_avg = noaa_raw_pr.groupby(noaa_raw_pr.index.month).mean().sort_index()

eccc_raw_tmax = get_canesm2(output_freq,"ECCC",eccc_station_IDs, 'historical',"tmax",raw_files_dir,1986)
bch_raw_tmax = get_canesm2(output_freq,"BCH",bch_station_IDs, 'historical',"tmax",raw_files_dir,1986)
#noaa_raw_tmax = get_canesm2(output_freq,noaa_station_IDs,WRF_files,"tmax")

eccc_raw_tmax_avg = eccc_raw_tmax.groupby(eccc_raw_tmax.index.month).mean().sort_index()
bch_raw_tmax_avg = bch_raw_tmax.groupby(bch_raw_tmax.index.month).mean().sort_index()
#noaa_raw_tmax_avg = noaa_raw_tmax.groupby(noaa_raw_tmax.index.month).mean().sort_index()

eccc_raw_tmin = get_canesm2(output_freq,"ECCC",eccc_station_IDs, 'historical',"tmin",raw_files_dir,1986)
bch_raw_tmin = get_canesm2(output_freq,"BCH",bch_station_IDs, 'historical',"tmin",raw_files_dir,1986)
#noaa_raw_tmin = get_canesm2(output_freq,noaa_station_IDs,WRF_files,"tmin")

eccc_raw_tmin_avg = eccc_raw_tmin.groupby(eccc_raw_tmin.index.month).mean().sort_index()
bch_raw_tmin_avg = bch_raw_tmin.groupby(bch_raw_tmin.index.month).mean().sort_index()
#noaa_raw_tmin_avg = noaa_raw_tmin.groupby(noaa_raw_tmin.index.month).mean().sort_index()

raw_pr = pd.concat([eccc_raw_pr_avg,bch_raw_pr_avg],axis=1)
raw_tmax = pd.concat([eccc_raw_tmax_avg,bch_raw_tmax_avg],axis=1)
raw_tmin = pd.concat([eccc_raw_tmin_avg,bch_raw_tmin_avg],axis=1)

#%%
rcm_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/' + run + '/'


eccc_rcm_pr = get_canrcm4(output_freq,"ECCC",eccc_station_IDs, 'historical',"pr",rcm_files_dir)
bch_rcm_pr = get_canrcm4(output_freq,"BCH",bch_station_IDs, 'historical',"pr",rcm_files_dir)
#noaa_rcm_pr = get_canrcm4(output_freq,noaa_station_IDs,WRF_files,"pr")

eccc_rcm_pr_avg = eccc_rcm_pr.groupby(eccc_rcm_pr.index.month).mean().sort_index()
bch_rcm_pr_avg = bch_rcm_pr.groupby(bch_rcm_pr.index.month).mean().sort_index()
#noaa_rcm_pr_avg = noaa_rcm_pr.groupby(noaa_rcm_pr.index.month).mean().sort_index()

eccc_rcm_tmax = get_canrcm4(output_freq,"ECCC",eccc_station_IDs, 'historical',"tmax",rcm_files_dir)
bch_rcm_tmax = get_canrcm4(output_freq,"BCH",bch_station_IDs, 'historical',"tmax",rcm_files_dir)
#noaa_rcm_tmax = get_canrcm4(output_freq,noaa_station_IDs,WRF_files,"tmax")

eccc_rcm_tmax_avg = eccc_rcm_tmax.groupby(eccc_rcm_tmax.index.month).mean().sort_index()
bch_rcm_tmax_avg = bch_rcm_tmax.groupby(bch_rcm_tmax.index.month).mean().sort_index()
#noaa_rcm_tmax_avg = noaa_rcm_tmax.groupby(noaa_rcm_tmax.index.month).mean().sort_index()

eccc_rcm_tmin = get_canrcm4(output_freq,"ECCC",eccc_station_IDs, 'historical',"tmin",rcm_files_dir)
bch_rcm_tmin = get_canrcm4(output_freq,"BCH",bch_station_IDs, 'historical',"tmin",rcm_files_dir)
#noaa_rcm_tmin = get_canrcm4(output_freq,noaa_station_IDs,WRF_files,"tmin")

eccc_rcm_tmin_avg = eccc_rcm_tmin.groupby(eccc_rcm_tmin.index.month).mean().sort_index()
bch_rcm_tmin_avg = bch_rcm_tmin.groupby(bch_rcm_tmin.index.month).mean().sort_index()
#noaa_rcm_tmin_avg = noaa_rcm_tmin.groupby(noaa_rcm_tmin.index.month).mean().sort_index()

rcm_pr = pd.concat([eccc_rcm_pr_avg,bch_rcm_pr_avg],axis=1)
rcm_tmax = pd.concat([eccc_rcm_tmax_avg,bch_rcm_tmax_avg],axis=1)
rcm_tmin = pd.concat([eccc_rcm_tmin_avg,bch_rcm_tmin_avg],axis=1)
#%%

def koppen(tmax,tmin,pr):
    
    highs = tmax.to_dict(orient='list')
    highs = {key: np.array(value) for key, value in highs.items()}
    
    lows = tmin.to_dict(orient='list')
    lows = {key: np.array(value) for key, value in lows.items()}
    
    precip = pr.to_dict(orient='list')
    precip = {key: np.array(value) for key, value in precip.items()}
    
    climate = {}
    
    for city in pr.columns:
        avgtemp = (highs[city] + lows[city]) / 2.0
        totalprecip = sum(precip[city])
        climate[city] = ''
    
        # Group A (Tropical)
        if min(avgtemp) >= 18.0:
            # Tropical Rainforest
            if min(precip[city]) >= 60.0:
                climate[city] = 'Af'
                continue
            # Tropical Monsoon
            elif min(precip[city]) < 60.0 and (min(precip[city]) / totalprecip) > 0.04:
                climate[city] = 'Am'
                continue
            else:
                # Tropical Savanna Dry Summer
                if np.where(precip[city]==min(precip[city]))[0][0] >= 6 and np.where(precip[city]==min(precip[city]))[0][0] <= 8:
                    climate[city] = 'As'
                    continue
                # Tropical Savanna Dry Winter
                else:
                    climate[city] = 'Aw'
                    continue
    
        # Group B (Arid and Semiarid)
        aridity = np.mean(avgtemp) * 20.0
        warmprecip = sum(precip[city][3:9])
        coolprecip = sum(precip[city][0:3]) + sum(precip[city][9:12])
        if warmprecip / totalprecip >= 0.70:
            aridity = aridity + 280.0
        elif warmprecip / totalprecip >= 0.30 and warmprecip / totalprecip < 0.70:
            aridity = aridity + 140.0
        else:
            aridity = aridity + 0.0
    
        # Arid Desert (BW)
        if totalprecip / aridity < 0.50:
            # Hot Desert (BWh)
            if np.mean(avgtemp) > 18.0:
                climate[city] = 'BWh'
                continue
            # Cold Desert (BWk)
            else:
                climate[city] = 'BWk'
                continue
    
        if 'A' in climate[city]:
            continue
    
        # Semi-Arid/Steppe (BS)
        elif totalprecip / aridity >= 0.50 and totalprecip / aridity < 1.00:
            # Hot Semi-Arid (BSh)
            if np.mean(avgtemp) > 18.0:
                climate[city] = 'BSh'
                continue
            # Cold Semi-Arid (BSk)
            else:
                climate[city] = 'BSk'
                continue
    
        if 'B' in climate[city]:
            continue
    
        # Group C (Temperate)
        sortavgtemp = avgtemp
        sortavgtemp.sort()
        tempaboveten = np.shape(np.where(avgtemp>10.0))[1]
        coldwarmratio = max(max(precip[city][0:2]),precip[city][11]) / min(precip[city][5:8])
        warmcoldratio = max(precip[city][5:8]) / min(min(precip[city][0:2]),precip[city][11])
        if min(avgtemp) >= 0.0 and min(avgtemp) <= 18.0 and max(avgtemp) >= 10.0:
            # Humid Subtropical (Cfa)
            if min(avgtemp) > 0.0 and max(avgtemp) > 22.0 and tempaboveten >= 4.0:
                climate[city] = 'Cfa'
            # Temperate Oceanic (Cfb)
            elif min(avgtemp) > 0.0 and max(avgtemp) < 22.0 and tempaboveten >= 4.0:
                climate[city] = 'Cfb'
            # Subpolar Oceanic (Cfc)
            elif min(avgtemp) > 0.0 and tempaboveten >= 1 and tempaboveten <= 3:
                climate[city] = 'Cfc'
    
            # Monsoon-influenced humid subtropical (Cwa)
            if min(avgtemp) > 0.0 and max(avgtemp) > 22.0 and tempaboveten >= 4 and warmcoldratio > 10.0:
                climate[city] = 'Cwa'
            # Subtropical Highland/Temperate Oceanic with Dry Winter (Cwb)
            elif min(avgtemp) > 0.0 and max(avgtemp) < 22.0 and tempaboveten >= 4 and warmcoldratio > 10.0:
                climate[city] = 'Cwb'
            # Cold Subtropical Highland/Subpolar Oceanic with Dry Winter (Cwc)
            elif min(avgtemp) > 0.0 and tempaboveten >= 1 and tempaboveten <= 3 and warmcoldratio > 10.0:
                climate[city] = 'Cwc'
    
            # Hot summer Mediterranean (Csa)
            if min(avgtemp) > 0.0 and max(avgtemp) > 22.0 and tempaboveten >= 4 and \
                coldwarmratio >= 3.0 and min(precip[city][5:8]) < 30.0:
                climate[city] = 'Csa'
            # Warm summer Mediterranean (Csb)
            elif min(avgtemp) > 0.0 and max(avgtemp) < 22.0 and tempaboveten >= 4 and \
                coldwarmratio >= 3.0 and min(precip[city][5:8]) < 30.0:
                climate[city] = 'Csb'
            # Cool summer Mediterranean (Csc)
            elif min(avgtemp) > 0.0 and tempaboveten >= 1 and tempaboveten <= 3 and \
                coldwarmratio >= 3.0 and min(precip[city][5:8]) < 30.0:
                climate[city] = 'Csc'
    
            if 'C' in climate[city]:
                continue
    
        # Group D (Continental)
        if min(avgtemp) < 0.0 and max(avgtemp) > 10.0:
            # Hot summer humid continental (Dfa)
            if max(avgtemp) > 22.0 and tempaboveten >= 4:
                climate[city] = 'Dfa'
            # Warm summer humid continental (Dfb)
            elif max(avgtemp) < 22.0 and tempaboveten >= 4:
                climate[city] = 'Dfb'
            # Subarctic (Dfc)
            elif tempaboveten >= 1 and tempaboveten <= 3:
                climate[city] = 'Dfc'
            # Extremely cold subarctic (Dfd)
            elif min(avgtemp) < -38.0 and tempaboveten >=1 and tempaboveten <= 3:
                climate[city] = 'Dfd'
    
            # Monsoon-influenced hot humid continental (Dwa)
            if max(avgtemp) > 22.0 and tempaboveten >= 4 and warmcoldratio >= 10:
                climate[city] = 'Dwa'
            # Monsoon-influenced warm humid continental (Dwb)
            elif max(avgtemp) < 22.0 and tempaboveten >= 4 and warmcoldratio >= 10:
                climate[city] = 'Dwb'
            # Monsoon-influenced subarctic (Dwc)
            elif tempaboveten >= 1 and tempaboveten <= 3 and warmcoldratio >= 10:
                climate[city] = 'Dwc'
            # Monsoon-influenced extremely cold subarctic (Dwd)
            elif min(avgtemp) < -38.0 and tempaboveten >= 1 and tempaboveten <= 3 and warmcoldratio >= 10:
                climate[city] = 'Dwd'
    
            # Hot, dry continental (Dsa)
            if max(avgtemp) > 22.0 and tempaboveten >= 4 and coldwarmratio >= 3 and min(precip[city][5:8]) < 30.0:
                climate[city] = 'Dsa'
            # Warm, dry continental (Dsb)
            elif max(avgtemp) < 22.0 and tempaboveten >= 4 and coldwarmratio >= 3 and min(precip[city][5:8]) < 30.0:
                climate[city] = 'Dsb'
            # Dry, subarctic (Dsc)
            elif tempaboveten >= 1 and tempaboveten <= 3 and coldwarmratio >= 1 and coldwarmratio >= 3 and \
                min(precip[city][5:8]) < 30.0:
                climate[city] = 'Dsc'
            # Extremely cold, dry subarctic (Dsd)
            elif min(avgtemp) < -38.0 and tempaboveten >= 1 and tempaboveten <= 3 and coldwarmratio >= 3 and \
                min(precip[city][5:8]) < 30.0:
                climate[city] = 'Dsd'
    
            if 'D' in climate[city]:
                continue
    
        # Group E (Polar and alpine)
        if max(avgtemp) < 10.0:
            # Tundra (ET)
            if max(avgtemp) > 0.0:
                climate[city] = 'ET'
            # Ice cap (EF)
            else:
                climate[city] = 'EF'
                
    return(climate)
 
#%%

koppen_colors = [('#0000fe'), #Af
                 ('#0277ff'), #Am
                 ('#379ae5'), #As
                 ('#6cb1e5'), #Aw
                 ('#fe0100'), #BWh
                 ('#fe9695'), #BWk
                 ('#f5a300'), #Bsh
                 ('#ffdb63'), #Bsk
                 ('#ffff00'), #Csa
                 ('#c6c701'), #Csb
                 ('#969600'), #Csc
                 ('#96ff96'), #Cwa
                 ('#63c764'), #Cwb
                 ('#329633'), #Cwc
                 ('#c7ff4d'), #Cfa
                 ('#66ff33'), #Cfb
                 ('#32c702'), #Cfc
                 ('#ff00fe'), #Dsa
                 ('#c600c7'), #Dsb
                 ('#963295'), #Dsc
                 ('#966495'), #Dsd
                 ('#abb1ff'), #Dwa
                 ('#5a77db'), #Dwb
                 ('#4c51b5'), #Dwc
                 ('#320087'), #Dwd
                 ('#02ffff'), #Dfa
                 ('#37c7ff'), #Dfb
                 ('#007e7d'), #Dfc
                 ('#00455e'), #Dfd
                 ('#b2b2b2'), #ET
                 ('#686868')] #EF
   
koppen_cats = ['Af','Am','As','Aw',
               'BWh','BWk','BSh','BSk',
               'Csa','Csb','Csc',
               'Cwa','Cwb','Cwc',
               'Cfa','Cfb','Cfc',
               'Dsa','Dsb','Dsc','Dsd',
               'Dwa','Dwb','Dwc','Dwd',
               'Dfa','Dfb','Dfc','Dfd',
               'ET','EF']
       
 #%%

obs_climate = koppen(obs_tmax,obs_tmin,obs_pr)   
wrf_d03_climate = koppen(wrf_d03_tmax,wrf_d03_tmin,wrf_d03_pr)   
wrf_d02_climate = koppen(wrf_d02_tmax,wrf_d02_tmin,wrf_d02_pr)   
raw_climate = koppen(raw_tmax,raw_tmin,raw_pr)   
rcm_climate = koppen(rcm_tmax,rcm_tmin,rcm_pr)   

wrf_d03_climate_rcp45 = koppen(wrf_d03_tmax_rcp45,wrf_d03_tmin_rcp45,wrf_d03_pr_rcp45)   
wrf_d03_climate_rcp85 = koppen(wrf_d03_tmax_rcp85,wrf_d03_tmin_rcp85,wrf_d03_pr_rcp85)   

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

canrcm4_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/orog_CanRCM4.nc'
canrcm4_nc = Dataset(canrcm4_file, mode='r')
lat_canrcm4 = np.squeeze(canrcm4_nc.variables['lat'][:])
lon_canrcm4 = np.squeeze(canrcm4_nc.variables['lon'][:])
topo_canrcm4 = np.squeeze(canrcm4_nc.variables['orog'][:])

canesm2_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/orog_CanESM2.nc'
canesm2_nc = Dataset(canesm2_file, mode='r')
lat_canesm2 = np.squeeze(canesm2_nc.variables['lat'][:])
lon_canesm2 = np.squeeze(canesm2_nc.variables['lon'][:])
topo_canesm2 = np.squeeze(canesm2_nc.variables['orog'][:])
#%%

def plot_koppen(climate,name):
    label_color_mapping = {}
    for label, color in zip(koppen_cats, koppen_colors):
        label_color_mapping[label] = color
    
    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique),loc=(0.17,0.32),fontsize=15,framealpha=0.9)
    
    #fig1,ax1 = plot_all_d03(lon_d02,lat_d02,topo_d02,lon_d03,lat_d03,topo_d03)
    
    #fig1,ax1 = plot_all_d03_flexdomain(lon_d02,lat_d02,topo_d02)
    #fig1,ax1 = plot_all_d03_flexdomain(lon_canrcm4,lat_canrcm4,topo_canrcm4)
    fig1,ax1 = plot_all_d03_flexdomain(lon_canesm2,lat_canesm2,topo_canesm2)


    for i in range(len(obs_lats)):
        index = obs_lats.index[i]
        
        if index not in climate:
            continue
        
        color = label_color_mapping[climate[index]]
        plt.scatter(obs_lons.iloc[i],obs_lats.iloc[i],s=150,color=color,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=0.8,zorder=4,marker='o',label=climate[index])
    
    
    cmap = 'terrain'
    vmin=0
    vmax=3000
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal',alpha=0.7)
    
    cbar_ax.tick_params(labelsize=16)
    
    cbar_ax.set_xlabel('Elevation [m]',size=18) 
      
    legend_without_duplicate_labels(ax1)
    ax1.set_title('Koppen Climate Classifications for ' + name,fontsize=17)
    fig1.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/koppen/' + name + '.png', dpi=600,bbox_inches='tight')
    


#plot_koppen(obs_climate,'Observations')
#plot_koppen(wrf_d03_climate, 'CanESM2-WRF D03 historical')
#plot_koppen(wrf_d02_climate, 'CanESM2-WRF D02 historical')
plot_koppen(raw_climate, 'CanESM2 historical')
#plot_koppen(rcm_climate, 'CanRCM4 historical')

#plot_koppen(wrf_d03_climate_rcp45, 'CanESM2-WRF D03 RCP45')
#plot_koppen(wrf_d03_climate_rcp85, 'CanESM2-WRF D03 RCP85')

