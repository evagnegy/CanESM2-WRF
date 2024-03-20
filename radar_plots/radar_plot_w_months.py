import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import warnings
import sys
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic


variable = 'wind' #t or pr
run = 'historical' #historical rcp45 or rcp85
output_freq = "daily" #yearly monthly or daily

#%%

eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'

df = pd.read_csv(eccc_daily_stations,header=None)
eccc_station_IDs = list(df.iloc[:,4])

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])

station_IDs = np.hstack((eccc_station_IDs,bch_station_IDs))


#%%

eccc_stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/'
bch_stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/daily/BCH/'

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
    eccc_obs = get_eccc_obs(output_freq,eccc_station_IDs,eccc_stations_dir,variable)
    bch_obs = get_bch_obs(output_freq,bch_station_IDs,bch_stations_dir,variable)

wrf_d02_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_bch = get_canesm2(output_freq, "BCH", bch_station_IDs, run, variable, raw_files_dir,start_year)
rcm_bch = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, variable, rcm_files_dir)
pcic_bch = get_pcic(output_freq, "BCH", bch_station_IDs, run, variable, pcic_files_dir)

wrf_d02_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_eccc = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, variable, raw_files_dir,start_year)
rcm_eccc = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, variable, rcm_files_dir)
pcic_eccc = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, variable, pcic_files_dir)


#%%
drop_stations = []
for i in eccc_station_IDs:
    if pcic_eccc[i].sum() == 0:
        print(i)
        drop_stations.append(i)
        
        #%%
eccc_obs = eccc_obs.drop(columns=drop_stations)
wrf_d02_eccc = wrf_d02_eccc.drop(columns=drop_stations)
wrf_d03_eccc = wrf_d03_eccc.drop(columns=drop_stations)
raw_eccc = raw_eccc.drop(columns=drop_stations)
rcm_eccc = rcm_eccc.drop(columns=drop_stations)
pcic_eccc = pcic_eccc.drop(columns=drop_stations)

#%%

eccc_obs_1 = eccc_obs.loc[(eccc_obs.index.month==1)].mean()
eccc_obs_2 = eccc_obs.loc[(eccc_obs.index.month==2)].mean()
eccc_obs_3 = eccc_obs.loc[(eccc_obs.index.month==3)].mean()
eccc_obs_4 = eccc_obs.loc[(eccc_obs.index.month==4)].mean()
eccc_obs_5 = eccc_obs.loc[(eccc_obs.index.month==5)].mean()
eccc_obs_6 = eccc_obs.loc[(eccc_obs.index.month==6)].mean()
eccc_obs_7 = eccc_obs.loc[(eccc_obs.index.month==7)].mean()
eccc_obs_8 = eccc_obs.loc[(eccc_obs.index.month==8)].mean()
eccc_obs_9 = eccc_obs.loc[(eccc_obs.index.month==9)].mean()
eccc_obs_10 = eccc_obs.loc[(eccc_obs.index.month==10)].mean()
eccc_obs_11 = eccc_obs.loc[(eccc_obs.index.month==11)].mean()
eccc_obs_12 = eccc_obs.loc[(eccc_obs.index.month==12)].mean()

wrf_d02_eccc_1 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==1)].mean()
wrf_d02_eccc_2 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==2)].mean()
wrf_d02_eccc_3 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==3)].mean()
wrf_d02_eccc_4 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==4)].mean()
wrf_d02_eccc_5 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==5)].mean()
wrf_d02_eccc_6 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==6)].mean()
wrf_d02_eccc_7 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==7)].mean()
wrf_d02_eccc_8 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==8)].mean()
wrf_d02_eccc_9 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==9)].mean()
wrf_d02_eccc_10 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==10)].mean()
wrf_d02_eccc_11 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==11)].mean()
wrf_d02_eccc_12 = wrf_d02_eccc.loc[(wrf_d02_eccc.index.month==12)].mean()

wrf_d03_eccc_1 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==1)].mean()
wrf_d03_eccc_2 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==2)].mean()
wrf_d03_eccc_3 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==3)].mean()
wrf_d03_eccc_4 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==4)].mean()
wrf_d03_eccc_5 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==5)].mean()
wrf_d03_eccc_6 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==6)].mean()
wrf_d03_eccc_7 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==7)].mean()
wrf_d03_eccc_8 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==8)].mean()
wrf_d03_eccc_9 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==9)].mean()
wrf_d03_eccc_10 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==10)].mean()
wrf_d03_eccc_11 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==11)].mean()
wrf_d03_eccc_12 = wrf_d03_eccc.loc[(wrf_d03_eccc.index.month==12)].mean()

raw_eccc_1 = raw_eccc.loc[(raw_eccc.index.month==1)].mean()
raw_eccc_2 = raw_eccc.loc[(raw_eccc.index.month==2)].mean()
raw_eccc_3 = raw_eccc.loc[(raw_eccc.index.month==3)].mean()
raw_eccc_4 = raw_eccc.loc[(raw_eccc.index.month==4)].mean()
raw_eccc_5 = raw_eccc.loc[(raw_eccc.index.month==5)].mean()
raw_eccc_6 = raw_eccc.loc[(raw_eccc.index.month==6)].mean()
raw_eccc_7 = raw_eccc.loc[(raw_eccc.index.month==7)].mean()
raw_eccc_8 = raw_eccc.loc[(raw_eccc.index.month==8)].mean()
raw_eccc_9 = raw_eccc.loc[(raw_eccc.index.month==9)].mean()
raw_eccc_10 = raw_eccc.loc[(raw_eccc.index.month==10)].mean()
raw_eccc_11 = raw_eccc.loc[(raw_eccc.index.month==11)].mean()
raw_eccc_12 = raw_eccc.loc[(raw_eccc.index.month==12)].mean()

rcm_eccc_1 = rcm_eccc.loc[(rcm_eccc.index.month==1)].mean()
rcm_eccc_2 = rcm_eccc.loc[(rcm_eccc.index.month==2)].mean()
rcm_eccc_3 = rcm_eccc.loc[(rcm_eccc.index.month==3)].mean()
rcm_eccc_4 = rcm_eccc.loc[(rcm_eccc.index.month==4)].mean()
rcm_eccc_5 = rcm_eccc.loc[(rcm_eccc.index.month==5)].mean()
rcm_eccc_6 = rcm_eccc.loc[(rcm_eccc.index.month==6)].mean()
rcm_eccc_7 = rcm_eccc.loc[(rcm_eccc.index.month==7)].mean()
rcm_eccc_8 = rcm_eccc.loc[(rcm_eccc.index.month==8)].mean()
rcm_eccc_9 = rcm_eccc.loc[(rcm_eccc.index.month==9)].mean()
rcm_eccc_10 = rcm_eccc.loc[(rcm_eccc.index.month==10)].mean()
rcm_eccc_11 = rcm_eccc.loc[(rcm_eccc.index.month==11)].mean()
rcm_eccc_12 = rcm_eccc.loc[(rcm_eccc.index.month==12)].mean()

pcic_eccc_1 = pcic_eccc.loc[(pcic_eccc.index.month==1)].mean()
pcic_eccc_2 = pcic_eccc.loc[(pcic_eccc.index.month==2)].mean()
pcic_eccc_3 = pcic_eccc.loc[(pcic_eccc.index.month==3)].mean()
pcic_eccc_4 = pcic_eccc.loc[(pcic_eccc.index.month==4)].mean()
pcic_eccc_5 = pcic_eccc.loc[(pcic_eccc.index.month==5)].mean()
pcic_eccc_6 = pcic_eccc.loc[(pcic_eccc.index.month==6)].mean()
pcic_eccc_7 = pcic_eccc.loc[(pcic_eccc.index.month==7)].mean()
pcic_eccc_8 = pcic_eccc.loc[(pcic_eccc.index.month==8)].mean()
pcic_eccc_9 = pcic_eccc.loc[(pcic_eccc.index.month==9)].mean()
pcic_eccc_10 = pcic_eccc.loc[(pcic_eccc.index.month==10)].mean()
pcic_eccc_11 = pcic_eccc.loc[(pcic_eccc.index.month==11)].mean()
pcic_eccc_12 = pcic_eccc.loc[(pcic_eccc.index.month==12)].mean()

bch_obs_1 = bch_obs.loc[(bch_obs.index.month==1)].mean()
bch_obs_2 = bch_obs.loc[(bch_obs.index.month==2)].mean()
bch_obs_3 = bch_obs.loc[(bch_obs.index.month==3)].mean()
bch_obs_4 = bch_obs.loc[(bch_obs.index.month==4)].mean()
bch_obs_5 = bch_obs.loc[(bch_obs.index.month==5)].mean()
bch_obs_6 = bch_obs.loc[(bch_obs.index.month==6)].mean()
bch_obs_7 = bch_obs.loc[(bch_obs.index.month==7)].mean()
bch_obs_8 = bch_obs.loc[(bch_obs.index.month==8)].mean()
bch_obs_9 = bch_obs.loc[(bch_obs.index.month==9)].mean()
bch_obs_10 = bch_obs.loc[(bch_obs.index.month==10)].mean()
bch_obs_11 = bch_obs.loc[(bch_obs.index.month==11)].mean()
bch_obs_12 = bch_obs.loc[(bch_obs.index.month==12)].mean()

wrf_d02_bch_1 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==1)].mean()
wrf_d02_bch_2 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==2)].mean()
wrf_d02_bch_3 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==3)].mean()
wrf_d02_bch_4 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==4)].mean()
wrf_d02_bch_5 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==5)].mean()
wrf_d02_bch_6 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==6)].mean()
wrf_d02_bch_7 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==7)].mean()
wrf_d02_bch_8 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==8)].mean()
wrf_d02_bch_9 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==9)].mean()
wrf_d02_bch_10 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==10)].mean()
wrf_d02_bch_11 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==11)].mean()
wrf_d02_bch_12 = wrf_d02_bch.loc[(wrf_d02_bch.index.month==12)].mean()

wrf_d03_bch_1 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==1)].mean()
wrf_d03_bch_2 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==2)].mean()
wrf_d03_bch_3 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==3)].mean()
wrf_d03_bch_4 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==4)].mean()
wrf_d03_bch_5 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==5)].mean()
wrf_d03_bch_6 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==6)].mean()
wrf_d03_bch_7 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==7)].mean()
wrf_d03_bch_8 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==8)].mean()
wrf_d03_bch_9 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==9)].mean()
wrf_d03_bch_10 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==10)].mean()
wrf_d03_bch_11 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==11)].mean()
wrf_d03_bch_12 = wrf_d03_bch.loc[(wrf_d03_bch.index.month==12)].mean()

raw_bch_1 = raw_bch.loc[(raw_bch.index.month==1)].mean()
raw_bch_2 = raw_bch.loc[(raw_bch.index.month==2)].mean()
raw_bch_3 = raw_bch.loc[(raw_bch.index.month==3)].mean()
raw_bch_4 = raw_bch.loc[(raw_bch.index.month==4)].mean()
raw_bch_5 = raw_bch.loc[(raw_bch.index.month==5)].mean()
raw_bch_6 = raw_bch.loc[(raw_bch.index.month==6)].mean()
raw_bch_7 = raw_bch.loc[(raw_bch.index.month==7)].mean()
raw_bch_8 = raw_bch.loc[(raw_bch.index.month==8)].mean()
raw_bch_9 = raw_bch.loc[(raw_bch.index.month==9)].mean()
raw_bch_10 = raw_bch.loc[(raw_bch.index.month==10)].mean()
raw_bch_11 = raw_bch.loc[(raw_bch.index.month==11)].mean()
raw_bch_12 = raw_bch.loc[(raw_bch.index.month==12)].mean()

rcm_bch_1 = rcm_bch.loc[(rcm_bch.index.month==1)].mean()
rcm_bch_2 = rcm_bch.loc[(rcm_bch.index.month==2)].mean()
rcm_bch_3 = rcm_bch.loc[(rcm_bch.index.month==3)].mean()
rcm_bch_4 = rcm_bch.loc[(rcm_bch.index.month==4)].mean()
rcm_bch_5 = rcm_bch.loc[(rcm_bch.index.month==5)].mean()
rcm_bch_6 = rcm_bch.loc[(rcm_bch.index.month==6)].mean()
rcm_bch_7 = rcm_bch.loc[(rcm_bch.index.month==7)].mean()
rcm_bch_8 = rcm_bch.loc[(rcm_bch.index.month==8)].mean()
rcm_bch_9 = rcm_bch.loc[(rcm_bch.index.month==9)].mean()
rcm_bch_10 = rcm_bch.loc[(rcm_bch.index.month==10)].mean()
rcm_bch_11 = rcm_bch.loc[(rcm_bch.index.month==11)].mean()
rcm_bch_12 = rcm_bch.loc[(rcm_bch.index.month==12)].mean()

pcic_bch_1 = pcic_bch.loc[(pcic_bch.index.month==1)].mean()
pcic_bch_2 = pcic_bch.loc[(pcic_bch.index.month==2)].mean()
pcic_bch_3 = pcic_bch.loc[(pcic_bch.index.month==3)].mean()
pcic_bch_4 = pcic_bch.loc[(pcic_bch.index.month==4)].mean()
pcic_bch_5 = pcic_bch.loc[(pcic_bch.index.month==5)].mean()
pcic_bch_6 = pcic_bch.loc[(pcic_bch.index.month==6)].mean()
pcic_bch_7 = pcic_bch.loc[(pcic_bch.index.month==7)].mean()
pcic_bch_8 = pcic_bch.loc[(pcic_bch.index.month==8)].mean()
pcic_bch_9 = pcic_bch.loc[(pcic_bch.index.month==9)].mean()
pcic_bch_10 = pcic_bch.loc[(pcic_bch.index.month==10)].mean()
pcic_bch_11 = pcic_bch.loc[(pcic_bch.index.month==11)].mean()
pcic_bch_12 = pcic_bch.loc[(pcic_bch.index.month==12)].mean()
#%%

obs_1 = pd.concat([eccc_obs_1,bch_obs_1])
obs_2 = pd.concat([eccc_obs_2,bch_obs_2])
obs_3 = pd.concat([eccc_obs_3,bch_obs_3])
obs_4 = pd.concat([eccc_obs_4,bch_obs_4])
obs_5 = pd.concat([eccc_obs_5,bch_obs_5])
obs_6 = pd.concat([eccc_obs_6,bch_obs_6])
obs_7 = pd.concat([eccc_obs_7,bch_obs_7])
obs_8 = pd.concat([eccc_obs_8,bch_obs_8])
obs_9 = pd.concat([eccc_obs_9,bch_obs_9])
obs_10 = pd.concat([eccc_obs_10,bch_obs_10])
obs_11 = pd.concat([eccc_obs_11,bch_obs_11])
obs_12 = pd.concat([eccc_obs_12,bch_obs_12])

wrf_d02_1 = pd.concat([wrf_d02_eccc_1,wrf_d02_bch_1])
wrf_d02_2 = pd.concat([wrf_d02_eccc_2,wrf_d02_bch_2])
wrf_d02_3 = pd.concat([wrf_d02_eccc_3,wrf_d02_bch_3])
wrf_d02_4 = pd.concat([wrf_d02_eccc_4,wrf_d02_bch_4])
wrf_d02_5 = pd.concat([wrf_d02_eccc_5,wrf_d02_bch_5])
wrf_d02_6 = pd.concat([wrf_d02_eccc_6,wrf_d02_bch_6])
wrf_d02_7 = pd.concat([wrf_d02_eccc_7,wrf_d02_bch_7])
wrf_d02_8 = pd.concat([wrf_d02_eccc_8,wrf_d02_bch_8])
wrf_d02_9 = pd.concat([wrf_d02_eccc_9,wrf_d02_bch_9])
wrf_d02_10 = pd.concat([wrf_d02_eccc_10,wrf_d02_bch_10])
wrf_d02_11 = pd.concat([wrf_d02_eccc_11,wrf_d02_bch_11])
wrf_d02_12 = pd.concat([wrf_d02_eccc_12,wrf_d02_bch_12])

wrf_d03_1 = pd.concat([wrf_d03_eccc_1,wrf_d03_bch_1])
wrf_d03_2 = pd.concat([wrf_d03_eccc_2,wrf_d03_bch_2])
wrf_d03_3 = pd.concat([wrf_d03_eccc_3,wrf_d03_bch_3])
wrf_d03_4 = pd.concat([wrf_d03_eccc_4,wrf_d03_bch_4])
wrf_d03_5 = pd.concat([wrf_d03_eccc_5,wrf_d03_bch_5])
wrf_d03_6 = pd.concat([wrf_d03_eccc_6,wrf_d03_bch_6])
wrf_d03_7 = pd.concat([wrf_d03_eccc_7,wrf_d03_bch_7])
wrf_d03_8 = pd.concat([wrf_d03_eccc_8,wrf_d03_bch_8])
wrf_d03_9 = pd.concat([wrf_d03_eccc_9,wrf_d03_bch_9])
wrf_d03_10 = pd.concat([wrf_d03_eccc_10,wrf_d03_bch_10])
wrf_d03_11 = pd.concat([wrf_d03_eccc_11,wrf_d03_bch_11])
wrf_d03_12 = pd.concat([wrf_d03_eccc_12,wrf_d03_bch_12])

raw_1 = pd.concat([raw_eccc_1,raw_bch_1])
raw_2 = pd.concat([raw_eccc_2,raw_bch_2])
raw_3 = pd.concat([raw_eccc_3,raw_bch_3])
raw_4 = pd.concat([raw_eccc_4,raw_bch_4])
raw_5 = pd.concat([raw_eccc_5,raw_bch_5])
raw_6 = pd.concat([raw_eccc_6,raw_bch_6])
raw_7 = pd.concat([raw_eccc_7,raw_bch_7])
raw_8 = pd.concat([raw_eccc_8,raw_bch_8])
raw_9 = pd.concat([raw_eccc_9,raw_bch_9])
raw_10 = pd.concat([raw_eccc_10,raw_bch_10])
raw_11 = pd.concat([raw_eccc_11,raw_bch_11])
raw_12 = pd.concat([raw_eccc_12,raw_bch_12])

rcm_1 = pd.concat([rcm_eccc_1,rcm_bch_1])
rcm_2 = pd.concat([rcm_eccc_2,rcm_bch_2])
rcm_3 = pd.concat([rcm_eccc_3,rcm_bch_3])
rcm_4 = pd.concat([rcm_eccc_4,rcm_bch_4])
rcm_5 = pd.concat([rcm_eccc_5,rcm_bch_5])
rcm_6 = pd.concat([rcm_eccc_6,rcm_bch_6])
rcm_7 = pd.concat([rcm_eccc_7,rcm_bch_7])
rcm_8 = pd.concat([rcm_eccc_8,rcm_bch_8])
rcm_9 = pd.concat([rcm_eccc_9,rcm_bch_9])
rcm_10 = pd.concat([rcm_eccc_10,rcm_bch_10])
rcm_11 = pd.concat([rcm_eccc_11,rcm_bch_11])
rcm_12 = pd.concat([rcm_eccc_12,rcm_bch_12])

pcic_1 = pd.concat([pcic_eccc_1,pcic_bch_1])
pcic_2 = pd.concat([pcic_eccc_2,pcic_bch_2])
pcic_3 = pd.concat([pcic_eccc_3,pcic_bch_3])
pcic_4 = pd.concat([pcic_eccc_4,pcic_bch_4])
pcic_5 = pd.concat([pcic_eccc_5,pcic_bch_5])
pcic_6 = pd.concat([pcic_eccc_6,pcic_bch_6])
pcic_7 = pd.concat([pcic_eccc_7,pcic_bch_7])
pcic_8 = pd.concat([pcic_eccc_8,pcic_bch_8])
pcic_9 = pd.concat([pcic_eccc_9,pcic_bch_9])
pcic_10 = pd.concat([pcic_eccc_10,pcic_bch_10])
pcic_11 = pd.concat([pcic_eccc_11,pcic_bch_11])
pcic_12 = pd.concat([pcic_eccc_12,pcic_bch_12])

#%%

wrf_d02_bias_1 = abs(np.mean(wrf_d02_1 - obs_1))
wrf_d02_bias_2 = abs(np.mean(wrf_d02_2 - obs_2))
wrf_d02_bias_3 = abs(np.mean(wrf_d02_3 - obs_3))
wrf_d02_bias_4 = abs(np.mean(wrf_d02_4 - obs_4))
wrf_d02_bias_5 = abs(np.mean(wrf_d02_5 - obs_5))
wrf_d02_bias_6 = abs(np.mean(wrf_d02_6 - obs_6))
wrf_d02_bias_7 = abs(np.mean(wrf_d02_7 - obs_7))
wrf_d02_bias_8 = abs(np.mean(wrf_d02_8 - obs_8))
wrf_d02_bias_9 = abs(np.mean(wrf_d02_9 - obs_9))
wrf_d02_bias_10 = abs(np.mean(wrf_d02_10 - obs_10))
wrf_d02_bias_11 = abs(np.mean(wrf_d02_11 - obs_11))
wrf_d02_bias_12 = abs(np.mean(wrf_d02_12 - obs_12))

wrf_d03_bias_1 = abs(np.mean(wrf_d03_1 - obs_1))
wrf_d03_bias_2 = abs(np.mean(wrf_d03_2 - obs_2))
wrf_d03_bias_3 = abs(np.mean(wrf_d03_3 - obs_3))
wrf_d03_bias_4 = abs(np.mean(wrf_d03_4 - obs_4))
wrf_d03_bias_5 = abs(np.mean(wrf_d03_5 - obs_5))
wrf_d03_bias_6 = abs(np.mean(wrf_d03_6 - obs_6))
wrf_d03_bias_7 = abs(np.mean(wrf_d03_7 - obs_7))
wrf_d03_bias_8 = abs(np.mean(wrf_d03_8 - obs_8))
wrf_d03_bias_9 = abs(np.mean(wrf_d03_9 - obs_9))
wrf_d03_bias_10 = abs(np.mean(wrf_d03_10 - obs_10))
wrf_d03_bias_11 = abs(np.mean(wrf_d03_11 - obs_11))
wrf_d03_bias_12 = abs(np.mean(wrf_d03_12 - obs_12))

raw_bias_1 = abs(np.mean(raw_1 - obs_1))
raw_bias_2 = abs(np.mean(raw_2 - obs_2))
raw_bias_3 = abs(np.mean(raw_3 - obs_3))
raw_bias_4 = abs(np.mean(raw_4 - obs_4))
raw_bias_5 = abs(np.mean(raw_5 - obs_5))
raw_bias_6 = abs(np.mean(raw_6 - obs_6))
raw_bias_7 = abs(np.mean(raw_7 - obs_7))
raw_bias_8 = abs(np.mean(raw_8 - obs_8))
raw_bias_9 = abs(np.mean(raw_9 - obs_9))
raw_bias_10 = abs(np.mean(raw_10 - obs_10))
raw_bias_11 = abs(np.mean(raw_11 - obs_11))
raw_bias_12 = abs(np.mean(raw_12 - obs_12))

rcm_bias_1 = abs(np.mean(rcm_1 - obs_1))
rcm_bias_2 = abs(np.mean(rcm_2 - obs_2))
rcm_bias_3 = abs(np.mean(rcm_3 - obs_3))
rcm_bias_4 = abs(np.mean(rcm_4 - obs_4))
rcm_bias_5 = abs(np.mean(rcm_5 - obs_5))
rcm_bias_6 = abs(np.mean(rcm_6 - obs_6))
rcm_bias_7 = abs(np.mean(rcm_7 - obs_7))
rcm_bias_8 = abs(np.mean(rcm_8 - obs_8))
rcm_bias_9 = abs(np.mean(rcm_9 - obs_9))
rcm_bias_10 = abs(np.mean(rcm_10 - obs_10))
rcm_bias_11 = abs(np.mean(rcm_11 - obs_11))
rcm_bias_12 = abs(np.mean(rcm_12 - obs_12))

pcic_bias_1 = abs(np.mean(pcic_1 - obs_1))
pcic_bias_2 = abs(np.mean(pcic_2 - obs_2))
pcic_bias_3 = abs(np.mean(pcic_3 - obs_3))
pcic_bias_4 = abs(np.mean(pcic_4 - obs_4))
pcic_bias_5 = abs(np.mean(pcic_5 - obs_5))
pcic_bias_6 = abs(np.mean(pcic_6 - obs_6))
pcic_bias_7 = abs(np.mean(pcic_7 - obs_7))
pcic_bias_8 = abs(np.mean(pcic_8 - obs_8))
pcic_bias_9 = abs(np.mean(pcic_9 - obs_9))
pcic_bias_10 = abs(np.mean(pcic_10 - obs_10))
pcic_bias_11 = abs(np.mean(pcic_11 - obs_11))
pcic_bias_12 = abs(np.mean(pcic_12 - obs_12))

#%%
# Set data
df = pd.DataFrame({
'model': ['WRF_d03','WRF_d02','CanESM2','CanRCM4','PCIC'],
'Jan': [wrf_d03_bias_1,wrf_d02_bias_1,raw_bias_1,rcm_bias_1,pcic_bias_1],
'Feb': [wrf_d03_bias_2,wrf_d02_bias_2,raw_bias_2,rcm_bias_2,pcic_bias_2],
'Mar': [wrf_d03_bias_3,wrf_d02_bias_3,raw_bias_3,rcm_bias_3,pcic_bias_3],
'Apr': [wrf_d03_bias_4,wrf_d02_bias_4,raw_bias_4,rcm_bias_4,pcic_bias_4],
'May': [wrf_d03_bias_5,wrf_d02_bias_5,raw_bias_5,rcm_bias_5,pcic_bias_5],
'Jun': [wrf_d03_bias_6,wrf_d02_bias_6,raw_bias_6,rcm_bias_6,pcic_bias_6],
'Jul': [wrf_d03_bias_7,wrf_d02_bias_7,raw_bias_7,rcm_bias_7,pcic_bias_7],
'Aug': [wrf_d03_bias_8,wrf_d02_bias_8,raw_bias_8,rcm_bias_8,pcic_bias_8],
'Sep': [wrf_d03_bias_9,wrf_d02_bias_9,raw_bias_9,rcm_bias_9,pcic_bias_9],
'Oct': [wrf_d03_bias_10,wrf_d02_bias_10,raw_bias_10,rcm_bias_10,pcic_bias_10],
'Nov': [wrf_d03_bias_11,wrf_d02_bias_11,raw_bias_11,rcm_bias_11,pcic_bias_11],
'Dec': [wrf_d03_bias_12,wrf_d02_bias_12,raw_bias_12,rcm_bias_12,pcic_bias_12],
})
 
#%%
# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks(color="grey", size=7)
plt.ylim(0,6)
 

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable
 
labels = ['WRF 3km','WRF 15km','CanESM2','CanRCM4','PCIC (CanESM2)']
colors = ['C0','C1','C2','C3','C4']

for i in range(len(df)):
    values=df.loc[i].drop('model').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values,'.-',linewidth=1, label=labels[i])
    ax.fill(angles, values, colors[i], alpha=0.1)
 
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Show the graph
plt.show()


Python