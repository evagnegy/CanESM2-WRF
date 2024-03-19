import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import warnings
import sys
import numpy as np
from netCDF4 import Dataset
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic,plot_all_d03,plot_zoomed_in
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib
import math

#variable = 't' #t or pr
run = 'historical' #historical rcp45 or rcp85
output_freq = "daily" #hourly or 6hr

#%%

eccc_hourly_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations_hourly.csv'

df = pd.read_csv(eccc_hourly_stations,header=None)
eccc_station_IDs = list(df.iloc[:,4])

eccc_station_IDs.remove(631) #drop bc at d03 domain border


#%%

eccc_stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/'

WRF_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/' + run + '/'
raw_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/' + run + '/'
rcm_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/' + run + '/'


if run == 'historical':
    start_year = 1986
    end_year = 2005
else:
    start_year = 2046
    end_year = 2065  

#%%

if run == 'historical': #no station obs for rcps
    eccc_obs_t = get_eccc_obs(output_freq,eccc_station_IDs,eccc_stations_dir,"t")
    eccc_obs_pr = get_eccc_obs(output_freq,eccc_station_IDs,eccc_stations_dir,"pr")
    eccc_obs_tmax = get_eccc_obs(output_freq,eccc_station_IDs,eccc_stations_dir,"tmax")
    eccc_obs_wind = get_eccc_obs(output_freq,eccc_station_IDs,eccc_stations_dir,"wind")


#%%

wrf_d02_eccc_t = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, "t", WRF_files_dir,start_year)
wrf_d03_eccc_t = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, "t", WRF_files_dir,start_year)
raw_eccc_t = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, "t", raw_files_dir,start_year)
rcm_eccc_t = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, "t", rcm_files_dir)

wrf_d02_eccc_pr = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, "pr", WRF_files_dir,start_year)
wrf_d03_eccc_pr = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, "pr", WRF_files_dir,start_year)
raw_eccc_pr = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, "pr", raw_files_dir,start_year)
rcm_eccc_pr = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, "pr", rcm_files_dir)

wrf_d02_eccc_tmax = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, "tmax", WRF_files_dir,start_year)
wrf_d03_eccc_tmax = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, "tmax", WRF_files_dir,start_year)
raw_eccc_tmax = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, "tmax", raw_files_dir,start_year)
rcm_eccc_tmax = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, "tmax", rcm_files_dir)


wrf_d02_eccc_wind = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, "wind", WRF_files_dir,start_year)
wrf_d03_eccc_wind = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, "wind", WRF_files_dir,start_year)
raw_eccc_wind = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, "wind", raw_files_dir,start_year)
rcm_eccc_wind = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, "wind", rcm_files_dir)

        
#%%
obs_t_avg = eccc_obs_t.mean()
obs_pr_avg = eccc_obs_pr.mean()
obs_tmax_avg = eccc_obs_tmax.mean()
obs_wind_avg = eccc_obs_wind.mean()

wrf_d02_t_avg = wrf_d02_eccc_t.mean()
wrf_d02_pr_avg = wrf_d02_eccc_pr.mean()
wrf_d02_tmax_avg = wrf_d02_eccc_tmax.mean()
wrf_d02_wind_avg = wrf_d02_eccc_wind.mean()

wrf_d03_t_avg = wrf_d03_eccc_t.mean()
wrf_d03_pr_avg = wrf_d03_eccc_pr.mean()
wrf_d03_tmax_avg = wrf_d03_eccc_tmax.mean()
wrf_d03_wind_avg = wrf_d03_eccc_wind.mean()

raw_t_avg = raw_eccc_t.mean()
raw_pr_avg = raw_eccc_pr.mean()
raw_tmax_avg = raw_eccc_tmax.mean()
raw_wind_avg = raw_eccc_wind.mean()

rcm_t_avg = rcm_eccc_t.mean()
rcm_pr_avg = rcm_eccc_pr.mean()
rcm_tmax_avg = rcm_eccc_tmax.mean()
rcm_wind_avg = rcm_eccc_wind.mean()


#%%
wrf_d02_t_NRMSE = math.sqrt(np.square(np.subtract(obs_t_avg,wrf_d02_t_avg)).mean())/np.mean(obs_t_avg)
wrf_d02_pr_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_avg,wrf_d02_pr_avg)).mean())/np.mean(obs_pr_avg)
wrf_d02_tmax_NRMSE = math.sqrt(np.square(np.subtract(obs_tmax_avg,wrf_d02_tmax_avg)).mean())/np.mean(obs_tmax_avg)
wrf_d02_wind_NRMSE = math.sqrt(np.square(np.subtract(obs_wind_avg,wrf_d02_wind_avg)).mean())/np.mean(obs_wind_avg)

wrf_d03_t_NRMSE = math.sqrt(np.square(np.subtract(obs_t_avg,wrf_d03_t_avg)).mean())/np.mean(obs_t_avg)
wrf_d03_pr_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_avg,wrf_d03_pr_avg)).mean())/np.mean(obs_pr_avg)
wrf_d03_tmax_NRMSE = math.sqrt(np.square(np.subtract(obs_tmax_avg,wrf_d03_tmax_avg)).mean())/np.mean(obs_tmax_avg)
wrf_d03_wind_NRMSE = math.sqrt(np.square(np.subtract(obs_wind_avg,wrf_d03_wind_avg)).mean())/np.mean(obs_wind_avg)

raw_t_NRMSE = math.sqrt(np.square(np.subtract(obs_t_avg,raw_t_avg)).mean())/np.mean(obs_t_avg)
raw_pr_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_avg,raw_pr_avg)).mean())/np.mean(obs_pr_avg)
raw_tmax_NRMSE = math.sqrt(np.square(np.subtract(obs_tmax_avg,raw_tmax_avg)).mean())/np.mean(obs_tmax_avg)
raw_wind_NRMSE = math.sqrt(np.square(np.subtract(obs_wind_avg,raw_wind_avg)).mean())/np.mean(obs_wind_avg)

rcm_t_NRMSE = math.sqrt(np.square(np.subtract(obs_t_avg,rcm_t_avg)).mean())/np.mean(obs_t_avg)
rcm_pr_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_avg,rcm_pr_avg)).mean())/np.mean(obs_pr_avg)
rcm_tmax_NRMSE = math.sqrt(np.square(np.subtract(obs_tmax_avg,rcm_tmax_avg)).mean())/np.mean(obs_tmax_avg)
rcm_wind_NRMSE = math.sqrt(np.square(np.subtract(obs_wind_avg,rcm_wind_avg)).mean())/np.mean(obs_wind_avg)

#%%
# Set data
df = pd.DataFrame({
'model': ['WRF_d03','WRF_d02','CanESM2','CanRCM4'],
't ': [wrf_d03_t_NRMSE,wrf_d02_t_NRMSE,raw_t_NRMSE,rcm_t_NRMSE],
'tmax': [wrf_d03_tmax_NRMSE,wrf_d02_tmax_NRMSE,raw_tmax_NRMSE,rcm_tmax_NRMSE],
'pr': [wrf_d03_pr_NRMSE,wrf_d02_pr_NRMSE,raw_pr_NRMSE,rcm_pr_NRMSE],
'wind': [wrf_d03_wind_NRMSE,wrf_d02_wind_NRMSE,raw_wind_NRMSE,rcm_wind_NRMSE]

})
 
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
plt.ylim(0,1)
 

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable
 
labels = ['WRF 3km','WRF 15km','CanESM2','CanRCM4']
colors = ['C0','C1','C2','C3']

for i in range(len(df)):
    values=df.loc[i].drop('model').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values,'.-',linewidth=1, label=labels[i])
    ax.fill(angles, values, colors[i], alpha=0.1)
 
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Show the graph
plt.show()


