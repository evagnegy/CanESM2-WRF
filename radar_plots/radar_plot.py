
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

run = 'historical' #historical rcp45 or rcp85
output_freq = "yearly" #yearly monthly or daily

#%%

eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'

df = pd.read_csv(eccc_daily_stations,header=None)
eccc_station_IDs = list(df.iloc[:,4])

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])

station_IDs = np.hstack((eccc_station_IDs,bch_station_IDs))


#%%

eccc_stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/daily/ECCC/'
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
    eccc_obs_t = get_eccc_obs(output_freq,eccc_station_IDs,eccc_stations_dir,"t")
    bch_obs_t = get_bch_obs(output_freq,bch_station_IDs,bch_stations_dir,"t")
    eccc_obs_pr = get_eccc_obs(output_freq,eccc_station_IDs,eccc_stations_dir,"pr")
    bch_obs_pr = get_bch_obs(output_freq,bch_station_IDs,bch_stations_dir,"pr")
    eccc_obs_tmax = get_eccc_obs(output_freq,eccc_station_IDs,eccc_stations_dir,"tmax")
    bch_obs_tmax = get_bch_obs(output_freq,bch_station_IDs,bch_stations_dir,"tmax")
    eccc_obs_pr_d = get_eccc_obs('daily',eccc_station_IDs,eccc_stations_dir,"pr")
    bch_obs_pr_d = get_bch_obs('daily',bch_station_IDs,bch_stations_dir,"pr")

wrf_d02_bch_t = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, "t", WRF_files_dir,start_year)
wrf_d03_bch_t = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, "t", WRF_files_dir,start_year)
raw_bch_t = get_canesm2(output_freq, "BCH", bch_station_IDs, run, "t", raw_files_dir,start_year)
rcm_bch_t = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, "t", rcm_files_dir)
pcic_bch_t = get_pcic(output_freq, "BCH", bch_station_IDs, run, "t", pcic_files_dir)

wrf_d02_eccc_t = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, "t", WRF_files_dir,start_year)
wrf_d03_eccc_t = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, "t", WRF_files_dir,start_year)
raw_eccc_t = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, "t", raw_files_dir,start_year)
rcm_eccc_t = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, "t", rcm_files_dir)
pcic_eccc_t = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, "t", pcic_files_dir)

wrf_d02_bch_pr = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, "pr", WRF_files_dir,start_year)
wrf_d03_bch_pr = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, "pr", WRF_files_dir,start_year)
raw_bch_pr = get_canesm2(output_freq, "BCH", bch_station_IDs, run, "pr", raw_files_dir,start_year)
rcm_bch_pr = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, "pr", rcm_files_dir)
pcic_bch_pr = get_pcic(output_freq, "BCH", bch_station_IDs, run, "pr", pcic_files_dir)

wrf_d02_eccc_pr = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, "pr", WRF_files_dir,start_year)
wrf_d03_eccc_pr = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, "pr", WRF_files_dir,start_year)
raw_eccc_pr = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, "pr", raw_files_dir,start_year)
rcm_eccc_pr = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, "pr", rcm_files_dir)
pcic_eccc_pr = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, "pr", pcic_files_dir)

wrf_d02_bch_tmax = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, "tmax", WRF_files_dir,start_year)
wrf_d03_bch_tmax = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, "tmax", WRF_files_dir,start_year)
raw_bch_tmax = get_canesm2(output_freq, "BCH", bch_station_IDs, run, "tmax", raw_files_dir,start_year)
rcm_bch_tmax = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, "tmax", rcm_files_dir)
pcic_bch_tmax = get_pcic(output_freq, "BCH", bch_station_IDs, run, "tmax", pcic_files_dir)

wrf_d02_eccc_tmax = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, "tmax", WRF_files_dir,start_year)
wrf_d03_eccc_tmax = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, "tmax", WRF_files_dir,start_year)
raw_eccc_tmax = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, "tmax", raw_files_dir,start_year)
rcm_eccc_tmax = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, "tmax", rcm_files_dir)
pcic_eccc_tmax = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, "tmax", pcic_files_dir)

wrf_d02_bch_pr_d = get_wrf("daily", "BCH", bch_station_IDs, "d02", run, "pr", WRF_files_dir,start_year)
wrf_d03_bch_pr_d = get_wrf("daily", "BCH", bch_station_IDs, "d03", run, "pr", WRF_files_dir,start_year)
raw_bch_pr_d = get_canesm2("daily", "BCH", bch_station_IDs, run, "pr", raw_files_dir,start_year)
rcm_bch_pr_d = get_canrcm4("daily", "BCH", bch_station_IDs, run, "pr", rcm_files_dir)
pcic_bch_pr_d = get_pcic("daily", "BCH", bch_station_IDs, run, "pr", pcic_files_dir)

wrf_d02_eccc_pr_d = get_wrf("daily", "ECCC", eccc_station_IDs, "d02", run, "pr", WRF_files_dir,start_year)
wrf_d03_eccc_pr_d = get_wrf("daily", "ECCC", eccc_station_IDs, "d03", run, "pr", WRF_files_dir,start_year)
raw_eccc_pr_d = get_canesm2("daily", "ECCC", eccc_station_IDs, run, "pr", raw_files_dir,start_year)
rcm_eccc_pr_d = get_canrcm4("daily", "ECCC", eccc_station_IDs, run, "pr", rcm_files_dir)
pcic_eccc_pr_d = get_pcic("daily", "ECCC", eccc_station_IDs, run, "pr", pcic_files_dir)


#%%
drop_stations = []
for i in eccc_station_IDs:
    if pcic_eccc_pr[i].sum() == 0:
        print(i)
        drop_stations.append(i)
        
        #%%
eccc_obs_t = eccc_obs_t.drop(columns=drop_stations)
eccc_obs_pr = eccc_obs_pr.drop(columns=drop_stations)

wrf_d02_eccc_t = wrf_d02_eccc_t.drop(columns=drop_stations)
wrf_d03_eccc_t = wrf_d03_eccc_t.drop(columns=drop_stations)
raw_eccc_t = raw_eccc_t.drop(columns=drop_stations)
rcm_eccc_t = rcm_eccc_t.drop(columns=drop_stations)
pcic_eccc_t = pcic_eccc_t.drop(columns=drop_stations)

wrf_d02_eccc_pr = wrf_d02_eccc_pr.drop(columns=drop_stations)
wrf_d03_eccc_pr = wrf_d03_eccc_pr.drop(columns=drop_stations)
raw_eccc_pr = raw_eccc_pr.drop(columns=drop_stations)
rcm_eccc_pr = rcm_eccc_pr.drop(columns=drop_stations)
pcic_eccc_pr = pcic_eccc_pr.drop(columns=drop_stations)

wrf_d02_eccc_tmax = wrf_d02_eccc_tmax.drop(columns=drop_stations)
wrf_d03_eccc_tmax = wrf_d03_eccc_tmax.drop(columns=drop_stations)
raw_eccc_tmax = raw_eccc_tmax.drop(columns=drop_stations)
rcm_eccc_tmax = rcm_eccc_tmax.drop(columns=drop_stations)
pcic_eccc_tmax = pcic_eccc_tmax.drop(columns=drop_stations)
   
wrf_d02_eccc_pr_d = wrf_d02_eccc_pr_d.drop(columns=drop_stations)
wrf_d03_eccc_pr_d = wrf_d03_eccc_pr_d.drop(columns=drop_stations)
raw_eccc_pr_d = raw_eccc_pr_d.drop(columns=drop_stations)
rcm_eccc_pr_d = rcm_eccc_pr_d.drop(columns=drop_stations)
pcic_eccc_pr_d = pcic_eccc_pr_d.drop(columns=drop_stations) 
#%%
eccc_obs_t_avg = eccc_obs_t.mean()
bch_obs_t_avg = bch_obs_t.mean()
eccc_obs_pr_avg = eccc_obs_pr.mean()
bch_obs_pr_avg = bch_obs_pr.mean()
eccc_obs_tmax_avg = eccc_obs_tmax.mean()
bch_obs_tmax_avg = bch_obs_tmax.mean()
eccc_obs_pr_d_avg = eccc_obs_pr_d.mean()
bch_obs_pr_d_avg = bch_obs_pr_d.mean()

wrf_d02_eccc_t_avg = wrf_d02_eccc_t.mean()
wrf_d02_bch_t_avg = wrf_d02_bch_t.mean()
wrf_d02_eccc_pr_avg = wrf_d02_eccc_pr.mean()
wrf_d02_bch_pr_avg = wrf_d02_bch_pr.mean()
wrf_d02_eccc_tmax_avg = wrf_d02_eccc_tmax.mean()
wrf_d02_bch_tmax_avg = wrf_d02_bch_tmax.mean()
wrf_d02_eccc_pr_d_avg = wrf_d02_eccc_pr_d.mean()
wrf_d02_bch_pr_d_avg = wrf_d02_bch_pr_d.mean()

wrf_d03_eccc_t_avg = wrf_d03_eccc_t.mean()
wrf_d03_bch_t_avg = wrf_d03_bch_t.mean()
wrf_d03_eccc_pr_avg = wrf_d03_eccc_pr.mean()
wrf_d03_bch_pr_avg = wrf_d03_bch_pr.mean()
wrf_d03_eccc_tmax_avg = wrf_d03_eccc_tmax.mean()
wrf_d03_bch_tmax_avg = wrf_d03_bch_tmax.mean()
wrf_d03_eccc_pr_d_avg = wrf_d03_eccc_pr_d.mean()
wrf_d03_bch_pr_d_avg = wrf_d03_bch_pr_d.mean()

raw_eccc_t_avg = raw_eccc_t.mean()
raw_bch_t_avg = raw_bch_t.mean()
raw_eccc_pr_avg = raw_eccc_pr.mean()
raw_bch_pr_avg = raw_bch_pr.mean()
raw_eccc_tmax_avg = raw_eccc_tmax.mean()
raw_bch_tmax_avg = raw_bch_tmax.mean()
raw_eccc_pr_d_avg = raw_eccc_pr_d.mean()
raw_bch_pr_d_avg = raw_bch_pr_d.mean()

rcm_eccc_t_avg = rcm_eccc_t.mean()
rcm_bch_t_avg = rcm_bch_t.mean()
rcm_eccc_pr_avg = rcm_eccc_pr.mean()
rcm_bch_pr_avg = rcm_bch_pr.mean()
rcm_eccc_tmax_avg = rcm_eccc_tmax.mean()
rcm_bch_tmax_avg = rcm_bch_tmax.mean()
rcm_eccc_pr_d_avg = rcm_eccc_pr_d.mean()
rcm_bch_pr_d_avg = rcm_bch_pr_d.mean()

pcic_eccc_t_avg = pcic_eccc_t.mean()
pcic_bch_t_avg = pcic_bch_t.mean()
pcic_eccc_pr_avg = pcic_eccc_pr.mean()
pcic_bch_pr_avg = pcic_bch_pr.mean()
pcic_eccc_tmax_avg = pcic_eccc_tmax.mean()
pcic_bch_tmax_avg = pcic_bch_tmax.mean()
pcic_eccc_pr_d_avg = pcic_eccc_pr_d.mean()
pcic_bch_pr_d_avg = pcic_bch_pr_d.mean()
#%%

obs_t_avg = pd.concat([eccc_obs_t_avg,bch_obs_t_avg])
obs_pr_avg = pd.concat([eccc_obs_pr_avg,bch_obs_pr_avg])
obs_tmax_avg = pd.concat([eccc_obs_tmax_avg,bch_obs_tmax_avg])
obs_pr_d_avg = pd.concat([eccc_obs_pr_d_avg,bch_obs_pr_d_avg])

wrf_d02_t_avg = pd.concat([wrf_d02_eccc_t_avg,wrf_d02_bch_t_avg])
wrf_d02_pr_avg = pd.concat([wrf_d02_eccc_pr_avg,wrf_d02_bch_pr_avg])
wrf_d02_tmax_avg = pd.concat([wrf_d02_eccc_tmax_avg,wrf_d02_bch_tmax_avg])
wrf_d02_pr_d_avg = pd.concat([wrf_d02_eccc_pr_d_avg,wrf_d02_bch_pr_d_avg])

wrf_d03_t_avg = pd.concat([wrf_d03_eccc_t_avg,wrf_d03_bch_t_avg])
wrf_d03_pr_avg = pd.concat([wrf_d03_eccc_pr_avg,wrf_d03_bch_pr_avg])
wrf_d03_tmax_avg = pd.concat([wrf_d03_eccc_tmax_avg,wrf_d03_bch_tmax_avg])
wrf_d03_pr_d_avg = pd.concat([wrf_d03_eccc_pr_d_avg,wrf_d03_bch_pr_d_avg])

raw_t_avg = pd.concat([raw_eccc_t_avg,raw_bch_t_avg])
raw_pr_avg = pd.concat([raw_eccc_pr_avg,raw_bch_pr_avg])
raw_tmax_avg = pd.concat([raw_eccc_tmax_avg,raw_bch_tmax_avg])
raw_pr_d_avg = pd.concat([raw_eccc_pr_d_avg,raw_bch_pr_d_avg])

rcm_t_avg = pd.concat([rcm_eccc_t_avg,rcm_bch_t_avg])
rcm_pr_avg = pd.concat([rcm_eccc_pr_avg,rcm_bch_pr_avg])
rcm_tmax_avg = pd.concat([rcm_eccc_tmax_avg,rcm_bch_tmax_avg])
rcm_pr_d_avg = pd.concat([rcm_eccc_pr_d_avg,rcm_bch_pr_d_avg])

pcic_t_avg = pd.concat([pcic_eccc_t_avg,pcic_bch_t_avg])
pcic_pr_avg = pd.concat([pcic_eccc_pr_avg,pcic_bch_pr_avg])
pcic_tmax_avg = pd.concat([pcic_eccc_tmax_avg,pcic_bch_tmax_avg])
pcic_pr_d_avg = pd.concat([pcic_eccc_pr_d_avg,pcic_bch_pr_d_avg])

#%%
wrf_d02_t_NRMSE = math.sqrt(np.square(np.subtract(obs_t_avg,wrf_d02_t_avg)).mean())/np.mean(obs_t_avg)
wrf_d02_pr_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_avg,wrf_d02_pr_avg)).mean())/np.mean(obs_pr_avg)
wrf_d02_tmax_NRMSE = math.sqrt(np.square(np.subtract(obs_tmax_avg,wrf_d02_tmax_avg)).mean())/np.mean(obs_tmax_avg)
wrf_d02_pr_d_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_d_avg,wrf_d02_pr_d_avg)).mean())/np.mean(obs_pr_d_avg)

wrf_d03_t_NRMSE = math.sqrt(np.square(np.subtract(obs_t_avg,wrf_d03_t_avg)).mean())/np.mean(obs_t_avg)
wrf_d03_pr_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_avg,wrf_d03_pr_avg)).mean())/np.mean(obs_pr_avg)
wrf_d03_tmax_NRMSE = math.sqrt(np.square(np.subtract(obs_tmax_avg,wrf_d03_tmax_avg)).mean())/np.mean(obs_tmax_avg)
wrf_d03_pr_d_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_d_avg,wrf_d03_pr_d_avg)).mean())/np.mean(obs_pr_d_avg)

raw_t_NRMSE = math.sqrt(np.square(np.subtract(obs_t_avg,raw_t_avg)).mean())/np.mean(obs_t_avg)
raw_pr_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_avg,raw_pr_avg)).mean())/np.mean(obs_pr_avg)
raw_tmax_NRMSE = math.sqrt(np.square(np.subtract(obs_tmax_avg,raw_tmax_avg)).mean())/np.mean(obs_tmax_avg)
raw_pr_d_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_d_avg,raw_pr_d_avg)).mean())/np.mean(obs_pr_d_avg)

rcm_t_NRMSE = math.sqrt(np.square(np.subtract(obs_t_avg,rcm_t_avg)).mean())/np.mean(obs_t_avg)
rcm_pr_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_avg,rcm_pr_avg)).mean())/np.mean(obs_pr_avg)
rcm_tmax_NRMSE = math.sqrt(np.square(np.subtract(obs_tmax_avg,rcm_tmax_avg)).mean())/np.mean(obs_tmax_avg)
rcm_pr_d_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_d_avg,rcm_pr_d_avg)).mean())/np.mean(obs_pr_d_avg)

pcic_t_NRMSE = math.sqrt(np.square(np.subtract(obs_t_avg,pcic_t_avg)).mean())/np.mean(obs_t_avg)
pcic_pr_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_avg,pcic_pr_avg)).mean())/np.mean(obs_pr_avg)
pcic_tmax_NRMSE = math.sqrt(np.square(np.subtract(obs_tmax_avg,pcic_tmax_avg)).mean())/np.mean(obs_tmax_avg)
pcic_pr_d_NRMSE = math.sqrt(np.square(np.subtract(obs_pr_d_avg,pcic_pr_d_avg)).mean())/np.mean(obs_pr_d_avg)

#%%
# Set data
df = pd.DataFrame({
'model': ['WRF_d03','WRF_d02','CanESM2','CanRCM4','PCIC'],
't ': [wrf_d03_t_NRMSE,wrf_d02_t_NRMSE,raw_t_NRMSE,rcm_t_NRMSE,pcic_t_NRMSE],
'tmax': [wrf_d03_tmax_NRMSE,wrf_d02_tmax_NRMSE,raw_tmax_NRMSE,rcm_tmax_NRMSE,pcic_tmax_NRMSE],
'pr (Y)': [wrf_d03_pr_NRMSE,wrf_d02_pr_NRMSE,raw_pr_NRMSE,rcm_pr_NRMSE,pcic_pr_NRMSE],
'pr (D)': [wrf_d03_pr_d_NRMSE,wrf_d02_pr_d_NRMSE,raw_pr_d_NRMSE,rcm_pr_d_NRMSE,pcic_pr_d_NRMSE]

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


