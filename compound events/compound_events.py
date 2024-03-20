
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys
import numpy as np
import matplotlib.ticker as ticker
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic

output_freq = "daily" #yearly monthly or daily
#%%

eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'
noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations.csv'

df = pd.read_csv(eccc_daily_stations,header=None)
eccc_station_IDs = list(df.iloc[:,4])
eccc_station_names = list(df.iloc[:,1])

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])

df = pd.read_csv(noaa_daily_stations)
noaa_station_IDs = list(df.iloc[:,0])
noaa_station_names = list(df.iloc[:,1])
#%%

stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/'
WRF_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/' 

wrf_d03_noaa_tx_hist = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'historical', "tmax", WRF_files_dir + 'historical/',1986)
wrf_d03_noaa_tx_rcp45 = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'rcp45', "tmax", WRF_files_dir + 'rcp45/',2046)

wrf_d03_noaa_tn_hist = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'historical', "tmin", WRF_files_dir + 'historical/',1986)
wrf_d03_noaa_tn_rcp45 = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'rcp45', "tmin", WRF_files_dir + 'rcp45/',2046)

wrf_d03_noaa_pr_hist = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'historical', "pr", WRF_files_dir + 'historical/',1986)
wrf_d03_noaa_pr_rcp45 = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'rcp45', "pr", WRF_files_dir + 'rcp45/',2046)

#%%
station_ID = 'USC00451400'

data_hist = pd.DataFrame({'Tmin': wrf_d03_noaa_tn_hist[station_ID].values, 'Tmax': wrf_d03_noaa_tx_hist[station_ID].values, 'Precipitation': wrf_d03_noaa_pr_hist[station_ID]}, index=wrf_d03_noaa_pr_hist[station_ID].index)
data_rcp45 = pd.DataFrame({'Tmin': wrf_d03_noaa_tn_rcp45[station_ID].values, 'Tmax': wrf_d03_noaa_tx_rcp45[station_ID].values, 'Precipitation': wrf_d03_noaa_pr_rcp45[station_ID]}, index=wrf_d03_noaa_pr_rcp45[station_ID].index)

t_perc_max = 90
t_perc_min = 10

pr_perc_limit = 75

window_size_t = 5  
window_size_pr = 29  

window_half_t = int((window_size_t-1)/2)
window_half_pr = int((window_size_pr-1)/2)  

daysinyear = 366 

t_warmday_percentile,t_coldday_percentile,t_warmnight_percentile,t_coldnight_percentile=[],[],[],[]


for i in range(1,window_half_t+1):
    dayofyear_values = data_hist[(data_hist.index.dayofyear >= daysinyear-window_half_t+(i-1)) | (data_hist.index.dayofyear <= i+window_half_t)]
    t_warmday_percentile.append(np.percentile(dayofyear_values['Tmax'],t_perc_max))
    t_coldday_percentile.append(np.percentile(dayofyear_values['Tmax'],t_perc_min))
    t_warmnight_percentile.append(np.percentile(dayofyear_values['Tmin'],t_perc_max))
    t_coldnight_percentile.append(np.percentile(dayofyear_values['Tmin'],t_perc_min))

for i in range(window_half_t+1,daysinyear-2):
    dayofyear_values = data_hist[(data_hist.index.dayofyear >= i-window_half_t) & (data_hist.index.dayofyear <= i+window_half_t)]
    t_warmday_percentile.append(np.percentile(dayofyear_values['Tmax'],t_perc_max))
    t_coldday_percentile.append(np.percentile(dayofyear_values['Tmax'],t_perc_min))
    t_warmnight_percentile.append(np.percentile(dayofyear_values['Tmin'],t_perc_max))
    t_coldnight_percentile.append(np.percentile(dayofyear_values['Tmin'],t_perc_min))

j = window_half_t+1
for i in range(daysinyear-2,daysinyear+1): 
    j += -1
    dayofyear_values = data_hist[(data_hist.index.dayofyear >= i-window_half_t) | (data_hist.index.dayofyear <= (3-j))]
    t_warmday_percentile.append(np.percentile(dayofyear_values['Tmax'],t_perc_max))
    t_coldday_percentile.append(np.percentile(dayofyear_values['Tmax'],t_perc_min))
    t_warmnight_percentile.append(np.percentile(dayofyear_values['Tmin'],t_perc_max))
    t_coldnight_percentile.append(np.percentile(dayofyear_values['Tmin'],t_perc_min))


#%%
    
pr_percentile = []

pr_filtered_hist = data_hist[data_hist['Precipitation'] > 0.1]['Precipitation']
pr_filtered_hist = pr_filtered_hist.reindex(data_hist.index) #add the NaN dates back in

for i in range(1,window_half_pr+1):
    dayofyear_values = pr_filtered_hist[(pr_filtered_hist.index.dayofyear >= daysinyear-window_half_pr+(i-1)) | (pr_filtered_hist.index.dayofyear <= i+window_half_pr)]
    pr_percentile.append(np.nanpercentile(dayofyear_values,pr_perc_limit))
for i in range(window_half_pr+1,daysinyear-2):
    dayofyear_values = pr_filtered_hist[(pr_filtered_hist.index.dayofyear >= i-window_half_pr) & (pr_filtered_hist.index.dayofyear <= i+window_half_pr)]
    pr_percentile.append(np.nanpercentile(dayofyear_values,pr_perc_limit))
j = window_half_pr+1
for i in range(daysinyear-2,daysinyear+1): 
    j += -1
    dayofyear_values = pr_filtered_hist[(pr_filtered_hist.index.dayofyear >= i-window_half_pr) | (pr_filtered_hist.index.dayofyear <= (3-j))]
    pr_percentile.append(np.nanpercentile(dayofyear_values,pr_perc_limit))
     
#%%

t_warmday_percentiles_long,t_coldday_percentiles_long,t_warmnight_percentiles_long,t_coldnight_percentiles_long=[],[],[],[]
pr_percentiles_long = []
   
for date in data_hist.index:
    for i in range(1,daysinyear+1):
        if date.dayofyear == i:
            t_warmday_percentiles_long.append(t_warmday_percentile[i-1])
            t_coldday_percentiles_long.append(t_coldday_percentile[i-1])
            t_warmnight_percentiles_long.append(t_warmnight_percentile[i-1])
            t_coldnight_percentiles_long.append(t_coldnight_percentile[i-1])
            pr_percentiles_long.append(pr_percentile[i-1])

#%%


warmday_wet = data_hist[(data_hist['Tmax'] > t_warmday_percentiles_long) & (data_hist['Precipitation'] > pr_percentiles_long)]
coldday_wet = data_hist[(data_hist['Tmax'] < t_coldday_percentiles_long) & (data_hist['Precipitation'] > pr_percentiles_long)]
warmnight_wet = data_hist[(data_hist['Tmin'] > t_warmnight_percentiles_long) & (data_hist['Precipitation'] > pr_percentiles_long)]
coldnight_wet = data_hist[(data_hist['Tmin'] < t_coldnight_percentiles_long) & (data_hist['Precipitation'] > pr_percentiles_long)]

warmday_dry = data_hist[(data_hist['Tmax'] > t_warmday_percentiles_long) & (data_hist['Precipitation'] < 1)]
coldday_dry = data_hist[(data_hist['Tmax'] < t_coldday_percentiles_long) & (data_hist['Precipitation'] < 1)]
warmnight_dry = data_hist[(data_hist['Tmin'] > t_warmnight_percentiles_long) & (data_hist['Precipitation'] < 1)]
coldnight_dry = data_hist[(data_hist['Tmin'] < t_coldnight_percentiles_long) & (data_hist['Precipitation'] < 1)]

count_warmday_wet_peryear = len(warmday_wet)/20
count_coldday_wet_peryear = len(coldday_wet)/20
count_warmnight_wet_peryear = len(warmnight_wet)/20
count_coldnight_wet_peryear = len(coldnight_wet)/20

count_warmday_dry_peryear = len(warmday_dry)/20
count_coldday_dry_peryear = len(coldday_dry)/20
count_warmnight_dry_peryear = len(warmnight_dry)/20
count_coldnight_dry_peryear = len(coldnight_dry)/20

#%%


fig, ax = plt.subplots(figsize=(20, 5))

ax.plot(wrf_d03_noaa_t[station_ID].index.to_timestamp(),wrf_d03_noaa_t[station_ID].values,label="t",color="k",alpha=0.1)

y1, y2 = ax.get_ylim()
ax.fill_between(wrf_d03_noaa_t[station_ID].index.to_timestamp(), y1,y2, where=wrf_d03_noaa_t[station_ID] > t_95,
                 color='red', alpha=0.6, label='Above 95th percentile')

ax.set_ylabel("Temperature (deg C)",fontsize=12)

ax2 = ax.twinx()
ax2.plot(wrf_d03_noaa_pr[station_ID].index.to_timestamp(),wrf_d03_noaa_pr[station_ID].values,label="pr",color="C1",alpha=0.1)
y1, y2 = ax2.get_ylim()
ax2.fill_between(wrf_d03_noaa_pr[station_ID].index.to_timestamp(), y1,y2, where=wrf_d03_noaa_pr[station_ID] < pr_5,
                 color='blue', alpha=0.6, label='Above 95th percentile')

#plt.legend()        
ax2.set_ylabel("Precipitation (mm/day)",fontsize=12)


#plt.title(station_name + " observation station",fontsize=14)# + ": " + str(station_ID))
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))


