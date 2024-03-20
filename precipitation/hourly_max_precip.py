
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

variable = 'pr' #t or pr
output_freq = "hourly" #yearly monthly or daily (hourly for wrf)
#%%

eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'

df = pd.read_csv(eccc_daily_stations,header=None)
eccc_station_IDs = list(df.iloc[:,4])
eccc_station_names = list(df.iloc[:,1])

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])

all_stations = np.hstack((bch_station_IDs,eccc_station_IDs))
all_stations = [int(item) if item.isdigit() else item for item in all_stations]


WRF_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/'


#%%

wrf_d03_bch_hist = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'historical', variable, WRF_files_dir + 'historical/',1986)
wrf_d03_eccc_hist = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'historical', variable, WRF_files_dir + 'historical/',1986)

wrf_d03_bch_rcp45 = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'rcp45', variable, WRF_files_dir + 'rcp45/',2046)
wrf_d03_eccc_rcp45 = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'rcp45', variable, WRF_files_dir + 'rcp45/',2046)

wrf_d03_bch_rcp85 = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'rcp85', variable, WRF_files_dir + 'rcp85/',2046)
wrf_d03_eccc_rcp85 = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'rcp85', variable, WRF_files_dir + 'rcp85/',2046)

#%%
wrf_d03_bch_hist_wet = wrf_d03_bch_hist.copy()
wrf_d03_eccc_hist_wet = wrf_d03_eccc_hist.copy()
wrf_d03_bch_rcp45_wet = wrf_d03_bch_rcp45.copy()
wrf_d03_eccc_rcp45_wet = wrf_d03_eccc_rcp45.copy()
wrf_d03_bch_rcp85_wet = wrf_d03_bch_rcp85.copy()
wrf_d03_eccc_rcp85_wet = wrf_d03_eccc_rcp85.copy()

for i in [4,5,6,7,8,9]:
    wrf_d03_bch_hist_wet = wrf_d03_bch_hist_wet[wrf_d03_bch_hist_wet.index.month != i]
    wrf_d03_eccc_hist_wet = wrf_d03_eccc_hist_wet[wrf_d03_eccc_hist_wet.index.month != i]
    wrf_d03_bch_rcp45_wet = wrf_d03_bch_rcp45_wet[wrf_d03_bch_rcp45_wet.index.month != i]
    wrf_d03_eccc_rcp45_wet = wrf_d03_eccc_rcp45_wet[wrf_d03_eccc_rcp45_wet.index.month != i]
    wrf_d03_bch_rcp85_wet = wrf_d03_bch_rcp85_wet[wrf_d03_bch_rcp85_wet.index.month != i]
    wrf_d03_eccc_rcp85_wet = wrf_d03_eccc_rcp85_wet[wrf_d03_eccc_rcp85_wet.index.month != i]

#%%
perc = 1

wrf_d03_hist_wet = pd.concat([wrf_d03_bch_hist_wet,wrf_d03_eccc_hist_wet],axis=1)
wrf_d03_rcp45_wet = pd.concat([wrf_d03_bch_rcp45_wet,wrf_d03_eccc_rcp45_wet],axis=1)
wrf_d03_rcp85_wet = pd.concat([wrf_d03_bch_rcp85_wet,wrf_d03_eccc_rcp85_wet],axis=1)

wrf_d03_hist_wet = wrf_d03_hist_wet.drop(columns='year')
wrf_d03_rcp45_wet = wrf_d03_rcp45_wet.drop(columns='year')
wrf_d03_rcp85_wet = wrf_d03_rcp85_wet.drop(columns='year')

wrf_d03_hist_wet_avg = wrf_d03_hist_wet.quantile(perc,axis=1)
wrf_d03_rcp45_wet_avg = wrf_d03_rcp45_wet.quantile(perc,axis=1)
wrf_d03_rcp85_wet_avg = wrf_d03_rcp85_wet.quantile(perc,axis=1)


wrf_d03_bch_hist_wet['year'] = wrf_d03_bch_hist_wet.index.year
wrf_d03_bch_hist_wet_perc = wrf_d03_bch_hist_wet.groupby('year').quantile(perc)
wrf_d03_bch_rcp45_wet['year'] = wrf_d03_bch_rcp45_wet.index.year
wrf_d03_bch_rcp45_wet_perc = wrf_d03_bch_rcp45_wet.groupby('year').quantile(perc)
wrf_d03_bch_rcp85_wet['year'] = wrf_d03_bch_rcp85_wet.index.year
wrf_d03_bch_rcp85_wet_perc = wrf_d03_bch_rcp85_wet.groupby('year').quantile(perc)

wrf_d03_eccc_hist_wet['year'] = wrf_d03_eccc_hist_wet.index.year
wrf_d03_eccc_hist_wet_perc = wrf_d03_eccc_hist_wet.groupby('year').quantile(perc)
wrf_d03_eccc_rcp45_wet['year'] = wrf_d03_eccc_rcp45_wet.index.year
wrf_d03_eccc_rcp45_wet_perc = wrf_d03_eccc_rcp45_wet.groupby('year').quantile(perc)
wrf_d03_eccc_rcp85_wet['year'] = wrf_d03_eccc_rcp85_wet.index.year
wrf_d03_eccc_rcp85_wet_perc = wrf_d03_eccc_rcp85_wet.groupby('year').quantile(perc)

wrf_d03_hist_wet_perc = pd.concat([wrf_d03_bch_hist_wet_perc,wrf_d03_eccc_hist_wet_perc],axis=1)
wrf_d03_rcp45_wet_perc = pd.concat([wrf_d03_bch_rcp45_wet_perc,wrf_d03_eccc_rcp45_wet_perc],axis=1)
wrf_d03_rcp85_wet_perc = pd.concat([wrf_d03_bch_rcp85_wet_perc,wrf_d03_eccc_rcp85_wet_perc],axis=1)

hist_mean = round(wrf_d03_hist_wet_perc.mean().mean(),1)
rcp45_mean = round(wrf_d03_rcp45_wet_perc.mean().mean(),1)
rcp85_mean = round(wrf_d03_rcp85_wet_perc.mean().mean(),1)

#%%

fig, ax = plt.subplots(figsize=(11, 5))

for station_ID in all_stations:     
    
    plt.plot(wrf_d03_hist_wet_perc[station_ID].values,color="C0",alpha=0.3,linestyle='--')
    plt.plot(wrf_d03_rcp45_wet_perc[station_ID].values,color="C1",alpha=0.3,linestyle='--')
    plt.plot(wrf_d03_rcp85_wet_perc[station_ID].values,color="C2",alpha=0.3,linestyle='--')

plt.plot(wrf_d03_hist_wet_perc.mean(axis=1).values,label="Historical (avg: " + str(hist_mean) + ")",color="C0",linewidth=3)
plt.plot(wrf_d03_rcp45_wet_perc.mean(axis=1).values,label="RCP4.5 (avg: " + str(rcp45_mean) + ")",color="C1",linewidth=3)
plt.plot(wrf_d03_rcp85_wet_perc.mean(axis=1).values,label="RCP8.5 (avg: " + str(rcp85_mean) + ")",color="C2",linewidth=3)  

plt.legend(loc='upper right')

if perc != 1:
    plt.ylabel(str(perc*100)[:2] + "th Percentile for Hourly Precip [mm/hr]",fontsize=12)
else:
    plt.ylabel("Max value for Hourly Precip [mm/hr]",fontsize=12)


plt.xlabel("year from start_year",fontsize=12)

plt.title("Wet season (Oct-Mar) for all stations (dashed), with averages (solid)", fontsize=14)# + ": " + str(station_ID))


for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))


plt.xlim([0,19])

#plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/indiv_stations/timeseries/historical/' + variable + '/' + output_freq + '/' + agency + '_' + station_name + "_" + str(station_ID) + "_" + variable,bbox_inches='tight',dpi=62)


        
#%%  histograms

fig, ax = plt.subplots(figsize=(7, 4))

bins=25
   
plt.hist(wrf_d03_hist_wet_avg,bins,label="Historical",color="C0",density=True,histtype='step',linewidth=4)
plt.hist(wrf_d03_rcp45_wet_avg,bins,label="RCP4.5",color="C1",density=True,histtype='step',linewidth=3)
plt.hist(wrf_d03_rcp85_wet_avg,bins,label="RCP8.5",color="C2",density=True,histtype='step',linewidth=2)

plt.ylabel('PDF',fontsize=12)

if perc != 1:
    plt.xlabel(str(perc*100)[:2] + "th Percentile across all stations for hourly precip [mm/hr]",fontsize=12)
else:
    plt.xlabel("Max value across all stations for hourly precip [mm/hr]",fontsize=12)

plt.ylim(ymin=-0.01)

plt.legend(loc='upper right')

