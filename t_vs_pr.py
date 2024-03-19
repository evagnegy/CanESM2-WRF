import pandas as pd
import warnings
import sys
import numpy as np
from netCDF4 import Dataset
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_noaa_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic,plot_all_d03,plot_zoomed_in
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib
from scipy.stats import linregress
import math

output_freq = "yearly" #yearly monthly or daily
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
noaa_station_IDs = list(df["STATION"])
noaa_station_names = list(df["NAME"])

#%%

WRF_files_his = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/historical/'
WRF_files_rcp45 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/rcp45/'
WRF_files_rcp85 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/rcp85/'

#%%

wrf_d03_bch_his_pr = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'historical', 'pr', WRF_files_his,1986)
wrf_d03_eccc_his_pr = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'historical', 'pr', WRF_files_his,1986)
wrf_d03_noaa_his_pr = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'historical', 'pr', WRF_files_his,1986)

wrf_d03_bch_his_t = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'historical', 't', WRF_files_his,1986)
wrf_d03_eccc_his_t = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'historical', 't', WRF_files_his,1986)
wrf_d03_noaa_his_t = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'historical', 't', WRF_files_his,1986)

wrf_d03_bch_rcp45_pr = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'rcp45', 'pr', WRF_files_rcp45,2046)
wrf_d03_eccc_rcp45_pr = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'rcp45', 'pr', WRF_files_rcp45,2046)
wrf_d03_noaa_rcp45_pr = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'rcp45', 'pr', WRF_files_rcp45,2046)

wrf_d03_bch_rcp45_t = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'rcp45', 't', WRF_files_rcp45,2046)
wrf_d03_eccc_rcp45_t = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'rcp45', 't', WRF_files_rcp45,2046)
wrf_d03_noaa_rcp45_t = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'rcp45', 't', WRF_files_rcp45,2046)

wrf_d03_bch_rcp85_pr = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'rcp85', 'pr', WRF_files_rcp85,2046)
wrf_d03_eccc_rcp85_pr = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'rcp85', 'pr', WRF_files_rcp85,2046)
wrf_d03_noaa_rcp85_pr = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'rcp85', 'pr', WRF_files_rcp85,2046)

wrf_d03_bch_rcp85_t = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'rcp85', 't', WRF_files_rcp85,2046)
wrf_d03_eccc_rcp85_t = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'rcp85', 't', WRF_files_rcp85,2046)
wrf_d03_noaa_rcp85_t = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", 'rcp85', 't', WRF_files_rcp85,2046)

#%%
stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/'
noaa_obs = get_noaa_obs(output_freq,noaa_station_IDs,stations_dir,"t")

# remove stations not in the original list
for station in noaa_station_IDs:
    if station not in list(noaa_obs.columns):
        wrf_d03_noaa_his_pr.drop(station, inplace=True, axis=1)
        wrf_d03_noaa_his_t.drop(station, inplace=True, axis=1)
        wrf_d03_noaa_rcp45_pr.drop(station, inplace=True, axis=1)
        wrf_d03_noaa_rcp45_t.drop(station, inplace=True, axis=1)
        wrf_d03_noaa_rcp85_pr.drop(station, inplace=True, axis=1)
        wrf_d03_noaa_rcp85_t.drop(station, inplace=True, axis=1)

#%%

wrf_d03_eccc_his_pr_avg = wrf_d03_eccc_his_pr.mean().sort_index()
wrf_d03_bch_his_pr_avg = wrf_d03_bch_his_pr.mean().sort_index()
wrf_d03_noaa_his_pr_avg = wrf_d03_noaa_his_pr.mean().sort_index()
wrf_d03_his_pr_avg = pd.concat([wrf_d03_eccc_his_pr_avg,wrf_d03_bch_his_pr_avg,wrf_d03_noaa_his_pr_avg])
wrf_d03_his_pr_avg.index = wrf_d03_his_pr_avg.index.astype(str)
wrf_d03_his_pr_avg = wrf_d03_his_pr_avg.sort_index()

wrf_d03_eccc_his_t_avg = wrf_d03_eccc_his_t.mean().sort_index()
wrf_d03_bch_his_t_avg = wrf_d03_bch_his_t.mean().sort_index()
wrf_d03_noaa_his_t_avg = wrf_d03_noaa_his_t.mean().sort_index()
wrf_d03_his_t_avg = pd.concat([wrf_d03_eccc_his_t_avg,wrf_d03_bch_his_t_avg,wrf_d03_noaa_his_t_avg])
wrf_d03_his_t_avg.index = wrf_d03_his_t_avg.index.astype(str)
wrf_d03_his_t_avg = wrf_d03_his_t_avg.sort_index()

wrf_d03_eccc_rcp45_pr_avg = wrf_d03_eccc_rcp45_pr.mean().sort_index()
wrf_d03_bch_rcp45_pr_avg = wrf_d03_bch_rcp45_pr.mean().sort_index()
wrf_d03_noaa_rcp45_pr_avg = wrf_d03_noaa_rcp45_pr.mean().sort_index()
wrf_d03_rcp45_pr_avg = pd.concat([wrf_d03_eccc_rcp45_pr_avg,wrf_d03_bch_rcp45_pr_avg,wrf_d03_noaa_rcp45_pr_avg])
wrf_d03_rcp45_pr_avg.index = wrf_d03_rcp45_pr_avg.index.astype(str)
wrf_d03_rcp45_pr_avg = wrf_d03_rcp45_pr_avg.sort_index()

wrf_d03_eccc_rcp45_t_avg = wrf_d03_eccc_rcp45_t.mean().sort_index()
wrf_d03_bch_rcp45_t_avg = wrf_d03_bch_rcp45_t.mean().sort_index()
wrf_d03_noaa_rcp45_t_avg = wrf_d03_noaa_rcp45_t.mean().sort_index()
wrf_d03_rcp45_t_avg = pd.concat([wrf_d03_eccc_rcp45_t_avg,wrf_d03_bch_rcp45_t_avg,wrf_d03_noaa_rcp45_t_avg])
wrf_d03_rcp45_t_avg.index = wrf_d03_rcp45_t_avg.index.astype(str)
wrf_d03_rcp45_t_avg = wrf_d03_rcp45_t_avg.sort_index()

wrf_d03_eccc_rcp85_pr_avg = wrf_d03_eccc_rcp85_pr.mean().sort_index()
wrf_d03_bch_rcp85_pr_avg = wrf_d03_bch_rcp85_pr.mean().sort_index()
wrf_d03_noaa_rcp85_pr_avg = wrf_d03_noaa_rcp85_pr.mean().sort_index()
wrf_d03_rcp85_pr_avg = pd.concat([wrf_d03_eccc_rcp85_pr_avg,wrf_d03_bch_rcp85_pr_avg,wrf_d03_noaa_rcp85_pr_avg])
wrf_d03_rcp85_pr_avg.index = wrf_d03_rcp85_pr_avg.index.astype(str)
wrf_d03_rcp85_pr_avg = wrf_d03_rcp85_pr_avg.sort_index()

wrf_d03_eccc_rcp85_t_avg = wrf_d03_eccc_rcp85_t.mean().sort_index()
wrf_d03_bch_rcp85_t_avg = wrf_d03_bch_rcp85_t.mean().sort_index()
wrf_d03_noaa_rcp85_t_avg = wrf_d03_noaa_rcp85_t.mean().sort_index()
wrf_d03_rcp85_t_avg = pd.concat([wrf_d03_eccc_rcp85_t_avg,wrf_d03_bch_rcp85_t_avg,wrf_d03_noaa_rcp85_t_avg])
wrf_d03_rcp85_t_avg.index = wrf_d03_rcp85_t_avg.index.astype(str)
wrf_d03_rcp85_t_avg = wrf_d03_rcp85_t_avg.sort_index()
#%%

plt.figure(figsize=(10, 10),dpi=200)

plt.xlabel('Precipitation (mm/year)',fontsize=18)
plt.ylabel('Temperature (deg C)',fontsize=18)
    
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

t_rcp45_change = wrf_d03_rcp45_t_avg.mean() - wrf_d03_his_t_avg.mean()
t_rcp85_change = wrf_d03_rcp85_t_avg.mean() - wrf_d03_his_t_avg.mean()

t45 = str(round(t_rcp45_change,2))
t85 = str(round(t_rcp85_change,2))
 
pr_rcp45_change = ((wrf_d03_rcp45_pr_avg.mean() - wrf_d03_his_pr_avg.mean())/wrf_d03_his_pr_avg.mean())*100
pr_rcp85_change = ((wrf_d03_rcp85_pr_avg.mean() - wrf_d03_his_pr_avg.mean())/wrf_d03_his_pr_avg.mean())*100

pr45 = str(round(pr_rcp45_change,1))
pr85 = str(round(pr_rcp85_change,1))

    
plt.scatter(wrf_d03_his_pr_avg,wrf_d03_his_t_avg,color='C0',s=200,marker='.',label='Historical')
plt.scatter(wrf_d03_rcp45_pr_avg,wrf_d03_rcp45_t_avg,color='C1',s=200,marker='.',label='RCP45 (t: ' + t45 + "ºC, pr: " + pr45 + "%)")
plt.scatter(wrf_d03_rcp85_pr_avg,wrf_d03_rcp85_t_avg,color='C2',s=200,marker='.',label='RCP85 (t: ' + t85 + "ºC, pr: " + pr85 + "%)")

plt.scatter(wrf_d03_his_pr_avg.mean(),wrf_d03_his_t_avg.mean(),s=500,marker='X',color='C0',edgecolor='k')
plt.scatter(wrf_d03_rcp45_pr_avg.mean(),wrf_d03_rcp45_t_avg.mean(),s=500,marker='X',color='C1',edgecolor='k')
plt.scatter(wrf_d03_rcp85_pr_avg.mean(),wrf_d03_rcp85_t_avg.mean(),s=500,marker='X',color='C2',edgecolor='k')




# =============================================================================
# slope, intercept, r_value, p_value, std_err = linregress(wrf_d03_his_pr_avg, wrf_d03_his_t_avg)
# x = np.linspace(0, 4000) 
# line_of_best_fit = slope * x + intercept
# plt.plot(x, line_of_best_fit,'--', color='grey')
# 
# =============================================================================
    
plt.title('Annual', fontsize=22)
plt.legend(fontsize=20,loc='lower right')

plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/t_vs_pr/t_vs_pr_annual.png',bbox_inches='tight')


    