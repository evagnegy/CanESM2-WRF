import pandas as pd
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

variable = 'tmax' 
run = 'historical' #historical rcp45 or rcp85
output_freq = "daily" #yearly monthly or daily
#%%

eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
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


#if variable == "t":
#    label="Temperature [deg C]"
#elif variable == "pr":
#    label="Total Annual Precipitation [mm]"

#%%

if run == 'historical': #no station obs for rcps
    eccc_obs = get_eccc_obs(output_freq,eccc_station_IDs,stations_dir,variable)
    bch_obs = get_bch_obs(output_freq,bch_station_IDs,stations_dir,variable)
    wrf_d02_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
    wrf_d02_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
    raw_bch = get_canesm2(output_freq, "BCH", bch_station_IDs, run, variable, raw_files_dir,start_year)
    rcm_bch = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, variable, rcm_files_dir)
    pcic_bch = get_pcic(output_freq, "BCH", bch_station_IDs, run, variable, pcic_files_dir)
    raw_eccc = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, variable, raw_files_dir,start_year)
    rcm_eccc = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, variable, rcm_files_dir)
    pcic_eccc = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, variable, pcic_files_dir)


wrf_d03_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
wrf_d03_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, variable, WRF_files_dir,start_year)

#%%
for i in eccc_station_IDs:
    if pcic_eccc[i].sum() == 0:
        pcic_eccc.loc[:,i] = np.nan
    
#%%

perc = 0.99

if run == 'historical':
    bch_obs_perc = bch_obs.quantile(perc)
    eccc_obs_perc = eccc_obs.quantile(perc)
    wrf_d02_bch_perc = wrf_d02_bch.quantile(perc)
    raw_bch_perc = raw_bch.quantile(perc)
    rcm_bch_perc = rcm_bch.quantile(perc)
    pcic_bch_perc = pcic_bch.quantile(perc)
    wrf_d02_eccc_perc = wrf_d02_eccc.quantile(perc)
    raw_eccc_perc = raw_eccc.quantile(perc)
    rcm_eccc_perc = rcm_eccc.quantile(perc)
    pcic_eccc_perc = pcic_eccc.quantile(perc)

wrf_d03_bch_perc = wrf_d03_bch.quantile(perc)
wrf_d03_eccc_perc = wrf_d03_eccc.quantile(perc)


#%%

def count_true_occurances(df,days):
    result_df = pd.DataFrame()
    result_df_days = pd.DataFrame()
    
    for col in df.columns:
        cum_sum = 0
        result = []
        hw_lengths = []
        
        for i in range(len(df[col])-1):
            
            item = df[col][i]
            item_nxt = df[col][i+1]
            
            if item:
                cum_sum += 1
                if cum_sum >= 3 and item_nxt==False:
                    hw_lengths.append(cum_sum)
            else:
                cum_sum = 0
            result.append(cum_sum)
        
        result_df[f'cumsum_{col}'] = result
        result_df_days[f'avg_days_{col}'] = [round(np.mean(hw_lengths),2)]
    
    count_occurances = result_df.applymap(lambda x: x == days).sum()

        
    return count_occurances,np.squeeze(result_df_days.T)

if run == 'historical':
    bch_obs_hw,bch_obs_hw_days = count_true_occurances(bch_obs > bch_obs_perc,3)
    eccc_obs_hw,eccc_obs_hw_days = count_true_occurances(eccc_obs > eccc_obs_perc,3)
    wrf_d02_bch_hw,wrf_d02_bch_hw_days = count_true_occurances(wrf_d02_bch > wrf_d02_bch_perc,3)
    raw_bch_hw,raw_bch_hw_days = count_true_occurances(raw_bch > raw_bch_perc,3)
    rcm_bch_hw,rcm_bch_hw_days = count_true_occurances(rcm_bch > rcm_bch_perc,3)
    pcic_bch_hw,pcic_bch_hw_days = count_true_occurances(pcic_bch > pcic_bch_perc,3)
    wrf_d02_eccc_hw,wrf_d02_eccc_hw_days = count_true_occurances(wrf_d02_eccc > wrf_d02_eccc_perc,3)
    raw_eccc_hw,raw_eccc_hw_days = count_true_occurances(raw_eccc > raw_eccc_perc,3)
    rcm_eccc_hw,rcm_eccc_hw_days = count_true_occurances(rcm_eccc > rcm_eccc_perc,3)
    pcic_eccc_hw,pcic_eccc_hw_days= count_true_occurances(pcic_eccc > pcic_eccc_perc,3)


wrf_d03_bch_hw,wrf_d03_bch_hw_days = count_true_occurances(wrf_d03_bch > wrf_d03_bch_perc,3)
wrf_d03_eccc_hw,wrf_d03_eccc_hw_days = count_true_occurances(wrf_d03_eccc > wrf_d03_eccc_perc,3)

#%%

if run == 'historical':
    wrf_d02_bch_hw_bias = wrf_d02_bch_hw - bch_obs_hw
    raw_bch_hw_bias = raw_bch_hw - bch_obs_hw
    rcm_bch_hw_bias = rcm_bch_hw - bch_obs_hw
    pcic_bch_hw_bias = pcic_bch_hw - bch_obs_hw
    wrf_d02_eccc_hw_bias = wrf_d02_eccc_hw - eccc_obs_hw
    raw_eccc_hw_bias = raw_eccc_hw - eccc_obs_hw
    rcm_eccc_hw_bias = rcm_eccc_hw - eccc_obs_hw
    pcic_eccc_hw_bias = pcic_eccc_hw - eccc_obs_hw

    wrf_d02_bch_hw_days_bias = wrf_d02_bch_hw_days - bch_obs_hw_days
    raw_bch_hw_days_bias = raw_bch_hw_days - bch_obs_hw_days
    rcm_bch_hw_days_bias = rcm_bch_hw_days - bch_obs_hw_days
    pcic_bch_hw_days_bias = pcic_bch_hw_days - bch_obs_hw_days
    wrf_d02_eccc_hw_days_bias = wrf_d02_eccc_hw_days - eccc_obs_hw_days
    raw_eccc_hw_days_bias = raw_eccc_hw_days - eccc_obs_hw_days
    rcm_eccc_hw_days_bias = rcm_eccc_hw_days - eccc_obs_hw_days
    pcic_eccc_hw_days_bias = pcic_eccc_hw_days - eccc_obs_hw_days
    
wrf_d03_bch_hw_bias = wrf_d03_bch_hw - bch_obs_hw
wrf_d03_eccc_hw_bias = wrf_d03_eccc_hw - eccc_obs_hw

wrf_d03_bch_hw_days_bias = wrf_d03_bch_hw_days - bch_obs_hw_days
wrf_d03_eccc_hw_days_bias = wrf_d03_eccc_hw_days - eccc_obs_hw_days

#%%

for i in eccc_station_IDs:
    if pcic_eccc[i].sum() == 0:
        pcic_eccc_hw_bias.loc['cumsum_' + str(i)] = np.nan
        pcic_eccc_hw.loc['cumsum_' + str(i)] = np.nan
        
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

#%%

def plot_hw(eccc,bch,fig_name,title,vmin,vmax,cmap, title2, xlabel):
    fig,ax = plot_zoomed_in(lon_d02,lat_d02,topo_d02,lon_d03,lat_d03,topo_d03)
    
    plt.scatter(eccc_lons, eccc_lats, c=eccc,s=500,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    plt.scatter(bch_lons, bch_lats, c=bch,s=400,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3,marker='s')
    
    df_allstations = pd.concat([abs(eccc),abs(bch)])
    avg_allstations = round(df_allstations.mean())
    
    plt.title(title + ': ' + str(avg_allstations) + ' ' + title2,fontsize=20)

    cbar_ax = fig.add_axes([0.15, 0.08, 0.73, 0.03])
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),cax=cbar_ax, orientation='horizontal',extend='both')
    cbar_ax.tick_params(labelsize=14)
    cbar_ax.set_xlabel(xlabel,size=15) 
    
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/heatwaves/' + run + '/' + fig_name + '.png',bbox_inches='tight')

#%%

percentile = str(perc*100)[:2]

min_ = 0
max_ = 45
cbar = 'jet'
figname = percentile + '_def_' + run
xlabel = percentile + 'th perc. Tmax (deg C)'
title2 = 'deg C avg for ' + str(start_year) + '-' + str(end_year)

if run == 'historical':
    plot_hw(eccc_obs_perc,bch_obs_perc,'obs_stations_' + figname, 'Obs stations',min_,max_,cbar,title2,xlabel)
    plot_hw(wrf_d02_eccc_perc,wrf_d02_bch_perc,'canesm2_wrf_d02_' + figname, 'CanESM2-WRF D02',min_,max_,cbar,title2,xlabel)
    plot_hw(raw_eccc_perc,raw_bch_perc,'canesm2_raw_' + figname, 'CanESM2',min_,max_,cbar,title2,xlabel)
    plot_hw(rcm_eccc_perc,rcm_bch_perc,'canrcm4_' + figname, 'CanRCM4',min_,max_,cbar,title2,xlabel)
    plot_hw(pcic_eccc_perc,pcic_bch_perc,'pcic_' + figname, 'PCIC (CanESM2)',min_,max_,cbar,title2,xlabel)

plot_hw(wrf_d03_eccc_perc,wrf_d03_bch_perc,'canesm2_wrf_d03_' + figname, 'CanESM2-WRF D03',min_,max_,cbar,title2,xlabel)



#%%

percentile = '99'

min_ = 0
max_ = 20
cbar = 'jet'
figname = percentile + '_' + run
xlabel = 'Heatwaves (>=3 days ' + percentile + 'th perc. Tmax)'
title2 = 'Heatwave Events in ' + str(start_year) + '-' + str(end_year)

if run == 'historical':
    plot_hw(eccc_obs_hw,bch_obs_hw,'obs_stations_' + figname, 'Obs stations',min_,max_,cbar,title2,xlabel)
    plot_hw(wrf_d02_eccc_hw,wrf_d02_bch_hw,'canesm2_wrf_d02_' + figname, 'CanESM2-WRF D02',min_,max_,cbar,title2,xlabel)
    plot_hw(raw_eccc_hw,raw_bch_hw,'canesm2_raw_' + figname, 'CanESM2',min_,max_,cbar,title2,xlabel)
    plot_hw(rcm_eccc_hw,rcm_bch_hw,'canrcm4_' + figname, 'CanRCM4',min_,max_,cbar,title2,xlabel)
    plot_hw(pcic_eccc_hw,pcic_bch_hw,'pcic_' + figname, 'PCIC (CanESM2)',min_,max_,cbar,title2,xlabel)

plot_hw(wrf_d03_eccc_hw,wrf_d03_bch_hw,'canesm2_wrf_d03_' + figname, 'CanESM2-WRF D03',min_,max_,cbar,title2,xlabel)

#%%

min_ = 0
max_ = 10
cbar = 'jet'
figname =  percentile + '_' + run + '_avgdays'
xlabel = 'Avg Heatwaves (>=3 days ' + percentile + 'th perc. Tmax) Days'
title2 = 'Heatwave Avg Length in ' + str(start_year) + '-' + str(end_year)

if run == 'historical':
    plot_hw(eccc_obs_hw_days,bch_obs_hw_days,'obs_stations_' + figname, 'Obs stations',min_,max_,cbar,title2,xlabel)
    plot_hw(wrf_d02_eccc_hw_days,wrf_d02_bch_hw_days,'canesm2_wrf_d02_' + figname, 'CanESM2-WRF D02',min_,max_,cbar,title2,xlabel)
    plot_hw(raw_eccc_hw_days,raw_bch_hw_days,'canesm2_raw_' + figname, 'CanESM2',min_,max_,cbar,title2,xlabel)
    plot_hw(rcm_eccc_hw_days,rcm_bch_hw_days,'canrcm4_' + figname, 'CanRCM4',min_,max_,cbar,title2,xlabel)
    plot_hw(pcic_eccc_hw_days,pcic_bch_hw_days,'pcic_' + figname, 'PCIC (CanESM2)',min_,max_,cbar,title2,xlabel)

plot_hw(wrf_d03_eccc_hw_days,wrf_d03_bch_hw_days,'canesm2_wrf_d03_' + figname, 'CanESM2-WRF D03',min_,max_,cbar,title2,xlabel)

#%%
min_ = -20
max_ = 20
cbar = 'bwr'
figname = percentile + '_' + run + '_bias'
xlabel = 'Heatwaves (>=3 days ' + percentile + 'th perc. Tmax) Event Bias'
title2 = 'Heatwave Event bias in ' + str(start_year) + '-' + str(end_year)

plot_hw(wrf_d02_eccc_hw_bias,wrf_d02_bch_hw_bias,'canesm2_wrf_d02_' + figname, 'CanESM2-WRF D02',min_,max_,cbar,title2,xlabel)
plot_hw(wrf_d03_eccc_hw_bias,wrf_d03_bch_hw_bias,'canesm2_wrf_d03_' + figname, 'CanESM2-WRF D03',min_,max_,cbar,title2,xlabel)
plot_hw(raw_eccc_hw_bias,raw_bch_hw_bias,'canesm2_raw_' + figname, 'CanESM2',min_,max_,cbar,title2,xlabel)
plot_hw(rcm_eccc_hw_bias,rcm_bch_hw_bias,'canrcm4_' + figname, 'CanRCM4',min_,max_,cbar,title2,xlabel)
plot_hw(pcic_eccc_hw_bias,pcic_bch_hw_bias,'pcic_' + figname, 'PCIC (CanESM2)',min_,max_,cbar,title2,xlabel)

#%%

min_ = -5
max_ = 5
cbar = 'bwr'
figname = percentile + '_hist_days_bias'
xlabel = 'Heatwaves (>=3 days ' + percentile + 'th perc. Tmax) Days Bias'
title2 = 'Heatwave Days bias in ' + str(start_year) + '-' + str(end_year)

plot_hw(wrf_d02_eccc_hw_days_bias,wrf_d02_bch_hw_days_bias,'canesm2_wrf_d02_' + figname, 'CanESM2-WRF D02',min_,max_,cbar,title2,xlabel)
plot_hw(wrf_d03_eccc_hw_days_bias,wrf_d03_bch_hw_days_bias,'canesm2_wrf_d03_' + figname, 'CanESM2-WRF D03',min_,max_,cbar,title2,xlabel)
plot_hw(raw_eccc_hw_days_bias,raw_bch_hw_days_bias,'canesm2_raw_' + figname, 'CanESM2',min_,max_,cbar,title2,xlabel)
plot_hw(rcm_eccc_hw_days_bias,rcm_bch_hw_days_bias,'canrcm4_' + figname, 'CanRCM4',min_,max_,cbar,title2,xlabel)
plot_hw(pcic_eccc_hw_days_bias,pcic_bch_hw_days_bias,'pcic_' + figname, 'PCIC (CanESM2)',min_,max_,cbar,title2,xlabel)

