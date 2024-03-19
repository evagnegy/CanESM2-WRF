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

variable = 'pr' #t or pr
output_freq = "daily" #yearly monthly or daily

run = 'historical'

if run == 'historical':
    start_year = 1986
    end_year = 2005
else:
    start_year = 2046
    end_year = 2065  
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

    wrf_d03_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'historical', variable, WRF_files_dir,start_year)
    wrf_d03_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'historical', variable, WRF_files_dir,start_year)


# =============================================================================
# wrf_d03_bch_rcp45 = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'rcp45', variable, WRF_files_dir + 'rcp45/',2046)
# wrf_d03_eccc_rcp45 = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'rcp45', variable, WRF_files_dir + 'rcp45/',2046)
# 
# wrf_d03_bch_rcp85 = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", 'rcp85', variable, WRF_files_dir + 'rcp85/',2046)
# wrf_d03_eccc_rcp85 = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", 'rcp85', variable, WRF_files_dir + 'rcp85/',2046)
# =============================================================================
#wrf_d03_bch_maxday = wrf_d03_bch.groupby(pd.PeriodIndex(wrf_d03_bch.index, freq="D")).max()
#wrf_d03_eccc_maxday = wrf_d03_eccc.groupby(pd.PeriodIndex(wrf_d03_eccc.index, freq="D")).max()

#%%
bch_wet = bch_obs.copy()
eccc_wet = eccc_obs.copy()
wrf_d03_bch_wet = wrf_d03_bch.copy()
wrf_d03_eccc_wet = wrf_d03_eccc.copy()
wrf_d02_bch_wet = wrf_d02_bch.copy()
wrf_d02_eccc_wet = wrf_d02_eccc.copy()
raw_bch_wet = raw_bch.copy()
raw_eccc_wet = raw_eccc.copy()
rcm_bch_wet = rcm_bch.copy()
rcm_eccc_wet = rcm_eccc.copy()
pcic_bch_wet = pcic_bch.copy()
pcic_eccc_wet = pcic_eccc.copy()


for i in [4,5,6,7,8,9]:
    bch_wet = bch_wet[bch_wet.index.month != i]
    eccc_wet = eccc_wet[eccc_wet.index.month != i]
    wrf_d03_bch_wet = wrf_d03_bch_wet[wrf_d03_bch_wet.index.month != i]
    wrf_d03_eccc_wet = wrf_d03_eccc_wet[wrf_d03_eccc_wet.index.month != i]
    wrf_d02_bch_wet = wrf_d02_bch_wet[wrf_d02_bch_wet.index.month != i]
    wrf_d02_eccc_wet = wrf_d02_eccc_wet[wrf_d02_eccc_wet.index.month != i]
    raw_bch_wet = raw_bch_wet[raw_bch_wet.index.month != i]
    raw_eccc_wet = raw_eccc_wet[raw_eccc_wet.index.month != i]
    rcm_bch_wet = rcm_bch_wet[rcm_bch_wet.index.month != i]
    rcm_eccc_wet = rcm_eccc_wet[rcm_eccc_wet.index.month != i]
    pcic_bch_wet = pcic_bch_wet[pcic_bch_wet.index.month != i]
    pcic_eccc_wet = pcic_eccc_wet[pcic_eccc_wet.index.month != i]

#%%

perc = 0.95

bch_perc = bch_wet.quantile(perc)
eccc_perc = eccc_wet.quantile(perc)
wrf_d03_bch_perc = wrf_d03_bch_wet.quantile(perc)
wrf_d03_eccc_perc = wrf_d03_eccc_wet.quantile(perc)
wrf_d02_bch_perc = wrf_d02_bch_wet.quantile(perc)
wrf_d02_eccc_perc = wrf_d02_eccc_wet.quantile(perc)
raw_bch_perc = raw_bch_wet.quantile(perc)
raw_eccc_perc = raw_eccc_wet.quantile(perc)
rcm_bch_perc = rcm_bch_wet.quantile(perc)
rcm_eccc_perc = rcm_eccc_wet.quantile(perc)
pcic_bch_perc = pcic_bch_wet.quantile(perc)
pcic_eccc_perc = pcic_eccc_wet.quantile(perc)
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

def plot_pr_ext(eccc,bch,fig_name,title,vmin,vmax,cmap, title2, xlabel):
   
    fig,ax = plot_zoomed_in(lon_d02,lat_d02,topo_d02,lon_d03,lat_d03,topo_d03)
    
    plt.scatter(eccc_lons, eccc_lats, c=eccc,s=500,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    plt.scatter(bch_lons, bch_lats, c=bch,s=400,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3,marker='s')
    
    df_allstations = pd.concat([abs(eccc),abs(bch)])
    avg_allstations = round(df_allstations.mean(),1)
    
    plt.title(title + ': ' + str(avg_allstations) + ' (mm/day) ' + title2,fontsize=20)

    cbar_ax = fig.add_axes([0.15, 0.08, 0.73, 0.03])
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),cax=cbar_ax, orientation='horizontal',extend='max')
    cbar_ax.tick_params(labelsize=14)
    
    if perc != 1:
        cbar_ax.set_xlabel(str(perc*100)[:2] + "th Percentile for Daily Precip [mm/day]",size=15) 
    else:
        cbar_ax.set_xlabel("Max value for Daily Precip [mm/day]",size=15) 


    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/extreme_pr/' + fig_name + '.png',bbox_inches='tight')

#%%

percentile = str(perc*100)[:2]
min_ = 0
max_ = 75
cbar = 'jet'
figname = percentile + '_' + run
xlabel = percentile +'th Percentile for daily precipitation (mm/day)'
title2 = 'for ' + str(start_year) + '-' + str(end_year)

if run == 'historical':
    plot_pr_ext(eccc_perc,bch_perc,'obs_stations_' + figname, 'Obs stations',min_,max_,cbar,title2,xlabel)
    plot_pr_ext(wrf_d02_eccc_perc,wrf_d02_bch_perc,'canesm2_wrf_d02_' + figname, 'CanESM2-WRF D02',min_,max_,cbar,title2,xlabel)
    plot_pr_ext(raw_eccc_perc,raw_bch_perc,'canesm2_raw_' + figname, 'CanESM2',min_,max_,cbar,title2,xlabel)
    plot_pr_ext(rcm_eccc_perc,rcm_bch_perc,'canrcm4_' + figname, 'CanRCM4',min_,max_,cbar,title2,xlabel)
    plot_pr_ext(pcic_eccc_perc,pcic_bch_perc,'pcic_' + figname, 'PCIC (CanESM2)',min_,max_,cbar,title2,xlabel)

plot_pr_ext(wrf_d03_eccc_perc,wrf_d03_bch_perc,'canesm2_wrf_d03_' + figname, 'CanESM2-WRF D03',min_,max_,cbar,title2,xlabel)


# =============================================================================
# #%%
# vmin = -7
# vmax = 7
# cmap = 'bwr_r'
# 
# fig,ax = plot_zoomed_in(lon_d02,lat_d02,topo_d02,lon_d03,lat_d03,topo_d03)
# 
# plt.scatter(eccc_lons, eccc_lats, c=wrf_d03_eccc_hist_perc_bias85,s=500,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
# plt.scatter(bch_lons, bch_lats, c=wrf_d03_bch_hist_perc_bias85,s=400,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3,marker='s')
# 
# df_allstations = pd.concat([abs(wrf_d03_eccc_hist_perc_bias85),abs(wrf_d03_bch_hist_perc_bias85)])
# avg_allstations = round(df_allstations.mean(),1)
# 
# plt.title('RCP8.5-Hist (avg. across all stations: ' + str(avg_allstations) + ' [mm/hr])',fontsize=20)
# 
# cbar_ax = fig.add_axes([0.15, 0.08, 0.73, 0.03])
# fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),cax=cbar_ax, orientation='horizontal',extend='both')
# cbar_ax.tick_params(labelsize=14)
# 
# if perc != 1:
#     cbar_ax.set_xlabel('RCP8.5-Hist of ' + str(perc*100)[:2] + "th Percentile for Hourly Precip [mm/hr]",size=15) 
# else:
#     cbar_ax.set_xlabel("RCP8.5-Hist of max value for Hourly Precip [mm/hr]",size=15) 
# 
# 
# #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/' + fig_name + '.png',bbox_inches='tight')
# 
# =============================================================================
