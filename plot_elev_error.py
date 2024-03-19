#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:05:45 2024

@author: evagnegy
"""

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
import xarray as xr
from scipy.stats import linregress


variable = 'wind' #t or pr
run = 'historical' #historical rcp45 or rcp85
output_freq = "yearly" #yearly monthly or daily
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

eccc_lats = df.iloc[:,7]
eccc_lons = df.iloc[:,8]
eccc_lats.index = eccc_station_IDs
eccc_lons.index = eccc_station_IDs

eccc_elev = (df.iloc[:,11])
eccc_elev.index = eccc_station_IDs

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])

bch_lats = df['Y']
bch_lons = df['X']
bch_lats.index = bch_station_IDs
bch_lons.index = bch_station_IDs

bch_elev = (df["ELEV"])
bch_elev.index = bch_station_IDs

df = pd.read_csv(noaa_daily_stations)

noaa_station_IDs = list(df.iloc[:,0])
noaa_station_names = list(df.iloc[:,1])

noaa_lats = df.iloc[:,2]
noaa_lons = df.iloc[:,3]
noaa_lats.index = noaa_station_IDs
noaa_lons.index = noaa_station_IDs

noaa_elev = (df["ELEVATION"])
noaa_elev.index = noaa_station_IDs

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
    
wrf_d02_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_eccc = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, variable, raw_files_dir,start_year)
rcm_eccc = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, variable, rcm_files_dir)

wrf_d02_noaa = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_noaa = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_noaa = get_canesm2(output_freq, "NOAA", noaa_station_IDs, run, variable, raw_files_dir,start_year)
rcm_noaa = get_canrcm4(output_freq, "NOAA", noaa_station_IDs, run, variable, rcm_files_dir)



#%%

wrf_d02_eccc = wrf_d02_eccc.mask(eccc_obs.isnull())
wrf_d03_eccc = wrf_d03_eccc.mask(eccc_obs.isnull())
raw_eccc = raw_eccc.mask(eccc_obs.isnull())
rcm_eccc = rcm_eccc.mask(eccc_obs.isnull())

wrf_d02_noaa = wrf_d02_noaa.mask(noaa_obs.isnull())
wrf_d03_noaa = wrf_d03_noaa.mask(noaa_obs.isnull())
raw_noaa = raw_noaa.mask(noaa_obs.isnull())
rcm_noaa = rcm_noaa.mask(noaa_obs.isnull())

if variable != "wind":
    wrf_d02_bch = wrf_d02_bch.mask(bch_obs.isnull())
    wrf_d03_bch = wrf_d03_bch.mask(bch_obs.isnull())
    raw_bch = raw_bch.mask(bch_obs.isnull())
    rcm_bch = rcm_bch.mask(bch_obs.isnull())

wrf_bias_eccc_d02 = np.mean(wrf_d02_eccc) - np.mean(eccc_obs)
wrf_bias_noaa_d02 = np.mean(wrf_d02_noaa) - np.mean(noaa_obs)

wrf_bias_eccc = np.mean(wrf_d03_eccc) - np.mean(eccc_obs)
wrf_bias_noaa = np.mean(wrf_d03_noaa) - np.mean(noaa_obs)

raw_bias_eccc = np.mean(raw_eccc) - np.mean(eccc_obs)
raw_bias_noaa = np.mean(raw_noaa) - np.mean(noaa_obs)

rcm_bias_eccc = np.mean(rcm_eccc) - np.mean(eccc_obs)
rcm_bias_noaa = np.mean(rcm_noaa) - np.mean(noaa_obs)

if variable != "wind":
    wrf_bias_bch_d02 = np.mean(wrf_d02_bch) - np.mean(bch_obs)
    wrf_bias_bch = np.mean(wrf_d03_bch) - np.mean(bch_obs)
    raw_bias_bch = np.mean(raw_bch) - np.mean(bch_obs)
    rcm_bias_bch = np.mean(rcm_bch) - np.mean(bch_obs)
    
    wrf_bias_d02 = pd.concat([wrf_bias_eccc_d02,wrf_bias_bch_d02,wrf_bias_noaa_d02])
    wrf_bias_d02.index = wrf_bias_d02.index.astype(str)
    wrf_bias_d02 = wrf_bias_d02.sort_index()

    wrf_bias = pd.concat([wrf_bias_eccc,wrf_bias_bch,wrf_bias_noaa])
    wrf_bias.index = wrf_bias.index.astype(str)
    wrf_bias = wrf_bias.sort_index()

    raw_bias = pd.concat([raw_bias_eccc,raw_bias_bch,raw_bias_noaa])
    raw_bias.index = raw_bias.index.astype(str)
    raw_bias = raw_bias.sort_index()

    rcm_bias = pd.concat([rcm_bias_eccc,rcm_bias_bch,rcm_bias_noaa])
    rcm_bias.index = rcm_bias.index.astype(str)
    rcm_bias = rcm_bias.sort_index()
    
    lats = pd.concat([eccc_lats,bch_lats,noaa_lats])
    lats.index = lats.index.astype(str)
    lats = lats.sort_index()

    lons = pd.concat([eccc_lons,bch_lons,noaa_lons])
    lons.index = lons.index.astype(str)
    lons = lons.sort_index()
    
    elev = pd.concat([eccc_elev,bch_elev,noaa_elev])
    elev.index = elev.index.astype(str)
    elev = elev.sort_index()

    
else:
    wrf_bias_d02 = pd.concat([wrf_bias_eccc_d02,wrf_bias_noaa_d02])
    wrf_bias_d02.index = wrf_bias_d02.index.astype(str)
    wrf_bias_d02 = wrf_bias_d02.sort_index()
    
    wrf_bias = pd.concat([wrf_bias_eccc,wrf_bias_noaa])
    wrf_bias.index = wrf_bias.index.astype(str)
    wrf_bias = wrf_bias.sort_index()
    
    raw_bias = pd.concat([raw_bias_eccc,raw_bias_noaa])
    raw_bias.index = raw_bias.index.astype(str)
    raw_bias = raw_bias.sort_index()
    
    rcm_bias = pd.concat([rcm_bias_eccc,rcm_bias_noaa])
    rcm_bias.index = rcm_bias.index.astype(str)
    rcm_bias = rcm_bias.sort_index()

    lats = pd.concat([eccc_lats,noaa_lats])
    lats.index = lats.index.astype(str)
    lats = lats.sort_index()

    lons = pd.concat([eccc_lons,noaa_lons])
    lons.index = lons.index.astype(str)
    lons = lons.sort_index()

    elev = pd.concat([eccc_elev,noaa_elev])
    elev.index = elev.index.astype(str)
    elev = elev.sort_index()

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

canesm2_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/orog_CanESM2.nc'
canesm2_nc = Dataset(canesm2_file, mode='r')
lat_canesm2 = np.squeeze(canesm2_nc.variables['lat'][:])
lon_canesm2 = np.squeeze(canesm2_nc.variables['lon'][:])-360
topo_canesm2 = np.squeeze(canesm2_nc.variables['orog'][:])

lons_canesm2,lats_canesm2 = np.meshgrid(lon_canesm2,lat_canesm2)

canrcm4_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/orog_CanRCM4.nc'
canrcm4_nc = Dataset(canrcm4_file, mode='r')
lat_canrcm4 = np.squeeze(canrcm4_nc.variables['lat'][:])
lon_canrcm4 = np.squeeze(canrcm4_nc.variables['lon'][:])-360
topo_canrcm4 = np.squeeze(canrcm4_nc.variables['orog'][:])


def get_nearest_gridpoint(target_lat,target_lon,lat,lon,topo):
    abslat = np.abs(lat-target_lat)
    abslon= np.abs(lon-target_lon)
    x, y = np.where(np.maximum(abslon,abslat) == np.min(np.maximum(abslon,abslat)))
    stn_topo = topo[x[0], y[0]]
    return(stn_topo)


modeled_elev_d03,modeled_elev_d02,modeled_elev_raw,modeled_elev_rcm = [],[],[],[]
for i in range(len(elev)):
    stn = elev.index[i]
    target_lat = lats[stn]
    target_lon = lons[stn]
    
    modeled_elev_d03.append(get_nearest_gridpoint(target_lat,target_lon,lat_d03,lon_d03,topo_d03))
    modeled_elev_d02.append(get_nearest_gridpoint(target_lat,target_lon,lat_d02,lon_d02,topo_d02))
    modeled_elev_raw.append(get_nearest_gridpoint(target_lat,target_lon,lats_canesm2,lons_canesm2,topo_canesm2))
    modeled_elev_rcm.append(get_nearest_gridpoint(target_lat,target_lon,lat_canrcm4,lon_canrcm4,topo_canrcm4))

modeled_elev_d03 = pd.DataFrame(modeled_elev_d03, columns=['elev'], index=elev.index)
modeled_elev_d02 = pd.DataFrame(modeled_elev_d02, columns=['elev'], index=elev.index)
modeled_elev_raw = pd.DataFrame(modeled_elev_raw, columns=['elev'], index=elev.index)
modeled_elev_rcm = pd.DataFrame(modeled_elev_rcm, columns=['elev'], index=elev.index)



elev_bias_d03 = np.squeeze(modeled_elev_d03)-elev
elev_bias_d02 = np.squeeze(modeled_elev_d02)-elev
elev_bias_raw = np.squeeze(modeled_elev_raw)-elev
elev_bias_rcm = np.squeeze(modeled_elev_rcm)-elev

#%%

def plot_elev(data,vmin,vmax,cmap,title,fig_name,label="Elevation bias (m)",extend='both'):
    fig,ax = plot_all_d03(title)
          
    plt.scatter(lons, lats, c=data,s=250,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3,marker='o')

    cbar_ax = fig.add_axes([0.15, 0.08, 0.73, 0.03])
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),cax=cbar_ax, orientation='horizontal',extend=extend)
    cbar_ax.tick_params(labelsize=20)
    cbar_ax.set_xlabel(label,size=23) 

    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/elev_bias/' + title + '.png',bbox_inches='tight')


plot_elev(elev,0,2500,'terrain',"CanESM2-WRF D03","Obs_station_elevation","Elevation (m)",extend=None)

vmin=-1500
vmax=1500
cmap = 'bwr'  

plot_elev(elev_bias_d03,vmin,vmax,cmap,"CanESM2-WRF D03","CanESM2-WRF_D03_elev_bias")
plot_elev(elev_bias_d02,vmin,vmax,cmap,"CanESM2-WRF D02","CanESM2-WRF_D02_elev_bias")
plot_elev(elev_bias_raw,vmin,vmax,cmap,"CanESM2","CanESM2_elev_bias")
plot_elev(elev_bias_rcm,vmin,vmax,cmap,'CanRCM4',"CanRCM4_elev_bias")


#%%
def plot_elev_bias(elev_b,model_b,color,title,savename):
    plt.figure(figsize=(10, 10),dpi=200)
    
    plt.ylabel('Abs. Elevation Bias (m)',fontsize=24)
    if variable == "pr":
        plt.xlabel('Abs. Precipitation bias (mm/day)',fontsize=24)
        vmin=0
        vmax=3500/365.25
        model_b=model_b/365.25
    elif variable == "t":
        plt.xlabel('Abs. Temperature bias (\N{degree sign}C)',fontsize=24)
        vmin=0
        vmax=10
    elif variable=="wind":
        plt.xlabel('Abs. Wind speed bias (m/s)',fontsize=24)
        vmin=0
        vmax=4
        
        
    plt.xlim([vmin,vmax])
    plt.ylim([-20,1500])
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    slope, intercept, r_value, p_value, std_err = linregress(model_b,elev_b)
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope * x + intercept
    plt.plot(x, line_of_best_fit,'--', color='grey')
    
    r_squared = r_value ** 2
    plt.text(0.5, 1400, f'$R^2$ = {r_squared:.2f}', fontsize=25, color='k')


    plt.scatter(model_b,elev_b,color=color,s=200,marker='.')
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/elev_bias/ann_elev_bias_' + variable + '_' + savename + '.png',bbox_inches='tight')

plot_elev_bias(abs(elev_bias_d03),abs(wrf_bias),'C0','CanESM2-WRF D03',"canesm2_wrf_d03")
plot_elev_bias(abs(elev_bias_d02),abs(wrf_bias_d02),'C1','CanESM2-WRF D02',"canesm2_wrf_d02")
plot_elev_bias(abs(elev_bias_raw),abs(raw_bias),'C2','CanESM2',"canesm2_raw")
plot_elev_bias(abs(elev_bias_rcm),abs(rcm_bias),'C3','CanRCM4',"canrcm4")

#%%
def plot_elev_bias_both_dir(elev_b,model_b,color,title,savename):
    plt.figure(figsize=(10, 10),dpi=200)
    
    plt.ylabel('Elevation Bias (m)',fontsize=24)
    if variable == "pr":
        plt.xlabel('Precipitation bias (mm/day)',fontsize=24)
        vmin=-3500/365.25
        vmax=3500/365.25
        model_b=model_b/365.25
    elif variable == "t":
        plt.xlabel('Temperature bias (\N{degree sign}C)',fontsize=24)
        vmin=-10
        vmax=10
    elif variable=="wind":
        plt.xlabel('Wind speed bias (m/s)',fontsize=24)
        vmin=-4
        vmax=4
        
        
    plt.xlim([vmin,vmax])
    plt.ylim([-1500,1500])
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    slope, intercept, r_value, p_value, std_err = linregress(model_b,elev_b)
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope * x + intercept
    plt.plot(x, line_of_best_fit,'--', color='grey')
    
    r_squared = r_value ** 2
    plt.text(1.5, 1300, f'$R^2$ = {r_squared:.2f}', fontsize=25, color='k')


    plt.scatter(model_b,elev_b,color=color,s=200,marker='.')
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/elev_bias/ann_elev_bias_' + variable + '_' + savename + '_bothdir.png',bbox_inches='tight')

plot_elev_bias_both_dir((elev_bias_d03),(wrf_bias),'C0','CanESM2-WRF D03',"canesm2_wrf_d03")
plot_elev_bias_both_dir((elev_bias_d02),(wrf_bias_d02),'C1','CanESM2-WRF D02',"canesm2_wrf_d02")
plot_elev_bias_both_dir((elev_bias_raw),(raw_bias),'C2','CanESM2',"canesm2_raw")
plot_elev_bias_both_dir((elev_bias_rcm),(rcm_bias),'C3','CanRCM4',"canrcm4")
