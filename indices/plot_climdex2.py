#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:07:36 2024

@author: evagnegy
"""
import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import WRFDomainLib
import scipy

climdex_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/'
geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'

geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

lons = Dataset(climdex_path + 'csdi_hist.nc').variables['lon'][:]
lats = Dataset(climdex_path + 'csdi_hist.nc').variables['lat'][:]


def plot_climdex(gridded_data,title,vmin,vmax):
    if vmin==0:
        cmap='viridis'
        xlabel='Count'
    else:
        cmap='bwr'
        xlabel='Diff. Count'
        
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
    
    gridded_data[land_d03==0]=np.nan
    
    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap,vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)
    
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    ax1.text(-3680871, 700000, 'D03', va='top', ha='left',fontweight='bold', size=15, color='red', zorder=2)
    
    plt.title(title,fontsize=20)
    
    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(0, vmax+1, 0.5))
    cbar_ax.tick_params(labelsize=20)
    cbar_ax.set_xlabel(xlabel + " (avg. per year)",size=20) 
    
    

#%%

nc = Dataset(climdex_path + 'cdd_hist.nc').variables['cooling_degree_days'][:]
cdd_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'cdd_rcp45.nc').variables['cooling_degree_days'][:]
cdd_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'cdd_rcp85.nc').variables['cooling_degree_days'][:]
cdd_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(cdd_hist,"Cooling Degree Days (Historical)",0,1000)
plot_climdex(cdd_rcp45 - cdd_hist,"Cooling Degree Days (RCP4.5 - Historical)",-500,500)
plot_climdex(cdd_rcp85 - cdd_hist,"Cooling Degree Days (RCP8.5 - Historical)",-500,500)

#%%

nc_hist = np.squeeze(Dataset(climdex_path + 'csdi_hist.nc').variables['csdi_6'][:])
csdi_hist_std = np.squeeze(np.std(nc_hist,axis=0))
csdi_hist_avg = np.squeeze(np.mean(nc_hist,axis=0))

#nc = Dataset(climdex_path + 'csdi_rcp45_base_hist.nc').variables['csdi_6'][:]
#csdi_rcp45_base_hist_avg = np.squeeze(np.mean(nc,axis=0))

#nc = Dataset(climdex_path + 'csdi_rcp85_base_hist.nc').variables['csdi_6'][:]
#csdi_rcp85_base_hist_avg = np.squeeze(np.mean(nc,axis=0))

nc_rcp45 = np.squeeze(Dataset(climdex_path + 'csdi_rcp45_base_rcp45.nc').variables['csdi_6'][:])
csdi_rcp45_base_rcp45_std = np.squeeze(np.std(nc_rcp45,axis=0))
csdi_rcp45_base_rcp45_avg = np.squeeze(np.mean(nc_rcp45,axis=0))

nc_rcp85 = np.squeeze(Dataset(climdex_path + 'csdi_rcp85_base_rcp85.nc').variables['csdi_6'][:])
csdi_rcp45_base_rcp85_std = np.squeeze(np.std(nc_rcp85,axis=0))
csdi_rcp85_base_rcp85_avg = np.squeeze(np.mean(nc_rcp85,axis=0))

#%%
plot_climdex(csdi_hist_avg,"Cold Spell Duration Index (Historical)",0,8)
#plot_climdex(csdi_rcp45_base_hist_avg - csdi_hist_avg,"Cold Spell Duration Index (RCP4.5 - Historical)\nBase: Historical",-10,10)
#plot_climdex(csdi_rcp85_base_hist_avg - csdi_hist_avg,"Cold Spell Duration Index (RCP8.5 - Historical)\nBase: Historical",-10,10)
plot_climdex(csdi_rcp45_base_rcp45_avg - csdi_hist_avg,"Cold Spell Duration Index (RCP4.5 - Historical)\nBase: RCP4.5",-8,8)
plot_climdex(csdi_rcp85_base_rcp85_avg - csdi_hist_avg,"Cold Spell Duration Index (RCP8.5 - Historical)\nBase: RCP8.5",-8,8)


#%%
csdi_hist_p0 = np.percentile(nc_hist,0,axis=0)
csdi_hist_p25 = np.percentile(nc_hist,25,axis=0)
csdi_hist_p50 = np.percentile(nc_hist,50,axis=0)
csdi_hist_p75 = np.percentile(nc_hist,75,axis=0)
csdi_hist_p100 = np.percentile(nc_hist,100,axis=0)

plot_climdex(csdi_hist_p0,"Cold Spell Duration Index p0 (Historical)",0,16)
plot_climdex(csdi_hist_p25,"Cold Spell Duration Index p25 (Historical)",0,16)
plot_climdex(csdi_hist_p50,"Cold Spell Duration Index p50 (Historical)",0,16)
plot_climdex(csdi_hist_p75,"Cold Spell Duration Index p75 (Historical)",0,16)
plot_climdex(csdi_hist_p100,"Cold Spell Duration Index p100 (Historical)",0,16)

#%%
csdi_rcp45_p0 = np.percentile(nc_rcp45,0,axis=0)
csdi_rcp45_p25 = np.percentile(nc_rcp45,25,axis=0)
csdi_rcp45_p50 = np.percentile(nc_rcp45,50,axis=0)
csdi_rcp45_p75 = np.percentile(nc_rcp45,75,axis=0)
csdi_rcp45_p100 = np.percentile(nc_rcp45,100,axis=0)

plot_climdex(csdi_rcp45_p0,"Cold Spell Duration Index p0 (RCP4.5)",0,16)
plot_climdex(csdi_rcp45_p25,"Cold Spell Duration Index p25 (RCP4.5)",0,16)
plot_climdex(csdi_rcp45_p50,"Cold Spell Duration Index p50 (RCP4.5)",0,16)
plot_climdex(csdi_rcp45_p75,"Cold Spell Duration Index p75 (RCP4.5)",0,16)
plot_climdex(csdi_rcp45_p100,"Cold Spell Duration Index p100 (RCP4.5)",0,16)

#%%
csdi_rcp85_p0 = np.percentile(nc_rcp85,0,axis=0)
csdi_rcp85_p25 = np.percentile(nc_rcp85,25,axis=0)
csdi_rcp85_p50 = np.percentile(nc_rcp85,50,axis=0)
csdi_rcp85_p75 = np.percentile(nc_rcp85,75,axis=0)
csdi_rcp85_p100 = np.percentile(nc_rcp85,100,axis=0)

plot_climdex(csdi_rcp85_p0,"Cold Spell Duration Index p0 (RCP8.5)",0,16)
plot_climdex(csdi_rcp85_p25,"Cold Spell Duration Index p25 (RCP8.5)",0,16)
plot_climdex(csdi_rcp85_p50,"Cold Spell Duration Index p50 (RCP8.5)",0,16)
plot_climdex(csdi_rcp85_p75,"Cold Spell Duration Index p75 (RCP8.5)",0,16)
plot_climdex(csdi_rcp85_p100,"Cold Spell Duration Index p100 (RCP8.5)",0,16)

#%%
 
plot_climdex(csdi_hist_std,"Cold Spell Duration Index STD (Historical)",-8,8)
plot_climdex(csdi_rcp45_base_rcp45_std,"Cold Spell Duration Index STD (Historical)",-8,8)
plot_climdex(csdi_rcp45_base_rcp85_std,"Cold Spell Duration Index STD (Historical)",-8,8)

plot_climdex(csdi_hist_std/np.sqrt(20),"Cold Spell Duration Index SE (Historical)",-2,2)
plot_climdex(csdi_rcp45_base_rcp45_std/np.sqrt(20),"Cold Spell Duration Index SE (Historical)",-2,2)
plot_climdex(csdi_rcp45_base_rcp85_std/np.sqrt(20),"Cold Spell Duration Index SE (Historical)",-2,2)

#%%

rcp45_res = scipy.stats.ttest_ind(np.squeeze(nc_hist), np.squeeze(nc_rcp45), axis=0)
rcp85_res = scipy.stats.ttest_ind(np.squeeze(nc_hist), np.squeeze(nc_rcp85), axis=0)


#plot_climdex(rcp45_res.pvalue,"Cold Spell Duration Index Pvalue (Hist/RCP4.5)",0,1 )
plot_climdex(rcp85_res.pvalue,"Cold Spell Duration Index Pvalue (Hist/RCP8.5)",0,1 )

#%%

nc = Dataset(climdex_path + 'dlyfrzthw_hist.nc').variables['dlyfrzthw'][:]
dlyfrzthw_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'dlyfrzthw_rcp45.nc').variables['dlyfrzthw'][:]
dlyfrzthw_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'dlyfrzthw_rcp85.nc').variables['dlyfrzthw'][:]
dlyfrzthw_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(dlyfrzthw_hist,"Daily Freeze Thaw Cycles (Historical)",0,200)
plot_climdex(dlyfrzthw_rcp45 - dlyfrzthw_hist,"Daily Freeze Thaw Cycles (RCP4.5 - Historical)",-100,100)
plot_climdex(dlyfrzthw_rcp85 - dlyfrzthw_hist,"Daily Freeze Thaw Cycles (RCP8.5 - Historical)",-100,100)

#%%

nc = Dataset(climdex_path + 'dsfreq_hist.nc').variables['dry_spell_frequency'][:]
dsfreq_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'dsfreq_rcp45.nc').variables['dry_spell_frequency'][:]
dsfreq_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'dsfreq_rcp85.nc').variables['dry_spell_frequency'][:]
dsfreq_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(dsfreq_hist,"Dry Spell Frequency (Historical)",0,50)
plot_climdex(dsfreq_rcp45 - dsfreq_hist,"Dry Spell Frequency (RCP4.5 - Historical)",-10,10)
plot_climdex(dsfreq_rcp85 - dsfreq_hist,"Dry Spell Frequency (RCP8.5 - Historical)",-10,10)

#%%

nc = Dataset(climdex_path + 'dstot_hist.nc').variables['dry_spell_total_length'][:]
dstot_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'dstot_rcp45.nc').variables['dry_spell_total_length'][:]
dstot_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'dstot_rcp85.nc').variables['dry_spell_total_length'][:]
dstot_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(dstot_hist,"Dry Spell Total Length (Historical)",0,356)
plot_climdex(dstot_rcp45 - dstot_hist,"Dry Spell Total Length (RCP4.5 - Historical)",-30,30)
plot_climdex(dstot_rcp85 - dstot_hist,"Dry Spell Total Length (RCP8.5 - Historical)",-30,30)


#%%

nc = Dataset(climdex_path + 'fd_hist.nc').variables['frost_days'][:]
fd_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'fd_rcp45.nc').variables['frost_days'][:]
fd_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'fd_rcp85.nc').variables['frost_days'][:]
fd_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(fd_hist,"Frost Days (Historical)",0,356)
plot_climdex(fd_rcp45 - fd_hist,"Frost Days (RCP4.5 - Historical)",-100,100)
plot_climdex(fd_rcp85 - fd_hist,"Frost Days (RCP8.5 - Historical)",-100,100)

#%%

nc = Dataset(climdex_path + 'freshet_hist.nc').variables['freshet_start'][:]
freshet_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'freshet_rcp45.nc').variables['freshet_start'][:]
freshet_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'freshet_rcp85.nc').variables['freshet_start'][:]
freshet_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(freshet_hist,"Freshet Start (Historical)",0,356)
plot_climdex(freshet_rcp45 - freshet_hist,"Freshet Start (RCP4.5 - Historical)",-100,100)
plot_climdex(freshet_rcp85 - freshet_hist,"Freshet Start (RCP8.5 - Historical)",-100,100)

#%%

nc = Dataset(climdex_path + 'sdii_hist.nc').variables['sdii'][:]
sdii_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'sdii_rcp45.nc').variables['sdii'][:]
sdii_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'sdii_rcp85.nc').variables['sdii'][:]
sdii_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(sdii_hist,"Simple Daily Intensity Index (Historical)",0,30)
plot_climdex(sdii_rcp45 - sdii_hist,"Simple Daily Intensity Index (RCP4.5 - Historical)",-5,5)
plot_climdex(sdii_rcp85 - sdii_hist,"Simple Daily Intensity Index (RCP8.5 - Historical)",-5,5)
#%%


nc = Dataset(climdex_path + 'gsl_hist.nc').variables['growing_season_length'][:]
gsl_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'gsl_rcp45.nc').variables['growing_season_length'][:]
gsl_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'gsl_rcp85.nc').variables['growing_season_length'][:]
gsl_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(gsl_hist,"Growing Season Length (Historical)",0,356)
plot_climdex(gsl_rcp45 - gsl_hist,"Growing Season Length (RCP4.5 - Historical)",-100,100)
plot_climdex(gsl_rcp85 - gsl_hist,"Growing Season Length (RCP8.5 - Historical)",-100,100)

#%%

nc = Dataset(climdex_path + 'hdd_hist.nc').variables['heating_degree_days'][:]
hdd_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'hdd_rcp45.nc').variables['heating_degree_days'][:]
hdd_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'hdd_rcp85.nc').variables['heating_degree_days'][:]
hdd_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(hdd_hist,"Heating Degree Days (Historical)",0,10000)
plot_climdex(hdd_rcp45 - hdd_hist,"Heating Degree Days (RCP4.5 - Historical)",-2000,2000)
plot_climdex(hdd_rcp85 - hdd_hist,"Heating Degree Days (RCP8.5 - Historical)",-2000,2000)

#%%

nc = Dataset(climdex_path + 'hwfreq_hist.nc').variables['heat_wave_frequency'][:]
hwfreq_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'hwfreq_rcp45.nc').variables['heat_wave_frequency'][:]
hwfreq_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'hwfreq_rcp85.nc').variables['heat_wave_frequency'][:]
hwfreq_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(hwfreq_hist,"Heat Wave Frequency (Historical)",0,5)
plot_climdex(hwfreq_rcp45 - hwfreq_hist,"Heat Wave Frequency (RCP4.5 - Historical)",-5,5)
plot_climdex(hwfreq_rcp85 - hwfreq_hist,"Heat Wave Frequency (RCP8.5 - Historical)",-5,5)

#%%

nc = Dataset(climdex_path + 'hwi_hist.nc').variables['heat_wave_index'][:]
hwi_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'hwi_rcp45.nc').variables['heat_wave_index'][:]
hwi_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'hwi_rcp85.nc').variables['heat_wave_index'][:]
hwi_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(hwi_hist,"Heat Wave Index (Historical)",0,125)
plot_climdex(hwi_rcp45 - hwi_hist,"Heat Wave Index (RCP4.5 - Historical)",-50,50)
plot_climdex(hwi_rcp85 - hwi_hist,"Heat Wave Index (RCP8.5 - Historical)",-50,50)

#%%

nc = Dataset(climdex_path + 'hwtot_hist.nc').variables['heat_wave_total_length'][:]
hwtot_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'hwtot_rcp45.nc').variables['heat_wave_total_length'][:]
hwtot_rcp45 = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'hwtot_rcp85.nc').variables['heat_wave_total_length'][:]
hwtot_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(hwtot_hist,"Heat Wave Total Length (Historical)",0,20)
plot_climdex(hwtot_rcp45 - hwtot_hist,"Heat Wave Total Length (RCP4.5 - Historical)",-10,10)
plot_climdex(hwtot_rcp45 - hwtot_hist,"Heat Wave Total Length (RCP8.5 - Historical)",-10,10)

#%%

nc = Dataset(climdex_path + 'wsdi_hist.nc').variables['warm_spell_duration_index'][:]
wsdi_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'wsdi_rcp45_base_hist.nc').variables['warm_spell_duration_index'][:]
wsdi_rcp45_base_hist = np.squeeze(np.mean(nc,axis=0))

#nc = Dataset(climdex_path + 'wsdi_rcp85_base_hist.nc').variables['warm_spell_duration_index'][:]
#wsdi_rcp85_base_hist = np.squeeze(np.mean(nc,axis=0))

nc = Dataset(climdex_path + 'wsdi_rcp45_base_rcp45.nc').variables['warm_spell_duration_index'][:]
wsdi_rcp45_base_rcp45 = np.squeeze(np.mean(nc,axis=0))

#nc = Dataset(climdex_path + 'wsdi_rcp85_base_rcp85.nc').variables['warm_spell_duration_index'][:]
#wsdi_rcp85_base_rcp85 = np.squeeze(np.mean(nc,axis=0))

plot_climdex(wsdi_hist,"Warm Spell Duration Index (Historical)",0,15)
plot_climdex(wsdi_rcp45_base_hist - wsdi_hist,"Warm Spell Duration Index (RCP4.5 - Historical)\nBase: Historical",-60,60)
#plot_climdex(wsdi_rcp85_base_hist - wsdi_hist,"Warm Spell Duration Index (RCP8.5 - Historical)\nBase: Historical",-60,60)
plot_climdex(wsdi_rcp45_base_rcp45 - wsdi_hist,"Warm Spell Duration Index (RCP4.5 - Historical)\nBase: RCP4.5",-10,10)
#plot_climdex(wsdi_rcp85_base_rcp85 - wsdi_hist,"Warm Spell Duration Index (RCP8.5 - Historical)\nBase: RCP8.5",-10,10)
