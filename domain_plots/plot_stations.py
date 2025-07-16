#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:45:41 2023

@author: evagnegy
"""
import matplotlib.pyplot as plt 
import matplotlib as mpl
import cartopy.crs as ccrs
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import cartopy.feature as cf
from cartopy.io.shapereader import Reader
import WRFDomainLib
#%%
shapefile_filepath = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/shapefiles/'

bch_station_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'

eccc_daily_stations_wind = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations_hourly.csv'
eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'

noaa_daily_stations_wind = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations_wind.csv'
noaa_daily_stations_t = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations_tas.csv'
noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations.csv'

noaa_daily_stations_buoys = '//Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_buoys.csv'
eccc_daily_stations_buoys = '//Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_buoys.csv'



df = pd.read_csv(eccc_daily_stations,header=None)
eccc_station_IDs_all = list(df.iloc[:,4])
eccc_station_names = list(df.iloc[:,1])



eccc_lats = df.iloc[:,7]
eccc_lons = df.iloc[:,8]
eccc_lats.index = eccc_station_IDs_all
eccc_lons.index = eccc_station_IDs_all

eccc_elev = (df.iloc[:,11])
eccc_elev.index = eccc_station_IDs_all

df = pd.read_csv(bch_station_file)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])

bch_lats = df['Y']
bch_lons = df['X']
bch_lats.index = bch_station_IDs
bch_lons.index = bch_station_IDs

bch_elev = (df["ELEV"])
bch_elev.index = bch_station_IDs

df = pd.read_csv(noaa_daily_stations_t)
noaa_station_IDs_t = list(df.iloc[:,0])


df = pd.read_csv(noaa_daily_stations)

noaa_station_IDs_all = list(df.iloc[:,0])
noaa_station_names = list(df.iloc[:,1])



noaa_lats = df.iloc[:,2]
noaa_lons = df.iloc[:,3]
noaa_lats.index = noaa_station_IDs_all
noaa_lons.index = noaa_station_IDs_all

noaa_elev = (df["ELEVATION"])
noaa_elev.index = noaa_station_IDs_all

eccc_station_file_wind = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations_hourly.csv'
noaa_station_file_wind = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations_wind.csv'

df = pd.read_csv(eccc_station_file_wind,header=None)
eccc_station_IDs_wind = list(df.iloc[:,4])
eccc_station_IDs_wind = [ int(x) for x in eccc_station_IDs_wind ]

df = pd.read_csv(noaa_station_file_wind)
noaa_station_IDs_wind = list(df.iloc[:,0])


noaa_station_file_tas = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations_tas.csv'

df = pd.read_csv(noaa_station_file_tas)
noaa_station_IDs_tas = list(df.iloc[:,0])

df = pd.read_csv(noaa_daily_stations_buoys)
noaa_buoy_station_IDs = list(df["STATION_ID"])

noaa_buoy_lats = df['Y']
noaa_buoy_lons = df['X']

df = pd.read_csv(eccc_daily_stations_buoys)
eccc_buoy_station_IDs = list(df["STATION_ID"])

eccc_buoy_lats = df['Y']
eccc_buoy_lons = df['X']


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

def plot_all_d03():
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)


    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    cmap = 'terrain'
    vmin=0
    vmax=3000
    ax1.pcolormesh(lon_d02, lat_d02, topo_d02, cmap=cmap, vmin=vmin,vmax=vmax, alpha=0.4, transform=ccrs.PlateCarree(),zorder=0)
    ax1.pcolormesh(lon_d03, lat_d03, topo_d03, cmap=cmap, vmin=vmin,vmax=vmax, alpha=0.5, transform=ccrs.PlateCarree(),zorder=0)

    #ax1.coastlines('10m', linewidth=0.8)
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)

    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/13
    random_x_factor = corner_x3[0]/65

    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    ax1.text(-3712000, 700000, 'D03', va='top', ha='left',fontweight='bold', size=30, color='red', zorder=10)

    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())

    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', alpha=1,linewidth=1.5)
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,-49,4))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(0,81,4))

    ax1.text(corner_x3[0]+length_x[2]*-0.175, corner_y3[0]+length_y[2]*0.13, '44$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-40,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*-0.175, corner_y3[0]+length_y[2]*0.78, '48$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-38,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.935, corner_y3[0]+length_y[2]*0.55, '52$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-30,alpha=0.8)

    ax1.text(corner_x3[0]+length_x[2]*0.1, corner_y3[0]+length_y[2]*1.01, '132$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.53, corner_y3[0]+length_y[2]*1.01, '128$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.875, corner_y3[0]+length_y[2]*1.01, '124$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    #ax1.text(corner_x3[0]+length_x[2]*0.95, corner_y3[0]+length_y[2]*0.67, '120$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=58,alpha=0.8)
    #ax1.text(corner_x3[0]+length_x[2]*0.935, corner_y3[0]+length_y[2]*0.035, '116$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=59,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.1, corner_y3[0]+length_y[2]*-0.08, '124$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.53, corner_y3[0]+length_y[2]*-0.08, '120$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.9, corner_y3[0]+length_y[2]*-0.08, '116$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)




    return fig1,ax1




# =============================================================================
# stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/'
# noaa_obs = get_noaa_obs("yearly",noaa_station_IDs,stations_dir,"t")
# all_noaa = noaa_elev.index.values
# obs_noaa = noaa_obs.columns.values
# 
# diff = [x for x in all_noaa if x not in set(obs_noaa)]
# 
# #%% 
# elev = pd.concat([eccc_elev,bch_elev,noaa_elev])
# elev.index = elev.index.astype(str)
# elev = elev.sort_index()
# 
# =============================================================================
#%%
# plot with whole d03 box
fig1,ax1 = plot_all_d03()

vmin=0
vmax=3000

border = 'k'
for i in range(len(eccc_lats)):
    plt.scatter(eccc_lons.values[i],eccc_lats.values[i],c=eccc_elev.values[i],s=150,transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,zorder=4,marker='s',cmap='terrain',vmin=vmin,vmax=vmax)

for i in range(len(bch_lats)):
    plt.scatter(bch_lons.values[i],bch_lats.values[i],s=250,c=bch_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='^',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)

count=0
for i in range(len(noaa_lats)):
    plt.scatter(noaa_lons.values[i],noaa_lats.values[i],s=180,c=noaa_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)


plt.scatter(0,0,facecolors='none',s=150,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="ECCC station",marker='s')
plt.scatter(0,0,facecolors='none',s=220,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="BCH station",marker='^')
plt.scatter(0,0,facecolors='none',s=200,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="NOAA station",marker='o')
plt.legend(loc=(0.12,0.78),fontsize=16)


cmap = 'terrain'

cbar_ax = fig1.add_axes([0.175, 0.08, 0.68, 0.03])
fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
              cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal')

cbar_ax.tick_params(labelsize=20)

cbar_ax.set_xlabel('Elevation (m)',size=23) 
    

#fig1.savefig('/Users/evagnegy/Desktop/paper_figures/stations/all_stations_elev.png', dpi=600,bbox_inches='tight')


 #%% plot with whole d03 box
fig1,ax1 = plot_all_d03()

vmin=0
vmax=3000

border = 'k'
for i in range(len(eccc_lats)):
    if eccc_station_IDs_all[i] in eccc_station_IDs_wind:
        plt.scatter(eccc_lons.values[i],eccc_lats.values[i],c=eccc_elev.values[i],s=150,transform=ccrs.PlateCarree(),edgecolor='w',linewidth=1,zorder=4,marker='s',cmap='terrain',vmin=vmin,vmax=vmax)
    else:
        plt.scatter(eccc_lons.values[i],eccc_lats.values[i],c=eccc_elev.values[i],s=150,transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,zorder=4,marker='s',cmap='terrain',vmin=vmin,vmax=vmax)

for i in range(len(bch_lats)):
    plt.scatter(bch_lons.values[i],bch_lats.values[i],s=250,c=bch_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='^',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)

count=0
for i in range(len(noaa_lats)):

    if noaa_station_IDs_all[i] not in noaa_station_IDs_t: #stations that dont have temperature
        plt.scatter(noaa_lons.values[i],noaa_lats.values[i],s=180,c=noaa_elev.values[i],transform=ccrs.PlateCarree(),edgecolor='r',linewidth=1,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)

    elif noaa_station_IDs_all[i] in noaa_station_IDs_wind:
        plt.scatter(noaa_lons.values[i],noaa_lats.values[i],s=180,c=noaa_elev.values[i],transform=ccrs.PlateCarree(),edgecolor='w',linewidth=1,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)

    else:
        plt.scatter(noaa_lons.values[i],noaa_lats.values[i],s=180,c=noaa_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)

for i in range(len(noaa_buoy_lats)):
    plt.scatter(noaa_buoy_lons.values[i],noaa_buoy_lats.values[i],s=180,c=5,transform=ccrs.PlateCarree(),edgecolor='gold',linewidth=1.5,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)

for i in range(len(eccc_buoy_lats)):
    plt.scatter(eccc_buoy_lons.values[i],eccc_buoy_lats.values[i],s=150,c=5,transform=ccrs.PlateCarree(),edgecolor='gold',linewidth=1.5,marker='s',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)


plt.scatter(0,0,facecolors='none',s=150,edgecolor='k',linewidth=2,label="ECCC station",marker='s')
plt.scatter(0,0,facecolors='none',s=220,edgecolor='k',linewidth=2,label="BCH station",marker='^')
plt.scatter(0,0,facecolors='none',s=200,edgecolor='k',linewidth=2,label="NOAA station",marker='o')

plt.scatter(0,0,facecolors='none',s=0,edgecolor='none',linewidth=0,label=" ")

plt.plot(0,0,color='lightgray',linewidth=3,label="pr, t, wind")
plt.plot(0,0,color='k',linewidth=2.5,label="pr, t")
plt.plot(0,0,color='gold',linewidth=2.5,label="t, wind")
plt.plot(0,0,color='r',linewidth=2.5,label="pr")

plt.legend(loc=(0.12,0.3),fontsize=15)


cmap = 'terrain'

cbar_ax = fig1.add_axes([0.175, 0.08, 0.68, 0.03])
fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
              cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal')

cbar_ax.tick_params(labelsize=20)

cbar_ax.set_xlabel('Elevation (m)',size=23) 
    

#fig1.savefig('/Users/evagnegy/Desktop/paper_figures/stations/all_stations_elev_vars.png', dpi=600,bbox_inches='tight')


#%% plot pr (all stations)
fig1,ax1 = plot_all_d03()

vmin=0
vmax=3000

border = 'k'
for i in range(len(eccc_lats)):
    plt.scatter(eccc_lons.values[i],eccc_lats.values[i],c=eccc_elev.values[i],s=150,transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,zorder=4,marker='s',cmap='terrain',vmin=vmin,vmax=vmax)

for i in range(len(bch_lats)):
    plt.scatter(bch_lons.values[i],bch_lats.values[i],s=250,c=bch_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='^',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)


for i in range(len(noaa_lats)):
    plt.scatter(noaa_lons.values[i],noaa_lats.values[i],s=180,c=noaa_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)


plt.scatter(0,0,facecolors='none',s=150,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="ECCC station",marker='s')
plt.scatter(0,0,facecolors='none',s=250,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="BCH station",marker='^')
plt.scatter(0,0,facecolors='none',s=180,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="NOAA station",marker='o')


plt.legend(loc=(0.12,0.78),fontsize=15)


cmap = 'terrain'

cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
              cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal')

cbar_ax.tick_params(labelsize=16)

cbar_ax.set_xlabel('Elevation (m)',size=18) 
    
fig1.savefig('/Users/evagnegy/Desktop/paper_figures/stations/all_stations_elev_pr.png', dpi=600,bbox_inches='tight')

#%% plot tas (all stations)
fig1,ax1 = plot_all_d03()

vmin=0
vmax=3000

border = 'k'
for i in range(len(eccc_lats)):
    plt.scatter(eccc_lons.values[i],eccc_lats.values[i],c=eccc_elev.values[i],s=150,transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,zorder=4,marker='s',cmap='terrain',vmin=vmin,vmax=vmax)

for i in range(len(bch_lats)):
    plt.scatter(bch_lons.values[i],bch_lats.values[i],s=250,c=bch_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='^',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)


for i in range(len(noaa_lats)):
    if noaa_station_IDs[i] in noaa_station_IDs_tas:
        plt.scatter(noaa_lons.values[i],noaa_lats.values[i],s=180,c=noaa_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)


plt.scatter(0,0,facecolors='none',s=150,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="ECCC station",marker='s')
plt.scatter(0,0,facecolors='none',s=250,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="BCH station",marker='^')
plt.scatter(0,0,facecolors='none',s=180,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="NOAA station",marker='o')


plt.legend(loc=(0.12,0.78),fontsize=15)


cmap = 'terrain'

cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
              cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal')

cbar_ax.tick_params(labelsize=16)

cbar_ax.set_xlabel('Elevation (m)',size=18) 
    
fig1.savefig('/Users/evagnegy/Desktop/paper_figures/stations/all_stations_elev_tas.png', dpi=600,bbox_inches='tight')


#%% plot wind (all stations)
fig1,ax1 = plot_all_d03()

vmin=0
vmax=3000

border = 'k'
for i in range(len(eccc_lats)):
    if eccc_station_IDs[i] in eccc_station_IDs_wind:
        plt.scatter(eccc_lons.values[i],eccc_lats.values[i],c=eccc_elev.values[i],s=150,transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,zorder=4,marker='s',cmap='terrain',vmin=vmin,vmax=vmax)

for i in range(len(noaa_lats)):
    if noaa_station_IDs[i] in noaa_station_IDs_wind:
        plt.scatter(noaa_lons.values[i],noaa_lats.values[i],s=180,c=noaa_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)


plt.scatter(0,0,facecolors='none',s=150,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="ECCC station",marker='s')
plt.scatter(0,0,facecolors='none',s=180,transform=ccrs.PlateCarree(),edgecolor='k',linewidth=2,label="NOAA station",marker='o')


plt.legend(loc=(0.12,0.82),fontsize=15)


cmap = 'terrain'

cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
              cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal')

cbar_ax.tick_params(labelsize=16)

cbar_ax.set_xlabel('Elevation (m)',size=18) 
    
fig1.savefig('/Users/evagnegy/Desktop/paper_figures/stations/all_stations_elev_wind.png', dpi=600,bbox_inches='tight')




#%% zoomed in and without box

fig,ax = plot_zoomed_in()

for i in range(len(ECCC_lats)):
    if ECCC_IDs[i] in [409,631,636,557]:
        #skip bc too close to boundary
        continue
    if ECCC_IDs[i] == 82:
        #plot a daily EC station w label
        plt.scatter(ECCC_lons[i],ECCC_lats[i],s=150,color='r',transform=ccrs.PlateCarree(),edgecolor='w',linewidth=0.8,label="ECCC station (daily)",zorder=4)
    elif ECCC_IDs[i] == 118:
        #plot an hourly EC station w label
        plt.scatter(ECCC_lons[i],ECCC_lats[i],s=150,color='k',transform=ccrs.PlateCarree(),edgecolor='w',linewidth=0.8,label="ECCC station (hourly)",zorder=5)
    elif ECCC_IDs[i] in [145,155,192,277,640,889,951,1001,1056]:
        #plot an hourly EC station w/o label
        plt.scatter(ECCC_lons[i],ECCC_lats[i],s=150,color='k',transform=ccrs.PlateCarree(),edgecolor='w',linewidth=0.8,zorder=5)
    else:
        #plot a daily EC station w/o label
        plt.scatter(ECCC_lons[i],ECCC_lats[i],s=150,color='r',transform=ccrs.PlateCarree(),edgecolor='w',linewidth=0.8,zorder=4)




for i in range(len(BCH_lats)):
    if i == 0:
        plt.scatter(BCH_lons[i],BCH_lats[i],s=150,color='r',transform=ccrs.PlateCarree(),edgecolor='w',linewidth=0.8,label="BCH station (daily)",marker='s',zorder=4)
        plt.scatter(0,0,s=150,color='k',transform=ccrs.PlateCarree(),edgecolor='w',linewidth=0.8,label="BCH station (hourly)",marker='s',zorder=4)
    else:
        plt.scatter(BCH_lons[i],BCH_lats[i],s=150,color='r',transform=ccrs.PlateCarree(),edgecolor='w',linewidth=0.8,marker='s',zorder=4)

ax.add_geometries(Reader(shapefile_filepath + 'st99_d00/st99_d00.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)
ax.add_geometries(Reader(shapefile_filepath + 'province/province.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5,zorder=3)               

plt.legend(loc=(0.57,0.82),fontsize=15,framealpha=0.9)

#ax.text(-2330000, 935000, '(a)', size=26, color='k',fontweight='bold')

cmap = 'terrain'
vmin=0
vmax=3000

cbar_ax = fig.add_axes([0.2, 0.09, 0.62, 0.02])
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
              cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal',alpha=0.7)

cbar_ax.tick_params(labelsize=16)

cbar_ax.set_xlabel('Elevation [m]',size=18) 
    

fig.savefig('/Users/evagnegy/Desktop/paper_figures/stations/all_stations_zoom.png', dpi=600,bbox_inches='tight')



#%% plotting individual ECCC stations

for i in range(len(ECCC_lats)):
    if i in ECCC_IDs[i] in [409,631,636,557]:
        continue
    
    fig,ax = plot_zoomed_in()
    #fig,ax = plot_all_d03()
    
    name = ECCC_names[i]
    ID = ECCC_IDs[i]

    plt.scatter(ECCC_lons[i],ECCC_lats[i],marker='*',s=700,color='r',transform=ccrs.PlateCarree(),edgecolor='k',linewidth=0.8,label="ECCC station",zorder=3)
    plt.title("ECCC: " + name)
    
    plt.savefig('/Users/evagnegy/Desktop/paper_figures/stations/ECCC_' + str(ID) + '_' + name + '_zoom.png',bbox_inches='tight')
    plt.close()

#%% plotting individual BCH stations

for i in range(len(BCH_lats)):

    name = BCH_names[i]
    ID = BCH_IDs[i]
    
    #fig,ax = plot_zoomed_in()
    fig,ax = plot_all_d03()

    plt.scatter(BCH_lons[i],BCH_lats[i],marker='*',s=700,color='k',transform=ccrs.PlateCarree(),edgecolor='k',linewidth=0.8,label="BCH station",)
    plt.title("BC Hydro: " + name)
    
    plt.savefig('/Users/evagnegy/Desktop/paper_figures/stations/BCH_' + ID + '.png',bbox_inches='tight')
    #plt.close()

#%% plotting individual NOAA stations

for i in range(len(NOAA_lats)):

    name = NOAA_names[i]
    ID = NOAA_IDs[i]
    
    #fig,ax = plot_zoomed_in()
    fig,ax = plot_all_d03()

    plt.scatter(NOAA_lons[i],NOAA_lats[i],marker='*',s=700,color='k',transform=ccrs.PlateCarree(),edgecolor='k',linewidth=0.8,label="NOAA station",)
    plt.title("NOAA NCEI: " + name)
    
    plt.savefig('/Users/evagnegy/Desktop/paper_figures/stations/NOAA_' + ID + '.png',bbox_inches='tight')
    plt.close()
