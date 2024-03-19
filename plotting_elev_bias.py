import pandas as pd
import warnings
import sys
import numpy as np
from netCDF4 import Dataset
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib
import WRFDomainLib
import cartopy.feature as cf
import matplotlib as mpl


#%%

eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'
noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations.csv'

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
noaa_obs = get_noaa_obs("yearly",noaa_station_IDs,stations_dir,"t")
all_noaa = noaa_elev.index.values
obs_noaa = noaa_obs.columns.values

diff = [x for x in all_noaa if x not in set(obs_noaa)]

#%%
# =============================================================================
# geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
# geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
# lat_d03 = np.squeeze(geo_em_d03_nc.variables['XLAT_C'][:])
# lon_d03 = np.squeeze(geo_em_d03_nc.variables['XLONG_C'][:])
# topo_d03 = np.squeeze(geo_em_d03_nc.variables['HGT_M'][:])
# 
# geo_em_d02_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d02.nc'
# geo_em_d02_nc = Dataset(geo_em_d02_file, mode='r')
# lat_d02 = np.squeeze(geo_em_d02_nc.variables['XLAT_C'][:])
# lon_d02 = np.squeeze(geo_em_d02_nc.variables['XLONG_C'][:])
# topo_d02 = np.squeeze(geo_em_d02_nc.variables['HGT_M'][:])
# 
# canrcm4_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/orog_CanRCM4.nc'
# canrcm4_nc = Dataset(canrcm4_file, mode='r')
# lat_canrcm4 = np.squeeze(canrcm4_nc.variables['lat'][:])
# lon_canrcm4 = np.squeeze(canrcm4_nc.variables['lon'][:])
# topo_canrcm4 = np.squeeze(canrcm4_nc.variables['orog'][:])
# 
# #topo_canrcm4 = topo_canrcm4[:-1,:-1]
# #lat_canrcm4 = (lat_canrcm4[1:]+lat_canrcm4[:-1])/2
# #lon_canrcm4 = (lon_canrcm4[1:]+lon_canrcm4[:-1])/2  
# 
# canesm2_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/orog_CanESM2.nc'
# canesm2_nc = Dataset(canesm2_file, mode='r')
# lat_canesm2 = np.squeeze(canesm2_nc.variables['lat'][:])
# lon_canesm2 = np.squeeze(canesm2_nc.variables['lon'][:])
# topo_canesm2 = np.squeeze(canesm2_nc.variables['orog'][:])
# 
# #topo_canesm2 = topo_canesm2[:-1,:-1]
# #lat_canesm2 = (lat_canesm2[1:]+lat_canesm2[:-1])/2
# #lon_canesm2 = (lon_canesm2[1:]+lon_canesm2[:-1])/2  
# 
# WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
# wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
# 
# #title = 'CanESM2'
# title = 'CanESM2-WRF D03'
# 
# fig1 = plt.figure(figsize=(10, 10),dpi=200)
# ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
# 
# if title == 'CanESM2-WRF D03':
#     ax1.pcolormesh(lon_d02, lat_d02, topo_d02, cmap="terrain", vmin=0,vmax=3000, alpha=0.3, transform=ccrs.PlateCarree(),zorder=0)
#     ax1.pcolormesh(lon_d03, lat_d03, topo_d03, cmap="terrain", vmin=0,vmax=3000, alpha=0.4, transform=ccrs.PlateCarree(),zorder=0)
# elif title == 'CanESM2-WRF D02':
#     ax1.pcolormesh(lon_d02, lat_d02, topo_d02, cmap="terrain", vmin=0,vmax=3000, alpha=0.7, transform=ccrs.PlateCarree(),zorder=0)
# elif title == 'CanESM2':
#     ax1.pcolormesh(lon_canesm2, lat_canesm2, topo_canesm2, cmap="terrain", vmin=0,vmax=3000, alpha=0.7, transform=ccrs.PlateCarree(),zorder=0)
# elif title == 'CanRCM4':
#     ax1.pcolormesh(lon_canrcm4, lat_canrcm4, topo_canrcm4, cmap="terrain", vmin=0,vmax=3000, alpha=0.7, transform=ccrs.PlateCarree(),zorder=0)
# 
# =============================================================================
elev = pd.concat([eccc_elev,bch_elev,noaa_elev])
elev.index = elev.index.astype(str)
elev = elev.sort_index()

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
    ax1.pcolormesh(lon_d02, lat_d02, topo_d02, cmap=cmap, vmin=vmin,vmax=vmax, alpha=0.3, transform=ccrs.PlateCarree(),zorder=0)
    ax1.pcolormesh(lon_d03, lat_d03, topo_d03, cmap=cmap, vmin=vmin,vmax=vmax, alpha=0.4, transform=ccrs.PlateCarree(),zorder=0)

    #ax1.coastlines('10m', linewidth=0.8)
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)

    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/13
    random_x_factor = corner_x3[0]/65

    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    ax1.text(-3700000, 700000, 'D03', va='top', ha='left',fontweight='bold', size=24, color='red', zorder=2)

    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())

    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', alpha=1)
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,-49,4))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(0,81,4))

    ax1.text(corner_x3[0]+length_x[2]*0.16, corner_y3[0]+length_y[2]*-0.09, '44ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-40,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*-0.17, corner_y3[0]+length_y[2]*0.8, '48ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-43,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*0.365, corner_y3[0]+length_y[2]*0.995, '52ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-40,alpha=0.6)

    ax1.text(corner_x3[0]+length_x[2]*-0.16, corner_y3[0]+length_y[2]*0.18, '132ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=50,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*-0.165, corner_y3[0]+length_y[2]*0.705, '128ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=50,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*0.875, corner_y3[0]+length_y[2]*1.017, '124ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=55,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*0.96, corner_y3[0]+length_y[2]*0.62, '120ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=58,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*0.975, corner_y3[0]+length_y[2]*0.035, '116ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=58,alpha=0.6)




    return fig1,ax1

fig1,ax1 = plot_all_d03()

vmin=0
vmax=3000
color = 'r'
border = 'k'
for i in range(len(eccc_lats)):
    if eccc_station_IDs[i] in [409,631,636,557]:
        #skip bc too close to boundary
        continue
    plt.scatter(eccc_lons.values[i],eccc_lats.values[i],c=eccc_elev.values[i],s=150,transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,zorder=4,marker='s',cmap='terrain',vmin=vmin,vmax=vmax)

for i in range(len(bch_lats)):
    plt.scatter(bch_lons.values[i],bch_lats.values[i],s=250,c=bch_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='^',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)

count=0
for i in range(len(noaa_lats)):
    if noaa_station_IDs[i] in ['USC00455946','USC00457696','USC00458959','USW00024157']:
        #skip bc too close to boundary
        continue
    if noaa_station_IDs[i] in diff:
        plt.scatter(noaa_lons.values[i],noaa_lats.values[i],s=180,c=noaa_elev.values[i],transform=ccrs.PlateCarree(),edgecolor='r',linewidth=1,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)
        count=count+1
    else:
        plt.scatter(noaa_lons.values[i],noaa_lats.values[i],s=180,c=noaa_elev.values[i],transform=ccrs.PlateCarree(),edgecolor=border,linewidth=0.5,marker='o',zorder=4,cmap='terrain',vmin=vmin,vmax=vmax)
        count=count+1

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
    

fig1.savefig('/Users/evagnegy/Desktop/paper_figures/stations/all_stations_elev.png', dpi=600,bbox_inches='tight')




#for i in range(len(modeled_elev_d03['elev'])):
#    el_actual = elev[i]
#    el = modeled_elev_d03['elev'][i]
#    la = lats[i]
#    lo = lons[i]

    #plt.scatter(lo,la,c=el_actual-el,marker='o',cmap='bwr',s=150,transform=ccrs.PlateCarree(),vmin=-1500,vmax=1500,edgecolor='k',zorder=10)
    #plt.scatter(lo,la,c=el_actual,marker='o',cmap='terrain',s=150,transform=ccrs.PlateCarree(),vmin=0,vmax=3000,edgecolor='k',zorder=10,alpha=0.7)

#plt.title(title,fontsize=22)



# =============================================================================
# vmin=0
# vmax=3000
# cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
# fig1.colorbar(matplotlib.cm.ScalarMappable(cmap='terrain', norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
#               cax=cbar_ax, ticks=np.arange(0, vmax+1, 500), orientation='horizontal')
# 
# cbar_ax1 = fig1.add_axes([0.93, 0.15, 0.02, 0.7])
# fig1.colorbar(matplotlib.cm.ScalarMappable(cmap='bwr', norm=matplotlib.colors.Normalize(vmin=-1500, vmax=1500)),
#               cax=cbar_ax1, orientation='vertical')
# 
# =============================================================================

