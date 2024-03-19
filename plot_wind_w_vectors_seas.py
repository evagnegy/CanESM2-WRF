import matplotlib.pyplot as plt 
import matplotlib as mpl
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import rgb2hex 
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import  ScalarMappable
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import datetime
import matplotlib.cbook
from cartopy.io.shapereader import Reader
from scipy import stats
import WRFDomainLib
import cartopy.feature as cf
import matplotlib.colors as pltcol
import cftime

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

#%%
wind_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/wind_d03_seasmean_hist_uv.nc'
wind_hist_nc = Dataset(wind_hist_file,'r')
lats = wind_hist_nc.variables['lat'][:]
lons = wind_hist_nc.variables['lon'][:]

u_d03_hist = np.squeeze(wind_hist_nc.variables['u'][:])
v_d03_hist = np.squeeze(wind_hist_nc.variables['v'][:])

u_d03_hist_djf = u_d03_hist[0,:,:]
u_d03_hist_mam = u_d03_hist[1,:,:]
u_d03_hist_jja = u_d03_hist[2,:,:]
u_d03_hist_son = u_d03_hist[3,:,:]

v_d03_hist_djf = v_d03_hist[0,:,:]
v_d03_hist_mam = v_d03_hist[1,:,:]
v_d03_hist_jja = v_d03_hist[2,:,:]
v_d03_hist_son = v_d03_hist[3,:,:]

#wspd_d03_hist_djf = np.sqrt(u_d03_hist_djf**2 + v_d03_hist_djf**2)
#wspd_d03_hist_mam = np.sqrt(u_d03_hist_mam**2 + v_d03_hist_mam**2)
#wspd_d03_hist_jja = np.sqrt(u_d03_hist_jja**2 + v_d03_hist_jja**2)
#wspd_d03_hist_son = np.sqrt(u_d03_hist_son**2 + v_d03_hist_son**2)

wdir_d03_hist_djf = 270-np.degrees(np.arctan2(v_d03_hist_djf, u_d03_hist_djf)) #270- to flip to "from" dir and change to cardinal 
wdir_d03_hist_mam = 270-np.degrees(np.arctan2(v_d03_hist_mam, u_d03_hist_mam)) #270- to flip to "from" dir and change to cardinal 
wdir_d03_hist_jja = 270-np.degrees(np.arctan2(v_d03_hist_jja, u_d03_hist_jja)) #270- to flip to "from" dir and change to cardinal 
wdir_d03_hist_son = 270-np.degrees(np.arctan2(v_d03_hist_son, u_d03_hist_son)) #270- to flip to "from" dir and change to cardinal 


wind_hist_file_spd = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/wind_d03_seasmean_hist_wspd.nc'
wind_hist_nc_spd = Dataset(wind_hist_file_spd,'r')
wspd_d03_hist = np.squeeze(wind_hist_nc_spd.variables['wspd'][:])

wspd_d03_hist_djf = wspd_d03_hist[0,:,:]
wspd_d03_hist_mam = wspd_d03_hist[1,:,:]
wspd_d03_hist_jja = wspd_d03_hist[2,:,:]
wspd_d03_hist_son = wspd_d03_hist[3,:,:]

#%%

wind_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/wind_d02_seasmean_hist_uv.nc'
wind_hist_nc = Dataset(wind_hist_file,'r')
lats_d02 = wind_hist_nc.variables['lat'][:]
lons_d02 = wind_hist_nc.variables['lon'][:]

u_d02_hist = np.squeeze(wind_hist_nc.variables['u'][:])
v_d02_hist = np.squeeze(wind_hist_nc.variables['v'][:])

u_d02_hist_djf = u_d02_hist[0,:,:]
u_d02_hist_mam = u_d02_hist[1,:,:]
u_d02_hist_jja = u_d02_hist[2,:,:]
u_d02_hist_son = u_d02_hist[3,:,:]

v_d02_hist_djf = v_d02_hist[0,:,:]
v_d02_hist_mam = v_d02_hist[1,:,:]
v_d02_hist_jja = v_d02_hist[2,:,:]
v_d02_hist_son = v_d02_hist[3,:,:]

#wspd_d02_hist_djf = np.sqrt(u_d02_hist_djf**2 + v_d02_hist_djf**2)
#wspd_d02_hist_mam = np.sqrt(u_d02_hist_mam**2 + v_d02_hist_mam**2)
#wspd_d02_hist_jja = np.sqrt(u_d02_hist_jja**2 + v_d02_hist_jja**2)
#wspd_d02_hist_son = np.sqrt(u_d02_hist_son**2 + v_d02_hist_son**2)

wdir_d02_hist_djf = 270-np.degrees(np.arctan2(v_d02_hist_djf, u_d02_hist_djf)) #270- to flip to "from" dir and change to cardinal 
wdir_d02_hist_mam = 270-np.degrees(np.arctan2(v_d02_hist_mam, u_d02_hist_mam)) #270- to flip to "from" dir and change to cardinal 
wdir_d02_hist_jja = 270-np.degrees(np.arctan2(v_d02_hist_jja, u_d02_hist_jja)) #270- to flip to "from" dir and change to cardinal 
wdir_d02_hist_son = 270-np.degrees(np.arctan2(v_d02_hist_son, u_d02_hist_son)) #270- to flip to "from" dir and change to cardinal 


wind_hist_file_spd = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/wind_d02_seasmean_hist_wspd.nc'
wind_hist_nc_spd = Dataset(wind_hist_file_spd,'r')
wspd_d02_hist = np.squeeze(wind_hist_nc_spd.variables['wspd'][:])

wspd_d02_hist_djf = wspd_d02_hist[0,:,:]
wspd_d02_hist_mam = wspd_d02_hist[1,:,:]
wspd_d02_hist_jja = wspd_d02_hist[2,:,:]
wspd_d02_hist_son = wspd_d02_hist[3,:,:]

#%%
wind_canesm2_u_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/uas_day_CanESM2_historical_r1i1p1_19790101-20051231.nc'
wind_canesm2_u_hist_nc = Dataset(wind_canesm2_u_hist_file,'r')
wspd_hist_canesm2_u_all = np.squeeze(wind_canesm2_u_hist_nc.variables['uas'][:])

lat = wind_canesm2_u_hist_nc.variables['lat'][:]
lon = wind_canesm2_u_hist_nc.variables['lon'][:]
lons_gcm,lats_gcm = np.meshgrid(lon,lat)

wind_canesm2_v_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/vas_day_CanESM2_historical_r1i1p1_19790101-20051231.nc'
wind_canesm2_v_hist_nc = Dataset(wind_canesm2_v_hist_file,'r')
wspd_hist_canesm2_v_all = np.squeeze(wind_canesm2_v_hist_nc.variables['vas'][:])

time_gcm = wind_canesm2_u_hist_nc.variables['time'][:]
time_gcm_dates = cftime.num2date(time_gcm, wind_canesm2_u_hist_nc.variables['time'].units, calendar=wind_canesm2_u_hist_nc.variables['time'].calendar)

djf_indices = [i for i, dt in enumerate(time_gcm_dates) if dt.month in [12,1,2]]
mam_indices = [i for i, dt in enumerate(time_gcm_dates) if dt.month in [3,4,5]]
jja_indices = [i for i, dt in enumerate(time_gcm_dates) if dt.month in [6,7,8]]
son_indices = [i for i, dt in enumerate(time_gcm_dates) if dt.month in [9,10,11]]


u_hist_canesm2_djf = np.mean(wspd_hist_canesm2_u_all[djf_indices,:,:],axis=0)
u_hist_canesm2_mam = np.mean(wspd_hist_canesm2_u_all[mam_indices,:,:],axis=0)
u_hist_canesm2_jja = np.mean(wspd_hist_canesm2_u_all[jja_indices,:,:],axis=0)
u_hist_canesm2_son = np.mean(wspd_hist_canesm2_u_all[son_indices,:,:],axis=0)

v_hist_canesm2_djf = np.mean(wspd_hist_canesm2_v_all[djf_indices,:,:],axis=0)
v_hist_canesm2_mam = np.mean(wspd_hist_canesm2_v_all[mam_indices,:,:],axis=0)
v_hist_canesm2_jja = np.mean(wspd_hist_canesm2_v_all[jja_indices,:,:],axis=0)
v_hist_canesm2_son = np.mean(wspd_hist_canesm2_v_all[son_indices,:,:],axis=0)

#wspd_canesm2_hist_djf = np.sqrt(u_hist_canesm2_djf**2 + v_hist_canesm2_djf**2)
#wspd_canesm2_hist_mam = np.sqrt(u_hist_canesm2_mam**2 + v_hist_canesm2_mam**2)
#wspd_canesm2_hist_jja = np.sqrt(u_hist_canesm2_jja**2 + v_hist_canesm2_jja**2)
#wspd_canesm2_hist_son = np.sqrt(u_hist_canesm2_son**2 + v_hist_canesm2_son**2)

wdir_canesm2_hist_djf = 270-np.degrees(np.arctan2(v_hist_canesm2_djf, u_hist_canesm2_djf)) #270- to flip to "from" dir and change to cardinal 
wdir_canesm2_hist_mam = 270-np.degrees(np.arctan2(v_hist_canesm2_mam, u_hist_canesm2_mam)) #270- to flip to "from" dir and change to cardinal 
wdir_canesm2_hist_jja = 270-np.degrees(np.arctan2(v_hist_canesm2_jja, u_hist_canesm2_jja)) #270- to flip to "from" dir and change to cardinal 
wdir_canesm2_hist_son = 270-np.degrees(np.arctan2(v_hist_canesm2_son, u_hist_canesm2_son)) #270- to flip to "from" dir and change to cardinal 

wspd_canesm2_all = np.sqrt(wspd_hist_canesm2_u_all**2 + wspd_hist_canesm2_v_all**2)

wspd_canesm2_hist_djf = np.mean(wspd_canesm2_all[djf_indices,:,:],axis=0)
wspd_canesm2_hist_mam = np.mean(wspd_canesm2_all[mam_indices,:,:],axis=0)
wspd_canesm2_hist_jja = np.mean(wspd_canesm2_all[jja_indices,:,:],axis=0)
wspd_canesm2_hist_son = np.mean(wspd_canesm2_all[son_indices,:,:],axis=0)


#%%

wind_canrcm4_u_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/uas_NAM-22_CCCma-CanESM2_historical_r1i1p1_CCCma-CanRCM4_r2_day_19790101-20051231.nc'
wind_canrcm4_u_hist_nc = Dataset(wind_canrcm4_u_hist_file,'r')
wspd_hist_canrcm4_u_all = np.squeeze(wind_canrcm4_u_hist_nc.variables['uas'][:])

lats_rcm = wind_canrcm4_u_hist_nc.variables['lat'][:]
lons_rcm = wind_canrcm4_u_hist_nc.variables['lon'][:]


wind_canrcm4_v_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/vas_NAM-22_CCCma-CanESM2_historical_r1i1p1_CCCma-CanRCM4_r2_day_19790101-20051231.nc'
wind_canrcm4_v_hist_nc = Dataset(wind_canrcm4_v_hist_file,'r')
wspd_hist_canrcm4_v_all = np.squeeze(wind_canrcm4_v_hist_nc.variables['vas'][:])


time_rcm = wind_canrcm4_u_hist_nc.variables['time'][:]
time_rcm_dates = cftime.num2date(time_rcm, wind_canrcm4_u_hist_nc.variables['time'].units, calendar=wind_canrcm4_u_hist_nc.variables['time'].calendar)

djf_indices = [i for i, dt in enumerate(time_rcm_dates) if dt.month in [12,1,2]]
mam_indices = [i for i, dt in enumerate(time_rcm_dates) if dt.month in [3,4,5]]
jja_indices = [i for i, dt in enumerate(time_rcm_dates) if dt.month in [6,7,8]]
son_indices = [i for i, dt in enumerate(time_rcm_dates) if dt.month in [9,10,11]]


u_hist_canrcm4_djf = np.mean(wspd_hist_canrcm4_u_all[djf_indices,:,:],axis=0)
u_hist_canrcm4_mam = np.mean(wspd_hist_canrcm4_u_all[mam_indices,:,:],axis=0)
u_hist_canrcm4_jja = np.mean(wspd_hist_canrcm4_u_all[jja_indices,:,:],axis=0)
u_hist_canrcm4_son = np.mean(wspd_hist_canrcm4_u_all[son_indices,:,:],axis=0)

v_hist_canrcm4_djf = np.mean(wspd_hist_canrcm4_v_all[djf_indices,:,:],axis=0)
v_hist_canrcm4_mam = np.mean(wspd_hist_canrcm4_v_all[mam_indices,:,:],axis=0)
v_hist_canrcm4_jja = np.mean(wspd_hist_canrcm4_v_all[jja_indices,:,:],axis=0)
v_hist_canrcm4_son = np.mean(wspd_hist_canrcm4_v_all[son_indices,:,:],axis=0)

#wspd_canrcm4_hist_djf = np.sqrt(u_hist_canrcm4_djf**2 + v_hist_canrcm4_djf**2)
#wspd_canrcm4_hist_mam = np.sqrt(u_hist_canrcm4_mam**2 + v_hist_canrcm4_mam**2)
#wspd_canrcm4_hist_jja = np.sqrt(u_hist_canrcm4_jja**2 + v_hist_canrcm4_jja**2)
#wspd_canrcm4_hist_son = np.sqrt(u_hist_canrcm4_son**2 + v_hist_canrcm4_son**2)

wdir_canrcm4_hist_djf = 270-np.degrees(np.arctan2(v_hist_canrcm4_djf, u_hist_canrcm4_djf)) #270- to flip to "from" dir and change to cardinal 
wdir_canrcm4_hist_mam = 270-np.degrees(np.arctan2(v_hist_canrcm4_mam, u_hist_canrcm4_mam)) #270- to flip to "from" dir and change to cardinal 
wdir_canrcm4_hist_jja = 270-np.degrees(np.arctan2(v_hist_canrcm4_jja, u_hist_canrcm4_jja)) #270- to flip to "from" dir and change to cardinal 
wdir_canrcm4_hist_son = 270-np.degrees(np.arctan2(v_hist_canrcm4_son, u_hist_canrcm4_son)) #270- to flip to "from" dir and change to cardinal 

wspd_canrcm4_all = np.sqrt(wspd_hist_canrcm4_u_all**2 + wspd_hist_canrcm4_v_all**2)

wspd_canrcm4_hist_djf = np.mean(wspd_canrcm4_all[djf_indices,:,:],axis=0)
wspd_canrcm4_hist_mam = np.mean(wspd_canrcm4_all[mam_indices,:,:],axis=0)
wspd_canrcm4_hist_jja = np.mean(wspd_canrcm4_all[jja_indices,:,:],axis=0)
wspd_canrcm4_hist_son = np.mean(wspd_canrcm4_all[son_indices,:,:],axis=0)


#%%

WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

def plot_map(wspd,wdir,cmap,vmin,vmax,xlabel,fig_name,title,dx,dy,lons,lats,width,scale):
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    ax1.pcolormesh(lons, lats, wspd, cmap=cmap, transform=ccrs.PlateCarree(),zorder=0,vmin=vmin,vmax=vmax)
    
    #plt.scatter(eccc_lons, eccc_lats, c=eccc_change,s=300,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    #plt.scatter(bch_lons, bch_lats, c=bch_change,s=300,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    #plt.scatter(noaa_lons, noaa_lats, c=noaa_change,s=300,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    
    #ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    #ax1.text(-3680871, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
    
    #plt.title(title,fontsize=20)
    
    ax1.quiver(lons, lats, dx, dy,color='k',angles=270-wdir, width=width, scale=scale,transform=ccrs.PlateCarree(),pivot='mid')


    #ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    ax1.set_extent([-143, -108, 37, 58], crs=ccrs.PlateCarree())

    #cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    #fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
    #              cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(-80, vmax+1, 20))
    #cbar_ax.tick_params(labelsize=22)
    
    #cbar_ax.set_xlabel(xlabel,size=24) 
    
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/future_changes/' + fig_name + '.png',bbox_inches='tight')

def reduce_wind(wdir_dx,wdir_dy,space):
    wdir_dx_reduced = wdir_dx.copy()
    wdir_dy_reduced = wdir_dy.copy()
    
    #space = 3 #needs to be odd
    half = int(space/2) #half - 0.5 
    
    for i in range(len(wdir_dx_reduced)):
        if i % space != 0:
            wdir_dx_reduced[i, :] = np.nan
            wdir_dx_reduced[:, i] = np.nan
            wdir_dy_reduced[i, :] = np.nan
            wdir_dy_reduced[:, i] = np.nan
            
        for j in range(len(wdir_dx_reduced)):
            if i % space == 0 and j % space == 0:
                  
                if i == 0 and j == 0:
                    wdir_dx_reduced[i,j] = np.mean(wdir_dx[i:i+half+1,j:j+half+1])
                    wdir_dy_reduced[i,j] = np.mean(wdir_dy[i:i+half+1,j:j+half+1])
                elif i == 0:
                    wdir_dx_reduced[i,j] = np.mean(wdir_dx[i:i+half+1,j-half:j+half+1])
                    wdir_dy_reduced[i,j] = np.mean(wdir_dy[i:i+half+1,j-half:j+half+1])
                elif j == 0:
                    wdir_dx_reduced[i,j] = np.mean(wdir_dx[i-half:i+half+1,j:j+half+1])
                    wdir_dy_reduced[i,j] = np.mean(wdir_dy[i-half:i+half+1,j:j+half+1])
                else:
                    wdir_dx_reduced[i,j] = np.mean(wdir_dx[i-half:i+half+1,j-half:j+half+1])
                    wdir_dy_reduced[i,j] = np.mean(wdir_dy[i-half:i+half+1,j-half:j+half+1])
    
    return(wdir_dx_reduced,wdir_dy_reduced)
    
#%%
#wdir_dx_reduced_hist_djf,wdir_dy_reduced_hist_djf = reduce_wind(u_d03_hist_djf,v_d03_hist_djf,space=11)
#wdir_dx_reduced_hist_mam,wdir_dy_reduced_hist_mam = reduce_wind(u_d03_hist_mam,v_d03_hist_mam,space=11)
#wdir_dx_reduced_hist_jja,wdir_dy_reduced_hist_jja = reduce_wind(u_d03_hist_jja,v_d03_hist_jja,space=11)
#wdir_dx_reduced_hist_son,wdir_dy_reduced_hist_son = reduce_wind(u_d03_hist_son,v_d03_hist_son,space=11)

wdir_dx_d02_reduced_hist_djf,wdir_dy_d02_reduced_hist_djf = reduce_wind(u_d02_hist_djf,v_d02_hist_djf,space=7)
wdir_dx_d02_reduced_hist_mam,wdir_dy_d02_reduced_hist_mam = reduce_wind(u_d02_hist_mam,v_d02_hist_mam,space=7)
wdir_dx_d02_reduced_hist_jja,wdir_dy_d02_reduced_hist_jja = reduce_wind(u_d02_hist_jja,v_d02_hist_jja,space=7)
wdir_dx_d02_reduced_hist_son,wdir_dy_d02_reduced_hist_son = reduce_wind(u_d02_hist_son,v_d02_hist_son,space=7)

#wdir_dx_reduced_rcp45_djf,wdir_dy_reduced_rcp45_djf = reduce_wind(wdir_rcp45_dx_djf,wdir_rcp45_dy_djf)
#wdir_dx_reduced_rcp85_djf,wdir_dy_reduced_rcp85_djf = reduce_wind(wdir_rcp85_dx_djf,wdir_rcp85_dy_djf)

#wdir_dx_reduced_rcp45_mam,wdir_dy_reduced_rcp45_mam = reduce_wind(wdir_rcp45_dx_mam,wdir_rcp45_dy_mam)
#wdir_dx_reduced_rcp85_mam,wdir_dy_reduced_rcp85_mam = reduce_wind(wdir_rcp85_dx_mam,wdir_rcp85_dy_mam)

#wdir_dx_reduced_rcp45_jja,wdir_dy_reduced_rcp45_jja = reduce_wind(wdir_rcp45_dx_jja,wdir_rcp45_dy_jja)
#wdir_dx_reduced_rcp85_jja,wdir_dy_reduced_rcp85_jja = reduce_wind(wdir_rcp85_dx_jja,wdir_rcp85_dy_jja)

#wdir_dx_reduced_rcp45_son,wdir_dy_reduced_rcp45_son = reduce_wind(wdir_rcp45_dx_son,wdir_rcp45_dy_son)
#wdir_dx_reduced_rcp85_son,wdir_dy_reduced_rcp85_son = reduce_wind(wdir_rcp85_dx_son,wdir_rcp85_dy_son)

u_hist_canrcm4_djf_reduced,v_hist_canrcm4_djf_reduced = reduce_wind(u_hist_canrcm4_djf,v_hist_canrcm4_djf,space=7)
u_hist_canrcm4_mam_reduced,v_hist_canrcm4_mam_reduced = reduce_wind(u_hist_canrcm4_mam,v_hist_canrcm4_mam,space=7)
u_hist_canrcm4_jja_reduced,v_hist_canrcm4_jja_reduced = reduce_wind(u_hist_canrcm4_jja,v_hist_canrcm4_jja,space=7)
u_hist_canrcm4_son_reduced,v_hist_canrcm4_son_reduced = reduce_wind(u_hist_canrcm4_son,v_hist_canrcm4_son,space=7)


#wdir_dx_reduced_hist[land==0] = np.nan
    
#%%
    
#plot_map(wspd_d03_hist_djf,wdir_d03_hist_djf,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","DJF (hist)",wdir_dx_reduced_hist_djf,wdir_dy_reduced_hist_djf,lons,lats,width=0.0025, scale=100)
#plot_map(wspd_d03_hist_mam,wdir_d03_hist_mam,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","MAM (hist)",wdir_dx_reduced_hist_mam,wdir_dy_reduced_hist_mam,lons,lats,width=0.0025, scale=100)
#plot_map(wspd_d03_hist_jja,wdir_d03_hist_jja,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","JJA (hist)",wdir_dx_reduced_hist_jja,wdir_dy_reduced_hist_jja,lons,lats,width=0.0025, scale=100)
#plot_map(wspd_d03_hist_son,wdir_d03_hist_son,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","SON (hist)",wdir_dx_reduced_hist_son,wdir_dy_reduced_hist_son,lons,lats,width=0.0025, scale=100)

plot_map(wspd_d02_hist_djf,wdir_d02_hist_djf,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","DJF (hist)",wdir_dx_d02_reduced_hist_djf,wdir_dy_d02_reduced_hist_djf,lons_d02,lats_d02,width=0.003, scale=100)
plot_map(wspd_d02_hist_mam,wdir_d02_hist_mam,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","MAM (hist)",wdir_dx_d02_reduced_hist_mam,wdir_dy_d02_reduced_hist_mam,lons_d02,lats_d02,width=0.003, scale=100)
plot_map(wspd_d02_hist_jja,wdir_d02_hist_jja,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","JJA (hist)",wdir_dx_d02_reduced_hist_jja,wdir_dy_d02_reduced_hist_jja,lons_d02,lats_d02,width=0.003, scale=100)
plot_map(wspd_d02_hist_son,wdir_d02_hist_son,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","SON (hist)",wdir_dx_d02_reduced_hist_son,wdir_dy_d02_reduced_hist_son,lons_d02,lats_d02,width=0.003, scale=100)


#plot_map(wspd_rcp45_djf,wdir_rcp45_djf,'jet',0,10, 'Avg Wind Speed (m/s)',"rcp45_mean_wspd","DJF (RCP4.5)",wdir_dx_reduced_rcp45_djf,wdir_dy_reduced_rcp45_djf)
#plot_map(wspd_rcp85_djf,wdir_rcp85_djf,'jet',0,10, 'Avg Wind Speed (m/s)',"rcp85_mean_wspd","DJF (RCP8.5)",wdir_dx_reduced_rcp85_djf,wdir_dy_reduced_rcp85_djf)
#plot_map(wspd_rcp45_djf-wspd_hist_djf,wdir_hist_djf,'bwr',-0.5,0.5, 'Avg Wind Speed Change (m/s)',"rcp45_mean_wspd_change","DJF (RCP4.5-hist)",[],[])
#plot_map(wspd_rcp85_djf-wspd_hist_djf,wdir_hist_djf,'bwr',-0.5,0.5, 'Avg Wind Speed Change (m/s)',"rcp85_mean_wspd_change","DJF (RCP8.5-hist)",[],[])

#plot_map(wspd_rcp45_mam,wdir_rcp45_mam,'jet',0,10, 'Avg Wind Speed (m/s)',"rcp45_mean_wspd","MAM (RCP4.5)",wdir_dx_reduced_rcp45_mam,wdir_dy_reduced_rcp45_mam)
#plot_map(wspd_rcp85_mam,wdir_rcp85_mam,'jet',0,10, 'Avg Wind Speed (m/s)',"rcp85_mean_wspd","MAM (RCP8.5)",wdir_dx_reduced_rcp85_mam,wdir_dy_reduced_rcp85_mam)
#plot_map(wspd_rcp45_mam-wspd_hist_mam,wdir_hist_mam,'bwr',-0.5,0.5, 'Avg Wind Speed Change (m/s)',"rcp45_mean_wspd_change","MAM (RCP4.5-hist)",[],[])
#plot_map(wspd_rcp85_mam-wspd_hist_mam,wdir_hist_mam,'bwr',-0.5,0.5, 'Avg Wind Speed Change (m/s)',"rcp85_mean_wspd_change","MAM (RCP8.5-hist)",[],[])

#plot_map(wspd_rcp45_jja,wdir_rcp45_jja,'jet',0,10, 'Avg Wind Speed (m/s)',"rcp45_mean_wspd","JJA (RCP4.5)",wdir_dx_reduced_rcp45_jja,wdir_dy_reduced_rcp45_jja)
#plot_map(wspd_rcp85_jja,wdir_rcp85_jja,'jet',0,10, 'Avg Wind Speed (m/s)',"rcp85_mean_wspd","JJA (RCP8.5)",wdir_dx_reduced_rcp85_jja,wdir_dy_reduced_rcp85_jja)
#plot_map(wspd_rcp45_jja-wspd_hist_jja,wdir_hist_jja,'bwr',-0.5,0.5, 'Avg Wind Speed Change (m/s)',"rcp45_mean_wspd_change","JJA (RCP4.5-hist)",[],[])
#plot_map(wspd_rcp85_jja-wspd_hist_jja,wdir_hist_jja,'bwr',-0.5,0.5, 'Avg Wind Speed Change (m/s)',"rcp85_mean_wspd_change","JJA (RCP8.5-hist)",[],[])


#plot_map(wspd_rcp45_son,wdir_rcp45_son,'jet',0,10, 'Avg Wind Speed (m/s)',"rcp45_mean_wspd","SON (RCP4.5)",wdir_dx_reduced_rcp45_son,wdir_dy_reduced_rcp45_son)
#plot_map(wspd_rcp85_son,wdir_rcp85_son,'jet',0,10, 'Avg Wind Speed (m/s)',"rcp85_mean_wspd","SON (RCP8.5)",wdir_dx_reduced_rcp85_son,wdir_dy_reduced_rcp85_son)
#plot_map(wspd_rcp45_son-wspd_hist_son,wdir_hist_son,'bwr',-0.5,0.5, 'Avg Wind Speed Change (m/s)',"rcp45_mean_wspd_change","SON (RCP4.5-hist)",[],[])
#plot_map(wspd_rcp85_son-wspd_hist_son,wdir_hist_son,'bwr',-0.5,0.5, 'Avg Wind Speed Change (m/s)',"rcp85_mean_wspd_change","SON (RCP8.5-hist)",[],[])


#plot_map(wspd_canesm2_hist_djf,wdir_canesm2_hist_djf,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","DJF (hist)",u_hist_canesm2_djf,v_hist_canesm2_djf,lons_gcm,lats_gcm,width=0.004, scale=100)
#plot_map(wspd_canesm2_hist_mam,wdir_canesm2_hist_mam,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","MAM (hist)",u_hist_canesm2_mam,v_hist_canesm2_mam,lons_gcm,lats_gcm,width=0.004, scale=100)
#plot_map(wspd_canesm2_hist_jja,wdir_canesm2_hist_jja,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","JJA (hist)",u_hist_canesm2_jja,v_hist_canesm2_jja,lons_gcm,lats_gcm,width=0.004, scale=100)
#plot_map(wspd_canesm2_hist_son,wdir_canesm2_hist_son,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","SON (hist)",u_hist_canesm2_son,v_hist_canesm2_son,lons_gcm,lats_gcm,width=0.004, scale=100)
#plot_map(wspd_canesm2_ann,wdir_canesm2_ann,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","SON (hist)",wspd_hist_canesm2_u_all,wspd_hist_canesm2_v_all,lons_gcm,lats_gcm)



#plot_map(wspd_canrcm4_hist_djf,wdir_canrcm4_hist_djf,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","DJF (hist)",u_hist_canrcm4_djf_reduced,v_hist_canrcm4_djf_reduced,lons_rcm,lats_rcm,width=0.003, scale=100)
#plot_map(wspd_canrcm4_hist_mam,wdir_canrcm4_hist_mam,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","MAM (hist)",u_hist_canrcm4_mam_reduced,v_hist_canrcm4_mam_reduced,lons_rcm,lats_rcm,width=0.003, scale=100)
#plot_map(wspd_canrcm4_hist_jja,wdir_canrcm4_hist_jja,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","JJA (hist)",u_hist_canrcm4_jja_reduced,v_hist_canrcm4_jja_reduced,lons_rcm,lats_rcm,width=0.003, scale=100)
#plot_map(wspd_canrcm4_hist_son,wdir_canrcm4_hist_son,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","SON (hist)",u_hist_canrcm4_son_reduced,v_hist_canrcm4_son_reduced,lons_rcm,lats_rcm,width=0.003, scale=100)
#plot_map(wspd_canrcm4_ann,wdir_canrcm4_ann,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd","SON (hist)",wspd_hist_canrcm4_u_all,wspd_hist_canrcm4_v_all,lons_rcm,lats_rcm)
