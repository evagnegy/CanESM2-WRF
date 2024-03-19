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
from windrose import WindroseAxes
#%%


wind_hist_file = '/Users/evagnegy/Downloads/wind_sample_d03.nc'
wind_hist_nc = Dataset(wind_hist_file,'r')
wspd_hist = np.squeeze(wind_hist_nc.variables['wspd'][:])
wdir_hist = np.squeeze(wind_hist_nc.variables['wdir'][:])

lats = wind_hist_nc.variables['lat'][:]
lons = wind_hist_nc.variables['lon'][:]

wdir_hist_rad = np.radians(wdir_hist)


wdir_hist_dx = (wspd_hist) * np.cos(wdir_hist_rad)
wdir_hist_dy = (wspd_hist) * np.sin(wdir_hist_rad)


WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

def plot_climdex(data,cmap,vmin,vmax,xlabel,fig_name,dx,dy):
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    ax1.pcolormesh(lons, lats, data, cmap=cmap, transform=ccrs.PlateCarree(),zorder=0,vmin=vmin,vmax=vmax)
    
    #plt.scatter(eccc_lons, eccc_lats, c=eccc_change,s=300,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    #plt.scatter(bch_lons, bch_lats, c=bch_change,s=300,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    #plt.scatter(noaa_lons, noaa_lats, c=noaa_change,s=300,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),edgecolor='k',zorder=3)
    
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    #ax1.text(-3680871, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
    
    #plt.title(title,fontsize=20)
    
    ax1.quiver(lons, lats, dx, dy, color='red', width=0.005, scale=80,transform=ccrs.PlateCarree())


    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(-80, vmax+1, 20))
    cbar_ax.tick_params(labelsize=22)
    
    cbar_ax.set_xlabel(xlabel,size=24) 
    
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/future_changes/' + fig_name + '.png',bbox_inches='tight')

def reduce_wind(wdir_dx,wdir_dy):
    wdir_dx_reduced = wdir_dx.copy()
    wdir_dy_reduced = wdir_dy.copy()
    
    space = 11 #needs to be odd
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
for i in range(24):
    wdir_dx_reduced_hist,wdir_dy_reduced_hist = reduce_wind(wdir_hist_dx[i,:,:],wdir_hist_dy[i,:,:])

    wdir_dx_reduced_hist[land==0] = np.nan
    wdir_dy_reduced_hist[land==0] = np.nan
    
    plot_climdex(wspd_hist[i,:,:],'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd",wdir_dx_reduced_hist,wdir_dy_reduced_hist)