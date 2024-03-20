import matplotlib.pyplot as plt 
import matplotlib as mpl
import cartopy.crs as ccrs
from netCDF4 import Dataset
import numpy as np
import matplotlib.cbook
import WRFDomainLib
import cartopy.feature as cf

geo_em_d03_file = '/home/spfm000/space/CanESM2-WRF/scripts/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

#%%

wind_hist_file = '/home/spfm000/space/CanESM2-WRF/historical/variables_complete/wind_d03.nc'
wind_hist_nc = Dataset(wind_hist_file,'r')
lats = wind_hist_nc.variables['lat'][:]
lons = wind_hist_nc.variables['lon'][:]

print(wind_hist_nc)
print(wind_hist_nc.variables)

u_d03_hist_all = np.squeeze(wind_hist_nc.variables['u'][:])
v_d03_hist_all = np.squeeze(wind_hist_nc.variables['v'][:])

#calculates spd and dir at each hour
wspd_d03_hist_all = np.sqrt(u_d03_hist_all**2 + v_d03_hist_all**2)
wdir_d03_hist_all = 270-np.degrees(np.arctan2(v_d03_hist_all, u_d03_hist_all)) 

print(np.shape(wspd_d03_hist_all))

wspd_d03_hist = np.mean(wspd_d03_hist_all,axis=0)
wdir_d03_hist = np.mean(wdir_d03_hist_all,axis=0)

wspd_d03_hist_u = np.mean(u_d03_hist_all,axis=0)
wspd_d03_hist_v = np.mean(v_d03_hist_all,axis=0)

#%%

WPSFile = '/home/spfm000/space/CanESM2-WRF/scripts/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

def plot_map(wspd,wdir,cmap,vmin,vmax,xlabel,fig_name,dx,dy,lons,lats):
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
    
    ax1.quiver(lons, lats, dx, dy,color='k',angles=270-wdir, width=0.0025, scale=90,transform=ccrs.PlateCarree(),pivot='mid')


    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    #ax1.set_extent([-143, -108, 37, 58], crs=ccrs.PlateCarree())

    #cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    #fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
    #              cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(-80, vmax+1, 20))
    #cbar_ax.tick_params(labelsize=22)
    
    #cbar_ax.set_xlabel(xlabel,size=24) 
    
    plt.savefig('/home/spfm000/space/CanESM2-WRF/scripts/test.png',bbox_inches='tight')

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
    

wdir_dx_reduced_hist,wdir_dy_reduced_hist = reduce_wind(u_d03_hist_all,v_d03_hist_all)
#wdir_dx_reduced_rcp45,wdir_dy_reduced_rcp45 = reduce_wind(wdir_rcp45_dx,wdir_rcp45_dy)
#wdir_dx_reduced_rcp85,wdir_dy_reduced_rcp85 = reduce_wind(wdir_rcp85_dx,wdir_rcp85_dy)

#wspd_hist_canrcm4_u_reduced,wspd_hist_canrcm4_v_reduced = reduce_wind(wspd_hist_canrcm4_u,wspd_hist_canrcm4_v)

#wdir_dx_reduced_hist_d02,wdir_dy_reduced_hist_d02 = reduce_wind(wdir_hist_dx_d02,wdir_hist_dy_d02)


#wdir_dx_reduced_hist[land==0] = np.nan
    
    
plot_map(wspd_d03_hist,wdir_d03_hist,'jet',0,5, 'Avg Wind Speed (m/s)',"hist_mean_wspd",wdir_dx_reduced_hist,wdir_dy_reduced_hist,lons,lats)
#plot_map(wspd_rcp45,wdir_rcp45,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd",wdir_dx_reduced_rcp45,wdir_dy_reduced_rcp45)
#plot_map(wspd_rcp85,wdir_rcp85,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd",wdir_dx_reduced_rcp85,wdir_dy_reduced_rcp85)

#plot_map(wspd_rcp45-wspd,'bwr',-0.5,0.5, 'Avg Wind Speed (m/s)',"rcp45_mean_wspd_diff_w_ocean")
#plot_map(wspd_rcp85-wspd,'bwr',-0.5,0.5, 'Avg Wind Speed (m/s)',"rcp85_mean_wspd_diff_w_ocean")

#plot_map(wspd_canesm2_hist,wdir_canesm2_hist,'jet',0,5, 'Avg Wind Speed (m/s)',"hist_mean_wspd_canesm2",wspd_hist_canesm2_u,wspd_hist_canesm2_v,lons_gcm,lats_gcm)
#plot_map(wspd_canrcm4_hist,wdir_canrcm4_hist,'jet',0,5, 'Avg Wind Speed (m/s)',"hist_mean_wspd_canrcm4",wspd_hist_canrcm4_u_reduced,wspd_hist_canrcm4_v_reduced,lons_rcm,lats_rcm)

#plot_map(wspd_hist_d02,wdir_hist_d02,'jet',0,10, 'Avg Wind Speed (m/s)',"hist_mean_wspd_d02",wdir_dx_reduced_hist_d02,wdir_dy_reduced_hist_d02,lons_d02,lats_d02)
