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



cwdi_rcp45_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/rcp45/cwdi.nc'
cwdi_rcp45_nc = Dataset(cwdi_rcp45_file,'r')
cwdi_rcp45_wrt_hist = np.squeeze(cwdi_rcp45_nc.variables['cold_wave_duration_index_wrt_mean_of_reference_period'][:])
cwdi_rcp45_count = np.squeeze(cwdi_rcp45_nc.variables['cold_waves_per_time_period'][:])

cwdi_rcp85_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/rcp85/cwdi.nc'
cwdi_rcp85_nc = Dataset(cwdi_rcp85_file,'r')
cwdi_rcp85_wrt_hist = np.squeeze(cwdi_rcp85_nc.variables['cold_wave_duration_index_wrt_mean_of_reference_period'][:])
cwdi_rcp85_count = np.squeeze(cwdi_rcp85_nc.variables['cold_waves_per_time_period'][:])

hwdi_rcp45_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/rcp45/hwdi.nc'
hwdi_rcp45_nc = Dataset(hwdi_rcp45_file,'r')
hwdi_rcp45_wrt_hist = np.squeeze(hwdi_rcp45_nc.variables['heat_wave_duration_index_wrt_mean_of_reference_period'][:])
hwdi_rcp45_count = np.squeeze(hwdi_rcp45_nc.variables['heat_waves_per_time_period'][:])

hwdi_rcp85_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/rcp85/hwdi.nc'
hwdi_rcp85_nc = Dataset(hwdi_rcp85_file,'r')
hwdi_rcp85_wrt_hist = np.squeeze(hwdi_rcp85_nc.variables['heat_wave_duration_index_wrt_mean_of_reference_period'][:])
hwdi_rcp85_count = np.squeeze(hwdi_rcp85_nc.variables['heat_waves_per_time_period'][:])

rr1_rcp45_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/rcp45/rr1.nc'
rr1_rcp45_nc = Dataset(rr1_rcp45_file,'r')
rr1_rcp45 = np.squeeze(rr1_rcp45_nc.variables['wet_days_index_per_time_period'][:])

rr1_rcp85_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/rcp85/rr1.nc'
rr1_rcp85_nc = Dataset(rr1_rcp85_file,'r')
rr1_rcp85 = np.squeeze(rr1_rcp85_nc.variables['wet_days_index_per_time_period'][:])

rr1_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/historical/rr1.nc'
rr1_hist_nc = Dataset(rr1_hist_file,'r')
rr1_hist = np.squeeze(rr1_hist_nc.variables['wet_days_index_per_time_period'][:])

sdii_rcp45_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/rcp45/sdii.nc'
sdii_rcp45_nc = Dataset(sdii_rcp45_file,'r')
sdii_rcp45 = np.squeeze(sdii_rcp45_nc.variables['simple_daily_intensity_index_per_time_period'][:])


sdii_rcp85_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/rcp85/sdii.nc'
sdii_rcp85_nc = Dataset(sdii_rcp85_file,'r')
sdii_rcp85 = np.squeeze(sdii_rcp85_nc.variables['simple_daily_intensity_index_per_time_period'][:])

sdii_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/climdex/historical/sdii.nc'
sdii_hist_nc = Dataset(sdii_hist_file,'r')
sdii_hist = np.squeeze(sdii_hist_nc.variables['simple_daily_intensity_index_per_time_period'][:])

#%%

tmax_95_rcp45_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/percentiles/rcp45/t_d03_tmax_95p.nc'
tmax_95_rcp45_nc = Dataset(tmax_95_rcp45_file,'r')
tmax_95_rcp45 = np.squeeze(tmax_95_rcp45_nc.variables['T2'][:])-273.15

lats = tmax_95_rcp45_nc.variables['lat'][:]
lons = tmax_95_rcp45_nc.variables['lon'][:]

tmax_95_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/percentiles/historical/t_d03_tmax_95p.nc'
tmax_95_hist_nc = Dataset(tmax_95_hist_file,'r')
tmax_95_hist = np.squeeze(tmax_95_hist_nc.variables['T2'][:])-273.15


#%%

pr_95_rcp45_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/percentiles/rcp45/pr_d03_daily_95p.nc'
pr_95_rcp45_nc = Dataset(pr_95_rcp45_file,'r')
pr_95_rcp45 = np.squeeze(pr_95_rcp45_nc.variables['pr'][:])

pr_95_hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/percentiles/historical/pr_d03_daily_95p.nc'
pr_95_hist_nc = Dataset(pr_95_hist_file,'r')
pr_95_hist = np.squeeze(pr_95_hist_nc.variables['pr'][:])


#%%

tmax_data_rcp45 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/t_d03_tmax_mean_rcp45.nc'
tmax_rcp45 = Dataset(tmax_data_rcp45, mode='r');
tmax_rcp45_mean = np.squeeze(tmax_rcp45.variables['T2'][:,:,:])
    
tmax_data_hist = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/t_d03_tmax_mean_hist.nc'
tmax_hist = Dataset(tmax_data_hist, mode='r');
tmax_hist_mean = np.squeeze(tmax_hist.variables['T2'][:,:,:])
        
tmax_mean_diff = tmax_rcp45_mean-tmax_hist_mean
#%%

lats = cwdi_rcp45_nc.variables['lat'][:]
lons = cwdi_rcp45_nc.variables['lon'][:]


geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])



WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

def plot_climdex(data,cmap,vmin,vmax,xlabel,fig_name):
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
    ax1.text(-3680871, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
    
    #plt.title(title,fontsize=20)
    
    
    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(-80, vmax+1, 20))
    cbar_ax.tick_params(labelsize=22)
    
    cbar_ax.set_xlabel(xlabel,size=24) 
    
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/percentiles/' + fig_name + '.png',bbox_inches='tight')

#%%

colors_pr = ['#8c5109','#a4671b','#c7974a','#d4b775','#f5e7c6','#c7e7e2','#80cdc1','#35978f','#12766e','#01665e']
newcmp_pr = pltcol.LinearSegmentedColormap.from_list("custom", colors_pr,N=16) 
under = '#543005'
over = '#003c30'
newcmp_pr.set_over(over) #add the max arrow color
newcmp_pr.set_under(under) #add the min arrow color



plot_climdex(tmax_95_hist,'jet',0,40, 'Tmax 95p (deg C)',"hist_tmax_daily_95p")
plot_climdex(tmax_95_rcp45,'jet',0,40, 'Tmax 95p (deg C)' ,"rcp45_tmax_daily_95p")
plot_climdex(tmax_95_rcp45-tmax_95_hist, cm.get_cmap('YlOrRd', 24),0,5, 'Tmax $\Delta$ 95p (deg C)',"diff45_tmax_daily_95p")

plot_climdex(tmax_95_rcp45-tmax_95_hist-tmax_mean_diff, cm.get_cmap('YlOrRd', 24),0,2, 'Tmax $\Delta$ 95p - $\Delta$ mean (deg C)',"diff45_tmax_daily_95p_minusmean")
#%%

plot_climdex(pr_95_hist,'gist_ncar',0,80, 'Precipitation 95p (mm/day)',"hist_pr_daily_95p")
plot_climdex(pr_95_rcp45,'gist_ncar',0,80, 'Precipitation 95p (mm/day)',"rcp45_pr_daily_95p")
plot_climdex(pr_95_rcp45-pr_95_hist,newcmp_pr,-10,10, 'Precipitation $\Delta$ 95p (mm/day)',"diff45_pr_daily_95p")
plot_climdex((pr_95_rcp45-pr_95_hist)/pr_95_hist * 100,newcmp_pr,-50,50, 'Precipitation $\Delta$ 95p (%)',"diff45_perc_pr_daily_95p")


#%%
plot_climdex(tmax_95_rcp45-tmax_95_hist-tmax_mean_diff, 'bwr_r',-5,5, 'Daily Precipitation Bias (mm/day)',"diff45_tmax_daily_95p_minusmean")

#%%

#plot_climdex(cwdi_rcp45_count/20,'Blues',0,3,'CWDI for 2046-2065 w.r.t. 1986-2005, RCP4.5 ','Cold Wave Duration Index')
#plot_climdex(cwdi_rcp85_count/20,'Blues',0,3,'CWDI for 2046-2065 w.r.t. 1986-2005, RCP8.5 ','Cold Wave Duration Index')

#plot_climdex(hwdi_rcp45_count/20,'Reds',0,10,'HWDI for 2046-2065 w.r.t. 1986-2005, RCP4.5 ','Heat Wave Duration Index')
#plot_climdex(hwdi_rcp85_count/20,'Reds',0,10,'HWDI for 2046-2065 w.r.t. 1986-2005, RCP8.5 ','Heat Wave Duration Index')

rr1_hist[land==0] = np.nan
rr1_rcp45[land==0] = np.nan
rr1_rcp85[land==0] = np.nan

sdii_hist[land==0] = np.nan
sdii_rcp45[land==0] = np.nan
sdii_rcp85[land==0] = np.nan

rr1_45_diff = (rr1_rcp45-rr1_hist)/20
rr1_85_diff = (rr1_rcp85-rr1_hist)/20

sdii_45_diff = (sdii_rcp45-sdii_hist)/20
sdii_85_diff = (sdii_rcp85-sdii_hist)/20

sdii_85_diff[sdii_85_diff<0.36] =np.nan

#plot_climdex(rr1_45_diff,newcmp_pr,-15,15,'2046-2065 (reference: 1986-2005), RCP4.5','Difference in Annual Wet Days','wet_day_diff_rcp45')
#plot_climdex(rr1_85_diff,newcmp_pr,-15,15,'2046-2065 (reference: 1986-2005), RCP8.5 ','Difference in Annual Wet Days','wet_day_diff_rcp85')

#plot_climdex(sdii_45_diff,newcmp_pr,-0.2,0.2,'2046-2065 (reference: 1986-2005), RCP4.5 ','Difference in annual Simple Daily Intensity Index','sdii_diff_rcp45')
plot_climdex(sdii_85_diff,newcmp_pr,-0.2,0.2,'2046-2065 (reference: 1986-2005), RCP8.5 ','Difference in annual Simple Daily Intensity Index','sdii_diff_rcp85')
