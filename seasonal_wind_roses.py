
import numpy as np
from windrose import WindroseAxes
from netCDF4 import Dataset
import matplotlib.pyplot as plt 
import WRFDomainLib
import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import datetime
import glob
import cftime

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
lats_d03 = np.squeeze(geo_em_d03_nc.variables['XLAT_C'][:])
lons_d03 = np.squeeze(geo_em_d03_nc.variables['XLONG_C'][:])
topo_d03 = np.squeeze(geo_em_d03_nc.variables['HGT_M'][:])
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

#%%
def plot_windrose(wdir,wspd,title,savename):
    ax = WindroseAxes.from_ax()
    ax.bar(wdir, wspd, normed=True, opening=0.8, edgecolor='white',bins=[0,3,6,9,12])
    ax.legend(fontsize=14,loc='upper left')
    ax.set_yticks([10,20,30,40])
    ax.set_yticklabels([10,20,30,40],fontsize=20)

    ax.set_title(title,fontsize=20)
    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/windroses/' + savename + '.png',bbox_inches='tight')
    
def make_windrose_alldomain(file,period):

    nc = Dataset(file,'r')
    
    wspd_all = np.squeeze(nc.variables['wspd_all'][:])
    wdir_all = np.squeeze(nc.variables['wdir_all'][:])
    wspd_land = np.squeeze(nc.variables['wspd_land'][:])
    wdir_land = np.squeeze(nc.variables['wdir_land'][:])
    wspd_ocean = np.squeeze(nc.variables['wspd_ocean'][:])
    wdir_ocean = np.squeeze(nc.variables['wdir_ocean'][:])
    
    time = nc.variables['time'][:]
    time_dates = cftime.num2date(time, nc.variables['time'].units, calendar=nc.variables['time'].calendar)

    djf_indices = [i for i, dt in enumerate(time_dates) if dt.month in [12,1,2]]
    mam_indices = [i for i, dt in enumerate(time_dates) if dt.month in [3,4,5]]
    jja_indices = [i for i, dt in enumerate(time_dates) if dt.month in [6,7,8]]
    son_indices = [i for i, dt in enumerate(time_dates) if dt.month in [9,10,11]]
    
    wspd_all_djf = wspd_all[djf_indices]
    wspd_all_mam = wspd_all[mam_indices]
    wspd_all_jja = wspd_all[jja_indices]
    wspd_all_son = wspd_all[son_indices]
    
    wdir_all_djf = wdir_all[djf_indices]
    wdir_all_mam = wdir_all[mam_indices]
    wdir_all_jja = wdir_all[jja_indices]
    wdir_all_son = wdir_all[son_indices]

    wspd_land_djf = wspd_land[djf_indices]
    wspd_land_mam = wspd_land[mam_indices]
    wspd_land_jja = wspd_land[jja_indices]
    wspd_land_son = wspd_land[son_indices]
    
    wdir_land_djf = wdir_land[djf_indices]
    wdir_land_mam = wdir_land[mam_indices]
    wdir_land_jja = wdir_land[jja_indices]
    wdir_land_son = wdir_land[son_indices]
    
    wspd_ocean_djf = wspd_ocean[djf_indices]
    wspd_ocean_mam = wspd_ocean[mam_indices]
    wspd_ocean_jja = wspd_ocean[jja_indices]
    wspd_ocean_son = wspd_ocean[son_indices]
    
    wdir_ocean_djf = wdir_ocean[djf_indices]
    wdir_ocean_mam = wdir_ocean[mam_indices]
    wdir_ocean_jja = wdir_ocean[jja_indices]
    wdir_ocean_son = wdir_ocean[son_indices]
    
    #plot_windrose(wdir_all_djf,wspd_all_djf,'D03 Wind entire domain (DJF ' + period + ')',period+'_all')
    #plot_windrose(wdir_all_mam,wspd_all_mam,'D03 Wind entire domain (MAM ' + period + ')',period+'_all')
    #plot_windrose(wdir_all_jja,wspd_all_jja,'D03 Wind entire domain (JJA ' + period + ')',period+'_all')
    #plot_windrose(wdir_all_son,wspd_all_son,'D03 Wind entire domain (SON ' + period + ')',period+'_all')

    #plot_windrose(wdir_land_djf,wspd_land_djf,'D03 Wind over land (DJF ' + period + ')',period+'_all')
    #plot_windrose(wdir_land_mam,wspd_land_mam,'D03 Wind over land (MAM ' + period + ')',period+'_all')
    #plot_windrose(wdir_land_jja,wspd_land_jja,'D03 Wind over land (JJA ' + period + ')',period+'_all')
    #plot_windrose(wdir_land_son,wspd_land_son,'D03 Wind over land (SON ' + period + ')',period+'_all')

    plot_windrose(wdir_ocean_djf,wspd_ocean_djf,'D03 Wind over ocean (DJF ' + period + ')',period+'_all')
    plot_windrose(wdir_ocean_mam,wspd_ocean_mam,'D03 Wind over ocean (MAM ' + period + ')',period+'_all')
    plot_windrose(wdir_ocean_jja,wspd_ocean_jja,'D03 Wind over ocean (JJA ' + period + ')',period+'_all')
    plot_windrose(wdir_ocean_son,wspd_ocean_son,'D03 Wind over ocean (SON ' + period + ')',period+'_all')


#% this is for an average of all domain   

hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/wind_avgs/wind_weighted_means_hist.nc'
rcp45_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/wind_avgs/wind_weighted_means_rcp45.nc'
rcp85_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/wind_avgs/wind_weighted_means_rcp85.nc'

 
make_windrose_alldomain(hist_file,"hist")
make_windrose_alldomain(rcp45_file,"rcp45")
#make_windrose_alldomain(rcp85_file,"rcp85")