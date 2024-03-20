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

geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
lats_d03 = np.squeeze(geo_em_d03_nc.variables['XLAT_C'][:])
lons_d03 = np.squeeze(geo_em_d03_nc.variables['XLONG_C'][:])
topo_d03 = np.squeeze(geo_em_d03_nc.variables['HGT_M'][:])
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

geo_em_d02_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d02.nc'
geo_em_d02_nc = Dataset(geo_em_d02_file, mode='r')
lats_d02 = np.squeeze(geo_em_d02_nc.variables['XLAT_C'][:])
lons_d02 = np.squeeze(geo_em_d02_nc.variables['XLONG_C'][:])
topo_d02 = np.squeeze(geo_em_d02_nc.variables['HGT_M'][:])

#%%
def plot_windrose(wdir,wspd,title,savename):
    ax = WindroseAxes.from_ax()
    ax.bar(wdir, wspd, normed=True, opening=0.8, edgecolor='white',bins=[0,3,6,9,12])
    ax.legend(fontsize=14,loc='upper left')
    ax.set_yticks([5,10,15,20])
    ax.set_yticklabels([5,10,15,20],fontsize=20)

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
    
    plot_windrose(wdir_all,wspd_all,'D03 Wind entire domain (' + period + ')',period+'_all')
    plot_windrose(wdir_land,wspd_land,'D03 Wind over land (' + period + ')',period+'_land')
    plot_windrose(wdir_ocean,wspd_ocean,'D03 Wind over ocean (' + period + ')',period+'_ocean')

#%% this is for an average of all domain   

hist_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/wind_avgs/wind_weighted_means_hist.nc'
rcp45_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/wind_avgs/wind_weighted_means_rcp45.nc'
rcp85_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/wind_avgs/wind_weighted_means_rcp85.nc'

 
make_windrose_alldomain(hist_file,"hist")
make_windrose_alldomain(rcp45_file,"rcp45")
make_windrose_alldomain(rcp85_file,"rcp85")

#%% for point locations

WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

def windrose_plotoverlay(ax,ws,wd):
    ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white',bins=[0,3,6,9,12])
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_yticks([5,10,15,20])
    ax.set_yticklabels([5,10,15,20],fontsize=20)
    ax.set_ylim([0,25])
    
def plot_map(ws_889,wd_889,ws_94224,wd_94224,ws_24243,wd_24243,ws_24233,wd_24233,ws_24229,wd_24229,ws_277,wd_277,ws_202,wd_202,ws_640,wd_640,ws_155,wd_155,ws_381,wd_381,ws_1275,wd_1275,ws_1053,wd_1053,ws_348,wd_348,ws_118,wd_118):
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    ax1.pcolormesh(lons_d02, lats_d02, topo_d02, cmap='terrain', vmin=0,vmax=3000, alpha=0.3, transform=ccrs.PlateCarree(),zorder=0)
    ax1.pcolormesh(lons_d03, lats_d03, topo_d03, cmap='terrain', vmin=0,vmax=3000, alpha=0.4, transform=ccrs.PlateCarree(),zorder=0)

    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    #ax1.text(-3680871, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)
        
    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    # 889: -123.18, 49.2 YVR
    wrax1 = inset_axes(ax1, width=0.9, height=0.9, bbox_to_anchor=(1310,1050),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax1,ws_889,wd_889)
    
# =============================================================================
#     #USW00094224: -123.88326, 46.15694 coastal WA
#     wrax2 = inset_axes(ax1, width=1.5, height=1.5, bbox_to_anchor=(1040,760),axes_class=WindroseAxes)
#     windrose_plotoverlay(wrax2,ws_94224,wd_94224)
#     
#     #USW00024243: -120.53488, 46.56398 interior WA
#     wrax3 = inset_axes(ax1, width=1.5, height=1.5, bbox_to_anchor=(1380,600),axes_class=WindroseAxes)
#     windrose_plotoverlay(wrax3,ws_24243,wd_24243)
#     
#     #USW00024233: -122.31442, 47.44467 Seattle
#     wrax4 = inset_axes(ax1, width=1.2, height=1.2, bbox_to_anchor=(1270,780),axes_class=WindroseAxes)
#     windrose_plotoverlay(wrax4,ws_24233,wd_24233)
#     
#     #USW00024229: -122.60919, 45.59578 Portland
#     wrax5 = inset_axes(ax1, width=1.2, height=1.2, bbox_to_anchor=(1080,560),axes_class=WindroseAxes)
#     windrose_plotoverlay(wrax5,ws_24229,wd_24229)
# =============================================================================
    
    #277: -125.77,49.08 Tofino
    wrax6 = inset_axes(ax1, width=1.2, height=1.2, bbox_to_anchor=(1120,1200),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax6,ws_277,wd_277)
    
    #202: -127.37,50.68 Port Hardy
    wrax7 = inset_axes(ax1, width=1.2, height=1.2, bbox_to_anchor=(1150,1500),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax7,ws_202,wd_202)
    
    #640: -122.51,53.03 Quesnel
    wrax8 = inset_axes(ax1, width=1.5, height=1.5, bbox_to_anchor=(1770,1550),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax8,ws_640,wd_640)
    
    #155: -124.9,49.72 Comox
    wrax9 = inset_axes(ax1, width=0.8, height=0.8, bbox_to_anchor=(1210,1190),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax9,ws_155,wd_155)
    
    #381: -126.6,52.39 Bella Coola
    wrax10 = inset_axes(ax1, width=1.4, height=1.4, bbox_to_anchor=(1380,1680),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax10,ws_381,wd_381)
    
    #1275: -120.44,50.7 Kamloops
    wrax11 = inset_axes(ax1, width=1.1, height=1.1, bbox_to_anchor=(1700,1090),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax11,ws_1275,wd_1275)
    
    #1053: -119.6,49.46 Penticton
    wrax12 = inset_axes(ax1, width=1.1, height=1.1, bbox_to_anchor=(1660,875),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax12,ws_1053,wd_1053)
    
    #348: -122.95,50.13 Whistler
    wrax13 = inset_axes(ax1, width=0.8, height=0.8, bbox_to_anchor=(1410,1130),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax13,ws_348,wd_348)
    
    #118: -123.43,48.65 Victoria
    wrax14 = inset_axes(ax1, width=0.8, height=0.8, bbox_to_anchor=(1230,950),axes_class=WindroseAxes)
    windrose_plotoverlay(wrax14,ws_118,wd_118)
    
    
    
    #ax1.scatter(-122.36,49.03, s=300,color='r',transform=ccrs.PlateCarree(),zorder=100)


    #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/future_changes/' + fig_name + '.png',bbox_inches='tight')


def make_windrose_station(st,period):
    hist_file_d03 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/historical/wind_d03_ECCC_' + st + '.nc'
    nc = Dataset(hist_file_d03,'r')
    
    wspd = np.squeeze(nc.variables['wspd'][:])
    wdir = np.squeeze(nc.variables['wdir'][:])
    
    plot_windrose(wdir,wspd,'D03 Wind at ' + st + ' (' + period + ')',period+'_'+st)
    
def get_station(agency,st,period):
    file_d03 = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/' + period + '/wind_d03_' + agency + '_' + st + '.nc'
    nc = Dataset(file_d03,'r')
    
    wspd = np.squeeze(nc.variables['wspd'][:])
    wdir = np.squeeze(nc.variables['wdir'][:])
        
    return(wspd,wdir)

def get_station_obs_eccc(st):
    
    t_all_station,t_all_station_wdir = [],[]
    for year in np.arange(1986, 2005+1).tolist():
        t_df_temp,t_df_temp_wdir = [],[]
        for month in ['1','2','3','4','5','6','7','8','9','10','11','12']:
            file = glob.glob('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/hourly/ECCC/' + st + "/" + month + '/*' + str(year) + '_*.csv')[0]
            df = pd.read_csv(file) 
            t_df_temp.append(list(df["Wind Spd (km/h)"]))
            t_df_temp_wdir.append(list(df["Wind Dir (10s deg)"]))
            
            t_df = [item for sublist in t_df_temp for item in sublist]
            t_df_wdir = [item for sublist in t_df_temp_wdir for item in sublist]
            
        t_all_station.append(t_df)
        t_all_station_wdir.append(t_df_wdir)
    
    flat_list_wspd = [item for sublist in t_all_station for item in sublist]
    flat_list_wdir = [item for sublist in t_all_station_wdir for item in sublist]

    wspd = np.array(flat_list_wspd)/3.6
    wdir = np.array(flat_list_wdir)*10
    
    return(wspd,wdir)

#%% historical obs

wspd_889,wdir_889 = get_station_obs_eccc('889')
wspd_94224,wdir_94224 = [],[]
wspd_24243,wdir_24243 = [],[]
wspd_24233,wdir_24233 = [],[]
wspd_24229,wdir_24229 = [],[]
wspd_277,wdir_277 = get_station_obs_eccc('277')
wspd_202,wdir_202 = get_station_obs_eccc('202')
wspd_640,wdir_640 = get_station_obs_eccc('640')
wspd_155,wdir_155 = get_station_obs_eccc('155')
wspd_381,wdir_381 = get_station_obs_eccc('381')
wspd_1275,wdir_1275 = get_station_obs_eccc('1275')
wspd_1053,wdir_1053 = get_station_obs_eccc('1053')
wspd_348,wdir_348 = get_station_obs_eccc('348')
wspd_118,wdir_118 = get_station_obs_eccc('118')


wspd_889_removed = wspd_889[~np.isnan(wdir_889)]
wdir_889_removed = wdir_889[~np.isnan(wdir_889)]
wspd_277_removed = wspd_277[~np.isnan(wdir_277)]
wdir_277_removed = wdir_277[~np.isnan(wdir_277)]
wspd_202_removed = wspd_202[~np.isnan(wdir_202)]
wdir_202_removed = wdir_202[~np.isnan(wdir_202)]
wspd_640_removed = wspd_640[~np.isnan(wdir_640)]
wdir_640_removed = wdir_640[~np.isnan(wdir_640)]
wspd_155_removed = wspd_155[~np.isnan(wdir_155)]
wdir_155_removed = wdir_155[~np.isnan(wdir_155)]
wspd_381_removed = wspd_381[~np.isnan(wdir_381)]
wdir_381_removed = wdir_381[~np.isnan(wdir_381)]
wspd_1275_removed = wspd_1275[~np.isnan(wdir_1275)]
wdir_1275_removed = wdir_1275[~np.isnan(wdir_1275)]
wspd_1053_removed = wspd_1053[~np.isnan(wdir_1053)]
wdir_1053_removed = wdir_1053[~np.isnan(wdir_1053)]
wspd_348_removed = wspd_348[~np.isnan(wdir_348)]
wdir_348_removed = wdir_348[~np.isnan(wdir_348)]
wspd_118_removed = wspd_118[~np.isnan(wdir_118)]
wdir_118_removed = wdir_118[~np.isnan(wdir_118)]

plot_map(wspd_889_removed,wdir_889_removed,wspd_94224,wdir_94224,wspd_24243,wdir_24243,wspd_24233,wdir_24233,wspd_24229,wdir_24229,wspd_277_removed,wdir_277_removed,wspd_202_removed,wdir_202_removed,wspd_640_removed,wdir_640_removed,wspd_155_removed,wdir_155_removed,wspd_381_removed,wdir_381_removed,wspd_1275_removed,wdir_1275_removed,wspd_1053_removed,wdir_1053_removed,wspd_348_removed,wdir_348_removed,wspd_118_removed,wdir_118_removed)
    


#%% D03 historical

wspd_889_d03,wdir_889_d03 = get_station('ECCC','889','historical')
wspd_94224_d03,wdir_94224_d03 = get_station('NOAA','USW00094224','historical')
wspd_24243_d03,wdir_24243_d03 = get_station('NOAA','USW00024243','historical')
wspd_24233_d03,wdir_24233_d03 = get_station('NOAA','USW00024233','historical')
wspd_24229_d03,wdir_24229_d03 = get_station('NOAA','USW00024229','historical')
wspd_277_d03,wdir_277_d03 = get_station('ECCC','277','historical')
wspd_202_d03,wdir_202_d03 = get_station('ECCC','202','historical')
wspd_640_d03,wdir_640_d03 = get_station('ECCC','640','historical')
wspd_155_d03,wdir_155_d03 = get_station('ECCC','155','historical')
wspd_381_d03,wdir_381_d03 = get_station('ECCC','381','historical')
wspd_1275_d03,wdir_1275_d03 = get_station('ECCC','1275','historical')
wspd_1053_d03,wdir_1053_d03 = get_station('ECCC','1053','historical')
wspd_348_d03,wdir_348_d03 = get_station('ECCC','348','historical')
wspd_118_d03,wdir_118_d03 = get_station('ECCC','118','historical')

# =============================================================================
# wspd_889_d03_removed = wspd_889_d03[~np.isnan(wdir_889)]
# wdir_889_d03_removed = wdir_889_d03[~np.isnan(wdir_889)]
# wspd_277_d03_removed = wspd_277_d03[~np.isnan(wdir_277)]
# wdir_277_d03_removed = wdir_277_d03[~np.isnan(wdir_277)]
# wspd_202_d03_removed = wspd_202_d03[~np.isnan(wdir_202)]
# wdir_202_d03_removed = wdir_202_d03[~np.isnan(wdir_202)]
# wspd_640_d03_removed = wspd_640_d03[~np.isnan(wdir_640)]
# wdir_640_d03_removed = wdir_640_d03[~np.isnan(wdir_640)]
# wspd_155_d03_removed = wspd_155_d03[~np.isnan(wdir_155)]
# wdir_155_d03_removed = wdir_155_d03[~np.isnan(wdir_155)]
# wspd_381_d03_removed = wspd_381_d03[~np.isnan(wdir_381)]
# wdir_381_d03_removed = wdir_381_d03[~np.isnan(wdir_381)]
# wspd_1275_d03_removed = wspd_1275_d03[~np.isnan(wdir_1275)]
# wdir_1275_d03_removed = wdir_1275_d03[~np.isnan(wdir_1275)]
# wspd_1053_d03_removed = wspd_1053_d03[~np.isnan(wdir_1053)]
# wdir_1053_d03_removed = wdir_1053_d03[~np.isnan(wdir_1053)]
# wspd_348_d03_removed = wspd_348_d03[~np.isnan(wdir_348)]
# wdir_348_d03_removed = wdir_348_d03[~np.isnan(wdir_348)]
# wspd_118_d03_removed = wspd_118_d03[~np.isnan(wdir_118)]
# wdir_118_d03_removed = wdir_118_d03[~np.isnan(wdir_118)]
# 
# plot_map(wspd_889_d03_removed,wdir_889_d03_removed,[],[],[],[],[],[],[],[],wspd_277_d03_removed,wdir_277_d03_removed,wspd_202_d03_removed,wdir_202_d03_removed,wspd_640_d03_removed,wdir_640_d03_removed,wspd_155_d03_removed,wdir_155_d03_removed,wspd_381_d03_removed,wdir_381_d03_removed,wspd_1275_d03_removed,wdir_1275_d03_removed,wspd_1053_d03_removed,wdir_1053_d03_removed,wspd_348_d03_removed,wdir_348_d03_removed,wspd_118_d03_removed,wdir_118_d03_removed)
# 
# =============================================================================
plot_map(wspd_889_d03,wdir_889_d03,wspd_94224_d03,wdir_94224_d03,wspd_24243_d03,wdir_24243_d03,wspd_24233_d03,wdir_24233_d03,wspd_24229_d03,wdir_24229_d03,wspd_277_d03,wdir_277_d03,wspd_202_d03,wdir_202_d03,wspd_640_d03,wdir_640_d03,wspd_155_d03,wdir_155_d03,wspd_381_d03,wdir_381_d03,wspd_1275_d03,wdir_1275_d03,wspd_1053_d03,wdir_1053_d03,wspd_348_d03,wdir_348_d03,wspd_118_d03,wdir_118_d03)

#%%

wspd_889,wdir_889 = get_station('ECCC','889','rcp45')
#wspd_94224,wdir_94224 = get_station('NOAA','USW00094224','rcp45')
#wspd_24243,wdir_24243 = get_station('NOAA','USW00024243','rcp45')
#wspd_24233,wdir_24233 = get_station('NOAA','USW00024233','rcp45')
#wspd_24229,wdir_24229 = get_station('NOAA','USW00024229','rcp45')
wspd_277,wdir_277 = get_station('ECCC','277','rcp45')
wspd_202,wdir_202 = get_station('ECCC','202','rcp45')
wspd_640,wdir_640 = get_station('ECCC','640','rcp45')
wspd_155,wdir_155 = get_station('ECCC','155','rcp45')
wspd_381,wdir_381 = get_station('ECCC','381','rcp45')
wspd_1275,wdir_1275 = get_station('ECCC','1275','rcp45')
wspd_1053,wdir_1053 = get_station('ECCC','1053','rcp45')
wspd_348,wdir_348 = get_station('ECCC','348','rcp45')
wspd_118,wdir_118 = get_station('ECCC','118','rcp45')

plot_map(wspd_889,wdir_889,wspd_94224,wdir_94224,wspd_24243,wdir_24243,wspd_24233,wdir_24233,wspd_24229,wdir_24229,wspd_277,wdir_277,wspd_202,wdir_202,wspd_640,wdir_640,wspd_155,wdir_155,wspd_381,wdir_381,wspd_1275,wdir_1275,wspd_1053,wdir_1053,wspd_348,wdir_348,wspd_118,wdir_118)
    
#%%

wspd_889,wdir_889 = get_station('ECCC','889','rcp85')
# =============================================================================
# wspd_94224,wdir_94224 = get_station('NOAA','USW00094224','rcp85')
# wspd_24243,wdir_24243 = get_station('NOAA','USW00024243','rcp85')
# wspd_24233,wdir_24233 = get_station('NOAA','USW00024233','rcp85')
# wspd_24229,wdir_24229 = get_station('NOAA','USW00024229','rcp85')
# =============================================================================
wspd_277,wdir_277 = get_station('ECCC','277','rcp85')
wspd_202,wdir_202 = get_station('ECCC','202','rcp85')
wspd_640,wdir_640 = get_station('ECCC','640','rcp85')
wspd_155,wdir_155 = get_station('ECCC','155','rcp85')
wspd_381,wdir_381 = get_station('ECCC','381','rcp85')
wspd_1275,wdir_1275 = get_station('ECCC','1275','rcp85')
wspd_1053,wdir_1053 = get_station('ECCC','1053','rcp85')
wspd_348,wdir_348 = get_station('ECCC','348','rcp85')
wspd_118,wdir_118 = get_station('ECCC','118','rcp85')

plot_map(wspd_889,wdir_889,wspd_94224,wdir_94224,wspd_24243,wdir_24243,wspd_24233,wdir_24233,wspd_24229,wdir_24229,wspd_277,wdir_277,wspd_202,wdir_202,wspd_640,wdir_640,wspd_155,wdir_155,wspd_381,wdir_381,wspd_1275,wdir_1275,wspd_1053,wdir_1053,wspd_348,wdir_348,wspd_118,wdir_118)
    

#%%
#make_windrose_station('889','hist')

