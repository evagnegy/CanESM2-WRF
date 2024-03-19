

from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys
import numpy as np
import datetime
import matplotlib.ticker as ticker
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic
import WRFDomainLib
import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib import cm
import xarray as xr
import scipy
import matplotlib as mpl
import matplotlib.colors as pltcol

#%%

variable = 't'
period = 'hist'

gridded_data_path = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/'

#%%
geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])



#%%
if variable == "wspd":
    filename = "wind_d03_mean_" + period + "_wspd"
else:
    filename = variable + "_d03_mean_" + period 
    
if variable == "t":
    wrf_d03_var_hist = Dataset(gridded_data_path + filename + '.nc','r').variables["T2"][:]-273.15
else:
    wrf_d03_var_hist = Dataset(gridded_data_path + filename + '.nc','r').variables[variable][:]

lons = Dataset(gridded_data_path + filename + '.nc','r').variables['lon'][:]
lats = Dataset(gridded_data_path + filename + '.nc','r').variables['lat'][:]

#%%
def plot_map(gridded_data,vmin,vmax,cmap):
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    ax1.pcolormesh(lons, lats, gridded_data, cmap=cmap, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),zorder=0)

    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    #ax1.text(-3700000, 700000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2)

    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal',extend='both')#,ticks=np.arange(0, vmax+1, 0.5))
    cbar_ax.tick_params(labelsize=25)
    
    
    if variable == "t":
        cbar_ax.set_xlabel('Temperature ($\degree$C)',size=25)

    elif variable == "pr": 
        cbar_ax.set_xlabel("Precipitation (mm/day)",size=25)    

    elif variable == "wspd":
        cbar_ax.set_xlabel("Wind speed (m/s)",size=25)   

    
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/means/' + period + '_' + variable + '_' + '.png',bbox_inches='tight')



if variable in ["t", "tmax", "tmin"]:
    vmin= -10
    vmax = 15
    cmap = 'jet'

elif variable == "pr":
    vmin= 0
    vmax = 15
    cmap = 'gist_ncar'
    
elif variable == "wspd":
    vmin= 0
    vmax = 10
    cmap = 'jet'
    

plot_map(np.squeeze(wrf_d03_var_hist), vmin,vmax,cmap)
