import xarray as xr
import numpy as np
from windrose import WindroseAxes
from netCDF4 import Dataset
import matplotlib.pyplot as plt 
import sys

path = sys.argv[1]
year = sys.argv[2]


geo_em_d03_file = '/home/spfm000/space/CanESM2-WRF/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])

og_file = '/home/spfm000/space/CanESM2-WRF/' + path + '/variables_complete/wind_d03_' + year + '.nc'
output_file = '/home/spfm000/space/CanESM2-WRF/' + path + '/variables_complete/wind_weighted_means_' + year + '.nc'


time = Dataset(og_file,'r').variables['time'][:]


wind_ds = xr.open_dataset(og_file)

wind_ds_land_mask = wind_ds.where(land)
wind_ds_ocean_mask = wind_ds.where(land==False)

#fig1 = plt.figure(figsize=(10, 10),dpi=200)
#ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#ax1.pcolormesh(wind_ds.lons, wind_ds.lats, wind_ds.wspd[0,:,:], transform=ccrs.PlateCarree())

#fig1 = plt.figure(figsize=(10, 10),dpi=200)
#ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#ax1.pcolormesh(wind_ds.lons, wind_ds.lats, wind_ds_land_mask.wspd[0,:,:], transform=ccrs.PlateCarree())    

#fig1 = plt.figure(figsize=(10, 10),dpi=200)
#ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#ax1.pcolormesh(wind_ds.lons, wind_ds.lats, wind_ds_ocean_mask.wspd[0,:,:], transform=ccrs.PlateCarree())    

weights = np.cos(np.deg2rad(wind_ds.lat))
weights.name = "weights"

wspd_all_weighted = wind_ds.wspd.weighted(weights)
wdir_all_weighted = wind_ds.wdir.weighted(weights)
wspd_all_weighted_mean = wspd_all_weighted.mean(("y", "x"))
wdir_all_weighted_mean = wdir_all_weighted.mean(("y", "x"))

wspd_land_weighted = wind_ds_land_mask.wspd.weighted(weights)
wdir_land_weighted = wind_ds_land_mask.wdir.weighted(weights)
wspd_land_weighted_mean = wspd_land_weighted.mean(("y", "x"))
wdir_land_weighted_mean = wdir_land_weighted.mean(("y", "x"))

wspd_ocean_weighted = wind_ds_ocean_mask.wspd.weighted(weights)
wdir_ocean_weighted = wind_ds_ocean_mask.wdir.weighted(weights)
wspd_ocean_weighted_mean = wspd_ocean_weighted.mean(("y", "x"))
wdir_ocean_weighted_mean = wdir_ocean_weighted.mean(("y", "x"))

#%%

with Dataset(output_file, "w", format = "NETCDF4_CLASSIC") as nc_out:
    # Define dimensions
    nc_out.createDimension("time",len(time))
   # Create variables and attributes
    nc_wspd_all = nc_out.createVariable("wspd_all", "single", ("time",))
    nc_wdir_all = nc_out.createVariable("wdir_all", "single", ("time",))
    nc_wspd_land = nc_out.createVariable("wspd_land", "single", ("time",))
    nc_wdir_land = nc_out.createVariable("wdir_land", "single", ("time",))
    nc_wspd_ocean = nc_out.createVariable("wspd_ocean", "single", ("time",))
    nc_wdir_ocean = nc_out.createVariable("wdir_ocean", "single", ("time",))
        
    
    nc_time = nc_out.createVariable("time","single",("time",))
    nc_time.long_name = "time"
    nc_time.standard = "time"
    nc_time.calendar = "standard"
    nc_time.units = "hours since 1986-01-01 00:00:00"
            
    nc_time[:] = time
    nc_wspd_all[:] = wspd_all_weighted_mean[:]
    nc_wdir_all[:] = wdir_all_weighted_mean[:]
    nc_wspd_land[:] = wspd_land_weighted_mean[:]
    nc_wdir_land[:] = wdir_land_weighted_mean[:]
    nc_wspd_ocean[:] = wspd_ocean_weighted_mean[:]
    nc_wdir_ocean[:] = wdir_ocean_weighted_mean[:]
    




# =============================================================================
# 
# 
# ax = WindroseAxes.from_ax()
# ax.bar(wdir_land_weighted_mean, wspd_land_weighted_mean , normed=True, opening=0.8, edgecolor='white')
# ax.set_legend(fontsize=16)
# ax.set_title('D03 Wind over Land',fontsize=20)
# plt.savefig('/Users/evagnegy/Desktop/test.png',bbox_inches='tight')
# 
# 
# ax = WindroseAxes.from_ax()
# ax.bar(wdir_ocean_weighted_mean, wspd_land_weighted_mean , normed=True, opening=0.8, edgecolor='white')
# ax.set_legend(fontsize=16)
# ax.set_title('D03 Wind over Ocean',fontsize=20)
# plt.savefig('/Users/evagnegy/Desktop/test2.png',bbox_inches='tight')
# 
# 
# 
# =============================================================================
