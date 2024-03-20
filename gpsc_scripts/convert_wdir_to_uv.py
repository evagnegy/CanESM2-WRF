from __future__ import print_function


from netCDF4 import Dataset
from wrf import getvar, ALL_TIMES
import numpy as np
import os
import pandas as pd
import sys
import xclim
import xarray as xr

path = '/home/spfm000/space/CanESM2-WRF/historical/variables_complete/wind_d02.nc'

wind_ds = xr.open_dataset(path)
wind_spd = wind_ds['wspd']
wind_dir = wind_ds['wdir']

wind_spd_chunked = wind_spd.chunk({'time': 100})
wind_dir_chunked = wind_dir.chunk({'time': 100})

u_chunks = -wind_spd_chunked * np.sin(np.radians(wind_dir_chunked))
v_chunks = -wind_spd_chunked * np.cos(np.radians(wind_dir_chunked))

u = xr.concat(u_chunks.chunk({'time': -1}),dim='time')
v = xr.concat(v_chunks.chunk({'time': -1}),dim='time')

u_ds = xr.Dataset({'u': u})
v_ds = xr.Dataset({'v': v})

print(u_ds)

u_avg = u_ds.mean(dim='time')
v_avg = v_ds.mean(dim='time')

print(u_avg)

#avg_spd = np.sqrt(u_avg**2 + v_avg**2)
#avg_dir = np.degrees(np.arctan2(-u_avg,-v_avg)) % 360

#print(avg_dir)

#wind_ds_mean = xr.Dataset({'wspd_avg': avg_spd.to_array(), 'wdir_avg': avg_dir.to_array()}) 

u_avg.to_netcdf('wind_d02_u.nc')
#v_avg.to_netcdf('wind_d02_v.nc')

