
from netCDF4 import Dataset
import datetime
import numpy as np

dir = '/home/spfm000/space/CanESM2-WRF/final_netcdfs/'

file = 'test.nc'

print(Dataset(dir+file,'r').variables)
print(Dataset(dir+file,'r'))

"""
vars = ['tasmax','tasmin','tas','pr','wspd']
times = ['historical','rcp45','rcp85']

vars=['pr']
times=['rcp85']

for var in vars:
    for time in times:
        print(var + "_" + time)
        nc = Dataset(dir+var+"_daily_d03_"+time+".nc","a")
        #nc=Dataset(dir+var+"_"+time+".nc","a")

        nc.setncattr('institution','Fisheries and Oceans Canada')
        nc.setncattr('contact','egnegy@eoas.ubc.ca')
        nc.setncattr('history','Created in March 2024')
        nc.setncattr('frequency','daily')
        nc.setncattr('map_proj','lambert')
        nc.setncattr('ref_lat','49.00')
        nc.setncattr('ref_lon','-130.90')
        nc.setncattr('truelat1','49.0')
        nc.setncattr('truelat2','49.0')
        nc.setncattr('stand_lon','-77.0')


        if time == 'historical':
            nc.setncattr('description','CanESM2-WRF 3-km historical output (1986-2005)')
        elif time == 'rcp45':
            nc.setncattr('description','CanESM2-WRF 3-km RCP 4.5 output (2046-2065)')
        elif time == 'rcp85':
            nc.setncattr('description','CanESM2-WRF 3-km RCP 8.5 output (2046-2065)')

        if var.startswith('tasmax'):
            nc_var = nc.variables['tasmax']    
            nc_var.standard_name = 'air_temperature'
            nc_var.units = 'K'
            nc_var.long_name = 'Daily Maximum Near-Surface Air Temperature'

        elif var.startswith('tasmin'):
            nc_var = nc.variables['tasmin']
            nc_var.standard_name = 'air_temperature'
            nc_var.units = 'K'
            nc_var.long_name = 'Daily Minimum Near-Surface Air Temperature'

        elif var.startswith('tas_'):
            nc_var = nc.variables['tas']
            nc_var.standard_name = 'air_temperature'
            nc_var.units = 'K'
            nc_var.long_name = 'Daily Mean Near-Surface Air Temperature'

        elif var.startswith('pr'):
            nc_var = nc.variables['pr']
            nc_var.standard_name = 'precipitation_flux'
            nc_var.units = 'kg m-2 d-1'
            nc_var.long_name = 'Precipitation'

        elif var.startswith('wspd'):
            nc_var = nc.variables['wspd']
            nc_var.standard_name = 'wind_speed'
            nc_var.units = 'm s-1'
            nc_var.long_name = 'Daily Mean Near-Surface Wind Speed'

        nc.close()

"""
