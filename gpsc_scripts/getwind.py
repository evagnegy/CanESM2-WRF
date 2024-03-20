from __future__ import print_function


from netCDF4 import Dataset
from wrf import getvar, ALL_TIMES
import numpy as np
import os
import pandas as pd
import sys

#filedir = "/gpfs/fs2/dfo/hpcmc-pfm/dfo_odis/evg000/ERA5_WRF_runs/2016_FLOODING_run/WRF/"

variable=sys.argv[1]
filedir=sys.argv[2]
outdir=sys.argv[3]
year=sys.argv[4]
startmonth=int(sys.argv[5])
endmonth=int(sys.argv[6])
domain=sys.argv[7]

days = []
hours = []
months = []
for day in range(1,32):
    if day < 10:
        days.append("0" + str(day))
    else:
        days.append(str(day))
for hour in range(24):
    if hour < 10:
        hours.append("0" + str(hour))
    else:
        hours.append(str(hour))
for month in range(startmonth,endmonth+1):
    if month < 10:
        months.append("0" + str(month))
    else:
        months.append(str(month))


#for d0 in ["d01","d02","d03"]:
print("now on domain: " + domain)
allfiles = []
for month in months:
    for day in days:
        for hour in hours:
            try:
                current_file = Dataset(filedir+"/wrfout_" + domain + "_" + year + "-" + month + "-" + day + "_" + hour + ":00:00")
                allfiles.append(current_file)
                #current_file.close() 
            except:
                print("  skipping: wrfout_" + domain + "_" + year + "-" + month + "-" + day + "_" + hour + ":00:00")


if variable == "t":
    print("   getting temperature")
    var = getvar(allfiles, "T2", timeidx=ALL_TIMES,method="cat")
elif variable == "pr":
    print("   getting precip")
    var = getvar(allfiles, "RAINNC", timeidx=ALL_TIMES, method="cat")
    var2 = getvar(allfiles, "RAINC", timeidx=ALL_TIMES, method="cat")
elif variable == "wind":
    print("   getting wind")
    #var=getvar(allfiles, "uvmet10_wspd_wdir", timeidx=ALL_TIMES, method="cat")
    var=getvar(allfiles, "uvmet10", timeidx=ALL_TIMES, method="cat") 

times = getvar(allfiles, "times", timeidx=ALL_TIMES,method="cat")

print("   getting lats")
lats = getvar(allfiles, "lat", method="cat")
print("   getting lons") 
lons = getvar(allfiles, "lon", method="cat")
    

hours_since = [int((times[i]-times[0])/3600e9) for i in range(len(times))]
    
first_date = str(times[0])[37:47]
first_hour = str(times[0])[48:56]
time_units = "hours since " + first_date + " " + first_hour
print(str(times[0])[48:56])
print(int(times[0]))

print(np.shape(var))

time_size = np.shape(var)[1] 
lat_size =  np.shape(var)[2]
lon_size =  np.shape(var)[3]

file = outdir+'/' + str(year) + '_' + str(startmonth) + "-" + str(endmonth) + "_" + variable + "_" + domain + '.nc'
ncfile = Dataset(file,'w',format='NETCDF4_CLASSIC')

#create dimensions
ncfile.createDimension('lat',lat_size)
ncfile.createDimension('lon',lon_size)
ncfile.createDimension('time_d',time_size) 

print(np.shape(lat_size))
print(np.shape(lon_size))
print(np.shape(time_size))

#print("      saving variables to new file")
#define variables
latitude = ncfile.createVariable('lat','d',('lon','lat'))
longitude = ncfile.createVariable('lon','d',('lon','lat'))
    
if variable == "t":
    var_save = ncfile.createVariable('T2','d',('time_d','lon','lat'))
elif variable == "pr":
    var_save = ncfile.createVariable('pr','d',('time_d','lon','lat'))
elif variable == "wind":
    var_save1 = ncfile.createVariable('wspd','d',('time_d','lon','lat'))
    var_save2 = ncfile.createVariable('wdir','d',('time_d','lon','lat'))

print(variable)
#print(var_save)
#var_save = ncfile.createVariable('wind','d',('time_d','lon','lat'))
time_save = ncfile.createVariable('time','d',('time_d'))
    
time_save.long_name = "time"
time_save.units = time_units
latitude.units = "degrees_north"
longitude.units = "degrees_east"
    
var_save1.coordinates = "lon lat"
var_save2.coordinates = "lon lat"

longitude[:] = lons[:,:]
latitude[:] = lats[:,:]
if variable == "t":
    var_save[:] = var 
elif variable == "pr":
    var_save[:] = var + var2
elif variable == "wind":
#    var_save1[:] = var
    var_save1[:] = var[0,:,:,:]
    var_save2[:] = var[1,:,:,:]
time_save[:] = hours_since

#close ncfile
ncfile.close()


