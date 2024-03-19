#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 23:10:16 2022

@author: evagnegy
"""
import os
import pandas as pd

station_directory = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/hourly/ECCC/"
output_directory = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/CDO_station_files/"

dirs = os.listdir(station_directory)

for stationID in dirs:
    if stationID[0] != ".":
   
        filename = "ECCC_" + stationID + ".txt"
        
        file = os.listdir(station_directory + stationID + '/1' )[1]
        
        print(file)

        df = pd.read_csv(station_directory + stationID + "/1/" + file) 
    
 
        longitude = df.loc[0,"Longitude (x)"]
        latitude = df.loc[0,"Latitude (y)"]
        f=open(output_directory + filename, "w+")
        f.write("# CDO grid file\n")
        f.write("# Station " + stationID + "\n\n")
        f.write("gridtype = curvilinear\n")
        f.write("gridsize = 1\n")
        f.write("xsize = 1\n")
        f.write("ysize = 1\n\n")
        f.write("# Longitudes\n")
        f.write("xvals = " + str(longitude) + "\n\n")
        f.write("# Latitude\n")
        f.write("yvals = " + str(latitude) + "\n")
        f.close()
