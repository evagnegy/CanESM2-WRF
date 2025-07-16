#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 23:10:16 2022

@author: evagnegy
"""
import os
import pandas as pd

#station_directory = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/daily/ECCC_buoy/"

station_directory = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_buoys.csv'
output_directory = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/CDO_station_files/"

#dirs = os.listdir(station_directory)

dirs = ['46131','46132','46146','46204','46207','46134','46207','46206']

df = pd.read_csv(station_directory)

for i in range(len(dirs)):
    
    stationID = dirs[i]
    print(stationID)

    
    filename = "ECCC_buoy_" + stationID + ".txt"

    
    #file = os.listdir(station_directory + stationID + '/1' )[1]
    #file = station_directory + filename


    #df = pd.read_csv(station_directory + stationID + "/1/" + file) 

 
    #longitude = df.loc[0,"Longitude (x)"]
    #latitude = df.loc[0,"Latitude (y)"]
    longitude = df.loc[i,"X"]
    latitude = df.loc[i,"Y"]
    
    
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
