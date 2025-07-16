#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:28:13 2024

@author: evagnegy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os 
import datetime
import scipy.io

#%% NOAA
buoy_IDs = ['46041','desw1','46029','wpow1','46005']


for ID in buoy_IDs:
    print(ID)
    
    
    df_all = pd.DataFrame()
    
    for year in range(1986,2006):
        
        file = '/Users/evagnegy/Desktop/noaa_buoys/' + ID + 'h' + str(year) + '.txt'
    
        if not os.path.exists(file):
            continue
        
        df = pd.read_csv(file, delimiter='\s+') 
        
        df.replace(999, np.nan, inplace=True)
        df.replace(99, np.nan, inplace=True)
        
        #years before 1999 use "YY"
        if 'YYYY' in df.columns:
            df['DateTime'] = df['YYYY'].astype(str) + '-' + df['MM'].astype(str).str.zfill(2) + '-' + df['DD'].astype(str).str.zfill(2) + ' ' + df['hh'].astype(str).str.zfill(2)
        else:
            df['DateTime'] = '19' + df['YY'].astype(str) + '-' + df['MM'].astype(str).str.zfill(2) + '-' + df['DD'].astype(str).str.zfill(2) + ' ' + df['hh'].astype(str).str.zfill(2)
        
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H')
                    
        df_all = df_all.append(df, ignore_index=True)
    
    df_all.set_index('DateTime', inplace=True)
    
    plt.figure()
    plt.scatter(df_all.index,df_all['ATMP'])
    
    
#%%

buoy_IDs = ['46041','desw1','46029','wpow1','46005']

start_year = 1986
end_year = 2005

var = "t"

years = np.arange(start_year, end_year+1).tolist()
days = pd.date_range(datetime.date(start_year,1,1),datetime.date(end_year+1,1,1)-datetime.timedelta(days=1),freq='d')

hours = pd.date_range(datetime.date(start_year,1,1),datetime.date(end_year+1,1,2)-datetime.timedelta(days=1),freq='H')[:-1]

if var=="t":
    var_col = "ATMP"
elif var=="wind":
    var_col="WSPD"
    
df_all = pd.DataFrame()

for ID in buoy_IDs:
    print(ID)
    
    
    df_st = pd.DataFrame()
    
    for year in range(1986,2006):
        
        file = '/Users/evagnegy/Desktop/noaa_buoys/' + ID + 'h' + str(year) + '.txt'
    
        if not os.path.exists(file):
            continue
        
        df = pd.read_csv(file, delimiter='\s+') 
        
        df.replace(999, np.nan, inplace=True)
        df.replace(99, np.nan, inplace=True)
        
        #years before 1999 use "YY"
        if 'YYYY' in df.columns:
            df['DateTime'] = df['YYYY'].astype(str) + '-' + df['MM'].astype(str).str.zfill(2) + '-' + df['DD'].astype(str).str.zfill(2) + ' ' + df['hh'].astype(str).str.zfill(2)
        else:
            df['DateTime'] = '19' + df['YY'].astype(str) + '-' + df['MM'].astype(str).str.zfill(2) + '-' + df['DD'].astype(str).str.zfill(2) + ' ' + df['hh'].astype(str).str.zfill(2)
        
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H')
                    
        df_st = df_st.append(df, ignore_index=True)
    
    df_st.set_index('DateTime', inplace=True)
    
    df_var = df_st[var_col]
    df_var.name = ID
    df_var = df_var[~df_var.index.duplicated(keep='first')]
    df_var_filled = df_var.reindex(hours)

    
    df_all = df_all.append(df_var_filled)
    

df_hourly = df_all.T

yearly_missing = df_hourly.resample('Y').apply(lambda x: x.isna().sum())
monthly_missing = df_hourly.resample('M').apply(lambda x: x.isna().sum())
daily_missing = df_hourly.resample('D').apply(lambda x: x.isna().sum())

yearly_expected = df_hourly.resample('Y').apply(lambda x: len(x))
monthly_expected = df_hourly.resample('M').apply(lambda x: len(x))
daily_expected = df_hourly.resample('D').apply(lambda x: len(x))

years_to_nan = yearly_missing > (0.1 * yearly_expected)
months_to_nan = monthly_missing > (0.1 * monthly_expected)
days_to_nan = daily_missing > (0.1 * daily_expected)

    
df_daily = df_hourly.groupby(pd.PeriodIndex(df_hourly.index, freq="D")).mean()
df_monthly = df_hourly.groupby(pd.PeriodIndex(df_hourly.index, freq="M")).mean()
df_yearly = df_hourly.groupby(pd.PeriodIndex(df_hourly.index, freq="Y")).mean()

years_to_nan.index = df_yearly.index
months_to_nan.index = df_monthly.index
days_to_nan.index = df_daily.index
 
df_yearly = df_yearly.mask(years_to_nan)
df_monthly = df_monthly.mask(months_to_nan)
df_daily = df_daily.mask(days_to_nan)


#%% DFO (from internet)

buoy_IDs = ['46131','46132','46146','46204','46207']


for ID in buoy_IDs:
    print(ID)
    
        
    file = '/Users/evagnegy/Desktop/dfo_buoys/c' + ID + '.csv'

    df = pd.read_csv(file) 
    
    df.set_index('DATE', inplace=True)
    df.index = pd.to_datetime(df.index)
    
    df = df[(df.index >= '1986-01-01 00:00') & (df.index < '2006-01-01 00:00')]

    plt.figure()
    plt.scatter(df.index,df['DRYT'])
    
    
#%% DFO  (from Amber)

buoy_IDs = ['46131']#,'46132','46146','46204','46207']


for ID in buoy_IDs:
    print(ID)
    
    mat = scipy.io.loadmat('/Users/evagnegy/Downloads/DFO_data/MB_' + ID + '_HM.mat')

    temp = mat['MB_' + ID + '_HM_VALUES'][:,4]
    times = mat['MB_' + ID + '_HM_VALUES'][:,0]
    
    times_py = (pd.to_datetime(times-719529,unit='D')).to_pydatetime()
    
    df = pd.DataFrame(temp,index=times_py)
    df.index = pd.to_datetime(df.index)
    
    df = df[(df.index >= '1986-01-01 00:00') & (df.index < '2006-01-01 00:00')]


    plt.figure()
    plt.scatter(df.index,df[0])


#%%


#def get_eccc_buoys(output_freq,station_IDs,stations_dir,var): 
buoy_IDs = ['46132','46132','46146','46204','46207']

start_year = 1986
end_year = 2005

df_all = pd.DataFrame()

for ID in buoy_IDs:
    
    mat = scipy.io.loadmat(stations_dir + 'daily/ECCC_buoy/MB_' + str(ID) + '_HM.mat')

    temp = mat['MB_' + str(ID) + '_HM_VALUES'][:,4]
    times = mat['MB_' + str(ID) + '_HM_VALUES'][:,0]
    
    times_py = (pd.to_datetime(times-719529,unit='D')).to_pydatetime()
    
    df = pd.DataFrame(temp,index=times_py)
    df.index = pd.to_datetime(df.index)
    
    df = df.rename_axis("DateTime")
    df.index = df.index.round(freq='H')

    df = df[(df.index >= '1986-01-01 00:00') & (df.index < '2006-01-01 00:00')]
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H')

    df_all = pd.concat([df_all,df],axis=1)
    



