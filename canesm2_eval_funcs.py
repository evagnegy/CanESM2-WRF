import numpy as np
import pandas as pd
import datetime
import glob
import math
from netCDF4 import Dataset
import netCDF4
import WRFDomainLib
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl
from scipy.interpolate import griddata

def get_eccc_obs(output_freq,station_IDs,stations_dir,var): 
    
    start_year = 1986
    end_year = 2005
    
    years = np.arange(start_year, end_year+1).tolist()
    days = pd.date_range(datetime.date(start_year,1,1),datetime.date(end_year+1,1,1)-datetime.timedelta(days=1),freq='d')
    
    if var=="t":
        var_col = "Mean Temp (°C)"
    elif var=="pr":
        var_col = "Total Rain (mm)"
    elif var=="tmax":
        var_col = "Max Temp (°C)"
    elif var=="tmin":
        var_col = "Min Temp (°C)"
    elif var=="wind":
        var_col="Wind Spd (km/h)"
    
    t_all,t_avg = [],[]
    
    for station_ID in station_IDs:
        
        t_all_station,t_avg_station = [],[]
        for year in years:
            if var=="wind":
                t_df_temp = []
                for month in ['1','2','3','4','5','6','7','8','9','10','11','12']:
           
                    file = glob.glob(stations_dir + 'hourly/ECCC/' + str(station_ID) + "/" + month + '/*' + str(year) + '_*.csv')[0]
      
                    df = pd.read_csv(file) 
                    df_daily = df.groupby(pd.PeriodIndex(df['Date/Time (UTC)'], freq="D")).mean()     
                    t_df_temp.append(list(df_daily[var_col]))
                    
                t_df = [item for sublist in t_df_temp for item in sublist]
                
                
            else:  
                file = glob.glob(stations_dir + 'daily/ECCC/' + str(station_ID) + "/*" + str(year) + '_*.csv')[0]
               
                df = pd.read_csv(file)
                t_df = list(df[var_col])
            
            count_nans = sum(math.isnan(i) for i in t_df)
            if count_nans > len(t_df)*0.1: #remove station if more than 10% of year is missing
                year_avg = np.nan
            else:
                if var=="t" or var=="wind" or var=="tmax" or var=="tmin":
                    year_avg = np.nanmean(t_df)
                elif var=="pr":
                    year_avg = np.nansum(t_df)
                #elif var=='tmax':
                #    year_avg = np.nanmax(t_df)

            
            t_all_station.append(t_df)
            t_avg_station.append(year_avg)
        
        flat_list = [item for sublist in t_all_station for item in sublist]
        
        t_all.append(flat_list)
        t_avg.append(t_avg_station)
                  
            
    df_daily_temp = pd.DataFrame(t_all)
    df_yearly_temp = pd.DataFrame(t_avg)
    
    if var == "wind": #convert to m/s
        df_daily_temp = df_daily_temp/3.6
        df_yearly_temp = df_yearly_temp/3.6
        
        
    df_daily = df_daily_temp.T
    df_yearly = df_yearly_temp.T
    
    df_daily.columns = station_IDs
    df_yearly.columns = station_IDs
    
    df_daily.index = pd.to_datetime(days,format='%Y-%m-%d')
    df_yearly.index = pd.to_datetime(years,format='%Y')
    df_yearly.index = df_yearly.index.to_period('A-DEC')
    
    df_daily_toavg = df_daily.reset_index()
    df_daily_toavg['Dates'] = days
    df_daily_toavg.drop(['index'], axis=1)
    
    if var == "t" or var == "tmax" or var == "tmin" or var == "wind":
        df_monthly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="M")).mean()
    elif var == "pr":
        df_monthly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="M")).sum()

            
    if output_freq == "daily":
        return df_daily
    elif output_freq == "monthly":
        return df_monthly
    elif output_freq == "yearly":
        return df_yearly
    
def get_bch_obs(output_freq,station_IDs,stations_dir,var): 

    if var == "t":

        df_min = pd.read_csv(stations_dir + '/daily/BCH/BCH_tnId.csv',index_col=0)
        df_max = pd.read_csv(stations_dir + '/daily/BCH/BCH_txId.csv',index_col=0)
        
        df_min.index = pd.to_datetime(df_min.index).date
        df_max.index = pd.to_datetime(df_max.index).date

        df = (df_min + df_max)/2 


    elif var == "pr":
        df = pd.read_csv(stations_dir + '/daily/BCH/BCH_p24Id.csv',index_col=0)
        df.index = pd.to_datetime(df.index).date
        
    elif var == "tmax":
        df = pd.read_csv(stations_dir + '/daily/BCH/BCH_txId.csv',index_col=0)
        df.index = pd.to_datetime(df.index).date
    elif var == "tmin":
        df = pd.read_csv(stations_dir + '/daily/BCH/BCH_tnId.csv',index_col=0)
        df.index = pd.to_datetime(df.index).date

    # remove stations not in the original list
    for station in list(df.columns):
        if station not in station_IDs:
            df.drop(station, inplace=True, axis=1)

    start_date = datetime.date(1986, 1, 1)
    end_date = datetime.date(2005, 12, 31)

    df_daily = df.loc[start_date:end_date]
    
    days = pd.date_range(start_date,end_date,freq='d')

    df_daily_toavg = df_daily.reset_index()
    df_daily_toavg['Dates'] = days
    df_daily_toavg.drop(['index'], axis=1)

    df_daily.index = days
    
    if var=="t":
        df_monthly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="M")).mean()
        df_yearly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="Y")).mean()
    elif var == "pr":
        df_monthly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="M")).sum()
        df_yearly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="Y")).sum()
    elif var == "tmax":
        df_monthly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="M")).mean()
        df_yearly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="Y")).mean()
    elif var == "tmin":
        df_monthly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="M")).mean()
        df_yearly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="Y")).mean()


    if output_freq == "daily":
        return df_daily
    elif output_freq == "monthly":
        return df_monthly
    elif output_freq == "yearly":
        return df_yearly

def get_noaa_obs(output_freq,station_IDs,stations_dir,var): 
    
    df_1 = pd.read_csv(stations_dir + '/daily/NOAA/1986-1990.csv',index_col=0,low_memory=False)
    df_2 = pd.read_csv(stations_dir + '/daily/NOAA/1991-1995.csv',index_col=0,dtype='str')
    df_3 = pd.read_csv(stations_dir + '/daily/NOAA/1996-2000.csv',index_col=0,dtype='str')
    df_4 = pd.read_csv(stations_dir + '/daily/NOAA/2001-2005.csv',index_col=0,dtype='str')

    df_all = pd.concat([df_1,df_2,df_3,df_4])

    if var == "t":
        df_min = df_all[['TMIN','DATE']]
        df_max = df_all[['TMAX','DATE']]
        
        pivoted_df_min = df_min.pivot_table(index=df_min.index, columns='DATE', values='TMIN')
        pivoted_df_max = df_max.pivot_table(index=df_max.index, columns='DATE', values='TMAX')
        
        
        df = (pivoted_df_min + pivoted_df_max)/2
        
    elif var == "pr":
        df_pr = df_all[['PRCP','DATE']]
        df = df_pr.pivot_table(index=df_pr.index, columns='DATE', values='PRCP')
     
    elif var == "tmax":
         df_max = df_all[['TMAX','DATE']]
         
         df = df_max.pivot_table(index=df_max.index, columns='DATE', values='TMAX')
    elif var == "tmin":
         df_max = df_all[['TMIN','DATE']]
         
         df = df_max.pivot_table(index=df_max.index, columns='DATE', values='TMIN')
    elif var == "wind":
        df_wnd = df_all[['AWND','DATE']]
        df = df_wnd.pivot_table(index=df_wnd.index, columns='DATE', values='AWND')
         
         
         
         
    df_daily = df.transpose()
    df_daily.index = pd.to_datetime(df_daily.index).date
    
    # remove stations not in the original list
    for station in list(df_daily.columns):
        if station not in station_IDs:
            df_daily.drop(station, inplace=True, axis=1)
    
    start_date = datetime.date(1986, 1, 1)
    end_date = datetime.date(2005, 12, 31)
    days = pd.date_range(start_date,end_date,freq='d')
    
    df_daily_toavg = df_daily.reset_index()
    df_daily_toavg['Dates'] = days
    df_daily_toavg.drop(['index'], axis=1)
    
    df_daily.index = days
    
    
    
    if var=="t" or var == "tmax" or var == "tmin" or var == "wind":
        df_monthly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="M")).mean()
        df_yearly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="Y")).mean()
    elif var == "pr":
        df_monthly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="M")).sum()
        df_yearly = df_daily_toavg.groupby(pd.PeriodIndex(df_daily_toavg['Dates'], freq="Y")).sum()

        
    
    if output_freq == "daily":
        return df_daily
    elif output_freq == "monthly":
        return df_monthly
    elif output_freq == "yearly":
        return df_yearly


def get_wrf(output_freq, agency, station_IDs, domain, run, var, model_files_dir,start_year):
    
    df_hourly_all = pd.DataFrame()
    df_daily_all = pd.DataFrame()
    df_monthly_all = pd.DataFrame()
    df_yearly_all = pd.DataFrame()
    
            
    if var == "t" or var == "tmax" or var=="tmin":
        wrf_var_name = 'T2'
        file_var = "t"
    elif var == "pr":
        wrf_var_name = "pr"
        file_var = "pr"
    elif var == "wind":
        wrf_var_name = 'wspd'
        file_var = "wind"
        
    for station_ID in station_IDs:
        
        wrf_st_file = model_files_dir + file_var + "_" + domain + "_" + agency + '_' + str(station_ID) + ".nc"
    
        nc = Dataset(wrf_st_file, mode='r')
        wrf_var_tmp = np.squeeze(nc.variables[wrf_var_name][:])
        wrf_time = np.squeeze(nc.variables['time'][:])
        
        
        if var=="t" or var=="tmax" or var=="tmin":
            wrf_var=(wrf_var_tmp-273.15).round(1)
        elif var=="pr":
            wrf_var = []
            for i in range(1,len(wrf_var_tmp)):
                value=wrf_var_tmp[i]-wrf_var_tmp[i-1]
                if value <0:
                    wrf_var.append(0)
                else:
                    wrf_var.append(value)
        elif var == "wind":
            wrf_var = wrf_var_tmp.copy()
                    
        date_wrf = [] 
        start = datetime.datetime(start_year,1,1)
        for hour in wrf_time:
            delta = datetime.timedelta(hours=hour)     # Create a time delta object from the number of days
            date_wrf.append(start + delta)     # Add the specified number of days to 1990
        
        
        df_wrf = pd.DataFrame()
        df_wrf[station_ID] = wrf_var
        
        if var=="t" or var=="tmax" or var == "wind" or var=="tmin":
            df_wrf['Dates'] = date_wrf
        elif var=="pr":
            df_wrf['Dates'] = date_wrf[1:]
            
        df_wrf['Dates'] = pd.to_datetime(df_wrf.Dates)
        

        #remove the 0 hour from jan 1st every year bc its not continuous with the rest of the run
        if var=="pr":
            df_wrf.loc[(df_wrf['Dates'].dt.day==1) & (df_wrf['Dates'].dt.month==1) & (df_wrf['Dates'].dt.hour==0)] = np.nan
    
    
        if var=="t" or var=="wind":
            wrf_yearly = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="Y")).mean()
            wrf_monthly = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="M")).mean()
            wrf_daily = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="D")).mean()
            #wrf_hourly = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="H")).mean()
            
        elif var=="pr":
            wrf_yearly = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="Y")).sum()
            wrf_monthly = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="M")).sum()
            wrf_daily = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="D")).sum()
            #wrf_hourly = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="H")).sum()

        elif var=="tmax":
            wrf_daily = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="D")).max()
            
            wrf_yearly = wrf_daily.groupby(pd.PeriodIndex(wrf_daily['Dates'], freq="Y")).mean()
            wrf_monthly = wrf_daily.groupby(pd.PeriodIndex(wrf_daily['Dates'], freq="M")).mean()

           # wrf_yearly = wrf_yearly.drop(['Dates'], axis=1)
           # wrf_monthly = wrf_monthly.drop(['Dates'], axis=1)
           # wrf_daily = wrf_daily.drop(['Dates'], axis=1)

        elif var=="tmin":
            wrf_daily = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="D")).min()

            wrf_monthly = wrf_daily.groupby(pd.PeriodIndex(wrf_daily['Dates'], freq="M")).mean()
            wrf_yearly = df_wrf.groupby(pd.PeriodIndex(df_wrf['Dates'], freq="Y")).mean()
            
            #wrf_yearly = wrf_yearly.drop(['Dates'], axis=1)
            #wrf_monthly = wrf_monthly.drop(['Dates'], axis=1)
            #wrf_daily = wrf_daily.drop(['Dates'], axis=1)

        df_daily_all = pd.concat([df_daily_all,wrf_daily],axis=1)
        df_monthly_all = pd.concat([df_monthly_all,wrf_monthly],axis=1)
        df_yearly_all = pd.concat([df_yearly_all,wrf_yearly],axis=1)
        #df_hourly_all = pd.concat([df_hourly_all,wrf_hourly],axis=1)

        

    if output_freq == "daily":
        return df_daily_all
    elif output_freq == "monthly":
        return df_monthly_all
    elif output_freq == "yearly":
        return df_yearly_all
    elif output_freq == "hourly":
        return df_hourly_all

def get_canesm2(output_freq, agency, station_IDs, run, var, model_files_dir,start_year):
    raw_yearly_all = pd.DataFrame()
    raw_monthly_all = pd.DataFrame()
    raw_daily_all = pd.DataFrame()

    if var == "t":
        raw_var_name = 'tas'
    elif var == "pr":
        raw_var_name = "pr"
    elif var == "tmax":
        raw_var_name = "tasmax"
    elif var == "tmin":
        raw_var_name = "tasmin"
    elif var == "wind":
        raw_var_name = "ws"
                
    for station_ID in station_IDs:
        raw_st_file = model_files_dir + var + "_" + agency + '_' + str(station_ID) + ".nc"
        
        nc = Dataset(raw_st_file, mode='r')
        raw_data_temp = np.squeeze(nc.variables[raw_var_name][:])

        if var=="t" or var=="tmax" or var=='tmin':
            raw_data=(raw_data_temp-273.15).round(1)
        elif var=="pr":
            raw_data_temp[raw_data_temp < 0] = 0
            raw_data=raw_data_temp*60*60*24
        elif var=="wind":
            raw_data=raw_data_temp.copy()
            
        date_raw_cft = netCDF4.num2date(nc.variables['time'][:], nc.variables['time'].units, nc.variables['time'].calendar)
          
        date_raw = []
        for i in range(len(date_raw_cft)):
            date_raw.append(datetime.datetime.strptime(str(date_raw_cft[i]),'%Y-%m-%d %H:%M:%S'))
                      
        df_raw = pd.DataFrame()
        df_raw[station_ID] = raw_data
        df_raw['Dates'] = date_raw
        df_raw['Dates'] = pd.to_datetime(df_raw.Dates)

        if var=="t" or var=="wind":
            raw_yearly_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="Y")).mean()
            raw_monthly_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="M")).mean()
            raw_daily_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="D")).mean()
        
        elif var=="pr":
            raw_yearly_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="Y")).sum()
            raw_monthly_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="M")).sum()
            raw_daily_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="D")).sum()
        
        elif var=="tmax":
            raw_daily_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="D")).mean()
            raw_yearly_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="Y")).mean()
            raw_monthly_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="M")).mean()
            
            #raw_yearly_temp = raw_yearly_temp.drop(['Dates'], axis=1)
            #raw_monthly_temp = raw_monthly_temp.drop(['Dates'], axis=1)
            #raw_daily_temp = raw_daily_temp.drop(['Dates'], axis=1)


        elif var=="tmin":
            raw_daily_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="D")).mean()
            raw_monthly_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="M")).mean()
            raw_yearly_temp = df_raw.groupby(pd.PeriodIndex(df_raw['Dates'], freq="Y")).mean()
    

        raw_yearly_all = pd.concat([raw_yearly_all,raw_yearly_temp],axis=1)
        raw_monthly_all = pd.concat([raw_monthly_all,raw_monthly_temp],axis=1)
        raw_daily_all = pd.concat([raw_daily_all,raw_daily_temp],axis=1)

    raw_yearly = raw_yearly_all.loc[str(start_year):str(start_year+20)]
    raw_monthly = raw_monthly_all.loc[str(start_year)+'-01':str(start_year+20)+'-12']
    raw_daily = raw_daily_all.loc[str(start_year)+'-01-01':str(start_year+20)+'-12-31']

    if output_freq == 'yearly':
        return raw_yearly
    elif output_freq == 'monthly':
        return raw_monthly 
    elif output_freq == 'daily':
        return raw_daily

def get_canrcm4(output_freq, agency, station_IDs, run, var, model_files_dir):

    rcm_yearly = pd.DataFrame()
    rcm_monthly = pd.DataFrame()
    rcm_daily = pd.DataFrame()

    if var == "t":
        rcm_var_name = 'tas'
    elif var == "pr":
        rcm_var_name = "pr"
    elif var == "tmax":
        rcm_var_name = "tmax"
    elif var == "tmin":
        rcm_var_name = "tmin"
    elif var == "wind":
        rcm_var_name = "sfcWind"
        
    for station_ID in station_IDs:
        rcm_station_file = model_files_dir + var + "_" + agency + "_" + str(station_ID) + ".nc"
        
        nc = Dataset(rcm_station_file, mode='r')
        rcm_data_temp = np.squeeze(nc.variables[rcm_var_name][:])
        
        if var=="t" :
            rcm_data=(rcm_data_temp-273.15).round(1)
        elif var=="pr":
            rcm_data_temp[rcm_data_temp < 0] = 0
            rcm_data=rcm_data_temp*60*60*24
        elif var == "tmax" or var=='tmin':
            rcm_data=rcm_data_temp.round(1)
        elif var == "wind":
            rcm_data=rcm_data_temp.copy()    
     
            
        date_rcm_cft = netCDF4.num2date(nc.variables['time'][:], nc.variables['time'].units, nc.variables['time'].calendar)
          
        date_rcm = []
        for i in range(len(date_rcm_cft)):
            date_rcm.append(datetime.datetime.strptime(str(date_rcm_cft[i]),'%Y-%m-%d %H:%M:%S'))
                    
        df_rcm = pd.DataFrame()
        df_rcm[station_ID] = rcm_data
        df_rcm['Dates'] = date_rcm
        df_rcm['Dates'] = pd.to_datetime(df_rcm.Dates)

        if var=="t" or var=='wind':
            rcm_yearly_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="Y")).mean()
            rcm_monthly_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="M")).mean()
            rcm_daily_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="D")).mean()
        
        elif var=="pr":
            rcm_yearly_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="Y")).sum()
            rcm_monthly_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="M")).sum()
            rcm_daily_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="D")).sum()
        
        elif var=="tmax":
            rcm_yearly_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="Y")).mean()
            rcm_monthly_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="M")).mean()
            rcm_daily_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="D")).mean()

            #rcm_yearly_temp = rcm_yearly_temp.drop(['Dates'], axis=1)
            #rcm_monthly_temp = rcm_monthly_temp.drop(['Dates'], axis=1)
            #rcm_daily_temp = rcm_daily_temp.drop(['Dates'], axis=1)
        elif var=="tmin":
            rcm_yearly_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="Y")).mean()
            rcm_monthly_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="M")).mean()
            rcm_daily_temp = df_rcm.groupby(pd.PeriodIndex(df_rcm['Dates'], freq="D")).mean()
            
            
        rcm_yearly = pd.concat([rcm_yearly,rcm_yearly_temp],axis=1)
        rcm_monthly = pd.concat([rcm_monthly,rcm_monthly_temp],axis=1)
        rcm_daily = pd.concat([rcm_daily,rcm_daily_temp],axis=1)

        start_date = pd.Period('1986-1-1')
        end_date = pd.Period('2005-12-31')
        rcm_daily = rcm_daily.loc[start_date:end_date]

        start_date = pd.Period('1986')
        end_date = pd.Period('2005')
        rcm_yearly = rcm_yearly.loc[start_date:end_date]
        
        start_date = pd.Period('1986-1')
        end_date = pd.Period('2005-12')
        rcm_monthly = rcm_monthly.loc[start_date:end_date]
        
    if output_freq == 'yearly':
        return rcm_yearly
    elif output_freq == 'monthly':
        return rcm_monthly 
    elif output_freq == 'daily':
        return rcm_daily
    
def get_pcic(output_freq, agency, station_IDs, run, var, model_files_dir):

    yearly = pd.DataFrame()
    monthly = pd.DataFrame()
    daily = pd.DataFrame()

    for station_ID in station_IDs:
        
        if var == "t":
            tmin_station_file = model_files_dir + "tmin_" + agency + "_" + str(station_ID) + ".nc"
            tmax_station_file = model_files_dir + "tmax_" + agency + "_" + str(station_ID) + ".nc"
            
            nc = Dataset(tmin_station_file, mode='r')
            nc_tmax = Dataset(tmax_station_file, mode='r')
            data_temp_tmin = np.squeeze(nc.variables['tasmin'][:])
            data_temp_tmax = np.squeeze(nc_tmax.variables['tasmax'][:])
            
            data = (data_temp_tmin + data_temp_tmax)/2
        
        elif var == "pr":
            station_file = model_files_dir + var + "_" + agency + "_" + str(station_ID) + ".nc"
        
            nc = Dataset(station_file, mode='r')
            data = np.squeeze(nc.variables[var][:])
            
            data[data < 0] = 0
            
        elif var == "tmax":
            tmax_station_file = model_files_dir + "tmax_" + agency + "_" + str(station_ID) + ".nc"
            
            nc = Dataset(tmax_station_file, mode='r')
            data = np.squeeze(nc.variables['tasmax'][:])
                
            
        date_cft = netCDF4.num2date(nc.variables['time'][:], nc.variables['time'].units, nc.variables['time'].calendar)
          
        date = []
        for i in range(len(date_cft)):
            date.append(datetime.datetime.strptime(str(date_cft[i]),'%Y-%m-%d %H:%M:%S'))
                    
        df = pd.DataFrame()
        df[station_ID] = data
        df['Dates'] = date
        df['Dates'] = pd.to_datetime(df.Dates)
    
        if var=="t":
            yearly_temp = df.groupby(pd.PeriodIndex(df['Dates'], freq="Y")).mean()
            monthly_temp = df.groupby(pd.PeriodIndex(df['Dates'], freq="M")).mean()
            daily_temp = df.groupby(pd.PeriodIndex(df['Dates'], freq="D")).mean()
        
        elif var=="pr":
            yearly_temp = df.groupby(pd.PeriodIndex(df['Dates'], freq="Y")).sum()
            monthly_temp = df.groupby(pd.PeriodIndex(df['Dates'], freq="M")).sum()
            daily_temp = df.groupby(pd.PeriodIndex(df['Dates'], freq="D")).sum()
            
        elif var=="tmax":
            yearly_temp = df.groupby(pd.PeriodIndex(df['Dates'], freq="Y")).max()
            monthly_temp = df.groupby(pd.PeriodIndex(df['Dates'], freq="M")).max()
            daily_temp = df.groupby(pd.PeriodIndex(df['Dates'], freq="D")).max()
    
            yearly_temp = yearly_temp.drop(['Dates'], axis=1)
            monthly_temp = monthly_temp.drop(['Dates'], axis=1)
            daily_temp = daily_temp.drop(['Dates'], axis=1)
            
        yearly = pd.concat([yearly,yearly_temp],axis=1)
        monthly = pd.concat([monthly,monthly_temp],axis=1)
        daily = pd.concat([daily,daily_temp],axis=1)
    
    
    if output_freq == 'yearly':
        return yearly
    elif output_freq == 'monthly':
        return monthly 
    elif output_freq == 'daily':
        return daily
    
  





def plot_all_d03(title):
    
    geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
    geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
    lat_d03 = np.squeeze(geo_em_d03_nc.variables['XLAT_C'][:])
    lon_d03 = np.squeeze(geo_em_d03_nc.variables['XLONG_C'][:])
    topo_d03 = np.squeeze(geo_em_d03_nc.variables['HGT_M'][:])

    geo_em_d02_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d02.nc'
    geo_em_d02_nc = Dataset(geo_em_d02_file, mode='r')
    lat_d02 = np.squeeze(geo_em_d02_nc.variables['XLAT_C'][:])
    lon_d02 = np.squeeze(geo_em_d02_nc.variables['XLONG_C'][:])
    topo_d02 = np.squeeze(geo_em_d02_nc.variables['HGT_M'][:])
    
    canrcm4_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/orog_CanRCM4.nc'
    canrcm4_nc = Dataset(canrcm4_file, mode='r')
    lat_canrcm4 = np.squeeze(canrcm4_nc.variables['lat'][:])
    lon_canrcm4 = np.squeeze(canrcm4_nc.variables['lon'][:])
    topo_canrcm4 = np.squeeze(canrcm4_nc.variables['orog'][:])

    #topo_canrcm4 = topo_canrcm4[:-1,:-1]
    #lat_canrcm4 = (lat_canrcm4[1:]+lat_canrcm4[:-1])/2
    #lon_canrcm4 = (lon_canrcm4[1:]+lon_canrcm4[:-1])/2  
    
    canesm2_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/orog_CanESM2.nc'
    canesm2_nc = Dataset(canesm2_file, mode='r')
    lat_canesm2 = np.squeeze(canesm2_nc.variables['lat'][:])
    lon_canesm2 = np.squeeze(canesm2_nc.variables['lon'][:])
    topo_canesm2 = np.squeeze(canesm2_nc.variables['orog'][:])
    
    #topo_canesm2 = topo_canesm2[:-1,:-1]
    #lat_canesm2 = (lat_canesm2[1:]+lat_canesm2[:-1])/2
    #lon_canesm2 = (lon_canesm2[1:]+lon_canesm2[:-1])/2  
    
    pcic_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_PCIC/tasmin_day_CanESM2_historical_rcp45.nc'
    pcic_nc = Dataset(pcic_file, mode='r')
    lat_pcic = np.squeeze(pcic_nc.variables['lat'][:])
    lon_pcic = np.squeeze(pcic_nc.variables['lon'][:])
    var_pcic = np.squeeze(pcic_nc.variables['tasmin'][0,:,:])
    lon_pcic,lat_pcic=np.meshgrid(lon_pcic,lat_pcic)

    
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)


    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    if title == 'CanESM2-WRF D03':
        ax1.pcolormesh(lon_d02, lat_d02, topo_d02, cmap="terrain", vmin=0,vmax=2500, alpha=0.3, transform=ccrs.PlateCarree(),zorder=0)
        ax1.pcolormesh(lon_d03, lat_d03, topo_d03, cmap="terrain", vmin=0,vmax=2500, alpha=0.4, transform=ccrs.PlateCarree(),zorder=0)
    elif title == 'CanESM2-WRF D02':
        ax1.pcolormesh(lon_d02, lat_d02, topo_d02, cmap="terrain", vmin=0,vmax=2500, alpha=0.7, transform=ccrs.PlateCarree(),zorder=0)
    elif title == 'CanESM2':
        ax1.pcolormesh(lon_canesm2, lat_canesm2, topo_canesm2, cmap="terrain", vmin=0,vmax=2500, alpha=0.7, transform=ccrs.PlateCarree(),zorder=0)
    elif title == 'CanRCM4':
        ax1.pcolormesh(lon_canrcm4, lat_canrcm4, topo_canrcm4, cmap="terrain", vmin=0,vmax=2500, alpha=0.7, transform=ccrs.PlateCarree(),zorder=0)
    elif "PCIC" in title:
        
        lon_source_1d = lon_d03[:-1,:-1].ravel()
        lat_source_1d = lat_d03[:-1,:-1].ravel()
        lon_target_1d = lon_pcic.ravel()
        lat_target_1d = lat_pcic.ravel()

        var_source_1d = topo_d03.ravel()

        topo_regridded_1d = griddata((lon_source_1d, lat_source_1d), var_source_1d, (lon_target_1d, lat_target_1d), method='linear')
        topo_regridded = topo_regridded_1d.reshape(lon_pcic.shape)

        topo_pcic = np.ma.masked_array(topo_regridded, mask=var_pcic.mask)
        topo_pcic = topo_pcic.filled(np.nan)
              
        ax1.pcolormesh(lon_pcic, lat_pcic, topo_pcic, cmap="terrain", vmin=0,vmax=2500, alpha=0.7, transform=ccrs.PlateCarree(),zorder=0)


    #ax1.coastlines('10m', linewidth=0.8)
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)

    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/13
    random_x_factor = corner_x3[0]/65

    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=5, edgecolor='red', zorder=2))
    #ax1.text(-3680871, 720000, 'D03', va='top', ha='left',fontweight='bold', size=32, color='red', zorder=4)

    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())

    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', alpha=1,lw=1.5)
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,-49,4))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(0,81,4))

# =============================================================================
#     ax1.text(corner_x3[0]+length_x[2]*0.15, corner_y3[0]+length_y[2]*-0.07, '44ºN', va='top', ha='left', size=18, color='k', zorder=2,rotation=-38,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*-0.17, corner_y3[0]+length_y[2]*0.81, '48ºN', va='top', ha='left', size=18, color='k', zorder=2,rotation=-43,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.365, corner_y3[0]+length_y[2]*0.995, '52ºN', va='top', ha='left', size=18, color='k', zorder=2,rotation=-40,alpha=0.8)
# 
#     ax1.text(corner_x3[0]+length_x[2]*-0.18, corner_y3[0]+length_y[2]*0.14, '128ºW', va='top', ha='left', size=18, color='k', zorder=2,rotation=50,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*-0.18, corner_y3[0]+length_y[2]*0.68, '132ºW', va='top', ha='left', size=18, color='k', zorder=2,rotation=50,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.875, corner_y3[0]+length_y[2]*1.0, '124ºW', va='top', ha='left', size=18, color='k', zorder=2,rotation=55,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.94, corner_y3[0]+length_y[2]*0.57, '120ºW', va='top', ha='left', size=18, color='k', zorder=2,rotation=55,alpha=0.8)
#     ax1.text(corner_x3[0]+length_x[2]*0.91, corner_y3[0]+length_y[2]*-0.01, '116ºW', va='top', ha='left', size=18, color='k', zorder=2,rotation=59,alpha=0.8)
# =============================================================================

    ax1.text(corner_x3[0]+length_x[2]*-0.175, corner_y3[0]+length_y[2]*0.13, '44$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-40,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*-0.175, corner_y3[0]+length_y[2]*0.78, '48$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-38,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.935, corner_y3[0]+length_y[2]*0.55, '52$\degree$N', va='top', ha='left', size=18, color='k', zorder=10,rotation=-30,alpha=0.8)

    ax1.text(corner_x3[0]+length_x[2]*0.1, corner_y3[0]+length_y[2]*1.01, '132$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.53, corner_y3[0]+length_y[2]*1.01, '128$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.875, corner_y3[0]+length_y[2]*1.01, '124$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    #ax1.text(corner_x3[0]+length_x[2]*0.95, corner_y3[0]+length_y[2]*0.67, '120$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=58,alpha=0.8)
    #ax1.text(corner_x3[0]+length_x[2]*0.935, corner_y3[0]+length_y[2]*0.035, '116$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=59,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.1, corner_y3[0]+length_y[2]*-0.08, '124$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.53, corner_y3[0]+length_y[2]*-0.08, '120$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)
    ax1.text(corner_x3[0]+length_x[2]*0.9, corner_y3[0]+length_y[2]*-0.08, '116$\degree$W', va='top', ha='left', size=18, color='k', zorder=10,rotation=10,alpha=0.8)



    return fig1,ax1

def plot_all_d03_flexdomain(lons,lats,topo):
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)


    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)

    ax1.pcolormesh(lons, lats, topo, cmap="terrain", vmin=0,vmax=3000, alpha=0.7, transform=ccrs.PlateCarree(),zorder=0)

    #ax1.coastlines('10m', linewidth=0.8)
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)

    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/13
    random_x_factor = corner_x3[0]/65

    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    ax1.text(-3680871, 700000, 'D03', va='top', ha='left',fontweight='bold', size=15, color='red', zorder=2)

    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())

    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', alpha=1)
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,-49,4))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(0,81,4))

    ax1.text(corner_x3[0]+length_x[2]*0.16, corner_y3[0]+length_y[2]*-0.09, '44ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-40,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*-0.17, corner_y3[0]+length_y[2]*0.8, '48ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-43,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*0.365, corner_y3[0]+length_y[2]*0.995, '52ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-40,alpha=0.6)

    ax1.text(corner_x3[0]+length_x[2]*-0.16, corner_y3[0]+length_y[2]*0.18, '128ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=50,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*-0.165, corner_y3[0]+length_y[2]*0.705, '132ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=50,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*0.875, corner_y3[0]+length_y[2]*1.017, '124ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=55,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*0.96, corner_y3[0]+length_y[2]*0.62, '120ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=58,alpha=0.6)
    ax1.text(corner_x3[0]+length_x[2]*0.975, corner_y3[0]+length_y[2]*0.035, '116ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=58,alpha=0.6)

    return fig1,ax1

def plot_zoomed_in(lon_d02,lat_d02,topo_d02,lon_d03,lat_d03,topo_d03):

    lccproj = ccrs.LambertConformal(central_longitude=-77-20, central_latitude=49,standard_parallels=(49, 49), globe=None, cutoff=-50)

    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)


    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=lccproj)

    ax1.pcolormesh(lon_d02, lat_d02, topo_d02, cmap="terrain", vmin=0,vmax=3000, alpha=0.3, transform=ccrs.PlateCarree(),zorder=0)
    ax1.pcolormesh(lon_d03, lat_d03, topo_d03, cmap="terrain", vmin=0,vmax=3000, alpha=0.4, transform=ccrs.PlateCarree(),zorder=0)

    ax1.coastlines('10m', linewidth=0.8)
    ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS)
    ax1.add_feature(cf.STATES)

    # d03 box

    r2 = mpl.patches.Rectangle((-2287000, 430000),  903000, 903000,fill=None, lw=3, edgecolor='red', zorder=2)
    t2 = mpl.transforms.Affine2D().rotate_deg(15) + ax1.transData
    r2.set_transform(t2)

    ax1.add_patch(r2)
    ax1.text(-2230000, 790000, 'D03', va='top', ha='left',fontweight='bold', size=25, color='red', zorder=2,rotation=15)

    ax1.set_extent([-128, -119, 48, 54], crs=ccrs.PlateCarree())

    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', alpha=1)
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,-49,2))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(0,81,2))


    ax1.text(-2210000, 360000, '48ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-25,alpha=0.6)
    ax1.text(-2230000, 610000, '50ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-25,alpha=0.6)
    ax1.text(-1500000, 575000, '52ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-18,alpha=0.6)
    ax1.text(-1550000, 820000, '54ºN', va='top', ha='left', size=10, color='k', zorder=2,rotation=-18,alpha=0.6)

    ax1.text(-2210000, 900000, '130ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=64,alpha=0.6)
    ax1.text(-2180000, 250000, '126ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=65,alpha=0.6)
    ax1.text(-2235000, 505000, '128ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=65,alpha=0.6)
    ax1.text(-1900000, 950000, '126ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=65,alpha=0.6)
    ax1.text(-1750000, 970000, '124ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=65,alpha=0.6)
    ax1.text(-1645000, 860000, '122ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=70,alpha=0.6)
    ax1.text(-1555000, 710000, '120ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=70,alpha=0.6)
    ax1.text(-1485000, 460000, '118ºW', va='top', ha='left', size=10, color='k', zorder=2,rotation=73,alpha=0.6)

    return fig1,ax1
