#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:20:03 2024

@author: evagnegy
"""

""" old percentile function
t_perc_max = 90
t_perc_min = 10
pr_perc_limit = 75

window_size_t = 5    #needs to be odd
window_size_pr = 29  #needs to be odd

window_half_t = int((window_size_t-1)/2)
window_half_pr = int((window_size_pr-1)/2)  

daysinyear = 366 

def get_percentiles(wrf_d03_time_hist,wrf_d03_tx_hist,wrf_d03_tn_hist,wrf_d03_pr_hist):
    day_of_years = [dt.timetuple().tm_yday for dt in wrf_d03_time_hist]
    
    t_warmday_percentile,t_coldday_percentile,t_warmnight_percentile,t_coldnight_percentile=[],[],[],[]
    
    j = window_half_t+1
    for i in range(1,daysinyear+1):
        print(i)
        if i in range(1,window_half_t+1):
            rolling_days = [dt for dt, day in zip(wrf_d03_time_hist, day_of_years) if day >= daysinyear-window_half_t+(i-1) or day <= i+window_half_t]
            
        elif i in range(window_half_t+1,daysinyear-2):
            rolling_days = [dt for dt, day in zip(wrf_d03_time_hist, day_of_years) if day >= i-window_half_t and day <= i+window_half_t]
      
        elif i in range(daysinyear-2,daysinyear+1): 
            j += -1
            rolling_days = [dt for dt, day in zip(wrf_d03_time_hist, day_of_years) if day >= i-window_half_t or day <= (3-j)]
        
        selected_tx_hist = [wrf_d03_tx_hist[i,:,:] for i, dt in enumerate(wrf_d03_time_hist) if dt in rolling_days]
        selected_tn_hist = [wrf_d03_tn_hist[i,:,:] for i, dt in enumerate(wrf_d03_time_hist) if dt in rolling_days]
        selected_pr_hist = [wrf_d03_pr_hist[i,:,:] for i, dt in enumerate(wrf_d03_time_hist) if dt in rolling_days]
        
        t_warmday_percentile.append(np.percentile(selected_tx_hist,t_perc_max,axis=0))
        t_coldday_percentile.append(np.percentile(selected_tx_hist,t_perc_min,axis=0))
        t_warmnight_percentile.append(np.percentile(selected_tn_hist,t_perc_max,axis=0))
        t_coldnight_percentile.append(np.percentile(selected_tn_hist,t_perc_min,axis=0))
    
        
    pr_percentile = []
    
    j = window_half_pr+1
    for i in range(1,daysinyear+1):
        print(i)
        if i in range(1,window_half_pr+1):
            rolling_days = [dt for dt, day in zip(wrf_d03_time_hist, day_of_years) if day >= daysinyear-window_half_pr+(i-1) or day <= i+window_half_pr]
        elif i in range(window_half_pr+1,daysinyear-2):
            rolling_days = [dt for dt, day in zip(wrf_d03_time_hist, day_of_years) if day >= i-window_half_pr and day <= i+window_half_pr]
      
        elif i in range(daysinyear-2,daysinyear+1): 
            j += -1
            rolling_days = [dt for dt, day in zip(wrf_d03_time_hist, day_of_years) if day >= i-window_half_pr or day <= (3-j)]
            
        selected_pr_hist = [wrf_d03_pr_hist[i,:,:] for i, dt in enumerate(wrf_d03_time_hist) if dt in rolling_days]
    
        selected_pr_hist_wet = np.array(selected_pr_hist)
        selected_pr_hist_wet[selected_pr_hist_wet<1] = np.nan
    
        pr_percentile.append(np.nanpercentile(selected_pr_hist_wet,pr_perc_limit,axis=0))
    

    
    t_warmday_percentiles_long,t_coldday_percentiles_long,t_warmnight_percentiles_long,t_coldnight_percentiles_long=[],[],[],[]
    pr_percentiles_long = []
       
    for date in wrf_d03_time_hist:
        for i in range(1,daysinyear+1):
            if date.timetuple().tm_yday == i:
    
                t_warmday_percentiles_long.append(t_warmday_percentile[i-1])
                t_coldday_percentiles_long.append(t_coldday_percentile[i-1])
                t_warmnight_percentiles_long.append(t_warmnight_percentile[i-1])
                t_coldnight_percentiles_long.append(t_coldnight_percentile[i-1])
                pr_percentiles_long.append(pr_percentile[i-1])

    return(t_warmday_percentiles_long, t_coldday_percentiles_long, t_warmnight_percentiles_long,t_coldnight_percentiles_long, pr_percentiles_long)

#t_warmday_percentiles_long_hist, t_coldday_percentiles_long_hist, t_warmnight_percentiles_long_hist,t_coldnight_percentiles_long_hist, pr_percentiles_long_hist = get_percentiles(wrf_d03_time_hist,wrf_d03_tx_hist,wrf_d03_tn_hist,wrf_d03_pr_hist)
#t_warmday_percentiles_long_rcp45, t_coldday_percentiles_long_rcp45, t_warmnight_percentiles_long_rcp45,t_coldnight_percentiles_long_rcp45, pr_percentiles_long_rcp45 = get_percentiles(wrf_d03_time_fut,wrf_d03_tx_rcp45,wrf_d03_tn_rcp45,wrf_d03_pr_rcp45)
#t_warmday_percentiles_long_rcp85, t_coldday_percentiles_long_rcp85, t_warmnight_percentiles_long_rcp85,t_coldnight_percentiles_long_rcp85, pr_percentiles_long_rcp85 = get_percentiles(wrf_d03_time_fut,wrf_d03_tx_rcp85,wrf_d03_tn_rcp85,wrf_d03_pr_rcp85)

"""