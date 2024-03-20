
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys
import numpy as np
import matplotlib.ticker as ticker
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic

variable = 't' #t or pr
run = 'rcp45' #historical rcp45 or rcp85
output_freq = "monthly" #yearly monthly or daily
#%%

eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'
noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations.csv'

df = pd.read_csv(eccc_daily_stations,header=None)
eccc_station_IDs = list(df.iloc[:,4])
eccc_station_names = list(df.iloc[:,1])

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])

df = pd.read_csv(noaa_daily_stations)
noaa_station_IDs = list(df.iloc[:,0])
noaa_station_names = list(df.iloc[:,1])
#%%

eccc_stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/ECCC/'
bch_stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/BCH/'

WRF_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_WRF/' + run + '/'
raw_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/' + run + '/'
rcm_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/' + run + '/'
pcic_files_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_PCIC/' + run + '/'


if run == 'historical':
    start_year = 1986
    end_year = 2005
else:
    start_year = 2046
    end_year = 2065  


if variable == "t":
    label="Temperature [deg C]"
elif variable == "pr":
    label="Total Annual Precipitation [mm]"

#%%

if run == 'historical': #no station obs for rcps
    eccc_obs = get_eccc_obs(output_freq,eccc_station_IDs,eccc_stations_dir,variable)
    bch_obs = get_bch_obs(output_freq,bch_station_IDs,bch_stations_dir,variable)


wrf_d03_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_bch = get_canesm2(output_freq, "BCH", bch_station_IDs, run, variable, raw_files_dir,start_year)
rcm_bch = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, variable, rcm_files_dir)
pcic_bch = get_pcic(output_freq, "BCH", bch_station_IDs, run, variable, pcic_files_dir)



# =============================================================================
# wrf_d03_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
# raw_eccc = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, variable, raw_files_dir,start_year)
# rcm_eccc = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, variable, rcm_files_dir)
# pcic_eccc = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, variable, pcic_files_dir)
# =============================================================================

#%%

#bch_obs.index = pd.to_datetime(bch_obs.index,format='%Y-%m-%d')

#eccc_obs_wet = eccc_obs.copy() 
bch_obs_wet = bch_obs.copy()
raw_bch_wet = raw_bch.copy() 
#raw_eccc_wet = raw_eccc.copy()
wrf_d03_bch_wet = wrf_d03_bch.copy() 
#wrf_d03_eccc_wet = wrf_d03_eccc.copy()
rcm_bch_wet = rcm_bch.copy() 
#rcm_eccc_wet = rcm_eccc.copy()
pcic_bch_wet = pcic_bch.copy() 
#pcic_eccc_wet = pcic_eccc.copy()

for i in [4,5,6,7,8,9]:
    bch_obs_wet = bch_obs_wet[bch_obs_wet.index.month != i]
    #eccc_obs_wet = eccc_obs_wet[eccc_obs_wet.index.month != i]

    wrf_d03_bch_wet = wrf_d03_bch_wet[wrf_d03_bch_wet.index.month != i]
    #wrf_d03_eccc_wet = wrf_d03_eccc_wet[wrf_d03_eccc_wet.index.month != i]

    raw_bch_wet = raw_bch_wet[raw_bch_wet.index.month != i]
    #raw_eccc_wet = raw_eccc_wet[raw_eccc_wet.index.month != i]

    rcm_bch_wet = rcm_bch_wet[rcm_bch_wet.index.month != i]
    #rcm_eccc_wet = rcm_eccc_wet[rcm_eccc_wet.index.month != i]

    pcic_bch_wet = pcic_bch_wet[pcic_bch_wet.index.month != i]
    #pcic_eccc_wet = pcic_eccc_wet[pcic_eccc_wet.index.month != i]



wrf_d03_bch_95 = wrf_d03_bch_wet.quantile(0.95)
#wrf_d03_eccc_95 = wrf_d03_eccc_wet.quantile(0.95)

wrf_d03_bch_99 = wrf_d03_bch_wet.quantile(0.99)
#wrf_d03_eccc_99 = wrf_d03_eccc_wet.quantile(0.99)

wrf_d03_bch_max = wrf_d03_bch_wet.quantile(1)
#wrf_d03_eccc_max = wrf_d03_eccc_wet.quantile(1)
#%% yearly averages

def plot_historical_timeseries(output_freq,station_IDs,station_names,historical,agency,wrf,raw,rcm,pcic):
    for i in range(len(station_IDs)):
        
        station_ID = station_IDs[i]
        station_name = station_names[i]
        
    
        fig, ax = plt.subplots(figsize=(11, 5))
           
        if run == "historical":
            plt.plot(historical[station_ID].index.strftime('%Y'),historical[station_ID].values,'--',label=agency + "station data",color="black",linewidth=2)
        
        plt.plot(wrf[station_ID].index.strftime('%Y'),wrf[station_ID].values,label="CanESM2-WRF d03",color="C0")
    
        plt.plot(raw[station_ID].index.strftime('%Y'),raw[station_ID].values,'-.',label="CanESM2",color="C3")
        plt.plot(rcm[station_ID].index.strftime('%Y'),rcm[station_ID].values,'-.',label="CanRCM4",color="C1")
        plt.plot(pcic[station_ID].index.strftime('%Y'),pcic[station_ID].values,'-.',label="PCIC (CanESM2)",color="C5")

    
        plt.legend()        
        plt.ylabel(label,fontsize=12)
        plt.title(station_name + " observation station",fontsize=14)# + ": " + str(station_ID))
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))
    
        #plt.ylim([600,2250])
        #plt.ylim([4.4,12])
        if agency == 'BCH':
            plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/indiv_stations/timeseries/historical/' + variable + '/' + output_freq + '/' + agency + "_" + str(station_ID) + "_" + variable,bbox_inches='tight',dpi=62)
        elif agency == "ECCC":
            plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/indiv_stations/timeseries/historical/' + variable + '/' + output_freq + '/' + agency + '_' + station_name + "_" + str(station_ID) + "_" + variable,bbox_inches='tight',dpi=62)
        plt.close()
    
    
#plot_historical_timeseries(output_freq,bch_station_IDs,bch_station_names,bch_obs,"BCH",wrf_d03_bch,raw_bch,rcm_bch,pcic_bch)
#plot_historical_timeseries(output_freq,eccc_station_IDs,eccc_station_names,eccc_obs,"ECCC",wrf_d03_eccc,raw_eccc,rcm_eccc,pcic_eccc)

#%% precip cumsum

def plot_cumsum_pr(output_freq,station_names,station_IDs,historical,agency,wrf,raw,rcm,pcic):

    for i in range(len(station_IDs)):

        station_ID = station_IDs[i]
        station_name = station_names[i]
        
        fig, ax = plt.subplots(figsize=(11, 5))
        
        if run == "historical":
            plt.plot(historical[station_ID].index.strftime('%Y'),np.cumsum(list(historical[station_ID]))/1000,'--',label="station data",color="black",linewidth=2)
    
        plt.plot(wrf[station_ID].index.strftime('%Y'),np.cumsum(list(wrf[station_ID]))/1000,label="CanESM2-WRF d03",color="C0")            
        plt.plot(raw[station_ID].index.strftime('%Y'),np.cumsum(list(raw[station_ID]))/1000,'-.',label="CanESM2",color="C3")
        plt.plot(rcm[station_ID].index.strftime('%Y'),np.cumsum(list(rcm[station_ID]))/1000,'-.',label="CanRCM4",color="C1")
        plt.plot(pcic[station_ID].index.strftime('%Y'),np.cumsum(list(pcic[station_ID]))/1000,'-.',label="PCIC (CanESM2)",color="C5")

        plt.legend()
       
        plt.ylabel('Accumulated Yearly Precipitation [m]',fontsize=12)
        plt.title(station_name + " observation station",fontsize=14)# + ": " + str(station_ID))
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))
    
        #plt.ylim([600,2250])
        #plt.ylim([4.4,12])
        plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/indiv_stations/accum_precip_timeseries/historical/' + agency + "_" + str(station_ID) + "_yrly_cumsum",bbox_inches='tight',dpi=62)
        plt.close()

plot_cumsum_pr(output_freq,bch_station_names,bch_station_IDs,bch_obs,"BCH",wrf_d03_bch,raw_bch,rcm_bch,pcic_bch)
plot_cumsum_pr(output_freq,eccc_station_names,eccc_station_IDs,eccc_obs,"ECCC",wrf_d03_eccc,raw_eccc,rcm_eccc,pcic_eccc)

        
#%% histograms

def plot_hists(output_freq,station_IDs,station_names,historical,agency,wrf,raw,rcm,pcic):

    for i in range(len(station_IDs)):
        
        station_ID = station_IDs[i]
        station_name = station_names[i]
        print(station_ID)
    
        fig, ax = plt.subplots(figsize=(7, 4))
           
        bins=30
        if run == "historical":
            plt.hist(historical[station_ID],bins,label="station data",color="black",linewidth=2,density=True,histtype='step')
        
        plt.hist(wrf[station_ID],bins,label="CanESM2-WRF d03",color="C0",density=True,histtype='step')
        plt.hist(raw[station_ID],bins,label="CanESM2",color="C3",density=True,histtype='step')
        plt.hist(rcm[station_ID],bins,label="CanRCM4",color="C1",density=True,histtype='step')
        
        if agency == "ECCC":
            if variable == "tas":
                if all(pcic_eccc[station_ID].isna()):
                    print('skipping pcic ' + str(station_ID))
                else:
                    plt.hist(pcic[station_ID],bins,label="PCIC (CanESM2)",color="C5",density=True,histtype='step')
            elif variable == "pr":
                if pcic_eccc[station_ID].sum() == 0:
                    print('skipping pcic ' + str(station_ID))
                else:
                    plt.hist(pcic[station_ID],bins,label="PCIC (CanESM2)",color="C5",density=True,histtype='step')
        else:
            plt.hist(pcic[station_ID],bins,label="PCIC (CanESM2)",color="C5",density=True,histtype='step')

                    
        plt.legend(loc='upper right')
        
        plt.ylabel('PDF',fontsize=12)
        #plt.xlabel('Daily temperatures [deg C]',fontsize=12)
        plt.xlabel('Monthly accum precip [mm/mon]',fontsize=12)
        #plt.xlabel('Daily accum precip [mm/day]',fontsize=12)


        plt.title(station_name + " observation station (Oct-March)",fontsize=14)# + ": " + str(station_ID))
        #for axis in [ax.xaxis, ax.yaxis]:
        #    axis.set_major_locator(ticker.MaxNLocator(integer=True))
    
        #plt.ylim([600,2250])
        #plt.ylim(ymin=-0.01)
        #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/indiv_stations/tas_histograms/historical/' + agency + "_" + str(station_ID) + "_daily_tas",bbox_inches='tight',dpi=62)
        plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/indiv_stations/pr_histograms/historical/' + agency + "_" + str(station_ID) + "_" + output_freq + "_pr_wet",bbox_inches='tight',dpi=62)
        #plt.close()
 
#plot_hists(output_freq,bch_station_IDs,bch_station_names,bch_obs,"BCH",wrf_d03_bch,raw_bch,rcm_bch,pcic_bch)

plot_hists(output_freq,bch_station_IDs,bch_station_names,bch_obs_wet,"BCH",wrf_d03_bch_wet,raw_bch_wet,rcm_bch_wet,pcic_bch_wet)
#plot_hists(output_freq,eccc_station_IDs,eccc_station_names,eccc_obs_wet,"ECCC",wrf_d03_eccc_wet,raw_eccc_wet,rcm_eccc_wet,pcic_eccc_wet)
   
 #%% histograms hourly

def plot_hists_hr(output_freq,station_IDs,station_names,agency,wrf,wrf_95,wrf_99,wrf_max):

     for i in range(len(station_IDs)):
         
         station_ID = station_IDs[i]
         station_name = station_names[i]
         print(station_ID)
     
         fig, ax = plt.subplots(figsize=(7, 4))
            
         bins=100

         plt.hist(wrf[station_ID],bins,label="CanESM2-WRF d03",color="C0",density=True,histtype='step',linewidth=2)
         plt.axvline(wrf_95[station_ID], color='k', linestyle='dashed', linewidth=1)
         plt.axvline(wrf_99[station_ID], color='k', linestyle='dashed', linewidth=1)
         plt.axvline(wrf_max[station_ID], color='k', linestyle='dashed', linewidth=1)


         plt.legend(loc='upper right')
         
         plt.ylabel('PDF',fontsize=12)
         #plt.xlabel('Daily temperatures [deg C]',fontsize=12)
         plt.xlabel('Hourly precip rates [mm/hr]',fontsize=12)

         plt.title(station_name + " observation station (Oct-March)",fontsize=14)# + ": " + str(station_ID))
         #for axis in [ax.xaxis, ax.yaxis]:
         #    axis.set_major_locator(ticker.MaxNLocator(integer=True))
     
         #plt.ylim([600,2250])
         plt.ylim(ymin=-0.1)
         #plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/indiv_stations/tas_histograms/historical/' + agency + "_" + str(station_ID) + "_daily_tas",bbox_inches='tight',dpi=62)
         plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/indiv_stations/pr_histograms/historical/' + agency + "_" + str(station_ID) + "_hourly_pr_wet",bbox_inches='tight',dpi=62)
         #plt.close()
    
plot_hists_hr(output_freq,bch_station_IDs,bch_station_names,"BCH",wrf_d03_bch_wet)
        
#plot_hists_hr(output_freq,bch_station_IDs,bch_station_names,"BCH",wrf_d03_bch_wet,wrf_d03_bch_95,wrf_d03_bch_99,wrf_d03_bch_max)
#plot_hists_hr(output_freq,eccc_station_IDs,eccc_station_names,"ECCC",wrf_d03_eccc_wet,wrf_d03_eccc_95,wrf_d03_eccc_99,wrf_d03_eccc_max)
      
    