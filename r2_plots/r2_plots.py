import pandas as pd
import warnings
import sys
import numpy as np
from netCDF4 import Dataset
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/scripts/')
from canesm2_eval_funcs import get_eccc_obs, get_bch_obs,get_noaa_obs,get_wrf,get_canesm2,get_canrcm4,get_pcic,plot_all_d03,plot_zoomed_in
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib
from scipy.stats import linregress
import math
import matplotlib as mpl

variable = 'wind' #t or pr
run = 'historical' #historical rcp45 or rcp85
output_freq = "yearly" #yearly monthly or daily
#%%

eccc_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv'
bch_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv'
noaa_daily_stations = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations.csv'

df = pd.read_csv(eccc_daily_stations,header=None)
eccc_station_IDs = list(df.iloc[:,4])
eccc_station_names = list(df.iloc[:,1])

eccc_elev = (df.iloc[:,11])
eccc_elev.index = eccc_station_IDs

df = pd.read_csv(bch_daily_stations)
bch_station_IDs = list(df["STATION_NO"])
bch_station_names = list(df["STATION_NA"])

bch_elev = (df["ELEV"])
bch_elev.index = bch_station_IDs

df = pd.read_csv(noaa_daily_stations)
noaa_station_IDs = list(df["STATION"])
noaa_station_names = list(df["NAME"])

noaa_elev = (df["ELEVATION"])
noaa_elev.index = noaa_station_IDs
#%%

stations_dir = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_obs_data/'

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


#%%

if run == 'historical': #no station obs for rcps
    eccc_obs = get_eccc_obs(output_freq,eccc_station_IDs,stations_dir,variable)
    bch_obs = get_bch_obs(output_freq,bch_station_IDs,stations_dir,variable)
    noaa_obs = get_noaa_obs(output_freq,noaa_station_IDs,stations_dir,variable)

wrf_d02_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_bch = get_wrf(output_freq, "BCH", bch_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_bch = get_canesm2(output_freq, "BCH", bch_station_IDs, run, variable, raw_files_dir,start_year)
rcm_bch = get_canrcm4(output_freq, "BCH", bch_station_IDs, run, variable, rcm_files_dir)
#pcic_bch = get_pcic(output_freq, "BCH", bch_station_IDs, run, variable, pcic_files_dir)

wrf_d02_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_eccc = get_wrf(output_freq, "ECCC", eccc_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_eccc = get_canesm2(output_freq, "ECCC", eccc_station_IDs, run, variable, raw_files_dir,start_year)
rcm_eccc = get_canrcm4(output_freq, "ECCC", eccc_station_IDs, run, variable, rcm_files_dir)
#pcic_eccc = get_pcic(output_freq, "ECCC", eccc_station_IDs, run, variable, pcic_files_dir)


wrf_d02_noaa = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d02", run, variable, WRF_files_dir,start_year)
wrf_d03_noaa = get_wrf(output_freq, "NOAA", noaa_station_IDs, "d03", run, variable, WRF_files_dir,start_year)
raw_noaa = get_canesm2(output_freq, "NOAA", noaa_station_IDs, run, variable, raw_files_dir,start_year)
rcm_noaa = get_canrcm4(output_freq, "NOAA", noaa_station_IDs, run, variable, rcm_files_dir)



# remove stations not in the original list
for station in noaa_station_IDs:
    if station not in list(noaa_obs.columns):
        wrf_d02_noaa.drop(station, inplace=True, axis=1)
        wrf_d03_noaa.drop(station, inplace=True, axis=1)
        raw_noaa.drop(station, inplace=True, axis=1)
        rcm_noaa.drop(station, inplace=True, axis=1)
        noaa_elev.drop(station,inplace=True)

        

#%%

if variable !="wind":
    eccc_obs_avg = eccc_obs.mean().sort_index()
    bch_obs_avg = bch_obs.mean().sort_index()
    noaa_obs_avg = noaa_obs.mean().sort_index()
    
    obs_avg = pd.concat([eccc_obs_avg,bch_obs_avg,noaa_obs_avg])
    obs_avg.index = obs_avg.index.astype(str)
    obs_avg = obs_avg.sort_index()
    
    elev = pd.concat([eccc_elev,bch_elev,noaa_elev])
    elev.index = elev.index.astype(str)
    elev = elev.sort_index()
    
    
    wrf_eccc_d02_avg = wrf_d02_eccc.mean().sort_index()
    wrf_bch_d02_avg = wrf_d02_bch.mean().sort_index()
    wrf_noaa_d02_avg = wrf_d02_noaa.mean().sort_index()
    
    wrf_eccc_d03_avg = wrf_d03_eccc.mean().sort_index()
    wrf_bch_d03_avg = wrf_d03_bch.mean().sort_index()
    wrf_noaa_d03_avg = wrf_d03_noaa.mean().sort_index()
    
    raw_eccc_avg = raw_eccc.mean().sort_index()
    raw_bch_avg = raw_bch.mean().sort_index()
    raw_noaa_avg = raw_noaa.mean().sort_index()
    
    rcm_eccc_avg = rcm_eccc.mean().sort_index()
    rcm_bch_avg = rcm_bch.mean().sort_index()
    rcm_noaa_avg = rcm_noaa.mean().sort_index()

else:
    eccc_obs_avg = eccc_obs.mean().sort_index()
    noaa_obs_avg = noaa_obs.mean().sort_index()
    
    obs_avg = pd.concat([eccc_obs_avg,noaa_obs_avg])
    obs_avg.index = obs_avg.index.astype(str)
    obs_avg = obs_avg.sort_index()
    
    elev = pd.concat([eccc_elev,noaa_elev])
    elev.index = elev.index.astype(str)
    elev = elev.sort_index()
    
    
    wrf_eccc_d02_avg = wrf_d02_eccc.mean().sort_index()
    wrf_noaa_d02_avg = wrf_d02_noaa.mean().sort_index()
    
    wrf_eccc_d03_avg = wrf_d03_eccc.mean().sort_index()
    wrf_noaa_d03_avg = wrf_d03_noaa.mean().sort_index()
    
    raw_eccc_avg = raw_eccc.mean().sort_index()
    raw_noaa_avg = raw_noaa.mean().sort_index()
    
    rcm_eccc_avg = rcm_eccc.mean().sort_index()
    rcm_noaa_avg = rcm_noaa.mean().sort_index()


#%%

def plot_scatter(model_eccc_avg,model_bch_avg,model_noaa_avg,vmin,vmax,color,title,unit,figname):
    plt.figure(figsize=(10, 10),dpi=200)
    
    plt.scatter(eccc_obs_avg,model_eccc_avg,color=color,s=200,marker='^',label='ECCC')
    #plt.scatter(bch_obs_avg,model_bch_avg,color=color,s=150,marker='o',label='BCH')
    plt.scatter(noaa_obs_avg,model_noaa_avg,color=color,s=150,marker='s',label='NOAA')

    plt.xlim([vmin,vmax])
    plt.ylim([vmin,vmax])
    
    plt.xlabel('Observed ' + unit,fontsize=18)
    plt.ylabel('Simulated ' + unit,fontsize=18)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    if variable != "wind":
        model_avg = pd.concat([model_eccc_avg,model_bch_avg,model_noaa_avg])
        model_avg.index = model_avg.index.astype(str)
        model_avg = model_avg.sort_index()
        
    else:
        model_avg = pd.concat([model_eccc_avg,model_noaa_avg])
        model_avg.index = model_avg.index.astype(str)
        model_avg = model_avg.sort_index()
            
    #print(min(model_avg))
    #print(min(obs_avg))
    
    #print(max(model_avg))
    #print(max(obs_avg))

    slope, intercept, r_value, p_value, std_err = linregress(obs_avg, model_avg)
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope * x + intercept
    plt.plot(x, line_of_best_fit,'--', color='grey')
    plt.plot(x, x, color='k')
    
    MSE = np.square(np.subtract(obs_avg,model_avg)).mean() 
    RMSE = math.sqrt(MSE)
  
    if variable == "pr":
        r=0
    else:
        r=2
        
        
    plt.scatter(-999,-999,color='white',s=0,label='$R^2$ = ' + str(round(r_value**2,2)))
    plt.scatter(-999,-999,color='white',s=0,label='$RMSE$ = ' + str(round(RMSE,r)))
    
    plt.title(title, fontsize=22)
    plt.legend(fontsize=20)

    
    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/r2/' + figname + '.png',bbox_inches='tight')

if variable == "t":
    vmin = -7.5
    vmax = 15
    title = 'Mean annual air temperature: '
    unit = '(deg C)'
    plot_scatter(wrf_eccc_d03_avg,wrf_bch_d03_avg,wrf_noaa_d03_avg,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_annual_tas_r2')
    plot_scatter(wrf_eccc_d02_avg,wrf_bch_d02_avg,wrf_noaa_d02_avg,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_annual_tas_r2')
    plot_scatter(raw_eccc_avg,raw_bch_avg,raw_noaa_avg,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_annual_tas_r2')
    plot_scatter(rcm_eccc_avg,rcm_bch_avg,rcm_noaa_avg,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_annual_tas_r2')

elif variable == "pr":
    vmin = 0
    vmax = 6000
    title = 'Mean annual total precipitation: '
    unit = '(mm/year)'
    plot_scatter(wrf_eccc_d03_avg,wrf_bch_d03_avg,wrf_noaa_d03_avg,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_annual_pr_r2')
    plot_scatter(wrf_eccc_d02_avg,wrf_bch_d02_avg,wrf_noaa_d02_avg,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_annual_pr_r2')
    plot_scatter(raw_eccc_avg,raw_bch_avg,raw_noaa_avg,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_annual_pr_r2')
    plot_scatter(rcm_eccc_avg,rcm_bch_avg,rcm_noaa_avg,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_annual_pr_r2')

elif variable == "wind":
    vmin = 0
    vmax = 5
    title = 'Wind '
    unit = '(m/s)'
    plot_scatter(wrf_eccc_d03_avg,[],wrf_noaa_d03_avg,vmin,vmax,'C0',title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_annual_wind_r2')
    plot_scatter(wrf_eccc_d02_avg,[],wrf_noaa_d02_avg,vmin,vmax,'C1',title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_annual_wind_r2')
    plot_scatter(raw_eccc_avg,[],raw_noaa_avg,vmin,vmax,'C2',title + 'CanESM2', unit,'canesm2_raw_annual_wind_r2')
    plot_scatter(rcm_eccc_avg,[],rcm_noaa_avg,vmin,vmax,'C3',title + 'CanRCM4', unit,'canrcm4_annual_wind_r2')


#%%
def plot_elev(model_eccc_avg,model_bch_avg,model_noaa_avg,vmin,vmax,title,unit,figname):
    fig = plt.figure(figsize=(10, 10),dpi=200)
    
    model_avg = pd.concat([model_eccc_avg,model_bch_avg,model_noaa_avg])
    model_avg.index = model_avg.index.astype(str)
    model_avg = model_avg.sort_index()
    
    
    plt.scatter(obs_avg,model_avg,c=elev,cmap='terrain',s=150,marker='o',edgecolor='k',vmin=0,vmax=3000)
    
    plt.xlim([vmin,vmax])
    plt.ylim([vmin,vmax])
    
    plt.xlabel('Observed ' + unit,fontsize=18)
    plt.ylabel('Simulated ' + unit,fontsize=18)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    

    slope, intercept, r_value, p_value, std_err = linregress(obs_avg, model_avg)
    x = np.linspace(vmin, vmax) 
    line_of_best_fit = slope * x + intercept
    plt.plot(x, line_of_best_fit,'--', color='grey')
    plt.plot(x, x, color='k')
    
    MSE = np.square(np.subtract(obs_avg,model_avg)).mean() 
    RMSE = math.sqrt(MSE)
  
    if variable == "pr":
        r=0
    else:
        r=1
        
    plt.scatter(-999,-999,color='white',s=0,label='$R^2$ = ' + str(round(r_value**2,2)))
    plt.scatter(-999,-999,color='white',s=0,label='$RMSE$ = ' + str(round(RMSE,r)))
    
    plt.title(title, fontsize=22)
    plt.legend(fontsize=20)

    cbar_ax = fig.add_axes([0.93, 0.12, 0.03, 0.76])
    fig.colorbar(mpl.cm.ScalarMappable(cmap='terrain', norm=mpl.colors.Normalize(vmin=0, vmax=3000)),
                  cax=cbar_ax, ticks=np.arange(0, 3000+1, 500), orientation='vertical')
    cbar_ax.tick_params(labelsize=16)
    cbar_ax.set_ylabel('Elevation [m]',size=18) 

    plt.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/r2/' + figname + '_elev.png',bbox_inches='tight')

if variable == "t":
    vmin = -7.5
    vmax = 15
    title = 'Mean annual air temperature: '
    unit = '(deg C)'
    plot_elev(wrf_eccc_d03_avg,wrf_bch_d03_avg,wrf_noaa_d03_avg,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_annual_tas_r2')
    plot_elev(wrf_eccc_d02_avg,wrf_bch_d02_avg,wrf_noaa_d02_avg,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_annual_tas_r2')
    plot_elev(raw_eccc_avg,raw_bch_avg,raw_noaa_avg,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_annual_tas_r2')
    plot_elev(rcm_eccc_avg,rcm_bch_avg,rcm_noaa_avg,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_annual_tas_r2')

elif variable == "pr":
    vmin = 0
    vmax = 6000
    title = 'Mean annual total precipitation: '
    unit = '(mm/year)'
    plot_elev(wrf_eccc_d03_avg,wrf_bch_d03_avg,wrf_noaa_d03_avg,vmin,vmax,title + 'CanESM2-WRF D03', unit,'canesm2_wrf_d03_annual_pr_r2')
    plot_elev(wrf_eccc_d02_avg,wrf_bch_d02_avg,wrf_noaa_d02_avg,vmin,vmax,title + 'CanESM2-WRF D02', unit,'canesm2_wrf_d02_annual_pr_r2')
    plot_elev(raw_eccc_avg,raw_bch_avg,raw_noaa_avg,vmin,vmax,title + 'CanESM2', unit,'canesm2_raw_annual_pr_r2')
    plot_elev(rcm_eccc_avg,rcm_bch_avg,rcm_noaa_avg,vmin,vmax,title + 'CanRCM4', unit,'canrcm4_annual_pr_r2')



    