import numpy as np
from netCDF4 import Dataset
import WRFDomainLib
import cartopy.feature as cf
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib
import matplotlib as mpl
import matplotlib.colors as pltcol

file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/avgs/historical/pr_d03_mon.nc'
nc = Dataset(file, mode='r')
pr_hist = nc.variables['pr'][:]


file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/avgs/historical/t_d03_tmin_mon.nc'
nc = Dataset(file, mode='r')
tmin_hist = nc.variables['T2'][:]-273.15

file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/avgs/historical/t_d03_tmax_mon.nc'
nc = Dataset(file, mode='r')
tmax_hist = nc.variables['T2'][:]-273.15


file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/avgs/rcp45/pr_d03_mon.nc'
nc = Dataset(file, mode='r')
pr_rcp45 = nc.variables['pr'][:]

file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/avgs/rcp45/t_d03_tmin_mon.nc'
nc = Dataset(file, mode='r')
tmin_rcp45 = nc.variables['T2'][:]-273.15

file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/avgs/rcp45/t_d03_tmax_mon.nc'
nc = Dataset(file, mode='r')
tmax_rcp45 = nc.variables['T2'][:]-273.15

file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/avgs/rcp85/pr_d03_mon.nc'
nc = Dataset(file, mode='r')
pr_rcp85 = nc.variables['pr'][:]

file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/avgs/rcp85/t_d03_tmin_mon.nc'
nc = Dataset(file, mode='r')
tmin_rcp85 = nc.variables['T2'][:]-273.15

file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/avgs/rcp85/t_d03_tmax_mon.nc'
nc = Dataset(file, mode='r')
tmax_rcp85 = nc.variables['T2'][:]-273.15


geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'
geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
lat_d03 = np.squeeze(geo_em_d03_nc.variables['XLAT_C'][:])
lon_d03 = np.squeeze(geo_em_d03_nc.variables['XLONG_C'][:])
landmask = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])


     
#%%

def koppen(tmax,tmin,pr):
    climate = pr.copy()[0,:,:]
    climate[:] = np.nan
    
    for i in range(np.shape(tmax_hist)[1]):
        print(i)
        for j in range(np.shape(tmax_hist)[2]):
            highs = tmax[:,i,j]
            lows = tmin[:,i,j]
            precip = pr[:,i,j]
    
            avgtemp = (highs + lows) / 2.0
            totalprecip = sum(precip)
            
            # Group A (Tropical)
            if min(avgtemp) >= 18.0:
                # Tropical Rainforest
                if min(precip) >= 60.0:
                    climate[i,j] = 0#'Af'
                    continue
                # Tropical Monsoon
                elif min(precip) < 60.0 and (min(precip) / totalprecip) > 0.04:
                    climate[i,j] = 1#'Am'
                    continue
                else:
                    # Tropical Savanna Dry Summer
                    if np.where(precip==min(precip))[0][0] >= 6 and np.where(precip==min(precip))[0][0] <= 8:
                        climate[i,j] = 2#'As'
                        continue
                    # Tropical Savanna Dry Winter
                    else:
                        climate[i,j] = 3#'Aw'
                        continue
                    
                    
            # Group B (Arid and Semiarid)
            aridity = np.mean(avgtemp) * 20.0
            warmprecip = sum(precip[3:9])
            coolprecip = sum(precip[0:3]) + sum(precip[9:12])
            if warmprecip / totalprecip >= 0.70:
                aridity = aridity + 280.0
            elif warmprecip / totalprecip >= 0.30 and warmprecip / totalprecip < 0.70:
                aridity = aridity + 140.0
            else:
                aridity = aridity + 0.0
        
            # Arid Desert (BW)
            if totalprecip / aridity < 0.50:
                # Hot Desert (BWh)
                if np.mean(avgtemp) > 18.0:
                    climate[i,j] = 4#'BWh'
                    continue
                # Cold Desert (BWk)
                else:
                    climate[i,j] = 5#'BWk'
                    continue
        
            if climate[i,j] in [0,1,2,3]:
                continue
        
            # Semi-Arid/Steppe (BS)
            elif totalprecip / aridity >= 0.50 and totalprecip / aridity < 1.00:
                # Hot Semi-Arid (BSh)
                if np.mean(avgtemp) > 18.0:
                    climate[i,j] = 6#'BSh'
                    continue
                # Cold Semi-Arid (BSk)
                else:
                    climate[i,j] = 7#'BSk'
                    continue
        
            if climate[i,j] in [4,5,6,7]:
                continue
        
            # Group C (Temperate)
            sortavgtemp = avgtemp
            sortavgtemp.sort()
            tempaboveten = np.shape(np.where(avgtemp>10.0))[1]
            coldwarmratio = max(max(precip[0:2]),precip[11]) / min(precip[5:8])
            warmcoldratio = max(precip[5:8]) / min(min(precip[0:2]),precip[11])
            if min(avgtemp) >= 0.0 and min(avgtemp) <= 18.0 and max(avgtemp) >= 10.0:
                # Humid Subtropical (Cfa)
                if min(avgtemp) > 0.0 and max(avgtemp) > 22.0 and tempaboveten >= 4.0:
                    climate[i,j] = 8#'Cfa'
                # Temperate Oceanic (Cfb)
                elif min(avgtemp) > 0.0 and max(avgtemp) < 22.0 and tempaboveten >= 4.0:
                    climate[i,j] = 9#'Cfb'
                # Subpolar Oceanic (Cfc)
                elif min(avgtemp) > 0.0 and tempaboveten >= 1 and tempaboveten <= 3:
                    climate[i,j] = 10#'Cfc'
        
                # Monsoon-influenced humid subtropical (Cwa)
                if min(avgtemp) > 0.0 and max(avgtemp) > 22.0 and tempaboveten >= 4 and warmcoldratio > 10.0:
                    climate[i,j] = 11#'Cwa'
                # Subtropical Highland/Temperate Oceanic with Dry Winter (Cwb)
                elif min(avgtemp) > 0.0 and max(avgtemp) < 22.0 and tempaboveten >= 4 and warmcoldratio > 10.0:
                    climate[i,j] = 12#'Cwb'
                # Cold Subtropical Highland/Subpolar Oceanic with Dry Winter (Cwc)
                elif min(avgtemp) > 0.0 and tempaboveten >= 1 and tempaboveten <= 3 and warmcoldratio > 10.0:
                    climate[i,j] = 13#'Cwc'
        
                # Hot summer Mediterranean (Csa)
                if min(avgtemp) > 0.0 and max(avgtemp) > 22.0 and tempaboveten >= 4 and \
                    coldwarmratio >= 3.0 and min(precip[5:8]) < 30.0:
                    climate[i,j] = 14#'Csa'
                # Warm summer Mediterranean (Csb)
                elif min(avgtemp) > 0.0 and max(avgtemp) < 22.0 and tempaboveten >= 4 and \
                    coldwarmratio >= 3.0 and min(precip[5:8]) < 30.0:
                    climate[i,j] = 15#'Csb'
                # Cool summer Mediterranean (Csc)
                elif min(avgtemp) > 0.0 and tempaboveten >= 1 and tempaboveten <= 3 and \
                    coldwarmratio >= 3.0 and min(precip[5:8]) < 30.0:
                    climate[i,j] = 16#'Csc'
        
                if climate[i,j] in [8,9,10,11,12,13,14,15,16]:
                    continue
        
            # Group D (Continental)
            if min(avgtemp) < 0.0 and max(avgtemp) > 10.0:
                # Hot summer humid continental (Dfa)
                if max(avgtemp) > 22.0 and tempaboveten >= 4:
                    climate[i,j] = 17#'Dfa'
                # Warm summer humid continental (Dfb)
                elif max(avgtemp) < 22.0 and tempaboveten >= 4:
                    climate[i,j] = 18#'Dfb'
                # Subarctic (Dfc)
                elif tempaboveten >= 1 and tempaboveten <= 3:
                    climate[i,j] = 19#'Dfc'
                # Extremely cold subarctic (Dfd)
                elif min(avgtemp) < -38.0 and tempaboveten >=1 and tempaboveten <= 3:
                    climate[i,j] = 20#'Dfd'
        
                # Monsoon-influenced hot humid continental (Dwa)
                if max(avgtemp) > 22.0 and tempaboveten >= 4 and warmcoldratio >= 10:
                    climate[i,j] = 21#'Dwa'
                # Monsoon-influenced warm humid continental (Dwb)
                elif max(avgtemp) < 22.0 and tempaboveten >= 4 and warmcoldratio >= 10:
                    climate[i,j] = 22#'Dwb'
                # Monsoon-influenced subarctic (Dwc)
                elif tempaboveten >= 1 and tempaboveten <= 3 and warmcoldratio >= 10:
                    climate[i,j] = 23#'Dwc'
                # Monsoon-influenced extremely cold subarctic (Dwd)
                elif min(avgtemp) < -38.0 and tempaboveten >= 1 and tempaboveten <= 3 and warmcoldratio >= 10:
                    climate[i,j] = 24#'Dwd'
        
                # Hot, dry continental (Dsa)
                if max(avgtemp) > 22.0 and tempaboveten >= 4 and coldwarmratio >= 3 and min(precip[5:8]) < 30.0:
                    climate[i,j] = 25#'Dsa'
                # Warm, dry continental (Dsb)
                elif max(avgtemp) < 22.0 and tempaboveten >= 4 and coldwarmratio >= 3 and min(precip[5:8]) < 30.0:
                    climate[i,j] = 26#'Dsb'
                # Dry, subarctic (Dsc)
                elif tempaboveten >= 1 and tempaboveten <= 3 and coldwarmratio >= 1 and coldwarmratio >= 3 and \
                    min(precip[5:8]) < 30.0:
                    climate[i,j] = 27#'Dsc'
                # Extremely cold, dry subarctic (Dsd)
                elif min(avgtemp) < -38.0 and tempaboveten >= 1 and tempaboveten <= 3 and coldwarmratio >= 3 and \
                    min(precip[5:8]) < 30.0:
                    climate[i,j] = 28#'Dsd'
        
                if climate[i,j] in [17,18,19,20,21,22,23,24,25,26,27,28]:
                    continue
        
            # Group E (Polar and alpine)
            if max(avgtemp) < 10.0:
                # Tundra (ET)
                if max(avgtemp) > 0.0:
                    climate[i,j] = 29#'ET'
                # Ice cap (EF)
                else:
                    climate[i,j] = 30#'EF'
                    
    return(climate)
     
#%%

koppen_colors = [('#0000fe'), #Af
                 ('#0277ff'), #Am
                 ('#379ae5'), #As
                 ('#6cb1e5'), #Aw
                 ('#fe0100'), #BWh
                 ('#fe9695'), #BWk
                 ('#f5a300'), #Bsh
                 ('#ffdb63'), #Bsk
                 ('#ffff00'), #Csa
                 ('#c6c701'), #Csb
                 ('#969600'), #Csc
                 ('#96ff96'), #Cwa
                 ('#63c764'), #Cwb
                 ('#329633'), #Cwc
                 ('#c7ff4d'), #Cfa
                 ('#66ff33'), #Cfb
                 ('#32c702'), #Cfc
                 ('#ff00fe'), #Dsa
                 ('#c600c7'), #Dsb
                 ('#963295'), #Dsc
                 ('#966495'), #Dsd
                 ('#abb1ff'), #Dwa
                 ('#5a77db'), #Dwb
                 ('#4c51b5'), #Dwc
                 ('#320087'), #Dwd
                 ('#02ffff'), #Dfa
                 ('#37c7ff'), #Dfb
                 ('#007e7d'), #Dfc
                 ('#00455e'), #Dfd
                 ('#b2b2b2'), #ET
                 ('#686868')] #EF
   
koppen_cats = ['Af','Am','As','Aw',
               'BWh','BWk','BSh','BSk',
               'Csa','Csb','Csc',
               'Cwa','Cwb','Cwc',
               'Cfa','Cfb','Cfc',
               'Dsa','Dsb','Dsc','Dsd',
               'Dwa','Dwb','Dwc','Dwd',
               'Dfa','Dfb','Dfc','Dfd',
               'ET','EF']
       

# makes the colorbar
cmap_koppen = pltcol.LinearSegmentedColormap.from_list("custom", koppen_colors,N=31) 



 #%%

 
wrf_d03_hist_climate = koppen(tmax_hist,tmin_hist,pr_hist)   

wrf_d03_rcp45_climate = koppen(tmax_rcp45,tmin_rcp45,pr_rcp45)   

wrf_d03_rcp85_climate = koppen(tmax_rcp85,tmin_rcp85,pr_rcp85)   

#%%

def plot_koppen(climate,name):


    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
    
    climate[landmask == 0] = np.nan
    vmin=0
    vmax=31
    ax1.pcolormesh(lon_d03, lat_d03, climate, cmap=cmap_koppen, transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)
    
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    cbar_ax = fig1.add_axes([0.2, 0.09, 0.62, 0.02])
    fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap_koppen, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)),
                  cax=cbar_ax, orientation='horizontal')

          
    ax1.set_title('Koppen Climate Classifications for ' + name,fontsize=17)
    #fig1.savefig('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/spatial_maps/koppen/' + name + '.png', dpi=600,bbox_inches='tight')
    


plot_koppen(wrf_d03_hist_climate,'CanESM2-WRF D03 historical')
#plot_koppen(wrf_d03_rcp45_climate,'CanESM2-WRF D03 RCP4.5')
#plot_koppen(wrf_d03_rcp85_climate,'CanESM2-WRF D03 RCP8.5')
