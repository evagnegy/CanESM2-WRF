from __future__ import print_function

import numpy as np
import os
import pandas as pd
import sys
import xclim
import xarray as xr

hist_path = '~/space/CanESM2-WRF/historical/variables_complete/'
rcp45_path = '~/space/CanESM2-WRF/rcp45/variables_complete/'
rcp85_path = '~/space/CanESM2-WRF/rcp85/variables_complete/'


#tas_hist = xr.open_dataset(hist_path+'t_d03_daily_newdim.nc')['T2'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})
tn_hist = xr.open_dataset(hist_path+'t_d03_tmin_daily_newdim.nc')['T2'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})
tx_hist = xr.open_dataset(hist_path+'t_d03_tmax_daily_newdim.nc')['T2'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})
#pr_hist = xr.open_dataset(hist_path+'pr_d03_daily_newdim.nc')['pr'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})

#tas_hist.attrs['units'] = 'K'
tn_hist.attrs['units'] = 'K'
tx_hist.attrs['units'] = 'K'
#pr_hist.attrs['units'] = 'mm/day'

#tas_rcp45 = xr.open_dataset(rcp45_path+'t_d03_daily_newdim.nc')['T2'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})
#tas_rcp85 = xr.open_dataset(rcp85_path+'t_d03_daily_newdim.nc')['T2'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})

tn_rcp45 = xr.open_dataset(rcp45_path+'t_d03_tmin_daily_newdim.nc')['T2'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})
tn_rcp85 = xr.open_dataset(rcp85_path+'t_d03_tmin_daily_newdim.nc')['T2'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})

tx_rcp45 = xr.open_dataset(rcp45_path+'t_d03_tmax_daily_newdim.nc')['T2'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})
tx_rcp85 = xr.open_dataset(rcp85_path+'t_d03_tmax_daily_newdim.nc')['T2'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})

#tas_rcp45.attrs['units'] = 'K'
#tas_rcp85.attrs['units'] = 'K'

tn_rcp45.attrs['units'] = 'K'
tn_rcp85.attrs['units'] = 'K'

tx_rcp45.attrs['units'] = 'K'
tx_rcp85.attrs['units'] = 'K'

#pr_rcp45 = xr.open_dataset(rcp45_path+'pr_d03_daily_newdim.nc')['pr'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})
#pr_rcp85 = xr.open_dataset(rcp85_path+'pr_d03_daily_newdim.nc')['pr'].chunk({'time':-1, 'latitude': 20, 'longitude': 20})

#pr_rcp45.attrs['units'] = 'mm/day'
#pr_rcp85.attrs['units'] = 'mm/day'


#pr_hist = pr_hist.where(pr_hist > 1, drop=True)
#pr_rcp45 = pr_rcp45.where(pr_rcp45 > 1, drop=True)
#pr_rcp85 = pr_rcp85.where(pr_rcp85 > 1, drop=True)


#tas10p_hist = xclim.core.calendar.percentile_doy(tas_hist,window=5,per=10.0)
#tas10p_rcp45 = xclim.core.calendar.percentile_doy(tas_rcp45,window=5,per=10.0)
#tas10p_rcp85 = xclim.core.calendar.percentile_doy(tas_rcp85,window=5,per=10.0)


#tas90p_hist = xclim.core.calendar.percentile_doy(tas_hist,window=5,per=90.0)
#tas90p_rcp45 = xclim.core.calendar.percentile_doy(tas_rcp45,window=5,per=90.0)
#tas90p_rcp85 = xclim.core.calendar.percentile_doy(tas_rcp85,window=5,per=90.0)

tn10p_hist = xclim.core.calendar.percentile_doy(tn_hist,window=5,per=10.0)
#tn90p_hist = xclim.core.calendar.percentile_doy(tn_hist,window=5,per=90.0)
#tx10p_hist = xclim.core.calendar.percentile_doy(tx_hist,window=5,per=10.0)
tx90p_hist = xclim.core.calendar.percentile_doy(tx_hist,window=5,per=90.0)
#pr75p_hist = xclim.core.calendar.percentile_doy(pr_hist,window=29,per=75.0)

#tn10p_rcp45 = xclim.core.calendar.percentile_doy(tn_rcp45,window=5,per=10.0)
#tn90p_rcp45 = xclim.core.calendar.percentile_doy(tn_rcp45,window=5,per=90.0)
#tx10p_rcp45 = xclim.core.calendar.percentile_doy(tx_rcp45,window=5,per=10.0)
#tx90p_rcp45 = xclim.core.calendar.percentile_doy(tx_rcp45,window=5,per=90.0)
#pr75p_rcp45 = xclim.core.calendar.percentile_doy(pr_rcp45,window=29,per=75.0)

#tn10p_rcp85 = xclim.core.calendar.percentile_doy(tn_rcp85,window=5,per=10.0)
#tn90p_rcp85 = xclim.core.calendar.percentile_doy(tn_rcp85,window=5,per=90.0)
#tx10p_rcp85 = xclim.core.calendar.percentile_doy(tx_rcp85,window=5,per=10.0)
#tx90p_rcp85 = xclim.core.calendar.percentile_doy(tx_rcp85,window=5,per=90.0)
#pr75p_rcp85 = xclim.core.calendar.percentile_doy(pr_rcp85,window=29,per=75.0)

#tn10p_hist.to_netcdf('~/space/CanESM2-WRF/climdex/tn10p_hist.nc')
#tn90p_hist.to_netcdf('~/space/CanESM2-WRF/climdex/tn90p_hist.nc')
#tx10p_hist.to_netcdf('~/space/CanESM2-WRF/climdex/tx10p_hist.nc')
#tx90p_hist.to_netcdf('~/space/CanESM2-WRF/climdex/tx90p_hist.nc')
##pr75p_hist.to_netcdf('~/space/CanESM2-WRF/climdex/pr75p_hist.nc')

#tn10p_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/tn10p_rcp45.nc')
#tn90p_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/tn90p_rcp45.nc')
#tx10p_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/tx10p_rcp45.nc')
#tx90p_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/tx90p_rcp45.nc')
#pr75p_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/pr75p_rcp45.nc')

#tn10p_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/tn10p_rcp85.nc')
#tn90p_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/tn90p_rcp85.nc')
#tx10p_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/tx10p_rcp85.nc')
#tx90p_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/tx90p_rcp85.nc')
#pr75p_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/pr75p_rcp85.nc')

#tas10p_hist.to_netcdf('~/space/CanESM2-WRF/climdex/tas10p_hist.nc')
#tas90p_hist.to_netcdf('~/space/CanESM2-WRF/climdex/tas90p_hist.nc')

#tas10p_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/tas10p_rcp45.nc')
#tas90p_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/tas90p_rcp45.nc')

#tas10p_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/tas10p_rcp85.nc')
#tas90p_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/tas90p_rcp85.nc')


#csdi_hist = xclim.indicators.atmos.cold_spell_duration_index(tasmin=tn_hist,tasmin_per=tn10p_hist,window=6,freq='MS')
csdi_rcp45 = xclim.indicators.atmos.cold_spell_duration_index(tasmin=tn_rcp45,tasmin_per=tn10p_hist,window=6,freq='MS')
csdi_rcp85 = xclim.indicators.atmos.cold_spell_duration_index(tasmin=tn_rcp85,tasmin_per=tn10p_hist,window=6,freq='MS')


#csdi_hist.to_netcdf('~/space/CanESM2-WRF/climdex/csdi_hist_mon.nc')
csdi_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/csdi_rcp45_mon_base_hist.nc')
csdi_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/csdi_rcp85_mon_base_hist.nc')

#cdd = xclim.indicators.atmos.cooling_degree_days(tas=tas_rcp85,thresh='18.0 degC')
#dlyfrzthw = xclim.indicators.atmos.daily_freezethaw_cycles(tasmin=tn_rcp85,tasmax=tx_rcp85,thresh_tasmin='0 degC',thresh_tasmax='0 degC')
#sdii = xclim.indicators.atmos.daily_pr_intensity(pr=pr_rcp85,thresh='1 mm/day')
#freshet_start = xclim.indicators.atmos.freshet_start(tas=tas_hist,thresh='0 degC')

#ds_freq_hist = xclim.indicators.atmos.dry_spell_frequency(pr=pr_hist, thresh='1.0 mm', window=3,freq='MS')
#ds_freq_rcp45 = xclim.indicators.atmos.dry_spell_frequency(pr=pr_rcp45, thresh='1.0 mm', window=3,freq='MS')
#ds_freq_rcp85 = xclim.indicators.atmos.dry_spell_frequency(pr=pr_rcp85, thresh='1.0 mm', window=3,freq='MS')


#ds_freq_hist.to_netcdf('~/space/CanESM2-WRF/climdex/dsfreq_hist_mon.nc')
#ds_freq_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/dsfreq_rcp45_mon.nc')
#ds_freq_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/dsfreq_rcp85_mon.nc')

#ws_freq_hist = xclim.indicators.atmos.wet_spell_frequency(pr=pr_hist, thresh='1.0 mm', window=3,freq='MS')
#ws_freq_rcp45 = xclim.indicators.atmos.wet_spell_frequency(pr=pr_rcp45, thresh='1.0 mm', window=3,freq='MS')
#ws_freq_rcp85 = xclim.indicators.atmos.wet_spell_frequency(pr=pr_rcp85, thresh='1.0 mm', window=3,freq='MS')


#ws_freq_hist.to_netcdf('~/space/CanESM2-WRF/climdex/wsfreq_hist_mon.nc')
#ws_freq_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/wsfreq_rcp45_mon.nc')
#ws_freq_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/wsfreq_rcp85_mon.nc')


#wd_hist = xclim.indicators.atmos.wetdays(pr=pr_hist, thresh='1.0 mm', freq='MS')
#wd_rcp45 = xclim.indicators.atmos.wetdays(pr=pr_rcp45, thresh='1.0 mm', freq='MS')
#wd_rcp85 = xclim.indicators.atmos.wetdays(pr=pr_rcp85, thresh='1.0 mm', freq='MS')


#wd_hist.to_netcdf('~/space/CanESM2-WRF/climdex/wd_hist_mon.nc')
#wd_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/wd_rcp45_mon.nc')
#wd_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/wd_rcp85_mon.nc')


#ds_tot = xclim.indicators.atmos.dry_spell_total_length(pr=pr_rcp85, thresh='1.0 mm', window=3)
#fd = xclim.indicators.atmos.frost_days(tasmin=tn_rcp85, thresh='0 degC')
#gsl_45 = xclim.indicators.atmos.growing_season_length(tas=tas_rcp45, thresh='5.0 degC', window=6)
#gsl_85 = xclim.indicators.atmos.growing_season_length(tas=tas_rcp85, thresh='5.0 degC', window=6)

#hwfreq_hist = xclim.indicators.atmos.heat_wave_frequency(tasmin=tn_hist, tasmax=tx_hist, thresh_tasmin='22.0 degC', thresh_tasmax='30 degC', window=3)
#hwfreq_rcp45 = xclim.indicators.atmos.heat_wave_frequency(tasmin=tn_rcp45, tasmax=tx_rcp45, thresh_tasmin='22.0 degC', thresh_tasmax='30 degC', window=3)
#hwfreq_rcp85 = xclim.indicators.atmos.heat_wave_frequency(tasmin=tn_rcp45, tasmax=tx_rcp85, thresh_tasmin='22.0 degC', thresh_tasmax='30 degC', window=3)

#hwi_hist = xclim.indicators.atmos.heat_wave_index(tasmax=tx_hist, thresh='25.0 degC', window=5)
#hwi_rcp45 = xclim.indicators.atmos.heat_wave_index(tasmax=tx_rcp45, thresh='25.0 degC', window=5)
#hwi_rcp85 = xclim.indicators.atmos.heat_wave_index(tasmax=tx_rcp85, thresh='25.0 degC', window=5)

#hwtotlen_hist = xclim.indicators.atmos.heat_wave_total_length(tasmin=tn_hist, tasmax=tx_hist, thresh_tasmin='22.0 degC', thresh_tasmax='30 degC', window=3)
#hwtotlen_rcp45 = xclim.indicators.atmos.heat_wave_total_length(tasmin=tn_rcp45, tasmax=tx_rcp45, thresh_tasmin='22.0 degC', thresh_tasmax='30 degC', window=3)
#hwtotlen_rcp85 = xclim.indicators.atmos.heat_wave_total_length(tasmin=tn_rcp45, tasmax=tx_rcp85, thresh_tasmin='22.0 degC', thresh_tasmax='30 degC', window=3)

#hdd_hist = xclim.indicators.atmos.heating_degree_days(tas=tas_hist,thresh='17.0 degC')
#hdd_rcp45 = xclim.indicators.atmos.heating_degree_days(tas=tas_rcp45,thresh='17.0 degC')
#hdd_rcp85 = xclim.indicators.atmos.heating_degree_days(tas=tas_rcp85,thresh='17.0 degC')

#id_hist = xclim.indicators.atmos.ice_days(tasmax=tx_hist,thresh='0 degC')
#id_rcp45 = xclim.indicators.atmos.ice_days(tasmax=tx_rcp45,thresh='0 degC')
#id_rcp85 = xclim.indicators.atmos.ice_days(tasmax=tx_rcp85,thresh='0 degC')

#wsdi_hist = xclim.indicators.atmos.warm_spell_duration_index(tasmax=tx_hist,tasmax_per=tx90p_hist,window=6,freq='MS')
#wsdi_rcp45_basehist = xclim.indicators.atmos.warm_spell_duration_index(tasmax=tx_rcp45,tasmax_per=tx90p_hist,window=6,freq='MS')
#wsdi_rcp85_basehist = xclim.indicators.atmos.warm_spell_duration_index(tasmax=tx_rcp85,tasmax_per=tx90p_hist,window=6,freq='MS')
wsdi_rcp45 = xclim.indicators.atmos.warm_spell_duration_index(tasmax=tx_rcp45,tasmax_per=tx90p_hist,window=6,freq='MS')
wsdi_rcp85 = xclim.indicators.atmos.warm_spell_duration_index(tasmax=tx_rcp85,tasmax_per=tx90p_hist,window=6,freq='MS')



#wsdi_hist.to_netcdf('~/space/CanESM2-WRF/climdex/wsdi_hist_mon_base_hist.nc')
wsdi_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/wsdi_rcp45_mon_base_hist.nc')
wsdi_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/wsdi_rcp85_mon_base_hist.nc')


#warmwet_hist = xclim.indicators.atmos.warm_and_wet_days(tas=tas_hist, pr=pr_hist, tas_per=tas90p_hist, pr_per=pr75p_hist)
#warmwet_rcp45 = xclim.indicators.atmos.warm_and_wet_days(tas=tas_rcp45, pr=pr_rcp45, tas_per=tas90p_rcp45, pr_per=pr75p_rcp45)
#warmwet_rcp85 = xclim.indicators.atmos.warm_and_wet_days(tas=tas_rcp85, pr=pr_rcp85, tas_per=tas90p_rcp85, pr_per=pr75p_rcp85)
#warmwet_rcp45_basehist = xclim.indicators.atmos.warm_and_wet_days(tas=tas_rcp45, pr=pr_rcp45, tas_per=tas90p_hist, pr_per=pr75p_hist)
#warmwet_rcp85_basehist = xclim.indicators.atmos.warm_and_wet_days(tas=tas_rcp85, pr=pr_rcp85, tas_per=tas90p_hist, pr_per=pr75p_hist)




#csdi.to_netcdf('~/space/CanESM2-WRF/climdex/csdi_rcp85_base_hist.nc')
#cdd.to_netcdf('~/space/CanESM2-WRF/climdex/cdd_rcp85.nc')
#dlyfrzthw.to_netcdf('~/space/CanESM2-WRF/climdex/dlyfrzthw_rcp85.nc')
#sdii.to_netcdf('~/space/CanESM2-WRF/climdex/sdii_rcp85.nc')
#freshet_start.to_netcdf('~/space/CanESM2-WRF/climdex/freshet_hist2.nc')

#ds_freq.to_netcdf('~/space/CanESM2-WRF/climdex/dsfreq_rcp85.nc')
#ds_tot.to_netcdf('~/space/CanESM2-WRF/climdex/dstot_rcp85.nc')
#fd.to_netcdf('~/space/CanESM2-WRF/climdex/fd_rcp85.nc')
#gsl_45.to_netcdf('~/space/CanESM2-WRF/climdex/gsl_rcp45.nc')
#gsl_85.to_netcdf('~/space/CanESM2-WRF/climdex/gsl_rcp85.nc')

#hwfreq_hist.to_netcdf('~/space/CanESM2-WRF/climdex/hwfreq_hist.nc')
#hwfreq_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/hwfreq_rcp45.nc')
#hwfreq_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/hwfreq_rcp85.nc')

#hwi_hist.to_netcdf('~/space/CanESM2-WRF/climdex/hwi_hist.nc')
#hwi_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/hwi_rcp45.nc')
#hwi_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/hwi_rcp85.nc')

#hwtotlen_hist.to_netcdf('~/space/CanESM2-WRF/climdex/hwtot_hist.nc')
#hwtotlen_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/hwtot_rcp45.nc')
#hwtotlen_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/hwtot_rcp85.nc')

#hdd_hist.to_netcdf('~/space/CanESM2-WRF/climdex/hdd_hist.nc')
#hdd_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/hdd_rcp45.nc')
#hdd_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/hdd_rcp85.nc')

#id_hist.to_netcdf('~/space/CanESM2-WRF/climdex/id_hist.nc')
#id_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/id_rcp45.nc')
#id_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/id_rcp85.nc')


#wsdi_hist.to_netcdf('~/space/CanESM2-WRF/climdex/wsdi_hist.nc')
#wsdi_rcp45_basehist.to_netcdf('~/space/CanESM2-WRF/climdex/wsdi_rcp45_base_hist.nc')
#wsdi_rcp85_basehist.to_netcdf('~/space/CanESM2-WRF/climdex/wsdi_rcp85_base_hist.nc')
#wsdi_rcp45_basercp.to_netcdf('~/space/CanESM2-WRF/climdex/wsdi_rcp45_base_rcp45.nc')
#wsdi_rcp85_basercp.to_netcdf('~/space/CanESM2-WRF/climdex/wsdi_rcp85_base_rcp85.nc')


#warmwet_hist.to_netcdf('~/space/CanESM2-WRF/climdex/warmwet_hist.nc')
#warmwet_rcp45_basehist.to_netcdf('~/space/CanESM2-WRF/climdex/warmwet_rcp45_base_hist.nc')
#warmwet_rcp85_basehist.to_netcdf('~/space/CanESM2-WRF/climdex/warmwet_rcp85_base_hist.nc')
#warmwet_rcp45.to_netcdf('~/space/CanESM2-WRF/climdex/warmwet_rcp45_base_rcp45.nc')
#warmwet_rcp85.to_netcdf('~/space/CanESM2-WRF/climdex/warmwet_rcp85_base_rcp85.nc')





