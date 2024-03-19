 #!/bin/bash -l

station_list=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv
#station_list=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations_hourly.csv
#station_list=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations_wind.csv

agency='ECCC'
model='CanESM2_PCIC'

#grid_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/wspd_day_CanESM2_historical.nc
#grid_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_raw/tas_day_CanESM2_historical_r1i1p1_18500101-20051231.nc
grid_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_PCIC/tasmin_day_CanESM2_historical_rcp45.nc
#grid_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/${model}/pr_NAM-22.nc
#grid_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/${model}/sfcWind-22.nc


while IFS=, read -r indx loc prov id1 StationID id2 id3 lat lon id4 id5 elev year1 year2 year3 year4 year5 year6 yeaer7 year8;
#while IFS=, read -r StationID Name lats lons el;
do
    echo "on $StationID"

    #files with lat/lon of each station
    station_file=${agency}_${StationID}.txt
    station_desc_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/CDO_station_files/$station_file

    #place you want them to save
    weights_WRF=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/weight_files/${model}/weights_${agency}_${StationID}.nc

    cdo -f nc genbic,${station_desc_file} ${grid_file} ${weights_WRF}


done < "$station_list"

echo
echo 'Verification complete!'
echo
