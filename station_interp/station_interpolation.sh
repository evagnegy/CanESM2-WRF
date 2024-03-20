#!/bin/bash -l
 
# bash script to get the weights for remapping NAM and GFS grb2 output to a list of stations. 
#It only needs ran once, and it can use any hour file from each IBC. More IBCs will have to be added
 
 
var='wind' 
model='CanRCM4'
agency="ECCC"

station_list=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations_hourly.csv
#station_list=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/BCH_d03_stations.csv

run='historical' #historical or rcp45 or rcp85

if [ $run == 'historical' ]; then
	#dates='18500101-20051231'
	dates='19790101-20051231'
else
	dates='20060101-21001231'
fi


#model_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/${model}/tasmax_day_CanESM2_${run}_r1i1p1_${dates}.nc
model_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/${model}/sfcWind_NAM-22.nc
#model_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/${model}/tasmax_day_CanESM2_historical_rcp45.nc
#model_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/${model}/wspd_day_CanESM2_historical.nc


#while IFS=, read -r Index Name Prov climID stationID wmoID tcID lats_d lons_d lats lons el year1 year2 year1h year2h year1d year2d year1m year2m;
while IFS=, read -r Station_name Elev X Y StationID flag;
do
    echo $StationID
		
		station_file=${agency}_${StationID}.txt

	station_desc_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/CDO_station_files/$station_file


	weights_WRF=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/weight_files/${model}/weights_${agency}_${StationID}_wind.nc
            
	output_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/${model}/${run}/${var}_${agency}_${StationID}.nc


	if [ -f "$output_file" ]; then
		echo "$output_file already exists."
	else	
			cdo -b F64 -f nc remap,${station_desc_file},${weights_WRF} ${model_file} ${output_file}
	fi

done < "$station_list"

echo
echo 'Verification complete!'
echo

