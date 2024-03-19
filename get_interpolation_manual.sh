#!/bin/bash -l
 
var="pr"
WRFPATH=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_PCIC/
#station_list=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations_wind.csv
station_list=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/ECCC_d03_stations.csv


WRF_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_PCIC/pr_day_CanESM2_historical_rcp45.nc

#while IFS=, read -r stationID Name lats lons el;
while IFS=, read -r indx loc prov id1 stationID id2 id3 lat lon id4 id5 elev year1 year2 year3 year4 year5 year6 yeaer7 year8;

do
        echo $stationID
 		#station_file=NOAA_${stationID}.txt
 		station_file=ECCC_${stationID}.txt


		station_desc_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/CDO_station_files/$station_file
 
		#weights=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/weight_files/CanRCM4/weights_NOAA_${stationID}.nc
        weights=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/weight_files/CanESM2_PCIC/weights_ECCC_${stationID}.nc

 		#output_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanRCM4/historical/${var}_NOAA_${stationID}.nc
 		output_file=/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_model_data/CanESM2_PCIC/historical/${var}_ECCC_${stationID}.nc


		if [ -f "$output_file" ]; then
			echo "$output_file already exists."
		else	
 			cdo -f nc -b F64 remap,${station_desc_file},${weights} ${WRF_file} ${output_file}
		fi

done < "$station_list"

echo
echo 'Verification complete!'
echo
#done
