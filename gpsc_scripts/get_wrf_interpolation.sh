#!/bin/bash -l
 
var=$1
domain=$2
WRFPATH=$3
station_list=$4

echo $var
echo $domain

#for year in {1986..1990}
#do

	WRF_file=${WRFPATH}/variables_complete/${var}_d0${domain}.nc

	while IFS=, read -r stationID Name lats lons el;
	#while IFS=, read -r name prov id stationID id2 id3 lat lon id4 id5 elev year1 year2 year3 year4 year5 year6 year7 year8;
	do
        	echo $stationID
 		station_file=NOAA_${stationID}.txt
		#station_file=ECCC_${stationID}.txt

		station_desc_file=/gpfs/fs7/dfo/hpcmc/pfm/spfm000/CanESM2-WRF/station_files/$station_file
		#station_desc_file=/home/spfm000/evg000/gpfs7/verification/station_files/st_${stationID}.txt  
		
		weights_WRF=/gpfs/fs7/dfo/hpcmc/pfm/spfm000/CanESM2-WRF/weight_files/CanESM2-WRF-D0${domain}/weights_${stationID}.nc
		#weights_WRF=/gpfs/fs7/dfo/hpcmc/pfm/spfm000/CanESM2-WRF/weight_files/CanESM2-WRF-D0${domain}/weights_${stationID}.nc 

 		output_file=/gpfs/fs7/dfo/hpcmc/pfm/spfm000/CanESM2-WRF/rcp45/variables_stations/${var}_d0${domain}_NOAA_${stationID}.nc
		echo $output_file


		if [ -f "$output_file" ]; then
			echo "$output_file already exists."
		else	
 			cdo -f nc remap,${station_desc_file},${weights_WRF} ${WRF_file} ${output_file}
		fi

	done < "$station_list"

	echo
	echo 'Verification complete!'
	echo
#done
