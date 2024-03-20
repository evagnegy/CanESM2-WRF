
#!/bin/sh

domain=3
station_list="/gpfs/fs7/dfo/hpcmc/pfm/spfm000/CanESM2-WRF/obs/ECCC_d03_stations_hourly.csv"

#while IFS=, read -r stationID Name lats lons el;
while IFS=, read -r name prov id stationID id2 id3 lat lon id4 id5 elev year1 year2 year3 year4 year5 year6 year7 year8;  
do
	echo $stationID
	WRF_file=/gpfs/fs7/dfo/hpcmc/pfm/spfm000/CanESM2-WRF/models_for_weights/CanESM2-D0${domain}.nc
	
	#station_desc_file=/gpfs/fs7/dfo/hpcmc/pfm/spfm000/CanESM2-WRF/station_files/NOAA_${stationID}.txt
	station_desc_file=/home/spfm000/evg000/gpfs7/verification/station_files/st_${stationID}.txt 

	#place you want them to save
	output_weights_WRF=/gpfs/fs7/dfo/hpcmc/pfm/spfm000/CanESM2-WRF/weight_files/CanESM2-WRF-D0${domain}/weights_ECCC_${stationID}.nc
	

	cdo -f nc genbic,${station_desc_file} ${WRF_file} ${output_weights_WRF}

done < "$station_list"

