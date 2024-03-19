

station_file="/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/NOAA_d03_stations.csv"

while IFS=, read -r Station_ID Name lat lon elev;
do

	echo $Station_ID
	for year in {1986..2005}
	do 
		for month in {1..12}
		do
			echo $Station_ID
    		wget --content-disposition "https://www.ncei.noaa.gov/access/services/data/v1?dataset=hourly-summaries&stations=${Station_ID}&startDate=1986-01-01&endDate=2005-12-31&dataTypes=&format=csv&includeStationName=true&includeStationLocation=true&units=metric" -P /Users/evagnegy/Desktop/test_noaa/${Station_ID}/${month}
		done
	done

done < "$station_file"
