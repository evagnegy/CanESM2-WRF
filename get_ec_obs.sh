

station_file="/Users/evagnegy/Desktop/CanESM2_WRF_Eval/station_metadata_lists/eccc_hourly_test.csv"

while IFS=, read -r Name Province Climate_ID Station_ID WMO_ID TC_ID lats_deg lons_deg lats lons elevation first_year last_year first_year_hourly last_year_hourly first_year_daily last_year_daily first_year_monthly second_year_monthly;
do

	echo $Station_ID
	for year in {1986..2005}
	do 
		for month in {1..12}
		do
			echo $Station_ID
    		wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=${Station_ID}&Year=${year}&Month=${month}&time=UTC&timeframe=1&submit= Download+Data" -P /Users/evagnegy/Desktop/test/${Station_ID}/${month}
		done
	done

done < "$station_file"
