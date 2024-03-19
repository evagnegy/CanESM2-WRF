

Station_ID=202

echo $Station_ID
for year in {1986..2005}
do 
	echo $Station_ID
	wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=${Station_ID}&Year=${year}&timeframe=2&submit= Download+Data" -P /Users/evagnegy/Desktop/test/${Station_ID}/${month}

done


