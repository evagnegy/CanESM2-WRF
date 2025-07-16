#!/bin/bash


#DIRECTORY="/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/daily" 
DIRECTORY='/Volumes/EVA/gridded_model_data/'

for file in "$DIRECTORY"/*.nc; do
    if [[ -f "$file" ]]; then
        ncrename -d lat,lats "$file"
        ncrename -d lon,lons "$file"
	echo "Renamed lat and lons in $(basename "$file")"
    fi
done

echo "All files processed."
