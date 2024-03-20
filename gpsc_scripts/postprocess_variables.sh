
verif_dir=/home/spfm000/space/CanESM2-WRF/historical/variables_complete/

var="wind"
d="d02"

cd ${verif_dir}

#for year in {2046..2065}
for year in {1986..2005}
do
	cdo mergetime ${year}_1-4_${var}_${d}.nc ${year}_5-8_${var}_${d}.nc ${year}_9-12_${var}_${d}.nc ${year}_${var}_${d}.nc 		
done
        
cdo mergetime 1986_${var}_${d}.nc 1987_${var}_${d}.nc 1988_${var}_${d}.nc 1989_${var}_${d}.nc 1990_${var}_${d}.nc 1991_${var}_${d}.nc 1992_${var}_${d}.nc 1993_${var}_${d}.nc 1994_${var}_${d}.nc 1995_${var}_${d}.nc 1996_${var}_${d}.nc 1997_${var}_${d}.nc 1998_${var}_${d}.nc 1999_${var}_${d}.nc 2000_${var}_${d}.nc 2001_${var}_${d}.nc 2002_${var}_${d}.nc 2003_${var}_${d}.nc 2004_${var}_${d}.nc 2005_${var}_${d}.nc ${var}_${d}.nc

#cdo mergetime 2046_${var}_d03.nc 2047_${var}_d03.nc 2048_${var}_d03.nc 2049_${var}_d03.nc 2050_${var}_d03.nc 2051_${var}_d03.nc 2052_${var}_d03.nc 2053_${var}_d03.nc 2054_${var}_d03.nc 2055_${var}_d03.nc 2056_${var}_d03.nc 2057_${var}_d03.nc 2058_${var}_d03.nc 2059_${var}_d03.nc 2060_${var}_d03.nc 2061_${var}_d03.nc 2062_${var}_d03.nc 2063_${var}_d03.nc 2064_${var}_d03.nc 2065_${var}_d03.nc ${var}_d03.nc

	#rm ${new_dir}/${station}/19*.nc
	#rm ${new_dir}/${station}/20*.nc
#done
