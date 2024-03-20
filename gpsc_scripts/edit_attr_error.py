
import netCDF4 as nc

def copy_netcdf_with_attributes(input_file, output_file, global_attrs=None, var_attrs=None):
    # Open the original netCDF file
    with nc.Dataset(input_file, 'r') as original_nc:
        # Create a new netCDF file
        with nc.Dataset(output_file, 'w') as new_nc:
            # Copy global attributes
            #if global_attrs:
            #    for attr_name, attr_value in global_attrs.items():
            #        setattr(new_nc, attr_name, attr_value)
            #else:
            for attr_name in original_nc.ncattrs():
                setattr(new_nc, attr_name, getattr(original_nc, attr_name))

            # Copy dimensions
            for dim_name, dim in original_nc.dimensions.items():
                new_nc.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)

            # Copy variables and their attributes
            for var_name, var in original_nc.variables.items():
                new_var = new_nc.createVariable(var_name, var.dtype, var.dimensions)
                new_var[:] = var[:]
                for attr_name in var.ncattrs():
                    setattr(new_var, attr_name, getattr(var, attr_name))
                # Add or update variable attributes if provided
                if var_attrs and var_name in var_attrs:
                    for attr_name, attr_value in var_attrs[var_name].items():
                        setattr(new_var, attr_name, attr_value)

if __name__ == "__main__":
    # Input and output file paths
    input_file_path = '/home/spfm000/space/CanESM2-WRF/final_netcdfs/pr_daily_d03_rcp45.nc'
    output_file_path = '/home/spfm000/space/CanESM2-WRF/final_netcdfs/pr_rcp45.nc'

    # Define global attributes to add
    global_attributes = {
        'description': 'CanESM2-WRF 3-km RCP 8.5 output (2046-2065)',
    
        'institution':'Fisheries and Oceans Canada',
        'contact':'egnegy@eoas.ubc.ca',
        'history':'Created in March 2024',
        'frequency':'daily',
        'map_proj':'lambert',
        'ref_lat':'49.00',
        'ref_lon':'-130.90',
        'truelat1':'49.0',
        'truelat2':'49.0',
        'stand_lon':'-77.0'

    }

    # Define variable attributes to add or update
    variable_attributes = {
        'pr': {
            'long_name': 'Precipitation',
            'units': 'kg m-2 d-1',
            'standard_name': 'precipitation_flux'

        }
        # Add more variable names and attributes as needed
    }

    # Call the function to copy netCDF file with attributes
    copy_netcdf_with_attributes(input_file_path, output_file_path, global_attrs=global_attributes, var_attrs=variable_attributes)
    print("NetCDF file copied with attributes successfully!")

