

from netCDF4 import Dataset
import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import WRFDomainLib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl
import matplotlib.patches as mpatches

#%%
t_file_hist = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/t_d03_mean_hist.nc"
pr_file_hist = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/pr_d03_mean_hist.nc"
wind_file_hist = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/wind_d03_mean_hist_wspd.nc"


t_file_rcp45 = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/t_d03_mean_rcp45.nc"
pr_file_rcp45 = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/pr_d03_mean_rcp45.nc"
wind_file_rcp45 = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/wind_d03_mean_rcp45_wspd.nc"

t_file_rcp85 = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/t_d03_mean_rcp85.nc"
pr_file_rcp85 = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/pr_d03_mean_rcp85.nc"
wind_file_rcp85 = "/Users/evagnegy/Desktop/CanESM2_WRF_Eval/gridded_model_data/means/wind_d03_mean_rcp85_wspd.nc"



geo_em_d03_file = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/geo_em.d03.nc'

t_ds = xr.open_dataset(t_file_hist,decode_times=False)
pr_ds = xr.open_dataset(pr_file_hist,decode_times=False)
wind_ds = xr.open_dataset(wind_file_hist,decode_times=False)

t_ds_rcp45 = xr.open_dataset(t_file_rcp45,decode_times=False)
pr_ds_rcp45 = xr.open_dataset(pr_file_rcp45,decode_times=False)
wind_ds_rcp45 = xr.open_dataset(wind_file_rcp45,decode_times=False)

t_ds_rcp85 = xr.open_dataset(t_file_rcp85,decode_times=False)
pr_ds_rcp85 = xr.open_dataset(pr_file_rcp85,decode_times=False)
wind_ds_rcp85 = xr.open_dataset(wind_file_rcp85,decode_times=False)

geo_em_d03_nc = Dataset(geo_em_d03_file, mode='r')
land_d03 = np.squeeze(geo_em_d03_nc.variables['LANDMASK'][:])
mask_d03 = xr.DataArray(land_d03,dims=('x','y'))

t_hist = np.squeeze(t_ds['T2'])-273.15
pr_hist = np.squeeze(pr_ds['pr'][:])
wind_hist = np.squeeze(wind_ds['wspd'])

t_rcp45 = np.squeeze(t_ds_rcp45['T2'])-273.15
pr_rcp45 = np.squeeze(pr_ds_rcp45['pr'][:])
wind_rcp45 = np.squeeze(wind_ds_rcp45['wspd'])

t_rcp85 = np.squeeze(t_ds_rcp85['T2'])-273.15
pr_rcp85 = np.squeeze(pr_ds_rcp85['pr'][:])
wind_rcp85 = np.squeeze(wind_ds_rcp85['wspd'])

#t_hist = t_hist.where(mask_d03)
#pr_hist = pr_hist.where(mask_d03)
#wind_hist = wind_hist.where(mask_d03)

lats = np.squeeze(Dataset(t_file_hist, mode='r').variables['lat'][:])
lons = np.squeeze(Dataset(t_file_hist, mode='r').variables['lon'][:])
#%%
variables = ['t','pr','wspd']
#ds_clim = xr.Dataset({'t': t_hist, 'pr': pr_hist, 'wspd': wind_hist})
#ds_clim = xr.merge([xr.Dataset({'t': t_rcp45}),xr.Dataset({'pr': pr_rcp45}),xr.Dataset({'wspd': wind_rcp45})], compat="override")
ds_clim = xr.merge([xr.Dataset({'t': t_rcp85}),xr.Dataset({'pr': pr_rcp85}),xr.Dataset({'wspd': wind_rcp85})], compat="override")

#variables = ['t','pr']
#ds_clim = xr.Dataset({'t': t_hist, 'pr': pr_hist})
#ds_clim = xr.merge([xr.Dataset({'t': t_rcp45}),xr.Dataset({'pr': pr_rcp45})], compat="override")
#ds_clim = xr.merge([xr.Dataset({'t': t_rcp85}),xr.Dataset({'pr': pr_rcp85})], compat="override")

#variables = ['pr','wspd']
#ds_clim = xr.Dataset({'pr': pr_hist, 'wspd': wind_hist})


for n_clusters in range(4,5): 


    colors = [plt.cm.tab10(i) for i in range(n_clusters)]
    
    # Convert to feature array
    features = np.array([ds_clim[v].to_numpy().flatten() for v in variables])
    
    # Mask the feature array
    mask_vec = mask_d03.to_numpy().flatten() > 0.
    features_masked = features[:,mask_vec]
    mask_masked     = mask_vec[mask_vec]
    features_masked = features_masked.transpose()
            
    # Scale the features
    scaler = StandardScaler()
    scaler.fit(features_masked)
    scaled_features = scaler.transform(features_masked)
    
# =============================================================================
#     # Perform the clustering
#     kmeans_region = KMeans(n_clusters = n_clusters, n_init = 10)
#     kmeans_region.fit(scaled_features)
#     labels = kmeans_region.labels_
#     cluster_id = np.zeros(features.shape[1])*np.nan
#     cluster_id[mask_vec] = labels
#     cluster_id = cluster_id.reshape(ds_clim[variables[0]].shape)
#     ds_clim['cluster'] = (('y','x'),cluster_id)
#     
#     # Clusters permutation
#     d_permut = dict(zip(np.array([-float(ds_clim.lat.where(ds_clim.cluster==i).min()) for i in range(n_clusters)]).argsort(),range(n_clusters))) 
#     d_permut_1 = {d_permut[k]:k for k in d_permut}
#     f_permut = lambda x: d_permut[x] if x in d_permut else np.nan
#     f_permut_1 = lambda x: d_permut_1[x] if x in d_permut_1 else np.nan
#     # f_permut,f_permut_1 = get_f_permut(clusters_loc,ds_clim)
#     # print(", ".join([f"{i}:{f_permut(i)}" for i in range(n_clusters)]))
#     ds_clim['clstr'] = (('y','x'),np.vectorize(f_permut)(ds_clim.cluster))
#     centers = scaler.inverse_transform(kmeans_region.cluster_centers_)
#     centers = centers[[f_permut_1(i) for i in range(len(centers))]]
#     labels = [f_permut(l) for l in labels]
#     
#     ref_centers = centers
# =============================================================================
    
    
    centers = ref_centers

    shp = (len(scaled_features),len(centers))
    dist_centers = np.zeros(shp)*np.nan
    for nc,ctr in enumerate(scaler.transform(centers)):
        # print(ctr)
        dist_centers[:,nc] = np.linalg.norm(scaled_features-ctr,axis=1)
    
    labels = dist_centers.argmin(axis=1)

    cluster_id = np.zeros(features.shape[1])*np.nan
    cluster_id[mask_vec] = labels
    cluster_id = cluster_id.reshape(ds_clim[variables[0]].shape)
    ds_clim['clstr'] = (('y','x'),cluster_id)


    
    
    
    
    
    
    handles = []
    
    WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)
    
    fig1 = plt.figure(figsize=(10, 10),dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)
    
    for i in range(n_clusters):
        ax1.contourf(lons, lats, ds_clim.clstr.where(ds_clim.clstr==i), transform=ccrs.PlateCarree(),
        zorder=0,vmin=0,vmax=n_clusters,colors=[colors[i]])
    
        center = centers[i,:].reshape(1,-1)[0]
    
        label = f"{i}: " + ", ".join([f"{v}:{center[k]:.1f}" for k,v in enumerate(variables)])
        handles.append(mpatches.Patch(color=colors[i], label=label))
        
    #ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
    ax1.add_feature(cf.BORDERS,linewidth=0.5)
    ax1.add_feature(cf.STATES,linewidth=0.5)
    
    # d03 box
    corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
    random_y_factor = -corner_y3[0]/12.5
    random_x_factor = corner_x3[0]/65
    
    ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
    ax1.text(-3680871, 700000, 'D03', va='top', ha='left',fontweight='bold', size=15, color='red', zorder=2)
    
    ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())
    
    plt.legend(handles=handles, loc="lower right")
    


    n_vars = len(variables)
    cols = [colors[l] for l in labels]
    ntot = int(n_vars*(n_vars-1)/2)
    ncols = ntot if ntot<5 else 5
    nrows = - int(-ntot // ncols)
    nrows = nrows if nrows>1 else 1
    cr,cc = (nrows>1), (ncols>1)

    if len(variables)==2:
        fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10,10)) #,sharex=True,sharey=True)
    else:
        fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(18,6)) #,sharex=True,sharey=True)

    k=0
    for i,v1 in enumerate(variables):
        for j,v2 in enumerate(variables):
            if i<j:
                i0 = k%nrows
                j0 = k%ncols
                ax = axs[i0,j0] if (cr and cc) else (axs[i0] if cr else (axs[j0] if cc else axs))
                # axs[i0,j0].scatter(scaled_features[:,i], scaled_features[:,j],c=cols,s=1,alpha=0.2)
                ax.scatter(features_masked[:,i], features_masked[:,j],c=cols,s=1,alpha=0.2)
                ax.set_title(f"x={v1}, y={v2}")
                ax.grid()
                k += 1
                for c in range(n_clusters):
                    ax.scatter(centers[c,i],centers[c,j],c=[colors[c]],s=70,alpha=1,edgecolor='k')
                    #if (ys!=ref_period) and plot_mean_centers:
                    #    m1,m2 = float(ds_clim[v1].where(ds_clim.clstr==c).mean()), float(ds_clim[v2].where(ds_clim.clstr==c).mean())
                    #    ax.scatter([m1],[m2],c=[colors[c]],marker='x',s=500,alpha=1)
                
                ax.set_xlim([-10,15])
                ax.set_ylim([-0.5,20])
                
                
#%%

WPSFile = '/Users/evagnegy/Desktop/CanESM2_WRF_Eval/domain/namelist.wps.txt'
wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = WRFDomainLib.calc_wps_domain_info(WPSFile)

fig1 = plt.figure(figsize=(10, 10),dpi=200)
ax1 = fig1.add_subplot(1, 1, 1, projection=wpsproj)


ax1.pcolormesh(lons, lats, t_hist, transform=ccrs.PlateCarree(),
    zorder=0,vmin=0,vmax=n_clusters,cmap='jet')

#ax1.add_feature(cf.OCEAN, edgecolor='face', facecolor='lightblue', zorder=1)
ax1.add_feature(cf.BORDERS,linewidth=0.5)
ax1.add_feature(cf.STATES,linewidth=0.5)

# d03 box
corner_x3, corner_y3 = WRFDomainLib.reproject_corners(corner_lon_full[2,:], corner_lat_full[2,:], wpsproj, latlonproj)
random_y_factor = -corner_y3[0]/12.5
random_x_factor = corner_x3[0]/65

ax1.add_patch(mpl.patches.Rectangle((corner_x3[0]+random_x_factor, corner_y3[0]+random_y_factor),  length_x[2], length_y[2],fill=None, lw=3, edgecolor='red', zorder=2))
ax1.text(-3680871, 700000, 'D03', va='top', ha='left',fontweight='bold', size=15, color='red', zorder=2)

ax1.set_extent([-131, -119, 46, 52], crs=ccrs.PlateCarree())

