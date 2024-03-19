#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 22:29:13 2024

@author: evagnegy
"""

#In remaining subplots add k-means clustered images
#flatten the image to get a line of values
flatraster_t = t_hist.flatten()
flatraster_pr = pr_hist.flatten()
flatraster_t.mask = False
flatraster_pr.mask = False
flatraster_t = flatraster_t.data 
flatraster_pr = flatraster_pr.data 

runs = 1
clusters = []
for step in range(0,runs):

    #This scipy code clusters k-mean, code has same length as flattened
    #raster and defines which cluster the value corresponds to 
    flatraster_real_t = flatraster_t[~np.isnan(flatraster_t)] # get rid of nans
    flatraster_real_pr = flatraster_pr[~np.isnan(flatraster_pr)] # get rid of nans

    flatraster_real_all = np.column_stack((flatraster_real_t, flatraster_real_pr))
   
    values_all = flatraster_real_all.astype(float)

    centroids, variance = kmeans(values_all, 4,iter=10) # cluster 
    code, distance = vq(flatraster_real_all, centroids)
    new_index = np.argsort(np.argsort(centroids)) # sort centroids in ascending order
    new_code = [new_index[old_index] for old_index in code] # change indices to reflect the change in order, and keep consistent on the plotted map
    
    #Have to add the nans back in for plotting purposes 
    index1 = t_hist.shape[0]
    index2 = t_hist.shape[1]
    matrix = np.empty((index1,index2))
    matrix[:,:] = np.nan
    index = ~np.isnan(t_hist)
    
    print(len(new_code))
    matrix[index] = new_code 
  
     
    clusters.append(matrix)


clusters_avg = np.mean(clusters,axis=0) # gets the mean of all of the cluster runs    
clusters_avg = clusters_avg+1 #moves it from 0-3 to 1-4
clusters_avg = np.around(clusters_avg,0) #rounds all of the clusters to whole numbers (most are already whole)
