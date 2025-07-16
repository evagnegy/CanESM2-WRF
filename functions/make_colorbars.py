#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:14:48 2024

@author: evagnegy
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as pltcol
import matplotlib.colorbar as cb
import numpy as np
import pylab as pl
from matplotlib import cm



def make_colorbar(colors,lim):
    
    color_count = len(colors)-2
    
    #ticks = np.linspace(lim[0],lim[1],color_count+1) #location where I want ticks
    #ticks = np.linspace(lim[0],lim[1],6) #location where I want ticks
    ticks = np.linspace(lim[0],lim[1],5) #location where I want ticks

    print(ticks)

    #cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors[1:-1],N=20) #[1:-1] is to not include the min/max arrow colors
    cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors,N=1000) #[1:-1] is to not include the min/max arrow colors
    
    
# =============================================================================
#     N = abs(lim[0])+abs(lim[1])
#     cmap_temp = pltcol.LinearSegmentedColormap.from_list("custom", colors,N=N) #[1:-1] is to not include the min/max arrow colors
#     cmap_list = [cmap_temp(i) for i in range(N)]
#     cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap_list[1:-1], N=abs(lim[0])+abs(lim[1])) #[1:-1] is to not include the min/max arrow colors
# 
# =============================================================================
   

    cmap.set_over(colors[-1]) #add the max arrow color
    cmap.set_under(colors[0]) #add the min arrow color
    
    return(cmap,ticks)
     


# this is a function that plots a colorbar, where the input is a colormap (cmap)
def plot_cbar(cmap,lim,label,norm=1,ticks=1):
    
    pl.figure(figsize=(20, 1.5),dpi=250)
    pl.gca().set_visible(False)
    cax = pl.axes([0.1, 0.2, 0.8, 0.6])
    
    if norm==1:
        norm = pltcol.Normalize(vmin=lim[0], vmax=lim[1])
    if isinstance(ticks,int):
        ticks = None
         
    else:
        ticks=ticks
        #ticklabels = ['‒3','‒1.5','0','1.5','3']
        ticklabels = ['‒60','‒40','-20','0','20','40','60']
        ticklabels=[str(round(i)) for i in (ticks)] 
    
    cb.ColorbarBase(cax,cmap=cmap,norm=norm,orientation="horizontal", extend='both',ticks=ticks)
    
    cax.set_xticks(ticks,ticklabels)
    
    cax.xaxis.set_tick_params(size=35,width=2)
    pl.xticks(fontsize=90)
    pl.xlabel(label,fontsize=100)

    plt.show()
     
         

#%%
colors_tas = ['#383593','#4f74b2','#7dacd0','#b1d8e8','#e2f3f8','#fdffc1',
              '#fae194','#f6b066','#eb7249','#ce3a2f','#9e152a']

lim = [-12,18] 
label = 'Temperature (\N{degree sign}C)'

cmap,ticks = make_colorbar(colors_tas,lim)

plot_cbar(cmap,lim,label,ticks=ticks)


#%%

colors_tas_delta = ['#142f60','#3465aa','#5392c1','#99c4dd','#d3e5f0','#f7f7f7',
                    '#fadcc8','#eea785','#ce6451','#ab242f','#630921']

lim = [-5,5] 

cmap,ticks = make_colorbar(colors_tas_delta,lim)

label = 'Temperature Bias (\N{degree sign}C)'
plot_cbar(cmap,lim,label,ticks=ticks)


#%%
colors_pr = ['#f2ffda','#ecf8b3','#c9e9b5','#88ccbb','#56b4c3','#3c8fbe','#335da6','#2e3392','#121c57']

lim = [0,15] 
label = ' Precipitation (mm/day)'

cmap,ticks = make_colorbar(colors_pr,lim)

plot_cbar(cmap,lim,label,ticks=ticks)

#%%

colors_pr_delta = ['#386158','#41847e','#67aba5','#a0d6cd','#d5edea','#f7f7f7','#f3e7c4',
                   '#dbc37f','#b88234','#865214','#503009'][::-1]

lim = [-100,100] 

cmap,ticks = make_colorbar(colors_pr_delta,lim)

label = 'Precipitation Bias (%)'
plot_cbar(cmap,lim,label,ticks=ticks)

#%%

colors_wspd= ['#f2ffda','#ecf8b3','#c9e9b5','#88ccbb','#56b4c3','#3c8fbe','#335da6','#2e3392','#121c57']

lim = [0,12] 
label = 'Wind Speed (m/s)'

cmap,ticks = make_colorbar(colors_wspd,lim)

plot_cbar(cmap,lim,label,ticks=ticks)

#%%

colors_wspd_delta = ['#424c03','#41641a','#4b8c49','#79b17d','#aed0b2','#d7e3e0',
                     '#aec3d5','#7394b5','#3e6896','#294072','#2c194d'][::-1]
 

lim = [-100,100] 

cmap,ticks = make_colorbar(colors_wspd_delta,lim)

label = 'Wind Speed Bias (%)'
plot_cbar(cmap,lim,label,ticks=ticks)

#%%
cmap = cm.get_cmap('PRGn', 24)
 
ticks=np.array([-1500,-750,0,750,1500])
lim = [-1500,1500] 


label = 'Elevation Bias (m)'
plot_cbar(cmap,lim,label,ticks=ticks)

#%%

t_colors = ['#f2f8d4','#fdce62','#f28a2c','#e84c0f','#bd1717','#6f150c','#741744']
newcmp_t = pltcol.LinearSegmentedColormap.from_list("custom", t_colors,N=22)
newcmp_t = newcmp_t(np.linspace(0, 1, newcmp_t.N))[1:-1] 
newcmp_t = pltcol.LinearSegmentedColormap.from_list("custom", newcmp_t,N=20)
newcmp_t.set_over(t_colors[-1]) #add the max arrow color
newcmp_t.set_under(t_colors[0]) #add the min arrow color


lim = [0,5] 

ticks = np.linspace(lim[0],lim[1],6)

label = '$\Delta$ Temperature ($\degree$C)'
plot_cbar(newcmp_t,lim,label,ticks=ticks)


#%%
colors_pr = ['#543005','#8c5109','#a4671b','#c7974a','#d4b775','#f5e7c6','#c7e7e2','#80cdc1','#35978f','#12766e','#01665e','#003c30']
newcmp_pr = pltcol.LinearSegmentedColormap.from_list("custom", colors_pr,N=18)
newcmp_pr = newcmp_pr(np.linspace(0, 1, newcmp_pr.N))[1:-1] 
newcmp_pr = pltcol.LinearSegmentedColormap.from_list("custom", newcmp_pr,N=16)
newcmp_pr.set_over(colors_pr[-1]) #add the max arrow color
newcmp_pr.set_under(colors_pr[0]) #add the min arrow color

lim = [-80,80] 

ticks = np.array([-80,-40,0,40,80])
#ticks = np.array([-50,0,50])

label = '$\Delta$ Precipitation (%)'
plot_cbar(newcmp_pr,lim,label,ticks=ticks)

#%%

lim = [-20,20] 

colors_wspd_delta = ['#424c03','#41641a','#4b8c49','#79b17d','#aed0b2','#d7e3e0',
                     '#aec3d5','#7394b5','#3e6896','#294072','#2c194d'][::-1]
cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors_wspd_delta,N=22)
cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=20)
cmap.set_over(colors_wspd_delta[-1]) #add the max arrow color
cmap.set_under(colors_wspd_delta[0]) #add the min arrow color

ticks=np.array([-20,-10,0,10,20])
label = '$\Delta$ Wind Speed (%)'
plot_cbar(cmap,lim,label,ticks=ticks)

#%%

 
lim = [0,15] 
cmap = 'viridis'
ticks=np.array([0,5,10,15])

label = 'Count/year'
plot_cbar(cmap,lim,label,ticks=ticks)

#%%
lim = [0,12] 
ticks = np.array([0,2,4,6,8,10,12])

vmin= 0
vmax = 12
cmap = plt.get_cmap('Purples')
colors = [cmap(i / (11 - 1)) for i in range(11)]

cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors,N=11)
cmap = cmap(np.linspace(0, 1, cmap.N))[:-1] 
cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=12)
cmap.set_over(colors[-1]) #add the max arrow color

label = 'Count/year'
plot_cbar(cmap,lim,label,ticks=ticks)

#%%

lim = [-60,60] 
cmap = 'bwr'
ticks=np.array([-50,0,50])

label = '$\Delta$ Count/year'
plot_cbar(cmap,lim,label,ticks=ticks)

#%%

# =============================================================================
# colors_tas_delta = ['#142f60','#3465aa','#5392c1','#99c4dd','#d3e5f0','#f7f7f7',
#                     '#fadcc8','#eea785','#ce6451','#ab242f','#630921']
# newcmp_t = pltcol.LinearSegmentedColormap.from_list("custom", colors_tas_delta,N=26)
# newcmp_t = newcmp_t(np.linspace(0, 1, newcmp_t.N))[1:-1] 
# newcmp_t = pltcol.LinearSegmentedColormap.from_list("custom", newcmp_t,N=24)
# newcmp_t.set_over(colors_tas_delta[-1]) #add the max arrow color
# newcmp_t.set_under(colors_tas_delta[0]) #add the min arrow color
# =============================================================================

# =============================================================================
# lim = [0,8] 
# ticks = np.array([0,2,4,6,8])
# cmap = cm.get_cmap('YlOrRd', 16)
# =============================================================================

#lim = [-1,1] 
#ticks = np.array([-1,-0.5,0,0.5,1])
#lim = [-6,6] 
#ticks = np.array([-6,-3,0,3,6])
#lim = [-3,3] 
#ticks = np.array([-3,-1.5,0,1.5,3])

 
colors_wspd_delta = ['#424c03','#41641a','#4b8c49','#79b17d','#aed0b2','#d7e3e0',
                     '#aec3d5','#7394b5','#3e6896','#294072','#2c194d'][::-1]
cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors_wspd_delta,N=22)
cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=20)
cmap.set_over(colors_wspd_delta[-1]) #add the max arrow color
cmap.set_under(colors_wspd_delta[0]) #add the min arrow color
lim = [-20,20]
ticks=np.array([-20,-10,0,10,20])

#label = '$\Delta$ Tmin 5p ($\degree$C)'
#label = '$\Delta$ Tmax 95p ($\degree$C)'
label = '$\Delta$ Wind Speed (%)'
#label = '$\Delta$ Wspd 95p (m/s)'
plot_cbar(cmap,lim,label,ticks=ticks)

#%%

lim = [-5,5] 
ticks = np.array([-5,-2.5,0,2.5,5])

vmin= -5
vmax = 5
cmap = plt.get_cmap('PuOr_r')
colors = [cmap(i / (22 - 1)) for i in range(22)]

cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors,N=22)
cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=20)
cmap.set_over(colors[-1]) #add the max arrow color
cmap.set_under(colors[0]) #add the min arrow color

#label = '$\Delta$ Tmin 50p-5p ($\degree$C)'
label = '$\Delta$ Tmax 95p-50p ($\degree$C)'

plot_cbar(cmap,lim,label,ticks=ticks)

#%%


vmin= -60
vmax = 60
colors = ['#424c03','#41641a','#4b8c49','#79b17d','#aed0b2', '#cfe6d1','#eddaeb',
                     '#d6b4d2','#c98dc1','#ad49a0','#8c037a','#5c0250'][::-1]



cmap = pltcol.LinearSegmentedColormap.from_list("custom", colors,N=20)
cmap = cmap(np.linspace(0, 1, cmap.N))[1:-1] 
cmap = pltcol.LinearSegmentedColormap.from_list("custom", cmap,N=18)
cmap.set_over(colors[-1]) #add the max arrow color
cmap.set_under(colors[0]) #add the min arrow color

lim = [vmin,vmax] 
ticks=np.array([-60,-40,-20,0,20,40,60])

label = '$\Delta$ Count/year'
plot_cbar(cmap,lim,label,ticks=ticks)