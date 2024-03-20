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


# this is a function that plots a colorbar, where the input is a colormap (cmap)
def plot_cbar(cmap,lim,label,norm=1,ticklabels=1,ticks=1):
    
    pl.figure(figsize=(8, 0.6),dpi=250)
    pl.gca().set_visible(False)
    cax = pl.axes([0.1, 0.2, 0.8, 0.6])
    
    if norm==1:
        norm = pltcol.Normalize(vmin=lim[0], vmax=lim[1])
    if isinstance(ticks,int):
        ticks=None
    else:
        ticks=ticks
    cb.ColorbarBase(cax,cmap=cmap,norm=norm,orientation="horizontal", extend='both',ticks=ticks)
    
    if ticklabels!=1:
        cax.set_xticks(ticks,ticklabels)
    
    cax.xaxis.set_tick_params(size=12,width=2)
    pl.xticks(fontsize=24)
    pl.xlabel(label,fontsize=24)

    plt.show()
    
       
          
lim = [-5,5] 
cmap = 'bwr'
label = 'Temperature Bias (\N{degree sign}C)'

lim = [-5,5] 
cmap = 'bwr_r'
label = 'Precipitation Bias (mm/day)'

lim = [-3,3] 
cmap = 'bwr'
label = 'Wind Speed Bias (m/s)'

plot_cbar(cmap,lim,label)
