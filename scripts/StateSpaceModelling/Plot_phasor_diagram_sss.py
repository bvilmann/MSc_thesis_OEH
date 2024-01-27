# -*- coding: utf-8 -*-
"""
Plot functions for phasor diagram

@author: chhil
"""


import matplotlib.pyplot as plt
import numpy as np

def plot_phasors(P,C,L,ax=None,legend_loc=None,legend:bool=True,bbox_to_anchor=None,line_styles = None):
    # P = Phasors with columns being x and y coordinates
    # C = Color of phasors
    # L = Labels of phasors

    if line_styles is None:
        line_styles = ['-']*len(C)


    total = len(L)
    
    # Used for setting labels position
    labels_x_minus = 0
    labels_x_plus = 0
    
    # Store minimum and maximum values for limits
    rmax = 0

    if ax is None:
        fig, ax = plt.subplots(1,1,dpi=150,subplot_kw={'projection': 'polar'})
    else:
        pass

    for i in range(0,total):
        radius = np.hypot(P[i,0], P[i,1])
        angle = np.arctan2(P[i,1], P[i,0])        
        ax.annotate('', xytext = (0, 0),  xy = (angle,radius),
                     xycoords='data', textcoords='data',
                     arrowprops=dict(arrowstyle = '->',color = C[i], ls=line_styles[i],
                                     linewidth=1.5,shrinkA = 0, shrinkB = 0,alpha=0.5))
        
        
        # Plot label
    #    plt.text(angle+0.17,radius, L[i], color= C[i])
             
        # Update max
        if (radius > rmax):
            rmax = radius

        if legend:
            if (i==0):
                legend_elements=[plt.Line2D([0],[0],color=C[i], label=L[i])]
            else:
                legend_elements.extend([plt.Line2D([0],[0],color=C[i], label=L[i])])

    ld_zeros = np.floor(np.log10(rmax)) #no. of leading zeros
    ticksize = np.around( rmax/4, -int( ld_zeros ) ) 
    ax.set_ylim(0,4*ticksize)
    ax.set_rticks([ticksize, 2*ticksize, 3*ticksize, 4*ticksize])  # less radial ticks

    if legend:
        if legend_loc is None:
            ax.legend(handles=legend_elements, bbox_to_anchor=(1,1.05), loc='upper left',fontsize=6, labelspacing=0)
        else:
            if bbox_to_anchor is None:
                ax.legend(handles=legend_elements, loc=legend_loc, fontsize=6, labelspacing=0)
            else:
                ax.legend(handles=legend_elements, loc=legend_loc, bbox_to_anchor=bbox_to_anchor, fontsize=6, labelspacing=0)
    ax.grid(True,linestyle=':')
    ax.tick_params(axis='both', labelsize=5, pad=-4)             
    return
