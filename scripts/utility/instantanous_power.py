# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 08:10:49 2023

@author: bvilm
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def clarke_transform(I, invariant='amplitude'):
    """
    Perform Clarke (alpha-beta-0) transformation on three-phase currents or voltages.
    
    Parameters:
        I: list or numpy array
            The three-phase currents or voltages in the form [Ia, Ib, Ic].
        kappa: float, optional
            The scaling factor for the transformation matrix.
            Default value is 2/3, providing a power-invariant transformation.
    
    Returns:
        I_alpha, I_beta, I_0: float
            The alpha, beta, and zero-sequence components in the stationary reference frame.
    """
    if invariant == 'amplitude':
        kappa = np.sqrt(2/3)
    elif invariant == 'power':
        kappa = (2/3)
        
    # Define Clarke transformation matrix considering kappa
    clarke_matrix = kappa * np.array([
        [1, -0.5, -0.5],
        [0, np.sqrt(3)/2, -np.sqrt(3)/2],
        [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]
    ])
    
    # Ensure I is a column vector
    I = np.array(I).reshape(-1, 1)
    
    # Perform the transformation
    result_vector = np.dot(clarke_matrix, I)
    
    # Extract the alpha, beta, and zero components
    I_alpha = result_vector[0,0]
    I_beta = result_vector[1,0]
    I_0 = result_vector[2,0]
    
    return result_vector

T = 2
w = 2*np.pi*f
f = 50
t = np.linspace(0,T/f,1000)
V = 1
I = 1
phi = 60*np.pi/180
figscale = 1.5

# get signals
v = np.sqrt(2)*V*np.sin(w*t)
i = np.sqrt(2)*I*np.sin(w*t-phi)
p = V*I*np.cos(phi) - V*I*np.cos(2*w*t-phi)
P = V*I*np.cos(phi)*(1-np.cos(2*w*t))
Q =  - V*I*np.cos(2*w*t-phi)

s = np.cos(phi) - np.cos(2*w*t-phi)
# plot
fig, ax =  plt.subplots(2,1,dpi=150,figsize=(6*figscale,4*figscale))
ax[0].plot(t,v,label='$v(t)$')
ax[0].plot(t,i,color='C3',label='$i(t)$')

ax[1].plot(t,p,color='k')
ax[1].plot(t,s,color='grey',ls=':')
ax[1].plot(t,P,color='C3')
ax[1].plot(t,Q,color='C0')


ax[1].fill_between(t,0,p,where=(t>=0.03) & (p<=0),hatch=r'\\//',alpha=0.3,edgecolor='C5',facecolor='C5',label='Power returned to source')
ax[1].fill_between(t,0,p,where=(t>=0.03) & (p>=0),hatch=r'\\//',alpha=0.3,edgecolor='C2',facecolor='C2',label='Power delivered to load')

for i in range(2):
    ax[i].grid(ls=':')
    ax[i].axhline(0,color='k',lw=0.75,ls=':')
    ax[i].set(xlim=(0,T/f))
    ax[i].legend(loc='upper left')


#%%
fig, ax = plt.subplots(2,1,dpi=150)
v = lambda t,phi: np.sqrt(2)*1*np.sin(2*np.pi*50*t+phi*np.pi/180)
f = 50
t = np.linspace(0,T/f,1000)
va,vb,vc = v(t,0),v(t,-120),v(t,120)
ax[0].plot(t,va)
ax[0].plot(t,vb)
ax[0].plot(t,vc)

aby = np.array([clarke_transform([a,b,c]) for a,b,c in zip(va,vb,vc)])
ax[1].plot(t,aby[:,:,0])


