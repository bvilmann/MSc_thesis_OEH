# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:59:44 2024

@author: bvilm
"""

from CCM_class import ComponentConnectionMethod, StateSpaceSystem
from CCM_utils import plot_nyquist, plot_Zdq, plot_eigs, load_ccm, \
    load_init_conditions, state_space_simulation, x_plot, show_matrix, \
    complex_to_real_expanded, plot_eigvals_dict, plot_participation_factors,\
    plot_participation_factor_bars
from NetworkLoader import NetworkClass, form_network_ssm
from pscad_data_reader import PowerSourceDataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from numpy.linalg import inv
import pandas as pd
from tqdm import tqdm

# %% ==================== ICWG ====================
ccm_model = 'ssm_ccm - full'
# ccm_model = 'ssm_ccm - full - filter'
network = 'icwg' # select 'icwog', 'icwg', 'icwg_mva'
suffix = ''
save_ = True
xr = 10
Pload = 1
Snom = 2
pu_norm = (Pload/Snom)**(-1)

scr_OWF = 10
rv = 0.0
lv = 0.0

xr_vi = 2*np.pi*0.07/0.03
xr_vi = 0.5
theta = np.arctan(xr_vi)



Vbase = 300e3
Sbase = 1e9
Zbase = Vbase**2/Sbase

L_arm = 0.002/90

sens = {}

#%%
for xr in [0.5,2]:
    theta = np.arctan(xr_vi)
    for z_vi in tqdm(np.linspace(0,.1,21)):
    
        if z_vi == 0:
            lv = rv = 0
        else:
            zv = z_vi*(np.cos(theta) + 1j*np.sin(theta))
            rv = zv.real
            lv = zv.imag
    
        # Get initial conditions
        net = NetworkClass(verbose=False, scr_OWF=scr_OWF, scr_HSC1_icwg=20*pu_norm, xr_HSC1_icwg=xr, Pload=-1*Pload)
        init_cond = net.load_flow2init_cond('icwg',normalize_currents=False)
        
        Z = getattr(net,f'Z_{network}')
        # Yc = getattr(net,f'Y_{network}_cmplx')
    
        # net.lnd_icwg.draw()
    
        # ------------------------ NO VIRTUAL IMPEDANCE ------------------------ 
        # Load small-signal models
        ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_vi=1, L_v=lv, R_v=rv, L_arm = L_arm,
                                    system_input=['v_gD', 'v_gQ'],
                                    system_output=['i_oD', 'i_oQ'],
                                    # system_output=['i_od','i_oq'],
                                    ccm_name=ccm_model)
    
        ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10, x_vi=1, L_v=lv, R_v=rv, L_arm = L_arm,
                                    system_input=['v_gD', 'v_gQ'],
                                    system_output=['i_oD', 'i_oQ'],
                                    # system_output=['i_od','i_oq'],
                                    ccm_name=ccm_model)
    
        # Analyse system
        Ad = block_diag(ccm1.A, ccm2.A)
        Bd = np.hstack([block_diag(ccm1.B, ccm2.B), np.zeros((ccm2.B.shape[0] * 2, 2))])
        Cd = np.vstack([block_diag(ccm1.C, ccm2.C), np.zeros((2, ccm2.C.shape[1] * 2))])
    
        A = form_network_ssm(Ad, Bd, Cd, Z)
       
        lamb = np.linalg.eigvals(A)
    
        sens[round(z_vi,3)]=lamb

#%%
for k,v in sens.items():
    if (v.real> 0).any():
        print(k)
        # break
   
    
#%%

# Example data: Replace this with your actual data
plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
if 'filter' in ccm_model:
    axins_view1,axins_loc1 = (-10,1,-100,100),(0.1,0.775,.3,.2)
    axins_view1_vi,axins_loc1 = (-50,1,-100,100),(0.1,0.775,.3,.2)
    axins_view2,axins_loc2 = (-1000,10,-100,100),(0.1,0.05,.3,.2)
else:
    axins_view1,axins_loc1 =  (-10,2,-100,100),(0.35,0.6,.3,.3)
    axins_view1_vi,axins_loc1=(-50,1,-100,100),(0.35,0.6,.3,.3)
    axins_view2,axins_loc2 = (-800,10,-100,100),(0.35,0.1,.3,.3)

if not save_: 
    savefile=None
elif 'filter' in ccm_model:
    savefile=f'{plot_path}\\eigval_full_zv_sens_filter.pdf'
else:
    savefile=f'{plot_path}\\eigval_full_zv_sens{suffix}.pdf'

plot_eigvals_dict('$|Z_v|$',sens,save=savefile,y_offset=200,
                  insert_axis_args1=[axins_view1,axins_loc1],
                  insert_axis_kwargs1={'color':'white','ls':':'},
                  insert_axis_args2=[axins_view2,axins_loc2],
                  insert_axis_kwargs2={'color':'white','ls':':'},
                  # hline = {'color':'red','vmin':2.5*2*np.pi,'vmax':6.5*2*np.pi},
                  hline = {'color':'red','vmin':6*2*np.pi,'vmax':6*2*np.pi},
                  )


#%%
names = [n.latex_name for _, n in ccm1.x.iterrows()]
names = ['$\\Delta{' + n + '}_{,HSC1}$' for n in names] + ['$\\Delta{' + n + '}_{,HSC2}$' for n in names]

# names = [n.latex_name for _,n in ccm1.x.iterrows()]
lamb, Phi = np.linalg.eig(A)
print([(i+1,l) for i,l in enumerate(lamb) if l.real > 0 ])
Psi = inv(Phi)
P = Phi * Psi.T

# plot_participation_factors(abs(P), names)
# plot_participation_factor_bars(abs(P), names)


# plot_eigs(ccm1.A)


