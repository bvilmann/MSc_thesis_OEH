# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:59:44 2024

@author: bvilm
"""

from CCM_class import ComponentConnectionMethod, StateSpaceSystem
from CCM_utils import plot_nyquist, plot_Zdq, plot_eigs, load_ccm, \
    load_init_conditions, state_space_simulation, x_plot, show_matrix, \
    complex_to_real_expanded, plot_eigvals_dict, plot_participation_factors,\
    plot_participation_factor_bars,plot_participation_factors, eigen_properties_to_latex, participation_matrix_to_latex
from NetworkLoader import NetworkClass, form_network_ssm
from pscad_data_reader import PowerSourceDataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from numpy.linalg import inv
import pandas as pd
from tqdm import tqdm

tab_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\tab'
plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'

# %% ==================== ICWG ====================
ccm_model = 'ssm_ccm - full'
# ccm_model = 'ssm_ccm - full - filter'
network = 'icwg' # select 'icwog', 'icwg', 'icwg_mva'
suffix = '_1pu'
save_ = True
xr = 10
Pload = 1
Snom = 2
pu_norm = (Pload/Snom)**(-1)
scr_OWF = 10
rv = 0.0
Vbase = 300e3
Sbase = 1e9
Zbase = Vbase**2/Sbase

L_arm = 0.002/90

sens = {}

#%%
cnt = 0
for lv in tqdm(np.linspace(0,1,101)):
# for lv in tqdm(np.linspace(0,3,3)):
# for lv in tqdm(np.linspace(0,50,3)):
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

    sens[round(lv,3)]=lamb
    

    # if cnt == 2:
    #     names = [n.latex_name for _, n in ccm1.x.iterrows()]
    #     names = ['$\\Delta{' + n + '}_{,HSC1}$' for n in names] + ['$\\Delta{' + n + '}_{,HSC2}$' for n in names]
        
    #     # names = [n.latex_name for _,n in ccm1.x.iterrows()]
    #     lamb, Phi = np.linalg.eig(A)
    #     print([(i+1,l) for i,l in enumerate(lamb) if l.real > 0 ])
    #     Psi = inv(Phi)
    #     P = Phi * Psi.T
        
    #     savefile=f'{plot_path}\\participation_factor_Lv.pdf'
    #     plot_participation_factors(abs(P), names,save=savefile)
    #     # plot_participation_factors(abs(P), names)

    # if cnt == 2:
    #     dominating_states = [r'$\Delta\var{i}{o,HSC1}{dq}$, $\Delta\var{i}{o,HSC2}{dq}$',
    #                          r'$\Delta\var{i}{o,hpf,HSC1}{dq}$, $\Delta\var{i}{o,hpf,HSC2}{dq}$',
    #                          r'$\Delta\delta_{HSC1}$, $\Delta\delta_{HSC2}$, $\Delta\var{\zeta}{v,HSC1}{dq}$, $\Delta\var{\zeta}{v,HSC2}{dq}$',
    #                          ]  
    #     eigen_properties_to_latex(A,tab_path,'Lv',selection=[1,3,9],caption='Selected eigenvalue properties for bordering unstable system from $L_v = 0.01$.',dominating_states=dominating_states)

    cnt += 1

    
#%%
for k,v in sens.items():
    if (v.real> 0).any():
        print(k,v.real.max())
    else:
        print(k)
        # break
    
#%%

# Example data: Replace this with your actual data
plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
if 'filter' in ccm_model:
    axins_view1,axins_loc1 = (-50,1,-100,100),(0.1,0.775,.3,.2)
    axins_view1_vi,axins_loc1 = (-50,1,-100,100),(0.1,0.775,.3,.2)
    axins_view2,axins_loc2 = (-1000,10,-100,100),(0.1,0.05,.3,.2)
else:
    axins_view1,axins_loc1 =  (-50,2,-100,100),(0.35,0.6,.3,.3)
    axins_view1_vi,axins_loc1=(-50,1,-100,100),(0.35,0.6,.3,.3)
    axins_view2,axins_loc2 = (-800,10,-100,100),(0.35,0.1,.3,.3)

if not save_: 
    savefile=None
elif 'filter' in ccm_model:
    savefile=f'{plot_path}\\eigval_full_lv_sens_filter.pdf'
else:
    savefile=f'{plot_path}\\eigval_full_lv_sens{suffix}.pdf'

plot_eigvals_dict('$L_v$',sens,save=savefile,y_offset=200,
                  insert_axis_args1=[axins_view1,axins_loc1],
                  insert_axis_kwargs1={'color':'white','ls':':'},
                  insert_axis_args2=[axins_view2,axins_loc2],
                  insert_axis_kwargs2={'color':'white','ls':':'},
                  # show_labels=[1,2,3,4,9,10],
                  # show_labels=[1,2,3,4,9,10],
                  # force_labels=True,
                  # hline = {'color':'red','vmin':2.5*2*np.pi,'vmax':6.5*2*np.pi},
                  # hline = {'color':'red','vmin':6*2*np.pi,'vmax':6*2*np.pi},
                  )

#%%
# names = [n.latex_name for _, n in ccm1.x.iterrows()]
# names = ['$\\Delta{' + n + '}_{,HSC1}$' for n in names] + ['$\\Delta{' + n + '}_{,HSC2}$' for n in names]

# # names = [n.latex_name for _,n in ccm1.x.iterrows()]
# lamb, Phi = np.linalg.eig(A)
# print([(i+1,l) for i,l in enumerate(lamb) if l.real > 0 ])
# Psi = inv(Phi)
# P = Phi * Psi.T

#%%
# participation_matrix_to_latex(P, tab_path, 'pm_Lv', names, selected_rows:list=None, selected_columns, threshold=0.02, caption='')
net = NetworkClass(verbose=False, scr_OWF=scr_OWF, scr_HSC1_icwg=20*pu_norm, xr_HSC1_icwg=xr, Pload=-1*Pload)
init_cond = net.load_flow2init_cond('icwg',normalize_currents=False)

Z = getattr(net,f'Z_{network}')
# Yc = getattr(net,f'Y_{network}_cmplx')

# net.lnd_icwg.draw()

# ------------------------ NO VIRTUAL IMPEDANCE ------------------------ 
# Load small-signal models
ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_vi=1, L_v=0.01, R_v=0, L_arm = L_arm,
                            system_input=['v_gD', 'v_gQ'],
                            system_output=['i_oD', 'i_oQ'],
                            # system_output=['i_od','i_oq'],
                            ccm_name=ccm_model)

ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10, x_vi=1, L_v=0.01, R_v=0, L_arm = L_arm,
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

sens[round(lv,3)]=lamb


names = [n.latex_name for _, n in ccm1.x.iterrows()]
names = ['$\\Delta{' + n + '}_{,HSC1}$' for n in names] + ['$\\Delta{' + n + '}_{,HSC2}$' for n in names]

# names = [n.latex_name for _,n in ccm1.x.iterrows()]
lamb, Phi = np.linalg.eig(A)
print([(i+1,l) for i,l in enumerate(lamb) if l.real > 0 ])
Psi = inv(Phi)
P = Phi * Psi.T

savefile=f'{plot_path}\\participation_factor_Lv.pdf'
plot_participation_factors(abs(P), names,save=savefile)
# plot_participation_factors(abs(P), names)

dominating_states = [
    #r'$\Delta\var{i}{o,HSC1}{dq}$, $\Delta\var{i}{o,HSC2}{dq}$',
     #                 r'$\Delta\var{i}{o,hpf,HSC1}{dq}$, $\Delta\var{i}{o,hpf,HSC2}{dq}$',
                      r'$\Delta\delta_{HSC1}$, $\Delta\delta_{HSC2}$, $\Delta\var{\zeta}{v,HSC1}{dq}$, $\Delta\var{\zeta}{v,HSC2}{dq}$',
                      ]  
eigen_properties_to_latex(A,tab_path,'Lv',selection=[9],caption='Selected eigenvalue properties for bordering unstable system with $L_v = 0.01$.',dominating_states=dominating_states)




names = [n.latex_name for _, n in ccm1.x.iterrows()]
names = ['$\\Delta{' + n + '}_{,HSC1}$' for n in names] + ['$\\Delta{' + n + '}_{,HSC2}$' for n in names]

# participation_matrix_to_latex(P, tab_path, 'pm_Lv', names, selected_rows=[0,11,3,4,14,15,7,8,18,19], selected_columns=[0,2,8], threshold=0.01,mode_labels=['1,2','3,4','9,10'], caption='Selected participation factors for $L_v=0.01$.',absolute=True)
participation_matrix_to_latex(P, tab_path, 'pm_Lv', names, selected_rows=[0,11,3,4,14,15], selected_columns=[8], threshold=0.01,mode_labels=['9,10'], caption='Selected participation factors for $L_v=0.01$.',absolute=True)
participation_matrix_to_latex(P, tab_path, 'pm_Lv_full', names, threshold=0.02, caption='Selected participation factors for $L_v=0.01$.',absolute=True)
# plot_participation_factors(abs(P), names)
# plot_participation_factor_bars(abs(P), names)


# plot_eigs(ccm1.A)


