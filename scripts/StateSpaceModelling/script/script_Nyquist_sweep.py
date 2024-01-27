from CCM_class import ComponentConnectionMethod, StateSpaceSystem
from CCM_utils import plot_nyquist, plot_Zdq, plot_eigs, load_ccm, \
    load_init_conditions, state_space_simulation, x_plot, show_matrix, \
        complex_to_real_expanded,plot_nyquist_sweep
from NetworkLoader import NetworkClass, form_network_ssm
from pscad_data_reader import PowerSourceDataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from numpy.linalg import inv

from tqdm import tqdm

import pandas as pd
#%%
tf_dq_Lv = []

net = NetworkClass(verbose=False,scr_OWF=10,Pload=-1)

init_cond = net.load_flow2init_cond('single')[1]

xr_vi = 2*np.pi*0.07/0.03
theta = np.arctan(xr_vi)

for v in ['v_gD']:
    for i in ['i_oD']:
        for rv in tqdm(np.linspace(0,.2,5)):


            # ccm, init_data = load_ccm(vd0=vd0,vq0=vq0,id0=id0,iq0=iq0,d0=0,scr=0,xr=10,x_vi=1,system_input=[v],system_output=[i],ccm_name='ssm_ccm')
            ccm, init_data = load_ccm(**init_cond, scr=2, xr=5, x_vi=1, L_v=0, R_v=rv,
                                      system_input=[v],
                                      system_output=[i],
                                      ccm_name='ssm_ccm - load_conv')
            # ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
            # tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
            tf_dq_Lv.append({'ccm':ccm,'label':'$R_v=' +str(round(rv,4)) + '$'})

plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
savefile=f'{plot_path}\\nyquist_with_vi_Rv_sweep.pdf'
plot_nyquist_sweep(tf_dq_Lv,unit_circle=True,N=10000,w_start=-2,w_end=5,save= savefile)

#%%
tf_dq_Lv = []


for v in ['v_gD']:
    for i in ['i_oD']:
        for lv in tqdm(np.linspace(0,.2,5)):


            # ccm, init_data = load_ccm(vd0=vd0,vq0=vq0,id0=id0,iq0=iq0,d0=0,scr=0,xr=10,x_vi=1,system_input=[v],system_output=[i],ccm_name='ssm_ccm')
            ccm, init_data = load_ccm(**init_cond, scr=2, xr=5, x_vi=1, L_v=lv, R_v=0,
                                      system_input=[v],
                                      system_output=[i],
                                      ccm_name='ssm_ccm - load_conv')
            # ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
            # tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
            tf_dq_Lv.append({'ccm':ccm,'label':'$L_v=' +str(round(lv,4)) + '$'})

plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
savefile=f'{plot_path}\\nyquist_with_vi_Lv_sweep.pdf'
plot_nyquist_sweep(tf_dq_Lv,unit_circle=True,N=10000,w_start=-2,w_end=5,save= savefile)

#%%
tf_dq_Lv = []

for v in ['v_gD']:
    for i in ['i_oD']:
        for z_vi in tqdm(np.linspace(0,.2,5)):
        
            if z_vi == 0:
                lv = rv = 0
            else:
                zv = z_vi*(np.cos(theta) + 1j*np.sin(theta))
                rv = zv.real
                lv = zv.imag

            # ccm, init_data = load_ccm(vd0=vd0,vq0=vq0,id0=id0,iq0=iq0,d0=0,scr=0,xr=10,x_vi=1,system_input=[v],system_output=[i],ccm_name='ssm_ccm')
            ccm, init_data = load_ccm(**init_cond, scr=2, xr=5, x_vi=1, L_v=lv, R_v=rv,
                                      system_input=[v],
                                      system_output=[i],
                                      ccm_name='ssm_ccm - load_conv')
            # ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
            # tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
            tf_dq_Lv.append({'ccm':ccm,'label':'$|Z_v|=' +str(round(z_vi,4)) + '$'})

plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
savefile=f'{plot_path}\\nyquist_with_vi_Zv_sweep.pdf'
plot_nyquist_sweep(tf_dq_Lv,unit_circle=True,N=10000,w_start=-2,w_end=5,save= savefile)

