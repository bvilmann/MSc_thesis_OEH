from CCM_class import ComponentConnectionMethod, StateSpaceSystem
from CCM_utils import plot_nyquist, plot_Zdq, plot_eigs, load_ccm, \
    load_init_conditions, state_space_simulation, x_plot, show_matrix, \
        complex_to_real_expanded
from NetworkLoader import NetworkClass, form_network_ssm
from pscad_data_reader import PowerSourceDataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from numpy.linalg import inv

import pandas as pd
#%%
tf_dq_Lv = []

net = NetworkClass(verbose=False,scr_OWF=10,Pload=-1)

init_cond = net.load_flow2init_cond('single')[1]

for v in ['v_gD','v_gQ']:
    for i in ['i_oD','i_oQ']:
        # ccm, init_data = load_ccm(vd0=vd0,vq0=vq0,id0=id0,iq0=iq0,d0=0,scr=0,xr=10,x_vi=1,system_input=[v],system_output=[i],ccm_name='ssm_ccm')
        ccm, init_data = load_ccm(**init_cond, scr=2, xr=5, x_vi=1, L_v=0.07, R_v=0.03,
                                  system_input=[v],
                                  system_output=[i],
                                  ccm_name='ssm_ccm - load_conv')
        # ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
        # tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
        tf_dq_Lv.append({'ccm':ccm,'axis':f'{v[-1]}{i[-1]}'})

plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
savefile=f'{plot_path}\\nyquist_with_vi.pdf'
plot_nyquist(tf_dq_Lv,unit_circle=True,N=10000,w_start=-2,w_end=5,save= savefile)

