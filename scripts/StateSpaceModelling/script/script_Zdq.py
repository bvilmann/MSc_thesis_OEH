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

for Lv in [0,0.02,0.04,0.06,0.08]:
    for v in ['v_gD','v_gQ']:
        for i in ['i_oD','i_oQ']:
            # ccm, init_data = load_ccm(vd0=vd0,vq0=vq0,id0=id0,iq0=iq0,d0=0,scr=0,xr=10,x_vi=1,system_input=[v],system_output=[i],ccm_name='ssm_ccm')
            ccm, init_data = load_ccm(**init_cond, scr=0, xr=10, x_vi=1, L_v=Lv, R_v=0.00,
                                      system_input=[v],
                                      system_output=[i],
                                      ccm_name='ssm_ccm - load_conv')
            # ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
            # tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
            tf_dq_Lv.append({'ccm':ccm,
                                        'label':'$L_v='+f'{round(Lv,4)}$',
                                        'axis':f'{v[-1]}{i[-1]}'})

# plot_nyquist(tf_dq_Lv,unit_circle=True,N=10000,w_start=-3,w_end=5)

#%%
v = tf_dq_Lv
A = (v['ccm']).A  # Replace with your A matrix
B = (v['ccm']).B  # Replace with your B matrix
C = (v['ccm']).C  # Replace with your C matrix
D = (v['ccm']).D  # Replace with your C matrix



#%%
plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
plot_Zdq(tf_dq_Lv,save=f'{plot_path}\\impedance_plot_hsc_Lv.pdf')

#%%
tf_dq_Rv = []

net = NetworkClass(verbose=False,scr_OWF=10,Pload=-1)

init_cond = net.load_flow2init_cond('single')[1]

for Rv in [0,0.02,0.04,0.06,0.08]:
    for v in ['v_gD','v_gQ']:
        for i in ['i_oD','i_oQ']:
            # ccm, init_data = load_ccm(vd0=vd0,vq0=vq0,id0=id0,iq0=iq0,d0=0,scr=0,xr=10,x_vi=1,system_input=[v],system_output=[i],ccm_name='ssm_ccm')
            ccm, init_data = load_ccm(**init_cond, scr=0, xr=10, x_vi=1, L_v=0., R_v=Rv,
                                      system_input=[v],
                                      system_output=[i],
                                      ccm_name='ssm_ccm - load_conv')
            # ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
            # tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
            tf_dq_Rv.append({'ccm':ccm,
                                        'label':'$R_v='+f'{round(Rv,4)}$',
                                        'axis':f'{v[-1]}{i[-1]}'})

# plot_nyquist(tf_dq_Lv,unit_circle=True,N=10000,w_start=-3,w_end=5)

plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
plot_Zdq(tf_dq_Rv,save=f'{plot_path}\\impedance_plot_hsc_Rv.pdf')

#%%
tf_dq_Zv = []

net = NetworkClass(verbose=False,scr_OWF=10,Pload=-1)

init_cond = net.load_flow2init_cond('single')[1]

xr_vi = 2*np.pi*0.07/0.03
theta = np.arctan(xr_vi)

for Zv in [0.00,0.02,0.04,0.06,0.08]:
    if Zv == 0:
        lv = rv = 0
    else:
        zv = Zv*(np.cos(theta) + 1j*np.sin(theta))
        rv = zv.real
        lv = zv.imag/(2*np.pi)

    for v in ['v_gD','v_gQ']:
        for i in ['i_oD','i_oQ']:
            # ccm, init_data = load_ccm(vd0=vd0,vq0=vq0,id0=id0,iq0=iq0,d0=0,scr=0,xr=10,x_vi=1,system_input=[v],system_output=[i],ccm_name='ssm_ccm')
            ccm, init_data = load_ccm(**init_cond, scr=0, xr=10, x_vi=1, L_v=lv, R_v=rv,
                                      system_input=[v],
                                      system_output=[i],
                                      ccm_name='ssm_ccm - load_conv')
            # ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
            # tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
            tf_dq_Zv.append({'ccm':ccm,
                                        'label':'$|Z_v|='+f'{round(Zv,4)}$',
                                        'axis':f'{v[-1]}{i[-1]}'})
# plot_nyquist(tf_dq_Lv,unit_circle=True,N=10000,w_start=-3,w_end=5)


plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
plot_Zdq(tf_dq_Zv,save=f'{plot_path}\\impedance_plot_hsc_Zv.pdf')


#%%
tf_dq_Ltfr = []

net = NetworkClass(verbose=False,scr_OWF=10,Pload=-1)

init_cond = net.load_flow2init_cond('single')[1]

for Ltfr in [0,0.02,0.04,0.06,0.08]:
    for v in ['v_gD','v_gQ']:
        for i in ['i_oD','i_oQ']:
            # ccm, init_data = load_ccm(vd0=vd0,vq0=vq0,id0=id0,iq0=iq0,d0=0,scr=0,xr=10,x_vi=1,system_input=[v],system_output=[i],ccm_name='ssm_ccm')
            ccm, init_data = load_ccm(**init_cond, scr=10, xr=10, x_vi=0, L_v=0, R_v=0.0, L_tfr=Ltfr,R_tfr=0,
                                      system_input=[v],
                                      system_output=[i],
                                      ccm_name='ssm_ccm - load_conv')
            # ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
            # tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
            tf_dq_Ltfr.append({'ccm':ccm,
                                        'label':'$\\mathit{L}_{tfr}='+f'{round(Ltfr,4)}$',
                                        'axis':f'{v[-1]}{i[-1]}'})
# plot_nyquist(tf_dq_Lv,unit_circle=True,N=10000,w_start=-3,w_end=5)

plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
plot_Zdq(tf_dq_Ltfr,save=f'{plot_path}\\impedance_plot_hsc_Ltfr.pdf')

#%%
tf_dq_Rtfr = []

net = NetworkClass(verbose=False,scr_OWF=10,Pload=-1)

init_cond = net.load_flow2init_cond('single')[1]

for Rtfr in [0,0.02,0.04,0.06,0.08]:
    for v in ['v_gD','v_gQ']:
        for i in ['i_oD','i_oQ']:
            # ccm, init_data = load_ccm(vd0=vd0,vq0=vq0,id0=id0,iq0=iq0,d0=0,scr=0,xr=10,x_vi=1,system_input=[v],system_output=[i],ccm_name='ssm_ccm')
            ccm, init_data = load_ccm(**init_cond, scr=10, xr=10, x_vi=0, L_v=0, R_v=0.0, L_tfr=0.02,R_tfr=Rtfr,
                                      system_input=[v],
                                      system_output=[i],
                                      ccm_name='ssm_ccm - load_conv')
            # ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
            # tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
            tf_dq_Rtfr.append({'ccm':ccm,
                                        'label':'$\\mathit{R}_{tfr}='+f'{round(Rtfr,4)}$',
                                        'axis':f'{v[-1]}{i[-1]}'})
# plot_nyquist(tf_dq_Lv,unit_circle=True,N=10000,w_start=-3,w_end=5)

plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
plot_Zdq(tf_dq_Rtfr,save=f'{plot_path}\\impedance_plot_hsc_Rtfr.pdf')

