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
import control as ctrl

# %%
vd0 ,vq0 ,id0 ,iq0 ,d0 ,(t0, x0), vdq ,idq = load_init_conditions(t0=1.5)

ccm = load_ccm(vd0 ,vq0 ,id0 ,iq0 ,ccm_name='ssm_ccm' ,scr=10 ,xr=10 ,x_vi=0 ,system_input=['v_gd' ,'v_gq']
               ,system_output=['i_od' ,'i_oq'])

# x0 = np.hstack([x0[1,:]])
x0_ = np.zeros(11)
u = lambda t: np.where( t>= 0.1 ,[31 0 /30 0 -1 ,0] ,[0 ,0])  # Constant input
T ,dt  = 0.6, 250e-6
t = np.linspace(0, 0.6, int( T /dt))  # Simulation time from 0 to 10 seconds

t_ss, x_ss, y = state_space_simulation(ccm.A, ccm.B, ccm.C, ccm.D, x0_, u, t)

fig, ax = plt.subplots(2 ,1 ,dpi=150)

for i in range(2):
    ax[i].plot(t0 ,idq[: ,i ] -[id0 ,iq0][i] ,label=['$i_{d,pscad}$' ,'$i_{q,pscad}$'][i])
    ax[i].plot(t ,y[i ,:] ,label=['$i_{d}$' ,'$i_{q}$'][i])
    ax[i].legend(loc='center right')
    ax[i].grid(ls=':')
    # ax[i+2].plot(t0,vdq[:,i]-[vd0,vq0][i],label=['$v_{d,pscad}$','$v_{q,pscad}$'][i])
    # ax[i+2].plot(u[i,:],label=['$v_{d,u}$','$v_{q,u}$'][i])
    # ax[i+2].plot(t,y[i+2,:],label=['$v_{d}$','$v_{q}$'][i])
    # ax[i+2].legend(loc='center right')
    # ax[i+2].grid(ls=':')

plt.show()
plt.close()

ccm.show()

# %%
# PLOTS
# x_plot(t_ss,x_ss.T)
# x_plot(t_pscad,x_pscad)

mask = [i for i ,v in enumerate(ccm.x.name.values) if v not in ['P' ,'Q']]
x_ss_ = x_ss[mask ,:]
x_plot(t0 ,x0 - x0[0 ,:] ,t_sim= t_ss ,x_sim= x_ss_.T)

# %%
from CCM_class import ComponentConnectionMethod, StateSpaceSystem

vd0 ,vq0 ,id0 ,iq0 ,d0 ,(t0, x0), vdq ,idq = load_init_conditions(t0=2)

ccm = load_ccm(vd0 ,vq0 ,id0 ,iq0 ,d0=d0 ,ccm_name='ssm_ccm - orig' ,Vbase=300e3 ,scr=10 ,xr=10 ,x_vi=0
               ,system_input=['v_gd' ,'v_gq'] ,system_output=['i_od' ,'i_oq' ,'v_od' ,'v_oq'])

ss = ctrl.ss(ccm.A ,ccm.B ,ccm.C ,ccm.D)

dt = 250e-6
T = 0.6
t = np.linspace(0, T, int( T /dt))  # Adjust time range and steps as needed
u = np.where((t > 0.1 ) & (t < 0.9), 1 0 /300 ,0)  # Example input signal, replace with your actual input
u = np.vstack([u ,np.zeros_like(u)])
t ,y = ctrl.forced_response(ss, T=t, U=u)

fig, ax = plt.subplots(4 ,1 ,dpi=150)

for i in range(2):
    ax[i].plot(t0 ,idq[: ,i ] -[id0 ,iq0][i] ,label=['$i_{d,pscad}$' ,'$i_{q,pscad}$'][i])
    ax[i].plot(t ,y[i ,:] ,label=['$i_{d}$' ,'$i_{q}$'][i])
    ax[ i +2].plot(t0 ,vdq[: ,i ] -[vd0 ,vq0][i] ,label=['$v_{d,pscad}$' ,'$v_{q,pscad}$'][i])
    # ax[i+2].plot(u[i,:],label=['$v_{d,u}$','$v_{q,u}$'][i])
    ax[ i +2].plot(t ,y[ i +2 ,:] ,label=['$v_{d}$' ,'$v_{q}$'][i])
    ax[i].legend(loc='center right')
    ax[ i +2].legend(loc='center right')
    ax[i].grid(ls=':')
    ax[ i +2].grid(ls=':')

plt.show()
plt.close()

# ccm.show()

plot_eigs(ccm.A)