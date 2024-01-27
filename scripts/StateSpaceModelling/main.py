from CCM_class import ComponentConnectionMethod, StateSpaceSystem
from CCM_utils import plot_nyquist, plot_Zdq, plot_eigs, load_ccm, \
    load_init_conditions, state_space_simulation, x_plot, show_matrix, \
        complex_to_real_expanded, plot_participation_factors
from NetworkLoader import NetworkClass, form_network_ssm 
from pscad_data_reader import PowerSourceDataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from numpy.linalg import inv
import pandas as pd

asdf

#%% ==================== SINGLE ====================

net = NetworkClass(verbose=False,scr_OWF=10,Pload=-1)

init_cond = net.load_flow2init_cond('single')

ccm, init_data = load_ccm(**init_cond[1], scr=10, xr=10, x_net=1, x_vi=0, L_v = 0.07, R_v = 0.03,
                          system_input=['v_gD','v_gQ'],
                           system_output=['i_oD','i_oQ'],
                          # system_output=['i_od','i_oq'],
                          ccm_name='ssm_ccm - load_conv')

# ccm.show(save=False)
plot_eigs(ccm.A)

A = ccm.A
B = np.hstack([ccm.B,np.zeros_like(ccm.B)])
C = np.vstack([ccm.C,np.zeros_like(ccm.C)])

A = form_network_ssm(ccm.A,
                     np.hstack([ccm.B,np.zeros_like(ccm.B)]),
                     np.vstack([ccm.C,np.zeros_like(ccm.C)]),
                     net.Z_single)

plot_eigs(A)

#%% ==================== FULL ====================

# Get initial conditions
net = NetworkClass(verbose=False,scr_OWF=10,scr_HSC1=40,scr_HSC2=40,Pload=-1)
init_cond = net.load_flow2init_cond('full')

# Load small-signal models
ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_net=1, x_vi=0, L_v = 0.07, R_v = 0.03,
                          system_input=['v_gD','v_gQ'],
                           system_output=['i_oD','i_oQ'],
                          # system_output=['i_od','i_oq'],
                          ccm_name='ssm_ccm - full')

ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10, x_net=1, x_vi=0, L_v = 0.07, R_v = 0.03,
                          system_input=['v_gD','v_gQ'],
                           system_output=['i_oD','i_oQ'],
                          # system_output=['i_od','i_oq'],
                          ccm_name='ssm_ccm - full')

# Analyse system
# plot_eigs(ccm1.A)


Ad = block_diag(ccm1.A,ccm2.A)
Bd = np.hstack([block_diag(ccm1.B,ccm2.B),np.zeros((ccm2.B.shape[0]*2,4))])
Cd = np.vstack([block_diag(ccm1.C,ccm2.C),np.zeros((4,ccm.C.shape[1]*2))])
Z = net.Z_full


# net.lnd_full.draw()
for i, mod in enumerate([Ad,Bd,Cd,Z]):
    print('ABCZ'[i],mod.shape)
    # show_matrix(mod,title='ABCZ'[i])

# show_matrix(Bd @ Z,title='Bd @ Z')
# show_matrix(Z @ Cd,title='Z @ Cd')

A = form_network_ssm(Ad, Bd, Cd, Z)

plot_eigs(A)

#%% ==================== ICWG ====================

# for scr in [1.5,2,3,4,5,6,7,8,9,10]:
for scr in [12]:
    
    # Get initial conditions
    net = NetworkClass(verbose=False,scr_OWF=10,scr_HSC1_icwg=scr,xr_HSC1_icwg=10,Pload=-1)
    init_cond = net.load_flow2init_cond('icwg')
    
    # net.lnd_icwg.draw()
    
    # Load small-signal models
    ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_vi=0, L_v = 0.07, R_v = 0.03,
                              system_input=['v_gD','v_gQ'],
                               system_output=['i_oD','i_oQ'],
                              # system_output=['i_od','i_oq'],
                              ccm_name='ssm_ccm - full')
    
    ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10, x_vi=0, L_v = 0.07, R_v = 0.03,
                              system_input=['v_gD','v_gQ'],
                               system_output=['i_oD','i_oQ'],
                              # system_output=['i_od','i_oq'],
                              ccm_name='ssm_ccm - full')
    
    # Analyse system
    # plot_eigs(ccm1.A)
    Ad = block_diag(ccm1.A,ccm2.A)
    Bd = np.hstack([block_diag(ccm1.B,ccm2.B),np.zeros((ccm2.B.shape[0]*2,2))])
    Cd = np.vstack([block_diag(ccm1.C,ccm2.C),np.zeros((2,ccm2.C.shape[1]*2))])
    Z = net.Z_icwg
    
    
    for i, mod in enumerate([Ad,Bd,Cd,Z]):
        print('ABCZ'[i],mod.shape)
        # show_matrix(mod,title='ABCZ'[i])
    
    # show_matrix(Bd @ Z,title='Bd @ Z')
    # show_matrix(Z @ Cd,title='Z @ Cd')
    
    A = form_network_ssm(Ad, Bd, Cd, Z)
    
    plot_eigs(A)


#%%
lamb, Phi = np.linalg.eig(A)
Psi = inv(Phi)
P = Phi * Psi.T

names = [n.latex_name for _,n in ccm1.x.iterrows()]
names = ['$\\Delta{'+n +'}_{,HSC1}$' for n in names] + ['$\\Delta{'+n +'}_{,HSC2}$' for n in names]

# for i in range(len(names)):
#     fig,ax = plt.subplots(1,1,dpi=150)
#     ax.bar([i for i in range(len(names))],abs(P[:,i]),zorder=5)
#     ax.set_xticks([i for i in range(len(names))])
#     ax.set_xticklabels(names,rotation=90)
#     ax.grid(ls=':')
#     plt.title('$\\lambda_{'+f'{i+1}' + '}$')
#     plt.show()
#     plt.close()

    

# names = [n.latex_name for _,n in ccm1.x.iterrows()]
plot_participation_factors(abs(P),names)


