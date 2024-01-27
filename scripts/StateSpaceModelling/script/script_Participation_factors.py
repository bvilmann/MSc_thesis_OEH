from CCM_class import ComponentConnectionMethod, StateSpaceSystem
from CCM_utils import plot_nyquist, plot_Zdq, plot_eigs, load_ccm, \
    load_init_conditions, state_space_simulation, x_plot, show_matrix, \
    complex_to_real_expanded, plot_participation_factors, \
        plot_participation_factor_bars,plot_participation_factor_bars_sample,\
            eigen_properties_to_latex,plot_participation_factor_bars_sample2
from NetworkLoader import NetworkClass, form_network_ssm
from pscad_data_reader import PowerSourceDataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from numpy.linalg import inv
import pandas as pd
from Plot_phasor_diagram_sss import plot_phasors

tab_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\tab'
plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
# %% ==================== ICWG ====================

# for scr in [1.5,2,3,4,5,6,7,8,9,10]:
for scr in [9.5]:

    # Get initial conditions
    net = NetworkClass(verbose=False, scr_OWF=10, scr_HSC1_icwg=scr*2, xr_HSC1_icwg=10, Pload=-1)
    init_cond = net.load_flow2init_cond('icwg')

    # net.lnd_icwg.draw()

    # ------------------------ NO VIRTUAL IMPEDANCE ------------------------ 
    # Load small-signal models
    ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_vi=0, L_v=0.07, R_v=0.03,
                                system_input=['v_gD', 'v_gQ'],
                                system_output=['i_oD', 'i_oQ'],
                                # system_output=['i_od','i_oq'],
                                ccm_name='ssm_ccm - full')

    ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10, x_vi=0, L_v=0.07, R_v=0.03,
                                system_input=['v_gD', 'v_gQ'],
                                system_output=['i_oD', 'i_oQ'],
                                # system_output=['i_od','i_oq'],
                                ccm_name='ssm_ccm - full')

    # Analyse system
    # plot_eigs(ccm1.A)
    Ad = block_diag(ccm1.A, ccm2.A)
    Bd = np.hstack([block_diag(ccm1.B, ccm2.B), np.zeros((ccm2.B.shape[0] * 2, 2))])
    Cd = np.vstack([block_diag(ccm1.C, ccm2.C), np.zeros((2, ccm2.C.shape[1] * 2))])
    Z = net.Z_icwg

    for i, mod in enumerate([Ad, Bd, Cd, Z]):
        print('ABCZ'[i], mod.shape)
        # show_matrix(mod,title='ABCZ'[i])

    # show_matrix(Bd @ Z,title='Bd @ Z')
    # show_matrix(Z @ Cd,title='Z @ Cd')

    A = form_network_ssm(Ad, Bd, Cd, Z)
    # plot_eigs(A,xlim=(-500,10),ylim=(-50,50))
    
    # ------------------------ NO VIRTUAL IMPEDANCE ------------------------ 
    # Load small-signal models
    ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_vi=1, L_v=0.07, R_v=0.03,
                                system_input=['v_gD', 'v_gQ'],
                                system_output=['i_oD', 'i_oQ'],
                                # system_output=['i_od','i_oq'],
                                ccm_name='ssm_ccm - full')

    
    ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10, x_vi=1, L_v=0.07, R_v=0.03,
                                system_input=['v_gD', 'v_gQ'],
                                system_output=['i_oD', 'i_oQ'],
                                # system_output=['i_od','i_oq'],
                                ccm_name='ssm_ccm - full')

    # Analyse system
    # plot_eigs(ccm1.A)
    Ad = block_diag(ccm1.A, ccm2.A)
    Bd = np.hstack([block_diag(ccm1.B, ccm2.B), np.zeros((ccm2.B.shape[0] * 2, 2))])
    Cd = np.vstack([block_diag(ccm1.C, ccm2.C), np.zeros((2, ccm2.C.shape[1] * 2))])
    Z = net.Z_icwg

    for i, mod in enumerate([Ad, Bd, Cd, Z]):
        print('ABCZ'[i], mod.shape)
        # show_matrix(mod,title='ABCZ'[i])

    # show_matrix(Bd @ Z,title='Bd @ Z')
    # show_matrix(Z @ Cd,title='Z @ Cd')

    A_vi = form_network_ssm(Ad, Bd, Cd, Z)

    plot_eigs(A_vi,xlim=(-500,10),ylim=(-50,50))
    

    #    
    names = [n.latex_name for _, n in ccm1.x.iterrows()]
    names = ['$\\Delta{' + n + '}_{,HSC1}$' for n in names] + ['$\\Delta{' + n + '}_{,HSC2}$' for n in names]

    lamb, Phi = np.linalg.eig(A)
    Psi = inv(Phi)
    P = Phi * Psi.T
    # plot_participation_factors(abs(P), names,save=None)

    lamb, Phi_vi = np.linalg.eig(A_vi)
    Psi_vi = inv(Phi_vi)
    P_vi = Phi_vi * Psi_vi.T
    # plot_participation_factors(abs(P_vi), names,save=None)


#%%
dominating_states = [
                    # r'$\Delta\var{i}{o}{dq},\Delta\var{i}{o,hpf}{dq},\Delta\var{\zeta}{c}{dq}$',
                     # r'$\Delta\var{i}{o}{dq},\Delta\var{i}{o,hpf}{dq},\Delta\var{\zeta}{c}{dq}$',
                     r'$\Delta\delta,\Delta\var{\zeta}{V}{dq}$',
                     # r'$\Delta P,\Delta Q$',
                     # r'$\Delta P,\Delta Q$',
                     ]  

eigen_properties_to_latex(A,tab_path,'SCR10',selection=[9],caption='Selected eigenvalue properties for bordering unstable system from $Z(SCR=9.5)$.',dominating_states=dominating_states)


# %%
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

lamb, Phi = np.linalg.eig(A)
Psi = inv(Phi)
P = Phi * Psi.T

lamb, Phi_vi = np.linalg.eig(A_vi)
Psi_vi = inv(Phi_vi)
P_vi = Phi_vi * Psi_vi.T

names = [n.latex_name for _, n in ccm1.x.iterrows()]
names = ['$\\Delta{' + n + '}_{,HSC1}$' for n in names] + ['$\\Delta{' + n + '}_{,HSC2}$' for n in names]

# names = [n.latex_name for _,n in ccm1.x.iterrows()]
savefile=f'{plot_path}\\P_heatmap_scr_sens.pdf'
plot_participation_factors(abs(P), names,save=savefile)

# Swap columns to match P without vi
P_vi[:,[8,9,10,11]] = P_vi[:,[10,11,8,9]]
savefile=f'{plot_path}\\P_heatmap_scr_sens_vi.pdf'
plot_participation_factors(abs(P_vi), names,save=savefile)

#%%
# savefile=f'{plot_path}\\P_bars_scr_sens.pdf'
# plot_participation_factor_bars(abs(P), names,save=savefile)
# savefile=f'{plot_path}\\P_bars_scr_sens_vi.pdf'
# plot_participation_factor_bars(abs(P_vi), names,save=savefile)
#%%
selected_modes = [2,4,8,18,19]
selected_modes = [8]
mode_labels = ['3,4','5,6','9,10','19','20','21','22']
mode_labels = ['9,10']
Pmax = max(abs(P).max(),abs(P_vi).max())

savefile=f'{plot_path}\\P_bars_scr_sens.pdf'
plot_participation_factor_bars_sample(abs(P), names,selected_modes,save=savefile,mode_labels=mode_labels,vmax = 0.3,figsize=(6,3))
savefile=f'{plot_path}\\P_bars_scr_sens_vi.pdf'
plot_participation_factor_bars_sample(abs(P_vi), names,selected_modes,save=savefile,mode_labels=mode_labels,vmax = 0.3,figsize=(6,3))


asdf

#%% =========== PLOT MODE SHAPES =========== 
modes = [9]
states = [1,2,3,12,13,14]
states_ = [s - 1 for s in states]

L = [n for i, n in enumerate(names) if i in states_]
scale = 0.75
fig, ax0 = plt.subplots(1,1,figsize=(4*scale,4*scale),sharey=True,sharex=True,dpi=150,subplot_kw={'projection': 'polar'})
P = []
P_vi = []
m = 9
for s in states:
    P.append((Phi[s-1,m-1].real,Phi[s-1,m-1].imag))
    # P_vi.append((Phi_vi[s-1,m-1].real,Phi_vi[s-1,m-1].imag))

plot_phasors(np.array(P), ['gold','C3','C0']*2, L,bbox_to_anchor=(0.05,1.05),ax=ax0,legend=True,legend_loc='upper right',line_styles=['-']*3+['--']*3)
# plot_phasors(np.array(P_vi), [f'C{i}' for i in range(4)], L,ax=ax1,legend=False)

ax0.set_ylabel('$\\lambda_{9,10}$\n\n',fontsize=7)

ax0.set_ylim(0,0.4)
fig.tight_layout()
fig.align_ylabels()
savefile=f'{plot_path}\\mode_shapes_9.pdf'
plt.savefig(savefile)
plt.show()


# #%% =========== PLOT MODE SHAPES =========== 
# modes = [19,20]
# modes = [19,20]
# states = [2,3,13,14]
# states_ = [s - 1 for s in states]

# L = [n for i, n in enumerate(names) if i in states_]
# scale = 0.9
# fig, axs = plt.subplots(2,2,figsize=(4*scale,4*scale),sharey=True,sharex=True,dpi=150,subplot_kw={'projection': 'polar'})
# for j, m in enumerate(modes):
#     ax0, ax1 = axs[j,0], axs[j,1]
#     P = []
#     P_vi = []
#     for s in states:
#         P.append((Phi[s-1,m-1].real,Phi[s-1,m-1].imag))
#         P_vi.append((Phi_vi[s-1,m-1].real,Phi_vi[s-1,m-1].imag))
    
#     plot_phasors(np.array(P), [f'C{i}' for i in range(4)], L,bbox_to_anchor=(0.05,.1),ax=ax0,legend=(False,True)[j == 0],legend_loc='upper right')
#     plot_phasors(np.array(P_vi), [f'C{i}' for i in range(4)], L,ax=ax1,legend=False)
    
#     ax0.set_ylabel('$\\lambda_{'+str(m)+'}$\n\n',fontsize=7)
    
#     if j == 1:
#         ax0.set_xlabel('Without VI',fontsize=7)
#         ax1.set_xlabel('With VI',fontsize=7)

# # fig.tight_layout()
# fig.align_ylabels()
# savefile=f'{plot_path}\\mode_shapes.pdf'
# plt.savefig(savefile)
# plt.show()

# #%% =========== PLOT MODE SHAPES =========== 
# modes = [21,22]
# states = [1,12]
# states_ = [s - 1 for s in states]

# L = [n for i, n in enumerate(names) if i in states_]
# scale = 0.9
# fig, axs = plt.subplots(1,1,figsize=(4*scale,4*scale),sharey=True,sharex=True,dpi=150,subplot_kw={'projection': 'polar'})
# for j, m in enumerate(modes):
#     ax0, ax1 = axs[j,0], axs[j,1]
#     P = []
#     P_vi = []
#     for s in states:
#         P.append((Phi[s-1,m-1].real,Phi[s-1,m-1].imag))
#         P_vi.append((Phi_vi[s-1,m-1].real,Phi_vi[s-1,m-1].imag))
    
#     plot_phasors(np.array(P), [f'C{i}' for i in range(4)], L,bbox_to_anchor=(0.05,.1),ax=ax0,legend=(False,True)[j == 0],legend_loc='upper right')
#     plot_phasors(np.array(P_vi), [f'C{i}' for i in range(4)], L,ax=ax1,legend=False)
    
#     ax0.set_ylabel('$\\lambda_{'+str(m)+'}$\n\n',fontsize=7)
    
#     if j == 1:
#         ax0.set_xlabel('Without VI',fontsize=7)
#         ax1.set_xlabel('With VI',fontsize=7)
#     ax0.set_ylim(0,0.8)
#     ax1.set_ylim(0,0.8)
# # fig.tight_layout()
# fig.align_ylabels()
# savefile=f'{plot_path}\\mode_shapes_21_22.pdf'
# # plt.savefig(savefile)
# plt.show()



# #%%
# selected_modes = [2,4,8,18,19]
# mode_labels = ['3,4','5,6','9,10','19','20']
# Pmax = max(abs(P).max(),abs(P_vi).max())

# for i, mode in enumerate(selected_modes):
    
#     P_df = pd.DataFrame({'Without VI':abs(P),'With VI':abs(P)},index=names)

# savefile=f'{plot_path}\\P_bars_scr_sens_df.pdf'
# plot_participation_factor_bars_sample2(P_df,selected_modes,save=savefile,mode_labels=mode_labels,vmax = Pmax)

