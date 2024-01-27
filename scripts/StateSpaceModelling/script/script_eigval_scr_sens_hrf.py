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
ccm_model = 'ssm_ccm - full - filter'
network = 'icwg' # select 'icwog', 'icwg', 'icwg_mva'
suffix = '_overview'
additional = True
save_ = False
xr = 10
Pload = 1
Snom = 2
pu_norm = (Pload/Snom)**(-1)

scr_OWF = 10

Vbase = 300e3
Sbase = 1e9
Zbase = Vbase**2/Sbase

Lv = 0.07
Rv = 0.03

L_arm = 0.002

sens = {}
sens_vi = {}

exp_range = np.logspace(np.log10(1), np.log10(12), num=40)
# exp_range = np.logspace(np.log10(1), np.log10(12), num=4)

#%%

def Z_(scr=10,xr=10,V=300e3,Vbase=300e3,Sbase=1e9,S=0.5e9,normalize:bool=True):
    theta = np.arctan(xr)
    z = Vbase**2/(scr*Sbase)*(np.cos(theta)+1j*np.sin(theta))
    
    if normalize:
        z /= V**2/S

    return z

z = Z_(S=1e9)
z = Z_(S=1e9)
# print('scr=10,S=1e9:', 1/Z_(scr=10,S=1e9))
# print('scr=10,S=0.5e9: ',1/Z_(scr=10,S=0.5e9))
# print('scr=20,S=1e9',1/Z_(scr=20,S=1e9))
# print('scr=200,S=1e9:', Z_(scr=200,S=1e9,normalize=False))

# eigs_ = []
# for i in range(100):
#     eigs_.append(np.linalg.eigvals(net.complex_to_real_expanded(np.array([i*z]).reshape(1,1))))
# er = [e.real for e in eigs_]
# ei = [e.imag for e in eigs_]
# plt.scatter(er,ei)

#%%
cnt = 0
for scr in tqdm(exp_range):
# for scr in tqdm([1,8,10,11]):
# for scr in tqdm([10]):

    # Get initial conditions
    net = NetworkClass(verbose=False, scr_OWF=scr_OWF, scr_HSC1_icwg=scr*pu_norm, xr_HSC1_icwg=xr, Pload=-1*Pload)
    init_cond = net.load_flow2init_cond('icwg',normalize_currents=False)
    
    Z = getattr(net,f'Z_{network}')
    # Yc = getattr(net,f'Y_{network}_cmplx')

    # net.lnd_icwg.draw()

    # ------------------------ NO VIRTUAL IMPEDANCE ------------------------ 
    # Load small-signal models
    ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_vi=0, L_v=0, R_v=0, L_arm = L_arm,
                                system_input=['v_gD', 'v_gQ'],
                                system_output=['i_oD', 'i_oQ'],
                                # system_output=['i_od','i_oq'],
                                ccm_name=ccm_model)

    ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10, x_vi=0, L_v=0, R_v=0, L_arm = L_arm,
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

    sens[round(scr,3)]=lamb

    # ------------------------ WITH VIRTUAL IMPEDANCE ------------------------ 
    # Load small-signal models
    ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_vi=1, L_v=Lv, R_v=Rv, L_arm = L_arm,
                                system_input=['v_gD', 'v_gQ'],
                                system_output=['i_oD', 'i_oQ'],
                                # system_output=['i_od','i_oq'],
                                ccm_name=ccm_model)

    ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10,x_vi=1, L_v=Lv, R_v=Rv, L_arm = L_arm,
                                system_input=['v_gD', 'v_gQ'],
                                system_output=['i_oD', 'i_oQ'],
                                # system_output=['i_od','i_oq'],
                                ccm_name=ccm_model)

    # Analyse system
    # plot_eigs(ccm1.A)
    Ad = block_diag(ccm1.A, ccm2.A)
    Bd = np.hstack([block_diag(ccm1.B, ccm2.B), np.zeros((ccm2.B.shape[0] * 2, 2))])
    Cd = np.vstack([block_diag(ccm1.C, ccm2.C), np.zeros((2, ccm2.C.shape[1] * 2))])

    # for i, mod in enumerate([Ad, Bd, Cd, Z]):
    #     print('ABCZ'[i], mod.shape)
    # show_matrix(mod,title='ABCZ'[i])
    # show_matrix(Bd @ Z,title='Bd @ Z')
    # show_matrix(Z @ Cd,title='Z @ Cd')


    A_vi = form_network_ssm(Ad, Bd, Cd, Z)
   
    lamb = np.linalg.eigvals(A_vi)

    sens_vi[round(scr,3)]=lamb

    if cnt == 0:
        names = [n.latex_name for _, n in ccm1.x.iterrows()]
        names = ['$\\Delta{' + n + '}_{,HSC1}$' for n in names] + ['$\\Delta{' + n + '}_{,HSC2}$' for n in names]
        
        # names = [n.latex_name for _,n in ccm1.x.iterrows()]
        lamb, Phi = np.linalg.eig(A)
        print([(i+1,l) for i,l in enumerate(lamb) if l.real > 0 ])
        Psi = inv(Phi)
        P = Phi * Psi.T

        lamb_vi, Phi_vi = np.linalg.eig(A_vi)
        print([(i+1,l) for i,l in enumerate(lamb) if l.real > 0 ])
        Psi_vi = inv(Phi_vi)
        P_vi = Phi_vi * Psi_vi.T
        
        # plot_participation_factors(abs(P), names)
        # plot_participation_factors(abs(P_vi), names)
        # plot_participation_factor_bars(abs(P), names)
    cnt += 1

#%%

sens_ = {}
sens_vi_ = {}

if additional:
    for scr in tqdm([200]):
    # for scr in tqdm([1,3,12,30,200]):
        # Get initial conditions
        net = NetworkClass(verbose=False, scr_OWF=scr_OWF, scr_HSC1_icwg=scr*pu_norm, xr_HSC1_icwg=10, Pload=-1)
        init_cond = net.load_flow2init_cond('icwg')
        Z = getattr(net,f'Z_{network}')

        # net.lnd_icwg.draw()
    
        # ------------------------ NO VIRTUAL IMPEDANCE ------------------------ 
        # Load small-signal models
        ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_vi=0, L_v=0.07, R_v=0.03,
                                    system_input=['v_gD', 'v_gQ'],
                                    system_output=['i_oD', 'i_oQ'],
                                    # system_output=['i_od','i_oq'],
                                    ccm_name=ccm_model)
    
        ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10, x_net=1, x_vi=0, L_v=0.07, R_v=0.03,
                                    system_input=['v_gD', 'v_gQ'],
                                    system_output=['i_oD', 'i_oQ'],
                                    # system_output=['i_od','i_oq'],
                                    ccm_name=ccm_model)
    
        # Analyse system
        # plot_eigs(ccm1.A)
        Ad = block_diag(ccm1.A, ccm2.A)
        Bd = np.hstack([block_diag(ccm1.B, ccm2.B), np.zeros((ccm2.B.shape[0] * 2, 2))])
        Cd = np.vstack([block_diag(ccm1.C, ccm2.C), np.zeros((2, ccm2.C.shape[1] * 2))])
    
        # for i, mod in enumerate([Ad, Bd, Cd, Z]):
        #     print('ABCZ'[i], mod.shape)
        # show_matrix(mod,title='ABCZ'[i])
        # show_matrix(Bd @ Z,title='Bd @ Z')
        # show_matrix(Z @ Cd,title='Z @ Cd')
    
        A = form_network_ssm(Ad, Bd, Cd, Z)
       
        lamb = np.linalg.eigvals(A)
    
        sens_[round(scr,3)]=lamb
    
        # ------------------------ WITH VIRTUAL IMPEDANCE ------------------------ 
        # Load small-signal models
        ccm1, init_data1 = load_ccm(**init_cond[1], scr=0, xr=10, x_vi=1, L_v=0.07, R_v=0.03,
                                    system_input=['v_gD', 'v_gQ'],
                                    system_output=['i_oD', 'i_oQ'],
                                    # system_output=['i_od','i_oq'],
                                    ccm_name=ccm_model)
    
        ccm2, init_data2 = load_ccm(**init_cond[2], scr=0, xr=10,x_vi=1, L_v=0.07, R_v=0.03,
                                    system_input=['v_gD', 'v_gQ'],
                                    system_output=['i_oD', 'i_oQ'],
                                    # system_output=['i_od','i_oq'],
                                    ccm_name=ccm_model)
    
        # Analyse system
        # plot_eigs(ccm1.A)
        Ad = block_diag(ccm1.A, ccm2.A)
        Bd = np.hstack([block_diag(ccm1.B, ccm2.B), np.zeros((ccm2.B.shape[0] * 2, 2))])
        Cd = np.vstack([block_diag(ccm1.C, ccm2.C), np.zeros((2, ccm2.C.shape[1] * 2))])
    
        # for i, mod in enumerate([Ad, Bd, Cd, Z]):
        #     print('ABCZ'[i], mod.shape)
        # show_matrix(mod,title='ABCZ'[i])
        # show_matrix(Bd @ Z,title='Bd @ Z')
        # show_matrix(Z @ Cd,title='Z @ Cd')
    
        A_vi = form_network_ssm(Ad, Bd, Cd, Z)
       
        lamb = np.linalg.eigvals(A_vi)
    
        sens_vi_[round(scr,3)]=lamb



#%%
print('WITHOUT VIRTUAL IMPEDANCE')
for k,v in sens.items():

    if (v.real> 0).any():
        print(k,v[(v.real> 0)])
    else:
        pass
        # break
print('\nWITH VIRTUAL IMPEDANCE')
for k,v in sens_vi.items():

    if (v.real> 0).any():
        print(k,v[(v.real> 0)])
    else:
        # print()
        # break
        pass
   
# plt.plot(sens.keys(),[v.real.max() for k,v in sens.items()])
# plt.grid(ls=':')
    
#%%

# Example data: Replace this with your actual data
plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
if 'filter' in ccm_model:
    axins_view1,axins_loc1 = (-10,1,-50,50),(0.1,0.775,.3,.2)
    axins_view1_vi,axins_loc1 = (-50,1,-50,50),(0.1,0.775,.3,.2)
    axins_view2,axins_loc2 = (-1000,10,-100,100),(0.1,0.05,.3,.2)
    xlim = (-750,100)
    ylim = (-100,100)
    if suffix == '_overview':
        xlim = ylim = None
else:
    axins_view1,axins_loc1 =  (-10,1,-60,60),(0.3,0.6,.4,.3)
    axins_view1_vi,axins_loc1=(-50,1,-60,60),(0.3,0.6,.4,.3)
    axins_view2,axins_loc2 = (-800,10,-100,100),(0.35,0.1,.3,.3)
    xlim = (-750,500)
    ylim = (-1000,1000)

if not save_: 
    savefile=None
elif 'filter' in ccm_model:
    savefile=f'{plot_path}\\eigval_full_scr_sens_filter{suffix}.pdf'
else:
    savefile=f'{plot_path}\\eigval_full_scr_sens{suffix}.pdf'

plot_eigvals_dict('SCR',sens,save=savefile,y_offset=0,
                  insert_axis_args1=[axins_view1,axins_loc1],
                  insert_axis_kwargs1={'color':'black','ls':':'},
                  show_labels=[9,10,15,16],
                  # insert_axis_args2=[axins_view2,axins_loc2],
                  # insert_axis_kwargs2={'color':'white','ls':':'},
                  # hline = {'color':'red','vmin':2.5*2*np.pi,'vmax':6.5*2*np.pi},
                  hline = {'color':'red','vmin':2.0*2*np.pi,'vmax':6.25*2*np.pi},
                  # additional=sens_
                    xlim=xlim,
                    ylim=ylim,
                  )

if not save_: 
    savefile=None
elif 'filter' in ccm_model:
    savefile=f'{plot_path}\\eigval_full_scr_sens_vi_filter{suffix}.pdf'
else:
    savefile=f'{plot_path}\\eigval_full_scr_sens_vi{suffix}.pdf'

plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
plot_eigvals_dict('SCR',sens_vi,save=savefile,y_offset=200,
                  insert_axis_args1=[axins_view1_vi,axins_loc1],
                  insert_axis_kwargs1={'color':'black','ls':':'},
                  # insert_axis_args2=[axins_view2,axins_loc2],
                  # insert_axis_kwargs2={'color':'white','ls':':'},
                  show_labels=[9,10,15,16],
                  additional=sens_vi_,
                    xlim=xlim,
                    ylim=ylim,
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


