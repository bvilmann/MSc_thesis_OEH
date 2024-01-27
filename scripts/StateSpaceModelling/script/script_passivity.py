from CCM_class import ComponentConnectionMethod, StateSpaceSystem
from CCM_utils import plot_nyquist, plot_Zdq, plot_eigs, load_ccm, \
    load_init_conditions, state_space_simulation, x_plot, show_matrix, \
        complex_to_real_expanded,get_passivity
from NetworkLoader import NetworkClass, form_network_ssm
from pscad_data_reader import PowerSourceDataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from numpy.linalg import inv
import matplotlib as mpl
import pandas as pd
plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'

#%%
tf_dq_Lv = []
p_dict = {'ref':{},
          'Lv':{},
          'Rv':{},
          'Zv':{}}
suffix = '_resonance'
if suffix == '_resonance':    
    range_ = [0.05,0.1,0.2,0.4,0.8]
else:
    range_ = [0.05,0.1,0.2]
# range_ = [0.2,1,2]



net = NetworkClass(verbose=False,scr_OWF=10,Pload=-1)

init_cond = net.load_flow2init_cond('single')[1]


for Lv in [0]:
    tf_dq_Lv = []
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
    # plot_passivity(tf_dq_Lv,save=f'{plot_path}\\passivity.pdf')
    p = get_passivity(tf_dq_Lv)
    p_dict['ref'][round(Lv,4)] = p    

for Lv in range_:
    tf_dq_Lv = []
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
    # plot_passivity(tf_dq_Lv,save=f'{plot_path}\\passivity.pdf')
    p = get_passivity(tf_dq_Lv)
    p_dict['Lv'][round(Lv,4)] = p    

for Rv in range_:
    tf_dq_Rv = []
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
    p = get_passivity(tf_dq_Rv)
    p_dict['Rv'][round(Rv,4)] = p    

xr_vi = 2*np.pi*0.07/0.03
theta = np.arctan(xr_vi)

for Zv in range_:
    tf_dq_Zv = []
    if Zv == 0:
        lv = rv = 0
    else:
        zv = Zv*(np.cos(theta) + 1j*np.sin(theta))
        rv = zv.real
        lv = zv.imag

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

    p = get_passivity(tf_dq_Zv)
    p_dict['Zv'][round(Zv,4)] = p    

                
#%%
omega = np.logspace(-2, 5, 1000)  # Replace with your desired frequency range
dim=1
fig, ax = plt.subplots(1,1,dpi=150,figsize=(8*dim,4*dim))
if suffix == '_resonance':
    keys = ['ref','Lv']
else:
    keys = ['ref','Lv','Rv','Zv']

for key in keys:
    for i, (k,v) in enumerate(p_dict[key].items()):
        if key == 'ref':
            clr = 'k'
            ls = '--'
            label = 'Ref'
        elif key == 'Lv':
            clr = mpl.colormaps['Blues']((i + 1) / len(range_))
            ls = '-'
            label = '$L_v='+str(k)+'$'
        elif key == 'Rv':
            clr = mpl.colormaps['Greens']((i + 1) / len(range_))
            ls = '-'
            label = '$R_v='+str(k)+'$'
        elif key == 'Zv':
            clr = mpl.colormaps['Reds']((i + 1) / len(range_))
            ls = '-'
            label = '$|Z_v|='+str(k)+'$'

        ax.semilogx(omega / (2*np.pi),v,label=label,color=clr, ls = ls,zorder=(3,5)[key == 'ref'])

ax.set_yscale('symlog')  # Set the y-axis to symmetrical log scale
ax.set_ylim(-100,0.1)
ax.set(xlabel='Frequency [Hz]',ylabel='Passivity index')
ax.set_xlim(omega[0] / (2*np.pi),omega[-1] / (2*np.pi))
ax.grid(ls=':')
ax.axhline(0,color='k',lw=0.75)
   
ax.legend(ncol=1,bbox_to_anchor=(1,1),title='[pu]')
fig.tight_layout()
plt.savefig(f'{plot_path}\\passivity{suffix}.pdf')

#%%

# plot_passivity(tf_dq_Lv,save=f'{plot_path}\\passivity.pdf')

