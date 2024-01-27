from pscad_data_reader import PowerSourceDataLoader
import matplotlib.pyplot as plt
import numpy as np
import plot_utils
import matplotlib.patches as mpatches
asdf
#%% ==================================== POWER SOURCE - MODEL VALIDATION ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'model_validation_power_source'
loader = PowerSourceDataLoader(file, directory=path)

dim = [6,4]
scale=1.5
fig, ax = plt.subplots(3,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
comp = r'power_source(OWF 1)'

# Voltage
cnt = 0
for i, signal in enumerate(['Vrms','v_ll']):
    clrs = ['black','blue','gold','red']
    alpha = [1,0.5,0.5,0.5]
    t, vals = loader.get_measurement_data(f'Main(0)\\power_source(OWF 1):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[0].plot(t,vals[:,j],label=['$V_{RMS}$','$V_{LL}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax[0].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1

# Current
cnt = 0
for i, signal in enumerate(['I_amp','i_abc']):
    clrs = ['black','blue','gold','red',]
    alpha = [1,0.5,0.5,0.5]
    t, vals = loader.get_measurement_data(f'Main(0)\\power_source(OWF 1):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[1].plot(t,vals[:,j],label=['$I_{ref}$','$I_{RMS}$','$I_{RMS}$'][i],color=clrs[cnt],zorder=(2,5)[cnt==0],alpha=alpha[cnt])
        else:
            ax[1].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1

# Power
cnt = 0
for i, signal in enumerate(['Pref','P','Qref','Q']):
    clrs = ['C0','lightblue','C3','pink']
    t, vals = loader.get_measurement_data(f'Main(0)\\power_source(OWF 1):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[2].plot(t,vals[:,j],label=['$P_{ref}$','$P_{meas}$','$Q_{ref}$','$Q_{meas}$'][i],color=clrs[cnt])
        else:
            ax[2].plot(t, vals[:, j], color=clrs[cnt])
        cnt += 1


# Determine the width of the widest y-tick label across all subplots
# max_width = max([max([tick.get_window_extent().width for tick in ax_.yaxis.get_ticklabels()]) for ax_ in ax])

for i, axis in enumerate(ax):
    axis.grid()
    axis.legend(loc='lower left',ncol=1)
    ax[i].set_ylabel(['Voltage [kV]','Current [kA]','Power [MVA]'][i])


ax[-1].set(xlim=(0,2),xlabel='$t$ [s]')

fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\model_validation_power_source.pdf')

plt.show()


#%% ==================================== POWER SOURCE - QUANTIZER ====================================

file = r'model_validation_quantizer'
loader = PowerSourceDataLoader(file, directory=path)

dim = [6,2]
scale=1.5
fig, ax = plt.subplots(1,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
comp = r'Main(0)'

# Voltage
cnt = 0
t, signal = loader.get_measurement_data(f'Main(0):signal1')
_, quant = loader.get_measurement_data(f'Main(0):quant1')

for i, (s,q)in enumerate(zip(signal.T,quant.T)):
    clrs = ['blue','gold','red']
    ax.plot(t, s, color=clrs[i],alpha=0.3)
    ax.step(t, q, color=clrs[i],alpha=1)

ax.grid()
ax.set_ylabel('Voltage [kV]')

ax.set(xlim=(0.2,0.24),xlabel='$t$ [s]')

ax.set(xticks=np.arange(0.2,0.24,.01))

fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\model_validation_quantizer.pdf')

plt.show()


#%% ==================================== DVF PF validation - CONTROL VALIDATION ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'DVF_PF1_1para_strong'
loader = PowerSourceDataLoader(file, directory=path)

dim = [6,5]
scale=1.5
fig, ax = plt.subplots(4,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
comp = r'power_source(OWF 1)'

ax0,ax[0] = plot_utils.insert_axis(ax[0], (1.24,1.26,-1.2,1.2), (0.25,0.2,1/3,0.6),box_kwargs=dict(color='C3', ls='--'),lw=1)
ax[0].annotate('', 
             xy=(1.24,0), 
             xytext=(1.0,0), 
             ha='center',va='center',
             fontsize=10,
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             # bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue")
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="white")
             )
ax3,ax[3] = plot_utils.insert_axis(ax[3], (1.24,1.41,0.95,1.15), (0.25,0.4,1/3,0.4),box_kwargs=dict(color='C3', ls='--'),lw=1)
ax[3].annotate('', 
             xy=(1.25,1), 
             xytext=(1.0,0.5), 
             ha='center',va='center',
             fontsize=10,
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             # bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue")
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="white")
             )

# Voltage
cnt = 0
for i, signal in enumerate(['Vrms','v_ll']):
    clrs = ['black','blue','gold','red']
    alpha = [1,0.25,0.25,0.25]
    scale = [1/66] + [1/(66*np.sqrt(2))]*3
    t, vals = loader.get_measurement_data(f'Main(0)\\power_source(OWF 1):{signal}')
    vals = scale[i]*vals
    for j in range(vals.shape[1]):
        if j == 0:
            ax[0].plot(t,vals[:,j],label=['$V_{RMS}^\\mathit{OWF}$','$V_{LL}^\\mathit{OWF}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
            ax0.plot(t,vals[:,j],label=['$V_{RMS}^\\mathit{OWF}$','$V_{LL}^\\mathit{OWF}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax[0].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
            ax0.plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1
cnt = 0
for i, signal in enumerate(['Vd_ref']):
    clrs = ['#006400']
    alpha = [1]
    t, vals = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(0)\control_DVF(0)\control_Q_Vac_Vdq(0):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[0].plot(t,vals[:,j],label=['$V_\\mathit{d,ref}^\\mathit{MMC}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
            ax0.plot(t,vals[:,j],label=['$V_\\mathit{d,ref}^\\mathit{MMC}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax[0].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
            ax0.plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1

# Frequency
cnt = 0
for i, signal in enumerate(['f_0','f_ref']):
    clrs = ['black','blue']
    alpha = [1,0.5,0.5,0.5]
    t, vals = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(0)\control_DVF(0)\control_P_P_theta(0):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[1].plot(t,vals[:,j],label=['$f_\\mathit{0}$','$f_\\mathit{ref}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax[1].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1

# Power
cnt = 0
for i, signal in enumerate(['Pac_meas_filt','Pac_ref']):
    clrs = ['black','#006400']
    alpha = [1,0.5,0.5,0.5]
    t, vals = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(0)\control_DVF(0)\control_P_P_theta(0):{signal}')
    vals *=1e3
    for j in range(vals.shape[1]):
        if j == 0:
            ax[2].plot(t,vals[:,j],label=['$P_\\mathit{meas}^\\mathit{MMC}$','$P_\\mathit{ref}^\\mathit{MMC}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax[2].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1

cnt = 0
for i, signal in enumerate(['Pref']):
    clrs = ['blue']
    alpha = [1,0.5,0.5,0.5]
    t, vals = loader.get_measurement_data(f'Main(0)\power_source(OWF 1):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[2].plot(t,vals[:,j],label=['$P_\\mathit{ref}^\\mathit{OWF}$','$P_\\mathit{ref}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax[2].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1


# Voltage
cnt = 0
for i, signal in enumerate(['Vd','Vd_ref']):
    clrs = ['black','#006400','grey','red']
    alpha = [1,1]
    t, vals = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(0)\control_DVF(0)\control_Q_Vac_Vdq(0):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[3].plot(t,vals[:,j],label=['$V_\\mathit{d,meas}$','$V_\\mathit{d,ref}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
            ax3.plot(t,vals[:,j],label=['$V_\\mathit{d,meas}$','$V_\\mathit{d,ref}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax[3].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
            ax3.plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1
cnt = 0
ax3twin = ax[3]
for i, signal in enumerate(['Vq','Vq_ref']):
    clrs = ['black','#006400','grey','red']
    alpha = [1,1]
    t, vals = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(0)\control_DVF(0)\control_Q_Vac_Vdq(0):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax3twin.plot(t,vals[:,j],label=['$V_\\mathit{q,meas}$','$V_\\mathit{q,ref}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax3twin.plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1

events = {
    0.05:{'text':'DC-link done\nramping up',
          'axis':[3],
          'xy1':(0.05,0.75),
          'xy2':(0.25,0.5),
          'ha':'center',
          'va':'center',
          },
    0.25:{'text':'$P_\\mathit{ref}^\\mathit{OWF}$=\n100 MW',
          'axis':[2,1],
          'xy1':(0.25,20),
          'xy2':(0.5,50),
          'ha':'center',
          'va':'center',
          },
    0.75:{'text':'$P_\\mathit{ref}^\\mathit{OWF}$=\n50 MW',
          'axis':[2,1],
          'xy1':(0.75,25),
          'xy2':(0.85,25),
          'ha':'left',
          'va':'center',
          },
    1.25:{'text':'',
          'axis':[3],
          'xy1':(1.25,0.85),
          'xy2':(1.6,0.7),
          'ha':'center',
          'va':'center',
          },
    1.4:{'text':'$V_\\mathit{d}^\\mathit{MMC}$=\n1.1 pu',
          'axis':[3],
          'xy1':(1.4,0.85),
          'xy2':(1.6,0.7),
          'ha':'center',
          'va':'center',
          'box':True,
          },
    1.5:{'text':'$Q_\\mathit{ref}^\\mathit{OWF}=$\n20 MVAr',
          'axis':[0],
          'xy1':(1.5,0),
          'xy2':(1.35,0),
          'ha':'center',
          'va':'center',
          # 'box':True,
          },
    1.75:{'text':'$P_\\mathit{ref}^\\mathit{MMC}$=\n50 MW',
          'axis':[1,2],
          'xy1':(1.75,49.95),
          'xy2':(1.65,49.95),
          'ha':'right',
          'va':'center',
          'box':True,
          },
    }
for t, event in events.items():
    for i in event['axis']:
        ax[i].axvline(t,color='red',lw=1)
    if 'arrow' in list():
        arrowstyle = event['axis']
    else:
        arrowstyle = '->'
    if 'box' in list(event.keys()):
        ax[event['axis'][0]].annotate(event['text'], 
                     xy=event['xy1'], 
                     xytext=event['xy2'], 
                     ha=event['ha'],va=event['va'],
                     fontsize=10,
                     arrowprops=dict(facecolor='black', arrowstyle=arrowstyle),
                     # bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue")
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="white")
                     )
    
    else:        
        ax[event['axis'][0]].annotate(event['text'], 
                     xy=event['xy1'], 
                     xytext=event['xy2'], 
                     ha=event['ha'],va=event['va'],
                     fontsize=10,
                     arrowprops=dict(facecolor='black', arrowstyle=arrowstyle))

# brace = mpatches.ConnectionPatch((1.25, 0.9), (1.4, 0.9), "data", "data",
#                                  arrowstyle="-[,widthA=0.5,widthB=0.5")
# brace = mpatches.FancyBboxPatch((1.25, 0.9), 1.4-1.25, 0.05, boxstyle="square,pad=0", mutation_scale=20, mutation_aspect=0.5)
# plt.gca().add_patch(brace)

       


for i, axis in enumerate(ax):
    axis.grid()
    axis.legend(loc=('lower right','lower right')[i==3],ncol=(1,2)[i in [1,3]])
    ax[i].set_ylabel(['Voltage [pu]','PF-control 1\nFrequency [Hz]','PF-control 1\nActive power [MW]','Vac-control\nVoltage [pu]'][i])


ax[-1].set(xlim=(0,2),xlabel='$t$ [s]')

fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\DVF_PF1_control_val.pdf')

plt.show()        
        
#%% ==================================== DVF 1para - CONTROL VALIDATION ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'DVF_PF1_1para_strong'
loader = PowerSourceDataLoader(file, directory=path)

dim = [6,4]
scale=1.5
fig, ax = plt.subplots(3,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
comp = r'power_source(OWF 1)'

# Voltage
cnt = 0
for i, signal in enumerate(['Vrms','v_ll']):
    clrs = ['black','blue','gold','red']
    alpha = [1,0.5,0.5,0.5]
    t, vals = loader.get_measurement_data(f'Main(0)\\power_source(OWF 1):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[0].plot(t,vals[:,j],label=['$V_{RMS}$','$V_{LL}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax[0].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1

# Voltage
cnt = 0
for i, signal in enumerate(['Vrms','v_ll']):
    clrs = ['black','blue','gold','red']
    alpha = [1,0.5,0.5,0.5]
    t, vals = loader.get_measurement_data(f'Main(0)\\power_source(OWF 1):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[0].plot(t,vals[:,j],label=['$V_{RMS}$','$V_{LL}$'][i],color=clrs[cnt],alpha=alpha[cnt],zorder=(2,5)[cnt==0])
        else:
            ax[0].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1

# Current
cnt = 0
for i, signal in enumerate(['I_amp','i_abc']):
    clrs = ['black','blue','gold','red',]
    alpha = [1,0.5,0.5,0.5]
    t, vals = loader.get_measurement_data(f'Main(0)\\power_source(OWF 1):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[1].plot(t,vals[:,j],label=['$I_{ref}$','$I_{RMS}$','$I_{RMS}$'][i],color=clrs[cnt],zorder=(2,5)[cnt==0],alpha=alpha[cnt])
        else:
            ax[1].plot(t, vals[:, j], color=clrs[cnt],alpha=alpha[cnt])
        cnt += 1

# Power
cnt = 0
for i, signal in enumerate(['Pref','P','Qref','Q']):
    clrs = ['C0','lightblue','C3','pink']
    t, vals = loader.get_measurement_data(f'Main(0)\\power_source(OWF 1):{signal}')
    for j in range(vals.shape[1]):
        if j == 0:
            ax[2].plot(t,vals[:,j],label=['$P_{ref}$','$P_{meas}$','$Q_{ref}$','$Q_{meas}$'][i],color=clrs[cnt])
        else:
            ax[2].plot(t, vals[:, j], color=clrs[cnt])
        cnt += 1


# Determine the width of the widest y-tick label across all subplots
# max_width = max([max([tick.get_window_extent().width for tick in ax_.yaxis.get_ticklabels()]) for ax_ in ax])

for i, axis in enumerate(ax):
    axis.grid()
    axis.legend(loc='lower left',ncol=1)
    ax[i].set_ylabel(['Voltage [kV]','Current [kA]','Power [MVA]'][i])


ax[-1].set(xlim=(0,2),xlabel='$t$ [s]')

fig.align_ylabels(ax)
fig.tight_layout()

# plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\DVF_PF1.pdf')

plt.show()


#%% ==================================== DVF 1para - CONTROL VALIDATION ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'DVF_PF1_2para_unstable2'
loader = PowerSourceDataLoader(file, directory=path)


import matplotlib.pyplot as plt
import mpl_toolkits
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib import cbook
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
def fun(x,y,E,scr,xr,P_sys,mode='V'):
    Z = E**2/(scr*P_sys)
    theta = np.arctan(xr)
    R, X = Z*np.cos(theta), Z*np.sin(theta)    
    # print(R,X,E,P,Q)
    if mode == 'P':
        V, Q = x,y        
        return [-(R*V**2-np.sqrt(-Q**2*(R**2+X**2)**2+(R**2+X**2)*(E**2-2*Q*X)*V**2-X**2*V**4))/(R**2+X**2),
                -(R*V**2+np.sqrt(-Q**2*(R**2+X**2)**2+(R**2+X**2)*(E**2-2*Q*X)*V**2-X**2*V**4))/(R**2+X**2)]
        
    elif mode =='Q':
        P, V = x,y
        return [(-V**2*X + np.sqrt(-P**2*R**4 - 2*P**2*R**2*X**2 - P**2*X**4 - 2*P*R**3*V**2 - 2*P*R*V**2*X**2 - R**2*V**4 + R**2*V**2*E**2 + V**2*E**2*X**2))/(R**2 + X**2),
                (-V**2*X - np.sqrt(-P**2*R**4 - 2*P**2*R**2*X**2 - P**2*X**4 - 2*P*R**3*V**2 - 2*P*R*V**2*X**2 - R**2*V**4 + R**2*V**2*E**2 + V**2*E**2*X**2))/(R**2 + X**2)]
        
    elif mode =='V':
        P, Q = x,y
        return [np.sqrt(-(R*P+X*Q)+E**2/2+np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q))),
                np.sqrt(-(R*P+X*Q)+E**2/2-np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q)))]

            

x = 3

n= 2001

p = np.concatenate([np.linspace(0, x, n)])
q = np.concatenate([np.linspace(-x, x*.1, n)])
P, Q = np.meshgrid(p, q)
P, Q = P.flatten(), Q.flatten()
P, Q = np.meshgrid(p, q)

scrs = [3]
xrs = [3]
cnt = 0

fig,ax = plt.subplots(1,1,figsize =(6, 6),dpi=200,sharex=True,sharey=True)
V = np.flip(np.array(fun(P, Q, 1,scrs[cnt],xrs[cnt],1,mode='V'))[0,:].reshape(n,n),axis=0)

q_ = np.array(fun(p,1,1,scr,xr,1,mode='Q'))[0]
cmap = plt.cm.rainbow #or any other colormap

extent = [0,x,-x,x*.1]

grey = ax.imshow(V,cmap = plt.cm.Greys,norm = mpl.colors.Normalize(vmin=0, vmax=1.4),label='$\\mathcal{C}$', extent=extent)    
bwr = ax.imshow(np.where((V>=0.95) & (1.05>=V),V,np.nan),cmap = plt.cm.bwr,label='$\\mathcal{F}$', extent=extent)    


# Creating figure
ax.set_xlabel('$P_{OWF}$ [pu]')
ax.set_ylabel('$Q_{OWF}$ [pu]')

# plot 
non_nan_indices = ~np.isnan(q_)
# Filter p and q_ using the non-nan indices
p_clean = np.linspace(0, x, n)[non_nan_indices]
q_clean = q_[non_nan_indices]
last_idx = -10
ax.plot(p_clean[:last_idx],q_clean[:last_idx],zorder=10,color='black',ls='--')

p_sim = 205.02176/100
q_sim = -258.2177/100
ax.scatter([p_sim],[q_sim],color='gold')

mask = t<=11
t, Psim = loader.get_measurement_data(f'Main(0):P')
t, Qsim = loader.get_measurement_data(f'Main(0):Q')
Psim = Psim[mask]
Qsim = Qsim[mask]

ax.plot(Psim,Qsim,zorder=10,color='black',ls='--')


ax.grid(ls=':')
cnt+=1
# ax[-1,-1].legend()
# plt.show()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\PQV_scr3_xr3.pdf', bbox_inches="tight")

#%% ==================================== DVF 1para - CONTROL VALIDATION ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'DVF_PF1_2para_unstable2'
loader = PowerSourceDataLoader(file, directory=path)
fig,ax = plt.subplots(1,1,figsize =(6, 6),dpi=200,sharex=True,sharey=True)
t, Psim = loader.get_measurement_data(f'Main(0):P')
t, f_a2= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(1)\control_P_P_theta(1):f_ref')
t, f_a1 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(0)\control_P_P_theta(0):f_ref')
mask = t<=11

ax.plot(Psim[mask],f_a2[mask])
ax.plot(Psim[mask],f_a1[mask])

#%% ==================================== DVF - LOADSTEP - ANGLE STUDIES ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'DVF_PF1_2para_loadstep_kp_eq'
loader = PowerSourceDataLoader(file, directory=path)
dim = [6,6]
scale=1.5
fig, ax = plt.subplots(2,2,dpi=150,figsize=[scale*d_ for d_ in dim])
t, wt1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(2)\control_P_P_theta(2):wt')
t, wt2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(1)\control_P_P_theta(1):wt')

ax[0,0].scatter(np.cos(wt1),np.sin(wt1))
ax[0,1].scatter(np.cos(wt2),np.sin(wt2))

ax10 = ax[1,0].twinx()
ax11 = ax[1,1].twinx()
ax[1,0].hist(wt1)
ax[1,1].hist(wt2)

# ax10.hist(np.sqrt(np.cos(wt1)**2+np.sin(wt1)**2))
# ax11.hist(np.sqrt(np.cos(wt1)**2+np.sin(wt1)**2),alpha=0.5)

for i in range(2):        
    ax[0,i].set(aspect='equal')
    for j in range(2):        
        ax[i,j].grid(ls=':')

#%% ==================================== DVF - LOADSTEP ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'DVF_PF1_2para_loadstep_kp_eq'
loader = PowerSourceDataLoader(file, directory=path)
dim = [6,4]
scale=1.5
fig, ax = plt.subplots(2,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
ax1,ax[1] = plot_utils.insert_axis(ax[1], (1.55,1.7,50.1125,50.125), (1.25/2,0.15,1/3,.375),box_kwargs=dict(color='magenta', ls='-'),lw=2)
ax2 = ax[1].twinx()

t, P = loader.get_measurement_data(f'Main(0):P_owf1')
t, P1 = loader.get_measurement_data(f'Main(0):P_A1')
t, P2 = loader.get_measurement_data(f'Main(0):P_A2')
t, f_a1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(2)\control_P_P_theta(2):f_ref')
t, f_a2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(1)\control_P_P_theta(1):f_ref')
t, wt1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(2)\control_P_P_theta(2):wt')
t, wt2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(1)\control_P_P_theta(1):wt')

ax[0].plot(t,P,color='black',alpha=1,zorder=10,label='$P_\\mathit{OWF}$')
ax[0].plot([0,0.25,0.25,1,1,2],[0,0,100,100,200,200],color='black',alpha=0.25,zorder=10,label='$P_\\mathit{OWF,ref}$')
ax[0].plot(t,P1,color='C0',alpha=1,zorder=10,label='$P_\\mathit{HSC1,m_p=0.05}$')
ax[0].plot(t,P2,color='C3',alpha=1,zorder=10,label='$P_\\mathit{HSC2,m_p=0.05}$')
ax[0].plot(t,P1+P2,color='grey',ls=':',alpha=1,zorder=11)

ax[1].plot(t,f_a1,color='C0',alpha=1,zorder=10,label='$f_\\mathit{HSC1,m_p=0.05}$')
ax[1].plot(t,f_a2,color='C3',alpha=1,zorder=10,label='$f_\\mathit{HSC2,m_p=0.05}$')
ax1.plot(t,f_a1,color='C0',alpha=1,zorder=10,label='$f_\\mathit{HSC1,m_p=0.05}$')
ax1.plot(t,f_a2,color='C3',alpha=1,zorder=10,label='$f_\\mathit{HSC2,m_p=0.05}$')
# theta = np.where(wt2-wt1<=-1,wt2-wt1+2*np.pi,wt2-wt1)
# ax2.plot(t,theta,color='black',alpha=1,zorder=3,label='$\\theta=\\omega_\\mathrm{A2}t-\\omega_\\mathrm{A1}t$')

file = r'DVF_PF1_2para_loadstep_kp_var'
loader = PowerSourceDataLoader(file, directory=path)
t, P = loader.get_measurement_data(f'Main(0):P_owf1')
t, P1 = loader.get_measurement_data(f'Main(0):P_A1')
t, P2 = loader.get_measurement_data(f'Main(0):P_A2')
t, f_a1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(2)\control_P_P_theta(2):f_ref')
t, f_a2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(1)\control_P_P_theta(1):f_ref')
t, wt1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(2)\control_P_P_theta(2):wt')
t, wt2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(1)\control_P_P_theta(1):wt')
# theta = np.where(wt2-wt1<=-1,wt2-wt1+2*np.pi,wt2-wt1)
# ax2.plot(t,theta,color='grey',ls='--',alpha=1,zorder=3,label='$\\theta=\\omega_\\mathrm{A2}t-\\omega_\\mathrm{A1}t$')

ax[0].plot(t,P1+P2,color='grey',ls='--',alpha=1,zorder=11,label='$P_\\mathit{HSC1}+P_\\mathit{HSC2}$')
ax[0].plot([0,0.25,0.25,1,1,2],[0,0,50,50,50,50],color='purple',alpha=0.5,zorder=10,label='$P_\\mathit{HSC1,HSC2,ref}$')
ax[0].plot(t,P1,color='lightblue',ls='--',alpha=1,zorder=10,label='$P_\\mathit{HSC1,m_p=0.075}$')
ax[0].plot(t,P2,color='pink',ls='--',alpha=1,zorder=10,label='$P_\\mathit{HSC2,m_p=0.025}$')

ax[1].plot(t,f_a1,color='lightblue',ls='--',alpha=1,zorder=10,label='$f_\\mathit{HSC1,m_p=0.075}$')
ax[1].plot(t,f_a2,color='pink',ls='--',alpha=1,zorder=10,label='$f_\\mathit{HSC2,m_p=0.025}$')

ax[-1].set(xlim=(0,2),xlabel='$t$ [s]')
ax[0].set(ylim=(-10,250))
ax[1].set(ylim=(49.8,50.25))
for i in range(2):
    ax[i].grid(ls=':')
    ax[i].set_ylabel(['Active Power [MW]','Frequency [Hz]'][i])
    ax[i].legend(loc='upper left',fontsize=12,ncol=2)

# ax2.legend(loc='upper right',fontsize=10,ncol=1)

# BOX
ax[1].annotate('', 
             xy=(1.625,50.11), 
             xytext=(1.595,50.035), 
             ha='center',va='center',
             fontsize=10,
             arrowprops=dict(facecolor='black', arrowstyle='<-'),
             # bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue")
             # bbox=dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="white")
             )



ax1.axvline(1.593825+0.02,color='black',zorder=15,lw=0.7)
ax1.axvline(1.593825+0.04,color='black',zorder=15,lw=0.7)
ax1.text(*(1.593825+0.03,50.1185),'$\Delta t=$\n20 ms',ha='center',va='center',zorder=20,fontsize=7)
ax1.annotate('', 
              xy=(1.593825+0.019,50.117), 
              xytext=(1.593825+0.041,50.117), 
              ha='center',va='center',
              fontsize=10,
              arrowprops=dict(facecolor='black', arrowstyle='<->',lw=0.7),
              # bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue")
              # bbox=dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="white")
              )

fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\DVF1_loadstep.pdf', bbox_inches="tight")




#%% ==================================== DVF - LOAD STEP - STABLE LIMIT-CYCLE ====================================
# path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
# file = r'DVF_PF1_2para_loadstep_kp_eq'
# file = r'DVF_PF1_2para_loadstep_limit'
# loader = PowerSourceDataLoader(file, directory=path)
# dim = [6,4]
# scale=1.5
# fig, ax = plt.subplots(2,1,dpi=150,figsize=[scale*d_ for d_ in dim])

# t, P = loader.get_measurement_data(f'Main(0):P_owf1')
# t, P1 = loader.get_measurement_data(f'Main(0):P_A1')
# t, P2 = loader.get_measurement_data(f'Main(0):P_A2')
# t, f_a1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(2)\control_P_P_theta(2):f_ref')
# t, f_a2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(1)\control_P_P_theta(1):f_ref')

# N=6
# dt = 0.02
# colors = plt.cm.rainbow(np.linspace(0, 1, N))

# for i in range(N):
#     t0 = 1.7 + dt*i
#     T = 1
#     mask = (t>=1.3) & (t<=1.5+0.02*T) 
#     mask = (t>=t0)& (t<=t0+dt*T) 
#     p1 = P1[mask]
#     p2 = P2[mask]
#     f1 = f_a1[mask]
#     f2 = f_a2[mask]
#     # ax[0].plot(t[mask],P[mask],color='black',alpha=1,zorder=10,label='$P_\\mathit{OWF}$')
#     # ax[0].plot([0,0.25,0.25,1,1,2],[0,0,100,100,200,200],color='black',alpha=0.25,zorder=10,label='$P_\\mathit{OWF,ref}$')
#     # ax.plot(p1,f2,color='C3',alpha=(i+1)/N,color=colors[i],zorder=10,label='$P_\\mathit{A1,K_p=0.05}$')
#     ax[0].plot(p1,p2,color=colors[i],zorder=10,label='$t\\in'+f'[{t0},{t0+dt*T}]'+'$ s')
#     ax[0].scatter([p1[0]],[p2[0]],color=colors[i],marker='o',zorder=10+i)
#     ax[0].scatter([p1[-1]],[p2[-1]],color='black',marker='x',zorder=11+i+1,lw=0.75)

# ax1 = ax[1].twinx()


# mask = (t>=1.7)& (t<=1.7+dt*T*N) 
# t_ = t[mask]
# p1 = P1[mask]
# p2 = P2[mask]
# f1 = f_a1[mask]
# f2 = f_a2[mask]
# ax[1].plot(t_,p1,color='C0')
# ax1.plot(t_,p2,color='C3')


# ax[1].set(xlabel='$t$ [s]',xlim=(1.7,1.7+dt*T*N))
# ax[1].set_ylabel('$P_\\mathrm{A1}$ [MW]',color='C0')
# ax1.set_ylabel('$P_\\mathrm{A2}$ [pu]',color='C3')
# ax1.tick_params(axis='y', labelcolor='C3')
# ax[1].tick_params(axis='y', labelcolor='C0')
# ax[0].set(xlabel='$P_\\mathrm{A1}$ [MW]')
# first_legend = ax[0].legend(loc='upper right',fontsize=10,ncol=2)

# # Add the legend manually to the current Axes.
# ax[0].add_artist(first_legend)
# ax[0].set_ylabel('$P_{A2}$ [pu]')
# for i in range(2):
#     ax[i].grid(ls=':')

# # Creating custom legend with scatter markers
# from matplotlib.lines import Line2D
# legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cycle start', 
#                           markerfacecolor='k', markersize=8),
#                    Line2D([0], [0], marker='x', color='k', label='Cycle end', 
#                           markerfacecolor='k', markersize=8)]

# # Create the second legend and add the artist manually.
# ax[0].legend(handles=legend_elements, loc='lower left',fontsize=10,framealpha=0.9)

# # Creating custom legend with scatter markers
# from matplotlib.lines import Line2D
# legend_elements = [Line2D([0], [0], color='C0', label='$P_\\mathit{A1,K_p=0.05}$'),
#                    Line2D([0], [0], color='C3', label='$P_\\mathit{A2,K_p=0.05}$')]
# ax1.legend(handles=legend_elements, loc='center right',fontsize=10)


# fig.align_ylabels(ax)
# fig.tight_layout()

# plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\DVF1_loadstep_limit_cycle.pdf', bbox_inches="tight")


#%% ==================================== DVF - LOAD STEP - STABLE LIMIT-CYCLE ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
# file = r'DVF_PF1_2para_loadstep_kp_eq'
file = r'DVF_PF1_2para_flatrun_limit'
loader = PowerSourceDataLoader(file, directory=path)
dim = [6,4]
scale=1.5
fig, ax = plt.subplots(2,1,dpi=150,figsize=[scale*d_ for d_ in dim])

t, P = loader.get_measurement_data(f'Main(0):P_owf1')
t, P1 = loader.get_measurement_data(f'Main(0):P_A1')
t, P2 = loader.get_measurement_data(f'Main(0):P_A2')
t, f_a1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(2)\control_P_P_theta(2):f_ref')
t, f_a2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(1)\control_P_P_theta(1):f_ref')

N=6
dt = 0.02*3
colors = plt.cm.rainbow(np.linspace(0, 1, N))

for i in range(N):
    t0 = 1.7 + dt*i
    T = 1
    mask = (t>=1.3) & (t<=1.5+0.02*T) 
    mask = (t>=t0)& (t<=t0+dt*T) 
    p1 = P1[mask]
    p2 = P2[mask]
    f1 = f_a1[mask]
    f2 = f_a2[mask]
    # ax[0].plot(t[mask],P[mask],color='black',alpha=1,zorder=10,label='$P_\\mathit{OWF}$')
    # ax[0].plot([0,0.25,0.25,1,1,2],[0,0,100,100,200,200],color='black',alpha=0.25,zorder=10,label='$P_\\mathit{OWF,ref}$')
    # ax.plot(p1,f2,color='C3',alpha=(i+1)/N,color=colors[i],zorder=10,label='$P_\\mathit{A1,K_p=0.05}$')
    ax[0].plot(np.sqrt(p1*p2),np.sqrt(p1**2),color=colors[i],zorder=5,label='$t\\in'+f'[{round(t0,2)},{round(t0+dt*T,2)}]'+'$ s')
    # ax[0].scatter([p1[0]],[f2[0]],color=colors[i],marker='o',zorder=10+i)
    # ax[0].scatter([p1[-1]],[f2[-1]],color='black',marker='x',zorder=11+i+1,lw=0.75)

ax1 = ax[1].twinx()

v_t0 = 1.7258
ax[1].axvline(v_t0,color='k',lw=.75)
ax[1].axvline(v_t0+0.02*3,color='k',lw=.75)

mask = (t>=1.7)& (t<=1.7+dt*T*N) 
t_ = t[mask]
p1 = P1[mask]
p2 = P2[mask]
f1 = f_a1[mask]
f2 = f_a2[mask]
ax[1].plot(t_,p1,color='C0')
ax1.plot(t_,p2,color='C3')

ax[1].set(xlabel='$t$ [s]',xlim=(1.7,1.7+dt*T*N))
ax[1].set_ylabel('$P_\\mathrm{A1}$ [MW]',color='C0')
ax1.set_ylabel('$P_\\mathrm{A2}$ [pu]',color='C3')
ax1.tick_params(axis='y', labelcolor='C3')
ax[1].tick_params(axis='y', labelcolor='C0')
ax[0].set(xlabel='$\\sqrt{P_\\mathrm{A1}P_\\mathrm{A2}}$ [MW]')
ax[0].set_ylabel('$\\sqrt{{P_{A1}}^2}$ [pu]')

ax[0].legend(loc='lower left',fontsize=10,ncol=2)

# Add the legend manually to the current Axes.
# ax[0].add_artist(first_legend)
for i in range(2):
    ax[i].grid(ls=':')

# # Creating custom legend with scatter markers
# from matplotlib.lines import Line2D
# legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cycle start', 
#                           markerfacecolor='k', markersize=8),
#                    Line2D([0], [0], marker='x', color='k', label='Cycle end', 
#                           markerfacecolor='k', markersize=8)]

# Create the second legend and add the artist manually.
# ax[0].legend(handles=legend_elements, loc='lower left',fontsize=10,framealpha=0.9)

# Creating custom legend with scatter markers
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='C0', label='$P_\\mathit{A1,K_p=0.05}$'),
                   Line2D([0], [0], color='C3', label='$P_\\mathit{A2,K_p=0.05}$')]
ax1.legend(handles=legend_elements, loc='center right',fontsize=10)


fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\DVF1_flatrun_limit_cycle.pdf', bbox_inches="tight")


#%% ==================================== DVF 1para - CONTROL VALIDATION - REDISPATCH ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'DVF_PF1_2para_redispatch'
loader = PowerSourceDataLoader(file, directory=path)
dim = [6,4]
scale=1.5
fig, ax = plt.subplots(2,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])

t, P = loader.get_measurement_data(f'Main(0):P_owf1')
t, P1 = loader.get_measurement_data(f'Main(0):P_A1')
t, P2 = loader.get_measurement_data(f'Main(0):P_A2')
t, f_a1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(2)\control_P_P_theta(2):f_ref')
t, f_a2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(1)\control_P_P_theta(1):f_ref')
    
ax[0].plot(t,P,color='black',alpha=1,zorder=10,label='$P_\\mathit{OWF}$')
ax[0].plot([0,0.25,0.25,1,1,2],[0,0,100,100,200,200],color='black',alpha=0.25,zorder=10,label='$P_\\mathit{OWF,ref}$')
ax[0].plot([0,0.25,0.25,1,1,2],[0,0,50,50,100,100],color='purple',alpha=0.5,zorder=10,label='$P_\\mathit{HSC1,HSC2,ref}$')
ax[0].plot(t,P1,color='C0',alpha=1,zorder=10,label='$P_\\mathit{HSC1,m_p=0.075}$')
ax[0].plot(t,P2,color='C3',alpha=1,zorder=10,label='$P_\\mathit{HSC2,m_p=0.025}$')
ax[0].plot(t,P1+P2,color='grey',ls=':',alpha=1,zorder=11)

ax[1].plot([0,2],[50]*2,color='black',alpha=1,zorder=10,label='$f_0$')
ax[1].plot(t,f_a1,color='C0',alpha=1,zorder=10,label='$f_\\mathit{HSC1,m_p=0.075}$')
ax[1].plot(t,f_a2,color='C3',alpha=1,zorder=10,label='$f_\\mathit{HSC2,m_p=0.025}$')

ax[-1].set(xlim=(0,2),xlabel='$t$ [s]')
ax[0].set(ylim=(-10,250))
ax[1].set(ylim=(49.8,50.3))

for i in range(2):
    ax[i].grid(ls=':')
    ax[i].set_ylabel(['Active Power [MW]','Frequency [Hz]'][i])
    ax[i].legend(loc='upper left',fontsize=12,ncol=(2,1)[i==1])


fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\DVF1_redispatch.pdf', bbox_inches="tight")

#%% ==================================== DVF 1para - CONTROL VALIDATION - REDISPATCH ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'sec_reg'
loader = PowerSourceDataLoader(file, directory=path)
dim = [6,4]
scale=1.5
fig, ax = plt.subplots(2,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])

t, P = loader.get_measurement_data(f'Main(0):P_owf1')
t, P1 = loader.get_measurement_data(f'Main(0):P_A1')
t, P2 = loader.get_measurement_data(f'Main(0):P_A2')
# t, f_a1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(2)\control_P_P_theta(2):f_ref')
# t, f_a2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(1)\control_P_P_theta(1):f_ref')
t, f1 = loader.get_measurement_data(f'Main(0):x_HSC1')
t, f2 = loader.get_measurement_data(f'Main(0):x_HSC2')

t, phi = loader.get_measurement_data(f'Main(0):phi')

    
ax[0].plot(t,P,color='black',alpha=1,zorder=10,label='$P_\\mathit{OWF}$')
ax[0].plot([0,0.6,0.6,1,1,4],[0,0,1e3,1e3,1e3,1e3],color='black',alpha=0.25,zorder=10,label='$P_\\mathit{OWF,ref}$')
ax[0].plot([0,0.6,0.6,1,1,4],[0,0,400,400,400,400],color='purple',alpha=0.5,zorder=10,label='$P_\\mathit{HSC1,HSC2,ref}$')
ax[0].plot(t,P1,color='C0',alpha=1,zorder=10,label='$P_\\mathit{HSC1}$')
ax[0].plot(t,P2,color='C3',alpha=1,zorder=10,label='$P_\\mathit{HSC2}$')

ax[1].plot([0,4],[50]*2,color='black',alpha=1,zorder=10,label='$f_0$')
ax[1].plot(t,f1[:,-1],color='C0',alpha=1,zorder=10,label='$f_\\mathit{HSC1}$')
ax[1].plot(t,f2[:,-1],color='C3',alpha=1,zorder=10,label='$f_\\mathit{HSC2}$')

# ax[2].plot(t,phi,color='C3',alpha=1,zorder=10,label='$\\phi=\\phi_\\mathit{HSC1}-\\phi_\\mathit{HSC2}$')
# ax[2].plot([0,4],[0]*2,color='black',alpha=1,zorder=10)

ax[-1].set(xlim=(0,4),xlabel='$t$ [s]')

for i in range(2):
    ax[i].grid(ls=':')
    ax[i].set_ylabel(['Active Power [MW]','Frequency [Hz]','Angle [deg]'][i])
    ax[i].legend(loc='lower right',fontsize=11,ncol=(6,3)[i==1])
    # ax[i].axvline(1.5,color='k',ls='--',lw=0.75)


fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\test5_sec_reg.pdf', bbox_inches="tight")


#%% ==================================== DVF 1para - CONTROL VALIDATION - Forced power-sharing ====================================
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'DVF_PF1_2para_unstable2'
loader = PowerSourceDataLoader(file, directory=path)
dim = [6,4]
scale=1.5
fig, ax = plt.subplots(2,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])

# t, P = loader.get_measurement_data(f'Main(0):P_owf1')
t, P1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(1)\control_P_P_theta(1):Pac_meas_filt')
t, P2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(0)\control_P_P_theta(0):Pac_meas_filt')
t, P1ref= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(1)\control_P_P_theta(1):Pac_ref')
t, P2ref = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(0)\control_P_P_theta(0):Pac_ref')
t, f_a1= loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A1)\control_DVF(1)\control_P_P_theta(1):f_ref')
t, f_a2 = loader.get_measurement_data(f'Main(0)\MMC_DVF_PF1(A2)\control_DVF(0)\control_P_P_theta(0):f_ref')
    
# ax[0].plot(t,P,color='black',alpha=1,zorder=10,label='$P_\\mathit{OWF}$')
ax[0].plot(t,P1*1e3,color='C0',alpha=1,zorder=10,label='$P_\\mathit{HSC1,K_p=0.05}$')
ax[0].plot(t,P2*1e3,color='C3',alpha=1,zorder=10,label='$P_\\mathit{HSC2,K_p=0.05}$')
ax[0].plot(t,P1ref*1e3,color='lightblue',alpha=1,zorder=10,label='$P_\\mathit{HSC1,ref,K_p=0.05}$')
ax[0].plot(t,P2ref*1e3,color='pink',alpha=1,label='$P_\\mathit{HSC2,ref,K_p=0.05}$')
ax[0].plot(t,(P2ref+P1ref)*1e3/2,color='purple',alpha=0.5)
ax[0].text((21-12.5)/2+12.5,300,'Mean active power reference of A1 & A2',color='purple',alpha=0.5,va='bottom',ha='center',fontsize=10)

# ax[0].plot([0,0.25,0.25,1,1,2],[0,0,50,50,50,50],color='purple',alpha=0.5,zorder=10,label='$P_\\mathit{A1,A2,ref}$')
# ax[0].plot(t,P1+P2,color='grey',ls=':',alpha=1,zorder=11)

ax[1].plot([0,21],[50]*2,color='black',alpha=1,zorder=10,label='$f_{n}$')
ax[1].plot(t,f_a1,color='C0',alpha=1,zorder=10,label='$f_\\mathit{HSC1,K_p=0.05}$')
ax[1].plot(t,f_a2,color='C3',alpha=1,zorder=10,label='$f_\\mathit{HSC2,K_p=0.05}$')

ax[-1].set(xlim=(0,21),xlabel='$t$ [s]')
for i in range(2):
    ax[i].grid(ls=':')
    ax[i].set_ylabel(['Active Power [MW]','Frequency [Hz]'][i])
    ax[i].legend(loc=('lower left','lower left')[i==1],fontsize=10,ncol=(2,1)[i==1])

fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\DVF1_forced_power_sharing.pdf', bbox_inches="tight")

#%%
fig, ax = plt.subplots(1,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
dim = [6,2]
scale=1.5
mask = ((t>=0.25) & (t<=11)) & (P2ref.ravel() > 0)
ax.plot(t[mask],-P1[mask]/P2ref[mask],color='C0',label='$P_\\mathit{A1,meas/ref}$')
ax.plot(t[mask],P2[mask]/P2ref[mask],color='C3',label='$P_\\mathit{A2,meas/ref}$')
ax.plot(t[mask],(P2[mask]-P1[mask])/P2ref[mask]/2,color='grey',label='$P_\\mathit{mean,meas/ref}$')
ax.grid(ls=':')
ax.axhline(0.5,color='k',label='$P_\\mathit{ideal,meas/ref}$')

ax.set(xlim=(0,11),xlabel='$t$ [s]',ylim=(0,0.6))
ax.set_ylabel(['Active Power [MW]','Frequency [Hz]'][i])
ax.legend(loc='lower right',fontsize=10,ncol=2)

fig.align_ylabels(ax)
fig.tight_layout()


plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\DVF1_forced_power_sharing_ratio.pdf', bbox_inches="tight")


#%% =========================== VIRTUAL IMPEDANCE - MITIGATING INSTABILITY =========================== 
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'virtual_impedance_PoC'
loader = PowerSourceDataLoader(file, directory=path)

dim = [6,8]
scale=1.6
fig, ax = plt.subplots(6,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
ax0 = ax[0].twinx()
ax1 = ax[1].twinx()

# --- PLOTTING --- 
# voltages
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Vdq_meas',t0=0.5)
ax[0].plot(t,vals[:,0],label='$v_\\mathit{HSC_1}^d$',color='C3',alpha=1,zorder=4)
ax0.plot(t,vals[:,1],label='$v_\\mathit{HSC_1}^q$',color='C0',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):Vdq_meas',t0=0.5)
ax[0].plot(t,vals[:,0],label='$v_\\mathit{HSC_2}^d$',color='C3',ls=':',alpha=1,zorder=4)
ax0.plot(t,vals[:,1],label='$v_\\mathit{HSC_2}^q$',ls=':',color='C0',alpha=1,zorder=3)
# currents
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Idq_meas',t0=0.5)
ax[1].plot(t,vals[:,0],label='$i_\\mathit{HSC_1}^d$',color='C3',alpha=1,zorder=3)
ax1.plot(t,vals[:,1],label='$i_\\mathit{HSC_1}^q$',color='C0',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):Idq_meas',t0=0.5)
ax[1].plot(t,vals[:,0],label='$i_\\mathit{HSC_2}^d$',color='C3',ls=':',alpha=1,zorder=3)
ax1.plot(t,vals[:,1],label='$i_\\mathit{HSC_2}^q$',ls=':',color='C0',alpha=1,zorder=3)

# active power
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Pac_ref',t0=0.5)
ax[2].plot(t,vals[:],label='$P_\\mathit{HSC_1}^\\mathit{ref}$',color='C0',ls=':',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Pac_meas_filt',t0=0.5)
ax[2].plot(t,vals[:],label='$P_\\mathit{HSC_1}$',color='C0',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):Pac_ref',t0=0.5)
ax[2].plot(t,vals[:],label='$P_\\mathit{HSC_2}^\\mathit{ref}$',color='C3',ls=':',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):Pac_meas_filt',t0=0.5)
ax[2].plot(t,vals[:],label='$P_\\mathit{HSC_2}$',color='C3',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\power_source(OWF 1):Pref',t0=0.5)
ax[2].plot(t,vals[:]/1000,label='$P_\\mathit{OWF_1}^\\mathit{ref}$',color='k',ls=':',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\power_source(OWF 1):P',t0=0.5)
ax[2].plot(t,vals[:]/1000,label='$P_\\mathit{OWF_1}$',color='k',alpha=1,zorder=3)

# reactive power
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Qac_ref',t0=0.5)
ax[3].plot(t,vals[:],label='$Q_\\mathit{HSC_1}^\\mathit{ref}$',color='C0',ls=':',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Qac_meas_filt',t0=0.5)
ax[3].plot(t,vals[:],label='$Q_\\mathit{HSC_1}$',color='C0',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):Qac_ref',t0=0.5)
ax[3].plot(t,vals[:],label='$Q_\\mathit{HSC_2}^\\mathit{ref}$',color='C3',ls=':',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):Qac_meas_filt',t0=0.5)
ax[3].plot(t,vals[:],label='$Q_\\mathit{HSC_2}$',color='C3',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\power_source(OWF 1):Qref',t0=0.5)
ax[3].plot(t,vals[:]/1000,label='$P_\\mathit{OWF_1}^\\mathit{ref}$',color='k',ls=':',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\power_source(OWF 1):Q',t0=0.5)
ax[3].plot(t,vals[:]/1000,label='$P_\\mathit{OWF_1}$',color='k',alpha=1,zorder=3)

# frequency
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):f',t0=0.5)
ax[4].plot(t,vals[:],label='$f_\\mathit{HSC_1}$',color='C0',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):f',t0=0.5)
ax[4].plot(t,vals[:],label='$f_\\mathit{HSC_2}$',color='C3',alpha=1,zorder=3)

# angle
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Vdq_meas',t0=0.5)
delta1 = np.arctan(vals[:,1],vals[:,0])*180/np.pi 
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):vdq_vi',t0=0.5)
delta2 = np.arctan(vals[:,1],vals[:,0])*180/np.pi 
ax[5].plot(t,delta1,label='$\\phi_\\mathit{HSC_1}^\\mathit{meas}=\\mathrm{tan}^{-1}\\left(\\frac{v_\\mathit{meas}^q}{v_\\mathit{meas}^d}\\right)$',color='C0',alpha=1,zorder=3)
ax[5].plot(t,-delta2,label='$\\phi_\\mathit{HSC_1}^\\mathit{vi}=\\mathrm{tan}^{-1}\\left(\\frac{v_\\mathit{vi}^q}{v_\\mathit{vi}^d}\\right)$',color='C3',alpha=1,zorder=3)

t, vals = loader.get_measurement_data('Main(0):phi',t0=0.5)
# delta1 = np.arctan(vals[:,1],vals[:,0])*180/np.pi 
# t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):vdq_vi',t0=0.5)
# delta2 = np.arctan(vals[:,1],vals[:,0])*180/np.pi 
# ax[5].plot(t,delta1,label='$\\phi_\\mathit{HSC_2,meas}=\\mathrm{tan}^{-1}\\left(\\frac{v_\\mathit{meas}^q}{v_\\mathit{meas}^d}\\right)$',ls=':',color='C0',alpha=1,zorder=3)
# ax[5].plot(t,-delta2,label='$\\phi_\\mathit{HSC_2,vi}=\\mathrm{tan}^{-1}\\left(\\frac{v_\\mathit{vi}^q}{v_\\mathit{vi}^d}\\right)$',ls=':',color='C3',alpha=1,zorder=3)
ax[5].plot(t,vals,label='$\\phi=\\phi_\\mathit{HSC_2}^v-\\phi_\\mathit{HSC_1}^v$',ls='-',color='k',alpha=1,zorder=3)

# t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):vdq_vi',t0=0.5)
# t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):Vdq_meas',t0=0.5)
# t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):vdq_vi',t0=0.5)
# t, phi_owf = loader.get_measurement_data('Main(0)\power_source(OWF 1):theta_i',t0=0.5)
# phi_ = phi2-phi1
# phi_ = np.where(phi_<-3,phi_+2*np.pi,phi_)
# phi_*=180/np.pi
# ax[5].plot(t,phi_,label='$\\phi_\\mathit{HSC_2}-\\phi_\\mathit{HSC_1}$',color='C0',alpha=1,zorder=3)

# phi_ = phi_owf-phi1
# phi_ = np.where(phi_<-3,phi_+2*np.pi,phi_)
# phi_*=180/np.pi
# ax[5].plot(t,phi_,label='$\\phi_\\mathit{OWF_1}-\\phi_\\mathit{HSC_1}$',color='k',alpha=1,zorder=3)

# FORMATTING
shared_axes = [0,1]
fontsize=11
for i, axis in enumerate(ax):
    axis.grid(ls=':',color=('grey','C3')[i in shared_axes])    
    if i in shared_axes:        
        axis.legend(loc='upper left',ncol=2,fontsize=fontsize)
    elif i == 5:
        axis.legend(loc='upper right',ncol=1,fontsize=fontsize)        
    else:
        axis.legend(loc='lower right',ncol=6,fontsize=fontsize)

    ax[i].set_ylabel(['$v_d$ [pu]','$i_d$ [pu]','Active Power [pu]','Reactive power [pu]','Frequency [Hz]','Angle [deg]'][i],color=('k','C3')[i in shared_axes])
    if i in shared_axes:
        ax[i].tick_params(axis='y', colors='C3')
        
    if i == 0:
        ax0.set_ylabel('$v_q$ [pu]',color='C0')
        ax0.tick_params(axis='y', colors='C0')
        ax0.legend(loc='upper right',ncol=2,fontsize=fontsize)
        ax0.grid(ls=':',color='C0')
    if i == 1:  
        ax1.set_ylabel('$i_q$ [pu]',color='C0')
        ax1.tick_params(axis='y', colors='C0')
        ax1.legend(loc='upper right',ncol=2,fontsize=fontsize)
        ax1.grid(ls=':',color='C0')
        
ax[-1].set(xlim=(0,2),xlabel='$t$ [s]')

fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\virtual_impedance.pdf')

plt.show()


#%% =========================== VIRTUAL IMPEDANCE - MITIGATING INSTABILITY =========================== 
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'virtual_impedance_PoC'
loader = PowerSourceDataLoader(file, directory=path)

dim = [5,2]
scale=1.6
t1 = 1.275
fig, ax = plt.subplots(1,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])

# --- PLOTTING --- 
# active power
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Pac_ref',t0=0.5,t1=t1)
ax.plot(t,vals[:],label='$P_\\mathit{HSC_1}^\\mathit{ref}$',color='C0',ls=':',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Pac_meas_filt',t0=0.5,t1=t1)
ax.plot(t,vals[:],label='$P_\\mathit{HSC_1}$',color='C0',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):Pac_ref',t0=0.5,t1=t1)
ax.plot(t,vals[:],label='$P_\\mathit{HSC_2}^\\mathit{ref}$',color='C3',ls=':',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):Pac_meas_filt',t0=0.5,t1=t1)
ax.plot(t,vals[:],label='$P_\\mathit{HSC_2}$',color='C3',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\power_source(OWF 1):Pref',t0=0.5,t1=t1)
ax.plot(t,vals[:]/1000,label='$P_\\mathit{OWF_1}^\\mathit{ref}$',color='k',ls=':',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\power_source(OWF 1):P',t0=0.5,t1=t1)
ax.plot(t,vals[:]/1000,label='$P_\\mathit{OWF_1}$',color='k',alpha=1,zorder=3)

# FORMATTING
shared_axes = [0,1]
fontsize=11

ax.set_ylabel('Active Power [pu]',color='k')
ax.legend(loc='lower center',ncol=3,fontsize=fontsize)
ax.grid(ls=':')
        
ax.set(xlim=(0,t1-0.5),xlabel='$t$ [s]')

fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\virtual_impedance_unstable.pdf')

plt.show()

#%% =========================== VIRTUAL IMPEDANCE - MITIGATING INSTABILITY =========================== 
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'virtual_impedance_PoC'
loader = PowerSourceDataLoader(file, directory=path)

dim = [6,4]
scale=1.6
fig, ax = plt.subplots(2,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
t0 = 1.15
# --- PLOTTING --- 
# voltages
t, vals = loader.get_measurement_data('Main(0):v_lg_A1',t0=1.15,t1=1.4)
ax[0].plot(t+t0-0.5,vals[:,0],label='$v_\\mathit{lg,HSC_1}^a$',color='C3',alpha=1,zorder=4)
ax[0].plot(t+t0-0.5,vals[:,1],label='$v_\\mathit{lg,HSC_1}^b$',color='C0',alpha=1,zorder=3)
ax[0].plot(t+t0-0.5,vals[:,2],label='$v_\\mathit{lg,HSC_1}^c$',color='C1',alpha=1,zorder=3)

t, vals = loader.get_measurement_data('Main(0):i_abc_A1',t0=1.15,t1=1.4)
ax[1].plot(t+t0-0.5,vals[:,0],label='$i_\\mathit{HSC_1}^a$',color='C3',alpha=1,zorder=4)
ax[1].plot(t+t0-0.5,vals[:,1],label='$i_\\mathit{HSC_1}^b$',color='C0',alpha=1,zorder=3)
ax[1].plot(t+t0-0.5,vals[:,2],label='$i_\\mathit{HSC_1}^c$',color='C1',alpha=1,zorder=3)


# FORMATTING
shared_axes = []
fontsize=11
for i, axis in enumerate(ax):
    axis.grid(ls=':',color=('grey','C3')[i in shared_axes])    
    if i in shared_axes:        
        axis.legend(loc='upper left',ncol=2,fontsize=fontsize)
    elif i == 5:
        axis.legend(loc='upper left',ncol=1,fontsize=fontsize)        
    else:
        axis.legend(loc='lower right',ncol=6,fontsize=fontsize)

    ax[i].set_ylabel(['Voltage [kV]','Current [kA]','Active Power [GW]','Reactive power [GVAr]','Frequency [Hz]','Angle [$^\circ$]'][i],color=('k','C3')[i in shared_axes])
    if i in shared_axes:
        ax[i].tick_params(axis='y', colors='C3')
       
        
ax[-1].set(xlim=(0+t0-0.5,0.25+t0-0.5),xlabel='Time [s]')

fig.align_ylabels(ax)
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\virtual_impedance_PoC_phase.pdf')



# plt.show()


#%% =========================== VIRTUAL IMPEDANCE - MITIGATING INSTABILITY =========================== 
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
dim = [4,6]
scale=1.6
fig, ax = plt.subplots(5,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
for i in range(1,9):
    file = f'loop_validation_{i}'

    loader = PowerSourceDataLoader(file, directory=path)

    # --- PLOTTING --- 
    # voltages
    t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Vrms_pu',t0=0.5)
    alpha = (1,0.125)[(vals[:,0] >=1.1).any()]
    ax[0].plot(t,vals[:,0],label=i,alpha=alpha,zorder=i)

    # active power
    t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Pac_meas',t0=0.5)
    ax[1].plot(t,vals[:,0],alpha=alpha,zorder=i)

    # reactive power
    t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Qac_meas',t0=0.5)
    ax[2].plot(t,vals[:,0],alpha=alpha,zorder=i)

    # currents
    t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Idq_meas',t0=0.5)
    ax[3].plot(t,vals[:,0],alpha=alpha,zorder=i)
    ax[4].plot(t,vals[:,1],alpha=alpha,zorder=i)
    
# FORMATTING
shared_axes = [0,1]
fontsize=11
for i, axis in enumerate(ax):
    axis.grid(ls=':',color=('grey','C3')[i in shared_axes])    
    axis.legend(loc='center right',ncol=2,fontsize=fontsize)

    ax[i].set_ylabel(['$v_\\mathit{HSC_1}^\\mathit{RMS}$ [pu]','$P_\\mathit{HSC_1}^\\mathit{meas}$ [pu]','$Q_\\mathit{HSC_1}^\\mathit{meas}$ [pu]','$i_\\mathit{HSC_1}^\\mathit{d}$ [pu]','$i_\\mathit{HSC_1}^\\mathit{q}$ [pu]','Angle [deg]'][i],color=('k','C3')[i in shared_axes])
    if i in shared_axes:
        ax[i].tick_params(axis='y', colors='C3')
        
    if i == 0:
        ax0.set_ylabel('$v_q$ [pu]',color='C0')
        ax0.tick_params(axis='y', colors='C0')
        ax0.legend(loc='upper right',ncol=2,fontsize=fontsize)
        ax0.grid(ls=':',color='C0')
    if i == 1:  
        ax1.set_ylabel('$i_q$ [pu]',color='C0')
        ax1.tick_params(axis='y', colors='C0')
        ax1.legend(loc='upper right',ncol=2,fontsize=fontsize)
        ax1.grid(ls=':',color='C0')
        
    ax[i].set(xlim=(0,2),xlabel='$t$ [s]',ylim=(0,1.1))

fig.align_ylabels(ax)
fig.tight_layout()

# plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\loop_validation.pdf')

plt.show()

#%%
from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fftshift
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = f'small_signal_SCR_pert'

loader = PowerSourceDataLoader(file, directory=path)

# --- PLOTTING --- 
# voltages
t, i_owf = loader.get_measurement_data('Main(0):i_abc_owf1',t0=0.)
t, i_HSC = loader.get_measurement_data('Main(0):i_abc_A1',t0=0.0)

i_owf = i_owf[:,0]
i_HSC = i_HSC[:,0]

fig, ax = plt.subplots(2,1,dpi=150,figsize=(10, 6),sharex=True)
ax0 = ax[0].twinx()
ax1 = ax[1].twinx()

ax0.set_ylabel('   (a)',rotation=0)
ax1.set_ylabel('   (b)',rotation=0)

# Parameters
fs = 1/(t[1]-t[0])


frequencies, times, Sxx = spectrogram(i_HSC, fs)
frequencies_owf, times_owf, Sxx_owf = spectrogram(i_owf, fs)

spectrogram_mesh  = ax[0].pcolormesh(times, frequencies, 10*np.log10((Sxx)), shading='gouraud',cmap='Blues')
# spectrogram_mesh  = ax[0].pcolormesh(times, frequencies, 10*np.log10((Sxx)),cmap='Blues')


# divider = make_axes_locatable(ax[0])
cax = fig.add_axes([0.6815, 0.8125, 0.2, 0.025])  # Adjust these values as needed

# Create the colorbar in the created axes
cbar = plt.colorbar(spectrogram_mesh, cax=cax, orientation='horizontal')
cbar.set_label('Intensity [dB]')

# Move the colorbar to the top
cax.xaxis.set_ticks_position("bottom")
cax.xaxis.set_label_position("top")
# Change color of the tick labels

for label in cax.xaxis.get_ticklabels():
    label.set_color('white')  # Change 'red' to your desired color

cax.tick_params(axis='x', colors='white')


cbar.ax.xaxis.label.set_color('white')  # Change 'blue' to your desired color

# Apply Welch's method to estimate power spectral density
f, pd = welch(i_HSC[t>=3], fs, nperseg=80*80)

pd_offset = 0.05+2
ax[0].plot((pd)+pd_offset,f,color='C3')

# Detect peaks in the power spectral density
peaks, _ = find_peaks(pd, height=0.004)  # You can adjust the 'height' parameter as needed

# Plot the detected peaks
ax[0].plot(pd[peaks]+pd_offset,f[peaks], 'x', color='k')

# Optionally, you can annotate the peaks with their frequencies
for peak in peaks:
    ax[0].annotate(f"{pd[peak]:.2f} "+'kA$^2$'+f"/Hz @ {f[peak]:.2f} Hz", (pd[peak]+pd_offset,f[peak]), textcoords="offset points", xytext=(10,0), ha='left',va='center',color='white')

ax[1].annotate("$Z\\left(\\mathit{SCR}=10,\\mathit{X/R}=10\\right)$", (2.5,-2.5), textcoords="offset points", xytext=(0,0), ha='center',va='center',color='k')
ax[1].annotate("$Z\\left(\\mathit{SCR}=11,\\mathit{X/R}=10\\right)$", (3.5,-2.5), textcoords="offset points", xytext=(0,0), ha='center',va='center',color='k')

ax[0].set_ylim(0, 100)  # Limit frequency axis to 100 Hz for better visibility
ax[1].set_xlim(2, 5.8)  # Limit frequency axis to 100 Hz for better visibility
ax[1].axvline(3,color='k',lw=0.75,zorder=5)  # Limit frequency axis to 100 Hz for better visibility
ax[1].plot(t,i_HSC,color='C0',zorder=4,label='$i_\\mathit{HSC}^a$')
ax[1].plot(t,i_owf,color='C2',alpha =0.25,label='$i_\\mathit{OWF}^a$')
ax[1].legend(loc='upper left',ncols=2)
# plt.tight_layout()  # This will adjust spacing to minimize overlap and make everything fit

ax[0].set_ylabel('Frequency [Hz]')
ax[1].set_xlabel('Time [sec]')
ax[1].set_ylabel('Current [kA]')
# ax[0].set_title('Spectrogram')

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\ss_SCR_perturbation_.pdf')

plt.show()

#

#%% =========================== VIRTUAL IMPEDANCE - MITIGATING INSTABILITY =========================== 
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'virtual_impedance_1GW'
loader = PowerSourceDataLoader(file, directory=path)

dim = [4,6]
scale=1.6
fig, ax = plt.subplots(4,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
ax0 = ax[0].twinx()
ax1 = ax[1].twinx()
ax2 = ax[2].twinx()
ax3 = ax[3].twinx()
t0 = 1.64
t1 = 1.74
# --- PLOTTING --- 
# voltages
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(A1)\control_DVF(2):Vdq_meas',t0=t0,t1=t1)
ax[0].plot(t+t0,vals[:,0],label='$v_\\mathit{HSC_1}^d$',color='C0',alpha=1,zorder=3)
ax0.plot(t+t0,vals[:,1],color='C3',alpha=1,zorder=3)
# currents
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(A1)\control_DVF(2):Idq_meas',t0=t0,t1=t1)
ax[1].plot(t+t0,vals[:,0],color='C0',alpha=1,zorder=3)
ax1.plot(t+t0,vals[:,1],color='C3',alpha=1,zorder=3)
# currents
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(A1)\control_DVF(2):vdq_vi',t0=t0,t1=t1)
ax[2].plot(t+t0,vals[:,0],color='C0',alpha=1,zorder=3)
ax2.plot(t+t0,vals[:,1],color='C3',alpha=1,zorder=3)

t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(A1)\control_DVF(2):theta',t0=t0,t1=t1)
ax[3].plot(t+t0,vals[:],color='C0',alpha=1,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(A2)\control_DVF(1):theta',t0=t0,t1=t1)
ax3.plot(t+t0,vals[:],label='$i_\\mathit{HSC_1}^q$',color='C3',alpha=1,zorder=3)

ax[-1].set(xlim=(t0,t1))

# FORMATTING
shared_axes = [0,1,2,3]
fontsize=11
for i, axis in enumerate(ax):
    axis.grid(ls=':',color=('grey','C0')[i in shared_axes])    

    ax[i].set_ylabel(['$v_\\mathit{HSC_1,meas}^d$ [pu]','$i_\\mathit{HSC_1,meas}^d$ [pu]','$v_\\mathit{HSC_1,vi}^d$ [pu]','$\\theta_\\mathit{HSC_2}$ [pu]','Frequency [Hz]','Angle [$^\circ$]'][i],color=('k','C0')[i in shared_axes])
    if i in shared_axes:
        ax[i].tick_params(axis='y', colors='C0')
        
    if i == 0:
        ax0.set_ylabel('$v_\\mathit{HSC_1,meas}^q$ [pu]',color='C3')
        ax0.tick_params(axis='y', colors='C3')
        ax0.grid(ls=':',color='C3')
    if i == 1:  
        ax1.set_ylabel('$i_\\mathit{HSC_1,meas}^q$ [pu]',color='C3')
        ax1.tick_params(axis='y', colors='C3')
        ax1.grid(ls=':',color='C3')
    if i == 2:  
        ax2.set_ylabel('$v_\\mathit{HSC_1,vi}^q$ [pu]',color='C3')
        ax2.tick_params(axis='y', colors='C3')
        ax2.grid(ls=':',color='C3')
    if i == 3:  
        ax3.set_ylabel('$\\theta_\\mathit{HSC_2}$ [pu]',color='C3')
        ax3.tick_params(axis='y', colors='C3')
        ax3.grid(ls=':',color='C3')

diff_signal = np.diff(vals.flatten(), prepend=0)

# Find indices where the difference signal goes from negative to positive
min_indices = np.where(np.sign(diff_signal) == -1,True,False)

# Since np.diff reduces the length of the array by 1, add 1 to align with the original array
# Now, min_indices corresponds to the indices right after the minima in the original waveform

# Filter to get the time instances of the minima
min_times = t.flatten()[min_indices] + t0

for t in min_times:
    ax[-1].axvline(x=t, color='grey', linestyle='-', ymin=0, ymax=4.37, clip_on=False)


fig.align_ylabels(ax)
fig.align_ylabels([ax0,ax1,ax2,ax3])
fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\virtual_impedance_theta_perturbations.pdf')

plt.show()

#%% =========================== VIRTUAL IMPEDANCE - MITIGATING INSTABILITY =========================== 
path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'
file = r'virtual_impedance_1GW'
file = r'virtual_impedance_noatan'
loader = PowerSourceDataLoader(file, directory=path)

dim = [6,6]
scale=1.6
fig, ax = plt.subplots(4,1,dpi=150,sharex=True,figsize=[scale*d_ for d_ in dim])
ax0 = ax[0].twinx()
ax1 = ax[1].twinx()
ax2 = ax[2].twinx()
ax3 = ax[3].twinx()
t0 = 1.68
t1 = 1.78

# --- PLOTTING --- 
# voltages
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Vdq_meas',t0=t0,t1=t1)
ax[0].plot(t+t0,vals[:,0],label='$v_\\mathit{HSC_1}^d$',color='C0',alpha=0.25,ls='-',zorder=3)
ax0.plot(t+t0,vals[:,1],color='C3',alpha=0.25,ls='--',zorder=3)
# currents
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Idq_meas',t0=t0,t1=t1)
ax[1].plot(t+t0,vals[:,0],color='C0',alpha=0.25,ls='--',zorder=3)
ax1.plot(t+t0,vals[:,1],color='C3',alpha=0.25,ls='--',zorder=3)
# currents
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):vdq_vi',t0=t0,t1=t1)
ax[2].plot(t+t0,vals[:,0],color='C0',alpha=0.25,ls='--',zorder=3)
ax2.plot(t+t0,vals[:,1],color='C3',alpha=0.25,ls='--',zorder=3)

t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):theta',t0=t0-0.01,t1=t1-0.01)
ax[3].plot(t+t0,vals[:],color='C0',alpha=0.25,ls='--',zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):theta',t0=t0-0.01,t1=t1-0.01)
ax3.plot(t+t0,vals[:],label='$i_\\mathit{HSC_1}^q$',color='C3',alpha=0.25,ls='--',zorder=3)

diff_signal = np.diff(vals.flatten(), prepend=0)

# Find indices where the difference signal goes from negative to positive
min_indices = np.where(np.sign(diff_signal) == -1,True,False)

# Since np.diff reduces the length of the array by 1, add 1 to align with the original array
# Now, min_indices corresponds to the indices right after the minima in the original waveform

# Filter to get the time instances of the minima
min_times = t.flatten()[min_indices] + t0

for t in min_times:
    ax[-1].axvline(x=t, color='grey', linestyle='-', ymin=0, ymax=4.37, clip_on=False)


# # --- PLOTTING --- 
# file = r'virtual_impedance_noatan_4pi'
# loader = PowerSourceDataLoader(file, directory=path)
# # voltages
# t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Vdq_meas',t0=t0,t1=t1)
# ax[0].plot(t+t0,vals[:,0],label='$v_\\mathit{HSC_1}^d$',color='C0',alpha=0.25,ls='--',zorder=3)
# ax0.plot(t+t0,vals[:,1],color='C3',alpha=0.25,ls='--',zorder=3)
# # currents
# t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Idq_meas',t0=t0,t1=t1)
# ax[1].plot(t+t0,vals[:,0],color='C0',alpha=0.25,ls='--',zorder=3)
# ax1.plot(t+t0,vals[:,1],color='C3',alpha=0.25,ls='--',zorder=3)
# # currents
# t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):vdq_vi',t0=t0,t1=t1)
# ax[2].plot(t+t0,vals[:,0],color='C0',alpha=0.25,ls='--',zorder=3)
# ax2.plot(t+t0,vals[:,1],color='C3',alpha=0.25,ls='--',zorder=3)

# t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):theta',t0=t0,t1=t1)
# ax[3].plot(t+t0,vals[:],color='C0',alpha=0.25,ls='--',zorder=3)
# t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):theta',t0=t0,t1=t1)
# ax3.plot(t+t0,vals[:],label='$i_\\mathit{HSC_1}^q$',color='C3',alpha=0.25,ls='--',zorder=3)

# --- PLOTTING --- 
file = r'virtual_impedance_noatan_4pi'
loader = PowerSourceDataLoader(file, directory=path)
# voltages
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Vdq_meas',t0=t0,t1=t1)
ax[0].plot(t+t0,vals[:,0],label='$v_\\mathit{HSC_1}^d$',color='C0',ls='--',alpha=0.75,zorder=3)
ax0.plot(t+t0,vals[:,1],color='C3',ls='-',alpha=0.75,zorder=3)
# currents
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Idq_meas',t0=t0,t1=t1)
ax[1].plot(t+t0,vals[:,0],color='C0',ls='-',alpha=0.75,zorder=3)
ax1.plot(t+t0,vals[:,1],color='C3',ls='-',alpha=0.75,zorder=3)
# currents
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):vdq_vi',t0=t0,t1=t1)
ax[2].plot(t+t0,vals[:,0],color='C0',ls='-',alpha=0.75,zorder=3)
ax2.plot(t+t0,vals[:,1],color='C3',ls='-',alpha=0.75,zorder=3)

t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):theta',t0=t0-0.01,t1=t1-0.01)
ax[3].plot(t+t0,vals[:],color='C0',ls='-',alpha=0.75,zorder=3)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):theta',t0=t0-0.01,t1=t1-0.01)
ax3.plot(t+t0,vals[:],label='$i_\\mathit{HSC_1}^q$',color='C3',alpha=0.75,zorder=3)

# --- PLOTTING --- 
file = r'virtual_impedance_noatan_infpi'
loader = PowerSourceDataLoader(file, directory=path)
# voltages
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Vdq_meas',t0=t0,t1=t1)
ax[0].plot(t+t0,vals[:,0],label='$v_\\mathit{HSC_1}^d$',color='C0',alpha=0.5,ls='-',zorder=5,lw=3.5)
ax0.plot(t+t0,vals[:,1],color='C3',alpha=0.5,ls='-',zorder=5,lw=3.5)
# currents
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):Idq_meas',t0=t0,t1=t1)
ax[1].plot(t+t0,vals[:,0],color='C0',alpha=.5,ls='-',zorder=5,lw=3.5)
ax1.plot(t+t0,vals[:,1],color='C3',alpha=.5,ls='-',zorder=5,lw=3.5)
# currents
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):vdq_vi',t0=t0,t1=t1)
ax[2].plot(t+t0,vals[:,0],color='C0',alpha=.5,ls='-',zorder=5,lw=3.5)
ax2.plot(t+t0,vals[:,1],color='C3',alpha=.5,ls='-',zorder=5,lw=3.5)

t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC1)\control_DVF(2):theta',t0=t0-0.01,t1=t1-0.01)
ax[3].plot(t+t0,vals[:],color='C0',alpha=.5,ls='-',zorder=5,lw=3.5)
t, vals = loader.get_measurement_data('Main(0)\MMC_DVF_PF1(HSC2)\control_DVF(1):theta',t0=t0-0.01,t1=t1-0.01)
ax3.plot(t+t0,vals[:],label='$i_\\mathit{HSC_1}^q$',color='C3',alpha=.5,ls='-',zorder=5,lw=3.5)

ax[-1].set(xlim=(t0,t1))

# FORMATTING
shared_axes = [0,1,2,3]
fontsize=11
for i, axis in enumerate(ax):
    axis.grid(ls=':',color=('grey','C0')[i in shared_axes])    

    ax[i].set_ylabel(['$v_\\mathit{HSC_1,meas}^d$ [pu]','$i_\\mathit{HSC_1,meas}^d$ [pu]','$v_\\mathit{HSC_1,vi}^d$ [pu]','$\\vartheta_\\mathit{HSC_2}$ [pu]','Frequency [Hz]','Angle [$^\circ$]'][i],color=('k','C0')[i in shared_axes])
    if i in shared_axes:
        ax[i].tick_params(axis='y', colors='C0')
        
    if i == 0:
        ax0.set_ylabel('$v_\\mathit{HSC_1,meas}^q$ [pu]',color='C3')
        ax0.tick_params(axis='y', colors='C3')
        ax0.grid(ls=':',color='C3')
    if i == 1:  
        ax1.set_ylabel('$i_\\mathit{HSC_1,meas}^q$ [pu]',color='C3')
        ax1.tick_params(axis='y', colors='C3')
        ax1.grid(ls=':',color='C3')
    if i == 2:  
        ax2.set_ylabel('$v_\\mathit{HSC_1,vi}^q$ [pu]',color='C3')
        ax2.tick_params(axis='y', colors='C3')
        ax2.grid(ls=':',color='C3')
    if i == 3:  
        ax3.set_ylabel('$\\vartheta_\\mathit{HSC_2}$ [pu]',color='C3')
        ax3.tick_params(axis='y', colors='C3')
        ax3.grid(ls=':',color='C3')



fig.align_ylabels(ax)
fig.align_ylabels([ax0,ax1,ax2,ax3])
fig.tight_layout()

# plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\virtual_impedance_theta_perturbations2.pdf')

plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a meshgrid for the range of x and y values
x = np.linspace(-1, 1, 1000)
y = np.linspace(-1, 1, 1000)
X, Y = np.meshgrid(x, y)

# Calculate atan(y/x), handling division by zero
Z_atan = np.arctan(np.divide(Y, X, out=np.zeros_like(Y), where=X!=0))
Z_atan = np.arctan(Y/X)

# Calculate atan2(y, x)
Z_atan2 = np.arctan2(Y, X)

# Define the unit circle
theta = np.linspace(0, 2*np.pi, 1000)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
circle_z = np.zeros_like(theta)  # z is 0 for the unit circle in the x-y plane

# Create a figure and a 3D subplot for atan(y/x)
fig = plt.figure(figsize=(14, 7))

# First subplot for atan(y/x)
ax1 = fig.add_subplot(121, projection='3d')
X1 = np.where(X<0,X,np.nan)
Y1 = np.where(X<0,Y,np.nan)
Z1 = np.where(X<0,Z_atan,np.nan)
X2 = np.where(X>0,X,np.nan)
Y2 = np.where(X>0,Y,np.nan)
Z2 = np.where(X>0,Z_atan,np.nan)

surf1 = ax1.plot_surface(X1, Y1, Z1, cmap='viridis', edgecolor='none')
surf1_ = ax1.plot_surface(X2, Y2, Z2, cmap='viridis', edgecolor='none',zorder=5)
# surf1 = ax1.plot_surface(X1, Y1, Z_atan, cmap='viridis', edgecolor='none')

ax1.set_title('atan(y/x)')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
# ax1.plot(circle_x, circle_y, circle_z, color='r', linewidth=2)  # Plot unit circle
ax1.plot(circle_x, circle_y, np.arctan(circle_y/circle_x), color='r', linewidth=3,zorder=10)  # Plot unit circle

fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10) # Add a color bar

# Second subplot for atan2(y, x)
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_atan2, cmap='viridis', edgecolor='none')
ax2.plot(circle_x, circle_y, np.arctan2(circle_y, circle_x), color='r', linewidth=3,zorder=10)  # Plot unit circle
ax2.set_title('atan2(y, x)')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')

# Set the view angle for the first subplot
ax1.view_init(elev=45, azim=45-180)  # for example, elevation=20°, azimuth=30°

# Set the view angle for the second subplot
ax2.view_init(elev=45, azim=135-180)  # for example, elevation=20°, azimuth=30°


fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10) # Add a color bar

fig.tight_layout()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\virtual_impedance_theta_perturbations_atan.pdf')


# Show the plot
plt.show()

#%%

def z_(r,i):
    mag = np.sqrt(r**2+i**2)
    phi = np.arctan(i,r)
    z = mag*(np.cos(phi) + 1j*np.sin(phi))
    return z

def Z_(Snom,SCR,V,XR):
    Z = V**2/(Snom*SCR)
    theta = np.arctan(XR)
    R = Z*np.cos(theta)
    X = Z*np.sin(theta)
    return R,X

def t_(t,vals,t0,t1,mean=False):
    mask = (t>= t0) & (t<=t1)
    if len(vals.shape) == 1:
        vals = vals[mask]
    elif len(vals.shape) == 2:
        vals = vals[mask,:]
                
    if mean:
        vals = vals.mean(axis=0)
        
    return vals  

