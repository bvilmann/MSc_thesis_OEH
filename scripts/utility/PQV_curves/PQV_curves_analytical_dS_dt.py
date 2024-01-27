# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:17:16 2023

@author: bvilm
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from matplotlib import cm

def pqv_func(x,y,E,scr,xr,P_sys,mode='V'):
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

def dS_dt(p,q,scr,xr,delta_f,Vbase,Sbase,f=50,T=1):
    Vm = 1
    Vk = 1
    dt = 1/f*T   
    Z = Vbase**2/(scr*Sbase)
    Y = 1/Z
    theta = np.arctan(xr)
    R, X = Z*np.cos(theta), Z*np.sin(theta)       
    
    vartheta = np.arccos(R/(np.sqrt(R**2+X**2))*Vk/Vm + (p*Z)/(Vk*Vm))   
    
    dp = 2*np.pi*(delta_f)*Y*Vk*Vm*np.sin(vartheta) * dt
    dq = 2*np.pi*(delta_f)*Y*Vk*Vm*np.cos(vartheta) * dt
    
    # Calculate new location
    p = p + dp
    q = q + dq
    return p,q

def dSdt_dt(p,q,scr,xr,v,delta_f,Vbase,Sbase,dt = 1e-5):
    Vm = 1
    Vk = v
    Z = Vbase**2/(scr*Sbase)
    Y = 1/Z
    theta = np.arctan(xr)
    R, X = Z*np.cos(theta), Z*np.sin(theta)       
    
    vartheta = np.arccos(R/(np.sqrt(R**2+X**2))*Vk/Vm + (p*Z)/(Vk*Vm))   
    
    dp = 2*np.pi*(delta_f)*Y*Vk*Vm*np.sin(vartheta) * dt
    dq = 2*np.pi*(delta_f)*Y*Vk*Vm*np.cos(vartheta) * dt
    
    # Calculate new location
    p = p + dp
    q = q + dq
    return p,q

def t_increment():
    return

def critical_time(p0,q0,scr,xr,sbase=1,df:float = 0.1, dt = 1e-5):
    
    # Consider initial values    
    v0 = np.array(pqv_func(p0, q0, 1,scr,xr,1,mode='V'))[0]
    p_ = p0
    q_ = q0
    v_ = v0
    t_ = 0
    print(v_)
    
    # Prepare data storage
    t = [0]
    p = [p0]
    q = [q0]
    v = [v0]

    while ~np.isnan(v_):
        p_,q_ = dSdt_dt(p_, q_, scr, xr, v_, df, 1, 1,dt = dt)
        
        v_ = np.array(pqv_func(p_, q_, 1,scr,xr,1,mode='V'))[0]
        t_ += dt
        print(v_)
        

        p.append(p_)
        q.append(q_)
        v.append(v_)
        t.append(t_)
        
        pass

    df = pd.DataFrame({'t':t,'p':p,'q':q,'v':v}).set_index('t')
    
    return df

def plot_PVQ_surface(x=10,n=2001):
    scr = x
    xr = x

    p = np.concatenate([np.linspace(0, x, n)])
    q = np.concatenate([np.linspace(-x, x*.1, n)])
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()
    P, Q = np.meshgrid(p, q)
    
    fig,ax = plt.subplots(subplot_kw={"projection": "3d"},figsize =(6, 6),dpi=200)
    V = np.flip(np.array(pqv_func(P, Q, 1,scr,xr,1,mode='V'))[0,:].reshape(n,n),axis=0)
    
    # ax.plot_surface(Q,V, P, vmin=0, vmax=P.max(), cmap=cm.rainbow)
    # ax.set(xlabel='Q',ylabel='V',zlabel='P')
    ax.plot_surface(P,Q, V, vmin=0, vmax=P.max(), cmap=cm.rainbow)
    ax.set(xlabel='P',ylabel='Q',zlabel='V')
    return ax

#%%

ax = plot_PVQ_surface()

# ax.plot(df.p,df.q,df.v)


#%%

plt.style.use('_mpl-gallery')

# Make data
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)

#%%
scr = 10
xr = 10

q = np.array(pqv_func(1,1,1,scr,xr,1,mode='Q'))[0]


df = critical_time(1, 1, scr, xr)

ax = plot_PVQ_surface()

ax.view_init(0, 90, 90)


#%% Surface plot    
x = 100
scr = 100
xr = 10
n= 2001

scrs = [scr]
xrs = [xr]
cnt = 0

p = np.concatenate([np.linspace(0, x, n)])
q = np.concatenate([np.linspace(-x, x*.1, n)])
P, Q = np.meshgrid(p, q)
P, Q = P.flatten(), Q.flatten()
P, Q = np.meshgrid(p, q)


fig,ax = plt.subplots(1,1,figsize =(5,5),dpi=200,sharex=True,sharey=True)
V = np.flip(np.array(pqv_func(P, Q, 1,scrs[cnt],xrs[cnt],1,mode='V'))[0,:].reshape(n,n),axis=0)

q_ = np.array(pqv_func(p,1,1,scr,xr,1,mode='Q'))[0]
cmap = plt.cm.rainbow #or any other colormap

extent = [0,x*1,-x,x*.1]

grey = ax.imshow(V,cmap = plt.cm.Greys,norm = mpl.colors.Normalize(vmin=0, vmax=1.4),label='$\\mathcal{C}$', extent=extent)    
bwr = ax.imshow(np.where((V>=0.95) & (1.05>=V),V,np.nan),cmap = plt.cm.bwr,label='$\\mathcal{F}$', extent=extent)    

# Creating figure
ax.set_xlabel('$P$ [pu]')
ax.set_ylabel('$Q$ [pu]')

# plot 
non_nan_indices = ~np.isnan(q_)
# Filter p and q_ using the non-nan indices
p_clean = np.linspace(0, x, n)[non_nan_indices]
q_clean = q_[non_nan_indices]

last_idx = -10
p = p_clean[:last_idx]
q = q_clean[:last_idx]
dp,dq = dS_dt(p,q,scrs[cnt],xrs[cnt],1,1,1)

# Plot normal
ax.plot(p,q,zorder=10,color='black',ls='--')
points = ax.scatter(p,q, c=p, cmap='winter', s=10)

# Plot ds/dt line normal
# ax.plot(dp,dq,zorder=10,color='black',ls='--')
# points = ax.scatter(dp, dq, c=dp, cmap='rainbow', s=20)

# for i in np.linspace(0,len(p_clean[:last_idx]),10):
#     idx = int(np.floor(i))
#     if idx == len(p_clean[:last_idx]):
#         idx -= 1
#     ax.annotate('', xy=(p[idx], q[idx]), xytext=(dp[idx],dq[idx]),
#             arrowprops=dict(facecolor='blue', edgecolor='black', arrowstyle='<-'))

#     ax.arrow(x=p[idx], y=q[idx], dx=dp[idx]-p[idx], dy=dq[idx]-q[idx], head_width=0.05, head_length=0.1, fc='blue', ec='black')

# ax.quiver(p, q, dp, dq, np.sqrt(dp**2 + dq**2), scale=0.00005, pivot='mid', cmap='rainbow')
itvl = 103
amp =  np.sqrt((dp[::itvl]-p[::itvl])**2 + (dq[::itvl]-q[::itvl])**2)
amp = [i for i in range(len(amp))]
# ax.quiver(p[::50], q[::50], dp[::50]-p[::50], dq[::50]-q[::50],amp,scale=80, pivot='mid', color='k')
ax.quiver(p[::itvl], q[::itvl], dp[::itvl]-p[::itvl], dq[::itvl]-q[::itvl],amp,scale=100,  cmap='winter',zorder=5)

# p_sim = 205.02176/100
# q_sim = -258.2177/100
# ax.scatter([p_sim],[q_sim],color='gold')

ax.set(xlim=(extent[0],extent[1]),ylim=(extent[2],extent[3]))

ax.grid(ls=':')

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\PQV_ds_dt.pdf', bbox_inches="tight")

#%%
import matplotlib

fig,ax = plt.subplots(1,1,dpi=150,figsize=(5,5))
normalize = matplotlib.colors.Normalize(vmin=0.75, vmax=2)
d = np.linspace(0,np.pi,1000)

p = lambda x, z: 1/z*np.sin(d)

for i in [0.5,1,1.5,2]:
    ax.plot(d,p(d,i), color=cm.Reds_r(normalize(i)),label='$X='+f'{round(i,2)}'+'$ [pu]')

ax.set(yticks=[0,0.5,1,1.5,2],xticks=[0,np.pi*1/4,np.pi/2,np.pi*3/4,np.pi],xticklabels=['$0\\pi$','$1/4\\pi$','$1/2\\pi$','$3/4\\pi$','$1\\pi$'],xlim=(min(d),max(d)),ylim=(0,2.1),xlabel='Load angle [rad]',ylabel='Active power [pu]')

ax.axvline(np.pi*1/8,color='k',lw=0.75)
ax.axvline(np.pi*1/4.5,color='k',lw=0.75)

ax.legend()
ax.grid(ls=':')
plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\ds_dt_Pdelta.pdf', bbox_inches="tight")


#%% plot dS/dt as a function of impedance
scr = np.linspace(1,1e3,1000)
xrs = [1,10,100,1000]
last_idx = -10
scale = 1.2
# Get data
fig, ax = plt.subplots(2,1,dpi=150,sharex=True,figsize=(6*scale,4*scale))
for xr in xrs:
    q = np.array(pqv_func(1,1,1,scr,xr,1,mode='Q'))[0]

    # Filter p and q_ using the non-nan indices
    dp,dq = dS_dt(1,q,scr,xr,0.1,1,1)

    # points = ax.scatter(dp, dq, c=dp, cmap='viridis', s=20)

    ax[0].plot(scr,dp - 1,label=f'$X/R$={xr}')
    ax[0].set(ylim=(0,5),xlim=(0,500),ylabel='${\\Delta P}$ [pu]')
    ax[1].plot(scr,dq - q,label=f'$X/R$={xr}')
    ax[1].set(ylim=(0,5),xlim=(0,500),ylabel='${\\Delta Q}$ [pu]',xlabel='SCR [-]')
for i in range(2):
    ax[i].grid(ls=':')
    ax[i].legend(loc='upper left')


