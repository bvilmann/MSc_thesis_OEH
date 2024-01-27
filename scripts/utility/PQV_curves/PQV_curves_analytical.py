# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:32:42 2022

@author: bvilm & eag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
#plt.rcParams['font.size'] = '14'
def locus_PV(X,E,low,high):
    low_rng = low
    high_rng = high
    n = 1001
    
    deg = np.linspace(low_rng, high_rng, n)
    rad = np.radians(deg)
    
    power_max = np.power(E,2)/(2*X)*(np.sqrt(1+np.power(np.tan(rad),2))-np.tan(rad))
    Q_max = power_max*np.tan(rad)
    V_max = np.sqrt(-X*Q_max+(np.power(E,2)/2))
    
    return (1e-6)*power_max, (1e-3)*V_max

# Function to compute and plot a PV-curve given X, E and a list of load angles phi.
def PlotPV(X, E, phi):
    
    fig, ax = plt.subplots(1,1, figsize=(6, 4) , dpi=300)
    
    for i in phi:
        rad = np.radians(i)
        
        #1 - Determine maximum power, Pmax, for current value of phi
        Pmax = np.power(E,2)/(2*X)*(np.sqrt(1+np.power(np.tan(rad),2))-np.tan(rad))
        
        #2 - Generate a P vector going from zero to Pmax.
        pct = 0.9
        n_points = 1000
        P_vec1 = np.linspace(0, pct*Pmax, n_points, endpoint=False)
        P_vec2 = np.linspace(pct*Pmax, (1-1e-10)*Pmax, n_points, endpoint=True)
        
        P = np.concatenate((P_vec1, P_vec2),axis=0)
        Q = P*np.tan(rad)
        
        #3 - Calculate the upper and lower voltage solutions at each point in the P vector (in kV)
        V_plus = np.sqrt(-X*Q+(np.power(E,2)/2) + np.sqrt((np.power(E,4)/4)-np.power(X,2)*np.power(P,2)-np.power(E,2)*X*Q))
        V_minus = np.sqrt(-X*Q+(np.power(E,2)/2) - np.sqrt((np.power(E,4)/4)-np.power(X,2)*np.power(P,2)-np.power(E,2)*X*Q))
        
        #4 - Plot the voltage V as a function of the power P
        P_plot = 1e-6*P
        V_plus_plot = 1e-3*V_plus
        V_minus_plot = 1e-3*V_minus
        
        ax.plot(P_plot, V_plus_plot, color="blue", lw=1.25)
        ax.plot(P_plot, V_minus_plot, color="blue", lw=1.25)
        
        #5 - add a label to the curve containing the current value of phi in degrees 
        text_offset = 1.5
        ax.scatter(P_plot[-1], V_plus_plot[-1], 75, marker='x',color='blue',zorder=3)
        ax.text(P_plot[-1]+text_offset,V_plus_plot[-1]+text_offset,f'$\\theta={i}^o$',ha='left',va='center',rotation=90,zorder=3,fontsize=11)
        
        # Print to debug
        print('\ndegrees: %d (º), radians: %.3f (rad)' %(i,rad))
        print('Pmax: %.2f (MW)' %P_plot[-1])
        print('V_plus: %.2f (kV)' %V_plus_plot[-100])
        print('V_minus: %.2f (kV)' %V_minus_plot[-100])
        
    #6 Add labels on axes, title and plot locus of critical points (outside the for loop now)
    
    #6.1 Locus of critical points
    P_locus, V_locus = locus_PV(X, E, -14, 30)
    ax.plot(P_locus, V_locus, color="grey", ls="-.", lw=1.5, alpha=0.75,zorder=3)
    
    x_locus, y_locus = (P_locus[-1]-20, V_locus[-400]-15)  
    ax.text(x_locus, y_locus, "Locus of critical points", ha='center', va='center', zorder=3, fontsize=12)
    
    #6.2 Add title, labels, grid, etc.
    fig.tight_layout()
    plt.margins(x=0, y=0)
    plt.ylim([0,1.1*V_plus_plot[0]])
    plt.xlim([0,1.1*P_plot[-1]])
    ax.set_ylabel("Voltage [kV]")
    ax.set_xlabel("Active Power P [MW]")
    ax.grid(linestyle="-.", linewidth=0.5)  
    #plt.show()
        
    return fig,ax


def PV_point_cal(P,phi,E=132e3,R=0,X=100):
    rad = np.radians(phi)
    Q = P*np.tan(rad)

    v1 = np.sqrt(-(R*P+X*Q)+E**2/2+np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q)))*1e-3

    return v1

def PV_cal(E,P_sys,scr,xr,phi = 0):
    Z = E**2/(scr*P_sys)
    theta = np.arctan(xr)
    R, X = Z*np.cos(theta), Z*np.sin(theta)    

    # print(R,X)

    rad = 0
    rad = np.radians(phi)

    #1 - Determine maximum power, Pmax, for current value of phi
    # Pmax = np.power(E,2)/(2*X)*(np.sqrt(1+np.power(np.tan(rad),2))-np.tan(rad))

    
    #2 - Generate a P vector going from zero to Pmax.
    pct = 0.9
    n_points = 1000
    P = np.linspace(0, P_sys*max(SCR), 1000, endpoint=False)    
    Q = P*np.tan(rad)

    #3 - Calculate the upper and lower voltage solutions at each point in the P vector (in kV)
    V1 = np.sqrt(-(R*P+X*Q)+E**2/2+np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q)))

    
    P_plot = P/P_sys
    V_plus_plot = V1/E


    return P_plot, V_plus_plot 
   
#%% PV-curves 
E = 132e3
X = 100
phi = np.array([20,10,0,-5,-10])
fig,ax = PlotPV(X,E,phi)

p = np.array([60,60,80,80])*1e6
v = PV_point_cal(p,np.array([0,-10,0,-10]))

ax.scatter(p[[0,2]]*1e-6,v[[0,2]],zorder=3)
ax.scatter(p[[1,3]]*1e-6,v[[1,3]],zorder=3)
ax.axvline(60,lw=0.75,ls='-',color='grey',zorder=2)
ax.axvline(80,lw=0.75,ls='-',color='grey',zorder=2)


ax.text(70,25,'$\\Delta P$',ha='center',va='center')

ax.annotate('', (60, 20),xycoords='data',
            xytext=(80,20), textcoords='data',
            arrowprops=dict(facecolor='black',arrowstyle='<-'),
            horizontalalignment='center', verticalalignment='center')

# plt.show()
plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\UPEC 2022 - MDPI - edited\img\PV_single.pdf', bbox_inches="tight")


#%% PV-curves with SCR and X/R sensitivity

P = 8e6
E = 230e3
xr = 10
# fig, ax = plt.subplots(1,1,figsize=(6,4.5))
ps = []
vs = []
scrs = []
xrs = []

N = 100
SCR = [10,7.5,5,2.5,1]
SCR = np.arange(10,0.5,-.5)
SCR = list(np.linspace(0.25,10,175)) + [1,5]
norm = mpl.colors.Normalize(vmin=-len(SCR)*2, vmax=len(SCR))
XRs = [10,5,1]
# xrs = [10,5,1][0:3]
# pv = np.zeros((len(SCR),100))
fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=200)
phi=-10
for i, scr in enumerate(SCR):
    print(scr)
    for xr in XRs:
        
        # print(scr)
        # for xr in [10,5,1]:                    
        # xr = 1
        p,v = PV_cal(E,P,scr,xr,phi=phi)
        if xr == 1:
            if scr==1:                
                ax.plot(p,v,color=mpl.cm.Blues(len(SCR)),ls=':',alpha=0.99,zorder=3)
            elif scr == 5:               
                ax.plot(p,v,color=mpl.cm.Blues(len(SCR)),ls='--',alpha=0.99,zorder=3)
            elif scr == 10:                
                ax.plot(p,v,color=mpl.cm.Blues(len(SCR)),ls='-',alpha=0.99,zorder=3)
            else:                
                ax.plot(p,v,color=mpl.cm.Blues((len(SCR)+i*2+1)/(len(SCR)*3)),alpha=0.5)
        elif xr == 5:
            if scr==1:                
                ax.plot(p,v,color=mpl.cm.Greens(len(SCR)),ls=':',alpha=0.99,zorder=3)
            elif scr == 5:               
                ax.plot(p,v,color=mpl.cm.Greens(len(SCR)),ls='--',alpha=0.99,zorder=3)
            elif scr == 10:                
                ax.plot(p,v,color=mpl.cm.Greens(len(SCR)),ls='-',alpha=0.99,zorder=3)
            else:                
                ax.plot(p,v,color=mpl.cm.Greens((len(SCR)+i*2+1)/(len(SCR)*3)),alpha=0.5)
        elif xr == 10:
            if scr==1:                
                ax.plot(p,v,color=mpl.cm.Oranges(len(SCR)),ls=':',alpha=0.99,zorder=3)
            elif scr == 5:               
                ax.plot(p,v,color=mpl.cm.Oranges(len(SCR)),ls='--',alpha=0.99,zorder=3)
            elif scr == 10:                
                ax.plot(p,v,color=mpl.cm.Oranges(len(SCR)),ls='-',alpha=0.99,zorder=3)
            else:                
                ax.plot(p,v,color=mpl.cm.Oranges((len(SCR)+i*2+1)/(len(SCR)*3)),alpha=0.5)
        
        # pv[i,:] = v
        # ps.append(p)
        # vs.append(v)
        # scrs.append(scr)
        # xrs.append(xr)
        # for j in range(len(p)):
        #     ps.append(p[j])
        #     vs.append(v[j])
        #     scrs.append(scr)
        #     xrs.append(xr)
        
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=mpl.cm.Blues(len(SCR)), lw=1),
                Line2D([0], [0], color=mpl.cm.Greens(len(SCR)), lw=1),
                Line2D([0], [0], color=mpl.cm.Oranges(len(SCR)), lw=1),
                Line2D([0], [0], color='k',ls=':', lw=1),
                Line2D([0], [0], color='k',ls='--', lw=1),
                Line2D([0], [0], color='k', lw=1),
                ]

# Create the figure
# fig, ax = plt.subplots()
ax.legend(custom_lines, ['X/R=1', 'X/R=5', 'X/R=10','SCR=1', 'SCR=5', 'SCR=10'])

ax.axvline(1,ls=':',lw=0.75,color='k',zorder=5)
ax.axhline(1,ls=':',lw=0.75,color='k',zorder=5)

# df = pd.DataFrame({'p':ps,'v':vs,'scr':scrs,'xr':xrs})
# df10 = df[df.xr == 10][['p','v','scr']]

# fig, ax = plt.subplots(1,1,figsize=(6,6))
# d = df10.pivot(index='scr',columns='p',values='v')
# X,Y,Z = d.index,d.columns,d.values
# ax.imshow(Z,extent=[min(X),max(X),min(Y),max(Y)],origin="lower")
ax.set_ylabel('$V_{OWF}$ [pu]')
ax.set_xlabel('$P_{OWF}$ [pu]')
ax.set_ylim(0.80,1.0125)
ax.set_xlim(0,5.5)
ax.grid()


# df10.pivot(index='scr', columns='p', values='v')

# plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\UPEC 2022 - MDPI - edited\img\PV_mult.pdf', bbox_inches="tight")
            
#%%
# Creating dataset
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
        return [np.sqrt(-(R*P+X*Q)+E**2/2+np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q))),
                np.sqrt(-(R*P+X*Q)+E**2/2-np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q)))]
        
    
    elif mode =='V':
        P, Q = x,y
        return [np.sqrt(-(R*P+X*Q)+E**2/2+np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q))),
                np.sqrt(-(R*P+X*Q)+E**2/2-np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q)))]

# Initialization
n = 250
x = 10
scr, xr=3,3





# ======= Map solution space for V =======
import matplotlib.tri as mtri
for i in range(2):
    # Creating figure
    fig = plt.figure(figsize =(8, 5),dpi=200)
    ax = plt.axes(projection ='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.w_xaxis.gridlines.set_ls(':')
    ax.w_yaxis.gridlines.set_ls(':')
    ax.w_zaxis.gridlines.set_ls(':')
    ax.set_xlabel('$P_{OWF}$ [pu]')
    ax.set_ylabel('$Q_{OWF}$ [pu]')
    ax.set_zlabel('$V_r$ [pu]')
    ax.set_xlim(0,x)
    ax.set_ylim(-x,4)
    ax.set_zlim(0,1.4)
    if i == 0:
        ax.view_init(15, 5)
        cmap = plt.cm.rainbow #or any other colormap
        norm = mpl.colors.Normalize(vmin=0, vmax=1.6)
    else:
        ax.view_init(15, 30)
        cmap = plt.cm.bwr #or any other colormap
        norm = mpl.colors.Normalize(vmin=0.95, vmax=1.05)
    
    
    # https://matplotlib.org/stable/gallery/mplot3d/trisurf3d_2.html
    # https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
    p = np.concatenate([np.linspace(0, x*0.9, n),np.linspace(x*0.9,x, n)])
    q = np.concatenate([np.linspace(-x, x, n),np.linspace(-x, x, n)])
    P, Q = np.meshgrid(p, q)
    
    # Trianglurization
    
    
    P, Q = P.flatten(), Q.flatten()
    
    zs = np.array(fun(P, Q,1,scr,xr,1,mode='V'))
    
    V = np.hstack([zs[0,:],np.flip(zs[1,:])])
    
    idx = np.argwhere(~np.isnan(V))
    fo = n**2 - idx.shape[0] # filtered out
    
    P = np.hstack([P,np.flip(P)])
    Q = np.hstack([Q,np.flip(Q)])
    P = P[idx].flatten()
    Q = Q[idx].flatten()
    V = V[idx].flatten()
    tri = mtri.Triangulation(V,Q)
    
    surf = ax.plot_trisurf(P,Q,V, triangles=tri.triangles, cmap=cmap,lw=0.05,norm=norm)
    
    
    # surf = ax.plot_trisurf(P,Q,V, cmap=cmap, norm=norm, edgecolor ='grey',lw=0.25,alpha=0.9)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    
    # Paths
    phi = np.array([20,10,0,-5,-10,-2])
    phi = np.array([10,0,-10])
    for i in phi:
        rad = np.radians(i)
        
        #1 - Determine maximum power, Pmax, for current value of phi
        
        #2 - Generate a P vector going from zero to Pmax.
        pct = 0.9
        n_points = 50
        P_vec1 = np.linspace(0, pct*x, n_points, endpoint=False)
        P_vec2 = np.linspace(pct*x, (1-1e-10)*x, n_points, endpoint=True)
        
        P = np.concatenate((P_vec1, P_vec2),axis=0)
        Q = P*np.tan(rad)
        
        #3 - Calculate the upper and lower voltage solutions at each point in the P vector (in kV)
        # V_plus = np.sqrt(-X*Q+(np.power(E,2)/2) + np.sqrt((np.power(E,4)/4)-np.power(X,2)*np.power(P,2)-np.power(E,2)*X*Q))
        V_plus = np.array(fun(P,Q,1,scr,xr,1,mode='V'))[0,:]
    
        ax.plot(P,Q,V_plus,lw=2,alpha=0.8,zorder=3,color='k')
    
    
    plt.show()
    
    
    # plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\UPEC 2022 - MDPI - edited\img\PVQ_surf_'+f'{i}'+'.pdf', bbox_inches="tight")
    # plt.close()

            
#%%
# Creating dataset
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
        P, Q = x,y
        return [np.sqrt(-(R*P+X*Q)+E**2/2+np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q))),
                np.sqrt(-(R*P+X*Q)+E**2/2-np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q)))]
        
    elif mode =='V':
        P, Q = x,y
        return [np.sqrt(-(R*P+X*Q)+E**2/2+np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q))),
                np.sqrt(-(R*P+X*Q)+E**2/2-np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q)))]

# Initialization
n = 100
x = 10
scr, xr=3,3





# ======= Map solution space for V =======
import matplotlib.tri as mtri
for scr in [10,3]:
    for xr in [10,3]:
        # Creating figure
        fig = plt.figure(figsize =(8, 5),dpi=200)
        ax = plt.axes(projection ='3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.w_xaxis.gridlines.set_ls(':')
        ax.w_yaxis.gridlines.set_ls(':')
        ax.w_zaxis.gridlines.set_ls(':')
        ax.set_xlabel('$P_{OWF}$ [pu]')
        ax.set_ylabel('$Q_{OWF}$ [pu]')
        ax.set_zlabel('$V_r$ [pu]')
        # ax.text(0.5,0.5,0.5,'$V_r$ [pu]',rotation=90)

        ax.set_xlim(0,x)
        ax.set_ylim(-x,4)
        ax.set_zlim(0,1.4)

        ax.view_init(15, 45)
        cmap = plt.cm.rainbow #or any other colormap
        norm = mpl.colors.Normalize(vmin=0, vmax=1.6)
        
        # https://matplotlib.org/stable/gallery/mplot3d/trisurf3d_2.html
        # https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
        p = np.concatenate([np.linspace(0, x*0.9, n),np.linspace(x*0.9,x, n)])
        q = np.concatenate([np.linspace(-x, x, n),np.linspace(-x, x, n)])
        P, Q = np.meshgrid(p, q)
        
        # Trianglurization
        
        
        P, Q = P.flatten(), Q.flatten()
        
        zs = np.array(fun(P, Q,1,scr,xr,1,mode='V'))
        
        V = np.hstack([zs[0,:],np.flip(zs[1,:])])
        
        idx = np.argwhere(~np.isnan(V))
        fo = n**2 - idx.shape[0] # filtered out
        
        P = np.hstack([P,np.flip(P)])
        Q = np.hstack([Q,np.flip(Q)])
        P = P[idx].flatten()
        Q = Q[idx].flatten()
        V = V[idx].flatten()
        tri = mtri.Triangulation(V,Q)
        
        surf = ax.plot_trisurf(P,Q,V, triangles=tri.triangles, cmap=cmap,lw=0.05,norm=norm,zorder=1)
        
        
        # surf = ax.plot_trisurf(P,Q,V, cmap=cmap, norm=norm, edgecolor ='grey',lw=0.25,alpha=0.9)
        
        fig.colorbar(surf, shrink=0.5, aspect=10,extend='max',label='$V_r$ [pu]')
        
            
        # Paths
        phi = np.array([20,10,0,-5,-10,-2])
        phis = np.array([0,-10,-20])
        
        ax.plot([x,0,0],[-x,-x,4],[1,1,1],color='k',ls=':',lw=1,zorder=0)
        ax.plot([x,0,0],[0,0,0],[0,0,1.4],color='k',ls=':',lw=1,zorder=0)
        ax.plot([1,1,1],[-x,-x,4],[1.4,0,0],color='k',ls=':',lw=1,zorder=0)
        q_ = []
        v_ = []
        
        for j,phi in enumerate(phis):
            rad = np.radians(phi)
            
            #1 - Determine maximum power, Pmax, for current value of phi
            
            #2 - Generate a P vector going from zero to Pmax.
            pct = 0.9
            n_points = 50
            P_vec1 = np.linspace(0, pct*x, n_points, endpoint=False)
            P_vec2 = np.linspace(pct*x, (1-1e-10)*x, n_points, endpoint=True)
            
            P = np.concatenate((P_vec1, P_vec2),axis=0)
            Q = P*np.tan(rad)
            q_.append(np.tan(rad))            
            #3 - Calculate the upper and lower voltage solutions at each point in the P vector (in kV)
            # V_plus = np.sqrt(-X*Q+(np.power(E,2)/2) + np.sqrt((np.power(E,4)/4)-np.power(X,2)*np.power(P,2)-np.power(E,2)*X*Q))
            V_plus = np.array(fun(P,Q,1,scr,xr,1,mode='V'))[0,:]

            
            v_.append(np.array(fun(1,np.tan(rad),1,scr,xr,1,mode='V'))[0])
            # print(v_)

        
            ax.plot(P,Q,V_plus,lw=2,alpha=0.8,zorder=5,color=['k','grey','darkgrey'][j],label='$\\theta='+f'{phi}'+'^\\circ$')
            print(v_)

        ax.plot([0],[0],[1],marker='*',label='No load',color='white',zorder=8,markersize=10,markeredgecolor='black',mfc='gold')
        for j in range(3):
            if j ==0:   
                ax.plot([1],q_[j],v_[j],color='white',marker='.',zorder=8,markersize=8,markeredgecolor='black',mfc=cmap(v_[j]/1.6),label='Nom. load')
            else:
                ax.plot([1],q_[j],v_[j],color='white',marker='.',zorder=8,markersize=8,markeredgecolor='black',mfc=cmap(v_[j]/1.6))
                
                
        if scr == 10 and xr == 10:
            plt.legend(ncol=2,loc='lower center',fontsize=8, bbox_to_anchor=(0.5, 0.175))

        fig.tight_layout()
        plt.show()
                    
        # plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\UPEC 2022 - MDPI - edited\img\PVQ_surf_'+f'scr{scr}_xr{xr}'+'.pdf', bbox_inches="tight")
        # plt.close()


#%%
# Creating dataset
# def fun(P,Q,E,scr,xr,P_sys):
#     Z = E**2/(scr*P_sys)
#     theta = np.arctan(xr)
#     R, X = Z*np.cos(theta), Z*np.sin(theta)    
#     print(R,X,E,P,Q)
#     return np.sqrt(-(R*P+X*Q)+E**2/2+np.sqrt(E**4/4-(X*P-R*Q)**2-E**2*(R*P+X*Q)))
import matplotlib.pyplot as plt
import mpl_toolkits
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib import cbook
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid




n= 1001
x = 10

p = np.concatenate([np.linspace(0, x, n)])
q = np.concatenate([np.linspace(-x, x*.1, n)])
P, Q = np.meshgrid(p, q)
P, Q = P.flatten(), Q.flatten()
P, Q = np.meshgrid(p, q)


scrs = [10,10,3,3]
xrs = [10,3,10,3]
cnt = 0

fig,ax = plt.subplots(2,2,figsize =(6, 6),dpi=200,sharex=True,sharey=True)

for i in range(2):
    for j in range(2):
        # V = np.ones(P.shape)
        # for k in range(P.shape[0]):
        #     V[:,k] = fun(P[k,:],Q[:,k],1,scrs[cnt],xrs[cnt],1)
        V = np.flip(np.array(fun(P, Q, 1,scrs[cnt],xrs[cnt],1,mode='V'))[0,:].reshape(n,n),axis=0)


        cmap = plt.cm.rainbow #or any other colormap
            
        # bwr = ax[i,j].pcolormesh(P,Q,np.where((V>=0.95) & (1.05>=V),V,np.nan),cmap = plt.cm.bwr)    
        # grey = ax[i,j].pcolormesh(P,Q,np.where((V>=0.95) & (1.05>=V),np.nan,V),cmap = plt.cm.Greys,norm = mpl.colors.Normalize(vmin=0, vmax=1.4))    
        bwr = ax[i,j].imshow(np.where((V>0.95) & (1.05>V),V,np.nan),cmap = plt.cm.bwr,label='$\\mathcal{F}$')    

        grey = ax[i,j].imshow(np.where((V>=0.95) & (1.05>=V),np.nan,V),cmap = plt.cm.Greys,norm = mpl.colors.Normalize(vmin=0, vmax=1.4),label='$\\mathcal{C}$')    

        # if i == 1 and j == 1:
        #     ax1_divider = mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(ax[i,j])
        #     cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        #     cb1 = cax1.colorbar(grey, cax=cax1,extend='max')

        # elif i == 0 and j == 1:
        #     ax1_divider = mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(ax[i,j])
        #     cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        #     cb1 = cax1.colorbar(bwr, cax=cax1,extend='max')

        # plt.colorbar()

        # Creating figure
        ax[i,j].set_title(f'SCR={scrs[cnt]}, X/R={xrs[cnt]}')
        if i > 0:
            ax[i,j].set_xlabel('$P_{OWF}$ [pu]')
        if j < 1:
            ax[i,j].set_ylabel('$Q_{OWF}$ [pu]')
        # ax[i,j].set_xticks([np.arange(0,P.shape[0],5)])
        # ax[i,j].set_xticklabels(P[[0,10,20,30],[0,10,20,30]])
        # ax[i,j].set_yticks([np.arange(0,Q.shape[0],5)])
        rng = np.array(list(range(0,n+1,200)))
        qrng = [int(i/11*n)-1 for i in [1,3,5,7,9,11]]
        ax[i,j].set_xticks(rng)
        ax[i,j].set_yticks(qrng)
        ax[i,j].set_xticklabels(p[rng])
        ax[i,j].set_yticklabels([round(x,1) for x in np.flip(q)[qrng]])
        ax[i,j].grid()
        cnt+=1
# ax[-1,-1].legend()
# plt.show()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\UPEC 2022 - MDPI - edited\img\PVQ_ims.pdf', bbox_inches="tight")

#%%
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

# V = np.ones(P.shape)
# for k in range(P.shape[0]):
#     V[:,k] = fun(P[k,:],Q[:,k],1,scrs[cnt],xrs[cnt],1)
V = np.flip(np.array(fun(P, Q, 1,scrs[cnt],xrs[cnt],1,mode='V'))[0,:].reshape(n,n),axis=0)

q_ = np.array(fun(p,1,1,scr,xr,1,mode='Q'))[0]


cmap = plt.cm.rainbow #or any other colormap
    
# bwr = ax[i,j].pcolormesh(P,Q,np.where((V>=0.95) & (1.05>=V),V,np.nan),cmap = plt.cm.bwr)    
# grey = ax[i,j].pcolormesh(P,Q,np.where((V>=0.95) & (1.05>=V),np.nan,V),cmap = plt.cm.Greys,norm = mpl.colors.Normalize(vmin=0, vmax=1.4))    
# if i == 1 and j == 1:
#     ax1_divider = mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(ax[i,j])
#     cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
#     cb1 = cax1.colorbar(grey, cax=cax1,extend='max')

# elif i == 0 and j == 1:
#     ax1_divider = mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(ax[i,j])
#     cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
#     cb1 = cax1.colorbar(bwr, cax=cax1,extend='max')

# plt.colorbar()

extent = [0,x,-x,x*.1]

grey = ax.imshow(V,cmap = plt.cm.Greys,norm = mpl.colors.Normalize(vmin=0, vmax=1.4),label='$\\mathcal{C}$', extent=extent)    
bwr = ax.imshow(np.where((V>=0.95) & (1.05>=V),V,np.nan),cmap = plt.cm.bwr,label='$\\mathcal{F}$', extent=extent)    
# grey = ax.imshow(np.where((V>=0.95) & (1.05>=V),np.nan,V),cmap = plt.cm.Greys,norm = mpl.colors.Normalize(vmin=0, vmax=1.4),label='$\\mathcal{C}$', extent=extent)    

file = 'DVF_PF1_2para_unstable2'

# Creating figure
# ax[i,j].set_title(f'SCR={scrs[cnt]}, X/R={xrs[cnt]}')
ax.set_xlabel('$P_{OWF}$ [pu]')
ax.set_ylabel('$Q_{OWF}$ [pu]')
# ax[i,j].set_xticks([np.arange(0,P.shape[0],5)])
# ax[i,j].set_xticklabels(P[[0,10,20,30],[0,10,20,30]])
# ax[i,j].set_yticks([np.arange(0,Q.shape[0],5)])
# rng = np.array(list(range(0,n+1,200)))
# qrng = [int(i/11*n)-1 for i in [1,3,5,7,9,11]]
# ax.set_xticks(rng)
# ax.set_yticks(qrng)
# ax.set_xticklabels(p[rng])
# ax.set_yticklabels([round(x,1) for x in np.flip(q)[qrng]])

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

ax.grid(ls=':')
cnt+=1
# ax[-1,-1].legend()
# plt.show()

plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\05_model\img\PQV_scr3_xr3.pdf', bbox_inches="tight")


