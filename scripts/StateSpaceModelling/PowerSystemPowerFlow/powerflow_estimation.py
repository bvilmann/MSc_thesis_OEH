# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:38:22 2022

@author: bvilm
"""

from PowerSystemPowerFlow import PowerFlow as pf
import LoadNetworkData
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import conj


def cmplx(mag, ang, degrees:bool = True):
    phi = ang*(1,np.pi/180)[degrees]
    return mag * (np.cos(phi) + 1j*np.sin(phi))

#%%    

# settings
max_iter = 30 # Iteration settings
err_tol = 1e-4
n_pos = {1:(0,0),
         2:(1,1),
         3:(4,1),
         4:(5,0),
         5:(4,-1),
         6:(1,-1),
         7:(2.5,0),
         }
load_from_estimation = False
# Load the Network data ...
lnd = LoadNetworkData.LoadNetworkData()

# Adding lines and transformers (bus1, bus2, r, x, g, b)
if load_from_estimation:
    for i, (node1, node2, d) in enumerate(G.edges.data()):
        lnd.add_line(node1,node2,r[i],x[i],b=b[i],name=f'Z{i}')

else:
    lnd.add_line(1,2,0.04,0.52,name='Za',b=0.03)
    lnd.add_line(2,3,0.03,0.47,name='Zb',b=0.02)
    lnd.add_line(3,4,0.05,0.42,name='Zps',b=0.02)
    lnd.add_line(1,7,0.04,0.43,name='Zc',b=0.02)
    lnd.add_line(4,7,0.01,0.5,name='Zd',b=0.02)
    lnd.add_line(1,6,0.04,0.46,name='Ze',b=0.02)
    lnd.add_line(5,6,0.03,0.45,name='Zf',b=0.05)
    lnd.add_line(4,5,0.04,0.45,name='Zg',b=0.05)

# lnd.add_transformer(3,4,0,0.05,n=1,alpha=phi,name='Zps')

# Adding loads and generators
lnd.add_gen(1,0,0,v=1,theta=0)
lnd.add_gen(4,0.9,0,v=1)
lnd.add_load(2,0.42,0.05)
lnd.add_load(3,0,0)
lnd.add_load(5,0.43,0.06)
lnd.add_load(6,0.48,0.05)
lnd.add_load(7,0.41,0.04)

lnd.build()
lnd.draw()

V,success,n, J, F = pf.PowerFlowNewton(lnd.Ybus,lnd.Sbus,lnd.V0,lnd.pv_index,lnd.pq_index,max_iter,err_tol)

# Display results if the power flow analysis converged
if success:
    bus, branch = pf.DisplayResults(V,lnd)
    system = pf.getSystem(V,lnd)

    
    
    G, fig, ax = pf.plot_pf(bus,branch,n_pos)
    fig.tight_layout()
    # plt.show()
    plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\Special course - System identification of black-box dynamical systems\\img\\pf_graph.pdf')

# plt.close()

losses = abs(abs(branch['From Bus Inj.','P']) - abs(branch['To Bus Inj.','P']))


#%%
data = {
        'line':[],
        'r':[],
        'l':[],
        'b':[],
        }
for i, row in branch.iterrows():
    line = G.edges[row['From','Bus'],row['To','Bus']]
    data['line'].append(f'{line["k"]}-{line["m"]}')
    data['r'].append(line['Z'].real)
    data['l'].append(line['Z'].imag)
    data['b'].append(line['Y'].imag)

line_data = pd.DataFrame(data)
line_data.to_excel(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\pf_line_data.xlsx',header=True)

#%%
# Get undirected graph
G = G.to_undirected()

# Populate graph with needed data

for i,row in bus.iterrows():
    G.nodes[i]['V'] = cmplx(row['Voltage','Magn.'],row['Voltage','Angle'])
    # print(i,row['Voltage','Magn.'],row['Voltage','Angle'],G.nodes[i]['V'])
    G.nodes[i]['Vr'] = G.nodes[i]['V'].real
    G.nodes[i]['Vi'] = G.nodes[i]['V'].imag
    G.nodes[i]['id'] = i
    
    
for i, row in branch.iterrows():
    line = G.edges[row['From','Bus'],row['To','Bus']]
    line['k'] = int(row['From','Bus'])
    line['m'] = int(row['To','Bus'])

    # Get impedances
    line['Z'] = lnd.G.edges[line['k'],line['m']]['z']
    line['Y'] = lnd.G.edges[line['k'],line['m']]['yc']
    
    # Get power flow
    line['Sk'] = row['From Bus Inj.','P'] + 1j*row['From Bus Inj.','Q']
    line['Sm'] = row['To Bus Inj.','P'] + 1j*row['To Bus Inj.','Q']
    line['Pk'] = line['Sk'].real
    line['Pm'] = line['Sm'].real
    line['Qk'] = line['Sk'].imag
    line['Qm'] = line['Sm'].imag

    # line['Sk'] = line['S_in']
    # line['Sm'] = line['S_out']

    # Calculate line currents    
    line['Ik'] = conj(line['Sk']/G.nodes[line['k']]['V'])
    line['Im'] = conj(line['Sm']/G.nodes[line['m']]['V'])
    line['Ik,re'] = line['Ik'].real
    line['Im,re'] = line['Im'].real
    line['Ik,im'] = line['Ik'].imag
    line['Im,im'] = line['Im'].imag

    # Calculate shunt currents  
    # TODO: REFINE THIS ESTIMATION
    line['Ik,sh,re'] =   - line['Y'].imag * G.nodes[line['k']]['Vi']/1
    line['Ik,sh,im'] =     line['Y'].imag * G.nodes[line['k']]['Vr']/1
    line['Im,sh,re'] =   - line['Y'].imag * G.nodes[line['m']]['Vi']/1
    line['Im,sh,im'] =     line['Y'].imag * G.nodes[line['m']]['Vr']/1

    # Get voltages
    line['Vk']      = G.nodes[line['k']]['V']
    line['Vk,re']   = G.nodes[line['k']]['Vr']
    line['Vk,im']   = G.nodes[line['k']]['Vi']
    line['Vm']      = G.nodes[line['m']]['V']
    line['Vm,re']   = G.nodes[line['m']]['Vr']
    line['Vm,im']   = G.nodes[line['m']]['Vi']

    line['dV']      = line['Vk'] - line['Vm']
    line['dVr']      = line['Vk,re'] - line['Vm,re']
    line['dVi']      = line['Vk,im'] - line['Vm,im']

# summing up current injection
for node,row in bus.iterrows():

    connected_edges = G.edges(node, data=True)    

    # iterate through each edge and sum the edge_value
    S_line = complex(0,0)
    for _, _, edge in connected_edges:
        if node == edge['k']:
            S_line += -edge['Sk']
        else:
            S_line += -edge['Sm']

    S_line += G.nodes[node]['gen']

    G.nodes[node]['Sk'] = S_line
    G.nodes[node]['Pk'] = S_line.real
    G.nodes[node]['Qk'] = S_line.imag
    
    # Calculate shunt currents    
    G.nodes[node]['Ik,sh'] = conj(G.nodes[node]['Sk']/G.nodes[node]['V'])
    G.nodes[node]['Ik,sh,re'] = G.nodes[node]['Ik,sh'].real
    G.nodes[node]['Ik,sh,im'] = G.nodes[node]['Ik,sh'].imag
    
    G.nodes[node]['Ik,sh'] =    conj(G.nodes[node]['Sk']/G.nodes[node]['V'])
    G.nodes[node]['Yc'] =  Yc =  -conj(G.nodes[node]['Sk']/(G.nodes[node]['V']**2))

    G.nodes[node]['Ik,sh,re'] =   - Yc.imag * G.nodes[node]['Vi']
    G.nodes[node]['Ik,sh,im'] =     Yc.imag * G.nodes[node]['Vr']

    # G.nodes[node]['Ik,sh,re'] =   - * G.nodes[node]['Vi']
    # G.nodes[node]['Ik,sh,im'] =     G.nodes[node]['Y'].imag * G.nodes[node]['Vr']

    
    print(node,G.nodes[node]['Pk'],Yc)
 

#%%
def RLSE(H,R=None): #RECURSIVE LEAST SQUARES ESTIMATION
    # For Ax = b => x = inv(A.T @ A)
    N = len(G.edges)
    M = 3# Number of type parameters estimated
    A = np.zeros((N*4,M*N))
    B = np.zeros((4*N,1))

    
    if R is None:
        R= np.diag(np.ones(32))
    else:
        R=np.diag(R)
    
    # Handle edges
    for i, (node1, node2, d) in enumerate(G.edges.data()):
        idx1 = i*4
        idx2 = i*M
    
        A[idx1:idx1+4,idx2:idx2 + M] = np.array([
            [d['dVr'],   -d['dVi'], -d['Vk,im']],
            [d['dVr'],    d['dVr'],  d['Vk,re']],
            [-d['dVr'],   d['dVi'], -d['Vm,im']],
            [-d['dVi'],  -d['dVr'],  d['Vm,re']]
            ])
    
        B[idx1:idx1+4,0] = np.array([d['Ik,re'], 
                                     d['Ik,im'], 
                                     d['Im,re'], 
                                     d['Im,im']])
    
    X = (inv(A.T @ A) @ A.T) @ B

    return A, X, B

def LSE(G,R=None):
    # For Ax = b => x = inv(A.T @ A)
    N = len(G.edges)
    n = len(G.nodes)    
    M = 3 # Number of type parameters estimated
    A = np.zeros((2*n + 4*N,2*n + M*N))
    B = np.zeros((2*n + 4*N,1))
    

    A[:2*n,:2*n] = np.eye(2*n)
    B[0*n:1*n,0] = [G.nodes[i+1]['Vr'] for i in range(len(G.nodes))]
    B[1*n:2*n,0] = [G.nodes[i+1]['Vi'] for i in range(len(G.nodes))]
    
    if R is None:
        R= np.diag(np.ones(32))
    else:
        R=np.diag(R)

    A = np.zeros((4*N,M*N))
    B = np.zeros((4*N,1))
    # A = np.zeros((2*n + 4*N,2*n + M*N))
    # B = np.zeros((2*n + 4*N,1))

    # A[0*n:1*n,0*n:1*n] = np.diag([G.nodes[i+1]['Vr'] for i in range(len(G.nodes))])
    # A[1*n:2*n,1*n:2*n] = np.diag([G.nodes[i+1]['Vi'] for i in range(len(G.nodes))])
    # B[0*n:1*n,0] = [G.nodes[i+1]['Vr'] for i in range(len(G.nodes))]
    # B[1*n:2*n,0] = [G.nodes[i+1]['Vi'] for i in range(len(G.nodes))]
   
    # Create mappings
    node_to_row = {node: i for i, node in enumerate(G.nodes())}
    edge_to_col = {(min(edge), max(edge)): i for i, edge in enumerate(G.edges())}
    edge_to_col.update({(max(edge), min(edge)): i for i, edge in enumerate(G.edges())})

    # Reverse mappings
    row_to_node = {i: node for node, i in node_to_row.items()}
    col_to_edge = {i: edge for edge, i in edge_to_col.items()}

    rows_to_nodes = lambda x: tuple([row_to_node[elm] for elm in x])
    
    print("Node to row mapping:", node_to_row)
    print("Edge to column mapping:", edge_to_col)
    
    # Handle edges
    for i, (node1, node2, d) in enumerate(G.edges.data()):
        # indices
        nk_idx = node1-1
        nm_idx = node2-1
        # idx1 = i*4 + 2*n
        # idx2 = i*M + 2*n
        idx1 = i*4
        idx2 = i*M
        print(idx1,idx2)
        # Associate 
        # A[[idx1+0],[nk_idx,nk_idx+n]] += \
        #     -np.array([d['Pk']/(abs(d['Vk'])**2),d['Qk']/(abs(d['Vk'])**2)])
        # A[[idx1+1],[nk_idx+n,nk_idx]] += \
        #     -np.array([d['Pk']/(abs(d['Vk'])**2),-d['Qk']/(abs(d['Vk'])**2)])
        # A[[idx1+2],[nm_idx,nm_idx+n]] += \
        #     -np.array([d['Pm']/(abs(d['Vm'])**2),d['Qm']/(abs(d['Vm'])**2)])
        # A[[idx1+3],[nm_idx+n,nm_idx]] += \
        #     -np.array([d['Pm']/(abs(d['Vm'])**2),-d['Qm']/(abs(d['Vm'])**2)])
            

        
        # A[idx1:idx1+4,idx2:idx2 + 2] += np.array([
        #     [d['dVr'],   -d['dVi']],
        #     [d['dVi'],    d['dVr']],
        #     [-d['dVr'],   d['dVi']],
        #     [-d['dVi'],  -d['dVr']]
        #     ])
        
        # # Iterate thorugh incident matrix
        # # get the nodes
        # row1 = node_to_row[node1]
        # for k in G.neighbors(node1):
        #     edge = G.edges[(node1,k)]
        #     A[idx1:idx1+4,idx2+2 + 3*edge_to_col[node1,k]:idx2 + 3 + 3*edge_to_col[node1,k]] += np.array([
        #         [-edge['Vk,im']],
        #         [ edge['Vk,re']],
        #         [-edge['Vm,im']],
        #         [ edge['Vm,re']]
        #         ])                
        
        # for m in G.neighbors(node2):
        #     # if m != node1:
        #     edge = G.edges[(node2,m)]
        #     A[idx1:idx1+4,idx2+2 + 3*edge_to_col[node2,m]:idx2 + 3 + 3*edge_to_col[node2,m]] += np.array([
        #         [-edge['Vk,im']],
        #         [ edge['Vk,re']],
        #         [-edge['Vm,im']],
        #         [ edge['Vm,re']]
        #         ])                
                
       

        # for node_to_row[node1]:
        # edge_idx = rows_to_nodes(inc[:,i].nonzero()[0])


        # for j in range(len(G.nodes)):            
        #     # if element in incident matrix is one, then add voltages
        #     inc[:,1].nonzero()[0]
            
        #     if inc[j,i] == 1:               
        #         print(col_to_edge[j],node1,node2)
        #         edge = G.edges[()]
                
        #         A[idx1:idx1+4,idx2+2 + 3*:idx2 + 3] += np.array([
        #             [-edge['Vk,im']],
        #             [ edge['Vk,re']],
        #             [-edge['Vm,im']],
        #             [ edge['Vm,re']]
        #             ])
    
        # B[idx1:idx1+4,0] = np.array([
        #     d['Ik,re']    +   G.nodes[node1]['Ik,sh,re']/G.nodes[node1]['Vr'], 
        #     d['Ik,im']    +   G.nodes[node1]['Ik,sh,im']/G.nodes[node1]['Vi'], 
        #     d['Im,re']    +   G.nodes[node2]['Ik,sh,re']/G.nodes[node2]['Vr'], 
        #     d['Im,im']    +   G.nodes[node2]['Ik,sh,im']/G.nodes[node2]['Vi'], 
        #     ])
        
        # B[idx1:idx1+4,0] = np.array([
        #     d['Ik,re']    +   G.nodes[node1]['Ik,sh,re'], 
        #     d['Ik,im']  ,#  +   G.nodes[node1]['Ik,sh,im'], 
        #     d['Im,re']    +   G.nodes[node2]['Ik,sh,re'], 
        #     d['Im,im']  ,#  +   G.nodes[node2]['Ik,sh,im'], 
        #     ])
        
        A[idx1:idx1+4,idx2:idx2 + M] += np.array([
            [d['dVr'],   -d['dVi'], -d['Vk,im']],
            [d['dVi'],    d['dVr'],  d['Vk,re']],
            [-d['dVr'],   d['dVi'], -d['Vm,im']],
            [-d['dVi'],  -d['dVr'],  d['Vm,re']]
            ])
            
        B[idx1:idx1+4,0] = np.array([
            d['Ik,re'], 
            d['Ik,im'], 
            d['Im,re'], 
            d['Im,im'], 
            ])

        B[idx1:idx1+4,0] += np.array([
            d['Ik,sh,re'], 
            d['Ik,sh,im'], 
            d['Im,sh,re'], 
            d['Im,sh,im'], 
            ])
    
    X = (inv(A.T @ A) @ A.T) @ B
    
    # X = X[2*n:]

    return A, X, B

# X = X2
import pandas as pd
import scipy.stats as stats

# def confidenceInterval():
##%% RESIDUAL ANALYSIS
def residual_analysis(r,x,b,G, normalize=False,yerrs=False):
    N =len(G.edges)
    eps_r = np.zeros(N)
    eps_x = np.zeros(len(G.edges))
    eps_b = np.zeros(len(G.edges))
    
    for i, (node1, node2, d) in enumerate(G.edges.data()):
        eps_r[i] = (np.real(d['Z']) - r[i])/(1,np.real(d['Z']))[normalize]
        eps_x[i] = (np.imag(d['Z']) - x[i])/(1,np.imag(d['Z']))[normalize]
        eps_b[i] = (np.imag(d['Y']) - b[i])/(1,np.imag(d['Y']))[normalize]
     
    fig,ax = plt.subplots(3,1,dpi=150,sharex=True)
    for i in range(3):
        ax[i].grid()
        ax[i].set(ylabel='$\\varepsilon_{'+['R','X','B'][i]+'}$'+f' {("[p.u.]","[%]")[normalize]}')
    
    eps = np.concatenate([eps_r,eps_x,eps_b])
    eps_df = pd.DataFrame({'r':eps[:N],'l':eps[N:2*N],'b':eps[-N:],})
    if yerrs:
        yerr = {}
        for i, col in enumerate(eps_df.columns):
            std_residual = (eps_df[col]*100).std()
            margin_error = std_residual * stats.norm.ppf(0.975) # 95% CI, so 0.975 (which is 1 - (1 - 0.95)/2)
            yerr[col] = margin_error
    
        ax[0].bar([i for i in range(len(eps_r))],eps_r*(1,100)[normalize],zorder=3,yerr=yerr['r'],capsize=3)
        ax[1].bar([i for i in range(len(eps_r))],eps_x*(1,100)[normalize],zorder=3,yerr=yerr['l'],capsize=3)
        ax[2].bar([i for i in range(len(eps_r))],eps_b*(1,100)[normalize],zorder=3,yerr=yerr['b'],capsize=3)
    else:
        ax[0].bar([i for i in range(len(eps_r))],eps_r*(1,100)[normalize],zorder=3)
        ax[1].bar([i for i in range(len(eps_r))],eps_x*(1,100)[normalize],zorder=3)
        ax[2].bar([i for i in range(len(eps_r))],eps_b*(1,100)[normalize],zorder=3)
        
    
    ax[2].set_xticks([i for i in range(len(eps_r))])
    ax[2].set_xticklabels([str(list(G.edges)[i]) for i in range(len(G.edges))])
    ax[2].set_xlabel('Lines')


    fig.tight_layout()



    plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\Special course - System identification of black-box dynamical systems\\img\\pf_estimations_{normalize}.pdf')

    return eps

A, X, B = LSE(G)

fig, ax =plt.subplots(1,1,dpi=150)
ax.imshow(np.where(abs(np.hstack([A,B]))!=0,abs(np.hstack([A,B])),np.nan))
# ax.axvline(len(G.nodes)-0.5,color='k',lw=0.75,alpha=0.5)
# ax.axhline(len(G.nodes)-0.5,color='k',lw=0.75,alpha=0.5)
# ax.axvline(len(G.nodes)*2-0.5,color='k',lw=0.75)
# ax.axhline(len(G.nodes)*2-0.5,color='k',lw=0.75)
# ax.axvline(len(G.nodes)*2+len(G.edges)*3-0.5,color='k',lw=0.75)
ax.axvline(len(G.edges)*3-0.5,color='k',lw=0.75)

Z = np.array([1/complex(r,xl) for r,xl in zip(X[::3],X[1::3])])
r = np.real(Z)
x = np.imag(Z)
y = np.array([complex(0,y_) for y_ in X[2::3]])
b = np.imag(y)
fig.tight_layout()
ax.set(xlabel='index',ylabel='index')
plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\Special course - System identification of black-box dynamical systems\\img\\pf_matrix.pdf')

eps1 = residual_analysis(r,x,b,G)


#%%

data = {
        'Line':[],
        'R_{True}':[],
        'R_{Est.}':[],
        'X_{True}':[],
        'X_{Est.}':[],
        'B_{True}':[],
        'B_{Est.}':[],
        }
cnt = 0

for i, (node1, node2, d) in enumerate(G.edges.data()):
    line = d
    data['Line'].append(f'({line["k"]},{line["m"]})')
    data['R_{True}'].append(line['Z'].real)
    data['X_{True}'].append(line['Z'].imag)
    data['B_{True}'].append(line['Y'].imag)
    data['R_{Est.}'].append(r[cnt])
    data['X_{Est.}'].append(x[cnt])
    data['B_{Est.}'].append(b[cnt])
    
    cnt += 1 

line_data = pd.DataFrame(data)

line_data.to_excel(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\pf_line_data_estimated.xlsx',header=True)


#%% ==================== WEIGHTED LEAST SQUARES ==================== 
def WLS(X, Y, W):
    # Create a diagonal matrix using the weights
    W_sqrt = np.sqrt(np.diagflat(W))
    
    # Apply the square root of the weights to X and Y
    X_weighted = np.matmul(W_sqrt, X)
    Y_weighted = np.matmul(W_sqrt, Y)
    
    # Calculate the parameters
    X_weighted_transpose = np.transpose(X_weighted)
    XWX_inverse = inv(np.matmul(X_weighted_transpose, X_weighted))
    b = np.matmul(np.matmul(XWX_inverse, X_weighted_transpose), Y_weighted)
    
    return b

def create_weight_vector(G):
    N = 4
    W = np.zeros(len(G.edges)*N)
    for i, (n1,n2, d) in enumerate(G.edges.data()):
        W[i*N:i*N + N] = abs(d['Sk'])        

        W[i]   = abs(1e6*d['Sk'])        
        W[i+1] = 1/abs(1e6*d['Sk'])        
        W[i+2] = abs(1e6*d['Sk'])        
        W[i+3] = 1/abs(1e6*d['Sk'])        

    
    return W

def WLS(X, Y, W):
    # Create a diagonal matrix using the weights
    W_sqrt = np.sqrt(np.diagflat(W))
    
    # Apply the square root of the weights to X and Y
    X_weighted = W_sqrt @ X
    Y_weighted = W_sqrt @ Y
    
    # Calculate the parameters
    X_weighted_transpose = X_weighted.T
    XWX_inverse = inv(X_weighted_transpose @ X_weighted)
    b = XWX_inverse @ X_weighted_transpose @ Y_weighted
    
    return b

W = create_weight_vector(G)

# B = WLS(X, Y, W)
# B = WLS(A, B, W) #  WLS(X, Y, W), where Y = Xb + e

# Create a diagonal matrix using the weights
W_sqrt = np.sqrt(np.diagflat(W))

# Apply the square root of the weights to X and Y
X_weighted = W_sqrt @ A
Y_weighted = W_sqrt @ B 

# Calculate the parameters
X_weighted_transpose = X_weighted.T
XWX_inverse = inv(X_weighted_transpose @ X_weighted)
X = XWX_inverse @ X_weighted_transpose @ Y_weighted

Z = np.array([1/complex(r,xl) for r,xl in zip(X[::3],X[1::3])])
r = np.real(Z)
x = np.imag(Z)
y = np.array([complex(0,y_) for y_ in X[2::3]])
b = np.imag(y)

eps2 = residual_analysis(r,x,b,G)

fig,ax = plt.subplots(1,1,dpi=150)
plt.step([i for i in range(len(W))],W)

#%%
# theta, V_hat, system_hat = pf.MLE()
# r = theta.x[::3]
# x = theta.x[1::3]
# b = theta.x[2::3]

# #%%
# eps2 = residual_analysis(r,x,b,G)

# #%%
# fig,ax = plt.subplots(1,1,dpi=150)
# ax.bar([i for i in range(len(system))],system - system_hat)

#%%
# data = pf.MCS(7*10)
# min_J_index = data['J'].idxmin()
# min_J_parameters = data.loc[min_J_index, [col for col in data.columns if col != 'J']]


#%%
import numpy as np

class RLS:
    def __init__(self, n_features, n_outputs, beta_init=None, P_init=None):
        self.n_features = n_features
        self.beta = beta_init if beta_init is not None else np.zeros(n_outputs)
        self.P = P_init if P_init is not None else np.eye(n_features)

    def update(self, x, y):
        x = x.reshape(-1, 1)
        y = np.array([y])
        
        # Compute the gain vector
        K = np.dot(self.P, x) / (1 + np.dot(x.T, np.dot(self.P, x)))

        # Update the parameter estimate
        self.beta += K.flatten() * (y - np.dot(x.T, self.beta))

        # Update the covariance matrix
        self.P = self.P - np.dot(K, np.dot(x.T, self.P))

    def predict(self, x):
        x = x.reshape(-1, 1)
        return np.dot(x.T, self.beta)

# Example usage:
rls = RLS(n_features=2,n_outputs=2)

for _ in range(1000):
    x = np.random.rand(2)  # Input features
    y = np.dot(x, [2, -3]) + np.random.normal(0, 0.1)  # Output (with some noise)
    rls.update(x, y)

print(rls.beta)  # Should be close to [2, -3]
#%%
class RecursiveLeastSquares:
    def __init__(self, n_features,n_observations, lambda_=1, P_init=None, beta_init=None):
        self.n_features = n_features
        self.lambda_ = lambda_
        self.beta = beta_init if beta_init is not None else np.zeros(n_features)
        self.P = P_init if P_init is not None else np.eye(n_features) * 1000  # high uncertainty in initial estimates

    def update(self, x, y):
        x = np.array(x).reshape(-1, 1)
        y = np.array([y])
        
        # Compute the gain vector
        K = self.P @ x / (self.lambda_ + x.T @ self.P @ x)

        # Update the parameter estimate
        self.beta += (K * (y - x.T @ self.beta)).flatten()

        # Update the covariance matrix
        self.P = (self.P - K @ x.T @ self.P) / self.lambda_

    def predict(self, x):
        x = np.array(x).reshape(-1, 1)
        return np.dot(x.T, self.beta)

rls = RecursiveLeastSquares(n_features=len(X),n_observations=len(B))
# Assume data_stream and output_stream are generator functions or similar that provide
# the feature vectors and outputs one at a time
# for x, y in zip(data_stream, output_stream):
# preds = np.
for _ in range(1000):
    for i in range(len(B)):
        x = A[i]  # This is a vector of 24 features
        y = B[i]  # This is the corresponding output
        rls.update(x, y) 
        prediction = rls.predict(x)


print(rls.beta)  # Should be close to [2, -3]
print(B.ravel())  # Should be close to [2, -3]

#%%

# def LSE_(x, H, z_hx, W=None, tol=1e-6, max_iter=1000):
#     """
#     This function implements the state update process, including state correction and convergence check.

#     Parameters:
#     x: Current state vector
#     H: Jacobian matrix
#     W: Weight matrix
#     z_hx: Difference between actual and predicted measurements
#     tol: Convergence tolerance
#     max_iter: Maximum number of iterations

#     Returns:
#     x: Updated state vector
#     converged: Whether the solution has converged
#     """
#     # Initialize convergence flag
#     converged = False
    
#     if W is None: W = np.ones(H.shape)

#     for _ in range(max_iter):
#         # Calculate the correction vector
#         delta_x = np.linalg.inv(H.T @ W @ H) @ (H.T @ W @ z_hx)

#         # Update the state estimate
#         x += delta_x

#         # Check for convergence
#         if np.linalg.norm(delta_x) < tol:
#             converged = True
#             break

#     return x, converged

# LSE_(X,A,eps1)





