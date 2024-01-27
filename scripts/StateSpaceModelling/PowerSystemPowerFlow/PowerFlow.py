# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:37:00 2022

@author: bvilm
"""
import pandas as pd
import networkx as nx
import numpy as np
from numpy import diag
from numpy.linalg import inv, det
import copy
from scipy.optimize import minimize, LinearConstraint
import PowerSystemPowerFlow.LoadNetworkData
import matplotlib.colors as mcolors

from tqdm import tqdm

#%%
def load_power_system(theta=None):
    # Load the Network data ...
    lnd = LoadNetworkData.LoadNetworkData()

    # Adding lines and transformers (bus1, bus2, r, x, g, b)
    if theta is not None:
        lines = [(1, 2), (1, 7), (1, 6), (3, 2), (4, 3), (4, 7), (4, 5), (5, 6)]
        for i in range(8):
            node1, node2 = lines[i]
            idx = i*3
            # lnd.add_line(node1,node2,np.exp(theta[idx]),np.exp(theta[idx+1]),b=np.exp(theta[idx+2]),name=f'Z{i}')
            # print(lines[i],(theta[idx]),(theta[idx+1]),(theta[idx+2]))

            lnd.add_line(node1, node2,(theta[idx]),(theta[idx+1]),b=(theta[idx+2]),name=f'Z{i}')
            
    else:
        lnd.add_line(1,2,0.04,0.52,name='Za',b=0.02)
        lnd.add_line(2,3,0.03,0.47,name='Zb',b=0.02)
        lnd.add_line(3,4,0.05,0.42,name='Zps',b=0.02)
        lnd.add_line(1,7,0.04,0.43,name='Zc',b=0.02)
        lnd.add_line(4,7,0.01,0.5,name='Zd',b=0.02)
        lnd.add_line(1,6,0.04,0.46,name='Ze',b=0.02)
        lnd.add_line(5,6,0.03,0.45,name='Zf',b=0.02)
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
    
    return lnd

def MCS(n_iters,r0=0.055,x0= 0.526,b0=0.052,max_iter = 30, err_tol=1e-4):
    
    # Get ground truth
    lnd = load_power_system()
    V,success,n, J, F = pf.PowerFlowNewton(lnd.Ybus,lnd.Sbus,lnd.V0,lnd.pv_index,lnd.pq_index,max_iter,err_tol)

    system = getSystem(V,lnd)

    # Defining constraints (only reasonable for not optimization without log)
    N = len(lnd.G.edges)*3    

    data = {'J':[]}
    for i, line in enumerate(lnd.G.edges):   
        data[f'R_{line}'] = []            
        data[f'X_{line}'] = []            
        data[f'B_{line}'] = []            

    for it in tqdm(range(int(n_iters))):
        # Perturb parameters
        theta = np.zeros(N)
        cnt = 0
        for i, line in enumerate(lnd.G.edges):   
            for j in range(3):
                val = [r0,x0,b0][j]
                delta = val*0.5
                rand_val = np.random.uniform(low=val - delta, high=val + delta, size=None)
                theta[cnt] = rand_val
                try: 
                    data[f'{["R","X","B"][j]}_{line}'].append(rand_val)
                except KeyError as e:
                    data[f'{["R","X","B"][j]}_{line[::-1]}'].append(rand_val)
                    
                
                cnt += 1

        # create system
        lnd = load_power_system(theta)
            
        # Evaluating cost function
        J = ML_(system,lnd,max_iter , err_tol)
            
        data['J'].append(J)
            
    return pd.DataFrame(data)

def MLE(r0=0.055,x0= 0.526,b0=0.052,max_iter = 30, err_tol=1e-4):

    # Get ground truth
    lnd = load_power_system()
    V,success,n, J, F = pf.PowerFlowNewton(lnd.Ybus,lnd.Sbus,lnd.V0,lnd.pv_index,lnd.pq_index,max_iter,err_tol)

    system = getSystem(V,lnd)

    # Defining constraints (only reasonable for not optimization without log)
    N = len(lnd.G.edges)*3

    x0 = [[r0,x0,b0][i%3] for i in range(N)]

    constraints = (LinearConstraint(np.eye(N), lb=np.zeros(N), ub=np.exp(np.ones(N)), keep_feasible=False))
    
    # Minimization
    thetahat = minimize(ML_Wrapper,
                        args=(system, max_iter , err_tol),
                        x0=x0,
                        method='SLSQP',
                        jac = '2-point',
                        constraints=constraints,
                        options={
                            'disp':True,
                            'eps':1e-8,
                            'finite_diff_rel_step':1e-8,
                            },
                        )
    # thetahat.x = np.exp(thetahat.x)
    
    lnd = load_power_system(thetahat.x)
    V_hat, success, n, J, F = pf.PowerFlowNewton(lnd.Ybus,lnd.Sbus,lnd.V0,lnd.pv_index,lnd.pq_index,max_iter,err_tol,silence=True)
    system_hat = getSystem(V_hat,lnd)

    return thetahat, V_hat, system_hat

def ML_Wrapper(theta,system,max_iter, err_tol):

    theta = np.where(theta <= 0,1e-10,theta)
    
    # create system
    lnd = load_power_system(theta)
        
    # Evaluating cost function
    J = ML_(system,lnd,max_iter , err_tol)

    return J


def ML_(system,lnd,max_iter , err_tol):        

    # Get covariance and prediction error
    V_hat,success,n, J, F = pf.PowerFlowNewton(lnd.Ybus,lnd.Sbus,lnd.V0,lnd.pv_index,lnd.pq_index,max_iter,err_tol,silence=True)
        
    system_hat = getSystem(V_hat,lnd)
    
    # Get covariance and prediction error
    eps = system - system_hat

    #
    n = len(V_hat)
    m = len(lnd.G.edges)
    r_vm = 0.1*np.ones(n)
    r_vp = 0.05*np.ones(n)
    r_sgen = 4*np.ones(n*2)
    r_sload = 4*np.ones(n*2)
    r_sfrom = 0.1*np.ones(m*2)
    r_sto = 0.1*np.ones(m*2)
    R = np.diag(np.concatenate([r_vm,r_vp,r_sgen,r_sload,r_sfrom,r_sto]))

    R = np.diag(np.ones(len(eps)))

    # Evaluate the cost function from the batch sum
    J = 1/2*(np.log(det(R)) \
              + np.log(2*np.pi)\
              + eps.T @ inv(R) @ eps)
    J += (1e6,0)[success]
    
    # print(J)
    
    return J
    


#%%
# Calculate mismatch function
def calculate_F(Ybus,Sbus,V,pv_index,pq_index):

    Delta_S = Sbus - V * (Ybus.dot(V)).conj()
    Delta_P = np.real(Delta_S)    
    Delta_Q = np.imag(Delta_S)    
    
    F = np.concatenate((Delta_P[pv_index],Delta_P[pq_index],Delta_Q[pq_index]),axis=0)
    
    return F

# Check tolerances
def CheckTolerance(F,n,err_tol = 1e-7,silence=False):
    # Take the infinity norm of the mismatch function (taking the largest mismatch)
    normF = np.linalg.norm(F,np.inf)
    if not silence: print(f'Iteration {n} - error: {normF}')

    # Checking if the largest mismatch is less than the tolerance
    success = (0,1)[normF < err_tol]

    return success

def generate_Derivatives(Ybus,V):
    
    # Partial derivative with respect to voltage magnitude V
    J_ds_dVm= diag(V/np.absolute(V)) @ diag((Ybus.dot(V)).conj())+ \
        diag(V) @ (Ybus @ np.diag(V/np.absolute(V))).conj()
    
    # Partial derivative with respect to voltage angle theta
    J_dS_dTheta = 1j*np.diag(V) @ (np.diag(Ybus.dot(V))-Ybus.dot(np.diag(V))).conj()    
    
    return J_ds_dVm, J_dS_dTheta

def generate_Jacobian(J_dS_dVm,J_dS_dTheta,pv_index,pq_index):

    # Append PV and PQ indices for convenience
    pvpq_ind = np.append(pv_index, pq_index)

    # Create the sub-matrices
    J_11 = np.real(J_dS_dTheta[np.ix_(pvpq_ind, pvpq_ind)])
    J_12 = np.real(J_dS_dVm[np.ix_(pvpq_ind, pq_index)])
    J_21 = np.imag(J_dS_dTheta[np.ix_(pq_index, pvpq_ind)])
    J_22 = np.imag(J_dS_dVm[np.ix_(pq_index, pq_index)])
    
    # Partitioning
    J = np.block([[J_11,J_12],[J_21,J_22]])
    
    return J

def Update_Voltages(dx,V,pv_index,pq_index):
    # Note: difference between Python and Matlab when using indices
    N1 = 0; N2 = len(pv_index) # dx[N1:N2]-ang. on the pv buses
    N3 = N2; N4 = N3 + len(pq_index) # dx[N3:N4]-ang. on the pq buses
    N5 = N4; N6 = N5 + len(pq_index) # dx[N5:N6]-mag. on the pq buses

    Theta = np.angle(V); Vm = np.absolute(V)
    if len(pv_index)>0:
        Theta[pv_index] += dx[N1:N2]
    if len(pq_index)>0:
        Theta[pq_index] += dx[N3:N4]
        Vm[pq_index] += dx[N5:N6]

    V = Vm * np.exp(1j*Theta)

    return V

def PowerFlowNewton(Ybus,Sbus,V0,pv_index,pq_index,max_iter,err_tol,silence=False):
    success = 0 # Initialization of flag, counter and voltage
    n = 0
    V = V0
    if not silence: print(' iteration maximum P & Q mismatch (pu)')
    if not silence: print(' --------- ---------------------------')
    # Determine mismatch between initial guess and and specified value for P and Q
    F = calculate_F(Ybus,Sbus,V,pv_index,pq_index)

    success = CheckTolerance(F,n,err_tol,silence=silence) # Check if the desired tolerance is reached

    # Iterative newton-rhapson method
    while (not success) and (n < max_iter): # Start Newton iterations
        # Update counter
        n += 1

        # Generate the Jacobian matrix
        J_dS_dVm,J_dS_dTheta = generate_Derivatives(Ybus,V)
        J = generate_Jacobian(J_dS_dVm,J_dS_dTheta,pv_index,pq_index)

        # Compute step
        dx = np.linalg.solve(J,F)
        # print(dx)
        # print(np.linalg.inv(J) @ F)
        
        # Update voltages and check if tolerance is now reached
        V = Update_Voltages(dx,V,pv_index,pq_index)

        # Calculate mismatch function        
        F = calculate_F(Ybus,Sbus,V,pv_index,pq_index)
        
        # Check if infinity norm of F < epsilon
        success = CheckTolerance(F,n,err_tol,silence=silence)

    if success:
        if not silence: print(f'The Newton Rapson Power Flow Converged in {n} iterations!\nError tolerance: {err_tol}')
    else:
        if not silence: print('No Convergence!!\n Stopped after %d iterations without solution...' % (n,))
    return V,success,n, J, F
        
def getSystem(V, lnd):
    S = V * (lnd.Ybus.dot(V)).conj()
    P = np.real(S)
    Q = np.imag(S)
    Pgen = np.where((P>0) & (lnd.buscode != 1),P,np.nan)
    Qgen = np.where((P>0) & (lnd.buscode != 1),Q,np.nan)
    Pload = np.where(P<0,P,np.nan)
    Qload = np.where((lnd.buscode == 1),Q,np.nan)
    
    # Branch part
    Sfrom = V[lnd.br_f] * (lnd.Y_from.dot(V)).conj()
    Pfrom = np.real(Sfrom)
    Qfrom = np.imag(Sfrom)

    Sto = V[lnd.br_t] * (lnd.Y_to.dot(V)).conj()
    Pto = np.real(Sto)
    Qto = np.imag(Sto)

    # Concatenate data
    result_vector = np.concatenate([abs(V),np.angle(V),Pgen,Pload,Qgen,Qload,Pfrom,Pto,Qfrom,Qto])
    result_vector = np.nan_to_num(result_vector, nan = 0) 

    return result_vector 

def DisplayResults(V, lnd, n_digits = 3, silence:bool=True):
    # Creating print values
    vm, vphi = abs(V), np.angle(V)*180/np.pi
    S = V * (lnd.Ybus.dot(V)).conj()
    P = np.real(S)
    Q = np.imag(S)
    Pgen = np.where((P>0) & (lnd.buscode != 1),P,np.nan)
    Qgen = np.where((P>0) & (lnd.buscode != 1),Q,np.nan)
    Pload = np.where(P<0,P,np.nan)
    Qload = np.where((lnd.buscode == 1),Q,np.nan)

    # ==================== BUS PRINT ====================
    arrays = [['Bus', 'Voltage', 'Voltage', 'Generation', 'Generation', 'Load', 'Load'], ['Type', 'Magn.', 'Angle', 'P', 'Q', 'P', 'Q']]
    cols = pd.MultiIndex.from_arrays(arrays).values
    bus = pd.DataFrame(columns=cols)    
    bus.columns = pd.MultiIndex.from_tuples(cols)
    for i in range(len(V)):
        bus.at[i,cols[0]] = lnd.buscode[i]
        bus.at[i,cols[1]] = round(vm[i],n_digits)
        bus.at[i,cols[2]] = round(vphi[i],n_digits)
        bus.at[i,cols[3]] = round(Pgen[i],n_digits)
        bus.at[i,cols[4]] = round(Qgen[i],n_digits)
        bus.at[i,cols[5]] = -round(Pload[i],n_digits)
        bus.at[i,cols[6]] = -round(Qload[i],n_digits)

    bus.index.name = '#'
    bus.index += 1
    bus = bus.fillna('-')
    if not silence:
        print('#======================================================#')
        print('#                      BUS RESULTS                     #')
        print('#======================================================#')
        print(bus)
        print('#======================================================#\n')

    # ==================== BRANCH PRINT ====================
    # lnd.Ybus, lnd.Y_from, lnd.Y_to, lnd.br_f, lnd.br_t, lnd.buscode
    Sfrom = V[lnd.br_f] * (lnd.Y_from.dot(V)).conj()
    Pfrom = np.real(Sfrom)
    Qfrom = np.imag(Sfrom)

    Sto = V[lnd.br_t] * (lnd.Y_to.dot(V)).conj()
    Pto = np.real(Sto)
    Qto = np.imag(Sto)
    
    arrays = [['From', 'To', 'From Bus Inj.', 'From Bus Inj.', 'To Bus Inj.', 'To Bus Inj.'], ['Bus', 'Bus','P', 'Q', 'P', 'Q']]
    cols = pd.MultiIndex.from_arrays(arrays).values
    branch = pd.DataFrame(columns=cols)    
    branch.columns = pd.MultiIndex.from_tuples(cols)
    for i in range(len(lnd.br_f)):
        branch.at[i,cols[0]] = lnd.br_f[i] + 1
        branch.at[i,cols[1]] = lnd.br_t[i] + 1
        branch.at[i,cols[2]] = round(Pfrom[i],n_digits)
        branch.at[i,cols[3]] = round(Qfrom[i],n_digits)
        branch.at[i,cols[4]] = round(Pto[i],n_digits)
        branch.at[i,cols[5]] = round(Qto[i],n_digits)

    # bus = bus.set_index(('Bus','#'))
    # for col in bus.columns:
    #     bus[col] = round(bus[col],3)
    branch.index.name = '#'
    branch.index += 1
    branch = branch.fillna('-')

    if not silence:
        print('#======================================================#')
        print('#                      BRANCH FLOW                     #')
        print('#======================================================#')
        print(branch)
        print('#======================================================#\n')
    
    return bus, branch


import matplotlib.pyplot as plt
import matplotlib

def plot_pf(bus,branch,n_pos=None,seed = 2000):
    # Creating directional graph
    G = nx.DiGraph()
    
    # Appending edges to graph
    for i, row in branch.iterrows():
        # Order of edges (direction of the edge) depends on power flow
        if row[('From Bus Inj.','P')] < 0:
            G.add_edge(int(row[('To','Bus')]), int(row[('From','Bus')]), S_in = -complex(row[('To Bus Inj.','P')],row[('To Bus Inj.','Q')]),S_out = -complex(row[('From Bus Inj.','P')],row[('From Bus Inj.','Q')]),d='->')
        else:
            G.add_edge(int(row[('From','Bus')]), int(row[('To','Bus')]), S_in = -complex(row[('From Bus Inj.','P')],row[('From Bus Inj.','Q')]),S_out = -complex(row[('To Bus Inj.','P')],row[('To Bus Inj.','Q')]),d='->')

    # Update nodes with power generation and load
    for i,row  in bus.iterrows():
        # If bus generates or consumes power
        if isinstance(row[('Generation','P')],float):            
            # nx.set_node_attributes(G, {i: {'gen':complex(row[('Generation','P')],row[('Generation','Q')])}})        
            nx.set_node_attributes(G, {i: {'gen':-f"({row[('Generation','P')]},{row[('Generation','Q')]})"}})
        elif isinstance( row[('Load','P')],float):            
            nx.set_node_attributes(G, {i: {'gen':-f"({row[('Load','P')]},{row[('Load','Q')]})"}})
        else:
            nx.set_node_attributes(G, {i: {'gen':0}})

        # Create the node label for plotting        
        nx.set_node_attributes(G, {i: {'name':f'{i}\n{G.nodes[i]["gen"]}'}})        
        nx.set_node_attributes(G, {i: {'V':row[('Voltage','Magn.')]}})        

    # Create the edge label for plotting
    for e in list(G.edges):
        nx.set_edge_attributes(G, {e: {'name': f'IN: {-G.edges[e]["S_in"]}\nOUT: {G.edges[e]["S_out"]}'}})

    # Find maximum power flow in the system to colormap power flow in lines.
    Smax = max([abs(G[e[0]][e[1]]['S_in']) for e in list(G.edges)])
    cmap=plt.get_cmap('RdYlGn_r')
    
    # Append colormapped power flow 
    clrs = []
    for k, v in nx.get_edge_attributes(G,'S_in').items():
        clrs.append(abs(v)/Smax)

    # Convert rgba-tuples to hex
    clrs = [matplotlib.colors.to_hex(cmap(abs(abs(v)/Smax))) for k,v in nx.get_edge_attributes(G,'S_in').items()]

    # Create figure and draw network
    fig, ax = plt.subplots(1,1,figsize=(12,8),dpi=200)
    nx.draw_networkx_edges(G,edge_color='k',width=5.,pos=n_pos,ax=ax,arrowstyle='-|>')

    # Define the color mapping
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["blue", "lightblue", "green", "yellow", "red"])  
    # Get the values from the nodes
    values = [node[1]['V'] for node in G.nodes(data=True)]
    
    # Normalize the values to [0,1] for color mapping
    norm = mcolors.Normalize(vmin=0.9, vmax=1.1)

    if n_pos is None:
        n_pos = nx.spring_layout(G)

    nx.draw(G,n_pos,node_color=cmap(norm(values)),width=5,with_labels=True, arrowsize=30,edge_color=clrs,node_size = [2000]*len(G.nodes), labels=nx.get_node_attributes(G,'name'),ax=ax,font_size=16)

    nx.draw_networkx_edge_labels(G, n_pos, edge_labels=nx.get_edge_attributes(G,'name'),font_color='black',ax=ax,font_size=14)

    # hardcoded xlim to ensure plot is visible
    ax.set_xlim(-1,6)
    # Show the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)



    return G, fig, ax

