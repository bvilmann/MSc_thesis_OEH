# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:50:37 2023

@author: bvilm
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import block_diag
from numpy.linalg import inv,eig
from StateSpaceModelling import StateSpaceModelling as SSM
import control
import matplotlib.cm as cm

def insert_axis(ax, xy_coords: tuple, axin_pos: tuple,
                box_kwargs=dict(color='grey', ls='-'),
                box: bool = True,
                grid: bool = True,
                arrow_pos: tuple = None,
                arrowprops: dict = dict(arrowstyle='->', color='grey', alpha=0.75, ls='-'),
                arrow_kwargs: dict = dict(xycoords='data', textcoords='data'),
                lw = 0.75,
                ):
    xpos, ypos, width, height = axin_pos
    x1, x2, y1, y2 = xy_coords

    axin = ax.inset_axes([xpos, ypos, width, height])

    axin.set_xlim(x1, x2)
    axin.set_ylim(y1, y2)

    # Options
    if arrow_pos is not None:
        x1a, x2a, y1a, y2a = arrow_pos
        ax.annotate('', xy=(x1a, y1a),
                    xytext=(x2a, y2a),
                    arrowprops=arrowprops,
                    **arrow_kwargs
                    )

    if grid:
        axin.grid()
    if box:
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], lw=lw, **box_kwargs)

    return axin, ax


# Replace 'your_file.xlsx' with the path to your Excel file
path = r'C:\Users\bvilm\PycharmProjects\StateSpaceModelling\statespacemodels'
file_path = f'{path}\\ssm_ccm.xlsx'

#%% ======================== PLOT EXEMPLARY GRAPH

# Read the specific sheet 'L_map'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# df now contains the data from the 'L_map' sheet
u_c = df[(df.group == 'comp') & (df.vector == 'input')].sort_values(['comp','idx'], inplace=False)
y_c = df[(df.group == 'comp') & (df.vector == 'output')].sort_values(['comp','idx'], inplace=False)
u_s = df[(df.group == 'sys') & (df.vector == 'input')].sort_values(['comp','idx'], inplace=False)
y_s = df[(df.group == 'sys') & (df.vector == 'output')].sort_values(['comp','idx'], inplace=False)

# Create a directed graph
G = nx.DiGraph()

# Add any additional vertices if needed
G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}{n['comp']}_{n['name']}" for i, n in u_c.iterrows()])
G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}{n['comp']}_{n['name']}" for i, n in y_c.iterrows()])
G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}{n['comp']}_{n['name']}" for i, n in u_s.iterrows()])
G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}{n['comp']}_{n['name']}" for i, n in y_s.iterrows()])

node_clr = ['red']*len(u_c)+['pink']*len(y_c)+['blue']*len(u_s)+['lightblue']*len(y_s)


edges = []
edge_clrs = []
# Set the seed for reproducibility
for i, pair in enumerate([(y_c,u_c),(u_s,u_c),(y_c,y_s),(u_s,y_s)]):
    np.random.seed(42)
    from_,to_ = pair   
    from_ = from_.reset_index()
    to_ = to_.reset_index()
    for j, node_j in to_.iterrows():
        # print(j,node_j['name'])
        for k, node_k in from_.iterrows():
            # print(k,node_k['name'])
            # print(j,k)
            if node_j['name'] == node_k['name']:
                
                v_k = f"{node_k['group'][0]}{node_k['vector'][0]}{node_k['comp']}_{node_k['name']}" # from
                v_j = f"{node_j['group'][0]}{node_j['vector'][0]}{node_j['comp']}_{node_j['name']}" # to
                edges.append((v_k,v_j))
                edge_clrs.append(['C0','C1','C2','C4'][i])
            


# Add edges to the graph
G.add_edges_from(edges)

plt.figure(figsize=(8, 6), dpi=150)

# pos = nx.spring_layout(G, seed=46)  # The seed ensures consistent layout
pos = nx.circular_layout(G)  # The seed ensures consistent layout
pos = {}
dx, dy = 10,3
max_n_layer = 0
for i, layer in enumerate(['si','co','ci','so']):
    nodes = [n for n in G if n[:2] == layer]
    max_n_layer = max(max_n_layer,len(nodes))


for i, layer in enumerate(['si','co','ci','so']):
    nodes = [n for n in G if n[:2] == layer]
    n_layer = len(nodes)
    
    for j, node in enumerate(nodes):
        pos[node] = np.array([i*dx,-j*dy-(max_n_layer-n_layer*1.6)])        

# Now G is a directed graph with edges and vertices as defined
# Draw the graph
# nx.draw(G,pos, with_labels=True, node_color='lightblue', node_size=5, arrowstyle='->', arrowsize=10)
nx.draw_networkx_nodes(G, pos, node_color=node_clr)
nx.draw_networkx_edges(G, pos, edge_color=edge_clrs)
nx.draw_networkx_labels(G, pos)

# Show the plot
# Set equal aspect ratio
# plt.gca().set_aspect('equal', adjustable='box')

# Show the plot
plt.show()





#%% ================== DERIVE INTERCONNECTION MATRICES, L1, L2, L3, L4 ==================

# Read the specific sheet 'L_map'
df = pd.read_excel(file_path, sheet_name='HSC1_L')

# df now contains the data from the 'L_map' sheet
u_c = df[(df.group == 'comp') & (df.vector == 'input')].sort_values(['comp','idx'], inplace=False)
y_c = df[(df.group == 'comp') & (df.vector == 'output')].sort_values(['comp','idx'], inplace=False)
u_s = df[(df.group == 'sys') & (df.vector == 'input')].sort_values(['comp','idx'], inplace=False)
y_s = df[(df.group == 'sys') & (df.vector == 'output')].sort_values(['comp','idx'], inplace=False)

# Create a directed graph
G = nx.DiGraph()

# Add any additional vertices if needed
G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}_{n['name']}" for i, n in u_c.iterrows()])
G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}_{n['name']}" for i, n in y_c.iterrows()])
G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}_{n['name']}" for i, n in u_s.iterrows()])
G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}_{n['name']}" for i, n in y_s.iterrows()])

# print(u_c,y_c,u_s,y_s)
L = {}
edges = []
edge_clrs = []
# Set the seed for reproducibility
for i, pair in enumerate([(y_c,u_c),(u_s,u_c),(y_c,y_s),(u_s,y_s)]):
    np.random.seed(42)
    from_,to_ = pair   
    from_ = from_.reset_index()
    to_ = to_.reset_index()
    L[int(f'{i+1}')] = np.zeros((len(to_),len(from_)))
    # print(L[int(f'{i+1}')].shape)
    for j, node_j in to_.iterrows():
        # print(j,node_j['name'])
        for k, node_k in from_.iterrows():
            # print(k,node_k['name'])
            # print(j,k)
            if node_j['name'] == node_k['name']:
                L[int(f'{i+1}')][j,k] = 1
    
    
#%%
import xml.etree.ElementTree as ET
import re
class PSCAD_loader:
    
    def __init__(self,filepath:str):
        # Load and parse the XML file
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Regular expressions for matching patterns
        mmc_pattern = re.compile(r'MMC_DVF_PF1\(([^)]*)\)')
        owf_pattern = re.compile(r'power_source\(([^)]*)\)')
        
        # Dictionaries to store mappings
        mmc_mappings = {}
        owf_mappings = {}
        
        # Iterate over all 'Analog' elements
        for analog in root.findall(".//Analog"):
            name = analog.get('name')
            index = analog.get('index')
        
            # Check for MMC_DVF_PF1 pattern and update mappings
            mmc_match = mmc_pattern.search(name)
            if mmc_match:
                mmc_id = mmc_match.group(1)  # Capture the MMC ID
        
                # Initialize dictionary for this MMC if not already present
                if mmc_id not in mmc_mappings:
                    mmc_mappings[mmc_id] = {'Vdq_meas': [], 'Idq_meas': []}
        
                # Add indices for Vdq_meas and Idq_meas
                if 'Vdq_meas' in name:
                    mmc_mappings[mmc_id]['Vdq_meas'].append(index)
                elif 'Idq_meas' in name:
                    mmc_mappings[mmc_id]['Idq_meas'].append(index)
        
            # Check for power_source pattern and update mappings
            owf_match = owf_pattern.search(name)
            if owf_match:
                owf_id = owf_match.group(1)  # Capture the OWF ID
        
                # Initialize dictionary for this OWF if not already present
                if owf_id not in owf_mappings:
                    owf_mappings[owf_id] = {'Vrms': [], 'Irms': []}
        
                # Add indices for Vrms and Irms
                if 'Vrms' in name:
                    owf_mappings[owf_id]['Vrms'].append(index)
                elif 'Irms' in name:
                    owf_mappings[owf_id]['Irms'].append(index)
        
        # Print the MMC and OWF mappings
        print("MMC Mappings:")
        for mmc_id, mappings in mmc_mappings.items():
            print(f"  MMC {mmc_id}: Vdq_meas Indices {mappings['Vdq_meas']}, Idq_meas Indices {mappings['Idq_meas']}")
        
        print("\nOWF Mappings:")
        for owf_id, mappings in owf_mappings.items():
            print(f"  OWF {owf_id}: Vrms Indices {mappings['Vrms']}, Irms Indices {mappings['Irms']}")

        return

PSCAD_loader(r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results\virtual_impedance_noatan_infpi.infx')


#%%
class StateSpaceSystem:
    
    def __init__(self,filepath,key,ss,ref=True,vals:dict={},verbose=False):
        if ref and key == 'HSC':
            self.ssm = ssm = SSM(filepath,f'{key}1_var',f'{key}1_SS{ss}')
            self.L = df = pd.read_excel(file_path, sheet_name=f'{key}1_L')
        else:
            self.ssm = ssm = SSM(filepath,f'{key}i_var',f'{key}i_SS{ss}')
            self.L = df = pd.read_excel(file_path, sheet_name=f'{key}i_L')

        self.u = u = df[(df.group == 'comp') & (df.vector == 'input') & (df.comp == ss)].sort_values(['comp','idx'], inplace=False)
        self.y = y = df[(df.group == 'comp') & (df.vector == 'output') & (df.comp == ss)].sort_values(['comp','idx'], inplace=False)
        self.x = x = df[(df.group == 'comp') & (df.vector == 'state') & (df.comp == ss)].sort_values(['comp','idx'], inplace=False)
        self.a = a = df[(df.group == 'sys') & (df.vector == 'input') & (df.comp == ss)].sort_values(['comp','idx'], inplace=False)
        self.b = b = df[(df.group == 'sys') & (df.vector == 'output') & (df.comp == ss)].sort_values(['comp','idx'], inplace=False)
        
        # 

        # Change values based on input
        for k, v in vals.items():
            ssm.store.data[k] = v
        
        # Evaluate the numerical state space model
        ssm.eval_num_ss()

        # Identifying A, B, C, D matrices based on column and row names
        
        self.idx_row = idx_row = len([c for c in ssm.ss_eval.index if c[0].lower() == 'd' or c[0].lower() == 'x'])
        self.idx_col = idx_col = len([c for c in ssm.ss_eval.columns if c[0].lower() == 'x'])
        
        
        self.A = ssm.ss_num[:idx_row,:idx_col]
        self.B = ssm.ss_num[:idx_row,idx_col:]
        self.C = ssm.ss_num[idx_row:,:idx_col]
        self.D = ssm.ss_num[idx_row:,idx_col:]

        if verbose:
            print(f'A{ss}',self.A.shape,idx_row,idx_col,'\n',self.A)
            print(f'B{ss}',self.B.shape,idx_row,idx_col,'\n',self.B)
            print(f'C{ss}',self.C.shape,idx_row,idx_col,'\n',self.C)
            print(f'D{ss}',self.D.shape,idx_row,idx_col,'\n',self.D)
        
        return



#%%


class ComponentConnectionMethod:
    def __init__(self,subsystems,system_input=None, system_output=None):
        self.subsystems = subsystems
        I = lambda x: np.diag(np.ones(x))
        self.L = L_map = subsystems[0].L
        self.a = subsystems[0].a
        self.b = subsystems[0].b
        
        if system_input is not None and system_output is not None:
            print(L_map)
            # L_map = L_map[(L_map['name'] == system_input & L_map['group']=='sys' & L_map['vector']=='input') \
            L_map = L_map[((L_map['name'] == system_input) & (L_map['group']=='sys') & (L_map['vector']=='input')) \
                          |((L_map['name'] == system_output) & (L_map['group']=='sys') & (L_map['vector']=='output')) \
                          |(L_map['group']=='comp')]
            print(L_map)
                
        # Prepare dictionary for subsystem matrices
        for M in ['A','B','C','D']:
            setattr(self,M,{})
        
        # Get state space models of subsystems
        for i, ss in enumerate(subsystems):     
            self.A[i+1] = ss.A
            self.B[i+1] = ss.B
            self.C[i+1] = ss.C
            self.D[i+1] = ss.D

        self.u = u = pd.concat([ss.u for ss in subsystems], axis=0)
        self.x = x = pd.concat([ss.x for ss in subsystems], axis=0)
        self.y = y = pd.concat([ss.y for ss in subsystems], axis=0)

               
        # Create block diagonal matrix
        for M in ['A','B','C','D']:
            matrices = [matrix for key, matrix in getattr(self,M).items()]
            # print(M,'\n',getattr(self,M))
            getattr(self,M)[0] = block_diag(*matrices)
        
        # Get L-map
        self.L = L = self.get_interconnection_matrices(L_map)
        for k,v in L.items():
            setattr(self,f'L{k}',v)
        
        # Create system matrix
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        
        nx = len(A[0])
        ny = len(D[0] @ L[1])
        # print(A[0].shape,B[0].shape,D[0].shape,L[1].shape,C[0].shape,(D[0] @ L[1]).shape)
        A =  A[0] + B[0] @ L[1] @ inv(I(ny)-D[0] @ L[1]) @ C[0]
        B =  B[0] @ L[1] @ inv(I(ny)-D[0] @ L[1]) @ D[0] @ L[2]+B[0] @ L[2]
        C =  L[3] @ inv(I(ny)-D[0] @ L[1]) @ C[0]
        D =  L[3] @ inv(I(ny)-D[0] @ L[1]) @ D[0] @ L[2]+L[4]
        
        self.sys = {'A':A,
                    'B':B,
                    'C':C,
                    'D':D}

        # Shift variable names cmp => "_M" and sys => "M"
        for M in ['A','B','C','D']:
            setattr(self,f'_{M}',getattr(self, M))
            setattr(self,M,eval(f'{M}'))
    
        # Get eigenvalues
        self.lamb, R = eig(self.A)
        

        # Calculate participation factors            
        L = inv(R)
        self.P = P = R @ L.T
            
        return

    def plot_eigenvalues(self):
        
        return

    def get_interconnection_matrices(self,L_map):
        
        # Read the specific sheet 'L_map'
        df = L_map

        # df now contains the data from the 'L_map' sheet
        u_c = df[(df.group == 'comp') & (df.vector == 'input')].sort_values(['comp','idx'], inplace=False)
        y_c = df[(df.group == 'comp') & (df.vector == 'output')].sort_values(['comp','idx'], inplace=False)
        u_s = df[(df.group == 'sys') & (df.vector == 'input')].sort_values(['comp','idx'], inplace=False)
        y_s = df[(df.group == 'sys') & (df.vector == 'output')].sort_values(['comp','idx'], inplace=False)

        # Create a directed graph
        G = nx.DiGraph()

        # Add any additional vertices if needed
        G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}_{n['name']}" for i, n in u_c.iterrows()])
        G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}_{n['name']}" for i, n in y_c.iterrows()])
        G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}_{n['name']}" for i, n in u_s.iterrows()])
        G.add_nodes_from([ f"{n['group'][0]}{n['vector'][0]}_{n['name']}" for i, n in y_s.iterrows()])

        # print(u_c,y_c,u_s,y_s)
        L = {}
        edges = []
        edge_clrs = []
        # Set the seed for reproducibility
        for i, pair in enumerate([(y_c,u_c),(u_s,u_c),(y_c,y_s),(u_s,y_s)]):
            from_,to_ = pair   
            from_ = from_.reset_index()
            to_ = to_.reset_index()
            L[int(f'{i+1}')] = np.zeros((len(to_),len(from_)))
            # print(L[int(f'{i+1}')].shape)
            for j, node_j in to_.iterrows():
                # print(j,node_j['name'])
                for k, node_k in from_.iterrows():
                    # print(k,node_k['name'])
                    # print(j,k)
                    if node_j['name'] == node_k['name']:
                        L[int(f'{i+1}')][j,k] = 1

        return L
    

    def dynamic_simulation(self,x0,t0,t1,dt = 0.001):
        t = np.arange(t0,t1,dt)

        dx = np.zeros((len(self.lamb),len(t)), dtype=np.complex)
        
        for k in range(0,len(t)):
            dx[:,k] = self.Phi.dot(np.exp(self.lamb*t[k])*(self.Psi).dot(x0))

        data = {f'${self.ltx_names[i,0][0]}$': list(dx[i,:]) for i in range(len(self.lamb))}
        df = pd.DataFrame(data,index=list(t))        

        return df


    def plot_time_response(self, fig, ax, x0, t0=0, t1=5, xlabel=False):        

        fig, ax = plt.subplots(1,1)

   
        ax.plot(self.t_plot_tr, self.dx_plot_tr[0], label=self.filename)
        
        ax.grid(linestyle='-.', linewidth=.25)
        ax.grid(linestyle=':', which='minor', linewidth=.25, alpha=.5)
        if xlabel:
            ax.set_xlabel('time [s]')
        ax.set_ylabel(self.filename)

        fig.tight_layout()        

        plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\img\\{self.filename}_series.pdf')
        
        return fig, ax


    
    def solve(self):
        
        return

    
    def participation_factor(self,vmax=1.):
        lamb, R = eig(self.A)
        L = inv(R)
        P = self.P
                
        if P.shape[0] > 24:
            fig,ax = plt.subplots(1,1,figsize=(9,9),dpi=200)
        elif P.shape[0] == 24:
            fig,ax = plt.subplots(1,1,figsize=(8,8),dpi=200)
        else:
            fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=200)
        im = ax.imshow(np.where((abs(P)>0.02) & (abs(P)<=vmax),abs(P),np.nan),vmin=0, vmax=vmax)
        ax.imshow(np.where(abs(P)>vmax,1,np.nan),vmin=0, vmax=vmax,cmap='Reds',alpha=0.25)
        ax.set_xticks([i for i in range(len(self.x))])
        ax.set_yticks([i for i in range(len(self.x))])
        ax.set_xticklabels(['$\\lambda_{' + str(i) +'}$' for i in range(1,len(self.x)+1)])
        ax.set_yticklabels(['$'+str(x)+'$' for x in self.x['latex_name']])
        # fig.colorbar(im, ax=ax, location='right', anchor=(0.2, 0.2))

        # c = plt.colorbar(im, cax = fig.add_axes([0.78, 0.5, 0.03, 0.38]))

        from mpl_toolkits.axes_grid1 import make_axes_locatable
    
        divider = make_axes_locatable(ax)
    
        ax_cb = divider.append_axes("right", size="5%", pad=0.1)
        fig = ax.get_figure()
        fig.add_axes(ax_cb)
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm), cax=ax_cb,extend = 'max')

        ax_cb.yaxis.tick_right()
        # ax_cb.yaxis.set_tick_params(labelright=False)

        # Minor ticks
        ax.set_xticks(np.arange(-.5, P.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-.5, P.shape[0], 1), minor=True)

        if P.shape[0] > 24:
            ax.set_xticklabels([("$\\lambda_{"+str(i+1)+"}$","")[i%2 == 1] for i in range(len(P))])
        
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)

        fig.tight_layout()

        # plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\img\\{self.filename}_P.pdf')

        plt.show()
        plt.close()

        return

def plot_eigs(A,leg_loc=None,mode='complex',xlim=None,ylim=None):
    fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=200)
    marks = ['o','*','+','s','d']
    lamb = np.linalg.eigvals(A)
    for i, l in enumerate(lamb):
        if np.real(l) > 0:
            print(i,l)
            ax.scatter([np.real(l)],[np.imag(l)],marker='3',color='red',s=200,alpha=1,zorder=4)

        if mode=='complex':
            if np.imag(l) > 0:
                ax.scatter([np.real(l),np.real(l)],[np.imag(l),-np.imag(l)],label='$\\lambda_{'+str(i+1)+'}$',marker=marks[i//12])
        else:
            if np.imag(l) > 0:
                ax.scatter([np.real(l),np.real(l)],[np.imag(l),-np.imag(l)],label='$\\lambda_{'+str(i+1)+'}$',marker=marks[i//12])
            elif np.imag(l) < 0:
                continue
            else:
                ax.scatter([np.real(l)],[np.imag(l)],label='$\\lambda_{'+str(i+1)+'}$',marker=marks[i//12])
            
    ax.axhline(0,color='k',lw=0.75)
    ax.axvline(0,color='k',lw=0.75)
    # ax.set_title(self.filename)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if leg_loc is not None:
        ax.legend(loc=leg_loc,ncol=(1,3)[len(lamb) > 24])
    else:            
        ax.legend(ncol=(1,3)[len(lamb) > 24])

    ax.grid()

    # plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\img\\{self.filename}_eig.pdf')
    plt.show()
    plt.close()

    return

#
id_ = 0.5
HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.0,'V_oq':0, 'I_od':id_, 'I_oq':0 }
    
# ss = StateSpaceSystem(file_path,'HSC',1,vals=HSC1_init)

path = r'C:\Users\bvilm\PycharmProjects\StateSpaceModelling\statespacemodels'
file_path = f'{path}\\ssm_ccm.xlsx'

HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
HSCi = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=False) for i in range(1,5)]

ccm = ComponentConnectionMethod(HSC1)

ccm.participation_factor()
# plot_eigs(ccm.A,mode='real')

HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.02,'V_oq':0, 'I_od':id_, 'I_oq':0 }
HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]

#%%
lambs_L_tfr = {}
XR = 7.5
for i, SCR in enumerate([0.25,0.5,0.75] + list(np.arange(1,100+0.0001,2))):
    print(SCR)
    HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.02,'V_oq':0, 'I_od':id_, 'I_oq':0,'L_tfr':np.sin(np.arctan(XR))/SCR}
    HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
    ccm = ComponentConnectionMethod(HSC1)
    lambs_L_tfr[SCR] = np.linalg.eigvals(ccm.A)
    
    plot_eigs(ccm.A,mode='real')

#%%
lambs_SCR_vi = {}
XR = 7.5
for i, SCR in enumerate([1e-6,0.25,0.5,0.75] + list(np.arange(1,10+0.0001,1))):
    print(SCR)
    HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.02,'V_oq':0, 'I_od':id_, 'I_oq':0,'L_v':np.sin(np.arctan(XR))/SCR,'R_v':np.cos(np.arctan(XR))/SCR}
    HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
    ccm = ComponentConnectionMethod(HSC1)
    lambs_L_tfr[SCR] = np.linalg.eigvals(ccm.A)
    
    # plot_eigs(ccm.A,mode='real')

#%%
lambs_SCR_vi = {}
XR = 7.5
# for i, SCR in enumerate([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1] + list(np.arange(1,10+0.0001,1))):
for i, SCR in enumerate(np.linspace(1e-5,5,50)):
    # SCR = np.log(SCR)
    # print(SCR)
    HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.02,'V_oq':0, 'I_od':id_, 'I_oq':0,'L_v':np.sin(np.arctan(XR))/SCR,'R_v':np.cos(np.arctan(XR))/SCR}
    HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
    ccm = ComponentConnectionMethod(HSC1)
    lambs_SCR_vi[SCR] = np.linalg.eigvals(ccm.A)
    
    # plot_eigs(ccm.A,mode='real')


#%%
lambs_Lv = {}
for i, L_v in enumerate(np.arange(0,0.6+.0001,0.02)):
    # L_v /= 100
    print(L_v)
    HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.00,'V_oq':0, 'I_od':id_, 'I_oq':0,'L_v':L_v,'L_tfr':0}
    HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
    ccm = ComponentConnectionMethod(HSC1)
    lambs_Lv[L_v] = np.linalg.eigvals(ccm.A)
    plot_eigs(ccm.A,mode='real')

#%%
lambs_Rv = {}
for i, R_v in enumerate(np.arange(0,0.6+.0001,0.02)):
    print(R_v)
    HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.0,'V_oq':0, 'I_od':id_, 'I_oq':0,'R_v':R_v,'L_v':0.0,'L_tfr':0}
    HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
    ccm = ComponentConnectionMethod(HSC1)
    lambs_Rv[R_v] = np.linalg.eigvals(ccm.A)
    plot_eigs(ccm.A,mode='real')

#%%
lambs_Zv = {}
for i, L_v in enumerate(np.arange(0,0.05+.0001,0.01)):
    for j, R_v in enumerate(np.arange(0,0.5+.0001,0.01)):
        print(L_v,R_v)
        HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.02,'V_oq':0, 'I_od':id_, 'I_oq':0,'L_v':0,'R_v':R_v,'L_tfr':0}
        HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
        ccm = ComponentConnectionMethod(HSC1)
        lambs_Zv[(R_v,L_v)] = np.linalg.eigvals(ccm.A)
        
        # plot_eigs(ccm.A,mode='real')

#%%
fig, ax = plt.subplots(1,1,dpi=150)
for i, (k,v) in enumerate(lambs_Zv.items()):
    v -= np.finfo(np.complex128).eps 

    ax.scatter([k[0]],[k[1]],color=('green','red')[(v> 0).any()])

ax.set_aspect('equal')

#%%
# Prepare figure and colormap
fig, ax = plt.subplots(1,1,dpi=150)
cmap = matplotlib.colormaps['winter']
# Normalize the parameter values for color mapping
param_values = np.array(list(lambs_L_tfr.keys()))
norm = plt.Normalize(vmin=param_values.min(), vmax=param_values.max())

ax.axvline(0,color='k',lw=0.75)
# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)

for i, (k,v) in enumerate(lambs_L_tfr.items()):
    real_parts = [eig.real for eig in v]
    imag_parts = [eig.imag for eig in v]
    color = cmap(norm(k))
    ax.scatter(real_parts, imag_parts, color=color,marker='o',zorder=3)
    if (np.array(real_parts) > 0).any():
        cbar.ax.hlines(k,0,1,color='red',lw=0.75)

cbar.set_label('Parameter Value')
# ax.set(xlim=(-0.0001,0.00001),ylim=(-10,10))

ax.grid(ls=':')

# Label axes
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
# ax.set_title('Eigenvalues in the Complex Plane')

plt.show()

   
#%% ========= EIGENVALUE-BASED ANALYSIS ========= 
# Prepare figure and colormap
fig, ax = plt.subplots(1,1,dpi=150)
ax1, ax = insert_axis(ax, [-100,200,-1.1e4,1.1e4], [.35,0.6,0.4,0.3],dict(color='red', ls=':',zorder=5))
# ax2, ax = insert_axis(ax, [-100,200,-1.1e4,1.1e4], [.35,0.6,0.4,0.3],dict(color='red', ls=':',zorder=5))

cmap = matplotlib.colormaps['winter']
# Normalize the parameter values for color mapping
param_values = np.array(list(lambs_Lv.keys()))
norm = plt.Normalize(vmin=param_values.min(), vmax=param_values.max())

ax.axvline(0,color='k',lw=0.75)
ax1.axvline(0,color='k',lw=0.75)
# ax2.axvline(0,color='k',lw=0.75)

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)

for i, (k,v) in enumerate(lambs_Lv.items()):
    real_parts = [eig.real for eig in v]
    imag_parts = [eig.imag for eig in v]
    color = cmap(norm(k))
    ax.scatter(real_parts, imag_parts, color=color,marker='o',zorder=3)
    ax1.scatter(real_parts, imag_parts, color=color,marker='o',zorder=3)
    # ax2.scatter(real_parts, imag_parts, color=color,marker='o',zorder=3)
    if (np.array(real_parts) > 0).any():
        cbar.ax.hlines(k,0,1,color='red',lw=0.75)
    else:
        cbar.ax.hlines(k,0,1,color='C2',lw=0.75)

cbar.set_label('$L_v$ [pu]')
# ax.set(xlim=(-0.01,0.002),ylim=(-0.2,.2))

ax.grid(ls=':')

# Label axes
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
# ax.set_title('Eigenvalues in the Complex Plane')

plt.show()   
#%%
# Prepare figure and colormap
fig, ax = plt.subplots(1,1,dpi=150)
ax1, ax = insert_axis(ax, [-1750,200,-.05e-5,.05e-5], [.35,0.6,0.4,0.3],dict(color='red', ls=':',zorder=5))
ax2, ax = insert_axis(ax, [-4.5e5,-4.46e5,-.05e-5,.05e-5], [.35,0.1,0.4,0.3],dict(color='red', ls=':',zorder=5))
cmap = matplotlib.colormaps['winter']
# Normalize the parameter values for color mapping
param_values = np.array(list(lambs_Lv.keys()))
norm = plt.Normalize(vmin=param_values.min(), vmax=param_values.max())

ax.axvline(0,color='k',lw=0.75)
ax1.axvline(0,color='k',lw=0.75)

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)

for i, (k,v) in enumerate(lambs_Rv.items()):
    real_parts = [eig.real for eig in v]
    imag_parts = [eig.imag for eig in v]
    color = cmap(norm(k))
    ax.scatter(real_parts, imag_parts, color=color,marker='o',zorder=3)
    ax1.scatter(real_parts, imag_parts, color=color,marker='o',zorder=3)
    ax2.scatter(real_parts, imag_parts, color=color,marker='o',zorder=3)
    if (np.array(real_parts) > 0).any():
        cbar.ax.hlines(k,0,1,color='red',lw=0.75)

cbar.set_label('$R_v$ [pu]')
# ax.set(xlim=(-0.01,0.002),ylim=(-0.2,.2))

ax.grid(ls=':')

# Label axes
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
# ax.set_title('Eigenvalues in the Complex Plane')

plt.show()   

#%%


HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.00,'V_oq':0, 'I_od':id_, 'I_oq':0,'L_v':L_v,'L_tfr':np.sin(np.arctan(XR))/8}
HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
tf_dq = []
for v in ['v_od','v_oq']:
    for i in ['i_ld','i_lq']:
        ccm = ComponentConnectionMethod(HSC1,system_input=v,system_output=i)
        tf_dq.append({'ccm':ComponentConnectionMethod(HSC1,system_input=v,system_output=i),
                                    'label':'_',
                                    'axis':f'{v[-1]}{i[-1]}'})

#%%

# HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.00,'V_oq':0, 'I_od':id_, 'I_oq':0,'L_v':L_v,'L_tfr':np.sin(np.arctan(XR))/8}
# HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
tf_dq_Lv = []
for i, L_v in enumerate(np.arange(0,0.2+.0001,0.05)):
    L_v /= 90
    HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.00,'V_oq':0, 'I_od':id_, 'I_oq':0,'R_v':0,'L_v':L_v,'L_tfr':0}
    HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
    for v in ['v_od','v_oq']:
        for i in ['i_ld','i_lq']:
            # ccm = ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i])
            tf_dq_Lv.append({'ccm':ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i]),
                                        'label':'$L_v='+f'{round(L_v,4)}$',
                                        'axis':f'{v[-1]}{i[-1]}'})

#%%
# HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.00,'V_oq':0, 'I_od':id_, 'I_oq':0,'L_v':L_v,'L_tfr':np.sin(np.arctan(XR))/8}
# HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
tf_dq_Rv = []
for i, R_v in enumerate(np.arange(0,0.2+.0001,0.05)):
    R_v /= 90
    HSC1_init = {'I_ld':id_,'I_lq':0, 'V_od':1.00,'V_oq':0, 'I_od':id_, 'I_oq':0,'R_v':R_v,'L_v':0,'L_tfr':0}
    HSC1 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC1_init,ref=True) for i in range(1,5)]
    for v in ['v_od','v_oq']:
        for i in ['i_ld','i_lq']:
            ccm = ComponentConnectionMethod(HSC1,system_input=[v],system_output=[i])
            tf_dq_Rv.append({'ccm':ccm,
                                        'label':'$R_v='+f'{round(R_v,4)}$',
                                        'axis':f'{v[-1]}{i[-1]}'})

    
#%%
def plot_Zdq(tf_dq,ax=None,save=False):
    if ax is None:
        # Plot Bode plot for each input-output pair
        fig, ax = plt.subplots(4, 2, figsize=(10, 8),sharex=True)

    for ccm_dict in tf_dq:        
        v = ccm_dict
        key = ccm_dict['axis']
    
        if key == 'dd':
            ax0,ax1 = ax[0,0],ax[1,0]
        elif key == 'dq':
            ax0,ax1 = ax[0,1],ax[1,1]
        elif key == 'qd':
            ax0,ax1 = ax[2,0],ax[3,0]
        elif key == 'qq':
            ax0,ax1 = ax[2,1],ax[3,1]

        print(v)

        # Define the system matrices A, B, and C
        A = (v['ccm']).A  # Replace with your A matrix
        B = (v['ccm']).B  # Replace with your B matrix
        C = (v['ccm']).C  # Replace with your C matrix
        D = (v['ccm']).D  # Replace with your C matrix
        
        # Define the frequency range for which you want to plot the Bode plot
        # omega = np.logspace(-3, 4, 1000)  # Replace with your desired frequency range
        omega = np.logspace(-2, 5, 1000)  # Replace with your desired frequency range
        
        # Preallocate magnitude and phase arrays
        mag = np.zeros((len(omega), C.shape[0], B.shape[1]))
        phase = np.zeros_like(mag)
        
        # Compute the transfer function and Bode plot for each frequency
        for k, w in enumerate(omega):
            s = 1j * w
            # Calculate the transfer function matrix
            T = C @ inv(s * np.eye(A.shape[0]) - A) @ B
            # Store magnitude and phase for each input-output pair
            mag[k, :, :] = 20 * np.log10(abs(T))
            phase[k, :, :] = np.angle(T, deg=True)
                
        # Adjust these loops and indexing if your system has more inputs/outputs
        for i in range(C.shape[0]):
            for j in range(B.shape[1]):
                ax0.semilogx(omega/(2*np.pi), mag[:, i, j], label=v['label'])
                ax1.semilogx(omega/(2*np.pi), phase[:, i, j], label=v['label'])
        
        ax0.set_title('$Y_\\mathit{HSC}^\\mathit{' + str(key) + '}=i_o^'+key[1]+'/v_o^'+key[0]+'$')
        ax0.axvline(50,lw=0.75,color='k')
        ax1.axvline(50,lw=0.75,color='k')
        ax1.grid(ls=':')
        ax0.grid(ls=':')
        ax1.grid(ls=':')
        ax1.set(ylim=(-180,180))

        if key[1] == 'd':
            ax1.set_ylabel('Phase (degrees)')
            ax0.set_ylabel('Magnitude (dB)')
        if key[0] == 'q':
            ax1.set_xlabel('Frequency (Hz)')

    ax1.legend(loc='lower right')

    ax0.set(xlim=((omega/(2*np.pi)).min(),(omega/(2*np.pi)).max()))        
    
    fig.tight_layout()
    fig.align_ylabels()
    if isinstance(save,str):
        plt.savefig(save)        
    else:
        plt.show()        
   
    return

path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'

# plot_Zdq(tf_dq_Rv,save=f'{path}\\impedance_plot_hsc_Rv.pdf')
# plot_Zdq(tf_dq_Lv,save=f'{path}\\impedance_plot_hsc_Lv.pdf')
plot_Zdq(tf_dq_Rv)
plot_Zdq(tf_dq_Lv)

#%%
def evalp(P,w):
    p = np.array(P).ravel()
    py = np.array([(p[i]*w*1j)**i for i in range(len(p)-1,0,-1)])
    return  py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, freqresp
idx = 2

tf = control.ss2tf(tf_dq_Lv[idx]['ccm'].A,tf_dq_Lv[idx]['ccm'].B,tf_dq_Lv[idx]['ccm'].C,tf_dq_Lv[idx]['ccm'].D)

# Transfer function coefficients
num = np.array(tf.num).ravel()
den = np.array(tf.den).ravel()

# Create transfer function
sys = TransferFunction(num, den)

# Preallocate magnitude and phase arrays
omega = np.logspace(-4, 4, 1000)  # Frequency range from 0.0001 to 10,000 rad/s

# Compute the transfer function and Bode plot for each frequency
for k, w in enumerate(omega):
    s = 1j * w
    # Convert frequency to complex s-domain
    omega = 2 * np.pi * frequency
    s = 1j * omega
    
    # Evaluate the polynomial at s
    numerator_value = np.polyval(numerator_coeffs, s)

    # Calculate the transfer function matrix
    T = C @ inv(s * np.eye(A.shape[0]) - A) @ B
    # Store magnitude and phase for each input-output pair
    mag[k, :, :] = 20 * np.log10(abs(T))
    phase[k, :, :] = np.angle(T, deg=True)


# Frequency range

# Frequency response
w, h = freqresp(sys, w=w)

# Nyquist plot
# plt.figure()
fig,ax = plt.subplots(1,1,dpi=150)
ax.plot(h.real, h.imag, lw=2)
ax.plot(h.real, -h.imag, lw=2, linestyle='--')  # Mirror image for negative frequencies
ax.set(xlabel='Real Part',ylabel='Imaginary Part')
ax.grid(ls=':')
plt.show()

