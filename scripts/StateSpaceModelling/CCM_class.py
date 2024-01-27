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
from numpy import sin, cos

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

# PSCAD_loader(r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results\virtual_impedance_noatan_infpi.infx')


#%%
class StateSpaceSystem:
    def __init__(self,file_path,key,ss,ref=True,vals:dict={},verbose=False,comp_id=1):
        if ref and key == 'HSC':
            self.ssm = ssm = SSM(file_path,f'{key}1_var',f'{key}1_SS{ss}')
            self.L = df = pd.read_excel(file_path, sheet_name=f'{key}1_L')
        else:
            self.ssm = ssm = SSM(file_path,f'{key}i_var',f'{key}i_SS{ss}')
            self.L = df = pd.read_excel(file_path, sheet_name=f'{key}i_L')

        self.name = f'{key}{comp_id}'

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

        # Verbose print statement
        if verbose:
            print(f'A{ss}',self.A.shape,idx_row,idx_col,'\n',self.A)
            print(f'B{ss}',self.B.shape,idx_row,idx_col,'\n',self.B)
            print(f'C{ss}',self.C.shape,idx_row,idx_col,'\n',self.C)
            print(f'D{ss}',self.D.shape,idx_row,idx_col,'\n',self.D)
        
        return



#%%


class ComponentConnectionMethod:
    def __init__(self,subsystems,system_input:list=None, system_output:list=None):
        self.name = self.suffix = subsystems[0].name
        self.subsystems = subsystems
        I = lambda x: np.diag(np.ones(x))
        self.L = L_map = subsystems[0].L
        self.L_map = subsystems[0].L
        self.a = subsystems[0].a
        self.b = subsystems[0].b
        self.system_input = system_input
        self.system_output = system_output

        
        if system_input is not None and system_output is not None:
            # print(L_map)
            # L_map = L_map[(L_map['name'] == system_input & L_map['group']=='sys' & L_map['vector']=='input') \
            L_map = L_map[((L_map['name'].isin(system_input)) & (L_map['group']=='sys') & (L_map['vector']=='input')) \
                          |((L_map['name'].isin(system_output)) & (L_map['group']=='sys') & (L_map['vector']=='output')) \
                          |(L_map['group']=='comp')]
            # print(L_map)
                
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

        # Update the L maps.
        # self.u['name'] = self.u['name'].apply(lambda x: x + '_' + self.name)
        # self.x['name'] = self.x['name'].apply(lambda x: x + '_' + self.name)
        # self.y['name'] = self.y['name'].apply(lambda x: x + '_' + self.name)
        # self.u['latex_name'] = self.u['latex_name'].apply(lambda x: x + '_{' + self.name + '}')
        # self.x['latex_name'] = self.x['latex_name'].apply(lambda x: x + '_{' + self.name + '}')
        # self.y['latex_name'] = self.y['latex_name'].apply(lambda x: x + '_{' + self.name + '}')
        return

    def __repr__(self):
        # print('\nA:',pd.DataFrame(self.A,index=self.x.name,columns=self.x.name),
        #       '\nB:',pd.DataFrame(self.B,index=self.x.name,columns=self.u.name),
        #       '\nC:',pd.DataFrame(self.C,index=self.y.name,columns=self.x.name),
        #       '\nD:',pd.DataFrame(self.D,index=self.y.name,columns=self.u.name),
        #       sep='\n')
        # print('\nA:',pd.DataFrame(self.A),'\nB:',pd.DataFrame(self.B),'\nC:',pd.DataFrame(self.C),'\nD:',pd.DataFrame(self.D),sep='\n')
        return


    def show(self,save:bool=False):
        # ====================== SUBSYSTEMS ======================
        fig, ax = plt.subplots(1,1,dpi=150)
        M = np.vstack([np.hstack([self._A[0],self._B[0]]),
                      np.hstack([self._C[0],self._D[0]])])

        ax.imshow(np.where(M==0,np.nan,M))

        xlabels = ['$' + n + '$' for n in self.x.latex_name] + ['$' + n + '$' for n in self.u.latex_name]
        ylabels = ['$' + n + '$' for n in self.x.latex_name] + ['$' + n + '$' for n in self.y.latex_name]

        ax.set_xticks([i for i in range(len(xlabels))])
        ax.set_yticks([i for i in range(len(ylabels))])
        ax.set_xticklabels(xlabels,fontsize=6)
        ax.set_yticklabels(ylabels,fontsize=6)

        # Minor ticks
        ax.set_xticks(np.arange(-.5, len(xlabels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)

        ax.axvline(self.x.shape[0]-.5,color='k',lw=0.75)
        ax.axhline(self.x.shape[0]-.5,color='k',lw=0.75)

        text_alpha = 0.25
        ax.text(self.x.shape[0]/2-.5,self.x.shape[0]/2-.5, "A",ha='center',va='center', fontweight="bold",color='grey',fontsize=20,alpha=text_alpha,zorder=1)
        ax.text(self.x.shape[0]+self.u.shape[0]/2-.5,self.x.shape[0]/2-.5, "B",ha='center',va='center', fontweight="bold",color='grey',fontsize=20,alpha=text_alpha,zorder=1)
        ax.text(self.x.shape[0]/2-.5,self.x.shape[0]+self.y.shape[0]/2-.5, "C",ha='center',va='center', fontweight="bold",color='grey',fontsize=20,alpha=text_alpha,zorder=1)
        ax.text(self.x.shape[0]+self.u.shape[0]/2-.5,self.x.shape[0]+self.y.shape[0]/2-.5, "D",ha='center',va='center', fontweight="bold",color='grey',fontsize=20,alpha=text_alpha,zorder=1)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
        fig.tight_layout()

        if not save:
            plt.show()
        else:
            plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\06_state_space\img\ccm_comp.pdf',bbox_inches='tight')
        plt.close()

        # ====================== CCM SYSTEM ======================
        fig, ax = plt.subplots(1,1,dpi=150)
        M = np.vstack([np.hstack([self.A,self.B]),
                      np.hstack([self.C,self.D])])

        ax.imshow(np.where(M==0,np.nan,M))

        u = ['$' + n[1].latex_name + '$' for n in self.L_map.iterrows() if n[1].group =='sys' and n[1]['name'] in self.system_input]
        y = ['$' + n[1].latex_name + '$' for n in self.L_map.iterrows() if n[1].group =='sys' and n[1]['name'] in self.system_output]
        xlabels = ['$' + n + '$' for n in self.x.latex_name] + u
        ylabels = ['$' + n + '$' for n in self.x.latex_name] + y

        ax.set_xticks([i for i in range(len(xlabels))])
        ax.set_yticks([i for i in range(len(ylabels))])
        ax.set_xticklabels(xlabels,fontsize=8)
        ax.set_yticklabels(ylabels,fontsize=8)

        # Minor ticks
        ax.set_xticks(np.arange(-.5, len(xlabels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)

        ax.axvline(self.x.shape[0]-.5,color='k',lw=0.75)
        ax.axhline(self.x.shape[0]-.5,color='k',lw=0.75)

        text_alpha = 0.25
        ax.text(self.x.shape[0]/2-.5,self.x.shape[0]/2-.5, "A",ha='center',va='center', fontweight="bold",color='grey',fontsize=20,alpha=text_alpha,zorder=1)
        ax.text(self.x.shape[0]+len(u)/2-.5,self.x.shape[0]/2-.5, "B",ha='center',va='center', fontweight="bold",color='grey',fontsize=20,alpha=text_alpha,zorder=1)
        ax.text(self.x.shape[0]/2-.5,self.x.shape[0]+len(y)/2-.5, "C",ha='center',va='center', fontweight="bold",color='grey',fontsize=20,alpha=text_alpha,zorder=1)
        ax.text(self.x.shape[0]+len(u)/2-.5,self.x.shape[0]+len(y)/2-.5, "D",ha='center',va='center', fontweight="bold",color='grey',fontsize=20,alpha=text_alpha,zorder=1)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)

        fig.tight_layout()

        if not save:
            plt.show()
        else:
            plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\06_state_space\img\ccm_sys.pdf',bbox_inches='tight')
        plt.close()


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

        # if P.shape[0] > 24:
        #     ax.set_xticklabels([("$\\lambda_{"+str(i+1)+"}$","")[i%2 == 1] for i in range(len(P))])
        
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)

        fig.tight_layout()

        # plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\img\\{self.filename}_P.pdf')

        plt.show()
        plt.close()

        return




    
#%%

# path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\07_stability\img'
# plot_Zdq(tf_dq_Rv,save=f'{path}\\impedance_plot_hsc_Rv.pdf')
# plot_Zdq(tf_dq_Lv,save=f'{path}\\impedance_plot_hsc_Lv.pdf')

#%%
