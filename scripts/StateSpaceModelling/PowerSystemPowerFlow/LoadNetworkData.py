# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:10:34 2022

@author: bvilm
"""
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
# pd.set_option('max_columns', None)




class LoadNetworkData:
    def __init__(self,f = 50,V0=1):
        # System settings
        self.f = f
        self.V_init = V0
        self.omega = f*2*np.pi        
        self.G = nx.Graph()
        return

    def calc_RX(self, scr, xr, Vbase=1, Sbase=1, Fbase=50):
        Z = Vbase ** 2 / (Sbase * scr)
        R = Z * np.cos(np.arctan(xr))
        X = Z * np.sin(np.arctan(xr))
        return R, X

    # Method for adding a line to the system
    def add_line(self,bus1,bus2,scr=None,xr=None,r=None,x=None,g=0,b=0,name=None,Vbase:float=1,Sbase:float=1,Fbase:float=50):
        assert (scr is not None and xr is not None) or (r is not None and x is not None)

        if scr is not None and xr is not None:
            r, x = self.calc_RX(scr,xr,Vbase=Vbase,Sbase=Sbase)
            # print(r,x)
        else:
            Zbase = Vbase ** 2 / Sbase
            r = r/Zbase
            x = x/Zbase

        z = complex(r,abs(x))
        # print(z)
        yc = complex(g,abs(b))

        self.G.add_edge(bus1,bus2,z=z,y=1/z,yc=yc,r=r,x=x,g=g,b=b,t='line',name=f'{name}' + ("\n","")[name is None] +f'{(round(z.real,3),round(z.imag,3))} pu')

        attrs = {bus1: {"v": None, "theta": None},bus2: {"v": None, "theta": None}}
        nx.set_node_attributes(self.G, attrs)        

        return
    
    # Method for adding a transformer to the system
    def add_transformer(self,bus1,bus2,r,x,n=1,alpha=0,name=None):
        
        # Impedance
        z = complex(r,abs(x))
        y = 1/z

        # Phase shifting
        shift = n*(np.cos(alpha*np.pi/180)+np.sin(alpha*np.pi/180)*1j)
        a, b = np.real(shift), np.imag(shift)
                
        
        self.shift = shift
        # Phase shifting equation
        Y = np.array([[y/(a**2+b**2), -y/(a-1j*b)],
                      [-y/(a+1j*b), y]])

        name=f'{name}' + ("\n","")[name is None] +f'{z}'

        Z = Y**(-1)
        
        # Add transformer branch
        self.G.add_edge(bus1,bus2,y=y,Y=Y,y12=Y[0,1],y21=Y[1,0],y11=Y[0,0],y22=Y[1,1],bus1=bus1,bus2=bus2,t='transformer',name=name,alpha=alpha)

        # update eventual nodes
        attrs = {bus1: {"v": None, "theta": None},bus2: {"v": None, "theta": None}}
        nx.set_node_attributes(self.G, attrs)        
        return

    def add_static_load(self,bus,P=0,Q=0,V=1,Vbase = 1, Sbase = 1):
        # SEE KUNDUR p. 795 (Section 12.7: Small-signal stability of multimachine systems)
        p = P/Sbase
        q = Q/Sbase
        v = V/Vbase

        g = p/(v**2)
        b = q/(v**2)
        y = g + 1j*b

        if self.G.has_node(bus):
            # Update existing node attributes
            self.G.nodes[bus]['g_statload'] = self.G.nodes[bus].get('g_statload', 0) + g
            self.G.nodes[bus]['b_statload'] = self.G.nodes[bus].get('b_statload', 0) + b
            self.G.nodes[bus]['y_statload'] = self.G.nodes[bus].get('y_statload', 0) + y
        else:
            # Add new node with attributes
            self.G.add_node(bus, g_statload=g, b_statload=b, y_statload=y,p_statload=p,q_statload=q)
            self.G.add_edge(bus1,bus2,z=z,y=1/z,yc=yc,r=r,x=x,g=g,b=b,t='line',name=f'{name}' + ("\n","")[name is None] +f'{z} pu')

        return

    def add_nonlinear_load(self,bus,P=0,Q=0,V=1,Vbase = 1, Sbase = 1, m = 0, n = 0):
        # SEE KUNDUR p. 796-8 (Section 12.7.(b): Small-signal stability of multimachine systems)



        return


    def add_gen(self,bus,p,q,v=None,theta=None):
        attrs = {bus: {"pgen": p, "qgen": q,'v':v,'theta':theta}}
        nx.set_node_attributes(self.G, attrs)        
        return

    def add_load(self,bus,p,q):
        attrs = {bus: {"pload": p, "qload": q}}
        nx.set_node_attributes(self.G, attrs)        
        return
        
    def build(self):
        # Ybus, Sbus, V0, buscode, pq_index, pv_index, Y_from, Y_to, br_f, br_t, br_Y        
        N = len(self.G.nodes) # Number of nodes

        # CREATING INITIAL VOLTAGE GUESSES
        self.V0 = self.V_init*np.ones(N,dtype=complex)            

        # CONSTRUCTING ADMITTANCE BUS
        Y = np.zeros((N,N),dtype=complex)
        for i, n in enumerate(self.G.nodes):
            for e in self.G.edges(i+1):
                if self.G[e[0]][e[1]]['t'] == 'line':
                    Y[i,i] += self.G[e[0]][e[1]]['y'] + self.G[e[0]][e[1]]['yc']
            if 'y_statload' in self.G.nodes[n].keys():
                Y[i, i] += self.G.nodes[n]['y_statload']

        for e in self.G.edges:
            i, j = e[0] - 1, e[1] - 1
            if self.G[e[0]][e[1]]['t'] == 'line':
                Y[i,j] = -self.G[e[0]][e[1]]['y']
                Y[j,i] = -self.G[e[0]][e[1]]['y']            
            elif self.G[e[0]][e[1]]['t'] == 'transformer':
                Y[np.ix_(np.array(e)-1,np.array(e)-1)] +=  self.G[e[0]][e[1]]['Y']

        self.Ybus = Y
                
        # CONSTRUCTING SBUS
        buscode = np.zeros((N),dtype=int)     
        S = np.zeros((N),dtype=complex)     
        pq_index = []
        pv_index = []
        for i, n in enumerate(self.G.nodes):
            if 'pgen' in self.G.nodes[i+1]:
                S[i] += complex(self.G.nodes[i+1]['pgen'],self.G.nodes[i+1]['qgen'])
            if 'pload' in self.G.nodes[i+1]:
                S[i] -= complex(self.G.nodes[i+1]['pload'],self.G.nodes[i+1]['qload'])

            # defining bus codes
            if self.G.nodes[i+1]['v'] is not None and self.G.nodes[i+1]['theta'] is not None:
                buscode[i] = 3
            elif self.G.nodes[i+1]['v'] is not None:
                buscode[i] = 2                                    
            else:
                buscode[i] = 1                    

        self.Sbus = S
        self.buscode = buscode
        self.pq_index = np.where(buscode == 1)[0] # Find indices for all PQ-busses
        self.pv_index = np.where(buscode == 2)[0] # Find indices for all PV-busses
        self.ref = np.where(buscode == 3)[0] # Find index for ref bus
        
        # CREATING BRANCH MATRICES
        n_br, n_bus = len(self.G.edges), len(self.G.nodes)
        
        Y_from = np.zeros((n_br,n_bus),dtype=np.complex128)
        Y_to = np.zeros((n_br,n_bus),dtype=np.complex128)
        br_f = np.array(self.G.edges)[:,0] -1 # The from busses (python indices start at 0)
        br_t = np.array(self.G.edges)[:,1] -1 # The to busses

        for i, e in enumerate(list(self.G.edges)): # Fill in the matrices
            if self.G.edges[e]['t'] == 'line':
                Y_from[i,br_f[i]]   = self.G.edges[e]['y']
                Y_from[i,br_t[i]]   = -self.G.edges[e]['y']
                Y_to[i,br_f[i]]     = -self.G.edges[e]['y']
                Y_to[i,br_t[i]]     = self.G.edges[e]['y']

            elif self.G.edges[e]['t'] == 'transformer':
                Y_from[i,br_f[i]]   = (self.G.edges[e]['y11'])
                Y_from[i,br_t[i]]   = (self.G.edges[e]['y12'])
                Y_to[i,br_f[i]]     = (self.G.edges[e]['y21'])
                Y_to[i,br_t[i]]     = (self.G.edges[e]['y22'])

        self.Y_from = Y_from
        self.Y_to = Y_to
        self.br_f = br_f
        self.br_t = br_t

        return

    def Ybus_to_real(self,Y_cmpl):
        i, j = Y_cmpl.shape
        Y_real = np.zeros((i*2,j*2))
        Y_real[:i,:j] =  Y_cmpl.real
        Y_real[:i,j:] = -Y_cmpl.imag
        Y_real[i:,:j] =  Y_cmpl.imag
        Y_real[i:,j:] =  Y_cmpl.real

        return Y_real

    def draw(self,seed=20,title=None):
        np.random.seed(seed)
        pos = nx.spring_layout(self.G)
        np.random.seed(seed)
        nx.draw_networkx(self.G,with_labels=True, node_size = [500]*len(self.G.nodes))
        np.random.seed(seed)
        nx.draw_networkx_edge_labels(self.G,pos, edge_labels=nx.get_edge_attributes(self.G,'name'),font_color='red')
        if title is not None:
            plt.title(title)
        plt.show()
        plt.close()
        return
    



