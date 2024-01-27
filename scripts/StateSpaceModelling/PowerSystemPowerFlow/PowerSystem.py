# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:10:34 2022

@author: bvilm
"""
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import control as ctrl
# pd.set_option('max_columns', None)

class PS:
    def __init__(self,
                 f = 50,
                 Vbase=1,
                 Sbase=None,
                 Zbase=None,
                 ):

        # System settings
        self.f = f
        self.V_init = Vbase
        self.omega = f*2*np.pi        

        # Setting per unit system
        if Sbase is not None:
            self.per_unit = True
            self.Ibase = Sbase/(np.sqrt(3)*Vbase)
            self.Zbase = Vbase**2/Sbase
        elif Zbase is not None:
            self.per_unit = True
            self.Ibase = Vbase/(np.sqrt(3)*Zbase)
            self.Sbase = Vbase**2/Zbase
        else:
            self.per_unit = False
            print('Per-unit was not able to create, please provide voltage and impedance or power base.')

        self.G = nx.Graph()
        return
    
    # Method for adding a line to the system
    def add_line(self,bus1,bus2,r,x,name=None,c=0):
        z = complex(r,abs(x))

        self.G.add_edge(bus1,bus2,z=z,y=1/z,yc=1j*c*self.omega,t='line',name=f'{name}' + ("\n","")[name is None] +f'{z}')

        # attrs = {bus1: {"v": None, "theta": None},bus2: {"v": None, "theta": None}}
        # nx.set_node_attributes(self.G, attrs)

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
        # attrs = {bus1: ,bus2: {"v": None, "theta": None}}
        # nx.set_node_attributes(self.G, attrs)
        return

    
    def add_gen(self,bus,v):
        attrs = {bus: {"v": v}}
        nx.set_node_attributes(self.G, attrs)
        return

    def add_load(self,bus,name,z,mode='z'):
        print(self.G.nodes[bus])
        if f'z_{name}' in self.G.nodes[bus]:
            raise AttributeError(f'"z_{name}" already defined at bus: {bus}')
        attrs = {bus: {f"z_{name}": z}}
        nx.set_node_attributes(self.G, attrs)
        return
        
    def build(self,ss_mode='z'):
        # Ybus, Sbus, V0, buscode, pq_index, pv_index, Y_from, Y_to, br_f, br_t, br_Y        
        N = len(self.G.nodes) # Number of nodes

        # CREATING input matrix
        B = np.zeros(N,dtype=complex)

        # CONSTRUCTING ADMITTANCE BUS
        Y = np.zeros((N,N),dtype=complex)

        # Defining self admittance
        for i, n in enumerate(self.G.nodes):
            # adding line admittance
            for e in self.G.edges(i+1):
                if self.G[e[0]][e[1]]['t'] == 'line':
                    Y[i,i] += self.G[e[0]][e[1]]['y'] + self.G[e[0]][e[1]]['yc']

            # adding load admittance
            # print(i,n,self.G.nodes,self.G.nodes[n])
            zs = [k for k in self.G.nodes[n].keys() if 'z' == k.split('_')[0]]
            for z in zs:
                # print(n,k,self.G.nodes[n][z])
                Y[i, i] += 1/self.G.nodes[n][z]

        # Defining mutual admittance
        for e in self.G.edges:
            i, j = e[0] - 1, e[1] - 1
            if self.G[e[0]][e[1]]['t'] == 'line':
                Y[i,j] = -self.G[e[0]][e[1]]['y']
                Y[j,i] = -self.G[e[0]][e[1]]['y']            
            elif self.G[e[0]][e[1]]['t'] == 'transformer':
                Y[np.ix_(np.array(e)-1,np.array(e)-1)] +=  self.G[e[0]][e[1]]['Y']

        # Creating voltage dependent input matrix
        for i, n in enumerate(self.G.nodes):
            # adding line admittance
            if 'v' in self.G.nodes[n]:
                B[i] += Y[i,i]


            for e in self.G.edges(i+1):
                if self.G[e[0]][e[1]]['t'] == 'line':
                    Y[i,i] += self.G[e[0]][e[1]]['y'] + self.G[e[0]][e[1]]['yc']

            # adding load admittance
            # print(i,n,self.G.nodes,self.G.nodes[n])
            keys = [k for k in self.G.nodes[n].keys() if 'z' == k.split('_')[0]]
            for k in keys:
                # print(n,k,self.G.nodes[n][k])
                Y[i, i] += 1/self.G.nodes[n][k]

        # Create state space
        # ss_mode = 'z'
        # ss = ctrl.ss(Y,B,np.diag(np.ones(N)),0)
        self.Y = Y
        self.Z = Z = np.linalg.inv(Y)
        self.B = B

        print(Y)
        print(Z)
        print(B)

        return Z, Y, B

    def draw(self,seed=20):
        np.random.seed(seed)
        pos = nx.spring_layout(self.G)
        np.random.seed(seed)
        nx.draw(self.G,with_labels=True, node_size = [500]*len(self.G.nodes))
        np.random.seed(seed)
        nx.draw_networkx_edge_labels(self.G,pos, edge_labels=nx.get_edge_attributes(self.G,'name'),font_color='red')
        plt.close()
        return
    



