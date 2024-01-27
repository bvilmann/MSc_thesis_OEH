# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:50:37 2023

@author: bvilm
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        print(j,node_j['name'])
        for k, node_k in from_.iterrows():
            print(k,node_k['name'])
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
df = pd.read_excel(file_path, sheet_name='L_map')

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

print(u_c,y_c,u_s,y_s)
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
    print(L[int(f'{i+1}')].shape)
    for j, node_j in to_.iterrows():
        print(j,node_j['name'])
        for k, node_k in from_.iterrows():
            print(k,node_k['name'])
            # print(j,k)
            if node_j['name'] == node_k['name']:
                L[int(f'{i+1}')][j,k] = 1

for k,v in L.items():
    print(k,'\n',v)
    
    
#%%


path = r'C:\Users\bvilm\PycharmProjects\StateSpaceModelling\statespacemodels'
file_path = f'{path}\\ssm_ccm.xlsx'

# Read the Excel sheet
df = pd.read_excel(file_path, sheet_name='SS1', index_col=0,header=0)

# Identifying A, B, C, D matrices based on column and row names
state_vars = ['x1', 'x2', 'x3']  # Adjust as per your state variables
input_vars = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8']  # Adjust as per your input variables
output_vars = ['y1', 'y2', 'y3']  # Adjust as per your output variables

# Extract matrices
A_matrix = df.loc[state_vars, state_vars]
B_matrix = df.loc[state_vars, input_vars]
C_matrix = df.loc[output_vars, state_vars]
D_matrix = df.loc[output_vars, input_vars]

# Convert to pandas DataFrame (if needed)
A_df = pd.DataFrame(A_matrix)
B_df = pd.DataFrame(B_matrix)
C_df = pd.DataFrame(C_matrix)
D_df = pd.DataFrame(D_matrix)

# # Optionally, save these matrices as CSV files
# A_df.to_csv(f'{path}\\A_matrix.csv', index=False)
# B_df.to_csv(f'{path}\\B_matrix.csv', index=False)
# C_df.to_csv(f'{path}\\C_matrix.csv', index=False)
# D_df.to_csv(f'{path}\\D_matrix.csv', index=False)




















    
    
    
