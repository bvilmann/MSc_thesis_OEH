# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:46:01 2023

@author: bvilm
"""

import mhi.pscad
import pandas as pd
import os
def load_default_project_settings(proj,df):
    
    # 
    settings = df_proj.val.to_dict()
    
    for k, v in df_proj.type.to_dict().items():
        if v == 'float':
            settings[k] = float(settings[k])
        elif v == 'int':
            settings[k] = int(settings[k])
    
    proj.parameters()

    return 

def run_project(proj):
    proj.build()
    proj.run()
    return

def move_output_files(source_path,store_path,save_filename):
    # Move output files
    files = [f for f in os.listdir(f'{source_path}') if save_filename in f and f.split('.')[-1] in ['infx','out']]
    
    for f in files:
        os.rename(f'{source_path}\\{f}', f'{store_path}\\{f}')
    return


# class pscad:
    
#     def __init__(self,ws):
#         self.app = 
        
#         return
#%% =============== SETTINGS ================
plot = False
output_filename = 'asdf'

ws_dir = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad'
ws_name = 'OEH_ws'

ws = f"{ws_dir}\\{ws_name}.pswx"



#%% =============== LOAD APPLICATION, WORKSPACE, AND PROJECT ===============

# Launch application
pscad = mhi.pscad.launch()

# Load workspace
pscad.load(ws)

# Get project
proj = pscad.project('OEH')


#%% =============== Change variables ===============
df = pd.read_excel(r'C:\Users\bvilm\PycharmProjects\StateSpaceModelling\pscad_systems\pscad_sys.xlsx', sheet_name = 'loop_validation',index_col=0)
df_proj = pd.read_excel(r'C:\Users\bvilm\PycharmProjects\StateSpaceModelling\pscad_systems\pscad_sys.xlsx', sheet_name = 'project',index_col=0)

load_default_project_settings(proj,df_proj)

source_path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\OEH.gf46'
store_path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\results'

for i, r in df.iterrows():
    print(i,r)
    for k,v in r.to_dict().items():   
        for HSC in ['HSC1','HSC2']:
            print(f'{HSC}_{k}')
            objs = proj.find_all(f'{HSC}_{k}')
            assert len(objs) == 1
        
            for obj in objs:
                obj.parameters(Value=v)
                # print(obj,obj.get_parameters())
    # 
    save_filename = f'loop_validation_{i}'
    proj.parameters(
        PlotType = 1,
        output_filename = save_filename,
        )
    
    run_project(proj)

    move_output_files(source_path,store_path,save_filename)   



#%% =============== Change project settings ===============
# 
proj.parameters(
    time_duration=2.5,
    MrunType=0,
    PlotType=0,
    output_filename = output_filename,
    StartType = 0, # not from snap
    startup_filename=0,
    sample_step = 250,
    time_step = 5,
    )


