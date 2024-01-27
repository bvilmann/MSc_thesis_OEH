from PowerSystemPowerFlow.LoadNetworkData import LoadNetworkData
from PowerSystemPowerFlow import PowerFlow as pf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class NetworkClass:
    def __init__(self,Vbase = 1, Sbase = 1,plot_network=False,plot_Ybus=False,verbose:bool=False,OWF:bool=True,**kwargs):
        data = {'scr_HSC1': 10,
                'scr_HSC2': 10,
                'scr_HSC1_icwg': 10,
                'xr_HSC1_icwg': 10,
                'scr_OWF': 10,
                'xr_HSC1': 10,
                'xr_HSC2': 10,
                'xr_OWF': 10,
                'Pload': -1,
                'Qload': 0,
                'nonlinear_load_m': 0,
                'nonlinear_load_n': 0,
                'Vbase':Vbase,
                'Sbase':Sbase,
                }

        data.update(kwargs)        
        self.data = data
        if verbose:
            plot_network = True
            plot_Ybus = True

        self.plot_network = plot_network
        self.plot_Ybus = plot_Ybus

        # Calling different networks
        self._fullNetwork(data)
        self._single(data)
        self._InterConnectionOnly_woGen(data)
        self._InterConnectionOnly_wGen(data)
        self._InterConnectionOnly_wGen_nonlinear(data)
        self._fullNetwork(data)

        return

    def complex_to_real_expanded(self,matrix):
        rows, cols = matrix.shape
        real_matrix = np.zeros((2 * rows, 2 * cols))

        for i in range(rows):
            for j in range(cols):
                real_matrix[2 * i, 2 * j] = matrix[i, j].real
                real_matrix[2 * i, 2 * j + 1] = -matrix[i, j].imag
                real_matrix[2 * i + 1, 2 * j] = matrix[i, j].imag
                real_matrix[2 * i + 1, 2 * j + 1] = matrix[i, j].real

        return real_matrix

    def _fullNetwork(self,data):
        self.lnd_full = lnd = LoadNetworkData()
        lnd.add_line(1, 3, scr=data['scr_HSC1'], xr=data['xr_HSC1'], name='Z_HSC1', Vbase=data['Vbase'], Sbase=data['Sbase'])
        lnd.add_line(2, 3, scr=data['scr_HSC2'], xr=data['xr_HSC2'], name='Z_HSC2', Vbase=data['Vbase'], Sbase=data['Sbase'])
        lnd.add_line(3, 4, scr=data['scr_OWF'], xr=data['xr_OWF'], name='Z_OWF', Vbase=data['Vbase'], Sbase=data['Sbase'])

        #
        # Add static loads (negative = generating)
        lnd.add_static_load(4, P=data['Pload'])

        self.build_network_and_save_adm_matrix(lnd,'full')

        return

    def _single(self,data):
        self.lnd_single = lnd = LoadNetworkData()
        lnd.add_line(1, 2, scr=data['scr_OWF'], xr=data['xr_OWF'], name='Z_OWF', Vbase=data['Vbase'], Sbase=data['Sbase'])

        # Add static loads (negative = generating)
        lnd.add_static_load(1, P=1/500)
        lnd.add_static_load(2, P=data['Pload'])

        self.build_network_and_save_adm_matrix(lnd,'single')

        return

    def _InterConnectionOnly_woGen(self,data):
        self.lnd_icwog = lnd = LoadNetworkData()
        lnd.add_line(1,2,scr=data['scr_HSC1_icwg'],xr=data['xr_HSC1_icwg'],name='Z_HSC1',  Vbase=data['Vbase'], Sbase=data['Sbase'])
        lnd.add_line(2,3,scr=data['scr_OWF'], xr=data['xr_OWF'],name='Z_OWF',  Vbase=data['Vbase'], Sbase=data['Sbase'])

        self.build_network_and_save_adm_matrix(lnd,'icwog')

        return

    def _InterConnectionOnly_wGen(self,data):
        self.lnd_icwg = lnd = LoadNetworkData()

        lnd.add_line(1,2,scr=data['scr_HSC1_icwg'],xr=data['xr_HSC1_icwg'],name='Z_HSC1',  Vbase=data['Vbase'], Sbase=data['Sbase'])
        lnd.add_line(2,3,scr=data['scr_OWF'], xr=data['xr_OWF'],name='Z_OWF',  Vbase=data['Vbase'], Sbase=data['Sbase'])

        # Add static loads (negative = generating)
        lnd.add_static_load(3, P=data['Pload'])

        self.build_network_and_save_adm_matrix(lnd,'icwg')

        return

    def _InterConnectionOnly_wGen_nonlinear(self,data):
        self.lnd_icwg_mva = lnd = LoadNetworkData()

        lnd.add_line(1,2,scr=data['scr_HSC1_icwg'],xr=data['xr_HSC1_icwg'],name='Z_HSC1',  Vbase=data['Vbase'], Sbase=data['Sbase'])
        lnd.add_line(2,3,scr=data['scr_OWF'], xr=data['xr_OWF'],name='Z_OWF',  Vbase=data['Vbase'], Sbase=data['Sbase'])

        # Creating network
        self.build_network_and_save_adm_matrix(lnd,'icwg_mva')

        # Create nonlinear load
        m,n =  data['nonlinear_load_m'],data['nonlinear_load_n']
        bus = self.get_initial_conditions('icwg')[0]
        vm = bus.loc[3,'Voltage']['Magn.']
        va = bus.loc[3,'Voltage']['Angle']*np.pi/180
        vr = vm*np.cos(va)
        vi = vm*np.sin(va)

        Grr = data['Pload']/(vm**2)*((m-2)*(vr/vm)**2 + 1) + data['Qload']/(vm**2)*((n-2)*(vr*vi)/(vm**2))
        Bri = data['Qload']/(vm**2)*((n-2)*(vi/vm)**2 + 1) + data['Pload']/(vm**2)*((m-2)*(vr*vi)/(vm**2))
        Bir = data['Qload']/(vm**2)*((n-2)*(vr/vm)**2 + 1) - data['Pload']/(vm**2)*((m-2)*(vr*vi)/(vm**2))
        Gii = data['Pload']/(vm**2)*((m-2)*(vi/vm)**2 + 1) - data['Qload']/(vm**2)*((n-2)*(vr*vi)/(vm**2))

        Y = self.complex_to_real_expanded(lnd.Ybus)
        Y[4,4] += Grr
        Y[4,5] += Bri
        Y[5,4] += -Bir
        Y[5,5] += Gii

        self.Y_icwg_mva = Y
        self.Z_icwg_mva = np.linalg.inv(Y)

        return

    def build_network_and_save_adm_matrix(self,lnd,name):
        # Build network
        lnd.build()

        # Obtain Ybus

        # Y = lnd.Ybus_to_real()
        Y = lnd.Ybus
        # print(pd.DataFrame(Y))

        # Verbose options
        if self.plot_network:
            lnd.draw(title=name)
            plt.close()
        if self.plot_Ybus:
            plt.imshow(abs(Y))
            plt.title(name)
            plt.close()

        setattr(self,f'Y_{name}_cmplx',Y)
        setattr(self,f'Z_{name}_cmplx',np.linalg.inv(Y))
        setattr(self,f'Y_{name}',self.complex_to_real_expanded(Y))
        setattr(self,f'Z_{name}',self.complex_to_real_expanded(np.linalg.inv(Y)))

        return

    def get_initial_conditions(self,name,silence:bool=True):
        #
        data = self.data
        lnd = LoadNetworkData()
        if name == 'single':
            lnd.add_line(1,2, scr=data['scr_OWF'], xr=data['xr_OWF'], name='Z_OWF', Vbase=data['Vbase'], Sbase=data['Sbase'])
            lnd.add_gen(1, -1, 0, v=1, theta=0)
            lnd.add_gen(2,1,0)

        elif name == 'inter_wogen':
            xr = (data['scr_HSC1'] * data['xr_HSC1'] + data['scr_HSC2'] * data['xr_HSC2']) / (
                        data['scr_HSC1'] + data['scr_HSC2'])
            scr = (data['scr_HSC1'] * data['scr_HSC2']) / (data['scr_HSC1'] + data['scr_HSC2'])

            lnd.add_line(1, 2, scr=scr, xr=xr, name='Z_HSC1', Vbase=data['Vbase'], Sbase=data['Sbase'])

            lnd.add_gen(1,0,0,v=1,theta=0)

        elif name == 'icwg':

            lnd.add_line(1, 2, scr=data['scr_HSC1_icwg'], xr=data['xr_HSC1_icwg'], name='Z_HSC1', Vbase=data['Vbase'], Sbase=data['Sbase'])
            lnd.add_line(2, 3, scr=data['scr_OWF'], xr=data['xr_OWF'], name='Z_OWF', Vbase=data['Vbase'], Sbase=data['Sbase'])
            lnd.add_gen(1, -.5, 0, v=1, theta=0)
            lnd.add_gen(2,-.5,0,v=1)
            lnd.add_gen(3,1,0)

        elif name == 'full':
            lnd.add_line(1, 3, scr=data['scr_HSC1'], xr=data['xr_HSC1'], name='Z_HSC1', Vbase=data['Vbase'],
                         Sbase=data['Sbase'])
            lnd.add_line(2, 3, scr=data['scr_HSC2'], xr=data['xr_HSC2'], name='Z_HSC2', Vbase=data['Vbase'],
                         Sbase=data['Sbase'])
            lnd.add_line(3, 4, scr=data['scr_OWF'], xr=data['xr_OWF'], name='Z_OWF', Vbase=data['Vbase'],
                         Sbase=data['Sbase'])

            lnd.add_gen(1, -.5, 0, v=1, theta=0)
            lnd.add_gen(2,-0.5,0,v=1)
            lnd.add_gen(4,1,0)

        lnd.build()
        # lnd.draw()

        # Power flow solution
        max_iter = 30  # Iteration settings
        err_tol = 1e-4

        V,success,n, J, F = pf.PowerFlowNewton(lnd.Ybus,lnd.Sbus,lnd.V0,lnd.pv_index,lnd.pq_index,max_iter,err_tol,silence=silence)
        
        # Display results if the power flow analysis converged
        if success:
            bus, branch = pf.DisplayResults(V,lnd,silence=silence)
            system = pf.getSystem(V,lnd)

            # G, fig, ax = pf.plot_pf(bus,branch)
            # fig.tight_layout()
            # plt.show()
            # plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\Special course - System identification of black-box dynamical systems\\img\\pf_graph.pdf')
            return bus,branch,system 
        else:
            return

    def load_flow2init_cond(self,name,normalize_currents:bool = False):
        # Get bus data
        bus,_,_ = self.get_initial_conditions(name)

        # preparing custom indices
        init_conds = {}
        if name == 'single':
            idxs = [1]
        elif name == 'full':
            idxs = [1,2]
        elif name == 'icwg':
            idxs = [1,2]

        # populate initial conditions
        for idx in idxs:
            vm = bus.loc[idx]['Voltage','Magn.']
            d0 = bus.loc[idx]['Voltage','Angle'] / 180 * np.pi
            p = bus.loc[idx]['Load','P']
            q = bus.loc[idx]['Load','Q']
            if isinstance(p,str):
                p = 0
            if isinstance(q,str):
                q = 0
            S = complex(p,q)

            vd0 = vm*np.cos(d0)
            vq0 = vm*np.sin(d0)
            vdq = complex(vd0,vq0)
            idq = np.conjugate(S/vdq)
            i_norm = (1,abs(max(idq.real,idq.imag)))[normalize_currents]
            id0 = idq.real / i_norm
            iq0 = idq.imag / i_norm
            init_conds[idx] = {'vd0':vd0, 'vq0':vq0, 'id0':id0, 'iq0':iq0, 'd0':d0}

        return init_conds
    

def form_network_ssm(A_D,B_D,C_D,Z):
    A = A_D + B_D @ Z @ C_D
    A = A_D - B_D @ Z @ C_D
    return A





