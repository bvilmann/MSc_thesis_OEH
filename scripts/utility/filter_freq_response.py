import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
w = w_ = 1.3  # Natural frequency
z = 0.6  # Damping factor (for the 2nd order filters)

#%%
# Define the transfer functions
# 1st order low pass filter: w/(s+w)
num_1st_order = [w]
den_1st_order = [1, w]
H_1st_order = signal.TransferFunction(num_1st_order, den_1st_order)

num_1st_order_lag = [w]
den_1st_order_lag = [1, 0]
H_1st_order_lag = signal.TransferFunction(num_1st_order_lag, den_1st_order_lag)

# 2nd order low pass filter: w^2/(s^2+s*2*z*w+w^2)
num_2nd_order = [w**2]
den_2nd_order = [1, 2*z*w, w**2]
H_2nd_order = signal.TransferFunction(num_2nd_order, den_2nd_order)

# 2nd order high reject filter: (s*w+w^2)/(s^2+s*z*w+w^2)
num_2nd_order_high_reject = [w, w**2]
den_2nd_order_high_reject = [1, 2*z*w, w**2]
H_2nd_order_high_reject = signal.TransferFunction(num_2nd_order_high_reject, den_2nd_order_high_reject)

# 2nd order high reject filter: (s*w+w^2)/(s^2+s*z*w+w^2)
num_HPF = [1, 0]
den_HPF = [1,w]
HPF = signal.TransferFunction(num_HPF, den_HPF)

# Frequency range for Bode plot
frequencies = np.logspace(-2, 2, 500)

# Plotting
fig = plt.figure(figsize=(8, 6))

for H, label in zip([H_1st_order, H_2nd_order, H_2nd_order_high_reject,HPF], 
                    ['1st Order Low Pass', '2nd Order Low Pass', '2nd Order High Reject','1st Order High Pass']):
    w, mag, phase = signal.bode(H, w=frequencies)
    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag, label=label)
    plt.ylabel('Magnitude (dB)')
    plt.grid(which='both',ls=':')
    plt.xlim(frequencies[0],frequencies[-1])
    plt.legend(loc='upper right',fontsize=8,ncol=2)
    plt.ylim(-25,15)

    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Phase (degrees)')
    plt.xlim(frequencies[0],frequencies[-1])

plt.grid(which='both',ls=':')
fig.align_ylabels()

plt.tight_layout()
# plt.show()

plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\06_state_space\img'

plt.savefig(f'{plot_path}\\filter_freq_resp.pdf')

#%%
ss = H_2nd_order.to_ss()

ss = H_2nd_order_high_reject.to_ss()
# ss = H_1st_order.to_ss()
# ss = HPF.to_ss()
ss = H_1st_order_lag.to_ss()
ss = H_1st_order.to_ss()
print(f'w = {w_}')
print(f'z = {z}')
for abc in ['A','B','C','D']:
    print(abc,': ')        
    print(getattr(ss,abc))

