# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:01:03 2023

@author: bvilm
"""

import matplotlib.pyplot as plt
import numpy as np

# Function to plot a complex number as an arrow
def plot_complex(ax, z, **kwargs):
    return ax.annotate('', xy=(z.real, z.imag), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", **kwargs))

v = 2+1j
i = (1.1+2j)/2
z = 1.5+0.5j

v_vi_a_a = i.real*z.real
v_vi_a_b = i.imag*z.imag
v_vi_b_a = i.imag*z.real
v_vi_b_b = -i.real*z.imag

v_vi_a = v_vi_a_a +v_vi_a_b
v_vi_b = v_vi_b_a +v_vi_b_b

v_vi = v_vi_a + 1j*v_vi_b

phi_z = np.arctan(z.imag/z.real)
phi_i = np.arctan(i.imag/i.real)
phi_v = np.arctan(v.imag/v.real)
phi_vz = np.arctan(v_vi.imag/v_vi.real)


# v_vi *= np.exp(1j*phi_z)*np.exp(-1j*phi_i)
v_iz = i*z*np.exp(0j*phi_i)*np.exp(-2j*phi_z)
v_vi *= np.exp(0j*phi_v)


# List of complex numbers

complex_numbers = [v,i,z,i*z,v_iz,v_vi]


# Set up the plot
fig, ax = plt.subplots(dpi=200)
ax.set_xlim(0, 4)
ax.set_ylim(0, 3)
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

# Plot each complex number as an arrow
for i, z in enumerate(complex_numbers):
    plot_complex(ax, z, color=['blue','red','green','purple','indigo','skyblue'][i])

ax.set_aspect('equal')

ax.annotate('', xy=(v_vi_a_a+v.real, 0+v.imag), xytext=(v.real, v.imag),arrowprops=dict(arrowstyle="->"))
ax.annotate('', xy=(v_vi_a_a+v_vi_a_b+v.real, 0+v.imag), xytext=(v.real+v_vi_a_a, v.imag),arrowprops=dict(arrowstyle="->"))
ax.annotate('', xy=(v.real, v_vi_b_a+v.imag), xytext=(v.real, v.imag),arrowprops=dict(arrowstyle="->"))
ax.annotate('', xy=(v.real+0.1, v_vi_b_a+v_vi_b_b+v.imag), xytext=(v.real+0.1, v.imag+v_vi_b_a),arrowprops=dict(arrowstyle="->"))
ax.annotate('', xy=(v.real+v_vi_a, v.imag+v_vi_b), xytext=(v.real, v.imag),arrowprops=dict(arrowstyle="->"))


# Add grid and show plot
ax.grid(True)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Complex Number Vectors')
plt.show()

#%%
v = 2+1j
i = (1.1+2j)/2
z = 1.5+0.33j
r =z.real
x = 1j*z.imag

v_vi_a_a = i.real*z.real
v_vi_a_b = i.imag*z.imag
v_vi_b_a = i.imag*z.real
v_vi_b_b = -i.real*z.imag
v_vi_a = v_vi_a_a +v_vi_a_b
v_vi_b = v_vi_b_a +v_vi_b_b
v_z = v_vi_a + 1j*v_vi_b

v_vi_a_a = i.real*z.real
v_vi_a_b = i.imag*0
v_vi_b_a = i.imag*z.real
v_vi_b_b = -i.real*0
v_vi_a = v_vi_a_a +v_vi_a_b
v_vi_b = v_vi_b_a +v_vi_b_b
v_r = v_vi_a + 1j*v_vi_b

v_vi_a_a = i.real*0
v_vi_a_b = i.imag*z.imag
v_vi_b_a = i.imag*0
v_vi_b_b = -i.real*z.imag
v_vi_a = v_vi_a_a +v_vi_a_b
v_vi_b = v_vi_b_a +v_vi_b_b
v_x = v_vi_a + 1j*v_vi_b


# List of complex numbers

complex_numbers = [v,i,z,r,x]


# Set up the plot
fig, ax = plt.subplots(dpi=200)
ax.set_xlim(0, 4)
ax.set_ylim(0, 3)
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

# Plot each complex number as an arrow
for i, z in enumerate(complex_numbers):
    plot_complex(ax, z, color=['blue','red','green','lightgreen','cyan','lightblue','indigo','skyblue'][i])

ax.set_aspect('equal')

for i, vec in enumerate([v_z,v_r,v_x]):
    ax.annotate('', xy=(vec.real+v.real, vec.imag+v.imag), xytext=(v.real, v.imag),arrowprops=dict(arrowstyle="->",color=['lightblue','indigo','skyblue'][i]))
    

# ax.annotate('', xy=(v_vi_a_a+v_vi_a_b+v.real, 0+v.imag), xytext=(v.real+v_vi_a_a, v.imag),arrowprops=dict(arrowstyle="->"))
# ax.annotate('', xy=(v.real, v_vi_b_a+v.imag), xytext=(v.real, v.imag),arrowprops=dict(arrowstyle="->"))
# ax.annotate('', xy=(v.real+0.1, v_vi_b_a+v_vi_b_b+v.imag), xytext=(v.real+0.1, v.imag+v_vi_b_a),arrowprops=dict(arrowstyle="->"))
# ax.annotate('', xy=(v.real+v_vi_a, v.imag+v_vi_b), xytext=(v.real, v.imag),arrowprops=dict(arrowstyle="->"))


# Add grid and show plot
ax.grid(True)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Complex Number Vectors')
plt.show()


