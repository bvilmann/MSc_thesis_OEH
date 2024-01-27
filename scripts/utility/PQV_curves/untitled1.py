import matplotlib.pyplot as plt
import numpy as np

def van_der_pol(t, xy, mu):
    x, y = xy
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]


# Define the system of ODEs
def system(t, xy):
    x, y = xy
    dxdt = -y + 0.5*x
    dydt = x -  0.3*y
    return np.array([dxdt, dydt])

#%%

# Create a grid of points
vmin = -6
vmax = 6
x = np.linspace(vmin, vmax, 20)
y = np.linspace(vmin, vmax, 20)
X, Y = np.meshgrid(x, y)

# Evaluate the vector field
U, V = np.empty(X.shape), np.empty(Y.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xy = X[i, j], Y[i, j]
        U[i, j], V[i, j] = system(0, xy)



# Plot the vector field
fig,ax =plt.subplots(1,1,dpi=150)
plt.quiver(X, Y, U, V, np.sqrt(U**2 + V**2), scale=30, scale_units='inches', pivot='mid')
ax.set(xlim=[vmin, vmax],ylim=[vmin,vmax],xlabel='x',ylabel='y')
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def system(t, xy):
    x, y = xy
    dxdt = -y + 0.05*x
    dydt = x - 0.1*y
    return [dxdt, dydt]

t_span = [0, 100]  # from t=0 to t=10
initial_point = [0,5 ]  # starting at x=1, y=0
solution = solve_ivp(system, t_span, initial_point, t_eval=np.linspace(*t_span, 1000))

# Create a grid of points
vmin = -6
vmax = 6
x = np.linspace(vmin, vmax, 20)
y = np.linspace(vmin, vmax, 20)
X, Y = np.meshgrid(x, y)

# Evaluate the vector field
U, V = np.empty(X.shape), np.empty(Y.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xy = X[i, j], Y[i, j]
        U[i, j], V[i, j] = system(0, xy)


fig,ax =plt.subplots(1,1,dpi=150)

# Plot the vector field (phase portrait)
ax.quiver(X, Y, U, V, scale=10, scale_units='inches', pivot='mid', color='gray', alpha=0.5)

# Plot the trajectory from the specified initial point
ax.plot(solution.y[0], solution.y[1], color='blue',lw=0.25)

ax.scatter([initial_point[0]],[initial_point[1]],marker='x',color='k',lw=0.75,s=10)

# plt.quiver(X, Y, U, V, np.sqrt(U**2 + V**2), scale=30, scale_units='inches', pivot='mid')
ax.set(xlim=[vmin, vmax],ylim=[vmin,vmax],xlabel='x',ylabel='y')
ax.set(xlim=[vmin, vmax],ylim=[vmin,vmax])

plt.show()

#%% VAN DER POL OSCILLATOR

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def van_der_pol(t, xy, mu):
    x, y = xy
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]


t_span = [0, 100]  # from t=0 to t=10
initial_point = [0,5 ]  # starting at x=1, y=0
mu_value = 0.6  # example value for mu
solution = solve_ivp(van_der_pol, t_span, initial_point, args=(mu_value,), t_eval=np.linspace(*t_span, 1000))

# Create a grid of points
vmin = -3
vmax = 3
x = np.linspace(vmin, vmax, 20)
y = np.linspace(vmin, vmax, 20)
X, Y = np.meshgrid(x, y)

# Evaluate the vector field
U, V = np.empty(X.shape), np.empty(Y.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xy = X[i, j], Y[i, j]
        U[i, j], V[i, j] = van_der_pol(0, xy,mu_value)


fig,ax =plt.subplots(1,1,dpi=150)

# Plot the vector field (phase portrait)
# ax.quiver(X, Y, U, V, scale=10, scale_units='inches', pivot='mid', color='gray', alpha=0.5)
magnitude = np.sqrt(U**2 + V**2)
plt.quiver(X, Y, U, V, color=plt.cm.viridis(magnitude / magnitude.max()), scale=20, scale_units='inches', pivot='mid')
# ax.quiver(X, Y, U, V, np.sqrt(U**2 + V**2), scale=20, pivot='mid', cmap='RdYlBu_r')

# Plot the trajectory from the specified initial point
ax.plot(solution.y[0], solution.y[1], color='blue',lw=0.25)

ax.scatter([initial_point[0]],[initial_point[1]],marker='x',color='k',lw=0.75,s=10)

# plt.quiver(X, Y, U, V, np.sqrt(U**2 + V**2), scale=30, scale_units='inches', pivot='mid')
ax.set(xlim=[vmin, vmax],ylim=[vmin,vmax],xlabel='x',ylabel='y')
ax.set(xlim=[vmin, vmax],ylim=[vmin,vmax])

plt.show()

