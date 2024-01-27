import numpy as np
import matplotlib.pyplot as plt

# Function for first-order Taylor series approximation of sin(x)
def taylor_series_sin_first_order(x, a):
    return np.sin(a) + np.cos(a) * (x - a)

# Define the range and the function
x = np.linspace(-np.pi, np.pi, 400)
y = np.sin(x)

# Points of evaluation
points = [0.0, 1.125]
colors = ['red', 'green']

dim = 1.2
fig, ax = plt.subplots(1,1,figsize=(6*dim,3*dim))

# Plot the original function
plt.plot(x, y, label='sin(x)',zorder=4)

# Plot the first-order Taylor series approximations at specified points
for point, color in zip(points, colors):
    y_taylor = taylor_series_sin_first_order(x, point)
    plt.plot(x, y_taylor, label=f'Linear Approx. at {point}', color=color,ls='--',zorder=5)
    plt.scatter([point], [np.sin(point)], color=color,zorder=5)

# ax.title('First-Order Taylor Series Approximation of sin(x)')
ax.set_xlabel('$x$')
ax.set_ylabel('$sin(x)$')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

ax.set_ylim(-1.1,1.1)
ax.set_xlim(-np.pi,np.pi)
ax.grid(True)
ax.legend(loc='upper left')
fig.tight_layout()
path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\06_state_space\img'
plt.savefig(f'{path}\\taylor.pdf')
