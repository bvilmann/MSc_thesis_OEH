import numpy as np
import matplotlib.pyplot as plt
# Creating dataset
x = np.outer(np.linspace(-3, 3
                         , 32), np.ones(32))

dx1 = 32
x = np.outer(np.linspace(-3, 3
                         , dx1), np.ones(dx1))
y = x.copy().T # transpose
z = (np.cos(x **2) + np.cos(y **2) ) + np.cos(x) + np.cos(y )
z = (np.cos(x) + np.cos(y ) )

def fun(x,y):
    return np.exp(-3)*(np.cos(x)+np.sin(y))

z = np.exp(-3)*(np.cos(x)+np.sin(y))
# z = -np.exp(-(x+y))*(np.cos(x)+np.sin(y))

# Creating figure
fig = plt.figure(figsize =(8, 5),dpi=200)
ax = plt.axes(projection ='3d')

x1 = np.linspace(-1,1.5,7)
y1 = np.linspace(0.5,1.5,7)
ax.plot(x1,y1,fun(x1,y1),color='black',zorder=3,marker='x')

x2 = np.linspace(-1,1.5,7)
y2 = np.linspace(0.5,1.5,7)
ax.plot(x2,y2,fun(x2,y2),color='black',zorder=3,marker='x')
# Creating plot

ax.plot_surface(x, y, z, cmap = plt.get_cmap('inferno'), edgecolor ='none')
# ax.plot_surface(x, y, np.where(z<=0.01,z,np.nan), cmap = plt.get_cmap('inferno'), edgecolor ='none')
# ax.plot_surface(x, y, np.where(z>=0,z,np.nan), cmap = plt.get_cmap('Greys'), edgecolor ='none')

# fig.suptitle('$exp^{-3}\\left(\\cos(x)+sin(y)\\right)$')
# show plot
ax.set_xlabel('$\\hat{x}_1$')
ax.set_ylabel('$\\hat{x}_2$')
ax.set_zlabel('$\\Re{\\left(\\lambda_1\\right)}$')
plt.show()