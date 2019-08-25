from scipy import integrate
from scipy import special
import numpy as np
from cmath import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patch_geo_func import x_ep, y_ep
from sys import stdout
import warnings
from assign_attr import *
from repel_system import *
np.seterr(invalid = 'raise', under = 'ignore', over = 'ignore')
#LR_Pi_file = 'cortex_94-Ny/Ny-2-LR_Pi.bin'
#pos_file = 'server_data/test_3d_pos.bin'
#pos_file = 'server_data/test_low_3d_pos.bin'
LR_Pi_file = 'Ny-2-LR_Pi.bin'
#pos_file = 'sobol_test_low_3d_pos.bin'
pos_file = 'uniform_pos_file.bin'
nblock = 32 #5209 #32
blockSize = 1024

a = 0.635
b = 96.7
k = np.sqrt(140)*0.873145
ecc = 2.0 # must consistent with the corresponding variables in parameter.m and macro.ipynb
p0 = -np.pi/2
p1 = np.pi/2

grid = np.array([64,104])*2
nx = grid[0]
ny = grid[1]
W = x_ep(ecc,0,k,a,b)
d = (1+2/nx)*W/nx
x = np.linspace(-W/nx, W+W/nx, nx)
W = W+2*W/nx
H = d*ny
y = np.linspace(-H/2, H/2, ny)
mMap = macroMap(nx, ny, x, y, nblock, blockSize, LR_Pi_file, pos_file, a, b, k, ecc, p0, p1)
fig = plt.figure('macroMap')
ax1 = fig.add_subplot(121, projection='polar')
ax2 = fig.add_subplot(122)
#mMap.assign_pos_OD0()
#mMap.assign_pos_OD1()
mMap.pODready = False
mMap.plot_map(ax1,ax2,True,False)
ax1.set_thetamin(p0/np.pi*180)
ax1.set_thetamax(p1/np.pi*180)
ax1.set_rmax(2.0)
ax1.set_rmin(0.0)
ax1.grid(False)
ax1.tick_params(labelleft=False, labelright=True,
               labeltop=False, labelbottom=True)
#ax1.set_yticks([0,0.5,1.00,1.50,2.00])
ax2.set_aspect('equal')
fig.savefig('sobol_test_low_density_uniform.png', dpi = 1000)

# spread for VF
fig = plt.figure('vposR', dpi = 1000)
dx = mMap.x[1] - mMap.x[0]
dy = mMap.y[1] - mMap.y[0]
ax1 = fig.add_subplot(111)
ax1.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
ax1.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
ax1.set_aspect('equal') 
dt = np.power(2.0,-np.arange(5,6)).reshape(6-5,1)
dt = np.tile(dt,(1,25)).flatten()
firstTime = False
if firstTime is True:
    R = mMap.LR.copy()
    R[R < 0] = 0
    R[R > 0] = 1
    vposR = mMap.pos[:, mMap.ODlabel>0].copy()
else:
    with open('vposR.bin','rb') as f:
        vposR = np.fromfile(f).reshape(2, np.sum(mMap.ODlabel>0))
    with open('R.bin','rb') as f:
        R = np.fromfile(f).reshape(mMap.Pi.shape)
        print(np.sum(R>0))
spreaded = False
while spreaded is False:
    vposR, R, spreaded = mMap.spread_vpos(dt, vposR, R, seed = 17482321, ax = ax1)
    print(np.sum(R>0))
with open('vposR.bin','wb') as f:
    vposR.tofile(f)
with open('R.bin','wb') as f:
    R.tofile(f)
fig.savefig('spread_VF_R.png', dpi = 1000)
