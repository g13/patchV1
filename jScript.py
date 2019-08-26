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
mMap.assign_pos_OD1()
# spread for VF
fig = plt.figure('vposL', dpi = 1000)
dx = mMap.x[1] - mMap.x[0]
dy = mMap.y[1] - mMap.y[0]
ax1 = fig.add_subplot(121)
ax1.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
ax1.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
ax1.set_aspect('equal') 
ax2 = fig.add_subplot(122)
ax2.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
ax2.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
ax2.set_aspect('equal') 
with open('vposL.bin','rb') as f:
    vposL = np.fromfile(f).reshape(2, np.sum(mMap.ODlabel<0))
ax1.plot(np.vstack((vposL[0,:], mMap.pos[0,mMap.ODlabel<0])), np.vstack((vposL[1,:], mMap.pos[1,mMap.ODlabel<0])), '-b', lw = 0.01)
ax1.plot(vposL[0,:], vposL[1,:], ',k')
ax1.plot(mMap.pos[0,mMap.ODlabel<0], mMap.pos[1,mMap.ODlabel<0], ',m')
ax2.plot(vposL[0,:], vposL[1,:], ',k')
fig.savefig('spread_VF_L.png', dpi = 1000)

fig = plt.figure('vposR', dpi = 1000)
dx = mMap.x[1] - mMap.x[0]
dy = mMap.y[1] - mMap.y[0]
ax1 = fig.add_subplot(121)
ax1.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
ax1.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
ax1.set_aspect('equal') 
ax2 = fig.add_subplot(122)
ax2.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
ax2.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
ax2.set_aspect('equal') 
with open('vposR.bin','rb') as f:
    vposR = np.fromfile(f).reshape(2, np.sum(mMap.ODlabel>0))
ax1.plot(np.vstack((vposR[0,:], mMap.pos[0,mMap.ODlabel>0])), np.vstack((vposR[1,:], mMap.pos[1,mMap.ODlabel>0])), '-b', lw = 0.01)
ax1.plot(vposR[0,:], vposR[1,:], ',k')
ax1.plot(mMap.pos[0,mMap.ODlabel>0], mMap.pos[1,mMap.ODlabel>0], ',m')
ax2.plot(vposR[0,:], vposR[1,:], ',k')
fig.savefig('spread_VF_R.png', dpi = 1000)

mMap.save(VF_file = 'vpos.bin')
