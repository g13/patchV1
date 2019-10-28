import pdb
from scipy import integrate
from scipy import special
import numpy as np
from cmath import *
import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex
from patch_geo_func import x_ep, y_ep
from sys import stdout
import warnings
np.seterr(invalid = 'raise', under = 'ignore', over = 'ignore')
from assign_attr import *
from repel_system import *

LR_Pi_file = 'Ny-2-LR_Pi.bin'
#pos_file = 'server_data/test_3d_pos.bin'
#pos_file = 'server_data/test_low_3d_pos.bin'
#LR_Pi_file = 'Ny-2-LR_Pi.bin'
#pos_file = 'ss_low_3d_pos.bin'
pos_file = 'ss_low_closer_pos3.bin'
#pos_file = 'ss_pos_file.bin'
OD_file = 'ss_od_file.bin'
OR_file = 'ORcolor.bin'
vpos_file = 'vpos.bin'
nblock = 32 #5209 #32
blockSize = 1024
uniform_pos_file = 'uniform_' + pos_file

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
x = np.linspace(-W/(2*nx-4), W+W/(2*nx-4), nx)
W = W+W/(nx-2)
d = W/(nx-1)
H = d*ny
y = np.linspace(-H/2, H/2, ny)
#mMap = macroMap(nx, ny, x, y, nblock, blockSize, LR_Pi_file, 'uniform_' + pos_file, OR_file, a, b, k, ecc, p0, p1, posUniform = True, OD_file = OD_file)
mMap = macroMap(nx, ny, x, y, nblock, blockSize, LR_Pi_file, 'uniform_' + pos_file, OR_file, a, b, k, ecc, p0, p1, posUniform = True, OD_file = OD_file)
#mMap = macroMap(nx, ny, x, y, nblock, blockSize, LR_Pi_file, pos_file, OR_file, a, b, k, ecc, p0, p1, posUniform = False)
#mMap = macroMap(nx, ny, x, y, nblock, blockSize, LR_Pi_file, pos_file, OR_file, a, b, k, ecc, p0, p1, posUniform = False, OD_file = OD_file)
if mMap.pODready:
    print('OD_labels...')
    assert(np.sum(mMap.ODlabel>0) + np.sum(mMap.ODlabel<0) == mMap.networkSize)
    print('checked')

fig = plt.figure('macroMap',dpi=1000)
ax1 = fig.add_subplot(132, projection='polar')
ax2 = fig.add_subplot(131)
#mMap.pODready = False
mMap.plot_map(ax1, ax2, fig.dpi, pltOD = True, pltVF = False, pltOP = False)
ax1.set_thetamin(p0/np.pi*180)
ax1.set_thetamax(p1/np.pi*180)
ax1.set_rmax(2.0)
ax1.set_rmin(0.0)
ax1.grid(False)
ax1.tick_params(labelleft=False, labelright=True,
               labeltop=False, labelbottom=True)
ax2.set_aspect('equal')
fig.savefig('sobol_test_low_density_uniform.png')
mMap.save(OD_file = 'ss_od_file.bin')

dx = mMap.x[1] - mMap.x[0]
dy = mMap.y[1] - mMap.y[0]
print('#spread uniformly')
if not mMap.posUniform:
    fig = plt.figure('pos', dpi = 600)
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
    ax1.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
    ax1.set_aspect('equal')
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
    ax2.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
    ax2.set_aspect('equal')
    dt0 = np.power(2.0,-np.arange(6,7)).reshape(1,1)
    dt1 = np.power(2.0,-np.arange(8,9)).reshape(1,1)
    dt = np.hstack((np.tile(dt0,(15,1)).flatten(), np.tile(dt1,(5,1)).flatten()))
    oldpos, cL, cR, nL, nR = mMap.make_pos_uniform(dt, seed = 17482321, ax1 = ax1, ax2 = ax2)
    fig.savefig(pos_file+'.png', dpi = 2000)
    mMap.save(pos_file = uniform_pos_file)
else:
    mMap.check_pos()

print('#stretch for vpos')
fig = plt.figure('vpos', dpi = 600)
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
dt = np.power(2.0,-np.arange(6,7)).reshape(1,1)
dt = np.tile(dt,(1,25)).flatten()
fT = False # firstTime
tmpL = 'tmpL.bin'
tmpVF_L = 'tmpVF_L.bin'
mMap.spread_pos_VF(dt, tmpVF_L, tmpL, 'L', firstTime = fT, ax = ax1)
fT = True # firstTime
tmpR = 'tmpR.bin'
tmpVF_R = 'tmpVF_R.bin'
mMap.spread_pos_VF(dt, tmpVF_R, tmpR, 'R', firstTime = fT, ax = ax2)
fig.savefig('vpos_in_cortex.png', dpi = 2000)
