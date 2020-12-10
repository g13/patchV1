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
np.seterr(invalid = 'ignore', under = 'ignore', over = 'ignore')
from assign_attr import *
import functools
print = functools.partial(print, flush=True)

ndt0 = 5 # 15
ndt1 = 20 # 10
ndt = 10 #25
ncore = 32
parallel = True
k1 = 0.5
k2 = 0.25
p_scale = 2.25
b_scale = 1.125
roi_ratio = 8.0/3.0 
limit_ratio = 1.0
#pos_name = 'V1_pos_2D_lowDensity'
newPos = False
newOP = False 
newOD = False 

pos_name = 'fD_pos'
LR_Pi_file = './FullFledged/or-ft10-ext-or-ft10-nG4s-lr2/or-ft10-nG4s-lr2-4-LR_Pi10.bin'
OR_file = './FullFledged/or-ft10-ext-or-ft10-nG4s-lr2/or-ft10-nG4s-lr2-4/ORcolor-f0011.bin'
if newPos:
    pos_file = pos_name+'.bin'
else:
    pos_file = pos_name+'_adjusted.bin'

if newOP:
    OP_file = None
else:
    OP_file = pos_name + '-op_file.bin'

if newOD:
    OD_file = None
else:
    OD_file = pos_name + '-od_file.bin'

uniform_pos_file = 'uniform_' + pos_name + '.bin'

#mMap = macroMap(LR_Pi_file, pos_file, posUniform = False, OD_file = OD_file, OPgrid_file = OR_file, OP_file = OP_file)
mMap = macroMap(LR_Pi_file, uniform_pos_file, posUniform = True, OD_file = OD_file, OPgrid_file = OR_file, OP_file = OP_file)
"""
fig = plt.figure('macroMap',dpi=1200)
ax1 = fig.add_subplot(111)
#mMap.pODready = False
mMap.plot_map(ax1, None, None, fig.dpi, pltOD = True, pltVF = False, pltOP = True, forceOP = True)
ax1.set_aspect('equal')
fig.savefig(pos_name+'1.png')
"""

if newOD:
    mMap.save(OD_file = pos_name + '-od_file.bin')

if newOP:
    mMap.save(OP_file = pos_name + '-op_file.bin')

if newPos:
    mMap.save(pos_file = pos_name + '_adjusted.bin')

dx = mMap.x[1] - mMap.x[0]
dy = mMap.y[1] - mMap.y[0]
print('#spread uniformly')
if not mMap.posUniform:
    dt0 = np.power(2.0,-np.arange(10,11)).reshape(1,1)
    dt1 = np.power(2.0,-np.arange(11,12)).reshape(1,1)
    dt = np.hstack((np.tile(dt0,(ndt0,1)).flatten(), np.tile(dt1,(ndt1,1)).flatten()))
    if parallel:
        oldpos = mMap.make_pos_uniform_p(dt, p_scale, b_scale, pos_name+'2', ncore = ncore, ndt_decay = ndt0, roi_ratio = roi_ratio, k1 = k1, k2 = k2, chop_ratio = 0, spercent = 0.01, seed = -1, local_plot = False)
        print(f'mean: {np.mean(oldpos - mMap.pos, axis = 1)}')
        print(f'std: {np.std(oldpos - mMap.pos, axis = 1)}')
    else:
        fig = plt.figure('pos', dpi = 600)
        ax1 = fig.add_subplot(121)
        ax1.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
        ax1.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
        ax1.set_aspect('equal')
        ax2 = fig.add_subplot(122)
        ax2.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
        ax2.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
        ax2.set_aspect('equal')
        oldpos, cL, cR, nL, nR = mMap.make_pos_uniform(dt, seed = 17482321, ax1 = ax1, ax2 = ax2)
        fig.savefig(pos_name+'2.png', dpi = 2000)
        del fig
    mMap.save(pos_file = uniform_pos_file)
#print('check')
#mMap.check_pos()

print('#stretch for vpos')
limit_ratio = 2.0
#'''
fig = plt.figure('vposL', dpi = 600)
dx = mMap.x[1] - mMap.x[0]
dy = mMap.y[1] - mMap.y[0]
ax1 = fig.add_subplot(111)
ax1.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
ax1.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
ax1.set_aspect('equal')
dt0 = np.power(2.0,-np.arange(9,10)).reshape(1,1)
dt1 = np.power(2.0,-np.arange(9,10)).reshape(1,1)
dt = np.hstack((np.tile(dt0,(1,10)).flatten(), np.tile(dt1,(1,ndt)).flatten()))
#dt = np.power(2.0,-np.array([11,10])).reshape(2,1)
#dt = np.tile(dt,(1,ndt)).flatten()
print(dt)
ndt_decay = 10
fT = True # firstTime
ct = True 
tmpL = 'tmpL5'
tmpVF_L = 'tmpVF_L5'
ax1 = None
mMap.spread_pos_VF(dt, tmpVF_L, tmpL, 'L', p_scale = p_scale, b_scale = b_scale, firstTime = fT, continuous = ct, ax = ax1, ndt_decay = ndt_decay, roi_ratio = roi_ratio, k1 = k1, k2 = k2, ncore = ncore, limit_ratio = limit_ratio)
#mMap.spread_pos_VF(dt, tmpVF_L, tmpL, 'L', firstTime = fT, ax = None, ndt_decay = ndt_decay, ncore = ncore)
#fig.savefig('vposL_in_'+pos_name+'.png', dpi = 3000)
#'''
'''
fig = plt.figure('vposR', dpi = 600)
ax2 = fig.add_subplot(111)
ax2.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
ax2.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
ax2.set_aspect('equal')
dt0 = np.power(2.0,-np.arange(9,10)).reshape(1,1)
dt1 = np.power(2.0,-np.arange(10,11)).reshape(1,1)
dt = np.hstack((np.tile(dt0,(1,2)).flatten(), np.tile(dt1,(1,ndt)).flatten()))
dt = np.power(2.0,-np.array([11,10])).reshape(2,1)
dt = np.tile(dt,(1,ndt)).flatten()
print(dt)
ndt_decay = 0
fT = True# firstTime
ct = True
tmpR = 'tmpR5'
tmpVF_R = 'tmpVF_R5'
ax2 = None
mMap.spread_pos_VF(dt, tmpVF_R, tmpR, 'R', p_scale = p_scale, b_scale = b_scale, firstTime = fT, continuous = ct, ax = ax2, ndt_decay = ndt_decay, k1 = k1, k2 = k2, ncore = ncore, limit_ratio = limit_ratio)
#mMap.spread_pos_VF(dt, tmpVF_R, tmpR, 'R', firstTime = fT, ax = None, ndt_decay = ndt_decay, ncore = ncore)
#fig.savefig('vposR_in_'+pos_name+'.png', dpi = 3000)
mMap.save(VFxy_file = 'vpos_xy-' + pos_name + '.bin')
mMap.save(VFpolar_file = 'vpos_polar-' + pos_name + '.bin')
'''
