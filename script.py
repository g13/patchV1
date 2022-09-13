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

ndt0 = 1 # 15
ndt1 = 1 # 10
ndt = 1 #25
ncore = 32 
k1 = 1.0
k2 = 0.5
p_scale = 1.0
b_scale = 0.5
roi_ratio = 4.0
limit_ratio = 1.0
chop_ratio = 0
crop = 0.0
#pos_name = 'V1_pos_2D_lowDensity'
damp = 0.5
l_ratio = 0.5
l_k = 0.0

newPos = False 
newOP = True 
newOD = True 

#pos_name = 'fD
pos_name = 'fD3'
lrfile = pos_name + '-bound'

#if crop > 0:
#    LR_Pi_file = 'or-ft10-nG4s-lr2-4-LR_Pi10-3x.bin'
#    OR_file = 'ORcolor-f0011-3x.bin'
#else:
#    LR_Pi_file = 'or-ft10-nG4s-lr2-4-LR_Pi10-3x_adjusted.bin'
#    OR_file = 'ORcolor-f0011-3x_adjusted.bin'
LR_Pi_file = './FullFledged/or-ft10-ext-or-ft10-nG4s-lr2/or-ft10-nG4s-lr2-4-LR_Pi10.bin'
OR_file = './FullFledged/or-ft10-ext-or-ft10-nG4s-lr2/or-ft10-nG4s-lr2-4/ORcolor-f0011.bin'
if newPos:
    pos_file = pos_name+'.bin'
else:
    pos_file = pos_name+'_adjusted.bin'
    #pos_file = pos_name+'.bin'

if newOP:
    OP_file = None
else:
    OP_file = pos_name + '-op_file.bin'

if newOD:
    OD_file = None
else:
    OD_file = pos_name + '-od_file.bin'

uniform_pos_file = 'uniform_' + pos_name + '.bin'

mMap = macroMap(LR_Pi_file, pos_file, crop = crop, realign = False, posUniform = False, OD_file = OD_file, OPgrid_file = OR_file, OP_file = OP_file)
#mMap = macroMap(LR_Pi_file, uniform_pos_file, posUniform = True, OD_file = OD_file, OPgrid_file = OR_file, OP_file = OP_file)
#mMap = macroMap(LR_Pi_file, uniform_pos_file, posUniform = True, OD_file = OD_file, OPgrid_file = OR_file, OP_file = OP_file, VFxy_file = 'vpos-micro_xy-' + pos_name + '.bin')
"""
fig = plt.figure('macroMap',dpi=1200)
ax1 = fig.add_subplot(111)
#mMap.pODready = False
mMap.plot_map(ax1, None, None, fig.dpi, pltOD = True, pltVF = False, pltOP = True, forceOP = True)
ax1.set_aspect('equal')
fig.savefig(pos_name+'1.png')
"""

if newPos:
    mMap.save(pos_file = pos_name + '_adjusted.bin')

dx = mMap.x[1] - mMap.x[0]
dy = mMap.y[1] - mMap.y[0]
print('#spread uniformly')
#mMap.posUniform = True
if not mMap.posUniform:
    dt0 = np.power(2.0,-np.arange(10,11)).reshape(1,1)
    dt1 = np.power(2.0,-np.arange(10,11)).reshape(1,1)
    dt = np.hstack((np.tile(dt0,(ndt0,1)).flatten(), np.tile(dt1,(ndt1,1)).flatten()))

    mMap.make_pos_uniform_p(dt, p_scale, b_scale, pos_name+'2', ncore = ncore, ndt_decay = ndt0, roi_ratio = roi_ratio, k1 = k1, k2 = k2, damp = damp, l_ratio = l_ratio, l_k = l_k, chop_ratio = chop_ratio, spercent = 0.01, seed = -1, bfile = lrfile, local_plot = False)
    mMap.save(pos_file = uniform_pos_file)

if newOD:
    mMap.save(OD_file = pos_name + '-od_file.bin')

#mMap.assign_pos_OD2(bfile = lrfile)
#mMap.save(OD_file = pos_name + '-od_file.bin')

if newOP:
    mMap.save(OP_file = pos_name + '-op_file.bin')

'''
print('#stretch for vpos')
limit_ratio = 2.0
roi_ratio = 4.0
k1 = 1.0
k2 = 0.5
p_scale = 1.0
b_scale = 0.5
damp = 0.80
l_ratio = 0.0
l_k = 0.0
#dt0 = np.power(2.0,-np.arange(8,9)).reshape(1,1)
#dt1 = np.power(2.0,-np.arange(10,21)).reshape(1,1)
dt0 = np.power(2.0,-np.arange(10,11)).reshape(1,1)
dt1 = np.power(2.0,-np.arange(10,11)).reshape(1,1)
dt = np.hstack((np.tile(dt0,(1,1)).flatten(), np.tile(dt1,(1,ndt)).flatten()))
#dt = np.power(2.0,-np.array([11,10])).reshape(2,1)
#dt = np.tile(dt,(1,ndt)).flatten()
#ndt_decay = 100
# growing period: 
#'''
'''
ndt_decay = 1
fT = False # firstTime
read_lrfile = 'tmpL18-0'
read_vpfile = 'tmpVF_L18-ss'
ct = False 
tmpL = 'tmpL19'
tmpVF_L = 'tmpVF_L19'
mMap.spread_pos_VF(dt, tmpVF_L, tmpL, 'L', p_scale = p_scale, b_scale = b_scale, firstTime = fT, read_lrfile = read_lrfile, read_vpfile= read_vpfile, continuous = ct, ndt_decay = ndt_decay, roi_ratio = roi_ratio, k1 = k1, k2 = k2, damp = damp, l_ratio = l_ratio, l_k = l_k, ncore = ncore, limit_ratio = limit_ratio)
#mMap.spread_pos_VF(dt, tmpVF_L, tmpL, 'L', firstTime = fT, ax = None, ndt_decay = ndt_decay, ncore = ncore)
#'''
'''
ndt_decay = 1
fT = False # firstTime
read_lrfile = 'tmpR8-0'
read_vpfile = 'tmpVF_R8-ss'
ct = False 
tmpR = 'tmpR9'
tmpVF_R = 'tmpVF_R9'
mMap.spread_pos_VF(dt, tmpVF_R, tmpR, 'R', p_scale = p_scale, b_scale = b_scale, firstTime = fT, read_lrfile = read_lrfile, read_vpfile= read_vpfile, continuous = ct, ndt_decay = ndt_decay, roi_ratio = roi_ratio, k1 = k1, k2 = k2, damp = damp, l_ratio = l_ratio, l_k = l_k, ncore = ncore, limit_ratio = limit_ratio)
#'''
'''
mMap.save(VFxy_file = 'vpos-micro_xy-' + pos_name + '.bin')
mMap.save(VFpolar_file = 'vpos-micro_polar-' + pos_name + '.bin')
'''
'''
fig = plt.figure('map', dpi = 2000)
ax = fig.add_subplot(111)
mMap.plot_map(ax)
fig.savefig('map.png', dpi = 2000)
#'''
