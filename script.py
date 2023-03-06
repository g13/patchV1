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

ndt0 = 1 # 50
ndt1 = 1 # 50
ncore = 32 
k1 = 1.0
k2 = 0.5
p_scale = 1.0
b_scale = 0.5
roi_ratio = 4.0
limit_ratio = 1.0
chop_ratio = 0
crop = 0.0
damp = 0.5
l_ratio = 0.5
l_k = 0.0

newPos = False
newOP = False
newOD = False

fdr = '/root/autodl-tmp/public/resource/'
#pos_name = 'low_2Dpos_110mm2-56x1024'
pos_name = 'ld_test'
lrfile = fdr + pos_name + '-bound'

#LR_Pi_file = fdr + 'big_ibeta-2-LR_Pi21.bin'
#OR_file = fdr + 'big_ORcolor-f0022.bin'
LR_Pi_file = fdr + 'ibeta-3-LR_Pi21.bin'
OR_file = fdr + 'ORcolor-f0022.bin'
if newPos:
    pos_file = fdr + pos_name+'.bin'
else:
    pos_file = fdr + pos_name+'_adjusted.bin'

if newOP:
    OP_file = None
else:
    OP_file = fdr + pos_name + '-op_file.bin'

if newOD:
    OD_file = None
else:
    OD_file = fdr + pos_name + '-od_file.bin'

uniform_pos_file = fdr + 'V1_pos-' + pos_name + '.bin'

#mMap = macroMap(LR_Pi_file, pos_file, crop = crop, realign = False, posUniform = False, OPgrid_file = OR_file)
#mMap = macroMap(LR_Pi_file, uniform_pos_file, posUniform = True, OD_file = OD_file, OPgrid_file = OR_file, OP_file = OP_file)
mMap = macroMap(LR_Pi_file, uniform_pos_file, posUniform = True, OD_file = OD_file, OPgrid_file = OR_file, OP_file = OP_file, VFxy_file = fdr + 'vpos-xy-' + pos_name + '.bin')

"""
mMap.pODready = False
fig = plt.figure('macroMap',dpi=1000)
ax1 = fig.add_subplot(111)
mMap.plot_map(ax1, dpi = fig.dpi, pltOD = True, pltVF = False, pltOP = True, forceOP = True, ngridLine = 0)
ax1.set_aspect('equal')
fig.savefig(pos_name+'1.png')
"""

if newPos:
    mMap.save(pos_file = fdr + pos_name + '_adjusted.bin')

dx = mMap.x[1] - mMap.x[0]
dy = mMap.y[1] - mMap.y[0]
print('#spread uniformly')
if not mMap.posUniform:
    dt0 = np.power(2.0,-np.arange(8,9)).reshape(1,1)
    dt1 = np.power(2.0,-np.arange(10,11)).reshape(1,1)
    dt = np.hstack((np.tile(dt0,(ndt0,1)).flatten(), np.tile(dt1,(ndt1,1)).flatten()))

    mMap.make_pos_uniform_p(dt, p_scale, b_scale, pos_name+'2', ncore = ncore, ndt_decay = ndt0, roi_ratio = roi_ratio, k1 = k1, k2 = k2, damp = damp, l_ratio = l_ratio, l_k = l_k, chop_ratio = chop_ratio, spercent = 0.01, seed = -1, bfile = lrfile, local_plot = False)
    mMap.save(pos_file = uniform_pos_file)

if newOD:
    mMap.save(OD_file = fdr + pos_name + '-od_file.bin')

if newOP:
    mMap.save(OP_file = fdr + pos_name + '-op_file.bin')

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
#dt1 = np.power(2.0,-np.arange(8,9)).reshape(1,1)
dt0 = np.power(2.0,-np.arange(10,11)).reshape(1,1)
dt1 = np.power(2.0,-np.arange(10,11)).reshape(1,1)
dt = np.hstack((np.tile(dt0,(ndt0,1)).flatten(), np.tile(dt1,(ndt1,1)).flatten()))
ndt_decay = 100
# growing period: 
#'''

'''
ndt_decay = 1
fT = False # firstTime
ct = False
noSpread = False
i = 31
j = 0
read_lrfile = f'tmpL{i}-{j}'
read_vpfile = f'tmpVF_L{i}-ss'
tmpL = f'tmpL{i+1}'
tmpVF_L = f'tmpVF_L{i+1}'
fig = plt.figure('spread', dpi = 800)
ax = fig.add_subplot(111)
mMap.spread_pos_VF(dt, tmpVF_L, tmpL, 'L', p_scale = p_scale, b_scale = b_scale, firstTime = fT, read_lrfile = read_lrfile, read_vpfile= read_vpfile, continuous = ct, ndt_decay = ndt_decay, roi_ratio = roi_ratio, k1 = k1, k2 = k2, damp = damp, l_ratio = l_ratio, l_k = l_k, ncore = ncore, limit_ratio = limit_ratio, ax = ax, noSpread = noSpread)
#mMap.spread_pos_VF(dt, tmpVF_L, tmpL, 'L', firstTime = fT, ax = None, ndt_decay = ndt_decay, ncore = ncore)
ax.set_aspect('equal')
fig.savefig(f'spread-{tmpL}.png')
#'''

'''
ndt_decay = 1
fT = False # firstTime
ct = False 
noSpread = False
i = 23
j = 0
read_lrfile = f'tmpL{i}-{j}'
read_lrfile = f'tmpR{i}-{j}'
read_vpfile = f'tmpVF_R{i}-ss'
tmpR = f'tmpR{i+1}'
tmpVF_R = f'tmpVF_R{i+1}'
fig = plt.figure('spread', dpi = 800)
ax = fig.add_subplot(111)
mMap.spread_pos_VF(dt, tmpVF_R, tmpR, 'R', p_scale = p_scale, b_scale = b_scale, firstTime = fT, read_lrfile = read_lrfile, read_vpfile= read_vpfile, continuous = ct, ndt_decay = ndt_decay, roi_ratio = roi_ratio, k1 = k1, k2 = k2, damp = damp, l_ratio = l_ratio, l_k = l_k, ncore = ncore, limit_ratio = limit_ratio, ax = ax, noSpread = noSpread)
ax.set_aspect('equal')
fig.savefig(f'spread-{tmpR}.png')
#'''

#'''
#mMap.save(VFxy_file = fdr + 'vpos-xy-' + pos_name + '.bin')
mMap.save(VFpolar_file = fdr + 'V1_vpos-' + pos_name + '.bin')
#mMap.save(Feature_file = fdr + 'V1_feature-' + pos_name + '.bin')
mMap.save(allpos_file = fdr + 'V1_allpos-' + pos_name + '.bin')
#'''
'''
fig = plt.figure('map', dpi = 2000)
ax = fig.add_subplot(111)
mMap.plot_map(ax)
fig.savefig('map.png', dpi = 2000)
#'''
