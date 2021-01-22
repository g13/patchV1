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

ndt0 = 10 # 15
ndt1 = 15 # 10
ndt = 100 #25
ncore = 14 
parallel = True
k1 = 1.0
k2 = 0.5
p_scale = 2.0
b_scale = 0.2
roi_ratio = 3.5
limit_ratio = 1.0
chop_ratio = 0
crop = 0.0
#pos_name = 'V1_pos_2D_lowDensity'
newPos = False 
newOP = False 
newOD = False 

pos_name = 'grid_scale3x'
oLR_Pi_file = 'or-ft10-nG4s-lr2-4-LR_Pi10-3x.bin'
oOR_file = 'ORcolor-f0011-3x.bin'

LR_Pi_file = './FullFledged/or-ft10-ext-or-ft10-nG4s-lr2/or-ft10-nG4s-lr2-4-LR_Pi10_cropped.bin'
OR_file = './FullFledged/or-ft10-ext-or-ft10-nG4s-lr2/or-ft10-nG4s-lr2-4/ORcolor-f0011_cropped.bin'

with open(LR_Pi_file,'r') as f:
    a = np.fromfile(f, 'f8', count=1)[0]
    b = np.fromfile(f, 'f8', count=1)[0]
    k = np.fromfile(f, 'f8', count=1)[0]
    ecc = np.fromfile(f, 'f8', count=1)[0]
    nx = np.fromfile(f, 'i4', count=1)[0]
    ny = np.fromfile(f, 'i4', count=1)[0]
    Pi = np.reshape(np.fromfile(f, 'i4', count = nx*ny),(ny,nx))
    x = np.fromfile(f, 'f8', count = nx)
    y = np.fromfile(f, 'f8', count = ny)
    LR = np.reshape(np.fromfile(f, 'f8', count = nx*ny),(ny,nx))
    #print(x)
    #print(y)

    scale = 4
    nblock = 1
    nxs = scale*(nx-1) + 1
    nys = scale*(ny-1) + 1
    blockSize = nxs*nys
    dataDim = 2
    dx = x[1] - x[0]
    x0 = x.copy()
    for i in range(nx-1):
        x[i] += dx/20
    x = np.array([x, x0+dx/4, x0+2.5/4*dx, x0+3/4*dx]).T.flatten()[:-(scale-1)]
    assert(x.size == nxs)
    assert(x[0] > x0[0])
    assert(x[-1] <= x0[-1])
    dy = y[1] - y[0]
    y0 = y.copy()
    for i in range(ny-1):
        #if i < ny//2:
        y[i] += dy/20
        #else:
            #y[i] -= dy/10
    y = np.array([y, y0+dy/4, y0+2.5/4*dy, y0+3/4*dy]).T.flatten()[:-(scale-1)]
    assert(y[0] > y0[0])
    assert(y[-1] <= y0[-1])
    xx, yy = np.meshgrid(x, y)
    pos = np.array([xx.flatten(),yy.flatten()], dtype = 'f8') 
    print(nx, ny)
    print(nxs, nys)
    print(pos.shape)
    #print(x)
    #print(y)

q_file = 'og_pos.bin'
with open(q_file, 'wb') as f:
    np.array([nblock, nx*ny, dataDim]).astype('u4').tofile(f)
    xx0, yy0 = np.meshgrid(x0, y0)
    og_pos = np.array([xx0.flatten(),yy0.flatten()], dtype = 'f8') 
    og_pos.reshape((nblock,dataDim,nx*ny)).astype('f8').tofile(f)

pos_file = 'temp_grid_pos.bin'
with open(pos_file, 'wb') as f:
    np.array([nblock, blockSize, dataDim]).astype('u4').tofile(f)
    pos.reshape((nblock,dataDim,blockSize)).astype('f8').tofile(f)

scaleGrid = macroMap(LR_Pi_file, pos_file, crop = 0, realign = False, posUniform = False, OPgrid_file = OR_file, noAdjust = True)

# order matters
scaleGrid.assign_pos_OD2(bfile = 'temp', force = True)
scaleGrid.adjust_pos(bfile = 'temp')
scaleGrid.save(OD_file = 'temp_od.bin')

with open(oLR_Pi_file,'wb') as f:
    np.array([a, b, k, ecc]).astype('f8').tofile(f)
    np.array([nxs, nys]).astype('i4').tofile(f)
    Pi = scaleGrid.ODlabel.copy()
    Pi[Pi != 0] = 1
    for i in range(nys):
        print(Pi.reshape((nys,nxs))[i,:])
    Pi.astype('i4').tofile(f)
    x = np.array([x0, x0+dx/4, x0+2*dx/4, x0+3*dx/4]).T.flatten()[:-(scale-1)]
    y = np.array([y0, y0+dy/4, y0+2*dy/4, y0+3*dy/4]).T.flatten()[:-(scale-1)]
    x.astype('f8').tofile(f)
    y.astype('f8').tofile(f)
    scaleGrid.ODlabel.astype('f8').tofile(f)
OD = scaleGrid.ODlabel.reshape((nys,nxs))
print('====================OD===================')
for i in range(nys):
    print(OD[i,:]) 

scaleGrid.assign_pos_OP(True)

print([min(scaleGrid.op), max(scaleGrid.op)])
with open(oOR_file,'wb') as f:
    op = scaleGrid.op * 2*np.pi
    OPgrid_x = np.cos(op)
    OPgrid_y = np.sin(op)

    pick = np.logical_not(np.isnan(OPgrid_x))
    print([np.min(OPgrid_x[pick]), np.max(OPgrid_x[pick])])
    print([np.min(OPgrid_y[pick]), np.max(OPgrid_y[pick])])

    OPgrid_x.astype('f8').tofile(f)
    OPgrid_y.astype('f8').tofile(f)
