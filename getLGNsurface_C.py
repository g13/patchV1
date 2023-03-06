from scipy import integrate
import numpy as np
from cmath import *

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex
from patch_geo_func import x_ep, y_ep
from sys import stdout
import sys
import warnings
np.seterr(invalid = 'raise', under = 'ignore', over = 'ignore')
from LGN_surface import *

#posL_file = 'parvo_pos_I5_uniform.bin'
#posR_file = 'parvo_pos_C6_uniform.bin'
#ecc = 2.50 #in deg
fdr = sys.argv[1]
theme = sys.argv[2]
if fdr[-1] != '/': 
    fdr = fdr + '/'
print(fdr)
print(theme)
posL_file = fdr + 'parvo_pos_I5-' + theme + '.bin'
posR_file = fdr + 'parvo_pos_C6-' + theme + '.bin'
pos_file = posR_file
fn = fdr + 'LGN_uniformR-' + theme

with open(pos_file, 'rb') as f:
    _ = np.fromfile(f, 'u4', 1)
    ecc = np.fromfile(f, 'f8', 1)

ndt0 = 50
ndt1 = 50
ncore = 14
k1 = 1.0
k2 = 0.5
p_scale = 1.0
b_scale = 0.15
roi_ratio = 4.0
chop_ratio = 0.05

dt0 = np.power(2.0,-np.arange(11,12)).reshape(1,1)
dt1 = np.power(2.0,-np.arange(11,12)).reshape(1,1)
dt = np.hstack((np.tile(dt0,(ndt0,1)).flatten(), np.tile(dt1,(ndt1,1)).flatten()))

shape_file = 'LGN_shapeC.bin'

nx = 101
ny = 201

x = np.linspace(0, ecc, nx)
dx = x[1]-x[0]
x = np.linspace(-dx, ecc+dx, nx)

y = np.linspace(-ecc, ecc, ny)
dy = y[1]-y[0]
y = np.linspace(-ecc-dy, ecc+dy, ny)

xx, yy = np.meshgrid(x,y)
Pi = np.sqrt(xx*xx + yy*yy) - ecc < 0
Pi[xx<0] = 0
print(Pi.shape)
with open(shape_file, 'wb') as f:
    np.array([nx, ny], dtype = 'u4').tofile(f)
    x.tofile(f)
    y.tofile(f)
    Pi.astype('i4').tofile(f)
print([np.min(x), np.max(x)])
print([np.min(y), np.max(y)])

#pos_file = 'temp_posL3.bin';
LGN_surface = surface(shape_file, pos_file, x[-1])
old_pos = LGN_surface.pos.copy()
parallel_repel_file = fn + '.bin'
LGN_surface.make_pos_uniform(dt, p_scale, b_scale, fn, ncore = ncore, ndt_decay = ndt0, chop_ratio = chop_ratio, roi_ratio = roi_ratio, k1 = k1, k2 = k2, bfile = fn, vpfile = fn )
LGN_surface.save(parallel_file = parallel_repel_file)
fig = plt.figure(fn, figsize = (12,10), dpi = 1000)
ax = fig.add_subplot(121)
for i in range(LGN_surface.nLGN):
    ax.plot([old_pos[0,i], LGN_surface.pos[0,i]], [old_pos[1,i], LGN_surface.pos[1,i]], '-c', lw = 0.1)
LGN_surface.plot_surface(ax)
ax = fig.add_subplot(122)
ax.plot(LGN_surface.pos[0,:], LGN_surface.pos[1,:], ',k')
fig.savefig(fn+'-trace.png', dpi = 1000)
