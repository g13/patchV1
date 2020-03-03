from scipy import integrate
import numpy as np
from cmath import *

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex
from patch_geo_func import x_ep, y_ep
from sys import stdout
import warnings
np.seterr(invalid = 'raise', under = 'ignore', over = 'ignore')
from LGN_surface import *
from repel_system import *

posL_file = 'parvo_pos_I5_uniform.bin'
posR_file = 'parvo_pos_C6_uniform.bin'
shape_file = 'LGN_shape.bin'
ecc = 2.51 #in deg

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
pos_file = posR_file;
LGN_surfaceR = surface(shape_file, pos_file, 2.525)
old_pos = LGN_surfaceR.prep_pos(expand = True)
parallel_repel_file = 'p_repelR.bin'
LGN_surfaceR.save(parallel_file = parallel_repel_file)
fig = plt.figure('preped_pos', dpi = 1000)
ax = fig.add_subplot(111)
LGN_surfaceR.plot_surface(ax)
fig.savefig('preped_posR.png', dpi = 1000)
