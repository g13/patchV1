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
from LGN_surface import *
from repel_system import *

posL_file = 'parvo_pos_I5_cart.bin'
posR_file = 'parvo_pos_C6_cart.bin'
shape_file = 'LGN_shape.bin'
ecc = 2.5 #in deg

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

pos_file = 'temp_pos.bin'
LGN_surfaceL = surface(shape_file, pos_file, 2.525)
LGN_surfaceL.prep_pos()

fig = plt.figure('vf_pos', dpi = 900)
ax = fig.add_subplot(111)
dt0 = np.power(2.0,-np.arange(7,8)).reshape(1,1)
dt1 = np.power(2.0,-np.arange(7,8)).reshape(1,1)
dt = np.hstack((np.tile(dt0,(100,1)).flatten(), np.tile(dt1,(100,1)).flatten()))
LGN_surfaceL.make_pos_uniform(dt, ax = ax, b_scale = 1.0, p_scale = 2.5)

LGN_surfaceL.save_pos('temp_pos.bin')
