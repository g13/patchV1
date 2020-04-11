#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

output = "LGN_gallery.bin"
with open(output, 'rb') as f:
    nLGN = np.fromfile(f, 'u4', 1)[0]
    nType = np.fromfile(f, 'u4', 1)[0]
    nKernelSample = np.fromfile(f, 'u4', 1)[0]
    nSample = np.fromfile(f, 'u4', 1)[0]
    max_convol = np.fromfile(f, 'f4', nLGN)

output = "LGN.bin"
precision = 'f4'
with open(output, 'rb') as f:
    nLGN = np.fromfile(f, 'u4', 1)[0]
    LGN_type = np.fromfile(f, 'u4', nLGN)
    LGN_polar = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    LGN_ecc = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    LGN_rw = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    LGN_rh = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    LGN_orient = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    LGN_k = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    LGN_ratio = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    tau_R = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    tau_D = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    nR = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    nD = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    delay = np.fromfile(f, precision, 2*nLGN).reshape(2,nLGN)
    spont = np.fromfile(f, precision, nLGN)
    c50 = np.fromfile(f, precision, nLGN)
    sharpness = np.fromfile(f, precision, nLGN)
    coneType = np.fromfile(f, 'u4', 2*nLGN).reshape(2,nLGN)

output_fn = "outputB4V1.bin"
with open(output_fn, 'rb') as f:
    nt = np.fromfile(f, 'u4', count = 1)[0]
    dt = np.fromfile(f, 'f4', count = 1)[0]
    nLGN = np.fromfile(f, 'u4', count = 1)[0]
    LGNfr  = np.zeros((nt, nLGN), dtype=float)
    convol  = np.zeros((nt, nLGN), dtype=float)
    luminance  = np.zeros((nt, nLGN), dtype=float)
    contrast  = np.zeros((nt, 2, nLGN), dtype=float)
    for it in range(nt):
        LGNfr[it,:] = np.fromfile(f, 'f4', count = nLGN)
        convol[it,:] = np.fromfile(f, 'f4', count = nLGN)
        luminance[it,:] = np.fromfile(f, 'f4', count = nLGN)
        contrast[it,0,:] = np.fromfile(f, 'f4', count = nLGN)
        contrast[it,1,:] = np.fromfile(f, 'f4', count = nLGN)

ns = 5
iLGN = np.random.randint(nLGN, size =ns)

nsub = iLGN.size
print(nsub)
t = np.arange(nt)*dt + dt
fig = plt.figure('LGN', dpi = 600)
grid = gs.GridSpec(nsub, 1, figure = fig, hspace = 0.2)
for i in range(nsub):
    j = iLGN[i]
    ax = fig.add_subplot(grid[i,:])
    ax.plot(t, LGNfr[:,j], 'k', lw = 0.1)
    ax.set_ylim(bottom = 0)
    ax2 = ax.twinx()
    ax2.plot(t, convol[:,j].T/max_convol[j], 'g', lw = 0.1)
    print([np.max(convol[:,j]), max_convol[j]])
    ax2.plot(t, contrast[:,0,j], ':r', lw = 0.1)
    ax2.plot(t, contrast[:,1,j], ':b', lw = 0.1)
    ax2.plot(t, luminance[:,j], ':m', lw = 0.1)
    ax2.set_ylim([-1,1])
    if LGN_k[0,j] > 0:
        on_off = 'on'
    else:
        on_off = 'off'
    ax.set_title(f'LGN #{j}, ' + on_off)
fig.savefig('lgn-response.png')
