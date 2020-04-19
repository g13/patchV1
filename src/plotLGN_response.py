#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

import sys
if len(sys.argv) == 1:
    suffix = ""
else:
    suffix = sys.argv[1]

print(suffix)
if suffix:
    suffix = "_" + suffix 

#iLGN = np.array([3])
ns = 5

output = "LGN_gallery" + suffix + ".bin"
with open(output, 'rb') as f:
    nLGN = np.fromfile(f, 'u4', 1)[0]
    nType = np.fromfile(f, 'u4', 1)[0]
    nKernelSample = np.fromfile(f, 'u4', 1)[0]
    nSample = np.fromfile(f, 'u4', 1)[0]
    max_convol = np.fromfile(f, 'f4', nLGN)

output = "LGN" + suffix + ".bin"
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

if 'iLGN' not in locals():
    iLGN = np.random.randint(nLGN, size =ns)
print(iLGN)
ns = iLGN.size
print(ns)

output_fn = "outputB4V1" + suffix + ".bin"
with open(output_fn, 'rb') as f:
    nt = np.fromfile(f, 'u4', count = 1)[0]
    dt = np.fromfile(f, 'f4', count = 1)[0]
    nLGN = np.fromfile(f, 'u4', count = 1)[0]
    LGNfr  = np.zeros((nt,ns), dtype=float)
    convol  = np.zeros((nt,ns), dtype=float)
    luminance  = np.zeros((nt, ns), dtype=float)
    contrast  = np.zeros((nt, 2, ns), dtype=float)
    contrast_t  = np.zeros((nt, 2, nLGN), dtype=float)
    for it in range(nt):
        data = np.fromfile(f, 'f4', count = nLGN)
        LGNfr[it,:] = data[iLGN]
        data = np.fromfile(f, 'f4', count = nLGN)
        convol[it,:] = data[iLGN]
        data = np.fromfile(f, 'f4', count = nLGN)
        luminance[it,:] = data[iLGN]

        contrast_t[it,0,:] = np.fromfile(f, 'f4', count = nLGN)
        contrast[it,0,:] = contrast_t[it,0,iLGN]

        contrast_t[it,1,:] = np.fromfile(f, 'f4', count = nLGN)
        contrast[it,1,:] = contrast_t[it,1,iLGN]

t = np.arange(nt)*dt + dt
fig = plt.figure('LGN', dpi = 600)
grid = gs.GridSpec(ns, 1, figure = fig, hspace = 0.2)
for i in range(ns):
    j = iLGN[i]
    ax = fig.add_subplot(grid[i,:])
    ax.plot(t, LGNfr[:,i], 'k', lw = 0.1)
    ax.set_ylim(bottom = 0)
    ax2 = ax.twinx()
    ax2.plot(t, convol[:,i].T/max_convol[j], 'g', lw = 0.1)
    print([np.max(convol[:,i]), max_convol[j]])
    ax2.plot(t, contrast[:,0,i], ':r', lw = 0.1)
    ax2.plot(t, contrast[:,1,i], ':b', lw = 0.1)
    ax2.plot(t, luminance[:,i], ':m', lw = 0.1)
    ax2.set_ylim([-1,1])
    if LGN_k[0,j] > 0:
        on_off = 'on'
    else:
        on_off = 'off'
    ax.set_title(f'LGN #{j}, ' + on_off)
fig.savefig('lgn-response' + suffix + '.png')

fig = plt.figure('contrast', dpi = 300) 
ax = fig.add_subplot(111)
pick = LGN_k[0,:] > 0
ax.hist(np.max(abs(contrast_t[:,0,pick]), axis=0), range = (0,1), color = 'r', alpha = 0.5)
ax.hist(np.max(abs(contrast_t[:,1,pick]), axis=0), range = (0,1), color = 'b', alpha = 0.5)
pick = LGN_k[0,:] < 0 
ax.hist(np.max(abs(contrast_t[:,0,pick]), axis=0), range = (0,1), color = 'm', alpha = 0.5)
ax.hist(np.max(abs(contrast_t[:,1,pick]), axis=0), range = (0,1), color = 'c', alpha = 0.5)
fig.savefig('lgn-contrast' + suffix + '.png')
