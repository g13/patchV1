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
if len(sys.argv) > 1:
    output_suffix = sys.argv[1]
    if len(sys.argv) > 2:
        input_suffix = sys.argv[2]
    else:
        input_suffix = ""
else:
    input_suffix = ""
    output_suffix = ""

precision = 'f4'
print(output_suffix)
print(input_suffix)
if output_suffix:
    output_suffix = "_" + output_suffix 
if input_suffix:
    input_suffix = "_" + input_suffix 

plotResponseSample = True
plotContrastDist = True
plotStat = True
nLGN_1D = 16
nt_ = 4000
nstep = 1000
#iLGN = np.array([0,137,255])
#iLGN = np.array([6*16+3,7*16+3,8*16+3, 6*16+10,7*16+10,8*16+10])
#iLGN = np.arange(185133)
ns = 10

output = "LGN_gallery" + output_suffix + ".bin"
with open(output, 'rb') as f:
    nParvo = np.fromfile(f, 'u4', 1)[0]
    nType = np.fromfile(f, 'u4', 1)[0]
    nKernelSample = np.fromfile(f, 'u4', 1)[0]
    nSample = np.fromfile(f, 'u4', 1)[0]
    nMagno = np.fromfile(f, 'u4', 1)[0]
    mType = np.fromfile(f, 'u4', 1)[0]
    assert(mType == 1)
    mKernelSample = np.fromfile(f, 'u4', 1)[0]
    mSample = np.fromfile(f, 'u4', 1)[0]
    nParvo_I = np.fromfile(f, 'u4', 1)[0]
    nParvo_C = np.fromfile(f, 'u4', 1)[0]
    nMagno_I = np.fromfile(f, 'u4', 1)[0]
    nMagno_C = np.fromfile(f, 'u4', 1)[0]
    print(f'parvo:{(nParvo_I, nParvo_C)}, magno: {(nMagno_I, nMagno_C)}')
    nLGN = nParvo + nMagno
    max_convol = np.fromfile(f, precision, nLGN)

output = "LGN" + output_suffix + ".bin"
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

output_fn = "outputB4V1" + output_suffix + ".bin"
with open(output_fn, 'rb') as f:
    nt = np.fromfile(f, 'u4', count = 1)[0]
    dt = np.fromfile(f, precision, count = 1)[0]
    if nt_ > nt:
        nt_ = nt
    print(f'nt_= {nt_}/{nt}')
    if nstep > nt_:
        nstep = nt_
    print(f'nstep = {nstep}')
    interstep = nt_//nstep
    tt = np.arange(nstep)*interstep
    nLGN = np.fromfile(f, 'u4', count = 1)[0]
    LGNfr  = np.zeros((nstep,ns))
    frStat = np.zeros(nLGN)
    convol  = np.zeros((nstep,ns))
    convol_total = np.zeros((nstep,nLGN))
    luminance  = np.zeros((nstep, ns))
    contrast  = np.zeros((nstep, 2, nLGN))
    for it in range(nstep):
        if it > 0:
            f.seek(nLGN*5*4*(interstep-1), 1)
        data = np.fromfile(f, precision, count = nLGN)
        frStat = frStat + data
        LGNfr[it,:] = data[iLGN]
        data = np.fromfile(f, precision, count = nLGN)
        convol_total[it,:] = data
        convol[it,:] = data[iLGN]
        data = np.fromfile(f, precision, count = nLGN)
        luminance[it,:] = data[iLGN]
        contrast[it,:,:] = np.fromfile(f, precision, count = 2*nLGN).reshape(2,nLGN)

t = tt + dt
if plotResponseSample:
    fig = plt.figure('LGN', dpi = 600)
    grid = gs.GridSpec(ns, 1, figure = fig, hspace = 0.2)
    for i in range(ns):
        j = iLGN[i]
        ax = fig.add_subplot(grid[i,:])
        ax.plot(t, convol[:,i].T/max_convol[j], 'g', lw = 0.1)
        max_convol_irl = np.max(convol[:,i])
        print([max_convol_irl, max_convol[j], max_convol_irl/max_convol[j]])
        ax.plot(t, contrast[:,0,j], ':r', lw = 0.1)
        ax.plot(t, contrast[:,1,j], ':b', lw = 0.1)
        ax.plot(t, luminance[:,i], ':m', lw = 0.1)
        ax.set_ylim([-1.0,1.0])
        ax2 = ax.twinx()
        ax2.plot(t, LGNfr[:,i], 'k', lw = 0.1)
        ax2.set_ylim(bottom = 0)
        if LGN_k[0,j] > 0:
            on_off = 'on'
        else:
            on_off = 'off'
        ix = np.mod(j, nLGN_1D)
        iy = np.int(np.floor(j/nLGN_1D))
        ax2.set_title(f'LGN #{j} {(ix,iy)}, ' + on_off + f' fr: {np.mean(LGNfr[:,i]):.3f}/{np.max(LGNfr[:,i]):.3f}Hz')
    fig.savefig('lgn-response' + output_suffix + '.png')

if plotContrastDist:
    fig = plt.figure('contrast', dpi = 300)
    pick0 = LGN_k[0,:] > 0
    pick1 = LGN_k[0,:] < 0 
    ax = fig.add_subplot(221)
    ind0 = np.argmax(np.abs(contrast[:,0,pick0]), axis = 0)
    ind1 = np.argmax(np.abs(contrast[:,0,pick1]), axis = 0)
    ax.hist(contrast[ind0,0,pick0], color = 'r', bins = 10, alpha = 0.5)
    ax.hist(contrast[ind1,0,pick1], color = 'b', bins = 10, alpha = 0.5)
    ax = fig.add_subplot(222)
    ind0 = np.argmax(np.abs(contrast[:,1,pick0]), axis = 0)
    ind1 = np.argmax(np.abs(contrast[:,1,pick1]), axis = 0)
    ax.hist(contrast[ind0,1,pick0], color = 'm', bins = 10, alpha = 0.5)
    ax.hist(contrast[ind1,1,pick1], color = 'c', bins = 10, alpha = 0.5)
    ax = fig.add_subplot(212)
    ax.hist(frStat/nstep)
    fig.savefig('lgn-contrast' + output_suffix + '.png')

if plotStat:
    fig = plt.figure('LGN_activity', dpi = 600)
    ax = fig.add_subplot(221)
    max_convol_irl = np.array([np.max(convol_total[:,i]) for i in range(nLGN)])
    active_ratio = max_convol_irl/max_convol
    ax.set_xlabel('convol/max_convol')
    ax.hist(active_ratio, bins=10)
    ax = fig.add_subplot(222)
    ax.hist(max_convol, bins=10, alpha = 0.5, label='predef')
    ax.hist(np.max(LGNfr, axis = 1), bins = 10, alpha = 0.5, label = 'irl' )
    ax.legend()
    ax.set_xlabel('max fr')
    ax = fig.add_subplot(223)
    ax.hist(max_convol*spont, bins=10)
    ax.set_xlabel('spont fr')
    ax = fig.add_subplot(224)
    ax.hist(np.mean(LGNfr, axis = 0), bins=10)
    ax.set_xlabel('mean fr')
    fig.savefig('LGN_activity'+output_suffix+'.png')
