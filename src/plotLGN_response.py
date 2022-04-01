#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from readPatchOutput import *
from plotV1_response import movingAvg
from os import path
np.seterr(divide='warn', invalid='warn')

import sys
if len(sys.argv) < 6:
    print(sys.argv)
    raise Exception(' need all 6 arguments, no default values available')
else:
    output_suffix = sys.argv[1]
    input_suffix = sys.argv[2]
    data_fdr = sys.argv[3]
    fig_fdr = sys.argv[4]
    if sys.argv[5] == 'True':
        readNewSpike = True 
        print('read new spikes')
    else:
        readNewSpike = False

print(output_suffix)
print(input_suffix)
output_suffix = "_" + output_suffix 
input_suffix = "_" + input_suffix 

data_fdr = data_fdr + '/'
fig_fdr = fig_fdr + '/'

plotResponseSample = True
plotContrastDist = False 
plotStat = True 
nLGN_1D = 8
nt_ = 50000
nstep = 10000
seed = 1653783
#iLGN = np.array([12198, 24358, 1833,2575])
#iLGN = np.array([0,1,nLGN_1D*nLGN_1D+nLGN_1D-1,nLGN_1D*nLGN_1D+nLGN_1D])
ns = 10
FRbins = 25 # per sec
nsmooth = 10 

parameterFn = data_fdr + "patchV1_cfg" +output_suffix + ".bin"

LGN_spFn = data_fdr + "LGN_sp" + output_suffix + ".bin"

prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, mE, mI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, virtual_LGN = read_cfg(parameterFn)
print(f'frRatioLGN = {frRatioLGN}, convolRatio = {convolRatio}')

output = data_fdr + "LGN_gallery" + output_suffix + ".bin"
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
    print(f'parvo:{(nParvo_I, nParvo_C)}, {nType} types, magno: {(nMagno_I, nMagno_C)}, {mType} types')
    nLGN = nParvo + nMagno
    max_convol = np.fromfile(f, prec, nLGN)

output = data_fdr + "LGN" + output_suffix + ".bin"
with open(output, 'rb') as f:
    nLGN = np.fromfile(f, 'u4', 1)[0]
    LGN_type = np.fromfile(f, 'u4', nLGN)
    LGN_polar = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    LGN_ecc = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    LGN_rw = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    LGN_rh = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    LGN_orient = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    LGN_k = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    LGN_ratio = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    tau_R = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    tau_D = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    nR = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    nD = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    delay = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
    spont = np.fromfile(f, prec, nLGN)
    c50 = np.fromfile(f, prec, nLGN)
    sharpness = np.fromfile(f, prec, nLGN)
    coneType = np.fromfile(f, 'u4', 2*nLGN).reshape(2,nLGN)

if 'iLGN' not in locals():
    np.random.seed(seed)
    iLGN = np.random.randint(nLGN, size =ns)
ns = iLGN.size
print(f'sample {ns} LGN\'s id:{iLGN}')

for i in range(ns):
    print(f'{iLGN[i]}: {(LGN_ecc[0,iLGN[i]]*np.cos(LGN_polar[0,iLGN[i]]), LGN_ecc[0,iLGN[i]]*np.sin(LGN_polar[0,iLGN[i]]))}')

output_fn = data_fdr + "outputB4V1" + output_suffix + ".bin"
with open(output_fn, 'rb') as f:
    nt = np.fromfile(f, 'u4', count = 1)[0]
    dt = np.fromfile(f, prec, count = 1)[0]
    if nt_ > nt or nt_ == 0:
        nt_ = nt
        nstep = nt_
    print(f'dt = {dt}')
    print(f'nt_= {nt_}/{nt}')
    if nstep > nt_:
        nstep = nt_
    print(f'nstep = {nstep}')
    interstep = nt_//nstep
    if interstep != nt_/nstep:
        ntstep = nt_
        interstep = nt_//nstep
    tt = np.arange(nstep)*interstep
    t = tt * dt
    t_in_ms = nt_*dt
    t_in_sec = t_in_ms/1000
    print(f'interstep = {interstep}')
    print(t[0], t[-1])
    nLGN = np.fromfile(f, 'u4', count = 1)[0]
    print(f'nLGN = {nLGN}')
    LGNfr  = np.zeros((nstep,ns))
    frStat = np.zeros(nLGN)
    convol  = np.zeros((nstep,ns))
    convol_total = np.zeros((nstep,nLGN))
    luminance  = np.zeros((nstep, ns))
    contrast  = np.zeros((nstep, 2, nLGN))
    for it in range(nstep):
        if it > 0:
            f.seek(nLGN*5*4*(interstep-1), 1)
        data = np.fromfile(f, prec, count = nLGN)
        frStat = frStat + data
        LGNfr[it,:] = data[iLGN]
        data = np.fromfile(f, prec, count = nLGN)
        if virtual_LGN:
            data = data * max_convol
        convol_total[it,:] = data
        convol[it,:] = data[iLGN]
        data = np.fromfile(f, prec, count = nLGN)
        luminance[it,:] = data[iLGN]
        contrast[it,:,:] = np.fromfile(f, prec, count = 2*nLGN).reshape(2,nLGN)

LGN_spScatter = readLGN_sp(LGN_spFn, prec = prec, nstep = nt_)

nbins = int(FRbins*t_in_sec)
realLGN_fr = np.empty((nbins,nLGN))
for i in range(nLGN):
    sp0 = np.array(LGN_spScatter[i])
    sp = sp0[sp0 < t[-1]]
    sp_range = np.linspace(0, nt_, nbins+1)*dt
    counts, _ = np.histogram(sp, bins = sp_range)
    realLGN_fr[:,i] = movingAvg(counts/(1/FRbins), counts.size, nsmooth)

if plotResponseSample:
    fig = plt.figure('LGN', dpi = 600)
    grid = gs.GridSpec(ns, 1, figure = fig, hspace = 0.2)
    for i in range(ns):
        j = iLGN[i]
        ax = fig.add_subplot(grid[i,:])
        ax.plot(t, convol[:,i].T/max_convol[j], 'g', lw = 0.1)
        max_convol_irl = np.max(convol[:,i]*convolRatio)
        print(f'convolRatio = {convolRatio:.3f}')
        print(f'max(convol/max_convol): {max_convol_irl:.3f}/{max_convol[j]:.3f} = {max_convol_irl/max_convol[j]:.3f}')
        ax.plot(t, contrast[:,0,j], ':r', lw = 0.1, label = 'C')
        ax.plot(t, contrast[:,1,j], ':b', lw = 0.1, label = 'S')
        ax.plot(t, luminance[:,i], ':m', lw = 0.1, label = 'lum')
        sp0 = np.array(LGN_spScatter[j])
        print(f'{sp0.size} spikes')
        sp = sp0[sp0 < t[-1]]
        ax.plot(sp, np.zeros(sp.size), '*g', ms = 0.05)

        ax.set_ylim([-1.0,1.0])
        ax2 = ax.twinx()
        ax2.plot(t, LGNfr[:,i], 'k', lw = 0.2)

        nbins = int(FRbins*t_in_sec)
        sp_range = np.linspace(0, nt_, nbins+1)*dt
        counts, _ = np.histogram(sp, bins = sp_range)
        fr = movingAvg(counts/(1/FRbins), counts.size, nsmooth)
        ax2.plot((sp_range[:-1] + sp_range[1:])/2, fr, 'b', lw=0.1)
    
        if i == 0:
            ax.legend()
        ax2.set_ylim(bottom = 0)
        if LGN_k[0,j] > 0:
            on_off = 'on'
        else:
            on_off = 'off'
        ix = np.mod(j, nLGN_1D)
        iy = np.int(np.floor(j/nLGN_1D))
        ax2.set_title(f'#{j} {(ix,iy)}, {LGN_type[j]} ' + on_off + f' fr: {np.mean(LGNfr[:,i]):.3f}/{np.max(LGNfr[:,i]):.3f}Hz, {sp0.size/t[-1]*1000:.3f}Hz')
    fig.savefig(fig_fdr+'lgn-response' + output_suffix + '.png')

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
    fig.savefig(fig_fdr+'lgn-contrast' + output_suffix + '.png')

if plotStat:
    fig = plt.figure('LGN_activity', dpi = 600)
    ax = fig.add_subplot(221)
    max_convol_irl = np.max(convol_total, axis = 0)
    active_ratio = max_convol_irl/max_convol
    ax.set_xlabel('convol/max_convol')
    ax.hist(active_ratio, bins=12)
    ax = fig.add_subplot(222)
    ax.hist(max_convol, bins=12, alpha = 0.5, label='predef')
    maxLGN_fr = np.max(realLGN_fr, axis=0)
    Ron = maxLGN_fr[LGN_type == 0]
    Roff = maxLGN_fr[LGN_type == 1]
    Gon = maxLGN_fr[LGN_type == 2]
    Goff = maxLGN_fr[LGN_type == 3]
    On = maxLGN_fr[LGN_type == 4]
    Off = maxLGN_fr[LGN_type == 5]
    ax.hist((Ron, Roff, Gon, Goff, On, Off), bins = 12, alpha = 0.5)
    ax.legend(fontsize='xx-small')
    ax.set_xlabel('max fr')
    ax = fig.add_subplot(223)
    ax.hist(max_convol*spont, bins=10)
    ax.set_xlabel('spont fr')
    ax = fig.add_subplot(224)
    meanLGN_fr = np.mean(realLGN_fr, axis=0)
    Ron = meanLGN_fr[LGN_type == 0]
    Roff = meanLGN_fr[LGN_type == 1]
    Gon = meanLGN_fr[LGN_type == 2]
    Goff = meanLGN_fr[LGN_type == 3]
    On = meanLGN_fr[LGN_type == 4]
    Off = meanLGN_fr[LGN_type == 5]
    ax.hist((Ron,Roff,Gon,Goff,On,Off), bins=20, label=(f'Ron:{np.mean(Ron):.3f}',f'Roff:{np.mean(Roff):.3f}',f'Gon:{np.mean(Gon):.3f}',f'Goff:{np.mean(Goff):.3f}',f'On:{np.mean(On):.3f}',f'Off:{np.mean(Off):.3f}'))
    ax.legend(fontsize='xx-small')
    ax.set_xlabel('mean fr')
    fig.savefig(fig_fdr+'LGN_activity'+output_suffix+'.png')
