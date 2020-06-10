import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import cm
import sys
from readPatchOutput import *

if len(sys.argv) > 1:
    output_suffix = sys.argv[1]
    print(output_suffix)
    if len(sys.argv) > 2:
        conLGN_suffix = sys.argv[2]
        print(conLGN_suffix)
        if len(sys.argv) > 3:
            readNewSpike = True 
            print('read new spikes')
        else:
            readNewSpike = False
            print('read stored spikes')
    else:
        conLGN_suffix = ""
        readNewSpike = False
else:
    output_suffix = ""
    conLGN_suffix = ""
    readNewSpike = False

if output_suffix:
    output_suffix = "_" + output_suffix
if conLGN_suffix:
    conLGN_suffix = "_" + conLGN_suffix

#sample = np.array([0,1,2,768])
ns = 12
np.random.seed(7329443)
nt_ = 10000
nstep = 5000
step0 = 0
TF = 4
TFbins = 64
FRbins = 25
tbinSize = 1
nsmooth = 5
lw = 0.1

plotSample = True
plotRpStat = True 
plotRpCorr = True 
plotTempMod = True 
plotScatterFF = True
plotFF_sLGN = True
plotLR_rp = True

#plotSample = False
#plotRpStat = False 
#plotRpCorr = False 
#plotTempMod = False 
#plotScatterFF = False
#plotFF_sLGN = False
#plotLR_rp = False 

pSample = True
#pSpike = True
#pVoltage = True
#pCond = True
#pH = True
#pFeature = True
#pLR = True

#pSample = False
pSpike = False
pVoltage = False
pCond = False
pH = False
pFeature = False
pLR = False

mE = 768
mI = 256
blockSize = 1024

if plotRpStat or plotLR_rp or plotRpCorr:
    pSpike = True
    pCond = True

if plotRpCorr:
    plotTempMod = True

if plotFF_sLGN:
    pCond = True

if plotTempMod or plotScatterFF:
    pSpike = True

if plotLR_rp:
    pLR = True
    pFeature = True

if plotSample:
    pSpike = True
    pVoltage = True
    pCond = True

# const
rawDataFn = "rawData" + output_suffix + ".bin"
LGN_frFn = "LGN_fr" + output_suffix + ".bin"
LGN_spFn = "LGN_sp" + output_suffix 

LGN_V1_sFn = "LGN_V1_sList" + conLGN_suffix + ".bin"
LGN_V1_idFn = "LGN_V1_idList" + conLGN_suffix + ".bin"

featureFn = "V1_feature.bin"
LGN_vposFn = "LGN_vpos.bin"

spDataFn = "V1_spikes" + output_suffix

if plotFF_sLGN or plotSample or plotTempMod:
    LGN_V1_s = readLGN_V1_s0(LGN_V1_sFn)
    LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)
    nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type = readLGN_vpos(LGN_vposFn)
    LGN_fr = readLGN_fr(LGN_frFn)
    if readNewSpike:
        LGN_spScatter = readLGN_sp(LGN_spFn + '.bin')
        np.save(LGN_spFn + '.npy', LGN_spScatter)
    else:
        LGN_spScatter = np.load(LGN_spFn + '.npy', allow_pickle=True)
    print('LGN data read')

if plotRpCorr:
    _, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)


with open(rawDataFn, 'rb') as f:
    dt = np.fromfile(f, 'f4', 1)[0] 
    nt = np.fromfile(f, 'u4', 1)[0] 
    if step0 + nt_ >= nt:
        nt_ = nt - step0
    tstep = (nt_ + nstep - 1)//nstep
    nstep = nt_//tstep
    interval = tstep - 1
    print(f'plot {nstep} data points from the {nt_} time steps startingfrom step {step0}')
    nV1 = np.fromfile(f, 'u4', 1)[0] 
    haveH = np.fromfile(f, 'u4', 1)[0] 
    ngFF = np.fromfile(f, 'u4', 1)[0] 
    ngE = np.fromfile(f, 'u4', 1)[0] 
    ngI = np.fromfile(f, 'u4', 1)[0] 

nblock = nV1//blockSize
epick = np.hstack([np.arange(768) + iblock*blockSize for iblock in range(nblock)])
ipick = np.hstack([np.arange(256) + iblock*blockSize + 768 for iblock in range(nblock)])

print(f'nV1 = {nV1}; haveH = {haveH}')
print(f'ngFF: {ngFF}; ngE: {ngE}; ngI: {ngI}')

# readFeature
if pFeature or pLR or plotSample:
    featureType = np.array([0,1])
    feature, rangeFeature, minFeature, maxFeature = readFeature(featureFn, nV1, featureType)
    LR = feature[0,:]

if 'sample' not in locals():
    sample = np.random.randint(nV1, size = ns)
    dOP = np.abs((feature[1,:]-1/2)*np.pi - np.pi*1/4)
    dOP[dOP > np.pi/2] = np.pi - dOP[dOP > np.pi/2]
    pick = epick[nLGN_V1[epick] > 0]
    sample[0] = pick[np.argmin(dOP[pick])]
    sample[1] = pick[np.argmax(dOP[pick])]
    pick = epick[nLGN_V1[epick] == 0]
    sample[2] = pick[np.argmin(dOP[pick])]
    sample[3] = pick[np.argmax(dOP[pick])]
    pick = ipick[nLGN_V1[ipick] > 0]
    sample[4] = pick[np.argmin(dOP[pick])]
    sample[5] = pick[np.argmax(dOP[pick])]
    pick = ipick[nLGN_V1[ipick] == 0]
    sample[6] = pick[np.argmin(dOP[pick])]
    sample[7] = pick[np.argmax(dOP[pick])]
else:
    ns = sample.size
print(f'sampling {sample}')

# read spikes
if pSpike:
    if not readNewSpike:
        spScatter = np.load(spDataFn + '.npy', allow_pickle=True)
    else:
        with open(rawDataFn, 'rb') as f:
            f.seek(4*7, 1)
            spScatter = np.empty(nV1, dtype = object)
            for i in range(nV1):
                spScatter[i] = []
            for it in range(nt):
                data = np.fromfile(f, 'f4', nV1)
                tsps = data[data > 0]
                if tsps.size > 0:
                    idxFired = np.nonzero(data)[0]
                    k = 0
                    for j in idxFired:
                        nsp = np.int(np.floor(tsps[k]))
                        tsp = tsps[k] - nsp
                        if nsp > 1:
                            if 1-tsp > 0.5:
                                dtsp = tsp/nsp
                            else:
                                dtsp = (1-tsp)/nsp
                            tstart = tsp - (nsp//2)*dtsp
                            for isp in range(nsp):
                                spScatter[j].append((it + tstart+isp*dtsp)*dt)
                        else:
                            spScatter[j].append((it+tsp)*dt)
                        k = k + 1
                f.seek((1+(ngE + ngI + ngFF)*(1+haveH))*nV1*4, 1)
        np.save(spDataFn, spScatter)

# read voltage and conductances
if pVoltage or pCond:
    with open(rawDataFn, 'rb') as f:
        f.seek(4*7, 1)
        if pVoltage:
            v = np.empty((nV1, nstep), dtype = 'f4')
        if pCond:
            if pH:
                getH = haveH
            else:
                getH = 0
            gE = np.empty((1+getH, ngE, nV1, nstep), dtype = 'f4')
            gI = np.empty((1+getH, ngI, nV1, nstep), dtype = 'f4')
            gFF = np.empty((1+getH, ngFF, nV1, nstep), dtype = 'f4')

        f.seek((2+(ngE + ngI + ngFF)*(1+haveH))*nV1*4*step0, 1)
        for i in range(nstep):
            if pVoltage:
                f.seek(nV1*4, 1)
                v[:,i] = np.fromfile(f, 'f4', ngFF*nV1).reshape(ngFF,nV1)
            else:
                f.seek(2*nV1*4, 1)

            if pCond:
                gFF[0,:,:,i] = np.fromfile(f, 'f4', ngFF*nV1).reshape(ngFF,nV1)
                if haveH:
                    if pH:
                        gFF[1,:,:,i] = np.fromfile(f, 'f4', ngFF*nV1).reshape(ngFF,nV1)
                    else:
                        f.seek(ngFF*nV1*4, 1)

                gE[0,:,:,i] = np.fromfile(f, 'f4', ngE*nV1).reshape(ngE,nV1)
                gI[0,:,:,i] = np.fromfile(f, 'f4', ngI*nV1).reshape(ngI,nV1)
                if haveH :
                    if pH:
                        gE[1,:,:,i] = np.fromfile(f, 'f4', ngE*nV1).reshape(ngE,nV1)
                        gI[1,:,:,i] = np.fromfile(f, 'f4', ngI*nV1).reshape(ngI,nV1)
                    else:
                        f.seek((ngE+ngI)*nV1*4, 1)
            else:
                f.seek((ngE + ngI + ngFF)*(1+haveH)*nV1*4, 1)
    
            f.seek((2+(ngE + ngI + ngFF)*(1+haveH))*nV1*4*interval, 1)
    print("rawData read")

# temporal modulation
if plotTempMod:
    def get_FreqComp(data, ifreq):
        ndata = len(data)
        Fcomp = np.sum(data * np.exp(-2*np.pi*1j*ifreq*np.arange(ndata)/ndata))/ndata
        return np.array([np.abs(Fcomp)*2, np.angle(Fcomp, deg = True)])
    tTF = 1000/TF
    dtTF = tTF/TFbins
    edges = np.arange(TFbins+1) * dtTF
    t_tf = (edges[:-1] + edges[1:])/2
    n_stacks = int(np.floor(nt_*dt / tTF))
    stacks = np.zeros(TFbins) + n_stacks
    r_stacks = np.mod(nt_*dt, tTF)
    i_stack = np.int(np.floor(r_stacks/dtTF))
    j_stack = np.mod(r_stacks, dtTF)
    stacks[:i_stack] += 1
    stacks[i_stack] += j_stack
    stacks = stacks*dtTF/1000
    
    F0 = np.zeros(nV1)
    F1 = np.zeros((nV1, 2))
    F2 = np.zeros((nV1, 2))
    if pSample:
        sfig = plt.figure(f'sample-TF', dpi = 600)
        grid = gs.GridSpec(ns, 2, figure = sfig, hspace = 0.2)
        j = 0
    for i in range(nV1):
        tsps = np.array(spScatter[i])
        tsps = tsps[np.logical_and(tsps >= step0*dt, tsps < (step0+nt_)*dt)] - step0*dt
        if len(tsps) > 0:
            tsp = np.array([np.mod(x,tTF) for x in tsps])
            nsp = np.histogram(tsp, bins = edges)[0]/stacks
            F0[i] = np.mean(nsp)
            F1[i,:] = get_FreqComp(nsp, 1)
            F2[i,:] = get_FreqComp(nsp, 2)
        else:
            nsp = np.zeros(edges.size-1)
            F0[i] = 0
            F1[i,:] = np.zeros(2)
            F2[i,:] = np.zeros(2)
        if pSample:
            if i in sample:
                amp = np.abs(np.fft.rfft(nsp))/TFbins
                amp[1:] = amp[1:]*2
                ax = sfig.add_subplot(grid[j,0])
                ax.plot(t_tf, nsp)
                ax.set_title(f'sample{i}')
                ax = sfig.add_subplot(grid[j,1])
                ff = np.arange(TFbins//2+1) * TF
                ax.plot(ff, amp)
                ax.set_title(f'F1/F0 = {amp[1]/amp[0]}')
                j = j + 1
    if pSample:
        sfig.savefig('V1-sample_TF' + output_suffix + '.png')
        plt.close(sfig)

    sc_range = np.linspace(0,2,21)
    phase_range = np.linspace(-180, 180, 33)

    fig = plt.figure(f'F1F0-stats', dpi = 600)


    if plotRpCorr:
        F1F0 = F1[:,0]/F0
        F0_0 = F0 > 0
    target = F1[:,0]/F0
    ax = fig.add_subplot(221)
    pick = epick[np.logical_and(nLGN_V1[epick]>0,np.isfinite(target[epick]))]
    ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
    pick = ipick[np.logical_and(nLGN_V1[ipick]>0,np.isfinite(target[ipick]))]
    ax.hist(target[pick], bins = sc_range, color = 'b', alpha = 0.5)
    ax.set_title('simple')
    ax.set_xlabel('F1/F0')

    ax = fig.add_subplot(222)
    pick = epick[np.logical_and(nLGN_V1[epick]==0,np.isfinite(target[epick]))]
    ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
    pick = ipick[np.logical_and(nLGN_V1[ipick]==0,np.isfinite(target[ipick]))]
    ax.hist(target[pick], bins = sc_range, color = 'b', alpha = 0.5)
    ax.set_title('complex')
    ax.set_xlabel('F1/F0')

    target = F1[:,1]
    ax = fig.add_subplot(223)
    pick = epick[np.logical_and(nLGN_V1[epick]>0, F0[epick]>0)]
    ax.hist(target[pick], bins = phase_range, color = 'r', alpha = 0.5)
    pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0[ipick]>0)]
    ax.hist(target[pick], bins = phase_range, color = 'b', alpha = 0.5)
    ax.set_xlabel('F1 phase')
    
    ax = fig.add_subplot(224)
    pick = epick[np.logical_and(nLGN_V1[epick]==0, F0[epick]>0)]
    ax.hist(target[pick], bins = phase_range, color = 'r', alpha = 0.5)
    pick = ipick[np.logical_and(nLGN_V1[ipick]==0, F0[ipick]>0)]
    ax.hist(target[pick], bins = phase_range, color = 'b', alpha = 0.5)
    ax.set_xlabel('F1 phase')
    fig.savefig('V1-F1F0' + output_suffix + '.png')
    plt.close(fig)

    fig = plt.figure(f'F2F0-stats', dpi = 600)

    target = F2[:,0]/F0
    ax = fig.add_subplot(221)
    pick = epick[np.logical_and(nLGN_V1[epick]>0,np.isfinite(target[epick]))]
    ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
    pick = ipick[np.logical_and(nLGN_V1[ipick]>0,np.isfinite(target[ipick]))]
    ax.hist(target[pick], bins = sc_range, color = 'b', alpha = 0.5)
    ax.set_title('simple')
    ax.set_xlabel('F2/F0')

    ax = fig.add_subplot(222)
    pick = epick[np.logical_and(nLGN_V1[epick]==0,np.isfinite(target[epick]))]
    ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
    pick = ipick[np.logical_and(nLGN_V1[ipick]==0,np.isfinite(target[ipick]))]
    ax.hist(target[pick], bins = sc_range, color = 'b', alpha = 0.5)
    ax.set_title('complex')
    ax.set_xlabel('F2/F0')

    target = F2[:,1]
    ax = fig.add_subplot(223)
    pick = epick[np.logical_and(nLGN_V1[epick]>0, F0[epick]>0)]
    ax.hist(target[pick], bins = phase_range, color = 'r', alpha = 0.5)
    pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0[ipick]>0)]
    ax.hist(target[pick], bins = phase_range, color = 'b', alpha = 0.5)
    ax.set_xlabel('F2 phase')
    
    ax = fig.add_subplot(224)
    pick = epick[np.logical_and(nLGN_V1[epick]==0, F0[epick]>0)]
    ax.hist(target[pick], bins = phase_range, color = 'r', alpha = 0.5)
    pick = ipick[np.logical_and(nLGN_V1[ipick]==0, F0[ipick]>0)]
    ax.hist(target[pick], bins = phase_range, color = 'b', alpha = 0.5)
    ax.set_xlabel('F2 phase')
    
    fig.savefig('V1-F2F0' + output_suffix + '.png')
    plt.close(fig)

    def movingAvg(data, n, m, axis = -1):
        avg_data = np.empty(data.shape)
        if np.mod(m,2) == 0:
            m = m + 1
        s = (m-1)//2
        avg_data[:,:,:s] = np.stack([np.mean(data[:,:,:i+s], axis = -1) for i in range(1,s+1)], axis = 2)
        avg_data[:,:,-s:] = np.stack([np.mean(data[:,:,-2*s+i:], axis = -1) for i in range(s)], axis = 2)
        if n >= m:
            avg_data[:,:,s:-s] = np.stack([np.mean(data[:,:,i-s:i+s+1], axis = -1) for i in range(s,n-s)], axis = 2)
        return avg_data

    if pCond:
        dtgTF = tstep*dt
        gTFbins = np.round(tTF/dtgTF)
        if abs(gTFbins - tTF/dtgTF)/gTFbins > 1e-4:
            raise Exception(f'tstep*dt {tstep*dt} and tTF {tTF} is misaligned, gTFbins = {gTFbins}, tTF/dtgTF = {tTF/dtgTF}')
        else:
            gTFbins = int(gTFbins)
        t_gtf = np.arange(gTFbins) * dtgTF
        stacks = np.zeros(gTFbins) + n_stacks
        i_stack = np.int(np.floor(r_stacks/dtgTF))
        j_stack = np.mod(r_stacks, dtgTF)
        stacks[:i_stack] += 1
        stacks[i_stack] += j_stack

        preData = gFF[0,:,:,:].copy()
        data = np.zeros((preData.shape[0], preData.shape[1], gTFbins))
        for i in range(n_stacks-1):
            data = data + preData[:,:,i*gTFbins:(i+1)*gTFbins]
        data[:,:,:i_stack+1] = preData[:,:,(n_stacks-1)*gTFbins:(n_stacks-1)*gTFbins+i_stack+1]
        data = data/stacks

        if nsmooth > 0:
            target = movingAvg(data, gTFbins, nsmooth)
        else:
            target = data
        F0 = np.mean(target, axis = -1).T
        F1 = np.zeros((nV1, ngFF, 2))
        for i in range(nV1):
            for ig in range(ngFF):
                F1[i,ig,:] = get_FreqComp(target[ig,i,:], 1)
        if plotRpCorr:
            gFF_F1F0 = F1[:,:,0]/F0
            gFF_F0_0 = F0 > 0

        sfig = plt.figure('gFF-sample', dpi = 600)
        grid = gs.GridSpec(ns, 2, figure = sfig, hspace = 0.2)
        for i in range(ns):
            if nLGN_V1[sample[i]] > 0:
                ax = sfig.add_subplot(grid[i,0])
                ax.plot(t_gtf, data[0,sample[i],:], '-k', lw = 0.5)
                ax.plot(t_gtf, target[0,sample[i],:], '-g', lw = 0.5)
                ax.set_title(f'F1/F0 = {F1[sample[i],:,0]/F0[sample[i],:]}')
                ax.set_xlabel('time(ms)')
                ax = sfig.add_subplot(grid[i,1])
                amp = np.abs(np.fft.rfft(target[0,sample[i],:]))/gTFbins
                amp[1:] = amp[1:]*2
                ff = np.arange(gTFbins//2+1) * TF
                ipre = np.argwhere(ff<TF)[-1][0]
                ipost = np.argwhere(ff>=TF)[0][0]
                ax.plot(ff, amp, 'r', lw = 0.5, alpha = 0.5)
                ax.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5)
                ax.set_title(f'{sample[i]}')
                ax.set_yscale('log')
                ax.set_xlabel('Hz')
        sfig.savefig('gFF-sample'+output_suffix + '.png')
        plt.close(sfig)

        fig = plt.figure(f'gFF-TFstats', dpi = 600)
        grid = gs.GridSpec(ngFF, 2, figure = fig, hspace = 0.2)
        sc_range = np.linspace(0,2,21)
        for ig in range(ngFF):
            ax = fig.add_subplot(grid[ig,0])
            target = F1[epick,ig,0]/F0[epick,ig]
            target = target[np.isfinite(target)]
            ax.hist(target, bins = sc_range, color = 'r', alpha = 0.5)
            target = F1[ipick,ig,0]/F0[ipick,ig]
            target = target[np.isfinite(target)]
            ax.hist(target, bins = sc_range, color = 'b', alpha = 0.5)

            ax = fig.add_subplot(grid[ig,1])
            phase_range = np.linspace(-180, 180, 33)
            target = F1[epick,ig,1]
            target = target[F1[epick,ig,0] > 0]
            ax.hist(target, bins = phase_range, color = 'r', alpha = 0.5)
            target = F1[ipick,ig,1]
            target = target[F1[ipick,ig,0] > 0]
            ax.hist(target, bins = phase_range, color = 'b', alpha = 0.5)
            ax.set_title('F1 phase')
        fig.savefig('gFF-TFstat' + output_suffix + '.png')
        plt.close(fig)

tpick = step0 + np.arange(nstep)*tstep 
t = tpick*dt
assert(np.sum(epick) + np.sum(ipick) == np.sum(np.arange(nV1)))
trange = nt_*dt/1000
if plotRpStat or plotLR_rp or plotRpCorr:
    fr = np.array([np.asarray(x)[np.logical_and(x>=step0*dt, x<(nt_+step0)*dt)].size for x in spScatter])/trange

# sample
if plotSample:
    for i in range(ns):
        iV1 = sample[i]
        fig = plt.figure(f'V1-sample-{iV1}', dpi = 1200, figsize = [3,nLGN_V1[iV1]+2])
        grid = gs.GridSpec(nLGN_V1[iV1]+2, 2, figure = fig, hspace = 0.2)
        iblock = iV1//blockSize
        ithread = np.mod(iV1, blockSize)
        ax = fig.add_subplot(grid[0,:])
        if pSpike:
            tsp0 = np.array(spScatter[iV1])
            tsp = tsp0[np.logical_and(tsp0>=step0*dt, tsp0<(nt_+step0)*dt)]
            ax.plot(tsp, np.ones(len(tsp)), '*k', ms = 1.0)
        if pVoltage:
            ax.plot(t, v[iV1,:], '-k', lw = lw)
        if pCond:
            ax2 = ax.twinx()
            for ig in range(ngE):
                ax2.plot(t, gE[0,ig,iV1,:], '-r', lw = (ig+1)/ngE * lw)
            for ig in range(ngI):
                ax2.plot(t, gI[0,ig,iV1,:], '-b', lw = (ig+1)/ngI * lw)
            for ig in range(ngFF):
                ax2.plot(t, gFF[0,ig,iV1,:], '-g', lw = (ig+1)/ngFF * lw)
            if pH:
                for ig in range(ngFF):
                    ax2.plot(t, gFF[1,ig,iV1,:], ':g', lw = (ig+1)/ngFF * lw)
                for ig in range(ngE):
                    ax2.plot(t, gE[1,ig,iV1,:], ':r', lw = (ig+1)/ngE * lw)
                for ig in range(ngI):
                    ax2.plot(t, gI[1,ig,iV1,:], ':b', lw = (ig+1)/ngI * lw)
        ax.set_title(f'ID: {(iblock, ithread)}')

        if nLGN_V1[iV1] > 0:
            ax = fig.add_subplot(grid[1,0])
            frTmp = LGN_fr[:, LGN_V1_ID[iV1]]
            LGN_fr_sSum = np.sum(frTmp[tpick,:] * LGN_V1_s[iV1], axis=-1)
            ax.plot(t, LGN_fr_sSum, '-g', lw = 2*lw)

            nbins = int(FRbins*nt_*dt/1000*TF)
            sp_range = np.linspace(step0, step0+nt_, nbins+1)*dt
            spTmp = np.hstack(LGN_spScatter[LGN_V1_ID[iV1]])
            LGN_sp_total = spTmp[np.logical_and(spTmp>=step0*dt, spTmp<(nt_+step0)*dt)]
            counts, _ = np.histogram(LGN_sp_total, bins = sp_range)
            ax.hist((sp_range[:-1] + sp_range[1:])/2, bins = sp_range, color = 'b', weights = counts/(1/TF/FRbins), alpha = 0.5)
            ax.plot(LGN_sp_total, np.zeros(LGN_sp_total.size) + np.max(LGN_fr_sSum)/2, '*g', ms = 1.0)
            ax2 = ax.twinx()
            for ig in range(ngFF):
                ax2.plot(t, gFF[0,ig,iV1,:], ':g', lw = (ig+1)/ngFF * lw)

            ax = fig.add_subplot(grid[1,1])
            markers = ('^r', 'vg', '*g', 'dr', '^k', 'vb')
            iLGN_vpos = LGN_vpos[:, LGN_V1_ID[iV1]]
            iLGN_type = LGN_type[LGN_V1_ID[iV1]]

            for j in range(6):
                pick = iLGN_type == j
                ax.plot(iLGN_vpos[0,pick], iLGN_vpos[1,pick], markers[j], ms = 1.0)
            ax.set_aspect('equal')

        for j in range(nLGN_V1[iV1]):
            ax = fig.add_subplot(grid[2+j,0])
            iLGN_fr = LGN_fr[tpick, LGN_V1_ID[iV1][j]]
            ax.plot(t, iLGN_fr, '-g', lw = 2*lw)
            spTmp = np.array(LGN_spScatter[LGN_V1_ID[iV1][j]])
            iLGN_sp = spTmp[np.logical_and(spTmp>=step0*dt, spTmp<(nt_+step0)*dt)]
            counts, _ = np.histogram(iLGN_sp, bins = sp_range)
            ax.hist((sp_range[:-1] + sp_range[1:])/2, bins = sp_range, color = 'b', weights = counts/(1/TF/FRbins), alpha = 0.5)
            ax.plot(iLGN_sp, np.zeros(iLGN_sp.size)+np.max(iLGN_fr)/2, '*g', ms = 1.0)
            ax = fig.add_subplot(grid[2+j,1])
            for k in range(6):
                pick = iLGN_type == k
                ax.plot(iLGN_vpos[0,pick], iLGN_vpos[1,pick], markers[k], ms = 1.0)
            ax.plot(iLGN_vpos[0,j], iLGN_vpos[1,j], '*k')
            ax.set_aspect('equal')
        fig.savefig(f'V1-sample-{iV1}' + output_suffix + '.png')
        plt.close(fig)

# statistics
if plotRpStat:
    fig = plt.figure(f'rpStat', figsize = (8,4), dpi = 600)
    ax = fig.add_subplot(221)
    ax.hist(fr[epick], color = 'r', alpha = 0.5, label = 'Exc')
    ax.plot(0, sum(fr[epick] == 0), '*r')
    ax.hist(fr[ipick], color = 'b', alpha = 0.5, label = 'Inh')
    ax.plot(0, sum(fr[ipick] == 0), '*b')
    ax.set_title('fr')
    ax.set_xlabel('Hz')
    ax.legend()
    
    target = np.sum(gFF[0,:,:,:], axis = -1)/trange
    ax = fig.add_subplot(222)
    target = np.sum(target, axis = 0)
    ax.hist(target[epick], color = 'r', alpha = 0.5)
    ax.hist(target[ipick], color = 'b', alpha = 0.5)
    ax.set_title('gFF')
    
    target = np.sum(gE[0,:,:,:], axis = -1)/trange
    ax = fig.add_subplot(223)
    target = np.sum(target, axis = 0)
    ax.hist(target[epick], color = 'r', alpha = 0.5)
    #ax.plot(0, sum(target[epick] == 0), '*r')
    ax.hist(target[ipick], color = 'b', alpha = 0.5)
    #ax.plot(0, sum(target[ipick] == 0), '*r')
    ax.set_title('gE')
    
    target = np.sum(gI[0,:,:,:], axis = -1)/trange
    ax = fig.add_subplot(224)
    target = np.sum(target, axis = 0)
    ax.hist(target[epick], color = 'r', alpha = 0.5)
    ax.hist(target[ipick], color = 'b', alpha = 0.5)
    ax.set_title('gI')

    fig.savefig('V1-rpStats' + output_suffix + '.png')
    plt.close(fig)

if plotRpCorr:
    fig = plt.figure(f'rpCorr', figsize = (12,12), dpi = 600)
    grid = gs.GridSpec(4, 4, figure = fig, hspace = 0.3, wspace = 0.3)
    targetE = np.sum(gFF[0,:,:,:], axis = 0) + np.sum(gE[0,:,:,:], axis = 0)
    targetI = np.sum(gI[0,:,:,:], axis = 0)
    target0 = targetE/(targetE+targetI)
    target0[targetE == 0] = 0
    target = np.mean(target0, axis = -1)
    ax = fig.add_subplot(grid[0,0])
    image = HeatMap(target[epick], fr[epick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
    ax.set_xlabel('gExc/gTot')
    ax.set_ylabel('Exc. FR Hz')
    old_target = target.copy()
    
    ax = fig.add_subplot(grid[0,1])
    image = HeatMap(target[ipick], fr[ipick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
    ax.set_xlabel('gExc/gTot')
    ax.set_ylabel('Inh. FR Hz')

    target = F1F0
    ax = fig.add_subplot(grid[0,2])
    pick = epick[np.logical_and(nLGN_V1[epick]>0, F0_0[epick])]
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
    ax.set_xlabel('F1F0')
    ax.set_ylabel('Exc. FR Hz')

    ax = fig.add_subplot(grid[0,3])
    pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0_0[ipick])]
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
    ax.set_xlabel('F1F0)')
    ax.set_ylabel('Inh. FR Hz')

    igFF = 0
    target = gFF_F1F0[:,igFF]
    ax = fig.add_subplot(grid[1,0])
    pick = epick[np.logical_and(nLGN_V1[epick]>0,gFF_F0_0[epick,igFF])]
    active = pick.size/epick.size
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
    ax.set_title(f'active simple {active*100:.3f}%')
    ax.set_xlabel('gFF_F1/F0')
    ax.set_ylabel('Exc FR')

    ax = fig.add_subplot(grid[1,1])
    pick = ipick[np.logical_and(nLGN_V1[ipick]>0,gFF_F0_0[ipick,igFF])]
    active = pick.size/ipick.size
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
    ax.set_xlabel('gFF_F1/F0')
    ax.set_ylabel('Inh FR')
    ax.set_title(f'active simple {active*100:.3f}%')

    target = np.mean(np.sum(gE[0,:,:,:], axis = 0), axis = -1)/trange
    ax = fig.add_subplot(grid[1,2])
    pick = epick[nLGN_V1[epick]==0]
    active = pick.size/epick.size
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
    ax.set_xlabel('gE')
    ax.set_ylabel('C. Exc FR')
    ax.set_title(f'active complex {active*100:.3f}%')

    ax = fig.add_subplot(grid[1,3])
    pick = ipick[nLGN_V1[ipick]==0]
    active = pick.size/ipick.size
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
    ax.set_xlabel('gE')
    ax.set_ylabel('C. Inh FR')
    ax.set_title(f'active complex {active*100:.3f}%')

    target = feature[1,:]
    ax = fig.add_subplot(grid[2,0])
    pick = epick[nLGN_V1[epick]>0]
    active = pick.size/epick.size
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
    ax.set_xlabel('OP')
    ax.set_ylabel('S. Exc FR')

    ax = fig.add_subplot(grid[2,1])
    pick = ipick[nLGN_V1[ipick]>0]
    active = pick.size/ipick.size
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
    ax.set_xlabel('OP')
    ax.set_ylabel('S. Inh FR')

    ax = fig.add_subplot(grid[2,2])
    pick = epick[nLGN_V1[epick]==0]
    active = pick.size/epick.size
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
    ax.set_xlabel('OP')
    ax.set_ylabel('C. Exc FR')

    ax = fig.add_subplot(grid[2,3])
    pick = ipick[nLGN_V1[ipick]==0]
    active = pick.size/ipick.size
    image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
    ax.set_xlabel('OP')
    ax.set_ylabel('C. Inh FR')

    ytarget = gFF_F1F0[:,igFF]
    ax = fig.add_subplot(grid[3,0])
    pick = epick[np.logical_and(nLGN_V1[epick]>0, gFF_F0_0[epick,igFF])]
    image = HeatMap(target[pick], ytarget[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
    ax.set_xlabel('OP')
    ax.set_ylabel('Exc. gFF_F1F0')

    ax = fig.add_subplot(grid[3,1])
    pick = ipick[np.logical_and(nLGN_V1[ipick]>0, gFF_F0_0[ipick,igFF])]
    image = HeatMap(target[pick], ytarget[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
    ax.set_xlabel('OP')
    ax.set_ylabel('Inh. gFF_F1F0')

    ytarget = F1F0
    ax = fig.add_subplot(grid[3,2])
    pick = epick[np.logical_and(nLGN_V1[epick]>0, F0_0[epick])]
    image = HeatMap(target[pick], ytarget[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
    ax.set_xlabel('OP')
    ax.set_ylabel('Exc. F1F0')

    ax = fig.add_subplot(grid[3,3])
    pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0_0[ipick])]
    image = HeatMap(target[pick], ytarget[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
    ax.set_xlabel('OP')
    ax.set_ylabel('Inh. F1F0')

    fig.savefig('V1-rpCorr' + output_suffix + '.png')
    plt.close(fig)

if plotLR_rp:
    fig = plt.figure(f'LRrpStat', dpi = 600)
    ax = fig.add_subplot(221)
    ax.hist(fr[LR>0], color = 'r', alpha = 0.5, label = 'Contra')
    ax.plot(0, sum(fr[LR>0] == 0), '*r')
    ax.hist(fr[LR<0], color = 'b', alpha = 0.5, label = 'Ipsi')
    ax.plot(0, sum(fr[LR<0] == 0), '*b')
    ax.set_title('fr')
    ax.set_xlabel('Hz')
    ax.legend()

    target = np.sum(gFF[0,:,:,:], axis = -1)/trange
    ax = fig.add_subplot(222)
    target = np.sum(target, axis = 0)
    ax.hist(target[LR>0], color = 'r', alpha = 0.5)
    ax.hist(target[LR<0], color = 'b', alpha = 0.5)
    ax.set_title('gFF')
    
    target = np.sum(gE[0,:,:,:], axis = -1)/trange
    ax = fig.add_subplot(223)
    target = np.sum(target, axis = 0)
    ax.hist(target[LR>0], color = 'r', alpha = 0.5)
    ax.hist(target[LR<0], color = 'b', alpha = 0.5)
    ax.set_title('gE')
    
    target = np.sum(gI[0,:,:,:], axis = -1)/trange
    ax = fig.add_subplot(224)
    target = np.sum(target, axis = 0)
    ax.hist(target[LR>0], color = 'r', alpha = 0.5)
    ax.hist(target[LR<0], color = 'b', alpha = 0.5)
    ax.set_title('gI')
    fig.savefig('V1-LRrpStats' + output_suffix + '.png')
    plt.close(fig)

if plotFF_sLGN:
    gTarget = np.sum(gFF[0,:,:,:], axis = -1)/trange
    s_irl = np.sum(gTarget, axis = 0)
    FF_sSum = np.array([np.sum(s) for s in LGN_V1_s])
    fig = plt.figure(f'FF_sLGN', dpi = 600)
    ax = fig.add_subplot(111)
    ax.plot(FF_sSum , s_irl, '*k', ms = 0.1)
    ax.set_xlabel('LGN_V1_sSum')
    ax.set_ylabel('gFF_sum')
    fig.savefig('FF_sLGN' + output_suffix + '.png')
    plt.close(fig)
        
# scatter
if plotScatterFF:
    fig = plt.figure(f'scatterFF', dpi = 600)
    ax = fig.add_subplot(211)
    tsp = np.hstack([x for x in spScatter])
    isp = np.hstack([ix + np.zeros(len(spScatter[ix])) for ix in np.arange(nV1)])
    tpick = np.logical_and(tsp>=step0*dt, tsp<nt_*dt)
    tsp = tsp[tpick]
    isp = isp[tpick].astype(int)

    if pLR:
        eSpick = np.logical_and(np.mod(isp, blockSize) < mE , LR[isp]>0)
        iSpick = np.logical_and(np.mod(isp, blockSize) >= mE, LR[isp]>0)
        ax.plot(tsp[eSpick], isp[eSpick], ',r')
        ax.plot(tsp[iSpick], isp[iSpick], ',b')
        eSpick = np.logical_and(np.mod(isp, blockSize) < mE , LR[isp]<0)
        iSpick = np.logical_and(np.mod(isp, blockSize) >= mE, LR[isp]<0)
        ax.plot(tsp[eSpick], isp[eSpick], ',m')
        ax.plot(tsp[iSpick], isp[iSpick], ',g')
    else:
        eSpick = np.mod(isp, blockSize) < mE
        iSpick = np.mod(isp, blockSize) >= mE
        ax.plot(tsp[eSpick], isp[eSpick], ',r')
        ax.plot(tsp[iSpick], isp[iSpick], ',b')
    ax.set_xlim(step0*dt, nt_*dt)

    ax = fig.add_subplot(212)
    t = (nt_ - step0)*dt #ms
    nbins = t/tbinSize #every 1ms
    edges = step0*dt + np.arange(nbins+1) * tbinSize 
    t_tf = (edges[:-1] + edges[1:])/2
    ff = np.arange(nbins//2+1) * 1000/t
    ipre = np.argwhere(ff<TF)[-1][0]
    ipost = np.argwhere(ff>=TF)[0][0]

    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nblock*mE)
    amp = np.abs(np.fft.rfft(nsp))/nbins
    amp[1:] = amp[1:]*2
    ax.plot(ff, amp, 'r', lw = 0.5, alpha = 0.5)
    ax.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Hz')
    ax2 = ax.twinx()

    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nblock*mI)
    amp = np.abs(np.fft.rfft(nsp))/nbins
    amp[1:] = amp[1:]*2
    ax2.plot(ff, amp, 'b', lw = 0.5, alpha = 0.5)
    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5)
    ax2.set_yscale('log')

    fig.savefig('V1-scatterFF' + output_suffix + '.png')
    plt.close(fig)
