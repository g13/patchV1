import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import cm
import sys
from readPatchOutput import *
np.seterr(invalid = 'raise')


prec = 'f4'
#@profile
def plotV1_response(output_suffix, conLGN_suffix, conV1_suffix, readNewSpike):
    if conLGN_suffix:
        conLGN_suffix = "_" + conLGN_suffix
    if conV1_suffix:
        conV1_suffix = "_" + conV1_suffix
    #sample = np.array([0,1,2,768])
    SCsplit = 0
    ns = 20
    np.random.seed(7329443)
    nt_ = 2000
    nstep = 20000
    step0 = 0
    TF = 4 
    stiOri = np.pi/4;
    nOri = 8
    TFbins = 5
    FRbins = 5
    tbinSize = 1
    nsmooth = 0
    lw = 0.1
    
    #plotRpStat = True 
    #plotRpCorr = True 
    plotScatterFF = True
    #plotSample = True
    #plotLGNsCorr = True
    #plotTempMod = True 
    #plotExc_sLGN = True
    #plotLR_rp = True
    
    plotRpStat = False 
    plotRpCorr = False 
    #plotScatterFF = False
    plotSample = False
    plotLGNsCorr = False 
    plotTempMod = False 
    plotExc_sLGN = False
    plotLR_rp = False 
    
    pSample = True
    #pSpike = True
    #pVoltage = True
    #pCond = True
    #pH = True
    #pFeature = True
    #pLR = True
    pW = True
    pDep = True
    pLog = True 
    
    #pSample = False
    pSpike = False
    pVoltage = False
    pCond = False
    pH = False
    pFeature = False
    pLR = False
    #pW = False
    #pDep = False
    #pLog = False
    
    pSingleLGN = False
    pSC = True
    #pSC = False
    
    vE = 14/3.0
    vI = -2/3.0
    vL = 0.0
    gL_E = 0.05
    gL_I = 0.07
    #mE = 768
    mE = 896
    #mI = 256
    mI = 128 
    blockSize = 1024
    
    if prec == 'f8':
        sizeofPrec = 8
    else:
        if prec == 'f4':
            sizeofPrec =4
        else:
            print('not implemented')

    if plotRpStat or plotLR_rp or plotRpCorr:
        pSpike = True
        pCond = True
    
    if plotRpCorr or plotSample:
        plotTempMod = True
    
    if plotExc_sLGN:
        pCond = True
    
    if plotTempMod or plotScatterFF:
        pSpike = True
    
    if plotLR_rp:
        pLR = True
        pFeature = True
    
    if plotSample:
        pVoltage = True
        pSpike = True
        pCond = True
    
    if plotLGNsCorr:
        pSpike = True
        pCond = True
    
    # const
    if output_suffix:
        _output_suffix = "_" + output_suffix
    
    rawDataFn = "rawData" + _output_suffix + ".bin"
    LGN_frFn = "LGN_fr" + _output_suffix + ".bin"
    LGN_spFn = "LGN_sp" + _output_suffix 
    
    LGN_V1_sFn = "LGN_V1_sList" + conLGN_suffix + ".bin"
    LGN_V1_idFn = "LGN_V1_idList" + conLGN_suffix + ".bin"
    
    conStats_Fn = "conStats" + conV1_suffix + ".bin"
    featureFn = "V1_feature.bin"
    LGN_vposFn = "LGN_vpos.bin"
    
    spDataFn = "V1_spikes" + _output_suffix
    
    if output_suffix:
        output_suffix = output_suffix + "-"
    
    if plotExc_sLGN or plotSample or (plotTempMod and pCond):
        if readNewSpike:
            print('reading LGN connections...')
            LGN_V1_s = readLGN_V1_s0(LGN_V1_sFn)
            print('     ID...')
            LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)
            print('     vpos...')
            nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type = readLGN_vpos(LGN_vposFn)
            print('     fr...')
            LGN_fr = readLGN_fr(LGN_frFn)
            print('     spikes...')
            LGN_spScatter = readLGN_sp(LGN_spFn + '.bin')
            np.savez(LGN_spFn + '.npz', LGN_spScatter = LGN_spScatter, LGN_V1_s = LGN_V1_s, LGN_V1_ID = LGN_V1_ID, nLGN_V1 = nLGN_V1, LGN_vpos = LGN_vpos, LGN_type = LGN_type, LGN_fr = LGN_fr)
        else:
            print('loading LGN data...')
            with np.load(LGN_spFn + '.npz', allow_pickle=True) as data:
                nLGN_V1 = data['nLGN_V1']
                if plotExc_sLGN or plotSample:
                    LGN_V1_s = data['LGN_V1_s']
                if plotSample:
                    LGN_V1_ID = data['LGN_V1_ID']
                    LGN_fr = data['LGN_fr']
                    LGN_spScatter = data['LGN_spScatter']
                    LGN_vpos = data['LGN_vpos']
                    LGN_type = data['LGN_type']
        print('complete.')
    
    if plotRpCorr or (plotScatterFF and pSC) or plotRpStat:
        _, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)
    
    
    with open(rawDataFn, 'rb') as f:
        dt = np.fromfile(f, prec, 1)[0] 
        nt = np.fromfile(f, 'u4', 1)[0] 
        if step0 + nt_ >= nt:
            nt_ = nt - step0
        if nstep > nt_:
            nstep = nt_
        tstep = (nt_ + nstep - 1)//nstep
        nstep = nt_//tstep
        interval = tstep - 1
        print(f'plot {nstep} data points from the {nt_} time steps startingfrom step {step0}, total {nt} steps')
        nV1 = np.fromfile(f, 'u4', 1)[0] 
        iModel = np.fromfile(f, 'i4', 1)[0] 
        haveH = np.fromfile(f, 'u4', 1)[0] 
        ngFF = np.fromfile(f, 'u4', 1)[0] 
        ngE = np.fromfile(f, 'u4', 1)[0] 
        ngI = np.fromfile(f, 'u4', 1)[0] 
    if nt_*dt < 1000/TF:
        TF = 1000/(nt_*dt)
    print(f'TF = {TF}')
    
    print(f'using model {iModel}')
    
    if iModel == 0:
        vThres = 1.0
    if iModel == 1:
        vThres = 2.0
    
    nblock = nV1//blockSize
    epick = np.hstack([np.arange(mE) + iblock*blockSize for iblock in range(nblock)])
    ipick = np.hstack([np.arange(mI) + iblock*blockSize + mE for iblock in range(nblock)])
    
    gL = np.zeros(nV1)
    gL[epick] = gL_E
    gL[ipick] = gL_I
    
    print(f'nV1 = {nV1}; haveH = {haveH}')
    print(f'ngFF: {ngFF}; ngE: {ngE}; ngI: {ngI}')
    
    if plotSample or plotLGNsCorr:
        nType, ExcRatio, preN, preNS, _ = read_conStats(conStats_Fn)
    
    # readFeature
    if pFeature or pLR or plotSample or pSample or plotRpCorr:
        featureType = np.array([0,1])
        feature, rangeFeature, minFeature, maxFeature = readFeature(featureFn, nV1, featureType)
        LR = feature[0,:]
        OP = (feature[1,:]-1/2)*np.pi
        dOP = np.abs(OP - stiOri)
        dOP[dOP > np.pi/2] = np.pi - dOP[dOP > np.pi/2]
        dOri = np.pi/(2*nOri)
        eORpick = epick[dOP[epick] <= dOri]
        iORpick = ipick[dOP[ipick] <= dOri]
    
    # time range
    tpick = step0 + np.arange(nstep)*tstep 
    t = tpick*dt
    assert(np.sum(epick) + np.sum(ipick) == np.sum(np.arange(nV1)))
    t_in_ms = nt_*dt
    t_in_sec = t_in_ms/1000
    
    # read spikes
    if pSpike:
        if not readNewSpike:
            print('loading V1 spike...')
            with np.load(spDataFn + '.npz', allow_pickle=True) as data:
                spScatter = data['spScatter']
                fr = data['fr']
        else:
            print('reading V1 spike...')
            negativeSpike = False
            with open(rawDataFn, 'rb') as f:
                f.seek(sizeofPrec+4*7, 1)
                spScatter = np.empty(nV1, dtype = object)
                for i in range(nV1):
                    spScatter[i] = []
                for it in range(nt):
                    data = np.fromfile(f, prec, nV1)
                    if np.sum(data<0) > 0:
                        print(f'{np.arange(nV1)[data<0]} has negative spikes at {data[data<0]} + {it*dt}')
                        negativeSpike = True
                    tsps = data[data > 0]
                    pick = data < 1
                    assert((data[pick] == 0).all())
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
                    if iModel == 0:
                        f.seek((2+(ngE + ngI + ngFF)*(1+haveH))*nV1*sizeofPrec, 1)
                    if iModel == 1:
                        f.seek((3+(ngE + ngI + ngFF)*(1+haveH))*nV1*sizeofPrec, 1)
            if negativeSpike:
                #print('negative spikes exist')
                raise Exception('negative spikes exist')
            fr = np.array([np.asarray(x)[np.logical_and(x>=step0*dt, x<(nt_+step0)*dt)].size for x in spScatter])/t_in_sec
            np.savez(spDataFn, spScatter = spScatter, fr = fr)
        print('V1 spikes acquired')
    
    
    if plotSample or pSample:
        if 'sample' not in locals():
            sample = np.random.randint(nV1, size = ns)
            if False:
                pick = epick[nLGN_V1[epick] > np.mean(nLGN_V1[epick])]
                sample[2] = pick[np.argmin(dOP[pick])]
                sample[3] = pick[np.argmax(dOP[pick])]
                pick = epick[nLGN_V1[epick] == 0]
                sample[0] = pick[np.argmin(dOP[pick])]
                sample[1] = pick[np.argmax(dOP[pick])]
                pick = ipick[nLGN_V1[ipick] > np.mean(nLGN_V1[ipick])]
                sample[6] = pick[np.argmin(dOP[pick])]
                sample[7] = pick[np.argmax(dOP[pick])]
                pick = ipick[nLGN_V1[ipick] == 0]
                sample[4] = pick[np.argmin(dOP[pick])]
                sample[5] = pick[np.argmax(dOP[pick])]
        
            if False:
                pick = epick[nLGN_V1[epick] == 0]
                sample[0] = pick[np.argmin(fr[pick])]
                sample[1] = pick[np.argmax(fr[pick])]
        
                pick = epick[nLGN_V1[epick] > np.mean(nLGN_V1[epick])]
                sample[2] = pick[np.argmin(fr[pick])]
                sample[3] = pick[np.argmax(fr[pick])]
        
                pick = ipick[nLGN_V1[ipick] == 0]
                sample[4] = pick[np.argmin(fr[pick])]
                sample[5] = pick[np.argmax(fr[pick])]
        
                pick = ipick[nLGN_V1[ipick] > np.mean(nLGN_V1[ipick])]
                sample[6] = pick[np.argmin(fr[pick])]
                sample[7] = pick[np.argmax(fr[pick])]
        else:
            ns = sample.size
        print(f'sampling {[(s//blockSize, np.mod(s,blockSize)) for s in sample]}') 
    
    # read voltage and conductances
    if pVoltage or pCond or plotLGNsCorr or plotSample or (plotTempMod and pCond):
        print('reading rawData..')
        gE = np.zeros((ngE,nV1,2))
        gI = np.zeros((ngI,nV1,2))
        gFF = np.zeros((ngFF,nV1,2))
        w = np.zeros((nV1,2))
        v = np.zeros((nV1,2))
        depC = np.zeros((nV1,2))
    
        cE = np.zeros((ngE,nV1,2))
        cI = np.zeros((ngI,nV1,2))
        cFF = np.zeros((ngFF,nV1,2))
    
        s_v = np.zeros(nV1)
        s_gFF = np.zeros(nV1)
        s_gE = np.zeros(nV1)
        s_gI = np.zeros(nV1)
    
        if plotTempMod and pCond:
            tTF = 1000/TF
            _nstep = int(round(tTF/tstep))
            stepsPerBin = _nstep//TFbins
            if stepsPerBin != _nstep/TFbins:
                raise Exception(f'binning for periods of {tTF} can not be divided by sampling steps: {round(tTF/tstep):.0f} x {tstep} ms vs. {TFbins}')
            dtTF = tTF/TFbins
            n_stacks = int(np.floor(nt_*dt / tTF))
            r_stacks = np.mod(nt_*dt, tTF)
            stacks = np.zeros(TFbins) + n_stacks
            i_stack = np.int(np.floor(r_stacks/dtTF))
            j_stack = np.mod(r_stacks, dtTF)
            stacks[:i_stack] += 1
            stacks[i_stack] += j_stack
    
            per_gFF = np.zeros((nV1,TFbins))
            per_gE = np.zeros((nV1,TFbins))
            per_gI = np.zeros((nV1,TFbins))
            per_v = np.zeros((nV1,TFbins))
            tmp_gFF = np.zeros(nV1)
            tmp_gE = np.zeros(nV1)
            tmp_gI = np.zeros(nV1)
            tmp_v = np.zeros(nV1)
    
        if plotLGNsCorr or plotRpCorr:
            gFF_gTot_ratio = np.zeros((ngFF,nV1,2))
            gE_gTot_ratio = np.zeros((ngE,nV1,2))
            gI_gTot_ratio = np.zeros((ngI,nV1,2))
            gEt_gTot_ratio = np.zeros((nV1,2))
            s_gTot = np.zeros(nV1)
            
        with open(rawDataFn, 'rb') as f:
            f.seek(sizeofPrec+4*7, 1)
            if pDep:
                _depC = np.empty((ns, nstep), dtype = prec)
    
            if iModel == 1:
                if pW:
                    _w = np.empty((ns, nstep), dtype = prec)
            if pVoltage:
                _v = np.empty((ns, nstep), dtype = prec)
            if pCond:
                if pH:
                    getH = haveH
                else:
                    getH = 0
                _gE = np.empty((1+getH, ngE, ns, nstep), dtype = prec)
                _gI = np.empty((1+getH, ngI, ns, nstep), dtype = prec)
                _gFF = np.empty((1+getH, ngFF, ns, nstep), dtype = prec)
    
            if iModel == 0:
                f.seek((3+(ngE + ngI + ngFF)*(1+haveH))*nV1*sizeofPrec*step0, 1)
            if iModel == 1:
                f.seek((4+(ngE + ngI + ngFF)*(1+haveH))*nV1*sizeofPrec*step0, 1)
    
            per_it = 0
            per_nt = 0
            for i in range(nstep):
                f.seek(nV1*sizeofPrec, 1)
                data = np.fromfile(f, prec, nV1)
                depC[:,0] = depC[:,0] + data
                depC[:,1] = depC[:,1] + data*data
                _depC[:,i] = data[sample]
    
                if iModel == 1:
                    if pW:
                        data = np.fromfile(f, prec, nV1)
                        w[:,0] = w[:,0] + data
                        w[:,1] = w[:,1] + data*data
                        _w[:,i] = data[sample]
                    else:
                        f.seek(nV1*sizeofPrec, 1)
    
                if pVoltage:
                    s_v = np.fromfile(f, prec, nV1)
                    v[:,0] = v[:,0] + s_v 
                    v[:,1] = v[:,1] + s_v*s_v
                    _v[:,i] = s_v[sample]
    
                    tmp_v = tmp_v + s_v
                    if per_it == 0: 
                        per_v[:,per_nt] = per_v[:,per_nt] + tmp_v/stepsPerBin
                        tmp_v = np.zeros(nV1)
                else:
                    f.seek(nV1*sizeofPrec, 1)
    
                if pCond:
                    s_gFF = np.fromfile(f, prec, ngFF*nV1).reshape(ngFF,nV1)
                    gFF[:,:,0] = gFF[:,:,0] + s_gFF
                    gFF[:,:,1] = gFF[:,:,1] + s_gFF*s_gFF
                    _gFF[0,:,:,i] = s_gFF[:,sample]
                    x = s_gFF*(vE - s_v)
                    cFF[:,:,0] = cFF[:,:,0] + x
                    cFF[:,:,1] = cFF[:,:,1] + x*x
    
                    if haveH:
                        if pH:
                            data = np.fromfile(f, prec, ngFF*nV1).reshape(ngFF,nV1)
                            _gFF[1,:,:,i] = data[:,sample]
                        else:
                            f.seek(ngFF*nV1*sizeofPrec, 1)
    
                    s_gE = np.fromfile(f, prec, ngE*nV1).reshape(ngE,nV1)
                    gE[:,:,0] = gE[:,:,0] + s_gE
                    gE[:,:,1] = gE[:,:,1] + s_gE*s_gE
                    _gE[0,:,:,i] = s_gE[:,sample]
                    x = s_gE*(vE - s_v)
                    cE[:,:,0] = cE[:,:,0] + x
                    cE[:,:,1] = cE[:,:,1] + x*x
    
                    s_gI = np.fromfile(f, prec, ngI*nV1).reshape(ngI,nV1)
                    gI[:,:,0] = gI[:,:,0] + s_gI
                    gI[:,:,1] = gI[:,:,1] + s_gI*s_gI
                    _gI[0,:,:,i] = s_gI[:,sample]
                    x = s_gI*(vI - s_v)
                    cI[:,:,0] = cI[:,:,0] + x
                    cI[:,:,1] = cI[:,:,1] + x*x
    
                    if plotTempMod and pCond:
                        tmp_gFF = tmp_gFF + np.sum(s_gFF, axis = 0)
                        tmp_gE = tmp_gE + np.sum(s_gE, axis = 0)
                        tmp_gI = tmp_gI + np.sum(s_gI, axis = 0)
                        if per_it == 0: 
                            per_gFF[:,per_nt] = per_gFF[:,per_nt] + tmp_gFF/stepsPerBin
                            per_gE[:,per_nt] = per_gE[:,per_nt] + tmp_gE/stepsPerBin
                            per_gI[:,per_nt] = per_gI[:,per_nt] + tmp_gI/stepsPerBin
                            tmp_gFF = np.zeros(nV1)
                            tmp_gE = np.zeros(nV1)
                            tmp_gI = np.zeros(nV1)
    
                    if plotLGNsCorr or plotRpCorr:
                        s_gTot = np.sum(s_gE, axis = 0) + np.sum(s_gI, axis = 0) + np.sum(s_gFF, axis = 0) + gL
                        x = s_gFF/s_gTot
                        gFF_gTot_ratio[:,:,0] = gFF_gTot_ratio[:,:,0] + x
                        gFF_gTot_ratio[:,:,1] = gFF_gTot_ratio[:,:,1] + x*x
    
                        x = s_gE/s_gTot
                        gE_gTot_ratio[:,:,0] = gE_gTot_ratio[:,:,0] + x
                        gE_gTot_ratio[:,:,1] = gE_gTot_ratio[:,:,1] + x*x
    
                        x = s_gI/s_gTot
                        gI_gTot_ratio[:,:,0] = gI_gTot_ratio[:,:,0] + x
                        gI_gTot_ratio[:,:,1] = gI_gTot_ratio[:,:,1] + x*x
                        
                        x = (np.sum(s_gE,axis=0)+np.sum(s_gFF,axis=0))/s_gTot
                        gEt_gTot_ratio[:,0] = gEt_gTot_ratio[:,0] + x
                        gEt_gTot_ratio[:,1] = gEt_gTot_ratio[:,1] + x*x
    
                    if haveH :
                        if pH:
                            data = np.fromfile(f, prec, ngE*nV1).reshape(ngE,nV1)
                            _gE[1,:,:,i] = data[:,sample]
                            data = np.fromfile(f, prec, ngI*nV1).reshape(ngI,nV1)
                            _gI[1,:,:,i] = data[:,sample]
                        else:
                            f.seek((ngE+ngI)*nV1*sizeofPrec, 1)
    
                else:
                    f.seek((ngE + ngI + ngFF)*(1+haveH)*nV1*sizeofPrec, 1)
        
                if iModel == 0:
                    f.seek((3+(ngE + ngI + ngFF)*(1+haveH))*nV1*sizeofPrec*interval, 1)
                if iModel == 1:
                    f.seek((4+(ngE + ngI + ngFF)*(1+haveH))*nV1*sizeofPrec*interval, 1)

                per_it = np.mod(per_it+1, stepsPerBin)
                if per_it == 0:
                    per_nt = np.mod(per_nt + 1, TFbins)
    
        def getMeanStd(arr, n):
            if len(arr.shape) == 3:
                arr[:,:,0] /= n
                x = arr[:,:,1]/nstep - arr[:,:,0]*arr[:,:,0]
                x[x<0] = 0
                arr[:,:,1] = np.sqrt(x)
            else:
                if len(arr.shape) == 2:
                    arr[:,0] /= n
                    x = arr[:,1]/nstep - arr[:,0]*arr[:,0]
                    x[x<0] = 0
                    arr[:,1] = np.sqrt(x)
                else:
                    raise Exception('dimension of arr need to be 2 or 3')
            

        getMeanStd(gFF,nstep)
        getMeanStd(gE,nstep)
        getMeanStd(gI,nstep)
        getMeanStd(w,nstep)
        getMeanStd(v,nstep)
        getMeanStd(depC,nstep)
        getMeanStd(cFF,nstep)
        getMeanStd(cE,nstep)
        getMeanStd(cI,nstep)
    
        if plotLGNsCorr or plotRpCorr:
            getMeanStd(gFF_gTot_ratio,nstep)
            getMeanStd(gE_gTot_ratio,nstep)
            getMeanStd(gI_gTot_ratio,nstep)
            getMeanStd(gEt_gTot_ratio,nstep)
    
        if plotTempMod and pCond:
            per_v = per_v/stacks
            per_gE = per_gE/stacks
            per_gI = per_gI/stacks
            per_gFF = per_gFF/stacks
        print("rawData read")
    
    # temporal modulation
    if plotTempMod:
        def get_FreqComp(data, ifreq):
            if ifreq == 0:
                raise Exception('just use mean for zero comp')
            ndata = len(data)
            Fcomp = np.sum(data * np.exp(-2*np.pi*1j*ifreq*np.arange(ndata)/ndata))/ndata
            return np.array([np.abs(Fcomp)*2, np.angle(Fcomp, deg = True)])

        def movingAvg(data, n, m, axis = -1):
            avg_data = np.empty(data.shape)
            if np.mod(m,2) == 0:
                m = m + 1
            s = (m-1)//2
            if len(data.shape) == 1:
                avg_data[:s] = [np.mean(data[:i+s]) for i in range(1,s+1)]
                avg_data[-s:] = [np.mean(data[-2*s+i:]) for i in range(s)]
                if n >= m:
                    avg_data[s:-s] = [np.mean(data[i-s:i+s+1]) for i in range(s,n-s)]

            if len(data.shape) == 2:
                avg_data[:,:s] = np.stack([np.mean(data[:,:i+s], axis = -1) for i in range(1,s+1)], axis = 1)
                avg_data[:,-s:] = np.stack([np.mean(data[:,-2*s+i:], axis = -1) for i in range(s)], axis = 1)
                if n >= m:
                    avg_data[:,s:-s] = np.stack([np.mean(data[:,i-s:i+s+1], axis = -1) for i in range(s,n-s)], axis = 1)

            return avg_data
    
        edges = np.arange(TFbins+1) * dtTF
        t_tf = (edges[:-1] + edges[1:])/2
        
        F0 = np.zeros(nV1)
        F1 = np.zeros((nV1, 2))
        F2 = np.zeros((nV1, 2))
        if pSample:
            sfig = plt.figure(f'sample-TF', dpi = 600, figsize = [5.0, ns])
            grid = gs.GridSpec(ns, 2, figure = sfig, hspace = 0.2)
            j = 0
        for i in range(nV1):
            tsps = np.array(spScatter[i])
            tsps = tsps[np.logical_and(tsps >= step0*dt, tsps < (step0+nt_)*dt)] - step0*dt
            if len(tsps) > 0:
                tsp = np.array([np.mod(x,tTF) for x in tsps])
                nsp = np.histogram(tsp, bins = edges)[0]/(stacks * dtTF/1000)
                if nsmooth > 1:
                    smoothed_fr = movingAvg(nsp, TFbins, nsmooth)
                else:
                    smoothed_fr = nsp
                F0[i] = np.mean(smoothed_fr)
                F1[i,:] = get_FreqComp(smoothed_fr, 1)
                F2[i,:] = get_FreqComp(smoothed_fr, 2)
                if pSample and i in sample:
                    amp = np.abs(np.fft.rfft(nsp))/TFbins
                    amp[1:] = amp[1:]*2
                    ax = sfig.add_subplot(grid[j,0])
                    ax.plot(t_tf, nsp, '-k', lw = 0.5)
                    ax.set_ylim(bottom = 0)
                    if nsmooth > 1:
                        ax.plot(t_tf, smoothed_fr, '-g', lw = 0.5)
                    ax.set_title(f'FR {(i//blockSize,np.mod(i, blockSize))}')
                    ax = sfig.add_subplot(grid[j,1])
                    ff = np.arange(TFbins//2+1) * TF
                    ax.plot(ff, amp, 'g', lw = 0.5, alpha = 0.5)
                    if amp[0] > 0:
                        f1f0 = amp[1]/amp[0]
                        #assert(np.abs(f1f0 - F1[i,0]/F0[i])/f1f0 < 1e-8)
                    else:
                        f1f0 = 0
                    ax.set_title(f'F1/F0 = {f1f0:.3f}')
                    j = j + 1
            else:
                nsp = np.zeros(edges.size-1)
                smoothed_fr = nsp
                F0[i] = 0
                F1[i,:] = np.zeros(2)
                F2[i,:] = np.zeros(2)
        if pSample:
            sfig.savefig(output_suffix + 'V1-sample_TF' + '.png')
            plt.close(sfig)
    
        sc_range = np.linspace(0,2,21)
        phase_range = np.linspace(-180, 180, 33)
    
        fig = plt.figure(f'F1F0-stats', dpi = 600)
    
        F1F0 = np.zeros(F1.shape[0])
        F0_0 = F0 > 0
        F1F0[F0_0] = F1[F0_0,0]/F0[F0_0]
    
        target = F1F0
        ax = fig.add_subplot(221)
        pick = epick[np.logical_and(nLGN_V1[epick]>0, F0[epick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
        pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0[ipick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'b', alpha = 0.5)
    
        ax.set_title('simple')
        ax.set_xlabel('F1/F0')
    
        ax = fig.add_subplot(222)
        pick = epick[np.logical_and(nLGN_V1[epick]==0, F0[epick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
        pick = ipick[np.logical_and(nLGN_V1[ipick]==0, F0[ipick] > 0)]
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
        fig.savefig(output_suffix + 'V1-F1F0-stats' + '.png')
        plt.close(fig)
    
        fig = plt.figure(f'V1_OP-F1F0', dpi = 600)
        target = F1F0
        ax = fig.add_subplot(221)
    
        pick = eORpick[np.logical_and(nLGN_V1[eORpick] > 0, F0[eORpick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
        pick = iORpick[np.logical_and(nLGN_V1[iORpick] > 0, F0[iORpick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'b', alpha = 0.5)
    
        ax.set_title('simple')
        ax.set_xlabel('F1/F0')
    
        ax = fig.add_subplot(222)
        pick = eORpick[np.logical_and(nLGN_V1[eORpick] == 0, F0[eORpick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
        pick = iORpick[np.logical_and(nLGN_V1[iORpick] == 0, F0[iORpick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'b', alpha = 0.5)
        ax.set_title('complex')
        ax.set_xlabel('F1/F0')
    
        target = F1[:,1]
        ax = fig.add_subplot(223)
        pick = eORpick[np.logical_and(nLGN_V1[eORpick] > 0, F0[eORpick] > 0)]
        ax.hist(target[pick], bins = phase_range, color = 'r', alpha = 0.5)
        pick = iORpick[np.logical_and(nLGN_V1[iORpick] > 0, F0[iORpick] > 0)]
        ax.hist(target[pick], bins = phase_range, color = 'b', alpha = 0.5)
        ax.set_xlabel('F1 phase')
        
        ax = fig.add_subplot(224)
        pick = eORpick[np.logical_and(nLGN_V1[eORpick] == 0, F0[eORpick] > 0)]
        ax.hist(target[pick], bins = phase_range, color = 'r', alpha = 0.5)
        pick = iORpick[np.logical_and(nLGN_V1[iORpick] == 0, F0[iORpick] > 0)]
        ax.hist(target[pick], bins = phase_range, color = 'b', alpha = 0.5)
        ax.set_xlabel('F1 phase')
        fig.savefig(output_suffix + 'V1_OP-F1F0' + '.png')
        plt.close(fig)
    
        fig = plt.figure(f'F2F0-stats', dpi = 600)
    
        target = np.zeros(F2.shape[0])
        target[F0_0] = F2[F0_0,0]/F0[F0_0]
        ax = fig.add_subplot(221)
        pick = epick[np.logical_and(nLGN_V1[epick]>0, F0[epick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
        pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0[ipick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'b', alpha = 0.5)
        ax.set_title('simple')
        ax.set_xlabel('F2/F0')
    
        ax = fig.add_subplot(222)
        pick = epick[np.logical_and(nLGN_V1[epick]==0, F0[epick] > 0)]
        ax.hist(target[pick], bins = sc_range, color = 'r', alpha = 0.5)
        pick = ipick[np.logical_and(nLGN_V1[ipick]==0, F0[ipick] > 0)]
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
        
        fig.savefig(output_suffix + 'V1-F2F0' + '.png')
        plt.close(fig)
    
        if pCond:
            #dtgTF = tstep*dt
            #gTFbins = np.round(tTF/dtgTF)
            #if abs(gTFbins - tTF/dtgTF)/gTFbins > 1e-4:
            #    raise Exception(f'tstep*dt {tstep*dt} and tTF {tTF} is misaligned, gTFbins = {gTFbins}, tTF/dtgTF = {tTF/dtgTF}')
            #else:
            #    gTFbins = int(gTFbins)
            #t_gtf = np.arange(gTFbins) * dtgTF
            dtgTF = dtTF
            t_gtf = t_tf
            gTFbins = TFbins
    
            #stacks = np.zeros(gTFbins) + n_stacks
            #i_stack = np.int(np.floor(r_stacks/dtgTF))
            #j_stack = np.mod(r_stacks, dtgTF)
            #stacks[:i_stack] += 1
            #stacks[i_stack] += j_stack
    
            data = per_gFF
    
            if nsmooth > 1:
                target = movingAvg(data, gTFbins, nsmooth)
            else:
                target = data
            gFF_F0 = np.mean(target, axis = -1).T
            gFF_F1 = np.zeros((nV1, 2))
            for i in range(nV1):
                gFF_F1[i,:] = get_FreqComp(target[i,:], 1)
    
            gFF_F0_0 = gFF_F0 > 0
            gFF_F1F0 = np.zeros(nV1)
            tF1 = gFF_F1[:,0]
            gFF_F1F0[gFF_F0_0] = tF1[gFF_F0_0]/gFF_F0[gFF_F0_0]
    
            if pSample:
                sfig = plt.figure(f'gFF_TF-sample', dpi = 600, figsize = [5.0, ns])
                grid = gs.GridSpec(ns, 2, figure = sfig, hspace = 0.2)
                for i in range(ns):
                    if nLGN_V1[sample[i]] > 0:
                        ax = sfig.add_subplot(grid[i,0])
                        ax.plot(t_gtf, data[sample[i],:], '-k', lw = 0.5)
                        if nsmooth > 1:
                            ax.plot(t_gtf, target[sample[i],:], '-g', lw = 0.5)
                        ax.set_ylim(bottom = 0)
                        ax.set_title(f'F1/F0 = {gFF_F1F0[sample[i]]:.2f}')
                        ax.set_xlabel('time(ms)')
                        ax = sfig.add_subplot(grid[i,1])
                        amp = np.abs(np.fft.rfft(target[sample[i],:]))/gTFbins
                        amp[1:] = amp[1:]*2
                        ff = np.arange(gTFbins//2+1) * TF
                        ipre = np.argwhere(ff<TF)[-1][0]
                        ipost = np.argwhere(ff>=TF)[0][0]
                        ax.plot(ff, amp, 'g', lw = 0.5, alpha = 0.5)
                        ax.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5)
                        ax.set_title(f'gFF {(sample[i]//blockSize,np.mod(sample[i], blockSize))}')
                        ax.set_yscale('log')
                        ax.set_xlabel('Hz')
                sfig.savefig(output_suffix + 'gFF_TF-sample'+'.png')
                plt.close(sfig)
    
            fig = plt.figure(f'gFF-TFstats', dpi = 600)
            grid = gs.GridSpec(1, 2, figure = fig, hspace = 0.2)
            sc_range = np.linspace(0,2,21)
    
            ax = fig.add_subplot(grid[0,0])
            eSpick = epick[np.logical_and(nLGN_V1[epick] > SCsplit, F0[epick] > 0)]
            iSpick = ipick[np.logical_and(nLGN_V1[ipick] > SCsplit, F0[ipick] > 0)]
            e_active = F0[eSpick]
            i_active = F0[iSpick]
    
            colors = ['r', 'b']
            labels = [f'E: {e_active.size/epick.size*100:.3f}%', f'I: {i_active.size/ipick.size*100:.3f}%']
            data = [gFF_F1F0[eSpick], gFF_F1F0[iSpick]]
            ax.hist(data, bins = sc_range, color = colors, alpha = 0.5, label = labels)
            ax.legend()
            ax.set_xlabel('F1/F0')
    
            ax = fig.add_subplot(grid[0,1])
            phase_range = np.linspace(-180, 180, 33)
            data = [gFF_F1[eSpick,1], gFF_F1[iSpick,1]]
            ax.hist(data, bins = phase_range, color = colors, alpha = 0.5)
            ax.set_title('F1 phase')
    
            fig.savefig(output_suffix + 'gFF-TFstat' + '.png')
            plt.close(fig)
    
            fig = plt.figure(f'OPgFF-TFstats', dpi = 600)
            grid = gs.GridSpec(1, 2, figure = fig, hspace = 0.2)
            sc_range = np.linspace(0,2,21)
    
            ax = fig.add_subplot(grid[0,0])
    
            eSpick = eORpick[np.logical_and(nLGN_V1[eORpick] > SCsplit, F0[eORpick] > 0)]
            iSpick = iORpick[np.logical_and(nLGN_V1[iORpick] > SCsplit, F0[iORpick] > 0)]
            e_active = F0[eORpick]
            i_active = F0[iORpick]
    
            colors = ['r', 'b']
            labels = [f'E: {e_active.size/eORpick.size*100:.3f}%', f'I: {i_active.size/iORpick.size*100:.3f}%']
            data = [gFF_F1F0[eSpick], gFF_F1F0[iSpick]]
            ax.hist(data, bins = sc_range, color = colors, alpha = 0.5, label = labels)
            ax.legend()
            ax.set_xlabel('F1/F0')
    
            ax = fig.add_subplot(grid[0,1])
            phase_range = np.linspace(-180, 180, 33)
            data = [gFF_F1[eSpick,1], gFF_F1[iSpick,1]]
            ax.hist(data, bins = phase_range, color = colors, alpha = 0.5)
            ax.set_title('F1 phase')
    
            fig.savefig(output_suffix + 'OPgFF-TFstat' + '.png')
            plt.close(fig)
    
    # sample
    
    if plotSample:
        for i in range(ns):
            iV1 = sample[i]
            if not pSingleLGN:
                fig = plt.figure(f'V1-sample-{iV1}', dpi = 1200, figsize = [5.0, 3])
                grid = gs.GridSpec(3, 2, figure = fig, hspace = 0.2, wspace = 0.3)
            else:
                fig = plt.figure(f'V1-sample-{iV1}', dpi = 1200, figsize = [5.0,nLGN_V1[iV1]+3])
                grid = gs.GridSpec(nLGN_V1[iV1]+3, 2, figure = fig, hspace = 0.2, wspace = 0.3)
            iblock = iV1//blockSize
            ithread = np.mod(iV1, blockSize)
            ax = fig.add_subplot(grid[0,:])
            #if pSpike:
            tsp0 = np.array(spScatter[iV1])
            tsp = tsp0[np.logical_and(tsp0>=step0*dt, tsp0<(nt_+step0)*dt)]
            ax.plot(tsp, np.zeros(len(tsp))+vThres, '*k', ms = 1.0)
            #if pVoltage:
            ax.plot(t, _v[i,:], '-k', lw = lw)
            ax.plot(t, np.ones(t.shape), ':k', lw = lw)
            #if pCond:
            ax2 = ax.twinx()
            for ig in range(ngFF):
                ax2.plot(t, _gFF[0,ig,i,:], '-g', lw = (ig+1)/ngFF * lw)
            for ig in range(ngE):
                ax2.plot(t, _gE[0,ig,i,:], '-r', lw = (ig+1)/ngE * lw)
            for ig in range(ngI):
                ax2.plot(t, _gI[0,ig,i,:], '-b', lw = (ig+1)/ngI * lw)
            if pH:
                for ig in range(ngFF):
                    ax2.plot(t, _gFF[1,ig,i,:], ':g', lw = (ig+1)/ngFF * lw)
                for ig in range(ngE):
                    ax2.plot(t, _gE[1,ig,i,:], ':r', lw = (ig+1)/ngE * lw)
                for ig in range(ngI):
                    ax2.plot(t, _gI[1,ig,i,:], ':b', lw = (ig+1)/ngI * lw)
    
            ax.set_title(f'ID: {(iblock, ithread)}:({LR[iV1]:.0f},{OP[iV1]*180/np.pi:.0f})- LGN:{nLGN_V1[iV1]}({np.sum(LGN_V1_s[iV1]):.1f}), E{preN[0,iV1]}({preNS[0,iV1]:.1f}), I{preN[1,iV1]}({preNS[1,iV1]:.1f})')
            ax.set_ylim(bottom = min(0,np.min(_v[i,:])))
            ax.set_ylim(top = vThres*1.1)
            ax.set_yticks(np.linspace(0,1,11))
            ax2.set_ylim(bottom = 0)
    
            ax = fig.add_subplot(grid[1,:])
            current = np.zeros(_v[i,:].shape)
            if ithread < mE:
                igL = gL_E
            else:
                igL = gL_I
            cL = -igL*(_v[i,:]-vL)
            ax.plot(t, cL, '-c', lw = lw)
            ax.plot(t[-1], np.mean(cL), '*c', ms = lw)
            current = current + cL
            for ig in range(ngFF):
                cFF = -_gFF[0,ig,i,:]*(_v[i,:]-vE)
                ax.plot(t, cFF, '-g', lw = (ig+1)/ngFF * lw)
                ax.plot(t[-1], np.mean(cFF), '*g', ms = (ig+1)/ngFF * lw)
                current = current + cFF
            for ig in range(ngE):
                cE = -_gE[0,ig,i,:]*(_v[i,:]-vE)
                ax.plot(t, cE, '-r', lw = (ig+1)/ngE * lw)
                ax.plot(t[-1], np.mean(cE), '*r', ms = (ig+1)/ngE * lw)
                current = current + cE
            for ig in range(ngI):
                cI = -_gI[0,ig,i,:]*(_v[i,:]-vI)
                ax.plot(t, cI, '-b', lw = (ig+1)/ngI * lw)
                ax.plot(t[-1], np.mean(cI), '*b', ms = (ig+1)/ngI * lw)
                current = current + cI
    
    
            ax.plot(t, _depC[i,:], '-y', lw = lw)
            ax.plot(t[-1], np.mean(_depC[i,:]), '*y', ms = lw)
            current = current + _depC[i,:]
    
            if iModel == 1:
                if pW:
                    ax.plot(t, -_w[i,:], '-m', lw = lw)
                    ax.plot(t[-1], -np.mean(_w[i,:]), '*m', ms = lw)
                    current = current - _w[i,:]
    
            ax.plot(t, current, '-k', lw = lw)
            ax.plot(t, np.zeros(t.shape), ':k', lw = lw)
            mean_current = np.mean(current)
            ax.plot(t[-1], mean_current, '*k', ms = lw)
            if nLGN_V1[iV1] > 0:
                title = f'FR:{fr[iV1]:.3f}, F1F0:{F1F0[iV1]:.3f}, {gFF_F1F0[iV1]:.3f}(gFF)'
            else:
                title = f'FR:{fr[iV1]:.3f}, F1F0:{F1F0[iV1]:.3f}'
            ax.set_title(title)
            ax.set_ylabel('current')
    
            if nLGN_V1[iV1] > 0:
                ax = fig.add_subplot(grid[2,0])
                frTmp = LGN_fr[:, LGN_V1_ID[iV1]]
                LGN_fr_sSum = np.sum(frTmp[tpick,:] * LGN_V1_s[iV1], axis=-1)
                ax.plot(t, LGN_fr_sSum, '-g', lw = 2*lw)
    
                nbins = int(FRbins*t_in_sec*TF)
                sp_range = np.linspace(step0, step0+nt_, nbins+1)*dt
                spTmp = np.hstack(LGN_spScatter[LGN_V1_ID[iV1]])
                LGN_sp_total = spTmp[np.logical_and(spTmp>=step0*dt, spTmp<(nt_+step0)*dt)]
                counts, _ = np.histogram(LGN_sp_total, bins = sp_range)
                ax.hist((sp_range[:-1] + sp_range[1:])/2, bins = sp_range, color = 'b', weights = counts/(1/TF/FRbins), alpha = 0.5)
                ax.plot(LGN_sp_total, np.zeros(LGN_sp_total.size) + np.max(LGN_fr_sSum)/2, '*g', ms = 1.0)
                ax2 = ax.twinx()
                for ig in range(ngFF):
                    ax2.plot(t, _gFF[0,ig,i,:], ':g', lw = (ig+1)/ngFF * lw)
    
                ax = fig.add_subplot(grid[2,1])
                markers = ('^r', 'vg', '*g', 'dr', '^k', 'vb')
                iLGN_vpos = LGN_vpos[:, LGN_V1_ID[iV1]]
                iLGN_type = LGN_type[LGN_V1_ID[iV1]]
    
                for j in range(6):
                    pick = iLGN_type == j
                    ax.plot(iLGN_vpos[0,pick], iLGN_vpos[1,pick], markers[j], ms = 1.0)
                ax.set_aspect('equal')
            if pSingleLGN:
                for j in range(nLGN_V1[iV1]):
                    ax = fig.add_subplot(grid[3+j,0])
                    iLGN_fr = LGN_fr[tpick, LGN_V1_ID[iV1][j]]
                    ax.plot(t, iLGN_fr, '-g', lw = 2*lw)
                    spTmp = np.array(LGN_spScatter[LGN_V1_ID[iV1][j]])
                    iLGN_sp = spTmp[np.logical_and(spTmp>=step0*dt, spTmp<(nt_+step0)*dt)]
                    counts, _ = np.histogram(iLGN_sp, bins = sp_range)
                    ax.hist((sp_range[:-1] + sp_range[1:])/2, bins = sp_range, color = 'b', weights = counts/(1/TF/FRbins), alpha = 0.5)
                    ax.plot(iLGN_sp, np.zeros(iLGN_sp.size)+np.max(iLGN_fr)/2, '*g', ms = 1.0)
                    ax = fig.add_subplot(grid[3+j,1])
                    for k in range(6):
                        pick = iLGN_type == k
                        ax.plot(iLGN_vpos[0,pick], iLGN_vpos[1,pick], markers[k], ms = 1.0)
                    ax.plot(iLGN_vpos[0,j], iLGN_vpos[1,j], '*k')
                    ax.set_aspect('equal')
            fig.savefig(output_suffix + f'V1-sample-{iblock}-{ithread}' + '.png')
            plt.close(fig)
    
    # statistics
    if plotRpStat:
        fig = plt.figure(f'rpStats', figsize = (8,4), dpi = 600)
    
        eCpick = epick[nLGN_V1[epick] <= SCsplit]
        eSpick = epick[nLGN_V1[epick] > SCsplit]
        iCpick = ipick[nLGN_V1[ipick] <= SCsplit]
        iSpick = ipick[nLGN_V1[ipick] > SCsplit]
    
        _, bin_edges = np.histogram(fr, bins = 12)
    
        ax = fig.add_subplot(221)
        #ax.hist(fr[eSpick], bins = bin_edges, color = 'r', log = True, alpha = 0.5, label = 'ExcS')
        #ax.hist(fr[iSpick], bins = bin_edges, color = 'b', log = True, alpha = 0.5, label = 'InhS')
        #ax.hist(fr[eCpick], bins = bin_edges, color = 'm', log = True, alpha = 0.5, label = 'ExcC')
        #ax.hist(fr[iCpick], bins = bin_edges, color = 'g', log = True, alpha = 0.5, label = 'InhC')
        data = [fr[eSpick], fr[iSpick], fr[eCpick], fr[iCpick]]
        colors = ['r','b','m','g']
        labels = ['ExcS', 'InhS', 'ExcC', 'InhC']
        ax.hist(data, bins = 12, color = colors, log = True, alpha = 0.5, label = labels)
    
        nzero = sum(fr[eSpick] == 0)
        if nzero > 0:
            ax.plot(0, nzero, '*r')
        nzero = sum(fr[iSpick] == 0)
        if nzero > 0:
            ax.plot(0, nzero, '*b')
        nzero = sum(fr[eCpick] == 0)
        if nzero > 0:
            ax.plot(0, nzero, '*m')
        nzero = sum(fr[iCpick] == 0)
        if nzero > 0:
            ax.plot(0, nzero, '*g')
    
        ax.set_title('fr')
        ax.set_xlabel('Hz')
        ax.legend()
        
        target = np.sum(gFF[:,:,0], axis = 0)
        ax = fig.add_subplot(222)
        ax.hist(target[eSpick], color = 'r', alpha = 0.5)
        ax.hist(target[iSpick], color = 'b', alpha = 0.5)
        ax.set_title('gFF')
        
        target = np.sum(gE[:,:,0], axis = 0)
        ax = fig.add_subplot(223)
        ax.hist(target[eSpick], color = 'r', alpha = 0.5)
        ax.hist(target[iSpick], color = 'b', alpha = 0.5)
        ax.hist(target[eCpick], color = 'm', alpha = 0.5)
        ax.hist(target[iCpick], color = 'g', alpha = 0.5)
        ax.set_title('gE')
        
        target = np.sum(gI[:,:,0], axis = 0)
        ax = fig.add_subplot(224)
        ax.hist(target[eSpick], color = 'r', alpha = 0.5)
        ax.hist(target[iSpick], color = 'b', alpha = 0.5)
        ax.hist(target[eCpick], color = 'm', alpha = 0.5)
        ax.hist(target[iCpick], color = 'g', alpha = 0.5)
        ax.set_title('gI')
    
        fig.savefig(output_suffix + 'V1-rpStats' + '.png')
        plt.close(fig)
    
        fig = plt.figure(f'OP-rpStats', figsize = (8,4), dpi = 600)
    
        eCpick = eORpick[nLGN_V1[eORpick] <= SCsplit]
        eSpick = eORpick[nLGN_V1[eORpick] > SCsplit]
        iCpick = iORpick[nLGN_V1[iORpick] <= SCsplit]
        iSpick = iORpick[nLGN_V1[iORpick] > SCsplit]
    
        _, bin_edges = np.histogram(fr[np.hstack((eORpick, iORpick))], bins = 12)
        
        ax = fig.add_subplot(221)
        #ax.hist(fr[eSpick], bins = bin_edges, color = 'r', log = True, alpha = 0.5, label = 'ExcS')
    
        #ax.hist(fr[iSpick], bins = bin_edges, color = 'b', log = True, alpha = 0.5, label = 'InhS')
    
        #ax.hist(fr[eCpick], bins = bin_edges, color = 'm', log = True, alpha = 0.5, label = 'ExcC')
    
        #ax.hist(fr[iCpick], bins = bin_edges, color = 'g', log = True, alpha = 0.5, label = 'InhC')
        data = [fr[eSpick], fr[iSpick], fr[eCpick], fr[iCpick]]
        colors = ['r','b','m','g']
        labels = ['ExcS', 'InhS', 'ExcC', 'InhC']
        ax.hist(data, bins = 12, color = colors, log = True, alpha = 0.5, label = labels)
        nzero = sum(fr[eSpick] == 0)
        if nzero > 0:
            ax.plot(0, nzero, '*r')
        nzero = sum(fr[iSpick] == 0)
        if nzero > 0:
            ax.plot(0, nzero, '*b')
        nzero = sum(fr[eCpick] == 0)
        if nzero > 0:
            ax.plot(0, nzero, '*m')
        nzero = sum(fr[iCpick] == 0)
        if nzero > 0:
            ax.plot(0, nzero, '*g')
    
        ax.set_title('fr (OP only)')
        ax.set_xlabel('Hz')
        ax.legend()
        
        target = np.sum(gFF[:,:,0], axis = 0)
        ax = fig.add_subplot(222)
        _, bin_edges = np.histogram(target[np.hstack((eORpick, iORpick))], bins = 12)
        ax.hist(target[eSpick], bins = bin_edges, color = 'r', alpha = 0.5)
        ax.hist(target[iSpick], bins = bin_edges, color = 'b', alpha = 0.5)
        ax.set_title('gFF')
        
        target = np.sum(gE[:,:,0], axis = 0)
        ax = fig.add_subplot(223)
        _, bin_edges = np.histogram(target[np.hstack((eORpick, iORpick))], bins = 12)
        ax.hist(target[eSpick], bins = bin_edges, color = 'r', alpha = 0.5)
        ax.hist(target[iSpick], bins = bin_edges, color = 'b', alpha = 0.5)
        ax.hist(target[eCpick], bins = bin_edges, color = 'm', alpha = 0.5)
        ax.hist(target[iCpick], bins = bin_edges, color = 'g', alpha = 0.5)
        ax.set_title('gE')
        
        target = np.sum(gI[:,:,0], axis = 0)
        ax = fig.add_subplot(224)
        _, bin_edges = np.histogram(target[np.hstack((eORpick, iORpick))], bins = 12)
        ax.hist(target[eSpick], bins = bin_edges, color = 'r', alpha = 0.5)
        ax.hist(target[iSpick], bins = bin_edges, color = 'b', alpha = 0.5)
        ax.hist(target[eCpick], bins = bin_edges, color = 'm', alpha = 0.5)
        ax.hist(target[iCpick], bins = bin_edges, color = 'g', alpha = 0.5)
        ax.set_title('gI')
    
        fig.savefig(output_suffix + 'V1_OP-rpStats' + '.png')
        plt.close(fig)
    
    if plotRpCorr:
        fig = plt.figure(f'rpCorr', figsize = (12,12), dpi = 600)
        grid = gs.GridSpec(4, 4, figure = fig, hspace = 0.3, wspace = 0.3)
        target = np.sum(gE_gTot_ratio[:,:,0], axis = 0)
    
        ax = fig.add_subplot(grid[0,0])
        image = HeatMap(target[epick], fr[epick], 25, 25, ax, 'Reds', vmin = 0, log_scale = pLog)
        ax.set_xlabel('gExc/gTot')
        ax.set_ylabel('Exc. FR Hz')
        old_target = target.copy()
        
        ax = fig.add_subplot(grid[0,1])
        image = HeatMap(target[ipick], fr[ipick], 25, 25, ax, 'Blues', vmin = 0, log_scale = pLog)
        ax.set_xlabel('gExc/gTot')
        ax.set_ylabel('Inh. FR Hz')
    
        target = F1F0
        ax = fig.add_subplot(grid[0,2])
        pick = epick[np.logical_and(nLGN_V1[epick]>0, F0_0[epick])]
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = pLog)
        ax.set_xlabel('F1F0')
        ax.set_ylabel('ExcS. FR Hz')
    
        ax = fig.add_subplot(grid[0,3])
        pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0_0[ipick])]
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = pLog)
        ax.set_xlabel('F1F0')
        ax.set_ylabel('InhS. FR Hz')
    
        target = gFF_F1F0
        ax = fig.add_subplot(grid[1,0])
        pick = epick[np.logical_and(nLGN_V1[epick]>0,gFF_F0_0[epick])]
        active = np.sum(fr[pick]>0)/np.sum(nLGN_V1[epick]>0)
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = pLog)
        ax.set_title(f'active simple {active*100:.3f}%')
        ax.set_xlabel('gFF_F1/F0')
        ax.set_ylabel('ExcS FR')
    
        ax = fig.add_subplot(grid[1,1])
        pick = ipick[np.logical_and(nLGN_V1[ipick]>0,gFF_F0_0[ipick])]
        active = np.sum(fr[pick]>0)/np.sum(nLGN_V1[ipick]>0)
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = pLog)
        ax.set_xlabel('gFF_F1/F0')
        ax.set_ylabel('InhS FR')
        ax.set_title(f'active simple {active*100:.3f}%')
    
        target = np.sum(gE[:,:,0], axis = 0)
        ax = fig.add_subplot(grid[1,2])
        pick = epick[nLGN_V1[epick]==0]
        active = np.sum(fr[pick]>0)/np.sum(nLGN_V1[epick]==0)
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = pLog)
        ax.set_xlabel('gE')
        ax.set_ylabel('ExcC FR')
        ax.set_title(f'active complex {active*100:.3f}%')
    
        ax = fig.add_subplot(grid[1,3])
        pick = ipick[nLGN_V1[ipick]==0]
        active = np.sum(fr[pick]>0)/np.sum(nLGN_V1[ipick]==0)
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = pLog)
        ax.set_xlabel('gE')
        ax.set_ylabel('InhC FR')
        ax.set_title(f'active complex {active*100:.3f}%')
    
        target = OP*180/np.pi
        ax = fig.add_subplot(grid[2,0])
        pick = epick[nLGN_V1[epick]>0]
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('S. Exc FR')
    
        ax = fig.add_subplot(grid[2,1])
        pick = ipick[nLGN_V1[ipick]>0]
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('S. Inh FR')
    
        ax = fig.add_subplot(grid[2,2])
        pick = epick[nLGN_V1[epick]==0]
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('C. Exc FR')
    
        ax = fig.add_subplot(grid[2,3])
        pick = ipick[nLGN_V1[ipick]==0]
        image = HeatMap(target[pick], fr[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('C. Inh FR')
    
        ytarget = gFF_F1F0
        ax = fig.add_subplot(grid[3,0])
        pick = epick[np.logical_and(nLGN_V1[epick]>0, gFF_F0_0[epick])]
        image = HeatMap(target[pick], ytarget[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('Exc. gFF_F1F0')
    
        ax = fig.add_subplot(grid[3,1])
        pick = ipick[np.logical_and(nLGN_V1[ipick]>0, gFF_F0_0[ipick])]
        image = HeatMap(target[pick], ytarget[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('Inh. gFF_F1F0')
    
        ytarget = F1F0
        ax = fig.add_subplot(grid[3,2])
        pick = epick[np.logical_and(nLGN_V1[epick]>0, F0_0[epick])]
        image = HeatMap(target[pick], ytarget[pick], 25, 25, ax, 'Reds', vmin = 0, log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('Exc. F1F0')
    
        ax = fig.add_subplot(grid[3,3])
        pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0_0[ipick])]
        image = HeatMap(target[pick], ytarget[pick], 25, 25, ax, 'Blues', vmin = 0, log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('Inh. F1F0')
    
        fig.savefig(output_suffix + 'V1-rpCorr' + '.png')
        plt.close(fig)
    
    if plotLR_rp:
        fig = plt.figure(f'LRrpStat', dpi = 600)
        ax = fig.add_subplot(221)
        ax.hist(fr[LR>0], color = 'r', log = True, alpha = 0.5, label = 'Contra')
        ax.plot(0, sum(fr[LR>0] == 0), '*r')
        ax.hist(fr[LR<0], color = 'b', log = True, alpha = 0.5, label = 'Ipsi')
        ax.plot(0, sum(fr[LR<0] == 0), '*b')
        ax.set_title('fr')
        ax.set_xlabel('Hz')
        ax.legend()
    
        target = np.sum(gFF[:,:,0], axis = 0)
        ax = fig.add_subplot(222)
        ax.hist(target[LR>0], color = 'r', alpha = 0.5)
        ax.hist(target[LR<0], color = 'b', alpha = 0.5)
        ax.set_title('gFF')
        
        target = np.sum(gE[:,:,0], axis = 0)
        ax = fig.add_subplot(223)
        ax.hist(target[LR>0], color = 'r', alpha = 0.5)
        ax.hist(target[LR<0], color = 'b', alpha = 0.5)
        ax.set_title('gE')
        
        target = np.sum(gI[:,:,0], axis = 0)
        ax = fig.add_subplot(224)
        ax.hist(target[LR>0], color = 'r', alpha = 0.5)
        ax.hist(target[LR<0], color = 'b', alpha = 0.5)
        ax.set_title('gI')
        fig.savefig(output_suffix + 'V1-LRrpStats' + '.png')
        plt.close(fig)
    
    if plotExc_sLGN:
        FF_irl = np.sum(gFF[:,:,0], axis = 0)
        FF_sSum = np.array([np.sum(s) for s in LGN_V1_s])
        fig = plt.figure(f'FF_sLGN', dpi = 600)
        ax = fig.add_subplot(221)
        ax.plot(FF_sSum[epick], FF_irl[epick], '*r', ms = 0.1)
        ax.plot(FF_sSum[ipick], FF_irl[ipick], '*b', ms = 0.1)
        ax.set_xlabel('FF_sSum')
        ax.set_ylabel('gFF')
        ax.set_xlim(left = 0)
        ax.set_ylim(bottom = 0)
    
        cortical = np.sum(gE[:,:,0], axis = 0)
        ax = fig.add_subplot(222)
        ax.plot(FF_sSum[epick], cortical[epick], '*r', ms = 0.1)
        ax.plot(FF_sSum[ipick], cortical[ipick], '*b', ms = 0.1)
        ax.set_xlabel('FF_sSum')
        ax.set_ylabel('gE')
        ax.set_xlim(left = 0)
        ax.set_ylim(bottom = 0)
    
        ax = fig.add_subplot(223)
        ax.plot(FF_irl[epick], cortical[epick]+FF_irl[epick], '*r', ms = 0.1)
        ax.plot(FF_irl[ipick], cortical[ipick]+FF_irl[ipick], '*b', ms = 0.1)
        ax2 = ax.twinx()
        ax2.plot(FF_irl[epick], fr[epick], 'sr', ms = 0.1)
        ax2.plot(FF_irl[ipick], fr[ipick], 'sb', ms = 0.1)
        ax.set_xlabel('gFF')
        ax.set_ylabel('gE+gFF')
        ax2.set_ylabel('fr')
        ax.set_xlim(left = 0)
        ax.set_ylim(bottom = 0)
    
        corticalI = np.sum(gI[:,:,0], axis = 0)
        ax = fig.add_subplot(224)
        ax.plot(FF_sSum[epick], (cortical[epick]+FF_irl[epick])/corticalI[epick], '*r', ms = 0.1)
        ax.plot(FF_sSum[ipick], (cortical[ipick]+FF_irl[ipick])/corticalI[ipick], '*b', ms = 0.1)
        ax.set_xlabel('FF_sSum')
        ax.set_ylabel('(gE+gFF)/gI')
        ax.set_xlim(left = 0)
        ax.set_ylim(bottom = 0)
    
        fig.savefig(output_suffix + 'Exc_sLGN' + '.png')
        plt.close(fig)
            
    # scatter
    if plotScatterFF:
        fig = plt.figure(f'scatterFF', dpi = 1200)
        ax = fig.add_subplot(211)
        tsp = np.hstack([x for x in spScatter])
        isp = np.hstack([ix + np.zeros(len(spScatter[ix])) for ix in np.arange(nV1)])
        tpick = np.logical_and(tsp>=step0*dt, tsp<(step0+nt_)*dt)
        tsp = tsp[tpick]
        isp = isp[tpick].astype(int)
    
        ax.set_xlim(step0*dt, (step0+nt_)*dt)
    
        ax2 = fig.add_subplot(212)
        ax2_ = ax2.twinx()
        t = nt_ *dt #ms
        nbins = t/tbinSize #every 1ms
        edges = step0*dt + np.arange(nbins+1) * tbinSize 
        t_tf = (edges[:-1] + edges[1:])/2
        ff = np.arange(nbins//2+1) * 1000/t
    
        ipre = np.argwhere(ff<TF)[-1][0]
        ipost = np.argwhere(ff>=TF)[0][0]
    
    
        if pLR or pSC:
            if pLR:
                eSpick = np.logical_and(np.mod(isp, blockSize) < mE , LR[isp]>0)
                iSpick = np.logical_and(np.mod(isp, blockSize) >= mE, LR[isp]>0)
                ax.plot(tsp[eSpick], isp[eSpick], ',r')
                ax.plot(tsp[iSpick], isp[iSpick], ',b')
    
                if np.sum(eSpick) > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/np.sum(eSpick)
                    amp = np.abs(np.fft.rfft(nsp))/nbins
                    amp[1:] = amp[1:]*2
                    ax2.plot(ff, amp, 'r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5, label = 'exc R')
    
                if np.sum(iSpick) > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/np.sum(iSpick)
                    amp = np.abs(np.fft.rfft(nsp))/nbins
                    amp[1:] = amp[1:]*2
                    ax2_.plot(ff, amp, 'b', lw = 0.5, alpha = 0.5)
                    ax2_.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh R')
                ax2.set_yscale('log')
                ax2.set_xlabel('Hz')
    
                eSpick = np.logical_and(np.mod(isp, blockSize) < mE , LR[isp]<0)
                iSpick = np.logical_and(np.mod(isp, blockSize) >= mE, LR[isp]<0)
                ax.plot(tsp[eSpick], isp[eSpick], ',m')
                ax.plot(tsp[iSpick], isp[iSpick], ',g')
    
                if np.sum(eSpick) > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/np.sum(eSpick)
                    amp = np.abs(np.fft.rfft(nsp))/nbins
                    amp[1:] = amp[1:]*2
                    ax2.plot(ff, amp, 'm', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>m', ms = 0.5, alpha = 0.5, label = 'exc L')
    
                if np.sum(iSpick) > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/np.sum(iSpick)
                    amp = np.abs(np.fft.rfft(nsp))/nbins
                    amp[1:] = amp[1:]*2
                    ax2_.plot(ff, amp, 'g', lw = 0.5, alpha = 0.5)
                    ax2_.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<g', ms = 0.5, alpha = 0.5, label = 'inh L')
                ax2.set_yscale('log')
                ax2.set_xlabel('Hz')
    
            if pSC:
                eSpick = np.logical_and(np.mod(isp, blockSize) < mE , nLGN_V1[isp] == 0)
                iSpick = np.logical_and(np.mod(isp, blockSize) >= mE, nLGN_V1[isp] == 0)
                ax.plot(tsp[eSpick], isp[eSpick], ',r')
                ax.plot(tsp[iSpick], isp[iSpick], ',b')
    
                if np.sum(eSpick) > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/np.sum(eSpick)
                    amp = np.abs(np.fft.rfft(nsp))/nbins
                    amp[1:] = amp[1:]*2
                    ax2.plot(ff, amp, 'r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5, label = 'exc C')
    
                if np.sum(iSpick) > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/np.sum(iSpick)
                    amp = np.abs(np.fft.rfft(nsp))/nbins
                    amp[1:] = amp[1:]*2
                    ax2_.plot(ff, amp, 'b', lw = 0.5, alpha = 0.5)
                    ax2_.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh C')

                ax2_.set_yscale('log')
                ax2.set_xlabel('Hz')
    
                eSpick = np.logical_and(np.mod(isp, blockSize) < mE , nLGN_V1[isp] > 0)
                iSpick = np.logical_and(np.mod(isp, blockSize) >= mE, nLGN_V1[isp] > 0)
                ax.plot(tsp[eSpick], isp[eSpick], ',m')
                ax.plot(tsp[iSpick], isp[iSpick], ',g')
    
                if np.sum(eSpick) > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/np.sum(eSpick)
                    amp = np.abs(np.fft.rfft(nsp))/nbins
                    amp[1:] = amp[1:]*2
                    ax2.plot(ff, amp, 'm', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>m', ms = 0.5, alpha = 0.5, label = 'exc S')
    
                if np.sum(iSpick) > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/np.sum(iSpick)
                    amp = np.abs(np.fft.rfft(nsp))/nbins
                    amp[1:] = amp[1:]*2
                    ax2_.plot(ff, amp, 'g', lw = 0.5, alpha = 0.5)
                    ax2_.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<g', ms = 0.5, alpha = 0.5, label = 'inh S')

                ax2_.set_yscale('log')
                ax2.set_xlabel('Hz')
    
        else:
            eSpick = np.mod(isp, blockSize) < mE
            iSpick = np.mod(isp, blockSize) >= mE
            ax.plot(tsp[eSpick], isp[eSpick], ',r')
            ax.plot(tsp[iSpick], isp[iSpick], ',b')
        
            nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nblock*mE)
            amp = np.abs(np.fft.rfft(nsp))/nbins
            amp[1:] = amp[1:]*2
            ax2.plot(ff, amp, 'r', lw = 0.5, alpha = 0.5)
            ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5, label = 'exc')
            ax2.set_yscale('log')
            ax2.set_xlabel('Hz')
    
            nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nblock*mI)
            amp = np.abs(np.fft.rfft(nsp))/nbins
            amp[1:] = amp[1:]*2
            ax2_.plot(ff, amp, 'b', lw = 0.5, alpha = 0.5)
            ax2_.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh')
            ax2_.set_yscale('log')
    
        if nt == nt_:
            fig.savefig(output_suffix + 'V1-scatterFF' + '_full.png')
        else:
            fig.savefig(output_suffix + 'V1-scatterFF' + '.png')
        plt.close(fig)
    
    if plotLGNsCorr:
        fig = plt.figure(f'sLGN-corr', figsize = (12,12), dpi = 600)
        grid = gs.GridSpec(4, 4, figure = fig, hspace = 0.3, wspace = 0.3)
        
        gFF_target = np.sum(gFF[:,:,0], axis = 0)
        gE_target  = np.sum(gE[:,:,0], axis = 0)
        gI_target  = np.sum(gI[:,:,0], axis = 0)
    
        nsE = preNS[0,:]
        nsI = preNS[1,:]
        
        ax = fig.add_subplot(grid[0,0])
        image = HeatMap(gFF_target[epick], gE_target[epick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. gE')
        
        ax = fig.add_subplot(grid[0,1])
        image = HeatMap(gFF_target[ipick], gE_target[ipick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. gE')
        
        ax = fig.add_subplot(grid[1,0])
        image = HeatMap(gFF_target[epick], nsE[epick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. nsE')
        
        ax = fig.add_subplot(grid[1,1])
        image = HeatMap(gFF_target[ipick], nsE[ipick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. nsE')
        
        ax = fig.add_subplot(grid[2,0])
        image = HeatMap(gFF_target[epick], gI_target[epick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. gI')
        
        ax = fig.add_subplot(grid[2,1])
        image = HeatMap(gFF_target[ipick], gI_target[ipick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. gI')
        
        ax = fig.add_subplot(grid[0,2])
        image = HeatMap(gFF_target[epick], fr[epick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. FR Hz')
        
        ax = fig.add_subplot(grid[0,3])
        image = HeatMap(gFF_target[ipick], fr[ipick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. FR Hz')
        
        ax = fig.add_subplot(grid[3,0])
        image = HeatMap(gFF_target[epick], nsI[epick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. nsI')
        
        ax = fig.add_subplot(grid[3,1])
        image = HeatMap(gFF_target[ipick], nsI[ipick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
        ax.set_xlabel('inh. nsI')
        ax.set_ylabel('inh. nsI')
    
        target = np.sum(gE_gTot_ratio[:,:,0],axis = 0)
        ax = fig.add_subplot(grid[1,2])
        image = HeatMap(gFF_target[epick], target[epick] , 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. gE/gTot')
        
        ax = fig.add_subplot(grid[1,3])
        image = HeatMap(gFF_target[ipick], target[ipick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. gE/gTot')
    
        ax = fig.add_subplot(grid[2,2])
        image = HeatMap(gFF_target[epick], nsE[epick]/nsI[epick], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. nsE/nsI')
        
        ax = fig.add_subplot(grid[2,3])
        image = HeatMap(gFF_target[ipick], nsE[ipick]/nsI[ipick], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. nsE/nsI')
    
        ax = fig.add_subplot(grid[3,2])
        image = HeatMap(gFF_target[epick], gEt_gTot_ratio[epick,0], 25, 25, ax, 'Reds', vmin = 0, log_scale = True)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. gEt/gTot')
        
        ax = fig.add_subplot(grid[3,3])
        image = HeatMap(gFF_target[ipick], gEt_gTot_ratio[ipick,0], 25, 25, ax, 'Blues', vmin = 0, log_scale = True)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. gEt/gTot')
    
        fig.savefig(output_suffix + 'sLGN-corr' + '.png')
        plt.close(fig)
    print('plotting finished')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_suffix = sys.argv[1]
        print(output_suffix)
        if len(sys.argv) > 2:
            conLGN_suffix = sys.argv[2]
            print(conLGN_suffix)
            if len(sys.argv) > 3:
                conV1_suffix = sys.argv[3]
                print(conV1_suffix)
                if len(sys.argv) > 4:
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
        conV1_suffix = ""
        readNewSpike = False
    plotV1_response(output_suffix, conLGN_suffix, conV1_suffix, readNewSpike)
