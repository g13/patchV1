#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.gridspec as gs
import matplotlib.colors as clr
from matplotlib import cm
import sys
from readPatchOutput import *
np.seterr(invalid = 'raise')


#@profile
def plotV1_response_lFF(output_suffix0, res_suffix, conLGN_suffix, conV1_suffix, res_fdr, data_fdr, fig_fdr, TF, iOri, nOri, readNewSpike, usePrefData, collectMeanDataOnly, OPstatus):
    #sample = np.array([0,1,2,768])
    #sample = np.array([40, 58, 75])
    #sample = np.array([1338, 10235])
    sample = np.array([203,360,365,715,467,743,203,752]);

    SCsplit = 0
    nLGNorF1F0 = True
    ns = 10
    seed = 657890
    np.random.seed(seed)
    step0 = 0
    nt_ = 50000
    nstep = 2000
    if nOri > 0:
        stiOri = np.pi*np.mod(iOri/nOri, 1.0)
    else:
        OPstatus = 0

    heatBins = 25
    TFbins = 25
    FRbins = 25
    tbinSize = 1
    nsmooth = 0
    lw = 0.1
    
    #plotRpStat = True 
    #plotRpCorr = True
    #plotScatterFF = True
    plotSample = True
    #plotDepC = True # plot depC distribution over orientation
    #plotLGNsCorr = True
    #plotTempMod = True 
    #plotExc_sLGN = True
    #plotLR_rp = True
    
    plotRpStat = False 
    plotRpCorr = False 
    plotScatterFF = False
    #plotSample = False
    plotDepC = False
    plotLGNsCorr = False 
    plotTempMod = False 
    plotExc_sLGN = False
    plotLR_rp = False 
    
    pSample = True
    pVoltage = True
    pCond = True
    pGap = True
    #pH = True
    #pFeature = True
    #pLR = True
    pW = True
    pDep = True
    pLog = True 
    
    #pSample = False
    #pVoltage = False
    #pCond = False
    #pGap = False
    pH = False
    pFeature = False
    pLR = False
    #pW = False
    #pDep = False
    #pLog = False
    
    pSingleLGN = False
    pSC = True
    #pSC = False

    if pSC and pLR:
        print('pSC overrides pLR')
        pLR = False
    
    if plotRpStat or plotLR_rp or plotRpCorr:
        pCond = True
    
    if plotRpCorr or plotSample or plotLGNsCorr:
        plotTempMod = True
    
    if plotExc_sLGN:
        pCond = True
    
    if plotLR_rp:
        pLR = True
        pFeature = True
    
    if plotSample:
        pVoltage = True
        pCond = True
    
    if plotLGNsCorr:
        pCond = True
    
    if pCond:
        pVoltage = True
    # const
    if nOri > 0:
        output_suffix = output_suffix0 + '_' + str(iOri)
    else:
        output_suffix = output_suffix0
    _output_suffix = "_" + output_suffix
    res_suffix = "_" + res_suffix
    conLGN_suffix = "_" + conLGN_suffix
    conV1_suffix = "_" + conV1_suffix

    data_fdr = data_fdr+"/"
    res_fdr = res_fdr+"/"
    fig_fdr = fig_fdr+"/"
    
    rawDataFn = data_fdr + "rawData" + _output_suffix + ".bin"
    LGN_frFn = data_fdr + "LGN_fr" + _output_suffix + ".bin"
    LGN_spFn = data_fdr + "LGN_sp" + _output_suffix
    meanFn = data_fdr + "mean_data" + _output_suffix + ".bin"
    
    pref_file = data_fdr + 'cort_pref_' + output_suffix0 + '.bin'
    if nOri == 0:
        max_frFn = data_fdr + 'max_fr_' + output_suffix0 + '.bin'

    spDataFn = data_fdr + "V1_spikes" + _output_suffix
    parameterFn = data_fdr + "patchV1_cfg" +_output_suffix + ".bin"

    LGN_V1_sFn = res_fdr + "LGN_V1_sList" + conLGN_suffix + ".bin"
    LGN_V1_idFn = res_fdr + "LGN_V1_idList" + conLGN_suffix + ".bin"
    conStats_Fn = res_fdr + "conStats" + conV1_suffix + ".bin"
    V1_RFpropFn = res_fdr + "V1_RFprop" + conLGN_suffix + ".bin"
    LGN_vposFn = res_fdr + 'LGN_vpos'+ res_suffix + ".bin"
    featureFn = res_fdr + 'V1_feature' + res_suffix + ".bin"
    V1_allposFn = res_fdr + 'V1_allpos' + res_suffix + ".bin"
    V1_vposFn = res_fdr + 'V1_vpos' + res_suffix + ".bin"

    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, virtual_LGN = read_cfg(parameterFn)
    blockSize = typeAcc[-1]
    print(f'blockSize = {blockSize}')
    
    if output_suffix:
        output_suffix = output_suffix + "-"
    
    if readNewSpike:
        print('reading LGN connections...')
        LGN_V1_s = readLGN_V1_s0(LGN_V1_sFn)
        print('     ID...')
        LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)
        print('     vpos...')
        nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type = readLGN_vpos(LGN_vposFn)
        print('     fr...')
        LGN_fr = readLGN_fr(LGN_frFn, prec = prec)
        print('     spikes...')
        LGN_spScatter = readLGN_sp(LGN_spFn + ".bin", prec = prec)
        np.savez(LGN_spFn + '.npz', spName = 'LGN_spScatter', LGN_spScatter = LGN_spScatter, LGN_V1_s = LGN_V1_s, LGN_V1_ID = LGN_V1_ID, nLGN_V1 = nLGN_V1, LGN_vpos = LGN_vpos, LGN_type = LGN_type, LGN_fr = LGN_fr, nLGN = nLGN, nLGN_I = nLGN_I, nLGN_C = nLGN_C)
        print('complete.')
    else:
        if plotExc_sLGN or plotSample or (plotTempMod and pCond):
            print('loading LGN data...')
            with np.load(LGN_spFn + '.npz', allow_pickle=True) as data:
                nLGN_V1 = data['nLGN_V1']
                nLGN = data['nLGN']
                nLGN_I = data['nLGN_I']
                nLGN_C = data['nLGN_C']
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
    
    with open(V1_allposFn, 'r') as f:
        nblock, blockSize, dataDim = np.fromfile(f, 'u4', count=3)
        networkSize = nblock*blockSize
        print(f'dataDim = {dataDim}')
        assert(blockSize == typeAcc[-1])
        print([nblock,blockSize,networkSize,dataDim])
        V1_x0, V1_xspan, V1_y0, V1_yspan = np.fromfile(f, 'f8', count=4)
        print(f'x:{[V1_x0, V1_x0 + V1_xspan]}')
        print(f'y:{[V1_y0, V1_y0 + V1_yspan]}')
        _pos = np.reshape(np.fromfile(f, 'f8', count = networkSize*dataDim), (nblock, dataDim, blockSize))
        pos = np.zeros((2,networkSize))
        pos[0,:] = _pos[:,0,:].reshape(networkSize)
        pos[1,:] = _pos[:,1,:].reshape(networkSize)
        V1_vx0, V1_vxspan, V1_vy0, V1_vyspan = np.fromfile(f, 'f8', 4)
        print(f'vx:{[V1_vx0, V1_vx0 + V1_vxspan]}')
        print(f'vy:{[V1_vy0, V1_vy0 + V1_vyspan]}')
        vx, vy = np.fromfile(f, 'f8', 2*networkSize).reshape(2,networkSize)
    
    with open(rawDataFn, 'rb') as f:
        dt = np.fromfile(f, prec, 1)[0] 
        nt = np.fromfile(f, 'u4', 1)[0] 
        if step0 >= nt:
            step0 = 0
            print('step0 is too large, set back to 0.')
        if nt_ == 0 or step0 + nt_ >= nt:
            nt_ = nt - step0
        if nstep > nt_:
            nstep = nt_
        tstep = (nt_ + nstep - 1)//nstep
        tstep = min(round(10/dt),tstep)
        nstep = nt_//tstep
        interval = tstep - 1
        print(f'plot {nstep} data points from the {nt_} time steps startingfrom step {step0}, total {nt} steps, dt = {dt}')
        nV1 = np.fromfile(f, 'u4', 1)[0] 
        iModel = np.fromfile(f, 'i4', 1)[0] 
        mI = np.fromfile(f, 'u4', 1)[0] 
        haveH = np.fromfile(f, 'u4', 1)[0] 
        ngFF = np.fromfile(f, 'u4', 1)[0] 
        ngE = np.fromfile(f, 'u4', 1)[0] 
        ngI = np.fromfile(f, 'u4', 1)[0] 

    if TF == 0:
        TF = 1000/(nt_*dt)
    if nt_*dt < 1000/TF:
        TF = 1000/(nt_*dt)
    print(f'TF = {TF}')
    
    print(f'using model {iModel}')
    
    nblock = nV1//blockSize
    epick = np.hstack([np.arange(nE) + iblock*blockSize for iblock in range(nblock)])
    ipick = np.hstack([np.arange(nI) + iblock*blockSize + nE for iblock in range(nblock)])
    
    _gL = np.zeros(nV1)
    _gL[epick] = gL[0]
    _gL[ipick] = gL[1]
    
    print(f'nV1 = {nV1}; haveH = {haveH}')
    print(f'ngFF: {ngFF}; ngE: {ngE}; ngI: {ngI}')
    
    if plotSample or plotLGNsCorr:
        nType, ExcRatio, preN, preNS, _, _, _, _, _ = read_conStats(conStats_Fn)
    
    # readFeature
    if pFeature or pLR or plotSample or pSample or plotRpCorr or plotScatterFF or plotTempMod or plotRpStat:
        featureType = np.array([0,1])
        feature, rangeFeature, minFeature, maxFeature = readFeature(featureFn, nV1, featureType)
        LR = feature[0,:]
        print(f'feature0 range:{[np.min(feature[0,:]), np.max(feature[0,:])]}')
        print(f'feature1 range:{[np.min(feature[1,:]), np.max(feature[1,:])]}')
        if usePrefData:
            try:
                with open(pref_file, 'rb') as f:
                    fitted = np.fromfile(f, 'i4', 1)[0]
                    _nV1 = np.fromfile(f, 'u4', 1)[0]
                    assert(_nV1 == nV1)
                    if fitted == 1:
                        print('using fitted OP')
                    OP = np.fromfile(f, 'f4', nV1)
                    OP_preset = np.mod(feature[1,:] + 0.5, 1.0)*np.pi
                print(f'read OP from {pref_file}')
            except IOError:
                print(f'Could not open {pref_file}! no cortical OP available, use preset OP instead.') 
                usePrefData = False
                OP = np.mod(feature[1,:] + 0.5, 1.0)*np.pi

        else:
            print(f'using preset OP')
            OP = np.mod(feature[1,:] + 0.5, 1.0)*np.pi
        print(f'OPrange: {[np.min(OP), np.max(OP)]}')

        OPrange = np.arange(heatBins+1)/heatBins * 180
        if nOri > 0:
            dOP = np.abs(OP - stiOri)
            dOP[dOP > np.pi/2] = np.pi - dOP[dOP > np.pi/2]
            dOri = np.pi/(2*nOri)
        else:
            dOP = np.zeros(nV1)
            dOri = 1
        eORpick = epick[dOP[epick] <= dOri]
        iORpick = ipick[dOP[ipick] <= dOri]
    
    F1F0range = np.arange(heatBins+1)/heatBins * 2
    # time range
    tpick = step0 + np.arange(nstep)*tstep 
    t = tpick*dt
    assert(np.sum(epick) + np.sum(ipick) == np.sum(np.arange(nV1)))
    t_in_ms = nt_*dt
    t_in_sec = t_in_ms/1000
    print(f't = {t_in_sec}')
    
    # read spikes
    if not readNewSpike:
        print('loading V1 spike...')
        with np.load(spDataFn + '.npz', allow_pickle=True) as data:
            spScatter = data['spScatter']
    else:
        spScatter = readSpike(rawDataFn, spDataFn, prec, sizeofPrec, vThres)

    fr = np.array([x[np.logical_and(x>=step0*dt, x<(nt_+step0)*dt)].size for x in spScatter])/t_in_sec

    if nOri == 0:
        with open(max_frFn, 'wb') as f:
            fr.tofile(f)

    print('V1 spikes acquired')
    
    if plotSample or pSample:
        with open(V1_vposFn, 'rb') as f:
            _n = np.fromfile(f, 'u4', 1)[0]
            assert(_n == nV1)
            V1_ecc = np.fromfile(f, 'f8', _n)
            V1_polar = np.fromfile(f, 'f8', _n)
            cx0 = V1_ecc*np.cos(V1_polar)
            cy0 = V1_ecc*np.sin(V1_polar)

        with open(V1_RFpropFn, 'rb') as f:
            _n = np.fromfile(f, 'u4', 1)[0]
            assert(_n == nV1)
            cx = np.fromfile(f, 'f4', _n)
            cy = np.fromfile(f, 'f4', _n)
            a = np.fromfile(f, 'f4', _n)
            RFphase = np.fromfile(f, 'f4', _n)
            RFfreq = np.fromfile(f, 'f4', _n)
            baRatio = np.fromfile(f, 'f4', _n)

        if 'sample' not in locals():
            sample = np.random.randint(nV1, size = ns)
            if False:
                sample = np.argsort(sfreq)[-ns:]
                print(sample)

            if False:
                sample = np.zeros(12, dtype = int)

                pick = epick[nLGN_V1[epick] > np.mean(nLGN_V1[epick])]
                opick = pick[dOP[pick] <= dOri]
                if np.sum(opick) > 0:
                    sample[0] = opick[np.argmin(fr[opick])]
                    sample[1] = opick[np.argmax(fr[opick])]
                else:
                    sample[0] = np.random.randint(nV1)
                    sample[1] = np.random.randint(nV1)

                opick = pick[dOP[pick] >= (nOri/2-1)*dOri]
                if np.sum(opick) > 0:
                    sample[2] = opick[np.argmin(fr[opick])]
                    sample[3] = opick[np.argmax(fr[opick])]
                else:
                    sample[2] = np.random.randint(nV1)
                    sample[3] = np.random.randint(nV1)

                pick = epick[nLGN_V1[epick] == 0]
                opick = pick[dOP[pick] <= dOri]
                if np.sum(opick) > 0:
                    sample[4] = opick[np.argmin(fr[opick])]
                    sample[5] = opick[np.argmax(fr[opick])]
                else:
                    sample[4] = np.random.randint(nV1)
                    sample[5] = np.random.randint(nV1)

                opick = pick[dOP[pick] >= (nOri/2-1)*dOri]
                if np.sum(opick) > 0:
                    sample[6] = opick[np.argmin(fr[opick])]
                    sample[7] = opick[np.argmax(fr[opick])]
                else:
                    sample[6] = np.random.randint(nV1)
                    sample[7] = np.random.randint(nV1)

                pick = ipick[nLGN_V1[ipick] > np.mean(nLGN_V1[ipick])]
                opick = pick[dOP[pick] <= dOri]
                if np.sum(opick) > 0:
                    sample[8] = opick[np.argmin(fr[opick])]
                    sample[9] = opick[np.argmax(fr[opick])]
                else:
                    sample[8] = np.random.randint(nV1)
                    sample[9] = np.random.randint(nV1)

                opick = pick[dOP[pick] >= (nOri/2-1)*dOri]
                if np.sum(opick) > 0:
                    sample[10] = opick[np.argmin(fr[opick])]
                    sample[11] = opick[np.argmax(fr[opick])]
                else:
                    sample[10] = np.random.randint(nV1)
                    sample[11] = np.random.randint(nV1)

                #sample = np.zeros(4, dtype = int)
                #sample[0] = 0
                #sample[1] = 1
                #sample[2] = 1000
                #sample[3] = 1001
        
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
        ns = sample.size
        print(f'sampling {[(s//blockSize, np.mod(s,blockSize)) for s in sample]}') 
    
    # read voltage and conductances
    if pVoltage or pCond or plotLGNsCorr or plotSample or plotDepC or (plotTempMod and pCond):
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
        cGap = np.zeros((mI,2)) # gap junction current
    
        s_v = np.zeros(nV1)
        s_gFF = np.zeros(nV1)
        s_gE = np.zeros(nV1)
        s_gI = np.zeros(nV1)
        s_gap = np.zeros(mI)
    
        tTF = 1000/TF
        _nstep = int(round(tTF/dt))
        stepsPerBin = _nstep//TFbins
        if stepsPerBin != _nstep/TFbins:
            #raise Exception(f'binning for periods of {tTF} can not be divided by sampling steps: {round(tTF/tstep):.0f} x {tstep} ms vs. {TFbins}')
            stepsPerBin = 1
            TFbins = _nstep
        print(f'steps per TFbin is {stepsPerBin}, tstep = {tstep}, tTF = {tTF}, TFbins = {TFbins}')

        if plotTempMod and pCond:
            dtTF = tTF/TFbins
            n_stacks = int(np.floor(nt_*dt / tTF))
            r_stacks = np.mod(nt_*dt, tTF)
            stacks = np.zeros(TFbins) + n_stacks
            i_stack = np.int(np.floor(r_stacks/dtTF))
            j_stack = np.mod(r_stacks, dtTF)
            stacks[:i_stack] += 1
            stacks[i_stack] += j_stack/dtTF
            print(f'stacks: {stacks}')
    
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
            f.seek(sizeofPrec+4*8, 1)
            if plotSample:
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
                if pGap:
                    gap_pick = np.mod(sample, blockSize) >= nE
                    gap_sample = sample[gap_pick]//blockSize*nI + np.mod(sample[gap_pick], blockSize) - nE
                    gap_ns = gap_sample.size
                    _cGap = np.empty((gap_ns, nstep), dtype = prec)
    
            if iModel == 0:
                f.seek(((3+(ngE + ngI + ngFF)*(1+haveH))*nV1 + mI)*sizeofPrec*step0, 1)
            if iModel == 1:
                f.seek(((4+(ngE + ngI + ngFF)*(1+haveH))*nV1 + mI)*sizeofPrec*step0, 1)
    
            per_it = 0
            per_nt = 0
            nstack_count = 0
            for i in range(nstep):
                f.seek(nV1*sizeofPrec, 1)
                data = np.fromfile(f, prec, nV1)
                depC[:,0] = depC[:,0] + data
                depC[:,1] = depC[:,1] + data*data
                if plotSample and pDep:
                    _depC[:,i] = data[sample]
    
                if iModel == 1:
                    data = np.fromfile(f, prec, nV1)
                    w[:,0] = w[:,0] + data
                    w[:,1] = w[:,1] + data*data
                    if pW and plotSample:
                        _w[:,i] = data[sample]
    
                s_v = np.fromfile(f, prec, nV1)
                v[:,0] = v[:,0] + s_v 
                v[:,1] = v[:,1] + s_v*s_v
                if pVoltage and plotSample:
                    _v[:,i] = s_v[sample]
    
                
                if pCond and plotTempMod:
                    tmp_v = tmp_v + s_v
                    if np.mod(per_it+1, stepsPerBin) == 0:
                        per_v[:,per_nt] = per_v[:,per_nt] + tmp_v/stepsPerBin
                        tmp_v = np.zeros(nV1)
    
                s_gFF = np.fromfile(f, prec, ngFF*nV1).reshape(ngFF,nV1)
                gFF[:,:,0] = gFF[:,:,0] + s_gFF
                gFF[:,:,1] = gFF[:,:,1] + s_gFF*s_gFF
                x = s_gFF*(vE - s_v)
                cFF[:,:,0] = cFF[:,:,0] + x
                cFF[:,:,1] = cFF[:,:,1] + x*x
                if pCond and plotSample:
                    _gFF[0,:,:,i] = s_gFF[:,sample]
    
                if haveH:
                    if pH:
                        data = np.fromfile(f, prec, ngFF*nV1).reshape(ngFF,nV1)
                        _gFF[1,:,:,i] = data[:,sample]
                    else:
                        f.seek(ngFF*nV1*sizeofPrec, 1)
    
                s_gE = np.fromfile(f, prec, ngE*nV1).reshape(ngE,nV1)
                gE[:,:,0] = gE[:,:,0] + s_gE
                gE[:,:,1] = gE[:,:,1] + s_gE*s_gE
                x = s_gE*(vE - s_v)
                cE[:,:,0] = cE[:,:,0] + x
                cE[:,:,1] = cE[:,:,1] + x*x
                if pCond and plotSample:
                    _gE[0,:,:,i] = s_gE[:,sample]
    
                s_gI = np.fromfile(f, prec, ngI*nV1).reshape(ngI,nV1)
                gI[:,:,0] = gI[:,:,0] + s_gI
                gI[:,:,1] = gI[:,:,1] + s_gI*s_gI
                x = s_gI*(vI - s_v)
                cI[:,:,0] = cI[:,:,0] + x
                cI[:,:,1] = cI[:,:,1] + x*x
                if pCond and plotSample:
                    _gI[0,:,:,i] = s_gI[:,sample]

                if plotTempMod and pCond:
                    tmp_gFF = tmp_gFF + np.sum(s_gFF, axis = 0)
                    tmp_gE = tmp_gE + np.sum(s_gE, axis = 0)
                    tmp_gI = tmp_gI + np.sum(s_gI, axis = 0)
                    if np.mod(per_it+1, stepsPerBin) == 0: 
                        per_gFF[:,per_nt] = per_gFF[:,per_nt] + tmp_gFF/stepsPerBin
                        per_gE[:,per_nt] = per_gE[:,per_nt] + tmp_gE/stepsPerBin
                        per_gI[:,per_nt] = per_gI[:,per_nt] + tmp_gI/stepsPerBin
                        tmp_gFF = np.zeros(nV1)
                        tmp_gE = np.zeros(nV1)
                        tmp_gI = np.zeros(nV1)
    
                if plotLGNsCorr or plotRpCorr:
                    s_gTot = np.sum(s_gE, axis = 0) + np.sum(s_gI, axis = 0) + np.sum(s_gFF, axis = 0) + _gL
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
                    if pH and plotSample:
                        data = np.fromfile(f, prec, ngE*nV1).reshape(ngE,nV1)
                        _gE[1,:,:,i] = data[:,sample]
                        data = np.fromfile(f, prec, ngI*nV1).reshape(ngI,nV1)
                        _gI[1,:,:,i] = data[:,sample]
                    else:
                        f.seek((ngE+ngI)*nV1*sizeofPrec, 1)

                s_cGap = np.fromfile(f, prec, mI)
                cGap[:,0] = cGap[:,0] + s_cGap
                cGap[:,1] = cGap[:,1] + s_cGap*s_cGap
                if pGap and plotSample:
                    _cGap[:,i] = s_cGap[gap_sample]
    
        
                if iModel == 0:
                    f.seek(((3+(ngE + ngI + ngFF)*(1+haveH))*nV1 + mI)*sizeofPrec*interval, 1)
                if iModel == 1:
                    f.seek(((4+(ngE + ngI + ngFF)*(1+haveH))*nV1 + mI)*sizeofPrec*interval, 1)

                per_it = np.mod(per_it+1, stepsPerBin)
                if per_it == 0:
                    if per_nt == 0:
                        nstack_count = nstack_count + 1
                    per_nt = np.mod(per_nt + 1, TFbins)
        print(f'nstack = {nstack_count}, istack = {per_nt}, rstack = {per_it}')
    
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

        with open(meanFn, 'wb') as f:
            np.array([nV1,mI,ngFF,ngE,ngI], dtype = 'i4').tofile(f)
            fr.tofile(f)
            gFF.tofile(f)
            gE.tofile(f)
            gI.tofile(f)
            w.tofile(f)
            v.tofile(f)
            depC.tofile(f)
            cFF.tofile(f)
            cE.tofile(f)
            cI.tofile(f)
            cGap.tofile(f)

    
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
    def get_FreqComp(data, ifreq):
        if ifreq == 0:
            raise Exception('just use mean for zero comp')
        ndata = len(data)
        Fcomp = np.sum(data * np.exp(-2*np.pi*1j*ifreq*np.arange(ndata)/ndata))/ndata
        return np.array([np.abs(Fcomp)*2, np.angle(Fcomp, deg = True)])

    edges = np.arange(TFbins+1) * dtTF
    t_tf = (edges[:-1] + edges[1:])/2
    
    F0 = np.zeros(nV1)
    F1 = np.zeros((nV1, 2))
    F2 = np.zeros((nV1, 2))
    if plotTempMod and pSample and not collectMeanDataOnly:
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
            if plotTempMod and pSample and not collectMeanDataOnly and i in sample:
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
    if plotTempMod and pSample and not collectMeanDataOnly:
        sfig.savefig(fig_fdr+output_suffix + 'V1-sample_TF' + '.png')
        plt.close(sfig)
    
    F1F0 = np.zeros(F1.shape[0])
    F0_0 = F0 > 0
    F1F0[F0_0] = F1[F0_0,0]/F0[F0_0]

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

    data = per_gE
    if nsmooth > 1:
        target = movingAvg(data, gTFbins, nsmooth)
    else:
        target = data
    gE_F0 = np.mean(target, axis = -1).T
    gE_F1 = np.zeros((nV1, 2))
    for i in range(nV1):
        gE_F1[i,:] = get_FreqComp(target[i,:], 1)
    
    gE_F0_0 = gE_F0 > 0
    gE_F1F0 = np.zeros(nV1)
    tF1 = gE_F1[:,0]
    gE_F1F0[gE_F0_0] = tF1[gE_F0_0]/gE_F0[gE_F0_0]

    data = per_gI
    if nsmooth > 1:
        target = movingAvg(data, gTFbins, nsmooth)
    else:
        target = data
    gI_F0 = np.mean(target, axis = -1).T
    gI_F1 = np.zeros((nV1, 2))
    for i in range(nV1):
        gI_F1[i,:] = get_FreqComp(target[i,:], 1)
    
    gI_F0_0 = gI_F0 > 0
    gI_F1F0 = np.zeros(nV1)
    tF1 = gI_F1[:,0]
    gI_F1F0[gI_F0_0] = tF1[gI_F0_0]/gI_F0[gI_F0_0]

    with open(meanFn, 'ab') as f:
        F1F0.tofile(f)
        gFF_F1F0.tofile(f)
        gE_F1F0.tofile(f)
        gI_F1F0.tofile(f)

    if collectMeanDataOnly:
        return None

    if plotTempMod:
        sc_range = np.linspace(0,2,21)
        phase_range = np.linspace(-180, 180, 33)
        
        if OPstatus != 2: 
            fig = plt.figure(f'F1F0-stats', dpi = 600)
    
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
            fig.savefig(fig_fdr+output_suffix + 'V1-F1F0-stats' + '.png')
            plt.close(fig)
    
        if OPstatus != 0:
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

            fign = 'V1_OP-F1F0'
            if usePrefData:
                fign = fign + '_pref'

            fig.savefig(fig_fdr+output_suffix + fign + '.png')
            plt.close(fig)
    
        if OPstatus != 2:
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
            
            fig.savefig(fig_fdr+output_suffix + 'V1-F2F0' + '.png')
            plt.close(fig)
    
        if pCond:
            if pSample:
                sfig = plt.figure(f'gFF_TF-sample', dpi = 600, figsize = [5.0, ns])
                grid = gs.GridSpec(ns, 2, figure = sfig, hspace = 0.2)
                for i in range(ns):
                    if nLGN_V1[sample[i]] > 0:
                        ax = sfig.add_subplot(grid[i,0])
                        ax.plot(t_gtf, per_gFF[sample[i],:], '-k', lw = 0.5)
                        ax.plot(t_gtf[-1], np.sum(gFF[:,sample[i],0]), '*r')
                        if nsmooth > 1:
                            data = movingAvg(per_gFF, gTFbins, nsmooth)
                            ax.plot(t_gtf, data[sample[i],:], '-g', lw = 0.5)
                        ax.set_ylim(bottom = 0)
                        ax.set_title(f'F1/F0 = {gFF_F1F0[sample[i]]:.2f}')
                        ax.set_xlabel('time(ms)')
                        ax = sfig.add_subplot(grid[i,1])
                        amp = np.abs(np.fft.rfft(per_gFF[sample[i],:]))/gTFbins
                        amp[1:] = amp[1:]*2
                        ff = np.arange(gTFbins//2+1) * TF
                        ipre = np.argwhere(ff<TF)[-1][0]
                        ipost = np.argwhere(ff>=TF)[0][0]
                        ax.plot(ff, amp, 'g', lw = 0.5, alpha = 0.5)
                        ax.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5)
                        ax.set_title(f'gFF {(sample[i]//blockSize,np.mod(sample[i], blockSize))}')
                        ax.set_yscale('log')
                        ax.set_xlabel('Hz')
                sfig.savefig(fig_fdr+output_suffix + 'gFF_TF-sample'+'.png')
                plt.close(sfig)
        
            if OPstatus != 2:
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
        
                fig.savefig(fig_fdr+output_suffix + 'gFF-TFstat' + '.png')
                plt.close(fig)
        
            if OPstatus != 0:
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
        
                fig.savefig(fig_fdr+output_suffix + 'OPgFF-TFstat' + '.png')
                plt.close(fig)
    
    if plotSample:
        i_gap = 0
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
            tsp0 = np.array(spScatter[iV1])
            tsp = tsp0[np.logical_and(tsp0>=step0*dt, tsp0<(nt_+step0)*dt)]
            itype = np.nonzero(np.mod(iV1, blockSize) < typeAcc)[0][0]
            subrange = vT[itype] - vR[itype]
            _vThres = min(vR[itype] + subrange*2, np.max(_v[i,:])) 
            ax.plot(tsp, np.zeros(len(tsp))+_vThres, 'sk', ms = 0.5, mec = 'k', mfc = 'k', mew = 0.1)
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

            frTmp = LGN_fr[:, LGN_V1_ID[iV1]]
            LGN_fr_sSum = np.sum(frTmp[tpick,:] * LGN_V1_s[iV1], axis=-1)
            if np.max(LGN_fr_sSum) > 0:
                ax2.plot(t, LGN_fr_sSum/np.max(LGN_fr_sSum), '-g', lw = 2*lw)
            else:
                ax2.plot(t, np.zeros(LGN_fr_sSum.shape), '-g', lw = 2*lw)
            ax2.set_ylabel('conductance', fontsize = 'x-small')
            ax.set_ylabel('voltage', fontsize = 'x-small')

            if pH:
                for ig in range(ngFF):
                    ax2.plot(t, _gFF[1,ig,i,:], ':g', lw = (ig+1)/ngFF * lw)
                for ig in range(ngE):
                    ax2.plot(t, _gE[1,ig,i,:], ':r', lw = (ig+1)/ngE * lw)
                for ig in range(ngI):
                    ax2.plot(t, _gI[1,ig,i,:], ':b', lw = (ig+1)/ngI * lw)
    
            ax.set_title(f'ID: {(iblock, ithread)}:({LR[iV1]:.0f},{OP[iV1]*180/np.pi:.0f})- LGN:{nLGN_V1[iV1]}({np.sum(LGN_V1_s[iV1]):.1f}), E{preN[0,iV1]}({preNS[0,iV1]:.1f}), I{preN[1,iV1]}({preNS[1,iV1]:.1f})', fontsize = 'small')
            _vBot = min(vR[itype], np.min(_v[i,:]))
            ax.set_ylim(bottom = _vBot)
            ax.set_ylim(top = _vBot + (_vThres - _vBot) * 1.1)
            ax.tick_params(axis='both', labelsize='xx-small', direction='in')
            ax2.tick_params(axis='both', labelsize='xx-small', direction='in')

            ax2.set_ylim(bottom = 0)
            if min(_v[i,:]) > -100 and max(_v[i,:]) < 50:
                ax.yaxis.grid(True, which='minor', linestyle=':', linewidth = 0.1)
                ax.yaxis.grid(True, which='major', linestyle='-', linewidth = 0.1)
                ax.tick_params(which='minor', width=0.2)
                ax.yaxis.set_major_locator(MultipleLocator(5))
                ax.yaxis.set_minor_locator(MultipleLocator(1.0))
            ax.set_xlim(t[0],t[-1])
    
            ax = fig.add_subplot(grid[1,:])
            current = np.zeros(_v[i,:].shape)
            cL = -_gL[iV1]*(_v[i,:]-vL)
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
            if pGap and itype >= nTypeE:
                cGap = -_cGap[i_gap,:]
                ax.plot(t, cGap, ':k', lw = lw)
                ax.plot(t[-1], np.mean(cGap), 'sk', ms = lw)
                current = current + cGap
                i_gap = i_gap+1
    
            ax.plot(t, _depC[i,:], '-y', lw = lw)
            ax.plot(t[-1], np.mean(_depC[i,:]), '*y', ms = lw)
            current = current + _depC[i,:]
    
            if iModel == 1:
                if pW:
                    ax.plot(t, -_w[i,:], '-m', lw = lw)
                    ax.plot(t[-1], -np.mean(_w[i,:]), '*m', ms = lw)
                    current = current - _w[i,:]
    
            ax.plot(t, current, '-k', lw = lw)
            ax.plot(t, np.zeros(t.shape), ':k', lw = lw/2)
            mean_current = np.mean(current)
            ax.plot(t[-1], mean_current, '*k', ms = lw)
            if nLGN_V1[iV1] > 0:
                title = f'FR:{fr[iV1]:.3f}, F1F0:{F1F0[iV1]:.3f}, {gFF_F1F0[iV1]:.3f}(gFF)'
            else:
                title = f'FR:{fr[iV1]:.3f}, F1F0:{F1F0[iV1]:.3f}'
            ax.set_title(title, fontsize = 'small')
            ax.set_ylabel('current', fontsize = 'small')
            ax.tick_params(axis='both', labelsize='xx-small', direction='in')

            ax.yaxis.grid(True, which='minor', linestyle=':', linewidth = 0.1)
            ax.yaxis.grid(True, which='major', linestyle='-', linewidth = 0.1)
            ax.tick_params(which='minor', width=0.2)
            ax.set_xlim(t[0],t[-1])
    
            if nLGN_V1[iV1] > 0:

                if LR[iV1] > 0:
                    all_pos = LGN_vpos[:,nLGN_I:nLGN]
                    all_type = LGN_type[nLGN_I:nLGN]
                else:
                    all_pos = LGN_vpos[:,:nLGN_I]
                    all_type = LGN_type[:nLGN_I]

                print(f'LGN id for neuron {iV1}: {LGN_V1_ID[iV1]}')
                ax = fig.add_subplot(grid[2,0])
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
                ax.tick_params(axis='both', labelsize='xx-small', direction='in')
                ax2.tick_params(axis='both', labelsize='xx-small', direction='in')
    
                ax = fig.add_subplot(grid[2,1])
                        #   L-on L-off M-on  M-off  On   Off
                markers = ('^r', 'vg', '*g', 'dr', '^k', 'vb')
                iLGN_vpos = LGN_vpos[:, LGN_V1_ID[iV1]]
                iLGN_type = LGN_type[LGN_V1_ID[iV1]]
    
                ms = 1.0

                ax.plot(vx[iV1], vy[iV1], '*k', ms = ms)
                ax.plot(cx[iV1], cy[iV1], 'sk', ms = ms)

                iLGN_v1_s = LGN_V1_s[iV1]
                max_s = np.max(iLGN_v1_s)
                min_s = np.max([np.min(iLGN_v1_s), ms*0.25])
                if max_s == 0:
                    ms *= 0.25
                    max_s = 1 
                    min_s = 1
                for j in range(len(markers)):
                    pick = all_type == j
                    ax.plot(all_pos[0,pick], all_pos[1,pick], markers[j], ms = min_s/max_s*ms, mew = 0.0, alpha = 0.5)

                for j in range(nLGN_V1[iV1]):
                    jtype = iLGN_type[j]
                    ax.plot(iLGN_vpos[0,j], iLGN_vpos[1,j], markers[jtype], ms = iLGN_v1_s[j]/max_s*ms, mew = ms*0.5)

                orient = OP[iV1] + np.pi/2
                if usePrefData:
                    orient0 = OP_preset[iV1] + np.pi/2

                if usePrefData:
                    bx, by = ellipse(vx[iV1], vy[iV1], a[iV1], baRatio[iV1], orient0)
                else:
                    bx, by = ellipse(vx[iV1], vy[iV1], a[iV1], baRatio[iV1], orient)

                ax.plot(bx, by, '-b', lw = 0.1)

                x = np.array([np.min([np.min(iLGN_vpos[0,:]), vx[iV1], np.min(bx)]), np.max([np.max(iLGN_vpos[0,:]), vx[iV1], np.max(bx)])])
                y = np.array([np.min([np.min(iLGN_vpos[1,:]), vy[iV1], np.min(by)]), np.max([np.max(iLGN_vpos[1,:]), vy[iV1], np.max(by)])])
                x[1] = np.max([x[1], np.max(bx)])
                x[0] = np.min([x[0], np.min(bx)])
                y[1] = np.max([y[1], np.max(by)])
                y[0] = np.min([y[0], np.min(by)])
                ax.set_xlim(left = x[0] - (x[1]-x[0])*0.1, right = x[1] + (x[1]-x[0])*0.1)
                ax.set_ylim(bottom = y[0] - (y[1]-y[0])*0.1, top = y[1] + (y[1]-y[0])*0.1)
                if usePrefData:
                    x0 = x.copy()
                    y0 = y.copy()

                if np.tan(orient) > 1:
                    x = (y-cy[iV1]) / np.tan(orient) + cx[iV1]
                else:
                    y = np.tan(orient)*(x-cx[iV1]) + cy[iV1]

                #if RFphase[iV1] > np.pi
                #    phase = RFphase[iV1] - 2*np.pi
                #if RFphase[iV1] < -np.pi
                #    phase = RFphase[iV1] + 2*np.pi
                #r_phase = a[iV1]*(phase/np.pi)
                #if usePrefData:
                #    ax.plot(x, y, '-k', lw = 0.1)
                #    ax.plot(cx0[iV1] + rphase*np.cos(OP[iV1]), cy0[iV1] + rphase*np.sin(OP[iV1]), '*m')
                #else:
                #    ax.plot(x, y, ':k', lw = 0.1)
                #    ax.plot(cx0[iV1] + rphase*np.cos(OP[iV1]), cy0[iV1] + rphase*np.sin(OP[iV1]), '*m')
                if usePrefData:
                    ax.plot(x, y, '-k', lw = 0.1)
                else:
                    ax.plot(x, y, ':k', lw = 0.1)

                if usePrefData:
                    if np.diff(y)[0] > np.diff(x)[0]:
                        x0 = (y0-cy[iV1]) / np.tan(orient0) + cx[iV1]
                    else:
                        y0 = np.tan(orient0)*(x0-cx[iV1]) + cy[iV1]
                        ax.plot(x0, y0, ':b', lw = 0.15)
                
                ax.set_aspect('equal')
                ax.set_xlabel('deg', fontsize = 'x-small')
                ax.set_ylabel('deg', fontsize = 'x-small')
                ax.tick_params(axis='both', labelsize='xx-small', direction='in')

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
                        ax.plot(iLGN_vpos[0,pick], iLGN_vpos[1,pick], markers[k], ms = 0.1)
                    ax.plot(iLGN_vpos[0,j], iLGN_vpos[1,j], '*k')
                    ax.set_aspect('equal')

            fig.savefig(fig_fdr+output_suffix + f'V1-sample-{iblock}-{ithread}#{nLGN_V1[iV1]}' + '.png')
            plt.close(fig)
    
    # plot depC distribution over orientation
    if plotDepC and (OPstatus != 2 or usePrefData):
        fig = plt.figure(f'plotDepC', figsize = (8,4), dpi = 600)

        eSpick = epick[nLGN_V1[epick] > SCsplit]
        eCpick = epick[nLGN_V1[epick] <= SCsplit]
        iSpick = ipick[nLGN_V1[ipick] > SCsplit]
        iCpick = ipick[nLGN_V1[ipick] <= SCsplit]

        target = OP*180/np.pi
        ytarget = depC[:,0]
        ax = fig.add_subplot(221)
        pick =  eSpick
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax = fig.add_subplot(222)
        pick =  eCpick
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax = fig.add_subplot(223)
        pick =  iSpick
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax = fig.add_subplot(224)
        pick =  iCpick
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        fign = 'depC'
        if usePrefData:
            fign = fign + '_pref'
        fig.savefig(fig_fdr+output_suffix + fign + '.png')

    # statistics
    if plotRpStat:
        if OPstatus != 2:
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
                ax.plot(0, nzero, 'sr')
                ax.plot(np.mean(fr[eSpick]), nzero, '*r')
            nzero = sum(fr[iSpick] == 0)
            if nzero > 0:
                ax.plot(0, nzero, 'sb')
                ax.plot(np.mean(fr[iSpick]), nzero, '*b')
            nzero = sum(fr[eCpick] == 0)
            if nzero > 0:
                ax.plot(0, nzero, 'sm')
                ax.plot(np.mean(fr[eCpick]), nzero, '*m')
            nzero = sum(fr[iCpick] == 0)
            if nzero > 0:
                ax.plot(0, nzero, 'sg')
                ax.plot(np.mean(fr[iCpick]), nzero, '*g')
    
            ax.set_title('fr')
            ax.set_xlabel('Hz')
            ax.legend()
            
            target = np.sum(gFF[:,:,0], axis = 0)
            ax = fig.add_subplot(222)
            ax.hist(target[eSpick], bins = 12, color = 'r', alpha = 0.5)
            ax.hist(target[iSpick], bins = 12, color = 'b', alpha = 0.5)
            ax.set_title('gFF')
            
            target = np.sum(gE[:,:,0], axis = 0)
            ax = fig.add_subplot(223)
            #_, bin_edges = np.histogram(target[np.hstack((epick, ipick))], bins = 12)

            #ax.hist(target[eSpick], bins = bin_edges, color = 'r', alpha = 0.5)
            #ax.hist(target[iSpick], bins = bin_edges, color = 'b', alpha = 0.5)
            #ax.hist(target[eCpick], bins = bin_edges, color = 'm', alpha = 0.5)
            #ax.hist(target[iCpick], bins = bin_edges, color = 'g', alpha = 0.5)

            ax.hist(target[eSpick], bins = 12, color = 'r', alpha = 0.5)
            ax.hist(target[iSpick], bins = 12, color = 'b', alpha = 0.5)
            ax.hist(target[eCpick], bins = 12, color = 'm', alpha = 0.5)
            ax.hist(target[iCpick], bins = 12, color = 'g', alpha = 0.5)
            ax.set_title('gE')
            
            target = np.sum(gI[:,:,0], axis = 0)
            ax = fig.add_subplot(224)
            ax.hist(target[eSpick], bins = 12, color = 'r', alpha = 0.5)
            ax.hist(target[iSpick], bins = 12, color = 'b', alpha = 0.5)
            ax.hist(target[eCpick], bins = 12, color = 'm', alpha = 0.5)
            ax.hist(target[iCpick], bins = 12, color = 'g', alpha = 0.5)
            ax.set_title('gI')
    
            fig.savefig(fig_fdr+output_suffix + 'V1-rpStats' + '.png')
            plt.close(fig)
    
        if OPstatus != 0:
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
                ax.plot(0, nzero, 'sr')
                ax.plot(np.mean(fr[eSpick]), nzero, '*r')
            nzero = sum(fr[iSpick] == 0)
            if nzero > 0:
                ax.plot(0, nzero, 'sb')
                ax.plot(np.mean(fr[iSpick]), nzero, '*b')
            nzero = sum(fr[eCpick] == 0)
            if nzero > 0:
                ax.plot(0, nzero, 'sm')
                ax.plot(np.mean(fr[eCpick]), nzero, '*m')
            nzero = sum(fr[iCpick] == 0)
            if nzero > 0:
                ax.plot(0, nzero, 'sg')
                ax.plot(np.mean(fr[iCpick]), nzero, '*g')
    
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
            #ax.hist(target[eSpick], bins = bin_edges, color = 'r', alpha = 0.5)
            #ax.hist(target[iSpick], bins = bin_edges, color = 'b', alpha = 0.5)
            #ax.hist(target[eCpick], bins = bin_edges, color = 'm', alpha = 0.5)
            #ax.hist(target[iCpick], bins = bin_edges, color = 'g', alpha = 0.5)
            ax.hist(target[eSpick], bins = 12, color = 'r', alpha = 0.5)
            ax.hist(target[iSpick], bins = 12, color = 'b', alpha = 0.5)
            ax.hist(target[eCpick], bins = 12, color = 'm', alpha = 0.5)
            ax.hist(target[iCpick], bins = 12, color = 'g', alpha = 0.5)
            ax.set_title('gE')
            
            target = np.sum(gI[:,:,0], axis = 0)
            ax = fig.add_subplot(224)
            _, bin_edges = np.histogram(target[np.hstack((eORpick, iORpick))], bins = 12)
            #ax.hist(target[eSpick], bins = bin_edges, color = 'r', alpha = 0.5)
            #ax.hist(target[iSpick], bins = bin_edges, color = 'b', alpha = 0.5)
            #ax.hist(target[eCpick], bins = bin_edges, color = 'm', alpha = 0.5)
            #ax.hist(target[iCpick], bins = bin_edges, color = 'g', alpha = 0.5)
            ax.hist(target[eSpick], bins = 12, color = 'r', alpha = 0.5)
            ax.hist(target[iSpick], bins = 12, color = 'b', alpha = 0.5)
            ax.hist(target[eCpick], bins = 12, color = 'm', alpha = 0.5)
            ax.hist(target[iCpick], bins = 12, color = 'g', alpha = 0.5)
            ax.set_title('gI')
    
            fign = 'V1_OP-rpStats'
            if usePrefData:
                fign = fign + '_pref'

            fig.savefig(fig_fdr+output_suffix + fign + '.png')
            plt.close(fig)
    
    if plotRpCorr and (OPstatus != 2 or usePrefData):
        fig = plt.figure(f'rpCorr', figsize = (18,12), dpi = 600)
        grid = gs.GridSpec(4, 6, figure = fig, hspace = 0.3, wspace = 0.3)
        target = np.sum(gE_gTot_ratio[:,:,0], axis = 0)
    
        ax = fig.add_subplot(grid[0,0])
        image = HeatMap(target[epick], fr[epick], heatBins, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('gExc/gTot')
        ax.set_ylabel('Exc. FR Hz')
        old_target = target.copy()
        
        ax = fig.add_subplot(grid[0,1])
        image = HeatMap(target[ipick], fr[ipick], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('gExc/gTot')
        ax.set_ylabel('Inh. FR Hz')
    
        target = F1F0
        ax = fig.add_subplot(grid[0,2])
        pick = epick[np.logical_and(nLGN_V1[epick]>0, F0_0[epick])]
        image = HeatMap(target[pick], fr[pick], F1F0range, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('F1F0')
        ax.set_ylabel('ExcS. FR Hz')
    
        ax = fig.add_subplot(grid[0,3])
        pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0_0[ipick])]
        image = HeatMap(target[pick], fr[pick], F1F0range, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('F1F0')
        ax.set_ylabel('InhS. FR Hz')
    
        target = gFF_F1F0
        ax = fig.add_subplot(grid[1,0])
        pick = epick[np.logical_and(nLGN_V1[epick]>SCsplit,gFF_F0_0[epick])]
        if np.sum(nLGN_V1[epick]>SCsplit) > 0:
            active = np.sum(fr[pick]>0)/np.sum(nLGN_V1[epick]>SCsplit)
            image = HeatMap(target[pick], fr[pick], F1F0range, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
            ax.set_title(f'active simpleE {active*100:.3f}%')
        else:
            ax.set_title('no active simpleE')
        ax.set_xlabel('gFF_F1/F0')
        ax.set_ylabel('ExcS FR')
    
        ax = fig.add_subplot(grid[1,1])
        pick = ipick[np.logical_and(nLGN_V1[ipick]>SCsplit,gFF_F0_0[ipick])]
        if np.sum(nLGN_V1[ipick]>SCsplit) > 0:
            active = np.sum(fr[pick]>0)/np.sum(nLGN_V1[ipick]>SCsplit)
            image = HeatMap(target[pick], fr[pick], F1F0range, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
            ax.set_title(f'active simpleI {active*100:.3f}%')
        else:
            ax.set_title('no active simpleI')
        ax.set_xlabel('gFF_F1/F0')
        ax.set_ylabel('InhS FR')
    
        target = np.sum(gE[:,:,0], axis = 0)
        ax = fig.add_subplot(grid[1,2])
        pick = epick[nLGN_V1[epick]<=SCsplit]
        if np.sum(nLGN_V1[epick]<=SCsplit) > 0:
            active = np.sum(fr[pick]>0)/np.sum(nLGN_V1[epick]<=SCsplit)
            image = HeatMap(target[pick], fr[pick], heatBins, heatBins, ax, 'Reds', log_scale = pLog)
            ax.set_title(f'active complexE {active*100:.3f}%')
        else:
            ax.set_title('no ative complexE')
        ax.set_xlabel('gE')
        ax.set_ylabel('ExcC FR')
    
        ax = fig.add_subplot(grid[1,3])
        pick = ipick[nLGN_V1[ipick]>=SCsplit]
        if np.sum(nLGN_V1[ipick]>=SCsplit) > 0:
            active = np.sum(fr[pick]>0)/np.sum(nLGN_V1[ipick]>=SCsplit)
            image = HeatMap(target[pick], fr[pick], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
            ax.set_title(f'active complexI {active*100:.3f}%')
        else:
            ax.set_title(f'no active complexI')
        ax.set_xlabel('gE')
        ax.set_ylabel('InhC FR')

    
        target = np.sum(gFF[:,:,0], axis = 0)
        ax = fig.add_subplot(grid[1,4])
        image = HeatMap(target[epick], fr[epick], heatBins, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('gFF')
        ax.set_ylabel('Exc FR')
        ax.set_title(f'active {np.sum(fr[epick]>0)/epick.size*100:.3f}%')
    
        ax = fig.add_subplot(grid[1,5])
        image = HeatMap(target[ipick], fr[ipick], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('gFF')
        ax.set_ylabel('Inh FR')
        ax.set_title(f'active {np.sum(fr[ipick]>0)/ipick.size*100:.3f}%')
    
        target = OP*180/np.pi
        ax = fig.add_subplot(grid[2,0])
        if nLGNorF1F0:
            pick = epick[nLGN_V1[epick]>SCsplit]
            ax.set_ylabel('S_sc. Exc FR')
        else:
            pick = epick[F1F0[epick]>1]
            ax.set_ylabel('S_f1. Exc FR')
        image = HeatMap(target[pick], fr[pick], OPrange, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
    
        ax = fig.add_subplot(grid[2,1])
        #pick = ipick[nLGN_V1[ipick]>=SCsplit]
        if nLGNorF1F0:
            pick = ipick[nLGN_V1[ipick]>SCsplit]
            ax.set_ylabel('S_sc. Inh FR')
        else:
            pick = ipick[F1F0[ipick]>1]
            ax.set_ylabel('S_f1. Inh FR')
        image = HeatMap(target[pick], fr[pick], OPrange, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
    
        ax = fig.add_subplot(grid[2,2])
        #pick = epick[nLGN_V1[epick]<SCsplit]
        if nLGNorF1F0:
            pick = epick[nLGN_V1[epick]<=SCsplit]
            ax.set_ylabel('C_sc. Exc FR')
        else:
            pick = epick[F1F0[epick]<=1]
            ax.set_ylabel('C_f1. Exc FR')
        image = HeatMap(target[pick], fr[pick], OPrange, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
    
        ax = fig.add_subplot(grid[2,3])
        #pick = ipick[nLGN_V1[ipick]<SCsplit]
        if nLGNorF1F0:
            pick = ipick[nLGN_V1[ipick]<=SCsplit]
            ax.set_ylabel('C_sc. Inh FR')
        else:
            pick = ipick[F1F0[ipick]<=1]
            ax.set_ylabel('C_f1. Inh FR')
        image = HeatMap(target[pick], fr[pick], OPrange, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
    
        ytarget = gFF_F1F0
        ax = fig.add_subplot(grid[3,0])
        pick = epick[np.logical_and(nLGN_V1[epick]>0, gFF_F0_0[epick])]
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('Exc. gFF_F1F0')
    
        ax = fig.add_subplot(grid[3,1])
        pick = ipick[np.logical_and(nLGN_V1[ipick]>0, gFF_F0_0[ipick])]
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('Inh. gFF_F1F0')
    
        ytarget = F1F0
        ax = fig.add_subplot(grid[3,2])
        pick = epick[np.logical_and(nLGN_V1[epick]>0, F0_0[epick])]
        image = HeatMap(target[pick], ytarget[pick], OPrange, F1F0range, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('Exc. F1F0')
    
        ax = fig.add_subplot(grid[3,3])
        pick = ipick[np.logical_and(nLGN_V1[ipick]>0, F0_0[ipick])]
        image = HeatMap(target[pick], ytarget[pick], OPrange, F1F0range, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('Inh. F1F0')

        target = OP*180/np.pi
        ytarget = np.sum(gI[:,:,0], axis = 0)
        ax = fig.add_subplot(grid[2,4])
        pick = epick[nLGN_V1[epick]>0]
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('S. Exc gI')
    
        ax = fig.add_subplot(grid[2,5])
        pick = ipick[nLGN_V1[ipick]>0]
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('S. Inh gI')

        ax = fig.add_subplot(grid[3,4])
        pick = epick[nLGN_V1[epick]==0]
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('C. Exc gI')
    
        ax = fig.add_subplot(grid[3,5])
        pick = ipick[nLGN_V1[ipick]==0]
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('C. Inh gI')

        target = OP*180/np.pi
        ytarget = np.sum(gE[:,:,0], axis = 0)
        ax = fig.add_subplot(grid[0,4])
        pick = epick[nLGN_V1[epick]==0]
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Reds', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('C. Exc gE')
    
        ax = fig.add_subplot(grid[0,5])
        pick = ipick[nLGN_V1[ipick]==0]
        image = HeatMap(target[pick], ytarget[pick], OPrange, heatBins, ax, 'Blues', log_scale = pLog, intPick = False, tickPick1 = 5)
        ax.set_xlabel('OP')
        ax.set_ylabel('C. Inh gE')
        
        fign = 'V1-rpCorr'
        if usePrefData:
            fign = fign + '_pref'

        fig.savefig(fig_fdr+output_suffix + fign + '.png')
        plt.close(fig)
    
    if plotLR_rp and OPstatus != 2:
        fig = plt.figure(f'LRrpStat', dpi = 600)
        ax = fig.add_subplot(221)
        ax.hist(fr[LR>0], bins = 12, color = 'r', log = True, alpha = 0.5, label = 'Contra')
        ax.plot(0, sum(fr[LR>0] == 0), '*r')
        ax.hist(fr[LR<0], bins = 12, color = 'b', log = True, alpha = 0.5, label = 'Ipsi')
        ax.plot(0, sum(fr[LR<0] == 0), '*b')
        ax.set_title('fr')
        ax.set_xlabel('Hz')
        ax.legend()
    
        target = np.sum(gFF[:,:,0], axis = 0)
        ax = fig.add_subplot(222)
        ax.hist(target[LR>0], bins = 12, color = 'r', alpha = 0.5)
        ax.hist(target[LR<0], bins = 12, color = 'b', alpha = 0.5)
        ax.set_title('gFF')
        
        target = np.sum(gE[:,:,0], axis = 0)
        ax = fig.add_subplot(223)
        ax.hist(target[LR>0], bins = 12, color = 'r', alpha = 0.5)
        ax.hist(target[LR<0], bins = 12, color = 'b', alpha = 0.5)
        ax.set_title('gE')
        
        target = np.sum(gI[:,:,0], axis = 0)
        ax = fig.add_subplot(224)
        ax.hist(target[LR>0], bins = 12, color = 'r', alpha = 0.5)
        ax.hist(target[LR<0], bins = 12, color = 'b', alpha = 0.5)
        ax.set_title('gI')
        fig.savefig(fig_fdr+output_suffix + 'V1-LRrpStats' + '.png')
        plt.close(fig)
    
    if plotExc_sLGN and OPstatus != 2:
        FF_irl = np.sum(gFF[:,:,0], axis = 0)#
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
    
        fig.savefig(fig_fdr+output_suffix + 'Exc_sLGN' + '.png')
        plt.close(fig)
            
    # scatter
    if plotScatterFF:
        if OPstatus != 2 or usePrefData:
            fig = plt.figure(f'scatterFF', figsize = (12,6), dpi = 1200)
            ax = fig.add_subplot(221)
            ax3 = fig.add_subplot(223)
            tsp = np.hstack([x for x in spScatter])
            isp = np.hstack([ix + np.zeros(len(spScatter[ix])) for ix in np.arange(nV1)])
            tpick = np.logical_and(tsp>=step0*dt, tsp<(step0+nt_)*dt)
            tsp = tsp[tpick]
            isp = isp[tpick].astype(int)
        
            ax.set_xlim(step0*dt, (step0+nt_)*dt)
            ax3.set_xlim(step0*dt, (step0+nt_)*dt)
        
            ax2 = fig.add_subplot(224)
            #ax2_ = ax2.twinx()
            t = nt_ *dt #ms
            nbins = np.int(np.round(t/tbinSize)) #every 1ms
            edges = step0*dt + np.arange(nbins+1) * tbinSize 
            t_tf = (edges[:-1] + edges[1:])/2
            ff = np.arange(nbins//2+1) * 1000/t
        
            ipre = np.argwhere(ff<TF)[-1][0]
            ipost = np.argwhere(ff>=TF)[0][0]
            nsmoothFr = 10 
            nsmoothFreq = 1 
        
            red = np.array([0.0, 1, 1])
            blue = np.array([0.666666, 1, 1])
            magenta = np.array([0.8333333, 1, 1])
            green = np.array([0.333333, 1, 0.5])
            ms1 = 0.16
            ms2 = 0.04
            mk1 = '.'
            mk2 = 's'
            sat0 = 0.25
            ax11 = fig.add_subplot(243)
            ax12 = fig.add_subplot(244)

            pick = epick[LR[epick] > 0]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax11.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            pick = ipick[LR[ipick] > 0]
            nnI = pick.size
            if nnI > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax11.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            if pLR:
                mode = 'LR'
                eSpick = np.logical_and(np.mod(isp, blockSize) < nE , LR[isp]>0)
                iSpick = np.logical_and(np.mod(isp, blockSize) >= nE, LR[isp]>0)
                ax.plot(tsp[eSpick], isp[eSpick], ',r')
                ax.plot(tsp[iSpick], isp[iSpick], ',b')
        
                if np.sum(eSpick) > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'r', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5, label = 'exc R')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'r', lw = 0.5, alpha = 0.5)
        
                if np.sum(iSpick) > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'b', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh R')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'b', lw = 0.5, alpha = 0.5)

            pick = epick[LR[epick] < 0]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax11.scatter(pos[0,pick], pos[1,pick], s = ms1, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk1)

            pick = ipick[LR[ipick] < 0]
            nnI = pick.size
            if nnI > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax11.scatter(pos[0,pick], pos[1,pick], s = ms1, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk1)

            if pLR:
                eSpick = np.logical_and(np.mod(isp, blockSize) < nE , LR[isp]<0)
                iSpick = np.logical_and(np.mod(isp, blockSize) >= nE, LR[isp]<0)
                ax.plot(tsp[eSpick], isp[eSpick], ',m')
                ax.plot(tsp[iSpick], isp[iSpick], ',g')
        
                if np.sum(eSpick) > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(np.sum(LR[epick]<0)*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'm', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':m', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>m', ms = 0.5, alpha = 0.5, label = 'exc L')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'm', lw = 0.5, alpha = 0.5)
        
                if np.sum(iSpick) > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(np.sum(LR[ipick]<0)*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'g', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':g', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<g', ms = 0.5, alpha = 0.5, label = 'inh L')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'g', lw = 0.5, alpha = 0.5)
        
            pick = epick[nLGN_V1[epick]<=SCsplit]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax12.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            pick = ipick[nLGN_V1[ipick]<=SCsplit]
            nnI = pick.size
            if nnI > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax12.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            if pSC:
                mode = 'SC'
                eSpick = np.logical_and(np.mod(isp, blockSize) < nE , nLGN_V1[isp] == 0)
                iSpick = np.logical_and(np.mod(isp, blockSize) >= nE, nLGN_V1[isp] == 0)
                ax.plot(tsp[eSpick], isp[eSpick], ',m')
                ax.plot(tsp[iSpick], isp[iSpick], ',g')
                if np.sum(eSpick) > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'm', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':m', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>m', ms = 0.5, alpha = 0.5, label = 'exc C')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'm', lw = 0.5, alpha = 0.5)
        
                if np.sum(iSpick) > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'g', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':g', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<g', ms = 0.5, alpha = 0.5, label = 'inh C')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'g', lw = 0.5, alpha = 0.5)

            pick = epick[nLGN_V1[epick] > SCsplit]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax12.scatter(pos[0,pick], pos[1,pick], s = ms1, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk1)

            pick = ipick[nLGN_V1[ipick] > SCsplit]
            nnI = pick.size
            if nnI > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax12.scatter(pos[0,pick], pos[1,pick], s = ms1, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk1)
        
            if pSC:
                eSpick = np.logical_and(np.mod(isp, blockSize) < nE , nLGN_V1[isp] > 0)
                iSpick = np.logical_and(np.mod(isp, blockSize) >= nE, nLGN_V1[isp] > 0)
                ax.plot(tsp[eSpick], isp[eSpick], ',r')
                ax.plot(tsp[iSpick], isp[iSpick], ',b')
        
                if np.sum(eSpick) > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'r', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5, label = 'exc S')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'r', lw = 0.5, alpha = 0.5)
        
                if np.sum(iSpick) > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'b', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh S')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'b', lw = 0.5, alpha = 0.5)

            if not pSC and not pLR:
                mode = 'EI'
                eSpick = np.mod(isp, blockSize) < nE
                iSpick = np.mod(isp, blockSize) >= nE
                ax.plot(tsp[eSpick], isp[eSpick], ',r')
                ax.plot(tsp[iSpick], isp[iSpick], ',b')
            
                nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nblock*nE*tbinSize/1000)
                data = np.fft.rfft(nsp)
                amp = np.abs(data)/nbins
                amp[1:] = amp[1:]*2
                amp = amp*amp
                ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'r', lw = 0.5, alpha = 0.5)
                #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5, label = 'exc')
                ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'r', lw = 0.5, alpha = 0.5)
                nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nblock*nI*tbinSize/1000)
                data = np.fft.rfft(nsp)
                amp = np.abs(data)/nbins
                amp[1:] = amp[1:]*2
                amp = amp*amp
                ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'b', lw = 0.5, alpha = 0.5)
                #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh')
                ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'b', lw = 0.5, alpha = 0.5)
        
            ax2.set_yscale('log')
            ax2.set_xlabel('Hz')
            #ax2_.set_ylim(bottom = -180, top = 180)
            ax11.set_aspect('equal')
            ax11.set_title('OP-LR')
            ax12.set_title('OP-SC')
            ax12.set_aspect('equal')
            ax2.legend(loc = 'best')

            fign = 'V1-scatterFF'
            if usePrefData:
                fign = fign + '_pref'

            if nt == nt_:
                fig.savefig(fig_fdr+output_suffix + fign + '-' + mode + '_full.png')
            else:
                fig.savefig(fig_fdr+output_suffix + fign + '-' + mode + '.png')
            plt.close(fig)

        if OPstatus != 0:
            ### Scatter OP vs non-OP
            fig = plt.figure(f'scatterFF_OP', figsize = (12,6), dpi = 1200)
            ax = fig.add_subplot(221)
            ax3 = fig.add_subplot(223)

            tsp = np.hstack([x for x in spScatter])
            isp = np.hstack([ix + np.zeros(len(spScatter[ix])) for ix in np.arange(nV1)])
            tpick = np.logical_and(tsp>=step0*dt, tsp<(step0+nt_)*dt)
            tsp = tsp[tpick]
            isp = isp[tpick].astype(int)
        
            ax.set_xlim(step0*dt, (step0+nt_)*dt)
            ax3.set_xlim(step0*dt, (step0+nt_)*dt)
        
            ax2 = fig.add_subplot(224)
            #ax2_ = ax2.twinx()
            t = nt_ *dt #ms
            nbins = np.int(np.round(t/tbinSize)) #every 1ms
            edges = step0*dt + np.arange(nbins+1) * tbinSize 
            t_tf = (edges[:-1] + edges[1:])/2
            ff = np.arange(nbins//2+1) * 1000/t
        
            ipre = np.argwhere(ff<TF)[-1][0]
            ipost = np.argwhere(ff>=TF)[0][0]
            nsmoothFr = 10 
            nsmoothFreq = 1 
        
            red = np.array([0.0, 1, 1])
            blue = np.array([0.666666, 1, 1])
            magenta = np.array([0.8333333, 1, 1])
            green = np.array([0.333333, 1, 0.5])
            ms1 = 0.16
            ms2 = 0.04
            mk1 = '.'
            mk2 = 's'
            sat0 = 0.25
            ax11 = fig.add_subplot(243)
            ax12 = fig.add_subplot(244)

            pick = eORpick[LR[eORpick] > 0]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax11.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            pick = iORpick[LR[iORpick] > 0]
            nnI = pick.size
            if nnI > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax11.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            V1_range = np.arange(nV1)
            ## FFT Right eye
            npLR = 0
            if pLR:
                mode = 'LR'
                eSpick = np.logical_and(np.logical_and(np.mod(isp, blockSize) < nE , LR[isp]>0), dOP[isp] <= dOri)
                iSpick = np.logical_and(np.logical_and(np.mod(isp, blockSize) >= nE, LR[isp]>0), dOP[isp] <= dOri)
                eIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) < nE , LR>0), dOP <= dOri)
                iIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) >= nE, LR>0), dOP <= dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + npLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npLR = npLR + nOP
                    ax.plot(tsp[eSpick], isp_OP, ',r')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + npLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npLR = npLR + nOP
                    ax.plot(tsp[iSpick], isp_OP, ',b')

                ax.plot([step0*dt, (step0+nt_)*dt], [npLR, npLR], '-k', lw = 0.05)

                eSpick_bg = np.logical_and(np.logical_and(np.mod(isp, blockSize) < nE , LR[isp]>0), dOP[isp] > dOri)
                iSpick_bg = np.logical_and(np.logical_and(np.mod(isp, blockSize) >= nE, LR[isp]>0), dOP[isp] > dOri)
                eIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) < nE , LR>0), dOP > dOri)
                iIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) >= nE, LR>0), dOP > dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + npLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npLR = npLR + nOP
                    ax.plot(tsp[eSpick_bg], isp_OP, ',r')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + npLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npLR = npLR + nOP
                    ax.plot(tsp[iSpick_bg], isp_OP, ',b')
                ax.plot([step0*dt, (step0+nt_)*dt], [npLR, npLR], '-k', lw = 0.05)
        
                nnE = np.sum(eSpick)
                if  nnE > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'r', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5, label = 'exc R OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'r', lw = 0.5, alpha = 0.5)

                nnE = np.sum(eSpick_bg)
                if nnE > 0:
                    nsp = np.histogram(tsp[eSpick_bg], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), 'vr', ms = 0.5, alpha = 0.5, label = 'exc R non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), ':r', lw = 0.5, alpha = 0.5)
        
                nnI = np.sum(iSpick)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'b', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh R OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'b', lw = 0.5, alpha = 0.5)

                nnI = np.sum(iSpick_bg)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick_bg], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'b', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh R non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'b', lw = 0.5, alpha = 0.5)

            pick = eORpick[LR[eORpick] < 0]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax11.scatter(pos[0,pick], pos[1,pick], s = ms1, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk1)

            pick = iORpick[LR[iORpick] < 0]
            nnI = pick.size
            if nnI > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)
                
                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax11.scatter(pos[0,pick], pos[1,pick], s = ms1, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk1)

            ## FFT Left eye
            if pLR:
                eSpick = np.logical_and(np.logical_and(np.mod(isp, blockSize) < nE , LR[isp]<0), dOP[isp] <= dOri)
                iSpick = np.logical_and(np.logical_and(np.mod(isp, blockSize) >= nE, LR[isp]<0), dOP[isp] <= dOri)
                eIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) < nE , LR<0), dOP <= dOri)
                iIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) >= nE, LR<0), dOP <= dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + npLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npLR = npLR + nOP
                    ax.plot(tsp[eSpick], isp_OP, ',m')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + npLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npLR = npLR + nOP
                    ax.plot(tsp[iSpick], isp_OP, ',g')

                ax.plot([step0*dt, (step0+nt_)*dt], [npLR, npLR], '-k', lw = 0.05)

                eSpick_bg = np.logical_and(np.logical_and(np.mod(isp, blockSize) < nE , LR[isp]<0), dOP[isp] > dOri)
                iSpick_bg = np.logical_and(np.logical_and(np.mod(isp, blockSize) >= nE, LR[isp]<0), dOP[isp] > dOri)
                eIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) < nE , LR<0), dOP > dOri)
                iIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) >= nE, LR<0), dOP > dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + npLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npLR = npLR + nOP
                    ax.plot(tsp[eSpick_bg], isp_OP, ',m')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + npLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npLR = npLR + nOP
                    ax.plot(tsp[iSpick_bg], isp_OP, ',g')
                ax.plot([step0*dt, (step0+nt_)*dt], [npLR, npLR], '-k', lw = 0.05)
        
                nnE = np.sum(eSpick)
                if nnE > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'm', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':m', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>m', ms = 0.5, alpha = 0.5, label = 'exc L OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'm', lw = 0.5, alpha = 0.5)
                    
                nnE = np.sum(eSpick_bg)
                if nnE > 0:
                    nsp = np.histogram(tsp[eSpick_bg], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), ':m', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':m', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), 'vm', ms = 0.5, alpha = 0.5, label = 'exc L non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), ':m', lw = 0.5, alpha = 0.5)

                nnI = np.sum(iSpick)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'g', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':g', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<g', ms = 0.5, alpha = 0.5, label = 'inh L OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'g', lw = 0.5, alpha = 0.5)

                nnI = np.sum(iSpick_bg)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick_bg], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), ':g', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':g', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), 'vg', ms = 0.5, alpha = 0.5, label = 'inh L non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), ':g', lw = 0.5, alpha = 0.5)
        
            pick = eORpick[nLGN_V1[eORpick]<=SCsplit]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax12.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            pick = iORpick[nLGN_V1[iORpick]<=SCsplit]
            nnI = pick.size
            if nnI > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax12.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            npSC = 0
            if pSC:
                mode = 'SC'
                eSpick = np.logical_and(np.logical_and(np.mod(isp, blockSize) < nE , nLGN_V1[isp] <= SCsplit), dOP[isp] <= dOri)
                iSpick = np.logical_and(np.logical_and(np.mod(isp, blockSize) >= nE, nLGN_V1[isp] <= SCsplit), dOP[isp] <= dOri)
                eIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) < nE , nLGN_V1 <= SCsplit), dOP <= dOri)
                iIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) >= nE, nLGN_V1 <= SCsplit), dOP <= dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + npSC + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npSC = npSC + nOP
                    ax.plot(tsp[eSpick], isp_OP, ',m')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + npSC + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npSC = npSC + nOP
                    ax.plot(tsp[iSpick], isp_OP, ',g')

                ax.plot([step0*dt, (step0+nt_)*dt], [npSC, npSC], '-k', lw = 0.05)

                eSpick_bg = np.logical_and(np.logical_and(np.mod(isp, blockSize) < nE , nLGN_V1[isp] <= SCsplit), dOP[isp] > dOri)
                iSpick_bg = np.logical_and(np.logical_and(np.mod(isp, blockSize) >= nE, nLGN_V1[isp] <= SCsplit), dOP[isp] > dOri)
                eIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) < nE , nLGN_V1 <= SCsplit), dOP > dOri)
                iIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) >= nE, nLGN_V1 <= SCsplit), dOP > dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + npSC + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npSC = npSC + nOP
                    ax.plot(tsp[eSpick_bg], isp_OP, ',m')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + npSC + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npSC = npSC + nOP
                    ax.plot(tsp[iSpick_bg], isp_OP, ',g')
                ax.plot([step0*dt, (step0+nt_)*dt], [npSC, npSC], '-k', lw = 0.05)
   
                nnE = np.sum(eSpick)
                if nnE > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'm', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':m', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>m', ms = 0.5, alpha = 0.5, label = 'exc C OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'm', lw = 0.5, alpha = 0.5)
        
                nnE = np.sum(eSpick_bg)
                if nnE > 0:
                    nsp = np.histogram(tsp[eSpick_bg], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), ':m', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':m', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), 'vm', ms = 0.5, alpha = 0.5, label = 'exc C non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), ':m', lw = 0.5, alpha = 0.5)

                nnI = np.sum(iSpick)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'g', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':g', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<g', ms = 0.5, alpha = 0.5, label = 'inh C OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'g', lw = 0.5, alpha = 0.5)

                nnI = np.sum(iSpick_bg)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick_bg], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), ':g', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':g', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), 'vg', ms = 0.5, alpha = 0.5, label = 'inh C non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), ':g', lw = 0.5, alpha = 0.5)

            pick = eORpick[nLGN_V1[eORpick] > SCsplit]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax12.scatter(pos[0,pick], pos[1,pick], s = ms1, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk1)

            pick = iORpick[nLGN_V1[iORpick] > SCsplit]
            nnI = pick.size
            if nnI > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = np.mod(OP[pick]+np.pi/2, np.pi)/np.pi
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax12.scatter(pos[0,pick], pos[1,pick], s = ms1, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk1)
        
            if pSC:
                eSpick = np.logical_and(np.logical_and(np.mod(isp, blockSize) < nE , nLGN_V1[isp] > SCsplit), dOP[isp] <= dOri)
                iSpick = np.logical_and(np.logical_and(np.mod(isp, blockSize) >= nE, nLGN_V1[isp] > SCsplit), dOP[isp] <= dOri)
                eIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) < nE , nLGN_V1 > SCsplit), dOP <= dOri)
                iIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) >= nE, nLGN_V1 > SCsplit), dOP <= dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + npSC + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npSC = npSC + nOP
                    ax.plot(tsp[eSpick], isp_OP, ',r')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + npSC + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npSC = npSC + nOP
                    ax.plot(tsp[iSpick], isp_OP, ',b')
                ax.plot([step0*dt, (step0+nt_)*dt], [npSC, npSC], '-k', lw = 0.05)

                eSpick_bg = np.logical_and(np.logical_and(np.mod(isp, blockSize) < nE , nLGN_V1[isp] > SCsplit), dOP[isp] > dOri)
                iSpick_bg = np.logical_and(np.logical_and(np.mod(isp, blockSize) >= nE, nLGN_V1[isp] > SCsplit), dOP[isp] > dOri)
                eIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) < nE , nLGN_V1 > SCsplit), dOP > dOri)
                iIpick = np.logical_and(np.logical_and(np.mod(V1_range, blockSize) >= nE, nLGN_V1 > SCsplit), dOP > dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + npSC + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npSC = npSC + nOP
                    ax.plot(tsp[eSpick_bg], isp_OP, ',r')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + npSC + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    npSC = npSC + nOP
                    ax.plot(tsp[iSpick_bg], isp_OP, ',b')

                nnE = np.sum(eSpick)
                if nnE > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'r', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5, label = 'exc S OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'r', lw = 0.5, alpha = 0.5)
        
                nnE = np.sum(eSpick_bg)
                if nnE > 0:
                    nsp = np.histogram(tsp[eSpick_bg], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), 'vr', ms = 0.5, alpha = 0.5, label = 'exc S non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), ':r', lw = 0.5, alpha = 0.5)

                nnI = np.sum(iSpick)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'b', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh S OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'b', lw = 0.5, alpha = 0.5)
                    
                nnI = np.sum(iSpick_bg)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick_bg], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), 'vb', ms = 0.5, alpha = 0.5, label = 'inh S non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), ':b', lw = 0.5, alpha = 0.5)

            n_pSC_pLR = 0
            if not pSC and not pLR:
                mode = 'EI'
                eSpick = np.logical_and(np.mod(isp, blockSize) <  nE, dOP[isp] <= dOri)
                iSpick = np.logical_and(np.mod(isp, blockSize) >= nE, dOP[isp] <= dOri)
                eIpick = np.logical_and(np.mod(V1_range, blockSize) < nE , dOP <= dOri)
                iIpick = np.logical_and(np.mod(V1_range, blockSize) >= nE, dOP <= dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + n_pSC_pLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    n_pSC_pLR = n_pSC_pLR + nOP
                    ax.plot(tsp[eSpick], isp_OP, ',r')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + n_pSC_pLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    n_pSC_pLR = n_pSC_pLR + nOP
                    ax.plot(tsp[iSpick], isp_OP, ',b')

                ax.plot([step0*dt, (step0+nt_)*dt], [n_pSC_pLR, n_pSC_pLR], '-k', lw = 0.05)
            
                eSpick_bg = np.logical_and(np.mod(isp, blockSize) <  nE, dOP[isp] > dOri)
                iSpick_bg = np.logical_and(np.mod(isp, blockSize) >= nE, dOP[isp] > dOri)
                eIpick = np.logical_and(np.mod(V1_range, blockSize) < nE , dOP > dOri)
                iIpick = np.logical_and(np.mod(V1_range, blockSize) >= nE, dOP > dOri)

                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[eIpick]]
                nOP = np.sum(eIpick)
                if nOP > 0:
                    isp_OP = [ix + n_pSC_pLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    n_pSC_pLR = n_pSC_pLR + nOP
                    ax.plot(tsp[eSpick_bg], isp_OP, ',r')
                OP_len = [sum(np.logical_and(spScatter[ix]>=step0*dt, spScatter[ix]<(step0+nt_)*dt)) for ix in np.arange(nV1)[iIpick]]
                nOP = np.sum(iIpick)
                if nOP > 0:
                    isp_OP = [ix + n_pSC_pLR + np.zeros(OP_len[ix], dtype = int) for ix in np.arange(nOP)]
                    if nOP > 1:
                        isp_OP = np.hstack(isp_OP)
                    else:
                        isp_OP = np.array(isp_OP)

                    n_pSC_pLR = n_pSC_pLR + nOP
                    ax.plot(tsp[iSpick_bg], isp_OP, ',b')

                nnE = np.sum(eSpick)
                if nnE > 0:
                    nsp = np.histogram(tsp[eSpick], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'r', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '>r', ms = 0.5, alpha = 0.5, label = 'exc OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'r', lw = 0.5, alpha = 0.5)

                nnE = np.sum(eSpick_bg)
                if nnE > 0:
                    nsp = np.histogram(tsp[eSpick_bg], bins = edges)[0]/(nnE*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':r', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), 'vr', ms = 0.5, alpha = 0.5, label = 'exc non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), ':r', lw = 0.5, alpha = 0.5)
                
                nnI = np.sum(iSpick)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), 'b', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), '<b', ms = 0.5, alpha = 0.5, label = 'inh OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), 'b', lw = 0.5, alpha = 0.5)

                nnI = np.sum(iSpick_bg)
                if nnI > 0:
                    nsp = np.histogram(tsp[iSpick_bg], bins = edges)[0]/(nnI*tbinSize/1000)
                    data = np.fft.rfft(nsp)
                    amp = np.abs(data)/nbins
                    amp[1:] = amp[1:]*2
                    amp = amp*amp
                    ax2.plot(ff, movingAvg(amp, amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    #ax2_.plot(ff, movingAvg(np.angle(data, deg = True), amp.size, nsmoothFreq), ':b', lw = 0.5, alpha = 0.5)
                    ax2.plot(TF, amp[ipre] + (amp[ipost]-amp[ipre])*(TF-ff[ipre])/(ff[ipost]-ff[ipre]), 'vb', ms = 0.5, alpha = 0.5, label = 'inh non-OP')
                    ax3.plot(edges[:-1], movingAvg(nsp, nbins, nsmoothFr), ':b', lw = 0.5, alpha = 0.5)
        
            ax2.set_yscale('log')
            ax2.set_xlabel('Hz')
            #ax2_.set_ylim(bottom = -180, top = 180)
            ax11.set_aspect('equal')
            ax11.set_title('OP-LR')
            ax12.set_title('OP-SC')
            ax12.set_aspect('equal')
            ax3.legend(loc = 'best')

            fign = 'V1-scatterFF_OP'
            if usePrefData:
                fign = fign + 'pref'

            if nt == nt_:
                fig.savefig(fig_fdr+output_suffix + fign + '-' + mode + '_full.png')
            else:
                fig.savefig(fig_fdr+output_suffix + fign + '-' + mode + '.png')
            plt.close(fig)
    
    if plotLGNsCorr:
        fig = plt.figure(f'sLGN-corr', figsize = (12,12), dpi = 600)
        grid = gs.GridSpec(4, 4, figure = fig, hspace = 0.3, wspace = 0.3)
        
        gFF_target = np.sum(gFF[:,:,0], axis = 0)
        gE_target  = np.sum(gE[:,:,0], axis = 0)
        gI_target  = np.sum(gI[:,:,0], axis = 0)
    
        nsE = preNS[0,:]
        nsI = preNS[1,:]
        
        eSpick = eORpick
        iSpick = iORpick
        spick = np.hstack((eORpick,iORpick))

        ax = fig.add_subplot(grid[0,0])
        image = HeatMap(gFF_target[eSpick], gE_target[eSpick], heatBins, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. gE')
        
        ax = fig.add_subplot(grid[0,1])
        image = HeatMap(gFF_target[iSpick], gE_target[iSpick], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. gE')
        
        ax = fig.add_subplot(grid[1,0])
        image = HeatMap(gFF_target[eSpick], nsE[eSpick], heatBins, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. nsE')
        
        ax = fig.add_subplot(grid[1,1])
        image = HeatMap(gFF_target[iSpick], nsE[iSpick], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. nsE')
        
        ax = fig.add_subplot(grid[2,0])
        image = HeatMap(gFF_target[eSpick], gI_target[eSpick], heatBins, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. gI')
        
        ax = fig.add_subplot(grid[2,1])
        image = HeatMap(gFF_target[iSpick], gI_target[iSpick], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. gI')
        
        ax = fig.add_subplot(grid[0,2])
        image = HeatMap(gFF_target[eSpick], fr[eSpick], heatBins, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. FR Hz')
        
        ax = fig.add_subplot(grid[0,3])
        image = HeatMap(gFF_target[iSpick], fr[iSpick], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. FR Hz')
        
        ax = fig.add_subplot(grid[3,0])
        image = HeatMap(gFF_target[eSpick], nsI[eSpick], heatBins, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. nsI')
        
        ax = fig.add_subplot(grid[3,1])
        image = HeatMap(gFF_target[iSpick], nsI[iSpick], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. nsI')
    
        target = np.sum(gE_gTot_ratio[:,:,0],axis = 0)
        ax = fig.add_subplot(grid[1,2])
        image = HeatMap(gFF_target[eSpick], target[eSpick] , heatBins, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. gE/gTot')
        
        ax = fig.add_subplot(grid[1,3])
        image = HeatMap(gFF_target[iSpick], target[iSpick], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. gE/gTot')
    
        ax = fig.add_subplot(grid[2,2])
        image = HeatMap(gFF_target[spick], nLGN_V1[spick], heatBins, heatBins, ax, 'Greys', log_scale = pLog)
        ax.set_xlabel('gFF')
        ax.set_ylabel('nLGN')
        
        ax = fig.add_subplot(grid[2,3])
        image = HeatMap(gFF_F1F0[spick], nLGN_V1[spick], heatBins, heatBins, ax, 'Greys', log_scale = pLog)
        ax.set_xlabel('gFF-F1F0')
        ax.set_ylabel('nLGN')
    
        ax = fig.add_subplot(grid[3,2])
        image = HeatMap(gFF_target[eSpick], gEt_gTot_ratio[eSpick,0], heatBins, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('exc. gFF')
        ax.set_ylabel('exc. gEt/gTot')
        
        ax = fig.add_subplot(grid[3,3])
        image = HeatMap(gFF_target[iSpick], gEt_gTot_ratio[iSpick,0], heatBins, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('inh. gFF')
        ax.set_ylabel('inh. gEt/gTot')
    
        fig.savefig(fig_fdr+output_suffix + 'sLGN-corr' + '.png')
        plt.close(fig)
    print('plotting finished')

def movingAvg(data, n, m, axis = -1):
    if m == 1:
        return data
    else:
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

def ellipse(cx, cy, a, baRatio, orient, n = 50):
    b = a*baRatio
    print(f'major:{b}, minor:{a}')
    e = np.sqrt(1-1/baRatio/baRatio)
    theta = np.linspace(0, 2*np.pi, n)
    phi = orient + theta
    r = a/np.sqrt(1-np.power(e*np.cos(theta),2))
    x = cx + r*np.cos(phi)
    y = cy + r*np.sin(phi)
    return x, y

if __name__ == "__main__":

    if len(sys.argv) < 14:
        raise Exception('not enough argument for plotV1_response_lFF(output_suffix, res_suffix, conLGN_suffix, conV1_suffix, res_fdr, data_fdr, fig_fdr, TF, iOri, nOri, readNewSpike, usePrefData, collectMeanDataOnly, OPstatus)')
    else:
        output_suffix = sys.argv[1]
        print(output_suffix)
        res_suffix = sys.argv[2]
        print(res_suffix)
        conLGN_suffix = sys.argv[3]
        print(conLGN_suffix)
        conV1_suffix = sys.argv[4]
        print(conV1_suffix)
        res_fdr = sys.argv[5]
        print(res_fdr)
        data_fdr = sys.argv[6]
        print(data_fdr)
        fig_fdr = sys.argv[7]
        print(fig_fdr)
        TF = float(sys.argv[8])
        print(TF)
        iOri = int(sys.argv[9])
        nOri = int(sys.argv[10])
        print(f'{iOri}/{nOri}')
        if sys.argv[11] == 'True' or sys.argv[10] == '1':
            readNewSpike = True 
            print('read new spikes')
        else:
            readNewSpike = False
            print('read stored spikes')
        if sys.argv[12] == 'True' or sys.argv[11] == '1':
            usePrefData = True 
            print('using fitted data')
        else:
            usePrefData = False
            print('not using fitted data')
        if sys.argv[13] == 'True' or sys.argv[12] == '1':
            collectMeanDataOnly= True 
            print('collect mean data only')
        else:
            collectMeanDataOnly = False

        OPstatus = int(sys.argv[14])
        if OPstatus != 0 and OPstatus != 1 and OPstatus != 2:
            raise Exception(f'OPstatus = {OPstatus} but it can only be 0: no OP plots, 1: preset OP plots, 2: update OP plots only')
        else:
            if OPstatus == 0:
                print('no OP plots are plotted')
            if OPstatus == 1:
                print('preset OP plots are plotted')
            if OPstatus == 2:
                print('update OP plots only')

    print(sys.argv)
    plotV1_response_lFF(output_suffix, res_suffix, conLGN_suffix, conV1_suffix, res_fdr, data_fdr, fig_fdr, TF, iOri, nOri, readNewSpike, usePrefData, collectMeanDataOnly, OPstatus)
