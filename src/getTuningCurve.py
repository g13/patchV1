#!/usr/bin/env python
import numpy as np
import scipy as sp
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.special as special
import matplotlib
from sys import stdout
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.gridspec as gs
import matplotlib.colors as clr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
import sys
from readPatchOutput import *
from plotV1_response import ellipse
#import multiprocessing as mp
np.seterr(invalid = 'raise')

def gatherTuningCurve(output_suffix, res_suffix, conLGN_suffix, conV1_suffix, res_fdr, setup_fdr, data_fdr, fig_fdr, nOri, fitTC, fitDataReady):
    
    res_fdr = res_fdr+"/"
    data_fdr = data_fdr+"/"
    setup_fdr = setup_fdr+"/"
    fig_fdr = fig_fdr+"/"

    if output_suffix:
        output_suffix = "-" + output_suffix + "_"
    if res_suffix:
        res_suffix = "-" + res_suffix
    if conLGN_suffix:
        conLGN_suffix = "-" + conLGN_suffix
    if conV1_suffix:
        conV1_suffix = "-" + conV1_suffix

    seed = 17843143
    fr_thres = 1
    ns = 20
    ndpref = 10
    SCsplit = 1
    heatBins = 25
    sample = None
    #sample = np.array([91,72,78,54,84,91,8,6,8,52])*1024 + np.array([649,650,508,196,385,873,190,673,350,806])
    pLog = False

    #plotTC = False
    #plotFR = False
    #plotSample = False
    #plotLGNsSize = False
    #plotF1F0 = False
    #plotOSI = False
    #plotSpatialOP = False

    plotTC = True
    plotFR = True 
    plotSample = True
    plotLGNsSize = True
    plotF1F0 = True
    plotOSI = True
    plotSpatialOP = True

    if sample is not None:
        add_dprefToSample = False
    else:
        add_dprefToSample = True

    LGN_V1_ID_file = setup_fdr + 'LGN_V1_idList'+conLGN_suffix+'.bin'
    LGN_V1_s_file = setup_fdr + 'LGN_V1_sList'+conLGN_suffix+'.bin'
    V1_RFpropFn = setup_fdr + "V1_RFprop" + conLGN_suffix + ".bin"

    LGN_frFn = data_fdr + "LGN_fr" + output_suffix 
    LGN_propFn = data_fdr + "LGN" + output_suffix + "1.bin"

    pref_file = data_fdr+'cort_pref' + output_suffix[:-1] + '.bin'
    fit_file = data_fdr+'fit_data' + output_suffix[:-1] + '.bin'

    parameterFn = data_fdr+"patchV1_cfg" +output_suffix + "1.bin"

    sampleFn = data_fdr+"OS_sampleList" + output_suffix[:-1] + ".bin"

    LGN_vposFn = res_fdr + 'LGN_vpos'+ res_suffix + ".bin"
    featureFn = res_fdr + 'V1_feature' + res_suffix + ".bin"
    V1_allposFn = res_fdr + 'V1_allpos' + res_suffix + ".bin"

    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, mE, mI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, virtual_LGN, tonicDep, noisyDep = read_cfg(parameterFn)

    LGN_V1_s = readLGN_V1_s0(LGN_V1_s_file)
    LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_ID_file)
    nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type = readLGN_vpos(LGN_vposFn)
    nV1_0 = nLGN_V1.size

    with open(LGN_propFn, 'rb') as f:
        print(nLGN)
        print(prec)
        f.seek((nLGN+1)*4+sizeofPrec*4*nLGN, 1)
        LGN_rw = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
        LGN_rh = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)
        LGN_orient = np.fromfile(f, prec, 2*nLGN).reshape(2,nLGN)

    with open(V1_RFpropFn, 'rb') as f:
        _nV1 = np.fromfile(f, 'u4', 1)[0]
        assert(nV1_0 == _nV1)
        f.seek(4*_nV1* sizeofPrec, 1)
        sfreq = np.fromfile(f, prec, count = _nV1)

    LGN_fr = np.empty((nOri,nLGN)) 
    LGN_fr_weighted = np.empty((nOri, nV1_0)) 
    perOriStats_files = [data_fdr+"traceStats" + output_suffix + str(iOri+1) + ".bin" for iOri in range(nOri)]

    for i in range(nOri):
        with open(perOriStats_files[i], 'rb') as f:
            if i == 0:
                iModel = np.fromfile(f, 'i4', count=1)[0]
                nV1 = np.fromfile(f, 'i4', count=1)[0]
                nI = np.fromfile(f, 'i4', count=1)[0]
                if nV1 != nV1_0:
                    raise Exception(f'inconsistent cfg file({nV1_0}) vs. mean data file ({nV1})')
                ngFF = np.fromfile(f, 'i4', count=1)[0]
                ngE = np.fromfile(f, 'i4', count=1)[0]
                ngI = np.fromfile(f, 'i4', count=1)[0]
                fr = np.empty((nOri,nV1))
                gFF = np.empty((nOri,ngFF,nV1,2))
                gE = np.empty((nOri,ngE,nV1,2))
                gI = np.empty((nOri,ngI,nV1,2))
                w = np.empty((nOri,nV1,2))
                v = np.empty((nOri,nV1,2))
                depC = np.empty((nOri,nV1,2))
                cFF = np.empty((nOri,ngFF,nV1,2))
                cE = np.empty((nOri,ngE,nV1,2))
                cI = np.empty((nOri,ngI,nV1,2))
                cTotal = np.empty((nOri,nV1,2))
                cGap = np.empty((nOri,nI,2))
                cTotal_freq = np.empty((nOri,nV1,3))
                cTotal_percent = np.empty((nOri,nV1,5))
                cFF_freq = np.empty((nOri,nV1,3))
                cFF_percent = np.empty((nOri,nV1,5))
                cE_freq = np.empty((nOri,nV1,3))
                cE_percent = np.empty((nOri,nV1,5))
                cI_freq = np.empty((nOri,nV1,3))
                cI_percent = np.empty((nOri,nV1,5))
                depC_freq = np.empty((nOri,nV1,3))
                depC_percent = np.empty((nOri,nV1,5))
                if iModel == 1:
                    w_freq = np.empty((nOri,nV1,3))
                    w_percent = np.empty((nOri,nV1,5))

                F1F0    = np.empty((nOri,nV1))
                gFF_F1F0= np.empty((nOri,nV1))
                gE_F1F0 = np.empty((nOri,nV1))
                gI_F1F0 = np.empty((nOri,nV1))
            else:
                _iModel = np.fromfile(f, 'i4', count=1)[0]
                _nV1 = np.fromfile(f, 'i4', count=1)[0]
                _nI = np.fromfile(f, 'i4', count=1)[0]
                _ngFF = np.fromfile(f, 'i4', count=1)[0]
                _ngE = np.fromfile(f, 'i4', count=1)[0]
                _ngI = np.fromfile(f, 'i4', count=1)[0]
                if _nI != nI or _nV1 != nV1 or _ngFF != ngFF or _ngE != ngE or _ngI != ngI or _iModel != iModel:
                    raise Exception("mean data files are not consistent")
    
            fr[i,:] = np.fromfile(f, 'f8', count = nV1)
            gFF[i,:,:,:] = np.fromfile(f, 'f8', count = ngFF*2*nV1).reshape((ngFF,nV1,2))
            gE[i,:,:,:] = np.fromfile(f, 'f8', count = ngE*2*nV1).reshape((ngE,nV1,2))
            gI[i,:,:,:] = np.fromfile(f, 'f8', count = ngI*2*nV1).reshape((ngI,nV1,2))
            w[i,:,:] = np.fromfile(f, 'f8', count = 2*nV1).reshape((nV1,2))
            v[i,:,:] = np.fromfile(f, 'f8', count = 2*nV1).reshape((nV1,2))
            depC[i,:,:] = np.fromfile(f, 'f8', count = 2*nV1).reshape((nV1,2))
            cFF[i,:,:,:] = np.fromfile(f, 'f8', count = ngFF*2*nV1).reshape((ngFF,nV1,2))
            cE[i,:,:,:] = np.fromfile(f, 'f8', count = ngE*2*nV1).reshape((ngE,nV1,2))
            cI[i,:,:,:] = np.fromfile(f, 'f8', count = ngI*2*nV1).reshape((ngI,nV1,2))
            cTotal[i,:,:] = np.fromfile(f, 'f8', count = 2*nV1).reshape((nV1,2))
            cGap[i,:,:] = np.fromfile(f, 'f8', count = 2*nI).reshape((nI,2))

            cTotal_freq[i,:,:] = np.fromfile(f, 'f8', count = 3*nV1).reshape((nV1,3))
            cTotal_percent[i,:,:] = np.fromfile(f, 'f8', count = 5*nV1).reshape((nV1,5))
            cFF_freq[i,:,:] = np.fromfile(f, 'f8', count = 3*nV1).reshape((nV1,3))
            cFF_percent[i,:,:] = np.fromfile(f, 'f8', count = 5*nV1).reshape((nV1,5))
            cE_freq[i,:,:] = np.fromfile(f, 'f8', count = 3*nV1).reshape((nV1,3))
            cE_percent[i,:,:] = np.fromfile(f, 'f8', count = 5*nV1).reshape((nV1,5))
            cI_freq[i,:,:] = np.fromfile(f, 'f8', count = 3*nV1).reshape((nV1,3))
            cI_percent[i,:,:] = np.fromfile(f, 'f8', count = 5*nV1).reshape((nV1,5))
            depC_freq[i,:,:] = np.fromfile(f, 'f8', count = 3*nV1).reshape((nV1,3))
            depC_percent[i,:,:] = np.fromfile(f, 'f8', count = 5*nV1).reshape((nV1,5))
            if iModel == 1:
                w_freq[i,:,:] = np.fromfile(f, 'f8', count = 3*nV1).reshape((nV1,3))
                w_percent[i,:,:] = np.fromfile(f, 'f8', count = 5*nV1).reshape((nV1,5))
            F1F0[i,:] = np.fromfile(f, 'f8', count = nV1)
            gFF_F1F0[i,:] = np.fromfile(f, 'f8', count = nV1)
            gE_F1F0[i,:] = np.fromfile(f, 'f8', count = nV1)
            gI_F1F0[i,:] = np.fromfile(f, 'f8', count = nV1)

        _LGN_fr = readLGN_fr(LGN_frFn+f"{i+1}.bin", prec = prec)
        LGN_fr[i,:] = np.max(_LGN_fr, axis=0)
    
        for j in range(nV1):
            LGN_fr_weighted[i,j] = np.sum(LGN_fr[i,LGN_V1_ID[j]]*LGN_V1_s[j])

    nE = nV1 - nI
    blockSize = typeAcc[-1]
    typeAcc = np.hstack((0, typeAcc))
    nblock = nV1//blockSize

    epick = np.hstack([np.arange(mE) + iblock*blockSize for iblock in range(nblock)])
    ipick = np.hstack([np.arange(mI) + iblock*blockSize + mE for iblock in range(nblock)])
    assert(epick.size == nE)
    assert(ipick.size == nI)
    eSpick = epick[nLGN_V1[epick] > SCsplit]
    eCpick = epick[nLGN_V1[epick] <= SCsplit]
    iSpick = ipick[nLGN_V1[ipick] > SCsplit]
    iCpick = ipick[nLGN_V1[ipick] <= SCsplit]

    dOri = 180/nOri
    nOri_bins = nOri
    op = np.arange(nOri)/nOri*180
    opEdges = (np.arange(nOri_bins+1)-0.5)/nOri_bins*180
    dopEdges = np.linspace(-nOri_bins//2-0.5, nOri_bins//2+0.5, nOri+2)*180/nOri_bins
    op = np.append(op, 180)
    opEdges = np.append(opEdges, 180*(1+0.5/nOri_bins))

    featureType = np.array([0,1])
    feature, rangeFeature, minFeature, maxFeature = readFeature(featureFn, nV1, featureType)
    LR = feature[0,:]
    iPref = np.mod(feature[1,:] + 0.5, 1.0)*nOri # for ori
    print(f'iPref: {[np.min(iPref), np.max(iPref)]}')

    gFF_max = gFF[:,:,:,0].sum(axis = 1)*(1+gFF_F1F0)
    print(f'gFF_max(gFF*gFF_F1F0) shape: {gFF_max.shape}')
    #iPref_thal = np.argmax(gFF_F1F0, axis = 0)
    iPref_thal = np.argmax(gFF_max, axis = 0)
    iPref_thal_theory = np.argmax(LGN_fr_weighted, axis = 0)
    print([np.min(iPref_thal), np.max(iPref_thal), nOri])
    iPref_cort = np.argmax(fr, axis = 0)
    max_fr = fr.T.flatten()[iPref_cort + np.arange(nV1)*nOri]
    with open(data_fdr+'max_fr' + output_suffix[:-1] + '.bin', 'wb') as f:
        max_fr.tofile(f)

    epick_act = epick[max_fr[epick] > fr_thres]
    ipick_act = ipick[max_fr[ipick] > fr_thres]

    eSpick_act = eSpick[max_fr[eSpick] > fr_thres]
    eCpick_act = eCpick[max_fr[eCpick] > fr_thres]
    iSpick_act = iSpick[max_fr[iSpick] > fr_thres]
    iCpick_act = iCpick[max_fr[iCpick] > fr_thres]

    zero_fr_pref = np.round(iPref).astype(int)[max_fr == 0]
    zero_fr_pref[zero_fr_pref == 0] = nOri
    zero_fr_pref -= 1
    iPref_cort[max_fr == 0] = zero_fr_pref

    max_pick = iPref_thal + np.arange(nV1)*nOri

    gFF_F1F0_2nd = gFF_F1F0.T.flatten().copy()
    gFF_F1F0_2nd[max_pick] = 0
    gFF_F1F0_2nd = np.reshape(gFF_F1F0_2nd, (nV1, nOri)).T
    iPref_thal_2nd = np.argmax(gFF_F1F0_2nd, axis = 0)

    iPref_thal_for_ori = np.mod(iPref_thal + 1, nOri)
    iPref_thal_theory_for_ori = np.mod(iPref_thal_theory + 1, nOri)
    iPref_thal_2nd_for_ori = np.mod(iPref_thal_2nd + 1, nOri)
    iPref_cort_for_ori = np.mod(iPref_cort + 1, nOri)

    if sample is None:
        np.random.seed(seed)
        sample = np.random.randint(nV1, size = ns)
        if True:
            sample = np.zeros(9, dtype = int)

            #pick = epick[nLGN_V1[epick] > np.mean(nLGN_V1[epick])]
            pick = epick[nLGN_V1[epick] > SCsplit]
            #sample[0] = pick[np.argmin(max_fr[pick])]
            sample[0] = pick[np.argpartition(max_fr[pick], pick.size//2)[pick.size//2]] # find median
            sample[1] = pick[np.argmax(max_fr[pick])]
            sample[2] = np.random.choice(pick, 1)[0]

            pick = epick[nLGN_V1[epick] <= SCsplit]
            #sample[3] = pick[np.argmin(max_fr[pick])]
            sample[3] = pick[np.argpartition(max_fr[pick], pick.size//2)[pick.size//2]] # find median
            sample[4] = pick[np.argmax(max_fr[pick])]
            sample[5] = np.random.choice(pick, 1)[0]

            pick = ipick[nLGN_V1[ipick] > SCsplit]
            sample[6] = pick[np.argpartition(max_fr[pick], pick.size//2)[pick.size//2]] # find median
            #sample[6] = pick[np.argmin(max_fr[pick])]
            sample[7] = pick[np.argmax(max_fr[pick])]
            sample[8] = np.random.choice(pick, 1)[0]
    
    with open(sampleFn, 'wb') as f:
        np.array([sample.size],dtype='u4').tofile(f)
        sample.astype('u4').tofile(f)
    ns = sample.size

    nNoOpt = 0
    noOpt = []
    if fitTC:
        iPref_cort0 = iPref_cort
        iPref_cort0_for_ori = iPref_cort_for_ori
        def von_Mises(theta, a, b, s, theta0):
            return b + a * np.exp(s*(np.cos(2*(theta-theta0)) - 1))

        if fitDataReady:
            f = open(fit_file, 'rb')
            fit_max_fr = np.fromfile(f, 'f8', nV1)
            fit_pref = np.fromfile(f, 'f8', nV1)
            fit_gOSI = np.fromfile(f, 'f8', nV1)
            fit_a = np.fromfile(f, 'f8', nV1)
            fit_b = np.fromfile(f, 'f8', nV1)
            fit_s = np.fromfile(f, 'f8', nV1)
            noOpt = np.fromfile(f, 'u4')
            f.close()
            print('read fitted data from file')

            iPref_cort_for_ori = fit_pref/np.pi*nOri
            iPref_cort = np.round(iPref_cort_for_ori).astype(int)
            # deg = 0 takes half away from deg = 360
            iPref_cort[iPref_cort == 0] = nOri
            iPref_cort = iPref_cort - 1

        else:
            print(f', fitting')

            fit_max_fr = np.zeros(nV1)
            fit_pref = np.zeros(nV1)
            fit_gOSI = np.zeros(nV1)
            fit_a = np.zeros(nV1)
            fit_b = np.zeros(nV1)
            fit_s = np.zeros(nV1)
            # von Mises fit: b + a * exp(s*(cos(theta-theta0)-1)), 
            def von_Mises0(theta, a, s, theta0):
                return a * np.exp(s*(np.cos(2*(theta-theta0)) - 1))

            def jac_von_Mises0(theta, a, s, theta0):
                return np.array([np.exp(s*(np.cos(2*(theta-theta0)) - 1)), a*np.exp(s*(np.cos(2*(theta-theta0)) - 1))*(np.cos(2*(theta-theta0)) - 1),a*np.exp(s*(np.cos(2*(theta-theta0)) - 1))*(2*s)*np.sin(2*(theta-theta0))]).T

            def jac_von_Mises(theta, a, b, s, theta0):
                return np.array([np.exp(s*(np.cos(2*(theta-theta0)) - 1)), np.ones(theta.shape), a*np.exp(s*(np.cos(2*(theta-theta0)) - 1))*(np.cos(2*(theta-theta0)) - 1),a*np.exp(s*(np.cos(2*(theta-theta0)) - 1))*(2*s)*np.sin(2*(theta-theta0))]).T
                # max: max_fr = b + a, theta = theta0
                # min: min_fr = b + a*np.exp(-2s), theta = theta0 + np.pi/2

                # s = 1
                # b + a = max
                # b + a*exp(-2) = min

                # a = (max - min)/(1-exp(-2))
                # b = max-a
            theta = (np.arange(nOri)+1)/nOri*np.pi
            for i in range(nV1):
                q = fr[:,i]
                max_q = np.max(q)
                min_q = np.min(q)
                iq = 1
                if max_q > min_q:
                    min_q2 = min_q
                    while min_q2 == min_q and iq < fr.shape[0]:
                        min_q2 = np.partition(q,iq)[iq]
                        iq = iq + 1
                    average_min = (min_q+min_q2)/2
                    s0 = (np.log(max_q-min_q) - np.log(average_min-min_q))/2
                    a0 = (max_q - average_min)/(1-np.exp(-2*s0))
                    b0 = max_q - a0
                    if np.mod(nOri, 2) == 0:
                        theta0 = iPref_cort[i]/nOri*np.pi
                    else:
                        inext = np.mod(iPref_cort[i]+1,nOri)
                        iprev = np.mod(iPref_cort[i]+nOri-1,nOri)
                        if q[iprev] > q[inext]:
                            theta0 = (iPref_cort[i] + iPref_cort[iprev])/2
                        else:
                            theta0 = (iPref_cort[i] + iPref_cort[inext])/2
                        theta0 = theta0*np.pi

                    if b0 < 0: 
                        b = 0
                        a0 = max_q
                        p0 = [a0, s0, theta0]
                        bounds = ([0, 0, 0],[2*a0, np.inf, np.pi])
                        try:
                            params, _ = optimize.curve_fit(von_Mises0, theta, q, p0 = p0, bounds = bounds)#, jac = jac_von_Mises0)
                            a, s, fit_pref[i] = params
                        except RuntimeError as noOptimal:
                            #print(f'{noOptimal} for {i}th neuron')
                            a = a0
                            s = s0
                            fit_pref[i] = theta0
                            nNoOpt = nNoOpt + 1
                            noOpt.append(i)

                    else:
                        p0 = [a0, b0, s0, theta0]
                        bounds = ([0, 0, 0, 0],[2*a0, max_q, np.inf, np.pi])
                        try:
                            params, _ = optimize.curve_fit(von_Mises, theta, q, p0 = p0, bounds = bounds)#, jac = jac_von_Mises)
                            a, b, s, fit_pref[i] = params
                        except RuntimeError as noOptimal:
                            #print(f'{noOptimal} for {i}th neuron')
                            a = a0
                            b = b0
                            s = 1
                            fit_pref[i] = theta0
                            nNoOpt = nNoOpt + 1
                            noOpt.append(i)
                        except FloatingPointError as e:
                            print(f'{e}:\n p0 = {p0}')
                            raise e

                    ##try:
                    tmp_integral = integrate.quad(lambda x: np.exp(s*np.sqrt(1-x*x)), 0, 1)[0] - integrate.quad(lambda x: np.exp(-s*np.sqrt(1-x*x)), 0, 1)[0]
                    #except IntegrationWarning as intWarn:
                    #    print(f'{intWarn}: for a = {a} , b = {b}, s = {s}')
                    #    tmp_integral = integrate.quad(lambda x: np.exp(s*np.sqrt(1-x*x)), 0, 1)[0] - integrate.quad(lambda x: np.exp(-s*np.sqrt(1-x*x)), 0, 1)[0]
                    #    print(tmp_integral)
                    fit_gOSI[i] = tmp_integral/(b + 2*a*special.i0(s))*np.pi
                else:
                    a = 0
                    b = max_q
                    s = 0
                    fit_pref[i] = iPref_cort_for_ori[i]/nOri*np.pi 
                    fit_gOSI[i] = 0

                fit_a[i] = a
                fit_b[i] = b
                fit_s[i] = s
                fit_max_fr[i] = a + b

                if np.mod(i+1,nV1//100) == 0 or i == nV1-1:
                    stdout.write(f'\r{i/nV1*100:.2f}%')
            stdout.write('\n')
            print(f'{nNoOpt/nV1*100:.2f}% of the neurons has no optimal fit for a tuning curve')

            iPref_cort_for_ori = fit_pref/np.pi*nOri
            print(f'fit_pref: {[np.min(fit_pref), np.max(fit_pref)]}')
            iPref_cort = np.round(iPref_cort_for_ori).astype(int)
            iPref_cort[iPref_cort == 0] = nOri
            iPref_cort = iPref_cort - 1
            noOpt = np.array(noOpt)
            with open(fit_file, 'wb') as f:
                fit_max_fr.tofile(f)
                fit_pref.tofile(f)
                fit_gOSI.tofile(f)
                fit_a.tofile(f)
                fit_b.tofile(f)
                fit_s.tofile(f)
                noOpt.tofile(f)

        with open(pref_file, 'wb') as f:
            fitted = True
            np.array([fitted], dtype = 'i4').tofile(f)
            np.array([nV1], dtype = 'u4').tofile(f)
            fit_pref.astype('f4').tofile(f)
            (iPref_cort0_for_ori/nOri*np.pi).astype('f4').tofile(f)
    else:
        with open(pref_file, 'wb') as f:
            fitted = False
            np.array([fitted], dtype = 'i4').tofile(f)
            np.array([nV1], dtype = 'u4').tofile(f)
            (iPref_cort_for_ori/nOri*np.pi).astype('f4').tofile(f)

    with open(pref_file, 'ab') as f:
        (iPref_thal_for_ori/nOri*np.pi).astype('f4').tofile(f)
        (iPref/nOri*np.pi).astype('f4').tofile(f)

        noOpt = np.array(noOpt, dtype = 'i4')

    figPref = plt.figure('pref-hist', figsize = (14,6))
    ax = figPref.add_subplot(231)

    OPbins = (np.arange(0,nOri+2)-0.5)/nOri*180
    OPs = (OPbins[1:] + OPbins[:-1])/2
    OPs = OPs[1:]
    print(f'OPbins = {OPbins}')
    print(f'OPs = {OPs}')

    targetEt = iPref_thal_for_ori[eSpick_act]/nOri*180
    targetIt = iPref_cort_for_ori[eSpick_act]/nOri*180
    targetEc = iPref_thal_for_ori[iSpick_act]/nOri*180
    targetIc = iPref_cort_for_ori[iSpick_act]/nOri*180
    ax.hist((targetEt,targetIt,targetEc,targetIc), bins=opEdges, label = ('thal_E','thal_I','cort_E','cort_I'), alpha = 0.3, color = ('r','b','m','c'), density = True)

    ax.legend(fontsize='xx-small', bbox_to_anchor=(0.0,1), loc='lower left', ncol = 4)
    ax.set_xticks([60,120,180])
    ax.set_ylabel('act. simple')
    ax.set_xlabel('OP')

    ax = figPref.add_subplot(232)

    dPref = iPref_cort_for_ori[iSpick_act]-iPref_thal_for_ori[iSpick_act]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ax.hist(dPref*dOri, bins=dopEdges, label = 'dPref_IS', alpha = 0.3, color = 'b', density = True)

    dPref = iPref_cort_for_ori[eSpick_act]-iPref_thal_for_ori[eSpick_act]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ax.hist(dPref*dOri, bins=dopEdges, label = 'dPref_ES', alpha = 0.3, color = 'r', density = True)
    ax.set_title('cort vs. thal.')

    ax.legend(fontsize='xx-small')
    ax.set_xlabel('delta_op.')

    ax = figPref.add_subplot(234)

    dPrefIS = iPref_thal_for_ori[iSpick]-iPref[iSpick]
    pick = dPrefIS > nOri//2
    dPrefIS[pick] = nOri - dPrefIS[pick]
    pick = dPrefIS < -nOri//2
    dPrefIS[pick] = nOri + dPrefIS[pick]
    #ax.hist(dPrefIS*dOri, bins=dopEdges, label = 'dPref_IS', alpha = 0.3, color = 'b', density = True)

    dPrefES = iPref_thal_for_ori[eSpick]-iPref[eSpick]
    pick = dPrefES > nOri//2
    dPrefES[pick] = nOri - dPrefES[pick]
    pick = dPrefES < -nOri//2
    dPrefES[pick] = nOri + dPrefES[pick]
    #ax.hist(dPrefES*dOri, bins=dopEdges, label = 'dPref_ES', alpha = 0.3, color = 'r', density = True)

    dPrefIS0 = iPref_thal_theory_for_ori[iSpick]-iPref[iSpick]
    pick = dPrefIS0 > nOri//2
    dPrefIS0[pick] = nOri - dPrefIS0[pick]
    pick = dPrefIS0 < -nOri//2
    dPrefIS0[pick] = nOri + dPrefIS0[pick]
    #ax.hist(dPrefIS0*dOri, bins=dopEdges, label = 'dPref_IS0', alpha = 0.3, color = 'g', density = True)

    dPrefES0 = iPref_thal_theory_for_ori[eSpick]-iPref[eSpick]
    pick = dPrefES0 > nOri//2
    dPrefES0[pick] = nOri - dPrefES0[pick]
    pick = dPrefES0 < -nOri//2
    dPrefES0[pick] = nOri + dPrefES0[pick]
    #ax.hist(dPrefES0*dOri, bins=dopEdges, label = 'dPref_ES0', alpha = 0.3, color = 'y', density = True)
    ax.hist((dPrefES*dOri,dPrefIS*dOri,dPrefES0*dOri,dPrefIS0*dOri), bins=dopEdges, label = ('dPref_ES','dPref_IS','dPref_ES0','dPref_IS0'), alpha = 0.3, color = ('r','b','m','c'), density = True)
    ax.set_xlabel('pre vs thal.')
    ax.legend(fontsize='xx-small', bbox_to_anchor = (0,1), loc = 'upper left')

    #spick = np.concatenate((eSpick,iSpick))
    spick = eSpick_act
    dPref = iPref_thal_for_ori[spick]-iPref[spick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    dprefMax_pick = spick[np.argpartition(-np.absolute(dPref), ndpref)[:ndpref]]

    dprefMin_pick = spick[np.argpartition(np.absolute(dPref), ndpref)[:ndpref]]

    #dprefMin_pick = spick[np.absolute(dPref) == np.min(np.absolute(dPref))]

    ax = figPref.add_subplot(235)

    if ipick_act.size > 0:
        dPref = iPref_cort_for_ori[ipick_act]-iPref[ipick_act]
        pick = dPref > nOri//2
        dPref[pick] = nOri - dPref[pick]
        pick = dPref < -nOri//2
        dPref[pick] = nOri + dPref[pick]
        ax.hist(dPref*dOri, bins=dopEdges, label = 'dPref_I', alpha = 0.3, color = 'b', density = True)

    if eCpick_act.size > 0:
        dPref = iPref_cort_for_ori[eCpick_act]-iPref[eCpick_act]
        pick = dPref > nOri//2
        dPref[pick] = nOri - dPref[pick]
        pick = dPref < -nOri//2
        dPref[pick] = nOri + dPref[pick]
        ax.hist(dPref*dOri, bins=dopEdges, label = 'dPref_EC', alpha = 0.3, color = 'm', density = True)

    if eSpick_act.size > 0:
        dPref = iPref_cort_for_ori[eSpick_act]-iPref[eSpick_act]
        pick = dPref > nOri//2
        dPref[pick] = nOri - dPref[pick]
        pick = dPref < -nOri//2
        dPref[pick] = nOri + dPref[pick]
        ax.hist(dPref*dOri, bins=dopEdges, label = 'dPref_ES', alpha = 0.3, color = 'r', density = True)

    
    ax.set_xlabel('pre. vs cort.')

    ax = figPref.add_subplot(233)
    
    pick = iSpick_act[LR[iSpick_act] < 0]
    dPrefIS_L = iPref_thal_for_ori[pick]-iPref[pick]
    pick = dPrefIS_L > nOri//2
    dPrefIS_L[pick] = nOri - dPrefIS_L[pick]
    pick = dPrefIS_L < -nOri//2
    dPrefIS_L[pick] = nOri + dPrefIS_L[pick]
    ax.hist(dPrefIS_L*dOri, bins=dopEdges, label = 'dPref_IS_L', alpha = 0.3, color = 'b')

    pick = eSpick_act[LR[eSpick_act] < 0]
    dPrefES_L = iPref_thal_for_ori[pick]-iPref[pick]
    pick = dPrefES_L > nOri//2
    dPrefES_L[pick] = nOri - dPrefES_L[pick]
    pick = dPrefES_L < -nOri//2
    dPrefES_L[pick] = nOri + dPrefES_L[pick]
    ax.hist(dPrefES_L*dOri, bins=dopEdges, label = 'dPref_ES_L', alpha = 0.3, color = 'r')
    ax.set_title('ipsi thal vs. pre')

    
    ax = figPref.add_subplot(236)

    pick = iSpick_act[LR[iSpick_act] > 0]
    dPrefIS_R = iPref_thal_for_ori[pick]-iPref[pick]
    pick = dPrefIS_R > nOri//2
    dPrefIS_R[pick] = nOri - dPrefIS_R[pick]
    pick = dPrefIS_R < -nOri//2
    dPrefIS_R[pick] = nOri + dPrefIS_R[pick]
    ax.hist(dPrefIS_R*dOri, bins=dopEdges, label = 'dPref_IS_R', alpha = 0.3, color = 'b')

    pick = eSpick_act[LR[eSpick_act] > 0]
    dPrefES_R = iPref_thal_for_ori[pick]-iPref[pick]
    pick = dPrefES_R > nOri//2
    dPrefES_R[pick] = nOri - dPrefES_R[pick]
    pick = dPrefES_R < -nOri//2
    dPrefES_R[pick] = nOri + dPrefES_R[pick]
    ax.hist(dPrefES_R*dOri, bins=dopEdges, label = 'dPref_ES_R', alpha = 0.3, color = 'r')
    ax.set_xlabel('contra. thal vs. pre')

    figPref.savefig(fig_fdr + 'pref-hist' + output_suffix[:-1] + '.png', dpi = 150)

    fig = plt.figure('nLGN-pref', figsize = (6,4))
    maxLGN = np.max(nLGN_V1)
    ax = fig.add_subplot(221)
    target = nLGN_V1[eSpick]
    ytarget = dPrefES*dOri
    HeatMap(target, ytarget, np.arange(maxLGN)+1, dopEdges, ax, 'Reds', log_scale = pLog)
    ax = fig.add_subplot(222)
    target = nLGN_V1[iSpick]
    ytarget = dPrefIS*dOri
    HeatMap(target, ytarget, np.arange(maxLGN)+1, dopEdges, ax, 'Blues', log_scale = pLog)
    ax = fig.add_subplot(223)
    ax.hist(nLGN_V1[epick], np.arange(maxLGN+1))
    ax = fig.add_subplot(224)
    ax.hist(nLGN_V1[ipick], np.arange(maxLGN+1))
    fig.savefig(fig_fdr + 'nLGN-pref' + output_suffix[:-1] + '.png', dpi = 150)


    fig = plt.figure('nLGN-sfreq', figsize = (6,3.5))
    maxLGN = np.max(nLGN_V1)
    ax = fig.add_subplot(121)
    target = nLGN_V1[eSpick]
    ytarget = sfreq[eSpick]
    HeatMap(target, ytarget, np.arange(maxLGN)+1, maxLGN, ax, 'Reds', log_scale = pLog)
    ax.set_xlabel('nLGN')
    ax.set_ylabel('sfreq (cyc/deg)')
    ax = fig.add_subplot(122)
    target = nLGN_V1[iSpick]
    ytarget = sfreq[iSpick]
    HeatMap(target, ytarget, np.arange(maxLGN)+1, maxLGN, ax, 'Blues', log_scale = pLog)
    ax.set_xlabel('nLGN')
    ax.set_ylabel('sfreq (cyc/deg)')
    fig.savefig(fig_fdr + 'nLGN-sfreq' + output_suffix[:-1] + '.png', dpi = 150)

    fig = plt.figure('nLGN-OS_dist', figsize = (6,4))
    ax = fig.add_subplot(121)
    target = nLGN_V1[epick]
    ytarget = iPref_cort_for_ori[epick] * dOri
    HeatMap(target, ytarget, np.arange(maxLGN)+1, opEdges, ax, 'Reds', log_scale = pLog)
    ax.set_xlabel('nLGN')
    ax.set_ylabel('iPref_cort')

    ax = fig.add_subplot(122)
    target = nLGN_V1[epick]
    ytarget = iPref[epick] * dOri
    HeatMap(target, ytarget, np.arange(maxLGN)+1, opEdges, ax, 'Reds', log_scale = pLog)
    ax.set_xlabel('nLGN')
    ax.set_ylabel('iPref_preset')
    fig.savefig(fig_fdr + 'nLGN-OS_dist' + output_suffix[:-1] + '.png', dpi = 150)

    markers = ['^r', 'vg', '*g', 'dr', '^b', 'vb']
    type_labels = ['Red-On', 'Green-Off', 'Green-On', 'Red-Off', 'On', 'Off']
    type_c = ('r', 'g', 'g', 'r', 'k', 'b')

    ms = 4
    fig = plt.figure('OP_dOri', figsize = (5,5), dpi = 300)

    ax = fig.add_subplot(221)
    target = iPref_thal_for_ori[spick]*dOri
    spick = eSpick_act
    dPref = iPref_thal_for_ori[spick]-iPref[spick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ytarget = dPref*dOri
    HeatMap(target, ytarget, opEdges, dopEdges, ax, 'Reds', log_scale = pLog)
    ax.set_ylabel('op spread')
    ax = fig.add_subplot(222)
    spick = iSpick_act
    target = iPref_thal_for_ori[spick]*dOri
    dPref = iPref_thal_for_ori[spick]-iPref[spick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ytarget = dPref*dOri
    HeatMap(target, ytarget, opEdges, dopEdges, ax, 'Blues', log_scale = pLog)
    ax = fig.add_subplot(223)
    spick = eSpick_act
    target = iPref_cort_for_ori[spick]*dOri
    dPref = iPref_cort_for_ori[spick] - iPref_thal_for_ori[spick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ytarget = dPref*dOri
    HeatMap(target, ytarget, opEdges, dopEdges, ax, 'Reds', log_scale = pLog)
    ax.set_ylabel('op spread')
    ax.set_xlabel('orientation')
    ax = fig.add_subplot(224)
    spick = iSpick_act
    target = iPref_cort_for_ori[spick]*dOri
    dPref = iPref_cort_for_ori[spick] - iPref_thal_for_ori[spick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ytarget = dPref*dOri
    HeatMap(target, ytarget, opEdges, dopEdges, ax, 'Reds', log_scale = pLog)
    ax.set_xlabel('orientation')

    fig.savefig(fig_fdr + 'OP_dOri' + output_suffix[:-1] + '.png', dpi = 300)

    print(f'prefMax iV1: {dprefMax_pick}')
    fig = plt.figure('thal_set-prefMax', figsize = (10,10), dpi = 300)
    grid = gs.GridSpec((ndpref+3)//4, 4, figure = fig, hspace = 0.2)
    i = 0
    for iV1 in dprefMax_pick:
        ax = fig.add_subplot(grid[i//4,np.mod(i,4)])
        iLGN_vpos = LGN_vpos[:,LGN_V1_ID[iV1]]
        iLGN_type = LGN_type[LGN_V1_ID[iV1]]
        if LR[iV1] > 0:
            all_pos = LGN_vpos[:,nLGN_I:nLGN]
            all_type = LGN_type[nLGN_I:nLGN]
        else:
            all_pos = LGN_vpos[:,:nLGN_I]
            all_type = LGN_type[:nLGN_I]

        if nLGN_V1[iV1] > 0:
            iLGN_ms = LGN_V1_s[iV1]
            max_s = np.max(iLGN_ms)
            min_s = np.min(iLGN_ms)
            for j in range(len(markers)):
                pick = all_type == j
                ax.plot(all_pos[0,pick], all_pos[1,pick], markers[j], ms = min_s/max_s*ms, mec = None, mew = 0, alpha = 0.6)

            if plotLGNsSize:
                for j in range(nLGN_V1[iV1]):
                    ax.plot(iLGN_vpos[0,j], iLGN_vpos[1,j], markers[iLGN_type[j]], ms = iLGN_ms[j]/max_s*ms*1.5, mec = 'k', mew = ms/9)
                x0 = np.average(iLGN_vpos[0,:], weights = iLGN_ms)
                y0 = np.average(iLGN_vpos[1,:], weights = iLGN_ms)
            else:
                for j in range(len(markers)):
                    pick = iLGN_type == j
                    ax.plot(iLGN_vpos[0,pick], iLGN_vpos[1,pick], markers[j], ms = ms, mec = 'k', mew = ms/9)
                x0 = np.average(iLGN_vpos[0,:])
                y0 = np.average(iLGN_vpos[1,:])

            iSubR = 0
            iSubG = 0
            iSubOn = 0
            iSubOff = 0
            for j in range(nLGN_V1[iV1]):
                jtype = iLGN_type[j]

                lgn_id = LGN_V1_ID[iV1][j]
                ix, iy = ellipse(iLGN_vpos[0,j], iLGN_vpos[1,j], LGN_rw[0,lgn_id]*180/np.pi, LGN_rh[0,lgn_id]/LGN_rw[0,lgn_id], LGN_orient[0,lgn_id], n=25)
                ax.plot(ix, iy, color=type_c[jtype], ls='dotted', lw=0.1)

                if jtype == 0 or jtype == 3:
                    if iSubR == 0:
                        subRed = iLGN_vpos[:,j].copy()
                    else:
                        subRed += iLGN_vpos[:,j]
                    iSubR += 1 
                if jtype == 1 or jtype == 2:
                    if iSubG == 0:
                        subGreen = iLGN_vpos[:,j].copy()
                    else:
                        subGreen += iLGN_vpos[:,j]
                    iSubG += 1 
                if jtype == 4:
                    if iSubOn == 0:
                        subOn = iLGN_vpos[:,j].copy()
                    else:
                        subOn += iLGN_vpos[:,j]
                    iSubOn += 1 
                if jtype == 5:
                    if iSubOff == 0:
                        subOff = iLGN_vpos[:,j].copy()
                    else:
                        subOff += iLGN_vpos[:,j]
                    iSubOff += 1 

            SF_text = ''
            if iSubR > 0 and iSubG > 0:
                subRed /= iSubR
                subGreen /= iSubG
                SF_RG = 1/(2*np.linalg.norm(subRed-subGreen))
                ax.plot(subRed[0], subRed[1], 'sr', ms = ms, alpha = 0.5)
                ax.plot(subGreen[0], subGreen[1], 'sg', ms = ms, alpha = 0.5)
                SF_text += f'SF_RG: {SF_RG:.1f}'

            if iSubOn > 0 and iSubOff > 0:
                subOn /= iSubOn
                subOff /= iSubOff
                SF_OnOff = 1/(2*np.linalg.norm(subOn-subOff))
                ax.plot(subOn[0], subOn[1], 'sk', ms = ms, alpha = 0.5)
                ax.plot(subOff[0], subOff[1], 'sb', ms = ms, alpha = 0.5)
                SF_text += f'SF_OnOff: {SF_OnOff:.1f}'

            if max_fr[iV1] > 0:
                orient = iPref_cort_for_ori[iV1]/nOri*np.pi + np.pi/2
                if orient > np.pi:
                    orient = orient - np.pi

            orient_thal = iPref_thal_for_ori[iV1]/nOri*np.pi + np.pi/2
            if orient_thal > np.pi:
                orient_thal = orient_thal - np.pi

            orient_thal_theory = iPref_thal_theory_for_ori[iV1]/nOri*np.pi + np.pi/2
            if orient_thal_theory > np.pi:
                orient_thal_theory = orient_thal_theory - np.pi

            orient_thal_2nd = iPref_thal_2nd_for_ori[iV1]/nOri*np.pi + np.pi/2
            if orient_thal_2nd > np.pi:
                orient_thal_2nd = orient_thal_2nd - np.pi

            orient_set = feature[1,iV1]*np.pi
            if orient_set > np.pi:
                orient_set = orient_set - np.pi
            x = np.array([np.min(iLGN_vpos[0,:]), np.max(iLGN_vpos[0,:])])
            y = np.array([np.min(iLGN_vpos[1,:]), np.max(iLGN_vpos[1,:])])
            x_thal_theory = x.copy()
            y_thal_theory = y.copy()
            x_thal = x.copy()
            y_thal = y.copy()
            x_set = x.copy()
            y_set = y.copy()
            x_thal_2nd = x.copy()
            y_thal_2nd = y.copy()

            if np.diff(y)[0] > np.diff(x)[0]:
                daspect = y[1]-y[0]
            else:
                daspect = x[1]-x[0]
            ax.set_xlim(left = x[1] - daspect*1.1, right = x[0] + daspect*1.1)
            ax.set_ylim(bottom = y[1] - daspect*1.1, top = y[0] + daspect*1.1)
            if SF_text:
                ax.text(0.5, 0.85, SF_text +  'cycles/deg', transform=ax.transAxes, ha='center', fontsize='x-small')


            if np.diff(y)[0] > np.diff(x)[0]:
                if max_fr[iV1] > 0:
                    x = (y-y0) / np.tan(orient) + x0
                x_thal = (y_thal-y0) / np.tan(orient_thal) + x0
                x_thal_theory = (y_thal_theory-y0) / np.tan(orient_thal_theory) + x0
                x_set = (y_set-y0) / np.tan(orient_set) + x0
                x_thal_2nd = (y_thal_2nd-y0) / np.tan(orient_thal_2nd) + x0
            else:
                if max_fr[iV1] > 0:
                    y = np.tan(orient)*(x-x0) + y0
                y_thal = np.tan(orient_thal)*(x_thal-x0) + y0
                y_thal_theory = np.tan(orient_thal_theory)*(x_thal_theory-x0) + y0
                y_set = np.tan(orient_set)*(x_set-x0) + y0
                y_thal_2nd = np.tan(orient_thal_2nd)*(x_thal_2nd-x0) + y0

            ax.plot(x0, y0, '*b', ms = 1.0)
            if max_fr[iV1] > 0:
                ax.plot(x, y, '-k', lw = 1.0, label = 'cort.')
            ax.plot(x_set, y_set, ':r', lw = 1.5, label = 'preset')
            ax.plot(x_thal, y_thal, ':b', lw = 1.0, label = 'thal.')
            ax.plot(x_thal_theory, y_thal_theory, ':y', lw = 0.6, label = 'thal. theory.')
            ax.plot(x_thal_2nd, y_thal_2nd, ':c', lw = 0.3, label = 'thal2')

            ax.set_aspect('equal')
        else:
            raise Exception(f'the {i}th dprefMin ({iV1}) has no LGN input!')

        if i == dprefMax_pick.size-1:
            legend_elements = []
            for m, l in zip(markers, type_labels):
                legend_elements.append(Line2D([0], [0], marker=m[0], color=m[1], label = l))
            marker_legend = plt.legend(handles=legend_elements, bbox_to_anchor = (1,0), loc='lower left', fontsize='xx-small')
            ax.legend(fontsize='xx-small', bbox_to_anchor=(1,1), loc='upper left')
            ax.add_artist(marker_legend)

        ax.tick_params(axis = 'x', labelsize = 'xx-small')
        ax.tick_params(axis = 'y', labelsize = 'xx-small')
        iblock = iV1//blockSize
        ithread = np.mod(iV1, blockSize)
        title = f'{iblock}-{ithread}:'
        if max_fr[iV1] > 0:
            title = title + f'cort({orient*180/np.pi:.0f})\n'
        else: 
            title = title + '\n'
        title = title + f'thal({orient_thal*180/np.pi:.0f}-{orient_thal_theory*180/np.pi:.0f}), preset({orient_set*180/np.pi:.0f})'
        ax.set_title(title, fontsize = 'x-small')
        i = i+1
    fig.savefig(fig_fdr+'thal_set-prefMax'+conLGN_suffix + output_suffix[:-1] +'.png')

    with open(sampleFn, 'ab') as f:
        np.array([dprefMax_pick.size],dtype='u4').tofile(f)
        dprefMax_pick.astype('u4').tofile(f)
    if add_dprefToSample:
        sample = np.append(sample, dprefMax_pick)

    fig = plt.figure('thal_set-prefMin', figsize = (10,10), dpi = 300)
    grid = gs.GridSpec((ndpref+3)//4, 4, figure = fig, hspace = 0.2)
    i = 0
    for iV1 in dprefMin_pick:
        ax = fig.add_subplot(grid[i//4,np.mod(i,4)])
        iLGN_vpos = LGN_vpos[:,LGN_V1_ID[iV1]]
        iLGN_type = LGN_type[LGN_V1_ID[iV1]]
        if LR[iV1] > 0:
            all_pos = LGN_vpos[:,nLGN_I:nLGN]
            all_type = LGN_type[nLGN_I:nLGN]
        else:
            all_pos = LGN_vpos[:,:nLGN_I]
            all_type = LGN_type[:nLGN_I]

        if nLGN_V1[iV1] > 0:
            iLGN_ms = LGN_V1_s[iV1]
            max_s = np.max(iLGN_ms)
            min_s = np.min(iLGN_ms)
            for j in range(len(markers)):
                pick = all_type == j
                ax.plot(all_pos[0,pick], all_pos[1,pick], markers[j], ms = min_s/max_s*ms, mec = None, mew = 0, alpha = 0.6)

            if plotLGNsSize:
                for j in range(nLGN_V1[iV1]):
                    ax.plot(iLGN_vpos[0,j], iLGN_vpos[1,j], markers[iLGN_type[j]], ms = iLGN_ms[j]/max_s*ms*1.5, mec = 'k', mew = ms/9)
                x0 = np.average(iLGN_vpos[0,:], weights = iLGN_ms)
                y0 = np.average(iLGN_vpos[1,:], weights = iLGN_ms)
            else:
                for j in range(len(markers)):
                    pick = iLGN_type == j
                    ax.plot(iLGN_vpos[0,pick], iLGN_vpos[1,pick], markers[j], ms = ms, mec = 'k', mew = ms/4)
                x0 = np.average(iLGN_vpos[0,:])
                y0 = np.average(iLGN_vpos[1,:])

            iSubR = 0
            iSubG = 0
            iSubOn = 0
            iSubOff = 0
            for j in range(nLGN_V1[iV1]):
                jtype = iLGN_type[j]
                if jtype == 0 or jtype == 3:
                    if iSubR == 0:
                        subRed = iLGN_vpos[:,j].copy()
                    else:
                        subRed += iLGN_vpos[:,j]
                    iSubR += 1 
                if jtype == 1 or jtype == 2:
                    if iSubG == 0:
                        subGreen = iLGN_vpos[:,j].copy()
                    else:
                        subGreen += iLGN_vpos[:,j]
                    iSubG += 1 
                if jtype == 4:
                    if iSubOn == 0:
                        subOn = iLGN_vpos[:,j].copy()
                    else:
                        subOn += iLGN_vpos[:,j]
                    iSubOn += 1 
                if jtype == 5:
                    if iSubOff == 0:
                        subOff = iLGN_vpos[:,j].copy()
                    else:
                        subOff += iLGN_vpos[:,j]
                    iSubOff += 1 

            SF_text = ''
            if iSubR > 0 and iSubG > 0:
                subRed /= iSubR
                subGreen /= iSubG
                SF_RG = 1/(2*np.linalg.norm(subRed-subGreen))
                ax.plot(subRed[0], subRed[1], 'sr', ms = ms, alpha = 0.5)
                ax.plot(subGreen[0], subGreen[1], 'sg', ms = ms, alpha = 0.5)
                SF_text += f'SF_RG: {SF_RG:.1f}'

            if iSubOn > 0 and iSubOff > 0:
                subOn /= iSubOn
                subOff /= iSubOff
                SF_OnOff = 1/(2*np.linalg.norm(subOn-subOff))
                ax.plot(subOn[0], subOn[1], 'sk', ms = ms, alpha = 0.5)
                ax.plot(subOff[0], subOff[1], 'sb', ms = ms, alpha = 0.5)
                SF_text += f'SF_OnOff: {SF_OnOff:.1f}'
            if max_fr[iV1] > 0:
                orient = iPref_cort_for_ori[iV1]/nOri*np.pi + np.pi/2
                if orient > np.pi:
                    orient = orient - np.pi

            orient_thal = iPref_thal_for_ori[iV1]/nOri*np.pi + np.pi/2
            if orient_thal > np.pi:
                orient_thal = orient_thal - np.pi

            orient_thal_theory = iPref_thal_theory_for_ori[iV1]/nOri*np.pi + np.pi/2
            if orient_thal_theory > np.pi:
                orient_thal_theory = orient_thal_theory - np.pi

            orient_thal_2nd = iPref_thal_2nd_for_ori[iV1]/nOri*np.pi + np.pi/2
            if orient_thal_2nd > np.pi:
                orient_thal_2nd = orient_thal_2nd - np.pi

            orient_set = feature[1,iV1]*np.pi
            if orient_set > np.pi:
                orient_set = orient_set - np.pi
            x = np.array([np.min(iLGN_vpos[0,:]), np.max(iLGN_vpos[0,:])])
            y = np.array([np.min(iLGN_vpos[1,:]), np.max(iLGN_vpos[1,:])])
            x_thal_theory = x.copy()
            y_thal_theory = y.copy()
            x_thal = x.copy()
            y_thal = y.copy()
            x_set = x.copy()
            y_set = y.copy()
            x_thal_2nd = x.copy()
            y_thal_2nd = y.copy()

            if np.diff(y)[0] > np.diff(x)[0]:
                daspect = y[1]-y[0]
            else:
                daspect = x[1]-x[0]

            ax.set_xlim(left = x[1] - daspect*1.1, right = x[0] + daspect*1.1)
            ax.set_ylim(bottom = y[1] - daspect*1.1, top = y[0] + daspect*1.1)
            if SF_text:
                ax.text(0.5, 0.85, SF_text +  ' cyc/deg', transform=ax.transAxes, ha='center', fontsize='x-small')

            if np.diff(y)[0] > np.diff(x)[0]:
                if max_fr[iV1] > 0:
                    x = (y-y0) / np.tan(orient) + x0
                x_thal = (y_thal-y0) / np.tan(orient_thal) + x0
                x_thal_theory = (y_thal_theory-y0) / np.tan(orient_thal_theory) + x0
                x_set = (y_set-y0) / np.tan(orient_set) + x0
                x_thal_2nd = (y_thal_2nd-y0) / np.tan(orient_thal_2nd) + x0
            else:
                if max_fr[iV1] > 0:
                    y = np.tan(orient)*(x-x0) + y0
                y_thal = np.tan(orient_thal)*(x_thal-x0) + y0
                y_thal_theory = np.tan(orient_thal_theory)*(x_thal_theory-x0) + y0
                y_set = np.tan(orient_set)*(x_set-x0) + y0
                y_thal_2nd = np.tan(orient_thal_2nd)*(x_thal_2nd-x0) + y0

            ax.plot(x0, y0, '*b', ms = 1.0)
            if max_fr[iV1] > 0:
                ax.plot(x, y, '-k', lw = 1.0, label = 'cort.')
            ax.plot(x_set, y_set, ':r', lw = 1.5, label = 'preset')
            ax.plot(x_thal, y_thal, ':b', lw = 1.0, label = 'thal.')
            ax.plot(x_thal_theory, y_thal_theory, ':y', lw = 0.6, label = 'thal. theory.')
            ax.plot(x_thal_2nd, y_thal_2nd, ':c', lw = 0.3, label = 'thal2')

            ax.set_aspect('equal')
        else:
            raise Exception(f'the {i}th dprefMin ({iV1}) has no LGN input!')

        if i == dprefMin_pick.size-1:
            legend_elements = []
            for m, l in zip(markers, type_labels):
                legend_elements.append(Line2D([0], [0], marker=m[0], color=m[1], label = l))
            marker_legend = plt.legend(handles=legend_elements, bbox_to_anchor = (1,0), loc='lower left', fontsize='xx-small')
            ax.legend(fontsize='xx-small', bbox_to_anchor=(1,1), loc='upper left')
            ax.add_artist(marker_legend)

        ax.tick_params(axis = 'x', labelsize = 'xx-small')
        ax.tick_params(axis = 'y', labelsize = 'xx-small')
        iblock = iV1//blockSize
        ithread = np.mod(iV1, blockSize)
        title = f'{iblock}-{ithread}:'
        if max_fr[iV1] > 0:
            title = title + f'cort({orient*180/np.pi:.0f})\n'
        else: 
            title = title + '\n'
        title = title + f'thal({orient_thal*180/np.pi:.0f}-{orient_thal_theory*180/np.pi:.0f}), preset({orient_set*180/np.pi:.0f})'
        ax.set_title(title, fontsize = 'x-small')
        i = i+1
    fig.savefig(fig_fdr+'thal_set-prefMin'+conLGN_suffix+ output_suffix[:-1]+'.png')
    with open(sampleFn, 'ab') as f:
        np.array([dprefMin_pick.size],dtype='u4').tofile(f)
        dprefMin_pick.astype('u4').tofile(f)

    if add_dprefToSample:
        sample = np.append(sample, dprefMin_pick)

    def alignTC(iPref, nOri, q):
        n = iPref.size
        ori = np.arange(nOri)


        # nOri = 4
        # ori: 0, 1, 2, 3, +
        # midpos: 2
        # if iPref = 3
        # roll -1
        # aligned: 1,2,3,0, +
        # + = 1
        #
        aligned = np.tile(ori, (n, 1))
        if np.mod(nOri, 2) == 0:
            mid_pos = nOri//2
        else:
            mid_pos = (nOri-1)//2
            # actual mid_pos = (nOri-1)//2 + 0.5
            #
        assert(aligned.shape[0] == n)
        for i in range(n):
            aligned[i,:] = np.roll(aligned[i,:], mid_pos - iPref[i])
            if np.mod(nOri, 2) == 1: 
                inext = np.mod(iPref[i]+1,nOri)
                iprev = np.mod(iPref[i]+nOri-1,nOri)
                # decide the next highest fr to join the two mid pos
                if fr[iprev,i] > fr[inext,i]: #  high - max - low
                    aligned[i,:] = np.roll(aligned[i,:], 1)
                else: # low - max - high
                    pass

        aligned = np.hstack((aligned, np.reshape(aligned[:,0],(n,1))))

        return aligned

    aligned_thal = alignTC(iPref_thal, nOri, fr)
    aligned_cort = alignTC(iPref_cort, nOri, fr)

    aligned_fr = np.array([fr[aligned_cort[i,:],i] for i in range(nV1)])

    if plotSpatialOP:

        with open(V1_allposFn, 'r') as f:
            _nblock, _blockSize, dataDim = np.fromfile(f, 'u4', count=3)
            assert(nV1 == _nblock*_blockSize)
            coord_span = np.fromfile(f, 'f8', count=4)
            pos = np.reshape(np.fromfile(f, 'f8', count = 2*nV1), (2,nV1))

        fig = plt.figure('spatial_OP', figsize = (6,10))
        ax = fig.add_subplot(321)
        orient = iPref/nOri*180
        orient_cort = iPref_cort_for_ori/nOri*180
        orient_thal = iPref_thal_for_ori/nOri*180
        
        bars0,_ = np.histogram(orient[epick[nLGN_V1[epick]>SCsplit]], bins = OPbins)
        bars,_ = np.histogram(orient_cort[epick[nLGN_V1[epick]>SCsplit]], bins = OPbins)
        thal_bars,_ = np.histogram(orient_thal[epick[nLGN_V1[epick]>SCsplit]], bins = OPbins)
        # deal with pi period
        if not fitTC:
            assert(bars[-1] == 0) 
        assert(thal_bars[-1] == 0) 

        bars[-1] += bars[0]
        bars = bars[1:]
        bars0[-1] += bars0[0]
        bars0 = bars0[1:]
        thal_bars[-1] += thal_bars[0]
        thal_bars = thal_bars[1:]

        ax.bar(OPs, bars0, alpha = 0.5, width=180/nOri*0.8, label = 'preset')
        ax.bar(OPs, bars-bars0, alpha = 0.5, width=180/nOri*0.8, label = 'simu.-preset')
        ax.bar(OPs, thal_bars-bars0, alpha = 0.5, width=180/nOri*0.8, label = 'thal.-preset')
        color = np.tile(np.array([0,0.6,1], dtype = float), (nOri,1))
        color[:,0] = OPs/180
        color[color[:,0] < 0,0] = 0
        color[color[:,0] > 1,0] = 1.0
        ybot = ax.get_ylim()[0]
        for i in range(nOri):
            ax.plot(OPs[i], (ybot + np.min(bars-bars0))/2, '.', ms = 10, mfc=clr.hsv_to_rgb(color[i,:]), mew=0, fillstyle='full', alpha = 0.8)

        ax.legend(fontsize = 'xx-small', frameon = False, loc = 'lower center', ncol = 3, bbox_to_anchor=(0.5,1.0) )
        ax.set_ylabel(f'#E: nLGN>{SCsplit}')
        ax.set_xticks([60,120,180])

        ax = fig.add_subplot(323)
        bars0,_ = np.histogram(orient[epick[nLGN_V1[epick]<=SCsplit]], bins = OPbins)
        bars,_ = np.histogram(orient_cort[epick[nLGN_V1[epick]<=SCsplit]], bins = OPbins)
        # deal with pi period
        bars[-1] += bars[0]
        bars = bars[1:]
        bars0[-1] += bars0[0]
        bars0 = bars0[1:]
        ax.bar(OPs, bars0, alpha = 0.5, width=180/nOri*0.8)
        ax.bar(OPs, bars-bars0, alpha = 0.5, width=180/nOri*0.8)
        ax.set_ylabel(f'#E: nLGN<={SCsplit}')
        ax.set_xticks([60,120,180])

        ax = fig.add_subplot(325)
        bars0,_ = np.histogram(orient[ipick], bins = OPbins)
        bars,_ = np.histogram(orient_cort[ipick], bins = OPbins)
        # deal with pi period
        bars[-1] += bars[0]
        bars = bars[1:]
        bars0[-1] += bars0[0]
        bars0 = bars0[1:]
        ax.bar(OPs, bars0, alpha = 0.5, width=180/nOri*0.8)
        ax.bar(OPs, bars-bars0, alpha = 0.5, width=180/nOri*0.8)
        ax.set_ylabel(f'#I')
        ax.set_xlabel(f'OP (deg)')
        ax.set_xticks([60,120,180])

        def plotScatter(ori, ax, plotI = False):
            sat0 = 0.0
            ms1 = 0.08
            ms2 = 0.08
            ms3 = 0.08
            mk1 = 'o'
            mk2 = '^'
            mk3 = 's'
            # contra
            pick = epick[np.logical_and(LR[epick] > 0, nLGN_V1[epick] > SCsplit)]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(max_fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+max_fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = ori[pick]/180
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            pick = epick[np.logical_and(LR[epick] > 0, nLGN_V1[epick] <= SCsplit)]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(max_fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+max_fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = ori[pick]/180
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            pick = ipick[LR[ipick] > 0]
            nnI = pick.size
            if nnI > 0 and plotI:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(max_fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+max_fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = ori[pick]/180
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax.scatter(pos[0,pick], pos[1,pick], s = ms3, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk3)

            ms1 = 0.08
            ms2 = 0.08
            ms3 = 0.08
            mk1 = 'P'
            mk2 = 'v'
            mk3 = 'D'
            # ipsi
            pick = epick[np.logical_and(LR[epick] < 0, nLGN_V1[epick] > SCsplit)]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(max_fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+max_fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = ori[pick]/180
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            pick = epick[np.logical_and(LR[epick] < 0, nLGN_V1[epick] <= SCsplit)]
            nnE = pick.size
            if nnE > 0:
                color = np.tile(np.array([0,0,1], dtype = float), (nnE,1))
                frMax = np.max(max_fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+max_fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = ori[pick]/180
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax.scatter(pos[0,pick], pos[1,pick], s = ms2, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk2)

            pick = ipick[LR[ipick] < 0]
            nnI = pick.size
            if nnI > 0 and plotI:
                color = np.tile(np.array([0,0,1], dtype = float), (nnI,1))
                frMax = np.max(max_fr[pick])
                if frMax == 0:
                    color[:,1] = sat0
                else:
                    color[:,1] = sat0 + np.log(1+max_fr[pick]/frMax)/np.log(2)*(1-sat0)

                color[:,0] = ori[pick]/180
                color[color[:,0] < 0,0] = 0
                color[color[:,0] > 1,0] = 1
                ax.scatter(pos[0,pick], pos[1,pick], s = ms3, c = clr.hsv_to_rgb(color), edgecolors = 'none', marker = mk3)

        ax = fig.add_subplot(322)
        plotScatter(orient,ax)
        ax.set_title('preset. fr')
        ax.set_aspect('equal')

        ax = fig.add_subplot(324)
        plotScatter(orient_cort,ax)
        ax.set_title('sim. fr')
        ax.set_aspect('equal')

        ax = fig.add_subplot(326)
        plotScatter(orient_thal,ax)
        ax.set_title('thal. F1')
        ax.set_aspect('equal')

        fig.savefig(fig_fdr+'spatial_OP'+output_suffix[:-1] + '.png', dpi = 900)


    if plotTC:

        fig = plt.figure('population-heatTC', figsize = (14,16))
        grid = gs.GridSpec(5, 5, figure = fig, hspace = 0.3, wspace = 0.3)
        # TC based on LGN response

        ax = fig.add_subplot(grid[0,0])
        ytarget = np.array([fr[aligned_thal[i,:],i] for i in eSpick_act]).flatten()
        target = np.tile(op, (eSpick_act.size, 1)).flatten()
        assert(ytarget.size == target.size)
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.FR')
        ax.set_title('LGN-based simple.E')

        ax = fig.add_subplot(grid[1,0])
        ytarget = np.array([cFF[aligned_thal[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_thal[i,:],i] for i in eSpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cFF')

        ax = fig.add_subplot(grid[2,0])
        ytarget = np.array([cE[aligned_thal[i,:],:,i,0].sum(axis = 1) for i in eSpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cE')

        ax = fig.add_subplot(grid[3,0])
        ytarget = np.array([cI[aligned_thal[i,:],:,i,0].sum(axis = 1) for i in eSpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cI')

        ax = fig.add_subplot(grid[4,0])
        ytarget = np.array([depC[aligned_thal[i,:],i,0] for i in eSpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.depC')

        ax = fig.add_subplot(grid[0,1])
        ytarget = np.array([fr[aligned_thal[i,:],i] for i in ipick_act]).flatten()
        target = np.tile(op, (ipick_act.size, 1)).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.FR')
        ax.set_title('LGN-based I')

        ax = fig.add_subplot(grid[1,1])
        ytarget = np.array([cFF[aligned_thal[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_thal[i,:],i] for i in ipick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cFF')

        ax = fig.add_subplot(grid[2,1])
        ytarget = np.array([cE[aligned_thal[i,:],:,i,0].sum(axis = 1) for i in ipick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cE')

        ax = fig.add_subplot(grid[3,1])
        ytarget = np.array([cI[aligned_thal[i,:],:,i,0].sum(axis = 1) for i in ipick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cI')

        ax = fig.add_subplot(grid[4,1])
        ytarget = np.array([depC[aligned_thal[i,:],i,0] for i in ipick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.depC')

        # TC based on cortical response

        ax = fig.add_subplot(grid[0,2])
        ytarget = aligned_fr[eSpick_act,:].flatten()
        target = np.tile(op, (eSpick_act.size, 1)).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.FR')
        ax.set_title('cortical-based simple.E')

        ax = fig.add_subplot(grid[1,2])
        ytarget = np.array([cFF[aligned_cort[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_cort[i,:],i] for i in eSpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cFF')

        ax = fig.add_subplot(grid[2,2])
        ytarget = np.array([cE[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in eSpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cE')

        ax = fig.add_subplot(grid[3,2])
        ytarget = np.array([cI[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in eSpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cI')

        ax = fig.add_subplot(grid[4,2])
        ytarget = np.array([depC[aligned_cort[i,:],i,0] for i in eSpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.depC')

        ax = fig.add_subplot(grid[0,3])
        ytarget = aligned_fr[eCpick_act,:].flatten()
        target = np.tile(op, (eCpick_act.size, 1)).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.FR')
        ax.set_title('cortical-based complex.E')

        ax = fig.add_subplot(grid[1,3])
        ytarget = np.array([cFF[aligned_cort[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_cort[i,:],i] for i in eCpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cFF')

        ax = fig.add_subplot(grid[2,3])
        ytarget = np.array([cE[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in eCpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cE')

        ax = fig.add_subplot(grid[3,3])
        ytarget = np.array([cI[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in eCpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cI')

        ax = fig.add_subplot(grid[4,3])
        ytarget = np.array([depC[aligned_cort[i,:],i,0] for i in eCpick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.depC')

        ax = fig.add_subplot(grid[0,4])
        ytarget = aligned_fr[ipick_act,:].flatten()
        target = np.tile(op, (ipick_act.size, 1)).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.FR')
        ax.set_title('cortical based I')

        ax = fig.add_subplot(grid[1,4])
        ytarget = np.array([cFF[aligned_cort[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_cort[i,:],i] for i in ipick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cFF')

        ax = fig.add_subplot(grid[2,4])
        ytarget = np.array([cE[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in ipick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cE')

        ax = fig.add_subplot(grid[3,4])
        ytarget = np.array([cI[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in ipick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cI')

        ax = fig.add_subplot(grid[4,4])
        ytarget = np.array([depC[aligned_cort[i,:],i,0] for i in ipick_act]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.depC')

        fig.savefig(fig_fdr+'popTC'+output_suffix[:-1] + '.png', dpi = 900)

    if plotF1F0:

        fig = plt.figure('F1F0')

        nbins = 10
        F1F0_op = F1F0.T.flatten()[iPref_thal + np.arange(nV1)*nOri]
        ax = fig.add_subplot(221)
        ax.hist(F1F0_op[eSpick_act], bins = nbins, color = 'r', alpha = 0.3)
        ax.set_title('thal-based')

        ax = fig.add_subplot(223)
        ax.hist(F1F0_op[iSpick_act], bins = nbins, color = 'b', alpha = 0.3)

        F1F0_op = F1F0.T.flatten()[iPref_cort + np.arange(nV1)*nOri]
        ax = fig.add_subplot(222)
        ax.hist(F1F0_op[eSpick_act], bins = nbins, color = 'r', alpha = 0.3)
        ax.set_title('cort-based')
        ax2 = ax.twinx()
        ax2.hist(F1F0_op[eCpick_act], bins = nbins, color = 'm', alpha = 0.3)

        ax = fig.add_subplot(224)
        ax.hist(F1F0_op[iSpick_act], bins = nbins, color = 'b', alpha = 0.3)
        ax2 = ax.twinx()
        ax2.hist(F1F0_op[iCpick_act], bins = nbins, color = 'c', alpha = 0.3)
        fig.savefig(fig_fdr+'F1F0'+output_suffix[:-1] + '.png', dpi = 900)

        max_lgn = max(nLGN_V1)
        fig = plt.figure('F1F0-nLGN', figsize = (6, 2*max_lgn+1))
        grid = gs.GridSpec(max_lgn+1, 2, figure = fig, hspace = 0.2)
        F1F0_op = F1F0.T.flatten()[iPref_cort + np.arange(nV1)*nOri]
        for i in range(max_lgn+1):
            ax = fig.add_subplot(grid[i,0])
            pick = epick[np.logical_and(nLGN_V1[epick] == i, max_fr[epick] > fr_thres)]
            ax.hist(F1F0_op[pick], bins = nbins, color = 'r')
            ax = fig.add_subplot(grid[i,1])
            pick = ipick[np.logical_and(nLGN_V1[ipick] == i, max_fr[ipick] > fr_thres)]
            ax.hist(F1F0_op[pick], bins = nbins, color = 'b')
            ax.set_ylabel(f'# nLGN={i}')
            if i == max_lgn:
                ax.set_xlabel('F1F0 at PO')
        fig.savefig(fig_fdr+'F1F0-nLGN'+output_suffix[:-1] + '.png', dpi = 900)

    if plotFR:
        if fitTC:
            fr_target = fit_max_fr
        else:
            fr_target = max_fr

        max_lgn = max(nLGN_V1)
        fig = plt.figure('FR-nLGN', figsize = (6, 2*max_lgn+1))
        grid = gs.GridSpec(max_lgn+1, 2, figure = fig, hspace = 0.2)
        for i in range(max_lgn+1):
            ax = fig.add_subplot(grid[i,0])
            pick = epick[nLGN_V1[epick] == i]
            ax.hist(fr_target[pick], bins = nbins, color = 'r')
            pick = epick[np.logical_and(nLGN_V1[epick] == i, fr_target[epick] == 0)]
            ax.bar(0, pick.size, color = 'm', alpha = 0.3)
            ax = fig.add_subplot(grid[i,1])
            pick = ipick[nLGN_V1[ipick] == i]
            ax.hist(fr_target[pick], bins = nbins, color = 'b')
            pick = ipick[np.logical_and(nLGN_V1[ipick] == i, fr_target[ipick] == 0)]
            ax.bar(0, pick.size, color = 'c', alpha = 0.3)
            ax.set_ylabel(f'# nLGN={i}')
            if i == max_lgn:
                ax.set_xlabel('FR at PO')
        fig.savefig(fig_fdr+'FR-nLGN'+output_suffix[:-1] + '.png', dpi = 900)

    if plotSample:
        theta = np.linspace(0, np.pi, 37)
        itheta = np.arange(nOri)
        itheta = np.insert(itheta, 0, nOri-1)
        def plotPhase(arr, phase_range, color):
            i = 0
            for it in itheta:
                i_phase = 1000/arr[it, 0]
                phase = (arr[it,2]+np.pi)/(2*np.pi)*i_phase
                n_phase = int(phase_range/i_phase)
                r_phase = np.mod(phase_range, i_phase)
                if r_phase > i_phase:
                    n_phase += 1
                phase_point = np.arange(n_phase)*i_phase + phase
                ax.plot(np.zeros(n_phase) + op[i], phase_point, '_', c = color, alpha = 0.5)
                i += 1

        #sampleList = np.unique(np.hstack((sample,noOpt))).astype('i4')
        sampleList = sample
        i = 0
        lw = 0.5
        for iV1 in sampleList:
        #for iV1 in np.array(noOpt):
            iblock = iV1//blockSize
            ithread = np.mod(iV1, blockSize)
            fig = plt.figure('sampleTC-{iblock}-{ithread}', figsize = (6,6))
            grid = gs.GridSpec(4, 5, figure = fig, hspace = 0.3, wspace = 0.3)
            ax = fig.add_subplot(grid[:2,:2]) # fr, cE, cI, depC
            ax2 = ax.twinx()
            ax.plot(op, fr[itheta,iV1], '-ok')
            pref_title = f'V1:{iPref_cort_for_ori[iV1]*dOri:.0f}, gFF_F1F0:{iPref_thal_for_ori[iV1]*dOri:.0f}, preset:{iPref[iV1]*dOri:.0f}'
            if fitTC:
                ax.plot(theta/np.pi*180, von_Mises(theta, fit_a[iV1], fit_b[iV1], fit_s[iV1], fit_pref[iV1]), ':k')
                ax.set_title(f'a={fit_a[iV1]:.2f}, b={fit_b[iV1]:.2f}, s={fit_s[iV1]:.2f}, {fit_pref[iV1]*180/np.pi:.0f}, {iV1 in noOpt}\n' + pref_title, fontsize = 'xx-small')
            else:
                ax.set_title(pref_title, fontsize = 'x-small')

            ax.set_ylim(bottom = 0)

            # substrate layer of 0-20%, 80-100%
            ax2.fill_between(op, depC_percent[itheta,iV1,0], depC_percent[itheta,iV1,1], facecolor = 'm', alpha = 0.2)
            ax2.fill_between(op, depC_percent[itheta,iV1,3], depC_percent[itheta,iV1,4], facecolor = 'm', alpha = 0.2)
            if iModel:
                ax2.fill_between(op, w_percent[itheta,iV1,0], w_percent[itheta,iV1,1], facecolor = 'c', alpha = 0.2)
                ax2.fill_between(op, w_percent[itheta,iV1,3], w_percent[itheta,iV1,4], facecolor = 'c', alpha = 0.2)
            ax2.fill_between(op, cTotal_percent[itheta,iV1,0], cTotal_percent[itheta,iV1,1], facecolor = 'k', alpha = 0.2)
            ax2.fill_between(op, cTotal_percent[itheta,iV1,3], cTotal_percent[itheta,iV1,4], facecolor = 'k', alpha = 0.2)
            ax2.fill_between(op, cFF_percent[itheta,iV1,0], cFF_percent[itheta,iV1,1], facecolor = 'g', alpha = 0.2)
            ax2.fill_between(op, cFF_percent[itheta,iV1,3], cFF_percent[itheta,iV1,4], facecolor = 'g', alpha = 0.2)
            ax2.fill_between(op, cE_percent[itheta,iV1,0], cE_percent[itheta,iV1,1], facecolor = 'r', alpha = 0.2)
            ax2.fill_between(op, cE_percent[itheta,iV1,3], cE_percent[itheta,iV1,4], facecolor = 'r', alpha = 0.2)
            ax2.fill_between(op, cI_percent[itheta,iV1,0], cI_percent[itheta,iV1,1], facecolor = 'b', alpha = 0.2)
            ax2.fill_between(op, cI_percent[itheta,iV1,3], cI_percent[itheta,iV1,4], facecolor = 'b', alpha = 0.2)

            # substrate layer of 20%-80%
            ax2.fill_between(op, depC_percent[itheta,iV1,1], depC_percent[itheta,iV1,3], facecolor = 'm', alpha = 0.4)
            if iModel:
                ax2.fill_between(op, w_percent[itheta,iV1,1], w_percent[itheta,iV1,3], facecolor = 'c', alpha = 0.4)
            ax2.fill_between(op, cTotal_percent[itheta,iV1,1], cTotal_percent[itheta,iV1,3], facecolor = 'k', alpha = 0.4)
            ax2.fill_between(op, cFF_percent[itheta,iV1,1], cFF_percent[itheta,iV1,3], facecolor = 'g', alpha = 0.4)
            ax2.fill_between(op, cE_percent[itheta,iV1,1], cE_percent[itheta,iV1,3], facecolor = 'r', alpha = 0.4)
            ax2.fill_between(op, cI_percent[itheta,iV1,1], cI_percent[itheta,iV1,3], facecolor = 'b', alpha = 0.4)

            # mean, median
            ax2.plot(op, depC_percent[itheta,iV1,2], ':m', lw = lw)
            ax2.plot(op, depC[itheta,iV1,0], '--m', lw = lw)
            if iModel:
                ax2.plot(op, w_percent[itheta,iV1,2], ':c', lw = lw)
                ax2.plot(op, w[itheta,iV1,0], '--c', lw = lw)
            ax2.plot(op, cTotal_percent[itheta,iV1,2], ':k', lw = lw)
            ax2.plot(op, cTotal[itheta,iV1,0], '--k', lw = lw)
            ax2.plot(op, cFF_percent[itheta,iV1,2], ':g', lw = lw)
            ax2.plot(op, cFF[itheta,:,iV1,0].sum(1), '--g', lw = lw)
            ax2.plot(op, cE_percent[itheta,iV1,2], ':r', lw = lw)
            ax2.plot(op, cE[itheta,:,iV1,0].sum(1), '--r', lw = lw)
            ax2.plot(op, cI_percent[itheta,iV1,2], ':b', lw = lw)
            ax2.plot(op, cI[itheta,:,iV1,0].sum(1), '--b', lw = lw)

            left, right = ax2.get_xlim()
            ax2.plot([left, right], [0, 0], '--y', lw = 3*lw)
            ax2.set_xlim(left = left, right = right)
            ax.set_xlabel('orientation')
            ax.set_ylabel('fr')
            ax2.set_ylabel('current')

            ax = fig.add_subplot(grid[2:,:2]) # gE, gI, gapI
            if nLGN_V1[iV1] > 0:
                ax.plot(op, gFF_max[itheta,iV1], 'g')
            ax.plot(op, gE[itheta,:,iV1,0].sum(1), 'r')
            ax.plot(op, gI[itheta,:,iV1,0].sum(1), 'b')
            ax2 = ax.twinx()
            ax2.plot(op, gFF_F1F0[itheta,iV1], ':g')
            ax2.plot(op, F1F0[itheta,iV1], ':k')
            ax.set_xlabel('orientation')
            ax.set_ylabel('conductance')
            ax2.set_ylabel('F1/F0')
            if ithread < mI:
                iI = iblock*mI + ithread - mE
                ax.plot(op, cGap[itheta, iI, 0], 'c')

            ax = fig.add_subplot(grid[:2,3:]) # gE, gI, gapI
            msRatio = 5
            ax.plot(op, cTotal_freq[itheta,iV1, 0], '-k', alpha = 0.5)
            ax.scatter(op, cTotal_freq[itheta,iV1, 0], s = cTotal_freq[itheta,iV1, 1]*msRatio, c = 'k', alpha = 0.5)

            ax.plot(op, cFF_freq[itheta,iV1, 0], '-g', alpha = 0.5)
            ax.scatter(op, cFF_freq[itheta,iV1, 0], s = cFF_freq[itheta,iV1, 1]*msRatio, c = 'g', alpha = 0.5)

            ax.plot(op, cE_freq[itheta,iV1, 0], '-r', alpha = 0.5)
            ax.scatter(op, cE_freq[itheta,iV1, 0], s = cE_freq[itheta,iV1, 1]*msRatio, c = 'r', alpha = 0.5)

            ax.plot(op, cI_freq[itheta,iV1, 0], '-b', alpha = 0.5)
            ax.scatter(op, cI_freq[itheta,iV1, 0], s = cI_freq[itheta,iV1, 1]*msRatio, c = 'b', alpha = 0.5)

            ax.plot(op, depC_freq[itheta,iV1, 0], '-m', alpha = 0.5)
            ax.scatter(op, depC_freq[itheta,iV1, 0], s = depC_freq[itheta,iV1, 1]*msRatio, c = 'm', alpha = 0.5)
            ax.set_ylabel('Freq Hz')
            min_freq = np.min(np.hstack((cFF_freq[:,iV1,0], cE_freq[:,iV1,0],cI_freq[:,iV1,0],depC_freq[:,iV1,0],cTotal_freq[:,iV1,0])))

            if iModel:
                ax.plot(op, w_freq[itheta,iV1, 0], '-c')
                ax.scatter(op, w_freq[itheta,iV1, 0], s = w_freq[itheta,iV1, 1], c = 'c', alpha = 0.5)
                min_freq = min(min_freq, w_freq[:,iV1,0].min())

            ax = fig.add_subplot(grid[2:,3:]) # gE, gI, gapI
            
            phase_range = 1000/min_freq
            plotPhase(cTotal_freq[:,iV1,:], phase_range, 'k')
            plotPhase(cFF_freq[:,iV1,:], phase_range, 'g')
            plotPhase(cE_freq[:,iV1,:], phase_range, 'r')
            plotPhase(cI_freq[:,iV1,:], phase_range, 'b')
            plotPhase(depC_freq[:,iV1,:], phase_range, 'm')
            if iModel:
                plotPhase(w_freq[:,iV1,:], phase_range, 'c')
            ax.set_ylabel('Phase as time (ms)')
            ax.set_ylim(bottom = 0)

            if i < ns:
                fig.savefig(fig_fdr+ f'sampleTC-{iblock}-{ithread}#{nLGN_V1[iV1]}' +output_suffix[:-1]+ '.png', dpi = 300)
            else:
                if i < ns+ndpref:
                    fig.savefig(fig_fdr+ f'MaxDiffTC-{iblock}-{ithread}#{nLGN_V1[iV1]}' +output_suffix[:-1]+ '.png', dpi = 300)
                else:
                    fig.savefig(fig_fdr+ f'MinDiffTC-{iblock}-{ithread}#{nLGN_V1[iV1]}' +output_suffix[:-1]+ '-min.png', dpi = 300)

            plt.close(fig)
            i = i + 1

    def get_gOSI(tc, nOri):
        ntheta = tc.shape[0]
        nsample = tc.shape[1]
        theta = np.exp(np.tile(1j*np.arange(nOri)/nOri*2*np.pi, (nsample,1)))
        assert(theta.shape[1] == ntheta)
        assert(theta.shape[0] == nsample)
        gOSI = np.zeros(tc.shape[1])
        pick = np.max(tc, axis = 0) > 0
        gOSI[pick] = np.abs(np.sum(tc.T*theta, axis = 1)[pick])/(np.sum(tc.T,axis = 1)[pick])
        assert(gOSI.size == nsample)
        return gOSI

    def get_OSI(tc, nOri):
        assert(tc.shape[1] == nOri+1)
        tc_min = tc[:,0]
        if np.mod(nOri, 2) == 0:
            tc_max = tc[:,nOri//2]
        else:
            tc_max = tc[:,(nOri-1)//2]
        OSI = np.zeros(tc.shape[0])
        pick = tc_max>0
        OSI[pick] = (tc_max[pick] - tc_min[pick]) / (tc_min[pick] + tc_max[pick])
        return OSI

    if plotOSI:
        osi_range = np.linspace(0,1,11)
        if fitTC:
            gOSI = fit_gOSI
        else:
            gOSI = get_gOSI(fr, nOri)
        OSI = get_OSI(aligned_fr, nOri)

        fig = plt.figure('OSI_dist', figsize = (4, nType*2))
        grid = gs.GridSpec(nType, 2, figure = fig, hspace = 0.2)
        for iType in range(nType):
            type_pick = np.hstack([np.arange(typeAcc[iType], typeAcc[iType+1]) + iblock*blockSize for iblock in range(nblock)])
            if iType < nTypeE:
                c = ('r', 'm')
            else:
                c = ('b', 'c')
            
            type_Spick = type_pick[np.logical_and(nLGN_V1[type_pick] >  SCsplit, max_fr[type_pick] > fr_thres)]
            type_Cpick = type_pick[np.logical_and(nLGN_V1[type_pick] <= SCsplit, max_fr[type_pick] > fr_thres)]
            ax = fig.add_subplot(grid[iType,0])
            #ax.hist(gOSI[type_Spick], bins = osi_range, color = c[0])
            #ax.hist(gOSI[type_Cpick], bins = osi_range, color = c[1])
            ax.hist(gOSI[type_Spick], bins = 10, color = c[0])
            ax.hist(gOSI[type_Cpick], bins = 10, color = c[1])
            if iType == 0:
                ax.set_title('gOSI (1-CV)')
            ax = fig.add_subplot(grid[iType,1])
            ax.hist(OSI[type_Spick], bins = osi_range, color = c[0])
            ax.hist(OSI[type_Cpick], bins = osi_range, color = c[1])
            if iType == 0:
                ax.set_title('OSI')
        fig.savefig(fig_fdr+ f'TC-OSI' +output_suffix[:-1]+ '.png', dpi = 300)
        plt.close(fig)
            
if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 12:
        print(sys.argv)
        raise Exception('not enough argument for getTuningCurve(output_suffix, res_suffix, conLGN_suffix, conV1_suffix, res_fdr, setup_fdr, data_fdr, fig_fdr, nOri, fitTC, fitDataReady)')
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
        setup_fdr = sys.argv[6]
        print(setup_fdr)
        data_fdr = sys.argv[7]
        print(data_fdr)
        fig_fdr = sys.argv[8]
        print(fig_fdr)
        nOri = int(sys.argv[9])
        print(nOri)
        if sys.argv[10] == 'True':
            fitTC = True
            print('use TC fitted with von Mises function')
        else:
            fitTC = False
            print('won\'t use fitted TC')
        if sys.argv[11] == 'True':
            fitDataReady = True
        else:
            fitDataReady = False

    gatherTuningCurve(output_suffix, res_suffix, conLGN_suffix, conV1_suffix, res_fdr, setup_fdr, data_fdr,fig_fdr, nOri, fitTC, fitDataReady)
