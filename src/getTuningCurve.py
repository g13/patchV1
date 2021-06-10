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
from matplotlib import cm
import sys
from readPatchOutput import *
from global_vars import LGN_vposFn, featureFn, seed
#import multiprocessing as mp
np.seterr(invalid = 'raise')

def gatherTuningCurve(output_suffix, conLGN_suffix, conV1_suffix, outputfdr, nOri, fitTC, fitDataReady):
    if outputfdr:
        outputfdr = outputfdr+"/"

    output_suffix = "_" + output_suffix + "_"
    conLGN_suffix = "_" + conLGN_suffix
    conV1_suffix = "_" + conV1_suffix

    fr_thres = 1
    ns = 20
    ndpref = 10
    SCsplit = 0
    heatBins = 25
    sample = None
    pLog = False

    #plotTC = False
    #plotFR = False
    #plotSample = False
    plotLGNsSize = False
    #plotF1F0 = False
    #plotOSI = False

    plotTC = True
    plotFR = True 
    plotSample = True
    #plotLGNsSize = True
    plotF1F0 = True
    plotOSI = True

    LGN_V1_ID_file = 'LGN_V1_idList'+conLGN_suffix+'.bin'
    LGN_V1_s_file = 'LGN_V1_sList'+conLGN_suffix+'.bin'

    pref_file = 'cort_pref' + output_suffix[:-1] + '.bin'
    fit_file = 'fit_data' + output_suffix[:-1] + '.bin'

    parameterFn = "patchV1_cfg" +output_suffix + "1.bin"

    LGN_V1_s = readLGN_V1_s0(LGN_V1_s_file)
    LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_ID_file)
    nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type = readLGN_vpos(LGN_vposFn)

    nV1_0 = nLGN_V1.size
    
    mean_data_files = ["mean_data" + output_suffix + str(iOri+1) + ".bin" for iOri in range(nOri)]
    for i in range(nOri):
        with open(mean_data_files[i], 'rb') as f:
            if i == 0:
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
                cGap = np.empty((nOri,nI,2))
                F1F0    = np.empty((nOri,nV1))
                gFF_F1F0= np.empty((nOri,nV1))
                gE_F1F0 = np.empty((nOri,nV1))
                gI_F1F0 = np.empty((nOri,nV1))
            else:
                _nV1 = np.fromfile(f, 'i4', count=1)[0]
                _nI = np.fromfile(f, 'i4', count=1)[0]
                _ngFF = np.fromfile(f, 'i4', count=1)[0]
                _ngE = np.fromfile(f, 'i4', count=1)[0]
                _ngI = np.fromfile(f, 'i4', count=1)[0]
                if _nI != nI or _nV1 != nV1 or _ngFF != ngFF or _ngE != ngE or _ngI != ngI:
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
            cGap[i,:,:] = np.fromfile(f, 'f8', count = 2*nI).reshape((nI,2))
            F1F0[i,:] = np.fromfile(f, 'f8', count = nV1)
            gFF_F1F0[i,:] = np.fromfile(f, 'f8', count = nV1)
            gE_F1F0[i,:] = np.fromfile(f, 'f8', count = nV1)
            gI_F1F0[i,:] = np.fromfile(f, 'f8', count = nV1)
    
    nE = nV1 - nI
    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, mE, mI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn = read_cfg(parameterFn)
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

    iop = np.arange(nOri)
    op = np.arange(nOri)/nOri*180
    opEdges = (np.arange(nOri+1)-0.5)/nOri*180
    op = np.append(op, 180)
    iop = np.append(iop, 0)
    opEdges = np.append(opEdges, 180*(1+0.5/nOri))
    print(opEdges)

    iPref_thal = np.argmax(gFF_F1F0, axis = 0)
    print([np.min(iPref_thal), np.max(iPref_thal), nOri])
    iPref_cort = np.argmax(fr, axis = 0)


    max_pick = iPref_thal + np.arange(nV1)*nOri

    gFF_F1F0_2nd = gFF_F1F0.T.flatten().copy()
    gFF_F1F0_2nd[max_pick] = 0
    gFF_F1F0_2nd = np.reshape(gFF_F1F0_2nd, (nV1, nOri)).T
    iPref_thal_2nd = np.argmax(gFF_F1F0_2nd, axis = 0)

    iPref_thal_for_ori = np.mod(iPref_thal + 1, nOri)
    iPref_thal_2nd_for_ori = np.mod(iPref_thal_2nd + 1, nOri)
    iPref_cort_for_ori = np.mod(iPref_cort + 1, nOri)

    featureType = np.array([0,1])
    feature, rangeFeature, minFeature, maxFeature = readFeature(featureFn, nV1, featureType)
    LR = feature[0,:]
    print(min(LR), max(LR))
    #iPref = np.mod(np.round(feature[1,:]*nOri), nOri)
    iPref = np.mod(feature[1,:] + 0.5, 1.0)*nOri # for ori
    print(f'iPref: {[np.min(iPref), np.max(iPref)]}')

    max_fr = fr.T.flatten()[iPref_cort + np.arange(nV1)*nOri]
    with open('max_fr' + output_suffix[:-1] + '.bin', 'wb') as f:
        max_fr.tofile(f)

    if plotSample:
        if sample is None:
            np.random.seed(seed)
            sample = np.random.randint(nV1, size = ns)
            if False:
                sample = np.zeros(9, dtype = int)

                pick = epick[nLGN_V1[epick] > np.mean(nLGN_V1[epick])]
                sample[0] = pick[np.argmin(max_fr[pick])]
                sample[1] = pick[np.argmax(max_fr[pick])]
                sample[2] = np.random.choice(pick, 1)[0]

                pick = epick[nLGN_V1[epick] == 0]
                sample[3] = pick[np.argmin(max_fr[pick])]
                sample[4] = pick[np.argmax(max_fr[pick])]
                sample[5] = np.random.choice(pick, 1)[0]

                pick = ipick[nLGN_V1[ipick] > np.mean(nLGN_V1[ipick])]
                sample[6] = pick[np.argmin(max_fr[pick])]
                sample[7] = pick[np.argmax(max_fr[pick])]
                sample[8] = np.random.choice(pick, 1)[0]

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
            fit_theta = np.fromfile(f, 'f8', nV1)
            noOpt = np.fromfile(f, 'u4')
            f.close()
            print('read fitted data from file')

            iPref_cort_for_ori = fit_pref/np.pi*nOri
            iPref_cort = np.round(iPref_cort_for_ori).astype(int)
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
            fit_theta = np.zeros(nV1)
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
            theta = np.arange(nOri)/nOri*np.pi
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
                fit_theta[i] = fit_pref[i]
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
                fit_theta.tofile(f)
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
        (np.mod(iPref+1,nOri)/nOri*np.pi).astype('f4').tofile(f)

        noOpt = np.array(noOpt, dtype = 'i4')

    figPref = plt.figure('pref-hist', figsize = (7,4))
    ax = figPref.add_subplot(231)
    ax.hist(iPref_cort[ipick], bins=nOri, alpha = 0.3, color = 'c', label = 'cort_I', density = True)
    ax.hist(iPref_cort[epick], bins=nOri, alpha = 0.3, color = 'm', label = 'cort_E', density = True)
    ax.hist(iPref_thal[iSpick], bins=nOri, alpha = 0.3, color = 'b', label = 'thal_I', density = True)
    ax.hist(iPref_thal[eSpick], bins=nOri, alpha = 0.3, color = 'r', label = 'thal_E', density = True)
    #ax.legend()

    ax = figPref.add_subplot(232)

    dPref = iPref_cort_for_ori[iSpick]-iPref_thal_for_ori[iSpick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ax.hist(dPref, bins=nOri, label = 'dPref_IS', alpha = 0.3, color = 'b', density = True)

    dPref = iPref_cort_for_ori[eSpick]-iPref_thal_for_ori[eSpick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ax.hist(dPref, bins=nOri, label = 'dPref_ES', alpha = 0.3, color = 'r', density = True)

    ax.legend()
    ax.set_xlabel('delta pref LGN vs Cort.')

    ax = figPref.add_subplot(234)

    dPrefIS = iPref_thal_for_ori[iSpick]-iPref[iSpick]
    pick = dPrefIS > nOri//2
    dPrefIS[pick] = nOri - dPrefIS[pick]
    pick = dPrefIS < -nOri//2
    dPrefIS[pick] = nOri + dPrefIS[pick]
    ax.hist(dPrefIS, bins=nOri, label = 'dPrefIS_IS', alpha = 0.3, color = 'b', density = True)

    dPrefES = iPref_thal_for_ori[eSpick]-iPref[eSpick]
    pick = dPrefES > nOri//2
    dPrefES[pick] = nOri - dPrefES[pick]
    pick = dPrefES < -nOri//2
    dPrefES[pick] = nOri + dPrefES[pick]
    ax.hist(dPrefES, bins=nOri, label = 'dPref_ES', alpha = 0.3, color = 'r', density = True)

    ax.set_xlabel('delta pref preset vs Thal.')

    spick = np.concatenate((eSpick[nLGN_V1[eSpick]>1],iSpick[nLGN_V1[iSpick]>1]))
    dPref = iPref_thal_for_ori[spick]-iPref[spick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    dprefMax_pick = spick[np.argpartition(-np.absolute(dPref), ndpref)[:ndpref]]

    dprefMin_pick = spick[np.argpartition(np.absolute(dPref), ndpref)[:ndpref]]

    #dprefMin_pick = spick[np.absolute(dPref) == np.min(np.absolute(dPref))]

    ax = figPref.add_subplot(235)

    dPref = iPref_cort_for_ori[ipick]-iPref[ipick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ax.hist(dPref, bins=nOri, label = 'dPref_I', alpha = 0.3, color = 'b', density = True)

    dPref = iPref_cort_for_ori[eCpick]-iPref[eCpick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ax.hist(dPref, bins=nOri, label = 'dPref_EC', alpha = 0.3, color = 'm', density = True)

    dPref = iPref_cort_for_ori[eSpick]-iPref[eSpick]
    pick = dPref > nOri//2
    dPref[pick] = nOri - dPref[pick]
    pick = dPref < -nOri//2
    dPref[pick] = nOri + dPref[pick]
    ax.hist(dPref, bins=nOri, label = 'dPref_ES', alpha = 0.3, color = 'r', density = True)

    
    ax.set_xlabel('delta pref preset vs Cort.')

    ax = figPref.add_subplot(233)
    
    pick = iSpick[LR[iSpick] < 0.5]
    dPrefIS_L = iPref_thal_for_ori[pick]-iPref[pick]
    pick = dPrefIS_L > nOri//2
    dPrefIS_L[pick] = nOri - dPrefIS_L[pick]
    pick = dPrefIS_L < -nOri//2
    dPrefIS_L[pick] = nOri + dPrefIS_L[pick]
    ax.hist(dPrefIS_L, bins=nOri, label = 'dPref_IS_L', alpha = 0.3, color = 'b')

    pick = eSpick[LR[eSpick] < 0.5]
    dPrefES_L = iPref_thal_for_ori[pick]-iPref[pick]
    pick = dPrefES_L > nOri//2
    dPrefES_L[pick] = nOri - dPrefES_L[pick]
    pick = dPrefES_L < -nOri//2
    dPrefES_L[pick] = nOri + dPrefES_L[pick]
    ax.hist(dPrefES_L, bins=nOri, label = 'dPref_ES_L', alpha = 0.3, color = 'r')

    
    ax = figPref.add_subplot(236)

    pick = iSpick[LR[iSpick] > 0.5]
    dPrefIS_R = iPref_thal_for_ori[pick]-iPref[pick]
    pick = dPrefIS_R > nOri//2
    dPrefIS_R[pick] = nOri - dPrefIS_R[pick]
    pick = dPrefIS_R < -nOri//2
    dPrefIS_R[pick] = nOri + dPrefIS_R[pick]
    ax.hist(dPrefIS_R, bins=nOri, label = 'dPref_IS_R', alpha = 0.3, color = 'b')

    pick = eSpick[LR[eSpick] > 0.5]
    dPrefES_R = iPref_thal_for_ori[pick]-iPref[pick]
    pick = dPrefES_R > nOri//2
    dPrefES_R[pick] = nOri - dPrefES_R[pick]
    pick = dPrefES_R < -nOri//2
    dPrefES_R[pick] = nOri + dPrefES_R[pick]
    ax.hist(dPrefES_R, bins=nOri, label = 'dPref_ES_R', alpha = 0.3, color = 'r')

    figPref.savefig(outputfdr + 'pref-hist' + output_suffix[:-1] + '.png', dpi = 150)

    fig = plt.figure('nLGN-pref', figsize = (6,4))
    dOpEdges = opEdges - 90 + 90/nOri
    maxLGN = np.max(nLGN_V1)
    ax = fig.add_subplot(221)
    target = nLGN_V1[eSpick]
    ytarget = dPrefES * 180/nOri
    HeatMap(target, ytarget, np.arange(maxLGN)+1, dOpEdges, ax, 'Reds', log_scale = pLog)
    ax = fig.add_subplot(222)
    target = nLGN_V1[iSpick]
    ytarget = dPrefIS * 180/nOri
    HeatMap(target, ytarget, np.arange(maxLGN)+1, dOpEdges, ax, 'Blues', log_scale = pLog)
    ax = fig.add_subplot(223)
    ax.hist(nLGN_V1[epick], np.arange(maxLGN+1))
    ax = fig.add_subplot(224)
    ax.hist(nLGN_V1[ipick], np.arange(maxLGN+1))
    fig.savefig(outputfdr + 'nLGN-pref' + output_suffix[:-1] + '.png', dpi = 150)

    markers = ['^r', 'vg', 'og', 'sr', '*k', 'dk']
    ms = 1.5

    print(f'prefMax iV1: {dprefMax_pick}')
    fig = plt.figure('thal_set-prefMax', dpi = 300)
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
                    ax.plot(iLGN_vpos[0,j], iLGN_vpos[1,j], markers[iLGN_type[j]], ms = iLGN_ms[j]/max_s*ms, mec = 'k', mew = ms/4)
                x0 = np.average(iLGN_vpos[0,:], weights = iLGN_ms)
                y0 = np.average(iLGN_vpos[1,:], weights = iLGN_ms)
            else:
                for j in range(len(markers)):
                    pick = iLGN_type == j
                    ax.plot(iLGN_vpos[0,pick], iLGN_vpos[1,pick], markers[j], ms = ms, mec = 'k', mew = ms/4)
                x0 = np.average(iLGN_vpos[0,:])
                y0 = np.average(iLGN_vpos[1,:])

            orient = iPref[iV1]/nOri*np.pi + np.pi/2
            orient_thal = iPref_thal_for_ori[iV1]/nOri*np.pi + np.pi/2
            orient_thal_2nd = iPref_thal_2nd_for_ori[iV1]/nOri*np.pi + np.pi/2
            orient_set = feature[1,iV1]*np.pi
            x = np.array([np.min(iLGN_vpos[0,:]), np.max(iLGN_vpos[0,:])])
            y = np.array([np.min(iLGN_vpos[1,:]), np.max(iLGN_vpos[1,:])])
            x_thal = x.copy()
            y_thal = y.copy()
            x_set = x.copy()
            y_set = y.copy()
            x_thal_2nd = x.copy()
            y_thal_2nd = y.copy()

            ax.set_xlim(left = x[1] - (x[1]-x[0])*1.1, right = x[0] + (x[1]-x[0])*1.1)
            ax.set_ylim(bottom = y[1] - (y[1]-y[0])*1.1, top = y[0] + (y[1]-y[0])*1.1)

            if np.diff(y)[0] > np.diff(x)[0]:
                x = (y-y0) / np.tan(orient) + x0
                x_thal = (y_thal-y0) / np.tan(orient_thal) + x0
                x_set = (y_set-y0) / np.tan(orient_set) + x0
                x_thal_2nd = (y_thal_2nd-y0) / np.tan(orient_thal_2nd) + x0
            else:
                y = np.tan(orient)*(x-x0) + y0
                y_thal = np.tan(orient_thal)*(x_thal-x0) + y0
                y_set = np.tan(orient_set)*(x_set-x0) + y0
                y_thal_2nd = np.tan(orient_thal_2nd)*(x_thal_2nd-x0) + y0

            ax.plot(x0, y0, '*b', ms = 0.15)
            ax.plot(x, y, '-k', lw = 0.3, label = 'rounded pre')
            ax.plot(x_set, y_set, ':k', lw = 0.3, label = 'preset')
            ax.plot(x_thal, y_thal, ':b', lw = 0.5, label = 'thal')
            ax.plot(x_thal_2nd, y_thal_2nd, ':c', lw = 0.5, label = 'thal2')
            ax.set_aspect('equal')
        if i == 0:
            ax.legend()

        ax.set_title(f'rounded pre:{orient*180/np.pi:.1f}, thal:{orient_thal*180/np.pi:.1f}, preset:{orient_set*180/np.pi:.1f}')
        i = i+1
    fig.savefig(outputfdr+'thal_set-prefMax'+conLGN_suffix + output_suffix[:-1] +'.png')

    fig = plt.figure('thal_set-prefMin', dpi = 300)
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
                    ax.plot(iLGN_vpos[0,j], iLGN_vpos[1,j], markers[iLGN_type[j]], ms = iLGN_ms[j]/max_s*ms, mec = 'k', mew = ms/4)
                x0 = np.average(iLGN_vpos[0,:], weights = iLGN_ms)
                y0 = np.average(iLGN_vpos[1,:], weights = iLGN_ms)
            else:
                for j in range(len(markers)):
                    pick = iLGN_type == j
                    ax.plot(iLGN_vpos[0,pick], iLGN_vpos[1,pick], markers[j], ms = ms, mec = 'k', mew = ms/4)
                x0 = np.average(iLGN_vpos[0,:])
                y0 = np.average(iLGN_vpos[1,:])

            orient = iPref[iV1]/nOri*np.pi + np.pi/2
            orient_thal = iPref_thal_for_ori[iV1]/nOri*np.pi + np.pi/2
            orient_thal_2nd = iPref_thal_2nd_for_ori[iV1]/nOri*np.pi + np.pi/2
            orient_set = feature[1,iV1]*np.pi
            x = np.array([np.min(iLGN_vpos[0,:]), np.max(iLGN_vpos[0,:])])
            y = np.array([np.min(iLGN_vpos[1,:]), np.max(iLGN_vpos[1,:])])
            x_thal = x.copy()
            y_thal = y.copy()
            x_set = x.copy()
            y_set = y.copy()
            x_thal_2nd = x.copy()
            y_thal_2nd = y.copy()

            ax.set_xlim(left = x[1] - (x[1]-x[0])*1.1, right = x[0] + (x[1]-x[0])*1.1)
            ax.set_ylim(bottom = y[1] - (y[1]-y[0])*1.1, top = y[0] + (y[1]-y[0])*1.1)

            if np.diff(y)[0] > np.diff(x)[0]:
                x = (y-y0) / np.tan(orient) + x0
                x_thal = (y_thal-y0) / np.tan(orient_thal) + x0
                x_set = (y_set-y0) / np.tan(orient_set) + x0
                x_thal_2nd = (y_thal_2nd-y0) / np.tan(orient_thal_2nd) + x0
            else:
                y = np.tan(orient)*(x-x0) + y0
                y_thal = np.tan(orient_thal)*(x_thal-x0) + y0
                y_set = np.tan(orient_set)*(x_set-x0) + y0
                y_thal_2nd = np.tan(orient_thal_2nd)*(x_thal_2nd-x0) + y0

            ax.plot(x0, y0, '*b', ms = 0.15)
            ax.plot(x, y, '-k', lw = 0.3, label = 'rounded pre')
            ax.plot(x_set, y_set, ':k', lw = 0.3, label = 'preset')
            ax.plot(x_thal, y_thal, ':b', lw = 0.5, label = 'thal')
            ax.plot(x_thal_2nd, y_thal_2nd, ':c', lw = 0.5, label = 'thal2')
            ax.set_aspect('equal')
        if i == 0:
            ax.legend()

        ax.set_title(f'rounded pre:{orient*180/np.pi:.1f}, thal:{orient_thal*180/np.pi:.1f}, preset:{orient_set*180/np.pi:.1f}')
        i = i+1
    fig.savefig(outputfdr+'thal_set-prefMin'+conLGN_suffix+ output_suffix[:-1]+'.png')

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


    if plotTC:

        fig = plt.figure('population-heatTC', figsize = (12,16))
        grid = gs.GridSpec(5, 5, figure = fig, hspace = 0.3, wspace = 0.3)
        # TC based on LGN response

        ax = fig.add_subplot(grid[0,0])
        ytarget = np.array([fr[aligned_thal[i,:],i] for i in eSpick[max_fr[eSpick] > fr_thres]]).flatten()
        target = np.tile(op, (np.sum(max_fr[eSpick] > fr_thres), 1)).flatten()
        assert(ytarget.size == target.size)
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.FR')
        ax.set_title('LGN-based simple.E')

        ax = fig.add_subplot(grid[1,0])
        ytarget = np.array([cFF[aligned_thal[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_thal[i,:],i] for i in eSpick[max_fr[eSpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cFF')

        ax = fig.add_subplot(grid[2,0])
        ytarget = np.array([cE[aligned_thal[i,:],:,i,0].sum(axis = 1) for i in eSpick[max_fr[eSpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cE')

        ax = fig.add_subplot(grid[3,0])
        ytarget = np.array([cI[aligned_thal[i,:],:,i,0].sum(axis = 1) for i in eSpick[max_fr[eSpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cI')

        ax = fig.add_subplot(grid[4,0])
        ytarget = np.array([depC[aligned_thal[i,:],i,0] for i in eSpick[max_fr[eSpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.depC')

        ax = fig.add_subplot(grid[0,1])
        ytarget = np.array([fr[aligned_thal[i,:],i] for i in ipick[max_fr[ipick] > fr_thres]]).flatten()
        target = np.tile(op, (np.sum(max_fr[ipick]>fr_thres), 1)).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.FR')
        ax.set_title('LGN-based I')

        ax = fig.add_subplot(grid[1,1])
        ytarget = np.array([cFF[aligned_thal[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_thal[i,:],i] for i in ipick[max_fr[ipick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cFF')

        ax = fig.add_subplot(grid[2,1])
        ytarget = np.array([cE[aligned_thal[i,:],:,i,0].sum(axis = 1) for i in ipick[max_fr[ipick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cE')

        ax = fig.add_subplot(grid[3,1])
        ytarget = np.array([cI[aligned_thal[i,:],:,i,0].sum(axis = 1) for i in ipick[max_fr[ipick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cI')

        ax = fig.add_subplot(grid[4,1])
        ytarget = np.array([depC[aligned_thal[i,:],i,0] for i in ipick[max_fr[ipick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.depC')

        # TC based on cortical response

        ax = fig.add_subplot(grid[0,2])
        ytarget = aligned_fr[eSpick[max_fr[eSpick] > fr_thres],:].flatten()
        target = np.tile(op, (np.sum(max_fr[eSpick] > fr_thres), 1)).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.FR')
        ax.set_title('cortical-based simple.E')

        ax = fig.add_subplot(grid[1,2])
        ytarget = np.array([cFF[aligned_cort[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_cort[i,:],i] for i in eSpick[max_fr[eSpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cFF')

        ax = fig.add_subplot(grid[2,2])
        ytarget = np.array([cE[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in eSpick[max_fr[eSpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cE')

        ax = fig.add_subplot(grid[3,2])
        ytarget = np.array([cI[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in eSpick[max_fr[eSpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cI')

        ax = fig.add_subplot(grid[4,2])
        ytarget = np.array([depC[aligned_cort[i,:],i,0] for i in eSpick[max_fr[eSpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.depC')

        ax = fig.add_subplot(grid[0,3])
        ytarget = aligned_fr[eCpick[max_fr[eCpick] > fr_thres],:].flatten()
        target = np.tile(op, (np.sum(max_fr[eCpick] > fr_thres), 1)).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.FR')
        ax.set_title('cortical-based complex.E')

        ax = fig.add_subplot(grid[1,3])
        ytarget = np.array([cFF[aligned_cort[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_cort[i,:],i] for i in eCpick[max_fr[eCpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cFF')

        ax = fig.add_subplot(grid[2,3])
        ytarget = np.array([cE[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in eCpick[max_fr[eCpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cE')

        ax = fig.add_subplot(grid[3,3])
        ytarget = np.array([cI[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in eCpick[max_fr[eCpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.cI')

        ax = fig.add_subplot(grid[4,3])
        ytarget = np.array([depC[aligned_cort[i,:],i,0] for i in eCpick[max_fr[eCpick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Reds', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('E.depC')

        ax = fig.add_subplot(grid[0,4])
        ytarget = aligned_fr[ipick[max_fr[ipick] > fr_thres],:].flatten()
        target = np.tile(op, (np.sum(max_fr[ipick] > fr_thres), 1)).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.FR')
        ax.set_title('cortical based I')

        ax = fig.add_subplot(grid[1,4])
        ytarget = np.array([cFF[aligned_cort[i,:],:,i,0].sum(axis = 1)*gFF_F1F0[aligned_cort[i,:],i] for i in ipick[max_fr[ipick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cFF')

        ax = fig.add_subplot(grid[2,4])
        ytarget = np.array([cE[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in ipick[max_fr[ipick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cE')

        ax = fig.add_subplot(grid[3,4])
        ytarget = np.array([cI[aligned_cort[i,:],:,i,0].sum(axis = 1) for i in ipick[max_fr[ipick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.cI')

        ax = fig.add_subplot(grid[4,4])
        ytarget = np.array([depC[aligned_cort[i,:],i,0] for i in ipick[max_fr[ipick] > fr_thres]]).flatten()
        HeatMap(target, ytarget, opEdges, heatBins, ax, 'Blues', log_scale = pLog)
        ax.set_xlabel('OP')
        ax.set_ylabel('I.depC')

        fig.savefig(outputfdr+'popTC'+output_suffix[:-1] + '.png', dpi = 900)

    if plotF1F0:

        fig = plt.figure('F1F0')

        nbins = 10
        F1F0_op = F1F0.T.flatten()[iPref_thal + np.arange(nV1)*nOri]
        ax = fig.add_subplot(221)
        ax.hist(F1F0_op[eSpick[max_fr[eSpick]>fr_thres]], bins = nbins, color = 'r', alpha = 0.3)
        ax.set_title('thal-based')

        ax = fig.add_subplot(223)
        ax.hist(F1F0_op[iSpick[max_fr[iSpick]>fr_thres]], bins = nbins, color = 'b', alpha = 0.3)

        F1F0_op = F1F0.T.flatten()[iPref_cort + np.arange(nV1)*nOri]
        ax = fig.add_subplot(222)
        ax.hist(F1F0_op[eSpick[max_fr[eSpick]>fr_thres]], bins = nbins, color = 'r', alpha = 0.3)
        ax.set_title('cort-based')
        ax2 = ax.twinx()
        ax2.hist(F1F0_op[eCpick[max_fr[eCpick]>fr_thres]], bins = nbins, color = 'm', alpha = 0.3)

        ax = fig.add_subplot(224)
        ax.hist(F1F0_op[iSpick[max_fr[iSpick]>fr_thres]], bins = nbins, color = 'b', alpha = 0.3)
        ax2 = ax.twinx()
        ax2.hist(F1F0_op[iCpick[max_fr[iCpick]>fr_thres]], bins = nbins, color = 'c', alpha = 0.3)
        fig.savefig(outputfdr+'F1F0'+output_suffix[:-1] + '.png', dpi = 900)

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
        fig.savefig(outputfdr+'F1F0-nLGN'+output_suffix[:-1] + '.png', dpi = 900)

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
        fig.savefig(outputfdr+'FR-nLGN'+output_suffix[:-1] + '.png', dpi = 900)

    if plotSample:
        theta = np.linspace(0, np.pi, 37)
        #sampleList = np.unique(np.hstack((sample,noOpt))).astype('i4')
        sampleList = sample
        for iV1 in sampleList:
        #for iV1 in np.array(noOpt):
            iblock = iV1//blockSize
            ithread = np.mod(iV1, blockSize)
            fig = plt.figure('sampleTC-{iblock}-{ithread}', figsize = (6,4))
            grid = gs.GridSpec(2, 5, figure = fig, hspace = 0.3, wspace = 0.3)
            ax = fig.add_subplot(grid[:,:2]) # fr, cE, cI, depC
            ax2 = ax.twinx()
            ax.plot(op, fr[iop,iV1], 'k')
            if fitTC:
                ax.plot(theta/np.pi*180, von_Mises(theta, fit_a[iV1], fit_b[iV1], fit_s[iV1], fit_theta[iV1]), ':k')
                ax.set_title(f'a={fit_a[iV1]:.2f}, b={fit_b[iV1]:.2f}, s={fit_s[iV1]:.2f}, {fit_pref[iV1]*180/np.pi:.2f}, {iV1 in noOpt}, {[iPref_cort[iV1], iPref_cort0[iV1]]}')

            ax.set_ylim(bottom = 0)
            ax2.plot(op, cFF[iop,:,iV1,0].sum(axis = 1)*gFF_F1F0[iop, iV1], 'g')
            ax2.plot(op, np.zeros(op.size), ':y')
            ax2.plot(op, cE[iop,:,iV1,0].sum(axis = 1), 'r')
            ax2.plot(op, cI[iop,:,iV1,0].sum(axis = 1), 'b')
            ax.set_xlabel('orientation')
            ax.set_ylabel('fr')
            ax2.set_ylabel('current')

            ax = fig.add_subplot(grid[:,3:]) # gE, gI, gapI
            if nLGN_V1[iV1] > 0:
                ax.plot(op, gFF[iop,:,iV1,0].sum(axis = 1)*gFF_F1F0[iop, iV1], 'g')
            ax.plot(op, gE[iop,:,iV1,0].sum(axis = 1), 'r')
            ax.plot(op, gI[iop,:,iV1,0].sum(axis = 1), 'b')
            ax.set_xlabel('orientation')
            ax.set_ylabel('conductance')
            if ithread < mI:
                iI = iblock*mI + ithread - mE
                ax.plot(op, cGap[iop, iI, 0], 'c')

            fig.savefig(outputfdr+ f'sampleTC-{iblock}-{ithread}#{nLGN_V1[iV1]}' +output_suffix[:-1]+ '.png', dpi = 300)
            plt.close(fig)

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
        fig.savefig(outputfdr+ f'TC-OSI' +output_suffix[:-1]+ '.png', dpi = 300)
        plt.close(fig)
            
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
                    outputfdr = sys.argv[4]
                    print(outputfdr)
                    if len(sys.argv) > 5:
                        nOri = int(sys.argv[5])
                        print(nOri)
                        if len(sys.argv) > 6:

                            if sys.argv[6] == 'True' or sys.argv[6] == '1':
                                fitTC = True
                                print('use TC fitted with von Mises function')
                            else:
                                fitTC = False
                                print('won\'t use fitted TC')
                            if len(sys.argv) > 7:
                                if sys.argv[7] == 'True' or sys.argv[7] == '1':
                                    fitDataReady = True
                                else:
                                    fitDataReady = False
                            else:
                                fitDataReady = False
                        else:
                            fitTC = False
                            fitDataReady = False
                    else:
                        raise Exception('nOri is not set') 
                else:
                    raise Exception('outputfdr, nOri are not set') 
            else:
                raise Exception('conV1_suffix, outputfdr, nOri are not set') 
        else:
            raise Exception('conLGN_V1_suffix, conV1_suffix, outputfdr, nOri are not set') 
    else:
        raise Exception('output_suffix, conLGN_V1_suffix, conV1_suffix, outputfdr, nOri are not set') 
    print(sys.argv)

    gatherTuningCurve(output_suffix, conLGN_suffix, conV1_suffix, outputfdr, nOri, fitTC, fitDataReady)
