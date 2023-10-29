import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
np.seterr(invalid = 'raise')

def read_input_frames(fn):
    with open(fn) as f:
        virtual_LGN = np.fromfile(f, 'i4', 1)[0]
        nFrame, height, width = np.fromfile(f, 'u4', 3)
        if virtual_LGN < 0:
            initL, initM, initS = np.fromfile(f, 'f4', 3)
            print(f'blank frame init values:{[initL, initM, initS]}')
            buffer_ecc, ecc = np.fromfile(f, 'f4', 2)
            neye = np.fromfile(f,'u4', 1)[0]
            frames = np.fromfile(f, 'f4', height*width*nFrame*3).reshape(nFrame, 3, height, width)
        else:
            if virtual_LGN == 0:
                nInputType = 2
            if virtual_LGN == 1:
                nInputType = 4
            if virtual_LGN == 2:
                nInputType = 6
            ecc = np.fromfile(f, 'f4', 1)[0]
            neye = np.fromfile(f,'u4', 1)[0]
            frames = np.fromfile(f, 'f4', height*width*nFrame*nInputType).reshape(nFrame, nInputType, height, width)

    if inputType < 0:
        return nFrame, width, height, initL, initM, initS, frames, buffer_ecc, ecc, neye
    else:
        return nFrame, width, height, frames, ecc, neye

def read_cfg(fn, rn = False):
    with open(fn, 'rb') as f:
        sizeofPrec = np.fromfile(f, 'u4', 1)[0]

        if sizeofPrec == 8:
            prec = 'f8'
        else:
            assert(sizeofPrec == 4)
            prec = 'f4'
        print(f'using {prec}')

        vL = np.fromfile(f, prec, 1)[0]
        vE = np.fromfile(f, prec, 1)[0]
        vI = np.fromfile(f, prec, 1)[0]
        nType = np.fromfile(f, 'u4', 1)[0]
        nTypeE = np.fromfile(f, 'u4', 1)[0]
        nTypeI = np.fromfile(f, 'u4', 1)[0]
        vR = np.fromfile(f, prec, nType)
        vThres = np.fromfile(f, prec, nType)
        gL = np.fromfile(f, prec, nType)
        vT = np.fromfile(f, prec, nType)
        nTypeHierarchy = np.fromfile(f, 'u4', 2)
        typeAcc = np.fromfile(f, 'u4', nType)
        sRatioLGN = np.fromfile(f, prec, nType)
        sRatioV1 = np.fromfile(f, prec, nType*nType)
        frRatioLGN = np.fromfile(f, prec, 1)[0] 
        convolRatio = np.fromfile(f, prec, 1)[0] 
        frameRate = np.fromfile(f,'u4',1)[0] 
        inputFn_len = np.fromfile(f,'u4',1)[0] 
        print(f'frameRate = {frameRate}')
        inputFn = np.fromfile(f,f'a{inputFn_len}', 1)[0].decode('UTF-8')
        print(inputFn)
        nLGN = np.fromfile(f,'u4',1)[0] 
        nV1 = np.fromfile(f,'u4',1)[0] 
        nt = np.fromfile(f,'u4',1)[0] 
        dt = np.fromfile(f,prec,1)[0] 
        normViewDistance, L_x0, L_y0, R_x0, R_y0 = np.fromfile(f, prec, 5)
        tonicDep = np.fromfile(f,prec,nType) 
        noisyDep = np.fromfile(f,prec,nType) 
        iVirtual_LGN = np.fromfile(f, 'i4', 1)[0]
        if iVirtual_LGN == 1:
            virtual_LGN = True
        else:
            virtual_LGN = False

        nE = typeAcc[nTypeHierarchy[0]-1]
        nI = typeAcc[-1] - nE
        print(typeAcc)
        print(f'nE = {nE}, nI = {nI}')
        print(f'vL = {vL}, vI = {vI}, vE = {vE}') 
        print(f'vR = {vR}, gL = {gL}, vT = {vT}')
        print(f'sRatioLGN = {sRatioLGN}, sRatioV1 = {sRatioV1}')
        print(f'nLGN = {nLGN}, nV1 = {nV1}, dt = {dt}, nt = {nt}')
    if rn:
        return prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, nLGN, nV1, nt, dt, normViewDistance, L_x0, L_y0, R_x0, R_y0, virtual_LGN, tonicDep, noisyDep
    else:
        return prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, virtual_LGN, tonicDep, noisyDep

def readLGN_V1_s0(fn, rnLGN_V1 = False, prec='f4'):
    with open(fn, 'rb') as f:
        nList = np.fromfile(f, 'u4', 1)[0]
        if rnLGN_V1:
            nLGN_V1 = np.empty(nList, dtype = int)
        LGN_V1_s = np.empty(nList, dtype = object)
        maxList = np.fromfile(f, 'u4', 1)[0]
        for i in range(nList):
            listSize = np.fromfile(f, 'u4', 1)[0]
            if rnLGN_V1:
                nLGN_V1[i] = listSize
            assert(listSize <= maxList)
            LGN_V1_s[i] = np.fromfile(f, prec, listSize)
    if rnLGN_V1:
        return LGN_V1_s, nLGN_V1
    else:
        return LGN_V1_s

def readLGN_V1_ID(fn, rLGN_V1_ID = True, rnLGN_V1 = True):
    if not rLGN_V1_ID and not rnLGN_V1:
        pass
    with open(fn, 'rb') as f:
        nList = np.fromfile(f, 'u4', 1)[0]
        if rLGN_V1_ID:
            LGN_V1_ID = np.empty(nList, dtype = object)
        if rnLGN_V1:
            nLGN_V1 = np.zeros(nList, dtype = int)
        for i in range(nList):
            listSize = np.fromfile(f, 'u4', 1)[0]
            if rnLGN_V1:
                nLGN_V1[i] = listSize
            ID_list = np.fromfile(f, 'u4', listSize)
            if rLGN_V1_ID:
                LGN_V1_ID[i] = ID_list
    if rnLGN_V1 and rLGN_V1_ID:
        return LGN_V1_ID, nLGN_V1
    else:
        if rnLGN_V1:
            return nLGN_V1
        else:
            return LGN_V1_ID
    
def readLGN_vpos(fn, return_polar_ecc = False, prec='f4'):
    with open(fn, 'rb') as f:
        nLGN_I = np.fromfile(f, 'u4', 1)[0]
        nLGN_C = np.fromfile(f, 'u4', 1)[0]
        nLGN = nLGN_I + nLGN_C
        max_ecc = np.fromfile(f, prec, 1)[0]
        vCoordSpan = np.fromfile(f, prec, 4)
        LGN_vpos = np.fromfile(f, prec, nLGN*2).reshape(2, nLGN)
        LGN_type = np.fromfile(f, 'u4', nLGN).reshape(nLGN)
        LGN_polar, LGN_ecc = np.fromfile(f, prec, nLGN*2).reshape(2, nLGN)
    if return_polar_ecc:
        return nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type, LGN_polar, LGN_ecc
    else:
        return nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type

def read_conStats(fn, prec='f4'):
    with open(fn, 'rb') as f:
        nType = np.fromfile(f, 'u4', 1)[0]
        networkSize = np.fromfile(f, 'u4', 1)[0]
        ExcRatio = np.fromfile(f, 'u4', networkSize)
        preN = np.fromfile(f, 'u4', nType*networkSize).reshape(nType, networkSize)
        preN_avail = np.fromfile(f, 'u4', nType*networkSize).reshape(nType, networkSize)
        preNS = np.fromfile(f, prec, nType*networkSize).reshape(nType, networkSize)
        nTypeI = np.fromfile(f, 'u4', 1)[0]
        mI = np.fromfile(f, 'u4', 1)[0]
        preGapN = np.fromfile(f, 'u4', nTypeI*mI).reshape(nTypeI, mI)
        preGapNS = np.fromfile(f, prec, nTypeI*mI).reshape(nTypeI, mI)
    return nType, ExcRatio, preN, preNS, preN_avail, nTypeI, mI, preGapN, preGapNS

def linear_diff(v0, vs):
    return v0-vs
def circular_diff(v0, vs, minv, maxv):
    v_diff = v0 - vs
    v_range = maxv - minv
    pick = v_diff > v_range/2
    v_diff[pick] = v_diff[pick] - v_range  
    pick = v_diff <- v_range/2
    v_diff[pick] = v_diff[pick] + v_range  
    return v_diff

def readFeature(fn, networkSize, featureType, prec='f4'):
    with open(fn, 'rb') as f:
        nFeature = np.fromfile(f, 'u4', count = 1)[0]
        rnFeature = featureType.size
        if rnFeature > nFeature:
            raise Exception('requested #feature smaller than provided')
        feature = np.fromfile(f,prec, count = networkSize*rnFeature).reshape(rnFeature, networkSize)
    maxFeature = np.max(feature, axis = -1)
    minFeature = np.min(feature, axis = -1)
    rangeFeature = maxFeature - minFeature
    for i in range(rnFeature):
        if featureType[i] == 1:
            rangeFeature[i] = rangeFeature[i]/2
    return feature, rangeFeature, minFeature, maxFeature

def readLGN_fr(fn, prec='f4'):
    with open(fn, 'rb') as f:
        nt = np.fromfile(f, 'u4', count = 1)[0]
        nLGN = np.fromfile(f, 'u4', count = 1)[0]
        LGN_fr = np.fromfile(f,prec, count = np.int64(nt)*nLGN).reshape(nt, nLGN)
    return LGN_fr

def readLGN_sp(fn, prec='f4'):
    with open(fn, 'rb') as f:
        dt = np.fromfile(f, prec, count = 1)[0]
        nt = np.fromfile(f, 'u4', count = 1)[0]
        nLGN = np.fromfile(f, 'u4', count = 1)[0]
        LGN_spScatter = np.empty(nLGN, dtype = object) 
        print(nt, dt) 
        for i in range(nLGN):
            LGN_spScatter[i] = []
        for it in range(nt):
            tsp0 = np.fromfile(f, prec, count = nLGN)
            assert(np.sum(tsp0>0) == np.sum(tsp0>=1))
            tsps = tsp0[tsp0 > 0]
            if tsps.size > 0:
                idxFired = np.nonzero(tsp0>=1)[0]
                k = 0
                for j in idxFired:
                    nsp = int(np.floor(tsps[k]))
                    tsp = tsps[k] - nsp
                    if nsp > 1:
                        if 1-tsp > 0.5:
                            dtsp = tsp/nsp
                        else:
                            dtsp = (1-tsp)/nsp
                        tstart = tsp - (nsp/2)*dtsp
                        for isp in range(nsp):
                            LGN_spScatter[j].append((it + tstart+isp*dtsp)*dt)
                    else:
                        LGN_spScatter[j].append((it + tsp)*dt)
                    k = k + 1
                #if 24277 in idxFired:
                #    print(f'24277 fired at {LGN_spScatter[24277][-1]}')
        for i in range(nLGN):
            LGN_spScatter[i] = np.asarray(LGN_spScatter[i]).astype(prec)
    return LGN_spScatter 

def movingAvg2D(data, m):
    avg_data = np.empty(data.shape)
    n = data.shape[-1]
    if np.mod(m,2) == 0:
        m = m + 1
    s = (m-1)//2
    avg_data[:,:s] = np.stack([np.mean(data[:,:i+s], axis = -1) for i in range(1,s+1)], axis = 1)
    avg_data[:,-s:] = np.stack([np.mean(data[:,-2*s+i:], axis = -1) for i in range(s)], axis = 1)
    if n >= m:
        avg_data[:,s:-s] = np.stack([np.mean(data[:,i-s:i+s+1], axis = -1) for i in range(s,n-s)], axis = 1)
    return avg_data

def readFrCondV(fn, outputFn, spFn, step0 = 0, step1 = 10000, nstep = 1000, nsmooth = 5, prec='f4'):
    with open(fn, 'rb') as f:
        dt = np.fromfile(f, prec, 1)[0] 
        nt = np.fromfile(f, 'u4', 1)[0] 
        nV1 = np.fromfile(f, 'u4', 1)[0] 
        haveH = np.fromfile(f, 'u4', 1)[0] 
        ngFF = np.fromfile(f, 'u4', 1)[0] 
        ngE = np.fromfile(f, 'u4', 1)[0] 
        ngI = np.fromfile(f, 'u4', 1)[0] 

        v = np.empty((nV1, nstep), dtype = prec)
        gE = np.empty((ngE, nV1, nstep), dtype = prec)
        gI = np.empty((ngI, nV1, nstep), dtype = prec)
        gFF = np.empty((ngFF, nV1, nstep), dtype = prec)

        if step1 >= nt:
            step1 = nt 
        tstep = (step1-step0 + nstep - 1)//nstep
        nstep = (step1-step0)//tstep
        interval = tstep - 1
        print(f'plot from {step0}*dt to {step1}*dt with {nstep} data points, stepsize = {tstep}*dt')

        f.seek((2+(ngE + ngI + ngFF)*(1+haveH))*nV1*4*step0, 1)
        for i in range(nstep):
            f.seek(nV1*4, 1)
            v[:,i] = np.fromfile(f, prec, ngFF*nV1).reshape(ngFF,nV1)
            gFF[:,:,i] = np.fromfile(f, prec, ngFF*nV1).reshape(ngFF,nV1)
            if haveH:
                f.seek(ngFF*nV1*4, 1)

            gE[:,:,i] = np.fromfile(f, prec, ngE*nV1).reshape(ngE,nV1)
            gI[:,:,i] = np.fromfile(f, prec, ngI*nV1).reshape(ngI,nV1)
            if haveH:
                f.seek((ngE+ngI)*nV1*4, 1)
            f.seek((2+(ngE + ngI + ngFF)*(1+haveH))*nV1*4*interval, 1)

    print(f'membrane potential and conductance read from {fn}')
    spScatter = np.load(spFn, allow_pickle=True)
    fr0 = np.zeros(nV1, nstep)
    edges = np.linspace(step0, step1, nstep+1)
    assert(edges[1] - edges[0] == tstep)
    edges = edges*dt
    for i in range(nV1):
        tsp = spScatter[i]
        fr0[i,:], _ = np.histogram(tsp[np.logical_and(tsp>=step0*dt, tsp<step1*dt)], edges)/(tstep*dt)
    fr = movingAvg2D(fr0, nsmooth)
    return fr, gFF, gE, gI, v, edges[:-1]

def readSpike(rawDataFn, spFn, prec, sizeofPrec, vThres):
    max_vThres = np.max(vThres)
    assert(max_vThres < 1)
    with open(rawDataFn, 'rb') as f:
        dt = np.fromfile(f, prec, 1)[0] 
        nt = np.fromfile(f, 'u4', 1)[0] 
        nV1 = np.fromfile(f, 'u4', 1)[0] 
        iModel = np.fromfile(f, 'i4', 1)[0] 
        mI = np.fromfile(f, 'u4', 1)[0] 
        haveH = np.fromfile(f, 'u4', 1)[0] 
        ngFF = np.fromfile(f, 'u4', 1)[0] 
        ngE = np.fromfile(f, 'u4', 1)[0] 
        ngI = np.fromfile(f, 'u4', 1)[0] 
        print('reading V1 spike...')
        negativeSpike = False
        multi_spike = 0
        spScatter = np.empty(nV1, dtype = object)
        spCount = np.zeros(nV1, dtype=int)
        for i in range(nV1):
            spScatter[i] = np.empty(nt, dtype = prec)
        for it in range(nt):
            data = np.fromfile(f, prec, nV1)
            pick = np.logical_and(data>max_vThres, data < 1)
            if np.sum(pick) > 0:
                print(f'{np.arange(nV1)[pick]} has negative spikes at {data[pick]} + {it*dt}')
                negativeSpike = True
            tsps = data[data >= 1]
            pick = data < 1
            assert((data[pick] <= max_vThres).all())
            if tsps.size > 0:
                idxFired = np.nonzero(data >= 1)[0]
                k = 0
                for j in idxFired:
                    nsp = int(np.floor(tsps[k]))
                    tsp = tsps[k] - nsp
                    if nsp > 1:
                        #raise Exception(f'{nsp} spikes from {j} at time step {it}, sInfo = {tsps[k]}!')
                        multi_spike = multi_spike + nsp
                        if 1-tsp > 0.5:
                            dtsp = tsp/nsp
                        else:
                            dtsp = (1-tsp)/nsp
                        tstart = tsp - (nsp//2)*dtsp
                        for isp in range(nsp):
                            spScatter[j][spCount[j] + isp] = ((it + tstart+isp*dtsp)*dt)
                        spCount[j] = spCount[j] + nsp
                    else:
                        spScatter[j][spCount[j]] = (it + tsp)*dt
                        spCount[j] = spCount[j] + 1
                    k = k + 1
            if iModel == 0:
                f.seek(((2+(ngE + ngI + ngFF)*(1+haveH))*nV1 + mI)*sizeofPrec, 1)
            if iModel == 1:
                f.seek(((3+(ngE + ngI + ngFF)*(1+haveH))*nV1 + mI)*sizeofPrec, 1)
        if negativeSpike:
            #print('negative spikes exist')
            raise Exception('negative spikes exist')

        for i in range(nV1):
            spScatter[i] = spScatter[i][:spCount[i]]
        np.savez(spFn, spName = 'spScatter', spScatter = spScatter)
        print(f'number of multiple spikes in one dt: {multi_spike}')
        return spScatter

def HeatMap(d1, d2, range1, range2, ax, cm, log_scale = False, intPick = False, tickPick1 = None, tickPick2 = None, mostDropZero = True):
    if hasattr(range1, "__len__") and hasattr(range2, "__len__"):
        if len(range1) == 2 and len(range2) == 2:
            h, edge1, edge2 = np.histogram2d(d1, d2, range = [range1, range2])
        else:
            h, edge1, edge2 = np.histogram2d(d1, d2, bins = [range1, range2])
    else:
        if hasattr(range1, "__len__") and not hasattr(range2, "__len__"):
            if len(range1) == 2:
                h, edge1, edge2 = np.histogram2d(d1, d2, bins = [np.linspace(range1[0], range1[1], range2+1), range2])
            else:
                h, edge1, edge2 = np.histogram2d(d1, d2, bins = [range1, range2])
        elif not hasattr(range1, "__len__") and hasattr(range2, "__len__"):
            if len(range2) == 2:
                h, edge1, edge2 = np.histogram2d(d1, d2, bins = [range1, np.linspace(range2[0], range2[1], range1+1)])
            else:
                h, edge1, edge2 = np.histogram2d(d1, d2, bins = [range1, range2])
        else:
            h, edge1, edge2 = np.histogram2d(d1, d2, bins = [range1, range2])
    if log_scale:
        data = np.log(h.T + 1)
    else:
        data = h.T

    if mostDropZero:
        d2_0 = d2[d2>0]
        d1_0 = d1[d2>0]
        h0, _, _ = np.histogram2d(d1_0, d2_0, bins = [edge1, edge2])
        if log_scale:
            data0 = np.log(h0.T + 1)
        else:
            data0 = h0.T
    else:
        data0 = data
        

    tc = np.zeros(edge1.size-1)
    tm = np.zeros(edge1.size-1)
    tmost = np.zeros(edge1.size-1)
    nmost = max(int(edge2.size * 0.1),1)
    for i in range(edge1.size-1):
        binned = np.logical_and(d1 <= edge1[i+1], d1 > edge1[i])
        if not binned.any():
            tc[i] = np.nan
            tm[i] = np.nan
            tmost[i] = np.nan
        else:
            tc[i] = np.mean(d2[binned])
            tm[i] = np.median(d2[binned])
            imost_list = np.argpartition(-data0[:,i], nmost)[:nmost]
            if sum(data0[imost_list,i]) > 0:
                tmost[i] = np.average(imost_list, weights = data0[imost_list,i])
            else:
                tmost[i] = np.nan

    #image = ax.imshow(data, vmin = vmin, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap(cm))
    image = ax.imshow(data, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap(cm), alpha = 0.7)

    nTick = 5
    if not hasattr(tickPick1, "__len__") and tickPick1 is not None:
       nTick = tickPick1 

    if intPick:
        if tickPick1 is None or not hasattr(tickPick1, "__len__"):
            tickPick1 = np.arange(0,edge1.size,(edge1.size-1)//(nTick-1))

        ax.set_xticks(tickPick1)
        if np.abs(edge1[-1]) > 0.1:
            ax.set_xticklabels([f'{label:.1f}' for label in edge1[tickPick1]])
        else:
            ax.set_xticklabels([f'{label:.1e}' for label in edge1[tickPick1]])
    else:
        if tickPick1 is None or not hasattr(tickPick1, "__len__"):
            tickPick1 = np.linspace(0,1,nTick)

        ax.set_xticks(np.array(tickPick1) * (edge1.size-1.0) - 0.5)
        min_value = edge1[0]
        max_value = edge1[-1]
        if np.abs(edge1[-1]) > 0.1:
            ax.set_xticklabels([f'{min_value + label * (max_value-min_value):.2f}' for label in tickPick1])
        else:
            ax.set_xticklabels([f'{min_value + label * (max_value-min_value):.2e}' for label in tickPick1])

    nTick = 5
    if not hasattr(tickPick2, "__len__") and tickPick2 is not None:
       nTick = tickPick2

    if intPick:
        if tickPick2 is None:
            tickPick2 = np.arange(0,edge2.size,(edge2.size-1)//(nTick-1))

        ax.set_yticks(tickPick2)
        if np.abs(edge2[-1]) > 0.1:
            ax.set_yticklabels([f'{label:.1f}' for label in edge2[tickPick2]])
        else:
            ax.set_yticklabels([f'{label:.1e}' for label in edge2[tickPick2]])
    else:
        if tickPick2 is None or not hasattr(tickPick2, "__len__"):
            tickPick2 = np.linspace(0,1,nTick)

        ax.set_yticks(np.array(tickPick2) * (edge2.size-1.0) - 0.5)
        min_value = edge2[0]
        max_value = edge2[-1]
        if np.abs(edge2[-1]) > 0.1:
            ax.set_yticklabels([f'{min_value + label * (max_value-min_value):.2f}' for label in tickPick2])
        else:
            ax.set_yticklabels([f'{min_value + label * (max_value-min_value):.2e}' for label in tickPick2])
    #if cm == 'Reds':
    #    color = 'm'
    #elif cm == 'Blues':
    #    color = 'c'
    #else: 
    #    color = 'k'
    color = 'k'

    ax.plot(np.arange(edge1.size - 1), (tc - edge2[0])/(edge2[-1]-edge2[0])*(edge2.size - 1)-0.5, ':', c=color, lw = 1.0, ms = 1.5) # mean
    ax.plot(np.arange(edge1.size - 1), (tm - edge2[0])/(edge2[-1]-edge2[0])*(edge2.size - 1)-0.5, '*--', c=color, lw = 1.0, ms = 1.5) # median
    ax.plot(np.arange(edge1.size - 1), tmost, '-', c=color, lw = 3.0, alpha = 0.5) # mode
    #ax.set_aspect()
    return image

def TuningCurves(data, bins, percentile, ax, color, tick, ticklabel):
    # data (presyn at columns)
    fig = plt.figure()
    tmpAx = fig.add_subplot(111)
    binned, edges, _ = tmpAx.hist(data, bins = bins)
    binned = np.array(binned)
    plt.close(fig)
    mid = np.mean(binned, axis = 0)
    lower = np.percentile(binned, 50 - percentile, axis = 0)
    upper = np.percentile(binned, 50 + percentile, axis = 0)
    x = (edges[:-1] + edges[1:])/2
    ax.plot(x, mid, c = clr.hsv_to_rgb(color))
    lcolor = color
    lcolor[1] *= 0.6
    ax.fill_between(x, lower, upper, color = clr.hsv_to_rgb(lcolor), ls = '-', alpha = 0.6)

    bottom = np.percentile(binned, 0, axis = 0)
    top = np.percentile(binned, 100, axis = 0)
    ax.fill_between(x, bottom, top, color = clr.hsv_to_rgb(lcolor), ls = ':', alpha = 0.3)

    ax.set_xticks(tick)
    ax.set_xticklabels(ticklabel)
    ax.set_ylim(bottom = 0)
