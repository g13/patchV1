import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def read_cfg(fn):
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
        sRatioV1 = np.fromfile(f, prec, 1)[0]
        frRatioLGN = np.fromfile(f, prec, 1)[0] 
        convolRatio = np.fromfile(f, prec, 1)[0] 
        nE = typeAcc[nTypeHierarchy[0]-1]
        nI = typeAcc[-1] - nE
        print(typeAcc)
        print(f'nE = {nE}, nI = {nI}')
        print(f'vL = {vL}, vI = {vI}, vE = {vE}') 
        print(f'vR = {vR}, gL = {gL}, vT = {vT}')
        print(f'sRatioLGN = {sRatioLGN}, sRatioV1 = {sRatioV1}')
    return prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI

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
    
def readLGN_vpos(fn, prec='f4'):
    with open(fn, 'rb') as f:
        nLGN_I = np.fromfile(f, 'u4', 1)[0]
        nLGN_C = np.fromfile(f, 'u4', 1)[0]
        nLGN = nLGN_I + nLGN_C
        max_ecc = np.fromfile(f, prec, 1)[0]
        vCoordSpan = np.fromfile(f, prec, 4)
        LGN_vpos = np.fromfile(f, prec, nLGN*2).reshape(2, nLGN)
        LGN_type = np.fromfile(f, 'u4', nLGN).reshape(nLGN)
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
        LGN_fr = np.fromfile(f,prec, count = nt*nLGN).reshape(nt, nLGN)
    return LGN_fr

def readLGN_sp(fn, prec='f4'):
    with open(fn, 'rb') as f:
        dt = np.fromfile(f, prec, count = 1)[0]
        nt = np.fromfile(f, 'u4', count = 1)[0]
        nLGN = np.fromfile(f, 'u4', count = 1)[0]
        LGN_sp = np.fromfile(f,prec, count = nt*nLGN).reshape(nt, nLGN)
        LGN_spScatter = np.empty(nLGN, dtype = object) 
        for i in range(nLGN):
            LGN_spScatter[i] = []
        for it in range(nt):
            tsp0 = LGN_sp[it,:]
            tsps = tsp0[tsp0 > 0]
            if tsps.size > 0:
                idxFired = np.nonzero(tsp0)[0]
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
                            LGN_spScatter[j].append((it + tstart+isp*dtsp)*dt)
                    else:
                        LGN_spScatter[j].append((it+tsp)*dt)
                    k = k + 1
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

def readSpike(fn, spFn, prec = 'f4'):
    with open(rawDataFn, 'rb') as f:
        dt = np.fromfile(f, prec, 1)[0] 
        nt = np.fromfile(f, 'u4', 1)[0] 
        nV1 = np.fromfile(f, 'u4', 1)[0] 
        haveH = np.fromfile(f, 'u4', 1)[0] 
        ngFF = np.fromfile(f, 'u4', 1)[0] 
        ngE = np.fromfile(f, 'u4', 1)[0] 
        ngI = np.fromfile(f, 'u4', 1)[0] 

        spScatter = np.empty(nV1, dtype = object)
        for i in range(nV1):
            spScatter[i] = []
        for it in range(nt):
            data = np.fromfile(f, prec, nV1)
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
        np.save(spFn, spScatter)
    print(f'spikes read from {fn} stored in {spFn}')

def HeatMap(d1, d2, range1, range2, ax, cm, log_scale = False, intPick = True, tickPick1 = None, tickPick2 = None):
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
    tc = np.zeros(edge1.size-1)
    for i in range(edge1.size-1):
        binned = np.logical_and(d1 <= edge1[i+1], d1 > edge1[i])
        if not binned.any():
            tc[i] = np.NAN
        else:
            tc[i] = np.mean(d2[binned])

    #image = ax.imshow(data, vmin = vmin, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap(cm))
    image = ax.imshow(data, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap(cm))

    nTick = 5
    if not hasattr(tickPick1, "__len__") and tickPick1 is not None:
       nTick = tickPick1 

    if intPick:
        if tickPick1 is None or not hasattr(tickPick1, "__len__"):
            tickPick1 = np.arange(0,edge1.size,(edge1.size-1)//(nTick-1))

        ax.set_xticks(tickPick1)
        if np.abs(edge1[-1]) > 0.1:
            ax.set_xticklabels([f'{label:.2f}' for label in edge1[tickPick1]])
        else:
            ax.set_xticklabels([f'{label:.2e}' for label in edge1[tickPick1]])
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
            ax.set_yticklabels([f'{label:.2f}' for label in edge2[tickPick2]])
        else:
            ax.set_yticklabels([f'{label:.2e}' for label in edge2[tickPick2]])
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

    ax.plot(np.arange(edge1.size - 1), (tc - edge2[0])/(edge2[-1]-edge2[0])*(edge2.size - 1)-0.5, '*-k', lw = 1.0, ms = 1.5)
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
