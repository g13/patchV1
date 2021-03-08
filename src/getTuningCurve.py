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

def gatherTuningCurve(output_suffix, conLGN_suffix, nOri, plotSample, plotSample = True, plotThalTC = True, plotCortTC = True):

    parameterFn = "patchV1_cfg_" +output_suffix + ".bin"
    LGN_V1_idFn = "LGN_V1_idList" + conLGN_suffix + ".bin"
    _, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)
    nV1_0 = nLGN_V1.size
    sample = None

    nOri = 8
    mean_data_files = ["mean_data_" + output_suffix + str(iOri) + ".bin" for iOri in range(nOri)]
    for i in range(nOri):
        with open(mean_data_files[i], 'rb'):
            if i == 0:
                nV1 = np.fromfile(f, count=1)[0]
                if nV1 != nV1_0:
                    raise Exception("inconsistent cfg file vs. mean data file")
                ngFF = np.fromfile(f, count=1)[0]
                ngE = np.fromfile(f, count=1)[0]
                ngI = np.fromfile(f, count=1)[0]
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
                F1F0    = np.empty((nOri,nV1))
                gFF_F1F0= np.empty((nOri,nV1))
                gE_F1F0 = np.empty((nOri,nV1))
                gI_F1F0 = np.empty((nOri,nV1))
            else:
                _nV1 = np.fromfile(f, count=1)[0]
                _ngFF = np.fromfile(f, count=1)[0]
                _ngE = np.fromfile(f, count=1)[0]
                _ngI = np.fromfile(f, count=1)[0]
                if _nV1 != nV1 or _ngFF != ngFF or _ngE != ngE or _ngI != ngI:
                    raise Exception("mean data files are not consistent")
    
            fr[i,:] = np.fromfile(f, count = nV1)
            gFF[i,:,:,:] = np.fromfile(f, count = ngFF*2*nV1).reshape((ngFF,nV1,2))
            gE[i,:,:,:] = np.fromfile(f, count = ngE*2*nV1).reshape((ngE,nV1,2))
            gI[i,:,:,:] = np.fromfile(f, count = ngI*2*nV1).reshape((ngI,nV1,2))
            w[i,:,:] = np.fromfile(f, count = 2*nV1).reshape((nV1,2))
            v[i,:,:] = np.fromfile(f, count = 2*nV1).reshape((nV1,2))
            depC[i,:,:] = np.fromfile(f, count = 2*nV1).reshape((nV1,2))
            cFF[i,:,:,:] = np.fromfile(f, count = ngFF*2*nV1).reshape((ngFF,nV1,2))
            cE[i,:,:,:] = np.fromfile(f, count = ngE*2*nV1).reshape((ngE,nV1,2))
            cI[i,:,:,:] = np.fromfile(f, count = ngI*2*nV1).reshape((ngI,nV1,2))
            F1F0[i,:] = np.fromfile(f, count = nV1)
            gFF_F1F0[i,:] = np.fromfile(f, count = nV1)
            gE_F1F0[i,:] = np.fromfile(f, count = nV1)
            gI_F1F0[i,:] = np.fromfile(f, count = nV1)
    
    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, mE, mI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, = read_cfg(parameterFn)
    blockSize = typeAcc[-1]
    nblock = nV1//blockSize
    epick = np.hstack([np.arange(mE) + iblock*blockSize for iblock in range(nblock)])
    ipick = np.hstack([np.arange(mI) + iblock*blockSize + mE for iblock in range(nblock)])

    opRange = np.arange(nOri)/nOri*np.pi
    iPref_cort = np.argmax(fr, axis = 0)
    iPref_thal = np.argmax(gFF_F1F0, axis = 0)
    op_thal = opRange[iPref_thal] 

    def alignTC(iPref, opRange):
        op = opRange[iPref]

    def plotTC(:)

    max_fr = fr.T.flatten()[iPref_cort + np.range(nV1)*nOri]
    if plotSample:
        if sample is None:
            sample = np.random.randint(nV1, size = ns)
            if True:
                sample = np.zeros(12, dtype = int)

                pick = epick[nLGN_V1[epick] > np.mean(nLGN_V1[epick])]
                opick = pick[dOP[pick] <= dOri]
                sample[0] = opick[np.argmin(fr[opick])]
                sample[1] = opick[np.argmax(fr[opick])]
                opick = pick[dOP[pick] >= (nOri/2-1)*dOri]
                sample[2] = opick[np.argmin(fr[opick])]
                sample[3] = opick[np.argmax(fr[opick])]

                pick = epick[nLGN_V1[epick] == 0]
                opick = pick[dOP[pick] <= dOri]
                sample[4] = opick[np.argmin(fr[opick])]
                sample[5] = opick[np.argmax(fr[opick])]
                opick = pick[dOP[pick] >= (nOri/2-1)*dOri]
                sample[6] = opick[np.argmin(fr[opick])]
                sample[7] = opick[np.argmax(fr[opick])]

                pick = ipick[nLGN_V1[ipick] > np.mean(nLGN_V1[ipick])]
                opick = pick[dOP[pick] <= dOri]
                sample[8] = opick[np.argmin(fr[opick])]
                sample[9] = opick[np.argmax(fr[opick])]
                opick = pick[dOP[pick] >= (nOri/2-1)*dOri]
                sample[10] = opick[np.argmin(fr[opick])]
                sample[11] = opick[np.argmax(fr[opick])]

    for 

