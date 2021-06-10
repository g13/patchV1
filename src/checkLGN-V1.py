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
from global_vars import LGN_vposFn, featureFn, V1_allposFn, V1_vposFn
np.seterr(invalid = 'raise')
from getReceptiveField import acuity


disLGN = 1.2
conLGN_suffix = '_' + sys.argv[1]
outputfdr = sys.argv[2] + '/'
LGN_V1_sFn = "LGN_V1_sList" + conLGN_suffix + ".bin"
LGN_V1_idFn = "LGN_V1_idList" + conLGN_suffix + ".bin"
V1_RFpropFn = "V1_RFprop" + conLGN_suffix + ".bin"

LGN_V1_s = readLGN_V1_s0(LGN_V1_sFn)
LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)
nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type = readLGN_vpos(LGN_vposFn)

nV1 = nLGN_V1.size

np.random.seed(1324)
ns = 50
sample = np.random.randint(nV1, size = ns)
#sample = np.argpartition(-nLGN_V1, ns)[:ns]
#sample[-1] = 14*1024+153
#sample = np.array([49663, 21036, 49532, 49166, 50083, 49851, 21478, 49497, 49705, 49567])
#ns = sample.size

featureType = np.array([0,1])
feature, rangeFeature, minFeature, maxFeature = readFeature(featureFn, nV1, featureType)
LR = feature[0,:]
OP = np.mod(feature[1,:] + 0.5, 1.0)*np.pi

def ellipse(cx, cy, a, baRatio, orient, n = 50):
    b = a*baRatio
    #print(f'major:{b}, minor:{a}')
    e = np.sqrt(1-1/baRatio/baRatio)
    theta = np.linspace(0, 2*np.pi, n)
    phi = orient + theta
    r = a/np.sqrt(1-np.power(e*np.cos(theta),2))
    x = cx + r*np.cos(phi)
    y = cy + r*np.sin(phi)
    return x, y

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

with open(V1_allposFn, 'r') as f:
    nblock = np.fromfile(f, 'u4', count=1)[0]
    blockSize = np.fromfile(f, 'u4', count=1)[0]
    networkSize = nblock*blockSize
    dataDim = np.fromfile(f, 'u4', count=1)[0]
    print(f'dataDim = {dataDim}')
    print([nblock,blockSize,networkSize,dataDim])
    coord_span = np.fromfile(f, 'f8', count=4)
    V1_x0 = coord_span[0]
    V1_xspan = coord_span[1]
    V1_y0 = coord_span[2]
    V1_yspan = coord_span[3]
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


fig = plt.figure(f'nLGN-V1', dpi = 300)
ax = fig.add_subplot(111)
ax.hist(nLGN_V1, bins = np.max(nLGN_V1)+1)
fig.savefig(outputfdr + f'nLGN-V1_hist.png')

for i in range(ns):
    iV1 = sample[i]
    if nLGN_V1[iV1] > 1:
        fig = plt.figure(f'LGN-V1_sample-{iV1}', dpi = 150)
        ax = fig.add_subplot(111)
        if LR[iV1] > 0:
            all_pos = LGN_vpos[:,nLGN_I:nLGN]
            all_type = LGN_type[nLGN_I:nLGN]
        else:
            all_pos = LGN_vpos[:,:nLGN_I]
            all_type = LGN_type[:nLGN_I]
        
        ms = 2.0
        ax.plot(vx[iV1], vy[iV1], '*k', ms = ms)
        ax.plot(cx[iV1], cy[iV1], 'sk', ms = ms)

        markers = ('^r', 'vg', '*g', 'dr', '^k', 'vb')

        for j in range(len(markers)):
            pick = all_type == j
            #ax.plot(all_pos[0,pick], all_pos[1,pick], markers[j], ms = ms, mew = 0.0, alpha = 0.5)
            ax.plot(all_pos[0,pick], all_pos[1,pick], markers[j], ms = 0.25*ms, alpha = 0.5)

        if nLGN_V1[iV1] > 0:
            iLGN_vpos = LGN_vpos[:, LGN_V1_ID[iV1]]
            iLGN_type = LGN_type[LGN_V1_ID[iV1]]

            iLGN_v1_s = LGN_V1_s[iV1]
            max_s = np.max(iLGN_v1_s)
            min_s = np.max([np.min(iLGN_v1_s), ms*0.25])

            rLGN = acuity(V1_ecc[iV1])
            for j in range(nLGN_V1[iV1]):
                jx0, jy0 = ellipse(iLGN_vpos[0,j], iLGN_vpos[1,j], rLGN, 1.0, 0)
                jx1, jy1 = ellipse(iLGN_vpos[0,j], iLGN_vpos[1,j], rLGN*disLGN, 1.0, 0)
                jtype = iLGN_type[j]
                if jtype == 0 or jtype == 3:
                    ax.plot(jx0, jy0, ':r', lw = 0.1)
                    ax.plot(jx1, jy1, '-r', lw = 0.1)
                else:
                    ax.plot(jx0, jy0, ':g', lw = 0.1)
                    ax.plot(jx1, jy1, '-g', lw = 0.1)

                jtype = iLGN_type[j]
                ax.plot(iLGN_vpos[0,j], iLGN_vpos[1,j], markers[jtype], ms = iLGN_v1_s[j]/max_s*ms, mew = ms*1.0)

        orient = OP[iV1] + np.pi/2
        bx, by = ellipse(vx[iV1], vy[iV1], a[iV1], baRatio[iV1], orient)
        ax.plot(bx, by, '-b', lw = 0.1)

        if nLGN_V1[iV1] > 0:
            x = np.array([np.min([np.min(iLGN_vpos[0,:]), vx[iV1], np.min(bx)]), np.max([np.max(iLGN_vpos[0,:]), vx[iV1], np.max(bx)])])
            y = np.array([np.min([np.min(iLGN_vpos[1,:]), vy[iV1], np.min(by)]), np.max([np.max(iLGN_vpos[1,:]), vy[iV1], np.max(by)])])
        else:
            x = np.array([np.min([vx[iV1], np.min(bx)]), np.max([vx[iV1], np.max(bx)])])
            y = np.array([np.min([vy[iV1], np.min(by)]), np.max([vy[iV1], np.max(by)])])
        ax.set_xlim(left = x[0] - (x[1]-x[0])*0.1, right = x[1] + (x[1]-x[0])*0.1)
        ax.set_ylim(bottom = y[0] - (y[1]-y[0])*0.1, top = y[1] + (y[1]-y[0])*0.1)

        if np.tan(orient) > 1:
            x = (y-cy[iV1]) / np.tan(orient) + cx[iV1]
        else:
            y = np.tan(orient)*(x-cx[iV1]) + cy[iV1]
        ax.plot(x, y, ':k', lw = 0.1)
        iblock = iV1//blockSize
        ithread = np.mod(iV1,blockSize)
                    
        ax.set_aspect('equal')
        fig.savefig(outputfdr + f'LGN-V1_sample-{iblock}-{ithread}#{nLGN_V1[iV1]}.png')
        plt.close(fig)
        plt.close(fig)
