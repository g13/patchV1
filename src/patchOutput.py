import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.gridspec as gs

import sys
if len(sys.argv) == 1:
    suffix = ""
else:
    suffix = sys.argv[1]

if suffix:
    suffix = '_' + suffix
rawData_fn = 'rawData'+suffix + '.bin'
stimulus_fn = 'stimulus' + suffix + '.bin'
framedOutput_fn = 'outputFrame' + suffix + '.bin'
outputB4V1_fn = 'outputB4V1' + suffix + '.bin'
LGN_fr_fn = 'LGN_fr' + suffix + '.bin'
seed = 8746251

plotSampleTrace = False
#plotSampleTrace = True 
plot_gFFtsp = False
plotFrame = True 
#plotFrame = False 
if plotFrame:
    phyV1 = False
    visV1 = False
    visLGN = False
    nPixel = 0
    with open(framedOutput_fn, 'rb') as f:
        dt = np.fromfile(f, 'f4', count=1)[0]
        odt = np.fromfile(f, 'u4', count=1)[0]
        iFrameOutput = np.fromfile(f, 'u4', count=1)[0]
        print(dt, odt, iFrameOutput)
        if iFrameOutput == 1 or iFrameOutput == 3 or iFrameOutput == 5 or iFrameOutput == 7:
            phyV1 = True
            phySpec = np.fromfile(f, 'u4', count=2)
            phyWidth, phyHeight = phySpec[0], phySpec[1]
            nPixel_phyV1 = phyWidth * phyHeight
            nPixel = nPixel_phyV1
            print(f'phyV1: {phyWidth}x{phyHeight}')
        else:
            nPixel_phyV1 = 0
        if iFrameOutput == 2 or iFrameOutput == 3 or iFrameOutput == 6 or iFrameOutput == 7:
            visV1 = True
            visV1Spec = np.fromfile(f, 'u4', count=2)
            visV1Width, visV1Height = visV1Spec[0], visV1Spec[1]
            nPixel_visV1_hlf = visV1Width * visV1Height
            nPixel_visV1 = nPixel_visV1_hlf*2
            nPixel = nPixel + nPixel_visV1
            print(f'visV1: 2x{visV1Width}x{visV1Height}')
        else:
            nPixel_visV1 = 0
        if iFrameOutput == 4 or iFrameOutput == 5 or iFrameOutput == 6 or iFrameOutput == 7:
            visLGN = True
            visLGNspec = np.fromfile(f, 'u4', count=2)
            visLGNwidth, visLGNheight = visLGNspec[0], visLGNspec[1]
            nPixel_visLGN_hlf = visLGNwidth * visLGNheight
            nPixel_visLGN = nPixel_visLGN_hlf*2
            nPixel = nPixel + nPixel_visLGN
            print(f'visLGN: 2x{visLGNwidth}x{visLGNheight}')
        else:
            nPixel_visLGN = 0
        framedOutput = np.fromfile(f, 'f4')
    
    print(framedOutput.size)
    print(framedOutput.shape)
    ont = framedOutput.size//nPixel
    print(ont, nPixel)
    assert(framedOutput.size%nPixel == 0)
    
    framedOutput = framedOutput.reshape(ont,nPixel)
    ot = odt*dt
    if phyV1:
        phyV1frame = np.zeros((ont, phyHeight, phyWidth), dtype = 'f4')
    if visV1:
        visV1frame = np.zeros((ont, 2, visV1Height, visV1Width), dtype = 'f4')
    if visLGN:
        visLGNframe = np.zeros((ont, 2, visLGNheight, visLGNwidth), dtype = 'f4')
    for i in range(ont):
        if phyV1:
            phyV1frame[i,:,:] = framedOutput[i,:nPixel_phyV1].reshape(phyHeight,phyWidth)/ot
        if visV1:
            visV1frame[i,:,:,:] = framedOutput[i,nPixel_phyV1:(nPixel_phyV1+nPixel_visV1)].reshape(2, visV1Height, visV1Width)/ot
        if visLGN:
            visLGNframe[i,:,:,:] = framedOutput[i,(nPixel_phyV1+nPixel_visV1):nPixel].reshape(2, visLGNheight, visLGNwidth)/ot
    
    def meshgrid(x,y):
        X = np.tile(x,(len(y),1))
        Y = np.tile(y,(len(x),1)).T
        return X, Y
    
    for i in range(ont):
        if phyV1:
            fig = plt.figure(f'phyV1-{i}', dpi = 1024)
            ax = fig.add_subplot(111)
            #plt.Normalize, norm =
            image = ax.imshow(phyV1frame[i,::-1,:], aspect = 'equal', origin = 'lower')
            fig.colorbar(image, ax=ax)
            fig.savefig(f'phyV1-{i}.png')
            plt.close(fig)
        
        if visV1:
            fig = plt.figure(f'visV1-{i}', dpi = 1024)
            ax = fig.add_subplot(111)
            image = ax.imshow(np.hstack((visV1frame[i,0,::-1,:], visV1frame[i,1,::-1,:])), aspect = 'equal', origin = 'lower')
            fig.colorbar(image, ax=ax)
            fig.savefig(f'visV1-{i}.png')
            plt.close(fig)
        
        if visLGN:
            fig = plt.figure(f'visLGN-{i}', dpi = 1024)
            ax = fig.add_subplot(111)
            image = ax.imshow(np.hstack((visLGNframe[i,0,::-1,:], visLGNframe[i,1,::-1,:])), aspect = 'equal', origin = 'lower')
            fig.colorbar(image, ax=ax)
            fig.savefig(f'visLGN-{i}.png')
            plt.close(fig)

if plotSampleTrace:
    with open(rawData_fn, 'rb') as f:
        dt = np.fromfile(f, 'f4', count = 1)[0]
        nt = np.fromfile(f, 'u4', count = 1)[0]
        nV1 = np.fromfile(f, 'u4', count = 1)[0]
        print(f'{nt}, {nV1}')
        hWrite = np.fromfile(f, 'u4', count = 1)[0]
        ngTypeFF = np.fromfile(f, 'u4', count = 1)[0]
        ngTypeE = np.fromfile(f, 'u4', count = 1)[0]
        ngTypeI = np.fromfile(f, 'u4', count = 1)[0]
        print(f'{ngTypeE}, {ngTypeI}')
        spikeTrain = np.zeros((nt,nV1), dtype = 'f4')
        v = np.zeros((nt,nV1), dtype = 'f4')
        gFF = np.zeros((nt,ngTypeFF,nV1), dtype = 'f4')
        gE = np.zeros((nt,ngTypeE,nV1), dtype = 'f4')
        gI = np.zeros((nt,ngTypeI,nV1), dtype = 'f4')
        if hWrite:
            hFF = np.zeros((nt,ngTypeFF,nV1), dtype = 'f4')
            hE = np.zeros((nt,ngTypeE,nV1), dtype = 'f4')
            hI = np.zeros((nt,ngTypeI,nV1), dtype = 'f4')
        for i in range(nt):
            spikeTrain[i,:] = np.fromfile(f, 'f4', count = nV1) #sInfo: integer(nsp) + decimal(tsp, normalized)
            v[i,:] = np.fromfile(f, 'f4', count = nV1)
            for j in range(ngTypeFF):
                gFF[i,j,:] = np.fromfile(f, 'f4', count = nV1)
            if hWrite:
                for j in range(ngTypeFF):
                    hFF[i,j,:] = np.fromfile(f, 'f4', count = nV1)
            for j in range(ngTypeE):
                gE[i,j,:] = np.fromfile(f, 'f4', count = nV1)
            for j in range(ngTypeI):
                gI[i,j,:] = np.fromfile(f, 'f4', count = nV1)
            if hWrite:
                for j in range(ngTypeE):
                    hE[i,j,:] = np.fromfile(f, 'f4', count = nV1)
                for j in range(ngTypeI):
                    hI[i,j,:] = np.fromfile(f, 'f4', count = nV1)
        print('rawdata read,')
        nsp = spikeTrain.copy()
        nsp[nsp<0] = 0
        nsp[nsp>0] = np.ceil(nsp[nsp>0])
        avg_fr = np.sum(nsp.astype('u4'), axis=0)
        avg_v = np.mean(v, axis = 0)
        avg_gFF = np.mean(np.sum(gFF, axis = 1), axis = 0)
        avg_gE = np.mean(np.sum(gE, axis = 1), axis = 0)
        avg_gI = np.mean(np.sum(gI, axis = 1), axis = 0)
        print('averaged\n')

    if plot_gFFtsp:
        with open('LGN_V1_idList.bin', 'rb') as f: # takes long time
            nLGNperV1 = np.zeros(nV1, dtype = 'u4')
            LGN_V1_ID = np.empty(nV1, dtype = object)
            for i in range(nV1):
                nLGNperV1[i] = np.fromfile(f, 'u4', count = 1)[0]
                LGN_V1_ID[i] = np.fromfile(f, 'u4', count = nLGNperV1[i])
            print('LGN_V1 read,')
            
    with open(LGN_fr_fn, 'rb') as f:
        nt = np.fromfile(f, 'u4', count = 1)[0]
        nLGN = np.fromfile(f, 'u4', count = 1)[0]
        LGN_fr = np.fromfile(f, 'f4', count = nt*nLGN).reshape(nt,nLGN)
        print('LGN_fr read\n')

    grid = gs.GridSpec(2, 3, hspace = 0.3, wspace = 0.3)
    fig = plt.figure('hist', dpi=1024)
    ax = fig.add_subplot(grid[0])
    ax.hist(avg_fr/(nt*dt/1000))
    ax.set_title('fr')
    ax = fig.add_subplot(grid[1])
    ax.hist(avg_v)
    ax.set_title('v')
    ax = fig.add_subplot(grid[2])
    ax.hist(avg_gFF)
    ax.set_title('gFF')
    ax = fig.add_subplot(grid[3])
    ax.hist(avg_gE)
    ax.set_title('gE')
    ax = fig.add_subplot(grid[4])
    ax.hist(avg_gI)
    ax.set_title('gI')
    ax = fig.add_subplot(grid[5])
    ax.hist(np.sum(LGN_fr, axis=0)/nt)
    ax.set_title('LGN_fr')

    fig.savefig('histogram.png')
    
    fig = plt.figure('vgtsp', dpi = 1024)
    nsample = 4
    np.random.seed(seed)
    pick = np.random.randint(nV1, size = nsample)
    pick[0] = 0
    pick[1] = 1
    pick[2] = 768
    pick[3] = 2
    grid = gs.GridSpec(nsample, 1, hspace = 0.2)
    t = (np.arange(nt)+1)*dt
    for i in range(nsample):
        nid = pick[i]
        ax = fig.add_subplot(grid[i])
        if plot_gFFtsp:
            ax.set_title(f'nLGN: {nLGNperV1[nid]}')
        ax2 = ax.twinx()
        ax.plot(t, v[:,nid].T, '.-k', ms=0.1, lw = 0.1)
        ax2.plot(t, gFF[:,0,nid].T, '.-g', ms = 0.1, lw = 0.1)
        ax2.plot(t, gE[:,0,nid].T, '.-r', ms = 0.1, lw = 0.1)
        #ax2.plot(t, gE[:,1,nid].T, '.:r', ms = 0.1, lw = 0.1)
        ax2.plot(t, gI[:,0,nid].T, '.-b', ms = 0.1, lw = 0.1)
        ax2.set_ylim(0)
        for it in range(nt):
            tsp = spikeTrain[it,nid]
            if tsp > 0:
                size = np.ceil(tsp)/10
                ratio = tsp - np.floor(tsp)
                #print(v.T.shape, it, nid)
                if it > 0:
                    x = t[it-1] + ratio*dt
                    ax.plot(x, 1.0, '*k', ms = size)
                else:
                    ax.plot(dt*ratio, 1.0, '*k', ms = size)
            _, ymax = ax2.get_ylim()
            #if plot_gFFtsp:
            #    for j in range(nLGNperV1[nid]):
            #        iLGN = LGN_V1_ID[nid][j]
            #        tsp = LGN_sInfo[it,iLGN] - np.floor(LGN_sInfo[it,iLGN])
            #        if tsp > 0:
            #            size = np.ceil(tsp)/10
            #            ratio = tsp - np.floor(tsp)
            #            #print(v.T.shape, it, nid)
            #            if it > 0:
            #                x = t[it-1] + ratio*dt
            #                #ax2.plot(x, gFF[it-1,0,nid] + (gFF[it,0,nid]-gFF[it-1,0,nid])*ratio, '*g', ms = size)
            #                ax2.plot([x,x], [j/nLGNperV1[nid]*ymax, gFF[it-1,0,nid] + (gFF[it,0,nid]-gFF[it-1,0,nid])*ratio], '-g', lw = 0.1)
            #            else:
            #                #ax2.plot(dt*ratio, gFF[it,0,nid]*ratio, '*g', ms = size)
            #                ax2.plot([dt*ratio,dt*ratio], [j/nLGNperV1[nid]*ymax, gFF[it,0,nid]*ratio], '-g', lw = 0.1)

    fig.savefig(f'vgtsp-{i}.png')
