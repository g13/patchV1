import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.gridspec as gs
from img_proc import create_gif, crop_image
from readPatchOutput import read_cfg

import sys

plotSampleTrace = False
#plotSampleTrace = True 
plot_gFFtsp = False 
plotFrame = True 
plotPhyV1 = True 
plotVisV1 = True
plotVisLGN = True
#plotFrame = False 

seed = 8746251
nsample = 4

if len(sys.argv) < 9:
    raise Exception("need all 8 arguments: output_suffix, res_suffix, conLGN_suffix, conV1_suffix, res_fdr, setup_fdr, data_fdr, fig_fdr")
else:
    output_suffix = sys.argv[1]
    res_suffix = sys.argv[2]
    conLGN_suffix = sys.argv[3]
    conV1_suffix = sys.argv[4]
    res_fdr = sys.argv[5]
    setup_fdr = sys.argv[6]
    data_fdr = sys.argv[7]
    fig_fdr = sys.argv[8]

res_fdr = res_fdr + "/"
setup_fdr = setup_fdr + "/"
data_fdr = data_fdr + "/"
fig_fdr = fig_fdr + "/"

output_suffix = '-' + output_suffix
res_suffix = '-' + res_suffix
conLGN_suffix = '-' + conLGN_suffix
conV1_suffix = '-' + conV1_suffix

LGN_V1_idFn = setup_fdr + "LGN_V1_idList" + conLGN_suffix + ".bin"
rawData_fn = data_fdr + 'rawData' + output_suffix + '.bin'
framedOutput_fn =  data_fdr + 'outputFrame' + output_suffix + '.bin'
outputB4V1_fn = data_fdr + 'outputB4V1' + output_suffix + '.bin'
LGN_fr_fn = data_fdr + 'LGN_fr' + output_suffix + '.bin' 
parameterFn = data_fdr + "patchV1_cfg" + output_suffix + '.bin'

prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, _virtual_LGN, tonicDep, noisyDep = read_cfg(parameterFn)

if plotFrame:
    phyV1 = False
    visV1 = False
    visLGN = False
    nPixel = 0
    with open(framedOutput_fn, 'rb') as f:
        dt = np.fromfile(f, 'f4', count=1)[0]
        ot = np.fromfile(f, 'u4', count=1)[0]
        iFrameOutput = np.fromfile(f, 'u4', count=1)[0]
        print(dt, ot, iFrameOutput)
        if iFrameOutput == 1 or iFrameOutput == 3 or iFrameOutput == 5 or iFrameOutput == 7:
            phyV1 = True
            phyWidth, phyHeight = np.fromfile(f, 'u4', count=2)
            nPixel_phyV1 = phyWidth * phyHeight
            nPixel = nPixel_phyV1
            print(f'phyV1: {phyWidth}x{phyHeight}')
        else:
            nPixel_phyV1 = 0
        if iFrameOutput == 2 or iFrameOutput == 3 or iFrameOutput == 6 or iFrameOutput == 7:
            visV1 = True
            visV1Width, visV1Height = np.fromfile(f, 'u4', count=2)
            nPixel_visV1_hlf = visV1Width * visV1Height
            nPixel_visV1 = nPixel_visV1_hlf*2
            nPixel += nPixel_visV1
            print(f'visV1: 2x{visV1Width}x{visV1Height}')
        else:
            nPixel_visV1 = 0
        if iFrameOutput == 4 or iFrameOutput == 5 or iFrameOutput == 6 or iFrameOutput == 7:
            visLGN = True
            visLGNwidth, visLGNheight = np.fromfile(f, 'u4', count=2)
            nPixel_visLGN_hlf = visLGNwidth * visLGNheight
            nPixel_visLGN = nPixel_visLGN_hlf*2
            nPixel += nPixel_visLGN
            print(f'visLGN: 2x{visLGNwidth}x{visLGNheight}')
        else:
            nPixel_visLGN = 0
        framedOutput = np.fromfile(f, 'f4') # already as firing rate
    
    print(framedOutput.size)
    print(framedOutput.shape)
    ont = framedOutput.size//nPixel
    print(ont, nPixel)
    assert(framedOutput.size%nPixel == 0)
    
    framedOutput = framedOutput.reshape(ont,nPixel)
    odt = ot*dt
    if phyV1 and plotPhyV1:
        phyV1frame = np.zeros((ont, phyHeight, phyWidth), dtype = 'f4')
    if visV1 and plotVisV1:
        visV1frame = np.zeros((ont, 2, visV1Height, visV1Width), dtype = 'f4')
    if visLGN and plotVisLGN:
        visLGNframe = np.zeros((ont, 2, visLGNheight, visLGNwidth), dtype = 'f4')
    for i in range(ont):
        if phyV1 and plotPhyV1:
            phyV1frame[i,:,:] = framedOutput[i,:nPixel_phyV1].reshape(phyHeight,phyWidth)
        if visV1 and plotVisV1:
            visV1frame[i,:,:,:] = framedOutput[i,nPixel_phyV1:(nPixel_phyV1+nPixel_visV1)].reshape(2, visV1Height, visV1Width)
        if visLGN and plotVisLGN:
            visLGNframe[i,:,:,:] = framedOutput[i,(nPixel_phyV1+nPixel_visV1):nPixel].reshape(2, visLGNheight, visLGNwidth)
     
    if phyV1 and plotPhyV1:
        phyV1_vmin = phyV1frame.min()
        phyV1_vmax = phyV1frame.max()
    if visV1 and plotVisV1:
        visV1_vmin = visV1frame.min()
        visV1_vmax = visV1frame.max()
    if visLGN and plotVisLGN:
        visLGN_vmin = visLGNframe.min()
        visLGN_vmax = visLGNframe.max()

    def meshgrid(x,y):
        X = np.tile(x,(len(y),1))
        Y = np.tile(y,(len(x),1)).T
        return X, Y
    
    phyV1_fns = []
    visV1_fns = []
    visLGN_fns = []
    for i in range(ont):
        if phyV1 and plotPhyV1:
            fig = plt.figure(f'phyV1-{i}', dpi = 144)
            ax = fig.add_subplot(111)
            #plt.Normalize, norm =
            image = ax.imshow(phyV1frame[i,::-1,:], aspect = 'equal', origin = 'lower', vmin = phyV1_vmin, vmax = phyV1_vmax)
            fig.colorbar(image, ax=ax)
            fn = fig_fdr+f'phyV1-{i}' + output_suffix + '.png'
            fig.savefig(fn)
            phyV1_fns.append(fn)
            plt.close(fig)
        
        if visV1 and plotVisV1:
            fig = plt.figure(f'visV1-{i}', dpi = 144)
            ax = fig.add_subplot(111)
            image = ax.imshow(np.hstack((visV1frame[i,0,::-1,:], visV1frame[i,1,::-1,:])), aspect = 'equal', origin = 'lower', vmin = visV1_vmin, vmax = visV1_vmax)
            fig.colorbar(image, ax=ax)
            fn = fig_fdr+f'visV1-{i}' + output_suffix + '.png'
            fig.savefig(fn)
            visV1_fns.append(fn)
            plt.close(fig)
        
        if visLGN and plotVisLGN:
            fig = plt.figure(f'visLGN-{i}', dpi = 144)
            ax = fig.add_subplot(111)
            image = ax.imshow(np.hstack((visLGNframe[i,0,::-1,:], visLGNframe[i,1,::-1,:])), aspect = 'equal', origin = 'lower', vmin = visLGN_vmin, vmax = visLGN_vmax)
            fig.colorbar(image, ax=ax)
            fn = fig_fdr+f'visLGN-{i}' + output_suffix + '.png'
            fig.savefig(fn)
            visLGN_fns.append(fn)
            plt.close(fig)

    ot_list = [i*odt for i in range(ont)]
    text = [f't = {time}' for time in ot_list]
    if phyV1 and plotPhyV1:
        create_gif(phyV1_fns, fig_fdr + 'phyV1' + output_suffix, 0.5, text = text, fontsize = 10, l = 0.1, t = 0.02)
        for f in phyV1_fns:
            if os.path.exists(f):
                os.remove(f) 
    if visV1 and plotVisV1:
        create_gif(visV1_fns, fig_fdr + 'visV1' + output_suffix, 0.5, text = text, fontsize = 10, l = 0.1, t = 0.02)
        for f in visV1_fns:
            if os.path.exists(f):
                os.remove(f) 
    if visLGN and plotVisLGN:
        create_gif(visLGN_fns, fig_fdr + 'visLGN' + output_suffix, 0.5, text = text, fontsize = 10, l = 0.1, t = 0.02)
        for f in visLGN_fns:
            if os.path.exists(f):
                os.remove(f) 

if plotSampleTrace:
    with open(rawData_fn, 'rb') as f:
        dt = np.fromfile(f, 'f4', count = 1)[0]
        nt = np.fromfile(f, 'u4', count = 1)[0]
        nV1 = np.fromfile(f, 'u4', count = 1)[0]
        print(f'{nt}, {nV1}')
        iModel = np.fromfile(f, 'i4', 1)[0] 
        mI = np.fromfile(f, 'u4', 1)[0] 
        hWrite = np.fromfile(f, 'u4', count = 1)[0]
        ngTypeFF = np.fromfile(f, 'u4', count = 1)[0]
        ngTypeE = np.fromfile(f, 'u4', count = 1)[0]
        ngTypeI = np.fromfile(f, 'u4', count = 1)[0]
        print(f'{ngTypeE}, {ngTypeI}')
        spikeTrain = np.zeros((nt,nV1), dtype = 'f4')
        depC = np.zeros((nt,nV1), dtype = prec)
        if iModel == 1:
            w = np.zeros((nt,nV1), dtype = prec)
        v = np.zeros((nt,nV1), dtype = 'f4')
        gFF = np.zeros((nt,ngTypeFF,nV1), dtype = 'f4')
        gE = np.zeros((nt,ngTypeE,nV1), dtype = 'f4')
        gI = np.zeros((nt,ngTypeI,nV1), dtype = 'f4')
        gapI = np.zeros((nt,ngTypeI,mI), dtype = 'f4')
        if hWrite:
            hFF = np.zeros((nt,ngTypeFF,nV1), dtype = 'f4')
            hE = np.zeros((nt,ngTypeE,nV1), dtype = 'f4')
            hI = np.zeros((nt,ngTypeI,nV1), dtype = 'f4')
        for i in range(nt):
            spikeTrain[i,:] = np.fromfile(f, 'f4', count = nV1) #sInfo: integer(nsp) + decimal(tsp, normalized)
            depC[i,:] = np.fromfile(f, 'f4', count = nV1)
            if iModel == 1:
                w[i,:] = np.fromfile(f, 'f4', count = nV1)
                
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

            gapI[i,:] = np.fromfile(f, prec, mI)

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
        with open(LGN_V1_idFn, 'rb') as f: # takes long time
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

    fig.savefig(fig_fdr + f'histogram{output_suffix}.png')
    
    fig = plt.figure('vgtsp', dpi = 1024)
    np.random.seed(seed)
    if 'pick' not in locals():
        pick = np.random.randint(nV1, size = nsample)
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

    fig.savefig(fig_fdr + f'vgtsp-{i}{output_suffix}.png')
