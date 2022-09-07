import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.gridspec as gs
import matplotlib.cbook as cbook
from matplotlib import cm
import sys, os
from sys import stdout
from readPatchOutput import *
sys.path.append(os.path.realpath('..'))
from ext_signal import apply_sRGB_gamma, LMS2sRGB
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

ns = 50
tau = 200
tau_step = 8
t0 = 10 # where spike > t0 are counted
plotSnapshots = True
plotPop = False # calc props from the max deviation RF snapshot 
tau_pick = tau_step//3 # one of the steps from tau_steps for pop analysis
checkFrame = True

def LMS2RG_axis(sta, ns, ntau_step, nChannel, height, width, initL, initM):
    assert(nChannel == 2 or nChannel == 3)
    RG_axis = np.zeros((ns, ntau_step, height, width))
    vmin = np.zeros(ns)
    vmax = np.zeros(ns)
    for i in range(ns):
        data = sta[i,:,:,:,:].reshape(ntau_step,nChannel,height*width)
        L = data[:,0,:]- initL
        M = data[:,1,:]- initM
        R_plus = np.array([np.min(L), np.max(L)]) 
        G_minus = np.array([np.min(M), np.max(M)])

        #data = (dR*L - dG*M).reshape(ntau_step, height, width)
        data = (M - L).reshape(ntau_step, height, width)
        vmin[i] = np.min(data)
        vmax[i] = np.max(data)
        if np.abs(vmax[i]) > np.abs(vmin[i]):
            ratio = np.abs(vmax[i])/np.abs(vmin[i])
            vmin[i] = vmin[i]*ratio
        else:
            ratio = np.abs(vmin[i])/np.abs(vmax[i])
            vmax[i] = vmax[i]*ratio
        RG_axis[i, :, :, :] = data

    return RG_axis, vmin, vmax

def acuity(ecc): 
    acuityK = 0.2049795945022049
    logAcuity0 = 3.6741080244555278
    cpd = -acuityK*ecc + logAcuity0
    cpd = np.exp(cpd)
    return 1.0/cpd/4

def getAcuityAtEcc(ecc, deg = True):
    if not deg:
        ecc = ecc/np.pi*180
    rsig = 1
    acuityC = acuity(ecc)/180*np.pi/rsig; # from cpd to radius of center
    acuityS = acuityC*6
    return acuityC, acuityS

def vpos_to_ppos(x0, y0, ecc, polar, normViewDistance):
    r = np.tan(ecc)*normViewDistance
    x = x0 + r*np.cos(polar)
    y = y0 + r*np.sin(polar)
    return x, y

def getReceptiveField(spikeInfo, inputFn, frameRate, res_fdr, setup_fdr, data_fdr, fig_fdr, output_suffix, suffix, sampleList, nstep, dt):
    ns = sampleList.size
    nFrame, width, height, initL, initM, initS, frames, buffer_ecc, ecc, neye = read_input_frames(inputFn) 
    print(nFrame)
    print(width)
    print(height)
    n = len(spikeInfo)
    print(f'n = {n}')
    _output_suffix = '_' + output_suffix

    stepRate = int(round(1000/dt))
    print(stepRate, frameRate)
    moreDt = True
    if stepRate < frameRate:
    	moreDt = False
    # norm/denorm = the normalized advancing phase at each step, 
    # denorm is the mininum number of frames such that total frame length is a multiple of dt.
    denorm, norm = find_denorm(frameRate, stepRate , moreDt)
    exact_norm = np.empty(denorm, dtype=int) # to be normalized by denorm to form normalized phase
    current_norm = 0
    print(f'{denorm} exact phases in [0,1]:')
    for i in range(denorm):
        if current_norm>denorm:
            current_norm = current_norm - denorm

        exact_norm[i] = current_norm
        if i<denorm - 1:
    	    stdout.write(f'{current_norm}/{denorm}, ')
        else:
    	    stdout.write(f'{current_norm}/{denorm}\n')

        current_norm = current_norm + norm

    assert(current_norm == denorm)

    exact_it = np.empty(denorm, dtype = int)
    for i in range(denorm):
    	exact_it[i] = (i*stepRate)/frameRate # => quotient of "i*Tframe/Tdt"
    	print(f'{exact_it[i]} + {exact_norm[i]}/{denorm}')

    # the first i frames' accumulated length in steps = exact_it[i] + exact_norm[i]/denorm
    assert(np.mod((stepRate*denorm), frameRate) == 0)
    co_product = (stepRate*denorm)//frameRate
    # the number of non-zero minimum steps to meet frameLength * denorm with 0 phase
    ntPerFrame = co_product # tPerFrame in the units of dt/denorm
    print(co_product)
    
    step0 = int(round(t0/dt))
    ntau = int(round(tau/dt))
    if np.mod(ntau, tau_step) != 0:
        raise Exception('ntau need to be a multiple of tau_step')
    ntau_step = tau_step + 1
    it_tau = np.linspace(0,ntau,ntau_step).astype('u4')
    sta = np.zeros((ns, ntau_step, 3, height, width), dtype = 'f4')
    isp = np.zeros(ns, dtype = int)
    ssInfo = np.array([np.round(x[x > step0*dt]/dt) for x in spikeInfo[sampleList]], dtype = object) # rounded up to the nearest dt
    nsp = np.array([x.size for x in ssInfo], dtype = 'u4')
    print(f'average spikes per sample = {np.mean(nsp)}')

    
    it0 = max(0, step0-ntau) # frame-counting starts ntau steps before step0
    currentFrame = it0*denorm//ntPerFrame # [0, nstep*dt/frameRate]
    iFrame = np.mod(currentFrame, nFrame) # [0, nFrame]
    jFrame = np.mod(iFrame - 1 + nFrame, nFrame)
    inorm = np.mod(currentFrame, denorm)

    print(f'it0 = {it0}, step0 = {step0}, ntau = {ntau}, nstep = {nstep}')
    # initialize for spikes that have blank frames.
    for i in range(ns):
        if ssInfo[i].size > 0:
            pickNear = ssInfo[i] < ntau
            nNear = np.sum(pickNear)
            if nNear > 0:
                ssInfoNear = ssInfo[i][pickNear]
                isp = np.tile(ssInfoNear.reshape(nNear, 1), (1, ntau_step)) - np.tile(it_tau, (nNear,1))
                itau = np.nonzero(isp <= 0)[1]
                for j in itau:
                    sta[i, j, 0, :, : ] += initL
                    sta[i, j, 1, :, : ] += initM
                    sta[i, j, 2, :, : ] += initS

    if checkFrame:
        fig = plt.figure('check frames')
        grid_row = 3
        grid_col = (min(nFrame,12)+grid_row-1)//grid_row
        grid = gs.GridSpec(grid_row, grid_col, figure = fig)
    for it in range(it0, nstep):
        if (it+1)*denorm >= currentFrame*ntPerFrame:
            #print(f'{jFrame}->{iFrame} frame')
            fdt = exact_norm[inorm]/denorm # the start of iFrame happens at (it*dt, it*dt + fdt, (it+1)*dt)
            frame = frames[jFrame,:,:,:]*fdt + frames[iFrame,:,:,:]*(1-fdt)
            #print([np.mean(frame[0,:,:]), np.mean(frame[1,:,:]), np.mean(frame[2,:,:])])

            #print(f'start of frame{iFrame} at {currentFrame * 1000/frameRate:.3f} ms, between it = {it}->{it+1}, t =({it*dt},{(it+1)*dt}), fdt = {fdt:.3f} ms')
            assert(currentFrame * 1000/frameRate <= (it+1)*dt and it*dt < currentFrame * 1000/frameRate or it == it0)
            currentFrame = currentFrame + 1
            iFrame = np.mod(iFrame + 1, nFrame)
            jFrame = np.mod(jFrame + 1, nFrame)
            inorm = np.mod(inorm + 1, denorm)
            if checkFrame and currentFrame < min(nFrame,12):
                ax = fig.add_subplot(grid[iFrame//grid_col, np.mod(iFrame,grid_col)])

                data = frame.reshape(3,height*width)
                img = np.round(apply_sRGB_gamma(np.matmul(LMS2sRGB, data))*255).T.reshape(height,width,3).astype('u1')
                ax.imshow(img, aspect = 'equal', origin = 'lower')
                ax.plot(R_x0*width*np.ones(height+1), np.arange(height+1), ':k', lw = 0.02)
                ax.axis('off')
        else:
            frame = frames[jFrame,:,:,:]
        if plotPop:
            pass

        for i in range(ns):
            if ssInfo[i].size > 0:
                pickNear = np.logical_and(ssInfo[i] <= it + ntau, ssInfo[i] >= it)
                nNear = np.sum(pickNear)
                if nNear > 0:
                    ssInfoNear = ssInfo[i][pickNear]
                    isp = np.tile((ssInfoNear-it).reshape(nNear, 1), (1, ntau_step)) - np.tile(it_tau, (nNear,1))
                    itau = np.nonzero(isp == 0)[1]
                    for j in itau:
                        sta[i, j, :, :, : ] += frame
        #    sta frames
    ### don't do average here
    #for i in range(ns):
    #    if ssInfo[i].size > 0:
    #        sta[i, :, :, :, :] /= ssInfo[i].size
    if checkFrame:
        fig.savefig(fig_fdr + '/' + inputFn[:-4] + '.png', dpi = 300)

    with open(data_fdr + 'STA_samples' + _output_suffix + '-' + suffix + '.bin', 'wb') as f:
        np.array([ns, ntau_step, height, width], dtype='u4').tofile(f)
        np.array([buffer_ecc, ecc], dtype='f4').tofile(f)
        np.array([neye]).astype('u4').tofile(f)
        it_tau.tofile(f)
        np.array([dt, initL, initM, initS], dtype='f4').tofile(f)
        sampleList.astype('u4').tofile(f)
        nsp.tofile(f)
        sta.tofile(f)

    with open(data_fdr + 'STA_pop' + _output_suffix + '-' + suffix + '.bin', 'wb') as f:
        np.array([ns, height, width], dtype='u4').tofile(f)
        np.array([buffer_ecc, ecc], dtype='f4').tofile(f)
        np.array([neye]).astype('u4').tofile(f)
        np.array([tau_pick, it_tau[tau_pick]], dtype='u4').tofile(f)
        np.array([dt, initL, initM, initS], dtype='f4').tofile(f)
        nsp.tofile(f)
        #popSF.tofile(f)
        #pop.tofile(f)

def plotSta(isuffix, res_suffix, conLGN_suffix, output_suffix, res_fdr, setup_fdr, data_fdr, fig_fdr, nf, hasOP = False):

    if isuffix == 0:
        suffix = 'LGN'
    if isuffix == 1:
        suffix = 'V1'
    if isuffix != 0 and isuffix != 1:
        raise Exception('wrong isuffix value, 0 for LGN 1 for V1')

    if res_fdr[-1] != '/':
        res_fdr = res_fdr + '/'
    if data_fdr[-1] != '/':
        data_fdr = data_fdr + '/'
    if setup_fdr[-1] != '/':
        setup_fdr = setup_fdr + '/'

    res_suffix = "_" + res_suffix

    _output_suffix = '_' + output_suffix
    LGN_V1_idFn = "LGN_V1_idList_" + conLGN_suffix + ".bin"
    if nf == 0:
        parameterFn = "patchV1_cfg" +_output_suffix + ".bin"
    else:
        parameterFn = "patchV1_cfg" +_output_suffix + "_1.bin"
    parameterFn = data_fdr + parameterFn
    pref_file = data_fdr + 'cort_pref_' + output_suffix + '.bin'

    LGN_vposFn = res_fdr + 'LGN_vpos'+ res_suffix + ".bin"
    featureFn = res_fdr + 'V1_feature' + res_suffix + ".bin"
    V1_allposFn = res_fdr + 'V1_allpos' + res_suffix + ".bin"

    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, nLGN, nV1, nstep, dt, normViewDistance, L_x0, L_y0, R_x0, R_y0, virtual_LGN, _, _ = read_cfg(parameterFn, True)
    if virtual_LGN:
        raise Exception('not implemented for virtual_LGN')

    LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)
    nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type, polar0, ecc0 = readLGN_vpos(LGN_vposFn, True)
    ecc0 *= np.pi/180
    if suffix == 'V1':
        featureType = np.array([0,1])
        feature, rangeFeature, minFeature, maxFeature = readFeature(featureFn, nV1, featureType)
        LR = feature[0,:]
        with open(V1_allposFn, 'rb' ) as f:
            nblock, neuronPerBlock, posDim = np.fromfile(f, 'u4', 3)
            _ = np.fromfile(f, 'f8', 4)
            nV1 = nblock * neuronPerBlock
            _ = np.fromfile(f, 'f8', 4 + 2*nV1)
            vx, vy = np.fromfile(f, 'f8', 2*nV1).reshape(2,nV1)
            ecc1 = np.sqrt(vx*vx + vy*vy)*np.pi/180
            polar1 = np.arctan2(vy, vx)
            print(f'x: {[np.min(vx), np.max(vx)]}')
            print(f'y: {[np.min(vy), np.max(vy)]}')
            print(f'ecc: {[np.min(ecc1), np.max(ecc1)]}')
            print(f'polar: {[np.min(polar1), np.max(polar1)]}')

    if hasOP:
        if nf > 0:
            try:
                f = open(pref_file, 'rb')
            except IOError:
                print(f'{pref_file} not found run getTuningCurve.py first')
    else:
        pref_file = ''

    if nf > 0:
        sample_data_files = ["STA_samples" + _output_suffix + '_' + str(i+1) + '-' + suffix + '.bin' for i in range(nf)]
        pop_data_files = ["STA_pop" + _output_suffix + '_' + str(i+1) + '-' + suffix + '.bin' for i in range(nf)]
        ntrial = nf
    else:
        sample_data_files = ["STA_samples" + _output_suffix + '-' + suffix + '.bin']
        pop_data_files = ["STA_pop" + _output_suffix + '-' + suffix + '.bin']
        ntrial = 1

    for i in range(ntrial):
        with open(data_fdr + sample_data_files[i], 'rb') as f:
            if i == 0:
                ns, ntau_step, height, width = np.fromfile(f, 'u4', 4)
                buffer_ecc, ecc = np.fromfile(f, 'f4', 2)
                neye = np.fromfile(f, 'u4', 1)[0]
                it_tau = np.fromfile(f, 'u4', ntau_step)
                dt, initL, initM, initS = np.fromfile(f, 'f4', 4)
                sampleList = np.fromfile(f, 'u4', ns)

                nsp = np.zeros(ns, dtype = 'u4')
                sta = np.zeros((ns, ntau_step, 3, height, width), dtype = 'f4')
            else:
                _ns, _ntau_step, _height, _width = np.fromfile(f, 'u4', 4)
                _buffer_ecc, _ecc = np.fromfile(f, 'f4', 2)
                _neye = np.fromfile(f, 'u4', 1)[0]
                _it_tau = np.fromfile(f, 'u4', _ntau_step)
                _dt, _initL, _initM, _initS = np.fromfile(f, 'f4', 4)
                _sampleList = np.fromfile(f, 'u4', ns)

                assert(ns == _ns and _ntau_step == ntau_step and _height == height and _width == width)
                assert(buffer_ecc == _buffer_ecc and ecc == _ecc)
                assert(_neye == neye)
                assert((_it_tau - it_tau == 0).all())
                assert(_dt == dt)
                assert(_initL == initL and _initM == initM and _initS == initS)
                assert((_sampleList - sampleList == 0).all())

            nsp += np.fromfile(f, 'u4', ns)
            sta += np.fromfile(f, 'f4').reshape((ns, ntau_step, 3, height, width))
    ### average here 
    for i in range(ns):
        if nsp[i] > 0:
            sta[i, :, :, :, :] /= nsp[i]

    sta, vmin, vmax = LMS2RG_axis(sta, ns, ntau_step, 3, height, width, initL, initM)

    print(f'nsample = {ns}, ntau_step = {ntau_step}, height = {height}, width = {width}, dt = {dt}, inits = {[initL, initM, initS]}')
    print(f'it_tau = {it_tau}')
    print(f'sampleList = {sampleList}')
    max_itau = np.zeros(ns, dtype = 'i4') - 1
    max_pos = np.zeros((ns, 2), dtype = 'i4')
    for i in range(ns):
        if nsp[i] > 0:
            dev_data = np.abs(sta[i,:,:,:].reshape(ntau_step, width*height))
            max_itau[i] = np.argmax(np.max(dev_data, axis = -1))
            max_id = np.argmax(dev_data[max_itau[i],:])
            max_pos[i, 0] = np.mod(max_id,width)
            max_pos[i, 1] = max_id//width
    print(f'spikes: {nsp}')
    print(f'RF max deviation at tau: {max_itau}')


    if neye == 1:
        range_ecc = 2*(ecc+buffer_ecc)
    else:
        range_ecc = 2*(ecc+2*buffer_ecc)
        assert(neye == 2)
    
    degPerPixel_w = range_ecc / width 
    degPerPixel_h = range_ecc / height 

    if hasOP:
        if nf > 0:
            with open(pref_file, 'rb') as f:
                fitted = np.fromfile(f, 'i4', 1)[0]
                n = np.fromfile(f, 'u4', 1)[0]
                if fitted == 1:
                    OP_fitted = np.fromfile(f, 'f4', n)
                OP_cort = np.fromfile(f, 'f4', n)
                OP_thal = np.fromfile(f, 'f4', n)
                OP_preset = np.fromfile(f, 'f4', n)
        else:
            OP = np.mod(feature[1,:] + 0.5, 1.0)*180

    print('data collected')
    print(L_x0, L_y0, R_x0, R_y0)
    if plotSnapshots:
        for i in range(ns):
            id = sampleList[i]
            if nsp[i] == 0:
                print(f'skip #{id}')
                continue
            fig = plt.figure('sample RF snapshots')
            grid_row = 3
            grid_col = (ntau_step+grid_row-1)//grid_row
            #grid = gs.GridSpec(grid_row, grid_col, figure = fig, hspace = 0.05, wspace = 0.05)
            grid = gs.GridSpec(grid_row, grid_col, figure = fig)

            # normalize sta
            sta[i,:,:,:] = (sta[i,:,:,:] - vmin[i])/(vmax[i] - vmin[i])
            if isuffix == 0:
                if id < nLGN_I:
                    lr = 'left'
                    x, y = vpos_to_ppos(L_x0, L_y0, ecc0[id], polar0[id], normViewDistance)
                else:
                    lr = 'right'
                    x, y = vpos_to_ppos(R_x0, R_y0, ecc0[id], polar0[id], normViewDistance)
                r_center, r_surround = getAcuityAtEcc(ecc0[id])
                center0 = np.ceil((np.tan(ecc0[id] + r_center) - np.tan(ecc0[id]))*normViewDistance*width)
                surround0 = np.ceil((np.tan(ecc0[id] + r_surround) - np.tan(ecc0[id]))*normViewDistance*width)
                surround1 = surround0
                center1 = center0
                print(f'center: {center0}, surround: {surround0} pixels')
            else:
                if LR[id] < 0:
                    lr = 'left'
                    x, y = vpos_to_ppos(L_x0, L_y0, ecc1[id], polar1[id], normViewDistance)
                else:
                    lr = 'right'
                    x, y = vpos_to_ppos(R_x0, R_y0, ecc1[id], polar1[id], normViewDistance)
                if nLGN_V1[id] > 0:
                    ecc = np.min(ecc0[LGN_V1_ID[id]])
                else:
                    ecc = ecc1[id]

                r_center, r_surround = getAcuityAtEcc(ecc)
                center0 = np.ceil((np.tan(ecc + r_center) - np.tan(ecc))*normViewDistance*width)
                surround0 = np.ceil((np.tan(ecc + r_surround) - np.tan(ecc))*normViewDistance*width)
                ecc = np.max(ecc0[LGN_V1_ID[id]])
                r_center, r_surround = getAcuityAtEcc(ecc)
                center1 = np.ceil((np.tan(ecc + r_center) - np.tan(ecc))*normViewDistance*width)
                surround1 = np.ceil((np.tan(ecc + r_surround) - np.tan(ecc))*normViewDistance*width)
                print(f'center: {[center0, center1]}, surround: {[surround0, surround1]} pixels')

            print(f'{lr}-parent: {(x, y)}')
            print(f'max_pos: {max_pos[i,:]}')
            print(f'center: {(x*width,y*height)}')
            if hasOP:
                if nf > 0:
                    if fitted == 1:
                        pref = np.round(np.array([OP_fitted[id], OP_cort[id], OP_thal[id], OP_preset[id]])*180/np.pi).astype('i4')
                    else:
                        pref = np.round(np.array([OP_cort[id], OP_thal[id], OP_preset[id]])*180/np.pi).astype('i4')
                else:
                    pref = np.round(np.array(OP[id])*180/np.pi).astype('i4')

                if LR[id] < 0:
                    LR_x0, LR_y0 = L_x0, L_y0
                else:
                    LR_x0, LR_y0 = R_x0, R_y0

                if nLGN_V1[id] > 0:
                    #markers = (',r', ',g', ',g', ',r', ',r', ',g')
                    markers = ('^r', 'vg', '*g', 'dr', '^k', 'vb')
                    labels = ('L-on', 'L-off', 'M-on', 'M-off', 'On', 'Off')
                    LGN_x = np.zeros(nLGN_V1[id])
                    LGN_y = np.zeros(nLGN_V1[id])
                    for j in range(nLGN_V1[id]):
                        LGN_id = LGN_V1_ID[id][j]
                        if LGN_id < nLGN_I:
                            LGN_x[j], LGN_y[j] = vpos_to_ppos(L_x0, L_y0, ecc0[LGN_id], polar0[LGN_id], normViewDistance) 
                        else:
                            LGN_x[j], LGN_y[j] = vpos_to_ppos(R_x0, R_y0, ecc0[LGN_id], polar0[LGN_id], normViewDistance) 

                        print(f'    child: {(LGN_x[j], LGN_y[j])}')
            for itau in range(ntau_step):
                ax = fig.add_subplot(grid[itau//grid_col, np.mod(itau, grid_col)])
                
                #cmap_name = 'gray'
                cmap_name = 'PiYG'
                ax.imshow(sta[i,itau,:,:], cmap = cmap_name, aspect = 'equal', origin = 'lower')
                ax.plot(x*width, y*height, '.k', ms = 0.5, label = 'preset c.')
                if neye == 2:
                    ax.plot(R_x0*width*np.ones(height+1), np.arange(height+1), '-k', lw = 0.2)
                if hasOP and nLGN_V1[id] > 0:
                    for j in range(nLGN_V1[id]):
                        jtype = LGN_type[LGN_V1_ID[id][j]]
                        ax.plot(LGN_x[j]*width, LGN_y[j]*height, markers[jtype], ms = 0.5, label = labels[jtype])

                extent = [surround0, surround1]
                #extent = [center0, center1]
                if hasOP:
                    if nLGN_V1[id] > 0:
                        ax.set_xlim(np.min([x,np.min(LGN_x)])*width-extent[0], np.max([x,np.max(LGN_x)])*width + extent[1])
                        ax.set_ylim(np.min([y,np.min(LGN_y)])*height-extent[0], np.max([y,np.max(LGN_y)])*height + extent[1])
                else:
                    ax.set_xlim(x*width-extent[0], x*width + extent[1])
                    ax.set_ylim(y*height-extent[0],y*height + extent[1])

                xleft, xright = ax.get_xlim()
                ybot, ytop = ax.get_ylim()

                if max_pos[i,0] < xleft or max_pos[i,0] > xright or max_pos[i,1] < ybot or max_pos[i,1] > ytop:
                    max_inside = False 
                else:
                    ax.plot(max_pos[i,0], max_pos[i,1], '*b', ms = 0.5, label = 'max r.')
                    max_inside = True

                time = f'{it_tau[itau]*dt} ms '
                if itau == 0:
                    if hasOP:
                        title = f'{time}, {nsp[i]} spikes, OP:{pref}, max inside:{max_inside}'
                    else:
                        title = f'{time}, {nsp[i]} spikes, LGN type: {LGN_type[id]}, max inside:{max_inside}'
                else:
                    title = f'{time}'
                if itau == max_itau[i]:
                    title = f'max {title}'
                    ax.set_xlabel('deg')
                    ax.set_ylabel('deg')
                else:
                    ax.axis('off')
                if itau == 0:
                    ax.legend(fontsize = 'xx-small',bbox_to_anchor=(-0.05, 1), loc='upper right')
                        
                ax.set_title(title, fontsize = 8)
                
                xtick = ax.get_xticks()
                ytick = ax.get_yticks()
                if len(xtick) > 1:
                    _xtick = xtick[-1] - xtick[0]
                else:
                    _xtick = xtick[0]
                if len(ytick) > 1:
                    _ytick = ytick[-1] - ytick[0]
                else:
                    _ytick = ytick[0]

                if _xtick*degPerPixel_w < 1:
                    deg_tlx = '{pos:.2f}'
                else:
                    deg_tlx = '{pos:.0f}'

                if _ytick*degPerPixel_h < 1:
                    deg_tly = '{pos:.2f}'
                else:
                    deg_tly = '{pos:.0f}'

                xticklabel = [deg_tlx.format(pos = x*degPerPixel_w) for x in xtick] 
                yticklabel = [deg_tly.format(pos = y*degPerPixel_h) for y in ytick] 
                ax.set_xticks(xtick)
                ax.set_yticks(ytick)
                ax.set_xticklabels(xticklabel)
                ax.set_yticklabels(yticklabel)

            ax = fig.add_subplot(15,20,300)
            ax.plot(sta[i,:,max_pos[i,0], max_pos[i,1]], '-r', lw = 0.1)
            ax.plot(np.arange(ntau_step), np.zeros(ntau_step), ':k', lw = 0.1)
            ax.axis('off')
            if isuffix == 1:
                fig.savefig(fig_fdr + '/' + f'sampleRF_snapshots-{output_suffix}-{suffix}-{id//neuronPerBlock}-{np.mod(id,neuronPerBlock)}#{nLGN_V1[id]}'  + '.png', dpi = 2000)
            else:
                fig.savefig(fig_fdr + '/' + f'sampleRF_snapshots-{output_suffix}-{suffix}-{id}.png', dpi = 2000)

            plt.close(fig)

def find_denorm(u1, u2, MorN):
    if MorN: #u2 > u1
        m = u1
        if np.mod(u2,u1) == 0: # only one zero phase
            norm = 1
            return 1, 1
        n = u2 - u1*(u2//u1)
    else:
        m = u2
        if np.mod(u1,u2) == 0: # only one zero phase
            return 1, 1
        n = u1 - u2*(u1//u2)

    print(f'm = {m}, n = {n}')
    assert (m>n)
    for i in range(n,1,-1):
        if np.mod(n,i) == 0 and np.mod(m,i)==0:
            return m//i, n//i
    return m, 1

if __name__ == "__main__":
    if sys.argv[1] == 'collecting':
        if len(sys.argv) < 10:
            raise Exception('not enough argument for getReceptiveField.py collecting output_suffix, conLGN_suffix, res_fdr, setup_fdr, data_fdr, fig_fdr, LGN_or_V1, seed, one')
        output_suffix = sys.argv[2]
        conLGN_suffix = '_' + sys.argv[3]
        res_fdr = sys.argv[4]
        setup_fdr = sys.argv[5]
        data_fdr = sys.argv[6]
        fig_fdr = sys.argv[7]
        LGN_or_V1 = int(sys.argv[8])
        seed = int(sys.argv[9])
        if len(sys.argv) < 11:
            one = False
        else:
            if sys.argv[10] == True or sys.argv[10] == '1':
                one = True
            elif sys.argv[10] == False or sys.argv[10] == '0':
                one = False
            else:
                raise Exception('one can only be True(1) or False(0)')

        _output_suffix = '_' + output_suffix
        if data_fdr[-1] != '/':
            data_fdr = data_fdr + '/'

        parameterFn = data_fdr + "patchV1_cfg" +_output_suffix + ".bin"
        spDataFn = data_fdr + "V1_spikes" + _output_suffix
        LGN_spFn = data_fdr + "LGN_sp" + _output_suffix

        prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, nLGN, nV1, nstep, dt, normViewDistance, L_x0, L_y0, R_x0, R_y0, virtual_LGN, _, _ = read_cfg(parameterFn, True)

        np.random.seed(seed)
        if LGN_or_V1 == 0:
            with np.load(LGN_spFn + '.npz', allow_pickle=True) as data:
                spName = data['spName']
                spScatter = data[f'{spName}']

            sampleList = np.random.randint(nLGN, size = ns, dtype = 'u4')
            suffix = 'LGN'

        if LGN_or_V1 == 1:
            
            with np.load(spDataFn + '.npz', allow_pickle=True) as data:
                spName = data['spName']
                spScatter = data[f'{spName}']
            if not one:
                max_frFn = 'max_fr_' + output_suffix[:-2] + '.bin'
            else:
                max_frFn = 'max_fr_' + output_suffix + '.bin'

            with open(data_fdr + max_frFn, 'rb') as f:
                max_fr = np.fromfile(f, 'f8', nV1)
            blockSize = nE + nI
            nblock = nV1//blockSize
            print(f'nblock = {nblock}, blockSize = {blockSize}')
            epick = np.hstack([np.arange(nE) + iblock*blockSize for iblock in range(nblock)])
            ipick = np.hstack([np.arange(nI) + iblock*blockSize + nE for iblock in range(nblock)])

            #sampleList = epick[np.argpartition(-max_fr[epick], ns)[:ns]]
            #sampleList = epick[np.argpartition(-max_fr[epick], ns)[:ns]]
            sampleList = np.random.randint(nV1, size = ns, dtype = 'u4')
            suffix = 'V1'

        getReceptiveField(spScatter, inputFn, frameRate, res_fdr, setup_fdr, data_fdr, fig_fdr, output_suffix, suffix, sampleList, nstep, dt)

    if sys.argv[1] == 'plotting':
        if len(sys.argv) < 11:
            raise Exception('not enough argument for getReceptiveField.py plotting isuffix, res_suffix, conLGN_suffix, output_suffix, res_fdr, setup_fdr, data_fdr, fig_fdr, nf, LGN_or_V1')
        
        isuffix = int(sys.argv[2])
        res_suffix = sys.argv[3]
        conLGN_suffix = sys.argv[4]
        output_suffix = sys.argv[5]
        res_fdr = sys.argv[6]
        setup_fdr = sys.argv[7]
        data_fdr = sys.argv[8]
        fig_fdr = sys.argv[9]
        nf = int(sys.argv[10])
        if len(sys.argv) < 12:
            hasOP = False
        else:
            if sys.argv[11] == True or sys.argv[11] == '1':
                hasOP = True
            elif sys.argv[11] == False or sys.argv[11] == '0':
                hasOP = False
            else:
                raise Exception('hasOP can only be True(1) or False(0)')

        plotSta(isuffix, res_suffix, conLGN_suffix, output_suffix, res_fdr, setup_fdr, data_fdr, fig_fdr, nf, hasOP)
