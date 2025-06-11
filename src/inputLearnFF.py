# essential input files
# suffix: theme string $lgn in lFF.slurm
# seed: for randomize LGN connecction
# std_ecc: initial connections weights to be gaussian distributed if nonzero
# suffix0: theme string #lgn0 in lFF.slurm
# stage: retinal wave stages, takes 2 or 3
import numpy as np
import warnings
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from collections import Counter
mpl.rcParams.update({'font.family': 'CMU Sans Serif', 'axes.unicode_minus' : False})
mpl.rcParams.update({'mathtext.fontset': 'cm', 'mathtext.default':'regular'})
from patch_square import *
from scipy.stats import qmc

def inputLearnFF(inputFn, suffix, seed, std_ecc, suffix0, stage, res_fdr, setup_fdr, squareOrCircle, relay, binary_thres, sInput, retino_cross, retinotopy = 0): 
    #gap = True
    gap = False
    nc = 16
    t0_perc = 0.5
    relay_thres = 0.5
    cross_reldis = 0.45
    res_fdr = res_fdr + '/'
    setup_fdr = setup_fdr + '/'
    con_std = 0
    gapCon = 0.75
    gapStr = 0.5
    nmin = 6
    nmin_ext = 8
    bmin = 4
    icolor = {19: '#00a896', 113: '#147df5', 312: '#ff6d1f'}

    if std_ecc > 0:
        normed = True
    else:
        normed = False
    
    if stage == 2:
        u0 = 1.0
        u1 = 1.0
    else:
        u0 = 1.0
        u1 = 1.0
        ##u0 = 0.666667; # uniform range
        ##u1 = 1.333333;
    
    #### HERE ########
    if stage == 5:
        nother = 1
        set_other = np.array([[0.32],[0.85]])
    else:
        nother = 0
    
    same = False
    
    if stage == 2 or stage == 5:
        pCon = 0.95
        if retinotopy > 0:
            nLGN_1D = 24
        else:
            nLGN_1D = 16
            #nLGN_1D = 32
            #nLGN_1D = 48

        radiusRatio = 1.0
    
    if stage == 3:
        if relay:
            nLGN_1D = 16
        else:
            nLGN_1D = 8

        pCon = 0.95
        radiusRatio = 1.0
    
    ########
    
    if not len(suffix0)==0 :
        suffix0 = '-' + suffix0
    
    if not len(suffix)==0 :
        suffix = '-' + suffix
    
    fLGN_V1_ID = setup_fdr + 'LGN_V1_idList' + suffix + '.bin'
    fLGN_V1_s = setup_fdr + 'LGN_V1_sList' + suffix + '.bin'
    fV1_RFprop = setup_fdr + 'V1_RFprop' + suffix + '.bin'
    fLGN_switch = setup_fdr + 'LGN_switch' + suffix + '.bin'
    if retino_cross or gap:
        crossFn = setup_fdr + 'cross_pair' + suffix + '.bin'
        nblock = 32
        blockSize = 32
        mI = 24
        mE = 8
    else:
        nblock = 1
        mE = 992
        mI = 32
        #nblock = 32
        #mE = 24 
        #mI = 8
        blockSize = mE + mI

    fV1_vposFn = res_fdr + 'V1_vpos' + suffix0 + '.bin'
    fV1_allpos = res_fdr + 'V1_allpos' + suffix0 + '.bin'
    fV1_feature = res_fdr + 'V1_feature' + suffix0 + '.bin'
    fLGN_vpos = res_fdr + 'LGN_vpos' + suffix0 + '.bin'
    fLGN_surfaceID = res_fdr + 'LGN_surfaceID' + suffix0 + '.bin'
    #parvoMagno = 1 # parvo
    parvoMagno = 2
    #parvoMagno = 3 # both, don't use, only for testing purpose
    
    default_rng = np.random.default_rng(seed)
    initialConnectionStrength = 1.0
    
    eiStrength = 0.0
    ieStrength = 0.0
    nV1 = nblock * blockSize
    doubleOnOff = 1
    frameVisV1output = False
    
    with open(res_fdr + inputFn + '.cfg','rb') as f:
        nStage = np.fromfile(f, 'u4', 1)[0]
        nOri = np.fromfile(f, 'u4', nStage)
        nRep = np.fromfile(f, 'u4', nStage)
        frameRate = np.fromfile(f, 'f8', 1)[0]
        framesPerStatus = np.fromfile(f, 'u4', nStage)
        framesToFinish = np.fromfile(f, 'u4', nStage)
        max_ecc = np.fromfile(f, 'f8', 1)[0]
        SF = np.fromfile(f,'f8',2*nStage)
    print(f'nStage = {nStage}, nOri = {nOri}, nRep = {nRep}, frameRate = {frameRate}, framesPerStatus: {framesPerStatus}, framesToFinish: {framesToFinish}, max_ecc = {max_ecc}')

    if stage == 2 or stage == 3:
        ##### HERE ########
        if stage == 2:
            peakRate = 0.5
            # corresponds to the parameters set in ext_input.py
        if stage == 3:
            peakRate = 1.0
            absentRate = 1.0
        #############
        nStatus = nOri[0] * nRep[0]
        status = np.zeros((nStatus, 6))
        statusFrame = np.zeros(nStatus)
        for i in range(nOri[0]):
            for j in range(nRep[0]-1):
                statusFrame[i*nRep[0]+j] = framesPerStatus[0]
            statusFrame[(i+1)*nRep[0]-1] = framesToFinish[0]

        stageFrame = sum(statusFrame)
        nFrame = stageFrame
        print(np.transpose(statusFrame))
        print(sum(statusFrame))
        if 2 == stage:
            status[:, 4] = peakRate
            status[:, 5] = peakRate
        else:
            if 3 == stage:
                absentSeq = np.arange(nRep[0]-1, nStatus, nRep[0])
                status[:, 4] = peakRate
                status[absentSeq, 4] = absentRate
                status[:, 5] = 1.0
            else:
                warnings.warn('unexpected stage')
        reverse = np.zeros(nStatus)
    else:
        assert(nStage == 2)
        ##### HERE ########
        assert(stage == 5)
        peakRate = np.array([0.5,1.0])
        # corresponds to the parameters set in ext_input.py
        absentRate = 1.0
        nStatus = sum(nOri * nRep)
        status = np.zeros((nStatus, 6))
        statusFrame = np.zeros(nStatus)
        others = np.zeros((nother,nStatus))
        iStatus = 0
        for k in range(nOri.size):
            for i in range(nother):
                others[i, iStatus: iStatus + nOri[k] * nRep[k]] = set_other[i,k]
            iStatus = iStatus + nOri[k] * nRep[k]

        print(others)
        #############
        iFrame = 0
        for k in range(nOri.size):
            for i in range(nOri[k]):
                for j in range(nRep[k]):
                    jFrame = iFrame + i*nRep[k] + j
                    statusFrame[jFrame] = framesPerStatus[k]
                    if k == 2 and np.mod(j,nRep[k]) == 0:
                        status[jFrame, 4] = absentRate
                    else:
                        status[jFrame, 4] = peakRate[k]
                    status[jFrame, 5] = peakRate[k]
                statusFrame[iFrame + (i+1)* nRep[k] - 1] = framesToFinish[k]
            iFrame += nOri[k] * nRep[k]
            if k == 0:
                stageFrame = iFrame
            else:
                nFrame = iFrame
        print(np.transpose(statusFrame))
        print(np.sum(statusFrame))
        reverse = np.zeros(nStatus)
    
    nLGN = nLGN_1D * nLGN_1D
    if doubleOnOff:
        nLGN = nLGN * 2
    
    with open(fLGN_surfaceID, 'wb') as f:
        if doubleOnOff:
            LGN_idx, LGN_idy = np.meshgrid(np.arange(nLGN_1D * 2),np.arange(nLGN_1D))
        else:
            LGN_idx, LGN_idy = np.meshgrid(np.arange(nLGN_1D), np.arange(nLGN_1D))
        
        # tranform the coords into row-first order
        if doubleOnOff:
            np.array([2*nLGN_1D-1, nLGN_1D - 1], dtype = 'u4').tofile(f)
        else:
            np.array([nLGN_1D-1, nLGN_1D - 1], dtype = 'u4').tofile(f)
    
        LGN_idx.astype('i4').tofile(f)
        LGN_idy.astype('i4').tofile(f)

    if doubleOnOff:
        idx = LGN_idx
        idx = np.ceil((idx + 1) / 2) - 1
        LGN_x = (idx.flatten() - nLGN_1D / 2 + 0.5) / nLGN_1D * max_ecc / 0.5
    else:
        LGN_x = (LGN_idx.flatten() - nLGN_1D / 2 + 0.5) / nLGN_1D * max_ecc / 0.5
    
    LGN_y = (LGN_idy.flatten() - nLGN_1D / 2 + 0.5) / nLGN_1D * max_ecc / 0.5
    LGN_vpos0 = np.array([LGN_x[0:nLGN:2], LGN_y[0:nLGN:2]])
    print('LGN_vpos0 (either ON or OFF)')
    print(LGN_vpos0.shape)
    print(LGN_vpos0)
    #for i=1:nLGN_1D
#	for j=1:nLGN_1D
#		index = (i-1)*nLGN_1D + j;
#		fprintf(['(', num2str(LGN_vpos0(index,1),'#1.1f'), ', ', num2str(LGN_vpos0(index,2),'#1.1f'),')']);
#		if j==nLGN_1D
#			fprintf('\n');
#		else
#			fprintf(', ');
#		end
#	end
#end
    
    if retinotopy > 0:
        nLGN_circ = np.sum(np.sum(LGN_vpos0 * LGN_vpos0, axis = 0) <= max_ecc * max_ecc * np.power(1-retinotopy,2))
        nLGNperV1 = int(np.round(nLGN_circ * pCon))*2
    else:
        if squareOrCircle:
            nLGN_square = np.sum(np.max(np.abs(LGN_vpos0), axis = 0) <= max_ecc * radiusRatio)
            nLGNperV1 = int(np.round(nLGN_square * pCon))*2
        else:
            nLGN_circ = np.sum(np.sum(LGN_vpos0 * LGN_vpos0, axis = 0) <= max_ecc * max_ecc * radiusRatio * radiusRatio)
            nLGNperV1 = int(np.round(nLGN_circ * pCon))*2
    
    assert(np.mod(nLGNperV1,2) == 0)
    
    print(f'nLGNperV1 = {nLGNperV1}')
    disLGN = max_ecc*2/nLGN_1D
    input_width = np.zeros((nStage,2), dtype = 'f4')
    print(f'input width crosses')
    for i in range(nOri.size):
        input_width[i,0] = 1/SF[i*0]/2/disLGN
        input_width[i,1] = 1/SF[i*1]/2/disLGN
        print(f'part {i} on: {input_width[i,0]:.1f} LGNs')
        print(f'part {i} off: {input_width[i,1]:.1f} LGNs')

    epick = np.hstack([np.arange(mE) + iblock*blockSize for iblock in range(nblock)])
    ipick = np.hstack([np.arange(mI) + iblock*blockSize + mE for iblock in range(nblock)])

    with open(fV1_allpos, 'wb') as f:
        # not necessarily a ring
        rng = np.random.default_rng(seed+1)
        qdx = rng.permutation(np.arange(nV1))
        if retinotopy != 0:
            V1_pos = np.zeros((2,nV1))
            #candidate_pos = square_pos(2*max_ecc*retinotopy, int(round(nV1/np.pi*4)), [0,0], seed = seed)
            candidate_pos = fast_poisson_disk2d(2*max_ecc*retinotopy, int(round(nblock*mE/np.pi*4)), ratio = 0.76, seed = seed).T - max_ecc*retinotopy
            pdx = np.argpartition(np.linalg.norm(candidate_pos, axis = 0), nblock*mE)
            V1_pos[:,epick] = candidate_pos[:,pdx[:nblock*mE]]

            candidate_pos = fast_poisson_disk2d(2*max_ecc*retinotopy, int(round(nblock*mI/np.pi*4)), ratio = 0.76, seed = seed + 1).T - max_ecc*retinotopy
            pdx = np.argpartition(np.linalg.norm(candidate_pos, axis = 0), nblock*mI)
            V1_pos[:,ipick] = candidate_pos[:,pdx[:nblock*mI]]

            ecc = np.linalg.norm(V1_pos, axis = 0)
            polar = np.arctan2(V1_pos[1,:], V1_pos[0,:])
        else:
            ecc = np.zeros(nV1) + max_ecc
            polar = np.arange(nV1)/nV1*2*np.pi
            V1_pos = np.array([ecc*np.cos(polar), ecc*np.sin(polar)])
        np.array([nblock, blockSize, 2], dtype = 'u4').tofile(f)
        np.array([-max_ecc, 2*max_ecc, -max_ecc, 2*max_ecc], dtype = 'f8').tofile(f)
        V1_pos.astype('f8').tofile(f)
        np.array([-max_ecc, 2*max_ecc, -max_ecc, 2*max_ecc], dtype = 'f8').tofile(f)
        V1_pos.astype('f8').tofile(f)
    
    with open(fV1_vposFn, 'wb') as f:
        np.array([nV1], dtype = 'u4').tofile(f)
        ecc.astype('f8').tofile(f)
        polar.astype('f8').tofile(f)

    with open(fLGN_switch, 'wb') as f:
        np.array([nStatus], dtype = 'u4').tofile(f)
        status.astype('f4').tofile(f)
        statusFrame.astype('u4').tofile(f)
        reverse.astype('i4').tofile(f)
        np.array([nother], dtype = 'u4').tofile(f)
        if stage == 5:
            others.astype('f4').tofile(f)
    
    with open(fLGN_vpos, 'wb') as f:
        np.array([nLGN, 0], dtype = 'u4').tofile(f)
        interDis = max_ecc / (nLGN_1D - 1)
        np.array([max_ecc], dtype = 'f4').tofile(f)
        np.array([-max_ecc, 2*max_ecc, -max_ecc, 2*max_ecc], dtype = 'f4').tofile(f)
        # using uniform_retina, thus normalized to the central vision of a single eye
        print([np.min(LGN_x), np.max(LGN_x)])
        print([np.min(LGN_y), np.max(LGN_y)])
        LGN_x.astype('f4').tofile(f)
        LGN_y.astype('f4').tofile(f)

        # see preprocess/RFtype.h
        if parvoMagno == 1:
            LGN_type = default_rng.randint(4, size = nLGN)
        
        if parvoMagno == 2:
            LGN_type = np.zeros(LGN_idx.shape, dtype = 'u4')
            if doubleOnOff:
                LGN_type[:,0:LGN_idx.shape[1]:2] = 4
                LGN_type[:,1:LGN_idx.shape[1]:2] = 5
            else:
                type0 = np.zeros(nLGN_1D)
                type0[0:nLGN_1D:2] = 4
                type0[1:nLGN_1D:2] = 5
                type1 = np.zeros(nLGN_1D)
                type1[0:nLGN_1D:2] = 5
                type1[1:nLGN_1D:2] = 4
                for i in range(nLGN_1D):
                    if np.mod(i,2) == 0:
                        #LGN_type(:,i) = circshift(type, i);
                        LGN_type[i,:] = type0
                    else:
                        LGN_type[i,:] = type1
        
        if parvoMagno == 3:
            LGN_type = np.zeros(nLGN, dtype = 'u4')
            nParvo = default_rng.randint(nLGN)
            nMagno = nLGN - nParvo
            LGN_type[:nParvo] = default_rng.randint(4, size = nParvo)
            LGN_type[nParvo:] = 4 + default_rng.randint(2, size = nMagno)
        
        LGN_type.astype('u4').tofile(f)
        # in polar form
        LGN_ecc = np.sqrt(np.power(LGN_x, 2) + np.power(LGN_y,2))
        
        LGN_polar = np.arctan2(LGN_y,LGN_x)
        
        LGN_polar.astype('f4').tofile(f)
        LGN_ecc.astype('f4').tofile(f)
        np.array([doubleOnOff, nStage, stageFrame, nFrame], dtype = 'i4').tofile(f)
        input_width.tofile(f)
        print(LGN_type)
    
    # not in use, only self block is included
    fNeighborBlock = setup_fdr + 'neighborBlock' + suffix + '.bin'
    # not in use, may use if cortical inhibition is needed
    fV1_delayMat = setup_fdr + 'V1_delayMat' + suffix + '.bin'
    
    fV1_conMat = setup_fdr + 'V1_conMat' + suffix + '.bin'
    
    fV1_vec = setup_fdr + 'V1_vec' + suffix + '.bin'
    
    fV1_gapMat = setup_fdr + 'V1_gapMat' + suffix + '.bin'
    
    fV1_gapVec = setup_fdr + 'V1_gapVec' + suffix + '.bin'
    
    nearNeighborBlock = nblock
    cid = open(fV1_conMat, 'wb')
    did = open(fV1_delayMat, 'wb')
    gid = open(fV1_gapMat, 'wb')
    tmp = np.array([nearNeighborBlock], dtype = 'u4')
    tmp.tofile(cid)
    tmp.tofile(gid)
    tmp.tofile(did)
    gapMat = np.zeros((nblock,nearNeighborBlock,mI,mI))
    nmin_ext = min(mI-1,nmin_ext)
    nmin = min(nmin_ext-2,nmin);
    bmin = min(nmin-2,bmin);
    imin_ids = np.empty((nblock, mI, nmin_ext), dtype = 'u4')
    nGap = np.zeros((nblock,mI))
    gapRand = np.random.default_rng(seed+2)

    with open(fNeighborBlock, 'wb') as f:
        tmp = np.zeros(nblock, dtype = 'u4') + nearNeighborBlock
        tmp.tofile(f) # nNearNeighborBlock
        tmp.tofile(f) # nNeighborBlock
        nBlockId = np.zeros((nblock, nearNeighborBlock), dtype = 'u4')
        for i in range(nblock):
            nBlockId[i,:] = np.arange(nearNeighborBlock)
            nBlockId[i,0] = i
            nBlockId[i,i] = 0
        nBlockId.tofile(f)

    # number of E and I connecitons based on LGN connections
    nI = int(np.min([nLGNperV1 // 4,mI]))
    nE = int(np.min([nLGNperV1,mE]))
    for iblock in range(nblock):
        conMat = np.zeros((nearNeighborBlock,blockSize,blockSize))
        delayMat = np.zeros((nearNeighborBlock,blockSize,blockSize))
        id_i = iblock * blockSize + np.arange(blockSize)
        for iNeighbor in range(nearNeighborBlock):
            #(post, pre)
            conIE = np.zeros((mE,mI))
            for i in range(mI):
                ied = default_rng.choice(mE,nE)
                conIE[ied,i] = ieStrength
            conEI = np.zeros((mI,mE))
            for i in range(mE):
                eid = default_rng.choice(mI,nI)
                conEI[eid,i] = eiStrength
            conMat[iNeighbor, mE:, :mE] = conEI
            conMat[iNeighbor, :mE, mE:] = conIE
            for j in range(blockSize):
                id_j = nBlockId[iblock,iNeighbor] * blockSize + j
                distance = np.linalg.norm(V1_pos[:, id_i].T - V1_pos[:, id_j], axis = -1)
                delayMat[iNeighbor,j,:] = distance

        conMat.astype('f4').tofile(cid)
        delayMat.astype('f4').tofile(did)
        tmpMat = np.vstack(delayMat[:,mE:,mE:])
        ids = np.argpartition(tmpMat, nmin_ext, 0)[:nmin_ext, :].T
        dis = np.array([tmpMat[ids[i,:],i] for i in range(mI)])
        #print(dis.shape)
        arg_imin = np.argsort(dis, axis = -1)
        _imin = np.array([ids[i, arg_imin[i, :]] for i in range(mI)])
        #print(_imin.shape)
        imin_ids[iblock, :, :] = _imin

        #if retino_cross:
        #    ii = 368
        #    if iblock == ii//blockSize:
        #        nid = (nBlockId[iblock,:] * blockSize + np.tile(np.arange(mI)+mE, (nearNeighborBlock,1)).T).T.flatten()
        #        print(imin_ids[iblock, ii%blockSize-mE, :])
        #        print(nid[imin_ids[iblock, ii%blockSize-mE, :]])
        #        print(tmpMat[imin_ids[iblock, ii%blockSize-mE, :], ii%blockSize-mE])
    # set max nearest neighbor
    mmin = np.zeros(nblock*mI, dtype = int) + nmin
    r = np.linalg.norm(V1_pos[:,ipick], axis = 0)
    # those at the boundary have less nearest neighbor
    mmin[r > max_ecc*retinotopy*(1 - np.sqrt(np.pi/(mI*nblock))*0.8)] = bmin
    nedge = np.zeros(mI*nblock, dtype = int)
    # gap junction adjacency list
    connected = np.empty(mI*nblock, dtype = object)
    for i in range(mI*nblock):
        connected[i] = []
    for i_pass in range(nmin_ext):
        for iblock in default_rng.permutation(nblock):
            for i in default_rng.permutation(mI):
                i_idx = iblock*mI + i
                if nedge[i_idx] >= min(i_pass+1, mmin[i_idx]):
                    continue
                _imin_ids = imin_ids[iblock, i, :].copy()
                # index in ipick
                iNeighbors = _imin_ids//mI
                iimin_ids = nBlockId[iblock, iNeighbors]*mI + _imin_ids - (iNeighbors)*mI
                nedge_sorted = np.argsort(nedge[iimin_ids])
                for jdx in _imin_ids[nedge_sorted]:
                    iNeighbor = jdx//mI
                    jblock = nBlockId[iblock,iNeighbor] 
                    if jblock != iblock:
                        jNeighbor = -1
                        for k in range(nearNeighborBlock):
                            if nBlockId[jblock,k] == iblock:
                                jNeighbor = k
                                break
                        assert(jNeighbor != -1)
                    else:
                        jNeighbor = iNeighbor
                    j = jdx - iNeighbor*mI
                    j_idx = jblock*mI + j
                    # preconditions
                    if j_idx == i_idx or gapMat[iblock, iNeighbor, j, i] > 0 or nedge[j_idx] >= min(i_pass+1, mmin[j_idx]):
                        continue
                    # no cluster
                    co_neighbor = [neighbor for neighbor, count in Counter(connected[j_idx] + connected[i_idx]).items() if count > 1]
                    clustered = False
                    for q in range(len(co_neighbor)):
                        n1 = co_neighbor[q]
                        for p in range(q+1, len(co_neighbor)):
                            n2 = co_neighbor[p]
                            if n2 == n1:
                                continue
                            if n2 in connected[n1]:
                                clustered = True
                                break
                        if clustered:
                            break
                    if clustered:
                        continue
                    if gapRand.random() < gapCon:
                        gapMat[iblock, iNeighbor, j, i] = gapStr
                        gapMat[jblock, jNeighbor, i, j] = gapStr
                        nGap[iblock, i] += 1
                        nGap[jblock, j] += 1
                        nedge[i_idx] += 1
                        nedge[j_idx] += 1
                        connected[i_idx].append(j_idx)
                        connected[j_idx].append(i_idx)
                    break
         
    gapMat.astype('f4').tofile(gid)
    print(f'avg. gap junct. per neuron:{np.sum(nedge)/(mI*nblock)}, max: {np.max(nedge)}, min: {np.min(nedge)}')
    cid.close()
    gid.close()
    did.close()
    assert((nGap != 0).all())

    # outside the if-condition for debug/plotting purpose
    if retino_cross:
        ic = 0
        crossed = []
        avail = True
        bool_ipick = np.ones(ipick.size, dtype = bool)
        while ic < nc and bool_ipick.any():
            icross = default_rng.choice(ipick[bool_ipick], 1)[0]
            crossed.append(icross)
            # avoid immediate neighbors
            iblock = icross // blockSize
            i = icross % blockSize - mE
            bool_ipick[iblock*mI+i] = False
            neighbor_id = (nBlockId[iblock,:] * mI + np.tile(np.arange(mI), (nearNeighborBlock,1)).T).T 
            ii = neighbor_id[gapMat[iblock,:,:,i] > 0].flatten()
            bool_ipick[ii] = False
            ic += 1
            
        nc = len(crossed)
        crossed = np.sort(crossed)
        cross = np.zeros((2,nc), dtype = 'u4')
        i = 0
        _ipick = ipick[bool_ipick]
        for ic in crossed:
            d_angle = np.abs(polar[_ipick] - polar[ic])
            pick = d_angle > np.pi
            d_angle[pick] = 2*np.pi - d_angle[pick]
            #ids = _ipick[(np.abs(ecc[_ipick]-ecc[ic]) < 0.05*retinotopy*max_ecc)  & (nGap.flatten()[bool_ipick] > round(gapCon*nmin)) & (_ipick != ic) & (d_angle > np.pi/36)]
            #ids = _ipick[(np.abs(ecc[_ipick]-ecc[ic]) < 0.05*retinotopy*max_ecc)  & (nGap.flatten()[bool_ipick] > round(gapCon*nmin)) & (_ipick != ic)]
            ids = _ipick[(nGap.flatten()[bool_ipick] >= round(gapCon*nmin)) & (_ipick != ic)]
            ids = _ipick[_ipick != ic]
            if len(ids) == 0:
                raise Exception('no suitable cross target')
            target = ids[np.argmin(np.abs(np.linalg.norm(V1_pos[:,ids].T - V1_pos[:,ic], axis = -1) - cross_reldis*retinotopy*max_ecc))]
            cross[0,i] = ic
            cross[1,i] = target
            i += 1

        print(f'retino cross = {cross}')
        with open(crossFn, 'wb') as f:
            np.array([nc], dtype = 'u4').tofile(f)
            cross.tofile(f)

    with open(fV1_vec,'wb') as f:
        np.array([0], dtype = 'i4').tofile(f)
        np.zeros(nV1, dtype = 'u4').tofile(f)

    with open(fV1_gapVec,'wb') as f:
        np.zeros(mI*nblock, dtype = 'u4').tofile(f)

    fConnectome = setup_fdr + 'connectome_cfg' + suffix + '.bin'
    with open(fConnectome,'wb') as f:
        np.array([2, 1, mE, mI+mE], dtype = 'u4').tofile(f)

    fConStats = setup_fdr + 'conStats' + suffix + '.bin'
    with open(fConStats,'wb') as f:
        np.array([2, nV1], dtype = 'u4').tofile(f)
        np.ones(nV1, dtype = 'f4').tofile(f)
        np.zeros((2, nV1), dtype = 'u4').tofile(f)
        np.zeros((2, nV1), dtype = 'u4').tofile(f)
        np.zeros((2, nV1), dtype = 'f4').tofile(f)
        np.array([1, mI*nblock], dtype = 'u4').tofile(f)
        np.zeros(mI*nblock, dtype = 'u4').tofile(f)
        np.zeros(mI*nblock, dtype = 'f4').tofile(f)

    with open(fV1_feature,'wb') as f:
        nFeature = 2
        np.array([nFeature], dtype = 'u4').tofile(f)
        default_rng.random(size=(nV1,nFeature), dtype = 'f4').tofile(f)

    with open(fV1_RFprop,'wb') as f:
        np.array([nV1], dtype = 'u4').tofile(f)
        V1_pos.astype('f4').tofile(f)
        np.ones((4,nV1), dtype = 'f4').tofile(f) # a, phase, sfreq, baRatio

    nOn = np.zeros(nV1)
    nOff = np.zeros(nV1)
    if not (sInput == None):
        idFile = Path(fLGN_V1_ID)
        assert(idFile.exists())
        with open(fLGN_V1_ID, 'rb') as f:
            _nV1 = np.fromfile(f, 'u4', 1)[0]
            idx = np.zeros((_nV1,nLGNperV1), dtype = 'u4')
            for i in range(_nV1):
                _nLGNperV1 = np.fromfile(f, 'u4', 1)[0]
                idx[i,:] = np.fromfile(f, 'u4', _nLGNperV1)

        with open(sInput, 'rb') as f:
            nt, sampleInterval = np.fromfile(f, 'u4', 2)
            nst = int(np.floor(nt / sampleInterval))
            dt_ = np.fromfile(f, 'f4', 1)[0]
            nV1_, max_LGNperV1 = np.fromfile(f, 'u4', 2)
            assert(nV1_ == nV1)
            assert(nLGNperV1 == max_LGNperV1)
            t0 = min(nst, max(1,int(np.floor(nst*(1-t0_perc)))))
            f.seek(-nV1*nLGNperV1*4*t0, 2)
            sLGN = np.fromfile(f, 'f4', nV1 * nLGNperV1).reshape((nLGNperV1, nV1)).T
            thres_sLGN = sLGN[sLGN>0].mean()*relay_thres
            sLGN[sLGN < thres_sLGN] = 0
            # balance populational On-Off
            sLGN_bal = np.zeros(nV1)
            for i in range(nV1):
                _id = idx[i,:]
                onId = np.arange(nLGNperV1)[np.logical_and(np.mod(_id,2)==0, sLGN[i,:] > 0)]
                offId = np.arange(nLGNperV1)[np.logical_and(np.mod(_id,2)==1, sLGN[i,:] > 0)]
                sLGN_bal[i] = np.sum(sLGN[i,onId]) - np.sum(sLGN[i,offId])
            dsLGN_bal = np.mean(sLGN_bal)
            for i in range(nV1):
                _id = idx[i,:]
                onId = np.arange(nLGNperV1)[np.logical_and(np.mod(_id,2)==0, sLGN[i,:] > 0)]
                offId = np.arange(nLGNperV1)[np.logical_and(np.mod(_id,2)==1, sLGN[i,:] > 0)]
                sLGN[i,onId] *= (1 - dsLGN_bal/2/sLGN[i,onId].sum())
                sLGN[i,offId] *= (1 + dsLGN_bal/2/sLGN[i,offId].sum())
                assert((sLGN >= 0).all())

            print(f'relayed sLGN: max = {sLGN[sLGN>0].max()}, mean = {sLGN[sLGN>0].mean()}, min = {sLGN[sLGN>0].min()}')

    else:
        if retinotopy > 0:
            idx = np.zeros((nV1,nLGNperV1), dtype = 'u4')
            _nLGNperV1 = nLGNperV1//2
            for i in range(nV1):
                dis = np.linalg.norm(LGN_vpos0.T - V1_pos[:,i], axis = -1)
                assert(dis.size == nLGN//2)
                _idx = np.argpartition(dis, _nLGNperV1) # either on or off idx partitioned to the first _nLGNperV1 LGN
                id_on = default_rng.choice(_idx[:_nLGNperV1], nLGNperV1//2, replace = False)
                current_id = 0
                for j in range(nLGN_1D):
                    idj = id_on[np.logical_and(id_on >= j*nLGN_1D, id_on < (j+1)*nLGN_1D)]
                    idx[i,current_id:current_id+idj.size] = idj*2
                    current_id += idj.size

                id_off = default_rng.choice(_idx[:_nLGNperV1], nLGNperV1//2, replace = False)
                for j in range(nLGN_1D):
                    idj = id_off[np.logical_and(id_off >= j*nLGN_1D, id_off < (j+1)*nLGN_1D)]
                    idx[i,current_id:current_id+idj.size] = idj*2 + 1
                    current_id += idj.size
                assert(current_id == nLGNperV1)

            if normed:
                sLGN = np.zeros((nV1, nLGNperV1))
                ic = 0
                for i in range(nV1):
                    if retino_cross and (i in crossed):
                        assert(i == cross[0,ic])
                        idx[cross[0,ic],:] = idx[cross[1,ic],:].copy()
                        sLGN[i,:] = np.exp(- (np.power(LGN_x[idx[cross[0,ic],:]] - V1_pos[0,cross[1,ic]], 2) + np.power(LGN_y[idx[cross[0,ic],:]] - V1_pos[1,cross[1,ic]],2)) / (std_ecc*std_ecc))
                        ic += 1
                    else:
                        sLGN[i,:] = np.exp(- (np.power(LGN_x[idx[i,:]] - V1_pos[0,i], 2) + np.power(LGN_y[idx[i,:]] - V1_pos[1,i],2)) / (std_ecc*std_ecc))

            else:
                sLGN = u0 + default_rng.random(size = (nV1, nLGNperV1)) * (u1 - u0)
                if retino_cross:
                    for i in range(cross.shape[-1]):
                        idx[cross[0,i],:] = idx[cross[1,i],:].copy()
                        sLGN[cross[0,i],:] = sLGN[cross[1,i],:].copy()

            with open(fLGN_V1_ID, 'wb') as f:
                np.array([nV1], dtype = 'u4').tofile(f)
                for i in range(nV1):
                    np.array([nLGNperV1], dtype = 'u4').tofile(f)
                    idx[i,:].tofile(f)
        else:
            idx = np.zeros((nV1,nLGNperV1), dtype = 'u4')
            with open(fLGN_V1_ID, 'wb') as f:
                np.array([nV1], dtype = 'u4').tofile(f)
                for i in range(nV1):
                    np.array([nLGNperV1], dtype = 'u4').tofile(f)
                    if doubleOnOff:
                        idi = np.zeros(nLGNperV1, dtype = 'u4')
                        current_id = 0
                        if i == 0 or not same :
                            id_on = randq(nLGN_1D * nLGN_1D, nLGNperV1 // 2, LGN_vpos0, max_ecc * radiusRatio, con_std, squareOrCircle, default_rng)
                        for j in range(nLGN_1D):
                            idj = id_on[np.logical_and(id_on >= j*nLGN_1D, id_on < (j+1)*nLGN_1D)]
                            idi[current_id:current_id + idj.size] = idj * 2
                            current_id += idj.size
                            nOn[i] += idj.size
                        if i == 0 or not same :
                            id_off = randq(nLGN_1D * nLGN_1D, nLGNperV1 // 2, LGN_vpos0, max_ecc * radiusRatio, con_std, squareOrCircle, default_rng)
                        for j in range(nLGN_1D):
                            idj = id_off[np.logical_and(id_off >= j*nLGN_1D, id_off < (j+1)*nLGN_1D)]
                            idi[current_id:current_id + idj.size] = idj * 2 + 1
                            current_id += idj.size
                            nOff[i] += idj.size
                        idx[i,:] = idi
                        assert(current_id == nLGNperV1)
                    else:
                        idx[i,:] = default_rng.choice(nLGN, size = nLGNperV1, replace = False)
                    idx[i,:].tofile(f)
            if normed:
                sLGN = np.zeros((nV1, nLGNperV1))
                for i in range(nV1):
                    sLGN[i,:] = np.exp(- np.power(LGN_ecc[idx[i,:]] / std_ecc, 2))
            else:
                sLGN = u0 + default_rng.random(size = (nV1, nLGNperV1)) * (u1 - u0)
        print('     mean, std')
        print(f'sOn: {np.mean(nOn)}, {np.std(nOn)}')
        print(f'sOff: {np.mean(nOff)}, {np.std(nOff)}')

        for i in range(nV1):
            ss = sLGN[i, :]
            if binary_thres > 0:
                pick = (ss - np.min(ss)) / (np.max(ss) - np.min(ss)) > binary_thres
                ss[pick] = u1
                ss[np.logical_not(pick)] = u0

            _nLGNperV1 = np.sum(sLGN[i,:] > 0)
            sLGN[i, :] = ss / np.sum(ss) * initialConnectionStrength * _nLGNperV1
            assert((sLGN[i, :] > 0).all())
    
    print(f'sLGN: {[np.min(sLGN), np.mean(sLGN), np.max(sLGN)]}')

    with open(fLGN_V1_s, 'wb') as f:
        np.array([nV1, nLGNperV1], dtype = 'u4').tofile(f)
        for i in range(nV1):
            np.array([nLGNperV1], dtype = 'u4').tofile(f)
            sLGN[i, :nLGNperV1].astype('f4').tofile(f)

    fig = plt.figure('sLGN_V1-init', figsize = (7, 3*(nc+1)), dpi = 200)
    if retino_cross:
        j = 0
        clims = np.array([0,1])
        for i, ii in zip(cross[0,:], cross[1,:]):
            s = np.zeros(nLGN)
            s[idx[i, :]] = sLGN[i, :nLGNperV1]
            s = np.reshape(s, (nLGN_1D , nLGN_1D,  2))
            stmp0 = s[:, :, 0]
            local_max = np.amax(np.abs(stmp0))
            stmp = stmp0 / local_max
            ax = fig.add_subplot(nc+1,3,3*j+1)
            ax.set_title(f'neuron {i}-ON')
            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('Reds'))
            ax.plot((V1_pos[0,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, '*b')
            ax.plot((V1_pos[0,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, 'ob')
            ax.set_xticks([]) 
            ax.set_yticks([])

            stmp0 = s[:, :, 1]
            local_max = np.amax(np.abs(stmp0))
            stmp = stmp0 / local_max
            ax = fig.add_subplot(nc+1,3,3*j+2)
            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('Blues'))
            ax.plot((V1_pos[0,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, '*r')
            ax.plot((V1_pos[0,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, 'or')
            ax.set_title(f'neuron {i}-OFF')
            ax.set_xticks([]) 
            ax.set_yticks([])

            stmp0 = s.mean(-1)
            local_max = np.amax(np.abs(stmp0))
            stmp = stmp0 / local_max
            ax = fig.add_subplot(nc+1,3,3*j+3)
            #ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('Greys'))
            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('gray'))
            if i in icolor.keys():
                ax.plot((V1_pos[0,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, 'o', color = icolor[i])
            else:
                ax.plot((V1_pos[0,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, 'ok')
            ax.plot((V1_pos[0,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, 'ok')
            ax.set_title(f'neuron {i}')

            j += 1

        i = 121
        s = np.zeros(nLGN)
        s[idx[i, :]] = sLGN[i, :nLGNperV1]
        s = np.reshape(s, (nLGN_1D , nLGN_1D,  2))
        stmp0 = s.mean(-1)
        local_max = np.amax(np.abs(stmp0))
        stmp = stmp0 / local_max
        ax = fig.add_subplot(nc+1,3,(nc+1)*3)
        ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('Greys'))
        ax.plot((V1_pos[0,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, '*w')
        ax.set_title(f'normal neuron {i}')
    elif not (sInput == None):
        j = 0
        for i in default_rng.integers(nV1, size = nc):
            s = np.zeros(nLGN)
            s[idx[i, :]] = sLGN[i, :nLGNperV1]
            s = np.reshape(s, (nLGN_1D , nLGN_1D,  2))
            stmp0 = s[:, :, 0]
            local_max = np.amax(np.abs(stmp0))
            stmp = stmp0 / local_max
            ax = fig.add_subplot(nc+1,3,3*j+1)
            ax.set_title(f'neuron {i}-ON')
            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('Reds'))
            ax.set_xticks([]) 
            ax.set_yticks([])

            stmp0 = s[:, :, 1]
            local_max = np.amax(np.abs(stmp0))
            stmp = stmp0 / local_max
            ax = fig.add_subplot(nc+1,3,3*j+2)
            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('Blues'))
            ax.set_title(f'neuron {i}-OFF')
            ax.set_xticks([]) 
            ax.set_yticks([])

            stmp0 = s.mean(-1)
            local_max = np.amax(np.abs(stmp0))
            stmp = stmp0 / local_max
            ax = fig.add_subplot(nc+1,3,3*j+3)
            #ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('Greys'))
            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('gray'))
            j += 1

    fig.tight_layout()
    fig.savefig(f'{setup_fdr}sLGN_V1-init-{suffix}-sep.png')
    fig.savefig(f'{setup_fdr}sLGN_V1-init-{suffix}-sep.svg')
            
    fig = plt.figure(figsize = (4, 4), dpi = 150)
    ax = fig.add_subplot(111)
    fs = 12
    ms = 2
    ax.plot(LGN_x, LGN_y, ',k')
    ax.plot(V1_pos[0,epick], V1_pos[1,epick], '*', ms = ms, c='gray', alpha = 1.0, zorder = 1)
    ax.plot(V1_pos[0,ipick], V1_pos[1,ipick], 's', ms = ms, c='gray', alpha = 1.0, zorder = 1)
    plot_nid = False
    handles = []
    ms = 4
    if retino_cross:
        ci = 0
        color = mpl.colors.hsv_to_rgb([0, 1, 0.9])
        orig = mpl.lines.Line2D([0],[0], ls = 'None', marker = 'o', ms = ms, color = 'k', label = 'original')
        misplaced = mpl.lines.Line2D([0],[0], ls ='None', marker = '^', ms = ms, color = color, label = 'misplaced')
        handles.append(orig)
        handles.append(misplaced)
        for i, ii in zip(cross[0,:], cross[1,:]):
            #color = plt.get_cmap('turbo')(0.2 + 0.8*ci/(cross.shape[1]-1))
            color = mpl.colors.hsv_to_rgb([0, 1, 0.9])
            #color = 'g'
            if i in icolor.keys():
                ax.plot(V1_pos[0,i], V1_pos[1,i], 'o', c = icolor[i], ls = None, ms = ms)
            else:
                ax.plot(V1_pos[0,i], V1_pos[1,i], 'o', c = 'k', ls = None, ms = ms)

            ax.plot(V1_pos[0,ii], V1_pos[1,ii], '^', c = color, ls = None, ms = ms)
            ax.plot(V1_pos[0,[i,ii]], V1_pos[1,[i,ii]], '-', c = 'k', lw = 1.25, alpha = 1.0, zorder = 0)
            #ax.text(V1_pos[0,i], V1_pos[1,i], f'{i}', fontsize = 6, c = 'm')
            #ax.text(V1_pos[0,ii], V1_pos[1,ii], f'{ii}', fontsize = 6, c = 'c')
            ci += 1

    edges = 0
    lgap = mpl.lines.Line2D([0,0],[0,0], ls = '-', color= 'gray', label = 'gap junction')
    handles.append(lgap)
    for iblock in range(nblock):
        for i in range(mI):
            i_id = iblock*blockSize + mE + i
            if retino_cross:
                if i_id not in cross and plot_nid:
                    ax.text(V1_pos[0,i_id], V1_pos[1,i_id], f'{i_id}', fontsize = 4, c = 'k')
            for iNeighbor in range(nearNeighborBlock):
                jblock = nBlockId[iblock,iNeighbor]
                for j in range(mI-1,-1,-1):
                    j_id = jblock*blockSize + mE + j
                    if j_id <= i_id:
                        break
                    if gapMat[iblock, iNeighbor, j, i] > 0:
                        if jblock != iblock:
                            for k in range(nearNeighborBlock):
                                if nBlockId[jblock,k] == iblock:
                                    jNeighbor = k
                                    break
                        else:
                            jNeighbor = iNeighbor
                        assert(gapMat[jblock, jNeighbor, i, j] > 0)
                        ax.plot([V1_pos[0,i_id], V1_pos[0,j_id]], [V1_pos[1,i_id], V1_pos[1,j_id]], '-', color = 'gray', lw = 1, alpha = 0.8, zorder = 0)

                        edges += 1
    ax.legend(handles = handles, loc = 'lower center', handlelength = 1, fontsize = fs-1, ncols = 3, bbox_to_anchor = (0.5, 1.0))
    for h in handles:
        h.set(visible = False)
    ax.set_xlim([V1_pos[0,:].min()-0.5,V1_pos[0,:].max()+0.5])
    ax.set_ylim([V1_pos[1,:].min()-0.5,V1_pos[1,:].max()+0.5])
    ax.set_xlabel('V1 center x (deg)', fontsize = fs)
    ax.tick_params(labelsize = fs-1)
    ax.set_ylabel('V1 center y (deg)', fontsize = fs)

    print(f'edges = {edges}')
    print(f'sum of gapMat:{np.sum(gapMat)}')
    ax.set_aspect('equal')
    fig.savefig(setup_fdr + f'V1_positions{suffix}.png')
    fig.savefig(setup_fdr + f'V1_positions{suffix}.svg')

    return

def randq(m, n, pos, r0, stdev, squareOrCircle, rng):
    r2 = np.sum(np.power(pos, 2), axis = 0)
    idx = np.arange(m)
    if squareOrCircle:
        ndm = np.sum(np.max(np.abs(pos), axis = 0) > r0)
        pick = np.max(np.abs(pos), axis = 0) <= r0
    else:
        ndm = np.sum(r2 > r0 * r0)
        pick = r2 <= r0 * r0
    
    id_left = idx[pick]
    if stdev <= 0:
        ids = id_left[rng.choice(m - ndm, size = n, replace = False)]
    else:
        rands = rng.random(m - ndm)
        id0 = np.partition(rands / np.exp(- r2[pick] / (stdev * stdev)), n)
        ids = id_left[id0]
    
    assert(np.all(ids >= 0))
    assert(np.all(ids < m))
    return ids

if __name__ == "__main__":

    if len(sys.argv) < 12:
        print(" inputLearnFF(inputFn, suffix, seed, std_ecc, suffix0, stage, res_fdr, setup_fdr, squareOrCircle, relay, binary_thres, sInput, retino_cross, retinotopy = 0)")
    else:
        inputFn = sys.argv[1]
        suffix = sys.argv[2]
        seed = int(sys.argv[3])
        std_ecc = float(sys.argv[4])
        suffix0 = sys.argv[5]
        stage = int(sys.argv[6])
        res_fdr = sys.argv[7]
        setup_fdr = sys.argv[8]
        squareOrCircle = bool(sys.argv[9])
        if sys.argv[9] == 'True' or sys.argv[9] == 'true' or sys.argv[9] == '1':
            squareOrCircle = True
        else:
            squareOrCircle = False
        if sys.argv[10] == 'True' or sys.argv[10] == 'true' or sys.argv[10] == '1':
            relay = True
        else:
            relay = False
        binary_thres = float(sys.argv[11])
        if relay:
            sInput = sys.argv[12]
        else:
            sInput = None

        if sys.argv[13] == 'True' or sys.argv[13] == 'true' or sys.argv[13] == '1':
            retino_cross = True
        else:
            retino_cross = False

        if len(sys.argv) == 15:
            retinotopy = float(sys.argv[14])
        else:
            retinotopy = 0

        inputLearnFF(inputFn, suffix, seed, std_ecc, suffix0, stage, res_fdr, setup_fdr, squareOrCircle, relay, binary_thres, sInput, retino_cross, retinotopy)
