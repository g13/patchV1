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
from patch_square import *
from scipy.stats import qmc

def inputLearnFF(inputFn, suffix, seed, std_ecc, suffix0, stage, res_fdr, setup_fdr, squareOrCircle, relay, binary_thres, sInput, retinotopy = 0): 
    retinotopy = 0.35 # V1 visual field ratio of LGN pool size in 1D
    nc = 2
    retino_cross = True
    cross_reldis = 0.5
    res_fdr = res_fdr + '/'
    setup_fdr = setup_fdr + '/'
    con_std = 0
    gapCon = 0.1
    nmin = 6

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
    gapStr = 1
    
    if stage == 2 or stage == 5:
        pCon = 1.0
        nLGN_1D = 24
        radiusRatio = 1.0
    
    if stage == 3:
        if relay:
            #pCon = 0.8 # initial sparsity
            pCon = 0.8
            nLGN_1D = 16
            radiusRatio = 0.5
        else:
            pCon = 0.8
            nLGN_1D = 8
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
    nblock = 32
    blockSize = 32
    mE = 8
    mI = 24
    #nblock = 1;
    #blockSize = 1024;
    #mE = 768;
    #mI = 256;
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
            LGN_idx,LGN_idy = np.meshgrid(np.arange(nLGN_1D * 2),np.arange(nLGN_1D))
        else:
            LGN_idx,LGN_idy = np.meshgrid(np.arange(nLGN_1D),np.arange(nLGN_1D))
        
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
    print('LGN_vpos0')
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
        nLGNperV1 = int(np.round(nLGN_circ * 2 * pCon))
    else:
        if squareOrCircle:
            nLGN_square = np.sum(np.max(np.abs(LGN_vpos0), axis = 0) <= max_ecc * radiusRatio)
            nLGNperV1 = int(np.round(nLGN_square * 2 * pCon))
        else:
            nLGN_circ = np.sum(np.sum(LGN_vpos0 * LGN_vpos0, axis = 0) <= max_ecc * max_ecc * radiusRatio * radiusRatio)
            nLGNperV1 = int(np.round(nLGN_circ * 2 * pCon))
    
    if np.mod(nLGNperV1,2) == 1:
        nLGNperV1 = nLGNperV1 + 1
    
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
            candidate_pos = square_pos(2*max_ecc*retinotopy, int(round(nV1/np.pi*4)), [0,0], seed = seed)
            pdx = np.argpartition(np.linalg.norm(candidate_pos, axis = 0), nV1)
            V1_pos = np.zeros((2,nV1))
            V1_pos[:,qdx] = candidate_pos[:,pdx[:nV1]]
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
    imin_ids = np.empty((nblock, mI, nmin+1), dtype = 'u4')
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
            for i in range(blockSize):
                id_i = iblock * blockSize + i
                for j in range(blockSize):
                    id_j = nBlockId[iblock,iNeighbor] * blockSize + j
                    distance = np.linalg.norm(V1_pos[:, id_i] - V1_pos[:, id_j])
                    delayMat[iNeighbor,i,j] = distance

        conMat.astype('f4').tofile(cid)
        delayMat.astype('f4').tofile(did)
        tmpMat = np.hstack(delayMat[:,mE:,mE:])
        imin_ids[iblock, :, :] = np.argpartition(tmpMat, nmin+1, -1)[:,:nmin+1]

    edges = 0
    for iblock in range(nblock):
        for i in range(mI):
            i_idx = iblock*mI + i
            for jdx in imin_ids[iblock, i, :]:
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
                if i == j:
                    continue

                if j_idx < i_idx and jNeighbor*mI + i in imin_ids[jblock, j, :]:
                    continue
                assert(gapMat[jblock, jNeighbor, j, i] == 0 and gapMat[iblock, iNeighbor, i, j] == 0)
                if gapRand.random() < gapCon:
                    gapMat[iblock, iNeighbor, i, j] = gapStr
                    gapMat[jblock, jNeighbor, j, i] = gapStr
                    nGap[iblock, i] += 1
                    nGap[jblock, j] += 1
                    edges += 1
         
    for iblock in range(nblock):
        for i in range(mI):
            if nGap[iblock, i] == 0:
                min_gap = nmin + 1
                for jdx in imin_ids[iblock, i, :]:
                    iNeighbor = jdx//mI
                    jblock = nBlockId[iblock,iNeighbor] 
                    j = jdx - iNeighbor*mI
                    if i == j:
                        continue
                    if nGap[jblock,j] < min_gap:
                        min_jdx = jdx
                        min_gap = nGap[jblock,j]

                assert(min_gap < nmin + 1)
                jdx = min_jdx
                iNeighbor = jdx//mI
                jblock = nBlockId[iblock,iNeighbor] 
                j = jdx - iNeighbor*mI
                if jblock != iblock:
                    jNeighbor = -1
                    for k in range(nearNeighborBlock):
                        if nBlockId[jblock,k] == iblock:
                            jNeighbor = k
                            break
                    assert(jNeighbor != -1)
                else:
                    jNeighbor = iNeighbor

                assert(gapMat[iblock, iNeighbor, i, j] == 0 and gapMat[jblock, jNeighbor, j, i] ==0)
                gapMat[iblock, iNeighbor, i, j] = gapStr
                gapMat[jblock, jNeighbor, j, i] = gapStr

                nGap[iblock, i] += 1
                nGap[jblock, j] += 1
                edges += 1

    gapMat.astype('f4').tofile(gid)
    print(f'gap junctions:{edges}')
    print(f'sum of gapMat:{np.sum(gapMat)}')
    cid.close()
    gid.close()
    did.close()
    assert((nGap != 0).all())

    # outside the if-condition for debug/plotting purpose
    if retino_cross:
        crossed = np.sort(default_rng.choice(ipick, nc, replace = False))
        cross = np.zeros((nc,2), dtype = 'u4')
        i = 0
        for ic in crossed:
            d_angle = np.abs(polar[ipick] - polar[ic])
            pick = d_angle > np.pi
            d_angle[pick] = 2*np.pi - d_angle[pick]
            ids = ipick[(np.abs(ecc[ipick]-ecc[ic]) < 0.1*retinotopy*max_ecc)  & (nGap.flatten() > round(gapCon*nmin)) & (ipick != ic) & (d_angle > np.pi/18)]
            if len(ids) == 0:
                raise Exception('no suitable cross target')
            target = ids[np.argmin(np.abs(np.linalg.norm(V1_pos[:,ids].T - V1_pos[:,ic], axis = -1) - cross_reldis*retinotopy*max_ecc))]
            cross[0,i] = ic
            cross[1,i] = target
            i += 1
        print(f'retino cross = {cross}')

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

    if not (sInput == None):
        with open(sInput, 'rb') as f:
            nt, sampleInterval = np.fromfile(f, 'u4', 2)
            nst = int(np.floor(nt / sampleInterval))
            dt_ = np.fromfile(f, 'f4', 1)[0]
            nV1_, max_LGNperV1 = np.fromfile(f, 'u4', 2)
            assert(nV1_ == nV1)
            if max_LGNperV1 <= nLGNperV1:
                print(max_LGNperV1)
                print(nLGNperV1)
                assert(nLGNperV1 == max_LGNperV1)
                f.seek(4, 1)
                nLearnFF = np.fromfile(f, 'u4', 1)[0]
                f.seek(nLearnFF*4 + 1, 1)
                sLGN = np.fromfile(f, 'f4', nV1_ * max_LGNperV1).reshape((max_LGNperV1, nV1)).T
    else:
        if retinotopy > 0:
            idx = np.zeros((nV1,nLGNperV1), dtype = 'u4')
            _nLGNperV1 = int(round((nLGNperV1//2)/pCon))
            for i in range(nV1):
                dis = np.linalg.norm(LGN_vpos0.T - V1_pos[:,i], axis = -1)
                assert(dis.size == nLGN//2)
                _idx = np.argpartition(dis, _nLGNperV1)
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
                    if i in crossed:
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
                        if i == 0 or not same :
                            id_off = randq(nLGN_1D * nLGN_1D, nLGNperV1 // 2, LGN_vpos0, max_ecc * radiusRatio, con_std, squareOrCircle, default_rng)
                        for j in range(nLGN_1D):
                            idj = id_off[np.logical_and(id_off >= j*nLGN_1D, id_off < (j+1)*nLGN_1D)]
                            idi[current_id:current_id + idj.size] = idj * 2 + 1
                            current_id += idj.size
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

        for i in range(nV1):
            ss = sLGN[i, :]
            if binary_thres > 0:
                pick = (ss - np.min(ss)) / (np.max(ss) - np.min(ss)) > binary_thres
                ss[pick] = u1
                ss[np.logical_not(pick)] = u0

            _nLGNperV1 = np.sum(sLGN[i,:] > 0)
            sLGN[i, :] = ss / np.sum(ss) * initialConnectionStrength * _nLGNperV1
            assert((sLGN[i, :] > 0).all())
    
    print(np.sum(sLGN[74,:] > 0))
    print(f'sLGN: {[np.min(sLGN), np.mean(sLGN), np.max(sLGN)]}')

    with open(fLGN_V1_s, 'wb') as f:
        np.array([nV1, nLGNperV1], dtype = 'u4').tofile(f)
        for i in range(nV1):
            np.array([nLGNperV1], dtype = 'u4').tofile(f)
            sLGN[i, :nLGNperV1].astype('f4').tofile(f)

    fig = plt.figure('sLGN_V1-init', figsize = (7,6))
    if retino_cross:
        j = 0
        for i, ii in zip(cross[0,:], cross[1,:]):
            s = np.zeros(nLGN)
            s[idx[i, :]] = sLGN[i, :nLGNperV1]
            s = np.reshape(s, (nLGN_1D , nLGN_1D * 2))
            clims = np.array([0,1])
            stmp0 = s[:, 0:2*nLGN_1D:2]
            local_max = np.amax(np.abs(stmp0))
            stmp = stmp0 / local_max
            ax = fig.add_subplot(2,2,2*j+1)
            ax.set_title(f'neuron {i}-ON')
            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('Reds'))
            ax.plot((V1_pos[0,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, '*r')
            ax.plot((V1_pos[0,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, 'or')
            ax.set_xticks([]) 
            ax.set_yticks([])

            stmp0 = s[:, 1:2*nLGN_1D:2]
            local_max = np.amax(np.abs(stmp0))
            stmp = stmp0 / local_max
            ax = fig.add_subplot(2,2,2*j+2)
            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('Blues'))
            ax.plot((V1_pos[0,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,i]+max_ecc)/max_ecc/2*nLGN_1D-0.5, '*b')
            ax.plot((V1_pos[0,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, (V1_pos[1,ii]+max_ecc)/max_ecc/2*nLGN_1D-0.5, 'ob')
            ax.set_title(f'neuron {i}-OFF')
            ax.set_xticks([]) 
            ax.set_yticks([])
            j += 1
        fig.savefig(f'{setup_fdr}sLGN_V1-init-{cross[0]}-{cross[1]}{suffix}-sep.png')
            
    fig = plt.figure(figsize = (8,8), dpi = 500)
    ax = fig.add_subplot(111)
    ax.plot(LGN_x, LGN_y, ',k')
    ax.plot(V1_pos[0,epick], V1_pos[1,epick], '*', ms = 1, c='r', alpha = 0.6)
    ax.plot(V1_pos[0,ipick], V1_pos[1,ipick], 's', ms = 1, c='b', alpha = 0.6)
    if retino_cross:
        for i, ii in zip(cross[0,:], cross[1,:]):
            ax.plot(V1_pos[0,i], V1_pos[1,i], 'ok', ls = None, ms = 3)
            ax.plot(V1_pos[0,ii], V1_pos[1,ii], '^k', ls = None, ms = 3)
            ax.plot(V1_pos[0,[i,ii]], V1_pos[1,[i,ii]], ':k', lw = 0.1)
            ax.text(V1_pos[0,i], V1_pos[1,i], f'{i}', fontsize = 4, c = 'm')
            ax.text(V1_pos[0,ii], V1_pos[1,ii], f'{ii}', fontsize = 4, c = 'c')

    edges = 0
    for iblock in range(nblock):
        for i in range(mI):
            i_id = iblock*blockSize + mE + i
            if retino_cross:
                if i_id not in cross:
                    ax.text(V1_pos[0,i_id], V1_pos[1,i_id], f'{i_id}', fontsize = 4, c = 'k')
            for iNeighbor in range(nearNeighborBlock):
                jblock = nBlockId[iblock,iNeighbor]
                for j in range(mI-1,-1,-1):
                    j_id = jblock*blockSize + mE + j
                    if j_id <= i_id:
                        break
                    if gapMat[iblock, iNeighbor, i, j] > 0:
                        if jblock != iblock:
                            for k in range(nearNeighborBlock):
                                if nBlockId[jblock,k] == iblock:
                                    jNeighbor = k
                                    break
                        else:
                            jNeighbor = iNeighbor
                        assert(gapMat[jblock, jNeighbor, j, i] > 0)
                        ax.plot([V1_pos[0,i_id], V1_pos[0,j_id]], [V1_pos[1,i_id], V1_pos[1,j_id]], '-b', lw = 1, alpha = 0.6)

                        edges += 1

    ax.set_xlim([V1_pos[0,:].min()-0.5,V1_pos[0,:].max()+0.5])
    ax.set_ylim([V1_pos[1,:].min()-0.5,V1_pos[1,:].max()+0.5])

    print(f'edges = {edges}')
    print(f'sum of gapMat:{np.sum(gapMat)}')
    ax.set_aspect('equal')
    fig.savefig(setup_fdr + f'V1_positions{suffix}.png')

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
        print(" inputLearnFF(inputFn, suffix, seed, std_ecc, suffix0, stage, res_fdr, setup_fdr, squareOrCircle, relay, binary_thres, sInput, retinotopy = 0)")
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
        if len(sys.argv) == 14:
            retinotopy = sys.argv[13]
        else:
            retinotopy = 0

        inputLearnFF(inputFn, suffix, seed, std_ecc, suffix0, stage, res_fdr, setup_fdr, squareOrCircle, relay, binary_thres, sInput, retinotopy)
