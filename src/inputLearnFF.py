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

def randq(m, n, pos, r0, stdev, squareOrCircle):
    r2 = np.sum(np.power(pos*pos, 2), axis = 0)
    idx = np.arange(m)
    if squareOrCircle:
        ndm = np.sum(np.max(np.abs(pos), axis = 0) > r0)
        pick = np.max(np.abs(pos), axis = 0) <= r0
    else:
        ndm = np.sum(r2 > r0 * r0)
        pick = r2 <= r0 * r0
    
    id_left = idx[pick]
    if stdev <= 0:
        ids = id_left[np.random.choice(m - ndm, size = n, replace = False)]
    else:
        rands = np.random.rand(m - ndm)
        id0 = np.partition(rands / np.exp(- r2[pick] / (stdev * stdev)), n)
        ids = id_left[id0]
    
    assert(np.all(ids >= 0))
    assert(np.all(ids < m))
    return ids
    
def inputLearnFF(inputFn, suffix, seed, std_ecc, suffix0, stage, res_fdr, setup_fdr, squareOrCircle, relay, binary_thres, sInput): 
    res_fdr = res_fdr + '/'
    setup_fdr = setup_fdr + '/'
    con_std = 0
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
        pCon = 0.9
        nLGN_1D = 16
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
        suffix0 = '_' + suffix0
    
    if not len(suffix)==0 :
        suffix = '_' + suffix
    
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
    
    np.random.seed(seed)
    initialConnectionStrength = 1.0
    
    eiStrength = 0.0
    ieStrength = 0.0
    nblock = 32
    blockSize = 32
    mE = 24
    mI = 8
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
            for j in range(nRep[0]):
                statusFrame[i * nRep[0] + j] = framesPerStatus[0]
            statusFrame[i * nRep[0]] = statusFrame[i * nRep[0]] + framesToFinish[0]

        print(np.transpose(statusFrame))
        print(sum(statusFrame))
        if 2 == stage:
            status[:, 4] = peakRate
            status[:, 5] = peakRate
        else:
            if 3 == stage:
                #rands = rand(1,nStatus);
#absentSeq = rands > 0;
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
                    jFrame = iFrame + (i - 1) * nRep[k] + j
                    statusFrame[jFrame] = framesPerStatus[k]
                    if k == 2 and np.mod(j,nRep[k]) == 0:
                        status[jFrame, 4] = absentRate
                    else:
                        status[jFrame, 4] = peakRate[k]
                    status[jFrame, 5] = peakRate[k]
                statusFrame[iFrame + i * nRep[k]] = statusFrame[iFrame + i * nRep[k]] + framesToFinish[k]
            iFrame = iFrame + nOri[k] * nRep[k]
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
    
    if squareOrCircle:
        nLGN_square = np.sum(np.max(np.abs(LGN_vpos0), axis = 0) <= max_ecc * radiusRatio)
        nLGNperV1 = int(np.round(nLGN_square * 2 * pCon))
    else:
        nLGN_circ = np.sum(np.sum(LGN_vpos0 * LGN_vpos0, axis = 0) <= max_ecc * max_ecc * radiusRatio * radiusRatio)
        nLGNperV1 = np.round(nLGN_circ * 2 * pCon)
    
    if np.mod(nLGNperV1,2) == 1:
        nLGNperV1 = nLGNperV1 + 1
    
    nLGNperV1
    # number of E and I connecitons based on LGN connections
    nI = int(np.min([nLGNperV1 // 4,mI]))
    nE = int(np.min([nLGNperV1,mE]))
    with open(fLGN_V1_ID, 'wb') as f:
    
        np.array([nV1], dtype = 'u4').tofile(f)
        idx = np.zeros((nV1,nLGNperV1), dtype = 'u4')
        for i in range(nV1):
            np.array([nLGNperV1], dtype = 'u4').tofile(f)
            if doubleOnOff:
                idi = np.zeros(nLGNperV1, dtype = 'u4')
                current_id = 0
                if i == 0 or not same :
                    ids_on = randq(nLGN_1D * nLGN_1D, nLGNperV1 // 2, LGN_vpos0, max_ecc * radiusRatio, con_std, squareOrCircle)
                for j in range(nLGN_1D):
                    idj = ids_on[np.logical_and(ids_on >= j*nLGN_1D, ids_on < (j+1)*nLGN_1D)]
                    idi[current_id:current_id + idj.size] = idj * 2
                    current_id = current_id + idj.size
                if i == 0 or not same :
                    ids_off = randq(nLGN_1D * nLGN_1D, nLGNperV1 // 2, LGN_vpos0, max_ecc * radiusRatio, con_std, squareOrCircle)
                for j in range(nLGN_1D):
                    idj = ids_off[np.logical_and(ids_off >= j*nLGN_1D, ids_off < (j+1)*nLGN_1D)]
                    idi[current_id:current_id + idj.size] = idj * 2 + 1
                    current_id = current_id + idj.size
                idx[i,:] = idi
            else:
                idx[i,:] = np.random.choice(nLGN, size = nLGNperV1, replace = False)

            assert(len(np.unique(idx[i,:])) ==  nLGNperV1)
            idx[i,:].tofile(f)
    
    with open(fV1_allpos, 'wb') as f:
        # not necessarily a ring
        loop = np.linspace(0, np.pi, nV1 + 1)
        polar = loop[:nV1]
        ecc = np.zeros(nV1)
        V1_pos = np.array([np.cos(polar)*ecc, np.sin(polar)*ecc])
        print('V1_pos.shape')
        print(V1_pos.shape)
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
            LGN_type = np.random.randint(4, size = nLGN)
        
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
            nParvo = np.random.randint(nLGN)
            nMagno = nLGN - nParvo
            LGN_type[:nParvo] = np.random.randint(4, size = nParvo)
            LGN_type[nParvo:] = 4 + np.random.randint(2, size = nMagno)
        
        LGN_type.astype('u4').tofile(f)
        # in polar form
        LGN_ecc = np.sqrt(np.power(LGN_x, 2) + np.power(LGN_y,2))
        
        LGN_polar = np.arctan2(LGN_y,LGN_x)
        
        LGN_polar.astype('f4').tofile(f)
        LGN_ecc.astype('f4').tofile(f)
        np.array([doubleOnOff], dtype = 'i4').tofile(f)
        print(LGN_type)
    
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
        if normed:
            sLGN = np.zeros((nV1, nLGNperV1))
            for i in range(nV1):
                sLGN[i,:] = np.exp(- np.power(LGN_ecc[idx[i,:]] / std_ecc, 2))
        else:
            sLGN = u0 + np.random.rand(nV1, nLGNperV1) * (u1 - u0)

        for i in range(nV1):
            ss = sLGN[i, :]
            if binary_thres > 0:
                pick = (ss - np.min(ss)) / (np.max(ss) - np.min(ss)) > binary_thres
                ss[pick] = u1
                ss[np.logical_not(pick)] = u0
            sLGN[i, :] = ss / np.sum(ss) * initialConnectionStrength * nLGNperV1

    for i in range(nV1):
        assert((sLGN[i, :] > 0).all())
    
    print(f'sLGN: {[np.min(sLGN), np.mean(sLGN), np.max(sLGN)]}')

    with open(fLGN_V1_s, 'wb') as f:
        np.array([nV1, nLGNperV1], dtype = 'u4').tofile(f)
        for i in range(nV1):
            np.array([nLGNperV1], dtype = 'u4').tofile(f)
            sLGN[i, :nLGNperV1].astype('f4').tofile(f)
            if i == 1:
                fig = plt.figure('sLGN_V1-init', figsize = (3,7))
                s = np.zeros(nLGN)
                s[idx[i, :]] = sLGN[i, :nLGNperV1]
                s = np.reshape(s, (nLGN_1D , nLGN_1D * 2))
                clims = np.array([0,1])
                for itype in range(2):
                    stmp0 = s[:, itype:2*nLGN_1D:2]
                    local_max = np.amax(np.abs(stmp0))
                    stmp = stmp0 / local_max
                    ax = fig.add_subplot(2,1,itype+1)

                    ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('gray'))
                    ax.set_xticks([]) 
                    ax.set_yticks([])
                fig.savefig(f'{setup_fdr}sLGN_V1-init-{i}{suffix}-sep.png')
        
    # by setting useNewLGN to false, populate LGN.bin follow the format in patch.cu
    
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
    for iblock in range(nblock):
        conMat = np.zeros((nearNeighborBlock,blockSize,blockSize))
        delayMat = np.zeros((nearNeighborBlock,blockSize,blockSize))
        for iNeighbor in range(nearNeighborBlock):
            #(post, pre)
            conIE = np.zeros((mE,mI))
            for i in range(mI):
                ied = np.random.choice(mE,nE)
                conIE[ied,i] = ieStrength
            conEI = np.zeros((mI,mE))
            for i in range(mE):
                eid = np.random.choice(mI,nI)
                conEI[eid,i] = eiStrength
            conMat[iNeighbor, mE:, :mE] = conEI
            conMat[iNeighbor, :mE, mE:] = conIE
            # not necessary to calc distance
            for i in range(blockSize):
                for j in range(blockSize):
                    id_i = iblock * blockSize + i
                    id_j = iNeighbor * blockSize + j
                    delayMat[i,j] = np.sqrt(np.sum(np.power(V1_pos[:, id_i] - V1_pos[:, id_j],2)))
                    delayMat[j,i] = delayMat[i,j]
        conMat.astype('f4').tofile(cid)
        gapMat = np.zeros((mI,mI,nearNeighborBlock))
        gapMat.astype('f4').tofile(gid)
        delayMat.astype('f4').tofile(did)
    
    cid.close()
    gid.close()
    did.close()

    with open(fNeighborBlock, 'wb') as f:
        tmp = np.zeros(nblock, dtype = 'u4') + nearNeighborBlock
        tmp.tofile(f) # nNearNeighborBlock
        tmp.tofile(f) # nNeighborBlock
        nBlockId = np.zeros((nblock, nearNeighborBlock,nblock), dtype = 'u4')
        for i in range(nblock):
            nBlockId[i,:] = np.arange(nearNeighborBlock)
        nBlockId.tofile(f)

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
        np.random.rand(nV1,nFeature).astype('f4').tofile(f)

    with open(fV1_RFprop,'wb') as f:
        np.array([nV1], dtype = 'u4').tofile(f)
        V1_pos.astype('f4').tofile(f)
        np.ones((4,nV1), dtype = 'f4').tofile(f) # a, phase, sfreq, baRatio
    return

if __name__ == "__main__":

    if len(sys.argv) < 12:
        print(" inputLearnFF(inputFn, suffix, seed, std_ecc, suffix0, stage, res_fdr, setup_fdr, squareOrCircle, sInput, relay, binary_thres)")
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

    inputLearnFF(inputFn, suffix, seed, std_ecc, suffix0, stage, res_fdr, setup_fdr, squareOrCircle, relay, binary_thres, sInput)
