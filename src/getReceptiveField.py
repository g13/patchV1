import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
import sys
from readPatchOutput import *

def getReceptiveField(spFn, inputFn, frameRate, output_fdr, output_suffix, sampleList):

    nFrame, width, height, frames = read_input_frames(inputFn) 
    np.load(spFn, allow_pickle=True) as data:
        spName = data['spName']
        spikeInfo = data[spName]

    n = len(spikeInfo)
    print(f'n = {n}')
if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise Exception('not enough argument for getReceptiveField.py output_suffix, output_fdr, LGN_or_V1, readNewSpike')
        if len(sys.argv < 5):
            readNewSpike = False

    _output_suffix = '_' + out_suffix
    LGN_frFn = "LGN_fr" + _output_suffix + ".bin"
    LGN_V1_sFn = "LGN_V1_sList" + conLGN_suffix + ".bin"
    LGN_V1_idFn = "LGN_V1_idList" + conLGN_suffix + ".bin"
    LGN_vposFn = "parvo_float-micro.bin"
    rawDataFn = "rawData" + _output_suffix + ".bin"

    parameterFn = "patchV1_cfg" +_output_suffix + ".bin"
    spDataFn = "V1_spikes" + _output_suffix
    LGN_spFn = "LGN_sp" + _output_suffix

    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, nLGN, nV1 = read_cfg(parameterFn, True)


    if LGN_or_V1 == 0:
        if readNewSpike:
            print('reading LGN connections...')
            LGN_V1_s = readLGN_V1_s0(LGN_V1_sFn)
            print('     ID...')
            LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)
            print('     vpos...')
            nLGN_I, nLGN_C, nLGN, max_ecc, vCoordSpan, LGN_vpos, LGN_type = readLGN_vpos(LGN_vposFn)
            print('     fr...')
            LGN_fr = readLGN_fr(LGN_frFn, prec = prec)
            print('     spikes...')
            LGN_spScatter = readLGN_sp(LGN_spFn + ".bin", prec = prec)
            np.savez(LGN_spFn + '.npz', spName = 'LGN_spScatter', LGN_spScatter = LGN_spScatter, LGN_V1_s = LGN_V1_s, LGN_V1_ID = LGN_V1_ID, nLGN_V1 = nLGN_V1, LGN_vpos = LGN_vpos, LGN_type = LGN_type, LGN_fr = LGN_fr)
            print('complete.')
        spFn = LGN_spFn + '.npz'

    if LGN_or_V1 == 1:
        if readNewSpike:
            readSpike(rawDataFn, spDataFn, prec, sizeofPrec, max_vThres)
        spFn = spDataFn + '.npz'

    getReceptiveField(spFn, inputFn, frameRate, output_fdr, output_suffix, sampleList)
