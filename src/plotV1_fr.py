#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from readPatchOutput import *
np.seterr(invalid = 'raise')

def plotV1_fr(output_suffix0, data_fdr, fig_fdr, nOri, readNewSpike, ns):
    sample = np.array([290,311,600,685,108,134,744]);
    step0 = 0
    nt_ = 16000
    fr_window = 1000 #ms
    nit = 32

    if nOri > 0:
        output_suffix = output_suffix0 + '_' + str(iOri)
    else:
        output_suffix = output_suffix0
    _output_suffix = "_" + output_suffix

    if data_fdr[-1] != "/":
        data_fdr = data_fdr+"/"
    if fig_fdr[-1] != "/":
        fig_fdr = fig_fdr+"/"

    rawDataFn = data_fdr + "rawData" + _output_suffix + ".bin"
    spDataFn = data_fdr + "V1_spikes" + _output_suffix
    parameterFn = data_fdr + "patchV1_cfg" +_output_suffix + ".bin"

    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, nLGN, nV1, nt, dt, normViewDistance, L_x0, L_y0, R_x0, R_y0, virtual_LGN, _, _ = read_cfg(parameterFn, True)

    stepInterval = int(round(fr_window/dt))
    if step0 >= nt:
        step0 = 0
        print('step0 is too large, set back to 0.')
    if nt_ == 0 or step0 + nt_ >= nt:
        nt_ = nt - step0
    if stepInterval > nt_:
        stepInterval = nt_
    it = np.append(np.arange(step0,nt_,stepInterval), nt_)

    stepInterval0 = int(round(nt/nit))
    it0 = np.append(np.arange(0,nt,stepInterval0), nt)

    if not readNewSpike:
        with np.load(spDataFn + '.npz', allow_pickle=True) as data:
            spScatter = data['spScatter']
    else:
        spScatter = readSpike(rawDataFn, spDataFn, prec, sizeofPrec, vThres)

    if 'sample' not in locals():
        sample = np.random.randint(nV1, size = ns)
    for i in sample:
        fig = plt.figure(f'V1-fr-{i}', dpi = 300, figsize = [6,4])
        ax = fig.add_subplot(211)
        fr = np.array([sum(np.logical_and(spScatter[i]>=it[j]*dt, spScatter[i]<it[j+1]*dt)) for j in range(it.size-1)])/(stepInterval*dt)*1000
        ax.plot(it[:-1]*dt, fr, '-k', lw = 0.5)
        ax.set_xlabel('sample range (ms)')
        ax.set_ylabel('firing rate')

        ax = fig.add_subplot(212)
        fr = np.array([sum(np.logical_and(spScatter[i]>=it0[j]*dt, spScatter[i]<it0[j+1]*dt)) for j in range(it0.size-1)])/(stepInterval0*dt)*1000
        ax.plot(it0[:-1]*dt/1000, fr, '-k', lw = 0.5)
        ax.set_xlabel('full time (s)')

        fig.savefig(fig_fdr+output_suffix + f'V1-fr-{i}.png')
        plt.close(fig)

if __name__ == "__main__":
    print(sys.argv)
    print(len(sys.argv))
    if len(sys.argv) < 6:
        raise Exception('not enough argument for plotV1_fr(output_suffix0, data_fdr, fig_fdr, nOri, readNewSpike, ns)')
    else:
        output_suffix0 = sys.argv[1]
        print(output_suffix0)
        data_fdr = sys.argv[2]
        print(data_fdr)
        fig_fdr = sys.argv[3]
        print(fig_fdr)
        nOri = int(sys.argv[4])
        if sys.argv[5] == 'True' or sys.argv[5] == '1':
            readNewSpike = True 
            print('read new spikes')
        else:
            readNewSpike = False
            print('read stored spikes')
        if len(sys.argv) < 7:
            ns = 8
        else:
            ns = int(sys.argv[6])
        print(f'ns = {ns}')

    plotV1_fr(output_suffix0, data_fdr, fig_fdr, nOri, readNewSpike, ns)
