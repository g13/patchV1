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
    nt_ = 0
    fr_window = 125 #ms
    stepInterval = int(round(fr_window/dt))
    if step0 >= nt:
        step0 = 0
        print('step0 is too large, set back to 0.')
    if nt_ == 0 or step0 + nt_ >= nt:
        nt_ = nt - step0
    if stepInterval > nt_:
        stepInterval = nt_
    it = np.append(np.arange(step0,nt_,stepInterval), nt_)

    if nOri > 0:
        output_suffix = output_suffix0 + '_' + str(iOri)
    else:
        output_suffix = output_suffix0
    _output_suffix = "_" + output_suffix

    rawDataFn = data_fdr + "rawData" + _output_suffix + ".bin"
    spDataFn = data_fdr + "V1_spikes" + _output_suffix


    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, nLGN, nV1, nt, dt, normViewDistance, L_x0, L_y0, R_x0, R_y0, virtual_LGN = read_cfg(parameterFn, True)

    if not readNewSpike:
        with np.load(spDataFn + '.npz', allow_pickle=True) as data:
            spScatter = data['spScatter']
    else:
        spScatter = readSpike(rawDataFn, spDataFn, prec, sizeofPrec, vThres)

    if 'sample' not in locals():
        sample = np.random.randint(nV1, size = ns)
    for i in sample:
        fig = plt.figure(f'V1-fr-{i}', dpi = 600, figsize = [4,2])
        ax = sfig.add_subplot(111)
        fr = np.array([sum(np.logical_and(spScatter[i]>=it[j]*dt, x<it[j+1]*dt)) for j in range(t.size-1)])/(stepInterval*dt)*1000
        ax.plot(t[:-1], fr, '-k', lw = 0.5)
        fig.savefig(fig_fdr+output_suffix + f'V1-fr-{i}.png')
        plt.close(fig)

if __name__ == "__main__":

    if len(sys.argv) < 5:
        raise Exception('not enough argument for plotV1_fr(output_suffix0, data_fdr, fig_fdr, nOri, readNewSpike, ns)')
    else:
        output_suffix = sys.argv[1]
        print(output_suffix)
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
        if len(sys.argv) < 6:
            ns = 8
        else:
            ns = int(sys.argv[6])
        print(f'ns = {ns}')

    plotV1_fr(output_suffix0, data_fdr, fig_fdr, nOri, readNewSpike, ns)
