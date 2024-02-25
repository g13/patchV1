#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from readPatchOutput import *
from plotV1_response import movingAvg
#np.seterr(invalid = 'raise')

def plotV1_fr(output_suffix0, res_fdr, data_fdr, fig_fdr, inputFn, nOri, readNewSpike, ns):
    sample = np.arange(10)
    sample_nOri = 10
    sample_nOri0 = 0
    sample_cutoff = 1.0
    low = 10
    high = 100-low
    fr_bin = 20 #ms
    fr_window = 50 # unit: fr_bin
    tauLTP = 17
    tauTrip = 114
    tauLTD = 34
    LTP_window = tauTrip #ms
    LTD_window = tauLTD #ms
    plot_popFR = False

    if nOri > 0:
        output_suffix = output_suffix0 + '_' + str(iOri)
    else:
        output_suffix = output_suffix0
    _output_suffix = "-" + output_suffix

    if data_fdr[-1] != "/":
        data_fdr = data_fdr+"/"
    if fig_fdr[-1] != "/":
        fig_fdr = fig_fdr+"/"

    rawDataFn = data_fdr + "rawData" + _output_suffix + ".bin"
    spDataFn = data_fdr + "V1_spikes" + _output_suffix
    parameterFn = data_fdr + "patchV1_cfg" +_output_suffix + ".bin"
    LGN_spFn = data_fdr + "LGN_sp" + _output_suffix + ".bin"

    with open(res_fdr + '/' + inputFn + '.cfg','rb') as f:
        nStage = np.fromfile(f, 'u4', 1)[0]
        nOri = np.fromfile(f, 'u4', nStage)
        nRep = np.fromfile(f, 'u4', nStage)
        frameRate = np.fromfile(f, 'f8', 1)[0]
        framesPerStatus = np.fromfile(f, 'u4', nStage)[0]
        framesToFinish = np.fromfile(f, 'u4', nStage)[0]


    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, nLGN, nV1, nt, dt, normViewDistance, L_x0, L_y0, R_x0, R_y0, virtual_LGN, _, _ = read_cfg(parameterFn, True)

    blockSize = typeAcc[-1]
    nblock = nV1//blockSize
    epick = np.hstack([np.arange(nE) + iblock*blockSize for iblock in range(nblock)])
    ipick = np.hstack([np.arange(nI) + iblock*blockSize + nE for iblock in range(nblock)])

    stepInterval0 = int((1000/frameRate*framesToFinish)/dt)
    if stepInterval0 > nt:
        stepInterval0 = 1
        nt_ = nt
        stepInterval = 1
        print(f'runtime too short')
    else:
        print(f'number timestep per sweep to average firing rate over  {stepInterval0}')
        print(f'frames per status = {framesPerStatus}, frames to finish = {framesToFinish}')
        stepInterval = int(round(fr_bin/dt))
        nt_ = stepInterval0 * sample_nOri
        if nt_ > nt:
            nt_ = nt
            sample_nOri = nt_//stepInterval0

    it = np.arange(0, nt_, stepInterval) + sample_nOri0*stepInterval0
    it0 = np.arange(0, int(round(nt*sample_cutoff)), stepInterval0)

    if not readNewSpike:
        with np.load(spDataFn + '.npz', allow_pickle=True) as data:
            spScatter = data['spScatter']
    else:
        spScatter = readSpike(rawDataFn, spDataFn, prec, sizeofPrec, vThres)

    LGN_spScatter = readLGN_sp(LGN_spFn, prec = prec)
    nLGN = len(LGN_spScatter)

    if 'sample' not in locals():
        sample = np.random.randint(nV1, size = ns)
    for i in sample:
        fig = plt.figure(f'V1-fr-{i}', dpi = 300, figsize = [16,8])
        ax = fig.add_subplot(211)
        fr = np.array([sum(np.logical_and(spScatter[i]>=it[j]*dt, spScatter[i]<it[j+1]*dt)) for j in range(it.size-1)], dtype = float)/(stepInterval*dt)*1000
        smoothed_fr = movingAvg(fr, it.size-1, fr_window)
        ax.set_title(f'max fr = {fr.max():.1f} Hz')
        ax.plot(it[:-1]*dt/1000, smoothed_fr, '-k', lw = 0.5)
        ax.plot(it[:-1]*dt/1000, smoothed_fr, ',r')
        ax.set_xlabel('sample range (s)')
        ax.set_ylabel('firing rate')

        ax = fig.add_subplot(212)
        fr = np.array([sum(np.logical_and(spScatter[i]>=it0[j]*dt, spScatter[i]<it0[j+1]*dt)) for j in range(it0.size-1)], dtype = float)/(stepInterval0*dt)*1000
        smoothed_fr = movingAvg(fr, it0.size-1, max(sample_nOri,1))
        ax.set_title(f'max fr = {fr.max():.1f} Hz')
        ax.plot(it0[:-1]*dt/1000, smoothed_fr, '-k', lw = 0.5)
        ax.set_xlabel('full time (s)')

        fig.savefig(fig_fdr+output_suffix + f'V1-fr-{i}.png', bbox_inches='tight')
        plt.close(fig)

    if plot_popFR:
        fig = plt.figure(f'V1-popFR', dpi = 300, figsize = [16,8])
        ax = fig.add_subplot(211)
        denorm = stepInterval*dt/1000
        l_fr = np.array([np.percentile([sum(np.logical_and(spScatter[i]>=it[j]*dt, spScatter[i]<it[j+1]*dt)) for i in epick], low) for j in range(it.size-1)], dtype = float)/denorm
        h_fr = np.array([np.percentile([sum(np.logical_and(spScatter[i]>=it[j]*dt, spScatter[i]<it[j+1]*dt)) for i in epick], high) for j in range(it.size-1)], dtype = float)/denorm
        denorm *= len(epick)
        fr = np.array([sum([sum(np.logical_and(spScatter[i]>=it[j]*dt, spScatter[i]<it[j+1]*dt)) for i in epick]) for j in range(it.size-1)], dtype = float)/denorm
        smoothed_fr = movingAvg(fr, it.size-1, fr_window)
        max_frE = fr.max()
        ax.plot(it[:-1]*dt, smoothed_fr, '-r', lw = 0.5, label = 'exc. fr', alpha = 0.7)
        ax.fill_between(it[:-1]*dt, l_fr, y2 = h_fr, color = 'r', edgecolor = 'None', alpha = 0.5)
        denorm = stepInterval*dt/1000
        l_fr = np.array([np.percentile([sum(np.logical_and(spScatter[i]>=it[j]*dt, spScatter[i]<it[j+1]*dt)) for i in ipick], low) for j in range(it.size-1)], dtype = float)/denorm
        h_fr = np.array([np.percentile([sum(np.logical_and(spScatter[i]>=it[j]*dt, spScatter[i]<it[j+1]*dt)) for i in ipick], high) for j in range(it.size-1)], dtype = float)/denorm
        denorm *= len(ipick)
        fr = np.array([sum([sum(np.logical_and(spScatter[i]>=it[j]*dt, spScatter[i]<it[j+1]*dt)) for i in ipick]) for j in range(it.size-1)], dtype = float)/denorm
        smoothed_fr = movingAvg(fr, it.size-1, fr_window)
        max_frI = fr.max()
        ax.plot(it[:-1]*dt, smoothed_fr, '-b', lw = 0.5, label = 'inh. fr', alpha = 0.7)
        ax.fill_between(it[:-1]*dt, l_fr, y2 = h_fr, color = 'b', edgecolor = 'None', alpha = 0.5)
        total_spikesE = np.mean([sum(np.logical_and(spScatter[i] < it[-1]*dt, spScatter[i] >= it[0]*dt)) for i in epick])
        total_spikesI = np.mean([sum(np.logical_and(spScatter[i] < it[-1]*dt, spScatter[i] >= it[0]*dt)) for i in ipick])
        LGN_spikes = np.sum([sum(np.logical_and(LGN_spScatter[i] < it[-1]*dt, LGN_spScatter[i] >= it[0]*dt)) for i in range(nLGN)])
        denorm = stepInterval*dt/1000*nLGN
        LGN_fr = np.array([sum([sum(np.logical_and(LGN_spScatter[i]>=it[j]*dt, LGN_spScatter[i]<it[j+1]*dt)) for i in range(nLGN)]) for j in range(it.size-1)], dtype = float)/denorm
        smoothed_LGN_fr = movingAvg(LGN_fr, it.size-1, fr_window)
        ax.plot(it[:-1]*dt, smoothed_LGN_fr, '-g', lw = 0.5, label = 'LGN. fr', alpha = 0.7)
        #LTP, LTD = learning_rule_convol(LGN_spScatter, spScatter, epick, tauTrip, tauLTD, tauLTP, 3, 0, nt_, dt, fig_fdr + output_suffix)
        
        #if LTP != 0:
        #    LTD_LTP_ratio = LTD/LTP
        #else:
        #    LTD_LTP_ratio = 0
        #ax.set_title(f'E: {max_frE:.1f}Hz({total_spikesE:.1f}sp), I: {max_frI:.1f}Hz({total_spikesI:.1f}sp), LGN: {LGN_fr.max():.1f}Hz, LTP:{LTP:.1f}, LTD:{LTD:.1f}; {LTD_LTP_ratio:.1f}')
        ax.set_title(f'E: {max_frE:.1f}Hz({total_spikesE:.1f}sp), I: {max_frI:.1f}Hz({total_spikesI:.1f}sp), LGN: {LGN_fr.max():.1f}Hz')
        ax.set_ylabel('fr')
        ax.set_xlabel('sample range (ms)')
        ax = ax.twinx()
        for i in range(nLGN):
            scatter = LGN_spScatter[i][np.logical_and(LGN_spScatter[i] >= it[0] * dt, LGN_spScatter[i] < it[-1] * dt)]
            ax.plot(scatter, len(scatter)*[i], ',g')
        for i in range(nblock):
            for j in range(nE):
                idx = i*blockSize + j
                scatter = spScatter[idx][np.logical_and(spScatter[idx] >= it[0] * dt, spScatter[idx] < it[-1] * dt)]
                ax.plot(scatter, len(scatter)*[idx+nLGN], ',r')
            for j in range(nI):
                idx = i*blockSize + nE + j
                scatter = spScatter[idx][np.logical_and(spScatter[idx] >= it[0] * dt, spScatter[idx] < it[-1] * dt)]
                ax.plot(scatter, len(scatter)*[idx+nLGN], ',b')
        ax.set_ylabel('LGN + V1 index')

        ax = fig.add_subplot(212)
        denorm = stepInterval0*dt/1000
        l_fr = np.array([np.percentile([sum(np.logical_and(spScatter[i]>=it0[j]*dt, spScatter[i]<it0[j+1]*dt)) for i in epick], low) for j in range(it0.size-1)], dtype = float)/denorm
        h_fr = np.array([np.percentile([sum(np.logical_and(spScatter[i]>=it0[j]*dt, spScatter[i]<it0[j+1]*dt)) for i in epick], high) for j in range(it0.size-1)], dtype = float)/denorm
        denorm *= len(epick)
        fr = np.array([sum([sum(np.logical_and(spScatter[i]>=it0[j]*dt, spScatter[i]<it0[j+1]*dt)) for i in epick]) for j in range(it0.size-1)], dtype = float)/denorm
        max_frE = fr.max()
        smoothed_fr = movingAvg(fr, it.size-1, fr_window)

        V1_totalFrFn = data_fdr + "V1_totalFr" + _output_suffix + ".bin"
        with open(V1_totalFrFn, 'wb') as f:
            np.array([it0.size-1]).tofile(f)
            ((it0[:-1] + it0[1:])/2*dt).tofile(f)
            fr.tofile(f)

        ax.plot(it0[:-1]*dt/1000, smoothed_fr, '-r', lw = 0.5, label = 'exc. fr')
        ax.fill_between(it0[:-1]*dt/1000, l_fr, y2 = h_fr, color = 'r', edgecolor = 'None', alpha = 0.5)
        denorm = stepInterval0*dt/1000
        l_fr = np.array([np.percentile([sum(np.logical_and(spScatter[i]>=it0[j]*dt, spScatter[i]<it0[j+1]*dt)) for i in ipick], low) for j in range(it0.size-1)], dtype = float)/denorm
        h_fr = np.array([np.percentile([sum(np.logical_and(spScatter[i]>=it0[j]*dt, spScatter[i]<it0[j+1]*dt)) for i in ipick], high) for j in range(it0.size-1)], dtype = float)/denorm
        denorm *= len(ipick)
        fr = np.array([sum([sum(np.logical_and(spScatter[i]>=it0[j]*dt, spScatter[i]<it0[j+1]*dt)) for i in ipick]) for j in range(it0.size-1)], dtype = float)/denorm
        max_frI = fr.max()
        smoothed_fr = movingAvg(fr, it.size-1, fr_window)
        ax.plot(it0[:-1]*dt/1000, smoothed_fr, '-b', lw = 0.5, label = 'inh. fr')
        ax.fill_between(it0[:-1]*dt/1000, l_fr, y2 = h_fr, color = 'b', edgecolor = 'None', alpha = 0.7)
        total_spikesE = np.mean([len(spScatter[i]) for i in epick])
        total_spikesI = np.mean([len(spScatter[i]) for i in ipick])
        ax.set_title(f'max FR. (avg. spikes) E: {max_frE:.1f}Hz({total_spikesE:.1f}), I: {max_frI:.1f}Hz({total_spikesI:.1f})')
        ax.set_ylabel('fr')
        ax.set_xlabel('full time (s)')
        fig.tight_layout(pad = 0.5)
        fig.savefig(fig_fdr+output_suffix + f'V1-popFR.png', bbox_inches='tight')
        plt.close(fig)


def learning_rule_convol(LGN_sp, V1_sp, idx, tauTrip, tauLTD, tauLTP, ntau, t0, nt, dt, fig_prefix = None):
    exp_weightLTP = np.exp(-np.arange(round(tauLTP/dt)*ntau) * dt/tauLTP)
    exp_weightLTD = np.exp(-np.arange(round(tauLTD/dt)*ntau) * dt/tauLTD)
    exp_weightTrip = np.exp(-np.arange(round(tauTrip/dt)*ntau) * dt/tauTrip)
    V1_sps = np.zeros(nt-t0)
    V1_i = np.zeros(len(V1_sp), dtype = int)
    LGN_sps = np.zeros(nt-t0)
    LGN_i = np.zeros(len(LGN_sp), dtype = int)
    for j in range(t0, nt):
        for i in idx:
            if V1_i[i] <= len(V1_sp[i]):
                continue
            while V1_sp[i][V1_i[i]] < (j+1)*dt:
                V1_sps[j-t0] += 1
                V1_i[i] += 1
                if V1_i[i] >= len(V1_sp[i]):
                    break
        for i in range(len(LGN_sp)):
            if LGN_i[i] <= len(LGN_sp[i]):
                continue
            print('#', len(LGN_sp[i]), LGN_i[i])
            while LGN_sp[i][LGN_i[i]] < (j+1)*dt:
                LGN_sps[j-t0] += 1
                LGN_i[i] += 1
                print(' ', len(LGN_sp[i]), LGN_i[i])
                if LGN_i[i] >= len(LGN_sp[i]):
                    break
        if j % round((nt-t0)*0.1) == 0:
            print(f'binning {j/(nt-t0)*100:.1f}%', end = '\r')
    V1_sps /= len(idx)
    LTP_mavg = np.convolve(LGN_sps, exp_weightLTP, mode = 'same')
    LTD_mavg = np.convolve(V1_sps, exp_weightLTD, mode = 'same')
    Trip_mavg = np.convolve(V1_sps, exp_weightTrip, mode = 'same')
    print('')
    print('convolved')
    if fig_prefix is not None:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(np.arange(round(tauLTP/dt)*ntau), exp_weightLTP, '-g')
        ax.plot(np.arange(round(tauTrip/dt)*ntau), exp_weightTrip, '-r')
        ax.plot(np.arange(round(tauLTD/dt)*ntau), exp_weightLTD, '-b')
        ax = fig.add_subplot(222)
        ax.plot(np.arange(t0,nt), LGN_sps, '-g', lw = 0.1)
        ax.plot(np.arange(t0,nt), LTP_mavg, ':g', lw = 0.1)
        ax = fig.add_subplot(224)
        ax.plot(np.arange(t0,nt), V1_sps, '-r', lw = 0.1)
        ax.plot(np.arange(t0,nt), LTD_mavg, ':r', lw = 0.1)
        fig.savefig(fig_prefix + '-convolve_check.png')
    LTP = sum(LTP_mavg*Trip_mavg*V1_sps)
    LTD = sum(LTD_mavg*LGN_sps)
    return LTP, LTD

if __name__ == "__main__":
    print(sys.argv)
    print(len(sys.argv))
    if len(sys.argv) < 6:
        raise Exception('not enough argument for plotV1_fr(output_suffix0, res_fdr, data_fdr, fig_fdr, inputFn, nOri, readNewSpike, ns)')
    else:
        output_suffix0 = sys.argv[1]
        print(output_suffix0)
        res_fdr = sys.argv[2]
        print(res_fdr)
        data_fdr = sys.argv[3]
        print(data_fdr)
        fig_fdr = sys.argv[4]
        print(fig_fdr)
        inputFn = sys.argv[5]
        print(inputFn)
        nOri = int(sys.argv[6])
        if sys.argv[7] == 'True' or sys.argv[7] == '1':
            readNewSpike = True 
            print('read new spikes')
        else:
            readNewSpike = False
            print('read stored spikes')
        if len(sys.argv) < 9:
            ns = 10
        else:
            ns = int(sys.argv[8])
        print(f'ns = {ns}')

    plotV1_fr(output_suffix0, res_fdr, data_fdr, fig_fdr, inputFn, nOri, readNewSpike, ns)
