# connection strength heatmaps # better to choose from testLearnFF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import contourpy as ctr
from scipy.optimize import curve_fit
import scipy.stats as stat
import sys
import warnings
    
def outputLearnFF(seed, isuffix0, isuffix, osuffix, res_fdr, setup_fdr, data_fdr, fig_fdr, LGN_switch, mix, st, examSingle, use_local_max, stage, ns, examLTD):
    step0 = 0
    nt_ = 0
    avgOnly = False
    
    fig_fdr = fig_fdr+'/'
    res_fdr = res_fdr+'/'
    setup_fdr = setup_fdr+'/'
    data_fdr = data_fdr+'/'
    np.random.seed(seed)
    if not isuffix0 == '':
        isuffix0 = '-' + isuffix0
    
    if not isuffix == '' :
        isuffix = '-' + isuffix
    
    if not osuffix == '':
        osuffix = '-'+osuffix
    
    #### HERE ####
    top_thres = 0.8
    thres_out = 0.2
    os_out = 0.5
    nstep = 1000
    nbins = 20
    nit0 = 20
    
    #V1_pick = np.array([203,752,365,360,715,467,743]); # specify the IDs of V1 neurons to be sampled. If set, ns will be ignored.
    #V1_pick = np.array([203])
    nop = 12
    ############
    
    f_sLGN = data_fdr+'sLGN'+osuffix+'.bin'
    f_dsLGN = data_fdr+'dsLGN'+osuffix+'.bin'
    learnDataFn = data_fdr+'learnData_FF'+osuffix+'.bin'
    V1_frFn = data_fdr+'max_fr'+osuffix+'.bin'

    fV1_allpos = res_fdr+'V1_allpos'+isuffix0+'.bin'
    fLGN_vpos = res_fdr+'LGN_vpos'+isuffix0+'.bin'

    fLGN_V1_ID = setup_fdr+'LGN_V1_idList'+isuffix+'.bin'
    fLGN_switch = setup_fdr+'LGN_switch'+isuffix+'.bin'
    fConnectome = setup_fdr+'connectome_cfg'+isuffix+'.bin'

    fStats = data_fdr + 'metric' + osuffix + '.bin'

    with open(fLGN_vpos, 'rb') as f:
        nLGN, nLGN_I = np.fromfile(f, 'u4', 2)
        print(f'Contra:{nLGN}, Ipsi:{nLGN_I}')
        f.seek(2 * 4, 1)
        xspan = np.fromfile(f, 'f4', 1)[0]
        print(f'span of LGN_vpos {xspan}')
        f.seek(2 * 4, 1)
        
        LGN_vpos = np.fromfile(f, 'f4', 2*(nLGN+nLGN_I)).reshape(2, nLGN + nLGN_I)
        LGN_type = np.fromfile(f, 'u4', nLGN+nLGN_I)
        types = np.unique(LGN_type)
        ntype = len(types)
        f.seek((nLGN + nLGN_I) * 2 * 4, 1)
        doubleOnOff, nStage, stageFrame, nFrame = np.fromfile(f, 'i4', 4)
        input_halfwidth = np.fromfile(f, 'f4', 2*nStage)/2

    # read the constants first only
    with open(f_sLGN, 'rb') as f:
        nt, sampleInterval = np.fromfile(f,'u4', 2)
        dt = np.fromfile(f, 'f4', 1)[0]
        nV1, max_LGNperV1 = np.fromfile(f, 'u4', 2)
        sRatio = np.fromfile(f, 'f4', 2)
        nLearnFF_E, nLearnFF_I = np.fromfile(f, 'u4', 2)
        nLearnFF = nLearnFF_E + nLearnFF_I
        sRatio = np.array(nLearnFF_E * [sRatio[0]]  + nLearnFF_I * [sRatio[1]])
        gmaxLGN = np.fromfile(f, 'f4', nLearnFF) * sRatio
        FF_InfRatio = np.fromfile(f, 'f4', 1)[0]
        nskip = (10 + nLearnFF)
        capacity = max_LGNperV1 * sRatio * FF_InfRatio / gmaxLGN / top_thres
        print(capacity, top_thres, gmaxLGN)
    
    with open(fV1_allpos, 'rb') as f:
        nblock = np.fromfile(f, 'u4', 1)[0]

    with open(fConnectome,'rb') as f:
        f.seek(2*4, 1)
        mE, blockSize = np.fromfile(f, 'u4', 2)
        mI = blockSize - mE
        synPerCon = np.fromfile(f, 'f4', 4)
        synPerFF = np.fromfile(f, 'f4', 2)
    nE = mE*nblock
    nI = mI*nblock

    epick = np.zeros(mE * nblock, dtype = 'u4')
    ipick = np.zeros(mI * nblock, dtype = 'u4')
    for i in range(nblock):
        epick[i*mE: (i+1)*mE] = i * blockSize + np.arange(mE)
        ipick[i*mI: (i+1)*mI] = i * blockSize + mE + np.arange(mI)
    
    try:
        with open(V1_frFn,'rb') as f:
            fr = np.fromfile(f, 'f8', nV1)
    except Exception as e:
        fr = np.zeros(nV1)
    
    #tstep
    if nt_ > nt or nt_ == 0:
        nt_ = nt
    
    if step0 > nt_:
        step0 = 0
    
    range_nt = nt_ - step0
    if range_nt == nt:
        rtime = ''
    else:
        rtime = np.array(['-t',num2str(step0 / nt * 100,'%.0f'),'_',num2str(nt_ / nt * 100,'%.0f'),'%'])
    
    if nstep > range_nt or nstep == 0:
        nstep = range_nt

    if sampleInterval > 1:
        dt = sampleInterval * dt
        nt_float = nt / sampleInterval
        nt = int(np.floor(nt / sampleInterval))
        nt_ = int(np.round(nt_ / sampleInterval))
        if nt_ > nt:
            nt_ = nt
        step0 = step0 // sampleInterval
        if step0 > nt_:
            step0 = 0
        range_nt = nt_ - step0
        if nstep > range_nt or nstep == 0:
            nstep = range_nt

    print(f'step0 = {step0}')
    print(f'nt_ = {nt_}')
    print(f'nstep = {nstep}')
    
    # read connection id
    with open(fLGN_V1_ID, 'rb') as f:
        LGN_V1_ID = np.zeros((nV1, max_LGNperV1), 'u4')
        nLGN_V1 = np.zeros(nV1, dtype = 'u4')
        f.seek(4, 1)
    
        for i in range(nV1):
            nLGN_V1[i] = np.fromfile(f, 'u4', 1)[0]
            assert(nLGN_V1[i] == max_LGNperV1)
            if nLGN_V1[i] > 0:
                LGN_V1_ID[i,:nLGN_V1[i]] = np.fromfile(f, 'u4', nLGN_V1[i])
                assert((LGN_V1_ID[i,:nLGN_V1[i]] >= 0).all())
                assert((LGN_V1_ID[i,:nLGN_V1[i]] < nLGN + nLGN_I).all())
    
    if 'V1_pick' not in locals():
        V1_pick = np.random.randint(nV1, size = ns)
    else:
        ns = V1_pick.size
    
    V1_pick = np.sort(V1_pick)
    print(V1_pick)
    nLGN_1D = int(np.round(np.sqrt(float(nLGN / 2))))
    print(LGN_type.reshape(nLGN_1D, 2*nLGN_1D))
    print(nLGN, nLGN_1D)
    #min_dis = xspan/(nLGN_1D-1);
    min_dis = 0
    if LGN_switch:
        with open(fLGN_switch, 'rb') as f:
            nStatus = np.fromfile(f, 'u4', 1)[0]
            status = np.fromfile(f,'f4', nStatus * 6)
            statusDur = np.fromfile(f, 'f4', nStatus)
            reverse = np.fromfile(f,'i4', nStatus)

        typeInput = np.array([np.sum(statusDur * (1 - reverse)), np.sum(statusDur * reverse)]) / np.sum(statusDur)
        nit = nStatus + 1
    else:
        nit = nit0
        typeInput = np.ones(ntype)
    
    if nit <= 1:
        nit = 2
    
    nrow = (nit + nit0 - 1)//nit0
    if nit > nt_ - step0:
        nit = nt_ - step0 
    
    #LGN_vpos0 = LGN_vpos.reshape(2,nLGN_1D, 2*nLGN_1D)
    #x0 = LGN_vpos0[0, 0, 0:2*nLGN_1D:2]
    #dx = x0[1] - x0[0]
    #r = np.linalg.norm(LGN_vpos0[:,:,0:2*nLGN_1D:2], axis = 0).flatten()
    if nLGN_1D % 2 == 1:
        v0 = np.arange(nLGN_1D).astype('f8') - (nLGN_1D-1)/2
        multiplier = 1
    else:
        v0 = np.arange(nLGN_1D).astype('f8')*2 - (nLGN_1D-1)
        multiplier = 2
    xx, yy = np.meshgrid(v0, v0)
    rLGN = np.sqrt(xx*xx + yy*yy)
    r = np.sort(np.unique(rLGN.flatten())/multiplier)
    min_dr = np.diff(r).min()
    rLGN = rLGN/multiplier
    qt = np.round(np.linspace(step0, nt_-1, nit)).astype('u4')
    ################## HERE ##################
    fout = open(fStats, 'wb')
    np.array([step0, nt_, nit, stage, nLGN_1D], dtype = int).tofile(fout)
    np.array([dt]).astype('f4').tofile(fout)
    input_halfwidth.astype('f4').tofile(fout)
    
    with open(f_sLGN, 'rb') as f:
        f.seek(nskip * 4, 1)
        fig = plt.figure('tOS-dist', figsize = (8,nit * 1.5), dpi = 300)
        fit_x = np.linspace(0, r[-1], 100)
        op_edges = np.linspace(0,360,nop)
        x_op = (op_edges[:-1] + op_edges[1:]) / 2
        lims = np.zeros((2,2))
        lims[:,0] = float('inf')
        lims[:,1] = - float('inf')
        it = np.diff(qt) - 1
        max20 = np.zeros((2, nit, max_LGNperV1 // 2))
        min0 = np.zeros((2, nit, max_LGNperV1 // 2))
        picked = np.zeros((2, nV1, nLGN_1D,nLGN_1D), dtype = bool)
        s_on = np.zeros((nV1, max_LGNperV1 // 2))
        s_off = np.zeros((nV1, max_LGNperV1 // 2))
        ron = np.zeros((nV1, max_LGNperV1 // 2))
        roff = np.zeros((nV1, max_LGNperV1 // 2))
        s_Don = np.zeros((r.size))
        s_Doff = np.zeros((r.size))
        rstd_on = np.zeros(nit)
        rstd_off = np.zeros(nit)
        _rstd_on = np.zeros((nV1,nit))
        _rstd_off = np.zeros((nV1,nit))
        def Gauss0(x, sigma, vmax):
            y = vmax*np.exp(-0.5*np.power(x,2)/(sigma*sigma))
            return y
        def Gauss1(x, sigma, vmax, shift, vmin):
            y = vmin + (vmax -vmin) * np.exp(-0.5*np.power(np.max(np.vstack((x-shift,np.zeros_like(x))), axis = 0),2)/(sigma*sigma))
            return y
        def choose_func(ydata, xdata):
            data = np.zeros(r.size)
            for j in range(r.size):
                pick = np.logical_and(xdata > r[j] - min_dr/2, xdata < r[j] + min_dr/2)
                if pick.any():
                    data[j] = np.mean(ydata[pick])
                else:
                    data[j] = np.nan
            pick = np.logical_not(np.isnan(data))
            dmin = np.min(data[pick])
            dmax = np.max(data[pick])
            return Gauss0, 0, dmin, dmax
            #if dmin/dmax > 0.615:
            #    gaussian_type = 0
            #    Gauss = Gauss0
            #else:
            #    gaussian_type = 1
            #    Gauss = Gauss1
            #return  Gauss, gaussian_type, dmin, dmax

        n0 = 0
        OnOff_balance = np.zeros(nit)
        OnOff_balance_p = np.zeros(nit)
        g_std_p = np.zeros(nit)
        ctr_p = np.zeros(nit)
        dis2c_p = np.zeros(nit)
        total_err = 0
        for i in range(nit):
            nerr = 0
            if i == 0:
                f.seek(max_LGNperV1 * nV1 * step0 * 4, 1)
            else:
                if it[i-1] > 0:
                    f.seek(max_LGNperV1 * nV1 * it[i-1] * 4, 1)
            sLGN = np.fromfile(f, 'f4', max_LGNperV1*nV1).reshape((max_LGNperV1, nV1)).T
            sLGN_grid = np.zeros((nV1,nLGN))
            for j in range(nV1):
                sLGN_grid[j, LGN_V1_ID[j, :nLGN_V1[j]]] = sLGN[j, :nLGN_V1[j]]

            sLGN_grid = np.reshape(sLGN_grid, (nV1, nLGN_1D, nLGN_1D * 2))
            sLGN_on = sLGN_grid[:,:,0:nLGN_1D * 2:2]
            sLGN_off = sLGN_grid[:,:,1:nLGN_1D * 2:2]
            if i == 0:
                picked[0,:,:,:] = sLGN_on > 0
                picked[1,:,:,:] = sLGN_off > 0
                nonzero_pick = np.logical_or(picked[0,:,:,:].any(0), picked[1,:,:,:].any(0))
                nonzero_sum = np.zeros((2, nLGN_1D, nLGN_1D))
                nonzero_sum[0,:,:] = picked[0,:,:,:].sum(0)
                nonzero_sum[1,:,:] = picked[1,:,:,:].sum(0)

            for j in range(nV1):
                if np.sum(picked[:,j,:,:]) > 0:
                    q_on = sLGN_on[j,:,:].flatten()
                    s_on[j,:] = q_on[picked[0,j,:,:].flatten()].flatten()
                    ron[j,:] = rLGN[picked[0,j,:,:]].flatten()
                    q_off = sLGN_off[j,:,:].flatten()
                    s_off[j,:] = q_off[picked[1,j,:,:].flatten()].flatten()
                    roff[j,:] = rLGN[picked[1,j,:,:]].flatten()

            binEdges = np.arange(max_LGNperV1 / 2+1)
            max20[0,i,:], _ = np.histogram(np.sum(s_on > gmaxLGN[0] * top_thres, axis = 1), bins = binEdges)
            max20[1,i,:], _ = np.histogram(np.sum(s_off > gmaxLGN[0] * top_thres, axis = 1), bins = binEdges)
            min0[0,i,:], _ = np.histogram(np.sum(s_on == 0, axis = 1), bins = binEdges)
            min0[1,i,:], _ = np.histogram(np.sum(s_off == 0, axis = 1), bins = binEdges)
            if i == nit-1:
                nmax = np.sum(s_on > gmaxLGN[0] * top_thres, axis = 1)
                if (nmax > 0).any():
                    avg_on_max = np.mean(nmax[nmax > 0])
                else:
                    avg_on_max = 0

                nmax = np.sum(s_off > gmaxLGN[0] * top_thres, axis = 1)
                if (nmax > 0).any():
                    avg_off_max = np.mean(nmax[nmax > 0])
                else:
                    avg_off_max = 0

                nmin = np.sum(s_on == 0, axis = 1)
                if (nmin > 0).any():
                    avg_on_min = np.mean(nmin[nmin > 0])
                else:
                    avg_on_min = 0

                nmin = np.sum(s_off == 0, axis = 1)
                if (nmin > 0).any():
                    avg_off_min = np.mean(nmin[nmin > 0])
                else:
                    avg_off_min = 0

            Gauss, gaussian_type_on, dmin, dmax = choose_func(s_on[epick,:], ron[epick,:])
            if gaussian_type_on == 0:
                p, _ = curve_fit(Gauss, ron[epick,:].flatten(), s_on[epick,:].flatten(), p0 = [nLGN_1D, dmax], bounds = ([0, dmin], [np.inf, np.inf]))
                fitted_on = Gauss(fit_x, p[0], p[1])
                rstd_on[i] = p[0]
            else:
                p, _ = curve_fit(Gauss, ron[epick,:].flatten(), s_on[epick,:].flatten(), p0 = [nLGN_1D/4, dmax, 1, dmin], bounds = ([0, dmin, 0, 0], [np.inf, np.inf, np.inf, dmax]))
                fitted_on = Gauss(fit_x, p[0], p[1], p[2], p[3])
                rstd_on[i] = p[0] + p[2]

            for j in range(nV1):
                if j in epick or nLearnFF_I > 0:
                    Gauss, gaussian_type, dmin, dmax = choose_func(s_on[j,:], ron[j,:])
                    if gaussian_type == 0:
                        p, _ = curve_fit(Gauss, ron[j,:], s_on[j,:], p0 = [nLGN_1D, dmax], bounds = ([0, dmin], [np.inf, np.inf]))
                        _rstd_on[j,i] = p[0]
                    else:
                        try:
                            p, _ = curve_fit(Gauss, ron[j,:], s_on[j,:], p0 = [nLGN_1D/4, dmax, 1, dmin], bounds = ([0, dmin, 0, 0], [np.inf, np.inf, np.inf, dmax]))
                            _rstd_on[j,i] = p[0] + p[2]
                        except RuntimeError:
                            nerr += 1
                            _fig = plt.figure(f'{i}-{j}')
                            _data = np.zeros(r.size)
                            for k in range(r.size):
                                pick = np.logical_and(ron[j,:] > r[k] - min_dr/2, ron[j,:] < r[k] + min_dr/2)
                                if pick.any():
                                    _data[k] = np.mean(s_on[j,pick])
                                else:
                                    _data[k] = np.nan

                            dmid = (_data.max() + _data.min())/2
                            _rstd_on[j,i] = r[np.argmin(np.abs(_data - dmid))]
                            _ax = _fig.add_subplot(111)
                            _ax.plot(ron[j,:], s_on[j,:], '.k')
                            _ax.plot(r, _data, '.b', alpha = 0.5)
                            _ax.plot(_rstd_on[j,i], dmid, '*r')
                            _fig.savefig(fig_fdr + f'est_rstd_on-t{j}-neuron{i}-{osuffix}.png')

            Gauss, gaussian_type_off, dmin, dmax = choose_func(s_off[epick,:], roff[epick,:])
            if gaussian_type_off == 0:
                p, _ = curve_fit(Gauss, roff[epick,:].flatten(), s_off[epick,:].flatten(), p0 = [nLGN_1D, dmax], bounds = ([0, dmin], [np.inf, np.inf]))
                fitted_off = Gauss(fit_x, p[0], p[1])
                rstd_off[i] = p[0]
            else:
                p, _ = curve_fit(Gauss, roff[epick,:].flatten(), s_off[epick,:].flatten(), p0 = [nLGN_1D/4, dmax, 1, dmin], bounds = ([0, dmin, 0, 0], [np.inf, np.inf, np.inf, dmax]))
                fitted_off = Gauss(fit_x, p[0], p[1], p[2], p[3])
                rstd_off[i] = p[0] + p[2]

            for j in range(nV1):
                if j in epick or nLearnFF_I > 0:
                    Gauss, gaussian_type, dmin, dmax = choose_func(s_off[j,:], roff[j,:])
                    if gaussian_type == 0:
                        p, _ = curve_fit(Gauss, roff[j,:], s_off[j,:], p0 = [nLGN_1D, dmax], bounds = ([0, dmin], [np.inf, np.inf]))
                        _rstd_off[j,i] = p[0]
                    else:
                        try:
                            p, _ = curve_fit(Gauss, roff[j,:], s_off[j,:], p0 = [nLGN_1D/4, dmax, 1, dmin], bounds = ([0, dmin, 0, 0], [np.inf, np.inf, np.inf, dmax]))
                            _rstd_off[j,i] = p[0] + p[2]
                        except RuntimeError:
                            nerr += 1
                            _fig = plt.figure(f'{i}-{j}')
                            _data = np.zeros(r.size)
                            for k in range(r.size):
                                pick = np.logical_and(roff[j,:] > r[k] - min_dr/2, roff[j,:] < r[k] + min_dr/2)
                                if pick.any():
                                    _data[k] = np.mean(s_off[j,pick])
                                else:
                                    _data[k] = np.nan

                            dmid = (_data.max() + _data.min())/2
                            _rstd_off[j,i] = r[np.argmin(np.abs(_data - dmid))]
                            _ax = _fig.add_subplot(111)
                            _ax.plot(roff[j,:], s_off[j,:], '.k')
                            _ax.plot(r, _data, '.b', alpha = 0.5)
                            _ax.plot(_rstd_off[j,i], dmid, '*r')
                            _fig.savefig(fig_fdr + f'est_rstd_off-t{j}-neuron{i}-{osuffix}.png')
            
            g_std_p[i] = stat.ttest_1samp(_rstd_on[epick,i] - _rstd_off[epick,i], 0).pvalue

            if stage == 2: 
                ax1 = fig.add_subplot(nit, 4, 4 * i + 2) # cell-normalized on vs. off
                ax2 = fig.add_subplot(nit, 4, 4 * i + 3) # cell-summed on and off strength
                ax3 = fig.add_subplot(nit, 4, 4 * i + 4) # on vs. off
                sLGN_on = np.sum(sLGN_on, axis = 0)[nonzero_pick]/nonzero_sum[0,:,:][nonzero_pick]
                sLGN_off = np.sum(sLGN_off, axis = 0)[nonzero_pick]/nonzero_sum[1,:,:][nonzero_pick]
                for j in range(r.size):
                    pick = np.logical_and(rLGN[nonzero_pick] > r[j] - min_dr/2, rLGN[nonzero_pick] < r[j] + min_dr/2)
                    if pick.any():
                        s_Don[j] = np.mean(sLGN_on[pick])
                        s_Doff[j] = np.mean(sLGN_off[pick])
                        ax3.plot(r[j], s_Don[j], '.r', ms = 1, alpha = 0.7)
                        ax3.plot(r[j], s_Doff[j], '.b', ms = 1, alpha = 0.7)
                ax3.plot(ron[epick,:], s_on[epick,:], ',r', alpha = 0.3)
                ax3.plot(roff[epick,:], s_off[epick,:], ',b', alpha = 0.3)

                if gaussian_type_on == 0:
                    ax3.plot(fit_x, fitted_on,':r', lw = 1)
                else:
                    ax3.plot(fit_x, fitted_on,'-r', lw = 1)
                if gaussian_type_off == 0:
                    ax3.plot(fit_x, fitted_off,':b', lw = 1)
                else:
                    ax3.plot(fit_x, fitted_off,'-b', lw = 1)
                ax3.set_ylim(bottom = 0)
                if i == 0:
                    ax3.set_title('str over dis')
                    ax3.set_ylabel('#avg str')
                if i == nit-1:
                    plt.xlabel('dis (unit LGN)')
                else:
                    ax3.set_xticks([]) 

            onS, offS, os, overlap, orient = determine_os_str(LGN_vpos, LGN_V1_ID, LGN_type, sLGN, nV1, nLGN_V1, min_dis, os_out, nLGN_1D)

            if stage == 2:
                ax = fig.add_subplot(nit, 4, 4 * i + 1)
            else:
                ax = fig.add_subplot(nit, 3, 3 * i + 1)

            _, binEdges, _ = ax.hist(onS[epick] - offS[epick], bins = 20, color = 'r', alpha = 0.7)
            OnOff_balance[i] = np.mean(onS[epick] - offS[epick])
            ax.hist(onS[ipick] - offS[ipick], bins = binEdges, color = 'b', alpha = 0.7)
            OnOff_balance_p[i] = stat.ttest_1samp(onS[epick] - offS[epick], 0).pvalue
            if binEdges[0] < lims[0,0]:
                lims[0,0] = binEdges[0]
            if binEdges[-1] > lims[0,1]:
                lims[0,1] = binEdges[-1]
            if i == 0:
                ax.set_title('On-Off balance')
                ax.set_xlabel('sOn-sOff')
                ax.set_ylabel('#V1')
            if i < nit-1:
                ax.set_xticks([]) 

            if stage == 3 or stage == 5:
                ax = fig.add_subplot(nit,3,3 * i + 2)
                #binned = np.digitize(orient * 180 / np.pi, bins = op_edges)
                #counts = np.zeros(nop-1)
                #for j in range(nop-1):
                #    counts[j] = np.sum(binned == j)
                #    if counts[j] > 0:
                #        counts[j] *= np.mean(os[binned == j])
                #ax.bar(x_op, counts, color = 'b')
                ax.hist(orient * 180 / np.pi, bins = op_edges, weights = os, color = 'b')

                if i == 0:
                    ax.set_title('OP dist')
                    ax.set_ylabel('#V1 weighted by os')
                if i == nit-1:
                    ax.set_xlabel('OP (deg)')
                else:
                    ax.set_xticks([]) 

                ax = fig.add_subplot(nit, 3, 3 * i + 3)
                _, binEdges, _ = ax.hist(os, bins = nop-1)
                if binEdges[1] < lims[1,0]:
                    lims[1,0] = binEdges[1]
                if binEdges[-1] > lims[1,1]:
                    lims[1,1] = binEdges[-1]
                if i == 0:
                    ax.set_title('OS dist')
                    ax.set_ylabel('#V1')
                if i == nit-1:
                    ax.set_xlabel('dis/(Ron+Roff)')
                else:
                    ax.set_xticks([]) 
            sys.stdout.write(f'\r{i}/{nit}, e: {nerr}')
            total_err += nerr
        print(f'#fit error: {total_err}')

        rstd_on *= np.sqrt(2*np.log(2))
        rstd_off *= np.sqrt(2*np.log(2))
        #pick = rstd_on > nLGN_1D/2
        #rstd_on[pick] = nLGN_1D/2
        #pick = rstd_off > nLGN_1D/2
        #rstd_off[pick] = nLGN_1D/2
        (picked[0, :, :, :].astype(int) - picked[1, :, :, :].astype(int)).astype(float).mean(axis = 0).tofile(fout)
        rstd_on.tofile(fout)
        rstd_off.tofile(fout)
        (np.mean(_rstd_on, axis = 0) * np.sqrt(2*np.log(2))).tofile(fout)
        (np.mean(_rstd_off, axis = 0) * np.sqrt(2*np.log(2))).tofile(fout)
        print('data collected')
        for i in range(nit):
            ax = plt.subplot(nit, 4, 4 * i + 1)
            ax.set_xlim(lims[0,:])
            if stage == 3 or stage == 5:
                ax = plt.subplot(nit, 4, 4 * i + 3)
                ax.set_xlim(lims[1,:])
        
        fig.savefig(f'{fig_fdr}tOS-dist{osuffix}{rtime}.png')
        plt.close(fig)
        print('tOS-dist finished')

        n_above = np.arange(max_LGNperV1/2)
        fig = plt.figure('max_min_tDist', figsize = (6,6), dpi = 300)
        ax = fig.add_subplot(2,2,1)
        counts = max20[0,:,:]
        if (counts > 0).any():
            normed_counts = counts/np.max(counts)
            counts[:,0] = 0
            counts[:,-1] = 0
            weighted_avg = np.array([np.argmax(counts[i,:]) for i in np.arange(nit)])
        else:
            normed_counts = np.zeros((nit, max_LGNperV1 // 2))
            weighted_avg = np.zeros(nit)
        weighted_avg.tofile(fout)
        cmap = 'Reds'
        ax.imshow(normed_counts.T, aspect = 'auto', origin = 'lower', cmap = plt.get_cmap(cmap))
        if (counts > 0).any():
            ax.plot(np.arange(nit), weighted_avg, ':k')

        ax.set_ylabel(f'# > {top_thres * 100:.0f}% upper bound')
        ax.set_title(f'on ({avg_on_max:.2f}/{capacity[0]:.2f})')

        ax = fig.add_subplot(2,2,2)
        counts = max20[1,:,:]
        if (counts > 0).any():
            normed_counts = counts/np.max(counts)
            counts[:,0] = 0
            counts[:,-1] = 0
            weighted_avg = np.array([np.argmax(counts[i,:]) for i in np.arange(nit)])
        else:
            normed_counts = np.zeros((nit, max_LGNperV1 // 2))
            weighted_avg = np.zeros(nit)
        weighted_avg.tofile(fout)
        cmap = 'Blues'
        ax.imshow(normed_counts.T, aspect = 'auto', origin = 'lower', cmap = plt.get_cmap(cmap))
        if (counts > 0).any():
            ax.plot(np.arange(nit), weighted_avg, ':k')

        ax.set_title(f'off ({avg_off_max:.2f}/{capacity[0]:.2f})')

        ax = fig.add_subplot(2,2,3)
        counts = min0[0,:,:]
        if (counts > 0).any():
            normed_counts = counts/np.max(counts)
            counts[:,0] = 0
            counts[:,-1] = 0
            weighted_avg = np.array([np.argmax(counts[i,:]) for i in np.arange(nit)])
        else:
            normed_counts = np.zeros((nit, max_LGNperV1 // 2))
            weighted_avg = np.zeros(nit)
        weighted_avg.tofile(fout)
        cmap = 'Reds'
        ax.imshow(normed_counts.T, aspect = 'auto', origin = 'lower', cmap = plt.get_cmap(cmap))
        if (counts > 0).any():
            ax.plot(np.arange(nit), weighted_avg, ':k')

        ax.set_ylabel('# == 0')
        ax.set_title(f'on: {avg_on_min:.2f}')
        ax.set_xlabel('sample time')
        ax = fig.add_subplot(2,2,4)
        counts = min0[1,:,:]
        if (counts > 0).any():
            normed_counts = counts/np.max(counts)
            counts[:,0] = 0
            counts[:,-1] = 0
            weighted_avg = np.array([np.argmax(counts[i,:]) for i in np.arange(nit)])
        else:
            normed_counts = np.zeros((nit, max_LGNperV1 // 2))
            weighted_avg = np.zeros(nit)
        weighted_avg.tofile(fout)
        cmap = 'Blues'
        ax.imshow(normed_counts.T, aspect = 'auto', origin = 'lower', cmap = plt.get_cmap(cmap))
        if (counts > 0).any():
            ax.plot(np.arange(nit), weighted_avg, ':k')

        ax.set_title(f'off: {avg_off_min:.2f}')
        fig.savefig(f'{fig_fdr}max_min_tDist{osuffix}{rtime}.png')
        plt.close(fig)
        print('max_min_tDist finished')
        f.seek(max_LGNperV1 * nV1 * (nt_ - 1) * 4 + nskip*4, 0)
        sLGN = np.fromfile(f, 'f4', max_LGNperV1* nV1).reshape(max_LGNperV1, nV1).T
    
    onS, offS, os, overlap, orient = determine_os_str(LGN_vpos, LGN_V1_ID, LGN_type, sLGN, nV1, nLGN_V1, min_dis, os_out, nLGN_1D)
    
    fig = plt.figure('stats-LGN_V1', figsize = (6,6), dpi = 300)
    ax = fig.add_subplot(2,2,1)
    dS = onS - offS
    #dS_edges = np.histogram_bin_edges(dS, bins = nbins)
    #countsE, _ = np.histogram(dS[epick], bins = dS_edges)
    #countsI, _ = np.histogram(dS[ipick], bins = dS_edges)
    #x_ds = (dS_edges[:-1] + dS_edges[1:]) / 2
    #ax.bar(x_ds, countsE, color = 'r', alpha = 0.5)
    #ax.bar(x_ds, countsI, color = 'b', alpha = 0.5)
    _, _edges, _ = ax.hist(dS[epick], bins = nbins, weights = os[epick], color = 'r', alpha = 0.5)
    ax.hist(dS[ipick], bins = _edges, weights = os[ipick], color = 'b', alpha = 0.5)
    ax.legend(['E','I'])
    ax.set_xlabel('sOn-sOff')
    ax.set_ylabel('#V1')
    ax = fig.add_subplot(2,2,2)
    binE = np.digitize(orient[epick] * 180 / np.pi, bins = op_edges)
    binI = np.digitize(orient[ipick] * 180 / np.pi, bins = op_edges)
    countsE = np.array([np.sum(binE == i) for i in range(nop-1)])
    countsI = np.array([np.sum(binI == i) for i in range(nop-1)])

    w = op_edges[1] - op_edges[0]
    ax.bar(x_op, countsE, width = w, color = 'r', alpha = 0.5)
    ax.bar(x_op, countsI, width = w, color = 'b', alpha = 0.5)
    ax.set_xlabel('OP (deg)')
    ax.set_ylabel('#V1')
    ax = fig.add_subplot(2,2,4)
    osE = os[epick]
    osI = os[ipick]
    overlapE = overlap[epick]
    overlapI = overlap[ipick]
    ocountsE = countsE.copy()
    ocountsI = countsI.copy()
    for i in range(nop-1):
        if countsE[i] > 0:
            countsE[i] *= np.mean(osE[binE == i])
            ocountsE[i] *= (1 - np.mean(overlapE[binE == i]))
        if countsI[i] > 0:
            countsI[i] *=  np.mean(osI[binI == i])
            ocountsI[i] *= (1 - np.mean(overlapI[binI == i]))
    
    ax.bar(x_op, countsE, width = w, color = 'r', alpha = 0.5)
    ax.bar(x_op, countsI, width = w, color = 'b', alpha = 0.5)
    ax.bar(x_op, -ocountsE, width = w, color = 'm', alpha = 0.5)
    ax.bar(x_op, -ocountsI, width = w, color = 'c', alpha = 0.5)

    ax.set_xlabel('OP (deg)')
    ax.set_ylabel('#V1 weighted by os')
    ax = fig.add_subplot(2,2,3)
    osEdges = np.linspace(0,1,nop)
    countsE, _ = np.histogram(osE, bins = osEdges)
    countsI, _ = np.histogram(osI, bins = osEdges)
    ocountsE, _ = np.histogram(overlapE, bins = osEdges)
    ocountsI, _ = np.histogram(overlapI, bins = osEdges)
    x_os = (osEdges[:-1] + osEdges[1:]) / 2
    w = osEdges[1] - osEdges[0]
    ax.bar(x_os, countsE, width = w, color = 'r', alpha = 0.5)
    ax.bar(x_os, countsI, width = w, color = 'b', alpha = 0.5)
    ax.bar(x_os, ocountsE, width = w, color = 'm', alpha = 0.5)
    ax.bar(x_os, ocountsI, width = w, color = 'c', alpha = 0.5)
    ax.set_xlabel('dis/(Ron+Roff) or overlap')
    ax.set_ylabel('#V1')
    fig.savefig(f'{fig_fdr}stats-LGN_V1{osuffix}{rtime}.png')
    plt.close(fig)
    print('stats finished')

    ################## HERE ##################
    
    if st == 2 or st == 1:
        sLGN_all = np.zeros((ns+2, nit, nLGN))
        if avgOnly:
            radius_t = np.zeros((ns+2, 2, nit))
            on_center = np.zeros((2, ns+2, nit))
            off_center = np.zeros((2, ns+2, nit))
            contour_area=  np.zeros((ns+1, 2, nit))
        else:
            radius_t = np.zeros((nV1+2, 2, nit))
            on_center = np.zeros((2, nV1+2, nit))
            off_center = np.zeros((2, nV1+2, nit))
            contour_area = np.zeros((nV1+2, 2, nit))
        with open(f_sLGN, 'rb') as f:
            f.seek(nskip * 4, 1)
            # skip times
                #ht = round(nt/2);
                #it = [0, ht-1, nt-1 - (ht+1)]
            for j in range(nit):
                if j == 0:
                    f.seek(max_LGNperV1 * nV1 * step0 * 4, 1)
                else:
                    if it[j - 1] > 0:
                        f.seek(max_LGNperV1 * nV1 * it[j - 1] * 4, 1)
                data = np.fromfile(f, 'f4', nV1*max_LGNperV1).reshape(max_LGNperV1, nV1).T
                if not avgOnly:
                    radius_t[:-2,:,j], on_center[:,:-2,j], off_center[:,:-2,j] = get_radius(nV1, data, nLGN_V1, LGN_vpos, LGN_type, LGN_V1_ID)
                    contour_area[:-2,:,j] = get_contour_area(nV1, data, nLGN_V1, LGN_V1_ID, nLGN_1D)
                q = 0
                for i in range(nV1):
                    if i in V1_pick:
                        iV1 = i
                        sLGN_all[q, j, LGN_V1_ID[iV1, :nLGN_V1[iV1]]] = data[iV1, :nLGN_V1[iV1]]
                        if avgOnly:
                            radius_t[q,:,j], on_center[:,q,j], off_center[:,q,j] = get_radius(1, data[iV1, :nLGN_V1[iV1]], nLGN_V1[iV1], LGN_vpos, LGN_type, LGN_V1_ID[iV1,:])
                            contour_area[q,:,j] = get_contour_area(1, data[iV1, :nLGN_V1[iV1]], nLGN_V1[iV1], LGN_V1_ID[iV1,:], nLGN_1D)
                        q += 1
                    if i in epick:
                        sLGN_all[-2, j, LGN_V1_ID[i, :nLGN_V1[i]]] += data[i, :nLGN_V1[i]]
                    if i in ipick and nLearnFF_I > 0:
                        sLGN_all[-1, j, LGN_V1_ID[i, :nLGN_V1[i]]] += data[i, :nLGN_V1[i]]
                # averaged
                sLGN_all[-2, j, :] /= nE
                radius_t[-2,:,j], on_center[:,-2,j], off_center[:,-2,j] = get_radius(1, sLGN_all[-2, j, :], nLGN, LGN_vpos, LGN_type, np.arange(nLGN))
                contour_area[-2,:,j] = get_contour_area(1, sLGN_all[-2, j, :], nLGN, np.arange(nLGN), nLGN_1D)
                if nLearnFF_I > 0:
                    sLGN_all[-1, j, :] /= nI
                    radius_t[-1,:,j], on_center[:,-1,j], off_center[:,-1,j] = get_radius(1, sLGN_all[-1, j, :], nLGN, LGN_vpos, LGN_type, np.arange(nLGN))
                    contour_area[-1,:,j] = get_contour_area(1, sLGN_all[-1, j, :], nLGN, np.arange(nLGN), nLGN_1D)
                ctr_p[j] = stat.ttest_1samp(np.sqrt(contour_area[:-2,0,j]) - np.sqrt(contour_area[:-2,1,j]), 0).pvalue
                dis2c_p[j] = stat.ttest_1samp(radius_t[:-2,0,j] - radius_t[:-2,1,j], 0).pvalue
            fig = plt.figure('t_radius', figsize = (4,2*(1+nLearnFF_I)), dpi = 300)
            ax1 = fig.add_subplot(1, (1+nLearnFF_I), 1)
            if avgOnly:
                # radius
                ax1.plot(qt*dt/1000, radius_t[-2,0,:], '-r', label = 'on radius')
                ax1.plot(qt*dt/1000, radius_t[-2,1,:], '-b', label = 'off radius')
                # contour
                ax1.plot(qt*dt/1000, np.sqrt(contour_area[-2,0,:])/2, ':m', label = 'on sqrt(area)/2')
                ax1.plot(qt*dt/1000, np.sqrt(contour_area[-2,1,:])/2, ':c', label = 'off sqrt(area)/2')
                # std
                ax1.plot(qt*dt/1000, rstd_on, ':r', label = 'on g-std.')
                ax1.plot(qt*dt/1000, rstd_off, ':b', label = 'off g-std.')

            else:
                # radius
                ax1.plot(qt*dt/1000, radius_t[epick,0,:].mean(axis = 0), '-r', label = 'on radius', lw  =1)
                ax1.fill_between(qt*dt/1000, np.percentile(radius_t[epick,0,:], 10, axis = 0), np.percentile(radius_t[epick,0,:], 90, axis = 0), edgecolor = 'None', color = 'r', alpha = 0.5, lw = 0)
                ax1.plot(qt*dt/1000, radius_t[epick,1,:].mean(axis = 0), '-b', label = 'off radius', lw = 1)
                ax1.fill_between(qt*dt/1000, np.percentile(radius_t[epick,1,:], 10, axis = 0), np.percentile(radius_t[epick,1,:], 90, axis = 0), edgecolor = 'None', color = 'b', alpha = 0.5, lw = 0)
                # contour
                contour_length = np.sqrt(contour_area[epick,0,:])/2
                ax1.plot(qt*dt/1000, contour_length.mean(axis = 0), '-m', label = 'on sqrt(area)/2', lw = 1)
                ax1.fill_between(qt*dt/1000, np.percentile(contour_length, 10, axis = 0), np.percentile(contour_length, 90, axis = 0), edgecolor = 'None', color = 'm', alpha = 0.5, lw = 0)
                contour_length = np.sqrt(contour_area[epick,1,:])/2
                ax1.plot(qt*dt/1000, contour_length.mean(axis = 0), '-c', label = 'off sqrt(area)/2', lw = 1)
                ax1.fill_between(qt*dt/1000, np.percentile(contour_length, 10, axis = 0), np.percentile(contour_length, 90, axis = 0), edgecolor = 'None', color = 'c', alpha = 0.5, lw = 0)

                # std
                ax1.plot(qt*dt/1000, rstd_on, ':r', label = 'on g-std.')
                ax1.plot(qt*dt/1000, rstd_off, ':b', label = 'off g-std.')


            ax1.set_xlim(qt[0]*dt/1000, qt[-1]*dt/1000)
            if input_halfwidth[0] == input_halfwidth[1]:
                ax1.axhline(y = input_halfwidth[0], xmin = 0, xmax = stageFrame/nFrame, ls = ':', color = 'k', label = 'input HW')
            else:
                ax1.axhline(y = input_halfwidth[0], xmin = 0, xmax = stageFrame/nFrame, ls = ':', color = 'r', label = 'on HW')
                ax1.axhline(y = input_halfwidth[1], xmin = 0, xmax = stageFrame/nFrame, ls = ':', color = 'b', label = 'off HW')
            if nStage > 1:
                if input_halfwidth[2] == input_halfwidth[3]:
                    ax1.axhline(y = input_halfwidth[2], xmin = stageFrame/nFrame, xmax = 1, ls = ':', color = 'k')
                else:
                    ax1.axhline(y = input_halfwidth[2], xmin = stageFrame/nFrame, xmax = 1, ls = ':', color = 'r')
                    ax1.axhline(y = input_halfwidth[3], xmin = stageFrame/nFrame, xmax = 1, ls = ':', color = 'b')

            ax1.legend(fontsize = 'xx-small', loc='upper left', bbox_to_anchor=(1, 1))
            ax1.set_ylim(0, nLGN_1D/2)
            ax1.set_xlabel('time')
            ax1.set_ylabel('avg. dis to center')
            if avgOnly:
                ax1.set_title(f'on diff: ({radius_t[-2,0,0]-radius_t[-2,0,-1]:.1f}, {(np.sqrt(contour_area[-2,0,0]) - np.sqrt(contour_area[-2,0,-1]))/2:.1f}, {rstd_on[0] - rstd_on[-1]:.1f})\n off diff: ({radius_t[-2,1,0]-radius_t[-2,1,-1]:.1f}, {(np.sqrt(contour_area[-2,1,0]) - np.sqrt(contour_area[-2,1,-1]))/2:.1f}, {rstd_off[0] - rstd_off[-1]:.1f})')
            else:
                ax1.set_title(f'on diff: {radius_t[epick,0,0].mean()-radius_t[epick,0,-1].mean():.1f}, off diff: {radius_t[epick,1,0].mean()-radius_t[epick,1,-1].mean():.1f}')
            if nLearnFF_I > 0:
                ax2 = fig.add_subplot(1, 2*(1+nLearnFF_I), 3)
                if avgOnly:
                    # radius
                    ax2.plot(qt*dt/1000, radius_t[-1,0,:], '-r', label = 'on radius')
                    ax2.plot(qt*dt/1000, radius_t[-1,1,:], '-b', label = 'off radius')
                    # contour
                    ax2.plot(qt*dt/1000, np.sqrt(contour_area[-1,0,:])/2, ':m', label = 'on sqrt(area)/2')
                    ax2.plot(qt*dt/1000, np.sqrt(contour_area[-1,1,:])/2, ':c', label = 'off sqrt(area)/2')

                else:
                    # radius
                    ax2.plot(qt*dt/1000, radius_t[ipick,0,:].mean(axis = 0), '-r', label = 'on radius', lw = 1)
                    ax2.fill_between(qt*dt/1000, np.percentile(radius_t[ipick,0,:], 10, axis = 0), np.percentile(radius_t[ipick,0,:], 90, axis = 0), edgecolor = 'None', color = 'r', alpha = 0.5, lw = 0)
                    ax2.plot(qt*dt/1000, radius_t[ipick,1,:].mean(axis = 0), '-b', label = 'off radius', lw = 1)
                    ax2.fill_between(qt*dt/1000, np.percentile(radius_t[ipick,1,:], 10, axis = 0), np.percentile(radius_t[ipick,1,:], 90, axis = 0), edgecolor = 'None', color = 'b', alpha = 0.5, lw = 0)
                    # contour
                    contour_length = np.sqrt(contour_area[ipick,0,:])/2
                    ax2.plot(qt*dt/1000, np.sqrt(contour_length.mean(axis = 0)), '-m', label = 'on sqrt(area)/2', lw = 1)
                    ax2.fill_between(qt*dt/1000, np.percentile(contour_length, 10, axis = 0), np.percentile(contour_length, 90, axis = 0), edgecolor = 'None', color = 'm', alpha = 0.5, lw = 0)
                    contour_length = np.sqrt(contour_area[ipick,1,:])/2
                    ax2.plot(qt*dt/1000, np.sqrt(contour_length.mean(axis = 0)), '-c', label = 'off sqrt(area)/2', lw = 1)
                    ax2.fill_between(qt*dt/1000, np.percentile(contour_length, 10, axis = 0), np.percentile(contour_length, 90, axis = 0), edgecolor = 'None', color = 'c', alpha = 0.5, lw = 0)

                ax2.set_xlim(qt[0]*dt/1000, qt[-1]*dt/1000)

                if input_halfwidth[0] == input_halfwidth[1]:
                    ax2.axhline(y = input_halfwidth[0], xmin = 0, xmax = stageFrame/nFrame, color = ':k', label = 'input HW', lw = 1)
                else:
                    ax2.axhline(y = input_halfwidth[0], xmin = 0, xmax = stageFrame/nFrame, color = ':r', label = 'on HW', lw = 1)
                    ax2.axhline(y = input_halfwidth[1], xmin = 0, xmax = stageFrame/nFrame, color = ':b', label = 'off HW', lw = 1)
                if nStage > 1:
                    if input_halfwidth[2] == input_halfwidth[3]:
                        ax2.axhline(y = input_halfwidth[2], xmin = stageFrame/nFrame, xmax = 1, color = ':k', lw = 1)
                    else:
                        ax2.axhline(y = input_halfwidth[2], xmin = stageFrame/nFrame, xmax = 1, color = ':r', lw = 1)
                        ax2.axhline(y = input_halfwidth[3], xmin = stageFrame/nFrame, xmax = 1, color = ':b', lw = 1)

                ax2.set_xlabel('time')
                ax2.set_ylabel('avg. dis to center')
                ax2.set_ylim(0, nLGN_1D/2)
            fig.savefig(f'{fig_fdr}t_radius{osuffix}.png', bbox_inches = 'tight')
            radius_t[-2,:,:].tofile(fout)
            (np.sqrt(contour_area[-2,:,:])/2).tofile(fout)
            OnOff_balance.tofile(fout)
            OnOff_balance_p.tofile(fout)
            g_std_p.tofile(fout)
            dis2c_p.tofile(fout)
            ctr_p.tofile(fout)

            fout.close()
            if not avgOnly:
                fig = plt.figure('radius_dist', figsize = (10,3), dpi = 300)
                ax = fig.add_subplot(231) 
                #_, edges, _ = ax.hist(radius_t[epick,0,-1], bins = 20, color = 'r', alpha = 0.7)
                #ax.hist(radius_t[epick,0,-1], bins = edges, color = ['r','b'], alpha = 0.7)
                ax.hist(radius_t[epick,:,-1], bins = 20, color = ['r','b'], alpha = 0.7)
                ax = fig.add_subplot(234)
                ax.hist(radius_t[epick,0,-1] - radius_t[epick,1,-1], bins = 20, color = 'gray', alpha = 0.7)
                ax = fig.add_subplot(232) 
                #_, edges, _ = ax.hist(np.sqrt(contour_area[epick,0,-1])/2, bins = 20, color = 'r', alpha = 0.7)
                #ax.hist(np.sqrt(contour_area[epick,1,-1])/2, bins = edges, color = 'b', alpha = 0.7)
                ax.hist(np.sqrt(contour_area[epick,:,-1])/2, bins = 20, color = ['r','b'], alpha = 0.7)
                ax = fig.add_subplot(235)
                ax.hist((np.sqrt(contour_area[epick,0,-1]) - np.sqrt(contour_area[epick,1,-1]))/2, bins = 20, color = 'gray', alpha = 0.7)
                ax = fig.add_subplot(233) 
                #_, edges, _ = ax.hist(_rstd_on[epick,-1], bins = 20, color = 'r', alpha = 0.7)
                #ax.hist(_rstd_off[epick,-1], bins = edges, color = 'b', alpha = 0.7)
                ax.hist([_rstd_on[epick,-1],_rstd_off[epick,-1]], bins = 20, color = ['r','b'], alpha = 0.7)
                ax = fig.add_subplot(236)
                ax.hist(_rstd_on[epick,-1] - _rstd_off[epick,-1], bins = 20, color = 'gray', alpha = 0.7)
                fig.savefig(f'{fig_fdr}radius_dist{osuffix}.png', bbox_inches = 'tight')

        _V1_pick = np.append(V1_pick, nV1)
        if nLearnFF_I > 0:
            nsq = ns + 2
        else:
            nsq = ns + 1
        for iq in range(nsq):
            iV1 = _V1_pick[iq]
            sLGN = sLGN_all[iq,:,:]
            gmax = np.max(np.abs(sLGN))
            if gmax == 0:
                continue
            if doubleOnOff == 0:
                fig = plt.figure(f'sLGN_V1-{iV1}', figsize = (nit,(2 - mix)), dpi = 300)
                sLGN = np.reshape(sLGN, (nit, nLGN_1D, nLGN_1D))
                if mix:
                    for i in range(nit):
                        ax = fig.add_subplot(1,nit + 1,i + 1)
                        stmp = sLGN[i,:,:]
                        offPick = LGN_type == 5
                        stmp[offPick] = - stmp[offPick]
                        local_max = np.max(np.abs(stmp))
                        stmp = stmp / gmax
                        stmp[np.abs(stmp) < local_max / gmax * thres_out] = 0
                        ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('gray'))

                        ax.set_title(f't{float(qt[i]) / nt * 100:.0f}%-n{np.sum(np.sum(stmp > 0))}-p{gmax / gmaxLGN[0] * 100:.0f}%', fontsize = 6)
                        if i == nit-1:
                            ax = fig.add_subplot(1,nit + 1,nit + 1)
                            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('gray'))
                            ax.set_xticks([])
                            ax.set_yticks([])
                            plt.colorbar()
                            if iV1 < nV1:
                                ax.set_title(f'{orient[iV1] * 180 / np.pi:.0f}deg {fr[iV1]:.2f}Hz')
                            else:
                                ax.set_title(f'total bias: {np.mean(orient * 180 / np.pi):.0f}deg {np.mean(fr):.2f}Hz')
                else:
                    for itype in range(ntype):
                        for i in range(nit):
                            ax = fig.add_subplot(ntype,nit + 1, itype * (nit + 1) + i + 1)
                            stmp = sLGN[i,:,:]
                            stmp[LGN_type != types[itype]] = 0
                            local_max = np.max(np.abs(stmp))
                            stmp = stmp / gmax
                            ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('gray'))
                            if itype == 0:
                                ax.set_title(f't{float(qt[i]) / nt * 100:.0f}%-n{np.sum(np.sum(stmp > 0))}-p{gmax / gmaxLGN[0] * 100:.0f}%', fontsize = 6)
                            if i == 0:
                                ax.set_ylabel(f'type: {types[itype]}')
                            if i == nit-1:
                                ax = plt.subplot(1,nit + 1, nit + 1)
                                im = imagesc(stmp,clims)
                                ax.set_xticks([])
                                ax.set_yticks([])
                                plt.colorbar()
                                if iV1 < nV1:
                                    ax.set_title(f'{orient[iV1] * 180 / np.pi:.0f}deg {fr[iV1]:.2f}Hz')
                                else:
                                    ax.set_title(f'total bias: {np.mean(orient * 180 / np.pi):.0f}deg {np.mean(fr):.2f}Hz')
            else:
                fig = plt.figure(f'sLGN_V1-{iV1}-sep', figsize = (nit0,ntype * nrow), dpi = 300)
                sLGN = np.reshape(sLGN, (nit, nLGN_1D, nLGN_1D, 2))
                for itype in range(ntype):
                    row = 0
                    for i in range(nit):
                        iplot = row * ntype * (nit0 + 1) + itype * (nit0 + 1)
                        if i >= nit0:
                            iplot += int(np.mod(i, nit0)) + 1
                        else:
                            iplot += i + 1
                        ax = fig.add_subplot(nrow * ntype, nit0 + 1, iplot)
                        stmp0 = sLGN[i,:,:,itype]
                        local_max = np.max(np.abs(stmp0))
                        if use_local_max == 1:
                            stmp = stmp0 / gmax
                        else:
                            stmp = stmp0 / gmaxLGN[0]

                        ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('gray'))
                        if itype == 0:
                            mc = '*r'
                            cc = ':r'
                            lc = 'r'
                            qcenter = on_center
                        else:
                            mc = '*b'
                            cc = ':b'
                            lc = 'b'
                            qcenter = off_center
                        if avgOnly:
                            radius = radius_t[iq,itype,i]
                            center = qcenter[:,iq,i]
                        else:
                            radius = radius_t[iV1,itype,i]
                            center = qcenter[:,iV1,i]

                        center = center/xspan * nLGN_1D + nLGN_1D/2
                        radius = radius/xspan * nLGN_1D
                        nc = 20
                        circle = np.array([[radius * np.cos(2*np.pi/nc*ic) for ic in range(nc)], [radius * np.sin(2*np.pi/nc*ic) for ic in range(nc)]])
                        ax.plot(center[0]-0.5, center[1]-0.5, mc)
                        ax.plot(center[0]-0.5 + circle[0,:], center[1]-0.5 + circle[1,:], cc, lw = 1.0, alpha = 0.7)
                        _stmp = stmp/stmp.max()
                        if (_stmp > 0.5).all():
                            level = (_stmp.max() + _stmp.min())/2
                        else:
                            level = 0.5
                        try:
                            ax.contour(_stmp, levels = [level], linewidths = 0.5, colors = lc)
                        except UserWarning:
                            print(_stmp, i, iq, avgOnly, level)

                        ax.set_yticks([])
                        ax.set_xticks([])
                        ax.set_aspect('equal')
                        local_nCon = np.sum(np.sum(stmp0 >= thres_out * gmaxLGN[0]))
                        if itype == 0:
                            ax.set_title(f't{float(qt[i]) / nt * 100:.0f}%-n{local_nCon}-p{local_max / gmaxLGN[0] * 100:.0f}%', fontsize = 5)
                        else:
                            ax.set_title(f'n{local_nCon}-p{local_max / gmaxLGN[0] * 100:.0f}%', fontsize = 5)
                        if np.mod(i,nit0) == 0:
                            ax.set_ylabel(f'type: {types[itype]}')
                        if i == nit-1:
                            ax = fig.add_subplot(nrow * ntype, nit0 + 1, iplot + 1)
                            stmp = stmp0 / gmax
                            stmp[stmp < thres_out] = 0
                            im = ax.imshow(stmp, aspect = 'equal', origin = 'lower', cmap = plt.get_cmap('gray'))
                            if itype == 0:
                                mc = '*r'
                                cc = ':r'
                                lc = 'r'
                                qcenter = on_center
                            else:
                                mc = '*b'
                                cc = ':b'
                                lc = 'b'
                                qcenter = off_center
                            if avgOnly:
                                radius = radius_t[iq,itype,i]
                                center = qcenter[:,iq,i]
                            else:
                                radius = radius_t[iV1,itype,i]
                                center = qcenter[:,iV1,i]
                            
                            center = center/xspan * nLGN_1D + nLGN_1D/2
                            radius = radius/xspan * nLGN_1D
                            nc = 20
                            circle = np.array([[radius * np.cos(2*np.pi/nc*ic) for ic in range(nc)], [radius * np.sin(2*np.pi/nc*ic) for ic in range(nc)]])
                            ax.plot(center[0]-0.5, center[1]-0.5, mc)
                            ax.plot(center[0]-0.5 + circle[0,:], center[1]-0.5 + circle[1,:], cc, lw = 1.0, alpha = 0.7)
                            _stmp = stmp/stmp.max()
                            if (_stmp > 0.5).all():
                                level = (_stmp.max() + _stmp.min())/2
                            else:
                                level = 0.5
                            ax.contour(_stmp, levels = [level], linewidths = 0.5, colors = lc)

                            pos_bbox = ax.get_position()
                            ax.set_yticks([])
                            ax.set_xticks([])
                            ax.set_aspect('equal')
                            ax.set_title(f'{local_nCon:.0f}', fontsize = 5)

                            pos = pos_bbox.get_points()
                            bar_width = (pos[1,0] - pos[0,0]) * 0.15
                            pad = bar_width
                            cax = plt.axes([pos[1,0] + pad, pos[0,1], bar_width, pos[1,1] - pos[0,1]])
                            plt.colorbar(im, cax = cax)

                        if np.mod(i+1,nit0) == 0:
                            row = row + 1

            if iV1 < nV1:
                if mix and doubleOnOff != 1:
                    fig.savefig(f'{fig_fdr}sLGN_V1-{iV1}{osuffix}-mix{rtime}.png')
                else:
                    fig.savefig(f'{fig_fdr}sLGN_V1-{iV1}{osuffix}-sep{rtime}.png')
            else:
                if mix and doubleOnOff != 1:
                    fig.savefig(f'{fig_fdr}avg_sLGN_V1{osuffix}-mix{rtime}.png')
                else:
                    fig.savefig(f'{fig_fdr}avg_sLGN_V1{osuffix}-sep{rtime}.png')
            plt.close(fig)
    print('sLGN finished')

    if st == 2 or st == 0:
        tstep = int(np.round(range_nt / nstep))
        it = np.arange(step0, nt_, tstep)
        nstep = it.size

        if examLTD:
            lAvg = np.zeros((nstep, nV1))
            data = np.zeros(nV1)
            with open(f_dsLGN, 'rb') as f:
                f.seek(9*4, 1)
                A_LTP = np.fromfile(f, 'f4', nLearnFF)*sRatio
                rLTD_E = np.fromfile(f, 'f4', nLearnFF_E)
                rLTD_I = np.fromfile(f, 'f4', nLearnFF_I)
                targetFR = np.fromfile(f, 'f4', 2)
                learnData_FF = np.fromfile(f, 'i4', 1)[0]
                print(f'rLTD = {rLTD_E, rLTD_I}, targetFR = {targetFR}, A_LTP = {A_LTP}') 
                if learnData_FF >= 2:
                    with open(learnDataFn, 'rb') as fq: 
                        fq.seek(10*4, 1)
                        dskip = 0
                        dskip += nLGN*4 # LGN_sInfo
                        dskip += nLGN*nLearnTypeFF*4 # vLTP
                        dskip += 2*nV1*4 # gFF, V1_sInfo
                        dskip += max_LGNperV1*nV1*4 # LGN_V1_s
                        dskip += 2*nLearnFF_E*nE*4 # vLTD_E, vTripE
                        dskip += 2*nLearnFF_I*nI*4 # vLTD_I, vTripI
                        fq.seek(dskip * step0, 1)
                        if nLearnFF_E > 0:
                            dataE = np.fromfile(fq, 'f4', 2*nE).reshape(nE, 2)
                            data[epick] = dataE[:,0]
                        if nLearnFF_I > 0:
                            dataI = np.fromfile(fq, 'f4', nI)
                            data[ipick] = dataI
                        lAvg[0,:] = data
                        for j in range(1,nstep):
                            fq.seek(dskip * (tstep - 1), 1)
                            if nLearnFF_E > 0:
                                dataE = np.fromfile(fq, 'f4', 2*nE).reshape(nE, 2)
                                data[epick] = dataE[:,0]
                            if nLearnFF_I > 0:
                                dataI = np.fromfile(fq, 'f4', nI)
                                data[ipick] = dataI
                            lAvg[j,:] = data
                else:
                    f.seek((2*nE + nI) * step0 * 4, 1)
                    if nLearnFF_E > 0:
                        dataE = np.fromfile(f, 'f4', 2*nE).reshape(nE, 2)
                        data[epick] = dataE[:,0]
                    if nLearnFF_I > 0:
                        dataI = np.fromfile(f, 'f4', nI)
                        data[ipick] = dataI
                    lAvg[0,:] = data
                    for j in range(1,nstep):
                        f.seek((2*nE + nI) * (tstep - 1) * 4, 1)
                        if nLearnFF_E > 0:
                            dataE = np.fromfile(f, 'f4', 2*nE).reshape(nE, 2)
                            data[epick] = dataE[:,0]
                        if nLearnFF_I > 0:
                            dataI = np.fromfile(f, 'f4', nI)
                            data[ipick] = dataI
                        lAvg[j,:] = data

            LTD = np.zeros((nstep, nV1))
            avgFR = np.zeros((nstep, nV1))
            avgFR[:,epick] = lAvg[:,epick]
            if nLearnFF_E > 0:
                LTD[:,epick] = np.power(lAvg[:,epick],2)*rLTD_E[0]
            avgFR[:,ipick] = lAvg[:,ipick]
            if nLearnFF_I > 0:
                LTD[:,ipick] = np.power(lAvg[:,ipick],2)*rLTD_I[0]

            fig = plt.figure('t_LTD', figsize = (10,4*(1+nLearnFF_I)), dpi = 300)
            ax = fig.add_subplot(1, 2*(1+nLearnFF_I), 1)
            ax.imshow(LTD[:,epick].T/np.max(LTD[:,epick]), aspect = 'auto', origin = 'lower', cmap = plt.get_cmap('Reds'))
            ax.set_xlabel('time step')
            ax.set_ylabel('neuron ID')
            ax = fig.add_subplot(1, 2*(1+nLearnFF_I), 2)
            ax.plot(it * dt/1000, LTD[:,epick].T.mean(axis = 0), '-r', lw = 0.1, alpha = 0.7)
            ax.plot(it * dt/1000, LTD[:,epick].T.std(axis = 0), '-m', alpha = 0.7, lw = 0.1)
            ax.axhline(A_LTP, ls = ':', color = 'r', alpha = 0.7, lw = 0.1)
            ax.set_xlabel('time')
            ax.set_ylabel('LTD')
            if nLearnFF_I > 0:
                ax = fig.add_subplot(1, 2*(1+nLearnFF_I), 3)
                ax.imshow(LTD[:,ipick].T/np.max(LTD[:,ipick]), aspect = 'auto', origin = 'lower', cmap = plt.get_cmap('Blues'))
                ax.set_ylabel('neuron ID')
                ax.set_xlabel('time step')
                ax = fig.add_subplot(1, 2*(1+nLearnFF_I), 4)
                ax.plot(it * dt/1000, LTD[:,ipick].T.mean(axis = 0), '-b', lw = 0.1, alpha = 0.7)
                ax.plot(it * dt/1000, LTD[:,ipick].T.std(axis = 0), '-c', alpha = 0.7, lw = 0.1)
                ax.axhline(A_LTP, ls = ':', color = 'b', alpha = 0.7, lw = 0.1)
                ax.set_xlabel('time')
                ax.set_ylabel('LTD')
            fig.savefig(f'{fig_fdr}t_LTD{osuffix}.png')

        qtt = np.floor(np.linspace(0, nstep-1, nit)).astype('i4')
        tLGN_all = np.zeros((ns, nstep, max_LGNperV1))
        with open(f_sLGN, 'rb') as f:
            f.seek(nskip * 4, 1)
            f.seek(max_LGNperV1 * nV1 * step0 * 4, 1)
            data = np.fromfile(f, 'f4', nV1*max_LGNperV1).reshape(max_LGNperV1, nV1).T
            tLGN_all[:,0,:] = data[V1_pick, :]
            for j in range(1,nstep):
                f.seek(max_LGNperV1 * nV1 * (tstep - 1) * 4, 1)
                data = np.fromfile(f, 'f4', nV1*max_LGNperV1).reshape(max_LGNperV1, nV1).T
                tLGN_all[:,j,:] = data[V1_pick, :]

        for iq in range(ns):
            iV1 = V1_pick[iq]
            tLGN = tLGN_all[iq,:,:]
            gmax = np.max(tLGN)
            if gmax == 0:
                continue
            if use_local_max != 1:
                gmax = gmaxLGN[0]
            gmin = np.min(tLGN)
            if examSingle:
                fig = plt.figure(f'tLGN_V1_single-{iV1}', figsize = (8,8), dpi = 300)
                for i in range(nLGN_V1[iV1]):
                    ip = LGN_V1_ID[iV1,i]
                    this_type = np.mod(LGN_type[ip],2)
                    sat = (tLGN[-1, i] - gmin) / gmax
                    val = 1.0
                    ip += 1
                    if this_type == 0:
                        hue = 0
                        ip = 2*(ip//(2*nLGN_1D)) * nLGN_1D + np.mod((ip+1)//2-1, nLGN_1D) + 1
                    else:
                        hue = 2 / 3
                        ip = (2*((ip+2*nLGN_1D-1)//(2*nLGN_1D))-1) * nLGN_1D + np.mod(ip//2-1, nLGN_1D) + 1
                    hsv = np.array([hue,sat,val]).T
                    ax = fig.add_subplot(2 * nLGN_1D, nLGN_1D, ip)
                    ax.plot(it * dt, tLGN[:,i] / gmax * 100, '-', color = clr.hsv_to_rgb(hsv))
                    ax.set_ylim(0, 100)
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.savefig(f'{fig_fdr}tLGN_V1_single-{iV1}{osuffix}{rtime}.png')
                plt.close(fig)

            fig = plt.figure(f'tLGN_V1-{iV1}', figsize = (8,9), dpi = 500)
            if LGN_switch:
                if examLTD:
                    ax = plt.subplot(31,3,(3*15 + 1, 3*15 + 2))
                else:
                    ax = plt.subplot(21,3,(3*10 + 1, 3*10 + 2))
                status_t = 0
                for i in range(nStatus):
                    current_nt = np.round(statusDur[i] * 1000 / dt)
                    current_t = np.arange(current_nt) * dt
                    ax.plot(status_t + current_t, np.zeros(current_nt) + reverse[i], 'k')
                    status_t = status_t + statusDur * 1000

                ax.set_xticks([])
                ax.set_yticks([])
            for i in range(ntype):
                if examLTD:
                    ax = fig.add_subplot(ntype+1, 3, (3*i+1, 3*i+2))
                else:
                    ax = fig.add_subplot(ntype, 3, (3*i+1, 3*i+2))
                ax.plot(it * dt/1000, tLGN[:,LGN_type[LGN_V1_ID[iV1, :nLGN_V1[iV1]]] == types[i]] / gmax * 100, '-', alpha = 0.5)
                ax.set_title(f'type{types[i]} input activation level {typeInput[i] * 100:.1f}%')
                ax.set_ylim(0,100)
                ax.set_ylabel('strength % of max')
                if i == ntype-1 and not examLTD:
                    ax.set_xlabel('s')
            edges = np.linspace(0,100,nbins)
            if examLTD:
                ax = fig.add_subplot(ntype+1, 3, (3*ntype+1, 3*ntype+2))
                ax.plot(it * dt/1000, LTD[:,i]/A_LTP[0]*100 , ':k', lw = 0.5, alpha = 0.5)
                ax.set_ylabel('LTD / LTP ratio %')
                ax2 = ax.twinx()
                ax2.plot(it * dt/1000, avgFR[:,i],'-b', lw = 0.5, alpha = 0.5)
                ax2.plot(it[-1] * dt/1000, targetFR[0], '*r', alpha = 0.5, label = 'target FR')
                ax2.set_ylabel('avg. filtered FR (Hz)')
                ax.set_xlabel('time (s)')
            cmap = plt.get_cmap('Set1')
            for i in range(nit):
                ax0 = fig.add_subplot(nit, 9, 9*i+7)
                ax = fig.add_subplot(nit, 9, (9*i+8, 9*i+9))
                ss = []
                for j in range(ntype):
                    data = tLGN[qtt[i], LGN_type[LGN_V1_ID[iV1,:nLGN_V1[iV1]]] == types[j]]
                    ss.append(data[data > 0] / gmax * 100)
                    ax0.bar(j, sum(data==0), alpha = 0.5, color = cmap(j/ntype))
                ax.hist(ss, bins = edges, alpha = 0.5, color = [cmap(j/ntype) for j in range(ntype)])
                if i == nit-1:
                    ax.set_xlabel(f'strength/max_cap(%), on:off= {onS[iV1] / offS[iV1]:.1f}')
                    ax0.set_xticks([0,1], labels = ['on', 'off'])
                else:
                    ax.set_xticks([])
                    ax0.set_xticks([])
                ax.set_yticks([])
                ax0.set_yticks([])
                ax0.set_ylim([0, max_LGNperV1/2])

            fig.tight_layout = True
            fig.savefig(f'{fig_fdr}tLGN_V1-{iV1}{osuffix}{rtime}.png')
            plt.close(fig)
    print('tLGN finished')
    return
    
def determine_os_str(pos, ids, types, s, n, m, min_dis, os_out, nLGN_1D): 
    orient = np.zeros(n)
    overlap = np.zeros(n)
    os = np.zeros(n)
    onS = np.zeros(n)
    offS = np.zeros(n)
    for i in range(n):
        all_id = ids[i, :m[i]]
        all_type = types[all_id]
        all_s = s[i,:m[i]]
        sLGN = np.zeros(nLGN_1D * nLGN_1D * 2)
        sLGN[all_id] = all_s
        sLGN = np.reshape(sLGN, (nLGN_1D, nLGN_1D * 2))
        on_s = sLGN[:,0:nLGN_1D * 2:2]
        off_s = sLGN[:,1:nLGN_1D * 2:2]
        overlap_pick = np.logical_and(on_s > 0,off_s > 0)
        if overlap_pick.any():
            max_s = np.maximum(on_s, off_s)
            min_s = np.minimum(on_s, off_s)
            smaller_size = np.min([np.sum(on_s > 0),np.sum(off_s > 0)])
            overlap[i] = np.sum(min_s[overlap_pick] / max_s[overlap_pick]) / smaller_size
        else:
            overlap[i] = 0
        on_s = all_s[all_type == 4]
        on_id = all_id[all_type == 4]
        sPick = on_s >= np.max(on_s) * os_out
        onPick = on_id[sPick]
        onS[i] = np.sum(on_s)
        on_pos = np.mean(pos[:,onPick], axis = 1)
        off_s = all_s[all_type == 5]
        off_id = all_id[all_type == 5]
        sPick = off_s >= np.max(off_s) * os_out
        offPick = off_id[sPick]
        offS[i] = np.sum(off_s)
        off_pos = np.mean(pos[:,offPick], axis = 1)
        dis_vec = np.array([on_pos[0] - off_pos[0], off_pos[1] - on_pos[1]])
        on_off_dis = np.sqrt(np.sum(dis_vec * dis_vec))
        if on_off_dis <= min_dis / 2:
            orient[i] = np.nan
            os[i] = 0
        else:
            orient[i] = np.arctan2(off_pos[1] - on_pos[1],on_pos[0] - off_pos[0])
            if orient[i] < 0:
                orient[i] += 2 * np.pi
            proj = dis_vec / on_off_dis
            if onPick.size > 1:
                rel_pos = pos[:,onPick] - on_pos.reshape(2,1)
                on_dis = np.sqrt(np.sum(rel_pos*rel_pos, axis = 0))
                pick = on_dis > 0
                cos_on = np.zeros(onPick.size)
                cos_on[pick] = np.sum((rel_pos[:,pick] / on_dis[pick]) * proj.reshape(2,1), axis = 0)
                proj_on = on_dis*cos_on
                max_on_p = np.max(proj_on[proj_on >= 0])
                max_on_m = np.max(np.abs(proj_on[proj_on <= 0]))
                r_on = (max_on_p + max_on_m) * max_on_p / max_on_m
                if np.isnan(r_on):
                    r_on = min_dis / 2
            else:
                r_on = min_dis / 2
            if offPick.size > 1:
                rel_pos = pos[:,offPick] - off_pos.reshape(2,1)
                off_dis = np.sqrt(np.sum(rel_pos * rel_pos, axis = 0))
                pick = off_dis > 0
                cos_off = np.zeros(offPick.size)
                cos_off[pick] = np.sum((rel_pos[:,pick] / off_dis[pick]) * proj.reshape(2,1), axis = 0)
                proj_off = off_dis*cos_off
                max_off_p = np.max(proj_off[proj_off >= 0])
                max_off_m = np.max(np.abs(proj_off[proj_off <= 0]))
                r_off = (max_off_p + max_off_m) * (max_off_p / max_off_m)
                if np.isnan(r_off):
                    r_off = min_dis / 2
            else:
                r_off = min_dis / 2
            os[i] = on_off_dis / (r_on + r_off)
    
    return onS, offS, os, overlap, orient
    
def get_radius(n, s, m, pos, types ,ids):
    if n == 1:
        s = np.reshape(s, (1,*s.shape))
        ids = np.reshape(ids, (1,*ids.shape))
        m = np.array([m])
    radius = np.zeros((n,2)) - 1
    on_center = np.zeros((2,n))
    off_center = np.zeros((2,n))
    for i in range(n):
        if m[i] > 0:
            all_id = ids[i, :m[i]]
            all_type = types[all_id]
            all_s = s[i,:m[i]]
            on_s = all_s[all_type == 4]
            off_s = all_s[all_type == 5]

            if (on_s > 0).any():
                on_id = all_id[all_type == 4]
                weights = on_s.copy()
                pick = weights < on_s.max()/2
                weights[pick] = 0
                on_center[:,i] = np.average(pos[:,on_id], axis = 1, weights = weights)
                radius[i, 0] = np.average(np.linalg.norm(pos[:,on_id].T - on_center[:,i], axis = 1), weights = weights)

            if (off_s > 0).any():
                off_id = all_id[all_type == 5]
                weights = off_s.copy()
                pick = weights < off_s.max()/2
                weights[pick] = 0
                off_center[:,i] = np.average(pos[:,off_id], axis = 1, weights = weights)
                radius[i, 1] = np.average(np.linalg.norm(pos[:,off_id].T - off_center[:,i], axis = 1), weights = weights)
        
    if n == 1:
        return radius.reshape(2), on_center.reshape(2), off_center.reshape(2)
    else:
        return radius, on_center, off_center

#def vertices_to_area(vertices):
#    print(vertices.shape)
#    if 
#    for v in vertices:
#        print(v)
        
def get_area(x, y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def point_inside_shape(point, shape):
    rel_vec = shape - point
    dcos = rel_vec[:-1,0]*rel_vec[1:,0] + rel_vec[:-1,1]*rel_vec[1:,1]
    dsin = rel_vec[:-1,0]*rel_vec[1:,1] - rel_vec[:-1,1]*rel_vec[1:,0]
    dangle = np.arctan2(dsin, dcos)
    if sum(dangle) > np.pi:
        return True
    else:
        return False

def get_contour_area(n, z, m, ids, l, level = 0.2):
    if n == 1:
        z = np.reshape(z, (1, *z.shape))
        ids = np.reshape(ids, (1,*ids.shape))
        m = np.array([m])
    area = np.zeros((n,2))
    for i in range(n):
        if m[i] > 0 and (z[i,:] > 0).any():
            data = np.zeros(l*l*2)
            data[ids[i, :m[i]]] = z[i,:]
            data = data.reshape(l, l, 2)
            for j in range(2):
                normed = data[:,:,j]/data[:,:,j].max()
                nrow = normed.shape[0]
                ncol = normed.shape[1]
                normed = np.hstack((np.zeros((nrow,1)), normed, np.zeros((nrow,1))))
                normed = np.vstack((np.zeros((1,ncol+2)), normed, np.zeros((1,ncol+2))))
                assert((normed[:,0] == 0).all())
                assert((normed[:,-1] == 0).all())
                assert((normed[0,:] == 0).all())
                assert((normed[-1,:] == 0).all())
                quadset = ctr.contour_generator(z = normed)
                shapes = quadset.lines(level)
                nshape = len(shapes)
                a = np.zeros(nshape)
                for k in range(nshape):
                    shape = shapes[k]
                    # assert closed shape
                    assert(shape.shape[0] >= 3)
                    assert(shape.shape[1] == 2)
                    assert(shape[0,0] == shape[-1,0])
                    assert(shape[0,1] == shape[-1,1])
                    a[k] = get_area(shape[:-1,0], shape[:-1,1])

                idx = (-a).argsort()
                area[i,j] = a[idx[0]]
                for k in range(1,nshape):
                    inside = False
                    for ii in range(k):
                        if point_inside_shape(shapes[k][0,:], shapes[ii]):
                            inside = True
                            break
                    if inside:
                        area[i,j] -= a[idx[k]]
                    else:
                        area[i,j] += a[idx[k]]

    if n == 1:
        return area.reshape(2)
    else:
        return area
    #return contour, area

if __name__ == '__main__':
    if len(sys.argv) < 14:
        print("outputLearnFF(seed, isuffix0, isuffix, osuffix, res_fdr, setup_fdr, data_fdr, fig_fdr, LGN_switch, mix, st, examSingle, use_local_max, waveStage, ns, examLTD)")
        print('no default argument available')
    else:
        seed = int(sys.argv[1])
        isuffix0 = sys.argv[2]
        isuffix = sys.argv[3]
        osuffix = sys.argv[4]
        res_fdr = sys.argv[5]
        setup_fdr = sys.argv[6]
        data_fdr = sys.argv[7]
        fig_fdr = sys.argv[8]
        if sys.argv[9] == 'True' or sys.argv[9] == 'true' or sys.argv[9] == '1':
            LGN_switch = True
        else:
            LGN_switch = False

        if sys.argv[10] == 'True' or sys.argv[10] == 'true' or sys.argv[10] == '1':
            mix = 1 
        else:
            mix = 0 

        st = int(sys.argv[11])
        if sys.argv[12] == 'True' or sys.argv[12] == 'true' or sys.argv[12] == '1':
            examSingle = True
        else:
            examSingle = False
        if sys.argv[13] == 'True' or sys.argv[13] == 'true' or sys.argv[13] == '1':
            use_local_max = True
        else:
            use_local_max = False
        stage = int(sys.argv[14])

        if len(sys.argv) > 15:
            ns = int(sys.argv[15])
            if len(sys.argv) > 16:
                if sys.argv[16] == 'True' or sys.argv[16] == 'true' or sys.argv[16] == '1':
                    examLTD = True
                else:
                    examLTD = False
            else:
                examLTD = False
        else:
            ns = 10


    outputLearnFF(seed, isuffix0, isuffix, osuffix, res_fdr, setup_fdr, data_fdr, fig_fdr, LGN_switch, mix, st, examSingle, use_local_max, stage, ns, examLTD)
