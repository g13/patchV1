import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stat
import sys

def random_trials_plot(data_fdr, fig_fdr, suffix, separate, use_idx, num, batch):
    if fig_fdr[-1] != '/':
        fig_fdr = fig_fdr+'/'
    if data_fdr[-1] != '/':
        data_fdr = data_fdr+'/'
    totalFr_binned = True
    if not separate:
        if not use_idx:
            num = []
            for key, value in batch.items():
                num += [f'{key}{i}' for i in range(0,value)]
        n = len(num)
        print(f'{n} trials')

        for i in range(n):
            with open(f'{data_fdr}metric-{suffix}{num[i]}.bin', 'rb') as f:
                if i == 0:
                    step0, nt_, nit, stage, nLGN_1D = np.fromfile(f, 'int', 5)
                    dt = np.fromfile(f, 'f4', 1)[0]
                    rstd = np.zeros((2,n,nit))
                    rstd_norm = np.zeros((2,n,nit))
                    max20 = np.zeros((2,n,nit))
                    min0 = np.zeros((2,n,nit))
                    radius = np.zeros((2,n,nit))
                    cl = np.zeros((2,n,nit))
                    OnOff_balance = np.zeros((n, nit))
                else:
                    _step0, _nt_, _nit, _stage, _nLGN_1D = np.fromfile(f, 'int', 5)
                    _dt = np.fromfile(f, 'f4', 1)[0]
                    assert(_nit == nit)
                    assert(_stage == stage)
                    assert(_nLGN_1D == nLGN_1D)

                # input_halfwidth
                if stage == 2 or stage == 3:
                    _ = np.fromfile(f, 'f4', 2) 
                else:
                    assert(stage == 5)
                    _ = np.fromfile(f, 'f4', 2*2)[:2]
                print(f'{data_fdr}metric-{suffix}{num[i]}.bin')
                _ = np.fromfile(f, 'f8', nLGN_1D * nLGN_1D).reshape(nLGN_1D, nLGN_1D) # initial_dist
                rstd[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                rstd_norm[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                max20[:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
                min0[:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
                radius[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                cl[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                OnOff_balance[i,:] = np.fromfile(f, 'f8', nit)
            if totalFr_binned:
                frFn = f'{data_fdr}V1_totalFr-{suffix}{num[i]}.bin'
                try:
                    with open(frFn, 'rb') as f:
                        if i == 0:
                            nt = np.fromfile(f, 'i8', 1)[0]
                            print(nt)
                            ft = np.fromfile(f, 'f8', nt)
                            fr = np.zeros((n, nt))
                        else:
                            _nt = np.fromfile(f, 'i8', 1)[0]
                            _ft = np.fromfile(f, 'f8', _nt)
                            assert(_nt == nt)
                            assert((_ft - ft).any() == False)
                        fr[i,:] = np.fromfile(f, 'f8', nt)
                except FileNotFoundError:
                    print(f"file not found: {frFn}, no firing rate plots")
                    totalFr_binned = False

        t = np.arange(nit)
        ylim = [0, nLGN_1D/2 + 0.5]
        fig = plt.figure('temporal_plots', figsize = (14,5), dpi = 200)
        ax = fig.add_subplot(241)
        fill_between(t, rstd, ax, 2, 'g. std', ylim = ylim)
        ax = fig.add_subplot(242)
        fill_between(t, rstd_norm, ax, 2, 'g. std by cell', ylim = ylim)
        ax = fig.add_subplot(243)
        fill_between(t, radius, ax, 2, 'avg. dis2c', ylim = ylim)
        ax = fig.add_subplot(244)
        if not totalFr_binned:
            fill_between(t, cl, ax, 2, 'sqrt(area)/2', ylim = ylim, xlabel = 't (au)')
        else:
            fill_between(t, cl, ax, 2, 'sqrt(area)/2', ylim = ylim)
        ax = fig.add_subplot(245)
        fill_between(t, OnOff_balance, ax, 1, 'On-Off', xlabel = 't (au)')
        ax = fig.add_subplot(246)
        fill_between(t, max20, ax, 2, '# top 20%', ylim = [0, nLGN_1D*nLGN_1D], xlabel = 't (au)')
        ax = fig.add_subplot(247)
        fill_between(t, min0, ax, 2, '# disconnected', ylim = [0, nLGN_1D*nLGN_1D], xlabel = 't (au)')
        if totalFr_binned:
            ax = fig.add_subplot(248)
            fill_between(ft/1000, fr, ax, 1, 'avg. FR', xlabel = 't (s)')
        plt.tight_layout(w_pad = 1)
        fig.savefig(fig_fdr + 'temporal_plots-' + suffix +'.png', bbox_inches = 'tight')

        tpoint = -1
        fig = plt.figure('end_dist', figsize = (14,8), dpi = 200)
        ax = fig.add_subplot(241)
        hist(rstd[:,:,tpoint], ax, 2, 'g. std')
        ax.text(0.1, 0.9, f'pvalue: {sig(rstd[:,:,tpoint],2):.3e}', transform = ax.transAxes)
        ax = fig.add_subplot(242)
        hist(rstd_norm[:,:,tpoint], ax, 2, 'g. std by cell')
        ax.text(0.1, 0.9, f'pvalue: {sig(rstd_norm[:,:,tpoint],2):.3e}', transform = ax.transAxes)
        ax = fig.add_subplot(243)
        hist(radius[:,:,tpoint], ax, 2, 'avg. dis2c')
        ax.text(0.1, 0.9, f'pvalue: {sig(radius[:,:,tpoint],2):.3e}', transform = ax.transAxes)
        ax = fig.add_subplot(244)
        hist(cl[:,:,tpoint], ax, 2, 'sqrt(area)/2')
        ax.text(0.1, 0.9, f'pvalue: {sig(cl[:,:,tpoint],2):.3e}', transform = ax.transAxes)
        ax = fig.add_subplot(245)
        hist(OnOff_balance[:,tpoint], ax, 1, 'On-Off')
        ax.text(0.1, 0.9, f'pvalue: {sig(OnOff_balance[:,tpoint],1):.3e}', transform = ax.transAxes)
        ax = fig.add_subplot(246)
        hist(max20[:,:,tpoint], ax, 2, '# top 20%')
        ax.text(0.1, 0.9, f'pvalue: {sig(max20[:,:,tpoint],2):.3e}', transform = ax.transAxes)
        ax = fig.add_subplot(247)
        hist(min0[:,:,tpoint], ax, 2, '# disconnected')
        ax.text(0.1, 0.9, f'pvalue: {sig(min0[:,:,tpoint],2):.3e}', transform = ax.transAxes)
        if totalFr_binned:
            ax = fig.add_subplot(248)
            hist(fr[:,tpoint-1], ax, 1, 'avg. FR')
        fig.savefig(fig_fdr + 'end_dist-' + suffix +'.png', bbox_inches = 'tight')
    else:
        nbatch = len(batch)
        keys = batch.keys()
        rstd = dict({})
        rstd_norm = dict({})
        max20 = dict({})
        min0 = dict({})
        radius = dict({})
        cl = dict({})
        OnOff_balance = dict({})
        if totalFr_binned:
            ft = dict({})
            fr = dict({})
        for key, ntrial in batch.items():
            for i in range(ntrial):
                with open(f'{data_fdr}metric-{suffix}{key}{i}.bin', 'rb') as f:
                    if i == 0:
                        step0, nt_, nit, stage, nLGN_1D = np.fromfile(f, 'int', 5)
                        dt = np.fromfile(f, 'f4', 1)[0]
                        rstd[key] = np.zeros((2,ntrial,nit))
                        rstd_norm[key] = np.zeros((2,ntrial,nit))
                        max20[key] = np.zeros((2,ntrial,nit))
                        min0[key] = np.zeros((2,ntrial,nit))
                        radius[key] = np.zeros((2,ntrial,nit))
                        cl[key] = np.zeros((2,ntrial,nit))
                        OnOff_balance[key] = np.zeros((ntrial,nit))
                    else:
                        _step0, _nt_, _nit, _stage, _nLGN_1D = np.fromfile(f, 'int', 5)
                        _dt = np.fromfile(f, 'f4', 1)[0]
                        assert(_nit == nit)
                        assert(_stage == stage)
                        assert(_nLGN_1D == nLGN_1D)

                    # input_halfwidth
                    if stage == 2 or stage == 3:
                        _ = np.fromfile(f, 'f4', 2) 
                    else:
                        assert(stage == 5)
                        _ = np.fromfile(f, 'f4', 2*2)[:2]
                    print(f'{data_fdr}metric-{suffix}{key}{i}.bin')
                    _ = np.fromfile(f, 'f8', nLGN_1D * nLGN_1D).reshape(nLGN_1D, nLGN_1D) # initial_dist
                    rstd[key][:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                    rstd_norm[key][:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                    max20[key][:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
                    min0[key][:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
                    radius[key][:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                    cl[key][:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                    OnOff_balance[key][i,:] = np.fromfile(f, 'f8', nit)

                if totalFr_binned:
                    frFn = f'{data_fdr}V1_totalFr-{suffix}{key}{i}.bin'
                    try:
                        with open(frFn, 'rb') as f:
                            if i == 0:
                                nt = np.fromfile(f, 'i8', 1)[0]
                                print(nt)
                                ft[key] = np.fromfile(f, 'f8', nt)
                                fr[key] = np.zeros((ntrial, nt))
                            else:
                                _nt = np.fromfile(f, 'i8', 1)[0]
                                _ft = np.fromfile(f, 'f8', nt)
                                assert(_nt == nt)
                                assert((_ft - ft[key]).any() == False)
                            fr[key][i,:] = np.fromfile(f, 'f8', nt)
                    except FileNotFoundError:
                        print(f"file not found: {frFn}, no firing rate plots")
                        totalFr_binned = False

        t = np.arange(nit)
        ylim = [0, nLGN_1D/2 + 0.5]
        fig = plt.figure('temporal_sep', figsize = (14,5), dpi = 200)
        ax = fig.add_subplot(241)
        temporal_sep(t, rstd, ax, 2, 'g. std', ylim = ylim, legend = True)
        ax = fig.add_subplot(242)
        temporal_sep(t, rstd_norm, ax, 2, 'g. std by cell', ylim = ylim)
        ax = fig.add_subplot(243)
        temporal_sep(t, radius, ax, 2, 'avg. dis2c', ylim = ylim)
        ax = fig.add_subplot(244)
        if not totalFr_binned:
            temporal_sep(t, cl, ax, 2, 'sqrt(area)/2', ylim = ylim)
        else:
            temporal_sep(t, cl, ax, 2, 'sqrt(area)/2', ylim = ylim, xlabel = 't (au)')
        ax = fig.add_subplot(245)
        temporal_sep(t, OnOff_balance, ax, 1, 'On-Off', xlabel = 't (au)')
        ax = fig.add_subplot(246)
        temporal_sep(t, max20, ax, 2, '# top 20%', ylim = [0, nLGN_1D*nLGN_1D], xlabel = 't (au)')
        ax = fig.add_subplot(247)
        temporal_sep(t, min0, ax, 2, '# disconnected', ylim = [0, nLGN_1D*nLGN_1D], xlabel = 't (au)')
        if totalFr_binned:
            ax = fig.add_subplot(248)
            norm_temporal_sep(fr, ax, 1, 'avg. FR', xlabel = 'normalized time %')
        plt.tight_layout(w_pad = 1)
        fig.savefig(fig_fdr + 'temporal_sep-' + suffix +'.png', bbox_inches = 'tight')

        tpoint = -1
        fig = plt.figure('end_sep', figsize = (20,8), dpi = 200)
        ax = fig.add_subplot(241)
        end_sep(rstd, tpoint, ax, 2, 'g. std', ylim = ylim)
        ax = fig.add_subplot(242)
        end_sep(rstd_norm, tpoint, ax, 2, 'g. std by cell', ylim = ylim)
        ax = fig.add_subplot(243)
        end_sep(radius, tpoint, ax, 2, 'avg. dis2c', ylim = ylim)
        ax = fig.add_subplot(244)
        end_sep(cl, tpoint, ax, 2, 'sqrt(area)/2', ylim = ylim)
        ax = fig.add_subplot(245)
        end_sep(OnOff_balance, tpoint, ax, 1, 'On-Off')
        ax = fig.add_subplot(246)
        end_sep(max20, tpoint, ax, 2, '# top 20%', ylim = [0, nLGN_1D*nLGN_1D])
        ax = fig.add_subplot(247)
        end_sep(min0, tpoint, ax, 2, '# disconnected', ylim = [0, nLGN_1D*nLGN_1D])
        if totalFr_binned:
            ax = fig.add_subplot(248)
            end_sep(fr, tpoint, ax, 1, 'avg. FR')
        plt.tight_layout(w_pad = 2)
        fig.savefig(fig_fdr + 'end_sep-' + suffix +'.png', bbox_inches = 'tight')

def sig(d, n, mean = 0):
    if n == 2:
        return stat.ttest_ind(d[0,:], d[1,:]).pvalue
    if n == 1:
        return stat.ttest_1samp(d, mean).pvalue

def hist(data, ax, n, xlabel, bins = 20):
    cmap = mpl.cm.get_cmap('Paired')
    if n == 2:
        ax.hist([data[0,:], data[1,:]], bins = bins, color = [cmap((6-0.5)/12), cmap((2-0.5)/12)])
    if n == 1:
        ax.hist(data, bins = bins, color = cmap((4-0.5)/12)) 
    ax.set_xlabel(xlabel)

def _fill_between(t, d, c1, c2, ax, alpha):
    mean = d.mean(axis = 0)
    low = np.percentile(d, 25, axis = 0)
    high = np.percentile(d, 75, axis = 0)
    ax.plot(t, mean, color = c1)
    ax.fill_between(t, low, high, color = c2, ec = 'None', alpha = 0.7)

def fill_between(t, data, ax, n, ylabel, ylim = None, xlabel = None):
    cmap = mpl.cm.get_cmap('Paired')
    if n == 2:
        _fill_between(t, data[0,:,:], cmap((6-0.5)/12), cmap((5-0.5)/12), ax, 0.7)
        _fill_between(t, data[1,:,:], cmap((2-0.5)/12), cmap((1-0.5)/12), ax, 0.7)
    if n == 1:
        _fill_between(t, data, cmap((4-0.5)/12), cmap((3-0.5)/12), ax, 0.7)

    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel) 
    else:
        ax.xaxis.set_ticklabels([])

def temporal_sep(t, d_dict, ax, n, ylabel, ylim = None, xlabel = None, legend = False):
    cmap = mpl.cm.get_cmap('Paired')
    c1 = cmap((6-0.5)/12)
    c2 = cmap((2-0.5)/12)
    c3 = cmap((4-0.5)/12)
    i = 0
    for data in d_dict.values(): 
        if n == 2:
            ax.plot(t, data[0,:,:].mean(axis = 0), '-', lw = (i+1)*0.5, color = c1, alpha = 0.7)
            ax.plot(t, data[1,:,:].mean(axis = 0), ':', lw = (i+1)*0.5, color = c2, alpha = 0.7)
            if i == 0 and legend:
                ax.legend(['On', 'Off']) 
        if n == 1:
            ax.plot(t, data.mean(axis = 0), lw = (i+1)*0.5, color = c3, alpha = 0.7)
        i += 1

    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel) 
    else:
        ax.xaxis.set_ticklabels([])


def norm_temporal_sep(d_dict, ax, n, ylabel, ylim = None, xlabel = None, legend = False):
    cmap = mpl.cm.get_cmap('Paired')
    c1 = cmap((6-0.5)/12)
    c2 = cmap((2-0.5)/12)
    c3 = cmap((4-0.5)/12)
    i = 0
    data_len = [d.shape[-1] for d in d_dict.values()]
    max_dl = np.max(data_len)
    for data in d_dict.values(): 
        t = np.linspace(0, 100, data_len[i])/max_dl
        if n == 2:
            ax.plot(t, data[0,:,:].mean(axis = 0), '-', lw = (i+1)*0.5, color = c1, alpha = 0.7)
            ax.plot(t, data[1,:,:].mean(axis = 0), ':', lw = (i+1)*0.5, color = c2, alpha = 0.7)
            if i == 0 and legend:
                ax.legend(['On', 'Off']) 
        if n == 1:
            ax.plot(t, data.mean(axis = 0), lw = (i+1)*0.5, color = c3, alpha = 0.7)
        i += 1

    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel) 
    else:
        ax.xaxis.set_ticklabels([])

def end_sep(d_dict, tpoint, ax, n, ylabel, ylim = None):
    cmap = mpl.cm.get_cmap('Paired')
    c1 = cmap((6-0.5)/12)
    c_1 = cmap((5-0.5)/12)
    c2 = cmap((2-0.5)/12)
    c_2 = cmap((1-0.5)/12)
    c3 = cmap((4-0.5)/12)
    c_3 = cmap((3-0.5)/12)
    if n == 2:
        data = np.array([d[:,:,tpoint] for d in d_dict.values()])
        p = np.zeros((2,len(d_dict)-1)) 
        p[0,:] = [sig(data[i:i+2,0,:], 2) for i in range(len(d_dict)-1)]
        p[1,:] = [sig(data[i:i+2,1,:], 2) for i in range(len(d_dict)-1)]
    else:
        data = np.array([d[:,tpoint] for d in d_dict.values()])
        p = [sig(data[i:i+2,:], 2) for i in range(len(d_dict)-1)]
    low = np.percentile(data, 25, axis = -1)
    high = np.percentile(data, 75, axis = -1)
    mean = data.mean(-1)
    xticklabel = [key for key in d_dict.keys()]
    ax2 = ax.twinx()
    x = np.arange(len(d_dict))
    xp = x[:-1] + 0.5
    if n == 2:
        ax.plot(x, mean[:,0], color = c1)
        ax.plot(x, mean[:,1], color = c2)
        ax.fill_between(x, low[:,0], high[:,0], color = c_1, ec = 'None', alpha = 0.7) 
        ax.fill_between(x, low[:,1], high[:,1], color = c_2, ec = 'None', alpha = 0.7) 
        ax2.plot(xp, p[0,:], '*', color = c1, alpha = 0.5) 
        ax2.plot(xp, p[1,:], '*', color = c2, alpha = 0.5) 
    if n == 1:
        ax.plot(x, mean, color = c3)
        ax.fill_between(x, low, high, color = c_3, ec = 'None', alpha = 0.7)
        ax2.plot(xp, p, '*', color = c3, alpha = 0.5) 
    ax2.plot(xp, np.zeros_like(xp) + 0.05,  ':k', lw = 1, alpha = 0.7) 
    ax2.set_ylabel('t. test')
    ax.set_xticks(np.arange(len(d_dict)), labels = xticklabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)

if __name__ == '__main__':
    data_fdr = sys.argv[1]
    fig_fdr = sys.argv[2]
    suffix = sys.argv[3] 
    n = int(sys.argv[4])
    if sys.argv[5] == 'True' or sys.argv[5] == '1':
        separate = True
    else:
        separate = False

    if sys.argv[6] == 'True' or sys.argv[6] == '1':
        use_idx = True
    else:
        use_idx = False

    if not use_idx:
        batch = dict()
        for i in range(n):
            batch[sys.argv[7+i*2]] = int(sys.argv[7+i*2+1])
        print(batch)
        num = None
    else:
        num = [int(sys.argv[7+i]) for i in range(n)]
        batch = None 

    random_trials_plot(data_fdr, fig_fdr, suffix, separate, use_idx, num, batch)
