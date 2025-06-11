import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stat
import matplotlib.gridspec as gs
import sys
from outputLearnFF import get_oneOverYscale
from plotV1_response import movingAvg
nblock = 1
mE = 992
mI = 32
blockSize = mE+mI
epick = np.hstack([np.arange(mE) + iblock*blockSize for iblock in range(nblock)])
nE = mE*nblock
lw0 = 1.0
dlw = 0.4

mpl.rcParams.update({'font.family': 'CMU Sans Serif', 'axes.unicode_minus' : False, 'mathtext.fontset': 'cm', 'mathtext.default':'regular'})

def get_colors(n, sat0 = 0.3, val0 = 0.1, val1 = 1.0):
    sat_green = np.zeros((n,3))
    sat_red = np.zeros((n,3))
    sat_blue = np.zeros((n,3))
    sat_green[:,0] = 1/3
    sat_red[:,0] = 0
    sat_blue[:,0] = 2/3

    #sat_green[:,1] = np.linspace(sat0, 1, n)
    sat_green[:,1] = sat0
    sat_green[:,2] = np.power(np.linspace(val0, val1, n), 0.5)
    #sat_green[:,2] = 0.5

    sat_red[:,1] = np.linspace(sat0, 1, n)
    #sat_red[:,1] = sat0
    #sat_red[:,2] = np.linspace(val0, val1, n)
    sat_red[:,2] = val1

    sat_blue[:,1] = np.linspace(sat0, 1, n)
    #sat_blue[:,1] = sat0
    #sat_blue[:,2] = np.linspace(val0, val1, n)
    sat_blue[:,2] = val1
    greens = mpl.colors.hsv_to_rgb(sat_green)
    reds = mpl.colors.hsv_to_rgb(sat_red)
    blues = mpl.colors.hsv_to_rgb(sat_blue)
    sat_black = np.zeros((n,3))
    sat_black[:,0] = 1/3
    sat_black[:,1] = 0
    sat_black[:,2] = 0.1
    blacks = mpl.colors.hsv_to_rgb(sat_black)
    return greens, reds, blues, blacks

def random_trials_plot(theme, data_fdr, fig_fdr, suffix, separate, use_idx, num, batch, single):
    plot_temp_sep = False
    plot_end_sep = False
    global prod_label
    theme0 = theme
    if theme0 == 'speed': 
        prod_label = [r'0.5$\times$', r'1$\times$', r'2$\times$', r'3$\times$', r'4$\times$']
        unit = '4 deg/s'
    if theme0 == 'width':
        prod_label = ['4', '6', '8', '10', '12']
        unit = '#LGN'
    if theme0 == 'w0':
        #prod_label = [r'$w^*-2$', r'$w^*-1$', r'$w^*$', r'$w^*+1$', r'$w^*+2$']
        prod_label = [r'$0.04$', r'$0.05$', r'$0.06$', r'$0.07$', r'$0.08$']
        theme0 = r'$w_0$'
    if theme0 == 'LTD-ratio':
        prod_label = [r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$', r'$0.6$']
        theme0 = 'LTD ratio'
    if theme0 == 'same_strength':
        prod_label = [r'0.5$\times$', r'0.75$\times$', r'1$\times$', r'1.25$\times$', r'1.5$\times$']
        theme0 = 'width, same strength'
        unit = ''
        #unit = 'standard width (10 degs)'
    if theme0 == 'homeo':
        prod_label = [r'$\gamma$', r'$\gamma$', r'2/3$\gamma$', r'1/3$\gamma$']
        theme0 = 'gamma'
        unit = ''

    if suffix[-1] != '-' and theme[0] != '-':
        theme = '-' + theme 
    save_svg = True
    if fig_fdr[-1] != '/':
        fig_fdr = fig_fdr+'/'
    if data_fdr[-1] != '/':
        data_fdr = data_fdr+'/'
    totalFr_binned = True
    if not separate:
        if not use_idx:
            num = []
            for key, value in batch.items():
                if not single:
                    num += [f'{key}{i}' for i in range(0,value)]
                else:
                    num += [f'{key}']
        n = len(num)
        print(f'{n} trials')
        for i in range(n):
            with open(f'{data_fdr}metric-{suffix}{num[i]}.bin', 'rb') as f:
                if i == 0:
                    step0, nt_, nit, stage, nV1, nLGN_1D, nradial_dis = np.fromfile(f, 'int', 7)
                    dt = np.fromfile(f, 'f4', 1)[0]
                    rstd = np.zeros((2,n,nit))
                    rstd_norm = np.zeros((2,n,nit))
                    s_sum = np.zeros((2,n,nV1,nit))
                    s_std = np.zeros((2,n,nV1,nit))
                    n_con = np.zeros((2,n,nit))
                    rdist = np.zeros((2,n,nradial_dis,nit))

                    max20 = np.zeros((2,n,nit))
                    min0 = np.zeros((2,n,nit))
                    radius = np.zeros((2,n,nit))
                    cl = np.zeros((2,n,nit))
                    OnOff_balance = np.zeros((n, nV1, nit))
                else:
                    _step0, _nt_, _nit, _stage, _nV1,  _nLGN_1D, _nradial_dis = np.fromfile(f, 'int', 7)
                    _dt = np.fromfile(f, 'f4', 1)[0]
                    assert(_nradial_dis == nradial_dis)
                    assert(_nit == nit)
                    assert(_nV1 == nV1)
                    assert(_stage == stage)
                    assert(_nLGN_1D == nLGN_1D)

                # input_halfwidth
                if stage == 2 or stage == 3:
                    _ = np.fromfile(f, 'f4', 2) 
                else:
                    assert(stage == 5)
                    _ = np.fromfile(f, 'f4', 2*2)[:2]
                print(f'{data_fdr}metric-{suffix}{num[i]}.bin')
                if i == 0:
                    radial_dis = np.fromfile(f, 'f4', nradial_dis)
                else:
                    _ = np.fromfile(f, 'f4', nradial_dis)
                _ = np.fromfile(f, 'f8', nLGN_1D * nLGN_1D).reshape(nLGN_1D, nLGN_1D) # initial_dist
                rstd[:,i,:] = 1/np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                rstd_norm[:,i,:] = 1/np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                s_sum[:,i,:,:] = np.fromfile(f, 'f8', 2*nV1*nit).reshape(2,nV1,nit)
                s_std[:,i,:,:] = np.fromfile(f, 'f8', 2*nV1*nit).reshape(2,nV1,nit)
                n_con[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                rdist[0,i,:,:] = np.fromfile(f, 'f8', nit*nradial_dis).reshape(nit,nradial_dis).T
                rdist[1,i,:,:] = np.fromfile(f, 'f8', nit*nradial_dis).reshape(nit,nradial_dis).T

                max20[:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
                min0[:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
                radius[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                cl[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                _ = np.fromfile(f, 'f8', nit)
                OnOff_balance[i,:,:] = s_sum[0,i,:,:] - s_sum[1,i,:,:]
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

        forward, inverse, yt, ytl, yt_neg, ytl_neg = get_oneOverYscale(nLGN_1D//2)
        yt_full = yt_neg.copy()
        ytl_full = ytl_neg.copy()
        yt_full = yt_full.extend(yt)
        ytl_full = ytl_full.extend(ytl)
        t = np.arange(nit)
        greens, reds, blues, blacks = get_colors(nit)
        fig = plt.figure('temporal_plots', figsize = (14,5), dpi = 200)
        ax = fig.add_subplot(241)
        #ylim = [0, nLGN_1D/2*np.sqrt(2) + 0.5]
        ylim = [0, nLGN_1D/2]
        try:
            state = fill_between(t, rstd, ax, 2, '1/g.std')
            ax.set_yscale('function', functions=(forward, inverse))
            if state == 0:
                ax.set_yticks(yt_full, ytl_full)
                ax.set_ylim([-1.5,1.5])
            elif state == 1:
                ax.set_yticks(yt, ytl)
                ax.set_ylim([0,1.5])
            else:
                ax.set_yticks(yt_neg, ytl_neg)
                ax.set_ylim([-1.5,0])
        except FloatingPointError:
            print('no g.std available')
        ax = fig.add_subplot(242)
        try:
            state = fill_between(t, rstd_norm, ax, 2, '1/g.std by cell')
            ax.set_yscale('function', functions=(forward, inverse))
            if state == 0:
                ax.set_yticks(yt_full, ytl_full)
                ax.set_ylim([-1.5,1.5])
            elif state == 1:
                ax.set_yticks(yt, ytl)
                ax.set_ylim([0,1.5])
            else:
                ax.set_yticks(yt_neg, ytl_neg)
                ax.set_ylim([-1.5,0])
        except FloatingPointError:
            print('no g.std available')

        ax = fig.add_subplot(243)
        fill_between(t, radius, ax, 2, 'avg. dis2c', ylim = ylim)
        ax.legend()
        ax = fig.add_subplot(244)
        if not totalFr_binned:
            fill_between(t, cl, ax, 2, r'$\sqrt{area}/2$', ylim = ylim, xlabel = 't (au)')
            ax.legend()
        else:
            fill_between(t, cl, ax, 2, r'$\sqrt{area}/2$', ylim = ylim)
            ax.legend()
        ax = fig.add_subplot(245)
        fill_between(t, OnOff_balance.reshape(n*nV1,nit), ax, 1, 'ON-OFF', xlabel = 't (au)')
        ax = fig.add_subplot(246)
        fill_between(t, max20, ax, 2, '# top 20%', ylim = [0, nLGN_1D*nLGN_1D], xlabel = 't (au)')
        ax = fig.add_subplot(247)
        fill_between(t, min0, ax, 2, '# disconnected', ylim = [0, nLGN_1D*nLGN_1D], xlabel = 't (au)')
        if totalFr_binned:
            ax = fig.add_subplot(248)
            fill_between(ft/1000, fr, ax, 1, 'avg. FR', xlabel = 't (s)')
        plt.tight_layout(w_pad = 1)
        fig.savefig(fig_fdr + 'temporal_plots-' + suffix + theme + '.png', bbox_inches = 'tight')
        if save_svg:
            fig.savefig(fig_fdr + 'temporal_plots-' + suffix + theme + '.svg', bbox_inches = 'tight')

        # production figures:

        fig = plt.figure('single_stats', figsize = (10, 4.5), dpi = 150)
        # on off balance
        label_fs = 'large'
        widths = [1.5, 1.5, 2.8, 0.5]
        grids = gs.GridSpec(ncols = 4, nrows = 2, figure = fig, width_ratios = widths, hspace = 0.4)
        data = OnOff_balance[:,epick,:].reshape(n*nE,nit) # per neuron per trial
        #data = OnOff_balance[:,epick,:].mean(1).reshape(n,nit) # per trial
        #data = OnOff_balance[:,epick,:].mean(0).reshape(nE,nit) # per neuron
        ax = fig.add_subplot(grids[0,2])
        fill_between(t, data, ax, 1, 'weight balance:\n ON-OFF', xlabel = 'normalized time (%)', fs = label_fs)
        ax.set_xticks([0, 2.5, 5, 7.5, 10], ['0', '25','50','75','100'])
        ax.set_ylim(data.min(),data.max())
        ax = fig.add_subplot(grids[0,3])
        chartBox = ax.get_position() 
        ax.set_position([chartBox.x0-0.02, chartBox.y0, chartBox.width, chartBox.height * 1.0]) 
        ax.hist(data[:,-1], bins = 20, weights = np.ones(data.shape[0])*100/data.shape[0], density = False, orientation = 'horizontal', rwidth = 0.85, color = 'g' , alpha = 0.6)
        ax.set_ylim(data.min(),data.max())
        #tick = ax.get_xticks()
        #ax.set_xticks(tick, labels = [f'{tck*100:.0f}' for tck in tick])
        #tick = [0, 10]
        #ax.set_xticks(tick)
        ax.set_xlabel('final dist. (%)', fontsize = 'small')
        ax.yaxis.set_label_position('right')
        ax.tick_params(axis = 'y', labelleft = False, labelsize = 'small')
        #ax.set_ylabel(f'p-value: {sig(data[:,-1],1,0,True):.2e}')

        # radial_dist 
        ax = fig.add_subplot(grids[1,0])
        rdist_single(rdist, radial_dis, 'ON', ax, c = reds, fs = label_fs)
        chartBox = ax.get_position() 
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height]) 
        ax = fig.add_subplot(grids[1,1])
        rdist_single(rdist, radial_dis, 'OFF', ax, c = blues, fs = label_fs)
        chartBox = ax.get_position() 
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height]) 
        # temporal
        ylim = [0, nLGN_1D/2 + 2]
        ax = fig.add_subplot(grids[1,2])
        single_temporal_r(t, radius, ax, 'weighted radius', ylim, 'normalized time (%)', reds[-1,:], blues[-1,:], 4, ['ON', 'OFF'], fs = label_fs)
        ax.legend(fontsize = 'small')
        ax.set_xticks([0, 2.5, 5, 7.5, 10], ['0', '25','50','75','100'])
        ax2 = ax.twinx()
        single_temporal_r(t, cl, ax2, r'$\sqrt{area}/2$', ylim, 'normalized time (%)', reds[4,:], blues[4,:], lc = 'gray', fs = label_fs)
        ax2.tick_params(axis = 'y', colors = 'gray')
        #ax2.yaxis.label.set_color('gray')
        ax2.spines['right'].set_color('gray')
        fig.savefig(fig_fdr + 'single_stats-' + suffix + theme + '.png', bbox_inches = 'tight')
        if save_svg:
            fig.savefig(fig_fdr + 'single_stats-' + suffix + theme + '.svg', bbox_inches = 'tight')

        tpoint = -1
        if n > 1:
            fig = plt.figure('end_dist', figsize = (14,8), dpi = 200)
            ax = fig.add_subplot(241)
            try:
                hist(rstd[:,:,tpoint], ax, 2, 'g. std')
                ax.text(0.1, 0.9, f'pvalue: {sig(rstd[:,:,tpoint],2):.3e}', transform = ax.transAxes)
            except ValueError:
                print('no g.std available')
            ax = fig.add_subplot(242)
            try:
                hist(rstd_norm[:,:,tpoint], ax, 2, 'g. std by cell')
                ax.text(0.1, 0.9, f'pvalue: {sig(rstd_norm[:,:,tpoint],2):.3e}', transform = ax.transAxes)
            except ValueError:
                print('no g.std available')
            ax = fig.add_subplot(243)
            hist(radius[:,:,tpoint], ax, 2, 'avg. dis2c')
            ax.text(0.1, 0.9, f'pvalue: {sig(radius[:,:,tpoint],2):.3e}', transform = ax.transAxes)
            ax = fig.add_subplot(244)
            hist(cl[:,:,tpoint], ax, 2, r'$\sqrt{area}/2$')
            ax.text(0.1, 0.9, f'pvalue: {sig(cl[:,:,tpoint],2):.3e}', transform = ax.transAxes)
            ax = fig.add_subplot(245)
            avg_OnOff_balance = OnOff_balance.mean(1)
            hist(avg_OnOff_balance[:,tpoint], ax, 1, 'ON-OFF')
            ax.text(0.1, 0.9, f'pvalue: {sig(avg_OnOff_balance[:,tpoint],1):.3e}', transform = ax.transAxes)
            ax = fig.add_subplot(246)
            hist(max20[:,:,tpoint], ax, 2, '# top 20%')
            ax.text(0.1, 0.9, f'pvalue: {sig(max20[:,:,tpoint],2):.3e}', transform = ax.transAxes)
            ax = fig.add_subplot(247)
            hist(min0[:,:,tpoint], ax, 2, '# disconnected')
            ax.text(0.1, 0.9, f'pvalue: {sig(min0[:,:,tpoint],2):.3e}', transform = ax.transAxes)
            if totalFr_binned:
                ax = fig.add_subplot(248)
                hist(fr[:,tpoint-1], ax, 1, 'avg. FR')
            fig.savefig(fig_fdr + 'end_dist-' + suffix + theme + '.png', bbox_inches = 'tight')
            if save_svg:
                fig.savefig(fig_fdr + 'end_dist-' + suffix + theme + '.svg', bbox_inches = 'tight')
    else:
        nbatch = len(batch)
        greens, reds, blues, blacks = get_colors(nbatch)
        keys = batch.keys()
        radial_dis = dict({})
        rstd = dict({})
        rstd_norm = dict({})
        s_sum = dict({})
        s_std = dict({})
        n_con = dict({})
        rdist = dict({})
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
                if ntrial == 1:
                    num_suffix = key
                else:
                    num_suffix = f'{key}{i}'
                with open(f'{data_fdr}metric-{suffix}{num_suffix}.bin', 'rb') as f:
                    if i == 0:
                        step0, nt_, nit, stage, nV1, nLGN_1D, nradial_dis = np.fromfile(f, 'int', 7)
                        dt = np.fromfile(f, 'f4', 1)[0]
                        rstd[key] = np.zeros((2,ntrial,nit))
                        rstd_norm[key] = np.zeros((2,ntrial,nit))
                        s_sum[key] = np.zeros((2,ntrial,nV1,nit))
                        s_std[key] = np.zeros((2,ntrial,nV1,nit))
                        n_con[key] = np.zeros((2,ntrial,nit))
                        rdist[key] = np.zeros((2,ntrial,nradial_dis,nit))
                        radial_dis[key] = np.zeros(nradial_dis)
                        max20[key] = np.zeros((2,ntrial,nit))
                        min0[key] = np.zeros((2,ntrial,nit))
                        radius[key] = np.zeros((2,ntrial,nit))
                        cl[key] = np.zeros((2,ntrial,nit))
                        OnOff_balance[key] = np.zeros((ntrial,nV1,nit))
                    else:
                        _step0, _nt_, _nit, _stage, _nV1, _nLGN_1D, _nradial_dis = np.fromfile(f, 'int', 7)
                        _dt = np.fromfile(f, 'f4', 1)[0]
                        assert(_nradial_dis == nradial_dis)
                        assert(_nit == nit)
                        assert(_stage == stage)
                        assert(_nLGN_1D == nLGN_1D)

                    # input_halfwidth
                    if stage == 2 or stage == 3:
                        _ = np.fromfile(f, 'f4', 2) 
                    else:
                        assert(stage == 5)
                        _ = np.fromfile(f, 'f4', 2*2)[:2]
                    print(f'{data_fdr}metric-{suffix}{num_suffix}.bin')
                    if i == 0:
                        radial_dis[key] = np.fromfile(f, 'f4', nradial_dis)
                        print(f'step0 = {step0}, nt = {nt_}, nit = {nit}, nV1 = {nV1}, nLGN_1D = {nLGN_1D}, nradial_dis = {nradial_dis}')
                    else:
                        _ = np.fromfile(f, 'f4', nradial_dis)
                    _ = np.fromfile(f, 'f8', nLGN_1D * nLGN_1D).reshape(nLGN_1D, nLGN_1D) # initial_dist
                    rstd[key][:,i,:] = 1/np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                    rstd_norm[key][:,i,:] = 1/np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                    s_sum[key][:,i,:,:] = np.fromfile(f, 'f8', 2*nV1*nit).reshape(2,nV1,nit)
                    s_std[key][:,i,:,:] = np.fromfile(f, 'f8', 2*nV1*nit).reshape(2,nV1,nit)
                    n_con[key][:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                    rdist[key][0,i,:,:] = np.fromfile(f, 'f8', nit*nradial_dis).reshape(nit,nradial_dis).T
                    rdist[key][1,i,:,:] = np.fromfile(f, 'f8', nit*nradial_dis).reshape(nit,nradial_dis).T
                    max20[key][:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
                    min0[key][:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
                    radius[key][:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                    cl[key][:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
                    _ = np.fromfile(f, 'f8', nit)
                    OnOff_balance[key][i,:,:] = s_sum[key][0,i,:,:] - s_sum[key][1,i,:,:]

                if totalFr_binned:
                    frFn = f'{data_fdr}V1_totalFr-{suffix}{num_suffix}.bin'
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

        forward, inverse, yt, ytl, yt_neg, ytl_neg = get_oneOverYscale(nLGN_1D//2)
        yt_full = yt_neg.copy()
        ytl_full = ytl_neg.copy()
        yt_full.extend(yt)
        ytl_full.extend(ytl)

        t = np.arange(nit)
        if plot_temp_sep:
            fig = plt.figure('temporal_sep', figsize = (14,5), dpi = 200)
            ax = fig.add_subplot(241)
            try:
                state = temporal_sep(t, rstd, ax, 2, 'g.std', legend = True)
                ax.set_yscale('function', functions=(forward, inverse))
                if state == 0:
                    ax.set_yticks(yt_full, ytl_full)
                    ax.set_ylim([-1.5,1.5])
                elif state == 1:
                    ax.set_yticks(yt, ytl)
                    ax.set_ylim([0,1.5])
                else:
                    ax.set_yticks(yt_neg, ytl_neg)
                    ax.set_ylim([-1.5,0])
            except FloatingPointError:
                print('no g.std available')

            ax = fig.add_subplot(242)
            try:
                state = temporal_sep(t, rstd_norm, ax, 2, 'g.std by cell')
                ax.set_yscale('function', functions=(forward, inverse))
                if state == 0:
                    ax.set_yticks(yt_full, ytl_full)
                    ax.set_ylim([-1.5,1.5])
                elif state == 1:
                    ax.set_yticks(yt, ytl)
                    ax.set_ylim([0,1.5])
                else:
                    ax.set_yticks(yt_neg, ytl_neg)
                    ax.set_ylim([-1.5,0])
            except FloatingPointError:
                print('no g.std available')

            ylim = [0, nLGN_1D/2*np.sqrt(2) + 0.5]
            ax = fig.add_subplot(243)
            temporal_sep(t, radius, ax, 2, 'avg. dis2c', ylim = ylim)
            ax = fig.add_subplot(244)
            if not totalFr_binned:
                temporal_sep(t, cl, ax, 2, r'$\sqrt{area}/2$', ylim = ylim)
            else:
                temporal_sep(t, cl, ax, 2, r'$\sqrt{area}/2$', ylim = ylim, xlabel = 't (au)')
            ax = fig.add_subplot(245)
            temporal_sep(t, OnOff_balance, ax, 1, r'ON-OFF', xlabel = 't (au)')
            if totalFr_binned:
                ax = fig.add_subplot(248)
                norm_temporal_sep(fr, ax, 1, 'avg. FR', xlabel = 'normalized time %')
            plt.tight_layout(w_pad = 1)
            fig.savefig(fig_fdr + 'temporal_sep-' + suffix + theme + '.png', bbox_inches = 'tight')
            if save_svg:
                fig.savefig(fig_fdr + 'temporal_sep-' + suffix + theme + '.svg', bbox_inches = 'tight')

        if plot_end_sep:
            tpoint = -1
            fig = plt.figure('end_sep', figsize = (20,8), dpi = 200)
            ax = fig.add_subplot(241)
            rdist_plot(rdist, radial_dis, tpoint, ax)

            '''
            state = end_sep(rstd, tpoint, ax, 2, 'g. std')
            ax.set_yscale('function', functions=(forward, inverse))
            if state == 0:
                ax.set_yticks(yt_full, ytl_full)
                ax.set_ylim([-1.5,1.5])
            elif state == 1:
                ax.set_yticks(yt, ytl)
                ax.set_ylim([0,1.5])
            else:
                ax.set_yticks(yt_neg, ytl_neg)
                ax.set_ylim([-1.5,0])
            '''

            ax = fig.add_subplot(242)
            try:
                state = end_sep(rstd_norm, tpoint, ax, 2, 'g. std by cell')
                ax.set_yscale('function', functions=(forward, inverse))
                if state == 0:
                    ax.set_yticks(yt_full, ytl_full)
                    ax.set_ylim([-1.5,1.5])
                elif state == 1:
                    ax.set_yticks(yt, ytl)
                    ax.set_ylim([0,1.5])
                else:
                    ax.set_yticks(yt_neg, ytl_neg)
                    ax.set_ylim([-1.5,0])
            except FloatingPointError:
                print('no g.std available')

            ax = fig.add_subplot(243)
            end_sep(radius, tpoint, ax, 2, 'avg. dis2c', ylim = ylim)
            ax = fig.add_subplot(244)
            end_sep(cl, tpoint, ax, 2, r'$\sqrt{area}/2$', ylim = ylim)
            ax = fig.add_subplot(245)
            end_sep(OnOff_balance, tpoint, ax, 1, 'ON-OFF')
            ax = fig.add_subplot(246)
            end_sep(max20, tpoint, ax, 2, '# top 20%', ylim = [0, nLGN_1D*nLGN_1D])
            ax = fig.add_subplot(247)
            end_sep(min0, tpoint, ax, 2, '# disconnected', ylim = [0, nLGN_1D*nLGN_1D])
            if totalFr_binned:
                ax = fig.add_subplot(248)
                end_sep(fr, tpoint, ax, 1, 'avg. FR')
            plt.tight_layout(w_pad = 2)
            fig.savefig(fig_fdr + 'end_sep-' + suffix + theme + '.png', bbox_inches = 'tight')
            if save_svg:
                fig.savefig(fig_fdr + 'end_sep-' + suffix + theme + '.svg', bbox_inches = 'tight')

        ylim = [0, nLGN_1D/2*np.sqrt(2) + 0.5]
        if totalFr_binned:
            fig = plt.figure('normed_fr', figsize = (12,6), dpi = 100)
            ax = fig.add_subplot(111)
            normed_fr = fr.copy()
            for key in normed_fr.keys(): 
                normed_fr[key] = fr[key]/np.tile(np.max(fr[key], axis = 1).reshape(fr[key].shape[0],1), (1,fr[key].shape[1]))
            norm_temporal_sep(normed_fr, ax, 1, 'normed. FR', xlabel = 'normalized time %')
            fig.savefig(fig_fdr + 'normed_fr-' + suffix + theme + '.png', bbox_inches = 'tight')
            if save_svg:
                fig.savefig(fig_fdr + 'normed_fr-' + suffix + theme + '.svg', bbox_inches = 'tight')

        if totalFr_binned:
            fig = plt.figure('temporal_detail', figsize = (4,10), dpi = 120)
        else:
            fig = plt.figure('temporal_detail', figsize = (4,8), dpi = 120)
        ax = fig.add_subplot(511)
        temporal_sep(t, radius, ax, 2, 'avg. dis2c', ylim = ylim, i0 = 3)
        ax = fig.add_subplot(512)
        temporal_sep(t, n_con, ax, 2, 'avg. #connected', ylim = [0, nLGN_1D*nLGN_1D], xlabel = 't (au)', i0 = 3)
        ax = fig.add_subplot(513)
        temporal_sep(t, s_sum, ax, 2, 'avg. total weight', i0 = 3)
        ax = fig.add_subplot(514)
        temporal_sep(t, s_std, ax, 2, 'avg. weight std.', ylim = [0, np.nan], xlabel = 't (au)', i0 = 3)
        if totalFr_binned:
            ax = fig.add_subplot(515)
            norm_temporal_sep(fr, ax, 1, 'avg. FR', xlabel = 'normalized time %', i0 = 2)
        plt.tight_layout(h_pad = 1)
        fig.savefig(fig_fdr + 'temporal_detail-' + suffix + theme + '.png', bbox_inches = 'tight')
        if save_svg:
            fig.savefig(fig_fdr + 'temporal_detail-' + suffix + theme + '.svg', bbox_inches = 'tight')

        # production figures
        
        if theme0 == 'width':
            fig = plt.figure('stats', figsize = (7.5, 3.2), dpi = 200, constrained_layout = True)
        else:
            fig = plt.figure('stats', figsize = (7.5, 3.5), dpi = 200, constrained_layout = True)
        label_fs = 11
        single_radius = {}
        for key, value in radius.items():
            single_radius[key] = (value[0,:,:] + value[1,:,:])/2

        single_cl = {}
        for key, value in cl.items():
            single_cl[key] = (value[0,:,:] + value[1,:,:])/2

        ylim = [0, nLGN_1D/2+1]
        ax = fig.add_subplot(232)
        temporal_sep(t, single_radius, ax, 1, 'weighted r.', ylim = ylim, c = greens, fw = True, fs = label_fs, theme = theme0)
        ax.set_xticks([0, 2.5, 5, 7.5, 10], ['0', '25', '50', '75', '100'])
        ax.set_yticks([0, 4, 8], ['0', '4', '8'])
        ax.set_xlabel('normalized t (%)', fontsize = label_fs)
        ax = fig.add_subplot(235)
        temporal_sep(t, single_cl, ax, 1, r'char. length', ylim = ylim, c = greens, fw = True, fs = label_fs, theme = theme0)
        ax.set_xticks([0, 2.5, 5, 7.5, 10], ['0', '25', '50', '75', '100'])
        ax.set_yticks([0, 4, 8], ['0', '4', '8'])
        ax.set_xlabel('normalized t (%)', fontsize = label_fs)
        tpoint = -1
        ax = fig.add_subplot(233)
        bar_end(single_radius, tpoint, ax, 'final weighted r.', ylim, greens, blacks, fs = label_fs, theme = theme0)
        if theme0 == 'speed' or theme0 == 'width':
            ax.set_xlabel(f'wave {theme0} ({unit})', fontsize = label_fs)
        else:
            ax.set_xlabel(f'{theme0}', fontsize = label_fs)

        ax = fig.add_subplot(234)
        if theme0 == 'speed':
            temporal_rdist(rdist, radial_dis, 'both', prod_label, ax, nLGN_1D/2, c = greens, fw = True, fs = label_fs)
        if theme0[:5] == 'width':
            if unit == '':
                temporal_rdist(rdist, radial_dis, 'both', prod_label, ax, nLGN_1D/2, c = greens, fw = True, fs = label_fs, normalized = True, legs = label_fs-2, theme = theme0)
            else:
                temporal_rdist(rdist, radial_dis, 'both', [f'{prod_label[i]} {unit[1:]}' for i in range(len(prod_label))], ax, nLGN_1D/2, c = greens, fw = True, fs = label_fs, normalized = True, legs = label_fs-2)
        if theme0 == r'$w_0$':
            temporal_rdist(rdist, radial_dis, 'both', prod_label, ax, nLGN_1D/2, c = greens, fw = True, fs = label_fs, theme = theme0, legs = label_fs-2)
        if theme0 == 'LTD ratio':
            temporal_rdist(rdist, radial_dis, 'both', prod_label, ax, nLGN_1D/2, c = greens, fw = True, fs = label_fs, theme = theme0, legs = label_fs-2)
        if theme0 == 'gamma':
            temporal_rdist(rdist, radial_dis, 'both', prod_label, ax, nLGN_1D/2, c = greens, fw = True, fs = label_fs, normalized = False, theme = theme0, legs = label_fs-2)
        ax.set_xticks([0, 2, 4, 6, 8])

        if totalFr_binned:
            ax = fig.add_subplot(231)
            norm_temporal_sep(fr, ax, 1, 'avg. FR (Hz)', xlabel = 'normalized t (%)', c = greens, fw = True, fs = label_fs)
        ax = fig.add_subplot(236)
        bar_end(single_cl, tpoint, ax, r'final char. length', ylim, greens, blacks, fs = label_fs, theme = theme0)
        if theme0 == 'speed' or theme0 == 'width':
            ax.set_xlabel(f'wave {theme0} ({unit})', fontsize = label_fs)
        else:
            ax.set_xlabel(f'{theme0}', fontsize = label_fs)

        fig.savefig(fig_fdr + 'stats-' + suffix + theme + '.png', bbox_inches = 'tight')
        if save_svg:
            fig.savefig(fig_fdr + 'stats-' + suffix + theme + '.svg', bbox_inches = 'tight')

def sig(d, n, mean = 0, print_result = False):
    if n == 2:
        return stat.ttest_ind(d[0,:], d[1,:]).pvalue
    if n == 1:
        if print_result:
            print(stat.ttest_1samp(d, mean).statistic)
            print(stat.ttest_1samp(d, mean).confidence_interval)
        return stat.ttest_1samp(d, mean).pvalue

def hist(data, ax, n, xlabel, bins = 20):
    cmap = mpl.colormaps['Paired']
    if n == 2:
        ax.hist([data[0,:], data[1,:]], bins = bins, color = [cmap((6-0.5)/12), cmap((2-0.5)/12)])
    if n == 1:
        ax.hist(data, bins = bins, color = cmap((4-0.5)/12)) 
    ax.set_xlabel(xlabel)

def _fill_between(t, d, c1, c2, ax, alpha, ls = '-', label = None):
    mean = d.mean(axis = 0)
    low = np.percentile(d, 10, axis = 0)
    high = np.percentile(d, 90, axis = 0)
    ax.plot(t, mean, ls = ls, color = c1, lw = 3*lw0, label = label)
    ax.fill_between(t, low, high, color = c2, ec = 'None', alpha = 0.7)

def fill_between(t, data, ax, n, ylabel, ylim = None, xlabel = None, fs = 'medium'):
    cmap = mpl.colormaps['Paired']
    has_pos = False
    has_neg = False
    if not has_pos and (data.flatten() > 0).any():
        has_pos = True
    if not has_neg and (data.flatten() < 0).any():
        has_neg = True
    if n == 2:
        _fill_between(t, data[0,:,:], cmap((6-0.5)/12), cmap((5-0.5)/12), ax, 0.7, label = 'OFF')
        _fill_between(t, data[1,:,:], cmap((2-0.5)/12), cmap((1-0.5)/12), ax, 0.7, ls = ':', label = 'OFF')
    if n == 1:
        _fill_between(t, data, cmap((4-0.5)/12), cmap((3-0.5)/12), ax, 0.7)

    ax.set_ylabel(ylabel, fontsize = fs)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize = fs) 
    else:
        ax.xaxis.set_ticklabels([])
    if has_pos and has_neg:
        return 0
    if has_pos and not has_neg:
        return 1
    if not has_pos and has_neg:
        return -1

def single_temporal_r(t, data, ax, ylabel, ylim, xlabel, c1, c2, ref = None, label = [None, None], lc = None, fs = 'medium'):
    lo = 0
    hi = 100
    d = data[0,:,:]
    mean = d.mean(axis = 0)
    low = np.percentile(d, lo, axis = 0)
    high = np.percentile(d, hi, axis = 0)
    ax.errorbar(t, mean, yerr = np.abs(np.array([mean-low, high-mean])), ls = '-', color = c1, lw = 2*lw0, capsize = 2, alpha = 0.7, label = label[0])
    d = data[1,:,:]
    mean = d.mean(axis = 0)
    low = np.percentile(d, lo, axis = 0)
    high = np.percentile(d, hi, axis = 0)
    ax.errorbar(t, mean, yerr = np.abs(np.array([mean-low, high-mean])), ls = ':', color = c2, lw = 2*lw0, capsize = 2, alpha = 0.7, label = label[1])
    if ref is not None:
        ax.plot(t,np.zeros(t.shape) + ref, ls = ':', c = 'gray', lw = 2*lw0, alpha = 0.7, label = 'input wave\nhalf-width')
    ax.set_xlabel(xlabel, fontsize = fs)
    if lc is None:
        ax.set_ylabel(ylabel, fontsize = fs)
    else:
        ax.set_ylabel(ylabel, color = lc, fontsize = fs)
    ax.set_ylim(ylim)

def temporal_sep(t, d_dict, ax, n, ylabel, ylim = None, xlabel = None, legend = False, c1 = None, c2 = None, c3 = None, i0 = 1, c = None, fw = False, fs = 'medium', theme = None):
    cmap = mpl.colormaps['Paired']
    if c1 is None:
        c1 = cmap((6-0.5)/12)
    else:
        c1 = cmap((c1-0.5)/12)
    if c2 is None:
        c2 = cmap((2-0.5)/12)
    else:
        c2 = cmap((c2-0.5)/12)
    if c3 is None:
        c3 = cmap((4-0.5)/12)
    else:
        c3 = cmap((c3-0.5)/12)
    if c is not None:
        assert(c.shape[-1] == 3)
        if n == 2:
            assert(len(c.shape) == 3)
            assert(c.shape[1] == 2)
        else:
            assert(len(c.shape) == 2)
    i = 0
    has_pos = False
    has_neg = False
    lw_ratio = 2
    if fw:
        _dlw = 0
    else:
        _dlw = dlw
    for data in d_dict.values(): 
        if n == 2:
            if len(data.shape) > 3:
                data = data.reshape(2,-1,data.shape[-1])
            d0 = data[0,:,:].mean(axis = 0)
            d1 = data[1,:,:].mean(axis = 0)

            if c is None:
                ax.plot(t, d0, '-', lw = lw_ratio*lw0 + i*_dlw, color = c1, alpha = 0.7)
                ax.plot(t, d1, ':', lw = lw_ratio*lw0 + i*_dlw, color = c2, alpha = 0.7)
            else:
                ax.plot(t, d0, '-', lw = lw_ratio*lw0 + i*_dlw, color = c[0,i,:], alpha = 0.7)
                ax.plot(t, d1, ':', lw = lw_ratio*lw0 + i*_dlw, color = c[1,i,:], alpha = 0.7)
            if i == 0 and legend:
                ax.legend(['ON', 'OFF']) 

            if not has_pos and (d0.flatten() > 0).any() or (d1.flatten() > 0).any():
                has_pos = True
            if not has_neg and (d0.flatten() < 0).any() or (d1.flatten() < 0).any():
                has_neg = True

        if n == 1:
            if len(data.shape) > 2:
                data = data.reshape(-1,data.shape[-1])
            d = data.mean(axis = 0)
            if c is None:
                ax.plot(t, d, lw = lw_ratio*lw0 + i*_dlw, color = c3, alpha = 0.7)
            else:
                if theme == 'width, same strength' and i == 0:
                    ax.plot(t, d, lw = lw_ratio*lw0 + i*_dlw, color = c[i,:], ls =':', alpha = 0.7)
                elif theme == 'gamma' and i == 0:
                    ax.plot(t, d, lw = lw_ratio*lw0 + i*_dlw, color = 'r', ls =':', alpha = 0.7)
                else:
                    ax.plot(t, d, lw = lw_ratio*lw0 + i*_dlw, color = c[i,:], alpha = 0.7)
            if not has_pos and (d.flatten() > 0).any():
                has_pos = True
            if not has_neg and (d.flatten() < 0).any():
                has_neg = True
        i += 1

    ax.set_ylabel(ylabel, fontsize = fs)
    if ylim is not None and not np.isnan(ylim).all():
        if np.isnan(ylim[1]):
            ax.set_ylim(bottom = ylim[0])
        elif np.isnan(ylim[0]):
            ax.set_ylim(top = ylim[0])
        else:
            ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize = fs) 
    else:
        ax.xaxis.set_ticklabels([])
    if has_pos and has_neg:
        return 0
    if has_pos and not has_neg:
        return 1
    if not has_pos and has_neg:
        return -1

def rdist_single(rdist, dis, label, ax, c, fixed_w = True, fs = 'medium'):
    ntp = rdist.shape[-1]
    lw = lw0 + dlw * np.arange(ntp)
    if fixed_w:
        _dlw = 0
    else:
        _dlw = dlw
    for i in range(ntp):
        match label:
            case 'ON':
                data = rdist[0,:,:,i].mean(0)
            case 'OFF':
                data = rdist[1,:,:,i].mean(0)
            case 'both':
                data = rdist[:,:,:,i].mean(0).mean(0)
        if i == 0 or i == ntp-1 or i == ntp//2:
            ax.plot(dis, data, c = c[i,:], lw = 2*lw0 + i*_dlw, alpha = 0.9, label = f'{i/(ntp-1)*100:.0f}% t')
        else:
            ax.plot(dis, data, c = c[i,:], lw = 2*lw0 + i*_dlw, alpha = 0.9)
    ax.legend(fontsize = 'small')
    if label == 'both':
        ax.set_ylabel('conn. weight', fontsize = fs)
    else:
        ax.set_ylabel(label + ' weight', fontsize = fs)
    ax.set_xlabel('radial distance', fontsize = fs)

def temporal_rdist(rdist, dis, dtype, label, ax, radius = 0, c = None, fw = False, fs = 'medium', normalized = False, legs = 'medium', theme = None):
    color = {'ON': 'r', 'OFF':'b', 'both' : 'g'}
    i = 0
    for key, value in rdist.items():
        if fw:
            lw = 2*lw0
        else:
            lw = 2*lw0 + i*dlw
        match dtype:
            case 'ON':
                data = value[0,:,:,-1].mean(0)
            case 'OFF':
                data = value[1,:,:,-1].mean(0)
            case 'both':
                data = value[:,:,:,-1].mean(0).mean(0)
        if radius == 0:
            rpick = np.ones(len(dis[key]), dtype = bool)
        else:
            rpick = dis[key] <= radius
        if normalized:
            data = data/np.max(data)
        if c is not None:
            if theme == 'width, same strength' and key == '10_ss_m':
                ax.plot(dis[key][rpick], data[rpick], c = c[i], ls = ':', lw = lw, alpha = 0.6, label = label[i])
            elif theme == 'gamma' and key == 'ss_m':
                ax.plot(dis[key][rpick], data[rpick], c = 'red', ls = ':', lw = lw, alpha = 0.6, label = f'{label[i]} (homeo.)')
            else:
                ax.plot(dis[key][rpick], data[rpick], c = c[i], lw = lw, alpha = 0.6, label = label[i])
        else:
            ax.plot(dis[key][rpick], data[rpick], c = color[dtype], lw = lw, alpha = 0.6, label = label[i])
        i += 1

    if theme is None:
        ax.legend(fontsize = legs, frameon = False)
    else:
        if theme == r'$w_0$':
            #ax.legend(fontsize = legs, frameon = False, ncols = 2, title = theme, handlelength = 1.0, handletextpad = 0.5, columnspacing = 1.0, loc = 'upper right')
            ax.legend(fontsize = legs, frameon = False, ncols = 3, handlelength = 1.0, handletextpad = 0.5, columnspacing = 1.0, loc = 'lower left', bbox_to_anchor = (0.1, 0.6))
            ax.text(0.92, 0.714, theme, fontsize = legs+2.5, transform = ax.transAxes, horizontalalignment = 'right')
            ylim = ax.get_ylim()
            ax.set_ylim(top = ylim[1]*1.5)
        if theme == 'LTD ratio':
            ax.legend(fontsize = legs, frameon = False, ncols = 2, handlelength = 1.0, handletextpad = 0.5, columnspacing = 1.0, loc = 'lower left', bbox_to_anchor = (0.45, 0.3), title = theme, title_fontsize = legs)
            ylim = ax.get_ylim()
            #ax.set_ylim(top = ylim[1]*1.3)
        if theme == 'width, same strength':
            ax.legend(fontsize = legs-1, frameon = False, ncols = 1, handlelength = 1.0, handletextpad = 0.3, columnspacing = 1.0, loc = 'lower left', bbox_to_anchor = (0.7, 0.25))
            ylim = ax.get_ylim()
            ax.set_xlim(right = 9)
        if theme == 'gamma':
            ax.legend(fontsize = legs-1, frameon = False, ncols = 1, handlelength = 1.0, handletextpad = 0.3, columnspacing = 1.0, loc = 'lower left', bbox_to_anchor = (0.55, 0.2), title = 'learning rate', title_fontsize = legs)
            ylim = ax.get_ylim()
            ax.set_xlim(right = 8)

    ax.tick_params(axis = 'both', labelsize = legs + 1)
    if normalized:
        ax.set_ylabel('norm. conn. weight', fontsize = fs)
    else:
        ax.set_ylabel('conn. weight', fontsize = fs)
    #ax.set_xlabel('radial dis. (#LGN)', fontsize = fs)
    ax.set_xlabel('radial distance', fontsize = fs)

def rdist_plot(rdist, dis, tpoint, ax):
    i = 0
    for key, value in rdist.items():
        ax.plot(dis[key], value[0,:,:,-1].mean(0), c = 'r', lw = lw0 + i*dlw, alpha = 0.6, label = key+'-ON')
        ax.plot(dis[key], value[1,:,:,-1].mean(0), c = 'b', lw = lw0 + i*dlw, alpha = 0.6, label = key+'-OFF')
        i += 1
    #ax.legend()
    ax.set_ylabel('conn. weight')
    ax.set_xlabel('radial distance (#LGN)')

def norm_temporal_sep(d_dict, ax, n, ylabel, ylim = None, xlabel = None, legend = False, i0 = 1, nbins = 25, c = None, fw = False, fs = 'medium'):
    cmap = mpl.colormaps['Paired']
    c1 = cmap((6-0.5)/12)
    c2 = cmap((2-0.5)/12)
    c3 = cmap((4-0.5)/12)
    data_len = [d.shape[-1] for d in d_dict.values()]
    i = 0
    lw_ratio = 2
    if fw:
        _dlw = 0
    else:
        _dlw = dlw
    for data in d_dict.values(): 
        data = data.mean(axis = 0)
        t = np.linspace(0, 100, data_len[i])
        if nbins > 0:
            data = movingAvg(data, data.size, data.size//nbins)
        if c is None:
            ax.plot(t, data, lw = lw_ratio*lw0 + i*_dlw, color = c3, alpha = 0.7)
        else:
            ax.plot(t, data, lw = lw_ratio*lw0 + i*_dlw, color = c[i,:], alpha = 0.7)
        i += 1

    ax.set_ylabel(ylabel, fontsize = fs)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize = fs) 
    else:
        ax.xaxis.set_ticklabels([])

def bar_end(d_dict, tpoint, ax, ylabel, ylim, cbar, cp, fs = 'medium', theme = None):
    n = len(d_dict)
    data = np.array([d.reshape(-1,d.shape[-1])[:,tpoint] if len(d.shape) > 2 else d[:,tpoint] for d in d_dict.values()])
    if data.shape[-1] > 1:
        p = [sig(data[i:i+2,:], 2) for i in range(n-1)]
    else:
        p = np.zeros(n-1)
    x = np.arange(n)
    xp = x[:-1] + 0.5
    for i in range(n):
        d_mean = data[i,:].mean()
        if theme == 'width, same strength' and i == 0:
            ax.bar(x[i], d_mean, color = 'None', ls = '--', lw = 1, ec = cbar[i,:])
        elif theme == 'gamma' and i == 0:
            ax.bar(x[i], d_mean, color = 'None', ls = '--', lw = 1, ec = 'r')
        else:
            ax.bar(x[i], d_mean, color = cbar[i,:])
        print(f'{ylabel}: {prod_label[i]}, {d_mean, np.std(data[i,:])}, [{np.percentile(data[i,:],10), np.percentile(data[i,:],90)}]')
        high = np.percentile(data[i,:],90)
        if theme == 'gamma' and i == 0:
            ax.errorbar(x[i], d_mean, yerr = np.array([d_mean - np.percentile(data[i,:],10), high - d_mean]).reshape(2,1), capsize = 4, capthick = 1, color = 'r')
        else:
            ax.errorbar(x[i], d_mean, yerr = np.array([d_mean - np.percentile(data[i,:],10), high - d_mean]).reshape(2,1), capsize = 4, capthick = 1, color = cbar[i,:])
        if i > 0 and p[i-1] > 0.005 and not (theme == 'gamma' and i == 1):
            prev_high = np.percentile(data[i-1,:],75)
            both_high = max(prev_high, high)
            ax.plot([x[i-1], x[i]], [both_high*1.1]*2, '-', color = cp[i,:], alpha = 0.7) 
            ax.plot([x[i-1], x[i-1]], [prev_high*1.05, both_high*1.1], '-', color = cp[i,:], alpha = 0.9) 
            ax.plot([x[i], x[i]], [high*1.05, both_high*1.1], '-', color = cp[i,:], alpha = 0.9) 
            print(f'T-test pvalue between {prod_label[i-1]} and {prod_label[i]} = {p[i-1]}')
    ax.set_xticks(x, labels = prod_label)
    ax.set_ylabel(ylabel, fontsize = fs)

def end_sep(d_dict, tpoint, ax, n, ylabel, ylim = None, c = None):
    cmap = mpl.colormaps['Paired']
    c1 = cmap((6-0.5)/12)
    c_1 = cmap((5-0.5)/12)
    c2 = cmap((2-0.5)/12)
    c_2 = cmap((1-0.5)/12)
    c3 = cmap((4-0.5)/12)
    c_3 = cmap((3-0.5)/12)
    has_pos = False
    has_neg = False
    if n == 2:
        data = np.array([d.reshape(2,-1,d.shape[-1])[:,:,tpoint] if len(d.shape) > 3 else d[:,:,tpoint] for d in d_dict.values() ])
        if not has_pos and (data.flatten() > 0).any():
            has_pos = True
        if not has_neg and (data.flatten() < 0).any():
            has_neg = True

        p = np.zeros((2,len(d_dict)-1)) 
        if data.shape[-1] > 1:
            p[0,:] = [sig(data[i:i+2,0,:], 2) for i in range(len(d_dict)-1)]
            p[1,:] = [sig(data[i:i+2,1,:], 2) for i in range(len(d_dict)-1)]
    else:
        data = np.array([d.reshape(-1,d.shape[-1])[:,tpoint] if len(d.shape) > 2 else d[:,tpoint] for d in d_dict.values()])
        if not has_pos and (data.flatten() > 0).any():
            has_pos = True
        if not has_neg and (data.flatten() < 0).any():
            has_neg = True
        if data.shape[-1] > 1:
            p = [sig(data[i:i+2,:], 2) for i in range(len(d_dict)-1)]
        else:
            p = np.zeros(len(d_dict)-1)
    low = np.percentile(data, 25, axis = -1)
    high = np.percentile(data, 75, axis = -1)
    print(f'{ylabel}: {np.std(data, axis = -1)}')
    mean = data.mean(-1)
    if prod_label is not None:
        xticklabel = prod_label
    else:
        xticklabel = [key for key in d_dict.keys()]
    ax2 = ax.twinx()
    x = np.arange(len(d_dict))
    xp = x[:-1] + 0.5
    if n == 2:
        ax.plot(x, mean[:,0], color = c1, lw = lw0*3)
        ax.plot(x, mean[:,1], color = c2, lw = lw0*3)
        ax.fill_between(x, low[:,0], high[:,0], color = c_1, ec = 'None', alpha = 0.7) 
        ax.fill_between(x, low[:,1], high[:,1], color = c_2, ec = 'None', alpha = 0.7) 
        ax2.plot(xp, p[0,:], '*', color = c1, alpha = 0.5) 
        ax2.plot(xp, p[1,:], '*', color = c2, alpha = 0.5) 
    if n == 1:
        ax.plot(x, mean, color = c3, lw = lw0*3)
        ax.fill_between(x, low, high, color = c_3, ec = 'None', alpha = 0.7)
        ax2.plot(xp, p, '*', color = c3, alpha = 0.5) 
    ax2.plot(xp, np.zeros_like(xp) + 0.05,  ':k', lw = lw0, alpha = 0.7) 
    ax2.set_ylabel('t. test')
    ax.set_xticks(np.arange(len(d_dict)), labels = xticklabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)

    if has_pos and has_neg:
        return 0
    if has_pos and not has_neg:
        return 1
    if not has_pos and has_neg:
        return -1

if __name__ == '__main__':
    theme = sys.argv[1]
    data_fdr = sys.argv[2]
    fig_fdr = sys.argv[3]
    suffix = sys.argv[4] 
    n = int(sys.argv[5])
    if sys.argv[6] == 'True' or sys.argv[6] == '1':
        separate = True
    else:
        separate = False

    if sys.argv[7] == 'True' or sys.argv[7] == '1':
        use_idx = True
    else:
        use_idx = False

    if sys.argv[8] == 'True' or sys.argv[8] == '1':
        single = True
    else:
        single = False

    if not use_idx:
        batch = dict()
        for i in range(n):
            batch[sys.argv[9+i*2]] = int(sys.argv[9+i*2+1])
            if single is True:
                assert(batch[sys.argv[9+i*2]] == 1)
        print(batch)
        num = None
    else:
        num = [i for i in range(n)]
        batch = None 

    random_trials_plot(theme, data_fdr, fig_fdr, suffix, separate, use_idx, num, batch, single)
