import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import sys

def series_plot(data_fdr, suffix, fig_fdr, suffix0, pvalue = 0.05):
    if fig_fdr[-1] != '/':
        fig_fdr = fig_fdr+'/'
    if data_fdr[-1] != '/':
        data_fdr = data_fdr+'/'
    n = len(suffix)
    print(suffix)
    input_halfwidth = np.zeros((2,n))
    for i in range(n):
        with open(data_fdr + 'metric_' + suffix[i] + '.bin', 'rb') as f:
            if i == 0:
                step0, nt_, nit, stage, nLGN_1D = np.fromfile(f, 'int', 5)
                dt = np.fromfile(f, 'f4', 1)[0]
                initial_dist = np.zeros((n,nLGN_1D,nLGN_1D))
                rstd = np.zeros((2,n,nit))
                rstd_norm = np.zeros((2,n,nit))
                max20 = np.zeros((2,n,nit))
                min0 = np.zeros((2,n,nit))
                radius = np.zeros((2,n,nit))
                cl = np.zeros((2,n,nit))
                EI_balance = np.zeros((n, nit))
                EI_balance_p = np.zeros((n, nit))
                g_std_p = np.zeros((n, nit))
                dis2c_p = np.zeros((n, nit))
                ctr_p = np.zeros((n, nit))
            else:
                _step0, _nt_, _nit, _stage, _nLGN_1D = np.fromfile(f, 'int', 5)
                _dt = np.fromfile(f, 'f4', 1)[0]
                assert(_nit == nit)
                assert(_stage == stage)
                assert(_nLGN_1D == nLGN_1D)

            if stage == 2 or stage == 3:
                input_halfwidth[:,i] = np.fromfile(f, 'f4', 2)
            else:
                assert(stage == 5)
                input_halfwidth[:,i] = np.fromfile(f, 'f4', 2*2)[:2]
            initial_dist[i, :, :] = np.fromfile(f, 'f8', nLGN_1D * nLGN_1D).reshape(nLGN_1D, nLGN_1D)
            rstd[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
            rstd_norm[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
            max20[:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
            min0[:,i,:] = np.fromfile(f, 'int', 2*nit).reshape(2,nit)
            radius[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
            cl[:,i,:] = np.fromfile(f, 'f8', 2*nit).reshape(2,nit)
            EI_balance[i,:] = np.fromfile(f, 'f8', nit)
            EI_balance_p[i,:] = np.fromfile(f, 'f8', nit)
            g_std_p[i,:] = np.fromfile(f, 'f8', nit)
            dis2c_p[i,:] = np.fromfile(f, 'f8', nit)
            ctr_p[i,:] = np.fromfile(f, 'f8', nit)

    fig = plt.figure('initial_dist', figsize = (3*n+3,3.5), dpi = 200)
    for i in range(n):
        ax = fig.add_subplot(1,n,i+1)
        cmap = 'bwr'
        im = ax.imshow(initial_dist[i, :, :], aspect = 'auto', origin = 'lower', cmap = plt.get_cmap(cmap))
        fig.colorbar(im, ax = ax)
        ax.set_title(f'sum:{np.sum(initial_dist[i,:,:].flatten())}, center sum:{np.sum(initial_dist[i,nLGN_1D//4:nLGN_1D*3//4,nLGN_1D//4:nLGN_1D*3//4].flatten()):.2e} \n on-off: {EI_balance[i, -1]:.3e}')
    fig.savefig(fig_fdr + 'initial_dist-' + suffix0 +'.png')

    fig = plt.figure('metric_plots', figsize = (18,4), dpi = 200)
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax22 = ax2.twinx()
    ax3 = fig.add_subplot(143)
    ax32 = ax3.twinx()
    ax4 = fig.add_subplot(144)
    ax42 = ax4.twinx()
    t = np.arange(nit)
    for i in range(n):
        ax1.plot(t, rstd[0,i,:], '-r', lw = i+1)
        ax1.plot(t, rstd[1,i,:], '-b', lw = i+1)
        ax1.set_ylim(0, nLGN_1D/2 + 0.5)

        pick = g_std_p[i,:] > pvalue
        ax2.plot(t, rstd_norm[0,i,:], '-r', lw = i+1)
        ax2.plot(t, rstd_norm[1,i,:], '-b', lw = i+1)
        ax2.plot(t[pick], rstd_norm[0,i,pick], '*r')
        ax2.plot(t[pick], rstd_norm[1,i,pick], '*b')
        ax2.set_ylim(0, nLGN_1D/2 + 0.5)
        ax22.plot(t, g_std_p[i,:], '-k', lw = i+1)

        pick = dis2c_p[i,:] > pvalue
        ax3.plot(t, radius[0,i,:], '-r', lw = i+1)
        ax3.plot(t, radius[1,i,:], '-b', lw = i+1)
        ax3.plot(t[pick], radius[0,i,pick], '*r')
        ax3.plot(t[pick], radius[1,i,pick], '*b')
        ax3.set_ylim(0, nLGN_1D/2 + 0.5)
        ax32.plot(t, dis2c_p[i,:], '-k', lw = i+1)

        pick = ctr_p[i,:] > pvalue
        ax4.plot(t, cl[0,i,:], '-r', lw = i+1)
        ax4.plot(t, cl[1,i,:], '-b', lw = i+1)
        ax4.plot(t[pick], cl[0,i,pick], '*r')
        ax4.plot(t[pick], cl[1,i,pick], '*b')
        ax4.set_ylim(0, nLGN_1D/2 + 0.5)
        ax42.plot(t, ctr_p[i,:], '-k', lw = i+1)

    ax1.set_title('total_averaged gauss. std.')
    ax2.set_title('cell-wise gauss. std.')
    ax3.set_title('averaged distance to center')
    ax4.set_title('sqrt(contour_area)')
    fig.savefig(fig_fdr + 'metric_plots-' + suffix0 +'.png', bbox_inches = 'tight')

    fig = plt.figure('min_max_plots', figsize = (11,4), dpi = 200)
    ax = fig.add_subplot(121)
    for i in range(n):
        ax.plot(t, max20[0,i,:], '-r', lw = (i+1)*0.3)
        ax.plot(t, max20[1,i,:], '-b', lw = (i+1)*0.3)
    ax.set_title('avg. #80%')
    ax = fig.add_subplot(122)
    for i in range(n):
        ax.plot(t, min0[0,i,:], '-r', lw = (i+1)*0.3)
        ax.plot(t, min0[1,i,:], '-b', lw = (i+1)*0.3)
    ax.set_title('avg. #0')
    fig.savefig(fig_fdr + 'min_max_plots-' + suffix0 +'.png')
    for i in range(n):
        print(i, EI_balance_p[i, -1])

if __name__ == '__main__':
    data_fdr = sys.argv[1]
    fig_fdr = sys.argv[2]
    suffix0 = sys.argv[3] 
    suffix = sys.argv[4:] 
    series_plot(data_fdr, suffix, fig_fdr, suffix0)
