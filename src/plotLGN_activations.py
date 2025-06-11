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
frame_rate = 60
#prod_label = [r'0.5$\times$', r'1$\times$', r'2$\times$', r'3$\times$', r'4$\times$']
#prod_label = ['2x', '3x', '4x']
#prod_t = np.array([1149, 575*2, 288*4, 192*6, 144*8])/frame_rate
prod_ratio = np.ones(5)
#prod_label = ['5', '7.5', '10', '12.5', '15']
prod_label = ['4', '6', '8', '10', '12']
prod_t = np.array([500, 537, 575, 612, 650])/frame_rate
prod_ratio = np.ones(5)
unit = '#LGN'
#prod_ratio = np.array([1, 0.43529])
lw0 = 1.0
dlw = 0.4
nLGNperV1 = 396

mpl.rcParams.update({'font.family': 'CMU Sans Serif', 'axes.unicode_minus' : False, 'mathtext.fontset': 'cm'})

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

def plotLGN_activations(theme, data_fdr, fig_fdr, suffix, single, batch):
    if suffix[-1] != '-' and theme[0] != '-':
        theme = '-' + theme 
    save_svg = True
    if fig_fdr[-1] != '/':
        fig_fdr = fig_fdr+'/'
    if data_fdr[-1] != '/':
        data_fdr = data_fdr+'/'

    nbatch = len(batch)
    greens, reds, blues, blacks = get_colors(nbatch)
    keys = batch.keys()
    fr = dict({})
    dt = dict({})
    dT = dict({})
    ii = 0
    for key, ntrial in batch.items():
        for i in range(ntrial):
            if single:
                num_suffix = key
            else:
                num_suffix = f'{key}{i}'
            with open(f'{data_fdr}LGN_activation-{suffix}{num_suffix}.bin', 'rb') as f:

                if i == 0:
                    _nt = np.fromfile(f, 'i4', 1)[0]
                    dt[key] = np.fromfile(f, 'f4', 1)[0]
                    # waves cannot be precisely aligned in one trial
                    approx_T = [0]
                    dT[key] = int(prod_t[ii]/dt[key])
                    k = 0
                    while approx_T[k] + dT[key] < _nt:
                        approx_T.append(int(round(k*prod_t[ii]/dt[key])))
                        k += 1
                    approx_T = approx_T[:k]
                    fr[key] = np.zeros((ntrial, len(approx_T), dT[key]))
                    print(prod_label[ii], k)
                else:
                    _ = np.fromfile(f, 'i4', 1)
                    _ = np.fromfile(f, 'f4', 1)
                _fr = np.fromfile(f, 'f8', _nt)
                for j in range(len(approx_T)):
                    fr[key][i,j,:] = _fr[approx_T[j]:approx_T[j] + dT[key]]/nLGNperV1
        ii += 1

    print(dT, dt)
    label_fs = 'medium'
    greens, reds, blues, blacks = get_colors(nbatch)
    if theme[1:] == 'width':
        fig = plt.figure('LGN_activation', figsize = (2.5,(nbatch + 1)*1.0), dpi = 120)
    else:
        fig = plt.figure('LGN_activation', figsize = (3.3, (nbatch + 1)*1.0), dpi = 120)
    i = 0
    activation = dict({})
    if theme[1:] == 'width':
        tlim = np.max([dT[key]*dt[key] for key in batch.keys()])
    for key, ntrial in batch.items():
        #nwave = 50 # fr[key].shape[1]
        nwave = fr[key].shape[1]
        activation[key] = np.zeros(ntrial*nwave)
        t = np.arange(dT[key])*dt[key]
        ax = fig.add_subplot(nbatch+1,1,i+2)
        data = fr[key][:,:nwave,:].reshape(ntrial*nwave,dT[key]) 
        #'''
        ax.plot(t, data.mean(0), color = greens[i], lw = 1)
        low = np.percentile(data, 10, axis = 0)
        high = np.percentile(data, 90, axis = 0)
        ax.fill_between(t, low, high, color = greens[i], alpha = 0.5, ec = 'None')
        #'''
        #ax.plot(t,data.T, alpha = 0.7)
        ylim = ax.get_ylim()
        ax.set_ylim(bottom = 0)
        if i == nbatch // 2:
            ax.set_ylabel('total LGN FR. (Hz)', fontsize = 'large')
        if theme[1:] == 'speed':
            tx, ty = 0.97, 0.76
            ax.text(tx, ty, f'{prod_label[i]} {theme[1:]}', transform = ax.transAxes, fontsize = label_fs, horizontalalignment = 'right')
            ax.set_ylim(top = 1.35*ylim[1])
        else:
            tx, ty = 0.95, 0.7
            ax.text(tx, ty, f'{prod_label[i]} {unit[1:]}', transform = ax.transAxes, fontsize = label_fs, horizontalalignment = 'right')
            ax.set_ylim(top = 1.15*ylim[1])

        if i < nbatch - 1 and len(np.unique(prod_ratio)) == 1:
            ax.set_xticklabels([])
        if i == nbatch - 1:
            ax.set_xlabel('t (s)', fontsize = label_fs)
        ax.set_xlim(left = 0)
        if theme[1:] == 'width':
            ax.set_xlim(right = tlim * 1.1)
        ax.tick_params(axis='both', which='major', labelsize=label_fs)
        activation[key] = np.sum(data, axis = 1) * prod_ratio[i]
        print(prod_label[i])
        print(activation[key].mean())
        print(activation[key].std())
        i += 1

    ax = fig.add_subplot(nbatch+1,1,1)
    i = 0
    for key in batch.keys():
        a_mean = activation[key].mean() 
        if i == 0:
            act0 = activation[key].mean()
            if theme[1:] == 'width':
                ratios = [1.0]
        else:
            print(f'{prod_label[0]} act_ratio over {prod_label[i]}: {act0/activation[key].mean()}')
            if theme[1:] == 'width':
                ratios.append(act0/activation[key].mean())


        ax.bar(i, a_mean, color = greens[i], alpha = 0.7)
        ax.errorbar(i, a_mean, yerr = np.abs(np.array([a_mean - np.percentile(activation[key], 10), np.percentile(activation[key], 90) - a_mean])).reshape(2,1), capsize = 4, capthick = 1.5, color = greens[i], alpha = 1.0)
        i += 1
    # F-test
    #_, p_value = stat.f_oneway(*[activation[key] for key in activation.keys()])
    d = [activation[key] for key in activation.keys()]
    p_value = stat.ttest_ind(d[0], d[1]).pvalue
    #ax.set_xticks(np.arange(nbatch), labels = prod_label, labelsize = label_fs)
    ax.set_xticks(np.arange(nbatch), labels = prod_label, fontsize = label_fs)
    if theme[1:] == 'width':
        ax.set_xlim(right = nbatch)
    ax.set_ylabel('avg. #spikes', fontsize = label_fs)
    if theme[1:] == 'speed':
        ax.set_title(f'LGN activation per 0.5x speed wave', fontsize = label_fs)
        chartBox = ax.get_position() 
        ax.set_position([chartBox.x0, chartBox.y0 + 0.125/nbatch, chartBox.width, chartBox.height*0.9]) 
    else:
        ax2 = ax.twinx()
        ax2.plot(np.arange(nbatch), ratios, '-*', color = 'gray', alpha = 0.5)
        ax2.set_ylim(0, top = 1.2)
        ax2.set_ylabel('initial conn. \n weight ratio', fontsize = label_fs)
        ax2.tick_params(axis = 'y', colors = 'gray')
        ax2.spines['right'].set_color('gray')
        ax.set_title(f'LGN activation per wave', fontsize = label_fs)
        chartBox = ax.get_position() 
        ax.set_position([chartBox.x0, chartBox.y0 + 0.125/nbatch, chartBox.width*0.85, chartBox.height*0.9]) 

    print(f'F-test p = {p_value:.2e}')
    ylim = ax.get_ylim()
    ax.set_ylim(top = ylim[1]*1.1)
    ax.tick_params(axis='y', which='major', labelsize=label_fs)
    if len(np.unique(prod_ratio)) > 1:
        fig.tight_layout()
    fig.savefig(fig_fdr + 'LGN_activation-' + suffix + theme + '.png', bbox_inches = 'tight')
    if save_svg:
        fig.savefig(fig_fdr + 'LGN_activation-' + suffix + theme + '.svg', bbox_inches = 'tight')

if __name__ == '__main__':
    theme = sys.argv[1]
    data_fdr = sys.argv[2]
    fig_fdr = sys.argv[3]
    suffix = sys.argv[4] 
    n = int(sys.argv[5])
    if sys.argv[6] == 'True' or sys.argv[6] == 'true' or sys.argv[6] == 1:
        single = True
    else:
        single = False

    batch = dict()
    for i in range(n):
        batch[sys.argv[7+i*2]] = int(sys.argv[7+i*2+1])
    print(batch)
    plotLGN_activations(theme, data_fdr, fig_fdr, suffix, single, batch)
