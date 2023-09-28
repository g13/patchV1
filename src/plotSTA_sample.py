import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.gridspec as gs
import matplotlib.cbook as cbook
from matplotlib import cm
import sys, os
from sys import stdout
from readPatchOutput import *
sys.path.append(os.path.realpath('..'))
from ext_signal import apply_sRGB_gamma, LMS2sRGB
import warnings
import os 
from img_proc import create_gif
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

def singleplot(name, sta, dt, fdr, nsp, n_id, single_v = True):
    nt = sta.shape[0]
    height = sta.shape[2]
    width = sta.shape[3]
    npixel2 = height*width
    data = sta.reshape(nt, 3, npixel2)
    if not single_v:
        img = np.zeros((nt, height,width,3))
        l = data[:,0,:] - np.mean(data[:,0,:])
        m = data[:,1,:] - np.mean(data[:,1,:])
        s = data[:,2,:] - np.mean(data[:,2,:])
        ldev = np.abs(l).max(-1)
        mdev = np.abs(m).max(-1)
        sdev = np.abs(s).max(-1)
        imax = np.argmax(np.array((ldev,mdev,sdev)).max(0))

        l_spat_imin = np.argmin(l.var(0)*np.sign(l.sum(0)))
        lmin = l[:,l_spat_imin]
        l_spat_imax = np.argmax(l.var(0)*np.sign(l.sum(0)))
        lmax = l[:,l_spat_imax]

        m_spat_imin = np.argmin(m.var(0)*np.sign(m.sum(0)))
        mmin = m[:,m_spat_imin]
        m_spat_imax = np.argmax(m.var(0)*np.sign(m.sum(0)))
        mmax = m[:,m_spat_imax]

        s_spat_imin = np.argmin(s.var(0)*np.sign(s.sum(0)))
        smin = s[:,s_spat_imin]
        s_spat_imax = np.argmax(s.var(0)*np.sign(s.sum(0)))
        smax = s[:,s_spat_imax]

    else:
        img = np.zeros((nt, height,width))
        _v = data.sum(1)/3
        _v = _v - np.mean(_v)
        imax = np.argmax(np.abs(_v).max(-1))

        v_spat_imin = np.argmin(_v.var(0)*np.sign(_v.sum(0)))
        vmin = _v[:,v_spat_imin]
        v_spat_imax = np.argmax(_v.var(0)*np.sign(_v.sum(0)))
        vmax = _v[:,v_spat_imax]

    for i in range(nt):
        if not single_v:
            img[i,:,:,:] = apply_sRGB_gamma(np.matmul(LMS2sRGB, data[i,:,:])).T.reshape(height,width,3)
        else:
            img[i,:,:] = apply_sRGB_gamma(np.matmul(LMS2sRGB, data[i,:,:])).T.reshape(height,width,3).sum(-1)/3
        
    if not single_v:
        pmin = np.min(img)
        pmax = np.max(img)
        if pmin == pmax:
            img = img.astype('u1')
        else:
            img = np.round((img - pmin)/(pmax-pmin)*255).astype('u1')
    else:
        pmin = np.min(img)
        pmax = np.max(img)
        if pmin != pmax:
            img = (img - pmin)/(pmax-pmin)

    fn = []
    for i in range(nt):
        fig = plt.figure(figsize = (4,6), dpi = 200)
        ax = fig.add_subplot(4,1,(1,3))
        if not single_v:
            im = ax.imshow(img[i,:,:,:], aspect = 'equal', origin = 'lower', cmap = 'gray', vmin = 0, vmax = 255)
            ax.set_title(f'#{n_id}-{nsp}spikes\n pixel range:{np.round(pmin*255):.0f} ~ {np.round(pmax*255):.0f}')
            ax.plot(l_spat_imax%width-0.5, l_spat_imax//width-0.5, '*', mfc = 'r', mec = 'k', alpha = 0.2)
            ax.plot(m_spat_imax%width-0.5, m_spat_imax//width-0.5, '*', mfc = 'g', mec = 'k', alpha = 0.2)
            ax.plot(s_spat_imax%width-0.5, s_spat_imax//width-0.5, '*', mfc = 'b', mec = 'k', alpha = 0.2)
            ax.plot(l_spat_imin%width-0.5, l_spat_imin//width-0.5, 's', mfc = 'r', mec = 'w', alpha = 0.2)
            ax.plot(m_spat_imin%width-0.5, m_spat_imin//width-0.5, 's', mfc = 'g', mec = 'w', alpha = 0.2)
            ax.plot(s_spat_imin%width-0.5, s_spat_imin//width-0.5, 's', mfc = 'b', mec = 'w', alpha = 0.2)
        else:
            im = ax.imshow(img[i,:,:], aspect = 'equal', origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
            ax.set_title(f'#{n_id}-{nsp}spikes\n pixel range:{pmin:.2f} ~ {pmax:.2f}')
            ax.plot(v_spat_imax%width-0.5, v_spat_imax//width-0.5, '*', mfc = 'w', mec = 'k',alpha = 0.5)
            ax.plot(v_spat_imin%width-0.5, v_spat_imin//width-0.5, 's', mfc = 'k', mec = 'w',alpha = 0.5)
        ax.set_xlabel('deg')
        ax.set_ylabel('deg')
        fig.colorbar(im, ax=ax, location='right', shrink=0.7)
        ax = fig.add_subplot(4,1,4)
        if not single_v:
            ax.plot(np.arange(nt)*dt, lmax, '-*r')
            ax.plot(np.arange(nt)*dt, mmax, '-*g')
            ax.plot(np.arange(nt)*dt, smax, '-*b')
            ax.plot(np.arange(nt)*dt, lmin, ':sr')
            ax.plot(np.arange(nt)*dt, mmin, ':sg')
            ax.plot(np.arange(nt)*dt, smin, ':sb')
            range_min, range_max = ax.get_ylim()
            ax.plot([i*dt, i*dt], [range_min, range_max], ':k', lw = 1.5)
        else:
            ax.plot(np.arange(nt)*dt, vmax, '-*k')
            ax.plot(np.arange(nt)*dt, vmin, ':sk')
            range_min, range_max = ax.get_ylim()
            ax.plot([i*dt, i*dt], [range_min, range_max], ':r', lw = 1.5)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('activation')
        if i == imax:
            fig.savefig(fdr + f'{name}-max{i}.png')
        filename = fdr + f'_{name}-frame{i}.png'
        fig.savefig(filename)
        plt.close(fig)
        fn.append(filename)
    create_gif(fn, fdr + f'{name}', nt)# crop = None, text = None, fontsize = 20, l = 0.1, t = 0.1)     
    for f in fn:
        os.remove(f)

def plotSTA_sample(suffix, data_fdr, fig_fdr):
    if data_fdr[-1] != "/":
        data_fdr = data_fdr+"/"
    if fig_fdr[-1] != "/":
        fig_fdr = fig_fdr+"/"

    _output_suffix = "-" + suffix
    staFn = data_fdr + "STA_sample" + _output_suffix + ".bin"

    parameterFn = data_fdr + "patchV1_cfg" + _output_suffix + ".bin"
    prec, sizeofPrec, vL, vE, vI, vR, vThres, gL, vT, typeAcc, nE, nI, sRatioLGN, sRatioV1, frRatioLGN, convolRatio, nType, nTypeE, nTypeI, frameRate, inputFn, _virtual_LGN, tonicDep, noisyDep = read_cfg(parameterFn)
    with open(staFn, 'rb') as f:
        w, h, nt = np.fromfile(f, 'u4', 3)
        dt = np.fromfile(f, 'f4', 1)[0]
        nSample = np.fromfile(f, 'u4', 1)[0]
        sample_id = np.fromfile(f, 'u4', nSample)
        STA_size = nSample * nt*w*h*3
        STA_nsp = np.fromfile(f, 'u4', nSample)
        STA = np.fromfile(f, 'f4', STA_size).reshape((nSample, nt, 3, w, h))

    for i in range(8):
        print(f'{STA_nsp[i]} spikes for neuron {sample_id[i]}')
        if STA_nsp[i] > 0:
            singleplot(f'STA-{sample_id[i]}-' + suffix, STA[i,:,:,:,:]/STA_nsp[i], dt, fig_fdr, STA_nsp[i], sample_id[i])
         
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(sys.argv)
        raise Exception('not enough argument for plotSTA_sample(suffix, data_fdr, fig_fdr)')
    else:
        suffix = sys.argv[1]
        data_fdr = sys.argv[2]
        fig_fdr = sys.argv[3]
        plotSTA_sample(suffix, data_fdr, fig_fdr)
