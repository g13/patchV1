from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import special
from cmath import *
import sobol_seq as ss
import os
os.environ['NUMBAPRO_CUDALIB']='C:/Users/gueux/Miniconda3/envs/py36_7/Library/bin'
np.seterr(all='raise')
special.seterr(all='raise')
sobol_set = True

# f-function related
def fp(e,p,ab,s1=0.76,s2=0.1821):
    if e == 0:
        #print('e -> 0, return value -> inf')
        #return inf
        return 0 
    else:
        return np.power(1/np.cosh(p), 2*s2/(np.power(e/ab,s1) + np.power(e/ab,-s1)))
    
def fpp(e,p,ab,s1=0.76,s2=0.1821):
    if p == 0:
        return 0
    else:
        return fp(e,p,ab,s1,s2)*p

d_sech = lambda x: -2*(np.exp(x)-np.exp(-x))/np.power(np.exp(x)+np.exp(-x),2)

def d_sech_log(e,ab,s1,s2):
    if e == 0:
        #print('e -> 0, return value -> inf')
        #return inf
        return 0
    else:
        e_ab_s1 = np.power(e/ab,s1)
        e_ab_s1r = np.power(ab/e,s1)
        return -2*s2*s1/e*(e_ab_s1-e_ab_s1r)/np.power(e_ab_s1+e_ab_s1r,2)

def d_fp_e(e,p,ab,s1=0.76,s2=0.1821,eval_fp=np.nan):
    if e == 0:
        #print('e -> 0, return value -> inf')
        #return inf
        return 0
    else:
        if np.isnan(eval_fp):
            eval_fp = fp(e,p,ab,s1,s2)
        return eval_fp*np.log(1/np.cosh(p)) * d_sech_log(e,ab,s1,s2)
    
def d_fp_p(e,p,ab,s1=0.76,s2=0.1821,eval_fp=np.nan):
    if e == 0:
        #print('e -> 0, return value -> inf')
        #return inf
        return 0
    
    if np.isnan(eval_fp):
        val = 2*s2/(np.power(e/ab,s1) + np.power(e/ab,-s1)) * fp(e,p,ab,s1,s2) * np.cosh(p) * d_sech(p)
    else:
        val = 2*s2/(np.power(e/ab,s1) + np.power(e/ab,-s1)) * eval_fp * np.cosh(p) * d_sech(p)
    
    return val
    
    
def d_fpp_e(e,p,ab,s1=0.76,s2=0.1821,eval_fp=np.nan):
    if p == 0:
        return 0 
    else:
        return p*d_fp_e(e,p,ab,s1,s2,eval_fp)
    
def d_fpp_p(e,p,ab,s1=0.76,s2=0.1821,eval_fp=np.nan):
    if p == 0:
        return 1
    else:
        if np.isnan(eval_fp):
            eval_fp = fp(e,p,ab,s1,s2)
        return p*d_fp_p(e,p,ab,s1,s2,eval_fp) + eval_fp
        
# e,p -> x,y mapping
def R_ep(e,p,ab,s1=0.76,s2=0.1821,eval_fp=np.nan,cos_eval_fp=np.nan):
    if np.isnan(eval_fp):
        eval_fp = fp(e,p,ab,s1,s2)
    if np.isnan(cos_eval_fp):
        cos_eval_fp = np.cos(eval_fp*p)
    return ab*ab + e*e + 2*ab*e*cos_eval_fp
    
def x_ep(e,p,k,a,b,s1=0.76,s2=0.1821):
    ratio = R_ep(e,p,a,s1,s2)/R_ep(e,p,b,s1,s2)
    return k/2*np.log(ratio) - k*np.log(a/b)

def e_x(x,k,a,b):
    xr = np.exp(2*x/k + 2*np.log(a/b)) 
    sqrtb2_4ac = np.sqrt(4*np.power(b*xr-a,2) - 4*(xr-1)*(b*b*xr-a*a))
    e0 = ((a-b*xr)*2 - sqrtb2_4ac)/2/(xr-1)
    return e0
    
def phi_ep(e,p,ab,s1=0.76,s2=0.1821):
    eval_fpp = fpp(e,p,ab,s1,s2)
    return np.arctan(e*np.sin(eval_fpp)/(ab+e*np.cos(eval_fpp)))

def y_ep(e,p,k,a,b,s1=0.76,s2=0.1821):
    return k*(phi_ep(e,p,a,s1,s2) - phi_ep(e,p,b,s1,s2))

# derivatives
def d_x_e(e,p,k,a,b,s1=0.76,s2=0.1821):
    eval_fp = fp(e,p,a,s1,s2)
    val = (a*np.cos(eval_fp*p) + e - a*e*np.sin(eval_fp*p)*d_fpp_e(e,p,a,s1,s2,eval_fp))/R_ep(e,p,a,s1,s2,eval_fp)
    
    eval_fp = fp(e,p,b,s1,s2)
    val -= (b*np.cos(eval_fp*p) + e - b*e*np.sin(eval_fp*p)*d_fpp_e(e,p,b,s1,s2,eval_fp))/R_ep(e,p,b,s1,s2,eval_fp)
    return val*k

def d_x_p(e,p,k,a,b,s1=0.76,s2=0.1821):
    eval_fp = fp(e,p,a,s1,s2)
    val = -a*e*np.sin(eval_fp*p)*d_fpp_p(e,p,a,s1,s2,eval_fp)/R_ep(e,p,a,s1,s2,eval_fp) 
    eval_fp = fp(e,p,b,s1,s2)
    val -= -b*e*np.sin(eval_fp*p)*d_fpp_p(e,p,b,s1,s2,eval_fp)/R_ep(e,p,b,s1,s2,eval_fp) 
    return val*k

def d_phi_p(e,p,ab,s1=0.76,s2=0.1821,eval_fp=np.nan):
    if np.isnan(eval_fp):
        eval_fp = fp(e,p,ab,s1,s2)
    return e*d_fpp_p(e,p,ab,s1,s2,eval_fp)*(e+ab*np.cos(eval_fp*p)) / R_ep(e,p,ab,s1,s2,eval_fp)

def d_y_p(e,p,k,a,b,s1=0.76,s2=0.1821):
    return k*(d_phi_p(e,p,a,s1,s2) - d_phi_p(e,p,b,s1,s2))

def d_phi_e(e,p,ab,s1=0.76,s2=0.1821,eval_fp=np.nan):
    if np.isnan(eval_fp):
        eval_fp = fp(e,p,ab,s1,s2)
    return (ab*np.sin(eval_fp*p) + (ab*e*np.cos(eval_fp*p) + e*e)*d_fpp_e(e,p,ab,s1,s2,eval_fp)) / R_ep(e,p,ab,s1,s2,eval_fp)

def d_y_e(e,p,k,a,b,s1=0.76,s2=0.1821):
    return k*(d_phi_e(e,p,a,s1,s2) - d_phi_e(e,p,b,s1,s2))

# intergrand
def model_block_ep(e,p,k,a,b,s1=0.76,s2=0.1821):
    eval_fp_a = fp(e,p,a,s1,s2)
    d_eval_fpp_a_e = d_fpp_e(e,p,a,s1,s2,eval_fp_a)
    d_eval_fpp_a_p = d_fpp_p(e,p,a,s1,s2,eval_fp_a)
    cos_eval_fpp_a = np.cos(eval_fp_a*p)
    sin_eval_fpp_a = np.sin(eval_fp_a*p)
    R_ep_a = R_ep(e,p,a,s1,s2,eval_fp_a,cos_eval_fpp_a)

    eval_fp_b = fp(e,p,b,s1,s2)
    d_eval_fpp_b_e = d_fpp_e(e,p,b,s1,s2,eval_fp_b)
    d_eval_fpp_b_p = d_fpp_p(e,p,b,s1,s2,eval_fp_b)
    cos_eval_fpp_b = np.cos(eval_fp_b*p)
    sin_eval_fpp_b = np.sin(eval_fp_b*p)
    R_ep_b = R_ep(e,p,b,s1,s2,eval_fp_b,cos_eval_fpp_b)

    val  = (a*cos_eval_fpp_a + e - a*e*sin_eval_fpp_a*d_eval_fpp_a_e)/R_ep_a
    val -= (b*cos_eval_fpp_b + e - b*e*sin_eval_fpp_b*d_eval_fpp_b_e)/R_ep_b
    dxe = val*k

    d_phi_a_p = e*d_eval_fpp_a_p*(e+a*cos_eval_fpp_a)/R_ep_a
    d_phi_b_p = e*d_eval_fpp_b_p*(e+b*cos_eval_fpp_b)/R_ep_b
    dyp = k*(d_phi_a_p - d_phi_b_p)

    val  = -a*e*sin_eval_fpp_a*d_eval_fpp_a_p/R_ep_a
    val -= -b*e*sin_eval_fpp_b*d_eval_fpp_b_p/R_ep_b
    dxp = val*k

    d_phi_a_e = (a*sin_eval_fpp_a + (a*e*cos_eval_fpp_a + e*e)*d_eval_fpp_a_e)/R_ep_a
    d_phi_b_e = (b*sin_eval_fpp_b + (b*e*cos_eval_fpp_b + e*e)*d_eval_fpp_b_e)/R_ep_b
    dye = k*(d_phi_a_e - d_phi_b_e)
    return dxe*dyp - dxp*dye

model_block_ep0 = lambda e,p,k,a,b,s1,s2: d_x_e(e,p,k,a,b,s1,s2)*d_y_p(e,p,k,a,b,s1,s2) - d_x_p(e,p,k,a,b,s1,s2)*d_y_e(e,p,k,a,b,s1,s2)

def is_point_inside(px, py, x, y, n, d='clockwise'):
    select = np.zeros(n, dtype=bool)
    for i in range(n):
        phase = np.unwrap(np.arctan2(y-py[i],x-px[i]))
        phase_path = np.sum(np.diff(phase))
        if d == 'clockwise':
            phase_path = -phase_path
        if (2*np.pi - phase_path) < 1e-10*2*np.pi:
            select[i] = True
    return select

def get_pos_3d(x,y,area,n,skip=602):
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    storming_area = (xmax-xmin)*(ymax-ymin)
    ratio = storming_area/area
    pos = np.empty((3,n), dtype=float)
    ntmp = np.ceil(n*ratio).astype(int)
    i = 0
    count = 0
    max_trial = 10
    while count < max_trial:
        #irands = np.empty((2,ntmp), dtype='uint32')
        #print(f'generating {count}: ({i}+{ntmp})/{n}')
        # sobol sequence
        if sobol_set:
            rands = ss.i4_sobol_generate(3, ntmp, skip=skip) 
            # add noise to avoid complete overlap
            rands = rands * (1 + np.random.normal(0, 0.05, (ntmp,3)))
        else:
            rands = np.random.rand(ntmp, 3) # transformed compared with sobol

        px = xmin + (xmax-xmin) * rands[:,0]
        py = ymin + (ymax-ymin) * rands[:,1]
        pz = 0.01 * rands[:,2]
        #print(f'selecting {count}: ({i}+{ntmp})/{n}')
        select = is_point_inside(px, py, x, y, ntmp)
        nselected = sum(select)
        if i+nselected > n:
            nselected = n-i
        pos[0,i:i+nselected] = (px[select])[:nselected]
        pos[1,i:i+nselected] = (py[select])[:nselected]
        pos[2,i:i+nselected] = (pz[select])[:nselected]
        #print(f'selected {count}: ({i}+{nselected})/{n}')
        i += nselected
        count += 1
        #print(f'{count}: {ntmp}/{n}')
        if i < n:
            ntmp = max(np.ceil((n-i)*ratio).astype(int),10)
            if sobol_set:
                skip = skip + ntmp
        else:
            assert(i==n)
            break
    if count == max_trial and i < n:
        raise Exception(f'after {count} round of sobol rands still left {ntmp} points to be assigned')
    return pos

def cut_blocks(pos,iblock,nblock,total_block,block_area,e0,e1,model_block,get_x,get_y,ax=None,skip=602,s1=0.76,s2=0.1821,blockSize=1024,block_tol=1e-10,max_it=10,bp=32):
    p = np.linspace(-np.pi/2,np.pi/2,nblock+1)
    e_range = np.exp(np.linspace(np.log(e0+1),np.log(e1+1),bp))-1
    cut_p = np.zeros(nblock) + p[0]
    for i in range(nblock):
        it = 0
        if i < nblock-1:
            if i == 0:
                p0 = p[0]
            else:
                p0 = cut_p[i-1]
            p1 = p[i+1]
            r = integrate.dblquad(model_block,e0,e1,p0,p1)
            area = r[0]
            delta = (block_area-area)/block_area
            while np.abs(delta) > block_tol and it < max_it:
                dblock = lambda e: model_block(p1,e)
                r = integrate.quad(dblock,e0,e1)
                darea = r[0]
                if it > 0:
                    r = integrate.dblquad(model_block,e0,e1,p0,p1)
                    area = r[0]
                    delta = (block_area-area)/block_area
                it += 1
                p1 += (block_area-area)/darea

                #stdout.write(f'\r{(iblock+i+1)/total_block*100:.3f}%: iter #{it} , {i+1}/{nblock}')
            cut_p[i] = p1
        else:
            p0 = cut_p[i-1]
            cut_p[-1] = p[-1]
            p1 = p[-1]
            #stdout.write(f'\r{(iblock+i+1)/total_block*100:.3f}%: iter #0 , {i+1}/{nblock}')
        #clockwise boudnary
        x = [get_x(ecc,p1) for ecc in e_range]
        y = [get_y(ecc,p1) for ecc in e_range]
        if ax is not None and i<nblock-1:
            ax.plot(x, y, 'k', lw = 0.2)
        p_block = np.linspace(p0,p1,bp)
        x += [get_x(e1,polar) for polar in np.flipud(p_block)]
        y += [get_y(e1,polar) for polar in np.flipud(p_block)]
        x += [get_x(ecc,p0) for ecc in np.flipud(e_range)]
        y += [get_y(ecc,p0) for ecc in np.flipud(e_range)]
        x += [get_x(e0,polar) for polar in p_block]
        y += [get_y(e0,polar) for polar in p_block]
        x += [x[0]]
        y += [y[0]]
        x = np.array(x)
        y = np.array(y)
        pos[iblock+i,:,:] = get_pos_3d(x,y,block_area,blockSize,skip)
        if ax is not None:
            ax.plot(pos[iblock+i,0,:],pos[iblock+i,1,:],',')
        stdout.write(f'\r{(iblock+i+1)/total_block*100:.3f}%: iter #{it} , {i+1}/{nblock}')
    #return pos

def plot_patch(a,b,k,ecc,softedge,total_target,ax=None,skip=602,s1=0.76,s2=0.1821,ret_fig=False,blockSize=32):

    model_block = lambda p, e: model_block_ep(e,p,k,a,b,s1,s2)
    np.random.seed(skip)
    get_x = lambda e, p:  x_ep(e,p,k,a,b,s1,s2)
    get_y = lambda e, p:  y_ep(e,p,k,a,b,s1,s2)

    e = np.exp(np.linspace(np.log(1),np.log(ecc+1),100))-1
    p = np.linspace(-np.pi/2,np.pi/2,100)
    xlim = x_ep(e[-1],0,k,a,b,s1,s2)
    tx = [get_x(x,np.pi/2) for x in e]
    bx = [get_x(x,-np.pi/2) for x in e]
    ty = [get_y(x,np.pi/2) for x in e]
    by = [get_y(x,-np.pi/2) for x in e]
    ylim = max(ty)
    fig_ratio = ylim/xlim
    if ax is None:
        fig = plt.figure('patch', figsize = (8, 8*fig_ratio), dpi=300)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
    rx = [get_x(ecc,x) for x in p]
    ry = [get_y(ecc,x) for x in p]
    ax.plot(tx, ty, 'k', lw = 0.5)
    ax.plot(bx, by, 'k', lw = 0.5)
    ax.plot(rx, ry, 'k', lw = 0.5)
    ## leave space for visual field expansion from compressed OD column stripes
    # characteristic inter-neuron distance
    adjusted_ecc = e_x(xlim-softedge,k,a,b)
    rx0 = [get_x(adjusted_ecc,x) for x in p]
    ry0 = [get_y(adjusted_ecc,x) for x in p]
    ax.plot(rx0, ry0, ':k', lw =0.5)
    r = integrate.dblquad(model_block,0,adjusted_ecc,-np.pi/2,np.pi/2)
    total_area = r[0]
    block_area = total_area/total_target
    
    # characteristic length for block
    pos = np.zeros((total_target,3,blockSize))
    cl = np.sqrt(block_area)
    print(f'characteristic block width = {cl}')


    # use longitudinal lines as initial values for Newton's iterative method 
    width_tol = 0.2
    nslice = np.round(xlim/cl*(1+width_tol)).astype(int)
    block_tol = 1e-10
    i = 0
    max_it = 10
    cut_ecc = np.zeros(nslice+1)
    xl = np.zeros(nslice+1)
    nblock_slice = np.zeros(nslice)
    cut_ecc[i] = 0
    cx = 0
    xl_old = cx
    ntarget_acquired = 0
    ecc0 = lambda x: (a-b*np.exp(x))/(np.exp(x) - 1)
    area_acquired = 0
    cover = False
    while ntarget_acquired < total_target and not cover: 
        i += 1
        cecc = ecc0((xl_old + cl)/k + np.log(a/b))
        if xlim-xl_old < (1+width_tol)*cl or cecc > adjusted_ecc:
            cut_ecc[i] = adjusted_ecc
            cecc = adjusted_ecc
            r = integrate.dblquad(model_block,cut_ecc[i-1],adjusted_ecc,-np.pi/2,np.pi/2)
            area = r[0]
            target = np.round(area/block_area).astype(int)
            assert(np.abs(target - area/block_area) < 1e-10*total_target)
            area_acquired += area
            cover = True
        else:
            cx = xl_old + cl
            assert(cecc > cut_ecc[i-1])
            xl = cl
            r = integrate.dblquad(model_block,cut_ecc[i-1],cecc,-np.pi/2,np.pi/2)
            area = r[0]
            current = area/block_area
            target = np.round(current).astype(int)
            delta = (target-current)/target
            target_area = target*block_area
            it = 0
            # Newton's method
            #print(f'starting with {current:.3f}/{target}')
            while (np.abs(delta) > block_tol*target and it < max_it) or xl < cl*(1-width_tol) or xl > cl*(1+width_tol):
                if xl > cl*(1+width_tol) and np.abs(delta) <= block_tol*target:
                    target -= 1
                    target_area -= block_area
                    it = 1
                    #print(f'xl {xl:.4e} > cl {cl:.4e} too big')

                if xl < cl*(1-width_tol) and np.abs(delta) <= block_tol*target:
                    if target+ntarget_acquired < total_target:
                        target += 1
                        target_area += block_area
                        it = 1
                    else:
                        break
                    #print(f'xl {xl:.4e} < cl {cl:.4e} too small')
                dslice = lambda p: model_block(p, cecc)
                r = integrate.quad(dslice,-np.pi/2,np.pi/2)
                darea = r[0]
                if it > 0:
                    r = integrate.dblquad(model_block,cut_ecc[i-1],cecc,-np.pi/2,np.pi/2)
                    area = r[0]
                    current = area/block_area
                    delta = (target-current)/target
                    cx = x_ep(cecc,0,k,a,b,s1,s2)
                    xl = cx - xl_old
                it += 1
                #print(f'{it}:{current:.4e}/{target} ~ {delta*100:.2f}%, ecc = {cecc:.4e}, darea = {darea:.4e}')
                cecc += (target_area-area)/darea
            #stdout.write(f'\r{it} iterations, xl = {xl:.4f}, {target} targets\n')
            xl_old += xl
            cut_ecc[i] = cecc
            slice_x = np.array([get_x(cecc,x) for x in p])
            slice_y = np.array([get_y(cecc,x) for x in p])
            ax.plot(slice_x, slice_y, 'k', lw=0.2)
            cx = get_x(cecc,0)
            nblock_slice[i-1] = target
            area_acquired += area

        cut_blocks(pos,ntarget_acquired,target,total_target,block_area,cut_ecc[i-1],cecc,model_block,get_x,get_y,ax,skip,s1,s2,blockSize)

        ntarget_acquired += target

        stdout.write(f'\r{ntarget_acquired/total_target*100:.3f}% slice #{i-1}')
        assert(np.abs(ntarget_acquired/total_target - area_acquired/total_area) < 1e-10*ntarget_acquired)
        assert(ntarget_acquired <= total_target)
        stdout.flush()

    assert(np.abs(area_acquired - total_area)/total_area < 1e-10*total_target)
    assert(ntarget_acquired == total_target)
    
    ax.set_aspect('equal')
    ax.set_ylim(-ylim,ylim)
    ax.set_xlim(0,xlim)
    if ret_fig:
        return fig, pos
    else:
        return pos
