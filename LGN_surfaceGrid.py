import pdb
from scipy import integrate
import numpy as np
from cmath import *

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sys import stdout
import warnings
np.seterr(invalid = 'raise', under = 'ignore', over = 'ignore')

def nPosInRectangle(point, x0, x1, y0, y1, leftInclude = False, bottomInclude = False):
    if leftInclude:
        x_ready = np.logical_and(point[0,:] >= x0, point[0,:] <= x1)
    else:
        x_ready = np.logical_and(point[0,:] > x0, point[0,:] <= x1)
    if bottomInclude:
        y_ready = np.logical_and(point[1,:] >= y0, point[1,:] <= y1)
    else:
        y_ready = np.logical_and(point[1,:] > y0, point[1,:] <= y1)
    assert(x_ready.shape == point[0,:].shape)
    assert(y_ready.shape == point[1,:].shape)
    xy_ready = np.logical_and(x_ready, y_ready)
    assert(xy_ready.size == point.shape[1])
    return np.sum(xy_ready), np.nonzero(xy_ready)[0]

exist_data = False 
dpi = 2000
if exist_data:
    with open('LGN_surfaceID.bin', 'rb') as f:
        xs = np.fromfile(f, 'f8', count = 3)
        xmin = xs[0]
        mid = xs[1]
        xmax = xs[2]
        ys = np.fromfile(f, 'f8', count = 2)
        ymin = ys[0]
        ymax = ys[1]
        ids = np.fromfile(f, 'u4', count = 2)
        xmax_id = ids[0]
        ymax_id = ids[1]
        pos_ind = np.fromfile(f)
        pos_ind_fill = np.fromfile(f, dtype=bool)
    
else:
    parallel_fileL = 'p_repel_finalL.bin'
    parallel_fileR = 'p_repel_finalR2.bin'
    with open(parallel_fileL, 'rb') as f:
        nL = np.fromfile(f, 'u4', count = 1)[0]
        posL = np.fromfile(f, 'f8', count = 2*nL).reshape(2,nL)
        nbL = np.fromfile(f, 'u4', count = 1)[0]
        bPosL = np.fromfile(f, 'f8', count = nbL*2*3).reshape(nbL,2,3)
        np.fromfile(f, 'u4', count = nbL)
        np.fromfile(f, 'f8', count = 2)
        areaL = np.fromfile(f, 'f8', count = 1)[0]
    
    xmin = np.min(bPosL[:,0,1])
    mid = np.max(bPosL[:,0,1])
    
    with open(parallel_fileR, 'rb') as f:
        nR = np.fromfile(f, 'u4', count = 1)[0]
        posR = np.fromfile(f, 'f8', count = 2*nR).reshape(2,nR)
        nbR = np.fromfile(f, 'u4', count = 1)[0]
        bPosR = np.fromfile(f, 'f8', count = nbR*2*3).reshape(nbR,2,3)
        np.fromfile(f, 'u4', count = nbR)
        np.fromfile(f, 'f8', count = 2)
        areaR = np.fromfile(f, 'f8', count = 1)[0]
    
    xmax = np.max(bPosR[:,0,1]) + mid - np.min(bPosR[:,0,1])
    posR[0,:] = posR[0,:] + mid - np.min(bPosR[:,0,1])
    pos = np.hstack((posL,posR))
    ymin = np.min(np.vstack((bPosL[:,1,1], bPosR[:,1,1])))
    ymax = np.max(np.vstack((bPosL[:,1,1], bPosR[:,1,1])))
    assert((pos[1,:] > ymin).all())
    assert((pos[1,:] < ymax).all())
    
    nLGN = nL + nR
    print(f'nLGN {nLGN} = {nL} + {nR}')
    pos_ind = np.zeros((2,nLGN), dtype = 'u4')
    pos_ind_fill = np.zeros(nLGN, dtype = bool)
    
    fig = plt.figure('surface_chop', dpi = dpi)
    ax = fig.add_subplot(111)
    
    per_unit_area = 2*np.sqrt(3) # in cl^2
    # can be empirical choice
    clL = np.sqrt(areaL/nL/per_unit_area)*1.6
    clR = np.sqrt(areaR/nR/per_unit_area)*1.6
    cl = clL
    dline_x = cl*1.5  # empirical
    dline_y = cl
    x = []
    y = []
    idy = []
    x.append(xmin)
    iy = 0
    total_belonged = 0
    new = False # first column of the right eye
    right = False # 
    while x[-1] < xmax:
        print('\n')
        if not right and iy > 0:
            dline_x = cl
        if right:
            cl = clR
            dline_y = cl
            if not new:
                dline_x = cl
            else:
                dline_x = cl*1.5
        x.append(x[-1] + dline_x)
        y.append([ymin])
        idy.append([])
        #print(x)
        #print(y)
        n, ind_x = nPosInRectangle(pos, x[-2], x[-1], ymin, ymax, len(x) == 2, True)
        if total_belonged + n >= nL and not right:
            old_n = n
            n, ind_x = nPosInRectangle(pos, x[-2], mid, ymin, ymax, len(x) == 2, True)
            print(f'x[-2] = {x[-2]}, x[-1] = {x[-1]}, mid = {mid}, {total_belonged} + {n} >= {nL}, {old_n}->{n}')
            x[-1] = mid
            right = True # next column starts for the right eye
            assert(total_belonged + n == nL)
        else:
            breaked = False
            if iy > 0 and n > 0: 
                # adjust current_x
                x_new = x[-2] + 2*(np.mean(pos[0,ind_x]) - x[-2])
                n_new, ind = nPosInRectangle(pos, x[-2], x_new, ymin, ymax, len(x) == 2, True)
                if n == n_new:
                    if (ind - ind_x==0).all():
                        print('updated')
                        n = n_new
                        x[-1] = x_new
                        ind_x = ind
        print(f'n = {n}, x = {x[-1]}, dx/dline_x = {(x[-1]-x[-2])/dline_x}')
        print(f'# neurons: {ind_x.size}')
        belonged = 0
        #print(f'ind_x = {ind_x}')
        while y[iy][-1] <= ymax:
            y[iy].append(y[iy][-1]+dline_y)
            m, ind = nPosInRectangle(pos[:, ind_x], x[-2], x[-1], y[iy][-2], y[iy][-1], len(x) == 2, len(y[iy]) == 2)
            if m > 1:
                count = 0
                breaked0 = False
                m0 = m
                while m!=1:
                    if count == 0:
                        y[iy][-1] = np.mean(pos[1,ind_x[ind]])
                    else:
                        y[iy][-1] = np.mean(pos[1,ind_x[ind][ind1]])
                    #dy = y[iy][-1] - y[iy][-2]
                    #if m > 1:
                    #    y[iy][-1] = y[iy][-2] + dy*0.8
                    #else: # m == 0
                    #    y[iy][-1] = y[iy][-2] + dy*1.2
                    m, ind1 = nPosInRectangle(pos[:, ind_x[ind]], x[-2], x[-1], y[iy][-2], y[iy][-1], len(x) == 2, len(y[iy]) == 2)
                    count = count + 1
                    if count >= m0:
                        print(f'ix = {iy}, iy = {len(y[iy])}, {m0} -> {m}')
                        breaked0 = True
                        break
                print(f'ind = {ind}, ind_x[ind][{ind1}] = {ind_x[ind][ind1]}')
                pos_ind[0,ind_x[ind][ind1]] = iy
                pos_ind[1,ind_x[ind][ind1]] = len(y[iy])-2
                pos_ind_fill[ind_x[ind][ind1]] = True
                idy[iy].append(ind_x[ind][ind1])
                assert(len(x)-2 >= 0)
                assert(len(y[iy])-2 >= 0)
                if breaked0:
                    breaked = True
                    break
            elif m == 1:
                pos_ind[0,ind_x[ind]] = iy
                pos_ind[1,ind_x[ind]] = len(y[iy])-2
                pos_ind_fill[ind_x[ind]] = True
                idy[iy].append(ind_x[ind])
            else:
                assert(m==0)
                assert(len(ind)==0)
    
            belonged = belonged + m
    
            if iy == 0 or new:
                dy = y[iy][-1] - y[iy][-2]
                if dy < cl:
                    dline_y = (cl-dy) + cl
            else:
                clen_y = len(y[iy])
                if clen_y < len(y[iy-1]):
                    dline_y = y[iy-1][clen_y] - y[iy][-1]
            #print(f'dy = {dline_y}, y[{iy}, {len(y[iy])}] = {y[iy][-1]}, {belonged}/{n}')
            if belonged > n:
                print(f"belonged {belonged}, {np.sum(pos_ind_fill[ind_x])} == {n}")
                #breaked = True
                #break
        if breaked:
            break
        print(f'# of rows: {len(y[iy])}')
        if len(x) > 2 and len(y[iy]) > len(y[iy-1]) and not new:
            print(f'{len(y[iy])} <= {len(y[iy-1])}')
            break
        if new:
            new = False
        if belonged < n:
            print(f"belonged {belonged}, {np.sum(pos_ind_fill[ind_x])} == {n}")
            miss = np.logical_not(pos_ind_fill[ind_x])
            print(ind_x[miss])
            nmiss = np.sum(miss)
            for i in ind_x[miss]:
                print([i, iy, len(y[iy])])
                ax.plot(pos[0,i], pos[1,i], '*r', ms = 0.05)
            #for i in ind_x[pos_ind_fill[ind_x]]:
                #ax.plot(pos[0,i], pos[1,i], '*k', ms = 0.05)
            break
        #if total_belonged > nL:
        #    break
        iy = iy + 1
        total_belonged = total_belonged + n
        if total_belonged == nL: 
            print('new')
            new = True
            other = len(y)
            
        stdout.write(f'\r progress: {total_belonged}/{nLGN}\n')
        #r0 = r1
    assert(total_belonged == nLGN)
    assert(pos_ind_fill.all())
    
    xmax_id = np.max(pos_ind[0,:])
    ymax_id = np.max(pos_ind[1,:])
    print(f'xid range: {[np.min(pos_ind[0,:]), xmax_id]}')
    print(f'yid range: {[np.min(pos_ind[1,:]), ymax_id]}')
    assert(np.min(pos_ind[0,:]) >= 0)
    assert(np.min(pos_ind[1,:]) >= 0)
    assert(xmax_id <= len(x)-2)
    n_max = np.max([len(y[0]), len(y[other])]) - 1 # n = lines-1
    assert(ymax_id <= n_max-1) # max id = n-1
    
    for i in range(len(y)):
        if n_max > len(y[i]):
            n0 = (n_max - len(y[i]))//2
            if n0 > 0:
                for k in range(len(idy[i])):
                    pos_ind[1, idy[i][k]] = pos_ind[1, idy[i][k]] + n0

figId = plt.figure('surface_Id', dpi = dpi)
axId = figId.add_subplot(111)
    
xx = np.tile(np.arange(len(x)), (2,1))
yy = np.tile(np.array([0,n_max]), (len(x),1)).T
axId.plot(xx, yy, '-k', lw = 0.02)

xx = np.tile(np.array([0, xmax_id+1]), (n_max,1)).T
yy = np.tile(np.arange(n_max), (2,1))
axId.plot(xx, yy, '-k', lw = 0.02)

axId.plot([other,other], [0, n_max], '-g', lw = 0.02)

axId.plot(xx, yy, '-k', lw = 0.02)
axId.plot(pos_ind[0,:]+0.5, pos_ind[1,:]+0.5, ',g')
for i in range(len(idy)):
    if len(idy[i]) > 0:
        axId.plot(pos_ind[0,idy[i][0]]+0.5, pos_ind[1,idy[i][0]]+0.5, ',r')
        axId.plot(pos_ind[0,idy[i][-1]]+0.5, pos_ind[1,idy[i][-1]]+0.5, ',r')

axId.set_aspect('equal')
figId.savefig('LGNsurface_ID.png', dpi = dpi)

xx = np.array([x,x])
yy = np.tile(np.array([ymin, ymax]).T,(len(x),1)).T
ax.plot(xx, yy, '-k', lw = 0.02) # draw all columns
for i in range(len(x)-1):
    iy = np.array(y[i])
    yy = np.vstack((iy[1:], iy[1:]))
    xx = np.tile(np.array([x[i+1], x[i]]).T, (iy.size-1,1)).T
    ax.plot(xx, yy, '-k', lw = 0.02) # draw all rows

ax.plot([mid,mid], [ymin, ymax], '-g', lw = 0.02)
ax.plot(pos[0,pos_ind_fill], pos[1,pos_ind_fill], ',k')
ax.plot(pos[0,np.logical_not(pos_ind_fill)], pos[1,np.logical_not(pos_ind_fill)], 'sr', ms = 0.1)


ax.set_aspect('equal')
fig.savefig('LGNsurface_grid.png', dpi = dpi)
with open('LGN_surfaceID.bin', 'wb') as f:
    np.array([xmin, mid, xmax]).tofile(f)
    np.array([ymin, ymax]).tofile(f)
    np.array([xmax_id, ymax_id]).astype('u4').tofile(f)
    pos_ind.tofile(f)
    pos_ind_fill.tofile(f)
