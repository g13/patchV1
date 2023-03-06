import numpy as np
from scipy.stats import qmc
from scipy import integrate
import warnings
import matplotlib.pyplot as plt
from sys import stdout
#warnings.filterwarnings("once", message = "The balance properties of Sobol' points require")
warnings.filterwarnings("ignore", message = "The balance properties of Sobol\' points require", category=UserWarning)

def square_pos(per_dis, n, center, rng = None):
    pos = np.zeros((2,n))
    if rng is None:
        sampler = qmc.Sobol(d=2)
        rands = sampler.random(n)
        pos[0,:] = (rands[:,0]-1/2)*per_dis + center[0]
        pos[1,:] = (rands[:,1]-1/2)*per_dis + center[1]
    else:
        m = int(np.sqrt(n))
        if m*m == n:
            x, y = np.meshgrid(np.arange(m) + 0.5,np.arange(m) + 0.5)
            x = x.flatten()/m
            y = y.flatten()/m
            assert(x.size == y.size)
            assert(x.size == n)
            pos[0,:] = center[0] - per_dis/2 + per_dis*x
            pos[1,:] = center[1] - per_dis/2 + per_dis*y
            
        else:
            sampler = qmc.Sobol(d = 2)
            sample = sampler.random(n = n)
            pos = qmc.scale(sample, [center[0] - per_dis/2, center[1] - per_dis/2], [center[0] + per_dis/2, center[1] + per_dis/2]).T
        # permutes of exc and inh
        idx = rng.permutation(np.arange(n))
        pos[0,:] = pos[0,idx]
        pos[1,:] = pos[1,idx]
    return pos

def square_OP(center, r, phase0, phase, pos, clockwise):
    dis = pos.T - center
    op = np.arctan2(dis[:,1], dis[:,0])/2
    op = op + phase/2
    if clockwise:
        op = -op
    op = op + phase0/2

    while (np.abs(op) >= np.pi/2).any():
        dpick1 = op > np.pi/2
        dpick2 = op < -np.pi/2
        op[dpick1] = op[dpick1] - np.pi
        op[dpick2] = op[dpick2] + np.pi
    pick = (np.abs(dis) < r).all(1)
    return op, pick


def assign_square_xyID(n, pos, ratio = 2):
    xyID = np.zeros((2,n), dtype = 'i4')
    w = int(np.ceil(np.sqrt(n)))
    xy = np.zeros((2,w*w), dtype = 'i4')
    xy[0,:] = np.tile(np.arange(w), w)
    xy[1,:] = np.repeat(np.arange(w), w)
    nonreplace_pick = np.ones(w*w, dtype = bool)
    pos_id = np.arange(w*w)
    pos = pos*w
    xy = xy.T
    max_dis = w
    for i in range(n):
        dis = np.max(np.abs(pos[:,i] - xy), axis = 1)
        local_pick = np.logical_and(dis <= max_dis, nonreplace_pick)
        if local_pick.sum() == 0:
            raise Exception(f'no free tile point within {max_dis}')
        local_dis = dis[local_pick]
        local_id = pos_id[local_pick]
        imin = np.argmin(local_dis)
        xyID[:,i] = xy[local_id[imin],:]
        if xyID[0,i] == 0 and xyID[1,i] == 0:
            print(f'{i}:{xyID[:,i]}')
        if xyID[0,i] == 0 and xyID[1,i] == 1:
            print(f'{i}:{xyID[:,i]}')
        if xyID[0,i] == 1 and xyID[1,i] == 0:
            print(f'{i}:{xyID[:,i]}')
        if xyID[0,i] == 1 and xyID[1,i] == 1:
            print(f'{i}:{xyID[:,i]}')
        nonreplace_pick[local_id[imin]] = False 
        stdout.write(f'\r {i/n*100:.1f}%')
    assert(np.logical_not(nonreplace_pick).sum() == n)
    assert(nonreplace_pick.sum() == w*w - n)
    stdout.write('\n')
    return xyID, w

def sobol_disk_pos(ecc, n):
    sampler = qmc.Sobol(d = 2)
    m = int(np.ceil(4*n/np.pi))
    sample = sampler.random(n = m)
    _pos = qmc.scale(sample, [- ecc, -ecc], [ecc, ecc]).T
    r = np.linalg.norm(_pos, axis = 0)
    pos = _pos[:, r < ecc]
    if pos.shape[1] < n:
        print(f'short of {n - pos.shape[1]} LGN cells')
        _ecc = ecc / np.sqrt(2)
        _sample = sampler.random(n = n-pos.shape[1])
        _pos = qmc.scale(_sample, [-_ecc, -_ecc], [_ecc, _ecc]).T
        print(_pos.shape)
        print(pos.shape)
        cart = np.hstack((pos, _pos))
    else:
        cart = pos[:, :n]
    return cart

def poisson_disk_pos(ecc, n, ratio, seed):    
    m = int(np.ceil(4*n/np.pi))
    _radius = ratio/np.sqrt(m)
    print(f'#candidates = {m}, radius = {_radius:.2f}, seed = {seed}')
    engine = qmc.PoissonDisk(d=2, radius = _radius, ncandidates = m, seed = seed)
    #sample = engine.random(n = m).T
    sample = engine.fill_space().T
    _pos = (sample - 0.5)*2*ecc
    r = np.linalg.norm(_pos, axis = 0)
    pos = _pos[:, r < ecc]
    if pos.shape[1] < n:
        print(f'short of {n - pos.shape[1]} LGN cells')
        while pos.shape[1] < n:
            m = int(np.ceil(4*(n-pos.shape[1])/np.pi))
            #sample = engine.random(n = m).T
            sample = engine.fill_space().T
            _pos = (sample - 0.5)*2*ecc
            r = np.linalg.norm(_pos, axis = 0)
            _pos = _pos[:, r < ecc]
            pos = np.hstack((pos, _pos))

        cart = pos
    if pos.shape[1] > n:
        print(f'exceeds {pos.shape[1] - n} LGN cells')
        idx = np.random.permutation(np.arange(pos.shape[1]))
        cart = pos[:,idx[:n]]
    else:
        cart = pos
    return cart


def poisson_disk_pos0(ecc, n, ratio, seed):    
    m = int(np.ceil(4*n/np.pi))
    _radius = ratio/np.sqrt(m)
    print(f'#candidates = {m}, radius = {_radius:.2f}, seed = {seed}')
    engine = qmc.PoissonDisk(d=2, radius = _radius, ncandidates = m, seed = seed)
    #sample = engine.random(n = m).T
    sample = engine.fill_space().T
    seed += 1
    _pos = (sample - 0.5)*2*ecc
    r = np.linalg.norm(_pos, axis = 0)
    pos = _pos[:, r < ecc]
    count = 0
    
    while pos.shape[1] != n and count < 100:
        if pos.shape[1] < n:
            print(f'short of {n - pos.shape[1]} LGN cells')
        else:
            print(f'extra {pos.shape[1] - n} LGN cells')
        # m = int(np.ceil(4*(n-pos.shape[1])/np.pi))
        # sample = engine.random(n = m).T
        engine = qmc.PoissonDisk(d=2, radius = _radius, ncandidates = m, seed = seed)
        sample = engine.fill_space().T
        seed += 1
        _pos = (sample - 0.5)*2*ecc
        r = np.linalg.norm(_pos, axis = 0)
        pos = _pos[:, r < ecc]
        count += 1
    if count == 100:
        raise Exception('exceeds trial count')
    cart = pos.copy()
    return cart, seed

#Fast Poisson Disk Sampling in Arbitrary Dimensions
#Robert Bridson∗
#University of British Columbia
def _try_new_seed(j, idx, isample, closed, pos, neighbor, cells, irow, icol, add_idx, sample_list, active_list, sample_pos, l, cell_size, notSquare):
    if len(neighbor[idx]) == 0: # get cell's neighbors if not already
        neighbor[idx] = _get_neighbor_idx(2, notSquare, irow[idx], icol[idx], l, add_idx)
    # get neighboring samples' position
    neighbor_cells = cells[neighbor[idx]]
    occupied = neighbor_cells[neighbor_cells >= 0]
    neighbor_pos = sample_pos[occupied,:]
    # get distances from neighboring samples
    r = np.linalg.norm(neighbor_pos - pos[j,:], axis = 1)
    if (r > cell_size).all(): # accept if no neighboring sample within minimum radius 
        isample += 1
        cells[idx] = isample
        sample_pos[isample, :] = pos[j, :]
        sample_list.append(idx)
        active_list.append(isample)
        #print(isample, idx, pos[j,:])
        closed = False 

    return isample, closed

def _try_add_sample(i0, isample, closed, pos, neighbor, cells, cx, cy, cr, irow, icol, add_idx, sample_list, active_list, sample_pos, n, l, cell_size, notSquare):
    for j in range(pos.shape[0]):
        for k in range(len(neighbor[i0])):
            idx = neighbor[i0][k]
            if cells[idx] < 0: # if cell not occupied 
                if pos[j,0] >= cx[idx] and pos[j,0] < cx[idx] + 1/l*cr[idx] and pos[j,1] >= cy[idx] and pos[j,1] < cy[idx] + 1/(l+notSquare): # if sample is in the cell
                    isample, closed = _try_new_seed(j, idx, isample, closed, pos, neighbor, cells, irow, icol, add_idx, sample_list, active_list, sample_pos, l, cell_size, notSquare)
                    if closed:
                        break;
        if isample == n-1: # reached required number of sample, exit
            break;
    return isample, closed
def _get_neighbor_idx(d, notSquare, row, col, l, add_idx):
    neighbor_idx = []
    for j in range(-d, d+1):
        if notSquare:
            ir = max(min(row + j, l),0)
        else:
            ir = max(min(row + j, l-1),0)
        _idx = ir*l + sum(add_idx < ir)
        for k in range(-d, d+1):
            if ir in add_idx:
                ic = max(min(col + k, l),0)
            else:
                ic = max(min(col + k, l-1),0)
            idx = _idx + ic
            if idx not in neighbor_idx:
                neighbor_idx.append(idx)
    return neighbor_idx

def fast_poisson_disk2d(len_of_edge, n, ratio = 1, m = 32, attempts = 1, attempts0 = 1, jiggle = 0.05, seed = 5617328, plot = False):
    rGen = np.random.default_rng(seed)
    sobol_sampler = qmc.Sobol(d = 2)
    l = int(np.sqrt(n))
    cell_size = 1/np.sqrt(n) * ratio
    dn = n - l*l
    if dn < l:
        notSquare = False 
    else:
        dn -= l
        notSquare = True
    insq = int(notSquare)
    add_idx = rGen.choice(np.arange(l+insq), dn, replace = False)
    print(f'grid = {l+insq}x{l} + {dn}, {cell_size}')
    cx = np.zeros(n)
    cy = np.zeros(n)
    cr = np.zeros(n)
    irow = np.zeros(n, dtype = int)
    icol = np.zeros(n, dtype = int)
    idx = 0
    for i in range(l+insq):

        if i in add_idx:
            ncol = l+1
        else:
            ncol = l

        cx[idx:idx+ncol] = np.arange(ncol)/ncol
        cy[idx:idx+ncol] = i/(l+insq)
        icol[idx:idx+ncol] = np.arange(ncol)
        irow[idx:idx+ncol] = i
        cr[idx:idx+ncol] = np.zeros(ncol) + l/(ncol)
        idx += ncol

    print(add_idx)
    assert(idx == n)

    cells = np.zeros(n, dtype = int) - 1
    sample_pos = np.zeros((n,2))
    idx = rGen.integers(low = 0, high = n, size = 1)[0]
    isample = 0
    cells[idx] = isample
    sample_pos[isample,0] = cx[idx] + rGen.random(1)*1/l*cr[idx]
    sample_pos[isample,1] = cy[idx] + rGen.random(1)*1/(l+insq)
    sample_list = [idx]
    active_list = [isample]
    neighbor = np.empty(n, dtype = object)
    for i in range(n):
        neighbor[i] = []
    neighbor[idx] = _get_neighbor_idx(2, notSquare, irow[idx], icol[idx], l, add_idx)
    n_new_seed = 0
    while len(active_list) > 0:
        iAttempt = 0
        closed = True 
        while iAttempt < attempts and closed:
            i = rGen.integers(low = 0, high = len(active_list), size = 1)[0]
            i0 = sample_list[active_list[i]]
            avail_cell = np.array(neighbor[i0])[cells[neighbor[i0]] < 0]
            nAvail = len(avail_cell)
            template_pos = rGen.random((nAvail,m,2))
            if nAvail > 0: # continue if there're cells available
                # get new sample positions in the annulus and available cells
                spos = np.vstack([np.vstack([cx[avail_cell[j]] + template_pos[j,:,0]*1/l*cr[avail_cell[j]], cy[avail_cell[j]] + template_pos[j,:,1]*1/(l+insq)]).T for j in range(nAvail)])
                #print(m*len(avail_cell))
                #print(spos.shape)
                pos0 = sample_pos[active_list[i],:]
                r = np.linalg.norm(spos - pos0, axis = 1)
                pick = np.logical_and(r < 2*cell_size, r > cell_size)
                pos = spos[pick, :]
                # accept or reject samples
                isample, closed = _try_add_sample(i0, isample, closed, pos, neighbor, cells, cx, cy, cr, irow, icol, add_idx, sample_list, active_list, sample_pos, n, l, cell_size, notSquare)
                if isample == n-1: # reached required number of sample, exit
                    break;
            if closed: # current attempt failed to add any new sample
                iAttempt += 1
                if iAttempt == attempts: # no more attempt to find new samples around this sample 
                    active_list.pop(i)
        if len(active_list) == 0 and isample < n-1:
            avail_cell = np.arange(n)[cells < 0]
            assert(len(avail_cell) > 0)
            #template_pos = rGen.random((m*attempts0,2))
            template_pos = sobol_sampler.random(n = m*attempts0)
            for i in avail_cell:
                pos = np.vstack([cx[i] + template_pos[:,0]*1/l*cr[i], cy[i] + template_pos[:,1]*1/(l+insq)]).T
                if len(neighbor[i]) == 0: # get cell's neighbors if not already
                    neighbor[i] = _get_neighbor_idx(2, notSquare, irow[i], icol[i], l, add_idx)
                # get neighboring samples' position
                neighbor_cells = cells[neighbor[i]]
                occupied = neighbor_cells[neighbor_cells >= 0]
                neighbor_pos = sample_pos[occupied,:]
                # get distances from neighboring samples
                
                for j in range(pos.shape[0]):
                    r = np.linalg.norm(neighbor_pos - pos[j,:], axis = 1)
                    if (r > cell_size).all(): # accept if no neighboring sample within minimum radius 
                        isample += 1
                        cells[i] = isample
                        sample_pos[isample, :] = pos[j, :]
                        sample_list.append(i)
                        active_list.append(isample)
                        #print('seed', isample, i, pos[j,:], 'in', i, cx[i], cy[i])
                        n_new_seed += 1
                        break;

                print(f':{len(avail_cell)}:{len(sample_list)}/{n}(-{(n-len(sample_list))/n*100:.1f}%)', end='\r')
        else:
            print(f'{len(active_list)}/{len(sample_list)}/{n}({len(sample_list)/n*100:.1f}%)', end='\r')
        if isample == n-1: # reached required number of sample, exit
            break;

    print(f'{n_new_seed + 1} seed(s) {len(sample_list)}/{n}({len(sample_list)/n*100:.1f}%)')
    sample_pos = sample_pos[:isample+1, :] + (rGen.random((isample+1, 2))-0.5) * cell_size/2
    if plot:
        fig = plt.figure(dpi = max(n//32, 90))
        ax = fig.add_subplot(111)
        ax.plot(sample_pos[:,0]*len_of_edge, sample_pos[:,1]*len_of_edge, ',r')
    
        idx = np.arange(isample)
        j = -1
        k = -1
        rmin = np.sqrt(2)
        rmax = 0
        for i in range(isample+1):
            pick = np.hstack((idx[:i], idx[i+1:]))
            r = np.linalg.norm(sample_pos[i,:] - sample_pos[pick,:], axis = 1)
            if np.max(r) > rmax:
                rmax = np.max(r)
            _k = np.argmin(r)
            if r[_k] < rmin:
                j = i
                k = pick[_k]
                rmin = r[_k]
            
        #print(j, k, rmin, rmax, cell_size)
        #print(sample_pos[j,:])
        #print(sample_pos[k,:])
        #print(sample_list[j], sample_list[k])
        #print(cells[sample_list[j]], cells[sample_list[k]])
        #print(neighbor[sample_list[j]])
        neighbors = cells[neighbor[sample_list[j]]]
        #print(neighbors)
        neighbors = neighbors[np.logical_and(neighbors < isample+1, neighbors >= 0)]
        ax.plot(sample_pos[[j,k],0], sample_pos[[j,k],1], '-b', lw = 0.2)
        ax.plot(sample_pos[j,0], sample_pos[j,1], 'ok', ms = 1)
        ax.plot(sample_pos[neighbors,0] * len_of_edge, sample_pos[neighbors,1] * len_of_edge, 'ob', ms = 1)
        neighbor_grid = np.array(neighbor[sample_list[j]])
        pick = neighbor_grid[cells[neighbor_grid] >= 0]
        ax.plot(cx[pick] * len_of_edge, cy[pick] * len_of_edge, 'xg', ms = 1)
        pick = neighbor_grid[cells[neighbor_grid] < 0]
        ax.plot(cx[pick] * len_of_edge, cy[pick] * len_of_edge, 'xk', ms = 1)
    
    return sample_pos * len_of_edge
        
