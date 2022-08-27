import numpy as np
from scipy.stats import qmc
import warnings
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
