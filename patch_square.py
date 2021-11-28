import numpy as np
from scipy.stats import qmc
import warnings
from sys import stdout
#warnings.filterwarnings("once", message = "The balance properties of Sobol' points require")
warnings.filterwarnings("ignore", message = "The balance properties of Sobol\' points require", category=UserWarning)
def square_pos(per_dis, n, center):
    sampler = qmc.Sobol(d=2)
    rands = sampler.random(n)
    pos = np.zeros((2,n))
    pos[0,:] = (rands[:,0]-1/2)*per_dis[0] + center[0]
    pos[1,:] = (rands[:,1]-1/2)*per_dis[1] + center[1]
    return pos

def square_OP(center, r, phase, pos, clockwise):
    dis = pos.T - center
    op = np.arctan2(dis[:,1], dis[:,0])/2 
    if clockwise:
        op = (np.pi - (op + np.pi/2)) - np.pi/2
        op = op - phase
    else:
        op = op + phase
    dpick1 = op > np.pi/2
    dpick2 = op < -np.pi/2
    op[dpick1] = op[dpick1] - np.pi
    op[dpick2] = op[dpick2] + np.pi
    pick = (np.abs(dis) < r).all(1)
    return op, pick


def assign_square_xyID(n, pos, ratio = 2):
    xyID = np.zeros((2,n), dtype = 'i4')
    w = int(np.sqrt(n))
    xy = np.zeros((2,w*w), dtype = 'i4')
    xy[0,:] = np.tile(np.arange(w), w)
    xy[1,:] = np.repeat(np.arange(w), w)
    nonreplace_pick = np.ones(w*w, dtype = bool)
    pos_id = np.arange(w*w)
    pos = pos*w
    xy = xy.T
    max_dis = 1
    #max_dis = 2*w*w
    for i in range(n):
        dis = np.max(np.abs(pos[:,i] - xy), axis = 1)
        # local algo
        local_pick = dis <= max_dis
        #count = 0
        # slow down x1000!!!
        #while sum(local_pick) == 0: 
        #    max_dis *= ratio
        #    local_pick = np.logical_and(nonreplace_pick,dis <= max_dis)
        #    count += 1
        local_dis = dis[local_pick]
        local_id = pos_id[local_pick]
        imin = np.argmin(local_dis)
        xyID[:,i] = xy[local_id[imin],:]
        nonreplace_pick[local_id[imin]] = False 
        #if count > 0:
        #    print(count, sum(local_pick))

        # global algo
        #dis[np.logical_not(nonreplace_pick)] = max_dis
        #imin = np.argmin(dis)
        #xyID[:,i] = xy[imin,:]
        #nonreplace_pick[imin] = False
        stdout.write(f'\r {i/n*100:.1f}%')
    stdout.write('\n')
    return xyID, w
