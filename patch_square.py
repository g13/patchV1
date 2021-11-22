import numpy as np
from scipy.stats import qmc

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
