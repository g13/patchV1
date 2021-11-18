import numpy as np
from scipy.stats import qmc

def square_pos(per_dis, w, h, center):
    sampler = qmc.Sobol(d=2)
    rands = sampler.random(w*h)
    pos = np.zeros((w*h,2))
    pos[:,0] = (w*rands[:,0]-w/2)*per_dis[0] + center[0]
    pos[:,1] = (h*rands[:,1]-h/2)*per_dis[1] + center[1]
    return pos
