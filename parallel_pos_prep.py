import numpy as np
from cmath import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patch_geo_func import x_ep, y_ep
from sys import stdout
import warnings
from assign_attr import *
np.seterr(invalid = 'raise', under = 'ignore', over = 'ignore')

LR_Pi_file = 'ibeta-4-LR_Pi21.bin'
pos_file = 'V1_pos_2D_lowDensity.bin'
uniform_pos_file = 'uniform_pos.bin'
#pos_file = 'uniform_pos1.bin'
#uniform_pos_file = 'uniform_pos2.bin'

mMap = macroMap(LR_Pi_file, pos_file, posUniform = False)
mMap.assign_pos_OD1()
#ndt0 = 20
#ndt = 1500
#tpower = 6
#dt = np.zeros(ndt) + np.power(2.0,-tpower)
#dt = np.hstack((np.zeros(ndt0) + dt[0]/(ndt0+1), dt)) 
#mMap.make_pos_uniform_parallel(dt, 1.5, 2.0, 'lowDensity_b', ncpu = 16, ndt0 = ndt0)
mMap.save(Parallel_spreadVF_file = 'p_spreadVF.bin', Parallel_uniform_file = 'p_uniform.bin')
