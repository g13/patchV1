from scipy import integrate
from scipy import special
import numpy as np
from cmath import *
import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex
from patch_geo_func import x_ep, y_ep, e_x
from sys import stdout
import warnings
import sobol_seq as ss
np.seterr(invalid = 'raise', under = 'ignore', over = 'ignore')
import matplotlib
matplotlib.use('Agg')

from assign_attr import *
from repel_system import *

bs = np.array([2])

fixMid = False
fixTip = True 
fixBack = True
theme0 = 'fTB'
nyFlank = 43
nyBack = 27

bp_ratio = 8.0
n_isample = 10
n_bsample = 5
assert(np.mod(nyBack,2) == 1)
assert(np.mod(nyFlank,2) == 1)
print('generate training points w/ or w/o temporal median and boundary fixed')

def nonFixedTrainingPoints(n, ecc, seed = 5346578):
    point = np.zeros((2,n))
    skip = 0
    i = 0
    rlim = [0, ecc]
    tlim = [-pi/2, pi/2]
    np.random.seed(seed)
    while i < n-1:
        ntmp = n-i;
        rands = ss.i4_sobol_generate(2, ntmp, skip=skip).T
        # add noise to avoid complete overlap
        pick = rands[1,:] < 0.05
        npick = np.sum(pick)
        rands[:,pick] = rands[:,pick] * (1 + np.abs(np.random.normal(0, 0.5, (2,npick))))
        rands[:,np.logical_not(pick)] = rands[:,np.logical_not(pick)] * (1 + np.random.normal(0, 1.0/np.sqrt(n), (2,ntmp-npick)))
        pt = tlim[0] + (tlim[1]-tlim[0]) * rands[0,:]
        pr = rlim[0] + (rlim[1]-rlim[0]) * rands[1,:]
        pick = np.logical_and(np.logical_and(pt < tlim[1], pt > tlim[0]), np.logical_and(pr < rlim[1], pr > rlim[0]))
        nselect = sum(pick)
        point[0, i:i+nselect] = pt[pick]
        point[1, i:i+nselect] = pr[pick]
        i += nselect
        skip += nselect

    return point

output = 'Tmat_cortex'
a = 0.635
b = 96.7
k = np.sqrt(140)*0.873145
ecc = 2.0 # must consistent with the corresponding variables in parameter.m and macro.ipynb

grid = np.array([64,104])*2
nx = grid[0]
ny = grid[1]
W = x_ep(ecc,0,k,a,b)
x = np.linspace(-W/(2*nx-4), W+W/(2*nx-4), nx)
W = W+W/(nx-2)
d = W/(nx-1)
H = d*ny
y = np.linspace(-H/2, H/2, ny)

nTx = 20
nTy = 40

if fixMid: # lend the mid points to tip and tail
    nMid = nTx
    if fixTip:
        nMid = nMid - 1
    if fixBack:
        nMid = nMid - 1
else:
    nMid = 0

trainSize = nTx*nTy
pos = np.zeros((1,3,trainSize))
n = nTx * nTy - nMid - nyFlank - nyBack

for b_scale in bs:
    print(b_scale)
    theme = theme0 + '-' + str(b_scale)
    output_file = output + '-'+ theme + '.bin'
    assert(np.mod(nTy,2) == 0)

    pol_point = nonFixedTrainingPoints(n, ecc)
    for i in range(n):
        pos[0,0,i] = x_ep(pol_point[1,i],pol_point[0,i],k,a,b)
        pos[0,1,i] = y_ep(pol_point[1,i],pol_point[0,i],k,a,b)

    xmax = x_ep(ecc,0,k,a,b)
    if fixMid:
        x_b = xmax/((nTx-1)/(b_scale/2.0)+2.0)
        xecc = np.linspace(x_b, xmax-x_b, nTx)
        if fixTip and fixBack:
            pos[0,0,n:n+nMid] = xecc[1:-1]
        else:
            if fixTip:
                pos[0,0,n:n+nMid] = xecc[1:]
            else:
                if fixBack:
                    pos[0,0,n:n+nMid] = xecc[:-1]
                else:
                    pos[0,0,n:n+nMid] = xecc

        pos[0,1,n:n+nMid] = 0

    xecc_b = np.linspace(0,xmax,(nyFlank+1)//2)
    ecc_b = e_x(xecc_b,k,a,b)
    bound = np.zeros((2,nyFlank+nyBack))
    #top
    hyFlank = (nyFlank-1)//2
    bound[0,:hyFlank] = [x_ep(e, pi/2, k, a, b) for e in ecc_b[1:]]
    bound[1,:hyFlank] = [y_ep(e, pi/2, k, a, b) for e in ecc_b[1:]]
    #bot
    bound[0,hyFlank:nyFlank-1] = [x_ep(e, -pi/2, k, a, b) for e in ecc_b[1:]]
    bound[1,hyFlank:nyFlank-1] = [y_ep(e, -pi/2, k, a, b) for e in ecc_b[1:]]
    #tip
    bound[0,nyFlank-1] = x_ep(ecc_b[0], 0.0, k, a, b)
    bound[1,nyFlank-1] = y_ep(ecc_b[0], 0.0, k, a, b)
    #back
    polar = np.linspace(-np.pi/2, np.pi/2, nyBack)
    bound[0,nyFlank:] = [x_ep(ecc, p, k, a, b) for p in polar]
    bound[1,nyFlank:] = [y_ep(ecc, p, k, a, b) for p in polar]

    pos[0,:2,n+nMid:] = bound

    print(f'{n} unfixed / {trainSize} total training points');
    for i in range(trainSize):
        test = (pos[0,:2,i] - pos[0,:2,i+1:].T == 0).all(-1)
        nd = np.sum(test.any())
        if nd > 0:
            did = np.nonzero(test)[0] + i+1
            print(f'#{i} is duplicated by #{did} at {pos[0,:2,i]}')
            if i >= n:
                print('fixed points have duplicates, go figure')
                assert(i < n)
            else:
                j = np.random.randint(n)
                while j in did:
                    j = np.random.randint(n)
                weight = np.random.uniform(0,1)
                pos[0,:2,i] = pos[0,:2,i]*weight + pos[0,:2,j]*(1-weight) # make sure cortex shape is not concave

    fixed = np.arange(n, trainSize)

    assert(np.isnan(pos).all() == False)
    with open(output_file, 'wb') as f:
        pos.tofile(f)

    fig = plt.figure('initial_points', dpi = 600)
    ax = fig.add_subplot(111)
    ax.plot(pos[0,0,:n], pos[0,1,:n],'*k', ms = 0.2)
    ax.plot(pos[0,0,n:], pos[0,1,n:],'or', ms = 0.2)
    ax.set_aspect('equal')
    fig.savefig(output+'-'+theme+'-initialp.png', dpi = 1000)

    LR_Pi_file = 'Ny-2-LR_Pi.bin'
    #nx = 128
    #ny = 208
    #with open('Ny-2-LR_Pi_new.bin','r') as f:
    #    Pi = np.reshape(np.fromfile(f, 'i4', count = nx*ny),(ny,nx))
    #    LR = np.reshape(np.fromfile(f, 'f8', count = nx*ny),(ny,nx))
    #LR = np.empty((ny,nx), dtype = 'i4')

    pos_file = output_file
    OR_file = None
    vpos_file = 'Tmat_VF.bin'
    uniform_pos_file = 'uniform_' + pos_file

    p0 = -np.pi/2
    p1 = np.pi/2

    #mMap = macroMap(nx, ny, x, y, 1, trainSize, LR_Pi_file, 'uniform_' + pos_file, OR_file, a, b, k, ecc, p0, p1, posUniform = True, OD_file = OD_file, VF_file = vpos_file)
    #mMap = macroMap(nx, ny, x, y, 1, trainSize, LR_Pi_file, 'uniform_' + pos_file, OR_file, a, b, k, ecc, p0, p1, posUniform = True)
    mMap = macroMap(nx, ny, x, y, 1, trainSize, LR_Pi_file, pos_file, a, b, k, ecc, p0, p1, posUniform = False)
    #mMap = macroMap(nx, ny, x, y, 1, trainSize, LR_Pi_file, pos_file, OR_file, a, b, k, ecc, p0, p1, posUniform = False, OD_file = OD_file)
    if mMap.pODready:
        assert(np.sum(mMap.ODlabel>0) + np.sum(mMap.ODlabel<0) == mMap.trainSize)

    fig = plt.figure('macroMap',dpi=1000)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(222, projection='polar')
    ax3 = fig.add_subplot(224, projection='polar')
    #mMap.pODready = False
    mMap.plot_map(ax1, ax2, ax3, fig.dpi, pltOD = True, pltVF = False, pltOP = False)

    ax1.set_aspect('equal')
    ax2.set_thetamin(p0/np.pi*180)
    ax2.set_thetamax(p1/np.pi*180)
    ax2.set_rmax(2.0)
    ax2.set_rmin(0.0)
    ax2.grid(False)
    ax2.tick_params(labelleft=False, labelright=True,
                   labeltop=False, labelbottom=True)

    ax3.set_thetamin(p0/np.pi*180)
    ax3.set_thetamax(p1/np.pi*180)
    ax3.set_rmax(2.0)
    ax3.set_rmin(0.0)
    ax3.grid(False)
    ax3.tick_params(labelleft=False, labelright=True,
                   labeltop=False, labelbottom=True)
    mMap.save(OD_file = 'T_od-'+theme+'.bin')

    # spread uniformly
    fig = plt.figure('trace', dpi = 600)
    dx = mMap.x[1] - mMap.x[0]
    dy = mMap.y[1] - mMap.y[0]
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
    ax1.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
    ax1.set_aspect('equal')
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
    ax2.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
    ax2.set_aspect('equal')
    dt0 = np.power(2.0,-np.arange(5,6)).reshape(1,1)
    dt1 = np.power(2.0,-np.arange(6,7)).reshape(1,1)
    dt = np.hstack((np.tile(dt0,(300,1)).flatten(), np.tile(dt1,(1,1)).flatten()))
    #dt = np.tile(dt0,(15,1))
    b_cl, i_cl = mMap.make_pos_uniformT(dt, seed = 17482321, ax1 = ax1, ax2 = ax2, b_scale = b_scale, fixed = fixed, ratio = bp_ratio)
    fig.savefig(output+'-'+theme+'.png', dpi = 1000)

    # for sampling radius of influence
    theta = np.linspace(0,2*np.pi,64)
    icx = i_cl*np.cos(theta)
    icy = i_cl*np.sin(theta)
    icx = np.append(icx, icx[0])
    icy = np.append(icy, icy[0])
    bcx = b_cl*np.cos(theta)
    bcy = b_cl*np.sin(theta)
    bcx = np.append(bcx, bcx[0])
    bcy = np.append(bcy, bcy[0])

    fig = plt.figure('final_pos', dpi = 600)
    ax = fig.add_subplot(111)
    ax.plot(mMap.pos[0,:n], mMap.pos[1,:n], '*k', ms = 0.2)
    ax.plot(mMap.pos[0,n:], mMap.pos[1,n:], '*r', ms = 0.2)
    ax.set_xlim(mMap.x[0]-dx/2, mMap.x[-1]+dx/2)
    ax.set_ylim(mMap.y[0]-dy/2, mMap.y[-1]+dy/2)
    isample = np.random.randint(n, size = n_isample)
    for i in range(n_isample):
        ax.plot(mMap.pos[0,isample[i]]+icx, mMap.pos[1,isample[i]]+icy, ':g', lw = 0.1)
    bsample = np.random.randint(n, trainSize, size = n_bsample)
    for i in range(n_bsample):
        ax.plot(mMap.pos[0,bsample[i]]+bcx, mMap.pos[1,bsample[i]]+bcy, ':m', lw = 0.1)
    ax.set_aspect('equal')
    fig.savefig(output+'-'+theme+'_finalp.png', dpi = 600)

    Training_pos = 'Training_pos-'+theme+'.bin'
    mMap.vpos = mMap.assign_pos_VF(True)
    ymax = y_ep(ecc,pi/2,k,a,b)
    with open(Training_pos, 'wb') as f:
        np.array([mMap.networkSize]).astype('i4').tofile(f)
        mMap.pos.tofile(f)
        np.array([0,xmax]).tofile(f)
        np.array([-ymax,ymax]).tofile(f)
        mMap.vpos.tofile(f)
