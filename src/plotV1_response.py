import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import sys
if len(sys.argv) == 1:
    suffix = ""
else:
    suffix = sys.argv[1]
print(suffix)
if suffix:
    suffix = "_" + suffix 

np.random.seed(7329445)
nt_ = 2400
nstep = 2400
step0 = 0

pSpike = True
pVoltage = True
pCond = True
pH = True
readSpike = False

lw = 0.1
sample = np.array([0,1,2,768])
ns = 4

# const
rawDataFn = "rawData" + suffix + ".bin"
spDataFn = "spData" + suffix
with open(rawDataFn, 'rb') as f:
    dt = np.fromfile(f, 'f4', 1)[0] 
    nt = np.fromfile(f, 'u4', 1)[0] 
    if step0 + nt_ >= nt:
        nt_ = nt - step0
    tstep = (nt_ + nstep - 1)//nstep
    nstep = nt_//tstep
    interval = tstep - 1
    t = (step0 + np.arange(nstep)*tstep)*dt
    print(f'plot {nstep} points from the {nt_} steps startingfrom step {step0}')
    nV1 = np.fromfile(f, 'u4', 1)[0] 
    haveH = np.fromfile(f, 'u4', 1)[0] 
    ngFF = np.fromfile(f, 'u4', 1)[0] 
    ngE = np.fromfile(f, 'u4', 1)[0] 
    ngI = np.fromfile(f, 'u4', 1)[0] 

print(f'nV1 = {nV1}; haveH = {haveH}')
print(f'ngFF: {ngFF}; ngE: {ngE}; ngI: {ngI}')
if 'sample' not in locals():
    sample = np.random.randint(nV1, size = ns)

# spikes
if pSpike:
    if readSpike:
        spScatter = np.load(spDataFn + '.npy', allow_pickle=True)
    else:
        with open(rawDataFn, 'rb') as f:
            f.seek(4*7, 1)
            spScatter = np.empty(nV1, dtype = object)
            for i in range(nV1):
                spScatter[i] = []
        
            f.seek((2+(ngE + ngI + ngFF)*(1+haveH))*nV1*4*step0, 1)
            for it in range(nt-step0):
                data = np.fromfile(f, 'f4', nV1)
                tsps = data[data > 0]
                if tsps.size > 0:
                    idxFired = np.nonzero(data)[0]
                    k = 0
                    for j in idxFired:
                        nsp = np.int(np.floor(tsps[k]))
                        tsp = tsps[k] - nsp
                        if nsp > 1:
                            if 1-tsp > 0.5:
                                dtsp = tsp/nsp
                            else:
                                dtsp = (1-tsp)/nsp
                            tstart = tsp - (nsp//2)*dtsp
                            for isp in range(nsp):
                                spScatter[j].append((it + tstart+isp*dtsp)*dt)
                        else:
                            spScatter[j].append((it+tsp)*dt)
                        k = k + 1
                f.seek((1+(ngE + ngI + ngFF)*(1+haveH))*nV1*4, 1)
        np.save(spDataFn, spScatter)

# voltage and conductances
if pVoltage or pCond:
    with open(rawDataFn, 'rb') as f:
        f.seek(4*7, 1)
        if pVoltage:
            v = np.empty((nV1, nstep), dtype = 'f4')
        if pCond:
            if pH:
                getH = haveH
            else:
                getH = 0
            gE = np.empty((1+getH, ngE, nV1, nstep), dtype = 'f4')
            gI = np.empty((1+getH, ngI, nV1, nstep), dtype = 'f4')
            gFF = np.empty((1+getH, ngFF, nV1, nstep), dtype = 'f4')

        f.seek((2+(ngE + ngI + ngFF)*(1+haveH))*nV1*4*step0, 1)
        for i in range(nstep):
            if pVoltage:
                f.seek(nV1*4, 1)
                v[:,i] = np.fromfile(f, 'f4', ngFF*nV1).reshape(ngFF,nV1)
            else:
                f.seek(2*nV1*4, 1)

            if pCond:
                gFF[0,:,:,i] = np.fromfile(f, 'f4', ngFF*nV1).reshape(ngFF,nV1)
                if haveH:
                    if pH:
                        gFF[1,:,:,i] = np.fromfile(f, 'f4', ngFF*nV1).reshape(ngFF,nV1)
                    else:
                        f.seek(ngFF*nV1*4, 1)

                gE[0,:,:,i] = np.fromfile(f, 'f4', ngE*nV1).reshape(ngE,nV1)
                gI[0,:,:,i] = np.fromfile(f, 'f4', ngI*nV1).reshape(ngI,nV1)
                if haveH :
                    if pH:
                        gE[1,:,:,i] = np.fromfile(f, 'f4', ngE*nV1).reshape(ngE,nV1)
                        gI[1,:,:,i] = np.fromfile(f, 'f4', ngI*nV1).reshape(ngI,nV1)
                    else:
                        f.seek((ngE+ngI)*nV1*4, 1)
            else:
                f.seek((ngE + ngI + ngFF)*(1+haveH)*nV1*4, 1)
    
            f.seek((2+(ngE + ngI + ngFF)*(1+haveH))*nV1*4*interval, 1)


fig = plt.figure(f'sample', dpi = 1000)
grid = gs.GridSpec(ns, 1, figure = fig, hspace = 0.2)
for i in range(ns):
    iV1 = sample[i]
    ax = fig.add_subplot(grid[i])
    if pSpike:
        tsp0 = np.array(spScatter[iV1])
        tsp = tsp0[tsp0<t[-1]]
        ax.plot(tsp, np.ones(len(tsp)), '*k', ms = 1.0)
    if pVoltage:
        ax.plot(t, v[iV1,:], '-k', lw = lw)
    if pCond:
        ax2 = ax.twinx()
        for ig in range(ngFF):
            ax2.plot(t, gFF[0,ig,iV1,:], '-g', lw = (ig+1)/ngFF * lw)
        for ig in range(ngE):
            ax2.plot(t, gE[0,ig,iV1,:], '-r', lw = (ig+1)/ngE * lw)
        for ig in range(ngI):
            ax2.plot(t, gI[0,ig,iV1,:], '-b', lw = (ig+1)/ngI * lw)
        if pH:
            for ig in range(ngFF):
                ax2.plot(t, gFF[1,ig,iV1,:], ':g', lw = (ig+1)/ngFF * lw)
            for ig in range(ngE):
                ax2.plot(t, gE[1,ig,iV1,:], ':r', lw = (ig+1)/ngE * lw)
            for ig in range(ngI):
                ax2.plot(t, gI[1,ig,iV1,:], ':b', lw = (ig+1)/ngI * lw)

fig.savefig('V1-sample' + suffix + '.png')
