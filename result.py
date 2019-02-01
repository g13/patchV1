### 
# Change the condition in list comprehension if non-spike value of tsp changes in algorithm in get_raster_and_fr

import numpy as np
import matplotlib.pyplot as plt

class Result:
    paramPrefix = 'p_ushy'
    vPrefix = 'v_ictorious'
    gEprefix = 'gE_nerous'
    gIprefix = 'gI_berish'
    spikeTrainPrefix = 's_uspicious'
    nSpikePrefix = 'n_arcotic'
    dataFileSuffix = '.bin'
    def __init__(self, theme, directory, pP = paramPrefix, vP = vPrefix, gEp = gEprefix, gIp = gIprefix, sTP = spikeTrainPrefix, nSP = nSpikePrefix, dFS = dataFileSuffix):
        self.theme = theme
        self.directory = directory
        self.paramPrefix = pP
        self.vPrefix = vP
        self.gEprefix = gEp
        self.gIprefix = gIp
        self.spikeTrainPrefix = sTP
        self.nSpikePrefix = nSP
        self.dataFormat = dFS
        self.gRead = False
        self.vRead = False
        self.spRead = False
        self.nspRead = False
        self.read_paramater()
        
    def read_paramater(self):
        fn = self.directory+'/'+self.paramPrefix+'-'+self.theme+self.dataFileSuffix
        with open(fn, 'rb') as f:
            size = np.fromfile(f, 'u4', count = 2)
            self.nE = size[0]
            self.nI = size[1]
            self.n = self.nE + self.nI
            gSize = np.fromfile(f, 'u4', count = 2)
            self.ngTypeE = gSize[0]
            self.ngTypeI = gSize[1]
            vData = np.fromfile(f, "f8", count = 4)
            self.vL = vData[0]
            self.vT = vData[1]
            self.vE = vData[2]
            self.vI = vData[3]
            gLdata = np.fromfile(f, "f8", count = 2)
            self.gL_E = gLdata[0]
            self.gL_I = gLdata[1]
            tRefData = np.fromfile(f, "f8", count = 2)
            self.tRef_E = tRefData[0]
            self.tRef_I = tRefData[1]
            self.nstep = np.fromfile(f, "u4", count = 1)[0]
            self.dt = np.fromfile(f, "f8", count = 1)[0]
            self.inputRate = np.fromfile(f, "f8", count = 1)[0]
            print(self.nstep, self.dt, self.inputRate)
            nTimer = np.fromfile(f, 'i4', count = 1)[0]
            self.timer = np.fromfile(f, 'f8', count = nTimer)

    def read_spikeTrain(self):
        EPS = np.finfo(float).eps
        fn = self.directory+'/'+self.spikeTrainPrefix+'-'+self.theme+self.dataFileSuffix
        data = np.fromfile(fn, dtype='d')
        spikeTrain = np.reshape(data, (self.nstep, self.n))
        self.raster = np.empty((self.n,), dtype=object)
        for i in range(self.n):
            index = np.nonzero(self.dt - spikeTrain[:,i] > EPS )[0]
            nSpike = len(index)
            self.raster[i] = np.empty(nSpike)
            for j in range(nSpike):
                self.raster[i][j] = spikeTrain[index[j],i] + index[j]*self.dt
        self.read_nSpikePerDt()
        self.spRead = True
        
    def read_nSpikePerDt(self):
        fn = self.directory+'/'+self.nSpikePrefix+'-'+self.theme+self.dataFileSuffix
        data = np.fromfile(fn, dtype='u4')
        self.nSpikePerDt = np.reshape(data, (self.nstep, self.n))
        self.nspRead = True
        
    def read_voltage_trace(self):
        fn = self.directory+'/'+self.vPrefix+'-'+self.theme+self.dataFileSuffix
        data = np.fromfile(fn, dtype='d')
        self.v =  np.reshape(data, (self.nstep+1, self.n))
        self.vRead = True
            
    def read_conductance_trace(self):
        fn = self.directory+'/'+self.gEprefix+'-'+self.theme+self.dataFileSuffix
        data = np.fromfile(fn, dtype='d')
        self.gE =  np.reshape(data, (self.nstep+1, self.ngTypeE, self.n))
        fn = self.directory+'/'+self.gIprefix+'-'+self.theme+self.dataFileSuffix
        data = np.fromfile(fn, dtype='d')
        self.gI =  np.reshape(data, (self.nstep+1, self.ngTypeI, self.n))
        self.gRead = True

    def get_raster(self, update = False):
        if not self.spRead or update:
            self.read_spikeTrain()
        return self.raster, self.nSpikePerDt
    
    def get_nSpikePerDt(self, update = False):
        if not self.nspRead or update:
            self.read_nSpikePerDt()
        return self.nSpikePerDt
    
    def get_voltage_trace(self, update = False):
        if not self.vRead or update:
            self.read_voltage_trace()
        return self.v

    def get_conductance_trace(self, update = False):
        if not self.gRead or update:
            self.read_conductance_trace()
        return self.gE, self.gI

    def get_fr(self, t, weights = None, update = False):
        # Centered moving average without weight
        # fr is not stored in class
        if not self.spRead or update:
            self.read_spikeTrain()
        ndt = int(t/dt)
        nbins = self.nstep//ndt
        if ndt%2 == 0:
            l = ndt//2-1
            r = ndt//2
        else:
            l = (ndt+1)//2
            r = (ndt+1)//2
        nfr = self.nstep-l-r

        fr = np.empty((self.n, nfr), dtype='d')
        if weights == None:
            for i in range(self.n):
                for idt in range(l, l+nfr):
                    fr[i,idt-l] = sum(self.nSpikePerDt[i, idt-l:idt+r])
        else:
            assert(weights.size == ndt)
            for i in range(self.n):
                for idt in range(l, l+nfr):
                    fr[i,idt-l] = sum(self.nSpikePerDt[i, idt-l:idt+r]*weights)
        return fr
    
    def get_accVolt(self, update = False):
        # accVolt is not stored in class
        if not self.vRead or update:
            self.read_voltage_trace()
        if not self.spRead or update:
            self.read_spikeTrain()

        accVolt = np.empty((self.n, self.nstep+1), dtype='d')
        for i in range(self.n):
            it = 0
            ns = 0
            for ir in range(self.raster[i].size):
                et = np.int(self.raster[i][ir]/self.dt) + 1
                accVolt[i, it:et] = ns*(self.vT-self.vL) + self.v[it:et, i]
                it = et
                if self.raster[i][ir] < 0:
                    print(i, ir, self.raster[i][ir], self.theme)
                    print(self.v[:, i])
                    assert(self.raster[i][ir] >= 0)
                ns = ns + self.nSpikePerDt[et-1,i]
            accVolt[i, it:self.nstep+1] = ns*(self.vT-self.vL) + self.v[it:self.nstep+1, i]
        return accVolt

class cResult(Result):
    paramPrefix = 'p_CPU'
    vPrefix = 'v_CPU'
    gEprefix = 'gE_CPU'
    gIprefix = 'gI_CPU'
    spikeTrainPrefix = 's_CPU'
    nSpikePrefix = 'n_CPU'
    dataFileSuffix = '.bin'
    def __init__(self, theme, directory, pP = paramPrefix, vP = vPrefix, gEp = gEprefix, gIp = gIprefix, sTP = spikeTrainPrefix, nSP = nSpikePrefix, dFS = dataFileSuffix):
        super().__init__(theme, directory, pP, vP, gEp, gIp, sTP, nSP, dFS)
        
        
class rResult(Result):
    paramPrefix = 'rp_CPU'
    vPrefix = 'rv_CPU'
    gEprefix = 'rgE_CPU'
    gIprefix = 'rgI_CPU'
    spikeTrainPrefix = 'rs_CPU'
    nSpikePrefix = 'rn_CPU'
    dataFileSuffix = '.bin'
    def __init__(self, theme, directory, pP = paramPrefix, vP = vPrefix, gEp = gEprefix, gIp = gIprefix, sTP = spikeTrainPrefix, nSP = nSpikePrefix, dFS = dataFileSuffix):
        super().__init__(theme, directory, pP, vP, gEp, gIp, sTP, nSP, dFS)
        
class sResult(Result):
    paramPrefix = 'p_ushy'
    vPrefix = 'v_ictorious'
    gEprefix = 'gE_nerous'
    gIprefix = 'gI_berish'
    spikeTrainPrefix = 's_uspicious'
    nSpikePrefix = 'n_arcotic'
    dataFileSuffix = '_ssc.bin'
    def __init__(self, theme, directory, pP = paramPrefix, vP = vPrefix, gEp = gEprefix, gIp = gIprefix, sTP = spikeTrainPrefix, nSP = nSpikePrefix, dFS = dataFileSuffix):
        super().__init__(theme, directory, pP, vP, gEp, gIp, sTP, nSP, dFS)
        
        
def raster_plot(r, r0, x1, x2):
    raster_plot.fignum += 1
    fig = plt.figure(raster_plot.fignum, dpi = 600)
    ax = fig.add_subplot(212)
    def plot(_r, offset, marker, ms):
        for j in range(3):
            for i in range(r0.n):
                ax.plot(_r.raster[i], np.zeros(_r.raster[i].size)+i+offset, marker, ms=ms)
    plot(r0, 0.0, 'sr', 0.1)
    #plot(r[0], 0.2, '>g', 0.1)
    #plot(r[1], 0.4, 'ob', 0.1)
    plot(r[0], 0.4, '>g', 0.1)
    plot(r[1], 0.4, 'ob', 0.1)
    #plot(r[11], 0.4, 'o', 0.1)
    ax.set_xlim(x1, x2)
    
def v_pair_plot(r0, r1, t, i, x1, x2, eps, geps, y1=-2/3, y2=1):
    v_pair_plot.fignum += 1
    fig = plt.figure(v_pair_plot.fignum, dpi = 600, figsize= (10,3))
    ax = fig.add_subplot(411)
    #ax = fig.add_subplot(211)
    r0.get_voltage_trace()
    r1.get_voltage_trace()
    ax.plot(t,r0.v[:,i],'-sr',lw=0.3,ms=0.1)
    ax.plot(t,r1.v[:,i],'-sg',lw=0.2,ms=0.1)
    ax.set_ylim(y1, y2)
    ax.set_xlim(x1, x2)
    ax.set_title(f'{i}')
    ax = fig.add_subplot(412)
    ax.plot(t,r0.v[:,i]-r1.v[:,i],'-sk',lw=0.3,ms=0.1)
    ax.set_xlim(x1, x2)
    ax.set_ylim(-eps, eps)
    ax = fig.add_subplot(413)
    r0.get_conductance_trace()
    r1.get_conductance_trace()
    ax.plot(t,r0.gE[:,0,i]-r1.gE[:,0,i],'-sr',lw=0.3,ms=0.1)
    ax.set_xlim(x1, x2)
    ax2 = ax.twinx()
    ax2.plot(t,r0.gI[:,0,i]-r1.gI[:,0,i],'-sb',lw=0.3,ms=0.1)
    #ax.set_ylim(-geps, geps)
    ax = fig.add_subplot(414)
    ax.plot(t,r0.gE[:,0,i],'-sr',lw=0.3,ms=0.1)
    ax.plot(t,r1.gE[:,0,i],':sr',lw=0.3,ms=0.1)
    ax.plot(t,r0.gI[:,0,i],'-sb',lw=0.3,ms=0.1)
    ax.plot(t,r1.gI[:,0,i],':sb',lw=0.3,ms=0.1)
    ax.set_xlim(x1, x2)
    
def v_g_plot(r, t, r0, i, x1, x2, y1=0, y2=1):
    v_g_plot.fignum += 1
    fig = plt.figure(v_g_plot.fignum, dpi = 600)
    ax = fig.add_subplot(111)
    t0 = r0.dt*np.arange(r0.nstep+1)
    ax.plot(t[1],r[1].v[:,i],'-sr',lw=0.3,ms=0.1)
    ax.plot(t[0],r[0].v[:,i],'-sg',lw=0.2,ms=0.1)
    ax.plot(t0,r0.v[:,i],'-sb',lw=0.1,ms=0.1)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    
def get_veff(r, i):
    if i < r.nE:
        gL = r.gL_E
    else:
        gL = r.gL_I
    gT = r.gE[:,0,i] + r.gI[:,0,i] + gL
    veff = (r.gE[:,0,i]*r.vE + r.gI[:,0,i]*r.vI + gL*r.vL)/gT
    return veff

def v_veff_plot(r, t, r0, i, x1, x2, y1=0.8, y2=1):
    fig = plt.figure(v_veff_plot.fignum, dpi = 600)
    ax = fig.add_subplot(211)
    ax.set_title(f'{i}')
    t0 = r0.dt*np.arange(r0.nstep+1)
        
    ax.plot(t[1],r[1].v[:,i],'-sr',lw=0.2,ms=0.1)
    ax.plot(t[1],get_veff(r[1], i), ':r', lw=0.2)
    
    ax.plot(t[0],r[0].v[:,i],'-og',lw=0.2,ms=0.1)
    ax.plot(t[0],get_veff(r[0], i), ':g', lw=0.2)
    
    ax.plot(t0,  r0.v[:,i],  '->b',lw=0.2,ms=0.1)
    ax.plot(t0,  get_veff(r0, i),   ':b', lw=0.2)
    ax.plot(t0, np.zeros(t0.size) + r0.vT, '-k', lw = 0.1)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    
    ax = fig.add_subplot(212)
    ax.plot(r[1].raster[i], np.zeros(r[1].raster[i].size)+2,'.r')
    ax.plot(r[0].raster[i], np.zeros(r[0].raster[i].size)+1,'.g')
    ax.plot(r0.raster[i], np.zeros(r0.raster[i].size),'.b')
    ax.set_xlim(x1, x2)
    ax.set_ylim(-0.1, 2.1)
    v_veff_plot.fignum += 1
    
def cross_pair_plot(r0, r1, t, i, x1, x2, eps, geps, y1=-2/3, y2=1):
    cross_pair_plot.fignum += 1
    fig = plt.figure(cross_pair_plot.fignum, dpi = 600, figsize= (10,3))
    ax = fig.add_subplot(411)
    jt = r0.nstep//r1.nstep
    tpick = np.arange(0, (r1.nstep+1)*jt, jt, dtype=int)
    r0.get_voltage_trace()
    r1.get_voltage_trace()
    ax.plot(t,r0.v[tpick,i],'-sr',lw=0.3,ms=0.1)
    ax.plot(t,r1.v[:,i],'-sg',lw=0.2,ms=0.1)
    ax.set_ylim(y1, y2)
    ax.set_xlim(x1, x2)
    ax.set_title(f'{i}')
    ax = fig.add_subplot(412)
    ax.plot(t,r0.v[tpick,i]-r1.v[:,i],'-sk',lw=0.3,ms=0.1)
    ax.set_xlim(x1, x2)
    ax.set_ylim(-eps, eps)
    ax = fig.add_subplot(413)
    r0.get_conductance_trace()
    r1.get_conductance_trace()
    ax.plot(t,r0.gE[tpick,0,i]-r1.gE[:,0,i],'-sr',lw=0.3,ms=0.1)
    ax.set_xlim(x1, x2)
    ax2 = ax.twinx()
    ax2.plot(t,r0.gI[tpick,0,i]-r1.gI[:,0,i],'-sb',lw=0.3,ms=0.1)
    #ax.set_ylim(-geps, geps)
    ax = fig.add_subplot(414)
    ax.plot(t,r0.gE[tpick,0,i],'-sr',lw=0.3,ms=0.1)
    ax.plot(t,r1.gE[:,0,i],':sr',lw=0.3,ms=0.1)
    ax.set_xlim(x1, x2)
    ax2 = ax.twinx()
    ax2.plot(t,r0.gI[tpick,0,i],'-sb',lw=0.3,ms=0.1)
    ax2.plot(t,r1.gI[:,0,i],':sb',lw=0.3,ms=0.1)
    
    