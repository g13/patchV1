from patch_geo_func import x_ep, y_ep, model_block_ep
from repel_system import *
import matplotlib.pyplot as plt
from sys import stdout
from scipy import integrate
import numpy as np
import time
class macroMap:
    def __init__(self, nx, ny, x, y, nblock, blockSize, LR_Pi_file, pos_file, a, b, k, ecc, p0, p1):
        self.nblock = nblock
        self.blockSize = blockSize
        self.networkSize = nblock * blockSize
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.k = k
        self.nx = nx
        self.ny = ny
        self.xx, self.yy = np.meshgrid(x,y)
        with open(pos_file,'r') as f:
            pos = np.reshape(np.fromfile(f,'f8',count = 3*self.networkSize),(nblock,3,blockSize))
        self.pos = np.zeros((2,self.networkSize))
        self.pos[0,:] = pos[:,0,:].reshape(self.networkSize)
        self.pos[1,:] = pos[:,1,:].reshape(self.networkSize)
        with open(LR_Pi_file,'r') as f:
            self.Pi = np.reshape(np.fromfile(f, 'i4', count = nx*ny),(ny,nx))
            self.LR = np.reshape(np.fromfile(f, 'f8', count = nx*ny),(ny,nx))
        self.LR[self.LR > 0] = 1
        self.LR[self.LR < 0] = -1
        self.LR[self.Pi <=0] = 0
        ratio = self.Pi.size/(np.sum(self.Pi>0))
        self.necc = np.round(nx * ratio).astype(int)
        self.npolar = np.round(ny * ratio).astype(int)
        self.ecc = ecc
        self.e_range = np.exp(np.linspace(np.log(1),np.log(ecc+1),self.necc))-1
        self.p_range = np.linspace(p0,p1,self.npolar)
        memoryRequirement = ((np.int64(self.necc-1)*np.int64(self.npolar-1)*8 + np.int64(self.networkSize*(nx + ny)))*8/1024/1024/1024)
        print(f'{self.necc}x{self.npolar}, ecc-polar grid houses {self.networkSize} neurons')
        print(f'require {memoryRequirement:.3f} GB')
        self.vx = np.empty((self.necc,self.npolar))
        self.vy = np.empty((self.necc,self.npolar))
        for ip in range(self.npolar):
            self.vx[:,ip] = [x_ep(e,self.p_range[ip],self.k,self.a,self.b) for e in self.e_range]
            self.vy[:,ip] = [y_ep(e,self.p_range[ip],self.k,self.a,self.b) for e in self.e_range]
        self.pODready = False
        self.pVFready = False
        self.OD_VF_reconciled = False 
            
    # memory requirement low, slower
    def assign_pos_OD0(self):
        ## determine which blocks to be consider in each subgrid to avoid too much iteration over the whole network
        # calculate reach of each block as its radius, some detached blocks may be included
        pos = self.pos.reshape(2,self.nblock,self.blockSize)
        blockPosX = np.reshape(np.mean(pos[0,:,:],1), (self.nblock,1))
        blockPosY = np.reshape(np.mean(pos[1,:,:],1), (self.nblock,1))
        max_dis2center = np.max(np.power(pos[0,:,:]-blockPosX,2) + np.power(pos[1,:,:]-blockPosY,2),1).reshape((self.nblock,1))
        # iterate over each grid
        corner = np.zeros((2,4))
        self.ODlabel = np.zeros((self.networkSize))
        for i in range(self.ny-1):
            for j in range(self.nx-1):
                nc = np.sum(self.Pi[i:i+2,j:j+2])
                ic = (self.Pi[i:i+2,j:j+2]>0).reshape(4)
                if nc > 0:
                    corner[:,0] = [self.x[j],   self.y[i]]
                    corner[:,1] = [self.x[j+1], self.y[i]]
                    corner[:,2] = [self.x[j],   self.y[i+1]]
                    corner[:,3] = [self.x[j+1], self.y[i+1]]
                    # see if any corner is closer to each block than its radius
                    dx = np.tile(corner[0,:],(self.nblock,1)) - blockPosX
                    dy = np.tile(corner[1,:],(self.nblock,1)) - blockPosY
                    d2 = np.power(dx,2) + np.power(dy,2)
                    blist = np.nonzero((d2 < max_dis2center).any(1))[0]
                    nb = blist.size
                    OD_of_grid = self.LR[i:i+2,j:j+2]
                    gridLR = np.sum(OD_of_grid[self.Pi[i:i+2,j:j+2]])
                    for ib in blist:
                        #pick neurons that are inside the subgrid
                        ipick = np.logical_and(pos[0,ib,:] > self.x[j], pos[0,ib,:] < self.x[j+1])
                        ipick = np.logical_and(ipick, np.logical_and(pos[1,ib,:] > self.y[i], pos[1,ib,:] < self.y[i+1]))
                        bpick = np.arange(ib*self.blockSize,(ib+1)*self.blockSize)[ipick]
                        npick = np.sum(ipick)
                        if npick > 0:
                            if gridLR == nc:
                                self.ODlabel[bpick] = 1
                            else:
                                if gridLR == -nc:
                                    self.ODlabel[bpick] = -1
                                else:
                                    dx = np.tile(pos[0,ib,ipick],(nc,1)) - corner[0,ic].reshape(nc,1)
                                    dy = np.tile(pos[1,ib,ipick],(nc,1)) - corner[1,ic].reshape(nc,1)
                                    d2 = np.power(dx,2) + np.power(dy,2)
                                    OD_flattened = OD_of_grid.reshape(4)[ic]
                                    OD_of_neuron = np.argmin(d2,0)
                                    self.ODlabel[bpick] = OD_flattened[OD_of_neuron]
                stdout.write(f'\r{(i*(self.nx-1)+j+1)/(self.nx-1)/(self.ny-1)*100:.3f}%, grid ({i+1},{j+1})/({self.ny-1},{self.nx-1})')
        stdout.write('\n')
        self.pODready = True
        return self.ODlabel

    # memory requirement high, faster 
    def assign_pos_OD1(self):
        x0 = self.x[:-1]
        x1 = self.x[1:]
        y0 = self.y[:-1]
        y1 = self.y[1:]

        self.ODlabel = np.zeros((self.networkSize))
        i = np.nonzero(np.logical_and(np.tile(self.pos[1,:],(self.ny-1,1))-y0.reshape(self.ny-1,1) > 0, np.tile(self.pos[1,:],(self.ny-1,1))-y1.reshape(self.ny-1,1) < 0).T)[1]
        j = np.nonzero(np.logical_and(np.tile(self.pos[0,:],(self.nx-1,1))-x0.reshape(self.nx-1,1) > 0, np.tile(self.pos[0,:],(self.nx-1,1))-x1.reshape(self.nx-1,1) < 0).T)[1]

        corner_x = np.zeros((self.ny-1,self.nx-1,4))
        corner_x[:,:,0] = self.xx[:-1,:-1]
        corner_x[:,:,1] = self.xx[1:,:-1]
        corner_x[:,:,2] = self.xx[:-1,1:]
        corner_x[:,:,3] = self.xx[1:,1:]
        corner_y = np.zeros((self.ny-1,self.nx-1,4))
        corner_y[:,:,0] = self.yy[:-1,:-1]
        corner_y[:,:,1] = self.yy[1:,:-1]
        corner_y[:,:,2] = self.yy[:-1,1:]
        corner_y[:,:,3] = self.yy[1:,1:]
        def get_d(ic):
            dx = self.pos[0,:] - corner_x[i,j,ic]
            dy = self.pos[1,:] - corner_y[i,j,ic]
            d2 = np.power(dx,2) + np.power(dy,2)
            return d2
        d2 = np.zeros((self.networkSize,4))
        for ic in range(4):
            d2[:,ic] = get_d(ic)
        d2[self.Pi[i,j]<=0,    0] = np.float('inf')
        d2[self.Pi[i+1,j]<=0,  1] = np.float('inf')
        d2[self.Pi[i,j+1]<=0,  2] = np.float('inf')
        d2[self.Pi[i+1,j+1]<=0,3] = np.float('inf')
        ipick = np.argmin(d2,1)
        #print(ipick.shape)
        self.ODlabel[ipick == 0] = self.LR[i,    j][ipick == 0]
        self.ODlabel[ipick == 1] = self.LR[i+1,  j][ipick == 1]
        self.ODlabel[ipick == 2] = self.LR[i,  j+1][ipick == 2]
        self.ODlabel[ipick == 3] = self.LR[i+1,j+1][ipick == 3]
        self.pODready = True
        return self.ODlabel
    # boundary define by 4 corners
    def define_bound(self, grid):
        ngrid = (self.nx-1) * (self.ny-1)
        bpGrid = np.empty((ngrid,2,3))
        btypeGrid = np.empty(ngrid, dtype = int)
        bgrid = np.stack((grid[:-1,:-1], grid[:-1,1:], grid[1:,1:], grid[1:,:-1]), axis=2)
        assert(bgrid.shape[0] == self.ny-1 and bgrid.shape[1] == self.nx-1 and bgrid.shape[2] == 4)
        bgrid = bgrid.reshape((ngrid,4))
        xx0 = self.xx[:-1,:-1].reshape(ngrid)
        xx1 = self.xx[:-1,1:].reshape(ngrid)
        yy0 = self.yy[:-1,:-1].reshape(ngrid)
        yy1 = self.yy[1:,:-1].reshape(ngrid)

        # unimplemented boundaries: if assertation error, increase the resolution of the grid
        unb = np.array([1,0,1,0], dtype = bool)
        unpick = (bgrid - unb == 0).all(-1)
        nunpick = np.sum(unpick)
        assert(nunpick == 0)
        unb = np.array([0,1,0,1], dtype = bool)
        unpick = (bgrid - unb == 0).all(-1)
        nunpick = nunpick + np.sum(unpick)
        assert(nunpick == 0)
        ## pick boundary
        # horizontal
        hb = np.array([0,0,1,1], dtype = bool)
        hpick = (bgrid - hb == 0).all(-1)
        hb = np.array([1,1,0,0], dtype = bool)
        hpick = np.logical_or(hpick, (bgrid - hb == 0).all(-1))
        nhpick = np.sum(hpick)
        if nhpick > 0:
            bpGrid[hpick,0,0] = xx0[hpick]
            bpGrid[hpick,0,1] = (xx0[hpick] + xx1[hpick])/2
            bpGrid[hpick,0,2] = xx1[hpick]
            bpGrid[hpick,1,:] = (yy0[hpick] + yy1[hpick]).reshape(nhpick,1)/2
        # vertical
        vb = np.array([1,0,0,1], dtype = bool)
        vpick = (bgrid - vb == 0).all(-1)
        vb = np.array([0,1,1,0], dtype = bool)
        vpick = np.logical_or(vpick, (bgrid - vb == 0).all(-1))
        nvpick = np.sum(vpick)
        if nvpick > 0:
            bpGrid[vpick,1,0] = yy0[vpick]
            bpGrid[vpick,1,1] = (yy0[vpick] + yy1[vpick])/2
            bpGrid[vpick,1,2] = yy1[vpick]
            bpGrid[vpick,0,:] = (xx0[vpick] + xx1[vpick]).reshape(nvpick,1)/2
        assert(np.logical_and(vpick,hpick).any() == False)
        ### first horizontal 0:1 then vertical 1:2
        def pick_boundary_in_turn(bpattern, xx, yy):
            pick = (bgrid - bpattern == 0).all(-1)
            bpattern = np.logical_not(bpattern)
            pick = np.logical_or(pick, (bgrid - bpattern == 0).all(-1))
            npick = np.sum(pick)
            if npick:
                bpGrid[pick,1,:2] = (yy0[pick] + yy1[pick]).reshape(npick,1)/2
                bpGrid[pick,0,1:] = (xx0[pick] + xx1[pick]).reshape(npick,1)/2
                bpGrid[pick,1,2] = yy[pick]
                bpGrid[pick,0,0] = xx[pick]
            return pick, npick

        # upper left
        ulpick, nulpick = pick_boundary_in_turn(np.array([1,0,0,0], dtype=bool), xx0, yy0)
        btype2pick = ulpick
        assert(np.logical_and(ulpick,hpick).any() == False)
        assert(np.logical_and(ulpick,vpick).any() == False)
        # lower left
        llpick, nllpick = pick_boundary_in_turn(np.array([0,0,0,1], dtype=bool), xx0, yy1)
        btype2pick = np.logical_or(btype2pick,llpick)
        assert(np.logical_and(llpick,hpick).any() == False)
        assert(np.logical_and(llpick,vpick).any() == False)
        assert(np.logical_and(llpick,ulpick).any() == False)
        # upper right 
        urpick, nurpick = pick_boundary_in_turn(np.array([0,1,0,0], dtype = bool), xx1, yy0)
        btype2pick = np.logical_or(btype2pick,urpick)
        assert(np.logical_and(urpick,hpick).any() == False)
        assert(np.logical_and(urpick,vpick).any() == False)
        assert(np.logical_and(urpick,ulpick).any() == False)
        assert(np.logical_and(urpick,llpick).any() == False)
        # lower right 
        lrpick, nlrpick = pick_boundary_in_turn(np.array([0,0,1,0], dtype = bool), xx1, yy1)
        btype2pick = np.logical_or(btype2pick,lrpick)
        assert(np.logical_and(lrpick,hpick).any() == False)
        assert(np.logical_and(lrpick,vpick).any() == False)
        assert(np.logical_and(lrpick,ulpick).any() == False)
        assert(np.logical_and(lrpick,llpick).any() == False)
        assert(np.logical_and(lrpick,urpick).any() == False)

        ## assemble
        btypeGrid[hpick] = 0
        btypeGrid[vpick] = 1
        btypeGrid[btype2pick] = 2
        nbtype2pick = np.sum([nulpick, nllpick, nurpick, nlrpick])
        nbtype = nhpick + nvpick + nbtype2pick

        nout = np.sum((bgrid-np.array([0,0,0,0], dtype = bool) == 0).all(-1))
        nin = np.sum((bgrid-np.array([1,1,1,1], dtype = bool) == 0).all(-1))
        assert(nbtype == ngrid - np.sum([nin, nout, nunpick]))

        btype = np.empty(nbtype, dtype = int)
        btype[:nhpick] = 0
        btype[nhpick:(nhpick+nvpick)] = 1
        btype[(nhpick+nvpick):] = 2
        bp  = np.empty((nbtype,2,3))
        bp[:nhpick,:,:] = bpGrid[hpick,:,:] 
        bp[nhpick:(nhpick+nvpick),:,:] = bpGrid[vpick,:,:]
        bp[(nhpick+nvpick):,:,:] = bpGrid[btype2pick,:,:]
        return bp, btype
    #awaits vectorization
    def assign_pos_VF0(self):
        self.vpos = np.empty((2,self.networkSize))
        d0 = np.zeros(self.npolar-1)
        d1 = np.zeros(self.npolar-1)
        ix0 = np.zeros(self.npolar-1,dtype='int')
        for i in range(self.networkSize):
            # find iso-polar lines whose xrange does not include x-coord of the neuron 
            mask = np.logical_and(self.pos[0,i] - self.vx[:-1,:-1] > 0, self.pos[0,i] - self.vx[1:,:-1] < 0)
            pmask = mask.any(0)
            pnull = np.logical_not(pmask)
            ix = np.nonzero(mask.T)[1]
            ix0[pmask] = ix
            ix0[pnull] = -1
            d0[pmask] = np.power(self.vx[ix,  np.arange(self.npolar-1)[pmask]] - self.pos[0,i],2) + np.power(self.vy[ix,  np.arange(self.npolar-1)[pmask]] - self.pos[1,i],2)
            d1[pmask] = np.power(self.vx[ix+1,np.arange(self.npolar-1)[pmask]] - self.pos[0,i],2) + np.power(self.vy[ix+1,np.arange(self.npolar-1)[pmask]] - self.pos[1,i],2)
            d0[pnull] = np.float('inf')
            d1[pnull] = np.float('inf')
            #find minimum distance to iso-ecc and iso-polar node indices
            dis = np.min(np.vstack([d0, d1]),0)
            idp = np.argmin(dis)
            idx = ix0[idp]
            ## interpolate VF-ecc to pos
            self.vpos[0,i] = np.exp(np.log(self.e_range[idx]+1) + (np.log(self.e_range[idx+1]+1) - np.log(self.e_range[idx]+1)) * np.sqrt(dis[idp])/(np.sqrt(d0[idp])+np.sqrt(d1[idp])))-1
            ## interpolate VF-polar to pos
            vp_y0 = y_ep(self.vpos[0,i],self.p_range[idp],self.k,self.a,self.b)
            vp_x0 = x_ep(self.vpos[0,i],self.p_range[idp],self.k,self.a,self.b)
            vp_y1 = y_ep(self.vpos[0,i],self.p_range[idp+1],self.k,self.a,self.b)
            vp_x1 = x_ep(self.vpos[0,i],self.p_range[idp+1],self.k,self.a,self.b)
            dp0 = np.sqrt(np.power(self.pos[0,i]-vp_x0,2) + np.power(self.pos[1,i]-vp_y0,2))
            dp1 = np.sqrt(np.power(self.pos[0,i]-vp_x1,2) + np.power(self.pos[1,i]-vp_y1,2))
            self.vpos[1,i] = self.p_range[idp] + (self.p_range[idp+1] - self.p_range[idp]) * dp0/(dp0+dp1)
            #assert(self.vpos[1,i] >= self.p_range[0] and self.vpos[1,i] <= self.p_range[-1])
            stdout.write(f'\rassgining visual field: {(i+1)/self.networkSize*100:.3f}%')
        stdout.write('\n')
        self.pVFready = True
        return self.vpos

    #was for assign_pos_VF0 but actually is wrong, the visual field position flipped inside the OD stripes
    def reconcile_OD_VF(self, seed=None):
        if self.OD_VF_reconciled == False:
            if self.pVFready == False:
                self.assign_pos_VF()
            if self.pODready == False:
                self.assign_pos_OD1()
            label = np.copy(self.ODlabel)

            np.random.seed(seed)
            np.random.shuffle(label)
            R0 = self.ODlabel>0
            L0 = self.ODlabel<0
            R1 = label>0
            L1 = label<0

            # positional Right shuffled to Left OD to sample visual field
            pR = np.logical_and(R0,L1)
            # positional Left shuffled to Right OD to sample visual field
            pL = np.logical_and(L0,R1)

            assert(np.logical_and(pR,pL).any()==False)
            nR = np.sum(pR)
            nL = np.sum(pL) 
            assert(nR == nL)
            print(f'shuffled {nR} pairs, requiring {nL*nR*8/1024/1024/1024:.3f} Gb')
            # calculate distance
            Rind = np.arange(self.networkSize)[pR]
            Lind = np.arange(self.networkSize)[pL]
            disMat = np.power(np.tile(self.pos[0,pR],(nL,1)) - self.pos[0,pL].reshape(nL,1),2) + np.power(np.tile(self.pos[1,pR],(nL,1)) - self.pos[1,pL].reshape(nL,1),2)
            # bring back visual field position
            ind = np.argsort(disMat,1)
            print('distance of shuffled pairs sorted')
            l_range = np.arange(nL)
            rsel = np.ones(nR,dtype=bool)
            lsel = np.ones(nR,dtype=bool)
            iq = np.zeros(nR,dtype=int)
            for i in range(nR):
                # pick next min from all rows
                head = disMat[l_range[lsel],ind[l_range[lsel],iq[lsel]]]
                # pos id for min of all pairs
                assert(head.size == np.sum(lsel))
                indL = l_range[lsel][np.argmin(head)]
                indR = ind[indL,iq[indL]]
                # update bool select vector
                assert(lsel[indL] == True)
                assert(rsel[indR] == True)
                assert(Rind[indR] != Lind[indL])
                lsel[indL] = False
                rsel[indR] = False
                for j in l_range[lsel]: 
                    while rsel[ind[j,iq[j]]] == False:
                        iq[j] = iq[j] + 1
                ### NOTE: deep copy is required when swap vector
                self.vpos[:,Lind[indL]], self.vpos[:,Rind[indR]] = self.vpos[:,Rind[indR]].copy(), self.vpos[:,Lind[indL]].copy()
                stdout.write(f'\rreconciling OD and VF: {(i+1)/nR*100:.3f}%')
            stdout.write('\n')
            self.OD_VF_reconciled = True

            return self.vpos

    def spread_pos(self, dt, particle_param = None, boundary_param = None, ax1 = None, ax2 = None, seed = None):
        pR = self.ODlabel > 0
        pL = self.ODlabel < 0
        nR = np.sum(pR)
        nL = np.sum(pL)
        model_block = lambda p, e: model_block_ep(e,p,self.k,self.a,self.b)
        area = integrate.dblquad(model_block,0,self.ecc,self.p_range[0],self.p_range[-1])[0]
        subgrid = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])

        # right OD boundary 
        LR = self.LR.copy()
        LR[self.Pi < 1] = 0
        LR[self.LR == -1] = 0
        OD_boundR, btypeR = self.define_bound(LR)
        # left OD boundary 
        LR[self.LR == -1] = 1
        LR[self.LR == 1] = 0
        OD_boundL, btypeL = self.define_bound(LR)

        cl_L = np.sqrt(area/nL)
        kL = 1
        self.pos[:,pL] = simulate_repel(area, subgrid, self.pos[:,pL], dt, OD_boundL, btypeL, ax = ax1, seed = seed)
        cl_R = np.sqrt(area/nR)
        kR = 1
        self.pos[:,pR] = simulate_repel(area, subgrid, self.pos[:,pR], dt, OD_boundR, btypeR, ax = ax2, seed = seed)

    def spread_vpos(self, m=0.5, rext=2, ax1 = None, ax2 = None):
        #self.pos as starting position 
        pR = self.ODlabel > 0
        pL = self.ODlabel < 0
        nR = np.sum(pR)
        nL = np.sum(pL)
        posL = self.pos[:,pL].copy()
        posR = self.pos[:,pR].copy()

        # outer boundary
        boundary, btype = self.define_bound(self.Pi)

        model_block = lambda p, e: model_block_ep(e,p,self.k,self.a,self.b)
        area = integrate.dblquad(model_block,0,self.ecc,self.p_range[0],self.p_range[-1])[0]
        # characteristic length between position
        #cl_L = np.sqrt(area/nL)
        #kL = (m/(area/nL))
        spread(posL, nL, boundary, btype, ax1)
        #cl_R = np.sqrt(area/nR)
        #kR = (m/(area/nR))
        spread(posR, nR, boundary, btype, ax2)
        
    def plot_map(self,ax1,ax2,pltOD=True,pltVF=True,ngridLine=4):

        if pltOD == True:
            if self.pODready == False:
                #self.assign_pos_OD0()
                self.assign_pos_OD1()
            ax2.plot(self.pos[0,self.ODlabel>0], self.pos[1,self.ODlabel>0],',k')
            ax2.plot(self.pos[0,self.ODlabel<0], self.pos[1,self.ODlabel<0],',w')

        if pltVF == True:
            if self.pVFready == False:
                self.assign_pos_VF()
            plt.sca(ax1)
            plt.polar(self.vpos[1,self.ODlabel>0], self.vpos[0,self.ODlabel>0],',k')
            plt.polar(self.vpos[1,self.ODlabel<0], self.vpos[0,self.ODlabel<0],',w')

        if ngridLine > 0:
            plt.sca(ax1)
            for ip in range(self.npolar):
                if ip % (self.npolar//ngridLine)== 0:
                    plt.polar(self.p_range[ip]+np.zeros(self.necc), self.e_range,':',c='0.5')
            for ie in range(self.necc):
                if ie % (self.necc//ngridLine)== 0:
                    plt.polar(self.p_range, self.e_range[ie]+np.zeros(self.npolar),':',c='0.5')
            for ip in range(self.npolar):
                if ip % (self.npolar//ngridLine)== 0:
                    ax2.plot(self.vx[:,ip], self.vy[:,ip],':',c='0.5')
            for ie in range(self.necc):
                if ie % (self.necc//ngridLine)== 0:
                    ax2.plot(self.vx[ie,:], self.vy[ie,:],':',c='0.5')
