from patch_geo_func import x_ep, y_ep
import matplotlib.pyplot as plt
from sys import stdout
import numpy as np
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
            
    # memory requirement low, slower
    def find_pos_in_OD(self):
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
    def find_OD_in_pos(self):
        x0 = self.x[:-1]
        x1 = self.x[1:]
        y0 = self.y[:-1]
        y1 = self.y[1:]

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

    #awaits vectorization
    def assign_VF(self):
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

    def plot_map(self,ax1,ax2,pltOD=True,ngridLine=4):

        if pltOD == True:
            if self.pODready == False:
                self.find_pos_in_OD()
                #self.find_OD_in_pos()

            ax2.plot(self.pos[0,self.ODlabel>0], self.pos[1,self.ODlabel>0], ',k')
            ax2.plot(self.pos[0,self.ODlabel<0], self.pos[1,self.ODlabel<0], ',w')

            if self.pVFready == False:
                self.assign_VF()
            plt.sca(ax1)
            plt.polar(self.vpos[1,self.ODlabel>0], self.vpos[0,self.ODlabel>0],',k')
            plt.polar(self.vpos[1,self.ODlabel<0], self.vpos[0,self.ODlabel<0],',w')

        if ngridLine > 0:
            plt.sca(ax1)
            for ip in range(self.npolar):
                if ip % (self.npolar//ngridLine)== 0:
                    plt.polar(self.p_range[ip]+np.zeros(self.necc),self.e_range,':',c='0.5')
            for ie in range(self.necc):
                if ie % (self.necc//ngridLine)== 0:
                    plt.polar(self.p_range, self.e_range[ie]+np.zeros(self.npolar),':',c='0.5')
            for ip in range(self.npolar):
                if ip % (self.npolar//ngridLine)== 0:
                    ax2.plot(self.vx[:,ip], self.vy[:,ip],':',c='0.5')
            for ie in range(self.necc):
                if ie % (self.necc//ngridLine)== 0:
                    ax2.plot(self.vx[ie,:],self.vy[ie,:],':',c='0.5')
