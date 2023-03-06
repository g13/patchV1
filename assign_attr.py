from patch_geo_func import x_ep, y_ep, model_block_ep, e_x
from repel_system import simulate_repel
from parallel_repel_system import simulate_repel_parallel
from p_repel_system import parallel_repel
from multiprocessing.sharedctypes import RawArray
import multiprocessing as mp
from ctypes import c_double, c_int32

import matplotlib.pyplot as plt
import matplotlib.cm as cm 

from sys import stdout
from scipy import integrate
import numpy as np
import time
import py_compile
py_compile.compile('assign_attr.py')
import functools
print = functools.partial(print, flush=True)
np.seterr(invalid = 'raise', divide = 'raise')

class macroMap:
    def __init__(self, LR_Pi_file, pos_file, crop = 0, realign = False, posUniform = False, OPgrid_file = None, OD_file = None, OP_file = None, VFxy_file = None, noAdjust = False):
        with open(pos_file,'r') as f:
            self.nblock = np.fromfile(f,'u4', count=1)[0]
            self.blockSize = np.fromfile(f,'u4', count=1)[0]
            self.dataDim = np.fromfile(f,'u4', count=1)[0]
            self.networkSize = self.nblock * self.blockSize
            print(f'nblock = {self.nblock}, blockSize = {self.blockSize}, posDim = {self.dataDim}')
            pos = np.reshape(np.fromfile(f,'f8',count = self.dataDim*self.networkSize),(self.nblock,self.dataDim,self.blockSize))
        self.pos = np.zeros((2,self.networkSize))
        self.pos[0,:] = pos[:,0,:].reshape(self.networkSize) # dont use transpose
        self.pos[1,:] = pos[:,1,:].reshape(self.networkSize)
        self.oldpos = self.pos.copy()
        if self.dataDim == 3:
            self.zpos = pos[:,2,:]

        with open(LR_Pi_file,'r') as f:
            self.a, self.b, self.k, self.ecc = np.fromfile(f, 'f8', 4)
            nx, ny = np.fromfile(f, 'i4', 2)
            self.Pi = np.fromfile(f, 'i4', count = nx*ny).reshape(ny,nx)
            #for i in range(ny):
            #    print(self.Pi[i, :])
            #print('====================')
            self.x = np.fromfile(f, 'f8', count = nx)
            x_max = max(self.pos[0,:])
            x_min = min(self.pos[0,:])
            if (x_min < self.x[0] or x_max > self.x[-1]) and not noAdjust:
                raise Exception(f'neuronal x-position {[x_min, x_max]} break outside of grid {[self.x[0], self.x[-1]]}')
            self.y = np.fromfile(f, 'f8', count = ny)
            y_max = max(self.pos[1,:])
            y_min = min(self.pos[1,:])
            if (y_min < self.y[0] or y_max > self.y[-1]) and not noAdjust:
                raise Exception(f'neuronal y-position {[y_min, y_max]} break outside of grid {[self.y[0], self.y[-1]]}')
            LR = np.fromfile(f, 'f8', count = nx*ny).reshape(ny,nx)

            if crop > 0:
                x_crop = x_ep(crop,0,self.k,self.a,self.b)
                self.nx = np.nonzero(self.x > x_crop)[0][0] + 1
                x_ecc = x_crop
                if x_max >= x_ecc:
                    raise Exception('cropping out neurons')
                self.ecc = crop
                self.x = self.x[:self.nx]
                y0 = np.nonzero(self.y < y_min)[0][-1]
                y1 = np.nonzero(self.y > y_max)[0][0]+1
                self.y = self.y[y0:y1]
                self.ny = self.y.size
                self.Pi = self.Pi[y0:y1,:self.nx]
            else:
                self.nx = nx
                self.ny = ny

            if realign:
                if crop == 0:
                    x_ecc = x_ep(self.ecc,0,self.k,self.a,self.b)

            if crop > 0 or realign:
                # seal the right side of the crop
                n = self.ny*self.nx*10
                p_range = np.linspace(-np.pi/2,np.pi/2,n)
                rx = np.array([x_ep(self.ecc, p,self.k,self.a,self.b) for p in p_range])
                ry = np.array([y_ep(self.ecc, p,self.k,self.a,self.b) for p in p_range])
                for iy in range(self.ny):
                    if self.y[iy] < ry[0] or self.y[iy] > ry[-1]:
                        self.Pi[iy,:] = 0
                        continue
                    else:
                        jy = np.nonzero(self.y[iy] <= ry)[0][0]

                    if jy == 0:
                        x = rx[0]
                    else:
                        r = (self.y[iy] - ry[jy-1])/(ry[jy]-ry[jy-1])
                        x = rx[jy-1] + r*(rx[jy]-rx[jy-1])

                    for ix in range(self.nx):
                        if self.x[ix] > x:
                            self.Pi[iy,ix] = 0

                if realign: # seal other sides of the crop
                    # left side
                    self.Pi[:, self.x < 0] = 0
                    # top 
                    e_range = np.zeros(n, dtype = float)
                    x_range = np.linspace(0,x_ecc,n)
                    e_range[1:] = np.array([e_x(x, self.k,self.a,self.b) for x in x_range[1:]])
                    
                    tx = np.array([x_ep(e, np.pi/2,self.k,self.a,self.b) for e in e_range])
                    ty = np.array([y_ep(e, np.pi/2,self.k,self.a,self.b) for e in e_range])
                    for iy in range(self.ny):
                        if self.y[iy] < ty[0]: 
                            continue
                        if self.y[iy] > ty[-1]: 
                            self.Pi[iy,:] = 0
                            break
                        else:
                            jy = np.nonzero(self.y[iy] <= ty)[0][0]
                            if jy == 0:
                                x = 0
                            else:
                                r = (self.y[iy] - ty[jy-1])/(ty[jy]-ty[jy-1])
                                x = tx[jy-1] + r*(tx[jy]-tx[jy-1])
                            for ix in range(self.nx):
                                if self.x[ix] < x:
                                    self.Pi[iy,ix] = 0
                    # bot 
                    bx = np.flip(np.array([x_ep(e, -np.pi/2,self.k,self.a,self.b) for e in e_range]))
                    by = np.flip(np.array([y_ep(e, -np.pi/2,self.k,self.a,self.b) for e in e_range]))
                    print(by)
                    for iy in range(self.ny):
                        if self.y[iy] > by[-1]:
                            break 
                        if self.y[iy] < by[0]: 
                            self.Pi[iy,:] = 0
                            continue
                        else:
                            jy = np.nonzero(self.y[iy] <= by)[0][0]
                            r = (self.y[iy] - by[jy-1])/(by[jy]-by[jy-1])
                            x = bx[jy-1] + r*(bx[jy]-bx[jy-1])
                            for ix in range(self.nx):
                                if self.x[ix] < x:
                                    self.Pi[iy,ix] = 0

                print(f'new x: #{self.nx}, {[self.x[0], self.x[-1]]}, y: #{self.ny}, {[self.y[0], self.y[-1]]}')
                print(f'cropped grid of {nx}x{ny} to {self.nx}x{self.ny}')
                #for i in range(self.ny):
                #    print(self.Pi[i, :])

            self.xx, self.yy = np.meshgrid(self.x, self.y)
            # double to int
            self.LR = np.empty((ny,nx), dtype = 'i4') 
            self.LR[LR > 0] = 1
            self.LR[LR < 0] = -1
            print(f'ODgrid: {np.sum(LR < 0)} contra, {np.sum(LR>0)} ipsi')
            if crop > 0:
                self.LR = self.LR[y0:y1,:self.nx]

            self.LR[self.Pi <=0] = 0
            #print('=======')
            #for i in range(self.ny):
            #    print(self.LR[i, :])

        ratio = self.Pi.size/(np.sum(self.Pi>0))
        self.necc = np.round(self.nx * ratio).astype(int)
        self.npolar = np.round(self.ny * ratio).astype(int)
        self.e_range = np.exp(np.linspace(np.log(1),np.log(self.ecc+1),self.necc))-1
        self.p_range = np.linspace(-np.pi/2,np.pi/2,self.npolar)

        self.model_block = lambda p, e: model_block_ep(e,p,self.k,self.a,self.b)
        self.area = integrate.dblquad(self.model_block,0,self.ecc,self.p_range[0],self.p_range[-1])[0]
        self.subgrid = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])
        memoryRequirement = ((np.int64(self.necc-1)*np.int64(self.npolar-1)*8 + np.int64(self.networkSize*(self.nx + self.ny)))*8/1024/1024/1024)
        print(f'{self.necc}x{self.npolar}, ecc-polar grid houses {self.networkSize} neurons')
        print(f'require {memoryRequirement:.3f} GB')
        self.vx = np.empty((self.necc,self.npolar))
        self.vy = np.empty((self.necc,self.npolar))
        for ip in range(self.npolar):
            self.vx[:,ip] = [x_ep(e,self.p_range[ip],self.k,self.a,self.b) for e in self.e_range]
            self.vy[:,ip] = [y_ep(e,self.p_range[ip],self.k,self.a,self.b) for e in self.e_range]
        
        if crop > 0 or realign:
            with open(LR_Pi_file[:-4] + '_adjusted.bin', 'wb') as f:
                np.array([self.a, self.b, self.k, self.ecc]).astype('f8').tofile(f)
                np.array([self.nx, self.ny]).astype('i4').tofile(f)
                self.Pi.astype('i4').tofile(f)
                self.x.astype('f8').tofile(f)
                self.y.astype('f8').tofile(f)
                self.LR.astype('f8').tofile(f)

        # read preset orientation preferences
        if OPgrid_file is not None:
            with open(OPgrid_file,'r') as f:
                self.OPgrid_x = np.reshape(np.fromfile(f, 'f8', count = nx*ny),(ny,nx))
                self.OPgrid_y = np.reshape(np.fromfile(f, 'f8', count = nx*ny),(ny,nx))
                if crop > 0:
                    self.OPgrid_x = self.OPgrid_x[y0:y1,:self.nx]
                    self.OPgrid_y = self.OPgrid_y[y0:y1,:self.nx]
                self.OPgrid = np.arctan2(self.OPgrid_y, self.OPgrid_x)
                if crop > 0 or realign:
                    with open(OPgrid_file[:-4]+'_adjusted.bin','wb') as f:
                        self.OPgrid_x.astype('f8').tofile(f)
                        self.OPgrid_y.astype('f8').tofile(f)

        if OP_file is not None:
            with open(OP_file,'r') as f:
                self.op = np.fromfile(f, 'f8')

        if VFxy_file is not None:
            with open(VFxy_file,'r') as f:
                np.fromfile(f, 'u4', count = 1)
                self.vpos = np.reshape(np.fromfile(f, 'f8', count = self.networkSize*2),self.pos.shape)

        if OD_file is not None:
            with open(OD_file,'rb') as f:
                self.ODlabel = np.fromfile(f, 'i4', count = self.networkSize)
                print('loading OD_labels...')
                print(f'contra {np.sum(self.ODlabel > 0)} + ipsi {np.sum(self.ODlabel < 0)} == total {self.networkSize}')
                print('checked')

        
        self.pODready = OD_file is not None
        self.pOPready = OP_file is not None 
        self.LR_boundary_defined = False 
        self.pVFready = VFxy_file is not None 
        self.vposLready = VFxy_file is not None
        self.vposRready = VFxy_file is not None
        self.posUniform = posUniform
        self.adjusted = False 
        self.noAdjust = noAdjust
        if not posUniform and not noAdjust:
            self.adjust_pos(bfile = pos_file[:-4] + '-bound')
        else:
            self.adjusted = True
            self.opick = None
        # not used
        self.OD_VF_reconciled = False 

    # interpolate orientation preference for each neuron in from the OP grid
    def assign_pos_OP(self, force):
        if not self.posUniform and not force:
            print('positions are not yet adjusted')
            return None
        i, j, coord = self.get_ij_grid(get_d2 = False, get_coord = True)

        def thetaRebound(a):
            pick = a > np.pi/2
            a[pick] = a[pick] - np.pi
            pick = a < -np.pi/2
            a[pick] = a[pick] + np.pi
            return a
        def thetaInterpolate(a, b, r):
            #assert((r>=0).all() and (r<=1).all())
            mean = thetaRebound(a + r*thetaRebound(b-a))
            return mean 

                                #bot-left     #top-left     #bot-right        #top-right
        opgrid_x = np.stack((self.OPgrid_x[i,j], self.OPgrid_x[i+1,j], self.OPgrid_x[i,j+1], self.OPgrid_x[i+1,j+1]), axis = -1)
        opgrid_y = np.stack((self.OPgrid_y[i,j], self.OPgrid_y[i+1,j], self.OPgrid_y[i,j+1], self.OPgrid_y[i+1,j+1]), axis = -1)
        assert((np.isnan(opgrid_x) == np.isnan(opgrid_y)).all())
        #  i,j
        # grid:
        #  1,0; 1,1 | 1; 3
        #  0,0; 0,1 | 0; 2 
        a = np.empty((self.networkSize))
        b = np.empty((self.networkSize))
        self.op = np.empty((self.networkSize))
        # if fully defined:
        pick = np.logical_not(np.isnan(opgrid_x).any(-1))
        if np.sum(pick) > 0:
            # interpolation direction left -> right, bottom -> top
            a = opgrid_x[pick,0] + coord[0,pick] * (opgrid_x[pick,2] - opgrid_x[pick,0])
            b = opgrid_x[pick,1] + coord[0,pick] * (opgrid_x[pick,3] - opgrid_x[pick,1])
            op_x = a + coord[1,pick] * (b - a)

            a = opgrid_y[pick,0] + coord[0,pick] * (opgrid_y[pick,2] - opgrid_y[pick,0])
            b = opgrid_y[pick,1] + coord[0,pick] * (opgrid_y[pick,3] - opgrid_y[pick,1])
            op_y = a + coord[1,pick] * (b - a)
            self.op[pick] = np.arctan2(op_y, op_x)

        # if bottom-left undefined:
        pick = np.logical_and(np.isnan(opgrid_x[:,0]), np.logical_not(np.isnan(opgrid_x[:,1:]).any(-1)))
        if np.sum(pick) > 0:
            # origin at top-right # 3 fan from top to right
            fraction = np.arctan2(1-coord[1,pick], 1-coord[0,pick])/(np.pi/2)
            #assert(np.sum(fraction > 1) == 0)
            a = opgrid_x[pick,3] + (opgrid_x[pick,1] - opgrid_x[pick,3])*coord[0,pick]
            b = opgrid_x[pick,3] + (opgrid_x[pick,2] - opgrid_x[pick,3])*coord[1,pick]
            op_x = a + fraction * (b - a)

            a = opgrid_y[pick,3] + (opgrid_y[pick,1] - opgrid_y[pick,3])*coord[0,pick]
            b = opgrid_y[pick,3] + (opgrid_y[pick,2] - opgrid_y[pick,3])*coord[1,pick]
            op_y = a + fraction * (b - a)
            self.op[pick] = np.arctan2(op_y, op_x)

        # if top-left undefined:
        pick = np.logical_and(np.isnan(opgrid_x[:,1]), np.logical_not(np.isnan(opgrid_x[:,[0,2,3]]).any(-1)))
        if np.sum(pick) > 0:
            # origin at bottom-right # 2 fan from bot to right
            fraction = np.arctan2(coord[1,pick], 1-coord[0,pick])/(np.pi/2)
            #assert(np.sum(fraction > 1) == 0)
            a = opgrid_x[pick,2] + (opgrid_x[pick,0] - opgrid_x[pick,2])*coord[0,pick]
            b = opgrid_x[pick,2] + (opgrid_x[pick,3] - opgrid_x[pick,2])*coord[1,pick]
            op_x = a + fraction * (b - a)

            a = opgrid_y[pick,2] + (opgrid_y[pick,0] - opgrid_y[pick,2])*coord[0,pick]
            b = opgrid_y[pick,2] + (opgrid_y[pick,3] - opgrid_y[pick,2])*coord[1,pick]
            op_y = a + fraction * (b - a)
            self.op[pick] = np.arctan2(op_y, op_x)
            
        # if bottom-right undefined:
        pick = np.logical_and(np.isnan(opgrid_x[:,2]), np.logical_not(np.isnan(opgrid_x[:,[0,1,3]]).any(-1)))
        if np.sum(pick) > 0:
            # origin at top-left # 1 fan from top to left
            fraction = np.arctan2(1-coord[1,pick], coord[0,pick])/(np.pi/2)
            #assert(np.sum(fraction > 1) == 0)
            a = opgrid_x[pick,1] + (opgrid_x[pick,3] - opgrid_x[pick,1])*coord[0,pick]
            b = opgrid_x[pick,1] + (opgrid_x[pick,0] - opgrid_x[pick,1])*coord[1,pick]
            op_x = a + fraction * (b - a)

            a = opgrid_y[pick,1] + (opgrid_y[pick,3] - opgrid_y[pick,1])*coord[0,pick]
            b = opgrid_y[pick,1] + (opgrid_y[pick,0] - opgrid_y[pick,1])*coord[1,pick]
            op_y = a + fraction * (b - a)
            self.op[pick] = np.arctan2(op_y, op_x)

        # if top-right undefined:
        pick = np.logical_and(np.isnan(opgrid_x[:,3]), np.logical_not(np.isnan(opgrid_x[:,:-1]).any(-1)))
        if np.sum(pick) > 0:
            # origin at top-left # 0 fan from bot to left
            fraction = np.arctan2(coord[1,pick], coord[0,pick])/(np.pi/2)
            #assert(np.sum(fraction > 1) == 0)
            a = opgrid_x[pick,0] + (opgrid_x[pick,2] - opgrid_x[pick,0])*coord[0,pick]
            b = opgrid_x[pick,0] + (opgrid_x[pick,1] - opgrid_x[pick,0])*coord[1,pick]
            op_x = a + fraction * (b - a)

            a = opgrid_y[pick,0] + (opgrid_y[pick,2] - opgrid_y[pick,0])*coord[0,pick]
            b = opgrid_y[pick,0] + (opgrid_y[pick,1] - opgrid_y[pick,0])*coord[1,pick]
            op_y = a + fraction * (b - a)
            self.op[pick] = np.arctan2(op_y, op_x)

        # if top vertices undefined:
        pick = np.logical_and(np.isnan(opgrid_x[:,[1,3]]).all(-1), np.logical_not(np.isnan(opgrid_x[:,[0,2]]).any(-1)))
        if np.sum(pick) > 0:
            op_x = opgrid_x[pick,0] + (opgrid_x[pick,2] - opgrid_x[pick,0])*coord[0,pick]
            op_y = opgrid_y[pick,0] + (opgrid_y[pick,2] - opgrid_y[pick,0])*coord[0,pick]
            self.op[pick] = np.arctan2(op_y, op_x)

        # if bottom vertices undefined:
        pick = np.logical_and(np.isnan(opgrid_x[:,[0,2]]).all(-1), np.logical_not(np.isnan(opgrid_x[:,[1,3]]).any(-1)))
        if np.sum(pick) > 0:
            op_x = opgrid_x[pick,1] + (opgrid_x[pick,3] - opgrid_x[pick,1])*coord[0,pick]
            op_y = opgrid_y[pick,1] + (opgrid_y[pick,3] - opgrid_y[pick,1])*coord[0,pick]
            self.op[pick] = np.arctan2(op_y, op_x)

        # if left vertices undefined:
        pick = np.logical_and(np.isnan(opgrid_x[:,[0,1]]).all(-1), np.logical_not(np.isnan(opgrid_x[:,[2,3]]).any(-1)))
        if np.sum(pick) > 0:
            op_x = opgrid_x[pick,2] + (opgrid_x[pick,3] - opgrid_x[pick,2])*coord[1,pick]
            op_y = opgrid_y[pick,2] + (opgrid_y[pick,3] - opgrid_y[pick,2])*coord[1,pick]
            self.op[pick] = np.arctan2(op_y, op_x)

        # if right vertices undefined:
        pick = np.logical_and(np.isnan(opgrid_x[:,[3,2]]).all(-1), np.logical_not(np.isnan(opgrid_x[:,[0,1]]).any(-1)))
        if np.sum(pick) > 0:
            op_x = opgrid_x[pick,0] + (opgrid_x[pick,1] - opgrid_x[pick,0])*coord[1,pick]
            op_y = opgrid_y[pick,0] + (opgrid_y[pick,1] - opgrid_y[pick,0])*coord[1,pick]
            self.op[pick] = np.arctan2(op_y, op_x)

        # if only bottom-left is defined:
        pick = np.logical_and(np.isnan(opgrid_x[:,1:]).all(-1), np.logical_not(np.isnan(opgrid_x[:,0])))
        if np.sum(pick) > 0:
            self.op[pick] = self.OPgrid[i[pick], j[pick]]

        # if only bottom-right is defined:
        pick = np.logical_and(np.isnan(opgrid_x[:,[0,1,3]]).all(-1), np.logical_not(np.isnan(opgrid_x[:,2])))
        if np.sum(pick) > 0: # 
            self.op[pick] = self.OPgrid[i[pick], j[pick]+1]

        # if only top-left is defined:
        pick = np.logical_and(np.isnan(opgrid_x[:,[0,2,3]]).all(-1), np.logical_not(np.isnan(opgrid_x[:,1])))
        if np.sum(pick) > 0: #
            self.op[pick] = self.OPgrid[i[pick]+1, j[pick]]

        # if only top-right is defined:
        pick = np.logical_and(np.isnan(opgrid_x[:,[0,1,2]]).all(-1), np.logical_not(np.isnan(opgrid_x[:,3])))
        if np.sum(pick) > 0: #
            self.op[pick] = self.OPgrid[i[pick]+1, j[pick]+1]

        if self.adjusted and not self.noAdjust:
            assert(np.sum(np.isnan(self.op)) == 0)
        self.pOPready = True

        self.op = (self.op + np.pi)/2/np.pi
        print('op ready')
        return self.op

    # new method based on OD1, relief boundary pressure 
    def adjust_pos(self, bfile = None, check = True):
        per_unit_area = np.sqrt(3)/2 # in cl^2
        cl = np.sqrt((self.area/self.networkSize)/per_unit_area)
        if cl > self.subgrid[0]/2 or cl > self.subgrid[1]/2:
            cl = 0.5 * np.min(self.subgrid/2)
            print('restricting buffer length to a quarter of subgrid size')
        print(f'buffer length = {cl}, subgrid: {self.subgrid}')
        buffer_l = cl/4
        print('get_ij_grid')
        it = 0
        while True:
            i, j, d2, coord = self.get_ij_grid(get_d2 = True, get_coord = True)
            ipick = np.argmin(d2,1)

            def get_22ij(index):
                ip = np.mod(index,2)
                jp = index//2
                return ip, jp

            ip, jp = get_22ij(ipick) 
            bpattern = np.array([self.Pi[i, j], self.Pi[i+1, j], self.Pi[i, j+1], self.Pi[i+1, j+1]]).T
            not_rogue = np.logical_not((bpattern - np.array([0,0,0,0]) == 0).all(-1))
            if not_rogue.sum() < self.networkSize:
                raise Exception('not all neurons are near the boundary')
            self.opick = np.logical_and(self.Pi[i + ip, j + jp] == 0, not_rogue)
            if np.sum(self.opick) == 0:
                break
            
            index = np.unique(np.ravel_multi_index([i[self.opick], j[self.opick]], (self.ny, self.nx)))
            # find all affected grids
            affected_i, affected_j = np.unravel_index(index, (self.ny, self.nx))
            for ai in range(affected_i.size):
                stdout.write(f'\rdealing with {ai+1}/{affected_i.size} boundaries...')
                ii = affected_i[ai]
                jj = affected_j[ai]
                                    # left-bot          # left-top          # right-bot         # right-top
                bpattern = np.array([self.Pi[ii, jj], self.Pi[ii+1, jj], self.Pi[ii, jj+1], self.Pi[ii+1, jj+1]])
                # left vertical bound
                if (bpattern - np.array([0,0,1,1]) == 0).all():
                    #print('lv')
                    gpick = np.logical_and(i == ii, j == jj) # inside grid
                    x_min = min(self.pos[0,gpick]) 
                    x0 = (self.x[jj] + self.x[jj+1])/2 + buffer_l
                    if x_min < x0:
                        # pick all above within row
                        rpick = np.logical_and(not_rogue, np.logical_and(i == ii, coord[1,:] > 0.5)) 
                        if np.sum(rpick) > 0:
                            x_max = max(self.pos[0,rpick])
                            if x_min == x_max:
                                self.pos[0,rpick] = x0
                            else:
                                self.pos[0,rpick] = x0 + (self.pos[0,rpick] - x_min)/(x_max - x_min) * (x_max - x0)
                        # pick all below within row
                        rpick = np.logical_and(not_rogue, np.logical_and(i == ii, coord[1,:] <= 0.5)) 
                        if np.sum(rpick) > 0:
                            x_max = max(self.pos[0,rpick])
                            if x_min == x_max:
                                self.pos[0,rpick] = x0
                            else:
                                self.pos[0,rpick] = x0 + (self.pos[0,rpick] - x_min)/(x_max - x_min) * (x_max - x0)
                        # update j, d2 of the whole row
                        ppick = np.logical_and(not_rogue, i == ii) 
                        i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,ppick])
                        j[ppick] = j_new
                        d2[ppick] = d2_new
                        coord[:,ppick] = coord_new
                    continue
                # right vertical bound
                if (bpattern - np.array([1,1,0,0]) == 0).all():
                    #print('rv')
                    gpick = np.logical_and(i == ii, j == jj)
                    x_max = max(self.pos[0,gpick])
                    x1 = (self.x[jj] + self.x[jj+1])/2 - buffer_l
                    if x_max > x1:
                        # pick all above within row
                        rpick = np.logical_and(not_rogue, np.logical_and(i == ii, coord[1,:] > 0.5))
                        if np.sum(rpick) > 0:
                            x_min = min(self.pos[0,rpick])
                            if x_min == x_max:
                                self.pos[0,rpick] = x1
                            else:
                                self.pos[0,rpick] = x1 + (self.pos[0,rpick] - x_max)/(x_max - x_min) * (x1 - x_min)
                        # pick all below within row
                        rpick = np.logical_and(not_rogue, np.logical_and(i == ii, coord[1,:] <= 0.5))
                        if np.sum(rpick) > 0:
                            x_min = min(self.pos[0,rpick])
                            if x_min == x_max:
                                self.pos[0,rpick] = x1
                            else:
                                self.pos[0,rpick] = x1 + (self.pos[0,rpick] - x_max)/(x_max - x_min) * (x1 - x_min)
                        # update j, d2 of the whole row
                        ppick = np.logical_and(not_rogue, i == ii)
                        i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,ppick])
                        j[ppick] = j_new
                        d2[ppick] = d2_new
                        coord[:,ppick] = coord_new
                    continue
                # lower horizontal bound
                if (bpattern - np.array([0,1,0,1]) == 0).all():
                    #print('lh')
                    gpick = np.logical_and(i == ii, j == jj)
                    y_min = min(self.pos[1,gpick])
                    y0 = (self.y[ii] + self.y[ii+1])/2 + buffer_l
                    if y_min < y0:
                        #pick all within column to the right
                        rpick = np.logical_and(not_rogue, np.logical_and(j == jj, coord[0,:] > 0.5))
                        if np.sum(rpick) > 0:
                            y_max = max(self.pos[1,rpick])
                            if y_min == y_max:
                                self.pos[1,rpick] = y0
                            else:
                                self.pos[1,rpick] = y0 + (self.pos[1,rpick] - y_min)/(y_max - y_min) * (y_max - y0)
                        #pick all within column to the left 
                        rpick = np.logical_and(not_rogue, np.logical_and(j == jj, coord[0,:] <= 0.5))
                        if np.sum(rpick) > 0:
                            y_max = max(self.pos[1,rpick])
                            if y_min == y_max:
                                self.pos[1,rpick] = y0
                            else:
                                self.pos[1,rpick] = y0 + (self.pos[1,rpick] - y_min)/(y_max - y_min) * (y_max - y0)
                        # update j, d2 of the whole column 
                        ppick = np.logical_and(not_rogue, j == jj)
                        i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,ppick])
                        i[ppick] = i_new
                        d2[ppick] = d2_new
                        coord[:,ppick] = coord_new
                    continue
                # upper horizontal bound
                if (bpattern - np.array([1,0,1,0]) == 0).all():
                    #print('uh')
                    gpick = np.logical_and(i == ii, j == jj)
                    y_max = max(self.pos[1,gpick])
                    y1 = (self.y[ii] + self.y[ii+1])/2 - buffer_l
                    if y_max > y1:
                        #pick all within column to the right
                        rpick = np.logical_and(not_rogue, np.logical_and(j == jj, coord[0,:] > 0.5))
                        if np.sum(rpick) > 0:
                            y_min = min(self.pos[1,rpick])
                            if y_min == y_max:
                                self.pos[1,rpick] = y1
                            else:
                                self.pos[1,rpick] = y1 + (self.pos[1,rpick] - y_max)/(y_max - y_min) * (y1 - y_min)
                        #pick all within column to the left 
                        rpick = np.logical_and(not_rogue, np.logical_and(j == jj, coord[0,:] <= 0.5))
                        if np.sum(rpick) > 0:
                            y_min = min(self.pos[1,rpick])
                            if y_min == y_max:
                                self.pos[1,rpick] = y1
                            else:
                                self.pos[1,rpick] = y1 + (self.pos[1,rpick] - y_max)/(y_max - y_min) * (y1 - y_min)
                        # update j, d2 of the whole column 
                        ppick = np.logical_and(not_rogue, j == jj)
                        i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,ppick])
                        i[ppick] = i_new
                        d2[ppick] = d2_new
                        coord[:,ppick] = coord_new
                    continue
                # bot-left corner
                if (bpattern - np.array([0,1,1,1]) == 0).all():
                    #print('bl')
                    apick = np.logical_and(i == ii, j == jj) # pick everything in the specified grid
                    aind0 = np.arange(self.networkSize)[apick]
                    aind = aind0[np.logical_and(coord[0, aind0] <= 0.5, coord[1, aind0] <= 0.5)] # points in the bottom-left quarter, outside the boundary 
                    if aind.size == 0:
                        continue

                    axind = aind0[np.logical_and(coord[0, aind0] > 0.5, coord[1, aind0] <= 0.5)] # points in the bottom-right quarter, inside the boundary
                    ayind = aind0[np.logical_and(coord[0, aind0] <= 0.5, coord[1, aind0] > 0.5)] # points in the top-left quarter, inside the boundary

                    ir = np.argmax(np.linalg.norm(coord[:,aind] - 0.5, axis = 0))
                    l = np.abs(coord[:,aind[ir]] - 0.5) 
                    x_thres = 0.5 - l[1]
                    y_thres = 0.5 - l[0]
                    xind = aind[np.logical_and(coord[0,aind] > x_thres, coord[1,aind] <= y_thres)] # bottom-right range of the quarter for xpick
                    yind = aind[np.logical_and(coord[0,aind] <= x_thres, coord[1,aind] > y_thres)] # top-left range of the quarter for ypick
                    rind = aind[np.logical_and(coord[0,aind] > x_thres, coord[1,aind] > y_thres)] # the top-right range of the quarter for random pick
                    nr = rind.size
                    rind = np.random.permutation(rind)
                    if nr % 2 == 0:
                        xrind = rind[:nr//2]
                        yrind = rind[nr//2:]
                    else:
                        if np.random.rand() > 0.5:
                            xrind = rind[:(nr+1)//2]
                            yrind = rind[(nr+1)//2:]
                        else:
                            xrind = rind[:(nr-1)//2]
                            yrind = rind[(nr-1)//2:]

                    if nr + xind.size + yind.size != aind.size:
                        raise Exception(f'{nr} + {xind.size} + {yind.size} != {aind.size}')

                    allpick = np.array([])
                    # pick lower half row
                    if xrind.size + xind.size  > 0:
                        x0 = (self.x[jj] + self.x[jj+1])/2 + buffer_l
                        pick = np.logical_and(np.logical_and(np.logical_and(i==ii, j>jj), coord[1,:] <= 0.5), not_rogue)
                        xind0 = np.arange(self.networkSize)[pick]
                        xpick = np.hstack((axind, xind, xind0, xrind))
                        x_max = max(self.pos[0,xpick])
                        x_min = min(self.pos[0,xpick])
                        if x_max == x_min:
                            self.pos[0,xpick] = x0
                        else:
                            self.pos[0,xpick] = x0 + (self.pos[0,xpick] - x_min)/(x_max - x_min) * (x_max - x0)
                        allpick = xpick
                    # pick left half column
                    if yrind.size + yind.size  > 0:
                        y0 = (self.y[ii] + self.y[ii+1])/2 + buffer_l
                        pick = np.logical_and(np.logical_and(np.logical_and(j==jj, i>ii), coord[0,:] <= 0.5), not_rogue)
                        yind0 = np.arange(self.networkSize)[pick]
                        ypick = np.hstack((ayind, yind, yind0, yrind))
                        y_max = max(self.pos[1,ypick])
                        y_min = min(self.pos[1,ypick])
                        if y_max == y_min:
                            self.pos[1,ypick] = y0
                        else:
                            self.pos[1,ypick] = y0 + (self.pos[1,ypick] - y_min)/(y_max - y_min) * (y_max - y0)
                        if allpick.size == 0:
                            allpick = ypick
                        else:
                            allpick = np.hstack((allpick, ypick))
                    #update i,j,d2 of the half row and column picked above
                    i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,allpick])
                    i[allpick] = i_new
                    j[allpick] = j_new
                    d2[allpick] = d2_new
                    coord[:,allpick] = coord_new
                    continue
                # top-left corner
                if (bpattern - np.array([1,0,1,1]) == 0).all():
                    #print('tl')
                    apick = np.logical_and(i == ii, j == jj)
                    aind0 = np.arange(self.networkSize)[apick]
                    aind = aind0[np.logical_and(coord[0, aind0] <= 0.5, coord[1, aind0] > 0.5)]
                    if aind.size == 0:
                        continue
                    axind = aind0[np.logical_and(coord[0, aind0] > 0.5, coord[1, aind0] > 0.5)]
                    ayind = aind0[np.logical_and(coord[0, aind0] <= 0.5, coord[1, aind0] <= 0.5)]

                    ir = np.argmax(np.linalg.norm(coord[:,aind] - 0.5, axis = 0))
                    l = np.abs(coord[:,aind[ir]] - 0.5)
                    x_thres = 0.5 - l[1]
                    y_thres = 0.5 + l[0]
                    xind = aind[np.logical_and(coord[0,aind] > x_thres, coord[1,aind] > y_thres)]
                    yind = aind[np.logical_and(coord[0,aind] <= x_thres, coord[1,aind] <= y_thres)]
                    rind = aind[np.logical_and(coord[0,aind] > x_thres, coord[1,aind] <= y_thres)]
                    nr = rind.size
                    rind = np.random.permutation(rind)
                    if nr % 2 == 0:
                        xrind = rind[:nr//2]
                        yrind = rind[nr//2:]
                    else:
                        if np.random.rand() > 0.5:
                            xrind = rind[:(nr+1)//2]
                            yrind = rind[(nr+1)//2:]
                        else:
                            xrind = rind[:(nr-1)//2]
                            yrind = rind[(nr-1)//2:]

                    if nr + xind.size + yind.size != aind.size:
                        raise Exception(f'{nr} + {xind.size} + {yind.size} != {aind.size}')

                    allpick = np.array([])
                    # pick upper half row
                    if xrind.size + xind.size  > 0:
                        x0 = (self.x[jj] + self.x[jj+1])/2 + buffer_l
                        pick = np.logical_and(np.logical_and(np.logical_and(i==ii, j>jj), coord[1,:] > 0.5), not_rogue)
                        xind0 = np.arange(self.networkSize)[pick]
                        xpick = np.hstack((axind, xind, xind0, xrind))
                        x_max = max(self.pos[0,xpick])
                        x_min = min(self.pos[0,xpick])
                        if x_min == x_max:
                            self.pos[0,xpick] = x0
                        else:
                            self.pos[0,xpick] = x0 + (self.pos[0,xpick] - x_min)/(x_max - x_min) * (x_max - x0)
                        allpick = xpick
                    # pick left half column 
                    if yrind.size + yind.size  > 0:
                        y1 = (self.y[ii] + self.y[ii+1])/2 - buffer_l
                        pick = np.logical_and(np.logical_and(np.logical_and(j==jj, i<ii), coord[0,:] <= 0.5), not_rogue)
                        yind0 = np.arange(self.networkSize)[pick]
                        ypick = np.hstack((ayind, yind, yind0, yrind))
                        y_max = max(self.pos[1,ypick])
                        y_min = min(self.pos[1,ypick])
                        if y_min == y_max:
                            self.pos[1,ypick] = y1
                        else:
                            self.pos[1,ypick] = y1 + (self.pos[1,ypick] - y_max)/(y_max - y_min) * (y1 - y_min)
                        if allpick.size == 0:
                            allpick = ypick
                        else:
                            allpick = np.hstack((allpick, ypick))
                    #update i,j,d2 of the half row and column picked above
                    i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,allpick])
                    i[allpick] = i_new
                    j[allpick] = j_new
                    d2[allpick] = d2_new
                    coord[:,allpick] = coord_new
                    continue
                # bot-right corner
                if (bpattern - np.array([1,1,0,1]) == 0).all():
                    #print('br')
                    apick = np.logical_and(i == ii, j == jj)
                    aind0 = np.arange(self.networkSize)[apick]
                    aind = aind0[np.logical_and(coord[0, aind0] > 0.5, coord[1, aind0] <= 0.5)]
                    if aind.size == 0:
                        continue
                    axind = aind0[np.logical_and(coord[0, aind0] <= 0.5, coord[1, aind0] <= 0.5)]
                    ayind = aind0[np.logical_and(coord[0, aind0] > 0.5, coord[1, aind0] > 0.5)]

                    ir = np.argmax(np.linalg.norm(coord[:,aind] - 0.5, axis = 0))
                    l = np.abs(coord[:,aind[ir]] - 0.5) 
                    x_thres = 0.5 + l[1]
                    y_thres = 0.5 - l[0]
                    xind = aind[np.logical_and(coord[0,aind] <= x_thres, coord[1,aind] <= y_thres)]
                    yind = aind[np.logical_and(coord[0,aind] > x_thres, coord[1,aind] > y_thres)]
                    rind = aind[np.logical_and(coord[0,aind] <= x_thres, coord[1,aind] > y_thres)]
                    nr = rind.size
                    rind = np.random.permutation(rind)
                    if nr % 2 == 0:
                        xrind = rind[:nr//2]
                        yrind = rind[nr//2:]
                    else:
                        if np.random.rand() > 0.5:
                            xrind = rind[:(nr+1)//2]
                            yrind = rind[(nr+1)//2:]
                        else:
                            xrind = rind[:(nr-1)//2]
                            yrind = rind[(nr-1)//2:]

                    if nr + xind.size + yind.size != aind.size:
                        raise Exception(f'{nr} + {xind.size} + {yind.size} != {aind.size}')

                    allpick = np.array([])
                    # pick lower half row
                    if xrind.size + xind.size  > 0:
                        x1 = (self.x[jj] + self.x[jj+1])/2 - buffer_l
                        pick = np.logical_and(np.logical_and(np.logical_and(i==ii, j < jj), coord[1,:] <= 0.5), not_rogue)
                        xind0 = np.arange(self.networkSize)[pick]
                        xpick = np.hstack((axind, xind, xind0, xrind))
                        x_max = max(self.pos[0,xpick])
                        x_min = min(self.pos[0,xpick])
                        if x_max == x_min:
                            self.pos[0,xpick] = x1
                        else:
                            self.pos[0,xpick] = x1 + (self.pos[0,xpick] - x_max)/(x_max - x_min) * (x1 - x_min)
                        allpick = xpick
                    # pick right half column
                    if yrind.size + yind.size  > 0:
                        y0 = (self.y[ii] + self.y[ii+1])/2 + buffer_l
                        pick = np.logical_and(np.logical_and(np.logical_and(j==jj, i > ii), coord[0,:] > 0.5), not_rogue)
                        yind0 = np.arange(self.networkSize)[pick]
                        ypick = np.hstack((ayind, yind, yind0, yrind))
                        y_max = max(self.pos[1,ypick])
                        y_min = min(self.pos[1,ypick])
                        if y_max == y_min:
                            self.pos[1,ypick] = y0
                        else:
                            self.pos[1,ypick] = y0 + (self.pos[1,ypick] - y_min)/(y_max - y_min) * (y_max - y0)
                        if allpick.size == 0:
                            allpick = ypick
                        else:
                            allpick = np.hstack((allpick, ypick))
                    #update i,j,d2 of the half row and column picked above
                    i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,allpick])
                    i[allpick] = i_new
                    j[allpick] = j_new
                    d2[allpick] = d2_new
                    coord[:,allpick] = coord_new
                    if ii == 36 and jj == 168:
                        print(coord[:,aind])
                        assert(np.logical_or(coord[0,aind] < 0.5, coord[1,aind] > 0.5).all())
                    continue
                # top-right corner
                if (bpattern - np.array([1,1,1,0]) == 0).all():
                    #print('tr')
                    apick = np.logical_and(i == ii, j == jj)
                    aind0 = np.arange(self.networkSize)[apick]
                    aind = aind0[np.logical_and(coord[0, aind0] > 0.5, coord[1, aind0] > 0.5)]
                    if aind.size == 0:
                        continue
                    axind = aind0[np.logical_and(coord[0, aind0] <= 0.5, coord[1, aind0] > 0.5)]
                    ayind = aind0[np.logical_and(coord[0, aind0] > 0.5, coord[1, aind0] <= 0.5)]

                    ir = np.argmax(np.linalg.norm(coord[:,aind] - 0.5, axis = 0))
                    l = np.abs(coord[:,aind[ir]] - 0.5) 
                    x_thres = 0.5 + l[1]
                    y_thres = 0.5 + l[0]
                    xind = aind[np.logical_and(coord[0,aind] <= x_thres, coord[1,aind] > y_thres)]
                    yind = aind[np.logical_and(coord[0,aind] > x_thres, coord[1,aind] <= y_thres)]
                    rind = aind[np.logical_and(coord[0,aind] <= x_thres, coord[1,aind] <= y_thres)]
                    nr = rind.size
                    rind = np.random.permutation(rind)
                    if nr % 2 == 0:
                        xrind = rind[:nr//2]
                        yrind = rind[nr//2:]
                    else:
                        if np.random.rand() > 0.5:
                            xrind = rind[:(nr+1)//2]
                            yrind = rind[(nr+1)//2:]
                        else:
                            xrind = rind[:(nr-1)//2]
                            yrind = rind[(nr-1)//2:]

                    if nr + xind.size + yind.size != aind.size:
                        raise Exception(f'{nr} + {xind.size} + {yind.size} != {aind.size}')

                    allpick = np.array([])
                    # pick upper half row
                    if xrind.size + xind.size  > 0:
                        x1 = (self.x[jj] + self.x[jj+1])/2 - buffer_l
                        pick = np.logical_and(np.logical_and(np.logical_and(i==ii, j < jj), coord[1,:] > 0.5), not_rogue)
                        xind0 = np.arange(self.networkSize)[pick]
                        xpick = np.hstack((axind, xind, xind0, xrind))
                        x_max = max(self.pos[0,xpick])
                        x_min = min(self.pos[0,xpick])
                        if x_max == x_min:
                            self.pos[0,xpick] = x1
                        else:
                            self.pos[0,xpick] = x1 + (self.pos[0,xpick] - x_max)/(x_max - x_min) * (x1 - x_min)
                        allpick = xpick
                    # pick right half column 
                    if yrind.size + yind.size  > 0:
                        y1 = (self.y[ii] + self.y[ii+1])/2 - buffer_l
                        pick = np.logical_and(np.logical_and(np.logical_and(j==jj, i < ii), coord[0,:] > 0.5), not_rogue)
                        yind0 = np.arange(self.networkSize)[pick]
                        ypick = np.hstack((ayind, yind, yind0, yrind))
                        y_max = max(self.pos[1,ypick])
                        y_min = min(self.pos[1,ypick])
                        if y_min == y_max:
                            self.pos[1,ypick] = y1
                        else:
                            self.pos[1,ypick] = y1 + (self.pos[1,ypick] - y_max)/(y_max - y_min) * (y1 - y_min)
                        if allpick.size == 0:
                            allpick = ypick
                        else:
                            allpick = np.hstack((allpick, ypick))
                    #update i,j,d2 of the half row and column picked above
                    i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,allpick])
                    i[allpick] = i_new
                    j[allpick] = j_new
                    d2[allpick] = d2_new
                    coord[:,allpick] = coord_new
                    continue
                # only bot-left corner
                if (bpattern - np.array([1,0,0,0]) == 0).all():
                    #print('obl')
                    gpick = np.logical_and(i==ii, j==jj)
                    opick = gpick
                    x_max = max(self.pos[0,gpick])
                    y_max = max(self.pos[1,gpick])
                    x1 = (self.x[jj] + self.x[jj+1])/2 - buffer_l
                    y1 = (self.y[ii] + self.y[ii+1])/2 - buffer_l

                    if x_xmax <= x1 and y_max <= y1:
                        continue
                    # squeeze y direction first
                    if y_max > y1:
                        ppick = np.logical_and(j == jj, not_rogue) 
                        y_min = min(self.pos[1,ppick])
                        if y_max == y_min:
                            self.pos[1,ppick] = y1
                        else:
                            self.pos[1,ppick] = y1 + (self.pos[1,ppick] - y_max)/(y_max - y_min) * (y1 - y_min)
                        opick = np.logical_or(opick, ppick)

                    # squeeze x direction
                    if x_max > x1:
                        ppick = np.logical_and(np.logical_and(i==ii, np.logical_or(j == jj, np.logical_and(j < jj, coord[1,:] < 0.5))), not_rogue)
                        x_min = min(self.pos[0,ppick])
                        if x_max == x_min:
                            self.pos[0,ppick] = x1
                        else:
                            self.pos[0,ppick] = x1 + (self.pos[0,ppick] - x_max)/(x_max - x_min) * (x1 - x_min)
                        opick = np.logical_or(opick, ppick)

                    i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,opick])
                    i[opick] = i_new
                    j[opick] = j_new
                    d2[opick] = d2_new
                    coord[:,opick] = coord_new
                    continue
                # only top-left corner
                if (bpattern - np.array([0,1,0,0]) == 0).all():
                    #print('otl')
                    gpick = np.logical_and(i==ii, j==jj)
                    opick = gpick
                    x_max = max(self.pos[0,gpick])
                    y_min = min(self.pos[1,gpick])
                    x1 = (self.x[jj] + self.x[jj+1])/2 - buffer_l
                    y0 = (self.y[ii] + self.y[ii+1])/2 + buffer_l
                    if y_min >= y0 and x_max <= x1:
                        continue
                    if y_min < y0:
                        ppick = np.logical_and(j == jj, not_rogue)
                        y_max = max(self.pos[1,ppick])
                        if y_max == y_min:
                            self.pos[1,ppick] = y0
                        else:
                            self.pos[1,ppick] = y0 + (self.pos[1,ppick] - y_min)/(y_max - y_min) * (y_max - y0)
                        opick = np.logical_or(opick, ppick)

                    if x_max > x1:
                        ppick = np.logical_and(np.logical_and(i==ii, np.logical_or(j == jj, np.logical_and(j < jj, coord[1,:] > 0.5))), not_rogue) # then x, so that more accommodated in x direction
                        x_min = min(self.pos[0,ppick])
                        if x_max == x_min:
                            self.pos[0,ppick] = x1
                        else:
                            self.pos[0,ppick] = x1 + (self.pos[0,ppick] - x_max)/(x_max - x_min) * (x1 - x_min)
                        opick = np.logical_or(opick, ppick)

                    i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,opick])
                    i[opick] = i_new
                    j[opick] = j_new
                    d2[opick] = d2_new
                    coord[:,opick] = coord_new
                    continue
                # only bot-right corner
                if (bpattern - np.array([0,0,1,0]) == 0).all():
                    #print('obr')
                    gpick = np.logical_and(i==ii, j==jj)
                    opick = gpick
                    y_max = max(self.pos[1,gpick])
                    x_min = min(self.pos[0,gpick])
                    x0 = (self.x[jj] + self.x[jj+1])/2 + buffer_l
                    y1 = (self.y[ii] + self.y[ii+1])/2 - buffer_l
                    if x_min >= x0 and y_max <= y1:
                        continue
                    if y_max > y1:
                        ppick = np.logical_and(j == jj, not_rogue)
                        y_min = min(self.pos[1,ppick])
                        if y_max == y_min:
                            self.pos[1,ppick] = y1
                        else:
                            self.pos[1,ppick] = y1 + (self.pos[1,ppick] - y_max)/(y_max - y_min) * (y1 - y_min)
                        opick = np.logical_or(opick, ppick)

                    if x_min < x0:
                        ppick = np.logical_and(np.logical_and(i==ii, np.logical_or(j == jj, np.logical_and(j > jj, coord[1,:] < 0.5))), not_rogue) # then x, so that more accommodated in x direction
                        x_max = max(self.pos[0,ppick])
                        if x_min == x_max:
                            self.pos[0,ppick] = x0
                        else:
                            self.pos[0,ppick] = x0 + (self.pos[0,ppick] - x_min)/(x_max - x_min) * (x_max - x0)
                        opick = np.logical_or(opick, ppick)

                    i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,opick])
                    i[opick] = i_new
                    j[opick] = j_new
                    d2[opick] = d2_new
                    coord[:,opick] = coord_new
                    continue
                # only top-right corner
                if (bpattern - np.array([0,0,0,1]) == 0).all():
                    #print('otr')
                    gpick = np.logical_and(i==ii, j==jj)
                    opick = gpick
                    x_min = min(self.pos[0,gpick])
                    y_min = min(self.pos[1,gpick])
                    x0 = (self.x[jj] + self.x[jj+1])/2 + buffer_l
                    y0 = (self.y[ii] + self.y[ii+1])/2 + buffer_l
                    if y_min >= y0 and x_min >= x0:
                        continue
                    if y_min < y0:
                        ppick = np.logical_and(j == jj, not_rogue)
                        y_max = max(self.pos[1,ppick])
                        if y_min == y_max:
                            self.pos[1,ppick] = y0
                        else:
                            self.pos[1,ppick] = y0 + (self.pos[1,ppick] - y_min)/(y_max - y_min) * (y_max - y0)
                        opick = np.logical_or(opick, ppick)
                    if x_min < x0:
                        ppick = np.logical_and(np.logical_and(i==ii, np.logical_or(j == jj, np.logical_and(j > jj, coord[1,:] > 0.5))), not_rogue) # then x, so that more accommodated in x direction
                        x_max = max(self.pos[0,ppick])
                        if x_min == x_max:
                            self.pos[0,ppick] = x0
                        else:
                            self.pos[0,ppick] = x0 + (self.pos[0,ppick] - x_min)/(x_max - x_min) * (x_max - x0)
                        opick = np.logical_or(opick, ppick)

                    i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,opick])
                    i[opick] = i_new
                    j[opick] = j_new
                    d2[opick] = d2_new
                    coord[:,opick] = coord_new
                    continue
            stdout.write(f'iter {it} done\n')
            it += 1
        print('adjusted.')
                        
        self.bound, self.btype = self.define_bound(self.Pi)

        if bfile is not None:
            with open(bfile + '.bin','wb') as f:
                nb = self.bound.shape[0]
                np.array([nb]).astype('i4').tofile(f)
                self.bound.copy().astype(float).tofile(f)

        self.adjusted = True

    # boundary define by 4 corners
    def define_bound(self, grid):
        ngrid = (self.nx-1) * (self.ny-1)
        bpGrid = np.empty((ngrid,2,3))
                            # low-left   # low-right    # top-right   # top-left
        bgrid = np.stack((grid[:-1,:-1], grid[:-1,1:], grid[1:,1:], grid[1:,:-1]), axis=2)
        assert(bgrid.shape[0] == self.ny-1 and bgrid.shape[1] == self.nx-1 and bgrid.shape[2] == 4)
        bgrid = bgrid.reshape((ngrid,4))
        xx0 = self.xx[:-1,:-1].reshape(ngrid)
        xx1 = self.xx[:-1,1:].reshape(ngrid)
        yy0 = self.yy[:-1,:-1].reshape(ngrid)
        yy1 = self.yy[1:,:-1].reshape(ngrid)

        # cross like boundaries
        bgrid_cross = np.zeros(bgrid.shape, dtype = int)
        bpGrid_cross = bpGrid.copy()

        unb = np.array([1,0,1,0], dtype = bool)
        unpick = (bgrid - unb == 0).all(-1)
        nunpick = np.sum(unpick)
        if nunpick > 0:
            indices = np.nonzero(unpick)[0]
            p, q = np.unravel_index(indices, (self.ny-1, self.nx-1))
            pick0 = self.Pi[p, q+1] == 0
            pick1 = self.Pi[p+1, q] == 0
            assert(np.logical_or(pick0, pick1).all())
            bgrid[indices, :] = np.array([1,0,0,0])
            bgrid_cross[indices, :] = np.array([0,0,1,0])

        unb = np.array([0,1,0,1], dtype = bool)
        unpick = (bgrid - unb == 0).all(-1)
        nunpick += np.sum(unpick)
        if nunpick > 0:
            indices = np.nonzero(unpick)[0]
            p, q = np.unravel_index(indices, (self.ny-1, self.nx-1))
            pick0 = self.Pi[p, q] == 0
            pick1 = self.Pi[p+1, q+1] == 0
            assert(np.logical_or(pick0, pick1).all())
            bgrid[indices, :] = np.array([0,1,0,0])
            bgrid_cross[indices, :] = np.array([0,0,0,1])

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
        def pick_boundary_in_turn(_bgrid_, bpattern, xx, yy, _bpGrid_ = bpGrid):
            pick = (_bgrid_ - bpattern == 0).all(-1)
            bpattern = np.logical_not(bpattern)
            pick = np.logical_or(pick, (_bgrid_ - bpattern == 0).all(-1))
            npick = np.sum(pick)
            if npick:
                _bpGrid_[pick,0,1:] = (xx0[pick] + xx1[pick]).reshape(npick,1)/2
                _bpGrid_[pick,1,:2] = (yy0[pick] + yy1[pick]).reshape(npick,1)/2
                _bpGrid_[pick,0,0] = xx[pick]
                _bpGrid_[pick,1,2] = yy[pick]
            return pick, npick

        # upper left
        ulpick, nulpick = pick_boundary_in_turn(bgrid, np.array([0,0,0,1], dtype=bool), xx0, yy1)
        btype2pick = ulpick
        assert(np.logical_and(ulpick,hpick).any() == False)
        assert(np.logical_and(ulpick,vpick).any() == False)
        # lower left
        llpick, nllpick = pick_boundary_in_turn(bgrid, np.array([1,0,0,0], dtype=bool), xx0, yy0)
        btype2pick = np.logical_or(btype2pick,llpick)
        assert(np.logical_and(llpick,hpick).any() == False)
        assert(np.logical_and(llpick,vpick).any() == False)
        assert(np.logical_and(llpick,ulpick).any() == False)
        # upper right 
        urpick, nurpick = pick_boundary_in_turn(bgrid, np.array([0,0,1,0], dtype = bool), xx1, yy1)
        btype2pick = np.logical_or(btype2pick,urpick)
        assert(np.logical_and(urpick,hpick).any() == False)
        assert(np.logical_and(urpick,vpick).any() == False)
        assert(np.logical_and(urpick,ulpick).any() == False)
        assert(np.logical_and(urpick,llpick).any() == False)
        # lower right 
        lrpick, nlrpick = pick_boundary_in_turn(bgrid, np.array([0,1,0,0], dtype = bool), xx1, yy0)
        btype2pick = np.logical_or(btype2pick,lrpick)
        assert(np.logical_and(lrpick,hpick).any() == False)
        assert(np.logical_and(lrpick,vpick).any() == False)
        assert(np.logical_and(lrpick,ulpick).any() == False)
        assert(np.logical_and(lrpick,llpick).any() == False)
        assert(np.logical_and(lrpick,urpick).any() == False)

        # get boundary points with crosses
        # upper left
        ulpick_cross, nulpick_cross = pick_boundary_in_turn(bgrid_cross, np.array([0,0,0,1], dtype=bool), xx0, yy1, _bpGrid_ = bpGrid_cross)
        btype2pick_cross = ulpick_cross
        # upper right 
        urpick_cross, nurpick_cross = pick_boundary_in_turn(bgrid_cross, np.array([0,0,1,0], dtype = bool), xx1, yy1, _bpGrid_ = bpGrid_cross)
        btype2pick_cross = np.logical_or(btype2pick_cross,urpick_cross)

        ## assemble
        nbtype2pick = np.sum([nulpick, nllpick, nurpick, nlrpick])
        nbtype2pick_cross = np.sum([nulpick_cross, nurpick_cross])
        if nbtype2pick_cross != nunpick:
            raise Exception(f'nbtype cross picked: {nbtype2pick_cross} != {nunpick}')
        nbtype = nhpick + nvpick + nbtype2pick + nunpick

        nout = np.sum((bgrid-np.array([0,0,0,0], dtype = bool) == 0).all(-1))
        nin = np.sum((bgrid-np.array([1,1,1,1], dtype = bool) == 0).all(-1))
        if nbtype != ngrid + nunpick - np.sum([nin, nout]):
            raise Exception(f'nbtype: {nbtype} != ngrid: {ngrid} - nin: {nin} - nout: {nout} + nunpick: {nunpick}')

        btype = np.empty(nbtype, dtype = int)
        btype[:nhpick] = 0
        btype[nhpick:(nhpick+nvpick)] = 1
        btype[(nhpick+nvpick):] = 2
        bp  = np.empty((nbtype,2,3))
        bp[:nhpick,:,:] = bpGrid[hpick,:,:] 
        bp[nhpick:(nhpick+nvpick),:,:] = bpGrid[vpick,:,:]
        bp[(nhpick+nvpick):(nhpick+nvpick+nbtype2pick),:,:] = bpGrid[btype2pick,:,:]
        bp[(nhpick+nvpick+nbtype2pick):,:,:] = bpGrid_cross[btype2pick_cross,:,:]
        return bp, btype

    #awaits vectorization
    def assign_pos_VF(self, straightFromPos = False):
        if not (hasattr(self, 'vpos') and self.vposLready and self.vposRready) and not straightFromPos:
            raise Exception('vpos is not ready')
        d0 = np.zeros(self.npolar-1)
        d1 = np.zeros(self.npolar-1)
        ix0 = np.zeros(self.npolar-1,dtype='int')
        if straightFromPos:
            vpos = self.pos.copy()
        else:
            vpos = self.vpos.copy()
        # top polar line excluded such that the coord of the nearest vertex will always be left and lower to the neuron's position, ready for interpolation
        for i in range(self.networkSize):
            pos = np.empty(2)
            pos[0] = vpos[0,i]
            pos[1] = vpos[1,i]
            # vx/vy(ecc,polar)
            mask = np.logical_and(pos[0] - self.vx[:-1,:-1] > 0, pos[0] - self.vx[1:,:-1] < 0)
            # find iso-polar lines whose xrange include x-coord of the neuron 
            pmask = mask.any(0)
            # find iso-polar lines whose xrange does not include x-coord of the neuron 
            pnull = np.logical_not(pmask)
            # find the preceding ecc index of the iso-polar lines that does includ the x-coord
            ix = np.nonzero(mask.T)[1]
            ix0[pmask] = ix
            ix0[pnull] = -1
            # calculate distance to the nearby VF grid vertex
            d0[pmask] = np.power(self.vx[ix,  np.arange(self.npolar-1)[pmask]] - pos[0],2) + np.power(self.vy[ix,  np.arange(self.npolar-1)[pmask]] - pos[1],2)
            d1[pmask] = np.power(self.vx[ix+1,np.arange(self.npolar-1)[pmask]] - pos[0],2) + np.power(self.vy[ix+1,np.arange(self.npolar-1)[pmask]] - pos[1],2)
            d0[pnull] = np.float('inf')
            d1[pnull] = np.float('inf')
            #find minimum distance to the nearest vertex
            dis = np.min(np.vstack([d0, d1]),0)
            # get the ecc and polar indices of the vertex
            idp = np.argmin(dis) # polar index
            idx = ix0[idp] # ecc index
            ## interpolate VF-ecc to pos
            vpos[0,i] = np.exp(np.log(self.e_range[idx]+1) + (np.log(self.e_range[idx+1]+1) - np.log(self.e_range[idx]+1)) * np.sqrt(dis[idp])/(np.sqrt(d0[idp])+np.sqrt(d1[idp])))-1
            ## interpolate VF-polar to pos
            vp_y0 = y_ep(vpos[0,i],self.p_range[idp],self.k,self.a,self.b)
            vp_x0 = x_ep(vpos[0,i],self.p_range[idp],self.k,self.a,self.b)
            vp_y1 = y_ep(vpos[0,i],self.p_range[idp+1],self.k,self.a,self.b)
            vp_x1 = x_ep(vpos[0,i],self.p_range[idp+1],self.k,self.a,self.b)
            dp0 = np.sqrt(np.power(pos[0]-vp_x0,2) + np.power(pos[1]-vp_y0,2))
            dp1 = np.sqrt(np.power(pos[0]-vp_x1,2) + np.power(pos[1]-vp_y1,2))
            vpos[1,i] = self.p_range[idp] + (self.p_range[idp+1] - self.p_range[idp]) * dp0/(dp0+dp1)
            #assert(vpos[1,i] >= self.p_range[0] and vpos[1,i] <= self.p_range[-1])
            stdout.write(f'\rassgining visual field: {(i+1)/self.networkSize*100:.3f}%')
        stdout.write('\n')
        vpos[1,:] = -vpos[1,:]
        print('dorsal-ventral flipped\n')
        self.pVFready = True
        return vpos

    def define_bound_LR(self):
        # right OD boundary 
        LR = self.LR.copy()
        LR[self.Pi < 1] = 0
        LR[self.LR == -1] = 0
        self.OD_boundR, self.btypeR = self.define_bound(LR)
        # left OD boundary 
        LR[self.LR == -1] = 1
        LR[self.LR == 1] = 0
        self.OD_boundL, self.btypeL = self.define_bound(LR)
        self.LR_boundary_defined = True

    def diffuse_bound(self, LR):
        print('diffuse the boudnary for spreading vpos')
        nLR = np.sum(LR > 0)
        tmpLR = LR[1:-1,1:-1]
        i, j = np.nonzero(tmpLR > 0)
        i = i + 1
        j = j + 1
        # spread out
        LR[i+1,j+1] = 1
        LR[i+1,j-1] = 1
        LR[i-1,j+1] = 1
        LR[i-1,j-1] = 1
        LR[i+1,j] = 1
        LR[i-1,j] = 1
        LR[i,j+1] = 1
        LR[i,j-1] = 1

        ngrid = (self.nx-1) * (self.ny-1)
        LRgrid = np.stack((LR[:-1,:-1], LR[:-1,1:], LR[1:,1:], LR[1:,:-1]), axis=2)
        LRgrid = LRgrid.reshape((ngrid,4))
        # pick corner-touching boundaries
        pick = (LRgrid - np.array([1,0,1,0]) == 0).all(-1)
        pick = np.logical_or(pick, (LRgrid - np.array([0,1,0,1]) == 0).all(-1))
        print(f'{sum(pick)} touches before')
        indices = np.nonzero(pick)[0]
        p, q = np.unravel_index(indices, (self.ny-1, self.nx-1))
        # merge
        LR[p  ,   q] = 1
        LR[p  , q+1] = 1 
        LR[p+1, q+1] = 1 
        LR[p+1,   q] = 1 
        # do not expand outside of cortex
        LR[self.Pi<=0] = 0

        spreaded = False 
        if np.sum(LR > 0) == np.sum(self.Pi > 0):
            spreaded = True
            print('spread finished')
        else:
            assert(nLR < np.sum(LR > 0))
        return LR, spreaded
         
    def check_pos(self, checkL = True, checkR = True):
        # check in boundary
        i, j, d2 = self.get_ij_grid(get_d2 = True, get_coord = False)
        ipick = np.argmin(d2,1)
        # in the shell
        if not (self.Pi[i,    j][ipick == 0] > 0).all():
            raise Exception(f'{np.sum(self.Pi[i, j][ipick == 0] == 0)} bot-left leak')

        if not (self.Pi[i+1,  j][ipick == 1] > 0).all():
            raise Exception(f'{np.sum(self.Pi[i+1, j][ipick == 1] == 0)} top-left leak')

        if not (self.Pi[i,  j+1][ipick == 2] > 0).all():
            raise Exception(f'{np.sum(self.Pi[i, j+1][ipick == 2] == 0)} bot-right leak')

        if not (self.Pi[i+1,j+1][ipick == 3] > 0).all():
            raise Exception(f'{np.sum(self.Pi[i+1,j+1][ipick == 3] == 0)} top-right leak')

        if checkL:
            # in the L stripe
            pL = self.ODlabel < 0
            assert((self.LR[i,    j][np.logical_and(ipick == 0, pL)] == -1).all())
            assert((self.LR[i+1,  j][np.logical_and(ipick == 1, pL)] == -1).all())
            assert((self.LR[i,  j+1][np.logical_and(ipick == 2, pL)] == -1).all())
            assert((self.LR[i+1,j+1][np.logical_and(ipick == 3, pL)] == -1).all())
            print('left boundary checked')

        if checkR:
            # in the R stripe
            pR = self.ODlabel > 0
            assert((self.LR[i,    j][np.logical_and(ipick == 0, pR)] == 1).all())
            assert((self.LR[i+1,  j][np.logical_and(ipick == 1, pR)] == 1).all())
            assert((self.LR[i,  j+1][np.logical_and(ipick == 2, pR)] == 1).all())
            assert((self.LR[i+1,j+1][np.logical_and(ipick == 3, pR)] == 1).all())
            print('right boundary checked')

    def assign_pos_OD2(self, bfile = None, force = False):
        i, j, d2 = self.get_ij_grid(get_d2 = True, get_coord = False)
        if force:
            d2[self.Pi[i,j]<=0,    0] = np.float('inf') #bot_left
            d2[self.Pi[i+1,j]<=0,  1] = np.float('inf') #top_left
            d2[self.Pi[i,j+1]<=0,  2] = np.float('inf') #bot_right
            d2[self.Pi[i+1,j+1]<=0,3] = np.float('inf') #top_right
        ipick = np.argmin(d2,1)

        if not self.noAdjust:
            def get_22ij(index):
                ip = np.mod(index,2)
                jp = index//2
                return ip, jp
            ip, jp = get_22ij(ipick) 
            self.opick = self.Pi[i + ip, j + jp] == 0
            if np.sum(self.opick) > 0:
                print(f'{np.sum(self.opick)} is not inside the ragged boundary')
                #raise Exception(f'{np.sum(self.opick)} is not inside the ragged boundary')


        print('assign ocular dominance preference to neuron according to their position in the cortex')
        self.ODlabel = np.zeros((self.networkSize), dtype = int)
        self.ODlabel[ipick == 0] = self.LR[i,    j][ipick == 0]
        self.ODlabel[ipick == 1] = self.LR[i+1,  j][ipick == 1]
        self.ODlabel[ipick == 2] = self.LR[i,  j+1][ipick == 2]
        self.ODlabel[ipick == 3] = self.LR[i+1,j+1][ipick == 3]
        if not self.noAdjust:
            print(f'{np.sum(self.ODlabel > 0)}-L + {np.sum(self.ODlabel < 0)}-R == {self.networkSize}-all')
            #raise Exception(f'{np.sum(self.ODlabel > 0)}L + {np.sum(self.ODlabel < 0)}R != all{self.networkSize}')

        self.pODready = True
        if self.LR_boundary_defined is False:
            self.define_bound_LR()
            if bfile is not None:
                with open(bfile + 'R.bin','wb') as f:
                    nb = self.OD_boundR.shape[0]
                    np.array([nb]).astype('i4').tofile(f)
                    self.OD_boundR.copy().astype(float).tofile(f)

                with open(bfile + 'L.bin','wb') as f:
                    nb = self.OD_boundL.shape[0]
                    np.array([nb]).astype('i4').tofile(f)
                    self.OD_boundL.copy().astype(float).tofile(f)

        return self.ODlabel

    def make_pos_uniform_p(self, dt, p_scale, b_scale, figname, ncore = 0, ndt_decay = 0, roi_ratio = 2.0, k1 = 1.0, k2 = 0.5, damp = 0.5, l_ratio = 0.0, l_k = 1.0, chop_ratio = 0, spercent = 0.01, seed = -1, bfile = None, local_plot = False, check = False):
        if not self.adjusted:
            self.adjust_pos(bfile = bfile)

        if ncore == 0:
            ncore = mp.cpu_count()
            print(f'{ncore} cores found')

        if chop_ratio == 0:
            chop_ratio = self.nblock/(self.nx*self.ny)
        nx = int(np.ceil(self.nx*chop_ratio))
        ny = int(np.ceil(self.ny*chop_ratio))
        if nx*ny < ncore:
            nx = int(np.ceil(np.sqrt(ncore) * self.nx/self.ny))
            ny = int(np.ceil(np.sqrt(ncore) * self.ny/self.nx))
        #nx = 2
        #ny = 2

        oldpos = self.pos.copy()

    ##  prepare shared memory
        raw_shared_cmem = RawArray(c_int32, np.zeros(ncore, dtype = 'i4'))
        shared_cmem = np.frombuffer(raw_shared_cmem, dtype = 'i4')

        raw_shared_dmem = RawArray(c_double, np.zeros(self.networkSize*8, dtype = 'f8'))
        raw_shared_imem = RawArray(c_int32, np.zeros(self.networkSize, dtype = 'i4'))
        shared_dmem = np.frombuffer(raw_shared_dmem, dtype = 'f8')
        shared_imem = np.frombuffer(raw_shared_imem, dtype = 'i4')

        # populate position in shared array and assign chop references
        pos = np.empty(self.nblock, dtype = object)
        for i in range(self.nblock):
            i0 = 2*i*self.blockSize
            i1 = i0 + 2*self.blockSize
            shared_dmem[i0:i1] = self.pos[:,i*self.blockSize:(i+1)*self.blockSize].reshape(2*self.blockSize).copy()
            pos[i] = shared_dmem[i0:i1].view().reshape((2,self.blockSize))
            i0 = i1

        raw_shared_bmem = RawArray(c_double, np.zeros(nx*ny*6, dtype = 'f8'))
        shared_bmem = np.frombuffer(raw_shared_bmem, dtype = 'f8') 
        raw_shared_nmem = RawArray(c_int32, np.zeros(nx*ny*6, dtype = 'i4')) # for neighbor_list and id
        shared_nmem = np.frombuffer(raw_shared_nmem, dtype = 'i4') 

        self.pos = parallel_repel(self.area, self.subgrid, pos, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, p_scale, self.bound, self.btype, b_scale, nx, ny, ncore, dt, spercent, figname, ndt_decay, 1.0, roi_ratio, k1, k2, seed, 1.0, damp, l_ratio, l_k, local_plot)
        if check:
            self.check_pos()

        self.posUniform = True
        
        self.assign_pos_OD2(bfile)
        return oldpos

    def spread_pos_VF(self, dt, vpfile, lrfile, LRlabel, seed = None, firstTime = True, read_lrfile = None, read_vpfile = None, continuous = True, particle_param = None, boundary_param = None, ax = None, p_scale = 2.0, b_scale = 1.0, ndt_decay = 0, roi_ratio = 2, k1 = 1.0, k2 = 0.5, ncore= 0, limit_ratio = 1.0, damp = 0.5, l_ratio = 0.5, l_k = 0.0, chop_ratio = 0, noSpread = False):
        if ncore == 0:
            ncore = mp.cpu_count()
            print(f'{ncore} cores found')
        if chop_ratio == 0:
            chop_ratio = self.nblock/(self.nx*self.ny)
        nx = int(np.ceil(self.nx*chop_ratio))
        ny = int(np.ceil(self.ny*chop_ratio))
        if nx*ny < ncore:
            nx = int(np.ceil(np.sqrt(ncore) * self.nx/self.ny))
            ny = int(np.ceil(np.sqrt(ncore) * self.ny/self.nx))

        oldpos = self.pos.copy()

        ngrid = np.sum(self.Pi).astype(int)
        if LRlabel == 'L':
            LRpick = self.ODlabel < 0
        else:
            assert(LRlabel == 'R')
            LRpick = self.ODlabel > 0
        nLR = np.sum(LRpick)
        print('#' +LRlabel+f': {nLR}')
        pickLR = LRpick.view().reshape((self.nblock,self.blockSize))

    ##  prepare shared memory
        raw_shared_cmem = RawArray(c_int32, np.zeros(ncore, dtype = 'i4'))
        shared_cmem = np.frombuffer(raw_shared_cmem, dtype = 'i4')
        raw_shared_dmem = RawArray(c_double, np.zeros(nLR*8, dtype = 'f8'))
        raw_shared_imem = RawArray(c_int32, np.zeros(nLR, dtype = 'i4'))
        shared_dmem = np.frombuffer(raw_shared_dmem, dtype = 'f8')
        shared_imem = np.frombuffer(raw_shared_imem, dtype = 'i4')

        if not hasattr(self,'vpos'):
            self.vpos = self.pos.copy() 
        if firstTime:
            LR = self.LR.copy()
            if LRlabel == 'L':
                LR[LR > 0] = 0
                LR[LR < 0] = 1
            else:
                LR[LR < 0] = 0
                LR[LR > 0] = 1
           # populate position in shared array
            #vposLR = np.empty(self.nblock, dtype = object)
            #vpos = self.pos.view().reshape((2,self.nblock,self.blockSize))
            chop_size = np.zeros(self.nblock, dtype = 'i4')
            k = 0
            for i in range(self.nblock):
                npick = np.sum(pickLR[i,:])
                if npick > 0:
                    chop_size[k] = npick
                    #vposLR[k] = vpos[:,i,pickLR[i,:]]
                    k = k + 1
            nchop0 = k
            #vposLR = vposLR[:k]
            chop_size0 = chop_size[:k].copy()
            
            #shared_dmem[:2*nLR] = np.hstack([p.flatten() for p in vposLR])
            vposLR = self.pos[:, LRpick].copy()
        else:
            if read_lrfile is None:
                raise Exception('lrfile not provided')
            if read_vpfile is None:
                raise Exception('vpfile not provided')

            with open(read_lrfile+'.bin','rb') as f:
                consts = np.fromfile(f, 'i4', 2)
                LR = np.fromfile(f, 'i4', count = np.prod(self.Pi.shape)).reshape(self.Pi.shape)

            with open(read_vpfile+'.bin','rb') as f:
                nchop0 = np.fromfile(f, 'i4', 1)[0]
                chop_size0 = np.fromfile(f, 'i4', nchop0)
                assert(sum(chop_size0) == nLR)
                vposLR = np.fromfile(f, 'f8').reshape(2, nLR)
        old_vpos = vposLR.copy()
        assert(np.sum(chop_size0) == nLR)
        raw_shared_bmem = RawArray(c_double, np.zeros(nx*ny*6, dtype = 'f8'))
        shared_bmem = np.frombuffer(raw_shared_bmem, dtype = 'f8') 
        raw_shared_nmem = RawArray(c_int32, np.zeros(nx*ny*6, dtype = 'i4')) # for neighbor_list and id
        shared_nmem = np.frombuffer(raw_shared_nmem, dtype = 'i4') 

        if ax is not None:
            if seed is None:
                seed = np.int64(time.time())
            np.random.seed(seed)
            nsLR = np.int(nLR*0.01)
            print(f'sample size: {nsLR}')
            pick = np.random.choice(nLR, nsLR, replace = False)
            #pick = np.arange(nLR)
        else:
            pick = None

        if continuous:
            spreaded = False # at least one time
            ip = 0 # plotting index
            ppos = None
            starting = True
            while spreaded is False:
                vposLR, LR, spreaded, ppos, ip, OD_bound = self.spread(dt, vposLR, nchop0, chop_size0, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, LR, nx, ny, lrfile, particle_param, boundary_param, ax, pick, ppos, ip, starting, figname = vpfile, p_scale = p_scale, b_scale = b_scale, ndt_decay = ndt_decay, roi_ratio = roi_ratio, k1 = k1, k2 = k2, ncore = ncore, limit_ratio = limit_ratio, damp = damp, l_ratio = l_ratio, l_k = l_k, chop_ratio = 0)
                starting = False
                print(f'#{ip}: {np.sum(LR).astype(int)}/{ngrid}')

                assert(vposLR.shape[1] == nLR)
                with open(vpfile + f'-{ip-1}.bin','wb') as f:
                    np.array([nchop0]).astype('i4').tofile(f)
                    chop_size0.astype('i4').tofile(f)
                    vposLR.tofile(f)
                print('data stored')
        else:
            ip = 0 # plotting index
            ppos = None
            starting = True
            vposLR, LR, spreaded, ppos, _, OD_bound = self.spread(dt, vposLR, nchop0, chop_size0, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, LR, nx, ny, lrfile, particle_param, boundary_param, ax, pick, ppos, ip, starting, figname = vpfile, p_scale = p_scale, b_scale = b_scale, ndt_decay = ndt_decay, roi_ratio = roi_ratio, k1 = k1, k2 = k2, ncore = ncore, limit_ratio = limit_ratio, damp = damp, l_ratio = l_ratio, l_k = l_k, chop_ratio = 0, seed = seed, plotOnly = False, noSpread = noSpread)
            print(f'spreaded = {spreaded}')
            assert(vposLR.shape[1] == nLR)
            with open(vpfile+'-ss.bin','wb') as f:
                np.array([nchop0]).astype('i4').tofile(f)
                chop_size0.astype('i4').tofile(f)
                vposLR.tofile(f)
            print('data stored')

        self.vpos[:, LRpick] = vposLR # when spreaded, return in shape (2,nLR)

        if ax is not None:
            # connecting points on grid sides
            ax.plot(OD_bound[:,0,[0,2]].squeeze(), OD_bound[:,1,[0,2]].squeeze(), ',r')
            # grid centers
            ax.plot(OD_bound[:,0,1].squeeze(), OD_bound[:,1,1].squeeze(), ',g')
            #if not continuous:
                # plot all displacement
            #ax.plot(np.vstack((self.pos[0,LRpick], old_vpos[0,:])), np.vstack((self.pos[1,LRpick], old_vpos[1,:])),'-,c', lw =0.01, alpha = 0.7)
            ax.plot(np.vstack((old_vpos[0,:], self.vpos[0,LRpick])), np.vstack((old_vpos[1,:], self.vpos[1,LRpick])),'-,m', lw =0.01, alpha = 0.7)
            # final positions
            ax.plot(self.vpos[0,LRpick], self.vpos[1,LRpick], ',k')

        # check only the relevant boundary
        if LRlabel == 'L':
            self.check_pos(True, False) 
        else:
            self.check_pos(False, True)

        if LRlabel == 'L':
            self.vposLready = True
        else:
            self.vposRready = True

    def spread(self, dt, vpos_flat, nchop, chop_size, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, LR, nx, ny, lrfile, particle_param = None, boundary_param = None, ax = None, pick = None, ppos = None, ip = 0, starting = True, figname = None, p_scale = 2.0, b_scale = 1.0, ndt_decay = 0, roi_ratio = 2, k1 = 1.0, k2 = 0.5, ncore = 0, limit_ratio = 1.0, damp = 0.5, l_ratio = 0.5, l_k = 0.0, chop_ratio = 0, seed = -1, plotOnly = False, noSpread = False):
        print(f'ax is {ax}')
        if starting and ax is not None:
            OD_bound, btype = self.define_bound(LR)
            # connecting points on grid sides
            ax.plot(OD_bound[:,0,[0,2]].squeeze(), OD_bound[:,1,[0,2]].squeeze(), ',r')
            # grid centers
            ax.plot(OD_bound[:,0,1].squeeze(), OD_bound[:,1,1].squeeze(), ',g')
        subarea = self.subgrid[0] * self.subgrid[1]

        old_area = subarea * np.sum(LR == 1)
        if not noSpread:
            LR, spreaded = self.diffuse_bound(LR)
        else:
            spreaded = False
        area = subarea * np.sum(LR == 1)
        ratio = area/old_area
        print(f'area increases {ratio*100:.3f}%')
        
        OD_bound, btype = self.define_bound(LR)

        with open(lrfile + f'-{ip}.bin','wb') as f:
            np.array([LR.size, OD_bound.shape[0]]).astype('i4').tofile(f)
            LR.astype('i4').tofile(f)
            OD_bound.copy().astype(float).tofile(f)

        if ax is not None:
            assert(pick is not None)
            if ppos is None:
                ppos = np.empty((2,2,pick.size)) # start/ending, x/y, selected id
                ppos[0,:,:] = vpos_flat[:,pick].copy()

        if not plotOnly:
            # input vpos as view of chops # return as (2, n) in the original order
            vpos_chop = np.empty(nchop, dtype = object)
            i0 = 0
            ii0 = 0
            for i in range(nchop):
                i1 = i0 + chop_size[i]
                ii1 = ii0 + chop_size[i]*2
                shared_dmem[ii0:ii1] = vpos_flat[:,i0:i1].flatten()
                vpos_chop[i] = shared_dmem[ii0:ii1].view().reshape((2,chop_size[i]))
                i0 = i1
                ii0 = ii1
        #
            vpos_flat = parallel_repel(area, self.subgrid, vpos_chop, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, p_scale, OD_bound, btype, b_scale, nx, ny, ncore, dt, 0, figname, ndt_decay, 1/np.sqrt(ratio), roi_ratio, k1 = k1, k2 = k2, seed = seed, limit_ratio = limit_ratio, damp = damp, l_ratio = l_ratio, l_k = l_k, local_plot = False)

        ip = ip + 1
        if ax is not None:
            ppos[ip%2,:,:] = vpos_flat[:,pick].copy()
            ax.plot(ppos[:,0,:].squeeze(), ppos[:,1,:].squeeze(), '-c', lw = 0.01)
            ax.plot(ppos[0,0,:], ppos[0,1,:], ',c')

        return vpos_flat, LR, spreaded, ppos, ip, OD_bound

    def get_ij_grid(self, get_d2 = True, get_coord = True):
        print('get the index of the nearest vertex for each neuron in its own grid')

        x0 = self.x[:-1]
        x1 = self.x[1:]
        y0 = self.y[:-1]
        y1 = self.y[1:]
        # index in y-dimension
        ipick = np.logical_and(np.tile(self.pos[1,:],(self.ny-1,1))-y0.reshape(self.ny-1,1) > 0, np.tile(self.pos[1,:],(self.ny-1,1))-y1.reshape(self.ny-1,1) <= 0).T
        i = np.zeros(self.networkSize, dtype = 'i4')
        pick = ipick.any(-1)
        i[pick] = np.nonzero(ipick)[1]
        if not np.sum(pick) == self.networkSize:
            nrogue = self.networkSize - np.sum(pick)
            print(f'{nrogue} particles out of {self.pos.shape[1]} have gone rogue')
            if not np.sum(self.pos[1,:] == self.y[0]) == nrogue:
                print(f'range: {np.max(pos[1,:])} > {self.y[-1]}, {np.min(pos[1,:])} <= {self.y[0]}')
                raise Exception(f'cant restrain rogue particles')
        print('indexed in y-dimension')

        # index in x-dimension
        jpick = np.logical_and(np.tile(self.pos[0,:],(self.nx-1,1))-x0.reshape(self.nx-1,1) > 0, np.tile(self.pos[0,:],(self.nx-1,1))-x1.reshape(self.nx-1,1) <= 0).T
        j = np.zeros(self.networkSize, dtype = 'i4')
        pick = jpick.any(-1)
        j[pick] = np.nonzero(jpick)[1]
        if not np.sum(pick) == self.networkSize:
            nrogue = self.networkSize - np.sum(pick)
            print(f'{nrogue} particles out of {self.pos.shape[1]} have gone rogue')
            if not np.sum(self.pos[0,:] == self.x[0]) == nrogue:
                print(f'range: {np.max(pos[0,:])} > {self.x[-1]}, {np.min(pos[0,:])} <= {self.x[0]}')
                raise Exception(f'cant restrain rogue particles')
        print('indexed in x-dimension')

        
        if not self.noAdjust:
            outcast = np.stack((self.Pi[i,j]<=0, self.Pi[i+1,j]<=0, self.Pi[i,j+1]<=0, self.Pi[i+1,j+1]<=0), axis = -1).all(-1)
            nout = np.sum(outcast)
            if nout > np.sqrt(self.nblock):
                raise Exception(f'too much outcasts: {nout}, something is wrong')
            else:
                while nout > 0:
                    indices = np.nonzero(outcast)[0]
                    for iout in indices:
                        iblock = iout//self.blockSize
                        mean_bpos = np.mean(self.pos[:,iblock*self.blockSize:(iblock+1)*self.blockSize], axis =-1)
                        iout_pos = self.pos[:,iout].copy()
                        self.pos[:,iout] = iout_pos + (mean_bpos - iout_pos)*np.random.rand()
                        
                    j[indices] = np.nonzero(np.logical_and(np.tile(self.pos[0,indices],(self.nx-1,1))-x0.reshape(self.nx-1,1) > 0, np.tile(self.pos[0,indices],(self.nx-1,1))-x1.reshape(self.nx-1,1) <= 0).T)[1]
                    i[indices] = np.nonzero(np.logical_and(np.tile(self.pos[1,indices],(self.ny-1,1))-y0.reshape(self.ny-1,1) > 0, np.tile(self.pos[1,indices],(self.ny-1,1))-y1.reshape(self.ny-1,1) <= 0).T)[1]

                    outcast = np.stack((self.Pi[i,j]<=0, self.Pi[i+1,j]<=0, self.Pi[i,j+1]<=0, self.Pi[i+1,j+1]<=0), axis = -1).all(-1)
                    old_nout = nout
                    nout = np.sum(outcast)
                    print(f'{old_nout - nout} outcasts corrected')
                print('no outcasts')


        if not get_d2 and not get_coord:
            return i, j
        else:
            if get_d2:
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
                print('calculate neurons\' cortical distance to the nearest vertices in the grid')
                d2 = np.zeros((4,self.networkSize))
                for ic in range(4):
                    dx = self.pos[0,:] - corner_x[i,j,ic]
                    dy = self.pos[1,:] - corner_y[i,j,ic]
                    d2[ic,:] = np.power(dx,2) + np.power(dy,2)

            if get_coord:
                corner_x = np.zeros((self.ny-1,self.nx-1,2))
                corner_x[:,:,0] = self.xx[:-1,:-1]
                corner_x[:,:,1] = self.xx[:-1,1:]
                corner_y = np.zeros((self.ny-1,self.nx-1,2))
                corner_y[:,:,0] = self.yy[:-1,:-1]
                corner_y[:,:,1] = self.yy[1:,:-1]
                coord = np.empty((2,self.networkSize))
                print('calculate neurons\' normalized coordinate in its grid')
                coord[0,:] = (self.pos[0,:] - corner_x[i,j,0])/(corner_x[i,j,1] - corner_x[i,j,0])
                coord[1,:] = (self.pos[1,:] - corner_y[i,j,0])/(corner_y[i,j,1] - corner_y[i,j,0])

            if get_coord and get_d2:
                return i, j, d2.T, coord
            elif get_coord:
                return i, j, coord
            else:
                return i, j, d2.T

    def get_ij_grid_local(self, pos, get_d2 = True, get_coord = True):
        x0 = self.x[:-1]
        x1 = self.x[1:]
        y0 = self.y[:-1]
        y1 = self.y[1:]
        # index in y-dimension
        ipick = np.logical_and(np.tile(pos[1,:],(self.ny-1,1))-y0.reshape(self.ny-1,1) > 0, np.tile(pos[1,:],(self.ny-1,1))-y1.reshape(self.ny-1,1) <= 0).T
        i = np.zeros(pos.shape[1], dtype = 'i4')
        pick = ipick.any(-1)
        i[pick] = np.nonzero(ipick)[1]
        if not np.sum(pick) == pos.shape[1]:
            nrogue = pos.size - np.sum(pick)
            print(f'rogue: {nrogue} == {np.sum(np.logical_or(pos[1,:] > self.y[-1], pos[1,:] <= self.y[0]))}')
            print(f'range: {np.max(pos[1,:])} > {self.y[-1]}, {np.min(pos[1,:])} <= {self.y[0]}')
            if not np.sum(pos[1,:] == self.y[0]) == nrogue:
                raise Exception(f'{nrogue} particles out of control')

        #print('indexed in y-dimension')
        # index in x-dimension
        jpick = np.logical_and(np.tile(pos[0,:],(self.nx-1,1))-x0.reshape(self.nx-1,1) > 0, np.tile(pos[0,:],(self.nx-1,1))-x1.reshape(self.nx-1,1) <= 0).T
        j = np.zeros(pos.shape[1], dtype = 'i4')
        pick = jpick.any(-1)
        j[pick] = np.nonzero(jpick)[1]
        if not j.size == pos.shape[1]:
            nrogue = pos.shape[1] - j.size
            print(f'rogue: {nrogue} == {np.sum(np.logical_or(pos[0,:] > self.x[-1], pos[0,:] <= self.x[0]))}')
            print(f'range: {np.max(pos[0,:])} > {self.x[-1]}, {np.min(pos[0,:])} <= {self.x[0]}')
            if not np.sum(pos[0,:] == self.x[0]) == nrogue:
                raise Exception(f'{nrogue} particles out of control')
        #print('indexed in x-dimension')

        if not self.noAdjust: 
            outcast = np.stack((self.Pi[i,j]<=0, self.Pi[i+1,j]<=0, self.Pi[i,j+1]<=0, self.Pi[i+1,j+1]<=0), axis = -1).all(-1)
            nout = np.sum(outcast)
            if nout > np.sqrt(self.nblock):
                raise Exception('too much outcasts: {nout}, something is wrong')
            else:
                while nout > 0:
                    indices = np.nonzero(outcast)[0]
                    for iout in indices:
                        iblock = iout//self.blockSize
                        mean_bpos = np.mean(pos[:, i == i[iout]], axis =-1)
                        iout_pos = pos[:,iout].copy()
                        pos[:,iout] = iout_pos + (mean_bpos - iout_pos)*np.random.rand()
                        
                    j[indices] = np.nonzero(np.logical_and(np.tile(pos[0,indices],(self.nx-1,1))-x0.reshape(self.nx-1,1) > 0, np.tile(pos[0,indices],(self.nx-1,1))-x1.reshape(self.nx-1,1) <= 0).T)[1]
                    i[indices] = np.nonzero(np.logical_and(np.tile(pos[1,indices],(self.ny-1,1))-y0.reshape(self.ny-1,1) > 0, np.tile(pos[1,indices],(self.ny-1,1))-y1.reshape(self.ny-1,1) <= 0).T)[1]

                    outcast = np.stack((self.Pi[i,j]<=0, self.Pi[i+1,j]<=0, self.Pi[i,j+1]<=0, self.Pi[i+1,j+1]<=0), axis = -1).all(-1)
                    old_nout = nout
                    nout = np.sum(outcast)
                    print(f'{old_nout - nout} outcasts corrected')


        if not get_d2 and not get_coord:
            return i, j
        else:
            if get_d2:
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
                #print('calculate neurons\' cortical distance to the nearest vertices in the grid')
                d2 = np.zeros((4,pos.shape[1]))
                for ic in range(4):
                    dx = pos[0,:] - corner_x[i,j,ic]
                    dy = pos[1,:] - corner_y[i,j,ic]
                    d2[ic,:] = np.power(dx,2) + np.power(dy,2)

            if get_coord:
                corner_x = np.zeros((self.ny-1,self.nx-1,2))
                corner_x[:,:,0] = self.xx[:-1,:-1]
                corner_x[:,:,1] = self.xx[:-1,1:]
                corner_y = np.zeros((self.ny-1,self.nx-1,2))
                corner_y[:,:,0] = self.yy[:-1,:-1]
                corner_y[:,:,1] = self.yy[1:,:-1]
                coord = np.empty((2,pos.shape[1]))
                #print('calculate neurons\' normalized coordinate in its grid')
                coord[0,:] = (pos[0,:] - corner_x[i,j,0])/(corner_x[i,j,1] - corner_x[i,j,0])
                coord[1,:] = (pos[1,:] - corner_y[i,j,0])/(corner_y[i,j,1] - corner_y[i,j,0])
            if not ((coord <= 1).all() and (coord >= 0).all()):
                #print('xmin')
                #pick = coord[0,:]<0
                #print(pos[0,pick])
                #print(corner_x[i[pick], j[pick],0])
                #print(corner_x[i[pick], j[pick],1])
                #print('xmax')
                #pick = coord[0,:]>1
                #print(pos[0,pick])
                #print(corner_x[i[pick], j[pick],0])
                #print(corner_x[i[pick], j[pick],1])
                #print('ymin')
                #pick = coord[1,:]<0
                #print(pos[1,pick])
                #print(corner_y[i[pick], j[pick],0])
                #print(corner_y[i[pick], j[pick],1])
                #print('ymax')
                #pick = coord[1,:]>1
                #print(pos[1,pick])
                #print(corner_y[i[pick], j[pick],0])
                #print(corner_y[i[pick], j[pick],1])
                raise Exception('outside of grid')

            if get_coord and get_d2:
                return i, j, d2.T, coord
            elif get_coord:
                return i, j, coord
            else:
                return i, j, d2.T

    def plot_map(self,ax1,ax2 = None, ax3 = None, dpi = 2000,pltOD=True,pltVF=False,pltOP=True,forceOP=False,ngridLine=6):
        if pltOD:
            if not self.pODready:
                #self.assign_pos_OD0()
                self.assign_pos_OD2()
                
            if pltOP:
                if not self.pOPready:
                    if self.assign_pos_OP(forceOP) is None:
                        print('Orientation Preference is not plotted')
            if pltOP and self.pOPready:
                s = np.power(dpi,2)/self.networkSize/100
                print(f'marker size = {s}')
                hsv = cm.get_cmap('hsv') # theta to ratio
                pick = self.ODlabel>0 
                if np.sum(pick) > 0:
                    ax1.scatter(self.pos[0,pick], self.pos[1,pick], s = s, linewidths=0.0, marker = '.', c = hsv(self.op[pick]))
                else:
                    print('no contra-lateral neuron')
                pick = self.ODlabel<0
                if np.sum(pick) > 0:
                    ax1.scatter(self.pos[0,pick], self.pos[1,pick], s = s, linewidths=0.0, marker = '.', c = hsv(self.op[pick]), alpha = 0.45)
                else:
                    print('no ipsi-lateral neuron')

                pick = self.Pi > 0
                ax1.plot(self.xx[pick], self.yy[pick], ',k')
                pick = self.Pi <= 0
                ax1.plot(self.xx[pick], self.yy[pick], ',', color = 'gray')

                ax1.plot([self.oldpos[0,:], self.pos[0,:]], [self.oldpos[1,:],self.pos[1,:]], '-k', lw = 0.01)
                ax1.plot(self.pos[0,:], self.pos[1,:], ',', color = 'blue')
                ax1.plot(self.oldpos[0,:], self.oldpos[1,:], ',', color = 'darkblue')
                pick = self.opick
                if pick is not None:
                    ax1.plot(self.pos[0,pick], self.pos[1,pick], ',', color = 'red')
                    ax1.plot(self.oldpos[0,pick], self.oldpos[1,pick], ',', color = 'darkred')
            else:
                #ax1.plot(self.pos[0,:], self.pos[1,:],',k')
                #ax1.plot(self.xx[self.Pi<=0], self.yy[self.Pi<=0],',r')
                #ax1.plot(self.xx[self.Pi>0], self.yy[self.Pi>0],',g')
                ax1.plot(self.pos[0,self.ODlabel>0], self.pos[1,self.ODlabel>0],',m')
                ax1.plot(self.pos[0,self.ODlabel<0], self.pos[1,self.ODlabel<0],',c')
            ax1.set_aspect('equal')
            if ngridLine > 0:
                for ip in range(self.npolar):
                    if ip % (self.npolar//ngridLine)== 0:
                        ax1.plot(self.vx[:,ip], self.vy[:,ip],':',c='0.5', lw = 0.1)
                for ie in range(self.necc):
                    if ie % (self.necc//ngridLine)== 0:
                        ax1.plot(self.vx[ie,:], self.vy[ie,:],':',c='0.5', lw = 0.1)

        if pltVF == True:
            if self.pVFready == False:
                raise Exception('get VF ready first: 1) make_pos_uniform for L and R. 2) spread_pos_VF for L and R. 3) assign_pos_VF')
            plt.sca(ax2)
            plt.polar(self.vpos[1,self.ODlabel>0], self.vpos[0,self.ODlabel>0],',m')
            if ngridLine > 0:
                for ip in range(self.npolar):
                    if ip % (self.npolar//ngridLine)== 0:
                        plt.polar(self.p_range[ip]+np.zeros(self.necc), self.e_range,':',c='0.5', lw = 0.1)
                for ie in range(self.necc):
                    if ie % (self.necc//ngridLine)== 0:
                        plt.polar(self.p_range, self.e_range[ie]+np.zeros(self.npolar),':',c='0.5', lw = 0.1)

            plt.sca(ax3)
            plt.polar(self.vpos[1,self.ODlabel<0], self.vpos[0,self.ODlabel<0],',m')
            if ngridLine > 0:
                for ip in range(self.npolar):
                    if ip % (self.npolar//ngridLine)== 0:
                        plt.polar(self.p_range[ip]+np.zeros(self.necc), self.e_range,':',c='0.5', lw = 0.1)
                for ie in range(self.necc):
                    if ie % (self.necc//ngridLine)== 0:
                        plt.polar(self.p_range, self.e_range[ie]+np.zeros(self.npolar),':',c='0.5', lw = 0.1)

    def save(self, pos_file = None, OD_file = None, OP_file = None, VFxy_file = None, VFpolar_file = None, Feature_file = None, Parallel_uniform_file = None, allpos_file = None, dataDim = 2, Parallel_spreadVF_file = None, fp = 'f4'):
        if pos_file is not None:
            with open(pos_file,'wb') as f:
                np.array([self.nblock, self.blockSize, self.dataDim]).astype('u4').tofile(f)        
                pos = np.empty((self.nblock, self.dataDim, self.blockSize))
                pos[:,0,:] = self.pos[0,:].reshape(self.nblock,self.blockSize)
                pos[:,1,:] = self.pos[1,:].reshape(self.nblock,self.blockSize)
                if self.dataDim == 3:
                    pos[:,2,:] = self.zpos
                pos.tofile(f)

        if OD_file is not None:
            if not self.pODready: 
                self.assign_pos_OD2()
            with open(OD_file,'wb') as f:
                self.ODlabel.astype('i4').tofile(f)

        if OP_file is not None:
            if not self.pOPready:
                if self.assign_pos_OP(True) is None:
                    raise Exception('failed to assign OP to neurons') 
            with open(OP_file,'wb') as f:
                self.op.tofile(f)

        if VFxy_file is not None:
            with open(VFxy_file,'wb') as f:
                np.array([self.networkSize]).astype('u4').tofile(f)
                self.vpos.tofile(f)

        if VFpolar_file is not None:
            vpos = self.assign_pos_VF()
            with open(VFpolar_file,'wb') as f:
                np.array([self.nblock, self.blockSize]).astype('u4').tofile(f)
                vpos.tofile(f)

        if Feature_file is not None:
            with open(Feature_file,'wb') as f:
                np.array([2]).astype('u4').tofile(f)
                self.ODlabel.astype(fp).tofile(f)
                self.op.astype(fp).tofile(f)

        if Parallel_uniform_file is not None:
            subarea = self.subgrid[0] * self.subgrid[1]
            area = subarea * np.sum(self.Pi > 0)
            A = self.Pi.copy()
            A[self.Pi <= 0] = 0
            A[self.Pi > 0] = 1
            boundPos, btype = self.define_bound(A)
            with open(Parallel_uniform_file, 'wb') as f:
                np.array([self.networkSize]).astype('u4').tofile(f)
                self.pos.tofile(f)
                np.array([btype.size]).astype('u4').tofile(f)
                boundPos.tofile(f)
                btype.astype('u4').tofile(f)
                print(np.min(btype), np.max(btype))
                self.subgrid.tofile(f)
                np.array([area]).astype('f8').tofile(f)

        if allpos_file is not None:
            subarea = self.subgrid[0] * self.subgrid[1]
            area = subarea * np.sum(self.Pi > 0)
            A = self.Pi.copy()
            A[self.Pi <= 0] = 0
            A[self.Pi > 0] = 1
            boundPos, _ = self.define_bound(A)
            x0 = np.min(boundPos[:,0,1]) 
            xspan = np.max(boundPos[:,0,1]) - x0
            y0 = np.min(boundPos[:,1,1])
            yspan = np.max(boundPos[:,1,1]) - y0
            vx0 = 0
            vxspan = self.ecc
            vy0 = -self.ecc
            vyspan = 2*self.ecc
            with open(allpos_file,'wb') as f:
                np.array([self.nblock, self.blockSize, dataDim]).astype('u4').tofile(f)        
                pos = np.empty((dataDim, self.networkSize))
                np.array([x0, xspan, y0, yspan], dtype = float).tofile(f)        
                pos[:2,:] = self.pos.copy()
                if dataDim == 3:
                    pos[2,:] = self.zpos.flatten()
                pos.tofile(f)
                vpos = self.assign_pos_VF()
                np.array([vx0, vxspan, vy0, vyspan], dtype = float).tofile(f)        
                vx = vpos[0,:] * np.cos(vpos[1,:])
                vy = vpos[0,:] * np.sin(vpos[1,:])
                np.vstack((vx,vy)).tofile(f)

        if Parallel_spreadVF_file is not None:
            subarea = self.subgrid[0] * self.subgrid[1]
            area = subarea * np.sum(self.Pi > 0)
            A = self.Pi.copy()
            A[self.Pi <= 0] = 0
            A[self.Pi > 0] = 1
            boundPos, btype = self.define_bound(A)
            with open(Parallel_spreadVF_file, 'wb') as f:
                np.array([self.networkSize]).astype('u4').tofile(f)
                self.pos.tofile(f)
                np.array([btype.size]).astype('u4').tofile(f)
                boundPos.tofile(f)
                btype.astype('u4').tofile(f)
                print(np.min(btype), np.max(btype))
                self.subgrid.tofile(f)
                np.array([area]).astype('f8').tofile(f)

    '''
######## rarely-used functions ##############
    #
    # memory requirement low, slower
    def assign_pos_OD0(self):
        print('slower, low mem requirement')
        ## determine which blocks to be consider in each subgrid to avoid too much iteration over the whole network
        # calculate reach of each block as its radius, some detached blocks may be included
        pos = self.pos.reshape(2,self.nblock,self.blockSize)
        blockPosX = np.reshape(np.mean(pos[0,:,:],1), (self.nblock,1))
        blockPosY = np.reshape(np.mean(pos[1,:,:],1), (self.nblock,1))
        max_dis2center = np.max(np.power(pos[0,:,:]-blockPosX,2) + np.power(pos[1,:,:]-blockPosY,2),1).reshape((self.nblock,1))
        # iterate over each grid
        corner = np.zeros((2,4))
        self.ODlabel = np.zeros((self.networkSize), dtype=int)
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

    def flip_pos_along_OD(self, seed=None):
        if self.OD_VF_reconciled == False:
            if self.pVFready == False:
                self.assign_pos_VF()
            if self.pODready == False:
                self.assign_pos_OD2()
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

    def make_pos_uniformT(self, dt, seed = None, particle_param = None, boundary_param = None, ax1 = None, ax2 = None, check = True, b_scale = 2.0, p_scale = 2.0, fixed = None, ratio = 2.0):
        subarea = self.subgrid[0] * self.subgrid[1]
        area = subarea * np.sum(self.Pi > 0)
        print(f'grid area: {area}, used in simulation')
        A = self.Pi.copy()
        A[self.Pi <= 0] = 0
        A[self.Pi > 0] = 1
        OD_bound, btype = self.define_bound(A)

        if fixed is None:
            nfixed = 0
        else:
            nfixed = fixed.size
        per_unit_area = 2*np.sqrt(3) # in cl^2
        i_cl = np.sqrt(area/per_unit_area/(self.networkSize - ratio*nfixed*np.power(b_scale/p_scale,2)))
        b_cl = b_scale/p_scale * i_cl

        oldpos = self.pos.copy()
        self.pos, convergence, nlimited = simulate_repel(area, self.subgrid, self.pos, dt, OD_bound, btype, boundary_param, particle_param, ax = ax1, seed = seed, p_scale = p_scale, b_scale = b_scale, fixed = fixed, mshape = '.', b_cl = b_cl, i_cl = i_cl)

        if check:
            self.check_pos(False, False) # check outer boundary only

        return b_cl, i_cl

    def make_pos_uniform(self, dt, seed = None, particle_param = None, boundary_param = None, ax1 = None, ax2 = None, check = True):
        if not self.pODready:
            self.assign_pos_OD2()
        pR = self.ODlabel > 0
        pL = self.ODlabel < 0
        nR = np.sum(pR)
        nL = np.sum(pL)
        areaL = self.area * nL/(nR+nL)
        areaR = self.area * nR/(nR+nL)
        print(f'smooth area: {areaL}, {areaR}')
        subarea = self.subgrid[0] * self.subgrid[1]
        areaR = subarea * np.sum(self.LR == 1)
        areaL = subarea * np.sum(self.LR == -1)
        print(f'grid area: {areaL}, {areaR}, used in simulation')
        if self.LR_boundary_defined is False:
            self.define_bound_LR()

        oldpos = self.pos.copy()
        self.pos[:,pL], convergenceL, nlimitedL = simulate_repel(areaL, self.subgrid, self.pos[:,pL], dt, self.OD_boundL, self.btypeL, boundary_param, particle_param, ax = ax1, seed = seed)
        if check:
            self.check_pos()
        self.pos[:,pR], convergenceR, nlimitedR = simulate_repel(areaR, self.subgrid, self.pos[:,pR], dt, self.OD_boundR, self.btypeR, boundary_param, particle_param, ax = ax2, seed = seed)
        if check:
            self.check_pos()

        if not self.posUniform:
            self.posUniform = True
        return oldpos, convergenceL, convergenceR, nlimitedL, nlimitedR

    # memory requirement high, faster
    def assign_pos_OD1(self, check = True):
        print('faster, high mem required')
        """ check duplicates beforehand
        duplicate = 0
        pos = self.pos.reshape(2,self.nblock,self.blockSize)
        for i in range(self.nblock):
            #assert(np.sum((self.pos[:,i] - self.pos[:,i+1:].T == 0).all(-1))==0)
            test_x = np.tile(pos[0,i,:], (self.blockSize,1)) - pos[0,i,:].T
            nd = 0
            if np.sum(test_x == 0) > self.blockSize:
                test_y = np.tile(pos[1,i,:], (self.blockSize,1)) - pos[1,i,:].T
                nd = (np.sum(np.logical_and(test_x, test_y)) - self.blockSize)/2
                duplicate = duplicate + nd
                if nd > 0:
                    print(f'{nd} neurons are duplicated by in block {i}')
            stdout.write(f'\r{(i+1)/self.nblock*100:.3f}%')
        stdout.write('\n')
        if duplicate > 0:
            print(f'before pos adjustment: {duplicate} points have duplicate(s)')
            assert(duplicate == 0)
        """
        print('get_ij_grid')
        i, j, d2 = self.get_ij_grid(get_d2 = True, get_coord = False)

        # pick neurons that is outside the discrete ragged boundary
        d2[self.Pi[i,j]<=0,    0] = np.float('inf') #bot_left
        d2[self.Pi[i+1,j]<=0,  1] = np.float('inf') #top_left
        d2[self.Pi[i,j+1]<=0,  2] = np.float('inf') #bot_right
        d2[self.Pi[i+1,j+1]<=0,3] = np.float('inf') #top_right
        ibp = (d2 == np.tile(np.array([np.float('inf'), np.float('inf'), np.float('inf'), np.float('inf')]), (self.networkSize,1))).any(-1)
        nbp = np.sum(ibp)
        print(f'#boundary points: {nbp} ')

        outcast = (d2 == np.tile(np.array([np.float('inf'), np.float('inf'), np.float('inf'), np.float('inf')]), (self.networkSize,1))).all(-1)
        if np.sum(outcast) > 0:
            raise Exception(f'grid area is not consistent with the area occupied by neurons, {np.sum(outcast)} out side of range.')
            
        per_unit_area = 2*np.sqrt(3) # in cl^2
        cl = np.sqrt((self.area/self.networkSize)/per_unit_area)
        if cl > self.subgrid[0]/2 or cl > self.subgrid[1]/2:
            cl = 0.5 * np.min(self.subgrid/2)
            print('restricting reflection length to a quarter of subgrid size')
        print((cl, self.subgrid))

        ipick = np.argmin(d2,1)
        
        # retain neurons inside the discrete ragged boundary
        outsideRaggedBound = np.stack((self.Pi[i,j] <= 0, self.Pi[i+1,j] <= 0, self.Pi[i,j+1] <= 0, self.Pi[i+1,j+1] <= 0), axis = -1).any(1)
        print(outsideRaggedBound.shape)
        print(f'adjust {np.sum(outsideRaggedBound)} positions near the boundary')
        x = ipick[outsideRaggedBound]//2
        y = np.mod(ipick[outsideRaggedBound],2)
        bx = self.xx[i[outsideRaggedBound] + y , j[outsideRaggedBound] + x]
        by = self.yy[i[outsideRaggedBound] + y , j[outsideRaggedBound] + x]

        indices = np.arange(self.networkSize)

        # bot_left
        bot_left = np.logical_and(x == 0, y == 0)
        if bot_left.any():
            oRB_copy = outsideRaggedBound.copy()
            oRB_copy[indices[outsideRaggedBound][np.logical_not(bot_left)]] = False

            cx = bx[bot_left] + self.subgrid[0]/2
            cy = by[bot_left] + self.subgrid[1]/2

            pick = cx - self.pos[0,oRB_copy] < cl
            ii_pick = indices[oRB_copy][pick]
            self.pos[0, ii_pick] = cx[pick] - cl - np.abs(self.pos[0, ii_pick] - cx[pick])/(self.subgrid[0]/2) * (self.subgrid[0]/2 - cl)
            pick = cy - self.pos[1,oRB_copy] < cl
            ii_pick = indices[oRB_copy][pick]
            self.pos[1, ii_pick] = cy[pick] - cl - np.abs(self.pos[1, ii_pick] - cy[pick])/(self.subgrid[1]/2) * (self.subgrid[1]/2 - cl)

        # bot_right
        bot_right = np.logical_and(x == 1, y == 0)
        if bot_right.any():
            oRB_copy = outsideRaggedBound.copy()
            oRB_copy[indices[outsideRaggedBound][np.logical_not(bot_right)]] = False

            cx = bx[bot_right] - self.subgrid[0]/2
            cy = by[bot_right] + self.subgrid[1]/2

            pick = self.pos[0,oRB_copy] - cx < cl
            ii_pick = indices[oRB_copy][pick]
            self.pos[0, ii_pick] = cx[pick] + cl + np.abs(self.pos[0, ii_pick] - cx[pick])/(self.subgrid[0]/2) * (self.subgrid[0]/2 - cl)
            pick = cy - self.pos[1,oRB_copy] < cl                                                                                        
            ii_pick = indices[oRB_copy][pick]                                                                                            
            self.pos[1, ii_pick] = cy[pick] - cl - np.abs(self.pos[1, ii_pick] - cy[pick])/(self.subgrid[1]/2) * (self.subgrid[1]/2 - cl)

        # top_left
        top_left = np.logical_and(x == 0, y == 1)
        if top_left.any():
            oRB_copy = outsideRaggedBound.copy()
            oRB_copy[indices[outsideRaggedBound][np.logical_not(top_left)]] = False

            cx = bx[top_left] + self.subgrid[0]/2
            cy = by[top_left] - self.subgrid[1]/2

            pick = cx - self.pos[0,oRB_copy] < cl
            ii_pick = indices[oRB_copy][pick]
            self.pos[0, ii_pick] = cx[pick] - cl - np.abs(self.pos[0, ii_pick] - cx[pick])/(self.subgrid[0]/2) * (self.subgrid[0]/2 - cl)
            pick = self.pos[1,oRB_copy] - cy < cl                                                                                        
            ii_pick = indices[oRB_copy][pick]                                                                                            
            self.pos[1, ii_pick] = cy[pick] + cl + np.abs(self.pos[1, ii_pick] - cy[pick])/(self.subgrid[1]/2) * (self.subgrid[1]/2 - cl)

        # top_right
        top_right = np.logical_and(x == 1, y == 1)
        if top_right.any():
            oRB_copy = outsideRaggedBound.copy()
            oRB_copy[indices[outsideRaggedBound][np.logical_not(top_right)]] = False

            cx = bx[top_right] - self.subgrid[0]/2
            cy = by[top_right] - self.subgrid[1]/2

            pick = self.pos[0,oRB_copy] - cx < cl
            ii_pick = indices[oRB_copy][pick]
            self.pos[0, ii_pick] = cx[pick] + cl + np.abs(self.pos[0, ii_pick] - cx[pick])/(self.subgrid[0]/2) * (self.subgrid[0]/2 - cl)
            pick = self.pos[1,oRB_copy] - cy < cl                                                                                        
            ii_pick = indices[oRB_copy][pick]                                                                                            
            self.pos[1, ii_pick] = cy[pick] + cl + np.abs(self.pos[1, ii_pick] - cy[pick])/(self.subgrid[1]/2) * (self.subgrid[1]/2 - cl)

        print('assign ocular dominance preference to neuron according to their position in the cortex')
        self.ODlabel = np.zeros((self.networkSize), dtype = int)
        self.ODlabel[ipick == 0] = self.LR[i,    j][ipick == 0]
        self.ODlabel[ipick == 1] = self.LR[i+1,  j][ipick == 1]
        self.ODlabel[ipick == 2] = self.LR[i,  j+1][ipick == 2]
        self.ODlabel[ipick == 3] = self.LR[i+1,j+1][ipick == 3]
        assert(np.sum(self.ODlabel > 0) + np.sum(self.ODlabel < 0) == self.networkSize)

        print('retract neurons from the OD-boundary to avoid extreme repelling force later')
        # move neuron away from LR boundary to avoid extreme repelling force.
        # collect neurons in heterogeneous LR grids
        LRgridType = np.stack((self.LR[i, j], self.LR[i+1, j], self.LR[i, j+1], self.LR[i+1, j+1]), axis=-1)
        #bpick = np.logical_or(np.sum(LRgridType, axis=-1) == -2, np.sum(LRgridType, axis=-1) == 2)
        bpick = np.logical_and(np.sum(LRgridType, axis=-1) != -4, np.sum(LRgridType, axis=-1) != 4)
        # exclude cortex boundary grids
        bpick = np.logical_and(bpick, np.logical_not(outsideRaggedBound))
        npick = np.sum(bpick)

        x = ipick[bpick]//2
        y = np.mod(ipick[bpick],2)
        bx = self.xx[i[bpick] + y , j[bpick] + x]
        by = self.yy[i[bpick] + y , j[bpick] + x]
        #self.pos[0,bpick] = bx + (self.pos[0,bpick] - bx)/3
        #self.pos[1,bpick] = by + (self.pos[1,bpick] - by)/3

        indices = np.arange(self.networkSize)
        # bot_left
        bot_left = np.logical_and(x == 0, y == 0)
        if bot_left.any():
            bp_copy = bpick.copy()
            bp_copy[indices[bpick][np.logical_not(bot_left)]] = False

            cx = bx[bot_left] + self.subgrid[0]/2
            cy = by[bot_left] + self.subgrid[1]/2

            pick = cx - self.pos[0,bp_copy] < cl
            ii_pick = indices[bp_copy][pick]
            self.pos[0, ii_pick] = cx[pick] - cl - np.abs(self.pos[0, ii_pick] - cx[pick])/(self.subgrid[0]/2) * (self.subgrid[0]/2 - cl)
            pick = cy - self.pos[1,bp_copy] < cl                                                                                         
            ii_pick = indices[bp_copy][pick]                                                                                             
            self.pos[1, ii_pick] = cy[pick] - cl - np.abs(self.pos[1, ii_pick] - cy[pick])/(self.subgrid[1]/2) * (self.subgrid[1]/2 - cl)

        # bot_right
        bot_right = np.logical_and(x == 1, y == 0)
        if bot_right.any():
            bp_copy = bpick.copy()
            bp_copy[indices[bpick][np.logical_not(bot_right)]] = False

            cx = bx[bot_right] - self.subgrid[0]/2
            cy = by[bot_right] + self.subgrid[1]/2

            pick = self.pos[0,bp_copy] - cx < cl
            ii_pick = indices[bp_copy][pick]
            self.pos[0, ii_pick] = cx[pick] + cl + np.abs(self.pos[0, ii_pick] - cx[pick])/(self.subgrid[0]/2) * (self.subgrid[0]/2 - cl)
            pick = cy - self.pos[1,bp_copy] < cl                                                                                         
            ii_pick = indices[bp_copy][pick]                                                                                             
            self.pos[1, ii_pick] = cy[pick] - cl - np.abs(self.pos[1, ii_pick] - cy[pick])/(self.subgrid[1]/2) * (self.subgrid[1]/2 - cl)

        # top_left
        top_left = np.logical_and(x == 0, y == 1)
        if top_left.any():
            bp_copy = bpick.copy()
            bp_copy[indices[bpick][np.logical_not(top_left)]] = False

            cx = bx[top_left] + self.subgrid[0]/2
            cy = by[top_left] - self.subgrid[1]/2

            pick = cx - self.pos[0,bp_copy] < cl
            ii_pick = indices[bp_copy][pick]
            self.pos[0, ii_pick] = cx[pick] - cl - np.abs(self.pos[0, ii_pick] - cx[pick])/(self.subgrid[0]/2) * (self.subgrid[0]/2 - cl)
            pick = self.pos[1,bp_copy] - cy < cl                                                                                         
            ii_pick = indices[bp_copy][pick]                                                                                             
            self.pos[1, ii_pick] = cy[pick] + cl + np.abs(self.pos[1, ii_pick] - cy[pick])/(self.subgrid[1]/2) * (self.subgrid[1]/2 - cl)

        # top_right
        top_right = np.logical_and(x == 1, y == 1)
        if top_right.any():
            bp_copy = bpick.copy()
            bp_copy[indices[bpick][np.logical_not(top_right)]] = False

            cx = bx[top_right] - self.subgrid[0]/2
            cy = by[top_right] - self.subgrid[1]/2

            pick = self.pos[0,bp_copy] - cx < cl
            ii_pick = indices[bp_copy][pick]
            self.pos[0, ii_pick] = cx[pick] + cl + np.abs(self.pos[0, ii_pick] - cx[pick])/(self.subgrid[0]/2) * (self.subgrid[0]/2 - cl)
            pick = self.pos[1,bp_copy] - cy < cl                                                                                         
            ii_pick = indices[bp_copy][pick]                                                                                             
            self.pos[1, ii_pick] = cy[pick] + cl + np.abs(self.pos[1, ii_pick] - cy[pick])/(self.subgrid[1]/2) * (self.subgrid[1]/2 - cl)

        # checks
        if check:
            self.check_pos(False, False) # check outer boundary only
            # check if exists overlap positions
            print('after pos adjustment: ')
            duplicate = 0
            pos = self.pos.reshape(2,self.nblock,self.blockSize)
            for i in range(self.nblock):
                #assert(np.sum((self.pos[:,i] - self.pos[:,i+1:].T == 0).all(-1))==0)
                test_x = np.tile(pos[0,i,:], (self.blockSize,1)) - pos[0,i,:].T
                nd = 0
                if np.sum(test_x == 0) > self.blockSize:
                    test_y = np.tile(pos[1,i,:], (self.blockSize,1)) - pos[1,i,:].T
                    nd = (np.sum(np.logical_and(test_x, test_y)) - self.blockSize)/2
                    duplicate = duplicate + nd
                    if nd > 0:
                        print(f'{nd} neurons are duplicated by in block {i}')
                #stdout.write(f'\r{(i+1)/self.nblock*100:.3f}%')
            #stdout.write('\n')
            if duplicate > 0:
                print(f'{duplicate} point(s) have duplicate(s)')
                assert(duplicate == 0)
            else:
                print(f'no duplicates')
        
        self.pODready = True
        return self.ODlabel
    def make_pos_uniform_parallel(self, dt, b_scale, p_scale, figname, ncpu = 16, ndt0 = 0, check = False):
        if not self.pODready:
            self.assign_pos_OD2()
        pR = self.ODlabel > 0
        pL = self.ODlabel < 0
        nR = np.sum(pR)
        nL = np.sum(pL)
        areaL = self.area * nL/(nR+nL)
        areaR = self.area * nR/(nR+nL)
        print(f'smooth area: {areaL}, {areaR}')
        subarea = self.subgrid[0] * self.subgrid[1]
        areaR = subarea * np.sum(self.LR == 1)
        areaL = subarea * np.sum(self.LR == -1)
        print(f'grid area: {areaL}, {areaR}, used in simulation')
        if self.LR_boundary_defined is False:
            self.define_bound_LR()

        oldpos = self.pos.copy()
        self.pos[:,pL] = simulate_repel_parallel(areaL, self.subgrid, self.pos[:,pL], dt, self.OD_boundL, self.btypeL, b_scale, p_scale, figname +'L', ndt0, ncpu)
        if check:
            self.check_pos()
        self.pos[:,pR] = simulate_repel_parallel(areaR, self.subgrid, self.pos[:,pR], dt, self.OD_boundR, self.btypeR, b_scale, p_scale, figname +'R', ndt0, ncpu)
        if check:
            self.check_pos()

        if not self.posUniform:
            self.posUniform = True
        return oldpos

    '''

