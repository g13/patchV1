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

class macroMap:
    def __init__(self, LR_Pi_file, pos_file, shrink = False, posUniform = False, OPgrid_file = None, OD_file = None, OP_file = None, VFxy_file = None):
        with open(pos_file,'r') as f:
            self.nblock = np.fromfile(f,'u4', count=1)[0]
            self.blockSize = np.fromfile(f,'u4', count=1)[0]
            self.dataDim = np.fromfile(f,'u4', count=1)[0]
            self.networkSize = self.nblock * self.blockSize
            print(f'nblock = {self.nblock}, blockSize = {self.blockSize}, posDim = {self.dataDim}')
            pos = np.reshape(np.fromfile(f,'f8',count = self.dataDim*self.networkSize),(self.nblock,self.dataDim,self.blockSize))
        self.pos = np.zeros((2,self.networkSize))
        self.pos[0,:] = pos[:,0,:].reshape(self.networkSize)
        self.pos[1,:] = pos[:,1,:].reshape(self.networkSize)
        if self.dataDim == 3:
            self.zpos = pos[:,2,:]
        with open(LR_Pi_file,'r') as f:
            self.a = np.fromfile(f, 'f8', count=1)[0]
            self.b = np.fromfile(f, 'f8', count=1)[0]
            self.k = np.fromfile(f, 'f8', count=1)[0]
            self.ecc = np.fromfile(f, 'f8', count=1)[0]
            nx = np.fromfile(f, 'i4', count=1)[0]
            ny = np.fromfile(f, 'i4', count=1)[0]
            self.Pi = np.reshape(np.fromfile(f, 'i4', count = nx*ny),(ny,nx))
            self.x = np.fromfile(f, 'f8', count = nx)
            x_max = max(self.pos[0,:])
            x_min = min(self.pos[0,:])
            if x_min < self.x[0] or x_max > self.x[-1]:
                raise Exception(f'neuronal x-position {[x_min, x_max]} break outside of grid {[self.x[0], self.x[-1]]}')
            self.y = np.fromfile(f, 'f8', count = self.ny)
            y_max = max(self.pos[1,:])
            y_min = min(self.pos[1,:])
            if y_min < self.y[0] or y_max > self.y[-1]:
                raise Exception(f'neuronal y-position {[y_min, y_max]} break outside of grid {[self.y[0], self.y[-1]]}')
            LR = np.reshape(np.fromfile(f, 'f8', count = nx*ny),(ny,nx))

            if shrink:
                self.nx = np.nonzero(self.x > x_max)[0][0] + 1
                self.x = self.x[:self.nx]
                self.ecc = e_x(self.x[-1], k, a, b)
                y0 = np.nonzero(self.y < y_min)[0][-1]
                y1 = np.nonzero(self.y > y_max)[0][0]+1
                self.y = self.y[y0:y1]
                self.ny = self.y.size
                self.Pi = self.Pi[y0:y1,:self.nx]
            else:
                self.nx = nx
                self.ny = ny

            self.xx, self.yy = np.meshgrid(self.x, self.y)
            # double to int
            self.LR = np.empty((self.ny,self.nx), dtype = 'i4') 
            self.LR[LR > 0] = 1
            self.LR[LR < 0] = -1
            if shrink:
                self.LR = self.LR[y0:y1,:self.nx]

            self.LR[self.Pi <=0] = 0
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
        
        # read preset orientation preferences
        if OPgrid_file is not None:
            with open(OPgrid_file,'r') as f:
                self.OPgrid_x = np.reshape(np.fromfile(f, 'f8', count = nx*ny),(ny,nx))
                self.OPgrid_y = np.reshape(np.fromfile(f, 'f8', count = nx*ny),(ny,nx))
                if shrink:
                    self.OPgrid_x = self.OPgrid_x[y0:y1,:self.nx]
                    self.OPgrid_y = self.OPgrid_y[y0:y1,:self.nx]
                self.OPgrid = np.arctan2(self.OPgrid_y, self.OPgrid_x)

        if OP_file is not None:
            with open(OP_file,'r') as f:
                self.op = np.fromfile(f, 'f8')
                if self.op.size != self.nx*self.ny:
                    raise Exception(f'wrong op file: {OP_file}, size of grid no match')
                else:
                    self.op = np.reshape(self.op,(self.ny,self.nx))

        if VFxy_file is not None:
            with open(VFxy_file,'r') as f:
                np.fromfile(f, 'u4', count = 1)
                self.vpos = np.reshape(np.fromfile(f, 'f8', count = self.networkSize*2),self.pos.shape)

        if OD_file is not None:
            with open(OD_file,'rb') as f:
                self.ODlabel = np.fromfile(f, 'i4', count = self.networkSize)
                print('loading OD_labels...')
                assert(np.sum(self.ODlabel > 0) + np.sum(self.ODlabel < 0) == self.networkSize)
                print('checked')

        
        self.pODready = OD_file is not None
        self.pOPready = OP_file is not None 
        self.LR_boundary_defined = False 
        self.pVFready = VFxy_file is not None 
        self.vposLready = VFxy_file is not None
        self.vposRready = VFxy_file is not None
        self.posUniform = posUniform
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

        assert(self.op.size == self.networkSize)
        assert(np.sum(np.isnan(self.op)) == 0)
        self.pOPready = True

        self.op = (self.op + np.pi)/2/np.pi
        print('op ready')
        return self.op

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
            assert(False)
            
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
        if nunpick > 0:
            indices = np.nonzero(unpick)[0]
            p, q = np.unravel_index(indices, (self.ny-1, self.nx-1))
            pick0 = self.Pi[p, q+1] == 0
            pick1 = self.Pi[p+1, q] == 0
            assert(np.logical_or(pick0, pick1).all())
            bgrid[indices[pick0], :] = np.array([1,1,1,0])
            bgrid[indices[pick1], :] = np.array([1,0,1,1])

        unb = np.array([0,1,0,1], dtype = bool)
        unpick = (bgrid - unb == 0).all(-1)
        nunpick = nunpick + np.sum(unpick)
        if nunpick > 0:
            indices = np.nonzero(unpick)[0]
            p, q = np.unravel_index(indices, (self.ny-1, self.nx-1))
            pick0 = self.Pi[p, q] == 0
            pick1 = self.Pi[p+1, q+1] == 0
            assert(np.logical_or(pick0, pick1).all())
            bgrid[indices[pick0], :] = np.array([1,1,0,1])
            bgrid[indices[pick1], :] = np.array([0,1,1,1])

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
        assert((self.Pi[i,    j][ipick == 0] > 0).all())
        assert((self.Pi[i+1,  j][ipick == 1] > 0).all())
        assert((self.Pi[i,  j+1][ipick == 2] > 0).all())
        assert((self.Pi[i+1,j+1][ipick == 3] > 0).all())

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
            self.assign_pos_OD1()
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

    def make_pos_uniform_p(self, dt, p_scale, b_scale, figname, ncore = 0, ndt_decay = 0, roi_ratio = 2.0, k1 = 1.0, k2 = 0.5, chop_ratio = 0, spercent = 0.01, seed = -1, local_plot = False, check = False):
        if not self.pODready:
            self.assign_pos_OD1()
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

        if chop_ratio == 0:
            chop_ratio = self.nblock/(self.nx*self.ny)
        oldpos = self.pos.copy()
        nx = np.round(self.nx*chop_ratio).astype('i4')
        ny = np.round(self.ny*chop_ratio).astype('i4')

    ##  prepare shared memory
        raw_shared_cmem = RawArray(c_int32, np.zeros(ncore, dtype = 'i4'))
        shared_cmem = np.frombuffer(raw_shared_cmem, dtype = 'i4')
    # Left
        posL = np.empty(self.nblock, dtype = object)
        pickL = pL.view().reshape((self.nblock,self.blockSize))
        pos = self.pos.view().reshape((2,self.nblock,self.blockSize))
        blockPick = np.ones(self.nblock)
        for i in range(self.nblock):
            if np.sum(pickL[i,:]) > 0:
                posL[i] = pos[:,i,pickL[i,:]]
            else:
                blockPick[i] = 0
        posL = posL[blockPick>0]
        blockL = posL.size
        blockSizeL = np.array([p.shape[1] for p in posL])
        nL = np.sum(pL)

        raw_shared_dmem = RawArray(c_double, np.zeros(nL*7, dtype = 'f8'))
        raw_shared_imem = RawArray(c_int32, np.zeros(nL, dtype = 'i4'))
        shared_dmem = np.frombuffer(raw_shared_dmem, dtype = 'f8')
        shared_imem = np.frombuffer(raw_shared_imem, dtype = 'i4')

        # populate position in shared array
        shared_dmem[:2*nL] = np.hstack([p.flatten() for p in posL])
        posL = np.empty(blockL, dtype = object)
        i0 = 0
        for i in range(blockL):
            i1 = i0 + blockSizeL[i]*2
            posL[i] = shared_dmem[i0:i1].reshape((2,blockSizeL[i]))
            i0 = i1
        assert(i0 == 2*nL)

        raw_shared_bmem = RawArray(c_double, np.zeros(nx*ny*6, dtype = 'f8'))
        shared_bmem = np.frombuffer(raw_shared_bmem, dtype = 'f8') 
        raw_shared_nmem = RawArray(c_int32, np.zeros(nx*ny*6, dtype = 'i4')) # for neighbor_list and id
        shared_nmem = np.frombuffer(raw_shared_nmem, dtype = 'i4') 

        self.pos[:,pL] = parallel_repel(areaL, self.subgrid, posL, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, p_scale, self.OD_boundL, self.btypeL, b_scale, nx, ny, ncore, dt, spercent, figname + 'L', ndt_decay, 1.0, roi_ratio, k1, k2, seed, 1.0, local_plot)
        if check:
            self.check_pos()

        posR = np.empty(self.nblock, dtype = object)
        pickR = pR.view().reshape((self.nblock,self.blockSize))
        pos = self.pos.view().reshape((2,self.nblock,self.blockSize))
        blockPick = np.ones(self.nblock)
        for i in range(self.nblock):
            if np.sum(pickR[i,:]) > 0:
                posR[i] = pos[:,i,pickR[i,:]]
            else:
                blockPick[i] = 0
        posR = posR[blockPick>0]
        blockR = posR.size
        blockSizeR = np.array([p.shape[1] for p in posR])
        nR = np.sum(pR)

        raw_shared_dmem = RawArray(c_double, np.zeros(nR*7, dtype = 'f8'))
        raw_shared_imem = RawArray(c_int32, np.zeros(nR, dtype = 'i4'))
        shared_dmem = np.frombuffer(raw_shared_dmem, dtype = 'f8')
        shared_imem = np.frombuffer(raw_shared_imem, dtype = 'i4')

        # populate position in shared array
        shared_dmem[:2*nR] = np.hstack([p.flatten() for p in posR])
        posR = np.empty(blockR, dtype = object)
        i0 = 0
        for i in range(blockR):
            i1 = i0 + blockSizeR[i]*2
            posR[i] = shared_dmem[i0:i1].reshape((2,blockSizeR[i]))
            i0 = i1
        assert(i0 == 2*nR)

        # no need to re-declare bmem, nmem and cmem

        self.pos[:,pR] = parallel_repel(areaR, self.subgrid, posR, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, p_scale, self.OD_boundR, self.btypeR, b_scale, nx, ny, ncore, dt, spercent, figname + 'R', ndt_decay, 1.0, roi_ratio, k1, k2, seed, 1.0, local_plot)
        if check:
            self.check_pos()

        if not self.posUniform:
            self.posUniform = True
        return oldpos

    def make_pos_uniform_parallel(self, dt, b_scale, p_scale, figname, ncpu = 16, ndt0 = 0, check = False):
        if not self.pODready:
            self.assign_pos_OD1()
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

    def spread_pos_VF(self, dt, vpfile, lrfile, LRlabel, seed = None, firstTime = True, continuous = True, particle_param = None, boundary_param = None, ax = None, p_scale = 2.0, b_scale = 1.0, ndt_decay = 0, roi_ratio = 2, k1 = 1.0, k2 = 0.5, ncore= 0, limit_ratio = 1.0, chop_ratio = 0):
        if ncore == 0:
            ncore = mp.cpu_count()
            print(f'{ncore} cores found')
        if chop_ratio == 0:
            chop_ratio = self.nblock/(self.nx*self.ny)

        oldpos = self.pos.copy()
        nx = np.round(self.nx*chop_ratio).astype('i4')
        ny = np.round(self.ny*chop_ratio).astype('i4')

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
        raw_shared_dmem = RawArray(c_double, np.zeros(nLR*7, dtype = 'f8'))
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
            with open(lrfile+'.bin','rb') as f:
                consts = np.fromfile(f, 'i4', 2)
                LR = np.fromfile(f, 'i4', count = np.prod(self.Pi.shape)).reshape(self.Pi.shape)

            with open(vpfile+'.bin','rb') as f:
                nchop0 = np.fromfile(f, 'i4', 1)[0]
                chop_size0 = np.fromfile(f, 'i4', nchop0)
                assert(sum(chop_size0) == nLR)
                vposLR = np.fromfile(f, 'f8').reshape(2, nLR)
                #i0 = 0
                #ii0 = 0
                #k = 0
                #for i in range(nchop0):
                #    if chop_size0[k] > 0:
                #        i1 = i0 + chop_size0[k]
                #        ii1 = ii0 + chop_size0[k]*2
                #        shared_dmem[ii0:ii1] = vposLR[:,i0:i1].flatten()
                #        k = k + 1
                #        i0 = i1
                #        ii0 = ii1
                #assert(ii0 == 2*nLR)

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
                vposLR, LR, spreaded, ppos, ip, OD_bound = self.spread(dt, vposLR, nchop0, chop_size0, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, LR, nx, ny, particle_param, boundary_param, ax, pick, ppos, ip, starting, figname = vpfile, p_scale = p_scale, b_scale = b_scale, ndt_decay = ndt_decay, roi_ratio = roi_ratio, k1 = k1, k2 = k2, ncore = ncore, limit_ratio = limit_ratio, chop_ratio = 0)
                starting = False
                print(f'#{ip}: {np.sum(LR).astype(int)}/{ngrid}')

                assert(vposLR.shape[1] == nLR)
                assert(np.sum(chop_size) == nLR)
                with open(vpfile + f'-{ip}.bin','wb') as f:
                    np.array([nchop0]).astype('i4').tofile(f)
                    chop_size0.astype('i4').tofile(f)
                    vposLR.tofile(f)
                with open(lrfile + f'-{ip}.bin','wb') as f:
                    np.array([LR.size, OD_bound.shape[0]]).astype('i4').tofile(f)
                    LR.astype('i4').tofile(f)
                    OD_bound.copy().astype(float).tofile(f)
                print('data stored')
        else:
            ip = 0 # plotting index
            ppos = None
            starting = True
            vposLR, LR, spreaded, ppos, _, OD_bound = self.spread(dt, vposLR, nchop0, chop_size0, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, LR, nx, ny, particle_param, boundary_param, ax, pick, ppos, ip, starting, figname = vpfile, p_scale = p_scale, b_scale = b_scale, ndt_decay = ndt_decay, k1 = k1, k2 = k2, ncore = ncore, limit_ratio = limit_ratio, chop_ratio = 0, seed = seed, plotOnly = False, noSpread = False)
            print(f'spreaded = {spreaded}')
            assert(vposLR.shape[1] == nLR)
            with open(vpfile+'-ss.bin','wb') as f:
                np.array([nchop0]).astype('i4').tofile(f)
                chop_size0.astype('i4').tofile(f)
                vposLR.tofile(f)
            with open(lrfile+'-ss.bin','wb') as f:
                np.array([LR.size, OD_bound.shape[0]]).astype('i4').tofile(f)
                LR.astype('i4').tofile(f)
                OD_bound.copy().astype(float).tofile(f)
            print('data stored')

        self.vpos[:, LRpick] = vposLR # when spreaded, return in shape (2,nLR)

        if ax is not None:
            # connecting points on grid sides
            ax.plot(OD_bound[:,0,[0,2]].squeeze(), OD_bound[:,1,[0,2]].squeeze(), ',r')
            # grid centers
            ax.plot(OD_bound[:,0,1].squeeze(), OD_bound[:,1,1].squeeze(), ',g')
            #if not continuous:
                # plot all displacement
            ax.plot(np.vstack((self.pos[0,LRpick], self.vpos[0,LRpick])), np.vstack((self.pos[1,LRpick], self.vpos[1,LRpick])),'-,c', lw =0.01)
            # final positions
            #ax.plot(self.vpos[0,LRpick], self.vpos[1,LRpick], ',k')

        # check only the relevant boundary
        if LRlabel == 'L':
            self.check_pos(True, False) 
        else:
            self.check_pos(False, True)

        if LRlabel == 'L':
            self.vposLready = True
        else:
            self.vposRready = True

    def spread(self, dt, vpos_flat, nchop, chop_size, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, LR, nx, ny, particle_param = None, boundary_param = None, ax = None, pick = None, ppos = None, ip = 0, starting = True, figname = None, p_scale = 2.0, b_scale = 1.0, ndt_decay = 0, roi_ratio = 2, k1 = 1.0, k2 = 0.5, ncore = 0, limit_ratio = 1.0, chop_ratio = 0, seed = -1, plotOnly = False, noSpread = False):
        if starting and ax is not None:
            OD_bound, btype = self.define_bound(LR)
            # connecting points on grid sides
            ax.plot(OD_bound[:,0,[0,2]].squeeze(), OD_bound[:,1,[0,2]].squeeze(), ',r')
            # grid centers
            ax.plot(OD_bound[:,0,1].squeeze(), OD_bound[:,1,1].squeeze(), ',g')
        subarea = self.subgrid[0] * self.subgrid[1]

        old_area = subarea * np.sum(LR == 1)
        if not noSpread or ip == 0:
            LR, spreaded = self.diffuse_bound(LR)
        else:
            spreaded = False
        area = subarea * np.sum(LR == 1)
        ratio = area/old_area
        print(f'area increases {ratio*100:.3f}%')
        
        OD_bound, btype = self.define_bound(LR)

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
            vpos_flat = parallel_repel(area, self.subgrid, vpos_chop, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, p_scale, OD_bound, btype, b_scale, nx, ny, ncore, dt, 0, figname, ndt_decay, 1/np.sqrt(ratio), roi_ratio, k1 = k1, k2 = k2, seed = seed, limit_ratio = limit_ratio, local_plot = False)

        ip = ip + 1
        if ax is not None:
            ppos[ip%2,:,:] = vpos_flat[:,pick].copy()
            ax.plot(ppos[:,0,:].squeeze(), ppos[:,1,:].squeeze(), '-,c', lw = 0.01)

        return vpos_flat, LR, spreaded, ppos, ip, OD_bound

    def get_ij_grid(self, get_d2 = True, get_coord = True):
        print('get the index of the nearest vertex for each neuron in its own grid')

        x0 = self.x[:-1]
        x1 = self.x[1:]
        y0 = self.y[:-1]
        y1 = self.y[1:]
        # index in y-dimension
        i = np.nonzero(np.logical_and(np.tile(self.pos[1,:],(self.ny-1,1))-y0.reshape(self.ny-1,1) > 0, np.tile(self.pos[1,:],(self.ny-1,1))-y1.reshape(self.ny-1,1) <= 0).T)[1]
        if not i.size == self.networkSize:
            nrogue = self.networkSize - i.size
            print(f'{nrogue} particles have gone rogue, {self.pos.shape}')
            assert(np.sum(np.logical_or(self.pos[1,:] > self.y[-1], self.pos[1,:] < self.y[0])) == nrogue)
            assert(i.size == self.networkSize)

        print('indexed in y-dimension')
        # index in x-dimension
        j = np.nonzero(np.logical_and(np.tile(self.pos[0,:],(self.nx-1,1))-x0.reshape(self.nx-1,1) > 0, np.tile(self.pos[0,:],(self.nx-1,1))-x1.reshape(self.nx-1,1) <= 0).T)[1]
        if not j.size == self.networkSize:
            nrogue = self.networkSize - j.size
            print(f'{nrogue} particles have gone rogue')
            assert(np.sum(np.logical_or(self.pos[1,:] > self.x[-1], self.pos[1,:] < self.x[0])) == nrogue)
            assert(j.size == self.networkSize)
        print('indexed in x-dimension')

        
        outcast = np.stack((self.Pi[i,j]<=0, self.Pi[i+1,j]<=0, self.Pi[i,j+1]<=0, self.Pi[i+1,j+1]<=0), axis = -1).all(-1)
        nout = np.sum(outcast)
        if nout > np.sqrt(self.nblock):
            raise Exception('too much outcasts: {nout}, something is wrong')
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
                print('calculate neurons\' cortical distance to the nearest vertex in the grid')
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

    def plot_map(self,ax1,ax2,ax3,dpi,pltOD=True,pltVF=True,pltOP=True,forceOP=False,ngridLine=4):
        if pltOD:
            if not self.pODready:
                #self.assign_pos_OD0()
                self.assign_pos_OD1()
                
            if pltOP:
                if not self.pOPready:
                    if self.assign_pos_OP(forceOP) is None:
                        print('Orientation Preference is not plotted')
            if pltOP and self.pOPready:
                s = (self.networkSize/self.area/1000/25*72/dpi)**2
                hsv = cm.get_cmap('hsv')
                pick = np.logical_not(np.isnan(self.OPgrid))
                                                                                                                    # theta to ratio
                ax1.scatter(self.xx[pick], self.yy[pick], s = s, linewidths=0.0, marker = '^', c= hsv(((self.OPgrid[pick]+np.pi)/2/np.pi).flatten()))
                pick = self.ODlabel>0 
                ax1.scatter(self.pos[0,pick], self.pos[1,pick], s = s, linewidths=0.0, marker = '.', c = hsv(self.op[pick]))
                pick = self.ODlabel<0 
                hsv_val = np.asarray(hsv(self.op[pick]), dtype = float)
                hsv_val[0,:] = 0.75
                ax1.scatter(self.pos[0,pick], self.pos[1,pick], s = s, linewidths=0.0, marker = '.', c =  hsv_val)
            else:
                #ax1.plot(self.pos[0,:], self.pos[1,:],',k')
                #ax1.plot(self.xx[self.Pi<=0], self.yy[self.Pi<=0],',r')
                #ax1.plot(self.xx[self.Pi>0], self.yy[self.Pi>0],',g')
                ax1.plot(self.pos[0,self.ODlabel>0], self.pos[1,self.ODlabel>0],',m')
                ax1.plot(self.pos[0,self.ODlabel<0], self.pos[1,self.ODlabel<0],',c')

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


    def save(self, pos_file = None, OD_file = None, OP_file = None, VFxy_file = None, VFpolar_file = None, Feature_file = None, Parallel_uniform_file = None, allpos_file = None, Parallel_spreadVF_file = None, fp = 'f4'):
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
                self.assign_pos_OD1()
                with open(OD_file,'wb') as f:
                    self.ODlabel.astype('i4').tofile(f)

        if OP_file is not None:
            if not self.pOPready:
                if not self.assign_pos_OP(True) is None:
                    with open(OP_file,'wb') as f:
                        self.op.tofile(f)

        if VFxy_file is not None:
            with open(VFxy_file,'wb') as f:
                np.array([self.networkSize]).astype('u4').tofile(f)
                self.vpos.tofile(f)

        if VFpolar_file is not None:
            vpos = self.assign_pos_VF()
            with open(VFpolar_file,'wb') as f:
                np.array([self.networkSize]).astype('u4').tofile(f)
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
            with open(pos_file,'wb') as f:
                np.array([self.nblock, self.blockSize, self.dataDim]).astype('u4').tofile(f)        
                pos = np.empty((self.nblock, self.dataDim, self.blockSize))
                np.array([x0, xspan, y0, yspan], dtype = float).tofile(f)        
                pos[:,0,:] = self.pos[0,:].reshape(self.nblock,self.blockSize)
                pos[:,1,:] = self.pos[1,:].reshape(self.nblock,self.blockSize)
                if self.dataDim == 3:
                    pos[:,2,:] = self.zpos
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
