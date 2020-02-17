from repel_system import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from sys import stdout
from scipy import integrate
import numpy as np
import time
import py_compile
py_compile.compile('LGN_surface.py')

class surface:
    def __init__(self, shape_file, pos_file, ecc):
        self.ecc = ecc
        with open(pos_file,'r') as f:
            self.nLGN = np.fromfile(f,'u4', count=1)[0]
            self.pos = np.fromfile(f, count = 2*self.nLGN).reshape(2,self.nLGN)
        rmax = np.max(np.sqrt(np.sum(self.pos*self.pos, axis = 0)))
        if rmax > self.ecc:
            raise Exception(f'{rmax} > {self.ecc}')
        with open(shape_file,'r') as f:
            self.nx = np.fromfile(f, 'u4', count = 1)[0]
            self.ny = np.fromfile(f, 'u4', count = 1)[0]
            self.x = np.fromfile(f, count = self.nx)
            self.y = np.fromfile(f, count = self.ny)
            self.xx, self.yy = np.meshgrid(self.x,self.y)
            # continue reading
            self.Pi = np.reshape(np.fromfile(f, 'i4', count = self.nx*self.ny),(self.ny,self.nx))
        # double to int
        self.subgrid = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])
        self.pos_uniform = False 

    # prepare pos
    def prep_pos(self, check = True):
        duplicate = 0
        for i in np.arange(self.nLGN):
            #assert(np.sum((self.pos[:,i] - self.pos[:,i+1:].T == 0).all(-1))==0)
            test = (self.pos[:,i] - self.pos[:,i+1:].T == 0).all(-1)
            nd = np.sum(test.any())
            if nd > 0:
                duplicate = duplicate + 1
                did = np.nonzero(test)[0] + i+1
                print(f'#{i} is duplicated by #{did} at {self.pos[:,i]}')
        if duplicate > 0:
            print(f'before pos adjustment: {duplicate} points have duplicate(s)')
            assert(duplicate == 0)

        i, j, d2 = self.get_ij_grid(get_d2 = True, get_coord = False)

        # pick neurons that is outside the discrete ragged boundary
        d2[self.Pi[i,j]<=0,    0] = np.float('inf')
        d2[self.Pi[i+1,j]<=0,  1] = np.float('inf')
        d2[self.Pi[i,j+1]<=0,  2] = np.float('inf')
        d2[self.Pi[i+1,j+1]<=0,3] = np.float('inf')
        nbp = np.sum((d2 == np.tile(np.array([np.float('inf'), np.float('inf'), np.float('inf'), np.float('inf')]), (self.nLGN,1))).any(-1))
        print(f'#boundary points: {nbp} ')
        assert(np.sum((d2 == np.tile(np.array([np.float('inf'), np.float('inf'), np.float('inf'), np.float('inf')]), (self.nLGN,1))).all(-1)) == 0)
        ipick = np.argmin(d2,1)
        
        # retain neurons inside the discrete ragged boundary
        outsideRaggedBound = ((self.Pi[i,j] <= 0) + (self.Pi[i+1,j] <= 0) + (self.Pi[i,j+1] <= 0) + (self.Pi[i+1,j+1] <= 0)).astype(bool)
        print(f'adjust {np.sum(outsideRaggedBound)} positions near the boundary')
        print(self.pos[:,733])
        print(self.pos[:,776])
        x = np.mod(ipick[outsideRaggedBound],2)
        y = ipick[outsideRaggedBound]//2
        bx = self.xx[i[outsideRaggedBound] + x , j[outsideRaggedBound] + y]
        by = self.yy[i[outsideRaggedBound] + x , j[outsideRaggedBound] + y]
        self.pos[0,outsideRaggedBound] = bx + (self.pos[0,outsideRaggedBound]-bx)/3
        self.pos[1,outsideRaggedBound] = by + (self.pos[1,outsideRaggedBound]-by)/3

        # checks
        if check:
            self.check_pos() # check outer boundary only
            # check if exists overlap positions
            print('after pos adjustment: ')
            duplicate = 0
            for i in np.arange(self.nLGN):
                #assert(np.sum((self.pos[:,i] - self.pos[:,i+1:].T == 0).all(-1))==0)
                test = (self.pos[:,i] - self.pos[:,i+1:].T == 0).all(-1)
                nd = np.sum(test.any())
                if nd > 0:
                    duplicate = duplicate + 1
                    did = np.nonzero(test)[0] + i+1
                    print(f'#{i} is duplicated by #{did} at {self.pos[:,i]}')
            if duplicate > 0:
                print(f'after pos adjustment: {duplicate} points have duplicate(s)')
                assert(duplicate == 0)
        
    # boundary define by 4 corners
    def define_bound(self, grid):
        print('defining the boundary midway through the grid')
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

    def check_pos(self):
        # check in boundary
        i, j, d2 = self.get_ij_grid(get_d2 = True, get_coord = False)
        ipick = np.argmin(d2,1)
        # in the shell
        assert((self.Pi[i,    j][ipick == 0] > 0).all())
        assert((self.Pi[i+1,  j][ipick == 1] > 0).all())
        assert((self.Pi[i,  j+1][ipick == 2] > 0).all())
        assert((self.Pi[i+1,j+1][ipick == 3] > 0).all())

    def make_pos_uniform(self, dt, seed = None, particle_param = None, boundary_param = None, ax = None, check = True, b_scale = 1.0, p_scale = 2.5):
        subarea = self.subgrid[0] * self.subgrid[1]
        area = subarea * np.sum(self.Pi > 0)
        print(f'grid area: {area}, used in simulation')
        A = self.Pi.copy()
        A[self.Pi <= 0] = 0
        A[self.Pi > 0] = 1
        bound, btype = self.define_bound(A)

        self.pos, convergence, nlimited, _ = simulate_repel(area, self.subgrid, self.pos, dt, bound, btype, boundary_param, particle_param, ax = ax, seed = seed, ns = self.nLGN, b_scale = b_scale, p_scale = p_scale)
        if check:
            self.check_pos() # check outer boundary only

        ax.set_aspect('equal')
        self.pos_uniform = True

    def get_ij_grid(self, get_d2 = True, get_coord = True):
        print('get the index of the nearest vertex for each neuron in its own grid')

        x0 = self.x[:-1]
        x1 = self.x[1:]
        y0 = self.y[:-1]
        y1 = self.y[1:]
        # index in y-dimension
        i = np.nonzero(np.logical_and(np.tile(self.pos[1,:],(self.ny-1,1))-y0.reshape(self.ny-1,1) > 0, np.tile(self.pos[1,:],(self.ny-1,1))-y1.reshape(self.ny-1,1) <= 0).T)[1]
        assert(i.size == self.nLGN)
        # index in x-dimension
        j = np.nonzero(np.logical_and(np.tile(self.pos[0,:],(self.nx-1,1))-x0.reshape(self.nx-1,1) > 0, np.tile(self.pos[0,:],(self.nx-1,1))-x1.reshape(self.nx-1,1) <= 0).T)[1]
        assert(j.size == self.nLGN)
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
                d2 = np.zeros((4,self.nLGN))
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
                coord = np.empty((2,self.nLGN))
                print('calculate neurons\' normalized coordinate in its grid')
                coord[0,:] = (self.pos[0,:] - corner_x[i,j,0])/(corner_x[i,j,1] - corner_x[i,j,0])
                coord[1,:] = (self.pos[1,:] - corner_y[i,j,0])/(corner_y[i,j,1] - corner_y[i,j,0])

            if get_coord and get_d2:
                return i, j, d2.T, coord
            elif get_coord:
                return i, j, coord
            else:
                return i, j, d2.T

    def gen_surface(self,ax):
        if not self.pos_uniform:
            print('make pos uniform first')
        ax.plot(self.pos[0,:], self.pos[1,:],',k')
        ax.set_aspect('equal')
        coordinate = -1
        return coordinate

    def save_pos(self, pos_file):
        with open(pos_file,'wb') as f:
            np.array([self.nLGN]).tofile(f)
            self.pos.tofile(f)
