import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from sys import stdout
from scipy import integrate
import numpy as np
import time
import py_compile
from patch_geo_func import model_block_ep
py_compile.compile('LGN_surface.py')

from p_repel_system import parallel_repel
from multiprocessing.sharedctypes import RawArray
import multiprocessing as mp
from ctypes import c_double, c_int32
k = np.sqrt(140)*0.873145

class surface:
    def __init__(self, shape_file, pos_file, ecc, k = k, a = 0.635, b = 96.7):
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

        ratio = self.Pi.size/(np.sum(self.Pi>0))
        self.necc = np.round(self.nx * ratio).astype(int)
        self.npolar = np.round(self.ny * ratio).astype(int)
        self.e_range = np.exp(np.linspace(np.log(1),np.log(self.ecc+1),self.necc))-1
        self.p_range = np.linspace(-np.pi/2,np.pi/2,self.npolar)

        self.model_block = lambda p, e: model_block_ep(e,p,k,a,b)
        self.area = integrate.dblquad(self.model_block,0,self.ecc,self.p_range[0],self.p_range[-1])[0]
        self.subgrid = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])

        self.pos_uniform = False 
        self.adjusted = False

    def adjust_pos(self, bfile = None, vpfile = None, check = True):
        print('get_ij_grid')
        i, j, d2, coord = self.get_ij_grid(get_d2 = True, get_coord = True)
        ipick = np.argmin(d2,1)

        def get_22ij(index):
            ip = np.mod(index,2)
            jp = index//2
            return ip, jp
        ip, jp = get_22ij(ipick) 
        bpattern = np.array([self.Pi[i, j], self.Pi[i+1, j], self.Pi[i, j+1], self.Pi[i+1, j+1]]).T
        print(f'bpattern shape = {bpattern.shape}')
        not_rogue = np.logical_not((bpattern - np.array([0,0,0,0]) == 0).all(-1))
        bpick = np.logical_and(self.Pi[i + ip, j + jp] == 0, not_rogue)
        
        index = np.unique(np.ravel_multi_index([i[bpick], j[bpick]], (self.ny, self.nx)))
        # find all affected grids
        affected_i, affected_j = np.unravel_index(index, (self.ny, self.nx))

        per_unit_area = np.sqrt(3)/2 # in cl^2
        cl = np.sqrt((self.area/self.nLGN)/per_unit_area)
        if cl > self.subgrid[0]/2 or cl > self.subgrid[1]/2:
            cl = 0.5 * np.min(self.subgrid/2)
            print('restricting buffer length to a quarter of subgrid size')
        print(f'buffer length = {cl}, subgrid: {self.subgrid}')
        buffer_l = cl/2

        for ai in range(affected_i.size):
            stdout.write(f'\rdealing with {ai+1}/{affected_i.size} boundaries...')
            ii = affected_i[ai]
            jj = affected_j[ai]
                                # left-bot          # left-top          # right-bot         # right-top
            bpattern = np.array([self.Pi[ii, jj], self.Pi[ii+1, jj], self.Pi[ii, jj+1], self.Pi[ii+1, jj+1]])
            # left vertical bound
            if (bpattern - np.array([0,0,1,1]) == 0).all():
                #print('lv')
                gpick = np.logical_and(i == ii, j == jj)
                x_min = min(self.pos[0,gpick])
                if x_min <= (self.x[jj] + self.x[jj+1])/2:
                    rpick = np.logical_and(not_rogue, np.logical_and(i == ii, coord[1,:] > 0.5))
                    if np.sum(rpick) > 0:
                        x_max = max(self.pos[0,rpick])
                        self.pos[0,rpick] = x_max + (self.pos[0,rpick] - x_max) * ((self.x[jj] + self.x[jj+1])/2 + buffer_l - x_max)/(x_min - x_max)

                    rpick = np.logical_and(not_rogue, np.logical_and(i == ii, coord[1,:] <= 0.5))
                    if np.sum(rpick) > 0:
                        x_max = max(self.pos[0,rpick])
                        self.pos[0,rpick] = x_max + (self.pos[0,rpick] - x_max) * ((self.x[jj] + self.x[jj+1])/2 + buffer_l - x_max)/(x_min - x_max)

                    #x_min0 = min(self.pos[0,gpick])
                    #assert(x_min0 > (self.x[jj] + self.x[jj+1])/2)

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
                if x_max >= (self.x[jj] + self.x[jj+1])/2:
                    rpick = np.logical_and(not_rogue, np.logical_and(i == ii, coord[1,:] > 0.5))
                    if np.sum(rpick) > 0:
                        x_min = min(self.pos[0,rpick])

                        if x_max == x_min:
                            self.pos[0,rpick] = (self.x[jj] + self.x[jj+1])/2 - buffer_l
                        else:
                            self.pos[0,rpick] = x_min + (self.pos[0,rpick] - x_min) * ((self.x[jj] + self.x[jj+1])/2 - buffer_l - x_min)/(x_max - x_min)

                    rpick = np.logical_and(not_rogue, np.logical_and(i == ii, coord[1,:] <= 0.5))
                    if np.sum(rpick) > 0:
                        x_min = min(self.pos[0,rpick])
                        if x_max == x_min:
                            self.pos[0,rpick] = (self.x[jj] + self.x[jj+1])/2 - buffer_l
                        else:
                            self.pos[0,rpick] = x_min + (self.pos[0,rpick] - x_min) * ((self.x[jj] + self.x[jj+1])/2 - buffer_l - x_min)/(x_max - x_min)

                    #x_max0 = max(self.pos[0,gpick])
                    #assert(x_max0 < (self.x[jj] + self.x[jj+1])/2)

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
                if y_min <= (self.y[ii] + self.y[ii+1])/2:
                    rpick = np.logical_and(not_rogue, np.logical_and(j == jj, coord[0,:] > 0.5))
                    if np.sum(rpick) > 0:
                        y_max = max(self.pos[1,rpick])
                        self.pos[1,rpick] = y_max + (self.pos[1,rpick] - y_max) * ((self.y[ii] + self.y[ii+1])/2 + buffer_l - y_max)/(y_min - y_max)
                    
                    rpick = np.logical_and(not_rogue, np.logical_and(j == jj, coord[0,:] <= 0.5))
                    if np.sum(rpick) > 0:
                        y_max = max(self.pos[1,rpick])
                        self.pos[1,rpick] = y_max + (self.pos[1,rpick] - y_max) * ((self.y[ii] + self.y[ii+1])/2 + buffer_l - y_max)/(y_min - y_max)

                    #y_min0 = min(self.pos[1,gpick])
                    #assert(y_min0 > (self.y[ii] + self.y[ii+1])/2)

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
                if y_max >= (self.y[ii] + self.y[ii+1])/2:
                    rpick = np.logical_and(not_rogue, np.logical_and(j == jj, coord[0,:] > 0.5))
                    if np.sum(rpick) > 0:
                        y_min = min(self.pos[1,rpick])
                        self.pos[1,rpick] = y_min + (self.pos[1,rpick] - y_min) * ((self.y[ii] + self.y[ii+1])/2 - buffer_l - y_min)/(y_max - y_min)

                    rpick = np.logical_and(not_rogue, np.logical_and(j == jj, coord[0,:] <= 0.5))
                    if np.sum(rpick) > 0:
                        y_min = min(self.pos[1,rpick])
                        self.pos[1,rpick] = y_min + (self.pos[1,rpick] - y_min) * ((self.y[ii] + self.y[ii+1])/2 - buffer_l - y_min)/(y_max - y_min)

                    #y_max0 = max(self.pos[1,gpick])
                    #assert(y_max0 < (self.y[ii] + self.y[ii+1])/2)

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
                aind0 = np.arange(self.nLGN)[apick]
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
                xrind = rind[:nr//2]
                yrind = rind[nr//2:]

                if nr + xind.size + yind.size != aind.size:
                    raise Exception(f'{nr} + {xind.size} + {yind.size} != {aind.size}')

                pick = np.logical_and(np.logical_and(np.logical_and(i==ii, j>jj), coord[1,:] <= 0.5), not_rogue)
                xind0 = np.arange(self.nLGN)[pick]

                pick = np.logical_and(np.logical_and(np.logical_and(j==jj, i>ii), coord[0,:] <= 0.5), not_rogue)
                yind0 = np.arange(self.nLGN)[pick]

                xpick = np.hstack((axind, xind, xind0, xrind))
                x_max = max(self.pos[0,xpick])
                x_min = min(self.pos[0,xpick])
                self.pos[0,xpick] = x_max + (self.pos[0,xpick] - x_max) * ((self.x[jj] + self.x[jj+1])/2 + buffer_l - x_max)/(x_min - x_max)

                ypick = np.hstack((ayind, yind, yind0, yrind))
                y_max = max(self.pos[1,ypick])
                y_min = min(self.pos[1,ypick])
                self.pos[1,ypick] = y_max + (self.pos[1,ypick] - y_max) * ((self.y[ii] + self.y[ii+1])/2 + buffer_l - y_max)/(y_min - y_max)

                #x_min0 = min(self.pos[0,xpick])
                #assert(x_min0 > (self.x[jj] + self.x[jj+1])/2)
                #y_min0 = min(self.pos[1,ypick])
                #assert(y_min0 > (self.y[ii] + self.y[ii+1])/2)
                
                allpick = np.hstack((xpick, ypick))
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
                aind0 = np.arange(self.nLGN)[apick]
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
                xrind = rind[:nr//2]
                yrind = rind[nr//2:]

                if nr + xind.size + yind.size != aind.size:
                    raise Exception(f'{nr} + {xind.size} + {yind.size} != {aind.size}')

                pick = np.logical_and(np.logical_and(np.logical_and(i==ii, j>jj), coord[1,:] > 0.5), not_rogue)
                xind0 = np.arange(self.nLGN)[pick]

                pick = np.logical_and(np.logical_and(np.logical_and(j==jj, i<ii), coord[0,:] <= 0.5), not_rogue)
                yind0 = np.arange(self.nLGN)[pick]

                xpick = np.hstack((axind, xind, xind0, xrind))
                x_max = max(self.pos[0,xpick])
                x_min = min(self.pos[0,xpick])
                self.pos[0,xpick] = x_max + (self.pos[0,xpick] - x_max) * ((self.x[jj] + self.x[jj+1])/2 + buffer_l - x_max)/(x_min - x_max)

                ypick = np.hstack((ayind, yind, yind0, yrind))
                y_max = max(self.pos[1,ypick])
                y_min = min(self.pos[1,ypick])
                self.pos[1,ypick] = y_min + (self.pos[1,ypick] - y_min) * ((self.y[ii] + self.y[ii+1])/2 - buffer_l - y_min)/(y_max - y_min)

                #y_max0 = max(self.pos[1,ypick])
                #assert(y_max0 < (self.y[ii] + self.y[ii+1])/2)
                #x_min0 = min(self.pos[0,xpick])
                #assert(x_min0 > (self.x[jj] + self.x[jj+1])/2)

                allpick = np.hstack((xpick, ypick))
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
                aind0 = np.arange(self.nLGN)[apick]
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
                xrind = rind[:nr//2]
                yrind = rind[nr//2:]

                if nr + xind.size + yind.size != aind.size:
                    raise Exception(f'{nr} + {xind.size} + {yind.size} != {aind.size}')

                pick = np.logical_and(np.logical_and(np.logical_and(i==ii, j < jj), coord[1,:] <= 0.5), not_rogue)
                xind0 = np.arange(self.nLGN)[pick]

                pick = np.logical_and(np.logical_and(np.logical_and(j==jj, i > ii), coord[0,:] > 0.5), not_rogue)
                yind0 = np.arange(self.nLGN)[pick]

                xpick = np.hstack((axind, xind, xind0, xrind))
                x_max = max(self.pos[0,xpick])
                x_min = min(self.pos[0,xpick])
                self.pos[0,xpick] = x_min + (self.pos[0,xpick] - x_min) * ((self.x[jj] + self.x[jj+1])/2 - buffer_l - x_min)/(x_max - x_min)

                ypick = np.hstack((ayind, yind, yind0, yrind))
                y_max = max(self.pos[1,ypick])
                y_min = min(self.pos[1,ypick])
                self.pos[1,ypick] = y_max + (self.pos[1,ypick] - y_max) * ((self.y[ii] + self.y[ii+1])/2 + buffer_l - y_max)/(y_min - y_max)

                #y_min0 = min(self.pos[1,ypick])
                #assert(y_min0 > (self.y[ii] + self.y[ii+1])/2)
                #x_max0 = max(self.pos[0,xpick])
                #assert(x_max0 < (self.x[jj] + self.x[jj+1])/2)

                allpick = np.hstack((xpick, ypick))
                i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,allpick])
                i[allpick] = i_new
                j[allpick] = j_new
                d2[allpick] = d2_new
                coord[:,allpick] = coord_new
                continue

            # top-right corner
            if (bpattern - np.array([1,1,1,0]) == 0).all():
                #print('tr')
                apick = np.logical_and(i == ii, j == jj)
                aind0 = np.arange(self.nLGN)[apick]
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
                xrind = rind[:nr//2]
                yrind = rind[nr//2:]

                if nr + xind.size + yind.size != aind.size:
                    raise Exception(f'{nr} + {xind.size} + {yind.size} != {aind.size}')

                pick = np.logical_and(np.logical_and(np.logical_and(i==ii, j < jj), coord[1,:] > 0.5), not_rogue)
                xind0 = np.arange(self.nLGN)[pick]

                pick = np.logical_and(np.logical_and(np.logical_and(j==jj, i < ii), coord[0,:] > 0.5), not_rogue)
                yind0 = np.arange(self.nLGN)[pick]

                xpick = np.hstack((axind, xind, xind0, xrind))
                x_max = max(self.pos[0,xpick])
                x_min = min(self.pos[0,xpick])
                self.pos[0,xpick] = x_min + (self.pos[0,xpick] - x_min) * ((self.x[jj] + self.x[jj+1])/2 - buffer_l - x_min)/(x_max - x_min)

                ypick = np.hstack((ayind, yind, yind0, yrind))
                y_max = max(self.pos[1,ypick])
                y_min = min(self.pos[1,ypick])
                self.pos[1,ypick] = y_min + (self.pos[1,ypick] - y_min) * ((self.y[ii] + self.y[ii+1])/2 - buffer_l - y_min)/(y_max - y_min)

                #y_max0 = max(self.pos[1,ypick])
                #assert(y_max0 < (self.y[ii] + self.y[ii+1])/2)
                #x_max0 = max(self.pos[0,xpick])
                #assert(x_max0 < (self.x[jj] + self.x[jj+1])/2)

                allpick = np.hstack((xpick, ypick))
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
                change = False 
                y_max = max(self.pos[1,gpick])
                if y_max > (self.y[ii] + self.y[ii+1])/2:
                    ppick = np.logical_and(j == jj, not_rogue) # squeeze y direction first
                    y_min = min(self.pos[1,ppick])
                    self.pos[1,ppick] = y_min + (self.pos[1,ppick] - y_min) * ((self.y[ii] + self.y[ii+1])/2 - buffer_l - y_min)/(y_max - y_min)

                    #y_max0 = max(self.pos[1,gpick])
                    #assert(y_max0 < (self.y[ii] + self.y[ii+1])/2)
                    opick = np.logical_or(opick, ppick)
                    change = True
                x_max = max(self.pos[0,gpick])
                if x_max > (self.x[jj] + self.x[jj+1])/2:
                    ppick = np.logical_and(np.logical_and(i==ii, np.logical_or(j == jj, np.logical_and(j < jj, coord[1,:] < 0.5))), not_rogue) # then x, so that more accommodated in x direction
                    x_min = min(self.pos[0,ppick])
                    self.pos[0,ppick] = x_min + (self.pos[0,ppick] - x_min) * ((self.x[jj] + self.x[jj+1])/2 - buffer_l - x_min)/(x_max - x_min)

                    #x_max0 = max(self.pos[0,gpick])
                    #assert(x_max0 < (self.x[jj] + self.x[jj+1])/2)
                    opick = np.logical_or(opick, ppick)
                    change = True
                if change:
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
                change = False 
                y_min = min(self.pos[1,gpick])
                if y_min < (self.y[ii] + self.y[ii+1])/2:
                    ppick = np.logical_and(j == jj, not_rogue)
                    y_max = max(self.pos[1,ppick])
                    self.pos[1,ppick] = y_max + (self.pos[1,ppick] - y_max) * ((self.y[ii] + self.y[ii+1])/2 + buffer_l - y_max)/(y_min - y_max)

                    #y_min0 = min(self.pos[1,gpick])
                    #assert(y_min0 > (self.y[ii] + self.y[ii+1])/2)
                    opick = np.logical_or(opick, ppick)
                    change = True 
                x_max = max(self.pos[0,gpick])
                if x_max > (self.x[jj] + self.x[jj+1])/2:
                    ppick = np.logical_and(np.logical_and(i==ii, np.logical_or(j == jj, np.logical_and(j < jj, coord[1,:] > 0.5))), not_rogue) # then x, so that more accommodated in x direction
                    x_min = min(self.pos[0,ppick])
                    self.pos[0,ppick] = x_min + (self.pos[0,ppick] - x_min) * ((self.x[jj] + self.x[jj+1])/2 - buffer_l - x_min)/(x_max - x_min)

                    #x_max0 = max(self.pos[0,gpick])
                    #assert(x_max0 < (self.x[jj] + self.x[jj+1])/2)
                    opick = np.logical_or(opick, ppick)
                    change = True
                if change:
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
                change = False
                y_max = max(self.pos[1,gpick])
                if y_max > (self.y[ii] + self.y[ii+1])/2:
                    ppick = np.logical_and(j == jj, not_rogue)
                    y_min = min(self.pos[1,ppick])
                    self.pos[1,ppick] = y_min + (self.pos[1,ppick] - y_min) * ((self.y[ii] + self.y[ii+1])/2 - buffer_l - y_min)/(y_max - y_min)

                    #y_max0 = max(self.pos[1,gpick])
                    #assert(y_max0 < (self.y[ii] + self.y[ii+1])/2)
                    opick = np.logical_or(opick, ppick)
                    change = True 
                x_min = min(self.pos[0,gpick])
                if x_min < (self.x[jj] + self.x[jj+1])/2:
                    ppick = np.logical_and(np.logical_and(i==ii, np.logical_or(j == jj, np.logical_and(j > jj, coord[1,:] < 0.5))), not_rogue) # then x, so that more accommodated in x direction
                    x_max = max(self.pos[0,ppick])
                    self.pos[0,ppick] = x_max + (self.pos[0,ppick] - x_max) * ((self.x[jj] + self.x[jj+1])/2 + buffer_l - x_max)/(x_min - x_max)

                    #x_min0 = min(self.pos[0,gpick])
                    #assert(x_min0 > (self.x[jj] + self.x[jj+1])/2)
                    opick = np.logical_or(opick, ppick)
                    change = True 
                if change:
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
                change = False
                y_min = min(self.pos[1,gpick])
                if y_min < (self.y[ii] + self.y[ii+1])/2:
                    ppick = np.logical_and(j == jj, not_rogue)
                    y_max = max(self.pos[1,ppick])
                    self.pos[1,ppick] = y_max + (self.pos[1,ppick] - y_max) * ((self.y[ii] + self.y[ii+1])/2 + buffer_l - y_max)/(y_min - y_max)

                    #y_min0 = min(self.pos[1,gpick])
                    #assert(y_min0 > (self.y[ii] + self.y[ii+1])/2)
                    opick = np.logical_or(opick, ppick)
                    change = True 
                x_min = min(self.pos[0,gpick])
                if x_min < (self.x[jj] + self.x[jj+1])/2:
                    ppick = np.logical_and(np.logical_and(i==ii, np.logical_or(j == jj, np.logical_and(j > jj, coord[1,:] > 0.5))), not_rogue) # then x, so that more accommodated in x direction
                    x_max = max(self.pos[0,ppick])
                    self.pos[0,ppick] = x_max + (self.pos[0,ppick] - x_max) * ((self.x[jj] + self.x[jj+1])/2 + buffer_l - x_max)/(x_min - x_max)

                    #x_min0 = min(self.pos[0,gpick])
                    #assert(x_min0 > (self.x[jj] + self.x[jj+1])/2)
                    opick = np.logical_or(opick, ppick)
                    change = True 
                if change:
                    i_new, j_new, d2_new, coord_new = self.get_ij_grid_local(self.pos[:,opick])
                    i[opick] = i_new
                    j[opick] = j_new
                    d2[opick] = d2_new
                    coord[:,opick] = coord_new
                continue
        stdout.write('done.\n')
                        
        i, j, d2 = self.get_ij_grid(get_d2 = True, get_coord = False)
        ipick = np.argmin(d2,1)
        ip, jp = get_22ij(ipick) 
        dropped = False
        bpick = self.Pi[i + ip, j + jp]
        for ind in range(4):
            pick = bpick[ipick == ind] == 0
            if np.sum(pick) > 0:
                print(f'@{ind}, {np.sum(pick)} drops out')
                dropped = True
        if dropped:
            print(self.pos[:,bpick == 0].T)
            print('adjust_pos not doing the job')
            #raise Exception('adjust_pos not doing the job')
        else:
            print('pos adjusted')

        self.bound, self.btype = self.define_bound(self.Pi)

        if bfile is not None:
            with open(bfile + '-bound.bin','wb') as f:
                nb = self.bound.shape[0]
                np.array([nb]).astype('i4').tofile(f)
                self.bound.copy().astype(float).tofile(f)

        if vpfile is not None:
            with open(vpfile + '-adjusted.bin','wb') as f:
                np.array([1, self.nLGN, 2]).astype('u4').tofile(f)        
                pos = np.empty((1, 2, self.nLGN))
                pos[:,0,:] = self.pos[0,:].reshape(1,self.nLGN)
                pos[:,1,:] = self.pos[1,:].reshape(1,self.nLGN)
                pos.tofile(f)

        self.adjusted = True

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

        if not (self.Pi[i,    j][ipick == 0] > 0).all():
            raise Exception(f'{np.sum(self.Pi[i, j][ipick == 0] == 0)} bot-left leak')

        if not (self.Pi[i+1,  j][ipick == 1] > 0).all():
            raise Exception(f'{np.sum(self.Pi[i+1, j][ipick == 1] == 0)} top-left leak')

        if not (self.Pi[i,  j+1][ipick == 2] > 0).all():
            raise Exception(f'{np.sum(self.Pi[i, j+1][ipick == 2] == 0)} bot-right leak')

        if not (self.Pi[i+1,j+1][ipick == 3] > 0).all():
            raise Exception(f'{np.sum(self.Pi[i+1,j+1][ipick == 3] == 0)} top-right leak')

    def make_pos_uniform(self, dt, p_scale, b_scale, figname, ncore = 0, ndt_decay = 0, roi_ratio = 2.0, k1 = 1.0, k2 = 0.5, chop_ratio = 1, spercent = 0.01, seed = -1, bfile = None, vpfile = None, local_plot = False, check = True):
        if not self.adjusted:
            self.adjust_pos(bfile = bfile, vpfile = vpfile)

        if ncore == 0:
            ncore = mp.cpu_count()
            print(f'{ncore} cores found')

        nx = int(np.ceil(self.nx*chop_ratio))
        ny = int(np.ceil(self.ny*chop_ratio))
        if nx*ny < ncore:
            nx = int(np.ceil(np.sqrt(ncore) * self.nx/self.ny))
            ny = int(np.ceil(np.sqrt(ncore) * self.ny/self.nx))

        ##  prepare shared memory
        raw_shared_cmem = RawArray(c_int32, np.zeros(ncore, dtype = 'i4'))
        shared_cmem = np.frombuffer(raw_shared_cmem, dtype = 'i4')

        raw_shared_dmem = RawArray(c_double, np.zeros(self.nLGN*8, dtype = 'f8'))
        raw_shared_imem = RawArray(c_int32, np.zeros(self.nLGN, dtype = 'i4'))
        shared_dmem = np.frombuffer(raw_shared_dmem, dtype = 'f8')
        shared_imem = np.frombuffer(raw_shared_imem, dtype = 'i4')

        # populate position in shared array and assign chop references
        pos = np.empty(1, dtype = object)
        shared_dmem[:2*self.nLGN] = self.pos.flatten()
        pos[0] = shared_dmem[:2*self.nLGN].view().reshape((2,self.nLGN))

        raw_shared_bmem = RawArray(c_double, np.zeros(nx*ny*6, dtype = 'f8'))
        shared_bmem = np.frombuffer(raw_shared_bmem, dtype = 'f8') 
        raw_shared_nmem = RawArray(c_int32, np.zeros(nx*ny*6, dtype = 'i4')) # for neighbor_list and id
        shared_nmem = np.frombuffer(raw_shared_nmem, dtype = 'i4') 

        subarea = self.subgrid[0] * self.subgrid[1]
        area = subarea * np.sum(self.Pi > 0)
        print(f'grid area: {area}, used in simulation')
        A = self.Pi.copy()
        A[self.Pi <= 0] = 0
        A[self.Pi > 0] = 1
        bound, btype = self.define_bound(A)

        self.pos = parallel_repel(area, self.subgrid, pos, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, p_scale, bound, btype, b_scale, nx, ny, ncore, dt, spercent, figname, ndt_decay, 1.0, roi_ratio, k1, k2, seed, 1.0, local_plot)

        if vpfile is not None:
            with open(vpfile+'-us.bin','wb') as f:
                np.array([1, self.nLGN]).astype('i4').tofile(f)
                self.pos.tofile(f)

        if check:
            self.check_pos() # check outer boundary only
        self.pos_uniform = True


    def get_ij_grid(self, get_d2 = True, get_coord = True):
        print('get the index of the nearest vertex for each neuron in its own grid')

        x0 = self.x[:-1]
        x1 = self.x[1:]
        y0 = self.y[:-1]
        y1 = self.y[1:]
        # index in y-dimension
        ipick = np.logical_and(np.tile(self.pos[1,:],(self.ny-1,1))-y0.reshape(self.ny-1,1) > 0, np.tile(self.pos[1,:],(self.ny-1,1))-y1.reshape(self.ny-1,1) <= 0).T
        i = np.zeros(self.nLGN, dtype = 'i4')
        pick = ipick.any(-1)
        i[pick] = np.nonzero(ipick)[1]
        if not np.sum(pick) == self.nLGN:
            nrogue = self.nLGN - np.sum(pick)
            print(f'{nrogue} particles out of {self.pos.shape[1]} have gone rogue')
            if not np.sum(self.pos[1,:] == self.y[0]) == nrogue:
                print(f'range: {np.max(pos[1,:])} > {self.y[-1]}, {np.min(pos[1,:])} <= {self.y[0]}')
                raise Exception(f'cant restrain rogue particles')
        print('indexed in y-dimension')

        # index in x-dimension
        jpick = np.logical_and(np.tile(self.pos[0,:],(self.nx-1,1))-x0.reshape(self.nx-1,1) > 0, np.tile(self.pos[0,:],(self.nx-1,1))-x1.reshape(self.nx-1,1) <= 0).T
        j = np.zeros(self.nLGN, dtype = 'i4')
        pick = jpick.any(-1)
        j[pick] = np.nonzero(jpick)[1]
        if not np.sum(pick) == self.nLGN:
            nrogue = self.nLGN - np.sum(pick)
            print(f'{nrogue} particles out of {self.pos.shape[1]} have gone rogue')
            if not np.sum(self.pos[0,:] == self.x[0]) == nrogue:
                print(f'range: {np.max(pos[0,:])} > {self.x[-1]}, {np.min(pos[0,:])} <= {self.x[0]}')
                raise Exception(f'cant restrain rogue particles')
        print('indexed in x-dimension')

        
        outcast = np.stack((self.Pi[i,j]<=0, self.Pi[i+1,j]<=0, self.Pi[i,j+1]<=0, self.Pi[i+1,j+1]<=0), axis = -1).all(-1)
        nout = np.sum(outcast)
        if nout > 0:
            raise Exception(f'{nout} outcasts found')

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

        outcast = np.stack((self.Pi[i,j]<=0, self.Pi[i+1,j]<=0, self.Pi[i,j+1]<=0, self.Pi[i+1,j+1]<=0), axis = -1).all(-1)
        nout = np.sum(outcast)
        if nout > 0:
            raise Exception(f'{nout} outcasts found')

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

    def plot_surface(self,ax):
        subarea = self.subgrid[0] * self.subgrid[1]
        area = subarea * np.sum(self.Pi > 0)
        print(f'grid area: {area}, used in simulation')
        A = self.Pi.copy()
        A[self.Pi <= 0] = 0
        A[self.Pi > 0] = 1
        bound, btype = self.define_bound(A)
        # connecting points on grid sides
        ax.plot(bound[:,0,[0,2]].squeeze(), bound[:,1,[0,2]].squeeze(), ',r')
        # grid centers
        ax.plot(bound[:,0,1].squeeze(), bound[:,1,1].squeeze(), ',g')
        # plot particles
        ax.plot(self.pos[0,:], self.pos[1,:],',k')
        ax.set_aspect('equal')
        coordinate = -1
        return coordinate

    def save(self, pos_file = None, parallel_file = None):
        if pos_file is not None:
            with open(pos_file,'wb') as f:
                np.array([self.nLGN]).tofile(f)
                self.pos.tofile(f)
        if parallel_file is not None:
            subarea = self.subgrid[0] * self.subgrid[1]
            area = subarea * np.sum(self.Pi > 0)
            A = self.Pi.copy()
            A[self.Pi <= 0] = 0
            A[self.Pi > 0] = 1
            boundPos, btype = self.define_bound(A)
            with open(parallel_file, 'wb') as f:
                np.array([self.nLGN]).astype('u4').tofile(f)
                self.pos.tofile(f)
                np.array([btype.size]).astype('u4').tofile(f)
                boundPos.tofile(f)
                btype.astype('u4').tofile(f)
                self.subgrid.tofile(f)
                np.array([area]).astype('f8').tofile(f)
