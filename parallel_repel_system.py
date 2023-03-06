from sys import stdout, argv

import multiprocessing as mp
from multiprocessing import Process, Lock, Barrier
from multiprocessing.sharedctypes import Value, RawArray
from ctypes import Structure, c_double, c_uint

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from repel_system import rec_boundary

import functools
print = functools.partial(print, flush=True)

from datetime import datetime
import numpy as np
import py_compile
py_compile.compile('parallel_repel_system.py')

class p_repel_system:
    def __init__(self, area, subgrid, initial_pos, bp, btype, p_scale, b_scale, n, nb, pid, index, barrier, lock, ax):
        self.n = n
        self.nb = nb
        self.m = index.size
        self.index = index
        self.pid = pid
        self.barrier = barrier
        self.lock = lock 

        per_unit_area = 2*np.sqrt(3) # in cl^2
        self.cl = np.sqrt((area/n)/per_unit_area)
        a_particle = self.cl*p_scale
        a_boundary = self.cl*b_scale

        self.particle = p_point_particle(initial_pos, a_particle, index.size, n)
        barrier.wait()
        self.boundary = p_rec_boundary(subgrid, bp, btype, a_boundary, pid, nb, ax)
        # will be used to limit displacement and velocity
        self.damp = 0.1
        self.default_limiting = self.cl
        self.limiting = np.zeros(self.m) + self.default_limiting
        if pid == 0:
            print(f'{self.nb} boundary points and {self.n} particles initialized')
            print(f'characteristic length (inter-particle-distance) = {self.cl}')
            print(f'in units of grids ({subgrid[0]:.3f},{subgrid[1]:.3f}):')
            print(f'    interparticle distance ({self.cl/subgrid[0]:.3f},{self.cl/subgrid[1]:.3f})')
            print(f'    radius of influence for particles ({self.particle.r0/subgrid[0]:.3f},{self.particle.r0/subgrid[1]:.3f})')
            print(f'    radius of influence for boundaries ({self.boundary.r0/subgrid[0]:.3f},{self.boundary.r0/subgrid[1]:.3f})')
            print(f'    default limiting of displacement in one dt: ({self.default_limiting/subgrid[0]:.3f}, {self.default_limiting/subgrid[1]:.3f})')

    def initialize(self):
        self.get_acc_update_limiting()
        self.barrier.wait()
        with self.lock:
            if self.pid == 0:
                print('initialized')

    def get_acc_update_limiting(self, pid):
        pos = self.particle.pos[:,self.index].copy()
        # for every particle picked
        for i in range(self.m):
            ## calculate repelling acceleratoin from the nearest boundary
            stdout.write(f'\r{(i+1)/self.m*100:.3f}%')
            stdout.flush()
            ds = self.boundary.pos[:,:,1] - pos[:,i].reshape(1,2)
            ib = np.argmin(np.sum(np.power(ds, 2), axis = -1))
            self.particle.acc[:,i], limiting = self.boundary.get_acc[ib](ib, pos[:,i])
            # update the limiting threshold
            if limiting > 0:
                self.limiting[i] =  np.min([limiting/2, self.default_limiting])

            ## accumulate acceleration from particles
            ds = pos[:,i].reshape(2,1) - self.particle.pos
            # pick a square first
            pick = np.max(np.abs(ds), axis = 0) < self.particle.r0
            pick[self.index[i]] = False
            if pick.any():
                ## calculate distance
                r = np.linalg.norm(ds[:,pick], axis=0)
                rpick = r < self.particle.r0
                pick[pick] = rpick
                ## calculate unit vector
                if rpick.any():
                    #assert((r[rpick] > 0).all())
                    ## untouched summed acceleration
                    direction = ds[:,pick]/r[rpick]
                    self.particle.acc[:,i] = self.particle.acc[:,i] + np.sum(self.particle.f(r[rpick])*direction, axis=-1)

    def next(self, dt, pid, final):
        # limitation on displacement and velocity are needed, but not acceleration
        nlimited = 0
        # get to new position by speed * dt + 1/2 * acc * dt^2
        # limit the change in displacement by the distance to the nearest boundary
        # calculate the new velocity, limit the final speed, not the change of velocity
        dpos = self.particle.vel*dt + 0.5*self.particle.acc * np.power(dt,2)
        # limit change in displacement
        dr = np.linalg.norm(dpos, axis = 0)
        large_dpos = dr > self.limiting
        dpos[:,large_dpos] = self.limiting[large_dpos] * dpos[:,large_dpos]/dr[large_dpos]
        dr[large_dpos] = self.limiting[large_dpos]

        self.particle.pos[:,self.index] = self.particle.pos[:,self.index] + dpos
        if not final:
            nlimited = np.sum(large_dpos)
            # number of freezed particles
            nfreeze = np.sum(dr == 0)
            print(f'limited {nlimited}, freezed {nfreeze}')
            # reset limiting threshold 
            self.limiting = np.zeros(self.m) + self.default_limiting
            ## update acceleration and velocity
            last_acc = self.particle.acc.copy()
            # wait for position update
            self.barrier.wait()
            self.get_acc_update_limiting()
            # wait for acc update from position
            self.barrier.wait()
            # use acc to change the course of velocity
            self.particle.vel = (1-self.damp) * self.particle.vel + 0.5*(self.particle.acc + last_acc)*dt
            # limit the absolute speed
            v = np.linalg.norm(self.particle.vel, axis = 0)
            large_vel =  v > self.limiting/dt
            self.particle.vel[:,large_vel] = self.limiting[large_vel]/dt * self.particle.vel[:,large_vel]/v[large_vel]
        else:
            self.barrier.wait()

        return self.particle.pos
    #return self.particle.pos, np.array([np.mean(dr), np.std(dr)]), nlimited, nfreeze

class p_point_particle:
    def __init__(self, initial_pos, abcl, m, n):
        self.pos = np.frombuffer(initial_pos, dtype = float).view().reshape((2,n))
        self.vel = np.zeros((2,m))
        self.acc = np.zeros((2,m))
       # potential Lennard-Jones Potential
        self.r0, self.f = get_r0_f(abcl)

class p_rec_boundary(rec_boundary):
    def __init__(self, subgrid, boundPos, boundType, abcl, pid, n, ax = None):
        self.pos = np.frombuffer(boundPos, dtype = float).view().reshape((n,2,3))
        btype = np.frombuffer(boundType, dtype = 'u4').view()
        self.subgrid = subgrid # (x,y)
        # set a radius to pick nearby particles
        self.rec = np.max(subgrid)
        if ax is not None and pid == 0:
            # connecting points on grid sides
            ax.plot(self.pos[:,0,[0,2]].squeeze(), self.pos[:,1,[0,2]].squeeze(), ',r')
            # grid centers
            ax.plot(self.pos[:,0,1].squeeze(), self.pos[:,1,1].squeeze(), ',g')

        self.get_acc = np.empty(n, dtype = object)
        for i in range(n):
            # same y-coord -> horizontal boundary
            if btype[i] == 0:
                assert((self.pos[i,1,1] - self.pos[i,1,:] == 0).all())
                self.get_acc[i] = self.get_ah
            # same x-coord -> vertical boundary
            elif btype[i] == 1:
                assert((self.pos[i,0,1] - self.pos[i,0,:] == 0).all())
                self.get_acc[i] = self.get_av
            else:
                # first horizontal then vertical
                if btype[i] == 2:
                    assert(self.pos[i,1,1] == self.pos[i,1,0])
                    self.get_acc[i] = self.get_ahv
                # first vertical then horizontal 
                else:
                    if btype[i] != 3:
                        raise Exception(f'btype: {btype[i]} is not implemented')
                    self.get_acc[i] = self.get_avh

        self.r0, self.f = get_r0_f(abcl)

def get_r0_f(abcl, k1 = 2, k2 = 1):
    r0 = np.power(k2/k1,-1/(k1-k2))*abcl
    #r0 = np.power(b*k2/a/k1,-1/(k1-k2))*cl
    #r0 = 2*abcl
    #f = lambda r: 2*np.power(abcl/r,3) - np.power(abcl/r,2)
    f = lambda r: k1*np.power(abcl/r,k1+1) - k2*np.power(abcl/r,k2+1)
    return r0, f

def parallel_repel(area, subgrid, nParticle, particlePos, p_scale, nBoundary, boundPos, boundType, b_scale, dt, pid, index, update_barrier, output_lock, spercent, figname, ndt0, seed = -1):
    if pid == 0:
        fig = plt.figure('parallel_repel', dpi = 1000)
        ax = fig.add_subplot(111)
        system = p_repel_system(area, subgrid, particlePos, boundPos, boundType, p_scale, b_scale, nParticle, nBoundary, pid, index, update_barrier, output_lock, ax)
    else:
        system = p_repel_system(area, subgrid, particlePos, boundPos, boundType, p_scale, b_scale, nParticle, nBoundary, pid, index, update_barrier, output_lock, None)

    if pid == 0:
        if spercent == 1.0:
            ns = nParticle
            spick = np.arange(ns)
        else:
            ns = int(spercent*system.particle.n)
            if seed < 0:
                seed = datetime.now()
            np.random.seed(seed)
            spick = np.random.choice(system.particle.n, ns, replace = False)
        pos = np.empty((2,2,ns))
        pos[0, :, :] = system.particle.pos[:,spick].copy()

    update_barrier.wait()
    system.initialize()

    #convergence = np.empty((dt.size,2))
    #nlimited = np.zeros(dt.size, dtype=int)
    final_limiting = system.default_limiting
    for i in range(dt.size):
        #pool.apply(system.next, args= (dt[i],p)) for p in patch
        #pos, convergence[i,:], nlimited[i], nfreeze = system.next(dt[i])
        if i < ndt0:
            system.default_limiting = final_limiting*(i+1)/(ndt0+1)
        if i == dt.size - 1:
            system.next(dt[i], True)
        else:
            system.next(dt[i], False)
        if pid == 0:
            stdout.write(f'\r{(i+1)/dt.size*100:.3f}%')
            stdout.flush()
    if pid == 0:
        stdout.write('\n')
        pos[1, :, :] = system.particle.pos[:,spick].copy()
        if spercent > 0.0:
            ax.plot(pos[:,0,:].squeeze(), pos[:,1,:].squeeze(), '-,c', lw = 0.01)

        ax.plot(system.particle.pos[0,:], system.particle.pos[1,:], ',k')
        if spercent > 0.0:
            ax.plot(pos[0,0,:], pos[0,1,:], ',m')
        ax.set_aspect('equal')
        fig.savefig(figname +'.png', dpi = 1000)
        fig = plt.figure('parallel_repel_final', dpi = 1000)
        ax = fig.add_subplot(111)
        # connecting points on grid sides
        ax.plot(system.boundary.pos[:,0,[0,2]].squeeze(), system.boundary.pos[:,1,[0,2]].squeeze(), ',r')
        # grid centers
        ax.plot(system.boundary.pos[:,0,1].squeeze(), system.boundary.pos[:,1,1].squeeze(), ',g')
        ax.plot(system.particle.pos[0,:], system.particle.pos[1,:], ',k')
        ax.set_aspect('equal')
        fig.savefig(figname +'_final.png', dpi = 1000)

def n_in_m(n, m, i):
    s = n//m
    r = np.mod(n, m)
    if i < r:
        return (s+1)*i + np.arange(s+1)
    else:
        return (s+1)*r + s*(i-r) + np.arange(s)

def simulate_repel_parallel(area, subgrid, pPos, dt, bPos, bType, b_scale, p_scale, figname, ndt0, n):
    particlePos = mp.sharedctypes.RawArray(c_double, pPos.flatten())
    boundPos = mp.sharedctypes.RawArray(c_double, bPos.flatten())
    boundType = mp.sharedctypes.RawArray(c_uint, bType)
    nParticle = pPos.shape[1]
    nBoundary = bPos.shape[0]

    update_barrier = Barrier(n)
    output_lock = Lock()
    procs = []
    for i in range(n):
        index = n_in_m(nParticle, n, i)
        print(f'patch size = {index.size}', flush = True)
        proc = Process(target = parallel_repel, args = (area, subgrid, nParticle, particlePos, p_scale, nBoundary, boundPos, boundType, b_scale, dt, i, index, update_barrier, output_lock, 1.0, figname, ndt0))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    with open(figname+'_final.bin', 'wb') as f:
        np.array([nParticle]).tofile(f)        
        np.frombuffer(particlePos, dtype=float).tofile(f)
        np.array([nBoundary]).tofile(f)        
        np.frombuffer(boundPos, dtype=float).tofile(f)
        np.frombuffer(boundType, dtype='u4').tofile(f)
        subgrid.tofile(f)
        np.array([area]).tofile(f)
    return np.frombuffer(particlePos, dtype = float).copy().reshape(2,nParticle)

if __name__ == '__main__':
    # defaults
    final_data_file = 'p_repel_finalR2.bin'
    initial_data_file = 'p_repel_finalR1.bin'
    n = 16
    p_scale = 2.0
    b_scale = 1.0
    ndt0 = 0
    ndt = 500
    tpower = 6
    figname = 'repelR'
    seed = -1

    if len(argv) > 1:
        final_data_file = argv[1]
        if len(argv) > 2:
            initial_data_file = argv[2]
            if len(argv) > 3:
                n = int(argv[3])
                if len(argv) > 4:
                    p_scale = float(argv[4])
                    if len(argv) > 5:
                        b_scale = float(argv[5])
                        if len(argv) > 6:
                            ndt = int(argv[6])
                            if len(argv) > 7:
                                tpower = int(argv[7])
    
    print(f'read data from {initial_data_file}')
    print(f'available cpu = {n}')
    print(f'p_scale = {p_scale}')
    print(f'b_scale = {b_scale}')
    print(f'dt: {ndt} of {np.power(2.0,-tpower)}')
    
    assert(n <= mp.cpu_count())
    dt = np.zeros(ndt) + np.power(2.0,-tpower)
    dt = np.hstack((np.zeros(ndt0) + dt[0]/(ndt0+1), dt)) 
    with open(initial_data_file, 'rb') as f:
        nParticle = np.fromfile(f, 'u4', count = 1)[0]
        particlePos = mp.sharedctypes.RawArray(c_double, np.fromfile(f, 'f8', count = 2*nParticle))

        nBoundary = np.fromfile(f, 'u4', count = 1)[0]
        boundPos = mp.sharedctypes.RawArray(c_double, np.fromfile(f, 'f8', count = nBoundary*2*3))
        boundType = mp.sharedctypes.RawArray(c_uint, np.fromfile(f, 'u4', count = nBoundary))
        subgrid = np.fromfile(f, 'f8', count = 2)
        area = np.fromfile(f, 'f8', count = 1)[0]

    update_barrier = Barrier(n)
    output_lock = Lock()
    procs = []
    for i in range(n):
        index = n_in_m(nParticle, n, i)
        print(f'patch size = {index.size}')
        proc = Process(target = parallel_repel, args = (area, subgrid, nParticle, particlePos, p_scale, nBoundary, boundPos, boundType, b_scale, dt, i, index, update_barrier, output_lock, 1.0, figname, ndt0, seed))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    with open(final_data_file, 'wb') as f:
        np.array([nParticle]).tofile(f)        
        np.frombuffer(particlePos, dtype=float).tofile(f)
        np.array([nBoundary]).tofile(f)        
        np.frombuffer(boundPos, dtype=float).tofile(f)
        np.frombuffer(boundType, dtype='u4').tofile(f)
        subgrid.tofile(f)
        np.array([area]).tofile(f)
