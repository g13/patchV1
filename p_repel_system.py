from sys import stdout, argv

import multiprocessing as mp
from multiprocessing import Process, Pool, Lock, Barrier
from multiprocessing.sharedctypes import Value, RawArray
from ctypes import Structure, c_double, c_int, c_uint, c_int32
#from viztracer import VizTracer
#import pprofile

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from repel_system import rec_boundary

import functools
print = functools.partial(print, flush=True)
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer

from datetime import datetime
import numpy as np
import py_compile
py_compile.compile('parallel_repel_system.py')

class p_repel_system:
    def __init__(self, area, subgrid, initial_pos, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, bp, btype, p_scale, b_scale, nx, ny, roi_ratio = 2.0, k1 = 1.0, k2 = 0.5, ncore = 0, limit_ratio = 1.0, damp = 0.5, l_ratio = 0.0, l_k = 1.0, ax = None):
        self.ncore = ncore
        per_unit_area = np.sqrt(3)/2 # in cl^2

        chop_size = np.array([x.shape[1] for x in initial_pos])
        self.n = sum(chop_size)
        self.cl = np.sqrt((area/self.n)/per_unit_area)
        a_particle = self.cl*p_scale
        a_boundary = self.cl*b_scale
        if roi_ratio <= 0:
            print('roi_ratio must be positive, set to 2 as default')
            roi_ratio = 2.0
        roi_ratio = roi_ratio/p_scale
        self.nb = btype.size

        self.boundary = p_rec_boundary(subgrid, bp, btype, a_boundary, self.nb, ax)
        self.particle = p_point_particle(initial_pos, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, self.n, chop_size, a_particle, self.boundary.xl, self.boundary.yl, nx, ny, k1, k2, roi_ratio, self.ncore)

        # will be used to limit displacement and velocity
        self.damp = damp
        self.l_ratio = l_ratio
        self.l_k = l_k
        self.default_limiting = self.cl/2 * limit_ratio
        #self.limiting = np.zeros(self.m) + self.default_limiting

        print(limit_ratio)
        print(f'{self.nb} boundary points and {self.particle.n} particles initialized')
        print(f'characteristic length (inter-particle-distance) = {self.cl}')
        print(f'in units of grids ({subgrid[0]:.3f},{subgrid[1]:.3f}):')
        print(f'    interparticle distance ({self.cl/subgrid[0]:.3f},{self.cl/subgrid[1]:.3f})')
        print(f'    radius of influence for particles ({self.particle.r0/subgrid[0]:.3f},{self.particle.r0/subgrid[1]:.3f})')
        print(f'    radius of influence for boundaries ({self.boundary.r0/subgrid[0]:.3f},{self.boundary.r0/subgrid[1]:.3f})')
        print(f'    default limiting of displacement in one dt: ({self.default_limiting/subgrid[0]:.3f}, {self.default_limiting/subgrid[1]:.3f})')

    def update_vel_acc_limiting(self, dt, cid, pid, ichop, nchop, b_ratio, lock):
        if self.particle.chop_size[cid] > 0:
            last_acc = self.particle.acc[cid].copy()
            vel = self.particle.vel[cid].view()
            limiting = self.particle.limiting[cid].view()
            self.update_acc_limiting(cid, pid, ichop, nchop, b_ratio, lock)
            # use acc to change the course of velocity
            #print(f'limiting #{pid}: {[np.min(self.particle.interdis[cid]), np.mean(self.particle.interdis[cid]), np.max(self.particle.interdis[cid])]/self.cl}:{self.cl}')
            damp = self.damp * self.cl/self.particle.interdis[cid]
            damp[damp > 1] = 1
            self.particle.dis_intended[cid] = np.mean(damp)
            self.particle.dis[cid] = np.std(damp)
            #damp[self.cl < self.particle.interdis[cid]] = 0
            #print(f'damp #{cid}: {[np.min(damp), np.mean(damp), np.max(damp)]}: {self.damp}')
            vel[:,:] = (vel[:,:] + (1-damp) * vel[:,:] + 0.5*(self.particle.acc[cid] + last_acc)*dt + (damp*damp*vel[:,:] - damp*last_acc)*dt*dt)/2
            #vel[:,:] = (1-self.damp) * vel[:,:] + 0.5*(self.particle.acc[cid] + last_acc)*dt
            #print(vel)
            # limit the absolute speed
            v = np.linalg.norm(vel, axis = 0)
            l_ratio = self.particle.interdis[cid]/self.cl
            #l_ratio[l_ratio < self.l_ratio] = 0
            l_ratio[l_ratio < self.l_ratio] = self.l_k * l_ratio[l_ratio < self.l_ratio]
            l_ratio[l_ratio > 1] = 1
            #l_ratio = 1
            large_vel =  v > limiting/dt * l_ratio
            vel[:,large_vel] = limiting[large_vel]/dt * vel[:,large_vel]/v[large_vel]

    def update_acc_limiting(self, cid, pid, ichop, nchop, b_ratio, lock):
        # no need to copy() pos here no write will happen here
        #if pid == self.printing_id:
        #    prof = pprofile.Profile()
        #    prof.enable()
        if self.particle.chop_size[cid] == 0:
            print(f'chop {cid} has zero particle')
            return
        pos = self.particle.pos[cid].view()
        # acc is free to update
        acc = self.particle.acc[cid].view()
        nint = np.zeros(self.particle.chop_size[cid])
        #r_max = np.zeros(self.particle.chop_size[cid])
        r_min = np.zeros(self.particle.chop_size[cid])
        #gotit = False
        #j = -1
        for i in range(self.particle.chop_size[cid]):
            #if i == 0:
            #    with lock:
            #        print(f'{cid}: acc[0] before {acc[:,i]}')
            ## calculate repelling acceleratoin from the nearest boundary
            if pid == self.printing_id:
                percent = (i+1)/self.particle.chop_size[cid]*100
                mod_size = max([self.particle.chop_size[cid]//10, 2])
                if (np.mod(i+1,mod_size) == 0 or i+1 == self.particle.chop_size[cid]):
                    stdout.write(f'\r acc: {ichop}/{nchop} - {i+1}/{self.particle.chop_size[cid]},{percent:.1f}%')
                    stdout.flush()

            #ds = np.mean(self.boundary.pos[:,:,:],axis=-1) - pos[:,i].reshape(1,2) # use the mean of the three guide points to detect cross-like boundaries.
            #ds = self.boundary.pos[:,:,1] - pos[:,i].reshape(1,2)
            #ib = np.argmin(np.sum(np.power(ds, 2), axis = -1))
            ds = self.boundary.midpos - np.tile(pos[:,i].reshape(2,1),(1,2))
            ib = np.argmin(np.min(np.sum(np.power(ds, 2), axis = -2), axis = -1))
            acc[:,i], limiting = self.boundary.get_acc(ib, pos[:,i])
            #if (acc[:,i] != 0).any():
            #    near_boundary = True
            #else:
            #    near_boundary = False 

            acc[:,i] *= b_ratio
            r0 = min(limiting, self.particle.r0)
            
            # update the limiting threshold
            self.particle.limiting[cid][i] =  np.min([limiting, self.default_limiting])

            nint[i] = 0
            ## accumulate acceleration from particles
            ds = pos[:,i].reshape(2,1) - pos
            pick = np.max(np.abs(ds), axis = 0) < r0
            pick[i] = False
            r = np.linalg.norm(ds[:,pick], axis=0)
            rpick = r < r0
            if rpick.any():
                pick[pick] = rpick
                direction = ds[:,pick]/r[rpick]
                #r_max[i] = np.max(r[rpick])
                acc[:,i] = acc[:,i] + np.sum(self.particle.f(r[rpick], self.cl)*direction, axis=-1)
                nint[i] += np.sum(rpick)
            rlist = r[rpick].tolist()

            # accumulate acceleration from neighboring particles
            npick = self.particle.neighbor_list[self.particle.chop_bid[cid],:] >= 0
            ids = self.particle.neighbor_list[self.particle.chop_bid[cid],npick]

            for neighbor in self.particle.block_cid[ids]:
                if neighbor == -1:
                    continue
                neighbor_center = np.array([np.sum(self.particle.block_bound[0,self.particle.chop_bid[neighbor],:])/2, np.sum(self.particle.block_bound[1,self.particle.chop_bid[neighbor],:])/2])
                # no need to copy() no write will happen here
                dis_neighbor = np.linalg.norm(neighbor_center - pos[:,i])
                if self.particle.block_halfDiag + r0 > dis_neighbor:
                    neighbor_pos = self.particle.pos[neighbor]
                    ds = pos[:,i].reshape(2,1) - neighbor_pos
                    # pick a square first
                    pick = np.max(np.abs(ds), axis = 0) < r0
                    if pick.any():
                        # calculate distance
                        r = np.linalg.norm(ds[:,pick], axis=0)
                        rpick = r < r0
                        r[r==0] = np.finfo(float).eps
                        # calculate unit vector
                        if rpick.any():
                            pick[pick] = rpick
                            # untouched summed acceleration
                            direction = ds[:,pick]/r[rpick]
                            acc[:,i] = acc[:,i] + np.sum(self.particle.f(r[rpick], self.cl)*direction, axis=-1)
                            nint[i] += np.sum(rpick)
                            r_neighbor_max = np.max(r[rpick])
                            #if r_neighbor_max > r_max[i]:
                            #    r_max[i] = r_neighbor_max
                        rlist = rlist +  r[rpick].tolist()
            nmin = min(len(rlist), 6)
            if nmin > 0:
                r_min[i] = np.mean(np.partition(rlist, nmin-1)[:nmin])
            else:
                r_min[i] = r0
            #if near_boundary:
            #    print(f'near_boundary damp = {self.damp*self.cl/r_min[i]}, limiting = {self.particle.limiting[cid][i]*r_min[i]/self.cl}')
            #if r_max[i] == 0:
            #    r_max[i] = r0

            #if j == i: 
            #    print(f'#{i}: {acc[:,i]} total from {nint[i]}, r0 = {r0}') 

            #if i == 0:
            #    with lock:
            #        print(f'{cid}: acc[0] after {self.particle.acc[cid][:,i]}')
        self.particle.nint[cid] = np.mean(nint)
        nint0 = nint.copy()
        nint[nint == 0] = 1
        self.particle.interdis[cid][:] = r_min
        #self.particle.interdis[cid][:] = r_max/np.sqrt(nint)
        #if pid == self.printing_id:
        #    prof.disable()
        #    prof.print_stats()

    def update_pos(self, dt, cid, final, lock):
        # limitation on displacement and velocity are needed, but not acceleration
        if self.particle.chop_size[cid] == 0:
            print(f'chop {cid} has zero particle')
            return 

        pos = self.particle.pos[cid].view()

        #pos0 = pos[:,np.array([0,1,2])].copy()
        #pos_shared = self.particle.shared_dmem[:3].copy()
        #acc0 = self.particle.acc[cid][:,np.array([0,1,2])]

        l_ratio = self.particle.interdis[cid]/self.cl
        #l_ratio[l_ratio < self.l_ratio] = 0
        l_ratio[l_ratio < self.l_ratio] = self.l_k * l_ratio[l_ratio < self.l_ratio]
        l_ratio[l_ratio > 1] = 1
        #l_ratio = 1
        limiting = self.particle.limiting[cid]*l_ratio
        #print(f'limiting #{cid}: {[np.min(limiting), np.mean(limiting), np.max(limiting)]/self.default_limiting}')

        # get to new position by speed * dt + 1/2 * acc * dt^2
        # limit the change in displacement by the distance to the nearest boundary
        # calculate the new velocity, limit the final speed, not the change of velocity
        
        dpos = self.particle.vel[cid]*dt
        #print(f'dpos component {[np.mean(self.particle.vel[cid]*dt), np.mean(0.5*self.particle.acc[cid] * np.power(dt,2))]}')
        # limit change in displacement
        dr = np.linalg.norm(dpos, axis = 0)
        self.particle.dis_intended[cid] = np.mean(dr)
        large_dpos = dr > limiting
        dpos[:,large_dpos] = limiting[large_dpos] * dpos[:,large_dpos]/dr[large_dpos]
        dr[large_dpos] = limiting[large_dpos]
        self.particle.dis[cid] = np.mean(dr)
        #if cid == 0:
            #old_pos = pos[0,:3].copy()
            #print(f'check dpos: {old_pos}')
        pos[:,:] = pos[:,:] + dpos
        #if cid == 0:
        #    print(f'check dpos: {pos[0,:3]- old_pos}')
        #    print(f'check dpos: {self.particle.shared_dmem[:3]}, {pos[0,:3]}, {self.particle.pos[0][0,:3]}')
        #    print(f'check dpos: {dpos[0,:3]} = {0.5*self.particle.acc[cid][0,:3]} * {np.power(dt,2)} = {0.5*self.particle.acc[cid][0,:3] * np.power(dt,2)}')

        # number of limited and frozen particles
        self.particle.nfrozen[cid] = np.sum(dr < self.default_limiting*0.01)
        self.particle.nlimited[cid] = np.sum(large_dpos)
        #with lock:
            #print(f'{cid}: change in pos[:,[0,1,2]] = {pos0 - pos[:,np.array([0,1,2])]}, cl = {self.particle.abcl}, dpos = {dpos[:,np.array([0,1,2])]}')

        if not final:
        # update chop_pos
            center_pos = np.mean(self.particle.pos[cid], axis=1)
            self.particle.chop_pos[:,cid] = center_pos
            self.particle.chop_maxD[cid] = np.max(np.linalg.norm(center_pos.reshape(2,1)-pos, axis = 0))

    def repel_near_chop(self, pid, dt, ndt_decay, cl_ratio, update_barrier, output_lock, nchop_total, mod_plot_fn = 'mod_plot'):
        # when simulating chop, update every loop
        chunk = self.particle.nchop//self.ncore
        remain = np.mod(self.particle.nchop, self.ncore)
        chunksize = chunk + (pid < remain)
        if pid < remain:
            chops_to_update = chunksize * pid + np.arange(chunksize)
        else:
            chops_to_update = (chunksize + 1) * remain +  (pid-remain) * chunksize + np.arange(chunksize)
        update_barrier.wait()
        self.particle.core_workload[pid] = np.sum(self.particle.chop_size[chops_to_update])
        update_barrier.wait()
        if pid == 0:
            print(f'workload c.v.: {np.std(self.particle.core_workload)/np.mean(self.particle.core_workload)}')

        self.printing_id = np.argmax(self.particle.core_workload)
        if pid == self.printing_id:
            print(f'chops: {chunk+1}x{remain}; {chunk}x{self.ncore-remain}')
            seed = int(datetime.now().timestamp())
            np.random.seed(seed)
        # when chopping up, fixed
        chunk = self.particle.nblock//self.ncore
        remain = np.mod(self.particle.nblock, self.ncore)
        blocksize = chunk + (pid < remain)
        if pid == 0:
            print([self.particle.nblock, self.ncore, chunk, remain, blocksize])
        if pid < remain:
            blocks_to_update = blocksize * pid + np.arange(blocksize)
        else:
            blocks_to_update = (blocksize + 1) * remain +  (pid-remain) * blocksize + np.arange(blocksize)
        if pid == self.printing_id:
            print(f'blocks: {chunk+1}x{remain}; {chunk}x{self.ncore-remain}')

        if pid == self.printing_id:
            tic = time.perf_counter() 

        # reach steady state limiting after ndt_decay
        if ndt_decay > 0:
            ss_limiting = self.default_limiting
            ss_cl = self.cl
            r0 = self.particle.r0
            # set
            self.default_limiting = ss_limiting*cl_ratio
            self.cl = ss_cl*cl_ratio
            self.particle.r0 = r0*cl_ratio
            if cl_ratio == 1:
                b0 = 0
                b_ratio = b0 + 1
            else:
                b_ratio = 1/cl_ratio
        else:
            b_ratio = 1

        # steady state limiting, cl, and r0
        # update acc and limiting
        nchop = chops_to_update.size
        for cid in chops_to_update:
            ichop = cid - chops_to_update[0] + 1
            self.update_acc_limiting(cid, pid, ichop, nchop, b_ratio, output_lock)
        update_barrier.wait()
        if pid == self.printing_id:
            toc = time.perf_counter() 
            stdout.write('\n')
            stdout.flush()
            print(f'initial acc limiting update took approx. {toc-tic:0.4f} sec')
            print(f'{dt.size} iterations to go')
            #print(f'mid pos: {self.particle.shared_dmem[spick]}')
            #print(f'mid vel: {self.particle.shared_dmem[2*self.particle.n + spick]}')
            #print(f'mid acc: {self.particle.shared_dmem[4*self.particle.n + spick]}')

        for i in range(dt.size):
        #   parallel update pos
            if pid == self.printing_id:
                print(f'iteration {i}')
                print(f'{self.nb} boundary points and {self.particle.n} particles initialized')
                print(f'b_ratio = {b_ratio}')
                print(f'characteristic length (inter-particle-distance) = {self.cl}')
                print(f'in units of grids ({self.boundary.subgrid[0]:.3f},{self.boundary.subgrid[1]:.3f}):')
                print(f'    interparticle distance ({self.cl/self.boundary.subgrid[0]:.3f},{self.cl/self.boundary.subgrid[1]:.3f})')
                print(f'    radius of influence for particles ({self.particle.r0/self.boundary.subgrid[0]:.3f},{self.particle.r0/self.boundary.subgrid[1]:.3f})')
                print(f'    radius of influence for boundaries ({self.boundary.r0/self.boundary.subgrid[0]:.3f},{self.boundary.r0/self.boundary.subgrid[1]:.3f})')
                print(f'    default limiting of displacement in one dt: ({self.default_limiting/self.boundary.subgrid[0]:.3f}, {self.default_limiting/self.boundary.subgrid[1]:.3f})')
                tic = time.perf_counter() 
            for cid in chops_to_update:
                self.update_pos(dt[i], cid, i == dt.size-1, output_lock)
            update_barrier.wait()
            if pid == self.printing_id:
                np.sum(self.particle.nfrozen)
                toc = time.perf_counter() 
                print(f'update pos took approx. {toc-tic:0.4f} sec, each particle travel {np.mean(self.particle.dis_intended):.3e}/{np.mean(self.particle.dis):.3e} (would/limited) mm , limited by {self.default_limiting:.3e} mm, #interacting particle on average: {np.mean(self.particle.nint):.1f}')
                print(f'limited {np.sum(self.particle.nlimited)/self.particle.n*100:.5f}%, {np.sum(self.particle.nfrozen)/self.particle.n*100:.2f}% frozen')
                #print(f'after pos: {self.particle.shared_dmem[spick]}')
                #print(f'after vel: {self.particle.shared_dmem[2*self.particle.n + spick]}')
                #print(f'after acc: {self.particle.shared_dmem[4*self.particle.n + spick]}')
                self.particle.nlimited[:] = 0
                self.particle.nfrozen[:] = 0
                self.particle.check(self.boundary.midpos, self.boundary.pos)
            update_barrier.wait()
            if i < dt.size - 1:
                if ndt_decay > 0:
                    if i < ndt_decay:
                        # reach steady state limiting after ndt_decay
                        self.default_limiting = ss_limiting*cl_ratio + ss_limiting*(1-cl_ratio)*(i+1)/ndt_decay
                        self.cl = ss_cl*cl_ratio + ss_cl*(1-cl_ratio)*(i+1)/ndt_decay
                        self.particle.r0 = r0*cl_ratio + r0*(1-cl_ratio)*(i+1)/ndt_decay
                        if cl_ratio == 1:
                            b_ratio = 1 + (1-(i+1)/ndt_decay)*b0
                        else:
                            b_ratio = 1 + (1-(i+1)/ndt_decay)*(1/cl_ratio-1)

                if pid == self.printing_id:
                    print(f'b_ratio = {b_ratio}')
                    tic = time.perf_counter() 
                for cid in chops_to_update:
                    ichop = cid - chops_to_update[0] + 1
                    self.update_vel_acc_limiting(dt[i], cid, pid, ichop, nchop, b_ratio, output_lock)
                update_barrier.wait()

                if pid == self.printing_id:
                    toc = time.perf_counter() 
                    stdout.write('\n')
                    stdout.flush()
                    print(f'update vel acc limiting took approx. {toc-tic:0.4f} sec')
                    print(f'damping mean:{np.mean(self.particle.dis_intended[self.particle.chop_size>0]):.3e}, std:{np.mean(self.particle.dis[self.particle.chop_size>0]):.3e}')
                # parallel chop up and update shared mem
                if pid == self.printing_id:
                    tic = time.perf_counter() 
                old_nchop = self.particle.nchop
                pos = np.empty(blocksize, dtype = object)
                acc = np.empty(blocksize, dtype = object)
                vel = np.empty(blocksize, dtype = object)
                limiting = np.empty(blocksize, dtype = object)
                interdis = np.empty(blocksize, dtype = object)
                back_map = np.empty(blocksize, dtype = object)
                for j in range(blocksize):
                    bid = blocks_to_update[j]
                    assert(bid < self.particle.nblock)
                    pos[j], acc[j], vel[j], limiting[j], interdis[j], back_map[j] = self.particle.parallel_chop_up(bid, pid, output_lock)

                if pid == self.printing_id:
                    self.particle.block_cid[:] = -1
                    self.particle.chop_bid[:] = -1

                update_barrier.wait()
                if pid == self.printing_id:
                    old_chopsize = self.particle.chop_size.copy()
                    self.particle.chop_size[:] = 0
                    #################
                    assert(np.sum(self.particle.block_size) == self.particle.n)

                update_barrier.wait()

                if pid == self.printing_id:
                    toc = time.perf_counter() 
                    print(f'parallel chop up took approx. {toc-tic:0.4f} sec')
                    tic = time.perf_counter() 
                self.particle.populate_shared_mem(pid, pos, vel, acc, limiting, interdis, back_map, blocks_to_update, output_lock)
                update_barrier.wait()
                if pid == self.printing_id:
                    toc = time.perf_counter() 
                    print(f'populate shared mem took approx. {toc-tic:0.4f} sec')
                # update pointers to shared memory
                if pid == self.printing_id:
                    tic = time.perf_counter() 
                self.particle.distribute_chops_from_shared_mem(pid)
                update_barrier.wait()
                if pid == self.printing_id:
                    toc = time.perf_counter() 
                    print(f'distribute chops from shared mem took approx. {toc-tic:0.4f} sec')
                # rebalance work load to process
                if old_nchop is not self.particle.nchop:
                    if pid == self.printing_id:
                        nchop_total.value = self.particle.nchop
                        print(f'{old_nchop} chops -> {self.particle.nchop} chops')
                    chunk = self.particle.nchop//self.ncore
                    remain = np.mod(self.particle.nchop, self.ncore)
                    chunksize = chunk + (pid < remain)
                    if pid < remain:
                        chops_to_update = (chunk + 1) * pid + np.arange(chunk + 1)
                    else:
                        chops_to_update = (chunk + 1) * remain +  (pid-remain) * chunk + np.arange(chunk)
                with output_lock:
                    if pid == self.printing_id:
                        chop_size_diff = self.particle.chop_size - old_chopsize
                        if np.sum(chop_size_diff) != 0:
                            print(chop_size_diff)
                            #############
                            assert(np.sum(chop_size_diff) == 0)
            with output_lock:
                if pid == self.printing_id:
                    print(f'================ {(i+1)/dt.size*100:.1f}%')
                    mod_plot = 10
                    if np.mod(i, mod_plot) == mod_plot-1:
                        flat_pos = self.particle.assemble()
                        with open(mod_plot_fn + '-pos_for_redraw-'+f'{i//mod_plot}'+'.bin', 'wb') as f:
                            np.array([flat_pos.shape[1]]).astype('i4').tofile(f)
                            flat_pos.astype(float).tofile(f)

class p_point_particle:
    def __init__(self, initial_pos, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, n, chop_size, abcl, xl, yl, nx, ny, k1 = 1.0, k2 = 0.5, roi_ratio = 4.0, ncore = 1):
        self.ncore = ncore
        self.nx = nx
        self.ny = ny
        self.xl = xl
        self.yl = yl
        self.nblock = self.nx * self.ny
        self.n = n
        print(f'{self.n} particles')
        print(f'{self.nx} x {self.ny} = {self.nblock} blocks')

        nchop = chop_size.size
        print(f'inputing # chop: {nchop}')
        self.pos = initial_pos

       # centers for inital chops
        chop_x = np.array([np.mean(pos[0,:]) for pos in self.pos])
        chop_y = np.array([np.mean(pos[1,:]) for pos in self.pos])
       # max distance away from center
        maxD = np.array([np.max(np.linalg.norm(np.array([chop_x[i], chop_y[i]]).reshape(2,1) - self.pos[i], axis = 0)) for i in range(nchop)])

        if np.min(np.hstack([pos[0,:] for pos in self.pos])) < self.xl[0]:
            print(f'{np.min(np.hstack([pos[0,:] for pos in self.pos]))} > {self.xl[0]}')
            assert(False)
        if np.max(np.hstack([pos[0,:] for pos in self.pos])) > self.xl[1]:
            print(f'{np.max(np.hstack([pos[0,:] for pos in self.pos]))} < {self.xl[1]}')
            assert(False)
        if np.min(np.hstack([pos[1,:] for pos in self.pos])) < self.yl[0]:
            print(f'{np.min(np.hstack([pos[1,:] for pos in self.pos]))} > {self.yl[0]}')
            assert(False)
        if np.max(np.hstack([pos[1,:] for pos in self.pos])) > self.yl[1]:
            print(f'{np.max(np.hstack([pos[1,:] for pos in self.pos]))} < {self.yl[1]}')
            assert(False)
       # setup blocks, read-only
        self.block_bound = np.zeros((2,self.ny,self.nx,2))

        xvec = np.linspace(self.xl[0], self.xl[1], self.nx+1)
        print(f'x: {xvec}')
        for i in range(self.nx):
            self.block_bound[0,:,i,:] = np.tile(xvec[i:i+2], (self.ny,1))

        yvec = np.linspace(self.yl[0], self.yl[1], self.ny+1)
        print(f'y: {yvec}')
        for j in range(self.ny):
            self.block_bound[1,j,:,:] = np.tile(yvec[j:j+2], (self.nx,1))

       # potential Lennard-Jones Potential
        self.k1 = k1
        self.k2 = k2
        r0 = np.power(self.k2/self.k1,-1/(self.k1-self.k2))*abcl 
        if abcl*roi_ratio > r0:
            self.r0 = r0
        else:
            self.r0 = abcl*roi_ratio

        self.block_bound = self.block_bound.reshape(2,self.nblock,2)
        self.block_halfDiag = np.sqrt(np.power((yvec[1]-yvec[0]),2) + np.power((xvec[1]-xvec[0]),2))/2
      # default neighbor list
        nwidth = int(np.ceil(self.r0/np.abs(xvec[1]-xvec[0])))
        nheight = int(np.ceil(self.r0/np.abs(yvec[1]-yvec[0])))
        self.neighbor_list = np.zeros((self.nblock, (2*nwidth+1)*(2*nheight+1)-1), dtype = 'i4') - 1
        print(self.neighbor_list.shape)
        for j in range(self.ny):
            for i in range(self.nx):
                k = j*self.nx + i
                ix = np.arange(i-nwidth ,i+nwidth+1)
                iy = np.arange(j-nheight,j+nheight+1)
                #print(f'{i,j}: {ix, iy}')
                list_x, list_y = np.meshgrid(ix, iy)
                list_id = (list_y * self.nx + list_x).flatten()
                # remove self-block
                k_id = nheight*(2*nwidth+1) + nwidth
                assert(list_id[k_id] == k)
                list_id = np.hstack((list_id[:k_id], list_id[k_id+1:]))
                list_x = np.hstack((list_x.flatten()[:k_id], list_x.flatten()[k_id+1:]))
                list_y = np.hstack((list_y.flatten()[:k_id], list_y.flatten()[k_id+1:]))
                # remove outlying neighbors
                list_id = np.unique(list_id[np.logical_and(np.logical_and(list_x >= 0, list_x < self.nx), np.logical_and(list_y >= 0, list_y < self.ny))])
                nl = list_id.size
                self.neighbor_list[k,:nl] = list_id
                assert((list_id >= 0).all())
                #print(f'{i,j}: {self.neighbor_list[k,:nl]}')

      # shared during parallel processing
        self.shared_dmem = shared_dmem
        self.shared_imem = shared_imem

        self.chop_pos = shared_bmem[:2*self.nblock].view().reshape((2, self.nblock)) # for chop center pos
        self.chop_maxD = shared_bmem[2*self.nblock:3*self.nblock].view() # for max distance away from chop center
        self.dis = shared_bmem[3*self.nblock:4*self.nblock].view() # for actual change in distance
        self.dis_intended = shared_bmem[4*self.nblock:5*self.nblock].view() # for intended change in distance without limiting
        self.nint = shared_bmem[5*self.nblock:6*self.nblock].view() # for average inter-distance

        self.block_cid = shared_nmem[:self.nblock].view()
        self.chop_bid = shared_nmem[self.nblock:2*self.nblock].view()
        self.block_size = shared_nmem[2*self.nblock:3*self.nblock].view()
        self.chop_size = shared_nmem[3*self.nblock:4*self.nblock].view() # for subsequent chops nchops <= nblock
        self.nlimited = shared_nmem[4*self.nblock:5*self.nblock].view() # for summing the number of limited particle
        self.nfrozen = shared_nmem[5*self.nblock:6*self.nblock].view() # for summing the number of freezed particle
        self.core_workload = shared_cmem.view()
      # initialize (why necessary?)
        self.shared_dmem[2*self.n: 4*self.n] = 0 # vel
        self.shared_dmem[4*self.n: 6*self.n] = 0 # acc
        self.shared_dmem[6*self.n: 7*self.n] = 0 # limiting
        self.block_cid[:] = -1
        self.chop_bid[:] = -1
        self.block_size[:] = 0
        self.chop_size[:] = 0
        self.nlimited[:] = 0
        self.nfrozen[:] = 0
        
       # initialize chops
       # default velocity and acceleration
        self.vel = np.empty(nchop, dtype = object)
        self.acc = np.empty(nchop, dtype = object)
        i0 = 0
        for i in range(nchop):
            i1 = i0 + 2*chop_size[i]
            self.vel[i] = self.shared_dmem[2*self.n + i0: 2*self.n + i1].view().reshape((2,chop_size[i]))
            self.acc[i] = self.shared_dmem[4*self.n + i0: 4*self.n + i1].view().reshape((2,chop_size[i]))
            i0 = i1

       # default back_map
        self.back_map = np.empty(nchop, dtype = object)
        indices0 = np.arange(self.n)
        i0 = 0
        for i in range(nchop):
            i1 = i0 + chop_size[i]
            self.shared_imem[i0:i1] = indices0[i0:i1]
            self.back_map[i] = self.shared_imem[i0:i1].view()
            i0 = i1

        '''
        rand_name = int(datetime.now().timestamp())
        fig = plt.figure(str(rand_name) + 'test_p', figsize = (12,4))
        ax1 = fig.add_subplot(131)
        self.nchop = nchop
        pos = self.assemble()
        ax1.plot(pos[0,:], pos[1,:], '.', ms = 0.5)
        '''

        #tic = time.perf_counter() 
        print('initial chop up')
        self.initial_chop_up(chop_x, chop_y, maxD)
        #toc = time.perf_counter() 
        #print(f'initial chop up in {toc-tic:0.4f} sec')
        print(f'{self.block_cid}')
        print(f'{self.nblock}')
        print(f'{self.nchop}')
        print(f'{self.chop_bid}')
        '''
        ax2 = fig.add_subplot(132)
        for i in range(self.nchop):
            ax2.plot(self.pos[i][0,:], self.pos[i][1,:], '.', ms = 0.5)

        ax3 = fig.add_subplot(133)
        pos = self.assemble()
        ax3.plot(pos[0,:], pos[1,:], '.', ms = 0.5)

        fig.savefig(str(rand_name) + 'test_p.png', dpi = 2000)
        '''

    def initial_chop_up(self, chop_x, chop_y, maxD):
        k = 0
        i0 = 0
        ii0 = 0
        tmp_pos = np.empty(self.nblock, dtype = object)
        tmp_backmap = np.empty(self.nblock, dtype = object)
        for k0 in range(self.nblock):
            dis = np.sqrt(np.power(np.sum(self.block_bound[0,k0,:])/2 - chop_x, 2) + np.power(np.sum(self.block_bound[1,k0,:])/2 - chop_y, 2))
            qpick = maxD + self.block_halfDiag >= dis

            if np.sum(qpick) > 0:
                nonzero = False
                tmpx = []
                tmpy = []
                tmp_map = []
                for (qpos, qmap) in zip(self.pos[qpick], self.back_map[qpick]):
                    qx = qpos[0,:]
                    qy = qpos[1,:]

                    pick = np.logical_and(np.logical_and(self.block_bound[0,k0,0] < qx, qx <= self.block_bound[0,k0,1]), np.logical_and(self.block_bound[1,k0,0] < qy, qy <= self.block_bound[1,k0,1]))
                    assert(pick.shape == qmap.shape)

                    if np.sum(pick) > 0:
                        tmpx = tmpx + qx[pick].tolist()
                        tmpy = tmpy + qy[pick].tolist()
                        tmp_map = tmp_map + qmap[pick].tolist()
                        nonzero = True

                if nonzero:
                    self.chop_size[k] = len(tmpx)
                    i1 = i0 + self.chop_size[k]
                    ii1 = ii0 + 2*self.chop_size[k]
                    tmp_pos[k] = np.array([tmpx,tmpy]).flatten() # cannot write to sharedmem now would overwrite the original data, self.pos is just view of shared_dmem[:2*self.n]
                    tmp_backmap[k] = np.array(tmp_map).astype('i4')
                    i0 = i1
                    ii0 = ii1
                    self.block_size[k0] = self.chop_size[k]
                    self.block_cid[k0] = k
                    self.chop_bid[k] = k0
                    k = k + 1

        self.nchop = k
        assert(ii0 == 2*self.n)
        assert(i0 == self.n)

        i0 = 0
        ii0 = 0
        self.pos = np.empty(self.nchop, dtype = object)
        self.vel = np.empty(self.nchop, dtype = object)
        self.acc = np.empty(self.nchop, dtype = object)
        self.limiting = np.empty(self.nchop, dtype = object)
        self.interdis = np.empty(self.nchop, dtype = object)
        self.back_map = np.empty(self.nchop, dtype = object)
        for i in range(self.nchop):
            i1 = i0 + self.chop_size[i]
            ii1 = ii0 + 2*self.chop_size[i]
            self.shared_dmem[ii0:ii1] = tmp_pos[i]
            self.shared_imem[i0:i1] = tmp_backmap[i]
            self.pos[i] = self.shared_dmem[ii0:ii1].view().reshape((2,self.chop_size[i]))
            self.vel[i] = self.shared_dmem[2*self.n + ii0: 2*self.n + ii1].view().reshape((2,self.chop_size[i]))
            self.acc[i] = self.shared_dmem[4*self.n + ii0: 4*self.n + ii1].view().reshape((2,self.chop_size[i]))
            self.limiting[i] = self.shared_dmem[6*self.n + i0: 6*self.n + i1].view()
            self.interdis[i] = self.shared_dmem[7*self.n + i0: 7*self.n + i1].view()
            self.back_map[i] = self.shared_imem[i0:i1].view()
            i0 = i1
            ii0 = ii1

    def parallel_chop_up(self, k0, pid, lock):
        # get neighbor index
        dis = np.linalg.norm(np.array([np.sum(self.block_bound[0,k0,:])/2,np.sum(self.block_bound[1,k0,:])/2]).reshape(2,1)  - self.chop_pos[:,:self.nchop], axis = 0)
        pick = self.chop_maxD[:self.nchop] + self.block_halfDiag >= dis
        
        pos = np.array([])
        vel = np.array([])
        acc = np.array([])
        limiting = np.array([])
        interdis = np.array([])
        back_map = np.array([])
        
        npick = np.sum(pick)
        self.block_size[k0] = 0
        if npick > 0:
            tmp_x = []
            tmp_y = []
            tmp_vx = []
            tmp_vy = []
            tmp_ax = []
            tmp_ay = []
            tmp_limiting = []
            tmp_interdis = []
            tmp_map = []
            nonzero = False
            cid = np.arange(self.nchop)[pick]
            for i in range(npick):
                j = cid[i]
                qx = self.pos[j][0,:]
                qy = self.pos[j][1,:]

                pick = np.logical_and(np.logical_and(self.block_bound[0,k0,0] < qx, qx <= self.block_bound[0,k0,1]), np.logical_and(self.block_bound[1,k0,0] < qy, qy <= self.block_bound[1,k0,1]))
                if np.sum(pick) > 0:
                    tmp_x = tmp_x + qx[pick].tolist()
                    tmp_y = tmp_y + qy[pick].tolist()
                    tmp_vx = tmp_vx + self.vel[j][0,pick].tolist()
                    tmp_vy = tmp_vy + self.vel[j][1,pick].tolist()
                    tmp_ax = tmp_ax + self.acc[j][0,pick].tolist()
                    tmp_ay = tmp_ay + self.acc[j][1,pick].tolist()
                    tmp_limiting = tmp_limiting + self.limiting[j][pick].tolist()
                    tmp_interdis = tmp_interdis + self.interdis[j][pick].tolist()
                    tmp_map = tmp_map + self.back_map[j][pick].tolist()
                    nonzero = True

            if nonzero:
                self.block_size[k0] = len(tmp_x)
                pos = np.array([tmp_x,tmp_y]).flatten()
                vel = np.array([tmp_vx,tmp_vy]).flatten()
                acc = np.array([tmp_ax,tmp_ay]).flatten()
                limiting = np.array(tmp_limiting)
                interdis = np.array(tmp_interdis)
                back_map = np.array(tmp_map, dtype = 'i4')
        return pos, vel, acc, limiting, interdis, back_map

    def populate_shared_mem(self, pid, pos, vel, acc, limiting, interdis, back_map, blocks_to_update, lock):
        self.nchop = np.sum(self.block_size > 0)
        k = 0
        i0 = 0
        ii0 = 0
        j = 0
        for i in range(self.nblock):
            if self.block_size[i] > 0:
                i1 = i0 + self.block_size[i]
                ii1 = ii0 + 2*self.block_size[i]

                if pid == 0:
                    self.chop_size[k] = self.block_size[i]
                    self.chop_bid[k] = i 
                    self.block_cid[i] = k
                    k = k + 1

                if i in blocks_to_update:
                    self.shared_dmem[ii0: ii1] = pos[j]
                    self.shared_dmem[2*self.n + ii0: 2*self.n + ii1] = vel[j]
                    self.shared_dmem[4*self.n + ii0: 4*self.n + ii1] = acc[j]
                    self.shared_dmem[6*self.n + i0: 6*self.n + i1] = limiting[j]
                    self.shared_dmem[7*self.n + i0: 7*self.n + i1] = interdis[j]
                    self.shared_imem[i0:i1] = back_map[j]
                    j = j + 1
                i0 = i1
                ii0 = ii1
            else:
                if i in blocks_to_update:
                    j = j + 1

        if pid == 0:
            assert(self.nchop == k)

    def distribute_chops_from_shared_mem(self, pid):
        self.pos = np.empty(self.nchop, dtype = object)
        self.vel = np.empty(self.nchop, dtype = object)
        self.acc = np.empty(self.nchop, dtype = object)
        self.limiting = np.empty(self.nchop, dtype = object)
        self.interdis = np.empty(self.nchop, dtype = object)
        self.back_map = np.empty(self.nchop, dtype = object)
        i0 = 0
        ii0 = 0
        for i in range(self.nchop):
            i1 = i0 + self.chop_size[i]
            ii1 = ii0 + 2*self.chop_size[i]
            self.pos[i] = self.shared_dmem[ii0:ii1].view().reshape((2,self.chop_size[i]))
            self.vel[i] = self.shared_dmem[2*self.n + ii0: 2*self.n + ii1].view().reshape((2,self.chop_size[i]))
            self.acc[i] = self.shared_dmem[4*self.n + ii0: 4*self.n + ii1].view().reshape((2,self.chop_size[i]))
            self.limiting[i] = self.shared_dmem[6*self.n + i0: 6*self.n + i1].view()
            self.interdis[i] = self.shared_dmem[7*self.n + i0: 7*self.n + i1].view()
            self.back_map[i] = self.shared_imem[i0:i1].view()
            i0 = i1
            ii0 = ii1

    #@timer
    def assemble(self):
        flat_pos = np.empty((2, self.n), dtype = float)
        i0 = 0
        ii0 = 0
        for i in range(np.sum(self.chop_size > 0)): # dont use self.nchop, self.pos, self.back_map, they are not shared outside multiprocessing
            i1 = i0 + self.chop_size[i]
            ii1 = ii0 + self.chop_size[i]*2
            flat_pos[:, self.shared_imem[i0:i1]] = self.shared_dmem[ii0:ii1].reshape((2,self.chop_size[i]))
            #flat_pos[0, self.back_map[i]] = self.pos[i][0, :]
            #flat_pos[1, self.back_map[i]] = self.pos[i][1, :]
            i0 = i1
            ii0 = ii1

        return flat_pos
    
    def check(self, midpos, bpos, pos = None):
        if pos is None:
            pos = self.assemble()
        if (pos[0,:] < self.xl[0]).any() or (pos[0,:] > self.xl[1]).any() or (pos[1,:] < self.yl[0]).any() or (pos[1,:] > self.yl[1]).any():
            pick = np.logical_or(pos[0,:] < self.xl[0], np.logical_or(pos[0,:] > self.xl[1],np.logical_or(pos[1,:] < self.yl[0],pos[1,:] > self.yl[1])))
            plist = np.arange(self.n)[pick]
            for i in plist:
                ds = midpos - np.tile(pos[:,i].reshape(2,1),(1,2))
                ib = np.argmin(np.min(np.sum(np.power(ds, 2), axis = -2), axis = -1))
                print(f'# {i} {pos[:,i]}, {bpos[ib,:,:]}')
            raise Exception(f'pos of shape {pos.shape} has particles escaped, xrange: {[np.min(pos[0,:]), np.max(pos[0,:])]} inside {[self.xl[0], self.xl[1]]}; yrange: {[np.min(pos[1,:]), np.max(pos[1,:])]} inside {[self.yl[0], self.yl[1]]}')
        else:
            print(f'pos of shape {pos.shape} is within the boundary, check passed')

    def f(self, r, cl):
        #r0 = 2*abcl
        #f = lambda r: 2*np.power(abcl/r,3) - np.power(abcl/r,2)
        force = self.k1*np.power(cl/r,self.k1+1) - self.k2*np.power(cl/r,self.k2+1)
        return force

class p_rec_boundary(rec_boundary):
    def __init__(self, subgrid, boundPos, boundType, abcl, n, ax = None):
        self.n = n
        self.shared_pos = RawArray(c_double, boundPos.flatten())
        self.pos = np.frombuffer(self.shared_pos, dtype = 'f8').view().reshape((n,2,3))
        midpos = np.stack([np.mean(boundPos[:,:,:2],axis=-1), np.mean(boundPos[:,:,1:],axis=-1)], axis = -1)
        self.shared_midpos = RawArray(c_double, midpos.flatten())
        self.midpos = np.frombuffer(self.shared_midpos, dtype = 'f8').view().reshape((n,2,2))

        self.xl = [np.min(self.pos[:,0,:].flatten()), np.max(self.pos[:,0,:].flatten())]
        self.yl = [np.min(self.pos[:,1,:].flatten()), np.max(self.pos[:,1,:].flatten())]

        self.btype = mp.sharedctypes.RawArray(c_uint, boundType)
        self.subgrid = subgrid # (x,y)
        # set a radius to pick nearby particles
        self.rec = np.max(subgrid)
        if ax is not None:
            # connecting points on grid sides
            ax.plot(self.pos[:,0,[0,2]].squeeze(), self.pos[:,1,[0,2]].squeeze(), ',r')
            # grid centers
            ax.plot(self.pos[:,0,1].squeeze(), self.pos[:,1,1].squeeze(), ',g')

        self.r0, self.f = get_r0_f(abcl)

    def get_acc(self, i, pos):
        if self.btype[i] == 0:
            # same y-coord -> horizontal boundary
            #assert((self.pos[i,1,1] - self.pos[i,1,:] == 0).all())
            r, d = self.get_ah(i, pos)
        elif self.btype[i] == 1:
            # same x-coord -> vertical boundary
            #assert((self.pos[i,0,1] - self.pos[i,0,:] == 0).all())
            r, d = self.get_av(i, pos)
        else:
            if self.btype[i] == 2:
                # first horizontal then vertical
                #assert(self.pos[i,1,1] == self.pos[i,1,0])
                r, d = self.get_ahv(i, pos)
            else:
                # first vertical then horizontal 
                #if self.btype[i] is not 3:
                #    raise Exception(f'btype: {self.btype[i]} is not implemented')
                r, d = self.get_avh(i, pos)
        return r, d

def get_r0_f(abcl, k1 = 1.0, k2 = 0.5, roi_ratio = 2):
    r0 = np.power(k2/k1,-1/(k1-k2))*abcl
    if abcl*roi_ratio < r0:
        r0 = abcl*roi_ratio
    #r0 = np.power(b*k2/a/k1,-1/(k1-k2))*cl
    #r0 = 2*abcl
    #f = lambda r: 2*np.power(abcl/r,3) - np.power(abcl/r,2)
    f = lambda r: k1*np.power(abcl/r,k1+1) - k2*np.power(abcl/r,k2+1)
    return r0, f

def parallel_repel(area, subgrid, particlePos, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, p_scale, boundPos, boundType, b_scale, nx, ny, ncore, dt, spercent, figname, ndt_decay, cl_ratio, roi_ratio, k1 = 1.0, k2 = 0.5, seed = -1, limit_ratio = 1.0, damp = 0.5, l_ratio = 0.0, l_k = 1.0, local_plot = True):
    if figname is None:
        figname = 'parallel_repel'
    if spercent > 0 and local_plot:
        fig = plt.figure(figname, dpi = 2000)
        ax = fig.add_subplot(111)
    else:
        ax = None

    system = p_repel_system(area, subgrid, particlePos, shared_dmem, shared_imem, shared_bmem, shared_nmem, shared_cmem, boundPos, boundType, p_scale, b_scale, nx, ny, roi_ratio, k1, k2, ncore, limit_ratio, damp, l_ratio, l_k, ax)
    n0  = system.particle.n
    if spercent > 0 and local_plot:
        if seed < 0:
            seed = int(datetime.now().timestamp())
        np.random.seed(seed)
        flat_pos = system.particle.assemble()
        spick = np.random.rand(system.particle.n) > spercent
        ns = np.sum(spick)
        sample_pos = np.zeros((2,2,ns))
        sample_pos[0,:,:] = flat_pos[:,spick].copy()
        del flat_pos

    update_barrier = Barrier(ncore)
    output_lock = Lock()
      
    nchop = Value(c_int32, 0, lock = output_lock)
    
    proc = np.empty(ncore, dtype = object)
    for i in range(ncore):
        proc[i] = Process(target = system.repel_near_chop, args = (i, dt, ndt_decay, cl_ratio, update_barrier, output_lock, nchop, figname))
        proc[i].start()

    for p in proc:
        p.join()

    pos = system.particle.assemble()
    #######
    assert(not np.isnan(pos.flatten()).any())
    system.particle.check(system.boundary.midpos,system.boundary.pos, pos = pos)
    
    print('write to file ...')
    with open(figname + '_for_redraw.bin', 'wb') as f:
        np.array([system.n]).astype('i4').tofile(f)
        pos.astype(float).tofile(f)

    with open(figname+'_final.bin', 'wb') as f:
        nchop_value = nchop.value
        np.array([nchop_value]).astype('i4').tofile(f)
        system.particle.chop_size[:nchop_value].astype('i4').tofile(f)
        system.particle.shared_dmem[:2*system.particle.n].tofile(f)
        #system.particle.vel.tofile(f)
        #system.particle.acc.tofile(f)
        #system.particle.limiting.tofile(f)
        np.array([system.nb]).tofile(f)
        boundPos.tofile(f)
        boundType.astype('u4').tofile(f)
        subgrid.tofile(f)
        np.array([area]).tofile(f)

    if np.sum(system.particle.chop_size[:nchop_value]) != pos.shape[1]:
        #print(f'chop sum {np.sum(system.particle.chop_size[:nchop_value])} flat sum: {pos.shape[1]} mismatch')
        raise Exception(f'chop sum {np.sum(system.particle.chop_size[:nchop_value])} flat sum: {pos.shape[1]} mismatch')
    if n0 != pos.shape[1]:
        #print('#particles: {n0} -> {pos.shape[1]}')
        raise Exception('#particles: {n0} -> {pos.shape[1]}')

    print('done')

    if local_plot:
        if spercent > 0:
            tic = time.perf_counter() 
            sample_pos[1, :, :] = pos[:,spick].copy()
            ax.plot(pos[0,:], pos[1,:], ',k')
            ax.plot(sample_pos[:,0,:], sample_pos[:,1,:], '-,c', lw = 0.01) # takes long time
            ax.plot(sample_pos[0,0,:], sample_pos[0,1,:], ',m')
            ax.set_aspect('equal')
            fig.savefig(figname +'.png', dpi = 2000)
            plt.close(fig)
            toc = time.perf_counter() 
            del ax
            del fig

        tic = time.perf_counter() 
        fig = plt.figure('parallel_repel_final', dpi = 2000)
        ax = fig.add_subplot(111)
        # connecting points on grid sides
        ax.plot(system.boundary.pos[:,0,[0,2]], system.boundary.pos[:,1,[0,2]], ',r')
        # grid centers
        ax.plot(system.boundary.pos[:,0,1], system.boundary.pos[:,1,1], ',g')
        ax.plot(pos[0,:], pos[1,:], ',k')
        ax.set_aspect('equal')
        fig.savefig(figname +'_final.png', dpi = 2000)
        toc = time.perf_counter() 
        print(f'final plot finished in {toc-tic:0.4f} sec')

    return pos

def n_in_m(n, m, i):
    s = n//m
    r = np.mod(n, m)
    if i < r:
        return (s+1)*i + np.arange(s+1)
    else:
        return (s+1)*r + s*(i-r) + np.arange(s)

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
        np.frombuffer(particlePos, dtype='f8').tofile(f)
        np.array([nBoundary]).tofile(f)        
        np.frombuffer(boundPos, dtype='f8').tofile(f)
        np.frombuffer(boundType, dtype='u4').tofile(f)
        subgrid.tofile(f)
        np.array([area]).tofile(f)
