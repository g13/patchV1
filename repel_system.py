import numpy as np
from sys import stdout
import py_compile
import time
py_compile.compile('repel_system.py')

class repel_system:
    def __init__(self, area, subgrid, initial_x, bp, btype, boundary_param = None, particle_param = None, initial_v = None, nlayer = 1, layer = None, soft_boundary = None, soft_btype = None,  enough_memory = False, p_scale = 2.5, b_scale = 1.0, fixed = None, b_cl = None, i_cl = None):
        self.nn = initial_x.shape[1]
        self.nb = bp.shape[0]
        if fixed is None:
            self.fixList = np.array([])
        else:
            self.fixList = fixed
        if b_cl is None:
            per_unit_area = 2*np.sqrt(3) # in cl^2
            self.cl = np.sqrt((area/self.nn)/per_unit_area)
            print(f'characteristic length (inter-particle-distance)')
            a_particle = self.cl*p_scale
            a_boundary = self.cl*b_scale
        else:
            assert(i_cl is not None)
            self.cl = b_cl/b_scale
            a_particle = i_cl
            a_boundary = b_cl
        # will be used to limit displacement and velocity
        self.damp = 0.1
        self.default_limiting = self.cl
        self.limiting = np.zeros(self.nn) + self.default_limiting
        if nlayer == -1:
            self.layerSupplied = True
            assert(nlayer == layer.shape[0])
            self.nlayer = nlayer
            self.layer = layer
        else:
            self.layerSupplied = False

        print('particle:')
        if particle_param is None:
            self.particle = point_particle(initial_x, L_J_potiential(a_particle,a_particle,2,1,a_particle), initial_v)
        else:
            self.particle = point_particle(initial_x, particle_param, initial_v)
        print('boundary:')
        if boundary_param is None:
            self.boundary = rec_boundary(subgrid, bp, btype, L_J_potiential(a_boundary,a_boundary,2,1,a_boundary), enough_memory)
        else:
            self.boundary = rec_boundary(subgrid, bp, btype, boundary_param, enough_memory)

        if soft_boundary is not None:
            assert(soft_btype is not None)
            self.soft_boundary = rec_boundary(subgrid, soft_boundary, soft_btype, None, enough_memory)

        print(f'{self.nb} boundary points and {self.nn} particles initialized')
        print(f'in units of grids ({subgrid[0]:.3f},{subgrid[1]:.3f}):')
        print(f'    interparticle distance ({self.cl/subgrid[0]:.3f},{self.cl/subgrid[1]:.3f})')
        print(f'    radius of influence for particles ({self.particle.r0/subgrid[0]:.3f},{self.particle.r0/subgrid[1]:.3f})')
        print(f'    radius of influence for boundaries ({self.boundary.r0/subgrid[0]:.3f},{self.boundary.r0/subgrid[1]:.3f})')
        print(f'    default limiting of displacement in one dt: ({self.default_limiting/subgrid[0]:.3f}, {self.default_limiting/subgrid[1]:.3f})')
        
        self.nlayer = nlayer
        self.enough_memory = enough_memory 

    def initialize(self):
        if self.layerSupplied is False:
            self.layer = np.empty(self.nlayer, dtype=object)
            if self.nlayer > 1:
                # transform to [x/y,index]
                bp = self.soft_boundary.pos[:,:,1].T
                dis = np.empty(self.nn)
                for i in range(self.nn):
                    ib = np.argmin(np.sum(np.power(bp - self.particle.pos[:,i].reshape(2,1),2), axis=0))
                    dis[i], _ = self.soft_boundary.get_r[ib](ib, self.particle.pos[:,i])
                    stdout.write(f'\rinitializing, {(i+1)/self.nn*100:.3f}%')

                print('\n')
                dis_range = np.linspace(np.min(dis), np.max(dis), self.nlayer+1)
                n = np.zeros(self.nlayer)
                for i in range(self.nlayer):
                    if i == self.nlayer-1:
                        self.layer[i] = np.nonzero(np.logical_and(dis >= dis_range[i], dis <= dis_range[i+1]))[0]
                    else:
                        self.layer[i] = np.nonzero(np.logical_and(dis >= dis_range[i], dis < dis_range[i+1]))[0]
                    n[i] = self.layer[i].size
                print(f'number of neurons in layers: {n}')
            else:
                self.layer[0] = np.arange(self.nn)

        self.get_acc_update_limiting(self.layer[0])
        print('initialized')
        if self.layerSupplied is False:
            return self.nlayer, self.layer
         
    def get_acc_update_limiting(self, pick = None):
        if pick is None:
            pos = self.particle.pos
            n = pos.shape[1]
            pick = np.arange(n)
        else:
            pos = self.particle.pos[:,pick]
            n = pos.shape[1]
        m = self.particle.pos.shape[1]
        # NOTE: not implemented
        if self.enough_memory:
            raise Exception('not implemented')
            ds = pos.reshape(2,n,1) - self.particle.pos.reshape(2,1,m)
            # dims 1: x/y. 2: target. 3: source.
            assert(ds.shape[0] == 2 and ds.shape[1] == n and ds.shape[2] == m)
            ## pick a square first
            square_pick = np.max(np.abs(ds), axis = 0) < self.particle.r0
            #assert(square_pick[index,index].all())
            index = np.arange(n)
            square_pick[index,index] = False
            ppick = square_pick.any(-1)
        else:
            # for every particle picked
            for i in range(n):

                ## calculate repelling acceleratoin from the nearest boundary
                ds = self.boundary.pos[:,:,1] - pos[:,i].reshape(1,2)
                ib = np.argmin(np.sum(np.power(ds, 2), axis = -1))
                acc, limiting = self.boundary.get_acc[ib](ib, pos[:,i])
                # update the limiting threshold
                if limiting > 0:
                    self.limiting[pick[i]] =  np.min([limiting/2, self.default_limiting])
                ## limiting acceleration from boundary
                # da = np.linalg.norm(acc)
                # if da > self.limiting[pick[i]]:
                    # acc = self.limiting[pick[i]] * acc/da
                self.particle.acc[:,pick[i]] = acc

                ## accumulate acceleration from particles
                ds = pos[:,i].reshape(2,1) - self.particle.pos
                # pick a square first
                ppick = np.max(np.abs(ds), axis = 0) < self.particle.r0
                ppick[pick[i]] = False
                if ppick.any():
                    ## calculate distance
                    r = np.linalg.norm(ds[:,ppick], axis=0)
                    assert(r.size == np.sum(ppick))
                    rpick = r < self.particle.r0
                    ppick[ppick] = rpick
                    ## calculate unit vector
                    if rpick.any():
                        ## untouched summed acceleration
                        direction = ds[:,ppick]/r[rpick]
                        self.particle.acc[:,pick[i]] = self.particle.acc[:,pick[i]] + np.sum(self.particle.f(r[rpick])*direction, axis=-1)
                        ## limiting the total summed acceleration
                        # dacc = np.sum(self.particle.f(r[rpick])*direction, axis=-1)
                        # da = np.linalg.norm(dacc, axis=0)
                        # if da > self.limiting[pick[i]]:
                        #   dacc = self.limiting[pick[i]] * dacc/da
                        # self.particle.acc[:,pick[i]] = dacc
                        ## limiting acceleration from individual interaction with particles
                        #direction = ds[:,ppick]/r[rpick]
                        #dacc = self.particle.f(r[rpick])*direction
                        #da = np.linalg.norm(dacc, axis=0)
                        #large_acc = da > self.limiting[pick[i]]
                        #dacc[:,large_acc] = self.limiting[pick[i]] * dacc[:,large_acc]/da[large_acc]
                        #self.particle.acc[:,pick[i]] = np.sum(dacc, axis=-1)

                #stdout.write(f'\rcalculate acceleration: {(i+1)/pick.size*100:.3f}%')
        #print('\n')

    def next(self, dt, layer_seq = None):
        if layer_seq is None:
            layer_seq = np.arange(self.nlayer)

        # limitation on displacement and velocity are needed, but not acceleration
        nlimited = 0
        # get to new position by speed * dt + 1/2 * acc * dt^2
        # limit the change in displacement by the distance to the nearest boundary
        # calculate the new velocity, limit the final speed, not the change of velocity
        if self.nlayer == 1:
            dpos = self.particle.vel*dt + 0.5*self.particle.acc * np.power(dt,2)
            # limit change in displacement
            dr = np.linalg.norm(dpos, axis = 0)
            large_dpos = dr > self.limiting
            dpos[:,large_dpos] = self.limiting[large_dpos] * dpos[:,large_dpos]/dr[large_dpos]
            if not self.fixList.size == 0:
                dpos[:,self.fixList] = 0
            self.particle.pos = self.particle.pos + dpos
            dr[large_dpos] = self.limiting[large_dpos]
            nlimited = np.sum(large_dpos)
        else:
            dr = np.empty(self.nn)
            # update position
            for i in layer_seq:
                pick = self.layer[i]
                dpos = self.particle.vel[:,pick]*dt + 0.5*self.particle.acc[:,pick] * np.power(dt,2)
                # limit position change
                dr[pick] = np.linalg.norm(dpos, axis = 0)
                large_dpos = dr[pick] > self.limiting[pick]
                dpos[:,large_dpos] = self.limiting[pick][large_dpos] * dpos[:,large_dpos]/dr[pick][large_dpos]
                self.particle.pos[:,pick] = self.particle.pos[:,pick] + dpos
                dr[pick][large_dpos] = self.limiting[pick][large_dpos]
                nlimited = nlimited + np.sum(large_dpos)

                if i < self.nlayer-1:
                    ## update acceleration and velocity
                    self.get_acc_update_limiting(self.layer[i+1])

        # number of freezed particles
        nfreeze = np.sum(dr == 0)
        # reset limiting threshold 
        self.limiting = np.zeros(self.nn) + self.default_limiting
        ## update acceleration and velocity
        last_acc = self.particle.acc.copy()
        self.get_acc_update_limiting()
        for i in layer_seq:
            pick = self.layer[i]
            # use acc to change the course of velocity
            self.particle.vel[:,pick] = (1-self.damp) * self.particle.vel[:,pick] + 0.5*(self.particle.acc[:,pick] + last_acc[:,pick])*dt
            # limit the absolute speed
            v = np.linalg.norm(self.particle.vel[:,pick], axis = 0)
            large_vel =  v > self.limiting[pick]/dt
            self.particle.vel[:,pick[large_vel]] = self.limiting[pick[large_vel]]/dt * self.particle.vel[:,pick[large_vel]]/v[large_vel]

        return self.particle.pos, np.array([np.mean(dr), np.std(dr)]), nlimited, nfreeze

class L_J_potiential:
    def __init__(self, a, b, k1, k2, cl):
        self.a = a
        self.b = b
        self.k1 = k1
        self.k2 = k2
        self.cl = cl
        self.r0 = np.power(b*k2/a/k1,-1/(k1-k2))*cl
        shift = a*np.power(cl/self.r0,k1) - b*np.power(cl/self.r0,k2)
        self.ph = lambda r: a*np.power(cl/r,k1) - b*np.power(cl/r,k2) - shift
        self.f = lambda r: k1/r*a*np.power(cl/r,k1) - k2/r*b*np.power(cl/r,k2)
    def plot(self, ax1, ax2, style):
        epsilon = np.finfo(float).eps
        n = 100
        r = np.linspace(0,self.r0,n) + epsilon
        y = self.f(r)
        ax1.plot(r,y,style)
        ax1.plot(self.r0, self.f(self.r0), '*r')
        ax1.plot(self.cl, self.f(self.cl), '*b')
        ax1.plot(r, np.zeros(n), ':g')
        ax1.set_title('force')
        y = self.ph(r)
        ax2.plot(r,y,style)
        ax2.plot(self.r0, self.ph(self.r0), '*r')
        ax2.plot(self.cl, self.ph(self.cl), '*b')
        ax2.plot(r, np.zeros(n), ':g')
        ax2.set_title('potential')
    def print(self):
        print(f'a = {self.a}')
        print(f'b = {self.b}')
        print(f'k1 = {self.k1}')
        print(f'k2 = {self.k2}')
        print(f'cl = {self.cl}')

class point_particle:
    def __init__(self, initial_x, param, initial_v = None):
        self.pos = initial_x
        self.n = self.pos.shape[1]
        if initial_v is not None:
            self.vel = initial_v
        else:
            self.vel = np.zeros((2,self.n))
        self.acc = np.zeros((2,self.n))

       # potential Lennard-Jones Potential
        self.r0 = param.r0
        self.f = param.f
        param.print()

class rec_boundary:
    def __init__(self, subgrid, pos, btype, param = None, enough_memory = False):
        self.pos = pos
        self.n = pos.shape[0]
        assert(pos.shape[1] == 2 and pos.shape[2] == 3)
        self.subgrid = subgrid # (x,y)
        # set a radius to pick nearby particles
        self.rec = np.max(subgrid)
        if param is not None:
            self.r0 = param.r0
            if self.r0 > self.rec:
                self.rec = self.r0
            param.print()
            self.f = param.f
            self.get_acc = np.empty(self.n, dtype = object)
            if enough_memory:
                raise Exception('not implemented')
                for i in range(self.n):
                    # same y-coord -> horizontal boundary
                    if btype[i] == 0:
                        assert((pos[i,1,1] - pos[i,1,:] == 0).all())
                        self.get_acc[i] = self.get_ah_vec
                    # same x-coord -> vertical boundary
                    elif btype[i] == 1:
                        assert((pos[i,0,1] - pos[i,0,:] == 0).all())
                        self.get_acc[i] = self.get_av_vec
                    else:
                        # first horizontal then vertical
                        if btype[i] == 2:
                            assert(pos[i,1,1] == pos[i,1,0])
                            self.get_acc[i] = self.get_ahv_vec
                        # first vertical then horizontal 
                        else:
                            if btype[i] is not 3:
                                raise Exception(f'btype: {btype[i]} is not implemented')
                            self.get_acc[i] = self.get_avh_vec
            else:
                for i in range(self.n):
                    # same y-coord -> horizontal boundary
                    if btype[i] == 0:
                        assert((pos[i,1,1] - pos[i,1,:] == 0).all())
                        self.get_acc[i] = self.get_ah
                    # same x-coord -> vertical boundary
                    elif btype[i] == 1:
                        assert((pos[i,0,1] - pos[i,0,:] == 0).all())
                        self.get_acc[i] = self.get_av
                    else:
                        # first horizontal then vertical
                        if btype[i] == 2:
                            assert(pos[i,1,1] == pos[i,1,0])
                            self.get_acc[i] = self.get_ahv
                        # first vertical then horizontal 
                        else:
                            if btype[i] is not 3:
                                raise Exception(f'btype: {btype[i]} is not implemented')
                            self.get_acc[i] = self.get_avh
        else:
            self.get_r = np.empty(self.n, dtype = object)
            print('the boundary is soft')
            for i in range(self.n):
                # same y-coord -> horizontal boundary
                if btype[i] == 0:
                    assert((pos[i,1,1] - pos[i,1,:] == 0).all())
                    self.get_r[i] = self.get_rh
                # same x-coord -> vertical boundary
                elif btype[i] == 1:
                    assert((pos[i,0,1] - pos[i,0,:] == 0).all())
                    self.get_r[i] = self.get_rv
                else:
                    # first horizontal then vertical
                    if btype[i] == 2:
                        assert(pos[i,1,1] == pos[i,1,0])
                        self.get_r[i] = self.get_rhv
                    # first vertical then horizontal 
                    else:
                        if btype[i] is not 3:
                            raise Exception(f'btype: {btype[i]} is not implemented')
                        self.get_r[i] = self.get_rvh

    # horizontal
    def get_rh(self, i, pos):
        d = pos[1] - self.pos[i,1,1]
        r = np.abs(d)
        return r, d
    def get_ah(self, i, pos):
        r, d = self.get_rh(i, pos)
        if r < self.r0:
            return np.array([0, np.copysign(self.f(r), d)]), r
        else:
            return np.array([0, 0]), r
        
    def get_rh_vec(self, i, pos):
        d = pos[1,:] - self.pos[i,1,1]
        r = np.abs(d)
        return r, d
    def get_ah_vec(self, i, pos):
        acc = np.zeros((2,pos.shape[1]))
        r, d = self.get_rh_vec(i, pos)
        pick =  r < self.r0
        if pick.any():
            acc[1,pick] = np.copysign(self.f(r[pick]), d[pick])
            print('from horizontal boundary')
            print(f'dis: {r[pick]}')
            print(f'acc: {acc[1,pick]}')
        return acc, r
    # vertical 
    def get_rv(self, i, pos):
        d = pos[0] - self.pos[i,0,1]
        r = np.abs(d)
        return r, d
    def get_av(self, i, pos):
        r, d = self.get_rv(i, pos)
        if r < self.r0:
            return np.array([np.copysign(self.f(r), d), 0]), r
        else:
            return np.array([0, 0]), r

    def get_rv_vec(self, i, pos):
        d = pos[0,:] - self.pos[i,0,1]
        r = np.abs(d)
        return r, d
    def get_av_vec(self, i, pos):
        acc = np.zeros((2,pos.shape[1]))
        r, d = self.get_rv_vec(i, pos)
        pick =  r < self.r0
        if pick.any():
            acc[0,pick] = np.copysign(self.f(r[pick]), d[pick])
            print('from vertical boundary')
            print(f'dis: {r[pick]}')
            print(f'acc: {acc[0,pick]}')
        return acc, r
    # horizontal & vertical 
    # turntype == 0: first two points horizontal
    # turntype == 2: first two points vertical 
    def get_rturn(self, i, pos, turntype):
        # inside the subgrid
        if (np.abs(pos - self.pos[i,:,1]) < self.subgrid).all():
            if np.sign(pos[0] - self.pos[i,0,1]) != np.sign(pos[0] - self.pos[i,0,1] - np.copysign(self.rec, self.pos[i,0,1] - self.pos[i,0,turntype])) and \
            np.sign(pos[1] - self.pos[i,1,1]) != np.sign(pos[1] - self.pos[i,1,1] - np.copysign(self.rec, self.pos[i,1,1] - self.pos[i,1,2-turntype])):
                # overlap of horizontal and vertical or the turning point
                r = np.linalg.norm(pos - self.pos[i,:,1], axis=0)
            else:
                # horizontal
                hlap = False
                if np.sign(pos[0] - self.pos[i,0,1]) != np.sign(pos[0] - self.pos[i,0,turntype]):
                    hlap = True
                    r = np.abs(pos[1] - self.pos[i,1,1])
                # vertical
                if np.sign(pos[1] - self.pos[i,1,1]) != np.sign(pos[1] - self.pos[i,1,2-turntype]):
                    if hlap is True:
                        r = np.min([r, np.abs(pos[0] - self.pos[i,0,1])])
                    else:
                        r = np.abs(pos[0] - self.pos[i,0,1])
        else:
            r = np.linalg.norm(pos - self.pos[i,:,1])
        return r, 0

    def get_turn(self, i, pos, turntype):
        acc = np.zeros(2)
        r = 0
        vv = False
        hh = False
        # vertical
        if np.sign(pos[0] - self.pos[i,0,1]) != np.sign(pos[0] - self.pos[i,0,turntype]):
            vv = True
            d = pos[1] - self.pos[i,1,1]
            ry = np.abs(d)
            r = ry
            if r < self.r0:
                acc[1] = np.copysign(self.f(r), d)
        # horizontal 
        if np.sign(pos[1] - self.pos[i,1,1]) != np.sign(pos[1] - self.pos[i,1,2-turntype]):
            hh = True
            d = pos[0] - self.pos[i,0,1]
            rx = np.abs(d)
            r = rx
            if r < self.r0:
                acc[0] = np.copysign(self.f(r), d)
        # turning section 
        if not vv and not hh:
            d = (pos - self.pos[i,:,1])
            r = np.linalg.norm(d, axis=0)
            #if np.sign(pos[0] - self.pos[i,0,1]) != np.sign(pos[0] - self.pos[i,0,1] - np.copysign(self.rec, self.pos[i,0,1] - self.pos[i,0,turntype])) and \
            #np.sign(pos[1] - self.pos[i,1,1]) != np.sign(pos[1] - self.pos[i,1,1] - np.copysign(self.rec, self.pos[i,1,1] - self.pos[i,1,2-turntype])):
            if r < self.r0:
                acc = self.f(r)*d/r
        elif vv and hh:
            r = np.min([rx, ry])
            
        assert(r > 0)
        return acc, r

    def get_rturn_vec(self, i, pos, turntype):
        r = np.zeros(pos.shape[1])
        # check horizontal
        ## pick if x is inbetween self.pos[i,0,0] and self.pos[i,0,1]
        hpick = np.sign(pos[0,:] - self.pos[i,0,1]) != np.sign(pos[0,:] - self.pos[i,0,turntype])
        if hpick.any():
            d = pos[1,hpick] - self.pos[i,1,1]
            r[hpick] = np.abs(d)
        # check vertical
        ## pick if y is inbetween self.pos[i,1,1] and self.pos[i,1,2]
        vpick = np.sign(pos[1,:] - self.pos[i,1,1]) != np.sign(pos[1,:] - self.pos[i,1,2-turntype])
        if vpick.any():
            rv_tmp = np.abs(pos[0,vpick] - self.pos[i,0,1])
            # choose the min distance between horizontal and vertical wing (vpick can have an overlap with hpick earlier)
            r[vpick] = np.min([rv_tmp, r[vpick]], axis = 0)
            # reverse zero r to rv_tmp values
            r[vpick] = np.max([rv_tmp, r[vpick]], axis = 0)
        # check turning region
        ## shrink to a quadrant of pos[:,1]
        tpick = np.sign(pos[0,:] - self.pos[i,0,1]) != np.sign(pos[0,:] - self.pos[i,0,1] - np.copysign(self.rec, self.pos[i,0,1] - self.pos[i,0,turntype]))
        tpick = np.logical_and(tpick, np.sign(pos[1,:] - self.pos[i,1,1]) != np.sign(pos[1,:] - self.pos[i,1,1] - np.copysign(self.rec, self.pos[i,1,1] - self.pos[i,1,2-turntype])))
        if tpick.any():
            d = (pos[:,tpick] - self.pos[i,:,1])
            r[tpick] = np.linalg.norm(d,axis=0)

        # different from r in get_acc, we want the distance to boundary not its lower limit
        outside = np.logical_not(np.logical_or(np.logical_or(tpick, vpick), hpick))
        r[outside] = np.linalg.norm(pos - self.pos[i,:,1].reshape(2,1), axis = 0)
        return r, 0

    def get_turn_vec(self, i, pos, turntype):
        acc = np.zeros((2,pos.shape[1]))
        r = np.zeros(pos.shape[1])
        # check horizontal
        ## pick if x is inbetween self.pos[i,0,0] and self.pos[i,0,1]
        hpick = np.sign(pos[0,:] - self.pos[i,0,1]) != np.sign(pos[0,:] - self.pos[i,0,turntype])
        if hpick.any():
            d = pos[1,hpick] - self.pos[i,1,1]
            rh_tmp = np.abs(d)
            r[hpick] = rh_tmp
            rpick = rh_tmp < self.r0
            hpick[hpick] = rpick
            if rpick.any():
                acc[1,hpick] = np.copysign(self.f(rh_tmp[rpick]), d[rpick])
                print('from horizontal wing')
                print(f'dis: {rh_tmp[rpick]}')
                print(f'acc: {acc[1,hpick]}')
        # check vertical
        ## pick if y is inbetween self.pos[i,1,1] and self.pos[i,1,2]
        vpick = np.sign(pos[1,:] - self.pos[i,1,1]) != np.sign(pos[1,:] - self.pos[i,1,2-turntype])
        if vpick.any():
            d = pos[0,vpick] - self.pos[i,0,1]
            rv_tmp = np.abs(d)
            rpick = rv_tmp < self.r0
            vpick[vpick] = rpick
            if rpick.any():
                acc[0,vpick] = np.copysign(self.f(rv_tmp[rpick]), d[rpick])
                print('from vertical wing')
                print(f'dis: {rv_tmp[rpick]}')
                print(f'acc: {acc[0,vpick]}')
            # choose the min distance between horizontal and vertical wing (vpick can have an overlap with hpick earlier)
            r[vpick] = np.min([rv_tmp, r[vpick]], axis = 0)
            # reverse zero r to rv_tmp values
            r[vpick] = np.max([rv_tmp, r[vpick]], axis = 0)
        # check turning region
        ## shrink to a quadrant of pos[:,1]
        tpick = np.sign(pos[0,:] - self.pos[i,0,1]) != np.sign(pos[0,:] - self.pos[i,0,1] - np.copysign(self.rec, self.pos[i,0,1] - self.pos[i,0,turntype]))
        tpick = np.logical_and(tpick, np.sign(pos[1,:] - self.pos[i,1,1]) != np.sign(pos[1,:] - self.pos[i,1,1] - np.copysign(self.rec, self.pos[i,1,1] - self.pos[i,1,2-turntype])))
        if tpick.any():
            d = (pos[:,tpick] - self.pos[i,:,1])
            r_tmp = np.linalg.norm(d,axis=0)
            r[tpick] = r_tmp
            rpick = r_tmp < self.r0
            tpick[tpick] = rpick
            if rpick.any():
                acc[:,tpick] = self.f(r_tmp[rpick])/r_tmp[rpick] * d[:,rpick]
                print('from the turning point')
                print(f'dis: {r_tmp[rpick]}')
                print(f'acc: {acc[:,tpick]}')
        outside = np.logical_not(np.logical_or(np.logical_or(tpick, vpick), hpick))
        r[outside] = np.max(np.abs(pos - self.pos[i,:,1].reshape(2,1)), axis = 0)
        assert((r[outside] > self.r0).all())
        return acc, r

    def get_rhv_vec(self, i, pos):
        return self.get_rturn_vec(i, pos, 0)

    def get_rvh_vec(self, i, pos):
        return self.get_rturn_vec(i, pos, 2)

    def get_rhv(self, i, pos):
        return self.get_rturn(i, pos, 0)

    def get_rvh(self, i, pos):
        return self.get_rturn(i, pos, 2)

    def get_ahv_vec(self, i, pos):
        return self.get_turn_vec(i, pos, 0)

    def get_avh_vec(self, i, pos):
        return self.get_turn_vec(i, pos, 2)

    def get_ahv(self, i, pos):
        return self.get_turn(i, pos, 0)

    def get_avh(self, i, pos):
        return self.get_turn(i, pos, 2)

def simulate_repel(area, subgrid, pos, dt, boundary, btype, boundary_param = None, particle_param = None, initial_v = None, nlayer = 1, layer = None, layer_seq = None, soft_boundary = None, soft_btype = None, ax = None, seed = None, ns = 1000, ret_vel = False, p_scale = 2.0, b_scale = 1.0, fixed = None, mshape = ',', b_cl = None, i_cl = None):
    # sample points to follow:
    print(boundary.size * pos.size/1024/1024/1024)
    if ax is not None:
        if soft_boundary is not None:
            ax.plot(soft_boundary[:,0,[0,2]].squeeze(), soft_boundary[:,1,[0,2]].squeeze(), ',b')
            ax.plot(soft_boundary[:,0,1].squeeze(), soft_boundary[:,1,1].squeeze(), ',g')
        # connecting points on grid sides
        ax.plot(boundary[:,0,[0,2]].squeeze(), boundary[:,1,[0,2]].squeeze(), ',r')
        # grid centers
        ax.plot(boundary[:,0,1].squeeze(), boundary[:,1,1].squeeze(), ',g')
    if ax is not None and ns == 0:
        print('no sample points selected')
    if ax is not None:
        if ns > 0: 
            if ns ==  pos.shape[1]:
                spick = np.arange(ns)
            else:
                if seed is None:
                    seed = np.int64(time.time())
                    print(f'seed = {seed}')
                np.random.seed(seed)
                if ns > pos.shape[1]:
                    ns = pos.shape[1]
                spick = np.random.choice(pos.shape[1], ns, replace = False)
            spos = np.empty((2,2,ns))
            spos[0,:,:] =  pos[:,spick]
            starting_pos = pos[:,spick].copy()
    # test with default bound potential param
    system = repel_system(area, subgrid, pos, boundary, btype, boundary_param, particle_param, initial_v, nlayer = nlayer, layer = layer, soft_boundary = soft_boundary, soft_btype = soft_btype, p_scale = p_scale, b_scale = b_scale, fixed = fixed, b_cl = b_cl, i_cl = i_cl)
    system.initialize()
    if dt is not None:
        convergence = np.empty((dt.size,2))
        nlimited = np.zeros(dt.size, dtype=int)
        for i in range(dt.size):
            pos, convergence[i,:], nlimited[i], nfreeze = system.next(dt[i], layer_seq)
            stdout.write(f'\r{(i+1)/dt.size*100:.3f}%, {nlimited[i]} particles\' displacement are limited, {nfreeze} particles freezed')
            if ax is not None and ns > 0:
                spos[(i+1)%2,:,:] =  pos[:,spick]
                ax.plot(spos[:,0,:].squeeze(), spos[:,1,:].squeeze(),'-,c', lw = 0.01)
    else:
        convergence = -1
        nlimited = -1

    if ax is not None:
        if system.nlayer == 1:
            ax.plot(pos[0,:], pos[1,:], mshape+'k', ms = 0.01)
        else:
            for i in np.arange(system.nlayer):
                ax.plot(pos[0,system.layer[i]], pos[1,system.layer[i]], mshape)
        if ns > 0:
            ax.plot(starting_pos[0,:], starting_pos[1,:], mshape+'m', ms = 0.01)
    print('\n')
    # normalize convergence of position relative the inter-particle-distance 
    convergence = convergence/system.cl
    if ret_vel is False:
        return pos, convergence, nlimited, system.layer
    else:
        return pos, convergence, nlimited, system.layer, system.particle.vel
