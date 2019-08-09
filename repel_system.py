import numpy as np
from sys import stdout

class repel_system:
    def __init__(self, area, subgrid, initial_x, bp, btype, boundary_param = None, particle_param = None, initial_v = None, nlayer = 1):
        self.nn = initial_x.shape[1]
        self.nb = bp.shape[0]
        self.cl = np.sqrt(area/self.nn)
        a = self.cl/10

        if particle_param is None:
            self.particle = point_particle(initial_x, L_J_potiential(a,a,2,1,self.cl))
        else:
            self.particle = point_particle(initial_x, particle_param)
        rec = np.max(subgrid)
        if boundary_param is None:
            self.boundary = rec_boundary(rec, bp, btype, L_J_potiential(a,a,2,1,self.cl))
        else:
            self.boundary = rec_boundary(rec, bp, btype, boundary_param)
        print(f'{self.nb} boundary points and {self.nn} particles initialized')
        
        self.nlayer = nlayer
        self.initialize()

    def initialize(self):
        self.layer = np.empty(self.nlayer, dtype=object)
        if self.nlayer > 1:
            # transform to [x/y,index]
            bp = self.boundary.pos[:,:,1].T
            dis = np.empty(self.nn)
            for i in range(self.nn):
                bpick = np.max(np.abs(bp - self.particle.pos[:,i]), axis=0) < self.boundary.r0
                dis[i] = np.min(np.linalg.norm(bp[:,bpick] - self.particle.pos[:,i]))
                stdout.write(f'\rinitializing, {(i+1)/self.nn*100}%')
            dis_range = np.fliplr(np.linspace(np.min(dis[i]), np.max(dis[i]), self.nlayer+1))
            for i in range(self.nlayer):
                if i == self.nlayer-1:
                    self.layer[i] = np.nonzero(dis >= dis_range[i] and dis <= dis_range[i+1])[0]
                else:
                    self.layer[i] = np.nonzero(dis >= dis_range[i] and dis < dis_range[i+1])[0]
        else:
            self.layer[0] = np.arange(self.nn)

        self.get_acc(self.layer[0])
        print('initialized')
         
    def get_acc(self, pick):
        pos = self.particle.pos[:,pick]
        # accumulate acceleration from boundary
        for i in range(self.nb):
            ds = pos - self.boundary.pos[i,:,1].reshape(2,1)
            # find the nearest boundary point
            bpick = np.max(np.abs(ds), axis = 0) < self.boundary.rec
            self.particle.acc[:,pick[bpick]] = self.boundary.get_acc[i](i, pos[:,pick[bpick]])

        # for every particle picked
        for i in range(pick.size):
            # accumulate acceleration from particles
            ds = pos - self.particle.pos
            ## pick a square first
            ppick = np.max(np.abs(ds), axis = 0) < self.particle.r0
            ## calculate distance
            r = np.linalg.norm(ds[:,ppick])
            rpick = r < self.particle.r0
            assert(rpick.size == np.sum(ppick))
            ## calculate unit vector
            direction = ds[:,ppick][:,rpick]/r[rpick]
            self.particle.acc[:,pick[i]] = acc + np.sum(self.particle.f(r[rpick])*direction, axis=1)

    def next(self, dt):
        # update position
        for i in range(self.nlayer):
            pick = self.layer[i]
            self.particle.pos[:,pick] = self.particle.pos[:,pick] + self.particle.vel[:,pick]*self.dt + 0.5*self.particle.acc[:,pick] * np.power(dt,2)
            if i < self.nlayer-1:
                self.get_acc(self.layer[i+1])
        # update velocity and acc
        for i in range(self.nlayer):
            pick = self.layer[i]
            last_acc = self.particle.acc[:,pick]
            self.get_acc(pick)
            self.particle.vel[:,pick] = self.particle.vel[:,pick] + 0.5*(self.particle.acc[:,pick]+last_acc)*dt
        return self.particle.pos

class L_J_potiential:
    def __init__(self, a, b, k1, k2, cl):
        self.a = a
        self.b = b
        self.k1 = k1
        self.k2 = k2
        self.cl = cl

class point_particle:
    def __init__(self, initial_x, param, initial_v = None):
        self.pos = initial_x
        self.n = self.pos.shape[1]
        if initial_v is not None:
            self.vel = initial_v
        else:
            self.vel = np.zeros((2,self.n))
        self.acc = np.empty((2,self.n))

       # potential Lennard-Jones Potential
        self.r0 = np.power(param.b*param.k2/param.a/param.k1,-1/(param.k1-param.k2))*param.cl
        shift = param.a*np.power(param.cl/self.r0,param.k1) - param.b*np.power(param.cl/self.r0,param.k2)
        #self.ph = lambda r: param.a*np.power(param.cl/r,param.k1) - param.b*np.power(param.cl/r.param.k2) - shift
        self.f = lambda r: param.k1/r*param.a*np.power(param.cl/r,param.k1) - param.k2/r*param.b*np.power(param.cl/r.param.k2)

class rec_boundary:
    def __init__(self, rec, pos, btype, param):
        self.pos = pos
        self.n = pos.shape[0]
        assert(pos.shape[1] == 2 and pos.shape[2] == 3)
        self.r0 = np.power(param.b*param.k2/param.a/param.k1,-1/(param.k1-param.k2))*param.cl
        # set a radius to pick nearby particles
        if self.r0 > rec:
            self.rec = self.r0
        else:
            self.rec = rec

        self.f = lambda r: param.k1/r*param.a*np.power(param.cl/r,param.k1) - param.k2/r*param.b*np.power(param.cl/r,param.k2)
        self.get_acc = np.empty(self.n, dtype = object)
        for i in range(self.n):
            # same y-coord -> horizontal boundary
            if (pos[i,1,1] - pos[i,1,:] == 0).all() and btype[i] == 0:
                self.get_acc[i] = self.get_rh
            # same x-coord -> vertical boundary
            elif (pos[i,0,1] - pos[i,0,:] == 0).all() and btype[i] == 1:
                self.get_acc[i] = self.get_rv
            else:
                # first horizontal then vertical
                if (pos[i,1,1] == pos[i,1,0]) and btype[i] == 2:
                    self.get_acc[i] = self.get_rhv
                # first vertical then horizontal 
                else:
                    if btype[i] is not 3:
                        raise Exception(f'btype: {btype[i]} is not implemented')
                    self.get_acc[i] = self.get_rvh
    # horizontal
    def get_rh(self, i, pos):
        acc = np.zeros((2,pos.shape[1]))
        d = pos[1,:] - self.pos[i,1,1]
        r = np.abs(d)
        pick =  r < self.r0
        acc[0,pick] = 0
        acc[1,pick] = np.copysign(self.f(r), d)
        return acc
    # vertical 
    def get_rv(self, i, pos):
        acc = np.zeros((2,pos.shape[1]))
        d = pos[0,:] - self.pos[i,0,1]
        r = np.abs(d)
        pick =  r < self.r0
        acc[1,pick] = 0
        acc[0,pick] = np.copysign(self.f(r), d)
        return acc 
    # horizontal & vertical 
    # turntype == 0: first two points horizontal
    # turntype == 2: first two points vertical 
    def get_turn(self, i, pos, turnype):
        acc = np.zeros((2,pos.shape[1]))
        # check horizontal
        ## pick if x is inbetween self.pos[i,0,0] and self.pos[i,0,1]
        hpick = np.logical_xor(np.sign(pos[0,:] - self.pos[i,0,1]), np.sign(pos[0,:] - self.pos[i,0,turntype]))
        d = pos[1,hpick] - self.pos[i,1,1]
        r = np.abs(d)
        hpick = np.logical_and(hpick, r < self.r0)
        acc[1,hpick] = np.copysign(self.f(r), d)
        # check vertical
        ## pick if y is inbetween self.pos[i,1,1] and self.pos[i,1,2]
        vpick = np.logical_xor(np.sign(pos[1,:] - self.pos[i,1,1]), np.sign(pos[1,:] - self.pos[i,1,2-turntype]))
        d = pos[0,vpick] - self.pos[i,0,1]
        r = np.abs(d)
        vpick = np.logical_and(vpick, r < self.r0)
        acc[0,vpick] = np.copysign(self.f(r), d)
        # check turning region
        ## shrink to a quadrant of pos[:,1]
        tpick = np.logical_xor(np.sign(pos[0,:] - self.pos[i,0,1]), np.sign(pos[0,:] - self.pos[i,0,1] - np.copysign(self.r0, pos[0,1] - pos[0,turntype])))
        tpick = np.logical_and(tpick, np.logical_xor(np.sign(pos[1,:] - self.pos[i,1,1]), np.sign(pos[1,:] - self.pos[i,1,1] - np.copysign(self.r0, pos[1,1] - pos[1,2-turntype]))))
        d = (pos[:,tpick] - self.pos[i,:,1])
        r = np.linalg.norm(d)
        acc[:,tpick] = self.f(r)/r * d
        return acc

    def get_rhv(self, i, pos):
        return self.get_turn(i, pos, 0)

    def get_rvh(self, i, pos):
        return self.get_turn(i, pos, 2)

def simulate_repel(area, subgrid, pos, dt, boundary, btype, particle_param = None, boundary_param = None, ax = None, seed = None, ns = 1000):
    # sample points to follow:
    if ax is not None:
        ax.plot(boundary[:,0,:].squeeze(), boundary[:,1,:].squeeze(), ',g')
    if ax is not None and ns == 0:
        print('no sample points selected')
    if ax is not None and ns > 0: 
        if seed is None:
            seed = time.time()
            print(f'seed = {seed}')
        np.random.seed(seed)
        spick = np.random.randint(0, pos.shape[1], (dt.size,ns))
        spos = np.empty((dt.size,2,ns))
    # test with default bound potential param
    system = repel_system(area, subgrid, pos, boundary, btype, boundary_param, particle_param)
    for i in range(dt.size):
        pos = system.next(dt[i])
        if ax is not None and ns > 0:
            spos[i,:,:] =  pos[:,spick]
    if ax is not None and ns > 0:
        ax.plot(spos[:,0,:].squeeze(), spos[:,1,:].squeeze())

    return pos
