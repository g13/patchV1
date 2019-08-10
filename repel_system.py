import numpy as np
from sys import stdout

class repel_system:
    def __init__(self, area, subgrid, initial_x, bp, btype, boundary_param = None, particle_param = None, initial_v = None, nlayer = 1, enough_memory = False):
        self.nn = initial_x.shape[1]
        self.nb = bp.shape[0]
        self.cl = np.sqrt(area/self.nn)
        a = self.cl/1000

        if particle_param is None:
            self.particle = point_particle(initial_x, L_J_potiential(a,a,2,1,self.cl))
        else:
            self.particle = point_particle(initial_x, particle_param)
        rec = np.max(subgrid)
        if boundary_param is None:
            self.boundary = rec_boundary(rec, bp, btype, L_J_potiential(a,a,2,1,self.cl), enough_memory)
        else:
            self.boundary = rec_boundary(rec, bp, btype, boundary_param, enough_memory)
        print(f'{self.nb} boundary points and {self.nn} particles initialized')
        
        self.nlayer = nlayer
        self.enough_memory = enough_memory 
        self.initialize()

    def initialize(self):
        self.layer = np.empty(self.nlayer, dtype=object)
        if self.nlayer > 1:
            # transform to [x/y,index]
            bp = self.boundary.pos[:,:,1].T
            if not self.enough_memory:
                dis = np.empty(self.nn)
                for i in range(self.nn):
                    dis[i] = np.min(np.sum(np.power(bp - self.particle.pos[:,i],2), axis=0))
                    stdout.write(f'\rinitializing, {(i+1)/self.nn*100:.3f}%')
            else:
                dis = np.min(np.sum(np.power(bp.reshape(2,1,self.bn) - self.particle.pos.reshape(2,self.nn,1), 2), axis=0), axis=-1)
            dis = np.sqrt(dis)

            print('\n')
            dis_range = np.fliplr(np.linspace(np.min(dis[i]), np.max(dis[i]), self.nlayer+1))
            for i in range(self.nlayer):
                if i == self.nlayer-1:
                    self.layer[i] = np.nonzero(dis >= dis_range[i] and dis <= dis_range[i+1])[0]
                else:
                    self.layer[i] = np.nonzero(dis >= dis_range[i] and dis < dis_range[i+1])[0]
        else:
            self.layer[0] = np.arange(self.nn)

        self.get_acc(self.layer[0])
        print('initialized\n')
         
    def get_acc(self, pick):
        pos = self.particle.pos[:,pick]
        # not necessarily faster not tested
        if self.enough_memory:
            ds = pos.reshape(2,pos.shape[1],1) - self.particle.pos.reshape(2,1,self.particle.pos.shape[1])
            # dims 1: x/y. 2: target. 3: source.
            assert(ds.shape[0] == 2 and ds.shape[1] == pos.shape[1] and ds.shape[2] == self.particle.pos.shape[1])
            ## pick a square first
            square_pick = np.max(np.abs(ds), axis = 0) < self.particle.r0
            index = np.arange(pick.size)
            assert(square_pick[index,index].all())
            square_pick[index,index] = False
            ppick = square_pick.any(-1)
            pp = np.arange(pick.size)[ppick]
            pick = pick[ppick]
            ## calculate distance
            for i in range(pick.size):
                dsi = ds[:,pp[i],square_pick[pp[i],:]]
                r = np.linalg.norm(dsi)
                assert(r.ndim == 1)
                rpick = r < self.particle.r0
                assert((self.particle.acc[:,pick[i]] < 1).all())
                ## calculate unit vector
                if rpick.any():
                    direction = dsi[rpick]/r[rpick]
                    self.particle.acc[:,pick[i]] = self.particle.acc[:,pick[i]] + np.sum(self.particle.f(r[rpick])*direction)
                stdout.write(f'\rget_acc: {(i+1)/pick.size*100:.3f}%')
                assert((self.particle.acc[:,pick[i]] < 1).all())
        else:
            # for every particle picked
            for i in range(pick.size):
                # get repel acceleratoin from the nearest boundary
                ds = np.max(np.abs(pos[:,i].reshape(1,2) - self.boundary.pos[:,:,1]), axis = -1)
                ib = np.argmin(ds)
                if ds[ib] < self.boundary.rec:
                    self.particle.acc[:,pick[i]] = self.boundary.get_acc[ib](ib, pos[:,i])
                else:
                    self.particle.acc[:,pick[i]] = 0
                #assert((np.abs(self.particle.acc[:,pick[i]]) < 1).all())
                # accumulate acceleration from particles
                ds = pos[:,i].reshape(2,1) - self.particle.pos
                ## pick a square first
                ppick = np.max(np.abs(ds), axis = 0) < self.particle.r0
                assert(ppick[pick[i]] == True)
                ppick[pick[i]] = False
                if ppick.any():
                    ## calculate distance
                    r = np.linalg.norm(ds[:,ppick])
                    rpick = r < self.particle.r0
                    ppick[ppick] = rpick
                    ## calculate unit vector
                    if rpick.any():
                        direction = ds[:,ppick]/r[rpick]
                        self.particle.acc[:,pick[i]] = self.particle.acc[:,pick[i]] + np.sum(self.particle.f(r[rpick])*direction, axis=1)
                stdout.write(f'\rget_acc: {(i+1)/pick.size*100:.3f}%')
                #assert((np.abs(self.particle.acc[:,pick[i]]) < 1).all())
        print('\n')

    def next(self, dt):
        # update position
        for i in range(self.nlayer):
            pick = self.layer[i]
            self.particle.pos[:,pick] = self.particle.pos[:,pick] + self.particle.vel[:,pick]*dt + 0.5*self.particle.acc[:,pick] * np.power(dt,2)
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
        self.f = lambda r: param.k1/r*param.a*np.power(param.cl/r,param.k1) - param.k2/r*param.b*np.power(param.cl/r,param.k2)

class rec_boundary:
    def __init__(self, rec, pos, btype, param, enough_memory = False):
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
        if enough_memory:
            for i in range(self.n):
                # same y-coord -> horizontal boundary
                if btype[i] == 0:
                    assert((pos[i,1,1] - pos[i,1,:] == 0).all())
                    self.get_acc[i] = self.get_rh_vec
                # same x-coord -> vertical boundary
                elif btype[i] == 1:
                    assert((pos[i,0,1] - pos[i,0,:] == 0).all())
                    self.get_acc[i] = self.get_rv_vec
                else:
                    # first horizontal then vertical
                    if btype[i] == 2:
                        assert(pos[i,1,1] == pos[i,1,0])
                        self.get_acc[i] = self.get_rhv_vec
                    # first vertical then horizontal 
                    else:
                        if btype[i] is not 3:
                            raise Exception(f'btype: {btype[i]} is not implemented')
                        self.get_acc[i] = self.get_rvh_vec
        else:
            for i in range(self.n):
                # same y-coord -> horizontal boundary
                if btype[i] == 0:
                    assert((pos[i,1,1] - pos[i,1,:] == 0).all())
                    self.get_acc[i] = self.get_rh
                # same x-coord -> vertical boundary
                elif btype[i] == 1:
                    assert((pos[i,0,1] - pos[i,0,:] == 0).all())
                    self.get_acc[i] = self.get_rv
                else:
                    # first horizontal then vertical
                    if btype[i] == 2:
                        assert(pos[i,1,1] == pos[i,1,0])
                        self.get_acc[i] = self.get_rhv
                    # first vertical then horizontal 
                    else:
                        if btype[i] is not 3:
                            raise Exception(f'btype: {btype[i]} is not implemented')
                        self.get_acc[i] = self.get_rvh

    # horizontal
    def get_rh(self, i, pos):
        d = pos[1] - self.pos[i,1,1]
        r = np.abs(d)
        if r < self.r0:
            return np.array([0, np.copysign(self.f(r), d)])
        
    def get_rh_vec(self, i, pos):
        hpick = np.sign(pos[0,:] - self.pos[i,0,0]) != np.sign(pos[0,:] - self.pos[i,0,2])
        assert(np.sum(hpick) == pos.shape[1])
        acc = np.zeros((2,pos.shape[1]))
        d = pos[1,hpick] - self.pos[i,1,1]
        r = np.abs(d)
        pick =  r < self.r0
        if pick.any():
            acc[0,pick] = 0
            acc[1,pick] = np.copysign(self.f(r[pick]), d[pick])
            print('from horizontal boundary')
            print(f'dis: {r[pick]}')
            print(f'acc: {acc[1,pick]}')
        return acc
    # vertical 
    def get_rv(self, i, pos):
        d = pos[0] - self.pos[i,0,1]
        r = np.abs(d)
        if r < self.r0:
            return np.array([np.copysign(self.f(r), d), 0])

    def get_rv_vec(self, i, pos):
        acc = np.zeros((2,pos.shape[1]))
        d = pos[0,:] - self.pos[i,0,1]
        r = np.abs(d)
        pick =  r < self.r0
        if pick.any():
            acc[1,pick] = 0
            acc[0,pick] = np.copysign(self.f(r[pick]), d[pick])
            print('from vertical boundary')
            print(f'dis: {r[pick]}')
            print(f'acc: {acc[0,pick]}')
        return acc 
    # horizontal & vertical 
    # turntype == 0: first two points horizontal
    # turntype == 2: first two points vertical 
    def get_turn(self, i, pos, turntype):
        acc = np.zeros(2)
        if np.sign(pos[0] - self.pos[i,0,1]) != np.sign(pos[0] - self.pos[i,0,1] - np.copysign(self.r0, self.pos[i,0,1] - self.pos[i,0,turntype])) and \
        np.sign(pos[1] - self.pos[i,1,1]) != np.sign(pos[1] - self.pos[i,1,1] - np.copysign(self.r0, self.pos[i,1,1] - self.pos[i,1,2-turntype])):
            d = (pos - self.pos[i,:,1])
            r = np.linalg.norm(d)
            if r < self.r0:
                acc = self.f(r)*d/r
        else:
            if np.sign(pos[0] - self.pos[i,0,1]) != np.sign(pos[0] - self.pos[i,0,turntype]):
                d = pos[1] - self.pos[i,1,1]
                r = np.abs(d)
                if r < self.r0:
                    acc[1] = np.copysign(self.f(r), d)

            if np.sign(pos[1] - self.pos[i,1,1]) != np.sign(pos[0] - self.pos[i,1,2-turntype]):
                d = pos[0] - self.pos[i,0,1]
                r = np.abs(d)
                if r < self.r0:
                    acc[0] = np.copysign(self.f(r), d)
        return acc

    def get_turn_vec(self, i, pos, turntype):
        acc = np.zeros((2,pos.shape[1]))
        # check horizontal
        ## pick if x is inbetween self.pos[i,0,0] and self.pos[i,0,1]
        hpick = np.sign(pos[0,:] - self.pos[i,0,1]) != np.sign(pos[0,:] - self.pos[i,0,turntype])
        if hpick.any():
            d = pos[1,hpick] - self.pos[i,1,1]
            r = np.abs(d)
            rpick = r < self.r0
            hpick[hpick] = rpick
            if rpick.any():
                acc[1,hpick] = np.copysign(self.f(r[rpick]), d[rpick])
                print('from horizontal wing')
                print(f'dis: {r[rpick]}')
                print(f'acc: {acc[1,hpick]}')
        # check vertical
        ## pick if y is inbetween self.pos[i,1,1] and self.pos[i,1,2]
        vpick = np.sign(pos[1,:] - self.pos[i,1,1]) != np.sign(pos[1,:] - self.pos[i,1,2-turntype])
        if vpick.any():
            d = pos[0,vpick] - self.pos[i,0,1]
            r = np.abs(d)
            rpick = r < self.r0
            vpick[vpick] = rpick
            if rpick.any():
                acc[0,vpick] = np.copysign(self.f(r[rpick]), d[rpick])
                print('from vertical wing')
                print(f'dis: {r[rpick]}')
                print(f'acc: {acc[0,vpick]}')
        # check turning region
        ## shrink to a quadrant of pos[:,1]
        tpick = np.sign(pos[0,:] - self.pos[i,0,1]) != np.sign(pos[0,:] - self.pos[i,0,1] - np.copysign(self.r0, self.pos[i,0,1] - self.pos[i,0,turntype]))
        tpick = np.logical_and(tpick, np.sign(pos[1,:] - self.pos[i,1,1]) != np.sign(pos[1,:] - self.pos[i,1,1] - np.copysign(self.r0, self.pos[i,1,1] - self.pos[i,1,2-turntype])))
        if tpick.any():
            d = (pos[:,tpick] - self.pos[i,:,1])
            r = np.linalg.norm(d)
            rpick = r < self.r0
            tpick[tpick] = rpick
            if rpick.any():
                acc[:,tpick] = self.f(r[rpick])/r[rpick] * d[rpick]
                print('from the turning point')
                print(f'dis: {r[rpick]}')
                print(f'acc: {acc[:,tpick]}')
        return acc

    def get_rhv_vec(self, i, pos):
        return self.get_turn_vec(i, pos, 0)

    def get_rvh_vec(self, i, pos):
        return self.get_turn_vec(i, pos, 2)

    def get_rhv(self, i, pos):
        return self.get_turn(i, pos, 0)

    def get_rvh(self, i, pos):
        return self.get_turn(i, pos, 2)

def simulate_repel(area, subgrid, pos, dt, boundary, btype, particle_param = None, boundary_param = None, ax = None, seed = None, ns = 100):
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
        spick = np.random.randint(0, pos.shape[1], ns)
        spos = np.empty((dt.size+1,2,ns))
        spos[0,:,:] =  pos[:,spick]
    # test with default bound potential param
    system = repel_system(area, subgrid, pos, boundary, btype, boundary_param, particle_param)
    for i in range(dt.size):
        pos = system.next(dt[i])
        if ax is not None and ns > 0:
            spos[i+1,:,:] =  pos[:,spick]
    if ax is not None and ns > 0:
        ax.plot(spos[:,0,:].squeeze(), spos[:,1,:].squeeze(),'-b', lw=0.1)
        ax.plot(spos[0,0,:].squeeze(), spos[0,1,:].squeeze(),',k')
        ax.plot(spos[1,0,:].squeeze(), spos[1,1,:].squeeze(),',r')

    return pos, spos
