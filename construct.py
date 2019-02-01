import numpy as np
#from numba import njit, prange
import math
import os
os.environ['NUMBAPRO_CUDALIB']='C:/Users/gueux/Miniconda3/envs/py36_7/Library/bin'
from pyculib import rand

class LineSegment(): # slope ~= 0
    def __init__(self, p0, p1, checkType = 'lr'):
        assert(p0.size == 2)
        assert(p1.size == 2)
        self.p0 = p0
        self.p1 = p1
        self.updateCheckType(checkType)
    
    def updateCheckType(self, checkType):
        self.checkType = checkType
        rel = self.p1-self.p0
        if checkType == 'lr':
            self.check = self.check_lr
            if rel[1] == 0:
                self.inverted_slope = np.float('inf')
            else:
                self.inverted_slope = rel[0]/rel[1]
        else:
            if checkType == 'bt':
                self.check = self.check_bt
                if rel[0] == 0:
                    self.slope = np.float('inf')
                else:
                    self.slope = rel[1]/rel[0]
            else:
                print('checkType undefined, using defaut (given y check x)')
                self.check = self.check_lr
        
    def check_bt(self, x):
        if self.slope == np.float('inf'):
            return self.p0[1]
        else:
            return self.p0[1] + (x-self.p0[0])*self.slope
        
    def check_lr(self, y):
        if self.inverted_slope == np.float('inf'):
            return self.p0[0]
        else:
            return self.p0[0] + (y-self.p0[1])*self.inverted_slope
    
    def xmin(self):
        return min(self.p0[0],self.p1[0])
    
    def xmax(self):
        return max(self.p0[0],self.p1[0])
    
    def ymin(self):
        return min(self.p0[1],self.p1[1])
    
    def ymax(self):
        return max(self.p0[1],self.p1[1])
    
    def generator(self, n):
        dy = (self.p1[1] - self.p0[1])/n
        for ir in range(n+1):
            y = self.p0[1] + dy*ir
            x = self.check_lr(y)
            yield x, y

class CircularSegment(): # angle < np.pi
    def __init__(self, x0, y0, radius, phase, angle, lr, checkType = 'lr'):
        #lr = -1 for the left or bottom segment of the circle
        #lr =  1 for the right or top
        assert(0 < angle and angle < np.pi)
        self.x0 = x0
        self.y0 = y0
        self.radius = radius
        self.phase = phase%(np.pi*2)
        self.angle = angle
        #print(f'myface = {self.phase}, angel = {self.angle}')
        self.lr = lr
        self.r2 = radius*radius
        self.end = self.angle + self.phase
        #print(f'x0 = {self.x0}, r={self.radius}: {self.xmin()}, {self.xmax()}')
        self.updateCheckType(checkType)
        
    def updateCheckType(self, checkType):
        self.checkType = checkType
        if checkType == 'lr':
            self.check = self.check_lr
        else:
            if checkType == 'bt':
                self.check = self.check_bt
            else:
                print('checkType undefined, using defaut (given y check x)')
                self.check = self.check_lr
        if (self.phase < np.pi and np.pi < self.end or self.phase < np.pi*2 and np.pi*2 < self.end) and self.checkType == 'bt':
            self.check = self.check_lr
            print('check type not matching, changed to given y check x')
        if (self.phase < np.pi*3/2 and np.pi*3/2 < self.end or self.phase < np.pi/2 and np.pi/2 < self.end) and self.checkType == 'lr':
            print('check type not matching, changing to given x check y')
            self.check = self.check_bt
    
    def copy(self, phase, angle):
        copied = CircularSegment(self.x0, self.y0, self.radius, phase, angle, self.lr)
        return copied
        
    def xmin(self):
        if self.phase < np.pi and np.pi < self.end:
            return self.x0 - self.radius
        else:
            return self.x0 + self.radius*min(np.cos(self.end), np.cos(self.phase))
            
    def xmax(self):
        if self.phase < 2*np.pi and 2*np.pi < self.end:
            return self.x0 + self.radius
        else:
            return self.x0 + self.radius*max(np.cos(self.end), np.cos(self.phase))
        
    def ymin(self):
        if self.phase < np.pi*3/2 and np.pi*3/2 < self.end:
            return self.y0 - self.radius
        else:
            return self.y0 + self.radius*min(np.sin(self.end), np.sin(self.phase))
        
    def ymax(self):
        if self.phase < np.pi/2 and np.pi/2 < self.end:
            return self.y0 + self.radius
        else:
            return self.y0 + self.radius*max(np.sin(self.end), np.sin(self.phase))
                
        
    def check_lr(self, y):
        return self.x0 + self.lr*np.sqrt(self.r2 - np.power(y-self.y0,2))
    
    def check_bt(self, x):
        return self.y0 + self.lr*np.sqrt(self.r2 - np.power(x-self.x0,2))
    
    def generator(self, ntheta):
        dtheta = self.angle/ntheta
        for itheta in range(ntheta+1):
            theta = self.phase + itheta*dtheta
            x = self.x0 + self.radius*np.cos(theta)
            y = self.y0 + self.radius*np.sin(theta)
            yield x, y

def draw(n, curve, ax, lw = 1, ls = '-'): # just having fun with generators
    x = np.empty(n+1)
    y = np.empty(n+1)
    i = 0
    for p, q in curve(n):
        x[i], y[i] = p, q
        i += 1
    ax.plot(x, y, ls, lw = lw)
    return x[0], y[0], x[-1], y[-1]

def poly_area(x, y, n):
    r = 0.0
    for i in range(n-1):
        r += x[i]*y[i+1] - x[i+1]*y[i]
    r += x[n-1]*y[0] - x[0]*y[n-1]
    return np.abs(r)/2   

def seg_area(angle, radius):
    theta = angle
    return radius*radius/2*(theta-np.sin(theta))

def generate_pos_2d(lcurve, rcurve, target_area, ly, ry, n, seed):
    nl = ly.size-1
    nr = ry.size-1
    assert(len(lcurve) == nl)
    assert(len(rcurve) == nr)
    assert(min(ly) == min(ry))
    assert(max(ly) == max(ry))
    
    def collect_2d(n, x, y):
        selected = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(nl):
                if ly[j] < y[i] and y[i] < ly[j+1]:
                    if lcurve[j].check(y[i]) > x[i]:
                        selected[i] = False
                        break
            if selected[i]:
                for j in range(nr):
                    if ry[j] < y[i] and y[i] < ry[j+1]:
                        if rcurve[j].check(y[i]) < x[i]:
                            selected[i] = False
                            break
        pos = np.array([x[selected], y[selected]])
        return pos, sum(selected)
    
    xmax = max([r.xmax() for r in rcurve])
    xmin = min([l.xmin() for l in lcurve])
    ymax = max(ly)
    ymin = min(ly)
    storming_area = (xmax-xmin) * (ymax-ymin);
    #print(xmax,xmin,ymax,ymin, storming_area)

    ratio = storming_area/target_area
    #print(ratio)
    # to do: if ratio > some threshold rotate the coordinates and implement methods for the corresponding change for the Class of curves, too
    ntmp = np.ceil(n*ratio).astype(int)
    i = 0
    pos = np.empty((2,n), dtype=float)
    prng = rand.PRNG(rndtype=rand.PRNG.XORWOW, seed=seed)
    #qrng = rand.QRNG(rndtype=rand.QRNG.SCRAMBLED_SOBOL32, ndim=2)
    #count = 0
    while (i < n):
        #irands = np.empty((2,ntmp), dtype='uint32')
        rands = np.empty(ntmp, dtype=float)
        
        prng.uniform(rands)
        #qrng.generate(irands)
        #rands = irands[0,:]/np.iinfo(np.dtype('uint32')).max
        x = xmin + (xmax-xmin) * rands
        
        prng.uniform(rands)
        #qrng.generate(irands)
        #rands = irands[1,:]/np.iinfo(np.dtype('uint32')).max
        y = ymin + (ymax-ymin) * rands
        
        acquired_pos, nselected = collect_2d(ntmp, x, y)
        if i+nselected > n:
            nselected = n-i
        pos[:,i:i+nselected] = acquired_pos[:, :nselected]
        i += nselected
        ntmp = np.ceil((n-i)*ratio).astype(int)
        #count += 1
    #print(count)
    return pos