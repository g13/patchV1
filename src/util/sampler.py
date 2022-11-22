import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from scipy.optimize import newton

def ring_dist(nsig, wv = 0.75, n = 1024): # get sampling density based on weight for std-1 gaussian kernel value(wv) and weight for std-1 gaussian kernel slope(ws) in a circle of radius(nsig)
    value = lambda x: np.exp(-x*x)
    ws = 1-wv
    if nsig > np.sqrt(2)/2:
        max_slope = np.sqrt(2)*np.exp(-0.5)
    else:
        max_slope = 2*nsig*np.exp(-nsig*nsig)
    slope = lambda x: 2*x*np.exp(-x*x)/max_slope
    acc_v = lambda x: np.pi*(1-np.exp(-x*x))
    acc_s = lambda x: 2*np.pi/max_slope*(0.5*np.sqrt(np.pi) * erf(x) - np.exp(-x*x)*x)
    n_ratio = n/(acc_v(nsig)*wv + acc_s(nsig)*ws)
    density = lambda x: (value(x)*wv + slope(x)*ws)*n_ratio
    func = lambda x: n_ratio*(acc_v(x)*wv + acc_s(x)*ws)
    fprime = lambda x: 2*np.pi*x*density(x)
    fprime2 = lambda x: 2*np.pi*density(x) + 2*np.pi*x*n_ratio*(wv*(-slope(x)*max_slope) + ws*(2*value(x)/max_slope - 2*x*slope(x)))
    
    r0 = np.sqrt(nsig/n)
    r1 = newton(lambda x: func(x) - 1, r0, fprime = fprime, fprime2 = fprime2)
    r0 = 2*r1 - r0
    r = [r1]
    m = [1]
    avg_ratio = 0
    for i in range(2,n+1):
        func0 = lambda x: func(x) - i
        r1 = newton(func0, r0, fprime = fprime, fprime2 = fprime2)
        avg_chord = 2*np.pi/(i-sum(m))*(r1 + r[-1])/2
        edge_ratio = avg_chord/(r1 - r[-1])
        if edge_ratio <= 1 or i == n:
            avg_ratio += edge_ratio
            r.append(r1)
            m.append(i-sum(m))
            r0 = 2*r1 - r0
    print(f'average areal espect ratio: chord/radius_seg = {avg_ratio/len(r):.2f}')
    return np.array(r), np.array(m)

def sample_phase_ring(r, m, randPhase = True, deg = True, norm_by_r = 1): # attribute sampling positions x, y and area as weight(w) to sample points(m) in ring at radius(r), and normalized by norm_by_r
    r0 = np.hstack((0, r))
    mid_r = (r0[1:] + r0[:-1])/2
    if m[0] == 1:
        mid_r[0] = 0
    n = sum(m)
    x = np.empty(n)
    y = np.empty(n)
    w = np.empty(n)
    m0 = np.hstack((0,np.cumsum(m)))
    if deg:
        deg2rad = np.pi/180
    else:
        deg2rad = 1
    for i in range(r.size):
        if randPhase:
            phase = np.random.rand()*2*np.pi
        else:
            phase = 0 
        polar_angles = np.arange(m[i])/m[i]*2*np.pi
        x[m0[i]:m0[i+1]] = mid_r[i] * np.cos(phase * deg2rad + polar_angles)
        y[m0[i]:m0[i+1]] = mid_r[i] * np.sin(phase * deg2rad + polar_angles)
        w[m0[i]:m0[i+1]] = np.pi*(r0[i+1]*r0[i+1] - r0[i]*r0[i])/m[i]
    return x/norm_by_r, y/norm_by_r, w/np.power(norm_by_r,2)

def sample_2d_gaussian(nsig, n, wv):
    print(f'nsig = {nsig}, wv = {wv}, n = {n}')
    r, m = ring_dist(nsig, wv, n)
    x, y, w = sample_phase_ring(r, m)
    print(f'x[0] = {x[0]}, y[0] = {y[0]}, w[0] = {w[0]}')
    print(f'x[-1] = {x[-1]}, y[-1] = {y[-1]}, w[-1] = {w[-1]}')
    return x, y, w
