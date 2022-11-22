import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from scipy.optimize import newton

def ring_dist(nsig, wv = 0.75, n = 1024, min_p = 1, baseRatio = 0.5, ratio_thres = 1.1825): # get sampling density based on weight for std-1 gaussian kernel value(wv) and weight for std-1 gaussian kernel slope(ws) in a circle of radius(nsig)
    sigma = 1/np.sqrt(2)
    r_max = nsig * sigma
    if wv > 0:
        r0 = r_max/np.sqrt(n)
        baseline = 1/r0  * baseRatio # minimal linear density when wv = 1.0
        value = lambda x: np.exp(-x*x) + baseline # default sig is 1/sqrt(2)
        ws = 1 - wv
        if r_max > np.sqrt(2)/2:
            max_slope = np.sqrt(2)*np.exp(-0.5)
        else:
            max_slope = 2*r_max*np.exp(-r_max*r_max)
        slope = lambda x: 2*x*np.exp(-x*x)/max_slope
        acc_v = lambda x: np.pi*((1-np.exp(-x*x)) + baseline*x*x)
        acc_s = lambda x: 2*np.pi/max_slope*(0.5*np.sqrt(np.pi) * erf(x) - np.exp(-x*x)*x) 
        n_ratio = n/(acc_v(r_max)*wv + acc_s(r_max)*ws)
        density = lambda x: (value(x)*wv + slope(x)*ws)*n_ratio
        func = lambda x: n_ratio*(acc_v(x)*wv + acc_s(x)*ws)
        fprime = lambda x: 2*np.pi*x*density(x)
        fprime2 = lambda x: 2*np.pi*density(x) + 2*np.pi*x*n_ratio*(wv*(-slope(x)*max_slope) + ws*(2*value(x)/max_slope - 2*x*slope(x)))
        r1 = newton(lambda x: func(x) - 1, r0, fprime = fprime, fprime2 = fprime2)
        r0 = 2*r1
        r = [r1]
    else:
        density = n/(np.pi*r_max*r_max)
        r1 = np.sqrt(1/(density * np.pi))
        r = [r1]

    m = [1]
    edge_ratio_old = 0
    avg_ratio = edge_ratio_old
    for i in range(2,n+1):
        if wv > 0:
            func0 = lambda x: func(x) - i
            r1 = newton(func0, r0, fprime = fprime, fprime2 = fprime2)
        else:
            r1 = np.sqrt(i/(density*np.pi))

        avg_chord = 2*np.pi/(i-sum(m))*(r1 + r[-1])/2
        edge_ratio = avg_chord/(r1 - r[-1])

        if edge_ratio <= ratio_thres and i-sum(m) >= min_p or i == n:
            if np.abs(edge_ratio - ratio_thres) < np.abs(edge_ratio_old - ratio_thres) or i == n:
                if i < n:
                    avg_ratio += edge_ratio
                r0 = 2*r1 - r[-1]
                r.append(r1)
                m.append(i-sum(m))

                edge_ratio_old = 0
            else:
                if i < n:
                    avg_ratio += edge_ratio_old
                r0 = 2*r_old - r[-1]
                r.append(r_old)
                m.append(i-sum(m)-1)

                edge_ratio_old = np.pi*(r1 + r[-1])/(r1 - r[-1])
            continue

        r_old = r1
        edge_ratio_old = edge_ratio


    print(f'average areal espect ratio (exclude first and last): chord/radius_seg = {avg_ratio/(len(r)-2):.2f}')
    print(f'last ring has {m[-1]} sample points, radius at {r[-1]:.3f}')
    assert(len(m) == len(r))
    assert(sum(m) == n)
    return np.array(r), np.array(m)

def sample_phase_ring(r, m, randPhase = True, deg = True, norm_by_r = 1): # attribute sampling positions x, y and area as weight(w) to sample points(m) in ring at radius(r), and normalized by norm_by_r
    norm_by_r /= np.sqrt(2)
    r0 = np.hstack((0, r))
    n = sum(m)
    x = np.empty(n)
    y = np.empty(n)
    w = np.empty(n)
    m0 = np.hstack((0,np.cumsum(m)))
    if deg:
        deg2rad = np.pi/180
    else:
        deg2rad = 1
    phase = 0
    for i in range(r.size):
        polar_angles = np.arange(m[i])/m[i]*2*np.pi
        r2_diff = r0[i+1]*r0[i+1] - r0[i]*r0[i]
        if i== 0 and m[i] == 1:
            exact_r = 0
        else:
            exact_r = np.sqrt(np.log(r2_diff/(np.exp(-r0[i]*r0[i]) - np.exp(-r0[i+1]*r0[i+1]))))
        if i > 0:
            #phase = phase + 1.5 * np.pi*2/m[i-1]
            phase = np.random.rand()*np.pi*2

        x[m0[i]:m0[i+1]] = exact_r * np.cos(phase * deg2rad + polar_angles)
        y[m0[i]:m0[i+1]] = exact_r * np.sin(phase * deg2rad + polar_angles)
        w[m0[i]:m0[i+1]] = np.pi * r2_diff / m[i]
    return x/norm_by_r, y/norm_by_r, w/np.power(norm_by_r,2)

def sample_2d_gaussian(nsig, n, wv):
    print(f'nsig = {nsig}, wv = {wv}, n = {n}')
    r, m = ring_dist(nsig, wv, n)
    x, y, w = sample_phase_ring(r, m)
    print(f'x[0] = {x[0]:.3f}, y[0] = {y[0]:.3f}, w[0] = {w[0]:.3f}')
    print(f'x[-1] = {x[-1]:.3f}, y[-1] = {y[-1]:.3f}, w[-1] = {w[-1]:.3f}')

    return x, y, w
