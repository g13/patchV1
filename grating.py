import numpy as np
import cv2 as cv
import functools
from ext_signal import *
#TODO: heterogeneous buffers, to save texture memory
def generate_random(amp, cSize, c0, fname, time, frameRate = 120, ecc = 2.5, buffer_ecc = 0.25, neye = 2, gtype = 'uniform', seed = None, shift = None):
    """
    buffer_ecc: buffering area, to avoid border problems in texture memory accesses
    neye == 2:
        frame from 2 visual fields: (ecc+2*buffer_ecc) x 2(ecc+buffer_ecc+buffer_ecc(unused)) (width x height)
        each has a temporal axis: [-buffer_ecc, ecc], and vertical axis [-ecc-buffer_ecc, ecc+buffer_ecc] in degree
    neye == 1:
        frame from a single visual fields: origin at the center 2(ecc+buffer_ecc) x 2(ecc+buffer_ecc) (width x height)
    """
    if neye == 1:
        npixel = int(round(2*(ecc+buffer_ecc)/cSize))
        a = npixel
    else:
        npixel = int(round(2*(ecc+2*buffer_ecc)/cSize))
        a = npixel//2
    b = npixel  
    FourCC = cv.VideoWriter_fourcc(*'HFYU')
    output = cv.VideoWriter(fname+'.avi', FourCC, frameRate, (npixel,npixel), True)

    if isinstance(time, (list, tuple, np.ndarray)):
        nseq = len(time)
    else:
        nseq = 1
        time = np.array([time])

    if isinstance(amp, (list, tuple, np.ndarray)):
        assert(len(amp) == nseq)
    else:
        amp = np.zeros(nseq) + amp

    ########### VIDEO encodes as BGR: 
    # rgb->bgr
    c0 = np.reshape(c0[::-1],(1,3))

    np.random.seed(seed)
    LMS = np.empty((nseq,), dtype=object)
    for i in range(nseq):
        t = time[i]
        nstep = np.round(frameRate * t)
        if not nstep == frameRate*t:
            raise Exception(f'time duration of sequence {i} is not in multiples of frame duration')
        nstep = int(nstep)
        if np.mod(nstep,2) != 0 and gtype == 'rotating':
            raise Exception(f'need even time step, current: {nstep}')
        print(f'{nstep} frames in total')

        LMS_seq = np.empty((nstep,)+(3, npixel, npixel), dtype=float)
        for it in range(nstep):
            if neye == 1:
                data = randomized(amp[i], a, b, c0, gtype)

            else:
                dataL = randomized(amp[i], a, b, c0, gtype)
                if shift is not None:
                    assert(shift == 0)
                    # TODO: implement nonzero shift 
                    dataR = dataL
                else:
                    dataR = randomized(amp[i], a, b, c0, gtype)

                data = np.concatenate((dataL,dataR), axis = 1)

            pixelData = np.reshape(np.round(data*255), (npixel,npixel,3)).astype('uint8')
            # bgr->lms
            LMS_seq[it,:,:,:] = data[:,:,::-1].reshape((npixel*npixel,3)).T.reshape((3,npixel,npixel))

            output.write(pixelData)
            #pixelData = np.reshape(np.round(data*255), (b,a,3))
            #cv.imshow('linear', pixelData)
            #cv.waitKey(0)
            #pixelData = adjust_gamma(pixelData, gamma = 2.2)
            #cv.imshow('gamma', pixelData)
            #cv.waitKey(0)
        LMS[i] = LMS_seq.copy()

    output.release()
    cv.destroyAllWindows()
    return LMS

def generate_grating(amp, spatialFrequency, temporalFrequency, direction, npixel, c1, c2, fname, time, phase, sharpness, frameRate = 120, ecc = 2.5, buffer_ecc = 0.25, gtype = 'drifting', neye = 2, bar = False, center = np.pi/2, wing = np.pi/2, genMovie = True):
    """
    spatialFrequency: cycle per degree
    temporalFrequency: Hz
    direction: 0-2pi in rad
    phase: 0-2pi in rad
    a: width of the half image in pixels 
    b: height of the image in pixels 
    c1, c2: the two opposite color in rgb values
    sharpness:  y = A/(1+exp(-sharpness*(x-0.5)) + C, y=x when sharpness = 0
    buffer_ecc: buffering area, to avoid border problems in texture memory accesses
    neye == 2:
        frame from 2 visual fields: (ecc+2*buffer_ecc) x 2(ecc+buffer_ecc+buffer_ecc(unused)) (width x height)
        each has a temporal axis: [-buffer_ecc, ecc], and vertical axis [-ecc-buffer_ecc, ecc+buffer_ecc] in degree
    neye == 1:
        frame from a single visual fields: origin at the center 2(ecc+buffer_ecc) x 2(ecc+buffer_ecc) (width x height)
    """
    if np.mod(npixel,2) != 0:
        raise Exception("need even pixel")
    if neye == 1:
        a = npixel
    else:
        a = npixel//2  
    b = npixel  
    if genMovie:
        FourCC = cv.VideoWriter_fourcc(*'HFYU')
        output = cv.VideoWriter(fname+'.avi', FourCC, frameRate, (npixel,npixel), True)

    if isinstance(time, (list, tuple, np.ndarray)):
        nseq = len(time)
    else:
        nseq = 1
        time = np.array([time])

    if isinstance(amp, (list, tuple, np.ndarray)):
        assert(len(amp) == nseq)
    else:
        amp = np.zeros(nseq) + amp

    if isinstance(spatialFrequency, (list, tuple, np.ndarray)):
        assert(len(spatialFrequency) == nseq)
    else:
        spatialFrequency = np.zeros(nseq) + spatialFrequency

    if isinstance(temporalFrequency, (list, tuple, np.ndarray)):
        assert(len(temporalFrequency) == nseq)
    else:
        temporalFrequency = np.zeros(nseq) + temporalFrequency

    if isinstance(direction, (list, tuple, np.ndarray)):
        assert(len(direction) == nseq)
    else:
        direction = np.zeros(nseq) + direction
        
    if isinstance(phase, (list, tuple, np.ndarray)):
        assert(len(phase) == nseq)
    else:
        phase = np.zeros(nseq) + phase

    if isinstance(sharpness, (list, tuple, np.ndarray)):
        assert(len(sharpness) == nseq)
    else:
        sharpness = np.zeros(nseq) + sharpness

    ########### VIDEO encodes as BGR: 
    # rgb->bgr
    c1 = np.reshape(c1[::-1],(1,3))
    c2 = np.reshape(c2[::-1],(1,3))

    if neye == 1:
        X, Y = meshgrid(np.linspace(-1,1,a)*(ecc+buffer_ecc)*np.pi/180, np.linspace(-1,1,b)*(ecc+buffer_ecc)*np.pi/180)
    else:
        X, Y = meshgrid((np.linspace(0,1,a)*(ecc+2*buffer_ecc)-buffer_ecc)*np.pi/180,np.linspace(-1,1,b)*(ecc+2*buffer_ecc)*np.pi/180)


    LMS = np.empty((nseq,), dtype=object)
    for i in range(nseq):
        t = time[i]
        nstep = int(np.round(frameRate*t))
        if not nstep == frameRate*t:
            nstep = int(np.round(frameRate*t))
            print(f'adjusted to {nstep} frames in total')
        else:
            print(f'exact {nstep} frames in total')

        if np.mod(nstep,2) != 0 and gtype == 'rotating':
            raise Exception(f'need even time step, current: {nstep}')

        radTF = temporalFrequency[i]*2*np.pi
        radSF = spatialFrequency[i]*180/np.pi*2*np.pi
        s = sharpness[i]
        print(f'sharpness={s}')
        @logistic(s)
        def grating(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y, bar, center = 0, wing = 0):
            return sine_wave(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y, bar, center, wing)

        if gtype == 'rotating':
            half = nstep//2 
            dl = np.linspace(0,np.pi/4,half)
            dr = np.linspace(0,np.pi/4,half)
            dd = np.hstack((dl, np.flip(dr)))
            
        dt = 1.0/frameRate
        #for it in range(1):
        if gtype not in ('drifting','rotating'):
            raise Exception(f'gtype {gtype} not implemented')

        LMS_seq = np.empty((nstep,)+(3, npixel, npixel), dtype=float)
        for it in range(nstep):
            t = it * dt
            if neye == 1:
                if gtype == 'rotating':
                    data = grating(amp[i], radTF, radSF, direction[i]-dd[it], a, b, c1, c2, phase[i], t, X, Y, bar, center, wing)
                if gtype == 'drifting':
                    data = grating(amp[i], radTF, radSF, direction[i], a, b, c1, c2, phase[i], t, X, Y, bar, center, wing)

            else:
                if gtype == 'rotating':
                    dataL = grating(amp[i], radTF, radSF, direction[i]-dd[it], a, b, c1, c2, phase[i], t, X, Y, bar, center, wing)
                    dataR = grating(amp[i], radTF, radSF, direction[i]+dd[it], a, b, c1, c2, phase[i], t, X, Y, bar, center, wing)
                if gtype == 'drifting':
                    dataL = grating(amp[i], radTF, radSF, direction[i], a, b, c1, c2, phase[i], t, X, Y, bar, center, wing)
                    dataR = grating(amp[i], radTF, radSF, direction[i], a, b, c1, c2, phase[i], t, X, Y, bar, center, wing)
                data = np.concatenate((dataL,dataR), axis = 1)

            pixelData = np.reshape(np.round(data*255), (npixel,npixel,3)).astype('uint8')
            # bgr->lms
            LMS_seq[it,:,:,:] = data[:,:,::-1].reshape((npixel*npixel,3)).T.reshape((3,npixel,npixel))

            if genMovie:
                output.write(pixelData)
            #pixelData = np.reshape(np.round(data*255), (b,a,3))
            #cv.imshow('linear', pixelData)
            #cv.waitKey(0)
            #pixelData = adjust_gamma(pixelData, gamma = 2.2)
            #cv.imshow('gamma', pixelData)
            #cv.waitKey(0)
        LMS[i] = LMS_seq.copy()

    if genMovie:
        output.release()
        cv.destroyAllWindows()
    return LMS

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 \
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

def logistic(sharpness):
    def decorator_logistic(function):
        @functools.wraps(function)
        def static_nolinearity(*args, **kwargs):
            x = function(*args, **kwargs)
            if sharpness > 0:
                exp_half = np.exp(sharpness/2)+1
                A = np.power(exp_half,2)/(np.exp(sharpness)-1)
                C = exp_half/(1-np.exp(sharpness))
                x = A/(1.0 + np.exp(-sharpness*(x-0.5))) + C
            else:
                x[x > 0.5] = 1
                x[x < 0.5] = 0
            return x
        return static_nolinearity
    return decorator_logistic

def sine_wave(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y, bar, center, wing):
    phi = (np.cos(direction)*X + np.sin(direction)*Y)*radSF - radTF*t
    if bar:
        #floor_phi = np.floor(phi/(2*np.pi)).astype(int)
        #phi_tmp = phi-floor_phi*2*np.pi
        pick = np.abs(phi + (phase + center)) > wing
        phi[pick] = -phase
    rel_color = np.reshape(1+amp*np.sin(phi + phase), (a*b,1))/2
    color = np.matmul(np.ones((a*b,1)), c1) + np.matmul(rel_color, (c2-c1))
    return color.reshape((b,a,3))/255


def randomized(amp, a, b, c0, gtype):
    if gtype == 'uniform':
        rel_color = 1+amp*np.random.uniform(size = (a*b,3))
    else:
        if gtype == 'normal':
            rel_color = 1+amp*np.random.randn(a*b,3)
        else:
            raise Exception(f'gtype: {gtype} not implemented')
    color = rel_color*c0
    color[color > 255] = 255
    color[color < 0] = 0
    return color.reshape((b,a,3))/255

def meshgrid(x,y):
    X = np.tile(x,(len(y),1))
    Y = np.tile(y,(len(x),1)).T
    return X, Y

# to check cuda prog output
def generate_from_float(fname, b, a, nt, cs_transform=LMS2sRGB, frameRate=60, suffix='bin'):
    with open(fname+'.'+suffix, 'rb') as f:
        data = np.fromfile(f, 'f4', count = nt*3*b*a).reshape((nt,3,b*a))

    FourCC = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter(fname+'.avi', FourCC, frameRate, (a,b), True)

    for it in range(nt):
        frameData = np.matmul(cs_transform, data[it,:,:]).T
        #print(frameData.shape)
        pixelData = np.reshape(np.round(frameData*255), (b,a,3)).astype('uint8')
        output.write(pixelData)
    
    output.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    video_fn = 'color_drifting_2i'
    stimulus_fn = video_fn + '.bin'
    crest = [255,0,0]
    valley = [0,255,0]
    buffer_deg = 1.0
    range_deg = 2.5 # eccentricity from the origin
    SF = 4
    TF = 8
    
    orient = np.array([np.pi*3/4, np.pi*1/4])
    time = np.array([1.2, 1.2])
    phase = 8.8*np.pi
    center = np.pi/2
    wing = np.pi/2
    sharpness = 1
    LMS_series = generate_grating(1.0, SF, TF, orient, 256, crest, valley, video_fn, time, phase, sharpness, frameRate = 120, ecc = range_deg, buffer_ecc = buffer_deg, gtype='drifting', neye = 2, bar = False, center = center, wing = wing)
    
    print(LMS_series.shape)
    nseq = LMS_series.size
    for i in range(nseq):
        print([np.min(LMS_series[i]), np.max(LMS_series[i])])
    
    with open(stimulus_fn, 'wb') as f:
        np.array([LMS_series[0].shape[0], LMS_series[0].shape[2], LMS_series[0].shape[3]], dtype='i4').tofile(f)
        mean_value = np.array([np.mean(LMS_series[0][:,0,:,:]), np.mean(LMS_series[0][:,1,:,:]), np.mean(LMS_series[0][:,2,:,:])])
        mean_value.astype('f4').tofile(f) # init_luminance
        np.array([buffer_deg, range_deg], dtype='f4').tofile(f)
        print([buffer_deg, range_deg])
        for i in range(nseq):
            print(i)
            LMS_series[i].astype('f4').tofile(f)
    print(mean_value)
