import numpy as np
import cv2 as cv
import functools
from ext_signal import *
#TODO: heterogeneous buffers, to save texture memory
def generate_grating(amp, spatialFrequency, temporalFrequency, direction, npixel, c1, c2, fname, time = 1, phase = 0, sharpness = 0, frameRate = 120, ecc = 2.5, buffer_ecc = 0.25, gtype = 'drifting'):
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
    frame from 2 visual fields: 2(ecc+buffer_ecc) x 2(ecc+buffer_ecc) (width x height)
    each has a temporal axis: [-buffer_ecc, ecc], and vertical axis [-ecc-buffer_ecc, ecc+buffer_ecc] in degree
    """
    if np.mod(npixel,2) != 0:
        raise Exception("need even pixel")
    nstep = np.round(frameRate * time)
    if np.mod(nstep,2) != 0 and gtype == 'rotating':
        raise Exception(f'need even time step, current: {nstep}')

    a = npixel//2  
    b = npixel  
    FourCC = cv.VideoWriter_fourcc(*'HFYU')
    output = cv.VideoWriter(fname+'.avi', FourCC, frameRate, (npixel,npixel), True)
    print(f'{nstep} frames in total')
    radTF = temporalFrequency*2*np.pi
    radSF = spatialFrequency*180/np.pi*2*np.pi
    ########### VIDEO encodes as BGR: 
    # rgb->bgr
    c1 = np.reshape(c1[::-1],(1,3))
    c2 = np.reshape(c2[::-1],(1,3))

    X, Y = meshgrid((np.linspace(0,1,a)*(ecc+buffer_ecc)-buffer_ecc)*np.pi/180,np.linspace(-1,1,b)*(ecc+buffer_ecc)*np.pi/180)

    print(f'sharpness={sharpness}')
    @logistic(sharpness)
    def grating(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y):
        return sine_wave(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y)

    half = nstep//2 
    dl = np.linspace(0,np.pi/4,half)
    dr = np.linspace(0,np.pi/4,half)
    dd = np.hstack((dl, np.flip(dr)))
        
    dt = 1.0/frameRate
    #for it in range(1):
    if gtype not in ('drifting','rotating'):
        raise Exception(f'gtype {gtype} not implemented')
    LMS = np.empty((nstep,)+(3, b, 2*a), dtype=float)
    for it in range(nstep):
        t = it * dt
        if gtype == 'rotating':
            dataL = grating(amp, radTF, radSF, direction-dd[it], a, b, c1, c2, phase, t, X, Y)
            dataR = grating(amp, radTF, radSF, direction+dd[it], a, b, c1, c2, phase, t, X, Y)
        if gtype == 'drifting':
            dataL = grating(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y)
            dataR = grating(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y)

        data = np.concatenate((dataL,dataR), axis = 1)
        pixelData = np.reshape(np.round(data*255), (b,2*a,3)).astype('uint8')
        # bgr->lms
        LMS[it,:,:,:] = data[:,:,::-1].reshape((b*2*a,3)).T.reshape((3,b,2*a))
        #pixelData = np.reshape(np.round(data*255), (b,a,3))
        #cv.imshow('linear', pixelData)
        #cv.waitKey(0)
        #pixelData = adjust_gamma(pixelData, gamma = 2.2)
        #cv.imshow('gamma', pixelData)
        #cv.waitKey(0)
        output.write(pixelData)
    
    output.release()
    cv.destroyAllWindows()
    return LMS

def generate_cyclop(amp, spatialFrequency, temporalFrequency, direction, npixel, c1, c2, fname, time = 1, phase = 0, sharpness = 0, frameRate = 120, ecc = 2.5, buffer_ecc = 0.25, gtype = 'drifting'):
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
    frame from a single visual fields: origin at the center 2(ecc+buffer_ecc) x 2(ecc+buffer_ecc) (width x height)
    """
    nstep = np.round(frameRate * time)
    if np.mod(nstep,2) != 0 and gtype == 'rotating':
        raise Exception(f'need even time step, current: {nstep}')

    a = npixel
    b = npixel  
    FourCC = cv.VideoWriter_fourcc(*'HFYU')
    output = cv.VideoWriter(fname+'.avi', FourCC, frameRate, (npixel,npixel), True)
    print(f'{nstep} frames in total')
    radTF = temporalFrequency*2*np.pi
    radSF = spatialFrequency*180/np.pi*2*np.pi
    ########### VIDEO encodes as BGR: 
    # rgb->bgr
    c1 = np.reshape(c1[::-1],(1,3))
    c2 = np.reshape(c2[::-1],(1,3))

    X, Y = meshgrid(np.linspace(-1,1,a)*(ecc+buffer_ecc)*np.pi/180, np.linspace(-1,1,b)*(ecc+buffer_ecc)*np.pi/180)

    print(f'sharpness={sharpness}')
    @logistic(sharpness)
    def grating(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y):
        return sine_wave(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y)

    half = nstep//2 
    dd = np.linspace(0,np.pi/4,half)
        
    dt = 1.0/frameRate
    #for it in range(1):
    if gtype not in ('drifting','rotating'):
        raise Exception(f'gtype {gtype} not implemented')
    LMS = np.empty((nstep,)+(3, b, a), dtype=float)
    for it in range(nstep):
        t = it * dt
        if gtype == 'rotating':
            data = grating(amp, radTF, radSF, direction-dd[it], a, b, c1, c2, phase, t, X, Y)
        if gtype == 'drifting':
            data = grating(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y)

        pixelData = np.reshape(np.round(data*255), (b,a,3)).astype('uint8')
        # bgr->lms
        LMS[it,:,:,:] = data[:,:,::-1].reshape((b*a,3)).T.reshape((3,b,a))
        #pixelData = np.reshape(np.round(data*255), (b,a,3))
        #cv.imshow('linear', pixelData)
        #cv.waitKey(0)
        #pixelData = adjust_gamma(pixelData, gamma = 2.2)
        #cv.imshow('gamma', pixelData)
        #cv.waitKey(0)
        output.write(pixelData)
    
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
            return x
        return static_nolinearity
    return decorator_logistic

def sine_wave(amp, radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y):
    rel_color = np.reshape(1+amp*np.sin((np.cos(direction)*X + np.sin(direction)*Y)*radSF - radTF*t + phase), (a*b,1))/2
    color = np.matmul(np.ones((a*b,1)), c1) + np.matmul(rel_color, (c2-c1))
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
