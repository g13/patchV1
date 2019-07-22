import numpy as np
import cv2 as cv
import functools
from ext_signal import *

def generate_grating(spatialFrequency, temporalFrequency, direction, a, b, c1, c2, fname, time = 1, phase = 0, sharpness = 0, frameRate=120):
    """
    spatialFrequency: cycle per degree
    temporalFrequency: Hz
    direction: 0-2pi in rad
    phase: 0-2pi in rad
    a: major axis of the ellipse, width of the image in pixels 
    b: minor axis of the ellipse, width of the image in pixels 
    c1, c2: the two opposite color in rgb values
    sharpness:  y = A/(1+exp(-sharpness*(x-0.5)) + C, y=x when sharpness = 0
    """
    FourCC = cv.VideoWriter_fourcc(*'H264')
    output = cv.VideoWriter(fname+'.avi', FourCC, frameRate, (a,b), True)
    nstep = np.round(frameRate * time)
    print(f'{nstep} frames in total')
    radTF = temporalFrequency*2*np.pi
    radSF = spatialFrequency*180/np.pi*2*np.pi
    c1 = np.reshape(c1[::-1],(1,3))
    c2 = np.reshape(c2[::-1],(1,3))

    #X, Y = meshgrid(np.arange(a)/a*2*h_max_ecc*np.pi/180,np.arange(b-1,-1,-1)/b*2*v_max_ecc*np.pi/180)
    X, Y = meshgrid(np.arange(a)/a*2*h_max_ecc*np.pi/180,np.arange(b)/b*2*v_max_ecc*np.pi/180)

    print(f'sharpness={sharpness}')
    @logistic(sharpness)
    def grating(radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y):
        return sine_wave(radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y)
    
    dt = 1.0/frameRate
    #for it in range(1):
    for it in range(nstep):
        t = it * dt
        data = grating(radSF, radSF, direction, a, b, c1, c2, phase, t, X, Y)

        pixelData = np.reshape(np.round(data*255), (b,a,3)).astype('uint8')
        #pixelData = np.reshape(np.round(data*255), (b,a,3))
        #cv.imshow('linear', pixelData)
        #cv.waitKey(0)
        #pixelData = adjust_gamma(pixelData, gamma = 2.2)
        #cv.imshow('gamma', pixelData)
        #cv.waitKey(0)
        output.write(pixelData)
    
    output.release()
    cv.destroyAllWindows()

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

def sine_wave(radTF, radSF, direction, a, b, c1, c2, phase, t, X, Y):
    rel_color = np.reshape(1+np.sin((np.cos(direction)*X + np.sin(direction)*Y)*radSF - radTF*t + phase), (a*b,1))/2
    color = np.matmul(np.ones((a*b,1)), c1) + np.matmul(rel_color, (c2-c1))
    return color/255

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
