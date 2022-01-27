import numpy as np
import cv2 as cv
import functools
from ext_signal import *
#TODO: heterogeneous buffers, to save texture memory
def generate_random(amp, radius, npixel, c, fname, time, frameRate = 120, ecc = 2.5, buffer_ecc = 0.25, neye = 2, gtype = 'randomPhase', seed = 567421, shift = None, inputLMS = True):
    """
    buffer_ecc: buffering area, to avoid border problems in texture memory accesses
    neye == 2:
        frame from 2 visual fields: (ecc+2*buffer_ecc) x 2(ecc+buffer_ecc+buffer_ecc(unused)) (width x height)
        each has a temporal axis: [-buffer_ecc, ecc], and vertical axis [-ecc-buffer_ecc, ecc+buffer_ecc] in degree
    neye == 1:
        frame from a single visual fields: origin at the center 2(ecc+buffer_ecc) x 2(ecc+buffer_ecc) (width x height)
    """
    if neye == 1:
        deg2pixel = npixel / (2*(ecc+buffer_ecc))
        a = npixel
    else:
        deg2pixel = npixel / (2*(ecc+2*buffer_ecc))
        a = npixel//2
        if neye != 2:
            print('neye need to be 1 or 2')
            return
        if np.mod(npixel,2) != 0:
            print('failed: npixel need to be even for neye == 2')
            return
    print(f'{1/deg2pixel} degree per pixel')
    radius_in_pixel = max(int(round(radius*deg2pixel)),1)
    if radius_in_pixel == 1:
        print('radius smaller than one pixel, mode changed to pixel')
        gtype = 'pixel'
    b = a*2  
    npixel = b
    FourCC = cv.VideoWriter_fourcc(*'FFV1')
    output = cv.VideoWriter(fname+'.avi', FourCC, frameRate, (npixel,npixel), True)
    print(f'frame size: {a}x{b}, radius in pixel {radius_in_pixel}')

    if isinstance(time, (list, tuple, np.ndarray)):
        nseq = len(time)
    else:
        nseq = 1
        time = np.array([time])

    if isinstance(amp, (list, tuple, np.ndarray)):
        assert(len(amp) == nseq)
    else:
        amp = np.zeros(nseq) + amp

    print(f'ecc = {ecc}, buffer_ecc = {buffer_ecc}')
    f = open(fname + '.bin', 'wb')
    np.array([-1]).astype('i4').tofile(f) 
    nFrame = int(np.round(frameRate*np.sum(time)))
    np.array([nFrame, npixel, npixel], dtype='i4').tofile(f)
    mean_value = np.mean(c, axis = 0)
    mean_value.astype('f4').tofile(f) # init_luminance
    np.array([buffer_ecc, ecc], dtype='f4').tofile(f)
    np.array([neye]).astype('u4').tofile(f)

    np.random.seed(seed)
    for i in range(nseq):
        t = time[i]
        nstep = np.round(frameRate * t)
        if not nstep == frameRate*t:
            raise Exception(f'time duration of sequence {i} is not in multiples of frame duration')
        nstep = int(nstep)
        print(f'{nstep} frames in total')

        for it in range(nstep):
            if neye == 1:
                if gtype == 'pixel':
                    data = randomStamp(amp[i], a, b, c)
                if gtype == 'randomPhase':
                    data = randomPhaseStamp(amp[i], a, b, radius_in_pixel, c)
                if gtype == 'fixedPhase':
                    data = randomPhaseStamp(amp[i], a, b, radius_in_pixel, c, True)
            else:
                if gtype == 'pixel':
                    dataL = randomized(amp[i], a, b, c0, gtype)
                if gtype == 'randomPhase':
                    dataL = randomPhaseStamp(amp[i], a, b, radius_in_pixel, c)
                if gtype == 'fixedPhase':
                    dataL = randomPhaseStamp(amp[i], a, b, radius_in_pixel, c, True)
                if shift is not None:
                    if shift == 0:
                        dataR = dataL
                    else:
                        raise Exception('only matching stimulus is implemented')
                else:
                    if gtype == 'pixel':
                        dataR = randomized(amp[i], a, b, c0, gtype)
                    if gtype == 'randomPhase':
                        dataR = randomPhaseStamp(amp[i], a, b, radius_in_pixel, c)
                    if gtype == 'fixedPhase':
                        dataR = randomPhaseStamp(amp[i], a, b, radius_in_pixel, c, True)

                data = np.concatenate((dataL,dataR), axis = 1)

            if inputLMS:
                # lms->rgb->bgr
                _LMS = data.reshape(npixel*npixel,3).T
                assert((_LMS>=0).all())
                assert((_LMS<=1).all())
                _sRGB = apply_sRGB_gamma(np.matmul(LMS2sRGB, _LMS))
                if (_sRGB<0).any() or (_sRGB>1).any():
                    print('sRGB space is not enough to represent the color')
                    print(f'{c1, c2}')
                    print(f'{np.min(_sRGB, axis = 1), np.max(_sRGB, axis = 1)}')
                    pick = _sRGB > 1
                    _sRGB[pick] = 1
                    pick = _sRGB < 0
                    _sRGB[pick] = 0
                pixelData = np.round(_sRGB*255).T.reshape(npixel,npixel,3)[:,:,::-1].astype('uint8')
                LMS_seq = _LMS.reshape((3,npixel,npixel))
            else: # input is sRGB
                pixelData = np.round(data*255).reshape(npixel,npixel,3).astype('uint8')
                # bgr->rgb->lms
                LMS_seq = np.matmul(sRGB2LMS, inverse_sRGB_gamma(data[:,:,::-1].reshape((npixel*npixel,3)).T)).reshape((3,npixel,npixel))

            #pixelData = np.reshape(np.round(data*255), (npixel,npixel,3)).astype('uint8')

            output.write(pixelData)
            #pixelData = np.reshape(np.round(data*255), (b,a,3))
            #cv.imshow('linear', pixelData)
            #cv.waitKey(0)
            #pixelData = adjust_gamma(pixelData, gamma = 2.2)
            #cv.imshow('gamma', pixelData)
            #cv.waitKey(0)
            LMS_seq.astype('f4').tofile(f)

    f.close()
    output.release()
    cv.destroyAllWindows()
    return

def generate_grating(amp, spatialFrequency, temporalFrequency, direction, npixel, c1, c2, fname, time, phase, sharpness, frameRate = 120, ecc = 2.5, buffer_ecc = 0.25, gtype = 'drifting', neye = 2, bar = False, center = np.pi/2, wing = np.pi/2, mask = None, maskData = None, inputLMS = False, genMovie = True):
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
        mask replace all masked pixels with the masked value
    """
    if np.mod(npixel,2) != 0:
        raise Exception("need even pixel")
    if neye == 1:
        a = npixel
    else:
        a = npixel//2  
        if neye != 2:
            print('neye need to be 1 or 2')
            return
        if np.mod(npixel,2) != 0:
            print('failed: npixel need to be even for neye == 2')
            return

    print(f'{npixel} degree per pixel')

    b = npixel  
    if genMovie:
        FourCC = cv.VideoWriter_fourcc(*'HFYU')
        #FourCC = cv.VideoWriter_fourcc(*'MP4V')
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

    if mask is not None:
        mask = np.reshape(np.repeat(mask, 3), (nseq, b,a,3))
        if maskData is None:
            raise Exception('mask data is not provided')
        else:
            if isinstance(maskData, (list, tuple, np.ndarray)):
                if isinstance(maskData, (list, tuple)):
                    maskData = np.array(list(maskData))

                if not np.array([i == j for i, j in zip(maskData.shape, (b,a,3))]).all() and len(maskData) != 3:
                    raise Exception('maskData shape does not match with stimulus size')
                else:
                    if np.array([i == j for i, j in zip(maskData.shape, (b,a,3))]).all():
                        maskData = np.tile(maskData, (nseq, 1))
                    elif maskData.size == 3:
                        maskData = np.tile(maskData, (nseq, b, a, 1)) 
                    else:
                        raise Exception('maskData takes array of shape (b,a,3) or (3,)')
            else:
                raise Exception('maskData takes array of shape (b,a,3) or (3,)')

    ########### VIDEO encodes as BGR: 
    if not inputLMS: # rgb->bgr
        c1_LMS = np.matmul(sRGB2LMS, inverse_sRGB_gamma(c1.reshape(3,1)))
        c2_LMS = np.matmul(sRGB2LMS, inverse_sRGB_gamma(c2.reshape(3,1)))
        mean_value = (c1_LMS+c2_LMS)/2
        c1 = c1[::-1]
        c2 = c2[::-1]
    else:
        c1_sRGB = apply_sRGB_gamma(np.matmul(LMS2sRGB, c1))
        c2_sRGB = apply_sRGB_gamma(np.matmul(LMS2sRGB, c2))
        print(f'crest in sRGB: {c1_sRGB}')
        print(f'valley in sRGB: {c2_sRGB}')
        if not (c1_sRGB<=1).all() or not (c1_sRGB>=0).all() or not (c2_sRGB<=1).all() or not (c2_sRGB>=0).all():
            raise Exception(f'crest and valley in LMS is out of the sRGB space')
        mean_value = (c1+c2)/2

    c1 = np.reshape(c1,(1,3))
    c2 = np.reshape(c2,(1,3))
    control = np.zeros(3)
    for i in range(3):
        if c1[0,i] >= c2[0,i]:
           control[i] = 1
        else: 
           control[i] = -1

    if neye == 1:
        X, Y = np.meshgrid(np.linspace(-1,1,a)*(ecc+buffer_ecc)*np.pi/180, np.linspace(-1,1,b)*(ecc+buffer_ecc)*np.pi/180)
        deg2pixel = npixel / (2*(ecc+buffer_ecc))
    else:
        X, Y = np.meshgrid((np.linspace(0,1,a)*(ecc+2*buffer_ecc)-buffer_ecc)*np.pi/180,np.linspace(-1,1,b)*(ecc+2*buffer_ecc)*np.pi/180)
        deg2pixel = npixel / (2*(ecc+2*buffer_ecc))

    print(f'{1/deg2pixel} degree per pixel')

    print(f'ecc = {ecc}, buffer_ecc = {buffer_ecc}')
    f = open(fname + '.bin', 'wb')
    np.array([-1]).astype('i4').tofile(f) 
    nFrame = np.sum(np.round(np.ceil(frameRate*time))).astype(int)
    print(nFrame)
    np.array([nFrame, npixel, npixel], dtype='i4').tofile(f)
    mean_value.astype('f4').tofile(f) # init_luminance
    np.array([buffer_ecc, ecc], dtype='f4').tofile(f)
    np.array([neye]).astype('u4').tofile(f)

    for i in range(nseq):
        t = time[i]
        nstep = int(np.round(frameRate*t))
        if not nstep == frameRate*t:
            nstep = int(np.ceil(frameRate*t))
            print(f'adjusted to {nstep} frames in total')
        else:
            print(f'exact {nstep} frames in total')

        if np.mod(nstep,2) != 0 and gtype == 'rotating':
            raise Exception(f'need even time step, current: {nstep}')

        radTF = temporalFrequency[i]*2*np.pi
        radSF = spatialFrequency[i]*180/np.pi*2*np.pi
        s = sharpness[i]
        print(f'sharpness={s}')
        #@logistic(s)
        def grating(amp, radTF, radSF, direction, a, b, c1, c2, control, s, phase, t, X, Y, bar, center = 0, wing = 0):
            return sine_wave(amp, radTF, radSF, direction, a, b, c1, c2, control, s, phase, t, X, Y, bar, center, wing)

        if gtype == 'rotating':
            half = nstep//2 
            dl = np.linspace(0,np.pi/4,half)
            dr = np.linspace(0,np.pi/4,half)
            dd = np.hstack((dl, np.flip(dr)))
            
        dt = 1.0/frameRate
        #for it in range(1):
        if gtype not in ('drifting','rotating'):
            raise Exception(f'gtype {gtype} not implemented')

        LMS_seq = np.empty((3, npixel, npixel), dtype=float)
        for it in range(nstep):
            t = it * dt
            if neye == 1:
                if gtype == 'rotating':
                    data = grating(amp[i], radTF, radSF, direction[i]-dd[it], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)
                if gtype == 'drifting':
                    data = grating(amp[i], radTF, radSF, direction[i], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)

                if mask is not None:
                    data[mask[i,:,:,:]] = maskData[i, mask[i,:,:,:]]
            else:
                if gtype == 'rotating':
                    dataL = grating(amp[i], radTF, radSF, direction[i]-dd[it], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)
                    dataR = grating(amp[i], radTF, radSF, direction[i]+dd[it], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)
                if gtype == 'drifting':
                    dataL = grating(amp[i], radTF, radSF, direction[i], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)
                    dataR = grating(amp[i], radTF, radSF, direction[i], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)

                if mask is not None:
                    assert(dataL.shape[0] == b)
                    assert(dataL.shape[1] == a)
                    assert(dataL.shape[2] == 3)
                    dataL[mask[i,:,:,:]] = maskData[i, mask[i,:,:,:]]
                    dataR[mask[i,:,:,:]] = maskData[i, mask[i,:,:,:]]
                data = np.concatenate((dataL,dataR), axis = 1)

            if inputLMS:
                # lms->rgb->bgr
                _LMS = data.reshape(npixel*npixel,3).T
                assert((_LMS>=0).all())
                assert((_LMS<=1).all())
                _sRGB = apply_sRGB_gamma(np.matmul(LMS2sRGB, _LMS))
                if (_sRGB<0).any() or (_sRGB>1).any():
                    print('sRGB space is not enough to represent the color')
                    #print(f'{c1, c2}')
                    print(f'{np.min(_sRGB, axis = 1), np.max(_sRGB, axis = 1)}')
                    pick = _sRGB > 1
                    _sRGB[pick] = 1
                    pick = _sRGB < 0
                    _sRGB[pick] = 0
                pixelData = np.round(_sRGB*255).T.reshape(npixel,npixel,3)[:,:,::-1].astype('uint8')
                LMS_seq = _LMS.reshape((3,npixel,npixel))
            else: # input is sRGB
                pixelData = np.round(data*255).reshape(npixel,npixel,3).astype('uint8')
                # bgr->rgb->lms
                LMS_seq = np.matmul(sRGB2LMS, inverse_sRGB_gamma(data[:,:,::-1].reshape((npixel*npixel,3)).T)).reshape((3,npixel,npixel))

            if genMovie:
                output.write(pixelData)
            #pixelData = np.reshape(np.round(data*255), (b,a,3))
            #cv.imshow('linear', pixelData)
            #cv.waitKey(0)
            #pixelData = adjust_gamma(pixelData, gamma = 2.2)
            #cv.imshow('gamma', pixelData)
            #cv.waitKey(0)
            LMS_seq.astype('f4').tofile(f)

    f.close()

    if genMovie:
        output.release()
        cv.destroyAllWindows()
    return

def generate_circular_mask(npixel, radius, seed, ecc, buffer_ecc, neye, center = None):
    """
    a: width of the half image in pixels 
    b: height of the image in pixels 
    buffer_ecc: buffering area, to avoid border problems in texture memory accesses
    neye == 2:
        frame from 2 visual fields: (ecc+2*buffer_ecc) x 2(ecc+buffer_ecc+buffer_ecc(unused)) (width x height)
        each has a temporal axis: [-buffer_ecc, ecc], and vertical axis [-ecc-buffer_ecc, ecc+buffer_ecc] in degree
    neye == 1:
        frame from a single visual fields: origin at the center 2(ecc+buffer_ecc) x 2(ecc+buffer_ecc) (width x height)
    mask replace all masked pixels with the masked value
    """
    if np.mod(npixel,2) != 0:
        raise Exception("need even pixel")
    if neye == 1:
        a = npixel
    else:
        a = npixel//2  
    b = npixel  

    if isinstance(radius, (list, tuple, np.ndarray)):
        rm = np.max(radius)
    else:
        rm = radius
    if center is None:
        np.random.seed(seed)
        if neye == 1:
            center = (np.random.rand(2)-0.5)*2*(ecc-rm)
        else:
            center = np.random.rand(2)*(ecc-2*rm)
        print(center)

    if isinstance(center, (list, tuple)):
        ncenter = len(center)//2
        center = np.array(list(center))
    else:
        ncenter = center.size//2
    center = np.reshape(center, (ncenter,2))

    if isinstance(radius, (list, tuple, np.ndarray)):
        if isinstance(radius, (list, tuple)):
            radius = np.array(list(radius))
        nseq = radius.size

        if ncenter == 1:
            center = np.tile(center, (nseq, 1))
        elif ncenter != nseq:
            raise Exception('center should have a shape of (nseq, 2), where nseq == radius.size')
            
    else:
        if len(center.shape) == 1:
            radius = np.array([radius])
            center = np.array([center])
        else:
            nseq = center.shape[0]
            radius = np.tile(radius, nseq)

    if neye == 1:
        x, y = np.meshgrid(np.linspace(-1,1,a)*(ecc+buffer_ecc), np.linspace(-1,1,b)*(ecc+buffer_ecc))
    else:
        x, y = np.meshgrid((np.linspace(0,1,a)*(ecc+2*buffer_ecc)-buffer_ecc), np.linspace(-1,1,b)*(ecc+2*buffer_ecc))

    print(f'xrange {[np.min(x), np.max(x)]}')
    print(f'yrange {[np.min(y), np.max(y)]}')
    mask = np.ones((nseq, b, a), dtype = bool)
    for iseq in range(nseq):
        nmasked = 0
        print(f'masking at {center[iseq]} for a radius of {radius[iseq]}')
        for i in range(a):
            for j in range(b):
                if np.linalg.norm([x[j,i] - center[iseq, 0], y[j,i] - center[iseq, 1]]) < radius[iseq]:
                    mask[iseq, j,i] = False
                    nmasked = nmasked + 1
            
        print(f'{nmasked} ({nmasked/b/a*100:.3f}%) pixel(s) left unmasked')
    return mask

def generate_retinal_wave(amp, spatialFrequency, temporalFrequency, waveSF, waveTF, direction, phase, sharpness, npixel, fname, time, frameRate = 120, ecc = 2.5, gtype = 'drifting', neye = 2, bar = False, center = np.pi/2, wing = np.pi/2, mask = None, maskData = None, ecc0 = 0, genMovie = True, virtual_LGN = 0, nrepeat = 1, reverse = False):
    if ecc0 == 0:
        ecc0 = ecc
    if np.mod(npixel,2) != 0:
        raise Exception("need even pixel")
    if neye == 1:
        a = npixel
    else:
        a = npixel//2  
        if neye != 2:
            print('neye need to be 1 or 2')
            return
        if np.mod(npixel,2) != 0:
            print('failed: npixel need to be even for neye == 2')
            return

    b = npixel  
    
    if waveSF == 0:
        waveSF = 1/(ecc*2)

    waveSF = waveSF*180/np.pi*2*np.pi
    waveTF = waveTF*2*np.pi

    if virtual_LGN == 0:
        nInputType = 2
    if virtual_LGN == 1:
        nInputType = 4
    if virtual_LGN == 2:
        nInputType = 6

    if genMovie:
        FourCC = cv.VideoWriter_fourcc(*'HFYU')
        #FourCC = cv.VideoWriter_fourcc(*'MP4V')
        output = np.empty(nInputType, dtype = object)
        for i in range(nInputType):
            output[i] = cv.VideoWriter(fname+f'_{i}.avi', FourCC, frameRate, (npixel,npixel), True)
        mixed_output = cv.VideoWriter(fname+'.avi', FourCC, frameRate, (npixel,npixel), True)

    # make sure size of arrays
    if isinstance(time, (list, tuple, np.ndarray)):
        nseq = len(time)
    else:
        nseq = 1
        time = np.array([time])

    amp = check_size(amp, nInputType, nseq, 'first', 'amp')
    spatialFrequency = check_size(spatialFrequency, nInputType, nseq, 'first', 'spatialFrequency')
    temporalFrequency = check_size(temporalFrequency, nInputType, nseq, 'first', 'temporalFrequency')
    direction = check_size(direction, nInputType, nseq, 'second', 'direction')
    phase = check_size(phase, nInputType, nseq, 'first', 'phase')
    sharpness = check_size(sharpness, nInputType, nseq, 'first', 'sharpness')

    ########### VIDEO encodes as BGR: 
    if neye == 1:
        X, Y = np.meshgrid(np.linspace(-1,1,a)*ecc*np.pi/180, np.linspace(-1,1,b)*ecc*np.pi/180)
        deg2pixel = npixel / (2*ecc)
    else:
        X, Y = np.meshgrid(np.linspace(0,1,a)*ecc*np.pi/180,np.linspace(-1,1,b)*ecc*np.pi/180)
        deg2pixel = npixel / (2*ecc)

    if mask is not None:
        if maskData is None:
            raise Exception('mask data is not provided')

        if len(mask.shape) == 3: 
            if mask.shape[0] != nseq:
                raise Exception('nseq mask does not match with time')
            if mask.shape[1] != b or mask.shape[2] != a:
                raise Exception('mask frame size does not match')
        else:
            if len(mask.shape) != 2:
                raise Exception('mask array size does not match')
            else:
                if mask.shape[0] != b or mask.shape[1] != a:
                    raise Exception('mask frame size does not match')

        if isinstance(maskData, (list, tuple, np.ndarray)):
            if isinstance(maskData, (list, tuple)):
                maskData = np.array(list(maskData))

            if len(maskData.shape) == 3:
                if maskData.shape[0] != nInputType:
                    raise Exception(f'maskData takes array of size nInputType = {nInputType} (determined by virtual_LGN) or a single value')
                else:
                    if maskData.shape[1] != b or maskData.shape[2] != a:
                        raise Exception('maskData frame size does not match')
            else:
                if len(maskData.shape) != 2:
                    raise Exception('mask array size does not match')
                if maskData.shape[1] != b or maskData.shape[2] != a:
                    raise Exception('maskData frame size does not match')
                else:
                    maskData = np.tile(maskData, (nInputType, nseq, 1))
        else:
            maskData = np.tile(maskData, (nInputType, nseq, b, a)) 

    if ecc0 < ecc:
        maskOut = np.zeros((b,a), dtype=bool);
        pick = np.logical_not(np.logical_and(np.abs(X) < ecc0*np.pi/180, np.abs(Y) < ecc0*np.pi/180))
        maskOut[pick] = True 

        maskOutData = np.zeros((b,a))
        maskOutData[pick] = 0.5

    print(f'{1/deg2pixel} degree per pixel')

    print(f'ecc = {ecc}')
    f = open(fname + '.bin', 'wb')
    np.array([virtual_LGN]).astype('i4').tofile(f) 
    nFrame = np.sum(np.round(np.ceil(frameRate*time))).astype(int)
    print(nFrame)
    np.array([nFrame, npixel, npixel], dtype='i4').tofile(f)
    np.array([ecc], dtype='f4').tofile(f)
    np.array([neye]).astype('u4').tofile(f)

    if gtype == 'rotating':
        half = nstep//2 
        dl = np.linspace(0,np.pi/4,half)
        dr = np.linspace(0,np.pi/4,half)
        dd = np.hstack((dl, np.flip(dr)))
                
    dt = 1.0/frameRate
    #for it in range(1):
    if gtype not in ('drifting','rotating'):
        raise Exception(f'gtype {gtype} not implemented')

    if reverse:
        typeSeq = np.arange(nInputType-1,-1,-1)
    else:
        typeSeq = np.arange(nInputType)
    
    nFrame = 0
    for i in range(nseq):
        t = time[i]
        nstep = int(np.round(frameRate*t))
        if not nstep == frameRate*t:
            nstep = int(np.ceil(frameRate*t))
            print(f'adjusted to {nstep} frames in total')
        else:
            print(f'exact {nstep} frames in total')

        if np.mod(nstep,2) != 0 and gtype == 'rotating':
            raise Exception(f'need even time step, current: {nstep}')

        for it in range(nstep):
            t = it * dt
            bw_seq = np.empty((nInputType, npixel, npixel), dtype=float)
            for jseq in range(nInputType):
                j = typeSeq[jseq]
                radTF = temporalFrequency[j,i]*2*np.pi
                radSF = spatialFrequency[j,i]*180/np.pi*2*np.pi
            
                if neye == 1:
                    if gtype == 'rotating':
                        data = retinal_wave(amp[j,i], radTF, radSF, direction[j,i]-dd[it], a, b, sharpness[j,i], phase[j,i], t, X, Y, bar, center, wing, waveSF, waveTF, nrepeat)
                    if gtype == 'drifting':
                        data = retinal_wave(amp[j,i], radTF, radSF, direction[j,i], a, b, sharpness[j,i], phase[j,i], t, X, Y, bar, center, wing, waveSF, waveTF, nrepeat)

                    if mask is not None:
                        data[mask[i,:,:]] = maskData[j,i, mask[i,:,:]]
                else:
                    if gtype == 'rotating':
                        dataL = retinal_wave(amp[j,i], radTF, radSF, direction[j,i]-dd[it], a, b, sharpness[j,i], phase[j,i], t, X, Y, bar, center, wing, waveSF, waveTF, nrepeat)
                        dataR = retinal_wave(amp[j,i], radTF, radSF, direction[j,i]+dd[it], a, b, sharpness[j,i], phase[j,i], t, X, Y, bar, center, wing, waveSF, waveTF, nrepeat)
                    if gtype == 'drifting':
                        dataL = retinal_wave(amp[j,i], radTF, radSF, direction[j,i], a, b, sharpness[j,i], phase[j,i], t, X, Y, bar, center, wing, waveSF, waveTF, nrepeat)
                        dataR = retinal_wave(amp[j,i], radTF, radSF, direction[j,i], a, b, sharpness[j,i], phase[j,i], t, X, Y, bar, center, wing, waveSF, waveTF, nrepeat)

                    if mask is not None:
                        assert(dataL.shape[0] == b)
                        assert(dataL.shape[1] == a)
                        dataL[mask[i,:,:]] = maskData[j,i, mask[i,:,:]]
                        dataR[mask[i,:,:]] = maskData[j,i, mask[i,:,:]]
                    data = np.concatenate((dataL,dataR), axis = 1)

                if ecc0 < ecc:
                    data[maskOut[:,:]] = maskOutData[maskOut[:,:]]

                #if i == 0 and it == 0 and j == 0:
                #    print(data)

                bw = np.repeat(data, 3).reshape(npixel,npixel,3)
                pixelData = np.round(bw*255).astype('uint8')
                bw_seq[jseq,:,:] = data
                if genMovie:
                    output[jseq].write(pixelData)
                #pixelData = np.reshape(np.round(data*255), (b,a,3))
                #cv.imshow('linear', pixelData)
                #cv.waitKey(0)
                #pixelData = adjust_gamma(pixelData, gamma = 2.2)
                #cv.imshow('gamma', pixelData)
                #cv.waitKey(0)
            bw_seq.astype('f4').tofile(f)
            nFrame = nFrame + 1

            if genMovie:
                mixed_data = (bw_seq[0,:,:] - bw_seq[1,:,:] + 1)/2
                pixelData = np.round(np.repeat(mixed_data, 3).reshape(npixel,npixel,3)*255).astype('uint8')
                mixed_output.write(pixelData)
            

    f.close()
    print(f'{nFrame} in total')

    if genMovie:
        for j in range(nInputType):
            output[j].release()
        mixed_output.release()
        cv.destroyAllWindows()
    return

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
            if sharpness != 1:
                if sharpness > 0:
                    exp_half = np.exp(sharpness/2)+1
                    A = np.power(exp_half,2)/(np.exp(sharpness)-1)
                    C = exp_half/(1-np.exp(sharpness))
                    x = A/(1.0 + np.exp(-sharpness*(x-0.5))) + C
                else:
                    x[x > 0.5] = 1
                    x[x < 0.5] = 0
            assert((x<=1).all())
            assert((x>=0).all())
            return x
        return static_nolinearity
    return decorator_logistic

def check_size(target, first, second, preferred = None, label = 'some var'):
    #print(f'before: {label} = {target}')
    if isinstance(target, (list, tuple, np.ndarray)):
        if not (len(target) == first or len(target) == second):
            raise Exception(f'{label} does not match') 
        else:
            if first == second and preferred is None: 
                raise Exception(f'{label} shape is ambiguous') 
            else:
                if preferred is None:
                    if len(target) == first:
                        target = np.tile(np.array([target]), (second,1)).T
                    else:
                        target = np.tile(np.array([target]), (first,1))
                else:
                    if preferred == 'first':
                        if len(target) == first:
                            target = np.tile(np.array([target]), (second,1)).T
                        else:
                            raise Exception(f'{label} shape is not consistent with the preferred {preferred} dimension') 
                        
                    else:
                        if preferred == 'second':
                            if len(target) == second:
                                target = np.tile(np.array([target]), (first,1))
                            else:
                                raise Exception(f'{label} shape is not consistent with the preferred {preferred} dimension') 
                        else:
                            raise Exception(f'unknown {preferred} dimension')
    else:
        target = np.tile(np.array([target]), (first, second))

    #print(f'after: {label} = {target}')
    assert(target.shape[0] == first)
    assert(target.shape[1] == second)
    return target

def sine_wave(amp, radTF, radSF, direction, a, b, c1, c2, control, sharpness, phase, t, X, Y, bar, center, wing):
    phi = (np.cos(direction)*X + np.sin(direction)*Y)*radSF - radTF*t
    if bar:
        #floor_phi = np.floor(phi/(2*np.pi)).astype(int)
        #phi_tmp = phi-floor_phi*2*np.pi
        pick = np.abs(phi + (phase + center)) > wing
        phi[pick] = -phase
    rel_color = np.reshape(1+amp*np.sin(phi + phase), (a*b,1))/2
    if sharpness != 1:
        if sharpness > 0:
            exp_half = np.exp(sharpness/2)+1
            A = np.power(exp_half,2)/(np.exp(sharpness)-1)
            C = exp_half/(1-np.exp(sharpness))
            rel_color = A/(1.0 + np.exp(-sharpness*(rel_color-0.5))) + C
        else:
            rel_color[rel_color > 0.5] = 1
            rel_color[rel_color < 0.5] = 0
        
    assert((rel_color <= 1.0).all())
    assert((rel_color >= 0.0).all())
    color = np.matmul(np.ones((a*b,1)), c1) + np.matmul(rel_color, (c2-c1))
    for i in range(3):
        if control[i] > 0:
            assert((color[:,i] <= c1[0,i]).all())
            assert((color[:,i] >= c2[0,i]).all())
        else:
            assert((color[:,i] >= c1[0,i]).all())
            assert((color[:,i] <= c2[0,i]).all())
        
    return color.reshape((b,a,3))

def retinal_wave(amp, radTF, radSF, direction, a, b, sharpness, phase, t, X, Y, bar, center, wing, waveSF, waveTF, nrepeat):
    wave_ratio = waveSF/radSF # period/wave_period
    # phase is alreayd wave phase
    phi0 = (np.cos(direction)*X + np.sin(direction)*Y)*waveSF - waveTF*t - phase
    wing0 = wing*wave_ratio
    center0 = np.pi - wing0
    pick = np.zeros(X.shape, dtype = bool)
    phi = np.zeros(X.shape) + center + wing
    for i in range(nrepeat):
        tmp_pick = np.abs((phi0 + i*2*np.pi) - center0) < wing0
        phi[tmp_pick] = (phi0[tmp_pick] + i*2*np.pi - center0 + wing0) / wave_ratio - wing + center

    bw = np.reshape((np.sin(phi) - np.sin(wing+center))/(1-np.sin(wing+center))*amp, (a*b,1))
    assert((bw >= 0).all())
    assert((bw <= 1).all())
    if sharpness != 1:
        if sharpness > 0:
            exp_half = np.exp(sharpness/2)+1
            A = np.power(exp_half,2)/(np.exp(sharpness)-1)
            C = exp_half/(1-np.exp(sharpness))
            bw = A/(1.0 + np.exp(-sharpness*(bw-0.5))) + C
        else:
            bw[bw > 5e-16] = 1

    assert((bw <= 1.0+5e-16).all())
    assert((bw >= 0.0-5e-16).all())
    bw[bw > 1] = 1
    bw[bw < 0] = 0
    return bw.reshape((b,a))

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

def randomPhaseStamp(amp, a, b, r, c, fixed = False):
    color = np.zeros((b*a,3))
    nc = c.shape[0]

    xphase0 = np.mod(a, r)
    if not fixed:
        xphase = np.random.randint(r)
    else:
        xphase = xphase0
    na = a // r + (xphase + xphase0 + r-1)//r
    yphase0 = np.mod(b, r)
    if not fixed:
        yphase = np.random.randint(r)
    else:
        yphase = yphase0
    
    nb = b // r + (yphase + yphase0 + r-1)//r
    stamp = np.random.choice(np.arange(nc), size = (nb,na))
    mat = np.repeat(np.repeat(stamp, r, axis = 1), r, axis = 0)
    matStamp = mat[yphase:yphase+b, xphase:xphase+a].reshape(b*a)
    for i in range(nc):
        color[matStamp==i,:] = c[i,:]

    color[color > 1] = 1
    color[color < 0] = 0
    return color.reshape((b,a,3))

def randomStamp(amp, a, b, c):
    color = np.zeros((b*a,3))
    nc = c.shape[0]
    stamp = np.random.choice(np.arange(nc), size = b*a)
    for i in range(nc):
        color[matStamp==i,:] = c[i,:]

    color[color > 255] = 255
    color[color < 0] = 0
    return color.reshape((b,a,3))/255

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
