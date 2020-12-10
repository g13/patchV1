import numpy as np
import cv2 as cv

#CIE RGB
RGB2XYZ = np.array([[0.49000, 0.31000, 0.20000],\
                    [0.17697, 0.81240, 0.01063],\
                    [0.00000, 0.01000, 0.99000]])/0.17697

#Hunt D65
#XYZ2LMS = np.array([[ 0.4002, 0.7076, -0.0808],\
#                    [-0.2263, 1.1653,  0.0457],\
#                    [      0,      0,  0.9182]])

#CAT02
XYZ2LMS = np.array([[ 0.7328, 0.4296, -0.1624],\
                    [-0.7036, 1.6975,  0.0061],\
                    [ 0.0030, 0.0136,  0.9834]])

XYZ2sRGB = np.array([[ 3.24096994, -1.53738318, -0.49861076],\
                     [-0.96924364,  1.87596750,  0.04155506],\
                     [ 0.05563008, -0.20397696,  1.05697151]])
# wiki sRGB
sRGB2XYZ = np.array([[0.41239080, 0.35758434, 0.18048079],\
                     [0.21263901, 0.71516868, 0.07219232],\
                     [0.01933082, 0.11919478, 0.95053215]])

#sBGR2LMS = np.matmul(XYZ2LMS,sRGB2XYZ[:,::-1])
#sRGB2LMS = sBGR2LMS[:,::-1]
sRGB2LMS = np.matmul(XYZ2LMS,sRGB2XYZ)
sBGR2LMS = sRGB2LMS[:,::-1]
LMS2sRGB = np.linalg.inv(sRGB2LMS)

RGB2LMS = np.matmul(XYZ2LMS,RGB2XYZ)
BGR2LMS = RGB2LMS[:,::-1]
LMS2RGB = np.linalg.inv(RGB2LMS)

LMS2XYZ = np.linalg.inv(XYZ2LMS)
print(f'lum in LMS = {LMS2XYZ[1,:]}')
print(np.matmul(LMS2XYZ, LMS2XYZ.T))
print(np.matmul(LMS2sRGB, LMS2sRGB.T))

def video_to_LMS_time_series(vid, start, end):
    cap = cv.VideoCapture(vid+'.avi')
    ret = True
    captured = False
    i = start
    
    (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
    if int(major_ver) < 3:
        frameRate = cap.get(cv.cv.CV_CAP_PROP_FPS)
        width = cap.get(cv.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.cv.CV_CAP_PROP_FRAME_HEIGHT)
    else:
        frameRate = cap.get(cv.CAP_PROP_FPS)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    frameRate = int(frameRate)
    height = int(height)
    width = int(width)
    print(f'fps: {frameRate}, size: {height}x{width}')
    FourCC = cv.VideoWriter_fourcc(*'HFYU')
    output = cv.VideoWriter(vid+'_LMS.avi', FourCC, frameRate, (width, height), True)

    while(i < end):
        ret, frame = cap.read()
        if ret:
            if i==start:
                # rotate axis of frame so that the shape is [frame, LMS, height, width]
                LMS = np.empty((end-start,)+(frame.shape[2],frame.shape[0],frame.shape[1]), dtype=float)

            frameToVid = img_to_LMS(frame)
            output.write((frameToVid*255).astype('uint8'))
            # video data is BGR, thus covert to RGB
            LMS[i-start] = frameToVid[:,:,::-1].reshape((height*width,3)).T.reshape(3,height,width)
            i += 1
        else:
            raise Exception('video file corrupted')
        
    output.release()
        
    cap.release()
    cv.destroyAllWindows()
    return LMS

def img_to_LMS(image):
    LMS = np.matmul(np.reshape(image.astype('float'), (image.size//3, 3)), sRGB2LMS.T)/255
    return LMS.reshape(image.shape)

def apply_sRGB_gamma(data):
    cast = np.copy(data).reshape(data.size)
    select = cast<0.0031308
    cast[select] = cast[select]*323/25
    select = np.logical_not(select)
    cast[select] = (211*np.power(cast[select], 5/12)-11) / 200
    return cast.reshape(data.shape)
    
def inverse_sRGB_gamma(data):
    cast = np.copy(data).reshape(data.size)
    select = cast<0.04045
    cast[select] = cast[select]*25/323
    select = np.logical_not(select)
    cast[select] = np.power((200*cast[select]+11)/211, 12/5)
    return cast.reshape(data.shape)
