import numpy as np
import cv2 as cv

#CIE RGB
RGB2XYZ = np.array([[0.49000, 0.31000, 0.20000],\
                    [0.17697, 0.81240, 0.01063],\
                    [0.00000, 0.01000, 0.99000]])/0.17697

#D65
XYZ2LMS = np.array([[ 0.4002, 0.7076, -0.0808],\
                        [-0.2263, 1.1653,  0.0457],\
                        [      0,      0,  0.9182]])
"""
XYZ2LMS = np.array([[ 0.38971, 0.68898, -0.07868],\
                    [-0.22981, 1.18340,  0.04641],\
                    [       0,       0,  1.00000]])
"""

sRGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],\
                     [0.2126, 0.7152, 0.0722],\
                     [0.0193, 0.1192, 0.9504]])

h_max_ecc = 90 #degree
v_max_ecc = 72 #degree

#sBGR2LMS = np.matmul(XYZ2LMS,sRGB2XYZ[:,::-1])
sBGR2LMS = np.matmul(XYZ2LMS,sRGB2XYZ[:,::-1])
sRGB2LMS = sBGR2LMS[:,::-1]
LMS2sRGB = np.linalg.inv(sRGB2LMS)

BGR2LMS = np.matmul(XYZ2LMS,RGB2XYZ[:,::-1])
RGB2LMS = BGR2LMS[:,::-1]
LMS2RGB = np.linalg.inv(sRGB2LMS)

def video_to_LMS_time_series(vid, start, end):
    cap = cv.VideoCapture(vid+'.avi')
    ret = True
    captured = False
    i = start
    while(i < end):
        ret, frame = cap.read()
        if ret:
            if i==start:
                LMS = np.empty((end-start,)+frame.shape)
            LMS[i-start] = img_to_LMS(frame)
            i += 1
            captured = True
        else:
            break
        
    cap.release()
    cv.destroyAllWindows()
    if captured:
        return LMS
    else:
        raise Exception('video file corrupted')

def img_to_LMS(image):
    LMS = np.matmul(np.reshape(image.astype('float'), (image.size//3, 3)), sRGB2LMS.T)/255
    return LMS.reshape(image.shape)

def sRGB_gamma(data):
    cast = np.copy(data).reshape(data.size)
    select = cast<0.0031308
    cast[select] = cast[select]*12.92
    select = True - select
    cast[select] = np.power(cast[select], 1/2.4) * 1.055 - 0.055
    return cast.reshape(data.shape)
    
def inverse_sRGB_gamma(data):
    cast = np.copy(data).reshape(data.size)
    select = cast<0.04045
    cast[select] = cast[select]/12.92
    select = True - select
    cast[select] = np.power((cast[select]+0.055)/1.055, 2.4)
    return cast.reshape(data.shape)
