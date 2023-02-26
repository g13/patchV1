import sys
import numpy as np

print(sys.argv)
print(len(sys.argv))

fdr = sys.argv[1]
theme = sys.argv[2]
if fdr[-1] != '/':
    fdr = fdr + '/'

fV1_pos = fdr + 'V1_pos-' + theme + '.bin'
fV1_vpos = fdr + 'V1_vpos-' + theme + '.bin'
fV1_allpos = fdr + 'V1_allpos-' + theme + '.bin'

with open(fV1_pos, 'rb') as f:
    nblock, blockSize, dataDim  = np.fromfile(f, 'u4', count = 3)
    nV1 =  nblock * blockSize
    pos = np.fromfile(f, dtype = float, count = 2*nV1).reshape(nblock,2,blockSize)
    x_min = pos[:,0,:].min()
    x_range = pos[:,0,:].max() - x_min
    y_min = pos[:,1,:].min()
    y_range = pos[:,1,:].max() - y_min
    print(x_min, x_range, y_min, y_range)
    x = pos[:,0,:].flatten()
    y = pos[:,1,:].flatten()
with open(fV1_vpos, 'rb') as f:
    nblock, blockSize = np.fromfile(f, 'u4', count = 2)
    assert(nV1 == nblock * blockSize)
    vpos = np.fromfile(f, dtype = float, count = 2*nV1).reshape(2,nV1)

with open(fV1_allpos, 'wb') as f:
        np.array([nblock, blockSize, dataDim], dtype = 'u4').tofile(f)
        np.array([x_min, x_range, y_min, y_range], dtype = float).tofile(f)
        x.tofile(f)
        y.tofile(f)
        vx = vpos[0,:] * np.cos(vpos[1,:])
        vy = vpos[0,:] * np.sin(vpos[1,:])
        vx_min = vx.min()
        vx_range = vx.max() - vx_min
        vy_min = vy.min()
        vy_range = vy.max() - vy_min
        np.array([vx_min, vx_range, vy_min, vy_range]).tofile(f)
        print(vx_min, vx_range, vy_min, vy_range)
        np.vstack((vx,vy)).tofile(f)
