import sys
import numpy as np
import matplotlib.pyplot as plt

print(sys.argv)
print(len(sys.argv))

fdr = sys.argv[1]
theme = sys.argv[2]
if fdr[-1] != '/':
    fdr = fdr + '/'
nE = 768

fV1_pos = fdr + 'V1_pos-' + theme + '.bin'
fV1_vpos = fdr + 'V1_vpos-' + theme + '.bin'
fV1_allpos = fdr + 'V1_allpos-' + theme + '.bin'

with open(fV1_pos, 'rb') as f:
    nblock, blockSize, dataDim  = np.fromfile(f, 'u4', count = 3)
    nV1 =  nblock * blockSize
    pos = np.fromfile(f, dtype = float, count = 2*nV1).reshape(nblock,2,blockSize)

fig = plt.figure('', dpi = 600)
ax = fig.add_subplot(111)
for i in range(nblock):
    ax.plot(pos[i,0,:nE], pos[i,1,:nE], '*', ms = 1)
    ax.plot(pos[i,0,nE:], pos[i,1,nE:], 's', ms = 1)
fig.savefig(fdr + 'V1pos-' + theme + '_check.png')
