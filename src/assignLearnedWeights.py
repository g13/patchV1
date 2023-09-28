import numpy as np
import warnings
import sys
import os 
import matplotlib.pyplot as plt
import shutil

def assignLearnedWeight(suffix, setup_fdr, in_fdr = None, FF_fn = None, cortical_fn = None):
    if FF_fn is None and cortical_fn is None:
        raise Exception("need either FF_fn or coritcal_fn")
    setup_fdr = setup_fdr + '/'
    in_fdr = in_fdr + '/'
    if in_fdr is not None:
        if FF_fn is not None:
            FF_fn = in_fdr + FF_fn
        if cortical_fn is not None:
            cortical_fn = in_fdr + cortical_fn

    if FF_fn is not None:
        fLGN_V1_ID = setup_fdr + 'LGN_V1_idList-' + suffix + '.bin'
        s_copy = setup_fdr + 'initial_LGN_V1_sList-' + suffix + '.bin'
        fLGN_V1_s = setup_fdr + 'LGN_V1_sList-' + suffix + '.bin'
        with open(FF_fn, 'rb') as f:
            nt, sampleInterval = np.fromfile(f,'u4', 2)
            dt = np.fromfile(f, 'f4', 1)[0]
            nV1, max_LGNperV1 = np.fromfile(f, 'u4', 2)

        sLGN_size = max_LGNperV1*nV1*4

        with open(FF_fn, 'rb') as f:
            f.seek(-sLGN_size, 2)
            sLGN = np.fromfile(f, 'f4', max_LGNperV1*nV1).reshape(max_LGNperV1, nV1).T
            
        shutil.copyfile(fLGN_V1_s, s_copy)
        max_s = np.max(sLGN.flatten())
        sum_s = np.sum(sLGN.flatten())
        print(f'sum of LGN str = {sum_s}, max = {max_s}')
        with open(fLGN_V1_s, 'rb+') as f:
            [_nV1, _max_LGNperV1] = np.fromfile(f, 'u4', 2)
            assert(_nV1 == nV1)
            assert(_max_LGNperV1 == max_LGNperV1)
            for i in range(_nV1):
                n = np.fromfile(f,'u4',1)[0] 
                if n > 0:
                    sLGN[i,:n].astype('f4').tofile(f)

        with open(fLGN_V1_s, 'rb') as f:
            [_nV1, _max_LGNperV1] = np.fromfile(f, 'u4', 2)
            assert(_nV1 == nV1)
            assert(_max_LGNperV1 == max_LGNperV1)
            max_s = 0
            sum_s = 0
            for i in range(_nV1):
                n = np.fromfile(f,'u4',1)[0] 
                if n > 0:
                    _sLGN = np.fromfile(f, 'f4', n)
                    sum_s += sum(_sLGN)
                    _max = np.max(_sLGN)
                    if _max > max_s:
                        max_s = _max
            print(f'check sum of LGN str = {sum_s}, max = {max_s}')


    if cortical_fn is not None:
        raise Exception('not implemented yet')
     
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(" assignLearnedWeight(suffix, setup_fdr, in_fdr = None, FF_fn = None, cortical_fn = None)")
    else:
        suffix = sys.argv[1]
        setup_fdr = sys.argv[2]
        if len(sys.argv) > 3:
            in_fdr = sys.argv[3]
            if len(sys.argv) > 4:
                FF_fn = sys.argv[4]
                if len(sys.argv) > 5:
                    cortical_fn = sys.argv[5]
                else:
                    cortical_fn = None
            else:
                cortical_fn = None
                FF_fn = None
        else:
            cortical_fn = None
            FF_fn = None
            in_fdr = None
            
    assignLearnedWeight(suffix, setup_fdr, in_fdr, FF_fn, cortical_fn)
