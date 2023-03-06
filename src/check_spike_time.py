from readPatchOutput import *

data_fdr = '/root/autodl-tmp/wd/data/'

n_diff = 3
max_difft = 5

def test_LGN(suffix):
    LGN_spFn = data_fdr + "LGN_sp" + suffix + ".bin"
    LGN_spScatter = readLGN_sp(LGN_spFn, prec = 'f4')
    return LGN_spScatter

def check_sp(all_sps):
    nsp_file = len(all_sps)
    sp0 = all_sps[0]
    sps = all_sps[1:]
    print(f'{nsp_file} files, {len(sp0)} neurons')
    earliest_isp = -1
    earliest_tsp = [-1.0, -1.0]
    q = 0
    for i in range(len(sp0)):
        for j in range(nsp_file-1):
            k = 0
            while k < min([len(sp0[i]), len(sps[j][i])]):
                if sp0[i][k] != sps[j][i][k]:
                    #print(f'{k}th spike from t-sp0[{i}]: {sp0[i][k]:.3f}, t-sp{j}[{i}]:{sps[j][i][k]:.3f}')
                    q = q+1
                    earlier_tsp = min([sp0[i][k],sps[j][i][k]])
                    if earlier_tsp < min(earliest_tsp) or earliest_isp == -1:
                        earliest_tsp = [sp0[i][k], sps[j][i][k]]
                        earliest_isp = i
                        break
                k = k + 1

    print(f'{q} pairs differ')
    print(f'earliest: {earliest_isp}') 
    print(earliest_tsp)
    return earliest_isp, earliest_tsp

def test_V1(suffix):
    rawDataFn = data_fdr + "rawData" + suffix + ".bin"
    spDataFn = data_fdr + "V1_spikes" + suffix
    spScatter = readSpike(rawDataFn, spDataFn, 'f4', 4, -20.0)
    return spScatter

print('LGN:')
LGN_sp1 = test_LGN('_minimal_test_2')
LGN_sp2 = test_LGN('_minimal_test_3')
LGN_isp, LGN_tsp = check_sp((LGN_sp1,LGN_sp2))

print('V1:')
V1_sp1 = test_V1('_minimal_test_2')
V1_sp2 = test_V1('_minimal_test_3')
V1_isp, V1_tsp = check_sp((V1_sp1,V1_sp2))


def check_if_not_same(fid0, fids, label, n, i, t):
    data0 = np.fromfile(fid0, prec, n)
    not_same = 0
    for f in fids:
        data = np.fromfile(f, prec, n)
        if not (data0 == data).all():
            idx = np.arange(len(data0))
            diff_id = idx[data != data0]
            m = min([diff_id.size, n_diff])
            print(f'{label} differs at time step {i}, {t}ms of dtype: {type(data0[diff_id[0]])}, print {m}/{diff_id.size} data points from {diff_id[:m]}')
            print(f"data0: {', '.join([repr(d) for d in data0[diff_id[:m]]])}")
            print(f"data:  {', '.join([repr(d) for d in data[diff_id[:m]]])}")
            #print(f"data0: {', '.join([float.hex(float(d)) for d in data0[diff_id[:m]]])}")
            #print(f"data:  {', '.join([float.hex(float(d)) for d in data[diff_id[:m]]])}")
            print(f"diff: {', '.join([repr(d) for d in data[diff_id[:m]] - data0[diff_id[:m]]])}")
            not_same = sum(data != data0)
    return not_same

sizeofPrec = 4
prec = 'f4'
def check_raw_data(suffixes):
    rawDataFns = [data_fdr + "rawData" + suffix + ".bin" for suffix in suffixes]
    rawData0 = rawDataFns[0] 
    with open(rawData0, 'rb') as fid0:
        dt = np.fromfile(fid0, prec, 1)[0] 
        nt, nV1 = np.fromfile(fid0, 'u4', 2)
        iModel = np.fromfile(fid0, 'i4', 1)[0] 
        mI, haveH, ngFF, ngE, ngI = np.fromfile(fid0, 'u4', 5) 
        fids = [open(rDf, 'rb') for rDf in rawDataFns[1:]]
        [f.seek(sizeofPrec + 4*8, 1) for f in fids]
        not_same = 0
        n_difft = 0
        for i in range(nt):
            [f.seek(nV1*sizeofPrec, 1) for f in fids]
            fid0.seek(nV1*sizeofPrec, 1)
            not_same += check_if_not_same(fid0, fids, 'depC', nV1, i, i*dt)
            if iModel == 1:
                not_same += check_if_not_same(fid0, fids, 'w', nV1, i, i*dt)
            not_same += check_if_not_same(fid0, fids, 'v', nV1, i, i*dt)
            for j in range(ngFF):
                not_same += check_if_not_same(fid0, fids, f'gFF{j}', nV1, i, i*dt)
            if haveH:
                for j in range(ngFF):
                    not_same += check_if_not_same(fid0, fids, f'hFF{j}', nV1, i, i*dt)
            for j in range(ngE):
                not_same += check_if_not_same(fid0, fids, f'gE{j}', nV1, i, i*dt)
            for j in range(ngI):
                not_same += check_if_not_same(fid0, fids, f'gI{j}', nV1, i, i*dt)
            if haveH :
                for j in range(ngE):
                    not_same += check_if_not_same(fid0, fids, f'hE{j}', nV1, i, i*dt)
                for j in range(ngI):
                    not_same += check_if_not_same(fid0, fids, f'hI{j}', nV1, i, i*dt)

            not_same += check_if_not_same(fid0, fids, 'cGap', mI, i, i*dt)
            if not_same > 0:
                print(f'{not_same} pairs of data dont match')
                n_difft += 1
            if n_difft > max_difft:
                break
        
suffixes = ['_minimal_test_2', '_minimal_test_3']
check_raw_data(suffixes)
