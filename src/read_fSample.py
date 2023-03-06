import numpy as np

n = 100
fdr = '/root/autodl-tmp/wd/data/'
for i in range(1, n+1):
    suffix = f'_minimal_test2_{i}'
    with open(fdr + 'sample_spikeCount' +  suffix + '.bin', 'rb') as f:
        if i == 1:
            sampleSize = np.fromfile(f, 'u4', 1)[0]
            sample_t0, t1 = np.fromfile(f, 'f4', 2)
            sampleID = np.fromfile(f, 'u4', sampleSize)
            sample_spikeCount = np.zeros((n, sampleSize), dtype = int)
        else:
            f.seek(4*(sampleSize + 3) ,1)
        sample_spikeCount[i-1,:] = np.fromfile(f, 'u4', sampleSize)

mean_nsp = np.mean(sample_spikeCount, axis = 0)
std_nsp = np.std(sample_spikeCount, axis = 0)
print(f'nsp std [{std_nsp.min()}, {std_nsp.mean()}, {std_nsp.max()}]')
print(f'nsp mean [{mean_nsp.min()}, {mean_nsp.mean()}, {mean_nsp.max()}]')
cov = std_nsp[mean_nsp > 0]/mean_nsp[mean_nsp > 0]
print(f'nsp cov [{cov.min()}, {cov.mean()}, {cov.max()}]')
idx0 = np.arange(sampleSize)[mean_nsp > 0]
idx = idx0[cov.argmax()]

print(mean_nsp[idx])
print(std_nsp[idx])
print(sample_spikeCount[:,idx])
