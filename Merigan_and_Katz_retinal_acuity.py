import numpy as np
def getCPD(ecc):
    # read from Merigan_and_Katz_retinal_acuity.svg using guides in inkscape
    h1 = 847.14235
    #h10 = 431.45730
    h40 = 182.28431
    e0 = 184.94661
    e3 = 295.77891
    #
    d40_1 = abs(h40 - h1)

    #d10_1 = abs(h10 - h1)
    #r10 = d10_1/d40_1
    #_r10 = np.log(10)/np.log(40)
    #print(f'log scale check: {_r10:.3f} ~ {r10:.3f}: {(r10-_r10)/_r10*100:.3f}%')

    r0 = abs(e0-h1)/d40_1
    r3 = abs(e3-h1)/d40_1
    cpd_0 = np.exp(r0 * np.log(40))
    cpd_3 = np.exp(r3 * np.log(40))
    #print((0, cpd_0))
    #print((3, cpd_3))
    # between eccentricity 0 and 3 assume power law function
    # log(cpd) = -k*ecc + log_cpd0
    k = -(np.log(cpd_3) - np.log(cpd_0))/3
    log_cpd0 = np.log(cpd_0)

    return np.exp(-k*ecc + log_cpd0)

def getAcuityEcc(ecc):
    return 1/getCPD(ecc)/4

print(getAcuityEcc(0.0))
print(getAcuityEcc(0.095))
