import sys
import os 

def clean_data(data_fdr, suffix):
    suffix = '-' + suffix
    data_fdr = data_fdr + '/'
    rawDataFn = data_fdr + "rawData" + suffix + ".bin"
    spDataFn = data_fdr + "V1_spikes" + suffix + ".npz"
    parameterFn = data_fdr + "patchV1_cfg" + suffix + ".bin"
    LGN_spFn = data_fdr + "LGN_sp" + suffix + ".bin"
    f_sLGN = data_fdr + 'sLGN' + suffix + '.bin'
    f_dsLGN = data_fdr + 'dsLGN' + suffix + '.bin'
    fLGN_fr = data_fdr + 'LGN_fr' + suffix + '.bin'
    fLGN = data_fdr + 'LGN' + suffix + '.bin'
    fLGN_gallery = data_fdr + 'LGN_gallery' + suffix + '.bin'
    fOutputB4V1 = data_fdr + 'outputB4V1' + suffix + '.bin'
    files = [rawDataFn, spDataFn, parameterFn, LGN_spFn, f_sLGN, f_dsLGN, fLGN, fLGN_fr, fLGN_gallery, fOutputB4V1]
    #files = [rawDataFn, spDataFn, parameterFn, LGN_spFn, f_sLGN, f_dsLGN, fLGN, fLGN_fr]
    for f in files:
        print(f, end = ' ')
        try:
            os.remove(f)
        except FileNotFoundError:
            print('not found')
            continue
        print('deleted')

if __name__ == "__main__":
    data_fdr = sys.argv[1]
    suffix = sys.argv[2]
    clean_data(data_fdr, suffix)
