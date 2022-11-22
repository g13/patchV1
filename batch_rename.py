import os
import re

def batch_rename(fdr, pattern, replace, suffix = None, verbose = True):
    if suffix is not None:
        files = [f for f in os.listdir(fdr) if os.path.isfile(os.path.join(fdr, f)) and re.search(f'.{suffix}$', f)]
    else:
        files = [f for f in os.listdir(fdr) if os.path.isfile(os.path.join(fdr, f))]
        
    if len(files) == 0:
        return -1
    for f in files:
        matched = re.search(pattern, f)
        if matched:
            new_name = re.sub(pattern, replace, f)
            os.rename(os.path.join(fdr,f), os.path.join(fdr,new_name))
            if verbose:
                print(f'{f} -> {new_name}')
    return 0