import numpy as np
import os.path as osp

from glob import glob
from scipy.spatial import cKDTree

class TerminalColors:
    BLACK = '\033[1;30m'
    RED = '\033[1;31m'
    GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[1;34m'
    MAGENTA = '\033[1;35m'
    CYAN = '\033[1;36m'
    WHITE = '\033[1;37m'
    RESET = '\033[0m'

def main():
    referee_root = osp.join('referee')
    referee_files = sorted(glob(osp.join(referee_root, "*.npy")))
    
    kdtree = cKDTree([np.load(referee_files[1]), np.load(referee_files[2])])
    loop_value, candidate_idx = kdtree.query(np.load(referee_files[0]), k=1)
    
    print(loop_value, candidate_idx)

if __name__ == "__main__":
    main()