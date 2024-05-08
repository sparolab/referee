import os
import numpy as np
import os.path as osp

from glob import glob
from util.descriptor.referee import *

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
    radar_root = osp.join('example')
    radar_files = sorted(glob(osp.join(radar_root, "*.png")))
        
    ## ReFeree
    ## We recommend the split ratio to 8 or 10.
    generation_time = referee(radar_files, "referee/", split_ratio=10)
    print("Descriptor Size: ", os.path.getsize(osp.join("referee/", '000000.npy')))
    
    print("Time: ", np.mean(generation_time[:, 0]))

if __name__ == "__main__":
    main()