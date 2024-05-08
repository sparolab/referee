import os
import argparse

import numpy as np
import os.path as osp
from glob import glob
import matplotlib.pyplot as plt

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

parser = argparse.ArgumentParser(description= "Radar Place Recognition Packages")
parser.add_argument('--desc', type = str, default = 'referee', help = 'we propose various methods (referee)')
parser.add_argument('--datasets_name', type = str, default = 'Sejong_02', help = 'we propose various methods (referee)')
args = parser.parse_args()

def main():
    radar_root = osp.join('example')
    radar_files = sorted(glob(osp.join(radar_root, "*.png")))
        
    ## ReFeree
    generation_time = referee(radar_files, "referee/", 10)
    print("Descriptor Size: ", os.path.getsize(osp.join("referee/", '000000.npy')))
    
    print("Time: ", np.mean(generation_time[:, 0]))

if __name__ == "__main__":
    main()