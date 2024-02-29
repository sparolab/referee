import os
import cv2
import time
import argparse

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.radar_utils import *

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

parser = argparse.ArgumentParser(description= "ReFeree")
parser.add_argument('--desc', type = str, default = 'referee', help = '')
parser.add_argument('--datasets_name', type = str, default = 'DDC_01', help = '')
args = parser.parse_args()

radar_root = osp.join('Datasets', args.datasets_name, 'polar')
radar_files = sorted([f for f in os.listdir(radar_root) if 'png' in f])

def main():
    generation_time = np.zeros((len(radar_files), 1))
    
    save_path = 'Description/' + args.datasets_name + '/referee/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # description size = radar_width / split_ratio
    split_ratio = 80

    for idx in tqdm(range(0, len(radar_files))):
        start_time = time.time()
        tmp = cv2.imread(osp.join(radar_root, radar_files[idx]), cv2.IMREAD_GRAYSCALE)

        # Radar features (Sarah Cen's 2018 feautres extration methods)
        targets = cen2018features(tmp.T)
        polar = targets_to_polar_image(targets, tmp.T.shape)

        # Radar Ringkey (Similar to Giseop Kim's Scan Context 2018)
        ringkey = np.sum(polar, axis=0)

        split_number = len(ringkey) // split_ratio
        split_ringkey = np.array_split(ringkey, split_number)
        final_ringkey = np.vstack([chunk.sum(axis=0) for chunk in split_ringkey]).reshape((split_number, 1))
        
        rfsd = np.count_nonzero(polar == 0, axis=0)
        split_rfsd = np.array_split(rfsd, split_number)
        final_rfsd = np.vstack([chunk.sum(axis=0) for chunk in split_rfsd]).reshape((split_number, 1))

        final_desc = final_ringkey * (1/final_rfsd)
        end_time = time.time()
        generation_time[idx, 0] = end_time - start_time

        np.save(save_path + str(idx).zfill(6) + '.npy', final_desc)
        # plt.imshow(final_rfsd, cmap = 'jet')
        # plt.pause(0.001)

    print("Descriptor Size: ", os.path.getsize(save_path + '000000.npy'))
    print("Time: ", np.mean(generation_time[:, 0]))

if __name__ == "__main__":
    main()