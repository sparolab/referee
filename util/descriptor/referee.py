from tqdm import tqdm
from .feature_extractor import *
import os.path as osp
import time

def referee(radar_files, save_path, split_ratio):
    generation_time = np.zeros((len(radar_files), 1))
    for idx in tqdm(range(len(radar_files))):
        start_time = time.time()
        polar = feature_extractor(radar_files, idx)
        
        polar_angle = polar.shape[0]
        split_number = polar_angle // split_ratio
        polar_cells = np.array_split(polar, split_number, axis=0)
        total_num = polar_cells[0].shape[0]*polar_cells[0].shape[1]
        zero_counts = list()
        for polar_cell in polar_cells:
            zero_count = 0
            for row in polar_cell:
                last_non_zero_index = np.where(row != 0)[0]
                if len(last_non_zero_index) != 0:
                    zero_cnt = np.count_nonzero(row[:last_non_zero_index[-1]] == 0)
                else:
                    zero_cnt = 0
                zero_count += zero_cnt
            zero_counts.append(zero_count)
        
        final_desc = np.array(zero_counts) / total_num
        if np.sum(final_desc.shape) != 50:
            print(idx, radar_files[idx], polar.shape)
        
        end_time = time.time()
        generation_time[idx, 0] = end_time - start_time
        
        np.linalg.norm(final_desc) 
        np.save(osp.join(save_path, str(idx).zfill(6) + '.npy'), final_desc)
    return generation_time