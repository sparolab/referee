import numpy as np
from tqdm import tqdm
import time
from scipy.spatial import cKDTree
from ..params import *

def extract_single_session_referee_loop_pair(descriptor, exclude_node, datasets, descriptor_name, param):
    print(len(descriptor))
    loop_idx  = np.zeros((len(descriptor), 3))
    loop_detection_time  = np.zeros((len(descriptor), 2))
    for current_idx in tqdm(range(len(descriptor))):
        if current_idx <= exclude_node:
            pass
        else:
            start_time = time.time()
            kdtree = cKDTree(descriptor[:current_idx - exclude_node])
            loop_value, candidate_idx = kdtree.query(descriptor[current_idx], k=1)

            end_time = time.time()
            loop_detection_time[current_idx, 0] = current_idx
            loop_detection_time[current_idx, 1] = end_time - start_time
            loop_idx[current_idx, 0] = current_idx
            loop_idx[current_idx, 1] = candidate_idx
            loop_idx[current_idx, 2] = loop_value

    print('Time: ', np.average(loop_detection_time[:, 1]))
    result_path = osp.join(results_path, 'single_session', datasets, descriptor_name)
    eval_path = osp.join(evaluation_path, 'single_session', datasets, descriptor_name)
    createDir(result_path)
    createDir(eval_path)
    np.savetxt(osp.join(result_path, descDirName(descriptor_name, param) + '.txt'), loop_idx)
    np.savetxt(osp.join(eval_path, descDirName(descriptor_name, param) + '.txt'), loop_detection_time)

def extract_multi_session_referee_loop_pair(descriptor1, descriptor2, datasets0, datasets1, descriptor_name, param):
    loop_idx  = np.zeros((len(descriptor1), 3))
    loop_detection_time  = np.zeros((len(descriptor1), 2))  
    for current_idx in tqdm(range(len(descriptor1))):
        start_time = time.time()

        kdtree = cKDTree(descriptor2)
        loop_value, candidate_idx = kdtree.query(descriptor1[current_idx], k=1)

        end_time = time.time()

        loop_detection_time[current_idx, 0] = current_idx
        loop_detection_time[current_idx, 1] = end_time - start_time
        loop_idx[current_idx, 0] = current_idx
        loop_idx[current_idx, 1] = candidate_idx
        loop_idx[current_idx, 2] = loop_value
        
    result_path = osp.join(results_path, 'multi_session', datasets0 + '_to_' + datasets1, descriptor_name)
    eval_path = osp.join(evaluation_path, 'multi_session', datasets0 + '_to_' + datasets1, descriptor_name)
    createDir(result_path)
    createDir(eval_path)
    np.savetxt(osp.join(result_path, descDirName(descriptor_name, param) + '.txt'), loop_idx)
    np.savetxt(osp.join(eval_path, descDirName(descriptor_name, param) + '.txt'), loop_detection_time)