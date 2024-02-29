import time

import numpy as np

from tqdm import tqdm
from scipy.spatial import KDTree, cKDTree

def extract_single_session_ours_loop_pair(descriptor, exclude_node, datasets, descriptor_name):
    loop_idx  = np.zeros((len(descriptor), 3))
    loop_detection_time  = np.zeros((len(descriptor), 2))
    for current_idx in tqdm(range(len(descriptor))):
        cosdist = []
        if current_idx <= exclude_node:
            pass
        else:
            start_time = time.time()
            # kdtree = cKDTree(descriptor[:current_idx - exclude_node] / np.linalg.norm(descriptor[:current_idx - exclude_node], axis=1)[:, np.newaxis])
            # loop_value, candidate_idx = kdtree.query(descriptor[current_idx] / np.linalg.norm(descriptor[current_idx]), k=1)
            kdtree = cKDTree(descriptor[:current_idx - exclude_node])
            loop_value, candidate_idx = kdtree.query(descriptor[current_idx], k=1)

            end_time = time.time()
            loop_detection_time[current_idx, 0] = current_idx
            loop_detection_time[current_idx, 1] = end_time - start_time
            loop_idx[current_idx, 0] = current_idx
            loop_idx[current_idx, 1] = candidate_idx
            loop_idx[current_idx, 2] = loop_value

    print('Time: ', np.average(loop_detection_time[:, 1]))
    np.savetxt('results/single_session_' + datasets + '_' + descriptor_name + '.txt', loop_idx)
    np.savetxt('evaluation/single_session_' + datasets + '_' + descriptor_name + '.txt', loop_detection_time)

def extract_multi_session_ours_loop_pair(descriptor1, descriptor2, datasets0, datasets1, descriptor_name):
    loop_idx  = np.zeros((len(descriptor1), 3))
    loop_detection_time  = np.zeros((len(descriptor1), 2))  
    for current_idx in tqdm(range(len(descriptor1))):
        start_time = time.time()
        
        # kdtree = cKDTree(descriptor2 / np.linalg.norm(descriptor2, axis=1)[:, np.newaxis])
        # loop_value, candidate_idx = kdtree.query(descriptor1[current_idx] / np.linalg.norm(descriptor1[current_idx]), k=1)
        kdtree = cKDTree(descriptor2)
        loop_value, candidate_idx = kdtree.query(descriptor1[current_idx], k=1)

        end_time = time.time()

        loop_detection_time[current_idx, 0] = current_idx
        loop_detection_time[current_idx, 1] = end_time - start_time
        loop_idx[current_idx, 0] = current_idx
        loop_idx[current_idx, 1] = candidate_idx
        loop_idx[current_idx, 2] = loop_value

    np.savetxt('results/multi_session_' + datasets0 + '_' + 'to_' + datasets1 + '_' + descriptor_name + '.txt', loop_idx)
    np.savetxt('evaluation/multi_session_' + datasets0 + '_' + 'to_' + datasets1 + '_' + descriptor_name + '.txt', loop_detection_time)
