import numpy as np

def generate_r_referee(polar, split_ratio):
    rfsd = np.count_nonzero(polar == 0, axis=0)        
    split_number = len(rfsd) // split_ratio
    
    split_rfsd = np.array_split(rfsd, split_number)
    final_desc = np.vstack([chunk.sum(axis=0) for chunk in split_rfsd]).reshape((split_number, 1))
        
    return final_desc


def generate_a_referee(polar, split_ratio):
    polar_angle = polar.shape[0]
    split_number = polar_angle // split_ratio
    polar_cells = np.array_split(polar, split_number, axis=0)
    total_num = polar_cells[0].shape[0]*polar_cells[0].shape[1]
    zero_counts = list()
    
    for polar_cell in polar_cells:
        zero_count = 0
        for row in polar_cell:
            last_non_zero_index = np.where(row != 0)[0]  # 0이 아닌 마지막 요소의 인덱스
            if len(last_non_zero_index) != 0:
                zero_cnt = np.count_nonzero(row[:last_non_zero_index[-1]] == 0)
            else:
                zero_cnt = 0
            zero_count += zero_cnt
        zero_counts.append(zero_count)
    
    final_desc = np.array(zero_counts) / total_num
    
    return final_desc