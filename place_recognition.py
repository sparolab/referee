import cv2
import argparse
import numpy as np
from glob import glob

from util.params import *
from util.recognition.recognition_utils import *
from util.recognition.referee import *

parser = argparse.ArgumentParser(description= "Radar Place Recognition Packages")
parser.add_argument('--desc', type = str, default = 'referee', help = 'we propose various methods (referee)')
parser.add_argument('--datasets_name', type = str, default = 'Riverside_03', help = 'DCC_01 / KAIST_03')
parser.add_argument('--multi_datasets1_name', type = str, default = 'Sejong_01', help = 'DCC_01 / KAIST_03')
parser.add_argument('--multi_datasets2_name', type = str, default = 'Sejong_02', help = 'DCC_01 / KAIST_03')
parser.add_argument('--session', type = str, default = 'multi', help = 'single session or multi session')
parser.add_argument('--extract_loop', type = bool, default = False, help = 'single session or multi session')
args = parser.parse_args()

def main():
    ## Extract Ground Truth
    ## =====================================================================================================================================================
    datasets = '-'.join(args.datasets_name.split("/"))
    exclude_node = 50 # Sejong & Oxford = 1000
    gt_threshold = 20
    pose_path = osp.join(dataset_path, args.datasets_name, 'poses.csv')
    position = getPosition(pose_path)
    
    pose_path1 = osp.join(dataset_path, args.multi_datasets1_name, 'poses.csv')
    position1 = getPosition(pose_path1)
    
    pose_path2 = osp.join(dataset_path, args.multi_datasets2_name, 'poses.csv')
    position2 = getPosition(pose_path2)
    
    descriptor_name = args.desc

    if args.session == 'single':
        gt_recall = extract_single_session_gt_pair(position, gt_threshold, exclude_node)
    else:
        gt_recall = extract_multi_session_gt_pair(position1, position2, gt_threshold)
    ## =====================================================================================================================================================

    if args.session == 'single':
        descriptor_dir = saveDescPath(args.datasets_name, descriptor_name, referee_split_ratio)
        # print(descriptor_dir)
        radar_fsd = get1DDescritor(descriptor_dir)
        if args.extract_loop:
            extract_single_session_referee_loop_pair(radar_fsd, exclude_node, datasets, descriptor_name, referee_split_ratio)
        result_path = osp.join(results_path, 'single_session', datasets, descriptor_name)
        loop_pair0 = np.loadtxt(osp.join(result_path, descDirName(descriptor_name, referee_split_ratio) + '.txt'))
        single_session_evaluation(loop_pair0, gt_recall, position, exclude_node, gt_threshold, datasets, descriptor_name, referee_split_ratio)
    
    if args.session == 'multi':
        descriptor_dir1 = saveDescPath(args.multi_datasets1_name, descriptor_name, referee_split_ratio)
        descriptor_dir2 = saveDescPath(args.multi_datasets2_name, descriptor_name, referee_split_ratio)
        radar_fsd1 = get1DDescritor(descriptor_dir1)
        radar_fsd2 = get1DDescritor(descriptor_dir2)
        dataset1 = '-'.join(args.multi_datasets1_name.split("/"))
        dataset2 = '-'.join(args.multi_datasets2_name.split("/"))
        if args.extract_loop:
            extract_multi_session_referee_loop_pair(radar_fsd1, radar_fsd2, dataset1, dataset2, descriptor_name, referee_split_ratio)
        result_path = osp.join(results_path, 'multi_session', dataset1 + '_to_' + dataset2, descriptor_name)
        loop_pair0 = np.loadtxt(osp.join(result_path, descDirName(descriptor_name, referee_split_ratio) + '.txt'))
        multi_session_evaluation(loop_pair0, gt_recall, position1, position2, gt_threshold, dataset1, dataset2, descriptor_name, referee_split_ratio)

if __name__ == "__main__":
    main()