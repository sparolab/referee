import argparse
import numpy as np

from utils.loop_utils import *
from utils.pose_utils import *
from utils.desc_utils import *
from utils.eval_utils import *

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
parser.add_argument('--desc', type = str, default = 'referee', help = '')
parser.add_argument('--datasets_name', type = str, default = 'DDC_01', help = '')
parser.add_argument('--multi_datasets1_name', type = str, default = 'Sejong_01', help = '')
parser.add_argument('--multi_datasets2_name', type = str, default = 'Sejong_02', help = '')
parser.add_argument('--session', type = str, default = 'multi', help = '')
args = parser.parse_args()

def main():
    ## Extract Ground Truth
    ## =====================================================================================================================================================
    datasets = 'Boreas_'
    exclude_node = 50   # Sejong & Oxford = 1000
    gt_threshold = 20
    
    pr_path = 'evaluation/pr_curve/'
    roc_path = 'evaluation/roc_curve/'
    pair_path = 'evaluation/matching_pair/'
    fig_path = 'figure/'
    results_path = 'results/'
    
    if not os.path.exists(pr_path):
        os.makedirs(pr_path)
    if not os.path.exists(roc_path):
        os.makedirs(roc_path)
    if not os.path.exists(pair_path):
        os.makedirs(pair_path)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if args.session == 'single':
        pose_path = 'Datasets/' + args.datasets_name + '/poses.csv'
        pose = getPose(pose_path)
        position = np.column_stack((pose[:, 1], pose[:, 2], pose[:, 3]))
        
        descriptor_dir = 'Description/' + args.datasets_name + '/referee/'
        descriptor_name = 'FSD'
        radar_fsd = get1DDescritor(descriptor_dir)
        
        extract_single_session_ours_loop_pair(radar_fsd, exclude_node, datasets + args.datasets_name, descriptor_name)
        
        loop_pair0 = np.loadtxt('results/single_session_' + datasets + args.datasets_name + '_' + descriptor_name + '.txt')
        single_session_evaluation(loop_pair0, position, exclude_node, gt_threshold, datasets + args.datasets_name, descriptor_name)
    
    if args.session == 'multi':
        # Load First Datasets' pose
        pose_path1 = 'Datasets/' + args.multi_datasets1_name + '/poses.csv'
        pose1 = getPose(pose_path1)
        position1 = np.column_stack((pose1[:, 1], pose1[:, 2], pose1[:, 3]))
        
        # Load Second Datasets' pose
        pose_path2 = 'Datasets/' + args.multi_datasets2_name + '/poses.csv'
        pose2 = getPose(pose_path2)
        position2 = np.column_stack((pose2[:, 1], pose2[:, 2], pose2[:, 3]))

        # Load First Datasets' Description
        descriptor_dir1 = 'Description/' + args.multi_datasets1_name + '/referee/'
        
        # Load First Datasets' Description
        descriptor_dir2 = 'Description/' + args.multi_datasets2_name + '/referee/'
        descriptor_name = 'FSD'
        
        radar_fsd1 = get1DDescritor(descriptor_dir1)
        radar_fsd2 = get1DDescritor(descriptor_dir2)
        
        extract_multi_session_ours_loop_pair(radar_fsd1, radar_fsd2, datasets + args.multi_datasets1_name, datasets + args.multi_datasets2_name, descriptor_name)
        
        loop_pair0 = np.loadtxt('results/multi_session_' + datasets + args.multi_datasets1_name + '_' + 'to_' + datasets + args.multi_datasets2_name + '_' + descriptor_name + '.txt')
        multi_session_evaluation(loop_pair0, position1, position2, gt_threshold, args.multi_datasets1_name, args.multi_datasets2_name, descriptor_name)

if __name__ == "__main__":
    main()