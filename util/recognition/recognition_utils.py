import os
import time
import natsort
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from tqdm import tqdm
import scipy
from scipy.spatial import KDTree, cKDTree
# from sklearn.neighbors import KDTree
from sklearn.metrics import auc, roc_curve, f1_score, roc_auc_score, precision_recall_curve, average_precision_score
from ..params import *

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

def find_best_index(data):
    min_fp = float('inf')  # fp가 가장 작은 값을 찾기 위해 무한대로 초기화
    max_tp = -1  # 해당하는 fp 중에서 tp가 가장 큰 값을 찾기 위해 -1로 초기화
    result_index = -1  # 결과 인덱스 초기화
    
    for index, (tp, fp) in enumerate(data):
        if fp < min_fp:
            # 더 작은 fp 값을 찾았다면, min_fp 업데이트와 함께 max_tp 및 result_index도 업데이트
            min_fp = fp
            max_tp = tp
            result_index = index
        elif fp == min_fp:
            # 같은 fp 값을 가진 항목 중 tp가 더 큰 경우에만 업데이트
            if tp > max_tp:
                max_tp = tp
                result_index = index
    
    return result_index

def getPose(pose_path):
    pose = np.loadtxt(pose_path, delimiter=',')
    return pose

def getPosition(pose_path):
    pose = getPose(pose_path)
    position = np.column_stack((pose[:, 1], pose[:, 2], pose[:, 3]))
    return position

def saveSingleResults(eval: str, dataset0: str, descriptor: str, param, value):
    path = osp.join(evaluation_path, "single_session", eval, dataset0, descriptor)
    createDir(path)
    np.savetxt(osp.join(path, descDirName(descriptor, param) + '.txt'), value)

def saveMultiResults(eval: str, dataset0: str, dataset1: str, descriptor: str, param, value):
    path = osp.join(evaluation_path, "multi_session", eval, dataset0 + '_to_' + dataset1, descriptor)
    createDir(path)
    np.savetxt(osp.join(path, descDirName(descriptor, param) + '.txt'), value)

#### Ground Truth
## =====================================================================================================================================================
def extract_single_session_gt_pair(position0, gt_threshold, exclude_node):
    gt_recall = 0
    ground_truth = np.zeros((len(position0), 2))
    for current_idx in tqdm(range(len(position0))):
        ground_truth[current_idx, 0] = current_idx
        if current_idx <= exclude_node:
            pass
        else:
            kdtree = cKDTree(position0[:current_idx - exclude_node])
            loop_value, closest_pose_idx = kdtree.query(position0[current_idx], k = 1)
            if loop_value <= gt_threshold:
                gt_recall += 1
                ground_truth[current_idx, 1] = closest_pose_idx
    return gt_recall

def extract_multi_session_gt_pair(position0, position1, gt_threshold):
    gt_recall = 0
    ground_truth = np.zeros((len(position0), 2))

    for current_idx in tqdm(range(len(position0))):
        ground_truth[current_idx, 0] = current_idx
        kdtree = cKDTree(position1)
        loop_value, closest_pose_idx = kdtree.query(position0[current_idx], k = 1)
        
        if loop_value <= gt_threshold:
            gt_recall += 1
            ground_truth[current_idx, 1] = closest_pose_idx

    return gt_recall
## =====================================================================================================================================================

## Evaluation Utils
## =====================================================================================================================================================
def single_session_evaluation(loop_pair0, ground_truth0, position0, exclude_node, gt_threshold, datasets0, description_name, param):
    arr = loop_pair0[:, 2]
    filtered_arr = arr[arr != 0]
    min_value = np.min(filtered_arr)
    max_value = np.max(filtered_arr)
    
    y_test = []
    y_score = []
    
    # loop_threshold  = np.linspace(min_value, max_value+(max_value-min_value)/100, 100)
    loop_threshold  = np.linspace(max_value, min_value, 100)
    
    # precision, recall, sensitivity, false_positive_rate = [1.], [0.], [], []
    # f1_scores = [2 * (recall[-1] * precision[-1]) / (recall[-1] + precision[-1])]
    precision, recall, sensitivity, false_positive_rate = [], [], [], []
    f1_scores = []
    query, candidate, gt_angle, estimate_angle = [], [], [], []
    tp_arr, fp_arr, fn_arr, tn_arr = [], [], [], []
    for loop_thresh_idx, loop_thresh in tqdm(enumerate(loop_threshold)):
        tp, fp, fn, tn = 0, 0, 0, 0
        success = 0
        for idx in range(len(loop_pair0)):
            if idx < exclude_node:
                pass
            else:
                similarity_distance = loop_pair0[idx, 2]
                translation_diff    = np.sqrt((position0[int(loop_pair0[idx, 0]), 0] - position0[int(loop_pair0[idx, 1]), 0])**2 +
                                              (position0[int(loop_pair0[idx, 0]), 1] - position0[int(loop_pair0[idx, 1]), 1])**2)
                if int(loop_pair0[idx, 0]) == 0:
                    pass
                if similarity_distance == 0:
                    continue
                elif similarity_distance < loop_thresh:
                    y_score.append(similarity_distance)
                    if translation_diff < gt_threshold:
                        tp += 1
                        y_test.append(0)
                    else:
                        fp += 1
                        y_test.append(1)

                else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                    if translation_diff < gt_threshold:
                        fn += 1
                    else:
                        tn += 1
        if tp + fp != 0:
            tp_arr.append(tp)
            fp_arr.append(fp)
            precision.append(tp / (tp + fp))
            print(f"{loop_thresh_idx} precision : {tp}, {fp}, {precision[-1]}")
            recall.append(tp / (tp + fn))
            sensitivity.append(tp / (tp + fn))
            false_positive_rate.append(fp / (fp + tn))
            if recall[-1] + precision[-1] != 0:
                f1 = 2 * (recall[-1] * precision[-1]) / (recall[-1] + precision[-1])
            else:
                f1 = 0
            f1_scores.append(f1)
    
    fpr, tpr, _ = roc_curve(y_test, y_score) # input 순서 : 실제 라벨, 예측 값
    
    fig1 = plt.figure()
    ax1 = fig1.subplots()
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, 1.1)
    ax1.plot(recall, precision)
    fig2 = plt.figure()
    ax2 = fig2.subplots()
    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    ax2.plot(false_positive_rate, sensitivity)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    z = np.linspace(0, 10, len(position0))
    # max_precision_index = np.argmax(recall)
    # max_precision_index = find_best_index(np.column_stack((tp_arr, fp_arr)))
    # max_precision_index = np.argmax(f1_scores[1:])
    max_precision_index = np.argmax(f1_scores)
    max_precision_thresh = loop_threshold[max_precision_index]
    max_precision_loop_pair = loop_pair0[(loop_pair0[:, 2] > 0)]
    max_precision_loop_pair = max_precision_loop_pair[(max_precision_loop_pair[:, 2] < max_precision_thresh)]
    
    ax3.plot(position0[:, 0], position0[:, 1], z, color='k')
    loop_or_not = []
    elev, azim = 30, -60
    tm, fm = 0, 0
    for idx in range(len(max_precision_loop_pair)):
        translation_diff = np.sqrt((position0[int(max_precision_loop_pair[idx, 0]), 0] - position0[int(max_precision_loop_pair[idx, 1]), 0])**2 +
                                   (position0[int(max_precision_loop_pair[idx, 0]), 1] - position0[int(max_precision_loop_pair[idx, 1]), 1])**2)
        if int(max_precision_loop_pair[idx, 0]) == 0:
            pass
        elif translation_diff < gt_threshold:
            tm += 1
            # print(TerminalColors.GREEN + "True Matching Pair : ", str(max_precision_loop_pair[idx, 0]), str(max_precision_loop_pair[idx, 1]) + TerminalColors.RESET, str(max_precision_loop_pair[idx, 2]))
            ax3.plot([position0[int(max_precision_loop_pair[idx, 0]), 0],    position0[int(max_precision_loop_pair[idx, 1]), 0]],
                    [position0[int(max_precision_loop_pair[idx, 0]), 1],    position0[int(max_precision_loop_pair[idx, 1]), 1]],
                    [z[int(max_precision_loop_pair[idx, 0])],               z[int(max_precision_loop_pair[idx, 1])]], c = 'limegreen',
                    linewidth=0.5, alpha=0.6)
            loop_or_not.append(1)
        else:
            fm += 1
            # print(TerminalColors.RED + "False Matching Pair : ", str(max_precision_loop_pair[idx, 0]), str(max_precision_loop_pair[idx, 1]) + TerminalColors.RESET, str(max_precision_loop_pair[idx, 2]))
            ax3.plot([position0[int(max_precision_loop_pair[idx, 0]), 0],    position0[int(max_precision_loop_pair[idx, 1]), 0]],
                    [position0[int(max_precision_loop_pair[idx, 0]), 1],    position0[int(max_precision_loop_pair[idx, 1]), 1]],
                    [z[int(max_precision_loop_pair[idx, 0])],               z[int(max_precision_loop_pair[idx, 1])]], c = 'red',
                    linewidth='0.5', alpha=0.6)
            loop_or_not.append(0)
        query.append(max_precision_loop_pair[idx, 0])
        candidate.append(max_precision_loop_pair[idx, 1])
    plt.axis('off')
    print(TerminalColors.GREEN + "Descriptor : ",    str(description_name)                         + TerminalColors.RESET)
    print(TerminalColors.GREEN + "Datasets : ",      str(datasets0)                                + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "Recall@1 : ",      str(tp/ground_truth0)                         + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "PR AUC_SCORE : ",  str(auc(recall, precision))                   + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "AP : ",            str(average_precision_score(y_test, y_score)) + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "ROC AUC_SCORE : ", str(roc_auc_score(y_test, y_score))           + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "F1_max_SCORE : ",  str(np.max(f1_scores))                        + TerminalColors.RESET)
    
    ax3.view_init(elev=elev, azim=azim)
    print("True Matching : ", tm)
    print("False Matching : ", fm)
    
    precision = np.array(precision)
    recall = np.array(recall)
    precision_recall = np.column_stack((recall, precision))
    auc_score = np.array([auc(false_positive_rate, sensitivity)])
    sensitivity = np.array(sensitivity)
    false_positive_rate = np.array(false_positive_rate)
    roc = np.column_stack((false_positive_rate, sensitivity))
    recall_at1 = np.array([tp/ground_truth0])
    pr_aucscore = np.array([auc(recall, precision)])
    roc_aucscore = np.array([roc_auc_score(y_test, y_score)])
    f1scores = np.array([f1_scores]).transpose()
    f1_recall = np.column_stack((recall, f1scores))
    ap = np.array([average_precision_score(y_test, y_score)])
    matching_pair = np.column_stack((np.array(query), np.array(candidate), np.array(loop_or_not)))

    saveSingleResults("recall@1", datasets0, description_name, param, recall_at1)
    saveSingleResults("pr_curve", datasets0, description_name, param, precision_recall)
    saveSingleResults("roc_curve", datasets0, description_name, param, roc)
    saveSingleResults("matching_pair", datasets0, description_name, param, matching_pair)
    saveSingleResults("pr_auc_score", datasets0, description_name, param, pr_aucscore)
    saveSingleResults("roc_aucscore", datasets0, description_name, param, roc_aucscore)
    saveSingleResults("f1_score", datasets0, description_name, param, f1_scores)
    saveSingleResults("f1_recall", datasets0, description_name, param, f1_recall)
    saveSingleResults("average_precision_score", datasets0, description_name, param, ap)
    
    fig_path = osp.join(figure_path, "single_session", datasets0)
    fprcurve_path = osp.join(fig_path, "pr_curve")
    froc_curve_path = osp.join(fig_path, "roc_curve")
    fpr_path = osp.join(fig_path, "place_recogntiion")
    createDir(fprcurve_path)
    createDir(froc_curve_path)
    createDir(fpr_path)
    fig1.savefig(osp.join(fprcurve_path, descDirName(description_name, param) + '.svg'))
    fig2.savefig(osp.join(froc_curve_path, descDirName(description_name, param) + '.svg'))
    fig3.savefig(osp.join(fpr_path, descDirName(description_name, param) + '.svg'))
    
def multi_session_evaluation(loop_pair0, ground_truth0, position0, position1, gt_threshold, datasets0, datasets1, descriptor, param):
    arr = loop_pair0[:, 2]
    filtered_arr = arr[arr != 0]
    min_value = np.min(filtered_arr)
    max_value = np.max(filtered_arr)
    
    y_test = []
    y_score = []
    
    loop_threshold  = np.linspace(min_value, max_value+(max_value-min_value)/100, 100)
    
    precision, recall, sensitivity, false_positive_rate = [1.], [0.], [], []
    f1_scores = [2 * (recall[-1] * precision[-1]) / (recall[-1] + precision[-1])]
    query, candidate, gt_angle, estimate_angle = [], [], [], []
    tp_arr, fp_arr, fn_arr, tn_arr = [], [], [], []
    for loop_thresh_idx, loop_thresh in tqdm(enumerate(loop_threshold)):
        tp, fp, fn, tn = 0, 0, 0, 0
        for idx in range(len(loop_pair0)):
            similarity_distance = loop_pair0[idx, 2]
            translation_diff    = np.sqrt((position0[int(loop_pair0[idx, 0]), 0] - position1[int(loop_pair0[idx, 1]), 0])**2 + 
                                          (position0[int(loop_pair0[idx, 0]), 1] - position1[int(loop_pair0[idx, 1]), 1])**2)

            if similarity_distance < loop_thresh:
                y_score.append(similarity_distance)
                if translation_diff < gt_threshold:
                    tp += 1
                    y_test.append(0)
                else:
                    fp += 1
                    y_test.append(1)
            else:
                if translation_diff < gt_threshold:
                    fn += 1
                else:
                    tn += 1
        if tp + fp != 0:
            tp_arr.append(tp)
            fp_arr.append(fp)
            precision.append(tp / (tp + fp))
            print(f"{loop_thresh_idx} precision : {tp}, {fp}, {precision[-1]}")
            recall.append(tp / (tp + fn))
            sensitivity.append(tp / (tp + fn))
            false_positive_rate.append(fp / (fp + tn))
            if recall[-1] + precision[-1] != 0:
                f1 = 2 * (recall[-1] * precision[-1]) / (recall[-1] + precision[-1])
            else:
                f1 = 0
            f1_scores.append(f1)
    
    # fpr, tpr, thresholds = roc_curve(y_test, y_score) # input 순서 : 실제 라벨, 예측 값
    
    print(datasets0, datasets1)
    print(TerminalColors.GREEN + "Descriptor : ",    str(descriptor)                               + TerminalColors.RESET)
    print(TerminalColors.GREEN + "Datasets : ",      str(datasets0), str(datasets1)                + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "Recall@1 : ",      str(tp/ground_truth0)                         + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "PR AUC_SCORE : ",  str(auc(recall, precision))                   + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "AP : ",            str(average_precision_score(y_test, y_score)) + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "ROC AUC_SCORE : ", str(roc_auc_score(y_test, y_score))           + TerminalColors.RESET)
    print(TerminalColors.BLUE  + "F1_max_SCORE : ",  str(np.max(f1_scores))                        + TerminalColors.RESET)

    fig1 = plt.figure()
    ax1 = fig1.subplots()
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, 1.1)
    ax1.plot(recall, precision)

    fig2 = plt.figure()
    ax2 = fig2.subplots()
    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    ax2.plot(false_positive_rate, sensitivity)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')

    z0 = np.linspace(0, 10, len(position0))
    z1 = np.linspace(20, 30, len(position1))
    # max_precision_index = np.argmax(precision)
    max_precision_index = np.argmax(f1_scores[1:])
    # max_precision_index = find_best_index(np.column_stack((tp_arr, fp_arr)))
    max_precision_thresh = loop_threshold[max_precision_index]
    max_precision_loop_pair = loop_pair0[(loop_pair0[:, 2] > 0)]
    max_precision_loop_pair = max_precision_loop_pair[(max_precision_loop_pair[:, 2] < max_precision_thresh)]

    ax3.plot(position0[:, 0], position0[:, 1], 5, color='k', zorder=1)
    ax3.plot(position1[:, 0], position1[:, 1], 25, color='k', zorder=1)
    tm, fm = 0, 0
    loop_or_not = []
    dataset = datasets0.split("-")
    elev, azim = 30, -60
    for idx in range(len(max_precision_loop_pair)):
        translation_diff = np.sqrt((position0[int(max_precision_loop_pair[idx, 0]), 0] - position1[int(max_precision_loop_pair[idx, 1]), 0])**2 + 
                                   (position0[int(max_precision_loop_pair[idx, 0]), 1] - position1[int(max_precision_loop_pair[idx, 1]), 1])**2)
        
        if translation_diff < gt_threshold:
            tm += 1
            # print(TerminalColors.GREEN + "True Matching Pair : ", str(max_precision_loop_pair[idx, 0]), str(max_precision_loop_pair[idx, 1]) + TerminalColors.RESET) 
            ax3.plot([position0[int(max_precision_loop_pair[idx, 0]), 0],    position1[int(max_precision_loop_pair[idx, 1]), 0]],
                     [position0[int(max_precision_loop_pair[idx, 0]), 1],    position1[int(max_precision_loop_pair[idx, 1]), 1]],
                     [5,              25], c = 'limegreen', zorder=0, linewidth="0.5", alpha=0.6)
            loop_or_not.append(1)
            
        else:
            fm += 1
            # print(TerminalColors.RED + "False Matching Pair : ", str(max_precision_loop_pair[idx, 0]), str(max_precision_loop_pair[idx, 1]) + TerminalColors.RESET) 
            ax3.plot([position0[int(max_precision_loop_pair[idx, 0]), 0],    position1[int(max_precision_loop_pair[idx, 1]), 0]],
                     [position0[int(max_precision_loop_pair[idx, 0]), 1],    position1[int(max_precision_loop_pair[idx, 1]), 1]],
                     [5,              25], c = 'red', zorder=0, linewidth="0.5", alpha=0.6)
            loop_or_not.append(0)

        query.append(max_precision_loop_pair[idx, 0])
        candidate.append(max_precision_loop_pair[idx, 1])
    
    ax3.view_init(elev=elev, azim=azim)
    
    print("True Matching : ", tm)
    print("False Matching : ", fm)

    plt.axis('off')
    # plt.show()

    precision = np.array(precision)
    recall = np.array(recall)
    precision_recall = np.column_stack((recall, precision))

    sensitivity = np.array(sensitivity)
    false_positive_rate = np.array(false_positive_rate)
    roc = np.column_stack((false_positive_rate, sensitivity))
    recall_at1 = np.array([tp/ground_truth0])
    pr_aucscore = np.array([auc(recall, precision)])
    roc_aucscore = np.array([roc_auc_score(y_test, y_score)])
    f1scores = np.array([f1_scores]).transpose()
    f1_recall = np.column_stack((recall, f1scores))
    ap = np.array([average_precision_score(y_test, y_score)])
    matching_pair = np.column_stack((np.array(query), np.array(candidate), np.array(loop_or_not)))
    
    saveMultiResults("recall@1", datasets0, datasets1, descriptor, param, recall_at1)
    saveMultiResults("pr_curve", datasets0, datasets1, descriptor, param, precision_recall)
    saveMultiResults("roc_curve", datasets0, datasets1, descriptor, param, roc)
    saveMultiResults("matching_pair", datasets0, datasets1, descriptor, param, matching_pair)
    saveMultiResults("pr_auc_score", datasets0, datasets1, descriptor, param, pr_aucscore)
    saveMultiResults("roc_auc_score", datasets0, datasets1, descriptor, param, roc_aucscore)
    saveMultiResults("f1_score", datasets0, datasets1, descriptor, param, f1scores)
    saveMultiResults("f1_recall", datasets0, datasets1, descriptor, param, f1_recall)
    saveMultiResults("average_precision_score", datasets0, datasets1, descriptor, param, ap)
    
    fig_path = osp.join(figure_path, "multi_session", datasets0 + '_to_' + datasets1)
    fprcurve_path = osp.join(fig_path, "pr_curve")
    froc_curve_path = osp.join(fig_path, "roc_curve")
    fpr_path = osp.join(fig_path, "place_recogntiion")
    createDir(fprcurve_path)
    createDir(froc_curve_path)
    createDir(fpr_path)
    fig1.savefig(osp.join(fprcurve_path, descDirName(descriptor, param) + '.svg'))
    fig2.savefig(osp.join(froc_curve_path, descDirName(descriptor, param) + '.svg'))
    fig3.savefig(osp.join(fpr_path, descDirName(descriptor, param) + '.svg'))
## =====================================================================================================================================================


## Descriptor Utils
## =====================================================================================================================================================
def descriptor_path(descriptor_dir):
    descriptor_dir = os.path.join(descriptor_dir)
    descriptorfile_list = os.listdir(descriptor_dir)
    descriptorfile_list = natsort.natsorted(descriptorfile_list)
    descriptor_fullpaths = [os.path.join(descriptor_dir, name) for name in descriptorfile_list]
    num_descriptors = len(descriptorfile_list)
    return descriptor_fullpaths, num_descriptors

def get1DDescritor(descriptor_dir):
    descriptor_fullpaths, num_descriptors = descriptor_path(descriptor_dir)
    desc_test = np.load(descriptor_fullpaths[0])
    try:
        dimension, list = desc_test.shape
        descriptor = np.zeros((num_descriptors, dimension))
        for i in range(num_descriptors):
            dd = np.load(descriptor_fullpaths[i])
            descriptor[i, :] = dd[:, 0]
    except ValueError:
        dimension = len(desc_test)
        descriptor = np.zeros((num_descriptors, dimension))
        for i in range(num_descriptors):
            dd = np.load(descriptor_fullpaths[i])
            descriptor[i, :] = dd
    return descriptor

def get2DDescritor(descriptor_dir):
    descriptor_fullpaths, num_descriptors = descriptor_path(descriptor_dir)
    desc_test = np.load(descriptor_fullpaths[0])
    dimension1, dimension2 = desc_test.shape
    descriptor = np.zeros((num_descriptors, dimension1, dimension2))
    for i in range(num_descriptors):
        descriptor[i, :, :] = np.load(descriptor_fullpaths[i])
        
    return descriptor
## =====================================================================================================================================================