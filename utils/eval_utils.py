import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, f1_score, roc_auc_score

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

def single_session_evaluation(loop_pair0, position0, exclude_node, gt_threshold, datasets0, description_name):
    arr = loop_pair0[:, 2]
    filtered_arr = arr[arr != 0]
    min_value = np.min(filtered_arr)
    max_value = np.max(filtered_arr)
    f1_scores = []
    
    y_test = []
    y_score = []
    
    loop_threshold  = np.linspace(max_value, min_value, 100)
    precision, recall = [], []
    query, candidate  = [], []
    
    for loop_thresh_idx, loop_thresh in tqdm(enumerate(loop_threshold)):
        tp, fp, fn, tn = 0, 0, 0, 0
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
                        tn += 1
                    else:
                        fn += 1
        try:
            precision.append(tp / (tp + fp))
            recall.append(tp / (tp + fn))
            f1 = 2 * ((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))
            f1_scores.append(f1)
        except ZeroDivisionError:
            pass
    
    fpr, tpr, _ = roc_curve(y_test, y_score) # input 순서 : 실제 라벨, 예측 값

    print(TerminalColors.GREEN + "Datasets : ", str(datasets0) + TerminalColors.RESET)
    print(TerminalColors.BLUE + "Recall@1 : ", str(precision[np.argmax(recall)]) + TerminalColors.RESET)
    print(TerminalColors.BLUE + "AUC_SCORE : ", str(roc_auc_score(y_test, y_score)) + TerminalColors.RESET)
    print(TerminalColors.BLUE + "F1_max_SCORE : ", str(max(f1_scores)) + TerminalColors.RESET)

    fig1 = plt.figure()
    ax1 = fig1.subplots()
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, 1.1)
    ax1.plot(recall, precision)
    fig2 = plt.figure()
    ax2 = fig2.subplots()
    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    ax2.plot(fpr, tpr)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    z = np.linspace(0, 10, len(position0))
    max_precision_index = np.argmax(precision)
    max_precision_thresh = loop_threshold[max_precision_index]
    max_precision_loop_pair = loop_pair0[(loop_pair0[:, 2] > 0)]
    max_precision_loop_pair = max_precision_loop_pair[(max_precision_loop_pair[:, 2] < max_precision_thresh)]
    ax3.plot(position0[:, 0], position0[:, 1], z)
    for idx in range(len(max_precision_loop_pair)):
        translation_diff = np.sqrt((position0[int(max_precision_loop_pair[idx, 0]), 0] - position0[int(max_precision_loop_pair[idx, 1]), 0])**2 +
                                   (position0[int(max_precision_loop_pair[idx, 0]), 1] - position0[int(max_precision_loop_pair[idx, 1]), 1])**2)
        if int(max_precision_loop_pair[idx, 0]) == 0:
            pass
        elif translation_diff < gt_threshold:
            print(TerminalColors.GREEN + "True Matching Pair : ", str(max_precision_loop_pair[idx, 0]), str(max_precision_loop_pair[idx, 1]) + TerminalColors.RESET, str(max_precision_loop_pair[idx, 2]))
            ax3.plot([position0[int(max_precision_loop_pair[idx, 0]), 0],    position0[int(max_precision_loop_pair[idx, 1]), 0]],
                    [position0[int(max_precision_loop_pair[idx, 0]), 1],    position0[int(max_precision_loop_pair[idx, 1]), 1]],
                    [z[int(max_precision_loop_pair[idx, 0])],               z[int(max_precision_loop_pair[idx, 1])]], c = 'limegreen')
        else:
            print(TerminalColors.RED + "False Matching Pair : ", str(max_precision_loop_pair[idx, 0]), str(max_precision_loop_pair[idx, 1]) + TerminalColors.RESET, str(max_precision_loop_pair[idx, 2]))
            ax3.plot([position0[int(max_precision_loop_pair[idx, 0]), 0],    position0[int(max_precision_loop_pair[idx, 1]), 0]],
                    [position0[int(max_precision_loop_pair[idx, 0]), 1],    position0[int(max_precision_loop_pair[idx, 1]), 1]],
                    [z[int(max_precision_loop_pair[idx, 0])],               z[int(max_precision_loop_pair[idx, 1])]], c = 'red')
    plt.axis('off')
    plt.show()
    precision = np.array(precision)
    recall = np.array(recall)
    precision_recall = np.column_stack((recall, precision))
    roc = np.column_stack((fpr, tpr))
    matching_pair = np.column_stack((np.array(query), np.array(candidate)))
    np.savetxt('evaluation/pr_curve/single_session_' + datasets0  + '_' + description_name + '.txt', precision_recall)
    np.savetxt('evaluation/roc_curve/single_session_' + datasets0  + '_' + description_name +'.txt', roc)
    np.savetxt('evaluation/matching_pair/single_session_' + datasets0  + '_' + description_name +'.txt', matching_pair)

    fig1.savefig('figure/' + 'single_session_' + datasets0 + '_' + description_name +'_pr_curve.png')
    fig2.savefig('figure/' + 'single_session_' + datasets0 + '_' + description_name +'_roc_curve.png')
    fig3.savefig('figure/' + 'single_session_' + datasets0 + '_' + description_name +'_place_recogntiion.png')
    
def multi_session_evaluation(loop_pair0, ground_truth0, position0, position1, gt_threshold, datasets0, datasets1, descriptor):
    arr = loop_pair0[:, 2]
    filtered_arr = arr[arr != 0]
    min_value = np.min(filtered_arr)
    max_value = np.max(filtered_arr)

    f1_scores = []
    
    y_test = []
    y_score = []

    loop_threshold  = np.linspace(max_value, min_value, 100)

    precision, recall = [], []
    query, candidate  = [], []

    for loop_thresh_idx, loop_thresh in tqdm(enumerate(loop_threshold)):
        tp, fp, fn, tn = 0, 0, 0, 0
        for idx in range(len(loop_pair0)):
            similarity_distance = loop_pair0[idx, 2]
            translation_diff    = np.sqrt((position0[int(loop_pair0[idx, 0]), 0] - position1[int(loop_pair0[idx, 1]), 0])**2 + 
                                          (position0[int(loop_pair0[idx, 0]), 1] - position1[int(loop_pair0[idx, 1]), 1])**2)

            if similarity_distance < loop_thresh:
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
        try:
            precision.append(tp / (tp + fp))
            recall.append(tp / (tp + fn))
            f1 = 2 * ((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))
            f1_scores.append(f1)
        except ZeroDivisionError:
            pass

    fpr, tpr, _ = roc_curve(y_test, y_score) # input 순서 : 실제 라벨, 예측 값
    
    print(datasets0, datasets1)
    print(TerminalColors.BLUE + "Recall@1 : ", str(precision[np.argmax(recall)]) + TerminalColors.RESET)
    print(TerminalColors.BLUE + "AUC_SCORE : ", str(roc_auc_score(y_test, y_score)) + TerminalColors.RESET)
    print(TerminalColors.BLUE + "F1_max_SCORE : ", str(max(f1_scores)) + TerminalColors.RESET)

    fig1 = plt.figure()
    ax1 = fig1.subplots()
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, 1.1)
    ax1.plot(recall, precision)

    fig2 = plt.figure()
    ax2 = fig2.subplots()
    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    ax2.plot(fpr, tpr)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')

    max_precision_index = np.argmax(precision)
    max_precision_thresh = loop_threshold[max_precision_index]
    max_precision_loop_pair = loop_pair0[(loop_pair0[:, 2] < max_precision_thresh)]

    ax3.plot(position0[:, 0], position0[:, 1], 5, color='k', zorder=1)
    ax3.plot(position1[:, 0], position1[:, 1], 25, color='k', zorder=1)
    tp = 0
    fp = 0
    loop_or_not = []
    for idx in range(len(max_precision_loop_pair)):
        translation_diff = np.sqrt((position0[int(max_precision_loop_pair[idx, 0]), 0] - position1[int(max_precision_loop_pair[idx, 1]), 0])**2 + 
                                   (position0[int(max_precision_loop_pair[idx, 0]), 1] - position1[int(max_precision_loop_pair[idx, 1]), 1])**2)
        
        if translation_diff < gt_threshold:
            tp += 1
            # print(TerminalColors.GREEN + "True Matching Pair : ", str(max_precision_loop_pair[idx, 0]), str(max_precision_loop_pair[idx, 1]) + TerminalColors.RESET) 
            ax3.plot([position0[int(max_precision_loop_pair[idx, 0]), 0],    position1[int(max_precision_loop_pair[idx, 1]), 0]],
                     [position0[int(max_precision_loop_pair[idx, 0]), 1],    position1[int(max_precision_loop_pair[idx, 1]), 1]],
                     [5,              25], c = 'limegreen', zorder=0, linewidth='0.5', alpha=0.2)
            loop_or_not.append(0)
        else:
            fp += 1
            # print(TerminalColors.RED + "False Matching Pair : ", str(max_precision_loop_pair[idx, 0]), str(max_precision_loop_pair[idx, 1]) + TerminalColors.RESET) 
            ax3.plot([position0[int(max_precision_loop_pair[idx, 0]), 0],    position1[int(max_precision_loop_pair[idx, 1]), 0]],
                     [position0[int(max_precision_loop_pair[idx, 0]), 1],    position1[int(max_precision_loop_pair[idx, 1]), 1]],
                     [5,              25], c = 'red', zorder=0, linewidth='0.5', alpha=0.2)
            loop_or_not.append(1)

        query.append(max_precision_loop_pair[idx, 0])
        candidate.append(max_precision_loop_pair[idx, 1])

    print("True Matching : ", tp)
    print("False Matching : ", fp)

    plt.axis('off')
    plt.show()

    precision = np.array(precision)
    recall = np.array(recall)
    precision_recall = np.column_stack((recall, precision))

    roc = np.column_stack((fpr, tpr))

    matching_pair = np.column_stack((np.array(query), np.array(candidate), np.array(loop_or_not)))

    np.savetxt('evaluation/pr_curve/multi_session_' + datasets0 + '_' + 'to_' + datasets1 + '_' + descriptor + '.txt', precision_recall)
    np.savetxt('evaluation/roc_curve/multi_session_' + datasets0  + '_' +  'to_' + datasets1 + '_' + descriptor + '.txt', roc)
    np.savetxt('evaluation/matching_pair/multi_session_' + datasets0 + '_' + 'to_' + datasets1 + '_' + descriptor + '.txt', matching_pair)

    fig1.savefig('figure/' + 'multi_session_' + datasets0 + '_to_' + datasets1 + '_' + descriptor + '_pr_curve.png')
    fig2.savefig('figure/' + 'multi_session_' + datasets0 + '_to_' + datasets1 + '_' + descriptor + '_roc_curve.png')
    fig3.savefig('figure/' + 'multi_session_' + datasets0 + '_to_' + datasets1 + '_' + descriptor + '_place_recogntiion.png')
