import os
import os.path as osp

pkg_path = "/home/bhbhchoi/Project/ReFeree_journal/Referee_git"
dataset_path = osp.join("/home/bhbhchoi/Project", "Datasets")
features_path = osp.join(pkg_path, "Features")
results_path = osp.join(pkg_path, "results")
evaluation_path = osp.join(pkg_path, "evaluation")
figure_path = osp.join(pkg_path, "figure")
desc_path = osp.join(pkg_path, "Description")

referee_split_ratio = 8

def saveDescPath(datasets_name, desc, param=""):
    path = osp.join(desc_path, datasets_name, descDirName(desc, param))
    createDir(path)
    return path

def saveResultPath(datasets_name, desc, param=""):
    path = osp.join(results_path, datasets_name, descDirName(desc, param))
    createDir(path)
    return path

def descParamsToStr(param):
    # 입력 변수의 유형에 따라 처리
    if isinstance(param, int):
        # 정수형(int)일 경우, 문자열로 변환
        return str(param)
    if isinstance(param, float):
        # 정수형(int)일 경우, 문자열로 변환
        return str(param)
    elif isinstance(param, tuple) and len(param) == 2:
        # 튜플일 경우, 각 요소를 문자열로 변환하고 'x'로 연결
        return f"{param[0]}x{param[1]}"
    elif isinstance(param, str):
        return ""

def descDirName(desc, param):
    return desc + "_" + descParamsToStr(param)

def createDir(path):
    if not osp.exists(path):
        os.makedirs(path)