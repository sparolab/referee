import numpy as np

def getPose(pose_path):
    pose = np.loadtxt(pose_path, delimiter=',', dtype=float)
    return pose