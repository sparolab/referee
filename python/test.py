import cv2
import argparse

from scipy.spatial import cKDTree

from utils.referee import *
from utils.feature_extractor import *

# ===========================================================================================================
parser = argparse.ArgumentParser(description= "ReFeree")

parser.add_argument('--r_referee_split_ratio', type = int, default = '80',  help = '')
parser.add_argument('--a_referee_split_ratio', type = int, default = '10',  help = '')

args = parser.parse_args()
# ===========================================================================================================

# ===========================================================================================================
# ================================================   Query   ================================================
# ===========================================================================================================
query_radar_image = cv2.imread("fig/000000.png", cv2.IMREAD_GRAYSCALE)
query_feature_image = cen2018features(query_radar_image.T)
query_binary_image = targets_to_polar_image(query_feature_image, query_radar_image.T.shape)

query_r_referee = generate_r_referee(query_binary_image, args.r_referee_split_ratio)
query_a_referee = generate_a_referee(query_binary_image, args.a_referee_split_ratio)

# ===========================================================================================================
# =============================================   Candidate 1   =============================================
# ===========================================================================================================
candidates1_radar_image = cv2.imread("fig/000001.png", cv2.IMREAD_GRAYSCALE)
candidates1_feature_image = cen2018features(candidates1_radar_image.T)
candidates1_binary_image = targets_to_polar_image(candidates1_feature_image, candidates1_radar_image.T.shape)

candidates1_r_referee = generate_r_referee(candidates1_binary_image, args.r_referee_split_ratio)
candidates1_a_referee = generate_a_referee(candidates1_binary_image, args.a_referee_split_ratio)

# ===========================================================================================================
# =============================================   Candidate 2   =============================================
# ===========================================================================================================
candidates2_radar_image = cv2.imread("fig/000010.png", cv2.IMREAD_GRAYSCALE)
candidates2_feature_image = cen2018features(candidates2_radar_image.T)
candidates2_binary_image = targets_to_polar_image(candidates2_feature_image, candidates2_radar_image.T.shape)

candidates2_r_referee = generate_r_referee(candidates2_binary_image, args.r_referee_split_ratio)
candidates2_a_referee = generate_a_referee(candidates2_binary_image, args.a_referee_split_ratio)

# ===========================================================================================================
# ==========================================   Place Recognition   ==========================================
# ===========================================================================================================
num_candidates = 2
dimension = len(query_r_referee)
database = np.zeros((num_candidates, dimension))
database[0, :] = candidates1_r_referee[:, 0]
database[1, :] = candidates2_r_referee[:, 0]

kdtree = cKDTree(database)
loop_value, candidate_idx = kdtree.query(query_r_referee[:, 0], k=1)

# ===========================================================================================================
# =====================================   Initial Heading Estimation   ======================================
# ===========================================================================================================
if candidate_idx == 0:
    print("Distance is : ", loop_value, " between query and candidates1!!")
    initial_cosdist = []
    for shift_index in range(len(candidates1_a_referee)):
        initial_cosine_similarity = np.dot(query_a_referee, candidates1_a_referee) / (np.linalg.norm(query_a_referee) * np.linalg.norm(candidates1_a_referee))
        initial_cosdist.append(initial_cosine_similarity)
        initial_angle = (np.argmin(initial_cosdist)) * (360/len(query_a_referee))
    print("Initial Heading is : ", initial_angle, " between query and candidates1!!")
else:
    print("Distance is : ", loop_value, " between query and candidates2!!")
    initial_cosdist = []
    for shift_index in range(len(candidates2_a_referee)):
        initial_cosine_similarity = np.dot(query_a_referee, candidates2_a_referee) / (np.linalg.norm(query_a_referee) * np.linalg.norm(candidates2_a_referee))
        initial_cosdist.append(initial_cosine_similarity)
        initial_angle = (np.argmin(initial_cosdist))  * (360/len(query_a_referee))
    print("Initial Heading is : ", initial_angle, " between query and candidates2!!")