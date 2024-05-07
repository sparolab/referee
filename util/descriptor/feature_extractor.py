import cv2
import numpy as np
import os.path as osp
from scipy import ndimage
from glob import glob
from ..params import *

def targets_to_polar_image(targets, shape):
    polar = np.zeros(shape)
    N = targets.shape[0]
    for i in range(0, N):
        polar[targets[i, 0], targets[i, 1]] = 255
    return polar

def targets_to_polar_intensity_image(targets, intensity_image):
    polar = np.zeros(intensity_image.shape)
    N = targets.shape[0]
    for i in range(0, N):
        polar[targets[i, 0], targets[i, 1]] = intensity_image[targets[i, 0], targets[i, 1]]
    return polar

def load_radar(example_path, fix_azimuths=False):
    """Decode a single Oxford Radar RobotCar Dataset radar example
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset Example png
    Returns:
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        fft_data (np.ndarray): Radar power readings along each azimuth
    """
    # Hard coded configuration to simplify parsing code
    encoder_size = 5600
    #t = float(example_path.split('.')[0]) * 1.0e-6
    raw_example_data = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    N = raw_example_data.shape[0]
    azimuth_step = 2 * np.pi / N
    if fix_azimuths:
        azimuths = np.zeros((N, 1), dtype=np.float32)
        for i in range(N):
            azimuths[i, 0] = i * azimuth_step
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32) / 255.
    return timestamps, azimuths, valid, fft_data

def cen2018features(fft_data: np.ndarray, min_range=58, zq=3.0, sigma_gauss=17) -> np.ndarray:
    """Extract features from polar radar data using the method described in cen_icra18
    Args:
        fft_data (np.ndarray): Polar radar power readings
        min_range (int): targets with a range bin less than or equal to this value will be ignored.
        zq (float): if y[i] > zq * sigma_q then it is considered a potential target point
        sigma_gauss (int): std dev of the gaussian filter used to smooth the radar signal
        
    Returns:
        np.ndarray: N x 2 array of feature locations (azimuth_bin, range_bin)
    """
    nazimuths = fft_data.shape[0]
    # w_median = 200
    # q = fft_data - ndimage.median_filter(fft_data, size=(1, w_median))  # N x R
    q = fft_data - np.mean(fft_data, axis=1, keepdims=True)
    # p = ndimage.gaussian_filter1d(q, sigma=sigma_gauss, truncate=3.0) # N x R
    # p = ndimage.gaussian_filter1d(q, sigma=17, truncate=3.0) # N x R 654/67
    p = ndimage.uniform_filter1d(q, size=17, mode='reflect') # N x R 656/65
    # p = ndimage.minimum_filter1d(q, size=20) # N x R
    # p = ndimage.generic_filter1d(q, filter_size=5) # N x R
    noise = np.where(q < 0, q, 0) # N x R
    nonzero = np.sum(q < 0, axis=-1, keepdims=True) # N x 1
    sigma_q = np.sqrt(np.sum(noise**2, axis=-1, keepdims=True) / nonzero) # N x 1

    def norm(x, sigma):
        return np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    nqp = norm(q - p, sigma_q)
    npp = norm(p, sigma_q)
    nzero = norm(np.zeros((nazimuths, 1)), sigma_q)
    y = q * (1 - nqp / nzero) + p * ((nqp - npp) / nzero)
    t = np.nonzero(y > zq * sigma_q) # thresholded signal
    # Extract peak centers
    current_azimuth = t[0][0]
    peak_points = [t[1][0]]
    peak_centers = []

    def mid_point(l):
        return l[len(l) // 2]

    for i in range(1, len(t[0])):
        if t[1][i] - peak_points[-1] > 1 or t[0][i] != current_azimuth:
            m = mid_point(peak_points)
            if m > min_range:
                peak_centers.append((current_azimuth, m))
            peak_points = []
        current_azimuth = t[0][i]
        peak_points.append(t[1][i])
    if len(peak_points) > 0 and mid_point(peak_points) > min_range:
        peak_centers.append((current_azimuth, mid_point(peak_points)))
    
    return np.asarray(peak_centers)

def load_radar_image(radar_files, idx):
    radar_image = cv2.imread(osp.join(radar_files[idx]), cv2.IMREAD_GRAYSCALE)
    if radar_image.shape[0] < radar_image.shape[1]:
            radar_image = radar_image.T
    return radar_image

def feature_extractor(radar_files, idx):
    tmp = load_radar_image(radar_files, idx)
    targets = cen2018features(tmp.T)
    polar = targets_to_polar_image(targets, tmp.T.shape)
    return polar