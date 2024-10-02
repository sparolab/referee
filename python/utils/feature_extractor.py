import numpy as np

from scipy import ndimage

def targets_to_polar_image(targets, shape):
    polar = np.zeros(shape)
    N = targets.shape[0]
    for i in range(0, N):
        polar[targets[i, 0], targets[i, 1]] = 255
    return polar

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
    p = ndimage.uniform_filter1d(q, size=sigma_gauss, mode='reflect') # N x R 682/31
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
