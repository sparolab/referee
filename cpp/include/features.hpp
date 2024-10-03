#pragma once
#include <Eigen/Dense>
#include <vector>
#include <opencv2/core.hpp>
#include "radar_utils.hpp"

/*!
   \brief Extract features from polar radar data using the method described in cen_icra18
   \param fft_data Polar radar power readings
   \param zq If y(i, j) > zq * sigma_q then it is considered a potential target point
   \param sigma_gauss std dev of the gaussian filter uesd to smooth the radar signal
   \param min_range We ignore the range bins less than this
   \param targets [out] Matrix of feature locations (azimuth_bin, range_bin, 1) x N
*/
double cen2018features(cv::Mat fft_data, float zq, int sigma_gauss, int min_range, Eigen::MatrixXd &targets);
