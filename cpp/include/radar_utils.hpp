#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <boost/algorithm/string.hpp>

#define CTS350 0
#define CIR204 1

void load_radar(std::string path, std::vector<int64_t> &timestamps, std::vector<double> &azimuths,
    std::vector<bool> &valid, cv::Mat &fft_data, int navtech_version = CTS350);

Eigen::MatrixXd targets_to_polar_image(cv::Mat &fft_data, Eigen::MatrixXd& targets);

void get_file_names(std::string datadir, std::vector<std::string> &radar_files, std::string extension = "");