#include <iostream>

#include "referee.hpp"
#include "features.hpp"
#include "radar_utils.hpp"

ReFereeRDB refereeR_db;
ReFereeADB refereeA_db;

ReFereeR getReFereeRDescriptor(std::string radar_file, float zq, int sigma_gauss, int min_range) {
    std::vector<int64_t> times;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    load_radar(radar_file, times, azimuths, valid, fft_data, CIR204); // use CIR204 for MulRan dataset

    Eigen::MatrixXd polar, targets;
    cen2018features(fft_data, zq, sigma_gauss, min_range, targets);

    polar = targets_to_polar_image(fft_data, targets);
    return ReFereeR(polar);
}

ReFereeA getReFereeADescriptor(std::string radar_file, float zq, int sigma_gauss, int min_range) {
    std::vector<int64_t> times;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    load_radar(radar_file, times, azimuths, valid, fft_data, CIR204); // use CIR204 for MulRan dataset

    Eigen::MatrixXd polar, targets;
    cen2018features(fft_data, zq, sigma_gauss, min_range, targets);

    polar = targets_to_polar_image(fft_data, targets);
    return ReFereeA(polar);
}

int main(int argc, char* argv[]) {
    // sensor params 
    int min_range = 58; // min range of radar points (bin)

    // cen2018 parameters
    float zq = 3.0;
    int sigma_gauss = 17;

    std::vector<int64_t> times;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    std::string datadir = "../examples";

    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);

    std::string query_data = "000000.png";
    std::string query_data_path = datadir + "/" + query_data;

    // Generate database
    for(std::string radar_file: radar_files) {
        if(radar_file == query_data)
            continue;
        refereeR_db.push_back(getReFereeRDescriptor(datadir + "/" + radar_file, zq, sigma_gauss, min_range));
        refereeA_db.push_back(getReFereeADescriptor(datadir + "/" + radar_file, zq, sigma_gauss, min_range));
    }

    cout << refereeR_db.size() << " radar data are loaded." << endl;

    // Load query
    ReFereeR queryR = getReFereeRDescriptor(query_data_path, zq, sigma_gauss, min_range);
    ReFereeA queryA = getReFereeADescriptor(query_data_path, zq, sigma_gauss, min_range);
    int loop_id = refereeR_db.detectLoopClosureID(queryR);
    double yaw_diff = refereeA_db.getYawDiff(queryA, loop_id);
    cout << "Query is matched with " << radar_files[loop_id] << " and angle difference is " << yaw_diff << endl;

    return 0;
}