// Pose-graph SLAM over radar odometry, with ReFeree closing the loops.
//
// A ROS 2 port of yeti_radar_odometry's posegraph.cpp. The original ran six
// threads over a dozen file-scope vectors guarded by three mutexes; the work is
// the same, but the keyframes live in one object (KeyframeStore), the loop
// registration in another (loop_registration), the file writing in a third
// (graph_io), and the node keeps only the graph.
//
// The g2o it writes is the same file a multi-session optimiser reads back as
// one session, so the two ends of the pipeline meet without a converter.

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "referee_interfaces/msg/referee_descriptor.hpp"
#include "referee_slam/graph_io.hpp"
#include "referee_slam/keyframe_store.hpp"
#include "referee_slam/loop_registration.hpp"

namespace referee_slam {

class PoseGraphNode : public rclcpp::Node {
 public:
  PoseGraphNode() : Node("referee_slam") {
    save_directory_ = declare_parameter<std::string>("save_directory", "/data/output/");
    keyframe_gap_m_ = declare_parameter<double>("keyframe_gap_m", 2.0);
    frame_id_ = declare_parameter<std::string>("frame_id", "odom");
    loop_period_ms_ = declare_parameter<int>("loop_detection_period_ms", 100);
    registration_.fitness_threshold = declare_parameter<double>("loop_fitness_threshold", 200.0);
    registration_.use_descriptor_yaw = declare_parameter<bool>("use_descriptor_yaw", true);

    initialiseNoise();
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    isam_ = std::make_unique<gtsam::ISAM2>(params);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "odom", 100, [this](nav_msgs::msg::Odometry::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(input_mutex_);
          odometry_.push_back(*msg);
        });
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "cloud_local", 100, [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(input_mutex_);
          clouds_.push_back(*msg);
        });
    descriptor_sub_ = create_subscription<referee_interfaces::msg::RefereeDescriptor>(
        "referee", 100, [this](referee_interfaces::msg::RefereeDescriptor::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(input_mutex_);
          descriptors_.push_back(*msg);
        });

    path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 10);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("pose", 10);

    graph_worker_ = std::thread(&PoseGraphNode::runGraph, this);
    loop_worker_ = std::thread(&PoseGraphNode::runLoopDetection, this);
  }

  ~PoseGraphNode() override {
    running_ = false;
    if (graph_worker_.joinable()) graph_worker_.join();
    if (loop_worker_.joinable()) loop_worker_.join();
    save();
  }

 private:
  void initialiseNoise() {
    prior_noise_ = gtsam::noiseModel::Diagonal::Variances(
        (gtsam::Vector(6) << 1e12, 1e12, 1e12, 1e12, 1e12, 1e12).finished());
    odometry_noise_ = gtsam::noiseModel::Diagonal::Variances(
        (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    // Loops are the measurements most likely to be wrong, so they get a robust
    // kernel rather than a tighter covariance.
    loop_noise_ = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1.0),
        gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished()));
  }

  // Pairs an odometry message with the cloud and descriptor of the same scan,
  // and keeps it only when the vehicle has moved far enough.
  void runGraph() {
    Pose6D previous{};
    double travelled = std::numeric_limits<double>::max();  // force the first frame in
    bool have_previous = false;

    while (running_ && rclcpp::ok()) {
      nav_msgs::msg::Odometry odom;
      sensor_msgs::msg::PointCloud2 cloud;
      referee_interfaces::msg::RefereeDescriptor descriptor;
      {
        std::lock_guard<std::mutex> lock(input_mutex_);
        if (odometry_.empty() || clouds_.empty() || descriptors_.empty()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
          continue;
        }
        odom = odometry_.front();   odometry_.pop_front();
        cloud = clouds_.front();    clouds_.pop_front();
        descriptor = descriptors_.front(); descriptors_.pop_front();
      }

      const Pose6D pose = fromOdometry(odom);
      if (have_previous) travelled += translationBetween(previous, pose);
      previous = pose;
      have_previous = true;
      if (travelled < keyframe_gap_m_) continue;
      travelled = 0.0;

      auto keyframe_cloud = std::make_shared<Cloud>();
      pcl::fromROSMsg(cloud, *keyframe_cloud);

      const int index = store_.add(pose, rclcpp::Time(odom.header.stamp).seconds(),
                                   keyframe_cloud, descriptor.range_descriptor,
                                   descriptor.angle_descriptor);
      addOdometryFactor(index);
      optimise();
      publishPath();
    }
  }

  void addOdometryFactor(int index) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    const gtsam::Pose3 pose = toGtsam(store_.pose(index));

    if (index == 0) {
      graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, pose, prior_noise_));
      initial_.insert(0, pose);
      return;
    }
    const gtsam::Pose3 previous = toGtsam(store_.pose(index - 1));
    const gtsam::Pose3 relative = previous.between(pose);
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(index - 1, index, relative, odometry_noise_));
    initial_.insert(index, pose);
    edges_.push_back(edgeLine(index - 1, index, relative));
  }

  // Proposes a loop for the newest keyframe, registers it, and adds the
  // constraint. Kept off the graph thread because registration is the slow part.
  void runLoopDetection() {
    while (running_ && rclcpp::ok()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(loop_period_ms_));
      const auto proposal = store_.proposeLoop();
      if (!proposal || proposal->previous_index == proposal->current_index) continue;

      const auto relative = registerLoop(store_, *proposal, registration_);
      if (!relative) continue;

      {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            proposal->previous_index, proposal->current_index, *relative, loop_noise_));
        edges_.push_back(
            edgeLine(proposal->previous_index, proposal->current_index, *relative));
      }
      RCLCPP_INFO(get_logger(), "loop %d <- %d", proposal->previous_index,
                  proposal->current_index);
      optimise();
    }
  }

  void optimise() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (initial_.empty()) return;
    isam_->update(graph_, initial_);
    isam_->update();
    graph_.resize(0);
    initial_.clear();

    estimate_ = isam_->calculateEstimate();
    for (std::size_t i = 0; i < estimate_.size(); ++i) {
      const auto pose = estimate_.at<gtsam::Pose3>(i);
      const auto rpy = pose.rotation().rpy();
      store_.setOptimisedPose(static_cast<int>(i),
                              Pose6D{pose.translation().x(), pose.translation().y(),
                                     pose.translation().z(), rpy(0), rpy(1), rpy(2)});
    }
  }

  void publishPath() {
    nav_msgs::msg::Path path;
    path.header.stamp = now();
    path.header.frame_id = frame_id_;
    for (const auto& pose : store_.optimisedPoses()) {
      geometry_msgs::msg::PoseStamped stamped;
      stamped.header = path.header;
      stamped.pose.position.x = pose.x;
      stamped.pose.position.y = pose.y;
      stamped.pose.position.z = pose.z;
      tf2::Quaternion q;
      q.setRPY(pose.roll, pose.pitch, pose.yaw);
      stamped.pose.orientation = tf2::toMsg(q);
      path.poses.push_back(stamped);
    }
    path_pub_->publish(path);
    if (!path.poses.empty()) {
      nav_msgs::msg::Odometry odom;
      odom.header = path.header;
      odom.pose.pose = path.poses.back().pose;
      odom_pub_->publish(odom);
    }
  }

  void save() {
    if (store_.size() == 0) return;
    std::lock_guard<std::mutex> lock(graph_mutex_);
    try {
      savePoseGraphG2o(save_directory_ + "singlesession_posegraph.g2o", store_.poses(), edges_);
      saveVerticesKitti(save_directory_ + "odom_poses_kitti.txt", store_.poses());
      if (!estimate_.empty())
        saveVerticesKitti(save_directory_ + "optimized_poses_kitti.txt", estimate_);
      RCLCPP_INFO(get_logger(), "wrote %zu keyframes to %s", store_.size(),
                  save_directory_.c_str());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "could not save: %s", e.what());
    }
  }

  std::string save_directory_, frame_id_;
  double keyframe_gap_m_;
  int loop_period_ms_;
  RegistrationParams registration_;

  KeyframeStore store_;
  std::mutex graph_mutex_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_, estimate_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  std::vector<std::string> edges_;
  gtsam::SharedNoiseModel prior_noise_, odometry_noise_, loop_noise_;

  std::mutex input_mutex_;
  std::deque<nav_msgs::msg::Odometry> odometry_;
  std::deque<sensor_msgs::msg::PointCloud2> clouds_;
  std::deque<referee_interfaces::msg::RefereeDescriptor> descriptors_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<referee_interfaces::msg::RefereeDescriptor>::SharedPtr descriptor_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  std::atomic<bool> running_{true};
  std::thread graph_worker_, loop_worker_;
};

}  // namespace referee_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<referee_slam::PoseGraphNode>());
  rclcpp::shutdown();
  return 0;
}
