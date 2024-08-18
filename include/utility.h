/*
Description: 读取参数，提供一些工具方法
Author     : Ji Qingshi
date       :
*/

#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// #include <opencv/imgproc.hpp> // for old version
#include <opencv2/imgproc.hpp> // compatible with ros-noetic

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h> // read and save pcd file
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>            // save or delete cloud in a certain space
#include <pcl_conversions/pcl_conversions.h> //???

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip> // concurrency for multi-thread
#include <array>
#include <thread>
#include <mutex>

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType
{
  VELODYNE,
  OUSTER
};
/**
 * 通用的全局参数配置的类,all parameters are stored in this class
 */
class ParamServer
{
 public:
  ros::NodeHandle nh;
  std::string robot_id;

  // Topics
  string pointCloudTopic;
  string imuTopic;
  string odomTopic; //???
  string gpsTopic;

  // Frames
  string lidarFrame;
  string baselinkFrame; // ???
  string odometryFrame;
  string mapFrame;

  // GPS Settings
  bool useImuHeadingInitialization;
  bool useGpsElevation;
  float gpsCovThreshold;
  float poseCovThreshold;

  // Save pcd
  bool savePCD;
  string savePCDDirectory;

  // Lidar Sensor Configuration
  SensorType sensor;
  int N_SCAN;
  int Horizon_SCAN;
  int downsampleRate;
  float lidarMinRange;
  float lidarMaxRange;

  // IMU
  float imuAccNoise;
  float imuGyrNoise;
  float imuAccBiasN;
  float imuGyrBiasN;
  float imuGravity;
  float imuRPYWeight;
  vector<double> extRotV;
  vector<double> extRPYV;
  vector<double> extTransV;
  Eigen::Matrix3d extRot; // rotation
  Eigen::Matrix3d extRPY; // unused varibles
  Eigen::Vector3d extTrans;
  Eigen::Quaterniond extQRPY;

  // LOAM
  float edgeThreshold; // curvature threshold
  float surfThreshold;
  int edgeFeatureMinValidNum; // if feature points are too few, do not registrate.
  int surfFeatureMinValidNum;

  // voxel filter paprams
  float odometrySurfLeafSize;  // 特征提取后，下采样面点
  float mappingCornerLeafSize; // 下采样地图点
  float mappingSurfLeafSize;

  float z_tollerance;        // z,pitch and yaw shouldn't be larger than this value for 2D situation;
  float rotation_tollerance; // 相邻两帧位姿的角度的变化阈值

  // CPU Params
  int numberOfCores;             // 最多可使用的cpu核心数量, for concurrency
  double mappingProcessInterval; // min interval between process of two consecutive clouds

  // Surrounding map: extract map from cached clouds by pose of clouds.
  float surroundingkeyframeAddingDistThreshold;  // threshold to judge whether this cloud should be a key frame
  float surroundingkeyframeAddingAngleThreshold; // 角度阈值
  float surroundingKeyframeDensity;              // pose of selected cloud won't be near to each other
  float surroundingKeyframeSearchRadius;         // select clouds whose pose is in this area

  // Loop closure
  bool loopClosureEnableFlag;          // 是否使用回环
  float loopClosureFrequency;          // fraquency to detect loop
  int surroundingKeyframeSize;         // 构建回环时使用的关键帧点云的最大数量
  float historyKeyframeSearchRadius;   // 在这个范围内寻找用于回环的关键帧
  float historyKeyframeSearchTimeDiff; // clouds that is close in time should be considered as loop
  int historyKeyframeSearchNum;        // 从回环帧前后这个数量的点云合并作为回环点云
  float historyKeyframeFitnessScore;   // score to be a loop by icp

  // global map visualization radius
  float globalMapVisualizationSearchRadius;
  float globalMapVisualizationPoseDensity;
  float globalMapVisualizationLeafSize;

  ParamServer()
  { // get parameters from params.yaml
    // <type>(keyword, varible, default value): get value of keyword and write it to varible
    nh.param<std::string>("/robot_id", robot_id, "roboat");

    nh.param<std::string>("lio_sam/pointCloudTopic", pointCloudTopic, "points_raw");
    nh.param<std::string>("lio_sam/imuTopic", imuTopic, "imu_correct");
    nh.param<std::string>("lio_sam/odomTopic", odomTopic, "odometry/imu");
    nh.param<std::string>("lio_sam/gpsTopic", gpsTopic, "odometry/gps");

    nh.param<std::string>("lio_sam/lidarFrame", lidarFrame, "base_link");
    nh.param<std::string>("lio_sam/baselinkFrame", baselinkFrame, "base_link");
    nh.param<std::string>("lio_sam/odometryFrame", odometryFrame, "odom");
    nh.param<std::string>("lio_sam/mapFrame", mapFrame, "map");

    nh.param<bool>("lio_sam/useImuHeadingInitialization", useImuHeadingInitialization, false);
    nh.param<bool>("lio_sam/useGpsElevation", useGpsElevation, false);
    nh.param<float>("lio_sam/gpsCovThreshold", gpsCovThreshold, 2.0);
    nh.param<float>("lio_sam/poseCovThreshold", poseCovThreshold, 25.0);

    nh.param<bool>("lio_sam/savePCD", savePCD, false);
    nh.param<std::string>("lio_sam/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");

    std::string sensorStr;
    nh.param<std::string>("lio_sam/sensor", sensorStr, "");

    if (sensorStr == "velodyne")
    {
      sensor = SensorType::VELODYNE;
    }
    else if (sensorStr == "ouster")
    {
      sensor = SensorType::OUSTER;
    }
    else
    {
      ROS_ERROR_STREAM("Invalid sensor type (must be either 'velodyne' or 'ouster'): " << sensorStr);
      ros::shutdown();
    }

    nh.param<int>("lio_sam/N_SCAN", N_SCAN, 16);
    nh.param<int>("lio_sam/Horizon_SCAN", Horizon_SCAN, 1800);
    nh.param<int>("lio_sam/downsampleRate", downsampleRate, 1);
    nh.param<float>("lio_sam/lidarMinRange", lidarMinRange, 1.0);
    nh.param<float>("lio_sam/lidarMaxRange", lidarMaxRange, 1000.0);

    nh.param<float>("lio_sam/imuAccNoise", imuAccNoise, 0.01);
    nh.param<float>("lio_sam/imuGyrNoise", imuGyrNoise, 0.001);
    nh.param<float>("lio_sam/imuAccBiasN", imuAccBiasN, 0.0002);
    nh.param<float>("lio_sam/imuGyrBiasN", imuGyrBiasN, 0.00003);
    nh.param<float>("lio_sam/imuGravity", imuGravity, 9.80511);
    nh.param<float>("lio_sam/imuRPYWeight", imuRPYWeight, 0.01);
    nh.param<vector<double>>("lio_sam/extrinsicRot", extRotV, vector<double>());
    nh.param<vector<double>>("lio_sam/extrinsicRPY", extRPYV, vector<double>());
    nh.param<vector<double>>("lio_sam/extrinsicTrans", extTransV, vector<double>());
    // http://blue-stone-w.top/blog/Eigen%E5%AD%A6%E4%B9%A0/, explaination for .data()
    extRot   = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
    extRPY   = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
    extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
    extQRPY  = Eigen::Quaterniond(extRPY);

    nh.param<float>("lio_sam/edgeThreshold", edgeThreshold, 0.1);
    nh.param<float>("lio_sam/surfThreshold", surfThreshold, 0.1);
    nh.param<int>("lio_sam/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
    nh.param<int>("lio_sam/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

    nh.param<float>("lio_sam/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
    nh.param<float>("lio_sam/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
    nh.param<float>("lio_sam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

    nh.param<float>("lio_sam/z_tollerance", z_tollerance, FLT_MAX); // FLT_MAX is max positive float
    nh.param<float>("lio_sam/rotation_tollerance", rotation_tollerance, FLT_MAX);

    nh.param<int>("lio_sam/numberOfCores", numberOfCores, 2);
    nh.param<double>("lio_sam/mappingProcessInterval", mappingProcessInterval, 0.15);

    nh.param<float>("lio_sam/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
    nh.param<float>("lio_sam/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
    nh.param<float>("lio_sam/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
    nh.param<float>("lio_sam/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

    nh.param<bool>("lio_sam/loopClosureEnableFlag", loopClosureEnableFlag, false);
    nh.param<float>("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0);
    nh.param<int>("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
    nh.param<float>("lio_sam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
    nh.param<float>("lio_sam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
    nh.param<int>("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
    nh.param<float>("lio_sam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

    nh.param<float>("lio_sam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
    nh.param<float>("lio_sam/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
    nh.param<float>("lio_sam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

    usleep(100);
  }
  /**
   * IMU数据转换到lidar坐标系下
   */
  // input state in IMU frame and output state in lidar frame
  sensor_msgs::Imu imuConverter(const sensor_msgs::Imu &imu_in)
  {
    sensor_msgs::Imu imu_out = imu_in;
    // rotate IMU data to front-left-up frame
    // rotate acceleration 加速度转到雷达系
    Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
    acc                           = extRot * acc; // use extrinsics (lidar -> IMU) ext是lidar到imu的外参中的R矩阵
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // rotate gyroscope 角速度转到雷达系
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
    gyr                        = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
    // rotate roll pitch yaw; using 9-axis IMU, there are pose data
    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
    Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation.x      = q_final.x(); // transform eigen data to ros data
    imu_out.orientation.y      = q_final.y();
    imu_out.orientation.z      = q_final.z();
    imu_out.orientation.w      = q_final.w();

    // check: if use 6-axis IMU, there is no RPY data
    if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() + q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1)
    {
      ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
      ros::shutdown();
    }

    return imu_out;
  }
};

// convert pcl cloud to msg cloud, publish and return it
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub,
                                      pcl::PointCloud<PointType>::Ptr thisCloud,
                                      ros::Time thisStamp,
                                      std::string thisFrame)
{
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp    = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->getNumSubscribers() != 0)
  {
    thisPub->publish(tempCloud);
  }
  return tempCloud;
}

// if msg is large, this function could be less-efficient
template <typename T>
double ROS_TIME(T msg)
{
  return msg->header.stamp.toSec();
}

// extract data in msg to varible
template <typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
  *angular_x = thisImuMsg->angular_velocity.x;
  *angular_y = thisImuMsg->angular_velocity.y;
  *angular_z = thisImuMsg->angular_velocity.z;
}
template <typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
  *acc_x = thisImuMsg->linear_acceleration.x;
  *acc_y = thisImuMsg->linear_acceleration.y;
  *acc_z = thisImuMsg->linear_acceleration.z;
}
template <typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
  double imuRoll, imuPitch, imuYaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

  *rosRoll  = imuRoll;
  *rosPitch = imuPitch;
  *rosYaw   = imuYaw;
}

float pointDistance(PointType p)
{
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}
float pointDistance(PointType p1, PointType p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

#endif