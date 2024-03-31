/*
Description: [match=registration,estimated=optimized,preint=preintegration,lo=lidarodometry,Euler=Euler Angle,rotation=orientation,vertice=node]

scan-map match and optimize; loop; gps fusion; factors graph optimization

rotation is described by Euler, not quaternion or rotation vector or rotation matrix

Author     : Ji Qingshi
date       :
*/

#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h> // Marginals类可以获得因子图中各个节点优化后的残差值
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

/* A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp) */
struct PointXYZIRPYT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                  // enforce SSE padding for correct memory alignment

// User defined point structures can be registered using PCL macros. http://wiki.ros.org/pcl_ros/cturtle
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

class mapOptimization : public ParamServer
{
 public:
  // gtsam
  NonlinearFactorGraph gtSAMgraph;
  Values initialEstimate, isamCurrentEstimate, optimizedEstimate;
  ISAM2 *isam;
  Eigen::MatrixXd poseCovariance;

  ros::Publisher pubLaserCloudSurround;
  ros::Publisher pubLaserOdometryGlobal;
  ros::Publisher pubLaserOdometryIncremental;
  ros::Publisher pubKeyPoses;
  ros::Publisher pubPath;

  ros::Publisher pubHistoryKeyFrames;
  ros::Publisher pubIcpKeyFrames; // key frame clouds near current position
  ros::Publisher pubRecentKeyFrames;
  ros::Publisher pubRecentKeyFrame;
  ros::Publisher pubCloudRegisteredRaw; // raw deskewed cloud
  ros::Publisher pubLoopConstraintEdge;

  ros::Subscriber subCloud;
  ros::Subscriber subGPS;
  ros::Subscriber subLoop; // 订阅自己从外部定义的回环

  ros::ServiceServer srvSaveMap;

  std::deque<nav_msgs::Odometry> gpsQueue;
  lio_sam::cloud_info cloudInfo;

  vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D; // xyz of key frame
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
  pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

  // corner feature set from odomOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
  // surf feature set from odomOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
  // downsampled corner featuer set from odomOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
  // downsampled surf featuer set from odomOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;

  pcl::PointCloud<PointType>::Ptr laserCloudOri; // feature point for map optimization
  pcl::PointCloud<PointType>::Ptr coeffSel;      // jacobi and residual for map optimization

  std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
  std::vector<PointType> coeffSelCornerVec;
  std::vector<bool> laserCloudOriCornerFlag;
  std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
  std::vector<PointType> coeffSelSurfVec;
  std::vector<bool> laserCloudOriSurfFlag;

  // we should transform key frame to global coordinate system and save them into local map for map optimization
  // this varible save transformed key frame that may be extracted to construct local map
  map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterICP; // 在回环检测中对局部地图下采样
  // for surrounding key poses of scan-to-map optimization
  pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;

  ros::Time timeLaserInfoStamp;
  double timeLaserInfoCur;

  // 将要用来优化的初始位姿变换; best pose after last map optimization; roll,pitch,yaw, x,y,z
  float transformTobeMapped[6];

  std::mutex mtx;
  std::mutex mtxLoopInfo;

  bool isDegenerate = false; // 判断点云是否退化
  cv::Mat matP;              // 作为临时使用的中间变量

  int laserCloudCornerFromMapDSNum = 0;
  int laserCloudSurfFromMapDSNum   = 0;
  int laserCloudCornerLastDSNum    = 0;
  int laserCloudSurfLastDSNum      = 0;

  // if add gps constraint or loop and these two constraints haven't used for optimization, will be "true"
  // "true" means need to update isam several times
  bool aLoopIsClosed = false;
  map<int, int> loopIndexContainer;      // 保存回环帧的序号from new to old
  vector<pair<int, int>> loopIndexQueue; // 回环帧的序号
  vector<gtsam::Pose3> loopPoseQueue;    // 保存回环帧之间的相对位姿
  vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
  deque<std_msgs::Float64MultiArray> loopInfoVec; // 自己从外部定义的回环

  nav_msgs::Path globalPath;

  Eigen::Affine3f transPointAssociateToMap;       // 把点变换进地图坐标系
  Eigen::Affine3f incrementalOdometryAffineFront; // last optimized pose, also used for initial guess
  Eigen::Affine3f incrementalOdometryAffineBack;  // current optimized pose

  mapOptimization()
  {
    ISAM2Params parameters; // 在定义ISAM2实例的时候存储参数的变量
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip      = 1;
    isam                            = new ISAM2(parameters);

    pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
    pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
    pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry", 1);
    pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 1);
    pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

    // 订阅分类好的cloud_info以及gps,用于后端优化和回环检测,
    // 注意gps接受的是nav_msgs::Odometry消息, 是通过robot_localization节点融合imu和gps数据得到的，这个过程没在这个程序里展示
    subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
    subGPS   = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
    // Float64 Multi Array; add loop manually to optimize
    subLoop = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

    // subscribe srv, which save map
    srvSaveMap = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);

    // 回环检测相关的一些历史帧
    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    pubIcpKeyFrames     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    // Marker Array???
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

    pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
    pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
    pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

    // 不同的特征进行滤波
    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    // for surrounding key poses of scan-to-map optimization
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity);

    allocateMemory();
  }

  void allocateMemory()
  {
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    // corner feature set from odomOptimization
    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
    // surf feature set from odomOptimization
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
    // downsampled corner featuer set from odomOptimization
    laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
    // downsampled surf featuer set from odomOptimization
    laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());

    laserCloudOri.reset(new pcl::PointCloud<PointType>());
    coeffSel.reset(new pcl::PointCloud<PointType>());

    laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i)
    {
      transformTobeMapped[i] = 0;
    }

    matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr &msgIn)
  {
    // extract time stamp
    timeLaserInfoStamp = msgIn->header.stamp;
    timeLaserInfoCur   = msgIn->header.stamp.toSec(); // rosTime to double

    // extract info and feature cloud
    cloudInfo = *msgIn; // copy data(not address) in "msgin" to "cloud info"
    pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
    pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

    std::lock_guard<std::mutex> lock(mtx);

    static double timeLastProcessing = -1; // time of last processed cloud
    // controll frequence of backend. adjust by calculation source
    if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
    {
      timeLastProcessing = timeLaserInfoCur;

      // map optimized pose + imu --> initial pose for map optimization
      updateInitialGuess();

      // as function name, construct local map(corner map and surf map)
      extractSurroundingKeyFrames();

      // 不同的leaf size进行下采样，主要是corner cloud和surface cloud
      downsampleCurrentScan();

      // 构建点到平面、点到直线的残差, 用高斯牛顿法进行优化
      // construct optimization function of point cloud registration and solve function
      scan2MapOptimization(); // scan to map registration

      // 添加factor,保存key pose之类的
      // judge whether is key frame according to registration result; update factor and optimal pose
      saveKeyFramesAndFactor();

      // correct global path/trajectory
      correctPoses();

      // publish lidar odom; 发布增量平滑后的odom
      publishOdometry();

      // publish visualized point cloud
      publishFrames();
    }
  }

  // as point cloud registration based on optimization, initial value is essential
  // (fast convergence, avoidance local solution)
  void updateInitialGuess()
  {
    // map optimized pose + imu --> initial pose for map optimization
    // 更新初始位姿, 来源可以是GPS ODOM, 也可以是上一帧的位姿, 存在transformTobeMapped中
    // save current transformation before any processing
    incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

    static Eigen::Affine3f lastImuTransformation;
    // initialization
    // if empty, means that no key frame(system was just initialized)
    if (cloudKeyPoses3D->points.empty())
    {
      // initial oriention is provided by magnetometer
      // 9-axis imu: accelerometer, gyroscope, magnetometer(oriention)
      transformTobeMapped[0] = cloudInfo.imuRollInit;
      transformTobeMapped[1] = cloudInfo.imuPitchInit;
      transformTobeMapped[2] = cloudInfo.imuYawInit;

      // for VIO or LIO (tight coupling with imu), 系统的不可观为4自由度(translation, yaw)
      // roll and pitch is 可观的 for imu(gravity)
      // 这里有磁力计将yaw对齐，但是也可以考虑不使用yaw, set yaw as zero
      if (!useImuHeadingInitialization)
      {
        transformTobeMapped[2] = 0;
      }
      // 获取初始的transform: save pose from magnetometer before return; translation is 0;
      lastImuTransformation = pcl::getTransformation(0, 0, 0,
                                                     cloudInfo.imuRollInit,
                                                     cloudInfo.imuPitchInit,
                                                     cloudInfo.imuYawInit);
      return;
    }

    // use imu pre-integration estimation for pose guess.
    // map optimization is necessary during imu preint, thus initialized system don't contain imu preint because no
    // map optimization is performed
    static bool lastImuPreTransAvailable = false; // whether recieve first frame of imupreint odom
    static Eigen::Affine3f lastImuPreTransformation;

    // odom可用的话, 使用imu odom作为初始位姿. 每帧imu在imuPreintexx.cpp中会实时进行预积分优化, 并存储其优化后的odom
    // if odom provided by imupreint is available
    if (cloudInfo.odomAvailable == true)
    {
      // convert imu preint from pcl to eigen; current imu preint
      Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
                                                         cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
      // if haven't recieved first frame of imupreint odom
      if (lastImuPreTransAvailable == false)
      {
        // save current preint odom data
        lastImuPreTransformation = transBack;
        lastImuPreTransAvailable = true;
      }
      else
      {
        // incre(delta pose) = current imu preint - last imu preint
        Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
        // pose by last map optimization
        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
        // final pose(initial pose) = last map optimized pose + incre pose
        Eigen::Affine3f transFinal = transTobe * transIncre;
        // convert from eigen to Euler Angle
        pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // now transformTobeMapped is initial guess

        // save current imu preint
        lastImuPreTransformation = transBack;
        // save 磁力计 orientatiion data in case odom is not available broken
        lastImuTransformation = pcl::getTransformation(0, 0, 0,
                                                       cloudInfo.imuRollInit,
                                                       cloudInfo.imuPitchInit,
                                                       cloudInfo.imuYawInit);
        return;
      }
    }

    // use imu incremental estimation for pose guess (only rotation)
    // if odom is not available, translation is unknown(set as zero) and update by only imu orientation(from 磁力计)
    if (cloudInfo.imuAvailable == true)
    {
      // same as mentioned above
      Eigen::Affine3f transBack  = pcl::getTransformation(0, 0, 0,
                                                          cloudInfo.imuRollInit,
                                                          cloudInfo.imuPitchInit,
                                                          cloudInfo.imuYawInit);
      Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
      Eigen::Affine3f transTobe  = trans2Affine3f(transformTobeMapped);
      Eigen::Affine3f transFinal = transTobe * transIncre;
      pcl::getTranslationAndEulerAngles(transFinal,
                                        transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
      // save imu before return (only rotation)
      lastImuTransformation = pcl::getTransformation(0, 0, 0,
                                                     cloudInfo.imuRollInit,
                                                     cloudInfo.imuPitchInit,
                                                     cloudInfo.imuYawInit);
      return;
    }
    // map optimization is based on k-nearest neighbor, so orentation is much more important than translation
    // so we use odom 6-DOF pose first. at least we need orientation
  }

  Eigen::Affine3f trans2Affine3f(float transformIn[])
  {
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
  }

  void extractSurroundingKeyFrames()
  {
    // if no key frame
    if (cloudKeyPoses3D->points.empty() == true)
    {
      return;
    }
    // 检测到了回环就提取回环帧,否则提取附近点云
    // 第一次进来loopClosureEnableFlag = false, 直接提取附近关键帧
    // if (loopClosureEnableFlag == true)
    // {
    //     extractForLoopClosure();
    // } else {
    //     extractNearby();
    // }

    extractNearby();
  }

  void extractForLoopClosure()
  {
    pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
    // 提取回环候选帧
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i)
    {
      if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
      {
        cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
      }
      else
      {
        break;
      }
    }
    extractCloud(cloudToExtract);
  }

  // 提取附近的点云帧, 包括corner和surface, cloudKeyPoses3D
  void extractNearby()
  {
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;     // save index of element from kd-tree
    std::vector<float> pointSearchSqDis; // save distance

    // extract all the nearby key poses and downsample them
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
    // extract key frame around last key frame(close in space) to construct local map
    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
    // save these key frames into cloud pointer; 满足要求的点云帧加到surroundingKeyPoses中
    for (int i = 0; i < (int)pointSearchInd.size(); ++i)
    {
      int id = pointSearchInd[i];
      surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }

    // in case key frame are too much
    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS); // here is not intensity
    // range-based for loop; traverse all points in key frame
    for (auto &pt : surroundingKeyPosesDS->points)
    {
      // find nearest key frame(original frame) of every downsized frame(pt)
      kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
      //  assign its intensity to pt. intensity is index of key frame
      pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
    }

    // also extract some latest key frames in case the robot rotates in one position(close in time)
    int numPoses = cloudKeyPoses3D->size();
    // traverse from last one to early one;
    for (int i = numPoses - 1; i >= 0; --i)
    {
      // 把10s内的关键帧也加到surroundingKeyPosesDS中; key frame generated in last 10s
      if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
      {
        surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
      }
      else
      {
        break;
      }
    }

    // construct local map with extracted key frame
    extractCloud(surroundingKeyPosesDS);
  }

  void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
  {
    // fuse the map
    laserCloudCornerFromMap->clear(); // save corner of local map
    laserCloudSurfFromMap->clear();   // save surf of local map
    for (int i = 0; i < (int)cloudToExtract->size(); ++i)
    {
      // some key frames is close to last key frame in time, ensure it's also close in space
      if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
      {
        continue;
      }

      // intensity is index of key frame, which this point locate
      int thisKeyInd = (int)cloudToExtract->points[i].intensity;
      // if this frame is in container, add this frame to map, else add it to container and to map
      if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())
      {
        // transformed cloud available; extract point from map into local map
        *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
        *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
      }
      else
      {
        // transformed cloud not available; transform cloud to global coordinate system(cloud info, 6D pose)
        pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        pcl::PointCloud<PointType> laserCloudSurfTemp   = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        *laserCloudCornerFromMap += laserCloudCornerTemp;
        *laserCloudSurfFromMap += laserCloudSurfTemp;
        // save transformed cloud; 角点和面点构成一个pair放入地图容器
        laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
      }
    } // endfor: have add all valid frames to map

    // Downsample the surrounding corner key frames (or map)
    downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
    // Downsample the surrounding surf key frames (or map)
    downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

    // clear map cache if too large
    if (laserCloudMapContainer.size() > 1000)
    {
      laserCloudMapContainer.clear();
    }
  }

  pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
  {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z,
                                                      transformIn->roll, transformIn->pitch, transformIn->yaw);

// use openmp parallel threads for fast process; use it when perform order in for loop make no difference
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
      const auto &pointFrom = cloudIn->points[i];
      // out = R * in + t
      cloudOut->points[i].x         = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
      cloudOut->points[i].y         = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
      cloudOut->points[i].z         = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
  }

  void downsampleCurrentScan()
  {
    // Downsample cloud from current scan
    laserCloudCornerLastDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(*laserCloudCornerLastDS);
    laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

    laserCloudSurfLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
  }

  void scan2MapOptimization()
  {
    // if no key frame, can't match
    if (cloudKeyPoses3D->points.empty())
    {
      return;
    }

    // there are enough feature points
    if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
    {
      // create kd-tree for corner and surf respectively(map)
      kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
      kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

      // iterative solution. use ceres to optimize in aloam, while this code and loam use GaussNewton
      for (int iterCount = 0; iterCount < 30; iterCount++)
      {
        laserCloudOri->clear();
        coeffSel->clear();

        // get two residual and two jacobi; 点到平面, 点到直线的残差, 这里写法还与aloam有点区别
        cornerOptimization();
        surfOptimization();

        // combine residuals and jacobis
        combineOptimizationCoeffs();

        // optimize: 高斯牛顿法迭代优化
        if (LMOptimization(iterCount) == true)
        {
          break;
        }
      }
      // now optimization has been performed

      transformUpdate();
    } // endif: enough feature points
    else
    {
      ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
    }
  }

  void cornerOptimization()
  {
    updatePointAssociateToMap();

    // 遍历点云, 构建点到直线的约束
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudCornerLastDSNum; i++)
    {                                      // traverse corner in this scan
      PointType pointOri, pointSel, coeff; // point original(in current lidar coordinate system)
                                           // point sel???
                                           // xyz is jacobi direction, intensity is residual
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      pointOri = laserCloudCornerLastDS->points[i]; // extract corner
      pointAssociateToMap(&pointOri, &pointSel);    // transform pose to global/map coordinate system with initial guess
      // find 5 nearest points from corner map (point, num, nearest points, distance)
      kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

      cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0)); // square size and be symmetrical
      cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0)); // save eigenvalues
      cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0)); // save eigenvectors

      // the farmost point of 5 nearest points is close means that
      // this point and its nearest points locate in same edge feature
      if (pointSearchSqDis[4] < 1.0)
      {                               // 1 meter
        float cx = 0, cy = 0, cz = 0; // coordinate of center
        // calculate average of coordinate
        for (int j = 0; j < 5; j++)
        {
          cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
          cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
          cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5; // center of gravity

        // calculate covariance matrix
        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0; // element of matrix
        for (int j = 0; j < 5; j++)
        {
          float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
          float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
          float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;
          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        // covariance matrix(symmetrical matrix)
        matA1.at<float>(0, 0) = a11;
        matA1.at<float>(0, 1) = a12;
        matA1.at<float>(0, 2) = a13;
        matA1.at<float>(1, 0) = a12;
        matA1.at<float>(1, 1) = a22;
        matA1.at<float>(1, 2) = a23;
        matA1.at<float>(2, 0) = a13;
        matA1.at<float>(2, 1) = a23;
        matA1.at<float>(2, 2) = a33;

        // 求协方差矩阵的特征值和特征向量; D1 is eigenvalues in the descending order; V1 is eigenvector
        cv::eigen(matA1, matD1, matV1);

        // check whether this point is corner; max eigenvalue is much bigger than other two, it's corner
        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
        {
          float x0 = pointSel.x; // define x0,y0,z0 is point O
          float y0 = pointSel.y;
          float z0 = pointSel.z;

          // direction of eigenvector corresponding to lagest eigenvalue = edge direction
          // get two point in this edge with center of 5 nearest points and edge direction
          float x1 = cx + 0.1 * matV1.at<float>(0, 0); // define x1,y1,z1 is point A
          float y1 = cy + 0.1 * matV1.at<float>(0, 1);
          float z1 = cz + 0.1 * matV1.at<float>(0, 2);
          float x2 = cx - 0.1 * matV1.at<float>(0, 0); // define x2,y2,z2 is point B
          float y2 = cy - 0.1 * matV1.at<float>(0, 1);
          float z2 = cz - 0.1 * matV1.at<float>(0, 2);
          // residual, vertial line direction from point to edge, jocabi direction)
          // define MO = cross multiplication of AO and BO; |MO|=a012
          float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));
          // distance between two points = |AB|
          float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
          // direction of vertial line from point O to edge AB = direction of cross multiplication of BA and OM
          // unit vector (vertial line) = direction of residual's gradient(decrease or increase???)
          float la  = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;
          float lb  = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;
          float lc  = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;
          float ld2 = a012 / l12; // residual:distance between point O and edge AB

          // Kernel function to reduce weight of residual(large residual-->small weight)
          float s = 1 - 0.9 * fabs(ld2); // s>0, because distance(this point,nearest point) < 1

          coeff.x         = s * la;
          coeff.y         = s * lb;
          coeff.z         = s * lc;
          coeff.intensity = s * ld2;

          // this is a valid constraint for map optimization; residual < 10cm
          if (s > 0.1)
          {
            laserCloudOriCornerVec[i]  = pointOri;
            coeffSelCornerVec[i]       = coeff; // xyz is jacobi direction, intensity is residual
            laserCloudOriCornerFlag[i] = true;  // this constraint is valid
          }
        }
      }
    }
  }

  // convert Euler angle to eigen
  void updatePointAssociateToMap()
  {
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
  }

  // transform pose to global/map coordinate system with initial guess
  void pointAssociateToMap(PointType const *const pi, PointType *const po)
  {
    // po = R * pi + t
    po->x         = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
    po->y         = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
    po->z         = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
    po->intensity = pi->intensity;
  }

  void surfOptimization()
  {
    updatePointAssociateToMap();
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudSurfLastDSNum; i++)
    {
      PointType pointOri, pointSel, coeff;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      // 寻找5个紧邻点, 计算其特征值和特征向量
      pointOri = laserCloudSurfLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

      Eigen::Matrix<float, 5, 3> matA0; // 5*3 存储5个紧邻点
      Eigen::Matrix<float, 5, 1> matB0;
      Eigen::Vector3f matX0;
      // plane formulation: Ax+By+Cz+1=0
      matA0.setZero();
      matB0.fill(-1);
      matX0.setZero();

      // the farmost point of 5 nearest points is valid
      if (pointSearchSqDis[4] < 1.0)
      {
        for (int j = 0; j < 5; j++)
        {
          matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
          matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
          matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
        }

        // solve superdefinite equation: AX = B; x is nomal vector of this plane
        // The solution (matrix inversion) method is QR factorization;
        // the system can be over-defined and/or the matrix src1 can be singular
        matX0 = matA0.colPivHouseholderQr().solve(matB0);

        // 法向量参数 ax+by+cz +d = 0
        float pa = matX0(0, 0);
        float pb = matX0(1, 0);
        float pc = matX0(2, 0);
        float pd = 1;

        // normalize x --> |x| = 1
        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        // 这里再次判断求解的方向向量和每个点相乘，最后结果是不是在误差范围内。
        bool planeValid = true;
        // if one point of 5 points is far from this plane, curvature of this plane is large and this is a invalid palne
        for (int j = 0; j < 5; j++)
        {
          // 5 point locate in the same plane

          if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x
                   + pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y
                   + pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd)
              > 0.2)
          {
            planeValid = false;
            break;
          }
        }

        if (planeValid) // 是有效的平面
        {
          // residual: distance between point and plane
          float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

          // how denonminator function??? |pointSel| is large, surf feature is more important than corner
          float s         = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
          coeff.x         = s * pa;
          coeff.y         = s * pb;
          coeff.z         = s * pc;
          coeff.intensity = s * pd2;

          // 误差在允许的范围内的话把这个点放到点云laserCloudOri中去，把对应的向量coeff放到coeffSel中
          if (s > 0.1)
          {
            laserCloudOriSurfVec[i]  = pointOri;
            coeffSelSurfVec[i]       = coeff;
            laserCloudOriSurfFlag[i] = true;
          }
        }
      }
    }
  }

  // combine corner and surf constraint; 把两类损失和协方差丢到laserCloudOri和coeffSel中, 后续进行联合优化
  void combineOptimizationCoeffs()
  {
    // combine corner coeffs
    for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
    {
      // "true" is valid constraint
      if (laserCloudOriCornerFlag[i] == true)
      {
        laserCloudOri->push_back(laserCloudOriCornerVec[i]); // point
        coeffSel->push_back(coeffSelCornerVec[i]);           // jacobi and residual
      }
    }
    // combine surf coeffs
    for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
    {
      if (laserCloudOriSurfFlag[i] == true)
      {
        laserCloudOri->push_back(laserCloudOriSurfVec[i]);
        coeffSel->push_back(coeffSelSurfVec[i]);
      }
    }
    // reset flag for next iteration
    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
  }

  bool LMOptimization(int iterCount)
  {
    // Gauss-Newton
    // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll
    // Y-X-Z in camera <=> Z-Y-X in lidar

    /** lidar to camera --> optimization --> camera to lidar ???**/

    // 高斯牛顿优化, 参考LOAM
    // lidar -> camera
    // calculate sin cos of rotation around xyz-axis
    float srx = sin(transformTobeMapped[1]); // y->x
    float crx = cos(transformTobeMapped[1]); // pitch in lidar frame
    float sry = sin(transformTobeMapped[2]); // z->y
    float cry = cos(transformTobeMapped[2]); // yaw in lidar frame
    float srz = sin(transformTobeMapped[0]); // x->z
    float crz = cos(transformTobeMapped[0]); // roll in lidar frame

    int laserCloudSelNum = laserCloudOri->size();
    // if constraints is few
    if (laserCloudSelNum < 50)
    {
      return false;
    }

    // 每个点的残差的旋转和平移的雅可比;旋转的雅可比需要求导,平移的雅可比为残差的方向
    cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0)); // A.transpose()
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));               // A.transpose() * A
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));  // 残差的大小
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));               // A.transpose() * B
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));                 // 步进的大小和方向

    PointType pointOri, coeff;

    for (int i = 0; i < laserCloudSelNum; i++)
    {
      // transform point and unit vector from point to line/surf to camera coordinate system
      // lidar -> camera
      pointOri.x      = laserCloudOri->points[i].y;
      pointOri.y      = laserCloudOri->points[i].z;
      pointOri.z      = laserCloudOri->points[i].x;
      coeff.x         = coeffSel->points[i].y;
      coeff.y         = coeffSel->points[i].z;
      coeff.z         = coeffSel->points[i].x;
      coeff.intensity = coeffSel->points[i].intensity;
      // in camera
      // calculate jacobi of rotation(rotate around yxz in camera <-> around zyx in lidar)
      // roll
      float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;
      // pitch
      float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;
      // yaw
      float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;
      // lidar -> camera
      matA.at<float>(i, 0) = arz; // jacobi of rotation
      matA.at<float>(i, 1) = arx;
      matA.at<float>(i, 2) = ary;
      matA.at<float>(i, 3) = coeff.z; // jacobi of translation
      matA.at<float>(i, 4) = coeff.x;
      matA.at<float>(i, 5) = coeff.y;
      matB.at<float>(i, 0) = -coeff.intensity; // residual
    }

    // JTJ * deltax = -JTe (J:Jacobi of e; e:residual, scalar quantity)
    // construct JTJ and -JTe matrix
    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    // QR factorization; the system can be over-defined and/or the matrix src1 can be singular
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR); // Ax=B->get x(deltax)

    // if it's the first iteration, check whether degenerated
    if (iterCount == 0)
    {
      cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

      cv::eigen(matAtA, matE, matV); // mat E: eigenvalue, max->min; mat V: eigenvector
      matV.copyTo(matV2);

      isDegenerate      = false;
      float eignThre[6] = {100, 100, 100, 100, 100, 100}; // eigenthresold
      // traverse eigen value from min to max
      for (int i = 5; i >= 0; i--)
      {
        // eigenvalue < eigenthresold,特征值太小,说明矩阵在相应的特征向量方向退化
        if (matE.at<float>(0, i) < eignThre[i])
        {
          for (int j = 0; j < 6; j++)
          {
            matV2.at<float>(i, j) = 0; // corresponding vector is set as zero
          }
          isDegenerate = true; // there is degenarated component
        }
        else
        {
          break;
        }
      }
      matP = matV.inv() * matV2; // eigenvector matrix without degenerated component
    }
    // for example, in a long corridor, cant estimate translation in direction of corridor,
    // thus, corresponding vector is degenerated and corresponding component of delta x should be zero

    // if degenerate, set degenerated component of delta x as zero
    if (isDegenerate)
    {
      cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
      matX.copyTo(matX2); // don't update
      matX = matP * matX2;
    }

    // update: x(k+1) = x(k) + delta x (for k-th iteration)
    transformTobeMapped[0] += matX.at<float>(0, 0);
    transformTobeMapped[1] += matX.at<float>(1, 0);
    transformTobeMapped[2] += matX.at<float>(2, 0);
    transformTobeMapped[3] += matX.at<float>(3, 0);
    transformTobeMapped[4] += matX.at<float>(4, 0);
    transformTobeMapped[5] += matX.at<float>(5, 0);

    float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) + pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) + pow(pcl::rad2deg(matX.at<float>(2, 0)), 2)); // unit is degree
    float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) + pow(matX.at<float>(4, 0) * 100, 2) + pow(matX.at<float>(5, 0) * 100, 2));                         // unit is cm

    if (deltaR < 0.05 && deltaT < 0.05)
    {
      return true; // pose has converged, end optomization
    }
    return false; // pose hasn't converged, keep optimizing
  }

  // weighted fuse transformTobeMapped and imu
  void transformUpdate()
  {
    // if state of 9-axis imu in global is available
    if (cloudInfo.imuAvailable == true)
    {
      // avoid that interpolation result is singular; pitch won't be close to 90°
      if (std::abs(cloudInfo.imuPitchInit) < 1.4)
      {
        // roll and pitch is observable from gravity, so weighted fuse pose from lidar estimated and magnetometer
        double imuWeight = imuRPYWeight;
        tf::Quaternion imuQuaternion;
        tf::Quaternion transformQuaternion;
        double rollMid, pitchMid, yawMid;

        // slerp: spherical linear interpolation: a linear smooth interpolation for quternion
        // slerp roll
        transformQuaternion.setRPY(transformTobeMapped[0], 0, 0); // convert to quternion
        imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
        // here:  final result = 0.99*lidar + 0.01*imu
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        transformTobeMapped[0] = rollMid; // interpolation result is final result

        // slerp pitch
        transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
        imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        transformTobeMapped[1] = pitchMid;
      }
    }

    // especially for indoor 2D situation; add constraints if 2D prior is known
    transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
    transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
    transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

    incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped); // convert to eigen
  }

  float constraintTransformation(float value, float limit)
  {
    if (value < -limit)
    {
      value = -limit;
    }
    if (value > limit)
    {
      value = limit;
    }

    return value;
  }

  void saveKeyFramesAndFactor()
  {
    // judge whether this frame is key frame by increment of translation and rotation
    if (saveFrame() == false)
    {
      return;
    }

    // if is key frame, add factor to isam
    addOdomFactor(); // odom factor

    addGPSFactor(); // gps factor

    addLoopFactor(); // loop factor

    // cout << "****************************************************" << endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // having added all factors, update iSAM graph model
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    // if add gps constraint or loop, isam need to optimize several times, because many vertices will be corrected
    if (aLoopIsClosed == true)
    {
      isam->update();
      isam->update();
      isam->update();
      isam->update();
      isam->update();
    }

    // clear constraint and vertice(they have been add to isam, so clearing them won't influence optimization)
    // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    // now optimization has been performed

    // save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    // optimized pose of newest key frame(index is "size-1")
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1); // 最新的pose
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");
    // 这里不断的增加关键帧到cloudKeyPoses3D、cloudKeyPoses6D中
    thisPose3D.x         = latestEstimate.translation().x();
    thisPose3D.y         = latestEstimate.translation().y();
    thisPose3D.z         = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x         = thisPose3D.x;
    thisPose6D.y         = thisPose3D.y;
    thisPose6D.z         = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
    thisPose6D.roll      = latestEstimate.rotation().roll();
    thisPose6D.pitch     = latestEstimate.rotation().pitch();
    thisPose6D.yaw       = latestEstimate.rotation().yaw();
    thisPose6D.time      = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    cout << isam->marginalCovariance(isamCurrentEstimate.size() - 1) << endl
         << endl;

    // 通过Marginals类可以获得各个节点优化后的残差值; save confidence of current pose
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);
    // print marginal covariances
    // cout << "x covariance:\n" << marginals.marginalCovariance(Symbol('x', 1)) << endl;

    // save updated transform as best/optimal estimation
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // save all the received edge and surf points
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    // copy cloud in current frame
    pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

    // save key frame cloud; space occupation will increase over time
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    // save path for visualization
    updatePath(thisPose6D);
  }

  bool saveFrame()
  {
    // if there isn't saved key frame, this frame is key frmae
    if (cloudKeyPoses3D->points.empty())
    {
      return true;
    }

    // last key frame
    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    // convert current pose to eigen
    Eigen::Affine3f transFinal   = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll) < surroundingkeyframeAddingAngleThreshold && abs(pitch) < surroundingkeyframeAddingAngleThreshold && abs(yaw) < surroundingkeyframeAddingAngleThreshold && sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
    {
      return false;
    }
    return true;
  }

  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
  {
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
  }

  void addOdomFactor()
  {
    // uunit: rad*rad, meter*meter;
    if (cloudKeyPoses3D->points.empty()) // if this frame is first key frame;
    {
      // low cofidence, especially unobservable translation and yaw
      // translation: we will use GPS data later, thus we set low confidence for translation
      // yaw: magnetometer may not be well calibrated
      // high confidence: pitch and roll calibrated by gravity
      noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());
      // add prior constraint to constraint 0th node(vertice)
      gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
      // add vertice info(0, initial value)
      initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped)); // "0" means "first" node
      // (index of node = index of key frame)
    }
    else // if not first key frame, add between factor
    {
      // high confidence: LO
      noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      gtsam::Pose3 poseFrom                          = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); // last key frame
      gtsam::Pose3 poseTo                            = trans2gtsamPose(transformTobeMapped);                 // current frame
      // between factor(index of last key frame(node), index of current(node), constraint, confidence)
      gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
      // add node info(index of node, initial value(prior pose))
      initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
  }

  gtsam::Pose3 trans2gtsamPose(float transformIn[])
  {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
  }

  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
  {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
  }

  void addGPSFactor()
  {
    if (gpsQueue.empty())
    {
      return;
    }

    // if no key frame, wait for system initialized and settles down
    if (cloudKeyPoses3D->points.empty())
    {
      return;
    }
    else
    {
      // if first key frame and last key frame are close, loop will be triggered (if loop, no need to use GPS)
      // or it means system just start
      if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
      {
        return;
      }
    }

    // pose covariance small(high confidence), no need to correct
    if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold)
    {
      return;
    }

    // last gps position
    static PointType lastGPSPoint;

    // pose的协方差比较大的时候才去添加gps factor
    while (!gpsQueue.empty())
    {
      // 时间戳对齐
      // if message too old
      if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
      {
        gpsQueue.pop_front();
      }
      // if message too new
      else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
      {
        break;
      }
      else // now gps time is close to current frame in time(-0.2 ~ 0.2)
      {
        nav_msgs::Odometry thisGPS = gpsQueue.front(); // get gps msg
        gpsQueue.pop_front();

        // GPS too noisy, skip
        float noise_x = thisGPS.pose.covariance[0];
        float noise_y = thisGPS.pose.covariance[7];
        float noise_z = thisGPS.pose.covariance[14];
        // if low confidence
        if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
        {
          continue;
        }

        // get position from gps msg
        float gps_x = thisGPS.pose.pose.position.x;
        float gps_y = thisGPS.pose.pose.position.y;
        float gps_z = thisGPS.pose.pose.position.z;
        // for gps, z usually is not as accurate as x and y, so we don't use z of gps
        // if gps z is accurate, we can use gps z to correct LO z(LO z is usually not accurate)
        if (!useGpsElevation)
        {
          gps_z   = transformTobeMapped[5]; // 此处假设gps的z不可信
          noise_z = 0.01;
        }

        // if x and y is too small, means that GPS not properly initialized (0,0,0)
        if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
        {
          continue;
        }

        // Add GPS every a few meters
        PointType curGPSPoint;
        curGPSPoint.x = gps_x;
        curGPSPoint.y = gps_y;
        curGPSPoint.z = gps_z;
        // add gps factor in a low frequence(interval > 5m)
        if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
        {
          continue;
        }
        else
        {
          lastGPSPoint = curGPSPoint;
        }

        gtsam::Vector Vector3(3);
        Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f); // gps factor noise
        noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
        // GPS factor in gstam (index/ID of vertice, position, noise)
        gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
        gtSAMgraph.add(gps_factor);
        // here we don't add value, because gps just provide constraint and
        // don't introduce varible that should be estimated

        // have added gps, need update isam several times
        aLoopIsClosed = true;
        break;
      } // endif: add this valid gps

    } // endwhile: have added all gps
  }

  void addLoopFactor()
  {
    // loop thread will detect loop and add loop to this queue
    if (loopIndexQueue.empty())
    {
      return;
    }
    // add all loops in queue
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
      int indexFrom = loopIndexQueue[i].first;  // current frame
      int indexTo   = loopIndexQueue[i].second; // prev frame
      // constraint between frames(between factor)
      gtsam::Pose3 poseBetween = loopPoseQueue[i];
      // confidence is ICP score
      gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
      // add constraint
      gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    // clear loop queue
    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
  }

  void updatePath(const PointTypePose &pose_in)
  {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp       = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id    = odometryFrame;
    pose_stamped.pose.position.x    = pose_in.x;
    pose_stamped.pose.position.y    = pose_in.y;
    pose_stamped.pose.position.z    = pose_in.z;
    tf::Quaternion q                = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
  }

  void correctPoses()
  {
    if (cloudKeyPoses3D->points.empty())
    {
      return;
    }

    // only loop and gps, which trigger global correction, can trigger to correct global trajectory
    if (aLoopIsClosed == true)
    {
      // clear map cache; many pose are corrected-->many cloud in map should be corrected, so we clear map container
      laserCloudMapContainer.clear();
      // clear path
      globalPath.poses.clear();
      // update key poses
      int numPoses = isamCurrentEstimate.size();

      // update all pose of key frames
      for (int i = 0; i < numPoses; ++i)
      {
        cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
        cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
        cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

        cloudKeyPoses6D->points[i].x     = cloudKeyPoses3D->points[i].x;
        cloudKeyPoses6D->points[i].y     = cloudKeyPoses3D->points[i].y;
        cloudKeyPoses6D->points[i].z     = cloudKeyPoses3D->points[i].z;
        cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
        cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
        cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

        updatePath(cloudKeyPoses6D->points[i]); // for visualization
      }

      aLoopIsClosed = false; // reset
    }
  }

  void publishOdometry()
  {
    // Publish odometry for ROS (global)
    nav_msgs::Odometry laserOdometryROS;
    laserOdometryROS.header.stamp          = timeLaserInfoStamp;
    laserOdometryROS.header.frame_id       = odometryFrame;
    laserOdometryROS.child_frame_id        = "odom_mapping";
    laserOdometryROS.pose.pose.position.x  = transformTobeMapped[3];
    laserOdometryROS.pose.pose.position.y  = transformTobeMapped[4];
    laserOdometryROS.pose.pose.position.z  = transformTobeMapped[5];
    laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0],
                                                                                     transformTobeMapped[1],
                                                                                     transformTobeMapped[2]);
    pubLaserOdometryGlobal.publish(laserOdometryROS);

    // Publish TF of lidar in odom coordinate
    static tf::TransformBroadcaster br;
    tf::Transform t_odom_to_lidar            = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                             tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
    tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
    br.sendTransform(trans_odom_to_lidar);

    // Publish odometry for ROS (incremental) (used for imu preint moduel to make odom smooth)
    static bool lastIncreOdomPubFlag = false;       // whether has data in odom, "false" means no data
    static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
    static Eigen::Affine3f increOdomAffine;         // incremental odometry in affine

    // when this if statement is first performed
    if (lastIncreOdomPubFlag == false)
    {
      lastIncreOdomPubFlag = true;
      laserOdomIncremental = laserOdometryROS;
      // when first perform, incremental is transformTobeMapped
      increOdomAffine = trans2Affine3f(transformTobeMapped);
    }
    else
    {
      // incre = current optimized pose - last optimized pose (incre is generated by scan-map match)
      // (pose is gotten by scan-map matching, not by gps or loop)
      // correction
      Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
      increOdomAffine             = increOdomAffine * affineIncre; // add incre to last pose
      float x, y, z, roll, pitch, yaw;
      pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);

      // if imu available, interpolate for roll and pitch
      if (cloudInfo.imuAvailable == true)
      {
        if (std::abs(cloudInfo.imuPitchInit) < 1.4)
        {
          double imuWeight = 0.1;
          tf::Quaternion imuQuaternion;
          tf::Quaternion transformQuaternion;
          double rollMid, pitchMid, yawMid;

          // slerp roll
          transformQuaternion.setRPY(roll, 0, 0);
          imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
          tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
          roll = rollMid;

          // slerp pitch
          transformQuaternion.setRPY(0, pitch, 0);
          imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
          tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
          pitch = pitchMid;
        }
      }
      laserOdomIncremental.header.stamp          = timeLaserInfoStamp;
      laserOdomIncremental.header.frame_id       = odometryFrame;
      laserOdomIncremental.child_frame_id        = "odom_mapping";
      laserOdomIncremental.pose.pose.position.x  = x;
      laserOdomIncremental.pose.pose.position.y  = y;
      laserOdomIncremental.pose.pose.position.z  = z;
      laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
      if (isDegenerate)
      {
        laserOdomIncremental.pose.covariance[0] = 1; // to mark whether odom is degenetated
      }
      else
      {
        laserOdomIncremental.pose.covariance[0] = 0;
      }
    } // endif: get incre odom
    pubLaserOdometryIncremental.publish(laserOdomIncremental);
  }

  void publishFrames()
  {
    if (cloudKeyPoses3D->points.empty())
    {
      return;
    }

    // publish key poses
    publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
    // Publish surrounding key frames
    publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
    // publish registered key frame
    if (pubRecentKeyFrame.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      // transform cloud to global coordinate
      *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
      *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
      publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish registered high-res raw cloud
    if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      *cloudOut                = *transformPointCloud(cloudOut, &thisPose6D);
      // publish original raw cloud
      publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish path
    if (pubPath.getNumSubscribers() != 0)
    {
      globalPath.header.stamp    = timeLaserInfoStamp;
      globalPath.header.frame_id = odometryFrame;
      pubPath.publish(globalPath);
    }
  }

  PointTypePose trans2PointTypePose(float transformIn[])
  {
    PointTypePose thisPose6D;
    thisPose6D.x     = transformIn[3];
    thisPose6D.y     = transformIn[4];
    thisPose6D.z     = transformIn[5];
    thisPose6D.roll  = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw   = transformIn[2];
    return thisPose6D;
  }

  void gpsHandler(const nav_msgs::Odometry::ConstPtr &gpsMsg)
  {
    gpsQueue.push_back(*gpsMsg);
  }

  void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr &loopMsg)
  {
    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    // whether invalid loop
    if (loopMsg->data.size() != 2)
    {
      return;
    }

    loopInfoVec.push_back(*loopMsg);

    // pop old loop
    while (loopInfoVec.size() > 5)
    {
      loopInfoVec.pop_front();
    }
  }

  bool saveMapService(lio_sam::save_mapRequest &req, lio_sam::save_mapResponse &res)
  {
    string saveMapDirectory;
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    if (req.destination.empty())
    {
      saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
    }
    else
    {
      saveMapDirectory = std::getenv("HOME") + req.destination;
    }
    cout << "Save destination: " << saveMapDirectory << endl;
    // create directory and remove old files;
    // 递归地删除该文件夹；c_str() 函数返回一个指向正规C字符串的指针常量, 内容与本 string 串相同
    int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
    // 若文件夹不存在则先创建文件夹，若文件夹存在则不做任何操作
    unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
    // save key frame transformations 为二进制文件
    pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
    pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
    // extract global point cloud map
    pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
    // 把关键帧转换进全局地图
    for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
    {
      *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
    }
    if (req.resolution != 0)
    {
      cout << "\n\nSave resolution: " << req.resolution << endl;

      // down-sample and save corner cloud
      downSizeFilterCorner.setInputCloud(globalCornerCloud);
      downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
      downSizeFilterCorner.filter(*globalCornerCloudDS);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
      // down-sample and save surf cloud
      downSizeFilterSurf.setInputCloud(globalSurfCloud);
      downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
      downSizeFilterSurf.filter(*globalSurfCloudDS);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
    }
    else
    {
      // save corner cloud
      pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
      // save surf cloud
      pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
    }

    // save global point cloud map
    *globalMapCloud += *globalCornerCloud;
    *globalMapCloud += *globalSurfCloud;

    int ret     = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
    res.success = ret == 0; //???

    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files completed\n"
         << endl;

    return true;
  }

  void loopClosureThread()
  {
    // if loop isn't enabled, exit this thread
    if (loopClosureEnableFlag == false)
    {
      return;
    }

    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
      // sleep for a time(depend on loop Frequency) after loop is performed once to reduce cpu occupation
      rate.sleep();
      // perform loop
      performLoopClosure();
      visualizeLoopClosure();
    }
  }

  // find and save loop; icp match; add constraint
  void performLoopClosure()
  {
    // no key frame, can't perform loop
    if (cloudKeyPoses3D->points.empty() == true)
    {
      return;
    }

    // pose3d and pose6d also are used in loop hander function, therefore copy key pose to avoid thread conflict
    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D; // (x,y,z)
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D; // (x,y,z,roll,pitch,yaw)
    mtx.unlock();

    // find keys
    int loopKeyCur; // index
    int loopKeyPre;

    // get external loop(added manually by ourselves) first
    // if(invalid loop or loop constraint has been added)  return; else have gotten loop frames, continue
    if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false) // false means no loop, found
    {
      if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false) // false means no loop, found
      {
        return;
      }
    }

    // get transformation between two loop frames by ICP
    // extract cloud
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    {
      // 0 means only itself
      loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
      // near key frames to construct local map
      loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
      // (points in this cloud are too few)
      if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
      {
        return;
      }
      if (pubHistoryKeyFrames.getNumSubscribers() != 0)
      {
        // publish local map to rviz
        publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
      }
    }

    // ICP Settings; use ICP to match frame and local map
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    // 设置对应点容忍最大距离
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6); // rotation
    icp.setRANSACIterations(0);           // 不进行RANSAC迭代

    // Align clouds: 将回环帧与local map进行匹配
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result); // perform ICP

    // inconverge or residual is too large; 通过icp score阈值判断是否匹配成功
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
    {
      return;
    }

    // publish corrected cloud to rviz
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation()); //???
      publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    // get transformation/correction(transform should be zero in loop, thus correction = ICP transform)
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose; correct by ICP transformation
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]); // transform before correction
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;               // pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); // affine to Euler

    // gtsam中添加回环的约束
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));         // corrected cur pose
    gtsam::Pose3 poseTo   = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]); // corrected prev pose
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore(); // set icp score as noise
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint between two frames
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre)); // index of two frames
    loopPoseQueue.push_back(poseFrom.between(poseTo));           // relative pose between loop
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    // add loop constriant
    loopIndexContainer[loopKeyCur] = loopKeyPre;
  }

  // if two frames are too far to loop but we think two frames can loop, we can add loop manually by ourselves ???
  bool detectLoopClosureExternal(int *latestID, int *closestID)
  {
    // this function is not used yet, please ignore it
    int loopKeyCur = -1;
    int loopKeyPre = -1;

    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    // queue for prior loop msg from outside
    if (loopInfoVec.empty())
    {
      return false;
    }

    // get loop info(time stamp)
    double loopTimeCur = loopInfoVec.front().data[0];
    double loopTimePre = loopInfoVec.front().data[1];
    loopInfoVec.pop_front();

    // too close in time
    if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
    {
      return false;
    }

    int cloudSize = copy_cloudKeyPoses6D->size();
    // if num of key frame is too small
    if (cloudSize < 2)
    {
      return false;
    }

    // latest key
    loopKeyCur = cloudSize - 1;
    // traverse all key frames to find key frame closest to loop Time Cur
    for (int i = cloudSize - 1; i >= 0; --i)
    {
      if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
      {
        loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity); // intensity的整数部分是序号
      }
      else
      {
        break;
      }
    }

    // previous key
    loopKeyPre = 0;
    // traverse all key frames to find key frame closest to loop Time Pre
    for (int i = 0; i < cloudSize; ++i)
    {
      if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
      {
        loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
      }
      else
      {
        break;
      }
    }

    // if same key frame
    if (loopKeyCur == loopKeyPre)
    {
      return false;
    }

    auto it = loopIndexContainer.find(loopKeyCur); // iterator
    // if (loop key cur has been saved in loop container)
    if (it != loopIndexContainer.end())
    {
      return false;
    }

    // index of 2 loop key frames
    *latestID  = loopKeyCur;
    *closestID = loopKeyPre;
    return true;
  }

  bool detectLoopClosureDistance(int *latestID, int *closestID)
  {
    // check whether latest key frame loop with other key frame
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
    {
      return false;
    }

    // find the closest history key frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
      int id = pointSearchIndLoop[i];
      //  check time difference: 两帧时间差也满足最小要求
      if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
      {
        loopKeyPre = id;
        break;
      }
    } // if stop too long, loop will be generated, too. we can use pose difference to check it

    // no loop
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
    {
      return false;
    }

    *latestID  = loopKeyCur;
    *closestID = loopKeyPre;
    return true;
  }

  // (near key frame cloud, index of current frame, num of near key frame)
  void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
  {
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
      int keyNear = key + i; // index of near frame
      if (keyNear < 0 || keyNear >= cloudSize)
      {
        continue;
      }
      // convert to global coordinate
      *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
      *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }
    // if no valid cloud/frame
    if (nearKeyframes->empty())
    {
      return;
    }

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
  }

  void visualizeLoopClosure()
  {
    if (loopIndexContainer.empty())
    {
      return;
    }

    visualization_msgs::MarkerArray markerArray;
    // loop nodes(add two frames in loop as node)
    // define info in node
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id    = odometryFrame;
    markerNode.header.stamp       = timeLaserInfoStamp;
    markerNode.action             = visualization_msgs::Marker::ADD;
    markerNode.type               = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns                 = "loop_nodes";
    markerNode.id                 = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x            = 0.3;
    markerNode.scale.y            = 0.3;
    markerNode.scale.z            = 0.3;
    markerNode.color.r            = 0;
    markerNode.color.g            = 0.8;
    markerNode.color.b            = 1;
    markerNode.color.a            = 1;
    // loop edges(add constraint between two frames)
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id    = odometryFrame;
    markerEdge.header.stamp       = timeLaserInfoStamp;
    markerEdge.action             = visualization_msgs::Marker::ADD;
    markerEdge.type               = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns                 = "loop_edges";
    markerEdge.id                 = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x            = 0.1;
    markerEdge.color.r            = 0.9;
    markerEdge.color.g            = 0.9;
    markerEdge.color.b            = 0;
    markerEdge.color.a            = 1;

    // traverse all loop
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
      int key_cur = it->first;
      int key_pre = it->second;
      geometry_msgs::Point p;
      p.x = copy_cloudKeyPoses6D->points[key_cur].x;
      p.y = copy_cloudKeyPoses6D->points[key_cur].y;
      p.z = copy_cloudKeyPoses6D->points[key_cur].z;
      // add node and edge
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
      p.x = copy_cloudKeyPoses6D->points[key_pre].x;
      p.y = copy_cloudKeyPoses6D->points[key_pre].y;
      p.z = copy_cloudKeyPoses6D->points[key_pre].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    // publish to visualize
    pubLoopConstraintEdge.publish(markerArray);
  }

  // global visualize thread
  void visualizeGlobalMapThread()
  {
    ros::Rate rate(0.2); // 按一定的频率发布全局地图
    while (ros::ok())
    {
      rate.sleep();
      publishGlobalMap();
    }

    // when ros is killed, save map；savePCD 再params.yaml文件中，选择是否保存地图
    if (savePCD == false)
    {
      return;
    }

    lio_sam::save_mapRequest req;
    lio_sam::save_mapResponse res;

    if (!saveMapService(req, res))
    {
      cout << "Fail to save map" << endl;
    }
  }

  // publish global map
  void publishGlobalMap()
  {
    // no subscriber, do not publish
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
    {
      return;
    }
    // if no key frame, no map; cloudKeyPoses3D 存的是关键帧的位姿
    if (cloudKeyPoses3D->points.empty() == true)
    {
      return;
    }

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>()); //???
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    // add all key frame to kdtree
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    // find key frame closest to current key frame
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    // save these key frames; 找到附近的点云帧并发布出来
    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
    {
      globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]); // frame
    }
    // downsample near selected key frames
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;                                                                                            // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

    // save index of downsampled key frames
    for (auto &pt : globalMapKeyPosesDS->points)
    {
      kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
      pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
    }

    // extract visualized and downsampled key frames that near to point in global map
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
    {
      if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
      {
        continue;
      }
      int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
      // transform pose of every frame to global coordinate
      *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points(transformed frames)
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "lio_sam");
  mapOptimization MO;
  ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
  // 两个线程，一边按固定的频率进行回环检测、添加约束边，另外一边进行地图发布和保存
  // loop is a independent thread from scan-match thread
  std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
  std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);
  ros::spin(); // 这里不间断的执行callback

  // join function will return only when its thread ends
  loopthread.join();
  visualizeMapThread.join();
  return 0;
}
