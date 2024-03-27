/*
Description: [estimated=optimized, preint=preintegration, lo=lidarodometry, transform=transformation]
high frequence odom, provide initial guess for map optimization
1.class TransformFusion:
pub pose( =imu preint+global pose(generated by map optimization node, which include loop closure) )
2.class IMUPreintegration:
imuHandler to
odometryHandler to
Author     : Ji Qingshi
date       :
*/

#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

/*
when difference between pose optimized by loop closure and preint is too big,
if we use these two constraint, preint node can't work well.
so we use old pose in preint optimization and pub pose propagated based on pose optimized by loop closure
*/
class TransformFusion : public ParamServer
{
  /* what is this class for
  pose = imu preint + global pose(generated by map optimization node, which include loop closure)
  pub pose in high frequence
  */
public:
  std::mutex mtx;

  ros::Subscriber subImuOdometry;
  ros::Subscriber subLaserOdometry;

  ros::Publisher pubImuOdometry;
  ros::Publisher pubImuPath;

  Eigen::Affine3f lidarOdomAffine;
  Eigen::Affine3f imuOdomAffineFront;
  Eigen::Affine3f imuOdomAffineBack;

  tf::TransformListener tfListener;
  tf::StampedTransform lidar2Baselink;

  double lidarOdomTime = -1; // save lo time; "-1" means there is not lo data
  deque<nav_msgs::Odometry> imuOdomQueue;

  TransformFusion()
  {
    if (lidarFrame != baselinkFrame)
    { // usually baselinkFrame is body frame
      // If you get an error "Lookup would require extrapolation(外推法) into the past" while running,
      // you can try this alternative(供选择的) code to call the listener:
      try
      {
        // ros Time(0) means newest
        // (destination_frame, original_frame, time of transform, ros::Duration(3.0))
        tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
        // look up transform between lidar Frame and baselinkFrame
        // save transform into lidartoBaselink
        tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
      }
      catch (tf::TransformException ex)
      {
        ROS_ERROR("%s", ex.what());
      }
    }

    // subscribe global pose generated in map optimization node
    subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
    // subscribe incremental pose generated in preint node
    subImuOdometry = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, &TransformFusion::imuOdometryHandler, this, ros::TransportHints().tcpNoDelay());

    // publisher 发布融合后的imu path和预积分完成优化后预测的odom
    pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
    pubImuPath = nh.advertise<nav_msgs::Path>("lio_sam/imu/path", 1);
  }

  // save global pose optimized by loop closure
  void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
  {
    std::lock_guard<std::mutex> lock(mtx);
    lidarOdomAffine = odom2affine(*odomMsg);
    lidarOdomTime = odomMsg->header.stamp.toSec();
  }

  // transform ros odom msg to eigen data structure
  Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
  {
    double x, y, z, roll, pitch, yaw;
    x = odom.pose.pose.position.x;
    y = odom.pose.pose.position.y;
    z = odom.pose.pose.position.z;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    return pcl::getTransformation(x, y, z, roll, pitch, yaw);
  }

  void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
  {
    // static tf
    // 定义一个广播，相当于发布话题时定义一个发布器
    static tf::TransformBroadcaster tfMap2Odom;
    // 定义存放转换信息（平动，转动）的变量
    // odom frame and map frame is same in map construction
    static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
    // send static tf
    // (transform, ros::Time::now(), "map", ”base_link“);sendTransform函数有很多重载
    // (存储变换关系的变量,广播tf使用的时间戳,父坐标系的名称,子坐标系的名称)
    tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

    std::lock_guard<std::mutex> lock(mtx);

    imuOdomQueue.push_back(*odomMsg);

    // get latest odometry (at current IMU stamp)
    if (lidarOdomTime == -1)
      return;
    while (!imuOdomQueue.empty())
    { // pop old imu data earlier than lo msg
      if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
      {
        imuOdomQueue.pop_front();
      }
      else
        break;
    }

    // calculate latest incremental in imu Odom Queue
    Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
    Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
    Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
    Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);

    // publish latest odometry; convert to odom msg format
    nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
    laserOdometry.pose.pose.position.x = x;
    laserOdometry.pose.pose.position.y = y;
    laserOdometry.pose.pose.position.z = z;
    laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
    pubImuOdometry.publish(laserOdometry);

    // publish tf
    // update tf
    static tf::TransformBroadcaster tfOdom2BaseLink;
    tf::Transform tCur; // current tf
    tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
    if (lidarFrame != baselinkFrame)
      tCur = tCur * lidar2Baselink; // updated tf
    // update tf from odom to base link (relative transform, time, original frame, destination frame)
    tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
    tfOdom2BaseLink.sendTransform(odom_2_baselink);

    // publish IMU path
    static nav_msgs::Path imuPath;     // contain stamp, pose, frame???
    static double last_path_time = -1; // "-1" means no last path msg
    double imuTime = imuOdomQueue.back().header.stamp.toSec();
    if (imuTime - last_path_time > 0.1)
    { // ensure frequence no more than 10 hz
      last_path_time = imuTime;
      geometry_msgs::PoseStamped pose_stamped; //???
      pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
      pose_stamped.header.frame_id = odometryFrame;
      pose_stamped.pose = laserOdometry.pose.pose;
      imuPath.poses.push_back(pose_stamped); // push back latest pose

      // erase old imu path data that one second earlier than lo time
      while (!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
      {
        imuPath.poses.erase(imuPath.poses.begin());
      }

      // pub path, which is predicted value that output by visual imu preint node
      if (pubImuPath.getNumSubscribers() != 0)
      {
        imuPath.header.stamp = imuOdomQueue.back().header.stamp;
        imuPath.header.frame_id = odometryFrame;
        pubImuPath.publish(imuPath);
      }
    }
  }
};

class IMUPreintegration : public ParamServer
{
public:
  std::mutex mtx;

  ros::Subscriber subImu;
  ros::Subscriber subOdometry;
  ros::Publisher pubImuOdometry;

  bool systemInitialized = false;

  gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise; // confidence of initial velocity
  gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
  gtsam::Vector noiseModelBetweenBias; // covariance of bias between two frames

  gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_; // setting up the IMU integration for optimization
  gtsam::PreintegratedImuMeasurements *imuIntegratorImu_; // setting up the IMU integration for IMU message thread

  std::deque<sensor_msgs::Imu> imuQueOpt;
  std::deque<sensor_msgs::Imu> imuQueImu;

  gtsam::Pose3 prevPose_;                 // previous pose; 6 degree of fredom
  gtsam::Vector3 prevVel_;                // 3 degree of fredom
  gtsam::NavState prevState_;             // state of last key-frame
  gtsam::imuBias::ConstantBias prevBias_; // imu bias of last key-frame

  gtsam::NavState prevStateOdom;             // real-time state from imu odometry
  gtsam::imuBias::ConstantBias prevBiasOdom; // real-time bias from imu odometry

  bool doneFirstOpt = false; // whether there is optimization performed, "false" means no optimization
                             // set as "true" in odometry Handler after first optimization
  // last imu time, "-1" means it hasn't been assigned
  double lastImuT_imu = -1; // use in imu handler,"-1"说明这是初始化后第一个imu数据
  double lastImuT_opt = -1; // use in optimization

  gtsam::ISAM2 optimizer;
  gtsam::NonlinearFactorGraph graphFactors; // constraint; like g2o edge(观测) or ceres parameterblock
  gtsam::Values graphValues;                // a map from keys to values
                                            // save estimated varibles; like g2o::Vertex  or ceres cost function

  const double delta_t = 0; // lidar and imu may be asynchronous, this is time difference

  int key = 1; // times of optimization; num of saved data for optimization in factor graph

  // transform between imu and lidar
  gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
                                        gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
  gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
                                        gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

  IMUPreintegration()
  {
    // 业务逻辑都在callback里面写, 两个数据是耦合关系, imu通过激光odom给出优化后的预积分预测
    // odom根据预测的位姿优化、融合出新的odom
    subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
    // this topic is sent by back end map optimization node
    subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5, &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

    // publisher 发布融合后的 imu path 和预积分完成优化后预测的odom ???
    pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic + "_incremental", 2000); // odometry/imu_incremental

    // 下面是预积分使用到的gtsam的一些参数配置; imu gravity, Acc Noise, Gyr Noise is in param.yaml
    // define direction of gravity;
    boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
    // acc white noise in continuous
    p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);
    // gyro white noise in continuous
    p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);
    // error committed in integrating position from velocities
    p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);
    // assume zero initial bias
    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());

    // high confidence of initial pose
    priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
    // low confidence of initial velocity
    priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
    // high confidence of initial bias
    priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
    // covariance of noraml lidar odom
    correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
    // covariance of degenerated lidar odom
    correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
    // covariance of bias between two frames
    noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

    // setting up the IMU integration for IMU message thread
    imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
    // setting up the IMU integration for optimization
    imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
  }

  void imuHandler(const sensor_msgs::Imu::ConstPtr &imu_raw)
  {
    std::lock_guard<std::mutex> lock(mtx);

    // transform state from IMU to lidar frame
    sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

    imuQueOpt.push_back(thisImu); // for optimization
    imuQueImu.push_back(thisImu); // to update newest IMU state

    // 检查有没有执行过一次优化,这里需要先在odomhandler中优化一次后再进行该函数后续的工作
    if (doneFirstOpt == false)
      return; // return if no optimization is performed

    double imuTime = ROS_TIME(&thisImu);
    // 获得时间间隔, 第一次为1/500,之后是两次imuTime间的差;
    double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
    lastImuT_imu = imuTime;

    // integrate this single imu message: add every new imu data into preint
    imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                            gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z),
                                            dt);

    // predict odometry: predict newest state by this new imu data
    gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

    // publish odometry
    nav_msgs::Odometry odometry;
    odometry.header.stamp = thisImu.header.stamp;
    odometry.header.frame_id = odometryFrame;
    odometry.child_frame_id = "odom_imu";

    // transform pose from imu to ldiar frame
    gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
    gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

    // pose: transform gtsam format to ros odometry msg format
    odometry.pose.pose.position.x = lidarPose.translation().x();
    odometry.pose.pose.position.y = lidarPose.translation().y();
    odometry.pose.pose.position.z = lidarPose.translation().z();
    odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
    odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
    odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
    odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

    // 与子tf的相对velocity
    odometry.twist.twist.linear.x = currentState.velocity().x();
    odometry.twist.twist.linear.y = currentState.velocity().y();
    odometry.twist.twist.linear.z = currentState.velocity().z();
    odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
    odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
    odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
    pubImuOdometry.publish(odometry);
  }

  // subscribe incremental odometry sent by map optimization node(LO)
  void odometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
  {
    std::lock_guard<std::mutex> lock(mtx);

    double currentCorrectionTime = ROS_TIME(odomMsg); // get time stamp

    // make sure we have imu data to integrate
    // 两个回调函数是互有联系的, 在imu的回调里就强调要完成一次优化才往下执行
    if (imuQueOpt.empty())
      return;

    // get odometry pose(data); 从雷达odom中取出位姿数据
    float p_x = odomMsg->pose.pose.position.x; // translation
    float p_y = odomMsg->pose.pose.position.y;
    float p_z = odomMsg->pose.pose.position.z;
    float r_x = odomMsg->pose.pose.orientation.x; // rotation
    float r_y = odomMsg->pose.pose.orientation.y;
    float r_z = odomMsg->pose.pose.orientation.z;
    float r_w = odomMsg->pose.pose.orientation.w;
    // whether this odometry is degenerated; "true" means "degenerated"
    bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
    // convert pose to gstam format
    gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

    /*
      // correction pose jumped, reset imu pre-integration
      if (currentResetId != imuPreintegrationResetId)
      {
        resetParams();
        imuPreintegrationResetId = currentResetId;
      }
    */
    // 0. initialize system; 第一帧进来初始化系统
    if (systemInitialized == false)
    {
      resetOptimization(); // 重置优化参数

      // pop IMU message older than this odometry msg
      while (!imuQueOpt.empty())
      {
        if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
        {
          lastImuT_opt = ROS_TIME(&imuQueOpt.front());
          imuQueOpt.pop_front();
        }
        else
          break;
      }

      // initial pose
      // transform pose from lidar to imu(because always process in imu coordinate in close-loop algorithm)
      prevPose_ = lidarPose.compose(lidar2Imu); // 雷达odom转到imu系下
      // PriorFactor,包括了位姿、速度和bias;加入PriorFactor在图优化中基本都是必需的前提
      // set initial pose and confidence(noise); prior pose(constraint), which estimated state should be close to
      // (first pose to be estimated, prior constraint, confidence/weight)
      gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
      // add constraint as a factor
      graphFactors.add(priorPose);

      // initial velocity
      prevVel_ = gtsam::Vector3(0, 0, 0);
      gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
      graphFactors.add(priorVel);

      // initial bias
      prevBias_ = gtsam::imuBias::ConstantBias(); // default value is 0
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
      graphFactors.add(priorBias);
      // now we have added all constraint factors for initialization

      // add values; set state with initial value; 除了因子外, 还要有节点value
      graphValues.insert(X(0), prevPose_);
      graphValues.insert(V(0), prevVel_);
      graphValues.insert(B(0), prevBias_);
      // now we have added all state factors for initialization
      // optimize once (factors, initial values)
      optimizer.update(graphFactors, graphValues); // optimize in isam optimizer using constraint and state
      graphFactors.resize(0);                      // set saved constraints and state with 0
      graphValues.clear();

      // interface of preintegration, initialize with initial bias
      imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
      imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

      key = 1;
      systemInitialized = true;
      return;
    }

    // reset graph for speed, 保存最后的噪声值
    // when add too many factors into isam optimizer, clear optimizer to ensure that optimization isn't too slow
    if (key == 100)
    { // num of factors is too large
      // get updated noise/confidence before reset
      gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
      gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
      gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
      // reset graph
      resetOptimization();

      // 重置之后还有类似与初始化的过程 区别在于噪声值不同
      // add newest pose velocity bias and respective covariance matrix into factor graph
      // add pose
      gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
      graphFactors.add(priorPose);
      // add velocity
      gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
      graphFactors.add(priorVel);
      // add bias
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
      graphFactors.add(priorBias);
      // add values
      graphValues.insert(X(0), prevPose_);
      graphValues.insert(V(0), prevVel_);
      graphValues.insert(B(0), prevBias_);
      // optimize once, 相当于初始化
      optimizer.update(graphFactors, graphValues);
      graphFactors.resize(0);
      graphValues.clear();

      key = 1;
    }

    // 1. integrate imu data and optimize(这里才开始主要的优化流程)
    // integrate imu data between two optimizations
    while (!imuQueOpt.empty())
    {
      // pop and integrate imu data that is between two optimizations
      sensor_msgs::Imu *thisImu = &imuQueOpt.front(); // get imu msg
      double imuTime = ROS_TIME(thisImu);
      if (imuTime < currentCorrectionTime - delta_t)
      { // get imu data earlier than odom
        // time difference of two frames; (imu time - last imu time); if not initialized, dt is small value
        double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
        // 进行预积分得到新的状态值,注意用到的是imu数据的加速度和角速度
        // 作者要求的9轴imu数据中欧拉角在本程序中没有任何用到, 全在地图优化里用到的
        // call preintegratoin interface to process imu data
        imuIntegratorOpt_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z),
                                                dt);
        // record current imu time;
        lastImuT_opt = imuTime;
        imuQueOpt.pop_front();
      }
      else
        break;
    }
    // add imu factor to graph; 利用两帧之间的IMU数据完成了预积分后增加imu因子到因子图中
    // after preintegration, transform it to preintegration constraint
    const gtsam::PreintegratedImuMeasurements &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
    // preintegration constraint relative state between 2 neighbor frames
    gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
    graphFactors.add(imu_factor);
    // add imu bias between factor
    // use constant to constraint bias, because difference of bias in neighbor frame is small
    graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                                                                        gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
    // add pose factor; 对应于作者论文中的因子图结构,就是与imu因子一起的 Lidar odometry factor
    gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
    // prior state
    gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise); // good idea
    graphFactors.add(pose_factor);
    // insert predicted values
    // use last state and preintegration constraint to predict current state; 插入预测的值 即因子图中x0 x1 x2……节点
    gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
    // set predicted values  as initial values and put them into factor graph
    graphValues.insert(X(key), propState_.pose());
    graphValues.insert(V(key), propState_.v()); // reckon by imu
    graphValues.insert(B(key), prevBias_);      // reckon by imu
    // optimize
    optimizer.update(graphFactors, graphValues);
    // every optimization adjust state small.
    // by author'experience optimization should be performed 2 times
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();

    // Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = optimizer.calculateEstimate(); // get optimized current state
    prevPose_ = result.at<gtsam::Pose3>(X(key));
    prevVel_ = result.at<gtsam::Vector3>(V(key));
    // 用位姿和速度得到navstate类型的变量
    prevState_ = gtsam::NavState(prevPose_, prevVel_);
    prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));

    // Reset the optimization preintegration object.
    imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
    // check optimization: 检查是否有失败情况,如有则重置参数
    if (failureDetection(prevVel_, prevBias_))
    { // abnormal status
      resetParams();
      return;
    }

    // imuIntegrateImu在每次odom触发优化后立刻获取最新的bias,然后对imu测量值imuQueImu执行bias改变的状态重传播处理
    // 2. after optimization, re-propagate imu odometry preintegration
    /* LO msg is gotten from key-frame, so its frequence is low
       Because imu more frequent than LO, so we have some imu data later than newest LO msg
       Use optimized state and these imu data to reckon real-time state
       so that nav system can output state in frequence as IMU. */
    prevStateOdom = prevState_;
    prevBiasOdom = prevBias_;
    // first pop imu message older than current correction data(current LO data)
    double lastImuQT = -1; // last imu time, "-1" means it hasn't been assigned; use in re-propagate
    while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
    {
      lastImuQT = ROS_TIME(&imuQueImu.front());
      imuQueImu.pop_front();
    }
    // repropogate
    if (!imuQueImu.empty())
    { // there are new imu data
      // reset bias use the newly optimized bias
      imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
      // integrate imu message from the beginning of this optimization
      // integrate imu msg left over; 利用imuQueImu中的数据进行预积分,主要区别旧在于上一行的更新了bias
      for (int i = 0; i < (int)imuQueImu.size(); ++i)
      {
        sensor_msgs::Imu *thisImu = &imuQueImu[i];
        double imuTime = ROS_TIME(thisImu);
        double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);
        // this is LO handler, don't reckon real-time state, add newest imu data into LO
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z),
                                                dt);
        lastImuQT = imuTime;
      }
    }
    ++key;
    doneFirstOpt = true;
  }

  // gtsam 相关优化参数重置
  void resetOptimization()
  {
    // initialize gtsam
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    optimizer = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newGraphFactors;
    graphFactors = newGraphFactors; // why redefine a varible???to clear this map

    gtsam::Values NewGraphValues;
    graphValues = NewGraphValues;
  }

  void resetParams()
  { // reset system and need reinitialize system
    lastImuT_imu = -1;
    doneFirstOpt = false;
    systemInitialized = false;
  }

  // 检测预计分失败的函数, 即时爆出错误,重置积分器
  bool failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
  {
    // check estimated velocity
    Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
    if (vel.norm() > 30)
    { // 108KM/h
      ROS_WARN("Large velocity, reset IMU-preintegration!");
      return true;
    }

    // check IMU
    Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
    Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
    if (ba.norm() > 1.0 || bg.norm() > 1.0)
    { // 1 m/s; 1 rad/s
      ROS_WARN("Large bias, reset IMU-preintegration!");
      return true;
    }

    return false;
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "roboat_loam");
  IMUPreintegration ImuP;
  TransformFusion TF;
  ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();

  return 0;
}