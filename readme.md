### introduction
* 原代码位置：https://github.com/TixiaoShan/LIO-SAM#sample-datasets

* 论文名称：LIO SAM - Tightly coupled Lidar Inertial Odometry via Smoothing and Mapping

* 本文件所在目录为包，不是ros的工作空间

### helpful links
* LIOSAM代码简介：https://zhuanlan.zhihu.com/p/352039509
* LIO-SAM论文阅读: https://zhuanlan.zhihu.com/p/153394930
* [论文](https://github.com/blue-stone-j/papers/blob/main/2020%20LIO%20SAM%20-%20Tightly%20coupled%20Lidar%20Inertial%20Odometry%20via%20Smoothing%20and%20Mapping.pdf)


### 可能遇到的问题

##### 之字形或跳动
如果激光雷达和IMU数据格式与LIO-SAM的要求一致，则可能是由于激光雷达和IMU数据的时间戳不同步而导致此问题。

##### 上下跳跃
base_link上下跳跃，可能是IMU外参标定错误。 

##### 
系统的性能在很大程度上取决于IMU测量的质量。IMU数据速率越高，系统精度越好。建议使用至少提供200Hz输出频率的IMU。受限于gtsam的计算精度，imu频率不能过高，最好维持在500Hz以下。


### discription of files
##### imageProjection
receive cloud, deskew and project cloud to pseudo-image.

##### featureExtraction
extract features.

##### imuPreintegration

##### mapOptimization