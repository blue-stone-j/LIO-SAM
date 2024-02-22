### introduction
* 原代码位置：https://github.com/TixiaoShan/LIO-SAM#sample-datasets

* 论文名称：LIO SAM - Tightly coupled Lidar Inertial Odometry via Smoothing and Mapping

* 本文件所在目录为包，不是ros的工作空间

### helpful links
* LIOSAM代码简介：https://zhuanlan.zhihu.com/p/352039509
* LIO-SAM论文阅读: https://zhuanlan.zhihu.com/p/153394930


### 可能遇到的问题

##### 之字形或跳动
如果激光雷达和IMU数据格式与LIO-SAM的要求一致，则可能是由于激光雷达和IMU数据的时间戳不同步而导致此问题。

##### 上下跳跃
base_link上下跳跃，可能是IMU外参标定错误。 

##### 
系统的性能在很大程度上取决于IMU测量的质量。IMU数据速率越高，系统精度越好。建议使用至少提供200Hz输出频率的IMU。