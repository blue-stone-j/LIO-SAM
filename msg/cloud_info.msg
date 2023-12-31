# Cloud Info
Header header

int32[] startRingIndex # a horizontal scan is a ring
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 

int64 imuAvailable  # whether imu can provide orientation
int64 odomAvailable # whether imu preint odom can provide initial pose of this cloud frame

# Attitude for LOAM initialization
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration and use in map optimization
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature