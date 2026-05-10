# Header
string frame_id
uint32 sequence_id
float64 timestamp

float64[6] desired_arm_joints_position
float64[6] desired_arm_joints_velocity

float64 desired_arm_gripper_position
float64 desired_arm_gripper_velocity

float64[6] arm_kp
float64[6] arm_kd
