# Description: This script is used to run a mujoco simulation in ROS2
import os

# Fail-safe: se non è stato scelto esplicitamente altro,
# ROS 2 comunica solamente sulla macchina locale.
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

print(
    "ROS 2 network mode:",
    "LOCALHOST" if os.environ["ROS_LOCALHOST_ONLY"] == "1" else "NETWORK",
)

# Fail-safe: se non è stato scelto esplicitamente altro,
# ROS 2 comunica solamente sulla macchina locale.
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

print(
    "ROS 2 network mode:",
    "LOCALHOST" if os.environ["ROS_LOCALHOST_ONLY"] == "1" else "NETWORK",
)

import sys
import shlex
import subprocess
from pathlib import Path

dir_path = Path(__file__).resolve().parent
sys.path.append(str(dir_path / ".."))

ros_ws = dir_path / "ros2_ws"
setup_bash = ros_ws / "install" / "setup.bash"

if not setup_bash.exists():
    print("Building the msgs first...")
    subprocess.run(["colcon", "build"], cwd=ros_ws, check=True)

if os.environ.get("LOCOMANIPULATION_TELEOP_ROS2_SOURCED") != "1":
    print("Sourcing ROS2 workspace and restarting script...")
    cmd = (
        f"source {shlex.quote(str(setup_bash))} && "
        "export LOCOMANIPULATION_TELEOP_ROS2_SOURCED=1 && "
        f"exec {shlex.quote(sys.executable)} "
        + " ".join(shlex.quote(arg) for arg in [str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    os.execv("/bin/bash", ["bash", "-c", cmd])

# Authors:
# Giulio Turrisi
import rclpy 
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from dls2_interface.msg import BaseState, BlindState, Imu, ControlSignal, ArmState, ArmTrajectoryGenerator, ArmControlSignal
from geometry_msgs.msg import PoseArray, Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy

import time
import numpy as np
import copy
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/mujoco_utils/")
sys.path.append(dir_path+"/../")

# Gym and Simulation related imports
import mujoco
import mujoco.viewer
import mujoco_utils
from heightmap import HeightMap

import config


class MujocoSimulationNode(Node):
    def __init__(self):
        super().__init__('Mujoco_Simulation_Node')

        self.simulation_dt = 0.002

        # Load the model and data.
        self.mjModel = mujoco.MjModel.from_xml_path(config.robot_model_path)
        self.mjData = mujoco.MjData(self.mjModel)
        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        self.mjData.qpos = self.mjModel.key_qpos[keyframe_id]
        self.viewer = mujoco.viewer.launch_passive(
                        self.mjModel,
                        self.mjData,
                        show_left_ui=False,
                        show_right_ui=False,
                        #key_callback=lambda x: self._key_callback(x),
                    )
        self.last_render_time = time.time()
        self.RENDER_FREQ = 30.0  # Hz
        self.HEIGHTMAP_FREQ = 15.0  # Hz
        self.step_num = 0

        # vision policies are the ones trained with the perceptive height scanner
        self.use_vision = config.training_env.get("perceptive_height_scanner") is not None
        if(self.use_vision):
            pattern_cfg = config.training_env["perceptive_height_scanner"]["pattern_cfg"]
            resolution_heightmap = pattern_cfg["resolution"]
            num_rows_heightmap = round(pattern_cfg["size"][0]/resolution_heightmap) + 1
            num_cols_heightmap = round(pattern_cfg["size"][1]/resolution_heightmap) + 1
            self.heightmap = HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=self.mjModel, mj_data=self.mjData)

        # Subscribers and Publishers
        self.publisher_base_state = self.create_publisher(BaseState,"/base_state", 1)
        self.publisher_blind_state = self.create_publisher(BlindState,"/blind_state", 1)
        self.publisher_imu = self.create_publisher(Imu,"/imu", 1)
        self.publisher_arm_blind_state = self.create_publisher(ArmState,"/arm_state", 1)
        self.publisher_heightmap = self.create_publisher(MarkerArray,"/height_scan_markers", 1)
        self.subscriber_trajectory_generator_arm = self.create_subscription(ArmTrajectoryGenerator,"/arm_trajectory_generator", self.get_arm_trajectory_generator_callback, 1)
        self.subscriber_control_signal_legs = self.create_subscription(ControlSignal,"/control_signal_legged", self.get_legs_control_signal_callback, 1)

        self.timer = self.create_timer(self.simulation_dt, self.compute_simulator_step_callback)

        # Desired PD
        qpos, qvel = self.mjData.qpos, self.mjData.qvel
        self.desired_arm_joints_position = copy.deepcopy(qpos[19:25])
        self.desired_legs_joints_position = copy.deepcopy(qpos[7:19])
        self.desired_gripper_position = copy.deepcopy(qpos[25])
        self.Kp_legs = 0
        self.Kd_legs = 0
        self.Kp_arm = 0
        self.Kd_arm = 0
        self.Kp_gripper = 0.0
        self.Kd_gripper = 0.0


    def get_arm_trajectory_generator_callback(self, msg):

        joints_position = np.array(msg.desired_arm_joints_position)
        self.desired_arm_joints_position = joints_position
        self.Kp_arm = np.array(msg.arm_kp)[0]
        self.Kd_arm = np.array(msg.arm_kd)[0]


    def get_legs_control_signal_callback(self, msg):
        
        # Desired leg joints position
        joints_position = np.array(msg.joints_position)
        self.desired_legs_joints_position = joints_position
        self.Kp_legs = np.array(msg.kp)[0]
        self.Kd_legs = np.array(msg.kd)[0]


    def compute_simulator_step_callback(self):
        
        # Get the current state of the robot -----------------------------------------------------
        qpos, qvel = self.mjData.qpos, self.mjData.qvel
        base_lin_vel = mujoco_utils.base_lin_vel(self.mjData, frame='world')
        base_ang_vel = mujoco_utils.base_ang_vel(self.mjData, frame='base')
        base_pos = mujoco_utils.base_pos(self.mjData)
        self.arm_joints_position = qpos[19:25]

        joints_pos_leg = qpos[7:19]
        joints_pos_arm = qpos[19:25]
        joints_pos_gripper = qpos[25]

        joints_vel_leg = qvel[6:18]
        joints_vel_arm = qvel[18:24]
        joints_vel_gripper = qvel[24]

        # Compute the PD torques ---------------------------------------------------------------
        temp_desired_legs_joints_position = copy.deepcopy(self.desired_legs_joints_position)
        
        error_joints_pos_leg = temp_desired_legs_joints_position - joints_pos_leg
        tau_leg = self.Kp_legs*error_joints_pos_leg - self.Kd_legs*joints_vel_leg

        error_joints_pos_arm = self.desired_arm_joints_position - joints_pos_arm
        tau_arm = self.Kp_arm*error_joints_pos_arm - self.Kd_arm*joints_vel_arm

        error_gripper_pos = self.desired_gripper_position - joints_pos_gripper
        tau_gripper = self.Kp_gripper*error_gripper_pos - self.Kd_gripper*joints_vel_gripper


        # Apply the torques and step the simulation ------------------------------------------------
        self.mjData.ctrl[0:12] = tau_leg
        self.mjData.ctrl[12:18] = tau_arm
        self.mjData.ctrl[18] = tau_gripper
        mujoco.mj_step(self.mjModel, self.mjData)
        self.step_num += 1


        # Publish Height Map (same convention as basic-locomotion run_simulator_ros2.py) ----------
        if self.use_vision and self.step_num % round((1.0/self.simulation_dt)/self.HEIGHTMAP_FREQ) == 0:
            base_ori_euler_xyz = mujoco_utils.base_ori_euler_xyz(self.mjData)
            heading_orientation_SO3 = mujoco_utils.heading_orientation_SO3(self.mjData)
            offset_world_frame = config.training_env["perceptive_height_scanner"]["offset"]["pos"] @ heading_orientation_SO3.T
            self.heightmap.update_height_map(self.mjData.qpos[0:3] + offset_world_frame, yaw=base_ori_euler_xyz[2])

            # Markers are published in the base frame
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, self.mjData.qpos[3:7])
            heightmap_local = (
                (self.heightmap.data.reshape(-1, 3) - self.mjData.qpos[0:3])
                @ rotation.reshape(3, 3)
            ).reshape(self.heightmap.num_rows, self.heightmap.num_cols, 3)

            # Inverse of the ordering conversion in run_controller_ros2.get_heightmap_callback.
            heightmap_publisher_order = np.flip(heightmap_local, axis=(0, 1)).transpose(1, 0, 2).reshape(-1, 3)
            heightmap_msg = MarkerArray()
            marker_stamp = self.get_clock().now().to_msg()
            for marker_id, point in enumerate(heightmap_publisher_order):
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = marker_stamp
                marker.ns = "height_scan"
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = point[2]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                marker.color.g = 1.0
                marker.color.a = 0.5
                heightmap_msg.markers.append(marker)
            self.publisher_heightmap.publish(heightmap_msg)


        # Publish the state of the robot ----------------------------------------------------------
        base_state_msg = BaseState()
        base_state_msg.pose.position = base_pos
        base_state_msg.pose.orientation = np.roll(self.mjData.qpos[3:7],-1)
        base_state_msg.velocity.linear = base_lin_vel
        base_state_msg.velocity.angular = base_ang_vel
        self.publisher_base_state.publish(base_state_msg)

        blind_state_msg = BlindState()
        blind_state_msg.joints_position = copy.deepcopy(self.mjData.qpos[7:19]).tolist()
        blind_state_msg.joints_velocity = copy.deepcopy(self.mjData.qvel[6:18]).tolist()
        self.publisher_blind_state.publish(blind_state_msg)

        imu_msg = Imu()
        imu_msg.linear_acceleration = self.mjData.sensordata[0:3]
        imu_msg.angular_velocity = self.mjData.sensordata[3:6]
        # For the quaternion, the order is [w, x, y, z] on mujoco, and [x, y, z, w] on DLS2
        imu_msg.orientation = np.roll(np.array(self.mjData.sensordata[9:13]), -1)
        self.publisher_imu.publish(imu_msg)

        arm_blind_state_msg = ArmState()
        arm_blind_state_msg.joints_position = self.mjData.qpos[19:25].tolist()
        arm_blind_state_msg.joints_velocity = self.mjData.qvel[18:24].tolist()
        self.publisher_arm_blind_state.publish(arm_blind_state_msg)


        # Render only at a certain frequency -----------------------------------------------------------------
        if time.time() - self.last_render_time > 1.0 / self.RENDER_FREQ:
            # Update the camera position
            self.viewer.cam.lookat[:] = base_pos

            if self.use_vision and self.heightmap.data is not None:
                mujoco_utils.render_spheres(
                    viewer=self.viewer,
                    positions=self.heightmap.data,
                    geom_ids=self.heightmap.geom_ids,
                    diameter=0.02,
                    color=[0, 1, 0, 0.5],
                )

            self.viewer.sync()
            self.last_render_time = time.time()




def create_executor():
    # EventsExecutor dispatches the callbacks in C++ (rclpy >= Kilted), with much less overhead than the
    # default SingleThreadedExecutor. It does not exist on Humble, where we fall back to SingleThreadedExecutor
    try:
        from rclpy.experimental import EventsExecutor
        print("Using EventsExecutor")
        return EventsExecutor()
    except ImportError:
        from rclpy.executors import SingleThreadedExecutor
        print("EventsExecutor not available in this ROS 2 distro, using SingleThreadedExecutor")
        return SingleThreadedExecutor()


#---------------------------
if __name__ == '__main__':
    
    print('Hello from the mujoco simulation node.')
    
    rclpy.init()
    mujoco_simulation_node = MujocoSimulationNode()
    rclpy.spin(mujoco_simulation_node, create_executor())

    mujoco_simulation_node.destroy_node()
    rclpy.shutdown()

    print("mujoco simulation node is stopped")
    exit(0)