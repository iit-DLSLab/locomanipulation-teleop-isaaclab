# Description: This script is used to run a mujoco simulation in ROS2

# Authors:
# Giulio Turrisi
import rclpy 
from rclpy.node import Node 
from dls2_interface.msg import BaseState, BlindState, TrajectoryGenerator, ArmState, ArmTrajectoryGenerator, ArmControlSignal
from geometry_msgs.msg import PoseArray, Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy

import time
import numpy as np
import copy
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/mujoco/")
sys.path.append(dir_path+"/../")

# Gym and Simulation related imports
import mujoco
import mujoco.viewer
import mujoco_utils
from heightmap import HeightMap


class MujocoSimulationNode(Node):
    def __init__(self):
        super().__init__('Mujoco_Simulation_Node')

        self.simulation_dt = 0.002

        # Load the model and data.
        self.mjModel = mujoco.MjModel.from_xml_path(dir_path+"/mujoco/models/scene_flat.xml")
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

        # Subscribers and Publishers
        self.publisher_base_state = self.create_publisher(BaseState,"/base_state", 1)
        self.publisher_blind_state = self.create_publisher(BlindState,"/blind_state", 1)
        self.publisher_arm_blind_state = self.create_publisher(ArmState,"/arm_state", 1)
        self.subscriber_trajectory_generator_arm = self.create_subscription(ArmTrajectoryGenerator,"/arm_trajectory_generator", self.get_arm_trajectory_generator_callback, 1)
        self.subscriber_trajectory_generator_legs = self.create_subscription(TrajectoryGenerator,"/trajectory_generator", self.get_legs_trajectory_generator_callback, 1)

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


    def get_legs_trajectory_generator_callback(self, msg):
        
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

        arm_blind_state_msg = ArmState()
        arm_blind_state_msg.joints_position = self.mjData.qpos[19:25].tolist()
        arm_blind_state_msg.joints_velocity = self.mjData.qvel[18:24].tolist()
        self.publisher_arm_blind_state.publish(arm_blind_state_msg)


        # Render only at a certain frequency -----------------------------------------------------------------
        if time.time() - self.last_render_time > 1.0 / self.RENDER_FREQ:
            # Update the camera position
            self.viewer.cam.lookat[:] = base_pos
            self.viewer.sync()
            self.last_render_time = time.time()




#---------------------------
if __name__ == '__main__':
    
    print('Hello from the mujoco simulation node.')
    
    rclpy.init()
    mujoco_simulation_node = MujocoSimulationNode()
    rclpy.spin(mujoco_simulation_node)

    mujoco_simulation_node.destroy_node()
    rclpy.shutdown()

    print("mujoco simulation node is stopped")
    exit(0)