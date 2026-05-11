# Description: This script is used to run the policy on the real robot

import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Joy
from dls2_interface.msg import BaseState, BlindState, TrajectoryGenerator, ArmState, ArmTrajectoryGenerator, ArmControlSignal
from geometry_msgs.msg import PoseArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor


import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation as R


import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/mujoco/")
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")


# Simulation related imports
import mujoco
import mujoco.viewer
import mujoco_utils
from heightmap import HeightMap


# Controller imports
from locomotion_policy_wrapper import LocomotionPolicyWrapper
from ik_mink import IKMink


import config
import threading


# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py


class TrashControlNode(Node):
    def __init__(self):
        super().__init__('Trash_Control_Node')

        self.simulation_dt = 0.002

        # Load the model and data.
        self.mjModel = mujoco.MjModel.from_xml_path(dir_path+"/mujoco/models/scene_rough.xml")
        self.mjData = mujoco.MjData(self.mjModel)
        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.mjData.qpos = self.mjModel.key_qpos[keyframe_id]
        
        self.use_detection_visualizer = False
        if self.use_detection_visualizer:
            self.visualizer_model = mujoco.MjModel.from_xml_path(dir_path+"/mujoco/models/z1/scene_floating.xml")
            self.visualizer_data = mujoco.MjData(self.visualizer_model)
            self.viewer = mujoco.viewer.launch_passive(
                            self.visualizer_model,
                            self.visualizer_data,
                            show_left_ui=False,
                            show_right_ui=False,
                        )
            self.last_render_time = time.time()
            self.RENDER_FREQ = 2.0  # Hz 


        # Initialization of variables used in the main control loop --------------------------------
        self.locomotion_policy = LocomotionPolicyWrapper(mjModel=self.mjModel)
        self.ik_mink_solver = IKMink()

        if(self.locomotion_policy.use_vision):
            resolution_heightmap = config.resolution_heightmap
            num_rows_heightmap = round(config.size_x_heightmap/resolution_heightmap) + 1
            num_cols_heightmap = round(config.size_y_heightmap/resolution_heightmap) + 1
            self.heightmap = HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=mjModel, mj_data=mjData) 

        self.arm_joints_position = np.zeros(6)  # 6 arm joints 
        self.arm_joints_velocity = np.zeros(6)  # 6 arm joints
        self.gripper_joint_position = 0
        self.gripper_joint_velocity = 0
        self.legs_joints_position = np.zeros(12)  # 12 leg joints
        self.legs_joints_velocity = np.zeros(12)  # 12 leg joints 
        self.desired_joint_pos_arm = np.zeros(6)
        self.desired_joint_pos_gripper = 0
        self.desired_joint_pos_leg = self.mjData.qpos[7:19]
        self.desired_pose_command = np.zeros(2)
        self.Kp_legs = config.Kp_stand_up_and_down
        self.Kd_legs = config.Kp_stand_up_and_down
        self.Kp_arm = config.Kp_arm
        self.Kd_arm = config.Kd_arm
        self.Kp_gripper = config.Kp_gripper
        self.Kd_gripper = config.Kd_gripper

        # --------------------------------------------------------------
        self.ref_base_lin_vel_H = np.array([0.0, 0.0, 0.0])  # Desired base linear velocity in the horizontal plane (x, y, z)
        self.ref_base_ang_yaw_dot = 0.0  # Desired base angular velocity around the vertical axis
        self.ref_ee_lin_vel = np.array([0.0, 0.0, 0.0])
        self.ref_ee_angular_vel = np.array([1.0, 0.0, 0.0, 0.0])
        self.ref_ee_lin_pos = np.array([0.0, 0.0, 0.0])

        # Interactive Command Line
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

        #self.console.isDown = True  # Only in this play_mujoco script
        #self.console.isRLActivated = False  # Only in this play_mujoco script

        # --------------------------------------------------------------
        # Subscribers and Publishers
        # callback "mutually exclusive"
        self.cb_control = MutuallyExclusiveCallbackGroup()

        # callback "mutually exclusive"
        self.cb_inputs = MutuallyExclusiveCallbackGroup()
        self.cb_joy = MutuallyExclusiveCallbackGroup()

        self.subscription_base_state = self.create_subscription(BaseState,"/base_state", self.get_base_state_callback, 1, callback_group=self.cb_inputs)
        self.subscription_blind_state = self.create_subscription(BlindState,"/blind_state", self.get_blind_state_callback, 1, callback_group=self.cb_inputs)
        self.subscription_arm_blind_state = self.create_subscription(ArmState,"/arm_state", self.get_arm_blind_state_callback, 1, callback_group=self.cb_inputs)

        self.subscription_joy = self.create_subscription(Joy,"/joy", self.get_joy_callback, 1, callback_group=self.cb_joy)
        
        
        self.publisher_trajectory_generator = self.create_publisher(TrajectoryGenerator,"/trajectory_generator", 1, callback_group=self.cb_inputs)
        self.publisher_arm_trajectory_generator = self.create_publisher(ArmTrajectoryGenerator,"/arm_trajectory_generator", 1, callback_group=self.cb_inputs)
        self.publisher_arm_control_signal = self.create_publisher(ArmControlSignal,"/arm_control_signal", 1, callback_group=self.cb_inputs)
        
        RL_FREQ = 1./(config.training_locomotion_env["sim"]["dt"]*config.training_locomotion_env["decimation"])  # Hz, frequency of the RL controller
        self.timer = self.create_timer(1.0/RL_FREQ, self.compute_rl_control, callback_group=self.cb_control)


        # Safety check to not do anything until a first base and blind state are received
        self.first_message_base_arrived = False
        self.first_message_legs_joints_arrived = False
        self.first_message_arm_joints_arrived = True
        self.last_joy_time = None

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        self.old_buttons = np.zeros(11)

    
    def get_joy_callback(self, msg):
        """
        Callback function to handle joystick input. Joystick used is a 
        8Bitdo Ultimate 2C Wireless Controller.
        """
        if(self.console.isArmJoystickActivated):
            filter_joystick = 0.7
            self.ref_ee_lin_vel[0] = self.ref_ee_lin_vel[0]*filter_joystick + (msg.axes[4]/20)*(1-filter_joystick)  # Forward/Backward
            self.ref_ee_lin_vel[1] = self.ref_ee_lin_vel[1]*filter_joystick + (msg.axes[0]/20)*(1-filter_joystick)  # Left/Right
            self.ref_ee_lin_vel[2] = self.ref_ee_lin_vel[2]*filter_joystick + (msg.axes[1]/20)*(1-filter_joystick)  # Up/Down

            self.ref_ee_lin_pos = self.ref_ee_lin_pos + self.ref_ee_lin_vel * 0.05
        else:
            filter_joystick = 0.7
            self.ref_base_lin_vel_H[0] = self.ref_base_lin_vel_H[0]*filter_joystick + (msg.axes[1]/3.5)*(1-filter_joystick)  # Forward/Backward
            self.ref_base_lin_vel_H[1] = self.ref_base_lin_vel_H[1]*filter_joystick + (msg.axes[0]/3.5)*(1-filter_joystick)  # Left/Right
            self.ref_base_ang_yaw_dot = self.ref_base_ang_yaw_dot*filter_joystick + (msg.axes[3]/2.)*(1-filter_joystick)  # Yaw

        self.last_joy_time = time.time()


        #kill the node if the button is pressed
        if msg.buttons[8] == 1:
            self.get_logger().info("Joystick button pressed, shutting down the node.") 
            # This will kill the robot hal
            os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o \"^[^ ]* *[0-9]*\" | grep -o \"[0-9]*\")")
            # This will kill the process running this script
            os.system("pkill -f run_controller_ros2.py") 
            exit(0)
        elif self.old_buttons[7] == 0 and msg.buttons[7] == 1:
            print("Locomotion activation")
            self.console.isRLActivated = not self.console.isRLActivated
            self.old_buttons[7] = 1
        elif(msg.axes[7] == 1.0):
            print("goUp")
            self.console.goUp()
        elif(msg.axes[7] == -1.0):
            print("goDown")
            self.console.goDown()

        elif(self.old_buttons[6] == 0 and msg.buttons[6] == 1):
            print("activateArm")
            self.console.isArmActivated = not self.console.isArmActivated
            self.old_buttons[6] = 1

        elif(self.old_buttons[0] == 0 and msg.buttons[0] == 1):
            print("Arm only Joystick")
            self.console.isArmJoystickActivated = not self.console.isArmJoystickActivated
            self.old_buttons[0] = 1





    def get_base_state_callback(self, msg):
        self.position = np.array(msg.pose.position) #world frame
        # For the quaternion, the order is [w, x, y, z] on mujoco, and [x, y, z, w] on DLS2
        self.orientation = np.roll(np.array(msg.pose.orientation), 1) #world frame
        self.linear_velocity = np.array(msg.velocity.linear) #world frame
        self.angular_velocity = np.array(msg.velocity.angular) #base frame

        self.first_message_base_arrived = True


    def get_blind_state_callback(self, msg):
        self.legs_joints_position = np.array(msg.joints_position)
        self.legs_joints_velocity = np.array(msg.joints_velocity)

        self.first_message_legs_joints_arrived = True


    def get_arm_blind_state_callback(self, msg):        
        self.arm_joints_position = np.array(msg.joints_position)
        self.arm_joints_velocity = np.array(msg.joints_velocity)

        self.first_message_arm_joints_arrived = True


    def compute_rl_control(self):        
        # Safety check to not do anything until a first base and blind state are received
        if(self.first_message_base_arrived==False or self.first_message_legs_joints_arrived==False or self.first_message_arm_joints_arrived==False):
            return
        
        # Safety check for joystick timeout
        if(self.last_joy_time is not None and time.time() - self.last_joy_time > 1.0):
            self.ref_base_lin_vel_H[0] = 0.0
            self.ref_base_lin_vel_H[1] = 0.0
            self.ref_base_ang_yaw_dot = 0.0
            print("Joystick timeout, stopping the robot")
            self.last_joy_time = None
    

        # Update the mujoco model
        self.mjData.qpos[0:3] = copy.deepcopy(self.position)
        self.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)

        self.mjData.qpos[3:7] = copy.deepcopy(self.orientation)
        self.mjData.qvel[3:6] = copy.deepcopy(self.angular_velocity)
        
        self.mjData.qpos[7:19] = copy.deepcopy(self.legs_joints_position)
        self.mjData.qvel[6:18] = copy.deepcopy(self.legs_joints_velocity)
        self.mjData.qpos[19:25] = copy.deepcopy(self.arm_joints_position)
        self.mjData.qvel[18:24] = copy.deepcopy(self.arm_joints_velocity)
        mujoco.mj_forward(self.mjModel, self.mjData)

        # Get the current state of the robot -----------------------------------------------------
        qpos, qvel = self.mjData.qpos, self.mjData.qvel
        base_lin_vel = mujoco_utils.base_lin_vel(self.mjData, frame='base')
        base_ang_vel = mujoco_utils.base_ang_vel(self.mjData, frame='base')
        base_ori_euler_xyz = mujoco_utils.base_ori_euler_xyz(self.mjData)
        heading_orientation_SO3 = mujoco_utils.heading_orientation_SO3(self.mjData)
        base_quat_wxyz = qpos[3:7]
        base_pos = mujoco_utils.base_pos(self.mjData)

        joints_pos_leg = qpos[7:19]
        joints_pos_arm = qpos[19:25]
        joints_pos_gripper = qpos[25]

        joints_vel_leg = qvel[6:18]
        joints_vel_arm = qvel[18:24]
        joints_vel_gripper = qvel[24]

    
        ref_base_lin_vel, ref_base_ang_vel = mujoco_utils.target_base_vel(self.mjData, self.ref_base_lin_vel_H, self.ref_base_ang_yaw_dot, frame='world')


        if(self.locomotion_policy.use_vision):
            self.heightmap.update_height_map(self.mjData.qpos[0:3], yaw=base_ori_euler_xyz[2])


        # IK controller --------------------------------------------------------------
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

        if(self.console.isArmActivated):
            self.desired_pose_command, \
                self.desired_joint_pos_arm, \
                ik_succeded = self.ik_mink_solver.compute(self.ref_ee_lin_pos, ee_quat, self.arm_joints_position, 
                                                        self.desired_pose_command, optimize_height=True, optimize_pitch=True)
        else:
            self.desired_joint_pos_arm = joints_pos_arm 

        # RL controller --------------------------------------------------------------
        if self.console.isRLActivated:            
            self.desired_joint_pos_leg = self.locomotion_policy.compute_control(
                        base_pos=base_pos, 
                        base_ori_euler_xyz=base_ori_euler_xyz, 
                        base_quat_wxyz=base_quat_wxyz,
                        base_lin_vel=base_lin_vel, 
                        base_ang_vel=base_ang_vel,
                        heading_orientation_SO3=heading_orientation_SO3,
                        joints_pos_leg=joints_pos_leg, 
                        joints_vel_leg=joints_vel_leg,
                        joints_pos_arm=joints_pos_arm,
                        ref_base_lin_vel=ref_base_lin_vel, 
                        ref_base_ang_vel=ref_base_ang_vel,
                        ref_pose_command=self.desired_pose_command,
                        heightmap_data=self.heightmap.data if self.locomotion_policy.use_vision else None)

            self.Kp_legs = self.locomotion_policy.Kp_walking
            self.Kd_legs = self.locomotion_policy.Kd_walking
        
        else:
            # Go up-and-down
            self.Kp_legs = self.locomotion_policy.Kp_stand_up_and_down
            self.Kd_legs = self.locomotion_policy.Kd_stand_up_and_down

        
        # Torque saturation for the legs
        max_torque = self.mjModel.actuator_ctrlrange[0:12, 1]
        max_torque = max_torque*0.95  # A margin for safety
        lower = (-max_torque + self.Kd_legs * joints_vel_leg) / self.Kp_legs
        upper = ( max_torque + self.Kd_legs * joints_vel_leg) / self.Kp_legs

        self.desired_joint_pos_leg = np.clip(
            self.desired_joint_pos_leg,
            joints_pos_leg + lower,
            joints_pos_leg + upper
        )
        


        # Send the desired positions to the trajectory generator --------------------------------            
        trajectory_generator_msg = TrajectoryGenerator()
        trajectory_generator_msg.timestamp = float(self.get_clock().now().nanoseconds)
        trajectory_generator_msg.joints_position = self.desired_joint_pos_leg.tolist()
        trajectory_generator_msg.joints_velocity = np.zeros(12).tolist()
        trajectory_generator_msg.kp = (np.ones(12)*self.Kp_legs).tolist()
        trajectory_generator_msg.kd = (np.ones(12)*self.Kd_legs).tolist()
        self.publisher_trajectory_generator.publish(trajectory_generator_msg)

        arm_trajectory_generator_msg = ArmTrajectoryGenerator()
        arm_trajectory_generator_msg.timestamp = float(self.get_clock().now().nanoseconds)
        arm_trajectory_generator_msg.desired_arm_joints_position = self.desired_joint_pos_arm.tolist()
        arm_trajectory_generator_msg.desired_arm_joints_velocity = (self.desired_joint_pos_arm*0.0).tolist()
        arm_trajectory_generator_msg.arm_kp = (np.ones(6)*self.Kp_arm).tolist()
        arm_trajectory_generator_msg.arm_kd = (np.ones(6)*self.Kd_arm).tolist()
        arm_trajectory_generator_msg.desired_arm_gripper_position = float(self.desired_joint_pos_gripper)
        self.publisher_arm_trajectory_generator.publish(arm_trajectory_generator_msg)


        # Compute the inverse dynamics
        M = np.zeros((self.mjModel.nv, self.mjModel.nv))
        mujoco.mj_fullM(self.mjModel, M, self.mjData.qM)
        M = M[18:24, 18:24]
        tau_arm = M @ (self.Kp_arm * (self.desired_joint_pos_arm - joints_pos_arm) - self.Kd_arm * joints_vel_arm)
        tau_arm += self.mjData.qfrc_bias[18:24]

        arm_control_signal_msg = ArmControlSignal()
        arm_control_signal_msg.desired_arm_joints_torque = tau_arm.tolist()
        arm_control_signal_msg.desired_arm_gripper_torque = 0.0  # Placeholder for gripper torque
        self.publisher_arm_control_signal.publish(arm_control_signal_msg)



#---------------------------
if __name__ == '__main__':
    
    print('Hello from the trash control ros node.')
    
    rclpy.init()
    trash_control_node = TrashControlNode()

    # Executor multithread (>=2)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(trash_control_node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        trash_control_node.destroy_node()
        rclpy.shutdown()

    #rclpy.spin(trash_control_node)
    #trash_control_node.destroy_node()
    #rclpy.shutdown()

    print("trash control ros node is stopped")
    exit(0)