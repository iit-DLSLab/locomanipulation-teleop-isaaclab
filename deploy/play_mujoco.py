# Description: This script is used to simulate the full model of the robot in mujoco

import time
import numpy as np


import sys
import os 
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path+"/mujoco/")
sys.path.append(file_path+"/../")
sys.path.append(file_path+"/../scripts/rsl_rl")


# Gym and Simulation related imports
import mujoco
import mujoco.viewer
import mujoco_utils
from heightmap import HeightMap


# Trash Policy imports
from locomotion_policy_wrapper import LocomotionPolicyWrapper
from ik_mink import IKMink


import config
import threading


class PlayMujoco:
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)

        self.simulation_dt = 0.002


        # Load the model and data.
        self.mjModel = mujoco.MjModel.from_xml_path(file_path + "/mujoco/models/scene_rough.xml")
        self.mjData = mujoco.MjData(self.mjModel)
        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.mjData.qpos = self.mjModel.key_qpos[keyframe_id]
        self.viewer = mujoco.viewer.launch_passive(
                        self.mjModel,
                        self.mjData,
                        show_left_ui=False,
                        show_right_ui=False,
                        #key_callback=lambda x: self._key_callback(x),
                    )
        
        joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        self.arm_dof_ids = np.array([self.mjModel.joint(name).id for name in joint_names])
        
        mujoco.mjv_defaultFreeCamera(self.mjModel, self.viewer.cam)
        self.last_render_time = time.time()
        self.RENDER_FREQ = 30.0  # Hz 


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
        self.legs_joints_position = np.zeros(12)  # 12 leg joints
        self.legs_joints_velocity = np.zeros(12)  # 12 leg joints
        
        self.desired_joint_pos_leg = self.mjData.qpos[7:19] 
        self.desired_joint_pos_arm = self.mjData.qpos[19:25] 
        self.desired_joint_pos_gripper = 0.0
        self.desired_pose_command_overwrite = np.zeros(2)
        self.Kp_legs = 0
        self.Kd_legs = 0
        self.Kp_arm = config.Kp_arm
        self.Kd_arm = config.Kd_arm
        self.Kp_gripper = config.Kp_gripper
        self.Kd_gripper = config.Kd_gripper


        # --------------------------------------------------------------
        self.ref_base_lin_vel_H = np.array([0.0, 0.0, 0.0])  # Desired base linear velocity in the horizontal plane (x, y, z)
        self.ref_base_ang_yaw_dot = 0.0  # Desired base angular velocity around the vertical axis

        # Interactive Command Line
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

        self.console.isDown = False  # Only in this play_mujoco script
        self.console.isRLActivated = True  # Only in this play_mujoco script


    def run(self):
        step_num = 1
        # Run the simulation
        while self.viewer.is_running():
            step_start = time.time()
            
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
            """reference_base_pose, \
                reference_joints_position, \
                ik_succeded = self.ik_mink_solver.compute(target_pos, target_quat, initial_joints_position, 
                                                        initial_base_pose, optimize_height=True, optimize_pitch=True)"""

            # RL controller --------------------------------------------------------------
            if self.console.isRLActivated and step_num % round(1 / (self.locomotion_policy.RL_FREQ * self.simulation_dt)) == 0:            
                
                """reference_base_pose, \
                    reference_joints_position, \
                    ik_succeded = self.ik_mink_solver.compute(target_pos, target_quat, initial_joints_position, 
                                                            initial_base_pose, optimize_height=True, optimize_pitch=True)"""

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
                            ref_pose_command=self.desired_pose_command_overwrite,
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
            
            # PD controller --------------------------------------------------------------
            error_joints_pos_leg = self.desired_joint_pos_leg - joints_pos_leg
            tau_leg = self.Kp_legs*error_joints_pos_leg - self.Kd_legs*joints_vel_leg

            error_joints_pos_arm = self.desired_joint_pos_arm - joints_pos_arm
            tau_arm = self.Kp_arm*error_joints_pos_arm - self.Kd_arm*joints_vel_arm

            # Compute the inverse dynamics
            M = np.zeros((self.mjModel.nv, self.mjModel.nv))
            mujoco.mj_fullM(self.mjModel, M, self.mjData.qM)
            M = M[18:24, 18:24]
            tau_arm += M @ (self.Kp_arm * (error_joints_pos_arm) - self.Kd_arm * joints_vel_arm)
            tau_arm += self.mjData.qfrc_bias[18:24]

            error_gripper_pos = self.desired_joint_pos_gripper - joints_pos_gripper
            tau_gripper = config.Kp_gripper*error_gripper_pos - config.Kd_gripper*joints_vel_gripper

            
            # Set control and mujoco step ----------------------------------------------------------------------
            self.mjData.ctrl[0:12] = tau_leg
            self.mjData.ctrl[12:18] = tau_arm 
            self.mjData.ctrl[18] = tau_gripper
            mujoco.mj_step(self.mjModel, self.mjData)
            step_num = step_num + 1


            # Sleep to match real-time ---------------------------------------------------------
            loop_elapsed_time = time.time() - step_start
            if(loop_elapsed_time < self.simulation_dt):
                time.sleep(self.simulation_dt - (loop_elapsed_time))


            # Render only at a certain frequency -----------------------------------------------------------------
            if time.time() - self.last_render_time > 1.0 / self.RENDER_FREQ:
                # Update the camera position
                self.viewer.cam.lookat[:] = base_pos
                self.viewer.sync()
                self.last_render_time = time.time()


if __name__ == "__main__":
    play_mujoco = PlayMujoco()
    play_mujoco.run()