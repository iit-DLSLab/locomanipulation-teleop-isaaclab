from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import mink
from mink.contrib.keyboard_teleop import keycodes

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")

import time
from scipy.spatial.transform import Rotation as R

"""@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == keycodes.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == keycodes.KEY_SPACE:
            self.pause = not self.pause"""

class IKMink:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(dir_path+"/mujoco/models/scene_floating.xml")
        self.data = mujoco.MjData(self.model)

        # Joints we wish to control.
        # fmt: off
        self.joint_names = [
            "basepitch", "basez", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]
        # fmt: on
        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])
        #self.actuator_ids = np.array([self.model.actuator(name).id for name in self.joint_names])

        self.configuration = mink.Configuration(self.model)

        self.end_effector_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.1,
            lm_damping=1.0,
        )

        self.posture_cost = np.zeros((self.model.nv,))
        self.posture_cost[2:] = 1e-3
        self.posture_task = mink.PostureTask(self.model, cost=self.posture_cost)

        self.immobile_base_cost = np.zeros((self.model.nv,))
        self.immobile_base_cost[0:2] = 5
        self.damping_task = mink.DampingTask(self.model, self.immobile_base_cost)

        self.tasks = [
            self.end_effector_task,
            self.posture_task,
        ]

        # Enable collision avoidance between the following geoms.
        self.collision_pairs = [
            (["base_collision1", "base_collision2", "base_collision3"],
            ["link2_collision1", "link2_collision2",
            "link3_collision1", "link3_collision2",
            "link4_collision",
            "link5_collision",
            "link6_collision",
            "link7_collision",
            "link8_collision"])
        ]
        self.collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=self.collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.02,
            collision_detection_distance=0.15,
        )

        self.limits = [
            mink.ConfigurationLimit(self.model),
            self.collision_avoidance_limit,
        ]

        # IK settings.
        self.solver = "daqp"
        self.pos_threshold = 1e-1
        self.ori_threshold = 1e-1
        self.max_iters = 20

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(self.model, self.data, "target", "attachment_site", "site")

    def compute(self, target_pos: np.ndarray, target_quat: np.ndarray, initial_joints_position: np.ndarray, initial_base_pose: np.ndarray, 
                optimize_height = False, optimize_pitch = False, visualize = False) -> [np.ndarray, np.ndarray, bool]:

        self.data.qpos[0:8] = np.concatenate((initial_base_pose, initial_joints_position))
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

        # Update task target.
        mocap_id = self.model.body("target").mocapid[0]
        self.data.mocap_pos[mocap_id] = target_pos
        self.data.mocap_quat[mocap_id] = target_quat

        T_wt = mink.SE3.from_mocap_name(self.model, self.data, "target")
        self.end_effector_task.set_target(T_wt)

        # Compute velocity and integrate into the next configuration.
        for i in range(self.max_iters):
            
            if(optimize_pitch and optimize_height):
                vel = mink.solve_ik(
                    self.configuration,
                    [*self.tasks, self.damping_task],
                    dt=0.005,
                    solver=self.solver,
                    damping=1e-3,
                    limits=self.limits,
                )
            else:   
                if(optimize_pitch == True and optimize_height == False):
                    # Create DOF freezing constraint for second joint.
                    frozen_dofs = [1]
                elif(optimize_pitch == False and optimize_height == True):
                    # Create DOF freezing constraint for first joint.
                    frozen_dofs = [0]
                else:
                    # Create DOF freezing constraint for first two joints.
                    frozen_dofs = [0, 1]
                
                freeze_task = mink.DofFreezingTask(model=self.model, dof_indices=frozen_dofs)
                vel = mink.solve_ik(
                    self.configuration,
                    [*self.tasks, self.damping_task],
                    constraints=[freeze_task],
                    dt=0.005,
                    solver=self.solver,
                    damping=1e-3,
                    limits=self.limits,
                )

            
            self.configuration.integrate_inplace(vel, 0.005)

            # Exit condition.
            err = self.end_effector_task.compute_error(self.configuration)
            pos_achieved = bool(np.linalg.norm(err[:3]) <= self.pos_threshold)
            ori_achieved = bool(np.linalg.norm(err[3:]) <= self.ori_threshold)
            if pos_achieved:
                ik_succeded = True
            else:
                ik_succeded = False

        final_base_pose = self.configuration.q[0:2] #base pitch, base z
        final_arm_joints = self.configuration.q[2:8]

        if visualize:
            self.visualize_ik(final_base_pose, final_arm_joints, target_pos, target_quat)

        return final_base_pose, final_arm_joints, ik_succeded
    

    def visualize_ik(self, base_pose: np.ndarray, arm_joints: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray) -> None:
        # Set final configuration
        self.data.qpos[0:2] = base_pose
        self.data.qpos[2:8] = arm_joints

        mujoco.mj_fwdPosition(self.model, self.data)

        # Launch viewer
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        while viewer.is_running():
            input("Press Enter to close the viewer...")
            viewer.close()


if __name__ == "__main__":
    
    ik_solver = IKMink()

    # Initial joint configuration
    initial_joints = np.array([0.0, -0.5, 0.5, 0.0, 1.0, 0.0])
    initial_base_pose = np.array([0.0, 0.0])  # pitch, z

    while True:

        # Define target position and orientation
        x_pos = np.random.uniform(0.4, 0.4)
        y_pos = np.random.uniform(-0.2, 0.2)
        z_pos = np.random.uniform(0.3, 0.6)
        target_pos = np.array([x_pos, y_pos, z_pos])

        roll_grasp = np.random.uniform(-1.8, 1.8)
        pitch_grasp = np.random.uniform(-1.8, 1.8)
        yaw_grasp = np.random.uniform(-1.8, 1.8)
        r = R.from_euler('xyz', [roll_grasp, pitch_grasp, yaw_grasp], degrees=False)
        target_quat = r.as_quat()

        # Compute IK
        final_base_pose, \
        final_arm_joints, \
        success = ik_solver.compute(target_pos, target_quat, initial_joints, initial_base_pose, 
                                    optimize_height=False, optimize_pitch=False, visualize=True)
        
        print("IK Success? ", success)

