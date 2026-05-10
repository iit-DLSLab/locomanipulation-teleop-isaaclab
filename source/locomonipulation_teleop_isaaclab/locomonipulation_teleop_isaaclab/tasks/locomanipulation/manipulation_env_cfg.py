import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.envs import ViewerCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG

from isaaclab.markers import VisualizationMarkersCfg

from locomanipulation_teleop.assets.aliengo_asset import ALIENGO_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import locomanipulation_teleop.tasks.custom_events as custom_events
import locomanipulation_teleop.tasks.custom_curriculums as custom_curriculums

import sys
import os 
import yaml 

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    scale_all_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )

    
    external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-1.0, 1.0),
        },
    )
    

    scale_all_joint_friction_model = EventTerm(
        func=custom_events.randomize_joint_friction_model,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
                "friction_distribution_params": (0.2, 2.0),
                "operation": "scale"},
    )


    scale_all_joint_armature_model = EventTerm(
        func=custom_events.randomize_joint_friction_model,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
                "armature_distribution_params": (0.0, 1.0),
                "operation": "scale"},
    )
    


    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (-5.0, 5.0),
            "damping_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )



@configclass
class ManipulationFlatEnvCfg(DirectRLEnvCfg):

    # Viewer
    #viewer = ViewerCfg(eye=(1.5, 1.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 6 + 2 # 6 for the arm, 2 for robot pose commands
    observation_space = 18 + 18 # 12 are the arm joints pos and vel
    observation_space += 9 # 9 the pose linear vel and angular, and orientation 
    observation_space += 7 # ee goal
    observation_space += 6 # action arm
    observation_space += 2 # pose commands 
    state_space = 0

    use_velocity_commands = True
    if(use_velocity_commands):
        action_space += 3 # robot vel commands
        observation_space +=3 # robot vel feedback


    # observation history
    use_observation_history = True
    history_length = 5
    if(use_observation_history):
        single_observation_space = observation_space # Placeholder. Later we may add map, but only from the latest obs
        observation_space *= history_length


    use_rma = False
    if(use_rma):
        rma_output_space = 12 # P gain
        rma_output_space += 12 # D gain 
        rma_output_space += 12 # friction static
        rma_output_space += 12 # friction dynamic
        rma_output_space += 12 # armature
        single_rma_observation_space = single_observation_space
        rma_observation_space = observation_space
        observation_space += rma_output_space
        rma_batch_size = 32
        rma_train_epochs = 500
        rma_lr = 1e-3
        rma_ep_saving_interval = 1000
        

    use_filter_actions = True

    
    # asymmetric ppo
    use_asymmetric_ppo = False
    if(use_asymmetric_ppo):
        state_space = observation_space
        state_space += 12 # P gain
        state_space += 12 # D gain
        #state_space += 1*17 # mass*num_bodies
        #state_space += 1*17 # inertia*num_bodies
        #state_space += 1 # wrench
        state_space += 12 # friction static
        state_space += 12 # friction dynamic
        state_space += 12 # armature
        #state_space += 1 # restitution



    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        #disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        #physx=PhysxCfg(
        #    gpu_max_rigid_contact_count=2**20,
        #    gpu_max_rigid_patch_count=2**24,
        #),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        #attach_yaw_only=True,
        ray_alignment='yaw',
        #pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.4, 1.0]),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # HACK to have gripper position
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/link06", 
        offset=ImuCfg.OffsetCfg(
            pos=(0.16, 0, 0)
        ), 
        debug_vis=False)
    
    # some marker for visualization
    """goal_object_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        },
    )

    ee_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        },
    )"""
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    frame_marker_cfg.prim_path = "/Visuals/myMarkers"
    #goal_object_marker_cfg = frame_marker_cfg
    #ee_marker_cfg = frame_marker_cfg

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()


    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # robot
    robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    desired_joints_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
                           'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',  
                           'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
                           'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6']


    # Desired clip actions
    desired_clip_actions = 6.0
    
    # Tracking reward scale
    ee_position_reward_scale = 1.5
    ee_final_orientation_reward_scale = 1.5
    ee_final_velocity_reward_scale = 1.5
    
    # Joint reward scale
    joints_torque_reward_scale = -2.5e-6
    joints_accel_reward_scale = -2.5e-8
    joints_vel_reward_scale = -2.5e-5
    joints_energy_reward_scale = -1e-4
    #joints_arm_position_reward_scale = -0.001
    joints_vel_smoothness_reward_scale = -1e-4
    joints_vel_final_reward_scale = 0.2
   
    
    # Undesired contacts reward scale
    undersired_contact_reward_scale = -1.0
    action_rate_reward_scale = -0.01
    action_smoothness_reward_scale = -0.001
    action_pose_and_vel_near_zero_reward_scale = -0.1

    # Robot base velocity tracking reward scale, to avoid
    # brutal motions during manipulation
    z_vel_reward_scale = -0.25
    ang_vel_reward_scale = -0.25


    # Loading the locomotion policy --------------------------------------------------------------------------------
    #try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    locomotion_policy_folder_path = dir_path + "/../../../../../tested_policies/locomotion/rough"

    # Load specific training parameters
    locomotion_policy_env_cfg = yaml.unsafe_load(open(locomotion_policy_folder_path + "/params/env.yaml", "r"))

    # state estimator locomotion
    use_imu = locomotion_policy_env_cfg["use_imu"]

    # action locomotion
    use_filter_actions_locomotion = locomotion_policy_env_cfg["use_filter_actions"]
    desired_clip_actions_locomotion = locomotion_policy_env_cfg["desired_clip_actions"]
    action_scale_locomotion = locomotion_policy_env_cfg["action_scale"]
    action_space_locomotion = locomotion_policy_env_cfg["action_space"]

    # periodic gait locomotion
    use_clock_signal = locomotion_policy_env_cfg["use_clock_signal"]
    # Desired step freq and duty factor (if periodic gait is used)
    desired_step_freq = locomotion_policy_env_cfg["desired_step_freq"]
    desired_duty_factor = locomotion_policy_env_cfg["desired_duty_factor"]
    desired_phase_offset = locomotion_policy_env_cfg["desired_phase_offset"]

    # observation locomotion
    observation_space_locomotion = locomotion_policy_env_cfg["observation_space"]
    use_observation_history_locomotion = locomotion_policy_env_cfg["use_observation_history"]
    history_length_locomotion = locomotion_policy_env_cfg["history_length"]
    if(use_observation_history_locomotion):
        single_observation_space_locomotion = locomotion_policy_env_cfg["single_observation_space"]
    #except:
    #    print("Error loading the locomotion policy parameters")  
    # -----------------------------------------------------------------------------------------------------------------



import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
@configclass
class ManipulationRoughBlindEnvCfg(ManipulationFlatEnvCfg):

    ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
        curriculum=False,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.2
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.10), platform_width=2.0,
            ),
            "star": terrain_gen.MeshStarTerrainCfg(
                proportion=0.1, num_bars=10, bar_width_range=(0.15, 0.20), bar_height_range=(0.05, 0.15), platform_width=2.0,
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.15, step_height_range=(0.05, 0.18), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.15, step_height_range=(0.05, 0.18), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
        },
    )

    """Rough terrains configuration."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=10,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )





@configclass
class ManipulationRoughVisionEnvCfg(ManipulationRoughBlindEnvCfg):
    # env
    #observation_space = 276
    observation_space = 429

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        #ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 1.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )