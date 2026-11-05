import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

robot = 'go2'  # 'go2', 'a2' (both with the piper_l arm)
scene = 'scene_rough'  # scene_flat, scene_rough

# ----------------------------------------------------------------------------------------------------------------
if(robot == "go2"):
    Kp_walking = 25.0
    Kd_walking = 1.5

    Kp_stand_up_and_down = 50.
    Kd_stand_up_and_down = 5.

    locomotion_policy_folder_path = dir_path + "/../tested_policies/" + robot + "/arm_moving_clock"

elif(robot == "a2"):
    Kp_walking = 50.
    Kd_walking = 3.

    Kp_stand_up_and_down = 50.
    Kd_stand_up_and_down = 5.

    locomotion_policy_folder_path = dir_path + "/../tested_policies/" + robot + "/arm_moving_clock"

# MuJoCo model of the robot with the arm, in the chosen scene
robot_model_path = dir_path + "/../robot_model/" + robot + "_piper_l_" + scene + ".xml"

Kp_arm = 10. *5
Kd_arm = 0.8 *5

Kp_gripper = 10.
Kd_gripper = 0.8

# Load specific training parameters
import yaml 
with open(locomotion_policy_folder_path + "/params/env.yaml", "r") as file:
    training_env = yaml.unsafe_load(file)

concurrent_state_est_network = locomotion_policy_folder_path + "/exported/concurrent_state_estimator.pth"
rma_network = locomotion_policy_folder_path + "/exported/rma.pth"

