import readline
import readchar
import time

import numpy as np
import copy
import mujoco


class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        self.isDown = True
        self.isRLActivated = False
        self.isArmActivated = False
        self.isArmJoystickActivated = False

        # Autocomplete setup
        self.commands = [
            "help", "ictp", "goUp", "goDown", "activateRL", "activateArm",  "ictp", "setKp", "setKd",
            "setBasePose", "setEefPose",
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None

    def goUp(self, ): 
        if(not self.isDown):
            print("The robot is already up")
            return

                            
        start_time = time.time()
        time_motion = 5.


        initial_joint_positions = copy.deepcopy(self.controller_node.legs_joints_position)

        keyframe_id = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        standUp_qpos = self.controller_node.mjModel.key_qpos[keyframe_id]
        reference_joint_positions = standUp_qpos[7:19]

        while(time.time() - start_time < time_motion):
            time_diff = time.time() - start_time
            alpha = time_diff / time_motion
            interpolated_positions = [
                (1 - alpha) * initial + alpha * reference
                for initial, reference in zip(initial_joint_positions, reference_joint_positions)
            ]

            self.controller_node.desired_joint_pos_leg = np.array(interpolated_positions)

            time.sleep(0.01)

        self.isDown = False

    def goDown(self, ):
        if(self.isDown):
            print("The robot is already down")
            return

        self.isDown = True
        self.isRLActivated = False

        start_time = time.time()
        time_motion = 5.

        temp = copy.deepcopy(self.controller_node.legs_joints_position)
        initial_joint_positions = temp
        
        keyframe_id = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        goDown_qpos = self.controller_node.mjModel.key_qpos[keyframe_id]
        reference_joint_positions = goDown_qpos[7:19]

        while(time.time() - start_time < time_motion):
            time_diff = time.time() - start_time
            alpha = time_diff / time_motion
            interpolated_positions = [
                (1 - alpha) * initial + alpha * reference
                for initial, reference in zip(initial_joint_positions, reference_joint_positions)
            ]

            self.controller_node.desired_joint_pos_leg = np.array(interpolated_positions)

            time.sleep(0.01)


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                if(input_string == "goUp"):
                    print("Going Up")
                    self.goUp()


                elif(input_string == "goDown"):
                    print("Going Down")
                    self.goDown()

                    
                elif(input_string == "activateRL"):
                    self.isRLActivated = not self.isRLActivated

                elif(input_string == "activateArm"):
                    self.isArmActivated = not self.isArmActivated
                    self.controller_node.desired_joint_pos_arm = np.zeros(6)

                elif(input_string == "help"):
                    self.print_all_commands()


                elif(input_string == "setKp"):
                    print("Kp stand_up_and_down: ", self.controller_node.locomotion_policy.Kp_stand_up_and_down)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kp_stand_up_and_down= float(temp)

                    print("Kp walking: ", self.controller_node.locomotion_policy.Kp_walking)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kp_walking = float(temp)

                    print("Kp arm: ", self.controller_node.manipulation_policy.Kp_arm)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.manipulation_policy.Kp_arm = float(temp)

                elif(input_string == "setKd"):
                    print("Kd stand_up_and_down: ", self.controller_node.locomotion_policy.Kd_stand_up_and_down)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kd_stand_up_and_down = float(temp)

                    print("Kd walking: ", self.controller_node.locomotion_policy.Kd_walking)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kd_walking = float(temp)

                    print("Kd arm: ", self.controller_node.manipulation_policy.Kd_arm)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.manipulation_policy.Kd_arm = float(temp)
                
                elif(input_string == "ictp"):
                    print("Interactive Keyboard Control")
                    print("w: Move Forward")
                    print("s: Move Backward")
                    print("a: Move Left")
                    print("d: Move Right")
                    print("q: Rotate Left")
                    print("e: Rotate Right")
                    print("0: Stop")
                    print("1: Pitch Up")
                    print("2: Reset Pitch")
                    print("3: Pitch Down")
                    print("Press any other key to exit")
                    while True:
                        command = readchar.readkey()
                        if(command == "w"):
                            self.controller_node.ref_base_lin_vel_H[0] += 0.1
                            print("w")
                        elif(command == "s"):
                            self.controller_node.ref_base_lin_vel_H[0] -= 0.1
                            print("s")
                        elif(command == "a"):
                            self.controller_node.ref_base_lin_vel_H[1] += 0.1
                            print("a")
                        elif(command == "d"):
                            self.controller_node.ref_base_lin_vel_H[1] -= 0.1
                            print("d")
                        elif(command == "q"):
                            self.controller_node.ref_base_ang_yaw_dot += 0.1
                            print("q")
                        elif(command == "e"):
                            self.controller_node.ref_base_ang_yaw_dot -= 0.1
                            print("e")
                        elif(command == "0"):
                            self.controller_node.ref_base_lin_vel_H[0] = 0
                            self.controller_node.ref_base_lin_vel_H[1] = 0
                            self.controller_node.ref_base_ang_yaw_dot = 0 
                            print("0")
                        elif(command == "1"):
                            self.pitch_delta -= 0.1
                            print("1")
                        elif(command == "2"):
                            self.pitch_delta = 0
                            print("2")
                        elif(command == "3"):
                            self.pitch_delta += 0.1
                            print("3")
                        else:
                            self.controller_node.ref_base_lin_vel_H[0] = 0
                            self.controller_node.ref_base_lin_vel_H[1] = 0
                            self.controller_node.ref_base_ang_yaw_dot = 0 
                            break

                elif input_string == "setBasePose":
                    print("Current Base Position: ", self.controller_node.desired_pose_command_overwrite)
                    temp = input("Enter Pitch (rad): ")
                    if(temp != ""):
                        self.controller_node.desired_pose_command_overwrite[0] = float(temp)
                    temp = input("Enter Height (m): ")
                    if(temp != ""):
                        self.controller_node.desired_pose_command_overwrite[1] = float(temp)  
                elif input_string == "setEefPose":
                    # set desired end-effector pose performs IK to compute the corresponding joint positions and set them as desired
                    temp_x = input("Enter Target X Position (m): ")
                    temp_y = input("Enter Target Y Position (m): ")
                    temp_z = input("Enter Target Z Position (m): ")

                    temp_x = float(temp_x) if temp_x != "" else 0.3
                    temp_y = float(temp_y) if temp_y != "" else 0.0
                    temp_z = float(temp_z) if temp_z != "" else 0.3

                    if temp_x != "" and temp_y != "" and temp_z != "":
                        target_pos = np.array([float(temp_x), float(temp_y), float(temp_z)])
                        
                        temp_qw = input("Enter Target W Orientation (quaternion w) [default 1.0]: ")
                        temp_qx = input("Enter Target X Orientation (quaternion x) [default 0.0]: ")
                        temp_qy = input("Enter Target Y Orientation (quaternion y) [default 0.0]: ")
                        temp_qz = input("Enter Target Z Orientation (quaternion z) [default 0.0]: ")
                        
                        qw = float(temp_qw) if temp_qw != "" else 1.0
                        qx = float(temp_qx) if temp_qx != "" else 0.0
                        qy = float(temp_qy) if temp_qy != "" else 0.0
                        qz = float(temp_qz) if temp_qz != "" else 0.0
                        target_quat = np.array([qw, qx, qy, qz])

                        initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                        initial_base_pose = copy.deepcopy(getattr(self.controller_node, 'desired_pose_command', getattr(self.controller_node, 'desired_pose_command_overwrite', np.zeros(2))))

                        reference_base_pose, reference_joints_position, ik_succeded = self.controller_node.ik_mink_solver.compute(
                            target_pos, target_quat, initial_joints_position, initial_base_pose, optimize_height=False, optimize_pitch=False)

                        if ik_succeded:
                            print("IK Succeeded. Setting desired arm joints.")
                            self.controller_node.desired_joint_pos_arm = reference_joints_position
                            if not self.isArmActivated:
                                print("Note: Arm is not activated. Use 'activateArm' command to execute.")
                        else:
                            print("IK Failed to find a solution.")
                    else:
                        print("Invalid input. X, Y, and Z positions are required.")

                elif input_string =="armHome":
                    self.controller_node.state_machine.armHome(self.controller_node.arm_joints_position)
                
                elif input_string =="armRest":
                    self.controller_node.state_machine.armRest(self.controller_node.arm_joints_position)
                
                elif input_string == "armPreReachObject":
                    self.controller_node.state_machine.armPreReachObject(self.controller_node.arm_joints_position)

                elif input_string == "armDance":
                    for j in range(10):
                        x_target_pos = np.random.uniform(0.3, 0.6)
                        y_target_pos = np.random.uniform(-0.3, 0.3)
                        z_target_pos = np.random.uniform(0.1, 0.3)
                        target_pos = [x_target_pos, y_target_pos, z_target_pos]
                        target_quat = ([ -0.7071, 0.0, -0.7071, 0])

                        initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                        initial_base_pose = copy.deepcopy(self.controller_node.desired_pose_command_overwrite)
                        
                        reference_base_pose, \
                            reference_joints_position, \
                            ik_succeded = self.controller_node.ik_mink_solver.compute(target_pos, target_quat, initial_joints_position, initial_base_pose)
                        
                        if ik_succeded:
                            time_motion = 3.
                            self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)


                elif input_string == "armReachObjectRL":
                    self.controller_node.state_machine.armReachObjectRL(self.controller_node.arm_joints_position)

                elif input_string == "armReachObjectIK":
                    self.controller_node.state_machine.armReachObjectIK(self.controller_node.arm_joints_position)

                elif input_string == "armReachBasket":
                    self.controller_node.state_machine.armReachBasket(self.controller_node.arm_joints_position)

                elif input_string == "armOpenBasket":
                    self.controller_node.state_machine.armOpenBasket(self.controller_node.arm_joints_position)

                elif input_string == "armCloseGripper":
                    print("Closing gripper")
                    self.controller_node.state_machine.change_state(gripper_state=GripperStateType.CLOSE) # CLOSE

                elif input_string == "armOpenGripper":
                    print("Opening gripper")
                    self.controller_node.state_machine.change_state(gripper_state=GripperStateType.OPEN) # OPEN

                elif input_string == "showDetectionVisualizer":
                    self.controller_node.use_detection_visualizer = not self.controller_node.use_detection_visualizer

                # elif input_string == "getTargetIKBase":
                #     target_pos, target_quat = self.controller_node.state_machine.detection_from_camera_to_base()
                #     print("Target IK Position Base: ", target_pos)
                #     print("Target IK Quaternion Base: ", target_quat)

                # elif input_string == "getTargetIKCamera":

                #     print("Target IK Position: ", self.controller_node.ik_goal_camera_frame)
                #     print("Target IK Quaternion: ", self.controller_node.ik_goal_orient_camera_frame)

                # elif input_string == "showIKFrames":
                #     target_pos, target_quat = self.controller_node.state_machine.detection_from_camera_to_base()

                #     initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                #     initial_base_pose = copy.deepcopy(self.controller_node.desired_pose_command_overwrite)

                #     _, _, ik_succeded = self.controller_node.ik_mink_solver.compute(target_pos, target_quat, initial_joints_position, initial_base_pose)


            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable keyboard commands:")
        print("help: Display all available messages")
        print("ictp: Interactive Keyboard Control")
        print("goUp: Move the robot to standing position")
        print("goDown: Move the robot to crouch position")
        print("activate: Activate/Deactivate RL policy for locomotion")
        print("setKp: Set the Kp values for the legs")
        print("setKd: Set the Kd values for the legs")
        print("setBasePose: Set desired base pitch and height")
        print("armHome: Move arm to home positionr\n")
        print("\nAvailable joystick commands:")
