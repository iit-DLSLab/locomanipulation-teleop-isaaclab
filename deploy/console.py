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
            "help", "ictp", "goUp", "goDown", "activate", "ictp", "setKp", "setKd",
            "activateArm", "setBasePose", "setEEPose", "armDance", "showIKVisualizer", "closeConsole"
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


    def goUp(self):
        if not self.isDown:
            print("The robot is already up")
            return

        start_time = time.time()
        time_motion = 5.0
        initial_joint_positions = copy.deepcopy(self.controller_node.legs_joints_position)
        keyframe_id = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        reference_joint_positions = self.controller_node.mjModel.key_qpos[keyframe_id][7:19]

        while time.time() - start_time < time_motion:
            alpha = (time.time() - start_time) / time_motion
            self.controller_node.desired_joint_pos_leg = (
                (1 - alpha) * initial_joint_positions + alpha * reference_joint_positions
            )
            time.sleep(0.01)

        self.isDown = False


    def goDown(self):
        if self.isDown:
            print("The robot is already down")
            return

        self.isDown = True
        wasRLActivated = self.isRLActivated
        self.isRLActivated = False
        start_time = time.time()
        time_motion = 5.0
        # Interpolate from the last commanded reference, not from the measured joints
        initial_joint_positions = copy.deepcopy(self.controller_node.desired_joint_pos_leg)
        if wasRLActivated:
            # The policy reference was tracked with the walking gains: rescale its offset from the
            # measured joints so the stand-down gains apply the same torque, otherwise
            # Kp_stand_up_and_down/Kp_walking times the torque pushes the robot in the air
            locomotion_policy = self.controller_node.locomotion_policy
            joints_pos = copy.deepcopy(self.controller_node.legs_joints_position)
            initial_joint_positions = joints_pos + (
                locomotion_policy.Kp_walking / locomotion_policy.Kp_stand_up_and_down
            ) * (initial_joint_positions - joints_pos)
        keyframe_id = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        reference_joint_positions = self.controller_node.mjModel.key_qpos[keyframe_id][7:19]

        while time.time() - start_time < time_motion:
            alpha = (time.time() - start_time) / time_motion
            self.controller_node.desired_joint_pos_leg = (
                (1 - alpha) * initial_joint_positions + alpha * reference_joint_positions
            )
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


                elif(input_string == "activate"):
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

                    print("Kp arm: ", self.controller_node.Kp_arm)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.Kp_arm = float(temp)


                elif(input_string == "setKd"):
                    print("Kd stand_up_and_down: ", self.controller_node.locomotion_policy.Kd_stand_up_and_down)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kd_stand_up_and_down = float(temp)

                    print("Kd walking: ", self.controller_node.locomotion_policy.Kd_walking)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kd_walking = float(temp)

                    print("Kd arm: ", self.controller_node.Kd_arm)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.Kd_arm = float(temp)

                elif(input_string == "ictp"):
                    print("Interactive Keyboard Control")
                    print("w: Move Forward")
                    print("s: Move Backward")
                    print("a: Move Left")
                    print("d: Move Right")
                    print("q: Rotate Left")
                    print("e: Rotate Right")
                    print("0: Stop")
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
                        else:
                            self.controller_node.ref_base_lin_vel_H[0] = 0
                            self.controller_node.ref_base_lin_vel_H[1] = 0
                            self.controller_node.ref_base_ang_yaw_dot = 0
                            break


                elif(input_string == "setBasePose"):
                    print("Current base pose command [pitch, height]: ", self.controller_node.desired_pose_command)
                    temp = input("Enter Pitch (rad): ")
                    if(temp != ""):
                        self.controller_node.desired_pose_command[0] = float(temp)
                    temp = input("Enter Height (m): ")
                    if(temp != ""):
                        self.controller_node.desired_pose_command[1] = float(temp)


                elif(input_string == "setEEPose"):
                    # The desired end-effector position is reached with IK on the arm joints
                    temp = input("Enter Target X Position (m) [default 0.3]: ")
                    x = float(temp) if temp != "" else 0.3
                    temp = input("Enter Target Y Position (m) [default 0.0]: ")
                    y = float(temp) if temp != "" else 0.0
                    temp = input("Enter Target Z Position (m) [default 0.3]: ")
                    z = float(temp) if temp != "" else 0.3
                    self.controller_node.ref_ee_lin_pos = np.array([x, y, z])


                elif(input_string == "armDance"):
                    for _ in range(10):
                        self.controller_node.ref_ee_lin_pos = np.array([
                            np.random.uniform(0.3, 0.6),
                            np.random.uniform(-0.3, 0.3),
                            np.random.uniform(0.1, 0.3),
                        ])
                        time.sleep(3)


                elif(input_string == "showIKVisualizer"):
                    self.controller_node.use_ik_visualizer = not self.controller_node.use_ik_visualizer


                elif(input_string == "closeConsole"):
                    print("Closing Console")
                    break


            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable Commands")
        print("help: Display all available messages")
        print("ictp: Interactive Keyboard Control")
        print("goUp: Make the robot stand up")
        print("goDown: Make the robot lie down")
        print("activate: Activate/Deactivate the RL policy during walking")
        print("setKp: Set the Kp gains for the PD controller")
        print("setKd: Set the Kd gains for the PD controller")
        print("activateArm: Activate/Deactivate the arm end-effector control")
        print("setBasePose: Set the desired base pitch and height")
        print("setEEPose: Set the desired end-effector position (reached with IK)")
        print("armDance: Move the end-effector to 10 random positions")
        print("showIKVisualizer: Show/Hide the IK visualizer")
        print("closeConsole: Close the console\n")
