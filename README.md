<div style="display: flex; justify-content: space-around;">
  <img src="https://img.shields.io/badge/IsaacLab%20-v2.3.2-green" alt="IsaacLab v2.3.0" style="margin-bottom: 1px;">
  <img src="https://img.shields.io/badge/rsl_rl%20-v3.3.0-brown" alt="rsl-rl v3.3.0" style="margin-bottom: 1px;">
  <img src="https://img.shields.io/badge/Mujoco%20-v3.7.0-blue" alt="Mujoco v3.7.0" style="margin-bottom: 1px;">
</div>

## Overview
An IsaacLab extension by DLS for performing teleoperation with Unitree Go2 + AgileX Piper.

Status: Work in progress - the train is based still on Aliengo, but it works on Go2. Arm IK can be sometimes jittering in some configuration.

Features:
- Locomotion policy able to adjust pose and carry a manipulator
- Manipulation controller using whole-body Inverse Kinematics with reduced model using [mink](https://github.com/kevinzakka/mink) + feedback linearization
- End-effector reference generation via joystick
- Sim-to-Sim in [Mujoco](https://github.com/google-deepmind/mujoco)
- Sim-to-Real in ROS2 compatible with our public low-level robot's hal for Go2 [unitree_ros2_dls](https://github.com/iit-DLSLab/unitree_ros2_dls) and Agilex Piper arm [piper-ros2-dls](https://github.com/iit-DLSLab/piper-ros2-dls)

## Cite this work

This work takes a lot of inspiration from our repo [trash-collection-isaaclab](https://github.com/iit-DLSLab/trash-collection-isaaclab). If you find it useful, please consider citing:

#### [BinWalker: Development and Field Evaluation of a Quadruped Manipulator Platform for Sustainable Litter Collection](https://arxiv.org/pdf/2603.10529)

```
@article{turrisi26littercollection,
  author = {Giulio Turrisi and Angelo Bratta and Giovanni Minelli and Gabriel Fischer Abati and Amir H. Rad and João Carlos Virgolino Soares and Claudio Semini},
  title = {BinWalker: Development and Field Evaluation of a Quadruped Manipulator Platform for Sustainable Litter Collection},
  journal = {arXiv},
  year = {2026}
}
```

## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi).
