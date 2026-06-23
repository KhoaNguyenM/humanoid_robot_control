# Roadmap and Notes

This file keeps the future roadmap, external deployment items, architecture notes, and result notes separate from the main repository README.

## Future Roadmap

- [ ] **RGB-D Camera and Depth Sensor Integration**
    - [ ] RGB-D camera and depth sensor integration.
- [ ] **Unitree G1 Locomotion and Teleoperation**
    - [ ] Testing stability and gait performance for Unitree G1 (loco & teleop).

## Deploy on Another Repo

- [ ] **Manipulation & Whole-Body Control (WBC)**
    - [ ] Implementation of WBC models (SONIC, TWIST2) to coordinate complex upper-limb manipulation with stable lower-body locomotion.
- [ ] **Full RL Training Pipeline**
    - [ ] Data Collection and Feature Preparation.
    - [ ] Model Development and Architecture Design.
    - [ ] Training, Hyperparameter Tuning, and Evaluation.
- [ ] **Autonomous AI Deployment**
    - [ ] Deploying trained RL/WBC models into the integrated system for real-time control.
- [ ] **VLA Deployment**
    - [ ] Deploying VLA model (GROOT N, pi,....)
- [ ] **Perceptro Robotics Model Deployment**

## System Architecture & Physics

### Robot Assets

Currently, the system is developed and validated on the following humanoid platforms:

- **Unitree H1**: Full-sized humanoid for high-performance locomotion.
- **Unitree G1**: Next-generation humanoid (29 DOF). Specifically utilizing the `g1_29dof_with_dex3` configuration for advanced manipulation.
    - *Reference:* [Unitree Sim IsaacLab](https://github.com/unitreerobotics/unitree_sim_isaaclab.git)

### Perception & Sensors

- **IMU Integration**: Real-time orientation and acceleration monitoring for balance.
- **Joint Feedback**: Precise tracking of joint positions, velocities, and efforts via ROS2.
- **Visual Perception**: (Planned) RGB-D sensors for spatial awareness and object detection.

## Results

### 1.1 ROS2 Control for Unitree H1 in Isaac Sim

Successfully controlled the Unitree H1 humanoid in Isaac Sim through ROS2, using policy observations from `/imu`, `/joint_states`, `/clock`, and command actions through `/joint_command`.

![H1 ROS2 policy control diagram](document/H1_diagram.png)

### 1.2 ROS2 Control for Unitree G1 in Isaac Sim

> [!NOTE]
> **Status: In Development**
> The project is currently in the active development phase. Performance metrics and demonstration recordings will be added here as milestones are reached.
