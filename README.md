# Humanoid Robot Control System

## Summary
The primary objective of this project is to achieve comprehensive and autonomous control over humanoid robots. We focus on researching advanced control policies—specifically Reinforcement Learning (RL) and Whole-Body Control (WBC)—and developing a robust framework for their practical application in both simulation and real-world environments.

## Development Roadmap
This roadmap tracks our progress toward full humanoid autonomy.

- [x] **Robot Simulation Setup**
    - [x] Integration of high-fidelity humanoid assets in NVIDIA Isaac Sim/Isaac Lab.
    - [x] Environment configuration for realistic physics interactions.
- [x] **Sensor & Signal Integration (ROS2 / OmniGraph)**
    - [x] IMU data acquisition and analysis.
    - [x] Joint state reading and command writing.
    - [ ] RGB-D Camera and Depth sensor integration. (Planned)
- [ ] **Locomotion Control (RL Policies)**
    - [ ] Implementation of vendor-provided locomotion policies.
    - [ ] Testing stability and gait performance for Unitree G1 and H1.
- [ ] **Manipulation & Whole-Body Control (WBC)**
    - [ ] Implementation of WBC models to coordinate complex upper-limb manipulation with stable lower-body locomotion.
- [ ] **Full RL Training Pipeline**
    - [ ] Data Collection and Feature Preparation.
    - [ ] Model Development and Architecture Design.
    - [ ] Training, Hyperparameter Tuning, and Evaluation.
- [ ] **Autonomous AI Deployment**
    - [ ] Deploying trained RL/WBC models into the integrated system for real-time control.

## System Architecture & Physics
### Robot Assets
Currently, the system is developed and validated on the following humanoid platforms:
*   **Unitree H1**: Full-sized humanoid for high-performance locomotion.
*   **Unitree G1**: Next-generation humanoid (29 DOF). Specifically utilizing the `g1_29dof_with_dex3` configuration for advanced manipulation.
    *   *Reference:* [Unitree Sim IsaacLab](https://github.com/unitreerobotics/unitree_sim_isaaclab.git)

### Perception & Sensors
*   **IMU Integration**: Real-time orientation and acceleration monitoring for balance.
*   **Joint Feedback**: Precise tracking of joint positions, velocities, and efforts via ROS2.
*   **Visual Perception**: (Planned) RGB-D sensors for spatial awareness and object detection.

## Results
> [!NOTE]
> **Status: In Development**
> The project is currently in the active development phase. Performance metrics and demonstration recordings will be added here as milestones are reached.
