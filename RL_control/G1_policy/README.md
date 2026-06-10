# Unitree G1 ROS 2 Policy Controller for Isaac Sim

This workspace runs the pretrained 12-DoF Unitree G1 locomotion policy from
`unitree_rl_gym` through ROS 2 Jazzy and the Isaac Sim ROS 2 bridge.

The ROS node is intentionally started **before** Isaac Sim starts publishing
simulation data. It loads the policy, waits for synchronized sensor messages,
and begins control on the first valid sample after Play is pressed.

## Package

```text
RL_control/G1_policy/
├── PLAN.md
├── README.md
└── g1_policy_controller/
    ├── g1_policy_controller/core.py
    ├── scripts/g1_policy_node.py
    ├── launch/g1_policy_controller.launch.py
    ├── policy/motion.pt
    └── test/test_g1_policy_controller.py
```

The packaged `motion.pt` is byte-identical to:

```text
unitree_rl_gym/deploy/pre_train/g1/motion.pt
```

SHA-256:

```text
cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d
```

## Requirements

The implementation was built and tested with:

- ROS 2 Jazzy
- Conda environment `env_sim`
- Python 3.12
- PyTorch 2.11
- NumPy 2.4
- `rclpy`
- `message_filters`
- `sensor_msgs`
- `geometry_msgs`

The Isaac Sim stage must already provide:

```text
/clock                 rosgraph_msgs/msg/Clock
/imu                   sensor_msgs/msg/Imu
/joint/joint_states    sensor_msgs/msg/JointState
/joint/joint_command   sensor_msgs/msg/JointState
```

The stage must place the robot at the correct home pose when Play is pressed.
This controller does not contain a startup interpolation or a move-to-home
state.

## Build

Run from a terminal:

```bash
cd /home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/G1_policy
conda activate env_sim
source /opt/ros/jazzy/setup.bash

colcon build --symlink-install --packages-select g1_policy_controller
source install/setup.bash
```

Run the tests with the same environment:

```bash
colcon test --packages-select g1_policy_controller
colcon test-result --verbose
```

## Startup Order

Use this order every time.

1. Open the configured G1 stage in Isaac Sim.
2. Do **not** press Play yet.
3. Open a terminal and prepare the environment:

```bash
cd /home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/G1_policy
conda activate env_sim
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

4. Start the ROS 2 controller:

```bash
ros2 launch g1_policy_controller g1_policy_controller.launch.py
```

The expected terminal output includes:

```text
Loaded G1 policy: ...
G1 policy controller is ready. Start Isaac Sim Play; the controller will run
on the first synchronized sensor sample.
```

5. Press Play in Isaac Sim.

After Play, Isaac Sim publishes the IMU, joint state, and clock. The node runs
its first inference on the first valid synchronized IMU/joint-state sample and
starts publishing `/joint/joint_command`.

There is no enable service, warm-up delay, or ramp. The LSTM starts with zero
memory and the velocity command starts at zero.

## Velocity Commands

The node subscribes to:

```text
/cmd_vel    geometry_msgs/msg/Twist
```

The command is interpreted as:

```text
linear.x  = desired forward velocity [m/s]
linear.y  = desired left velocity [m/s]
angular.z = desired yaw rate [rad/s]
```

Forward example:

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

Stop command:

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

Commands are clipped before observation scaling:

| Command | Limit |
|---|---:|
| Forward velocity `vx` | `[-0.8, 0.8] m/s` |
| Lateral velocity `vy` | `[-0.5, 0.5] m/s` |
| Yaw rate `wz` | `[-1.57, 1.57] rad/s` |

Like the NVIDIA H1 example, the most recent command is retained. There is no
automatic command timeout, so send an explicit zero command when motion should
stop.

## ROS Topics

| Direction | Default topic | Message |
|---|---|---|
| Subscribe | `/imu` | `sensor_msgs/msg/Imu` |
| Subscribe | `/joint/joint_states` | `sensor_msgs/msg/JointState` |
| Subscribe | `/cmd_vel` | `geometry_msgs/msg/Twist` |
| Publish | `/joint/joint_command` | `sensor_msgs/msg/JointState` |

All topic names and the policy path are launch arguments:

```bash
ros2 launch g1_policy_controller g1_policy_controller.launch.py --show-args
```

## Timing and Clock Behavior

Isaac Sim publishes the IMU and joint state at:

```text
sensor_dt = 0.005 s
sensor_rate = 200 Hz in simulation time
```

The G1 policy was trained at:

```text
policy_dt = 0.02 s
policy_rate = 50 Hz in simulation time
```

The controller uses exact timestamp synchronization through
`message_filters.TimeSynchronizer`.

At every synchronized 200 Hz sensor callback:

1. Validate IMU and joint-state data.
2. Run the LSTM on the first callback and then every fourth valid callback.
3. Commit the callback counter and policy step after successful processing.
4. Publish the newest 43-joint position target.

The action therefore changes at 50 Hz, while the latest target is sent to
Isaac Sim at 200 Hz when the sensor stream remains at 200 Hz. Like the H1
example, scheduling is callback-count based. If synchronized sensor callbacks
are dropped, the policy waits until four valid callbacks have been received,
so its cadence becomes slower than 50 Hz in simulation time.

Like the H1 controller, an inference slot is consumed only after policy
processing succeeds. If observation construction or inference fails, the
controller does not commit the callback count or gait phase. The next valid
sensor callback retries the same policy step and phase. No joint command is
published for the failed callback.

If simulation time moves backwards after Stop/Play or Reset, the node resets:

- LSTM hidden state
- LSTM cell state
- Previous action
- Current action
- Gait phase and policy step

The next valid sensor pair starts a new policy sequence immediately.

## Policy Observation

The TorchScript actor receives 47 `float32` values:

```text
observation shape = [1, 47]
```

| Indices | Size | Value |
|---|---:|---|
| `0:3` | 3 | `0.25 * pelvis angular velocity` |
| `3:6` | 3 | Projected gravity in the pelvis frame |
| `6:9` | 3 | `[2.0*vx, 2.0*vy, 0.25*wz]` |
| `9:21` | 12 | `q_leg - q_default` |
| `21:33` | 12 | `0.05 * dq_leg` |
| `33:45` | 12 | Raw action from the previous inference |
| `45` | 1 | `sin(2*pi*phase)` |
| `46` | 1 | `cos(2*pi*phase)` |

### Angular Velocity

The policy uses `Imu.angular_velocity` directly in the pelvis frame:

```text
obs[0:3] = 0.25 * [wx, wy, wz]
```

The expected axes are:

```text
+x forward
+y left
+z up
```

### Projected Gravity

ROS stores quaternion fields as `x, y, z, w`. The controller reorders them to
`w, x, y, z` before applying the Unitree deployment formula:

```text
gx =  2 * (-qz*qx + qw*qy)
gy = -2 * ( qz*qy + qw*qx)
gz =  1 - 2 * (qw*qw + qz*qz)
```

For an upright robot:

```text
projected_gravity ~= [0, 0, -1]
```

The policy does not use IMU linear acceleration. Projected gravity is a unit
direction vector and is not multiplied by `9.81`.

### Joint Position and Velocity

The Isaac Sim message contains 43 joints in an interleaved order. The
controller maps joints by name and extracts this exact policy order:

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
```

Default angles:

```text
[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
 -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
```

Position and velocity observations are:

```text
obs[9:21]  = q_leg - q_default
obs[21:33] = 0.05 * dq_leg
```

Waist, arm, wrist, and hand states are not included in the policy observation.

### Previous Action

`obs[33:45]` contains the raw 12-value output from the previous LSTM
inference. It is not a joint target and is not multiplied by the action scale.
It is updated only after the current inference completes.

### Gait Phase

The gait period is `0.8 s`, or 40 policy steps:

```text
phase = ((policy_step * 0.02) mod 0.8) / 0.8
```

To match the Python real-robot deployment, the first inference uses:

```text
phase = 0.025
```

## Policy Output and Joint Command

The policy outputs 12 raw actions:

```text
action shape = [1, 12]
```

Each leg target is:

```text
q_target = q_default + 0.25 * action
```

The ROS command contains all 43 joint names in the order currently published
by Isaac Sim:

- The 12 leg joints receive policy targets.
- The remaining 31 waist, arm, wrist, and Dex3 joints receive `0 rad`.
- Velocity targets are zero.
- Effort targets are zero.

The Isaac Sim articulation drives apply the configured stiffness and damping.
The ROS node does not calculate or publish torque.

## Similarities to the Isaac Sim H1 Example

This controller follows the same high-level design as:

```text
IsaacSim-ros_workspaces/jazzy_ws/src/
humanoid_locomotion_policy_example/h1_fullbody_controller
```

Both controllers:

- Are ROS 2 Python nodes.
- Load a TorchScript policy.
- Subscribe to `cmd_vel`, IMU, and joint state.
- Use `message_filters.TimeSynchronizer` for IMU and joint state.
- Use Isaac Sim time.
- Publish position targets as `sensor_msgs/msg/JointState`.
- Can be launched before Play and wait for simulation messages.
- Retain the most recent velocity command.

## Differences from the H1 Example

| Area | H1 example | This G1 controller |
|---|---|---|
| Policy input | 69 values | 47 values |
| Policy output | 19 full-body actions | 12 leg actions |
| Base linear velocity | Integrated from acceleration | Not observed |
| IMU linear acceleration | Used | Ignored |
| Joint position scale | H1 policy convention | `1.0` |
| Joint velocity scale | H1 policy convention | `0.05` |
| Action scale | `0.5` | `0.25` |
| Gait phase | Not included | Sine/cosine, `0.8 s` period |
| Policy scheduling | Every fourth callback | Every fourth valid callback |
| Failed inference | Counter is not advanced | Counter, phase, and LSTM are not advanced |
| Published joints | 19 controlled joints | All 43 G1/Dex3 joints |
| Upper body | Controlled by policy | Held at zero |
| Clock reset | Logs backward time | Resets LSTM and temporal state |
| Joint mapping | Missing joints become zero | Required joints are validated by name |

Both controllers now use callback-count decimation. The G1 controller still
uses sensor timestamps to detect a backward time jump and reset its LSTM and
temporal state.

## Validation and Safety Notes

The node rejects a synchronized sample when:

- The IMU and joint-state timestamps differ.
- The message does not contain exactly 43 joints.
- A required leg joint is missing.
- Joint arrays have inconsistent lengths.
- A quaternion, angular velocity, joint position, or joint velocity is not
  finite.
- The quaternion norm is zero.

If policy inference fails, the node restores the LSTM memory, retains the last
successful action, and retries the same gait phase on the next valid callback.
The failed callback does not publish a joint command.

The original deployment does not clip policy actions, and this implementation
keeps that behavior. It also does not add a joint-target limiter. Start with
zero or small velocity commands and verify the robot is at the configured home
pose before pressing Play.

## Stop

Send a zero velocity command, then stop the ROS node with `Ctrl+C`. Stop Isaac
Sim afterward. Starting the node again resets the LSTM memory.
