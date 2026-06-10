# Unitree G1 Policy Observation

This document describes the observation passed to the G1 policy in the
`unitree_rl_gym` repository. It covers the signal components, data sources,
coordinate frames, physical units, scaling factors, update rates, index order,
and temporal relationship between observations and actions.

## 1. Policy Scope

The physical G1 has 29 motors, but the policy in this repository does not
control all 29 DoFs.

- Policy input: `47` values.
- Policy output: `12` actions.
- The policy observes and controls only the 12 leg joints.
- The 3 waist joints and 14 arm joints, motors `12..28`, are held at fixed
  targets by PD controllers during real-robot deployment.
- Training uses `g1_12dof.urdf`, not the full 29-DoF dynamics model.

Relevant files:

- `legged_gym/envs/g1/g1_config.py`
- `resources/robots/g1_description/g1_12dof.urdf`
- `deploy/deploy_real/configs/g1.yaml`

Therefore, the most accurate description is:

> A 12-DoF locomotion policy deployed on a 29-DoF G1 robot.

## 2. Signal Flow Overview

At policy step `k`, the controller performs the following signal processing:

```text
IMU quaternion ------> projected gravity ---+
IMU gyroscope --------> angular velocity ----+
12 encoder positions -> joint offsets -------+
12 encoder velocities -> joint velocities ---+--> observation o_k (47)
remote/command -------> velocity command ----+
action a_(k-1) -------> previous action -----+
internal clock -------> sine/cosine phase ---+
                                               |
                                               v
                                      LSTM policy 47 -> 64
                                               |
                                               v
                                      MLP 64 -> 32 -> 12
                                               |
                                               v
                                           action a_k
                                               |
                                               v
                               q_target = q_default + 0.25*a_k
```

An observation is a snapshot at one policy step, but the policy is not purely
feed-forward. The exported `motion.pt` contains an LSTM with two internal
states:

```text
hidden_state: (1, 1, 64)
cell_state:   (1, 1, 64)
```

Consequently:

```text
a_k = policy(o_k, h_(k-1), c_(k-1))
```

Passing the same observation to the policy twice can produce different actions
when the LSTM state is different.

## 3. The 47-D Observation

The actor observation is:

```text
o_k = [
    0.25 * omega_body,                         # 3
    projected_gravity_body,                   # 3
    [2.0*cmd_vx, 2.0*cmd_vy, 0.25*cmd_wz],    # 3
    1.0 * (q - q_default),                    # 12
    0.05 * dq,                                # 12
    previous_action,                          # 12
    sin(2*pi*phase),                          # 1
    cos(2*pi*phase)                           # 1
]
```

Total dimension:

```text
3 + 3 + 3 + 12 + 12 + 12 + 1 + 1 = 47
```

After scaling, the values are dimensionless inputs passed directly to the
neural network. The repository does not use a running mean/variance
observation normalizer.

## 4. Complete Index Table

| Python index | Size | Signal | Physical unit before scaling | Scale | Policy input |
|---|---:|---|---|---:|---|
| `0:3` | 3 | Pelvis angular velocity | `rad/s` | `0.25` | `0.25*omega` |
| `3:6` | 3 | Projected gravity | unit vector | `1.0` | `R^T*[0,0,-1]` |
| `6:9` | 3 | Command `vx, vy, wz` | `m/s, m/s, rad/s` | `2, 2, 0.25` | scaled command |
| `9:21` | 12 | Joint position offset | `rad` | `1.0` | `q-q_default` |
| `21:33` | 12 | Joint velocity | `rad/s` | `0.05` | `0.05*dq` |
| `33:45` | 12 | Previous action | dimensionless | `1.0` | `a_(k-1)` |
| `45` | 1 | Sine of gait phase | dimensionless | `1.0` | `sin(2*pi*phase)` |
| `46` | 1 | Cosine of gait phase | dimensionless | `1.0` | `cos(2*pi*phase)` |

Python slices exclude the right endpoint. For example, `obs[0:3]` contains
elements `0`, `1`, and `2`.

## 5. Signal Details

### 5.1 Angular Velocity: `obs[0:3]`

This signal is the angular velocity of the pelvis/base expressed in the body
frame:

```text
omega_body = [omega_x, omega_y, omega_z]
```

Model axis convention:

- `+x`: forward.
- `+y`: left.
- `+z`: upward.

Therefore:

- `omega_x`: roll rate.
- `omega_y`: pitch rate.
- `omega_z`: yaw rate.

On the real robot, the signal comes from:

```python
low_state.imu_state.gyroscope
```

The G1 configuration uses `imu_type: "pelvis"`, so no torso-to-pelvis
transformation is applied. The quaternion and gyroscope are used directly from
the pelvis IMU.

The physical unit is `rad/s`. The policy input is:

```text
obs[0:3] = gyroscope [rad/s] * 0.25
```

For example, `omega_z = 1 rad/s` produces `obs[2] = 0.25`.

The deployment code does not apply an additional low-pass filter, moving
average, or other software filter. The controller samples the latest IMU value
available in `low_state`.

### 5.2 Projected Gravity: `obs[3:6]`

The policy does not receive the quaternion directly. The quaternion is
converted into the direction of gravity expressed in the body frame:

```text
g_body = R_world_to_body * [0, 0, -1]
```

This is a unit vector, not an acceleration in `m/s^2`. Its ideal magnitude is:

```text
||g_body|| = 1
```

The deployment quaternion order is:

```text
q = [qw, qx, qy, qz]
```

The code computes:

```text
g_x =  2*(-qz*qx + qw*qy)
g_y = -2*( qz*qy + qw*qx)
g_z =  1 - 2*(qw^2 + qz^2)
```

Useful reference poses:

| Pose | Approximate projected gravity |
|---|---|
| Upright | `[0, 0, -1]` |
| Rolled by `+10 deg` | `[0, -0.17365, -0.98481]` |
| Upside down | `[0, 0, +1]` |

Projected gravity carries roll and pitch information but is independent of
absolute yaw. This representation is suitable for locomotion because the
policy does not need to know the robot's absolute world heading.

Quaternion convention:

- Isaac Gym root state stores quaternions as `[x, y, z, w]`.
- `LowState` and the deployment helper use `[w, x, y, z]`.
- The two code paths use different helper functions appropriate to their
  respective conventions.

### 5.3 Velocity Command: `obs[6:9]`

The command has the following physical meaning:

```text
cmd = [desired_vx, desired_vy, desired_yaw_rate]
```

Units and signs:

- `desired_vx`: `m/s`; positive means forward.
- `desired_vy`: `m/s`; positive means left.
- `desired_yaw_rate`: `rad/s`; positive follows the right-hand rule around
  `+z`.

The policy receives:

```text
obs[6] = 2.0  * desired_vx
obs[7] = 2.0  * desired_vy
obs[8] = 0.25 * desired_yaw_rate
```

During training:

- `vx` and `vy` are sampled in `[-1, 1] m/s`.
- Planar commands with a norm less than or equal to `0.2 m/s` are set to zero.
- A new target command is sampled every `10 s`.
- With `heading_command=True`, training samples a target heading and converts
  heading error into a yaw-rate command at every policy step:

```text
cmd_wz = clip(0.5 * wrap_to_pi(target_heading - current_heading), -1, 1)
```

During real deployment:

```text
desired_vx = remote.ly       * 0.8
desired_vy = -remote.lx      * 0.5
desired_wz = -remote.rx      * 1.57
```

After scaling, the real command observation ranges are approximately:

```text
obs[6] in [-1.6, 1.6]
obs[7] in [-1.0, 1.0]
obs[8] in [-0.3925, 0.3925]
```

During training, the scaled yaw command is limited to `[-0.25, 0.25]`.
Maximum joystick yaw on the real robot is therefore outside the training
command distribution.

### 5.4 Joint Position Offset: `obs[9:21]`

This block does not contain absolute joint angles. It contains the difference
from the default pose:

```text
q_error = q_measured - q_default
obs[9:21] = q_error
```

The physical unit before scaling and the numerical unit after scaling are both
radians because `dof_pos_scale = 1.0`.

Joint order:

| Offset | Observation index | Motor | Joint | `q_default` rad |
|---:|---:|---:|---|---:|
| 0 | 9 | 0 | left hip pitch | `-0.1` |
| 1 | 10 | 1 | left hip roll | `0.0` |
| 2 | 11 | 2 | left hip yaw | `0.0` |
| 3 | 12 | 3 | left knee | `0.3` |
| 4 | 13 | 4 | left ankle pitch | `-0.2` |
| 5 | 14 | 5 | left ankle roll | `0.0` |
| 6 | 15 | 6 | right hip pitch | `-0.1` |
| 7 | 16 | 7 | right hip roll | `0.0` |
| 8 | 17 | 8 | right hip yaw | `0.0` |
| 9 | 18 | 9 | right knee | `0.3` |
| 10 | 19 | 10 | right ankle pitch | `-0.2` |
| 11 | 20 | 11 | right ankle roll | `0.0` |

This order comes from the movable-joint order in the URDF and matches the G1
motor indices. It is not determined by the key order in the
`default_joint_angles` dictionary.

### 5.5 Joint Velocity: `obs[21:33]`

The source is the velocity encoder signal for the same 12 joints and in the
same order as the position block:

```text
obs[21:33] = 0.05 * dq_measured
```

The physical unit is `rad/s`.

Examples:

```text
dq = 1.0 rad/s  -> observation = 0.05
dq = 10 rad/s   -> observation = 0.5
```

The deployment code does not estimate velocity using finite differences. It
uses `motor_state[i].dq` directly.

### 5.6 Previous Action: `obs[33:45]`

This block is the raw policy output from the previous policy step:

```text
obs_k[33:45] = action_(k-1)
```

It is not:

- A joint target in radians.
- A measured joint position.
- Torque.
- An action already multiplied by `0.25`.

The new action is converted into a position target using:

```text
q_target_k = q_default + 0.25 * action_k
```

For example, `action_k[0] = 0.4` produces the following left hip pitch target:

```text
q_target = -0.1 + 0.25*0.4 = 0.0 rad
```

Together with the LSTM, the previous action informs the policy about its most
recent control command, helps it infer the dynamic response, and discourages
abrupt action changes.

At the first inference step, the previous action is initialized to a zero
vector.

### 5.7 Gait Phase: `obs[45:47]`

The phase is an internal clock, not a contact signal:

```text
period = 0.8 s
phase = (time mod period) / period
obs[45] = sin(2*pi*phase)
obs[46] = cos(2*pi*phase)
```

Properties:

- Period: `0.8 s`.
- Gait-clock frequency: `1/0.8 = 1.25 Hz`.
- Policy steps per cycle: `0.8/0.02 = 40`.
- Phase increment per policy step: `1/40 = 0.025`.
- The sine/cosine pair remains continuous when phase wraps from `1` to `0`.

Training also creates a right-leg phase offset by `0.5` cycles for contact
reward calculation, but the actor receives only the single sine/cosine pair
shown above.

The clock is not synchronized or reset using foot contact. If a foot touches
the ground early or late, the phase continues according to elapsed time.

## 6. Update Rates

### 6.1 Isaac Gym Training

```text
physics_dt = 0.005 s
physics_frequency = 200 Hz
control_decimation = 4
policy_dt = 4 * 0.005 = 0.02 s
policy_frequency = 50 Hz
```

One action is held for four physics steps. After those four steps:

1. Root state, joint state, and contact state are updated.
2. Angular velocity and projected gravity are calculated.
3. Command and phase are updated.
4. A new observation is assembled.
5. The policy produces the action for the next control interval.

### 6.2 MuJoCo Deployment

```text
simulation_dt = 0.002 s
simulation_frequency = 500 Hz
control_decimation = 10
policy_dt = 10 * 0.002 = 0.02 s
policy_frequency = 50 Hz
```

PD torque is calculated at every 500 Hz physics step. Observation assembly and
neural-network inference run only once every ten physics steps, at 50 Hz.

### 6.3 Python Deployment on the Real Robot

```text
control_dt = 0.02 s
nominal policy frequency = 50 Hz
```

Each `run()` iteration:

1. Reads the latest `low_state` snapshot.
2. Assembles the observation.
3. Runs the LSTM policy.
4. Creates 12 joint-position targets.
5. Sends `LowCmd`.
6. Calls `sleep(0.02)`.

This is a nominal 50 Hz Python loop, not a hard real-time scheduler. Inference,
DDS, and operating-system scheduling can make the actual period longer than
20 ms. The internal IMU and encoder publication rates are not specified in
this repository; the policy samples the latest available snapshot at the
controller rate.

### 6.4 C++ Deployment on the Real Robot

- Observation assembly and policy inference: `50 Hz`.
- Low-command writer thread: `500 Hz`, repeatedly sending the latest command.

Sending commands at 500 Hz does not make the observation rate 500 Hz. The
action changes only once per 20 ms policy step.

### 6.5 Frequency Summary

| Signal or operation | Training | MuJoCo deployment | Real deployment |
|---|---:|---:|---:|
| Physics integration | `200 Hz` | `500 Hz` | Robot firmware; not defined here |
| Observation assembly | `50 Hz` | `50 Hz` | Nominal `50 Hz` |
| Policy/LSTM update | `50 Hz` | `50 Hz` | Nominal `50 Hz` |
| Previous-action update | `50 Hz` | `50 Hz` | Nominal `50 Hz` |
| Phase sample/update | `50 Hz` | `50 Hz` | Nominal `50 Hz` |
| Gait-phase fundamental | `1.25 Hz` | `1.25 Hz` | `1.25 Hz` |
| Python `LowCmd` send | N/A | N/A | Nominal `50 Hz` |
| C++ `LowCmd` send | N/A | N/A | `500 Hz`, repeats latest command |

## 7. Training Noise

Training adds independent uniform noise:

```text
noisy_obs = clean_obs + U(-1, 1) * noise_amplitude
```

| Block | Noise amplitude in observation | Equivalent physical magnitude |
|---|---:|---:|
| Angular velocity | `+/-0.05` | `+/-0.2 rad/s` |
| Projected gravity | `+/-0.05` | dimensionless |
| Command | `0` | `0` |
| Joint position | `+/-0.01` | `+/-0.01 rad` |
| Joint velocity | `+/-0.075` | `+/-1.5 rad/s` |
| Previous action | `0` | `0` |
| Phase sine/cosine | `0` | `0` |

Noise is added only to the 47-D actor observation. The critic's privileged
observation is not modified by noise in `G1Robot.compute_observations()`.

After observation assembly, training clips each value to `[-100, 100]`.
The current deployment code does not clip observations.

## 8. Actor and Critic Observations

The actor that produces actions receives 47 values and does not receive
measured base linear velocity.

During training, the critic receives a 50-D privileged observation:

```text
critic_obs = [
    2.0 * base_linear_velocity_body,   # 3
    actor_obs_without_noise            # 47
]
```

Base linear velocity has the physical unit `m/s` and a scale of `2.0`. This
signal only helps the critic learn the value function. It is not required on
the real robot and is not an input to the actor in `motion.pt`.

## 9. Complete Numerical Example

Assume the following signals at `t = 0.20 s`:

```text
gyro [rad/s] = [0.20, -0.10, 0.40]

pelvis orientation = roll +10 deg
quaternion [w,x,y,z] = [0.996195, 0.087156, 0, 0]

command = [0.40 m/s, -0.20 m/s, 0.60 rad/s]

q - q_default [rad] =
[ 0.05, -0.02,  0.03, -0.10,  0.04, -0.01,
 -0.04,  0.01, -0.02,  0.08, -0.03,  0.02]

dq [rad/s] =
[ 0.40, -0.20,  0.10, -1.00,  0.60, -0.30,
 -0.50,  0.20, -0.10,  0.80, -0.40,  0.30]

previous_action =
[ 0.12, -0.08,  0.03,  0.25, -0.16,  0.04,
 -0.10,  0.06, -0.02,  0.20, -0.12,  0.01]
```

Individual blocks:

```text
scaled gyro =
[0.050000, -0.025000, 0.100000]

projected gravity =
[0.000000, -0.173648, -0.984808]

scaled command =
[0.800000, -0.400000, 0.150000]

scaled dq =
[ 0.020000, -0.010000,  0.005000, -0.050000,
  0.030000, -0.015000, -0.025000,  0.010000,
 -0.005000,  0.040000, -0.020000,  0.015000]

phase = 0.20 / 0.8 = 0.25
sin(2*pi*phase) = 1
cos(2*pi*phase) = 0
```

The final 47-D observation is:

```text
[
  0.050000, -0.025000,  0.100000,
  0.000000, -0.173648, -0.984808,
  0.800000, -0.400000,  0.150000,

  0.050000, -0.020000,  0.030000, -0.100000,
  0.040000, -0.010000, -0.040000,  0.010000,
 -0.020000,  0.080000, -0.030000,  0.020000,

  0.020000, -0.010000,  0.005000, -0.050000,
  0.030000, -0.015000, -0.025000,  0.010000,
 -0.005000,  0.040000, -0.020000,  0.015000,

  0.120000, -0.080000,  0.030000,  0.250000,
 -0.160000,  0.040000, -0.100000,  0.060000,
 -0.020000,  0.200000, -0.120000,  0.010000,

  1.000000,  0.000000
]
```

### Python Code for the Example

```python
import numpy as np


def get_gravity_orientation(quaternion_wxyz):
    qw, qx, qy, qz = quaternion_wxyz
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def build_g1_observation(
    gyro,
    quaternion_wxyz,
    command,
    joint_position,
    joint_velocity,
    previous_action,
    time_s,
):
    default_angles = np.array(
        [
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        ],
        dtype=np.float32,
    )

    gyro = np.asarray(gyro, dtype=np.float32)
    command = np.asarray(command, dtype=np.float32)
    joint_position = np.asarray(joint_position, dtype=np.float32)
    joint_velocity = np.asarray(joint_velocity, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)

    phase = (time_s % 0.8) / 0.8

    obs = np.concatenate(
        [
            gyro * 0.25,
            get_gravity_orientation(quaternion_wxyz),
            command * np.array([2.0, 2.0, 0.25], dtype=np.float32),
            joint_position - default_angles,
            joint_velocity * 0.05,
            previous_action,
            np.array(
                [
                    np.sin(2.0 * np.pi * phase),
                    np.cos(2.0 * np.pi * phase),
                ],
                dtype=np.float32,
            ),
        ]
    )

    assert obs.shape == (47,)
    return obs


q_default = np.array(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    ],
    dtype=np.float32,
)

q_offset = np.array(
    [
        0.05, -0.02, 0.03, -0.10, 0.04, -0.01,
        -0.04, 0.01, -0.02, 0.08, -0.03, 0.02,
    ],
    dtype=np.float32,
)

obs = build_g1_observation(
    gyro=[0.20, -0.10, 0.40],
    quaternion_wxyz=[0.9961947, 0.0871557, 0.0, 0.0],
    command=[0.40, -0.20, 0.60],
    joint_position=q_default + q_offset,
    joint_velocity=[
        0.40, -0.20, 0.10, -1.00, 0.60, -0.30,
        -0.50, 0.20, -0.10, 0.80, -0.40, 0.30,
    ],
    previous_action=[
        0.12, -0.08, 0.03, 0.25, -0.16, 0.04,
        -0.10, 0.06, -0.02, 0.20, -0.12, 0.01,
    ],
    time_s=0.20,
)

print(obs.shape)
print(obs)
```

Output:

```text
(47,)
[ 0.05     -0.025     0.1       0.       -0.173648 -0.984808
  0.8      -0.4       0.15      0.05     -0.02      0.03
 -0.1       0.04     -0.01     -0.04      0.01     -0.02
  0.08     -0.03      0.02      0.02     -0.01      0.005
 -0.05      0.03     -0.015    -0.025     0.01     -0.005
  0.04     -0.02      0.015     0.12     -0.08      0.03
  0.25     -0.16      0.04     -0.1       0.06     -0.02
  0.2      -0.12      0.01      1.        0.      ]
```

## 10. Timing of One Policy Step

At time `t_k`, observations and actions are related as follows:

```text
state/sensors at t_k
       |
       v
o_k = [sensor_k, command_k, action_(k-1), phase_k]
       |
       v
(action_k, hidden_k, cell_k) =
    LSTMPolicy(o_k, hidden_(k-1), cell_(k-1))
       |
       v
q_target_k = q_default + 0.25*action_k
       |
       v
PD controller holds q_target_k until the next policy step
```

Because the observation contains `action_(k-1)`, `action_k` must not be written
into `obs[33:45]` before inference.

The ROS controller treats each policy step as a prepare/commit operation. The
candidate phase, LSTM update, previous action, and policy counter are committed
only after inference succeeds. If inference fails, LSTM memory is restored and
the next valid sensor callback retries the same `phase_k`.

## 11. Invariants for a New Controller

1. The input tensor must have shape `[batch, 47]`, normally `[1, 47]`.
2. The input dtype must be `float32`.
3. The block order and 12-joint order must remain unchanged.
4. Gyroscope and projected gravity must be expressed in the pelvis/body frame.
5. The deployment quaternion must use the `[w, x, y, z]` convention.
6. Joint positions must have `q_default` subtracted before policy inference.
7. Previous action must be the raw neural-network output from the prior step.
8. Observation assembly and LSTM inference should run at approximately `50 Hz`.
9. The phase period must remain `0.8 s`.
10. Reset the following values together when resetting the controller or robot:

```text
previous_action = zeros(12)
phase/counter = 0
policy.reset_memory()
```

11. Do not insert the 17 waist/arm joints into the 47-D observation.
12. Changing a scale, joint order, or update rate breaks the original
    observation contract and can destabilize the existing policy.
13. Do not advance the phase or policy counter when inference fails.

## 12. Current Deployment Differences and Risks

### 12.1 LSTM State Is Not Reset

The exported TorchScript policy provides `reset_memory()`, but the current
deployment scripts do not call it. If the robot is moved back to its default
pose while retaining the old hidden state, the first new action still contains
history from the previous motion.

### 12.2 Python and C++ Phase Differ by One Policy Tick

Python increments its counter before calculating phase, so its first inference
uses:

```text
phase = 0.02 / 0.8 = 0.025
```

C++ calculates phase before incrementing `time`, so its first inference uses:

```text
phase = 0
```

The difference is `0.02 s`, or one policy step.

### 12.3 Real Yaw Command Exceeds the Training Range

Training limits `cmd_wz` to `+/-1 rad/s`, while the real configuration permits
`+/-1.57 rad/s`. This intentionally introduces an observation distribution
shift at large joystick inputs.

### 12.4 The Real Python Loop Does Not Compensate for Processing Time

The code always calls `sleep(control_dt)` after inference. Its actual period is:

```text
T_actual = T_read + T_build_obs + T_inference + T_DDS + 0.02
```

The actual frequency can therefore be lower than 50 Hz. The MuJoCo loop
partially compensates for processing time using `time_until_next_step`; the
real Python loop does not.

### 12.5 Observation Has No Direct Contact Feedback

The policy does not directly know which foot is touching the ground. It must
infer contact from:

- Gait clock.
- Joint position and velocity.
- IMU angular velocity.
- Projected gravity.
- LSTM history.

This makes timing, LSTM state, and joint-state signals particularly important.

## 13. Important Source Files

- Actor and critic observation assembly:
  `legged_gym/envs/g1/g1_env.py`
- Scaling, noise, and simulation timestep:
  `legged_gym/envs/base/legged_robot_config.py`
- Policy/control timestep and PD control:
  `legged_gym/envs/base/legged_robot.py`
- G1 action count, observation count, and LSTM configuration:
  `legged_gym/envs/g1/g1_config.py`
- MuJoCo observation assembly:
  `deploy/deploy_mujoco/deploy_mujoco.py`
- Real Python observation assembly:
  `deploy/deploy_real/deploy_real.py`
- Real Python scaling and motor mapping:
  `deploy/deploy_real/configs/g1.yaml`
- Real C++ observation assembly:
  `deploy/deploy_real/cpp_g1/Controller.cpp`
- LSTM TorchScript exporter:
  `legged_gym/utils/helpers.py`
