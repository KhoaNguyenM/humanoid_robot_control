# ROS 2 Joint-State Processing for G1 Policy Observation

This document explains how to read `/joint/joint_states` and construct the
joint-related sections of the Unitree G1 locomotion-policy observation:

```text
obs[9:21]  = joint position offsets for 12 leg joints
obs[21:33] = scaled joint velocities for 12 leg joints
```

The ROS 2 topic publishes 43 joints, while the policy uses only 12 leg joints.
The required joints must be selected and reordered by name. The raw array order
published by ROS 2 must not be used as the policy order.

## 1. Policy Contract

The policy expects:

```text
obs[9:21]  = 1.0  * (q_policy_order - q_default)
obs[21:33] = 0.05 * dq_policy_order
```

| Observation indices | Size | Signal | Physical unit | Scale |
|---|---:|---|---|---:|
| `9:21` | 12 | Joint position relative to default pose | `rad` | `1.0` |
| `21:33` | 12 | Joint angular velocity | `rad/s` | `0.05` |

Both output blocks are dimensionless neural-network inputs after scaling.
Because the position scale is `1.0`, its numerical values remain equal to the
position offsets in radians.

The policy does not use:

- Waist joint positions or velocities.
- Arm joint positions or velocities.
- Hand joint positions or velocities.
- Joint effort from this topic.

## 2. ROS 2 Topic

The measured ROS 2 graph contains:

```text
Topic:          /joint/joint_states
Message type:   sensor_msgs/msg/JointState
Publisher:      1
Reliability:    RELIABLE
Durability:     VOLATILE
```

The message structure is:

```yaml
header:
  stamp:
    sec: ...
    nanosec: ...
  frame_id: ""

name:
- joint_name_0
- joint_name_1
- ...

position:
- position_0
- position_1
- ...

velocity:
- velocity_0
- velocity_1
- ...

effort:
- effort_0
- effort_1
- ...
```

For each array index `i`:

```text
name[i]
position[i]
velocity[i]
effort[i]
```

refer to the same joint.

The measured message had:

```text
name length:      43
position length:  43
velocity length:  43
effort length:    43
unique names:     43
duplicate names:  0
```

## 3. Units

All 12 policy joints are revolute joints. According to
`sensor_msgs/msg/JointState`:

- Position is in radians: `rad`.
- Velocity is in radians per second: `rad/s`.
- Effort is in newton-metres: `N*m`.

No degree-to-radian conversion is required.

The policy uses only position and velocity:

```text
q  = msg.position[index]  # rad
dq = msg.velocity[index]  # rad/s
```

The `effort` array is ignored when constructing this policy observation.

## 4. Required 12-Joint Policy Order

The policy joint order comes from the movable-joint order in
`g1_12dof.urdf`:

| Policy offset | Observation position index | Observation velocity index | Joint name | Default angle |
|---:|---:|---:|---|---:|
| 0 | 9 | 21 | `left_hip_pitch_joint` | `-0.1 rad` |
| 1 | 10 | 22 | `left_hip_roll_joint` | `0.0 rad` |
| 2 | 11 | 23 | `left_hip_yaw_joint` | `0.0 rad` |
| 3 | 12 | 24 | `left_knee_joint` | `0.3 rad` |
| 4 | 13 | 25 | `left_ankle_pitch_joint` | `-0.2 rad` |
| 5 | 14 | 26 | `left_ankle_roll_joint` | `0.0 rad` |
| 6 | 15 | 27 | `right_hip_pitch_joint` | `-0.1 rad` |
| 7 | 16 | 28 | `right_hip_roll_joint` | `0.0 rad` |
| 8 | 17 | 29 | `right_hip_yaw_joint` | `0.0 rad` |
| 9 | 18 | 30 | `right_knee_joint` | `0.3 rad` |
| 10 | 19 | 31 | `right_ankle_pitch_joint` | `-0.2 rad` |
| 11 | 20 | 32 | `right_ankle_roll_joint` | `0.0 rad` |

In code:

```python
POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

DEFAULT_JOINT_ANGLES = np.array(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    ],
    dtype=np.float32,
)
```

## 5. Published Order Is Not Policy Order

The measured ROS 2 message publishes waist, arm, leg, wrist, and hand joints in
an interleaved order.

For example, the first 12 published names were:

```text
0  waist_pitch_joint
1  left_shoulder_pitch_joint
2  right_shoulder_pitch_joint
3  waist_roll_joint
4  left_shoulder_roll_joint
5  right_shoulder_roll_joint
6  waist_yaw_joint
7  left_shoulder_yaw_joint
8  right_shoulder_yaw_joint
9  left_hip_pitch_joint
10 right_hip_pitch_joint
11 left_elbow_joint
```

Therefore, this is incorrect:

```python
q = np.asarray(msg.position[:12])
dq = np.asarray(msg.velocity[:12])
```

Those slices contain mostly waist and arm joints, not the 12 policy leg joints.

The measured message indices for the required policy order were:

```text
[9, 13, 17, 21, 25, 27, 10, 14, 18, 22, 26, 28]
```

However, these numeric indices should not be hard-coded. A publisher can change
its array order while preserving the same joint names. Always derive the
mapping from `msg.name`.

## 6. Name-Based Extraction

Create a mapping from joint name to message index:

```python
name_to_index = {
    name: index
    for index, name in enumerate(msg.name)
}
```

Then extract each required joint in policy order:

```python
q = np.array(
    [
        msg.position[name_to_index[name]]
        for name in POLICY_JOINT_NAMES
    ],
    dtype=np.float32,
)

dq = np.array(
    [
        msg.velocity[name_to_index[name]]
        for name in POLICY_JOINT_NAMES
    ],
    dtype=np.float32,
)
```

This approach deliberately ignores all 31 non-policy joints.

## 7. Joint Position Offset: `obs[9:21]`

The policy does not receive absolute joint positions. It receives the position
relative to the training default pose:

```text
q_offset = q_measured - q_default
```

The position scale is:

```text
dof_pos_scale = 1.0
```

Therefore:

```python
obs[9:21] = (q - DEFAULT_JOINT_ANGLES) * 1.0
```

Examples:

```text
left hip pitch:
q_measured = -0.05 rad
q_default  = -0.10 rad
observation = -0.05 - (-0.10) = +0.05

left knee:
q_measured = 0.40 rad
q_default  = 0.30 rad
observation = 0.40 - 0.30 = +0.10

right ankle pitch:
q_measured = -0.25 rad
q_default  = -0.20 rad
observation = -0.25 - (-0.20) = -0.05
```

Do not subtract zero from every joint. Hip pitch, knee, and ankle pitch have
non-zero default angles.

## 8. Joint Velocity: `obs[21:33]`

The policy receives measured joint velocity multiplied by:

```text
dof_vel_scale = 0.05
```

Therefore:

```python
obs[21:33] = 0.05 * dq
```

Examples:

```text
dq =  1.0 rad/s -> observation =  0.05
dq = -2.0 rad/s -> observation = -0.10
dq = 10.0 rad/s -> observation =  0.50
```

Use `msg.velocity` directly. Do not calculate finite differences from position
unless the publisher does not provide velocity.

The measured message provides a velocity array with all 43 entries, so no
finite-difference estimator is needed.

## 9. Processing the Measured Sample

For the measured message, name-based extraction produced:

| Policy offset | Message index | Joint | Position | Default | `obs` position | Velocity | `obs` velocity |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 9 | left hip pitch | `-0.0814` | `-0.1` | `0.0186` | `0.0` | `0.0` |
| 1 | 13 | left hip roll | `-0.0054` | `0.0` | `-0.0054` | `0.0` | `0.0` |
| 2 | 17 | left hip yaw | `-0.0139` | `0.0` | `-0.0139` | `0.0` | `0.0` |
| 3 | 21 | left knee | `0.2984` | `0.3` | `-0.0016` | `0.0` | `0.0` |
| 4 | 25 | left ankle pitch | `-0.2067` | `-0.2` | `-0.0067` | `0.0` | `0.0` |
| 5 | 27 | left ankle roll | `0.0083` | `0.0` | `0.0083` | `0.0` | `0.0` |
| 6 | 10 | right hip pitch | `-0.0973` | `-0.1` | `0.0027` | `0.0` | `0.0` |
| 7 | 14 | right hip roll | `0.0015` | `0.0` | `0.0015` | `0.0` | `0.0` |
| 8 | 18 | right hip yaw | `0.0012` | `0.0` | `0.0012` | `0.0` | `0.0` |
| 9 | 22 | right knee | `0.2990` | `0.3` | `-0.0010` | `0.0` | `0.0` |
| 10 | 26 | right ankle pitch | `-0.1998` | `-0.2` | `0.0002` | `0.0` | `0.0` |
| 11 | 28 | right ankle roll | `0.0007` | `0.0` | `0.0007` | `0.0` | `0.0` |

The resulting position block was:

```text
obs[9:21] =
[
  0.0186, -0.0054, -0.0139, -0.0016,
 -0.0067,  0.0083,  0.0027,  0.0015,
  0.0012, -0.0010,  0.0002,  0.0007
]
```

The robot was stationary, so the velocity block was:

```text
obs[21:33] =
[
  0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0
]
```

A stationary sample confirms array availability but does not validate dynamic
velocity signs or magnitudes. Controlled joint motion is required for that
test.

## 10. Measured Publish Rate

Five hundred `/joint/joint_states` messages were measured.

### Simulation Timestamp Rate

```text
Header timestamp interval: 0.005 s
Simulation-time rate:      200.000 Hz
Timestamp jitter:          negligible
```

### Wall-Clock Receive Rate

During the measurement:

```text
Wall-clock receive rate: approximately 80-86 Hz
Wall-clock interval:     approximately 10-17 ms
```

The simulator was running slower than real time. The message timestamps still
advanced by exactly `0.005 s`.

As with `/imu`, distinguish between:

```text
Simulation-time joint-state rate = 200 Hz
Wall-clock receive rate          = simulator-dependent
```

## 11. Synchronization with `/imu`

The IMU and joint-state topics were measured simultaneously:

```text
/imu samples:                 250
/joint/joint_states samples:  250
matching timestamps:          250
timestamp difference:         0.0 s
```

The measured publishers therefore provide IMU and joint-state snapshots at the
same 200 Hz simulation timestamps.

For a complete observation, use IMU and joint-state data from the same timestamp
when possible. At minimum, verify that their timestamp difference is much
smaller than the policy interval:

```text
policy_dt = 0.02 s
```

## 12. Policy Sampling at 50 Hz

The policy runs at:

```text
policy_dt = 0.02 s
policy_frequency = 50 Hz
```

The joint-state topic advances at:

```text
joint_state_dt = 0.005 s
joint_state_frequency = 200 Hz
```

There are four joint-state samples per policy interval:

```text
0.02 / 0.005 = 4
```

The controller should retain the latest validated joint state and update the
policy observation every `0.02 s` of simulation time.

Do not run the LSTM at every 200 Hz joint-state callback.

Use message timestamps or `/clock` for scheduling:

```python
if current_stamp - last_policy_stamp >= 0.02:
    run_policy_with_latest_sensor_data()
    last_policy_stamp = current_stamp
```

## 13. Complete Extraction Function

The following function selects only the 12 required joints, orders them
correctly, and produces the 24 policy values for `obs[9:33]`.

```python
import numpy as np


POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

DEFAULT_JOINT_ANGLES = np.array(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    ],
    dtype=np.float32,
)

DOF_POSITION_SCALE = 1.0
DOF_VELOCITY_SCALE = 0.05


def joint_state_to_policy_observation(msg):
    """Build obs[9:33] from sensor_msgs.msg.JointState."""
    name_count = len(msg.name)

    if len(msg.position) != name_count:
        raise ValueError(
            "JointState position array must match the name array"
        )
    if len(msg.velocity) != name_count:
        raise ValueError(
            "JointState velocity array must match the name array"
        )

    if len(set(msg.name)) != name_count:
        raise ValueError("JointState contains duplicate joint names")

    name_to_index = {
        name: index
        for index, name in enumerate(msg.name)
    }

    missing = [
        name
        for name in POLICY_JOINT_NAMES
        if name not in name_to_index
    ]
    if missing:
        raise ValueError(f"Missing policy joints: {missing}")

    q = np.array(
        [
            msg.position[name_to_index[name]]
            for name in POLICY_JOINT_NAMES
        ],
        dtype=np.float32,
    )

    dq = np.array(
        [
            msg.velocity[name_to_index[name]]
            for name in POLICY_JOINT_NAMES
        ],
        dtype=np.float32,
    )

    position_observation = (
        q - DEFAULT_JOINT_ANGLES
    ) * DOF_POSITION_SCALE

    velocity_observation = dq * DOF_VELOCITY_SCALE

    observation = np.concatenate(
        [
            position_observation,
            velocity_observation,
        ]
    ).astype(np.float32)

    if observation.shape != (24,):
        raise RuntimeError("Expected a 24-element joint observation")
    if not np.all(np.isfinite(observation)):
        raise ValueError("Joint observation contains NaN or infinity")

    return observation
```

Usage in the complete 47-D observation:

```python
joint_observation = joint_state_to_policy_observation(msg)

obs[9:21] = joint_observation[0:12]
obs[21:33] = joint_observation[12:24]
```

## 14. Cached Mapping for a High-Rate Controller

Building a dictionary at 200 Hz is usually acceptable, but a controller can
cache the mapping after validating the first message.

The cache must be rebuilt if `msg.name` changes:

```python
class PolicyJointExtractor:
    def __init__(self):
        self.cached_names = None
        self.policy_indices = None

    def update_mapping(self, names):
        names_tuple = tuple(names)
        if names_tuple == self.cached_names:
            return

        if len(set(names_tuple)) != len(names_tuple):
            raise ValueError("Duplicate joint names")

        name_to_index = {
            name: index
            for index, name in enumerate(names_tuple)
        }

        missing = [
            name
            for name in POLICY_JOINT_NAMES
            if name not in name_to_index
        ]
        if missing:
            raise ValueError(f"Missing policy joints: {missing}")

        self.policy_indices = np.array(
            [
                name_to_index[name]
                for name in POLICY_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self.cached_names = names_tuple

    def extract(self, msg):
        self.update_mapping(msg.name)

        if len(msg.position) != len(msg.name):
            raise ValueError("Invalid position array length")
        if len(msg.velocity) != len(msg.name):
            raise ValueError("Invalid velocity array length")

        all_positions = np.asarray(msg.position, dtype=np.float32)
        all_velocities = np.asarray(msg.velocity, dtype=np.float32)

        q = all_positions[self.policy_indices]
        dq = all_velocities[self.policy_indices]

        return np.concatenate(
            [
                q - DEFAULT_JOINT_ANGLES,
                0.05 * dq,
            ]
        ).astype(np.float32)
```

## 15. ROS 2 Subscriber Example

This node retains the latest joint state and constructs `obs[9:33]` at 50 Hz
of simulation time.

```python
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState


POLICY_DT = 0.02


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class G1JointObservationNode(Node):
    def __init__(self):
        super().__init__("g1_joint_observation")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.extractor = PolicyJointExtractor()
        self.last_policy_stamp = None

        self.create_subscription(
            JointState,
            "/joint/joint_states",
            self.joint_state_callback,
            qos,
        )

    def joint_state_callback(self, msg):
        current_stamp = stamp_to_seconds(msg.header.stamp)

        if self.last_policy_stamp is None:
            should_update = True
        else:
            should_update = (
                current_stamp - self.last_policy_stamp >= POLICY_DT - 1e-9
            )

        if not should_update:
            return

        joint_observation = self.extractor.extract(msg)

        position_observation = joint_observation[0:12]
        velocity_observation = joint_observation[12:24]

        self.last_policy_stamp = current_stamp

        self.get_logger().info(
            "obs[9:21] = "
            + np.array2string(position_observation, precision=6)
        )
        self.get_logger().info(
            "obs[21:33] = "
            + np.array2string(velocity_observation, precision=6)
        )


def main():
    rclpy.init()
    node = G1JointObservationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

In a complete policy controller, avoid logging at every policy step. Write the
result into a shared `float32` observation buffer instead.

## 16. Dynamic Validation

The measured robot was stationary, so all velocity values were zero. Validate
each joint dynamically before policy deployment.

For one joint at a time:

1. Command or manually move the joint in its positive URDF direction.
2. Confirm that the matching `msg.position` increases.
3. Confirm that `msg.velocity` is positive during the positive motion.
4. Confirm that only the intended policy element responds significantly.
5. Return the joint to the default pose and check that its position observation
   returns close to zero.

Example for left hip pitch:

```text
Joint name:                 left_hip_pitch_joint
Policy position element:    obs[9]
Policy velocity element:    obs[21]
Default angle:              -0.1 rad
```

If the measured position is `-0.05 rad` and velocity is `+0.4 rad/s`:

```text
obs[9]  = -0.05 - (-0.1) = +0.05
obs[21] = 0.05 * 0.4     = +0.02
```

## 17. Common Errors

### Taking the First 12 Published Joints

Incorrect:

```python
q = msg.position[:12]
```

Correct:

```python
q = [
    msg.position[name_to_index[name]]
    for name in POLICY_JOINT_NAMES
]
```

### Using Left-Right Interleaved Order

The ROS publisher places some left and right joints next to each other. The
policy expects all six left-leg joints followed by all six right-leg joints.

### Forgetting Default Position Subtraction

Incorrect:

```python
obs[9:21] = q
```

Correct:

```python
obs[9:21] = q - DEFAULT_JOINT_ANGLES
```

### Scaling Position by `0.05`

Incorrect:

```python
obs[9:21] = 0.05 * (q - q_default)
```

Correct:

```python
obs[9:21] = 1.0 * (q - q_default)
```

The `0.05` scale belongs only to velocity.

### Using Effort Instead of Velocity

Incorrect:

```python
obs[21:33] = 0.05 * msg.effort
```

Correct:

```python
obs[21:33] = 0.05 * selected_joint_velocities
```

### Running the Policy at 200 Hz

Incorrect:

```text
One LSTM inference for every joint-state callback
```

Correct:

```text
Retain the latest sample and infer at 50 Hz simulation-time
```

## 18. Final Checklist

Before using `/joint/joint_states` for policy inference, confirm:

- [ ] The topic type is `sensor_msgs/msg/JointState`.
- [ ] All `name`, `position`, and `velocity` arrays have matching lengths.
- [ ] Joint names are unique.
- [ ] All 12 required policy joint names are present.
- [ ] Joints are selected by name, not raw message index.
- [ ] The output order is six left-leg joints followed by six right-leg joints.
- [ ] Waist, arm, wrist, and hand joints are ignored.
- [ ] Joint positions are interpreted in radians.
- [ ] Joint velocities are interpreted in radians per second.
- [ ] The correct default angle is subtracted from each joint position.
- [ ] Position offsets are multiplied by `1.0`.
- [ ] Joint velocities are multiplied by `0.05`.
- [ ] The output dtype is `float32`.
- [ ] `obs[9:21]` has exactly 12 values.
- [ ] `obs[21:33]` has exactly 12 values.
- [ ] Joint states are paired with an IMU sample from the same timestamp when
      possible.
- [ ] Policy inference runs at `50 Hz` simulation-time.
- [ ] Controlled motion tests confirm every joint sign and velocity value.

Once these conditions are satisfied, `/joint/joint_states` provides all
information required to construct the joint-position and joint-velocity
sections of the G1 policy observation.
