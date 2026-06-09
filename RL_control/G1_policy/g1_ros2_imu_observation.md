# ROS 2 IMU Processing for G1 Policy Observation

This document explains how to read the ROS 2 `/imu` topic and construct the
first six values of the Unitree G1 locomotion-policy observation:

```text
obs[0:3] = scaled pelvis angular velocity
obs[3:6] = projected gravity in the pelvis frame
```

The policy input must match the observation convention used during training.
The IMU topic contains enough raw information for these six values, provided
that its coordinate frame is aligned with, or transformed into, the pelvis
frame.

## 1. Policy Contract

The first six policy inputs are:

```text
obs[0:6] = [
    0.25 * omega_pelvis_x,
    0.25 * omega_pelvis_y,
    0.25 * omega_pelvis_z,
    gravity_pelvis_x,
    gravity_pelvis_y,
    gravity_pelvis_z,
]
```

| Indices | Signal | Physical unit | Scale | Policy input unit |
|---|---|---|---:|---|
| `0:3` | Pelvis angular velocity | `rad/s` | `0.25` | Dimensionless |
| `3:6` | Gravity direction in pelvis frame | Unit vector | `1.0` | Dimensionless |

The expected pelvis-axis convention is:

```text
+x: robot forward
+y: robot left
+z: robot upward when standing
```

Relevant training code:

```python
self.base_ang_vel * self.obs_scales.ang_vel  # scale = 0.25
self.projected_gravity
```

## 2. ROS 2 Topic

The measured ROS 2 graph contains:

```text
Topic:          /imu
Message type:   sensor_msgs/msg/Imu
Publisher:      1
Frame ID:       pelvis_imu
Reliability:    RELIABLE
Durability:     VOLATILE
```

The relevant message fields are:

```yaml
header:
  stamp:
    sec: ...
    nanosec: ...
  frame_id: pelvis_imu

orientation:
  x: ...
  y: ...
  z: ...
  w: ...

angular_velocity:
  x: ...
  y: ...
  z: ...
```

The `linear_acceleration` field is not needed to build `obs[0:6]`.

## 3. Units and Quaternion Convention

According to `sensor_msgs/msg/Imu`:

- `angular_velocity` is measured in `rad/s`.
- `linear_acceleration` is measured in `m/s^2`.
- `orientation` is a unit quaternion.

ROS stores quaternion fields in this order:

```text
[x, y, z, w]
```

The existing Unitree deployment helper expects:

```text
[w, x, y, z]
```

Therefore, when calling a helper that expects `wxyz`, reorder the ROS fields:

```python
quaternion_wxyz = np.array(
    [
        msg.orientation.w,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
    ],
    dtype=np.float32,
)
```

Do not pass the ROS `xyzw` order directly to a function that expects `wxyz`.

## 4. Coordinate-Frame Requirement

The policy requires both signals in the pelvis frame:

```text
omega_pelvis
gravity_pelvis
```

The ROS message declares:

```text
header.frame_id = "pelvis_imu"
```

There are two possible cases.

### 4.1 `pelvis_imu` Is Aligned with the Pelvis Frame

If the IMU local axes satisfy:

```text
pelvis_imu +x = pelvis +x
pelvis_imu +y = pelvis +y
pelvis_imu +z = pelvis +z
```

then angular velocity can be used directly:

```python
omega_pelvis = np.array(
    [
        msg.angular_velocity.x,
        msg.angular_velocity.y,
        msg.angular_velocity.z,
    ],
    dtype=np.float32,
)
```

The orientation quaternion can also be used directly to calculate projected
gravity.

### 4.2 `pelvis_imu` Is Rotated Relative to the Pelvis

If the IMU prim has a non-identity mounting rotation, both angular velocity and
orientation must be transformed into the pelvis frame.

Let:

```text
R_P_I = rotation that maps vectors from IMU frame I to pelvis frame P
```

Then:

```text
omega_P = R_P_I * omega_I
gravity_P = R_P_I * gravity_I
```

The same static mounting rotation must be applied to both three-dimensional
signals. Transforming only angular velocity or only gravity creates an
inconsistent observation.

The transform should come from one of these sources:

- An identity mounting configuration verified in the Isaac Sim stage.
- A static transform published through `/tf_static`.
- A known fixed rotation configured directly in the observation node.

The current topic list does not contain `/tf` or `/tf_static`. If `pelvis_imu` is
not aligned with the pelvis, the observation node must be given the mounting
rotation explicitly or a TF publisher must be added.

## 5. Pelvis Angular Velocity: `obs[0:3]`

Read:

```python
omega_imu = np.array(
    [
        msg.angular_velocity.x,
        msg.angular_velocity.y,
        msg.angular_velocity.z,
    ],
    dtype=np.float32,
)
```

Transform it when required:

```python
omega_pelvis = rotation_pelvis_from_imu @ omega_imu
```

Apply the policy scale:

```python
obs[0:3] = 0.25 * omega_pelvis
```

Examples:

```text
omega_pelvis = [0, 0, 0] rad/s
obs[0:3]     = [0, 0, 0]

omega_pelvis = [0, 0, 1] rad/s
obs[0:3]     = [0, 0, 0.25]

omega_pelvis = [0.4, -0.2, 0.8] rad/s
obs[0:3]     = [0.1, -0.05, 0.2]
```

No degree-to-radian conversion should be performed. The ROS message already
uses `rad/s`.

### Axis Interpretation

With pelvis-aligned axes:

- Positive `omega_x`: positive roll rate.
- Positive `omega_y`: positive pitch rate.
- Positive `omega_z`: positive yaw rate.

The sign and axis order should be verified with controlled motions:

1. Rotate around pelvis `+x`; only `angular_velocity.x` should dominate.
2. Rotate around pelvis `+y`; only `angular_velocity.y` should dominate.
3. Rotate around pelvis `+z`; only `angular_velocity.z` should dominate.

## 6. Projected Gravity: `obs[3:6]`

The policy does not use the quaternion directly. It uses the world gravity
direction expressed in the pelvis frame:

```text
gravity_world = [0, 0, -1]
gravity_pelvis = R_world_to_pelvis * gravity_world
```

This is a dimensionless unit vector:

```text
||gravity_pelvis|| approximately 1
```

It is not the accelerometer reading and must not be multiplied by `9.81`.

### 6.1 Direct Formula from a ROS Quaternion

For a ROS quaternion:

```text
qx = msg.orientation.x
qy = msg.orientation.y
qz = msg.orientation.z
qw = msg.orientation.w
```

calculate:

```python
gravity_imu = np.array(
    [
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ],
    dtype=np.float32,
)
```

If the IMU axes and pelvis axes are aligned:

```python
gravity_pelvis = gravity_imu
```

Otherwise:

```python
gravity_pelvis = rotation_pelvis_from_imu @ gravity_imu
```

Finally:

```python
obs[3:6] = gravity_pelvis
```

No additional observation scale is applied.

### 6.2 Why `linear_acceleration` Is Not Used

Projected gravity is an orientation-derived direction vector. It remains useful
while the robot accelerates because it comes from the orientation estimate,
not directly from the instantaneous accelerometer vector.

Using normalized `linear_acceleration` as projected gravity is incorrect during
dynamic motion because it contains both gravity and body acceleration.

## 7. Expected Gravity for Different Robot Poses

Assuming the IMU axes are aligned with the pelvis axes:

| Robot pose | Approximate `gravity_pelvis` |
|---|---|
| Standing upright | `[0, 0, -1]` |
| Upside down | `[0, 0, +1]` |
| Rolled by `+10 deg` | `[0, -0.17365, -0.98481]` |
| Pitched by `+10 deg` | `[+0.17365, 0, -0.98481]` |
| Face down, about `+90 deg` pitch | Approximately `[+1, 0, 0]` |
| Face up, about `-90 deg` pitch | Approximately `[-1, 0, 0]` |

The measured sample was:

```text
orientation [x, y, z, w] =
[0.101190, 0.684975, -0.214107, 0.689005]

gravity_imu =
[0.987234, 0.153874, -0.041138]
```

Its magnitude is approximately one:

```text
||gravity_imu|| approximately 1.0
```

The dominant positive `x` component is consistent with a robot lying face down
with its pelvis forward axis pointing mostly toward the ground. This sample is
therefore not evidence of an invalid quaternion.

## 8. Measured Publish Rate

Five hundred `/imu` messages were measured.

### Simulation Timestamp

```text
Header timestamp interval: 0.005 s
Simulation-time rate:      200.000 Hz
Timestamp jitter:          negligible
```

The IMU data is therefore generated at the Isaac Gym training physics rate:

```text
physics_dt = 0.005 s
physics_frequency = 200 Hz
```

### Wall-Clock Reception

During the measurement:

```text
Wall-clock receive rate: approximately 85-90 Hz
Wall-clock interval:     approximately 10-20 ms
```

`/clock` showed a similar wall-clock rate. The simulator was running slower
than real time, while the message timestamps still advanced by exactly
`0.005 s`.

These are two different quantities:

```text
Simulation-time IMU rate = 200 Hz
Wall-clock receive rate  = simulator-dependent
```

## 9. Policy Sampling at 50 Hz

The policy was trained with:

```text
policy_dt = 0.02 s
policy_frequency = 50 Hz
```

Since `/imu` advances by `0.005 s` per sample, there are four IMU samples per
policy interval:

```text
0.02 / 0.005 = 4
```

The observation controller should use the newest IMU sample every `0.02 s` of
simulation time.

Do not infer the policy on every `/imu` callback. Doing so would run the LSTM at
200 Hz simulation-time instead of its trained 50 Hz rate.

### Recommended Scheduling

Use the message timestamp or `/clock`, not only wall-clock `sleep()`:

```python
if current_stamp - last_policy_stamp >= 0.02:
    run_policy_with_latest_imu()
    last_policy_stamp = current_stamp
```

This remains correct when Isaac Sim runs slower or faster than real time.

Simply processing every fourth callback also works when no messages are
dropped, but timestamp-based scheduling is more robust.

## 10. Covariance and Signal Validity

The measured message contains all-zero covariance matrices.

For `sensor_msgs/msg/Imu`, an all-zero covariance means:

```text
Covariance is unknown or not provided.
```

It does not mean that the sensor has zero noise.

Before using orientation, check:

```python
if msg.orientation_covariance[0] == -1.0:
    # Orientation is explicitly unavailable.
```

Before using angular velocity, check:

```python
if msg.angular_velocity_covariance[0] == -1.0:
    # Angular velocity is explicitly unavailable.
```

The current measured topic uses zero rather than `-1`, so both fields are
present but their uncertainty is unspecified.

Also validate quaternion magnitude:

```python
q_norm = np.linalg.norm([qx, qy, qz, qw])
```

A healthy orientation quaternion should satisfy:

```text
q_norm approximately 1
```

The measured quaternion norm was:

```text
1.00000002
```

## 11. Complete Processing Function

The following function converts one `sensor_msgs/msg/Imu` message into
`obs[0:6]`.

```python
import numpy as np


ANGULAR_VELOCITY_SCALE = 0.25


def imu_message_to_policy_observation(
    msg,
    rotation_pelvis_from_imu=None,
):
    """Build obs[0:6] from sensor_msgs/msg/Imu.

    Args:
        msg:
            A sensor_msgs.msg.Imu message.
        rotation_pelvis_from_imu:
            Optional 3x3 rotation matrix mapping vectors from the IMU frame
            into the pelvis frame. Use None only when both frames are aligned.

    Returns:
        A float32 NumPy array with shape (6,).
    """
    omega_imu = np.array(
        [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ],
        dtype=np.float32,
    )

    qx = float(msg.orientation.x)
    qy = float(msg.orientation.y)
    qz = float(msg.orientation.z)
    qw = float(msg.orientation.w)

    quaternion_norm = np.linalg.norm([qx, qy, qz, qw])
    if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-6:
        raise ValueError("Invalid IMU orientation quaternion")

    # Normalize to protect the gravity calculation against small drift.
    qx /= quaternion_norm
    qy /= quaternion_norm
    qz /= quaternion_norm
    qw /= quaternion_norm

    gravity_imu = np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )

    if rotation_pelvis_from_imu is None:
        omega_pelvis = omega_imu
        gravity_pelvis = gravity_imu
    else:
        rotation = np.asarray(rotation_pelvis_from_imu, dtype=np.float32)
        if rotation.shape != (3, 3):
            raise ValueError("rotation_pelvis_from_imu must have shape (3, 3)")

        omega_pelvis = rotation @ omega_imu
        gravity_pelvis = rotation @ gravity_imu

    obs_0_6 = np.concatenate(
        [
            ANGULAR_VELOCITY_SCALE * omega_pelvis,
            gravity_pelvis,
        ]
    ).astype(np.float32)

    if obs_0_6.shape != (6,):
        raise RuntimeError("Expected a six-element IMU observation")
    if not np.all(np.isfinite(obs_0_6)):
        raise ValueError("IMU observation contains NaN or infinity")

    return obs_0_6
```

## 12. ROS 2 Subscriber Example

This example subscribes to `/imu`, retains the newest sample, and builds the
six policy values at 50 Hz of simulation time.

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
from sensor_msgs.msg import Imu


POLICY_DT = 0.02


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class G1ImuObservationNode(Node):
    def __init__(self):
        super().__init__("g1_imu_observation")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.latest_imu = None
        self.last_policy_stamp = None

        self.create_subscription(Imu, "/imu", self.imu_callback, qos)

    def imu_callback(self, msg):
        self.latest_imu = msg
        current_stamp = stamp_to_seconds(msg.header.stamp)

        if self.last_policy_stamp is None:
            should_update = True
        else:
            should_update = (
                current_stamp - self.last_policy_stamp >= POLICY_DT - 1e-9
            )

        if not should_update:
            return

        # Use None only after verifying pelvis_imu is pelvis-aligned.
        obs_0_6 = imu_message_to_policy_observation(
            self.latest_imu,
            rotation_pelvis_from_imu=None,
        )

        self.last_policy_stamp = current_stamp
        self.get_logger().info(
            "obs[0:6] = " + np.array2string(obs_0_6, precision=6)
        )


def main():
    rclpy.init()
    node = G1ImuObservationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

In a full controller, this callback should normally update a shared observation
buffer rather than log every sample.

## 13. Processing the Measured Sample

Measured message:

```text
angular_velocity [rad/s] =
[0.0, 0.0, 0.0]

orientation [x, y, z, w] =
[0.101190, 0.684975, -0.214107, 0.689005]
```

Assuming `pelvis_imu` is aligned with the pelvis:

```text
scaled angular velocity =
0.25 * [0, 0, 0]
= [0, 0, 0]

projected gravity =
[0.987234, 0.153874, -0.041138]
```

The resulting first six policy values are:

```text
obs[0:6] =
[0.000000, 0.000000, 0.000000,
 0.987234, 0.153874, -0.041138]
```

Because the robot was lying face down, a gravity vector dominated by the
positive pelvis `x` component is expected. When the robot is moved upright,
the same calculation should produce a vector close to:

```text
[0, 0, -1]
```

## 14. Validation Procedure

Perform these tests before connecting the observation to the policy.

### 14.1 Static Upright Test

Hold the robot upright and stationary:

```text
obs[0:3] approximately [0, 0, 0]
obs[3:6] approximately [0, 0, -1]
```

### 14.2 Static Face-Down Test

Place the robot face down:

```text
obs[0:3] approximately [0, 0, 0]
obs[3:6] dominated by +x
```

The exact vector depends on how flat the pelvis lies and on the IMU mounting
rotation.

### 14.3 Positive Roll Test

Rotate the pelvis around `+x`:

```text
angular_velocity.x > 0
```

At a static roll of `+10 deg`:

```text
gravity approximately [0, -0.17365, -0.98481]
```

### 14.4 Positive Pitch Test

Rotate the pelvis around `+y`:

```text
angular_velocity.y > 0
```

At a static pitch of `+10 deg`:

```text
gravity approximately [+0.17365, 0, -0.98481]
```

### 14.5 Positive Yaw Test

Rotate around pelvis `+z`:

```text
angular_velocity.z > 0
```

Projected gravity should remain approximately unchanged during pure yaw.

### 14.6 Rate Test

Verify consecutive message timestamps:

```text
delta_stamp approximately 0.005 s
```

Verify policy updates:

```text
delta_policy_stamp approximately 0.02 s
```

## 15. Common Errors

### Wrong Quaternion Order

Incorrect:

```python
quaternion_wxyz = [
    msg.orientation.x,
    msg.orientation.y,
    msg.orientation.z,
    msg.orientation.w,
]
```

Correct:

```python
quaternion_wxyz = [
    msg.orientation.w,
    msg.orientation.x,
    msg.orientation.y,
    msg.orientation.z,
]
```

### Using Linear Acceleration as Projected Gravity

Incorrect:

```python
gravity = normalize(msg.linear_acceleration)
```

Correct:

```python
gravity = gravity_direction_from_orientation(msg.orientation)
```

### Running the LSTM at the IMU Rate

Incorrect:

```text
One policy inference per 200 Hz IMU callback
```

Correct:

```text
Keep the latest IMU sample and run policy at 50 Hz simulation-time
```

### Mixing Frames

Incorrect:

```text
angular velocity in pelvis_imu frame
gravity in pelvis frame
```

Correct:

```text
both angular velocity and gravity in pelvis frame
```

### Multiplying Gravity by 9.81

Incorrect:

```text
obs[3:6] = 9.81 * gravity_direction
```

Correct:

```text
obs[3:6] = gravity_direction
```

## 16. Final Checklist

Before using `/imu` for policy inference, confirm:

- [ ] `/imu` uses `sensor_msgs/msg/Imu`.
- [ ] `angular_velocity` is interpreted as `rad/s`.
- [ ] ROS quaternion order is handled as `xyzw`.
- [ ] Quaternion magnitude is close to one.
- [ ] `pelvis_imu` axes are aligned with pelvis axes, or a fixed rotation is
      applied.
- [ ] Angular velocity is transformed into the pelvis frame.
- [ ] Projected gravity is calculated from orientation.
- [ ] Projected gravity is not multiplied by `9.81`.
- [ ] Angular velocity is multiplied by `0.25`.
- [ ] Both signals are converted to `float32`.
- [ ] The final array has shape `(6,)`.
- [ ] Policy inference runs at `50 Hz` simulation-time.
- [ ] The newest IMU sample is used for each policy step.
- [ ] Upright gravity is close to `[0, 0, -1]`.
- [ ] Face-down gravity is dominated by the expected pelvis horizontal axis.
- [ ] Controlled roll, pitch, and yaw tests confirm axis signs.

Once these conditions are satisfied, `/imu` provides all information required
to construct the first six values of the G1 policy observation.
