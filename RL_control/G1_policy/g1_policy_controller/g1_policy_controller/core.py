from dataclasses import dataclass
from typing import Sequence

import numpy as np


NUM_ACTIONS = 12
NUM_OBSERVATIONS = 47

POLICY_DT_SECONDS = 0.02
POLICY_DECIMATION = 4
GAIT_PERIOD_SECONDS = 0.8

ANGULAR_VELOCITY_SCALE = 0.25
DOF_POSITION_SCALE = 1.0
DOF_VELOCITY_SCALE = 0.05
ACTION_SCALE = 0.25

COMMAND_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
COMMAND_LIMITS = np.array([0.8, 0.5, 1.57], dtype=np.float32)

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
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
    ],
    dtype=np.float32,
)


def _float32_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinity")
    return vector


def projected_gravity(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    quaternion = _float32_vector(quaternion_wxyz, 4, "quaternion")
    if float(np.linalg.norm(quaternion)) <= 1e-8:
        raise ValueError("quaternion norm must be non-zero")

    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def extract_policy_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(names) != len(positions) or len(names) != len(velocities):
        raise ValueError("joint name, position, and velocity arrays must match")
    if len(names) != len(set(names)):
        raise ValueError("joint state contains duplicate names")

    position_array = _float32_vector(positions, len(names), "joint positions")
    velocity_array = _float32_vector(velocities, len(names), "joint velocities")
    name_to_index = {name: index for index, name in enumerate(names)}

    missing = [name for name in POLICY_JOINT_NAMES if name not in name_to_index]
    if missing:
        raise ValueError(f"missing policy joints: {missing}")

    indices = np.array(
        [name_to_index[name] for name in POLICY_JOINT_NAMES],
        dtype=np.int64,
    )
    return position_array[indices], velocity_array[indices]


def build_observation(
    angular_velocity: Sequence[float],
    quaternion_wxyz: Sequence[float],
    command: Sequence[float],
    joint_position: Sequence[float],
    joint_velocity: Sequence[float],
    previous_action: Sequence[float],
    policy_step: int,
) -> np.ndarray:
    if policy_step < 1:
        raise ValueError("policy_step must start at 1")

    angular_velocity_array = _float32_vector(
        angular_velocity, 3, "angular velocity"
    )
    command_array = _float32_vector(command, 3, "command")
    joint_position_array = _float32_vector(
        joint_position, NUM_ACTIONS, "joint position"
    )
    joint_velocity_array = _float32_vector(
        joint_velocity, NUM_ACTIONS, "joint velocity"
    )
    previous_action_array = _float32_vector(
        previous_action, NUM_ACTIONS, "previous action"
    )

    clipped_command = np.clip(command_array, -COMMAND_LIMITS, COMMAND_LIMITS)
    phase = (
        (policy_step * POLICY_DT_SECONDS) % GAIT_PERIOD_SECONDS
    ) / GAIT_PERIOD_SECONDS

    observation = np.concatenate(
        [
            angular_velocity_array * ANGULAR_VELOCITY_SCALE,
            projected_gravity(quaternion_wxyz),
            clipped_command * COMMAND_SCALE,
            (joint_position_array - DEFAULT_JOINT_ANGLES)
            * DOF_POSITION_SCALE,
            joint_velocity_array * DOF_VELOCITY_SCALE,
            previous_action_array,
            np.array(
                [
                    np.sin(2.0 * np.pi * phase),
                    np.cos(2.0 * np.pi * phase),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32, copy=False)

    if observation.shape != (NUM_OBSERVATIONS,):
        raise RuntimeError(
            f"observation must have shape ({NUM_OBSERVATIONS},)"
        )
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation contains NaN or infinity")
    return observation


def build_full_joint_targets(
    joint_names: Sequence[str],
    action: Sequence[float],
) -> np.ndarray:
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("joint command contains duplicate names")

    action_array = _float32_vector(action, NUM_ACTIONS, "action")
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    missing = [name for name in POLICY_JOINT_NAMES if name not in name_to_index]
    if missing:
        raise ValueError(f"missing policy joints: {missing}")

    leg_targets = DEFAULT_JOINT_ANGLES + ACTION_SCALE * action_array
    targets = np.zeros(len(joint_names), dtype=np.float64)
    for policy_index, name in enumerate(POLICY_JOINT_NAMES):
        targets[name_to_index[name]] = float(leg_targets[policy_index])
    return targets


@dataclass(frozen=True)
class PolicyStepDecision:
    reset: bool
    infer: bool
    policy_step: int | None


class PolicyScheduler:
    def __init__(self) -> None:
        self.last_sensor_stamp_ns: int | None = None
        self.callback_count = 0
        self.policy_step = 0

    def reset(self) -> None:
        self.last_sensor_stamp_ns = None
        self.callback_count = 0
        self.policy_step = 0

    def update(self, sensor_stamp_ns: int) -> PolicyStepDecision:
        if sensor_stamp_ns < 0:
            raise ValueError("sensor timestamp must be non-negative")

        reset = (
            self.last_sensor_stamp_ns is not None
            and sensor_stamp_ns < self.last_sensor_stamp_ns
        )
        if reset:
            self.reset()

        self.last_sensor_stamp_ns = sensor_stamp_ns
        should_infer = self.callback_count % POLICY_DECIMATION == 0
        self.callback_count += 1
        if not should_infer:
            return PolicyStepDecision(reset=reset, infer=False, policy_step=None)

        self.policy_step += 1
        return PolicyStepDecision(
            reset=reset,
            infer=True,
            policy_step=self.policy_step,
        )
