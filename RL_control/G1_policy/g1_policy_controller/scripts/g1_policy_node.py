#!/usr/bin/env python3

from pathlib import Path

import message_filters
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu, JointState

from g1_policy_controller.core import (
    DEFAULT_JOINT_ANGLES,
    NUM_ACTIONS,
    NUM_OBSERVATIONS,
    PolicyScheduler,
    build_full_joint_targets,
    build_observation,
    extract_policy_joint_state,
)


EXPECTED_JOINT_COUNT = 43


def stamp_to_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class G1PolicyController(Node):
    def __init__(self) -> None:
        super().__init__("g1_policy_controller")

        self.declare_parameter("policy_path", "")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("joint_states_topic", "/joint/joint_states")
        self.declare_parameter("joint_command_topic", "/joint/joint_command")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.policy = self._load_policy()
        self.scheduler = PolicyScheduler()
        self.previous_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.has_policy_action = False

        simulation_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        command_topic = self.get_parameter("joint_command_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        joint_states_topic = self.get_parameter("joint_states_topic").value

        self.joint_publisher = self.create_publisher(
            JointState,
            command_topic,
            simulation_qos,
        )
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            cmd_vel_topic,
            self._cmd_vel_callback,
            10,
        )

        self.imu_subscriber = message_filters.Subscriber(
            self,
            Imu,
            imu_topic,
            qos_profile=simulation_qos,
        )
        self.joint_state_subscriber = message_filters.Subscriber(
            self,
            JointState,
            joint_states_topic,
            qos_profile=simulation_qos,
        )
        self.synchronizer = message_filters.TimeSynchronizer(
            [self.joint_state_subscriber, self.imu_subscriber],
            queue_size=10,
        )
        self.synchronizer.registerCallback(self._sensor_callback)

        self.get_logger().info(
            "G1 policy controller is ready. Start Isaac Sim Play; "
            "the controller will run on the first synchronized sensor sample."
        )

    def _load_policy(self):
        configured_path = str(self.get_parameter("policy_path").value)
        if configured_path:
            policy_path = Path(configured_path).expanduser()
        else:
            policy_path = (
                Path(get_package_share_directory("g1_policy_controller"))
                / "policy"
                / "motion.pt"
            )

        if not policy_path.is_file():
            raise FileNotFoundError(f"policy file not found: {policy_path}")

        policy = torch.jit.load(str(policy_path), map_location="cpu")
        policy.eval()
        if not hasattr(policy, "reset_memory"):
            raise RuntimeError("TorchScript policy does not export reset_memory()")
        policy.reset_memory()
        self.get_logger().info(f"Loaded G1 policy: {policy_path}")
        return policy

    def _cmd_vel_callback(self, message: Twist) -> None:
        command = np.array(
            [
                message.linear.x,
                message.linear.y,
                message.angular.z,
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(command)):
            self.get_logger().warning(
                "Ignoring /cmd_vel containing NaN or infinity",
                throttle_duration_sec=5.0,
            )
            return
        self.command = command

    def _reset_policy_state(self) -> None:
        self.policy.reset_memory()
        self.previous_action.fill(0.0)
        self.action.fill(0.0)
        self.has_policy_action = False

    def _sensor_callback(
        self,
        joint_state: JointState,
        imu: Imu,
    ) -> None:
        try:
            sensor_data = self._validate_and_extract(joint_state, imu)
        except ValueError as error:
            self.get_logger().warning(
                f"Ignoring invalid synchronized sensor sample: {error}",
                throttle_duration_sec=5.0,
            )
            return

        stamp_ns = stamp_to_nanoseconds(joint_state.header.stamp)
        decision = self.scheduler.update(stamp_ns)
        if decision.reset:
            self._reset_policy_state()
            self.get_logger().warning(
                "Simulation timestamp moved backwards; reset LSTM, action, "
                "and gait phase."
            )

        joint_position, joint_velocity, quaternion, angular_velocity = sensor_data
        if decision.infer:
            assert decision.policy_step is not None
            try:
                observation = build_observation(
                    angular_velocity=angular_velocity,
                    quaternion_wxyz=quaternion,
                    command=self.command,
                    joint_position=joint_position,
                    joint_velocity=joint_velocity,
                    previous_action=self.previous_action,
                    policy_step=decision.policy_step,
                )
                action = self._compute_action(observation)
            except (RuntimeError, ValueError) as error:
                self.get_logger().error(f"Policy inference failed: {error}")
                return

            self.action = action
            self.previous_action = action.copy()
            self.has_policy_action = True

        self.scheduler.commit(decision)

        if self.has_policy_action:
            target_positions = build_full_joint_targets(
                joint_state.name,
                self.action,
            )
            self._publish_joint_command(
                joint_state.name,
                target_positions,
            )

    def _validate_and_extract(
        self,
        joint_state: JointState,
        imu: Imu,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if stamp_to_nanoseconds(joint_state.header.stamp) != stamp_to_nanoseconds(
            imu.header.stamp
        ):
            raise ValueError("IMU and joint-state timestamps do not match")
        if len(joint_state.name) != EXPECTED_JOINT_COUNT:
            raise ValueError(
                f"expected {EXPECTED_JOINT_COUNT} joints, "
                f"got {len(joint_state.name)}"
            )
        if imu.orientation_covariance[0] == -1.0:
            raise ValueError("IMU orientation is unavailable")
        if imu.angular_velocity_covariance[0] == -1.0:
            raise ValueError("IMU angular velocity is unavailable")

        joint_position, joint_velocity = extract_policy_joint_state(
            joint_state.name,
            joint_state.position,
            joint_state.velocity,
        )
        quaternion = np.array(
            [
                imu.orientation.w,
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
            ],
            dtype=np.float32,
        )
        angular_velocity = np.array(
            [
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(quaternion)):
            raise ValueError("IMU quaternion contains NaN or infinity")
        if float(np.linalg.norm(quaternion)) <= 1e-8:
            raise ValueError("IMU quaternion norm must be non-zero")
        if not np.all(np.isfinite(angular_velocity)):
            raise ValueError("IMU angular velocity contains NaN or infinity")

        return joint_position, joint_velocity, quaternion, angular_velocity

    def _compute_action(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (NUM_OBSERVATIONS,):
            raise ValueError(
                f"expected observation shape ({NUM_OBSERVATIONS},), "
                f"got {observation.shape}"
            )

        memory_snapshot = {
            name: buffer.detach().clone()
            for name, buffer in self.policy.named_buffers()
            if name in {"hidden_state", "cell_state"}
        }
        try:
            with torch.inference_mode():
                observation_tensor = torch.from_numpy(observation).view(1, -1)
                output = self.policy(observation_tensor)
            action = output.detach().cpu().view(-1).numpy().astype(
                np.float32,
                copy=True,
            )
            if action.shape != (NUM_ACTIONS,):
                raise ValueError(
                    f"expected policy action shape ({NUM_ACTIONS},), "
                    f"got {action.shape}"
                )
            if not np.all(np.isfinite(action)):
                raise ValueError("policy action contains NaN or infinity")
            return action
        except Exception:
            with torch.no_grad():
                for name, buffer in self.policy.named_buffers():
                    if name in memory_snapshot:
                        buffer.copy_(memory_snapshot[name])
            raise

    def _publish_joint_command(
        self,
        joint_names,
        target_positions: np.ndarray,
    ) -> None:
        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.name = list(joint_names)
        command.position = target_positions.tolist()
        command.velocity = np.zeros(len(joint_names), dtype=np.float64).tolist()
        command.effort = np.zeros(len(joint_names), dtype=np.float64).tolist()
        self.joint_publisher.publish(command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = G1PolicyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
