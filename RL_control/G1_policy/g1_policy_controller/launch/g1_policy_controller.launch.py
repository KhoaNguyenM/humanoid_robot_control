from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    IfElseSubstitution,
    LaunchConfiguration,
    TextSubstitution,
)
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("g1_policy_controller"))
    default_policy_path = str(package_share / "policy" / "motion.pt")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "policy_path",
                default_value=default_policy_path,
                description="Path to the Unitree G1 TorchScript policy",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use the Isaac Sim /clock source",
            ),
            DeclareLaunchArgument(
                "imu_topic",
                default_value="/imu",
                description="Isaac Sim IMU topic",
            ),
            DeclareLaunchArgument(
                "joint_states_topic",
                default_value="/joint/joint_states",
                description="Isaac Sim joint-state topic",
            ),
            DeclareLaunchArgument(
                "joint_command_topic",
                default_value="/joint/joint_command",
                description="Isaac Sim joint-command topic",
            ),
            DeclareLaunchArgument(
                "cmd_vel_topic",
                default_value="/cmd_vel",
                description="Velocity command topic",
            ),
            DeclareLaunchArgument(
                "namespace",
                default_value="g1_01",
                description="Optional ROS namespace",
            ),
            DeclareLaunchArgument(
                "use_namespace",
                default_value="false",
                description="Apply the configured namespace",
            ),
            Node(
                package="g1_policy_controller",
                executable="g1_policy_node.py",
                name="g1_policy_controller",
                output="screen",
                namespace=IfElseSubstitution(
                    LaunchConfiguration("use_namespace"),
                    LaunchConfiguration("namespace"),
                    TextSubstitution(text=""),
                ),
                parameters=[
                    {
                        "policy_path": LaunchConfiguration("policy_path"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "imu_topic": LaunchConfiguration("imu_topic"),
                        "joint_states_topic": LaunchConfiguration(
                            "joint_states_topic"
                        ),
                        "joint_command_topic": LaunchConfiguration(
                            "joint_command_topic"
                        ),
                        "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                    }
                ],
            ),
        ]
    )
