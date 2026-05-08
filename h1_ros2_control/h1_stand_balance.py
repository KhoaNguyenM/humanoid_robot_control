#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

class H1StandBalanceNode(Node):
    def __init__(self):
        super().__init__('h1_stand_balance_node')
        
        # --- 1. Parameters ---
        self.policy_path = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/H1_control/h1_policy/h1_policy.pt"
        self.action_scale = 0.5
        self.num_joints = 19
        self.command = np.array([0.0, 0.0, 0.0]) # Stationary balance
        
        # --- 2. Joint Order (Custom) ---
        self.joint_names = [
            'left_hip_yaw', 'right_hip_yaw', 'torso', 'left_hip_roll', 'right_hip_roll',
            'left_shoulder_pitch', 'right_shoulder_pitch', 'left_hip_pitch', 'right_hip_pitch',
            'left_shoulder_roll', 'right_shoulder_roll', 'left_knee', 'right_knee',
            'left_shoulder_yaw', 'right_shoulder_yaw', 'left_ankle', 'right_ankle',
            'left_elbow', 'right_elbow'
        ]
        
        # We will read default positions from the first received joint state
        self.default_joint_pos = None

        # --- 3. Internal State ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.policy = torch.jit.load(self.policy_path).to(self.device)
        self.policy.eval()
        
        self.current_joint_pos = np.zeros(self.num_joints)
        self.current_joint_vel = np.zeros(self.num_joints)
        self.ang_vel = np.zeros(3)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.last_action = np.zeros(self.num_joints)
        
        self.imu_received = False
        self.joint_received = False
        self.control_start_time = None
        
        # --- 4. ROS2 Interface ---
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.cmd_pub = self.create_publisher(JointState, '/joint_command', 10)
        
        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("H1 Stand Balance Node started. Waiting for initial joint state to set default pose...")

    def imu_callback(self, msg):
        self.ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.imu_received = True

    def joint_callback(self, msg):
        # Temp storage to map the incoming names to our order
        temp_pos = np.zeros(self.num_joints)
        temp_vel = np.zeros(self.num_joints)
        
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                temp_pos[idx] = msg.position[i]
                temp_vel[idx] = msg.velocity[i]
        
        self.current_joint_pos = temp_pos
        self.current_joint_vel = temp_vel
        self.joint_received = True

        # Capture the very first valid joint state as the default (calibration)
        if self.default_joint_pos is None:
            self.default_joint_pos = np.copy(self.current_joint_pos)
            self.get_logger().info(f"Calibration Done! Default pose captured from simulation.")

    def compute_projected_gravity(self):
        rot = R.from_quat(self.quat)
        gravity_world = np.array([0.0, 0.0, -1.0])
        return rot.inv().apply(gravity_world)

    def control_loop(self):
        # Ensure we have both sensor data and the initial calibration pose
        if not (self.imu_received and self.joint_received and self.default_joint_pos is not None):
            return

        if self.control_start_time is None:
            self.control_start_time = self.get_clock().now()
            self.get_logger().info("Starting balance control logic...")

        now = self.get_clock().now()
        elapsed_sec = (now - self.control_start_time).nanoseconds / 1e9
        
        # Warm-up phase: Send default pose for first 0.5s to ensure stability
        if elapsed_sec < 0.5:
            self.send_command(self.default_joint_pos)
            return

        # Policy inference
        gravity_b = self.compute_projected_gravity()
        
        obs = np.zeros(69, dtype=np.float32)
        obs[0:3] = 0.0 # lin_vel placeholder
        obs[3:6] = self.ang_vel
        obs[6:9] = gravity_b
        obs[9:12] = self.command
        obs[12:31] = self.current_joint_pos - self.default_joint_pos
        obs[31:50] = self.current_joint_vel
        obs[50:69] = self.last_action
        
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action = self.policy(obs_tensor).cpu().numpy().flatten()
            self.last_action = action
            
        target_joint_pos = self.default_joint_pos + action * self.action_scale
        self.send_command(target_joint_pos)

    def send_command(self, positions):
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = self.joint_names
        cmd_msg.position = positions.tolist()
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = H1StandBalanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
