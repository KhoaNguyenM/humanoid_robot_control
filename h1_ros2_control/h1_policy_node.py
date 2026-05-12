#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

class H1PolicyControlNode(Node):
    def __init__(self):
        super().__init__('h1_policy_control_node')
        
        # --- 1. Parameters ---
        self.policy_path = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/H1_control/h1_policy/h1_policy.pt"
        self.action_scale = 0.5
        self.num_joints = 19
        self.command = np.array([0.0, 0.0, 0.0]) # Walk forward at 0.4 m/s
        
        # --- 2. Correct Joint Order (Verified via Debug Script) ---
        self.joint_names = [
            'left_hip_yaw', 'right_hip_yaw', 'torso', 'left_hip_roll', 'right_hip_roll',
            'left_shoulder_pitch', 'right_shoulder_pitch', 'left_hip_pitch', 'right_hip_pitch',
            'left_shoulder_roll', 'right_shoulder_roll', 'left_knee', 'right_knee',
            'left_shoulder_yaw', 'right_shoulder_yaw', 'left_ankle', 'right_ankle',
            'left_elbow', 'right_elbow'
        ]
        
        # Default positions mapped to the order above
        self.default_joint_pos = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0,        # yaw, torso, roll
            0.28, 0.28,                     # shoulder_pitch
            -0.28, -0.28,                   # hip_pitch
            0.0, 0.0,                       # shoulder_roll
            0.79, 0.79,                     # knee
            0.0, 0.0,                       # shoulder_yaw
            -0.52, -0.52,                   # ankle
            0.52, 0.52                      # elbow
        ])

        # --- 3. Internal State ---
        self.device = torch.device("cpu")
        self.policy = torch.jit.load(self.policy_path).to(self.device)
        self.policy.eval()
        
        self.current_joint_pos = np.zeros(self.num_joints)
        self.current_joint_vel = np.zeros(self.num_joints)
        self.ang_vel = np.zeros(3)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.last_action = np.zeros(self.num_joints)
        
        self.imu_received = False
        self.joint_received = False
        self.waiting_logged = False
        
        # --- 4. ROS2 Interface ---
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.cmd_pub = self.create_publisher(JointState, '/joint_command', 10)
        
        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("H1 Policy Node Updated with correct joint mapping.")

    def imu_callback(self, msg):
        self.ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.imu_received = True

    def joint_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                self.current_joint_pos[idx] = msg.position[i]
                self.current_joint_vel[idx] = msg.velocity[i]
        self.joint_received = True

    def compute_projected_gravity(self):
        rot = R.from_quat(self.quat)
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = rot.inv().apply(gravity_world)
        return gravity_body

    def control_loop(self):
        if not (self.imu_received and self.joint_received):
            if not self.waiting_logged:
                self.get_logger().info("Waiting for sensor data...")
                self.waiting_logged = True
            return

        if self.waiting_logged:
            self.get_logger().info("Data received! Starting control.")
            self.waiting_logged = False

        # 1. Prepare Observation (69 dim)
        gravity_b = self.compute_projected_gravity()
        
        obs = np.zeros(69, dtype=np.float32)
        obs[0:3] = 0.0 # lin_vel placeholder
        obs[3:6] = self.ang_vel
        obs[6:9] = gravity_b
        obs[9:12] = self.command
        obs[12:31] = self.current_joint_pos - self.default_joint_pos
        obs[31:50] = self.current_joint_vel
        obs[50:69] = self.last_action
        
        # 2. Inference
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action = self.policy(obs_tensor).cpu().numpy().flatten()
            self.last_action = action
            
        # 3. Scale and Apply Action
        target_joint_pos = self.default_joint_pos + action * self.action_scale
        
        # 4. Publish Command
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = self.joint_names
        cmd_msg.position = target_joint_pos.tolist()
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = H1PolicyControlNode()
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
