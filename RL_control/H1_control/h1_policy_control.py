import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
import onnxruntime as ort
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os

class H1PolicyController(Node):
    def __init__(self):
        super().__init__('h1_policy_controller')
        
        # 1. Configuration
        self.project_root = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control"
        self.policy_path = os.path.join(self.project_root, "unitree_sim_isaaclab/assets/model/policy.onnx")
        
        # Simulation Joint names (19 joints)
        self.sim_joint_names = [
            "left_hip_yaw", "right_hip_yaw", "torso", "left_hip_roll", "right_hip_roll",
            "left_shoulder_pitch", "right_shoulder_pitch", "left_hip_pitch", "right_hip_pitch",
            "left_shoulder_roll", "right_shoulder_roll", "left_knee", "right_knee",
            "left_shoulder_yaw", "right_shoulder_yaw", "left_ankle", "right_ankle",
            "left_elbow", "right_elbow"
        ]
        
        # Policy Joint order (27 joints expected for H1-2/G1 style observation)
        self.policy_joint_names = [
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle", "left_ankle_roll",
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle", "right_ankle_roll",
            "torso",
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"
        ]
        
        # Output Action names (12 joints - legs only)
        self.action_joint_names = [
            'left_hip_pitch', 'right_hip_pitch', 'left_hip_roll', 'right_hip_roll', 
            'left_hip_yaw', 'right_hip_yaw', 'left_knee', 'right_knee', 
            'left_ankle', 'right_ankle', 'left_ankle_roll', 'right_ankle_roll'
        ]

        # Normalization scales
        self.obs_scales = {
            "ang_vel": 0.25,
            "projected_gravity": 1.0,
            "commands": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "actions": 1.0
        }
        self.action_scale = 0.25
        
        # 2. Load Policy
        if not os.path.exists(self.policy_path):
            self.get_logger().error(f"Policy not found at {self.policy_path}")
            return
            
        self.ort_session = ort.InferenceSession(self.policy_path)
        input_shape = self.ort_session.get_inputs()[0].shape
        output_shape = self.ort_session.get_outputs()[0].shape
        self.get_logger().info(f"Loaded policy. Input shape: {input_shape}, Output shape: {output_shape}")
        
        # Observations: 3 (ang_vel) + 3 (grav) + 4 (cmd) + 27 (pos) + 27 (vel) + 27 (action) = 91
        self.num_obs_per_step = 91 
        self.history_len = input_shape[1] // self.num_obs_per_step
        self.obs_history = np.zeros((self.history_len, self.num_obs_per_step))
        
        # 3. State Variables
        self.sim_joint_pos = np.zeros(19)
        self.sim_joint_vel = np.zeros(19)
        self.current_ang_vel = np.zeros(3)
        self.current_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.last_action_27 = np.zeros(27) 
        self.commands = np.array([0.0, 0.0, 0.0, 0.0]) # Vx=0, Vy=0, YawRate=0 -> Stand Still
        
        # Safety Flags
        self.received_imu = False
        self.received_joint_states = False
        
        # 4. ROS2 Setup
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.publisher = self.create_publisher(JointState, '/joint_command', 10)
        
        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("H1 Policy Controller (Waiting for ROS2 topics...) Started.")

    def imu_callback(self, msg):
        self.current_ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.current_quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.received_imu = True

    def joint_state_callback(self, msg):
        if not msg.name: return
        for i, name in enumerate(msg.name):
            if name in self.sim_joint_names:
                idx = self.sim_joint_names.index(name)
                self.sim_joint_pos[idx] = msg.position[i]
                self.sim_joint_vel[idx] = msg.velocity[i]
        self.received_joint_states = True

    def get_projected_gravity(self):
        r = R.from_quat(self.current_quat)
        inv_rot = r.inv()
        gravity_world = np.array([0.0, 0.0, -1.0])
        return inv_rot.apply(gravity_world)

    def get_policy_state(self):
        pos_27 = np.zeros(27)
        vel_27 = np.zeros(27)
        for i, name in enumerate(self.policy_joint_names):
            if name in self.sim_joint_names:
                sim_idx = self.sim_joint_names.index(name)
                pos_27[i] = self.sim_joint_pos[sim_idx]
                vel_27[i] = self.sim_joint_vel[sim_idx]
        return pos_27, vel_27

    def control_loop(self):
        # Only execute if we have received initial state from simulation
        if not (self.received_imu and self.received_joint_states):
            return

        # 1. Prepare current observation
        proj_grav = self.get_projected_gravity()
        pos_27, vel_27 = self.get_policy_state()
        
        obs = np.concatenate([
            self.current_ang_vel * self.obs_scales["ang_vel"],
            proj_grav * self.obs_scales["projected_gravity"],
            self.commands * self.obs_scales["commands"],
            pos_27 * self.obs_scales["joint_pos"],
            vel_27 * self.obs_scales["joint_vel"],
            self.last_action_27 * self.obs_scales["actions"]
        ])
        
        # 2. Update history
        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = obs
        input_obs = self.obs_history.flatten().reshape(1, -1).astype(np.float32)

        # 3. Inference
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_obs}
        ort_outs = self.ort_session.run(None, ort_inputs)
        action_12 = ort_outs[0][0] 
        
        # 4. Map Action 12 back to sim joints (19)
        self.last_action_27[:12] = action_12 
        
        target_msg = JointState()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        
        for i, val in enumerate(action_12):
            joint_name = self.action_joint_names[i]
            if joint_name in self.sim_joint_names:
                target_msg.name.append(joint_name)
                target_msg.position.append(float(val * self.action_scale))
        
        # 5. Publish
        self.publisher.publish(target_msg)

def main(args=None):
    rclpy.init(args=args)
    node = H1PolicyController()
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
