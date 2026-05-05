import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import numpy as np
import os
import time

class ImuAnalyzer(Node):
    def __init__(self):
        super().__init__('imu_analyzer')
        self.subscription = self.create_subscription(
            Imu,
            '/imu',
            self.listener_callback,
            10)
        self.data = {
            'time': [],
            'accel_x': [], 'accel_y': [], 'accel_z': [],
            'gyro_x': [], 'gyro_y': [], 'gyro_z': []
        }
        self.start_time = None
        self.max_duration = 10.0 # seconds
        self.get_logger().info('IMU Analyzer started. Collecting data for 10 seconds...')

    def listener_callback(self, msg):
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        
        elapsed = current_time - self.start_time
        if elapsed > self.max_duration:
            return

        self.data['time'].append(elapsed)
        self.data['accel_x'].append(msg.linear_acceleration.x)
        self.data['accel_y'].append(msg.linear_acceleration.y)
        self.data['accel_z'].append(msg.linear_acceleration.z)
        self.data['gyro_x'].append(msg.angular_velocity.x)
        self.data['gyro_y'].append(msg.angular_velocity.y)
        self.data['gyro_z'].append(msg.angular_velocity.z)

    def save_results(self):
        if not self.data['time']:
            self.get_logger().warn("No data collected!")
            return

        # os.makedirs('sensor_graph', exist_ok=True)
        
        # Plot Acceleration
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.data['time'], self.data['accel_x'], label='X', alpha=0.8)
        plt.plot(self.data['time'], self.data['accel_y'], label='Y', alpha=0.8)
        plt.plot(self.data['time'], self.data['accel_z'], label='Z', alpha=0.8)
        plt.title('Linear Acceleration (m/s^2)')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Plot Angular Velocity
        plt.subplot(2, 1, 2)
        plt.plot(self.data['time'], self.data['gyro_x'], label='X', alpha=0.8)
        plt.plot(self.data['time'], self.data['gyro_y'], label='Y', alpha=0.8)
        plt.plot(self.data['time'], self.data['gyro_z'], label='Z', alpha=0.8)
        plt.title('Angular Velocity (rad/s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('imu_analysis.png', dpi=300)
        plt.close()

        # Print Analysis
        print("\n" + "="*40)
        print("      IMU DATA SUMMARY (10 Seconds)")
        print("="*40)
        print(f"{'Metric':<10} | {'Mean':>10} | {'Std Dev':>10} | {'Max':>10} | {'Min':>10}")
        print("-" * 58)
        for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            vals = np.array(self.data[key])
            print(f"{key:<10} | {vals.mean():>10.4f} | {vals.std():>10.4f} | {vals.max():>10.4f} | {vals.min():>10.4f}")
        print("="*40)
        print(f"Graph saved to: imu_analysis.png")

def main(args=None):
    rclpy.init(args=args)
    analyzer = ImuAnalyzer()
    
    start_wait = time.time()
    # Spin until we have enough data or timeout
    try:
        while rclpy.ok():
            rclpy.spin_once(analyzer, timeout_sec=0.1)
            if analyzer.start_time and (time.time() - analyzer.start_time) > analyzer.max_duration:
                break
            if not analyzer.start_time and (time.time() - start_wait) > 5.0:
                print("Timeout: No IMU data received after 5 seconds.")
                break
    except KeyboardInterrupt:
        pass
            
    analyzer.save_results()
    analyzer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
