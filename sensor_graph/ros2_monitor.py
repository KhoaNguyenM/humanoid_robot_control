import rclpy
from rclpy.node import Node
import time
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from collections import deque

class ROS2Monitor(Node):
    def __init__(self):
        super().__init__('ros2_monitor')
        self.topics = ['/imu', '/joint_states', '/odom', '/joint_command']
        self.stats = {topic: {'times': deque(maxlen=100), 'latencies': []} for topic in self.topics}
        
        self.create_subscription(Imu, '/imu', lambda msg: self.cb('/imu'), 10)
        self.create_subscription(JointState, '/joint_states', lambda msg: self.cb('/joint_states'), 10)
        self.create_subscription(Odometry, '/odom', lambda msg: self.cb('/odom'), 10)
        self.create_subscription(JointState, '/joint_command', lambda msg: self.cb('/joint_command'), 10)
        
        self.start_time = time.time()
        self.print_timer = self.create_timer(5.0, self.print_stats)
        print("Đang theo dõi tần số và độ trễ các topic ROS2 trong 10 giây...")

    def cb(self, topic):
        now = time.time()
        if self.stats[topic]['times']:
            last_time = self.stats[topic]['times'][-1]
            latency = now - last_time
            self.stats[topic]['latencies'].append(latency)
        self.stats[topic]['times'].append(now)

    def print_stats(self):
        print("\n" + "="*70)
        print(f"{'Topic':<25} | {'Hz (Tần số)':<15} | {'Latency (Độ trễ TB)':<20}")
        print("-" * 70)
        for topic in self.topics:
            latencies = self.stats[topic]['latencies']
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                hz = 1.0 / avg_latency if avg_latency > 0 else 0
                print(f"{topic:<25} | {hz:>10.2f} Hz | {avg_latency*1000:>12.2f} ms")
            else:
                print(f"{topic:<25} | {'No Data':>10} | {'N/A':>15}")
        print("="*70)

def main():
    rclpy.init()
    node = ROS2Monitor()
    try:
        # Run for approx 11 seconds to get two reports
        start = time.time()
        while rclpy.ok() and (time.time() - start < 11):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
