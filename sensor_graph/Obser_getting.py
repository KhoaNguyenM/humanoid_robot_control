import sys
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import tkinter as tk

class SensorSubscriber(Node):
    def __init__(self, imu_callback, odom_callback):
        super().__init__('sensor_gui_subscriber')
        
        # Đăng ký nhận dữ liệu từ topic /imu
        self.sub_imu = self.create_subscription(
            Imu,
            '/imu',
            self.listener_imu_callback,
            10)
            
        # Đăng ký nhận dữ liệu từ topic /odom/odom
        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom/odom',
            self.listener_odom_callback,
            10)
            
        self.imu_callback = imu_callback
        self.odom_callback = odom_callback

    def listener_imu_callback(self, msg):
        self.imu_callback(msg)
        
    def listener_odom_callback(self, msg):
        self.odom_callback(msg)

class SensorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor Monitor Example")
        self.root.configure(bg='#333333') # Màu nền tối giống như trong ảnh
        
        self.labels = {}
        fields = [
            'Acc x', 'Acc y', 'Acc z',
            'Gyro x', 'Gyro y', 'Gyro z',
            'Orient x', 'Orient y', 'Orient z', 'Orient w',
            'Vel x', 'Vel y', 'Vel z'  # Thêm trường Linear Velocity
        ]
        
        # Bảng màu tham khảo từ ảnh và tự thiết kế thêm cho Vel
        colors = {
            'Acc x': '#ffb3ba', 'Acc y': '#baffc9', 'Acc z': '#bae1ff',
            'Gyro x': '#e3d59e', 'Gyro y': '#c9d1a1', 'Gyro z': '#c4b5fd',
            'Orient x': '#d5d5c9', 'Orient y': '#dfb2d1', 'Orient z': '#a89aa8', 'Orient w': '#96c2a8',
            'Vel x': '#ffdfba', 'Vel y': '#ffffba', 'Vel z': '#baffc9' # Màu cam nhạt, vàng nhạt, xanh nhạt
        }
        
        for idx, field in enumerate(fields):
            # Tên biến (chữ màu xám sáng)
            tk.Label(root, text=field, font=('Arial', 11, 'bold'), fg='#aaaaaa', bg='#333333', width=10, anchor='w').grid(row=idx, column=0, padx=10, pady=5)
            
            # Giá trị biến
            var = tk.StringVar(value="0.0")
            self.labels[field] = var
            label = tk.Label(root, textvariable=var, font=('Arial', 11), bg=colors.get(field, 'white'), fg='black', width=15, anchor='e', relief='flat')
            label.grid(row=idx, column=1, padx=10, pady=5)

        self.latest_imu = None
        self.latest_odom = None
        self._update_gui_loop()

    def _update_gui_loop(self):
        # Tách biệt update UI và luồng ROS2 bằng cách cập nhật định kỳ 10Hz
        if self.latest_imu is not None:
            self._set_imu_vars(self.latest_imu)
        if self.latest_odom is not None:
            self._set_odom_vars(self.latest_odom)
            
        self.root.after(100, self._update_gui_loop)

    def update_imu_data(self, msg: Imu):
        self.latest_imu = msg
        
    def update_odom_data(self, msg: Odometry):
        self.latest_odom = msg
        
    def _set_imu_vars(self, msg: Imu):
        # IMU Data
        self.labels['Acc x'].set(f"{msg.linear_acceleration.x:.5f}")
        self.labels['Acc y'].set(f"{msg.linear_acceleration.y:.5f}")
        self.labels['Acc z'].set(f"{msg.linear_acceleration.z:.5f}")
        
        self.labels['Gyro x'].set(f"{msg.angular_velocity.x:.5f}")
        self.labels['Gyro y'].set(f"{msg.angular_velocity.y:.5f}")
        self.labels['Gyro z'].set(f"{msg.angular_velocity.z:.5f}")
        
        self.labels['Orient x'].set(f"{msg.orientation.x:.5f}")
        self.labels['Orient y'].set(f"{msg.orientation.y:.5f}")
        self.labels['Orient z'].set(f"{msg.orientation.z:.5f}")
        self.labels['Orient w'].set(f"{msg.orientation.w:.5f}")

    def _set_odom_vars(self, msg: Odometry):
        # Lấy Linear Velocity từ gói tin Odom (twist.twist.linear)
        self.labels['Vel x'].set(f"{msg.twist.twist.linear.x:.5f}")
        self.labels['Vel y'].set(f"{msg.twist.twist.linear.y:.5f}")
        self.labels['Vel z'].set(f"{msg.twist.twist.linear.z:.5f}")

def ros_spin_thread(node):
    rclpy.spin(node)

def main():
    rclpy.init()
    
    root = tk.Tk()
    gui = SensorGUI(root)
    
    # Khởi tạo ROS2 Node và truyền vào 2 callback
    sensor_node = SensorSubscriber(gui.update_imu_data, gui.update_odom_data)
    
    # Chạy ROS2 spin trong thread phụ
    spin_thread = threading.Thread(target=ros_spin_thread, args=(sensor_node,), daemon=True)
    spin_thread.start()
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        sensor_node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
