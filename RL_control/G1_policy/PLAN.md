# Ke hoach ROS 2 controller cho policy Unitree G1 trong Isaac Sim

## 1. Muc tieu

Xay dung mot ROS 2 Jazzy package chay policy locomotion G1 tu
`unitree_rl_gym/deploy/pre_train/g1/motion.pt` de dieu khien robot
`g1_29dof_with_dex3` trong Isaac Sim.

Controller se tham khao cach to chuc cua package H1:

`IsaacSim-ros_workspaces/jazzy_ws/src/humanoid_locomotion_policy_example/h1_fullbody_controller`

nhung phai giu chinh xac observation, action, tan so va joint order cua policy G1
trong `unitree_rl_gym`.

Thanh cong khi:

- Node nhan duoc `/imu` va `/joint/joint_states` o 200 Hz simulation time.
- Policy LSTM inference moi 4 callback, tuong duong 50 Hz khi sensor du 200 Hz.
- Joint target duoc publish tren `/joint/joint_command` o moi cap sensor dong bo.
- Robot bat dau chay policy ngay khi Isaac Sim Play va co cap sensor hop le dau tien.
- Khong co qua trinh ramp ve home, vi Isaac Sim da dat robot dung home khi Play.
- Observation gui vao policy co dung 47 gia tri, `float32`, dung thu tu va scale.
- Output 12 action duoc doi thanh position target cho 12 khop chan.
- Cac khop eo, tay, co tay va Dex3 duoc giu tai target home `0 rad`.

## 2. Pham vi thuc hien

Tao ROS package moi tai:

```text
RL_control/G1_policy/g1_policy_controller/
```

Package gom:

```text
g1_policy_controller/
├── CMakeLists.txt
├── package.xml
├── launch/
│   └── g1_policy_controller.launch.py
├── policy/
│   └── motion.pt
├── scripts/
│   └── g1_policy_node.py
└── test/
    └── test_g1_policy_controller.py
```

Khong sua policy, training code, Isaac Sim stage, OmniGraph, robot USD hoac package
H1 hien tai.

`policy/motion.pt` la ban sao byte-identical cua:

```text
unitree_rl_gym/deploy/pre_train/g1/motion.pt
```

## 3. Moi truong va dependency

Moi truong da kiem tra:

- Conda environment: `env_sim`
- Python: `3.12.13`
- PyTorch: `2.11.0+cu130`
- NumPy: `2.4.4`
- ROS 2: Jazzy
- `rclpy`, `message_filters`, `sensor_msgs`, `geometry_msgs` va
  `rosgraph_msgs` import duoc trong `env_sim`.

ROS package khai bao:

- `ament_cmake`
- `rclpy`
- `sensor_msgs`
- `geometry_msgs`
- `message_filters`
- `launch`
- `launch_ros`

PyTorch va NumPy duoc cung cap boi `env_sim`, khong cai bang `rosdep`.

## 4. ROS interface

### Subscribe

| Topic | Message | Muc dich |
|---|---|---|
| `/imu` | `sensor_msgs/msg/Imu` | Quaternion va angular velocity cua pelvis |
| `/joint/joint_states` | `sensor_msgs/msg/JointState` | Vi tri va van toc 43 khop |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Lenh `vx`, `vy`, `wz` |

### Publish

| Topic | Message | Muc dich |
|---|---|---|
| `/joint/joint_command` | `sensor_msgs/msg/JointState` | Position target cho robot |

Topic dung relative name trong source va duoc remap trong launch:

```text
imu                 -> /imu
joint_states        -> /joint/joint_states
joint_command       -> /joint/joint_command
cmd_vel             -> /cmd_vel
```

Cach nay giu kha nang them namespace giong package H1.

QoS cho IMU, joint state va joint command:

- Reliability: `RELIABLE`
- Durability: `VOLATILE`
- History: `KEEP_LAST`
- Depth: `10`

Day phu hop voi endpoint Isaac Sim da do truc tiep.

## 5. Khoi dong va vong doi node

Controller lam giong luong khoi dong H1:

1. Node khoi tao ROS, parameter, subscriber va publisher.
2. Load TorchScript policy.
3. Goi `policy.eval()` va `policy.reset_memory()`.
4. Khoi tao `previous_action = zeros(12)` va command bang zero.
5. Cho cap `/imu` va `/joint/joint_states` dong bo dau tien.
6. Kiem tra du lieu va mapping joint.
7. Chay inference dau tien ngay tren cap sensor hop le dau tien.
8. Publish joint command ngay sau khi co action.

Khong them:

- Ramp 2 giay.
- State chuyen tu home sang policy.
- Service hoac topic enable.
- Wall-clock `sleep()`.
- ROS timer rieng de chay policy.

Dieu kien van hanh la robot da o home chinh xac khi nguoi dung bam Play trong
Isaac Sim.

## 6. Dong bo va tan so

### Sensor

Isaac Sim dang publish:

```text
sensor_dt = 0.005 s
sensor_frequency = 200 Hz simulation time
```

`message_filters.TimeSynchronizer` dong bo exact timestamp cua:

- `/imu`
- `/joint/joint_states`

Queue size la `10`, giong package H1.

### Policy

Policy duoc train voi:

```text
policy_dt = 0.02 s
policy_frequency = 50 Hz
```

Node khong inference moi callback 200 Hz. Tai callback dong bo:

```text
if callback_count % 4 == 0:
    run_policy()
callback_count += 1
```

Callback hop le dau tien inference ngay. Voi sensor hoat dong dung 200 Hz,
policy inference moi 4 cap sensor, tuong duong 50 Hz. Neu callback bi mat,
controller van doi du 4 callback hop le nen policy se cham hon 50 Hz theo
simulation time.

### Command publication

Joint command duoc publish tai moi callback dong bo, danh dinh 200 Hz:

- O callback inference: cap nhat action va target moi.
- O ba callback con lai: publish lai target gan nhat.

Nhu vay action policy thay doi o 50 Hz, con Isaac Sim joint controller nhan va
giu target o 200 Hz.

### Simulation clock

Launch dat:

```text
use_sim_time = true
```

Node dung bo dem callback de scheduling. Timestamp trong sensor message duoc
dung de dong bo IMU/joint state va phat hien simulation time quay lui. `/clock`
cung cap ROS simulation time cho header command va logging.

Neu timestamp quay lui do Stop/Play hoac Reset:

- Goi `policy.reset_memory()`.
- Dat `previous_action = zeros(12)`.
- Dat action va target policy ve gia tri ban dau.
- Dat policy step va gait phase ve zero.
- Dat callback count ve zero.
- Inference lai ngay tren cap sensor hop le dau tien sau reset.

Khong ramp sau reset; Isaac Sim tiep tuc chiu trach nhiem dua robot ve home.

## 7. Observation G1 47 chieu

Observation phai la `numpy.float32`, shape `(47,)`, sau do doi thanh tensor
`torch.float32` shape `(1, 47)`.

```text
obs = [
    0.25 * pelvis_angular_velocity,                 # 3
    projected_gravity_in_pelvis,                   # 3
    [2.0 * vx, 2.0 * vy, 0.25 * wz],               # 3
    1.0 * (q_leg - q_default),                     # 12
    0.05 * dq_leg,                                 # 12
    previous_action,                               # 12
    sin(2*pi*phase),                               # 1
    cos(2*pi*phase),                               # 1
]
```

### IMU: `obs[0:6]`

`imu.orientation` cua ROS co field order `x, y, z, w`. Khi tinh theo helper cua
deploy G1, doi thanh:

```text
[w, x, y, z]
```

Angular velocity:

```text
obs[0:3] = 0.25 * [wx, wy, wz]
```

Projected gravity:

```text
gx =  2 * (-qz*qx + qw*qy)
gy = -2 * ( qz*qy + qw*qx)
gz =  1 - 2 * (qw*qw + qz*qz)
```

Khi robot dung thang:

```text
projected_gravity ~= [0, 0, -1]
```

Khong dung `linear_acceleration` va khong nhan `9.81`.

Gia dinh `pelvis_imu` thang hang voi pelvis:

```text
+x: phia truoc robot
+y: ben trai robot
+z: huong len
```

### Command: `obs[6:9]`

`/cmd_vel` mang don vi vat ly:

```text
vx = Twist.linear.x       [m/s]
vy = Twist.linear.y       [m/s]
wz = Twist.angular.z      [rad/s]
```

Clip truoc khi scale:

```text
vx in [-0.8, 0.8]
vy in [-0.5, 0.5]
wz in [-1.57, 1.57]
```

Sau do:

```text
obs[6:9] = [2.0*vx, 2.0*vy, 0.25*wz]
```

Khi chua nhan `/cmd_vel`, command bang zero. De giong H1, node giu lenh gan nhat
va khong tu dat zero theo wall-clock timeout.

### Joint state: `obs[9:33]`

Khong dung thu tu mang 43 khop tu Isaac Sim. Tao mapping bang `JointState.name`
va lay dung 12 khop theo thu tu policy:

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
```

Default angle:

```text
[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
 -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
```

Observation:

```text
obs[9:21]  = q_leg - q_default
obs[21:33] = 0.05 * dq_leg
```

Mapping duoc tao lai neu danh sach `JointState.name` thay doi.

### Previous action: `obs[33:45]`

Day la raw output cua policy o lan inference truoc:

```text
obs_k[33:45] = action_(k-1)
```

Khong nhan `action_scale` truoc khi dua vao observation.
Chi cap nhat `previous_action` sau khi inference hoan thanh.

### Gait phase: `obs[45:47]`

```text
period = 0.8 s
policy_dt = 0.02 s
40 policy steps / gait cycle
```

De khop Python deploy real cua G1:

```text
policy_step += 1
phase = (policy_step * 0.02 % 0.8) / 0.8
```

Lan inference dau tien dung:

```text
phase = 0.025
```

Observation:

```text
obs[45] = sin(2*pi*phase)
obs[46] = cos(2*pi*phase)
```

## 8. Policy inference

Load model:

```text
torch.jit.load(policy_path, map_location="cpu")
```

Inference:

```text
with torch.inference_mode():
    action = policy(obs_tensor)
```

Bat buoc kiem tra:

- Input shape `(1, 47)`.
- Output shape `(1, 12)` hoac co the flatten thanh `(12,)`.
- Input va output la finite.
- Policy co method `reset_memory()`.

Khong clip action trong controller de giu hanh vi giong deploy real goc.

Target 12 khop chan:

```text
q_target_leg = q_default + 0.25 * action
```

Policy chay tren CPU de tranh them dong bo CUDA trong callback ROS.

## 9. Joint command 43 khop

Message command:

```text
sensor_msgs/msg/JointState
```

Moi callback, `name` dung thu tu 43 khop cua `joint_state.name` gan nhat.
Mang position duoc tao theo ten:

- 12 khop chan: target tu policy.
- Tat ca khop con lai: `0 rad`.

Bao gom:

- 3 khop waist.
- 14 khop arm/wrist cua G1.
- 14 khop Dex3.

Message:

```text
header.stamp = node.get_clock().now().to_msg()
name          = 43 joint names
position      = 43 position targets
velocity      = 43 zeros
effort        = 43 zeros
```

Isaac Sim articulation drive da duoc cau hinh stiffness/damping theo:

```text
RL_control/G1_control/robot_gym_config/
g1_29dof_with_dex3_unitree_rl_gym_policy.yaml
```

ROS node chi publish position target, khong tu tinh torque PD.

## 10. Validation va xu ly loi

Khong inference va khong publish command dau tien cho den khi:

- IMU va joint state co cung timestamp.
- Co du 12 policy joint.
- `name`, `position`, `velocity` co length hop le.
- Quaternion, angular velocity, joint position va velocity la finite.
- Quaternion co norm khac zero.

Neu mot sample sau do khong hop le:

- Bo sample do.
- Khong cap nhat LSTM, previous action hoac phase.
- Ghi warning co throttle.
- Sample hop le tiep theo tiep tuc tu state truoc.

Neu policy output sai shape hoac co NaN/Inf:

- Khong cap nhat action.
- Khong cap nhat previous action.
- Ghi error.
- Khong publish target moi tai callback loi.

## 11. Parameter launch

Launch cung cap:

| Parameter | Default |
|---|---|
| `policy_path` | Policy duoc install trong package |
| `use_sim_time` | `true` |
| `imu_topic` | `/imu` |
| `joint_states_topic` | `/joint/joint_states` |
| `joint_command_topic` | `/joint/joint_command` |
| `cmd_vel_topic` | `/cmd_vel` |
| `namespace` | rong |
| `use_namespace` | `false` |

Khong tao `publish_period_ms`, vi command cadence duoc quyet dinh boi sensor
callback 200 Hz. Parameter nay co trong vi du H1 nhung khong duoc source H1 su
dung.

## 12. Cach build va chay

### Build lan dau

Tu root repository:

```bash
conda activate env_sim
source /opt/ros/jazzy/setup.bash
cd RL_control/G1_policy
colcon build --symlink-install --packages-select g1_policy_controller
source install/setup.bash
```

### Khoi dong moi lan

Terminal ROS:

```bash
cd /home/khoa-ng/Job_Project/My_Code/humanoid_robot_control
conda activate env_sim
source /opt/ros/jazzy/setup.bash
source RL_control/G1_policy/install/setup.bash
```

Trong Isaac Sim:

1. Mo stage G1 da cau hinh.
2. Bao dam OmniGraph publish `/clock`, `/imu`, `/joint/joint_states`.
3. Bao dam OmniGraph subscribe `/joint/joint_command`.
4. Chua bam Play.

Trong terminal, chay controller truoc:

```bash
ros2 launch g1_policy_controller g1_policy_controller.launch.py
```

Sau khi node bao da san sang, bam Play trong Isaac Sim. Robot duoc dat dung
home boi cau hinh Isaac Sim va policy bat dau khi node nhan cap IMU/joint state
dong bo dau tien.

Lenh di chuyen vi du:

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

Dung controller bang `Ctrl+C`. Khi chay lai node, LSTM bat dau voi memory zero.

## 13. Ke hoach kiem thu

### Unit test

1. Observation numerical:
   - Dung numerical example trong `g1_policy_observation.md`.
   - So sanh day du 47 gia tri.

2. Joint mapping:
   - Dung thu tu 43 khop mau dang duoc Isaac Sim publish.
   - Xac nhan 12 vi tri va van toc duoc sap dung policy order.

3. IMU:
   - Quaternion identity cho gravity `[0, 0, -1]`.
   - Roll 10 do cho ket qua phu hop tai lieu.
   - Quaternion zero bi reject.

4. Timing:
   - Xac nhan inference o callback 0, 4, 8, 12...
   - Xac nhan khoang timestamp khong anh huong den callback decimation.
   - Xac nhan 200 callback tao 50 inference.
   - Xac nhan command publish moi callback.

5. Temporal contract:
   - Observation dung action cu.
   - Previous action chi cap nhat sau inference.
   - Phase dau la `0.025`, chu ky 40 policy step.

6. Reset:
   - Timestamp quay lui goi `reset_memory()`.
   - Previous action, policy step va phase quay ve state ban dau.

7. Output:
   - Action shape 12 tao dung target `q_default + 0.25*action`.
   - Command co 43 ten, 43 position, 43 velocity va 43 effort.
   - 31 khop khong thuoc policy co target zero.

### Build test

```bash
cd RL_control/G1_policy
colcon build --symlink-install --packages-select g1_policy_controller
colcon test --packages-select g1_policy_controller
colcon test-result --verbose
```

### Integration test voi Isaac Sim

1. Kiem tra topic type va endpoint bang `ros2 topic info -v`.
2. Xac nhan IMU/joint timestamp cach nhau `0.005 s` va trung nhau.
3. Xac nhan `/joint/joint_command` co publisher sau khi launch node.
4. Do tan so:
   - Sensor/command xap xi 200 Hz simulation time.
   - Policy inference xap xi 50 Hz simulation time.
5. Chay zero `/cmd_vel`, robot giu can bang/step theo policy.
6. Thu `vx`, `vy`, `wz` nho truoc, sau do tang dan trong gioi han.
7. Stop/Play Isaac Sim va xac nhan LSTM reset, khong giu history cu.

## 14. Thu tu trien khai

1. Scaffold package ROS 2 va khai bao dependency.
2. Copy va kiem tra hash policy.
3. Viet helper joint mapping, quaternion/projected gravity va observation.
4. Viet node subscriber, exact-time synchronization va command publisher.
5. Them scheduling policy moi 4 callback giong package H1.
6. Them reset khi simulation time quay lui.
7. Viet launch file va parameter.
8. Viet unit test numerical va timing.
9. Build/test trong `env_sim`.
10. Integration test tren Isaac Sim voi topic hien tai.

## 15. Cac quyet dinh da khoa

- Khong ramp ve home.
- Khong can nut/service enable policy.
- Policy inference bat dau o sample dong bo dau tien.
- Sensor va command cadence la 200 Hz simulation time.
- Policy cadence la moi 4 callback, tuong duong 50 Hz khi sensor du 200 Hz.
- Timestamp chi dung cho exact synchronization va phat hien time reset.
- Dung `/cmd_vel` voi don vi vat ly.
- Dung exact synchronization cua IMU va joint state.
- Actor observation khong co base linear velocity.
- Khong dung IMU linear acceleration.
- Khong them 31 khop upper body/hand vao observation.
- Publish position target cho du 43 khop.
- Policy chay CPU va giu nguyen TorchScript LSTM goc.
