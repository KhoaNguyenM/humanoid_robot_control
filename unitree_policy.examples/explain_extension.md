# Giải thích Chi tiết Extension: `isaacsim.robot.policy.examples`

> **Phiên bản:** 4.1.11  
> **Đường dẫn cài đặt:** `exts/isaacsim.robot.policy.examples/`  
> **License:** Apache-2.0 © 2024–2025 NVIDIA CORPORATION & AFFILIATES

---

## 1. Tổng quan

Extension `isaacsim.robot.policy.examples` là một **module thực thi chính sách Reinforcement Learning (RL) trong Isaac Sim**. Mục tiêu của nó là cung cấp một khung (framework) chuẩn để:

1. **Tải** một policy đã được huấn luyện từ file (TorchScript `.pt`)
2. **Đọc cấu hình** môi trường huấn luyện từ file YAML
3. **Tính observation** từ trạng thái robot trong simulation
4. **Chạy inference** qua policy để ra action
5. **Áp dụng action** lên robot trong Isaac Sim

Extension cung cấp ví dụ cho 4 robot: **Unitree H1** (humanoid), **ANYmal** (quadruped), **Boston Dynamics Spot** (quadruped), và **Franka** (manipulator).

---

## 2. Cấu trúc Thư mục

```
isaac/exts/isaacsim.robot.policy.examples/
│
├── config/
│   └── extension.toml                  # Metadata, dependencies, test args
│
│
├── isaacsim/robot/policy/examples/
│   │
│   ├── __init__.py
│   │
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── policy_controller.py        # ★ Base class: PolicyController
│   │   └── config_loader.py            # ★ Parser YAML env config
│   │
│   ├── robots/
│   │   ├── __init__.py                 # Export tất cả robot policy classes
│   │   ├── h1.py                       # ★ H1FlatTerrainPolicy
│   │   ├── anymal.py                   # AnymalFlatTerrainPolicy
│   │   ├── spot.py                     # SpotFlatTerrainPolicy
│   │   └── franka.py                   # FrankaOpenDrawerPolicy
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── actuator_network.py         # SEA actuator networks (LSTM + MLP)
│   │
│   └── tests/
│       ├── test_h1.py
│       ├── test_anymal.py
│       ├── test_spot.py
│       └── test_franka.py
│
└── PACKAGE-LICENSES/

# Standalone scripts (chạy độc lập không cần UI):
isaac/standalone_examples/api/isaacsim.robot.policy.examples/
    ├── h1_standalone.py
    ├── anymal_standalone.py
    └── spot_standalone.py
```

---

## 3. Kiến trúc Phần mềm

### 3.1 Sơ đồ Kế thừa Lớp

```
omni.isaac.core → BaseController
                        │
                        ▼
              PolicyController          (controllers/policy_controller.py)
              ├── load_policy()         → load .pt TorchScript + parse YAML
              ├── initialize()          → set gains, limits, articulation props
              ├── _compute_action()     → torch inference
              ├── _compute_observation()→ abstract (override ở subclass)
              └── forward()             → abstract (override ở subclass)
                        │
          ┌─────────────┼──────────────────────┐
          ▼             ▼                       ▼
  H1FlatTerrainPolicy  AnymalFlatTerrainPolicy  SpotFlatTerrainPolicy ...
  (robots/h1.py)       (robots/anymal.py)       (robots/spot.py)
```

---

## 4. Module Chi tiết

### 4.1 `policy_controller.py` — Base Controller

Đây là lớp nền tảng của toàn bộ extension.

#### Constructor `__init__()`

```python
PolicyController(name, prim_path, root_path, usd_path, position, orientation)
```

- Kiểm tra nếu prim tại `prim_path` chưa tồn tại → tạo mới Xform và gắn USD reference
- Tạo `SingleArticulation` — đối tượng đại diện cho robot trong PhysX

#### `load_policy(policy_file_path, policy_env_path)`

```python
# Tải TorchScript policy
file_content = omni.client.read_file(policy_file_path)[2]  # đọc từ Nucleus server
file = io.BytesIO(memoryview(file_content).tobytes())
self.policy = torch.jit.load(file)                          # load TorchScript model

# Parse env config
self.policy_env_params = parse_env_config(policy_env_path)

# Lấy physics params từ YAML
self._decimation, self._dt, self.render_interval = get_physics_properties(...)
```

> **Lưu ý:** `omni.client.read_file()` hỗ trợ đọc cả từ **Nucleus server** (URL `http://`) lẫn đường dẫn local.

#### `initialize()`

```python
self.robot.initialize(physics_sim_view)
self.robot.get_articulation_controller().set_effort_modes("force")
self.robot.get_articulation_controller().switch_control_mode("position")

# Đọc PD gains, limits, default positions từ YAML
max_effort, max_vel, stiffness, damping, default_pos, default_vel = \
    get_robot_joint_properties(self.policy_env_params, self.robot.dof_names)

self.robot._articulation_view.set_gains(stiffness, damping)
self.robot._articulation_view.set_max_efforts(max_effort)
self.robot._articulation_view.set_max_joint_velocities(max_vel)
```

#### `_compute_action(obs: np.ndarray) → np.ndarray`

```python
with torch.no_grad():
    obs_tensor = torch.from_numpy(obs).view(1, -1).float()
    action = self.policy(obs_tensor).detach().view(-1).numpy()
return action
```

Chạy **forward pass** qua TorchScript model, không tính gradient (inference mode).

---

### 4.2 `config_loader.py` — Parser YAML

Module này đọc và giải mã file `h1_env.yaml` để lấy tham số cấu hình.

#### `parse_env_config(env_config_path)` → `dict`

```python
# Dùng custom YAML SafeLoader có thể bỏ qua các tag Python không biết
# Hỗ trợ !!python/tuple thay vì lỗi
data = yaml.load(file, Loader=SafeLoaderIgnoreUnknown)
```

Trả về dict Python từ toàn bộ YAML.

#### `get_robot_joint_properties(data, joint_names)` → `tuple`

Trả về: `(effort_limits, velocity_limits, stiffness, damping, default_pos, default_vel)`

Dùng **glob pattern matching** (`fnmatch`) để map tên joint:
```python
# Ví dụ pattern ".*_hip_yaw" khớp với "left_hip_yaw", "right_hip_yaw"
if fnmatch.fnmatch(joint, pattern.replace(".", "*") + "*"):
    stiffness_inorder.append(stiffness[pattern])
```

#### `get_physics_properties(data)` → `(decimation, dt, render_interval)`

Đọc từ `data['decimation']`, `data['sim']['dt']`, `data['sim']['render_interval']`.

#### Các hàm phụ khác:
| Hàm | Mô tả |
|-----|-------|
| `get_articulation_props(data)` | Trả về solver iteration counts, sleep threshold, self-collision |
| `get_observations(data)` | Trả về cấu trúc observation từ YAML |
| `get_action(data)` | Trả về cấu hình action |
| `get_physx_settings(data)` | Trả về cấu hình GPU PhysX |

---

### 4.3 `robots/h1.py` — `H1FlatTerrainPolicy`

Robot: **Unitree H1 Humanoid** — 19 DOF (degrees of freedom)

#### Policy Files (trên Nucleus Server)

```
Assets Root + /Isaac/Samples/Policies/H1_Policies/h1_policy.pt    ← TorchScript model
Assets Root + /Isaac/Samples/Policies/H1_Policies/h1_env.yaml     ← Env config
Assets Root + /Isaac/Robots/Unitree/H1/h1.usd                     ← Robot USD
```

#### Cấu hình (từ YAML `h1_env.yaml`)

| Tham số | Giá trị |
|---------|---------|
| `sim.dt` | 0.005 s (200 Hz physics) |
| `decimation` | 4 (policy chạy ở 50 Hz) |
| `render_interval` | 4 |
| `action_scale` | 0.5 |
| Số joints | 19 |

#### PD Gains từ YAML

| Nhóm khớp | Joints | Stiffness (K_p) | Damping (K_d) | Effort Limit |
|-----------|--------|-----------------|----------------|--------------|
| **Legs** | hip_yaw, hip_roll | 150.0 | 5.0 | 300 Nm |
| **Legs** | hip_pitch, knee | 200.0 | 5.0 | 300 Nm |
| **Legs** | torso | 200.0 | 5.0 | 300 Nm |
| **Feet** | ankle | 20.0 | 4.0 | 100 Nm |
| **Arms** | shoulder_pitch/roll/yaw, elbow | 40.0 | 10.0 | 300 Nm |

#### Default Joint Positions (từ `init_state.joint_pos`)

```yaml
.*_hip_yaw:    0.0   rad
.*_hip_roll:   0.0   rad
.*_hip_pitch: -0.28  rad
.*_knee:       0.79  rad
.*_ankle:     -0.52  rad
torso:         0.0   rad
.*_shoulder_pitch: 0.28 rad
.*_shoulder_roll:  0.0  rad
.*_shoulder_yaw:   0.0  rad
.*_elbow:      0.52  rad
```

#### `_compute_observation(command)` → `np.ndarray` (shape: 69)

```
Observation vector (69 chiều):
┌─────────────────────────────────────────────────────────┐
│ [0:3]   lin_vel_b     = R_BI @ lin_vel_I   (body frame) │
│ [3:6]   ang_vel_b     = R_BI @ ang_vel_I   (body frame) │
│ [6:9]   gravity_b     = R_BI @ [0, 0, -1]  (projected)  │
│ [9:12]  command       = [vx, vy, ω_z]                   │
│ [12:31] joint_pos_rel = joint_pos - default_pos (19 DOF)│
│ [31:50] joint_vel     = joint velocities        (19 DOF) │
│ [50:69] prev_action   = action từ bước trước   (19 DOF) │
└─────────────────────────────────────────────────────────┘
```

**Cách tính body-frame:**
```python
R_IB = quat_to_rot_matrix(q_IB)   # world → body rotation matrix
R_BI = R_IB.transpose()            # body → world inverse
lin_vel_b = R_BI @ lin_vel_I       # project world velocity to body frame
gravity_b = R_BI @ [0, 0, -1]     # project gravity to body frame
```

#### `forward(dt, command)`

```python
# Chỉ chạy policy mỗi `_decimation` bước physics
if self._policy_counter % self._decimation == 0:
    obs = self._compute_observation(command)
    self.action = self._compute_action(obs)
    self._previous_action = self.action.copy()

# Apply action: target_pos = default_pos + action * action_scale
action = ArticulationAction(
    joint_positions = self.default_pos + (self.action * self._action_scale)
)
self.robot.apply_action(action)
self._policy_counter += 1
```

> **Decimation:** Policy inference (50Hz) và physics step (200Hz) tách nhau để giảm tải tính toán.

---

### 4.4 `utils/actuator_network.py` — SEA Actuator Networks

Module này không được dùng cho H1, nhưng cung cấp hai loại actuator network cho các robot phức tạp hơn (như ANYmal).

#### `LstmSeaNetwork` — SEA với LSTM

Series Elastic Actuator (SEA) network dùng **LSTM 2 layers, 8 hidden units**:

```python
# Hidden state: (2 layers, 12 joints, 8 hidden units)
_hidden_state = torch.zeros((2, 12, 8))
_cell_state   = torch.zeros((2, 12, 8))

# Input mỗi step: position error + joint velocity
actuator_net_input[:, 0, 0] = action + default_pos - joint_pos  # pos error
actuator_net_input[:, 0, 1] = clip(joint_vel, -20, 20)          # velocity

# Output: torque, clipped tại ±80 Nm
torques = network(input, (hidden, cell)).clip(-80, 80)
```

#### `SeaNetwork` — SEA với MLP

MLP 3 lớp: `6 → 32 → 32 → 1` với activation **Softsign**:

```
Input (6 features mỗi joint):
  - vel_history[t-delay0], vel_history[t-delay1], vel_history[t]   ← velocity
  - pos_history[t-delay0], pos_history[t-delay1], pos_history[t]   ← position error

delay0 = 8 steps, delay1 = 3 steps (mô phỏng độ trễ actuator)

Output: torque = 20.0 × MLP(input)
```

Weights được load từ file CSV (không phải checkpoint PyTorch).

---

## 5. File Cấu hình `h1_env.yaml` — Phân tích Chi tiết

### 5.1 Simulation Settings

```yaml
sim:
  dt: 0.005              # Physics timestep: 200 Hz
  render_interval: 4     # Render mỗi 4 steps
  use_gpu_pipeline: true # GPU physics pipeline
  device: cuda:0
  physx:
    solver_type: 1       # TGS solver (1=TGS, 0=PGS)
    use_gpu: true
    enable_ccd: false    # Continuous collision detection tắt
    enable_stabilization: true
```

### 5.2 Observation Space (69 chiều)

```yaml
observations:
  policy:
    base_lin_vel:     noise: uniform(-0.1, 0.1)    # 3 dims
    base_ang_vel:     noise: uniform(-0.2, 0.2)    # 3 dims
    projected_gravity: noise: uniform(-0.05, 0.05) # 3 dims
    velocity_commands: (no noise)                  # 3 dims (vx, vy, ωz)
    joint_pos:        noise: uniform(-0.01, 0.01)  # 19 dims (relative to default)
    joint_vel:        noise: uniform(-1.5, 1.5)    # 19 dims
    actions:          (no noise)                   # 19 dims (last action)
    # height_scan: null  ← không dùng (flat terrain)
```

### 5.3 Action Space (19 chiều)

```yaml
actions:
  joint_pos:
    joint_names: [".*"]    # Tất cả 19 joints
    scale: 0.5             # action_scale
    use_default_offset: true  # target = action * scale + default_pos
```

### 5.4 Reward Functions

| Reward | Hàm | Weight |
|--------|-----|--------|
| Track linear velocity XY | `track_lin_vel_xy_yaw_frame_exp` | +1.0 |
| Track angular velocity Z | `track_ang_vel_z_world_exp` | +1.0 |
| Angular velocity XY penalty | `ang_vel_xy_l2` | -0.05 |
| Joint acceleration penalty | `joint_acc_l2` | -1.25e-7 |
| Action rate penalty | `action_rate_l2` | -0.005 |
| Feet air time | `feet_air_time_positive_biped` | +1.0 |
| Flat orientation | `flat_orientation_l2` | -1.0 |
| Ankle joint limits | `joint_pos_limits` | -1.0 |
| Termination penalty | `is_terminated` | -200.0 |
| Feet slide penalty | `feet_slide` | -0.25 |
| Hip deviation | `joint_deviation_l1` (hip_yaw, roll) | -0.2 |
| Arm deviation | `joint_deviation_l1` (shoulders, elbow) | -0.2 |
| Torso deviation | `joint_deviation_l1` (torso) | -0.1 |

### 5.5 Termination Conditions

```yaml
terminations:
  time_out:      # Episode timeout: 20s
  base_contact:  # Robot bị ngã (torso_link chạm đất > 1N)
```

---

## 6. Standalone Script `h1_standalone.py`

Script chạy ví dụ H1 mà không cần mở GUI Isaac Sim đầy đủ.

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Tạo world
my_world = World(stage_units_in_meters=1.0,
                 physics_dt=1/200,      # 200 Hz physics
                 rendering_dt=8/200)    # 25 Hz render

# Spawn robot
h1 = H1FlatTerrainPolicy(
    prim_path="/World/H1_0",
    position=np.array([0, 0, 1.05]),   # 1.05m trên mặt đất
)

# Command pattern tự động:
#   0–80 steps:   forward  [0.5, 0, 0]
#   80–130 steps: rotate   [0.5, 0, 0.5]
#   130–200 steps: sideways [0, 0, 0.5]
```

**Cách chạy:**
```bash
cd isaac
./python.sh standalone_examples/api/isaacsim.robot.policy.examples/h1_standalone.py
# Tùy chọn:
./python.sh h1_standalone.py --num-robots 4
./python.sh h1_standalone.py --env-url /Isaac/Environments/Grid/default_environment.usd
```

---

## 7. Luồng Thực thi Đầy đủ

```
[STARTUP]
Isaac Sim khởi động
    └─► Extension Manager load isaacsim.robot.policy.examples
            └─► Đăng ký menu "Robotics Examples > POLICY > Humanoid"

[LOAD]
Người dùng nhấn LOAD
    └─► H1FlatTerrainPolicy.__init__()
            ├─► Thêm h1.usd vào stage tại /World/H1
            ├─► load_policy(h1_policy.pt, h1_env.yaml)
            │       ├─► omni.client.read_file() → đọc từ Nucleus
            │       ├─► torch.jit.load() → load TorchScript model
            │       └─► parse_env_config() → đọc YAML
            └─► Tạo SingleArticulation object

[PLAY - First Step]
on_physics_step() được gọi lần đầu
    └─► robot.initialize()
            ├─► robot.initialize(physics_sim_view)
            ├─► switch_control_mode("position")
            ├─► set_gains(stiffness, damping)    ← từ YAML
            └─► set_max_efforts(max_effort)       ← từ YAML

[PLAY - Every Step]
Physics step (dt=0.005s, 200Hz)
    └─► on_physics_step(dt)
            └─► robot.forward(dt, command=[vx, vy, wz])
                    ├─► [counter % 4 == 0] → policy inference:
                    │       ├─► _compute_observation()
                    │       │       ├─► get_world_pose() → q_IB
                    │       │       ├─► R_BI = transpose(quat_to_rot_matrix(q_IB))
                    │       │       ├─► lin_vel_b = R_BI @ lin_vel_world
                    │       │       ├─► ang_vel_b = R_BI @ ang_vel_world
                    │       │       ├─► gravity_b = R_BI @ [0,0,-1]
                    │       │       └─► obs = concat(vel_b, ang_b, grav_b, cmd, Δpos, vel, prev_act)
                    │       └─► _compute_action(obs)
                    │               └─► policy(obs_tensor) → action (19,)
                    └─► apply_action(default_pos + action * 0.5)
```

---

## 8. Dependencies

Khai báo trong `config/extension.toml`:

```toml
[dependencies]
"isaacsim.core.api"       = {}   # World, SingleArticulation, BaseController
"isaacsim.core.nodes"     = {}   # OmniGraph nodes
"isaacsim.sensors.physics"= {}   # Physics-based sensors
"isaacsim.storage.native" = {}   # get_assets_root_path(), omni.client
```

**Python packages sử dụng:**
- `torch` — TorchScript model loading & inference
- `numpy` — Vector math, observation construction
- `yaml` — Đọc env config
- `omni.client` — Đọc file từ Nucleus server
- `carb` — Logging
- `omni.physx` — Flush physx changes