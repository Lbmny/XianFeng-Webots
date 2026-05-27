"""RescueRobot Controller - Webots Python 控制器
桥梁：Webots传感器 → NumPy数据 → PyTorch模型 → DeepSeek决策 → 运动指令
AI辅助生成比例：90%
"""

from controller import Supervisor, Keyboard
import numpy as np
import json
import sys
import os
from collections import deque

# 添加项目根目录到path，以便导入训练好的模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 创新点一：地形风险感知路径规划
from risk_mapper import RiskMapper
from path_planner import BaselinePlanner, RiskAwareAStar
from deepseek_planner import DeepSeekV4Planner

# 创新点三：能源自适应策略 + 极限信标
from energy_mdp import EnergyMDP
from energy_predictor import EnergyPredictor
from energy_llm import EnergyLLMDecider

# 创新点二：退化环境多模态检测 + 三模控制
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from detectors.yolo_dual_stream import DualStreamDetector
from detectors.detection_v4 import DetectionV4Analyzer
from benchmark_runner import run_all as run_benchmark


class DegradationEngine:
    """环境退化引擎 — Supervisor API 控制 Fog/Light + 图像后处理

    四级退化模式：
      1 = clean        无雾，满光
      2 = light_fog    轻雾，能见度50m
      3 = heavy_smoke  浓烟，能见度15m + Perlin噪声后处理
      4 = dark         全黑，能见度8m + 亮度降至15%
    """

    LEVELS = {
        "1": ("clean",       0,   1.0,  [0.7, 0.7, 0.7]),
        "2": ("light_fog",  50,   0.8,  [0.7, 0.7, 0.7]),
        "3": ("heavy_smoke", 15,  0.5,  [0.4, 0.4, 0.4]),
        "4": ("dark",        8,  0.05,  [0.1, 0.1, 0.1]),
    }

    def __init__(self, robot):
        self.robot = robot
        self.current = "clean"
        self.fog_node = None
        self.sun_node = None
        self._init_supervisor()

    def _init_supervisor(self):
        self.fog_node = self.robot.getFromDef("FOG")
        self.sun_node = self.robot.getFromDef("SUN")
        if self.fog_node is None:
            print("[Degradation] WARNING: FOG node not found — world params disabled")
        if self.sun_node is None:
            print("[Degradation] WARNING: SUN node not found — light control disabled")
        self._apply_world("clean")
        print("[Degradation] Engine ready — press 1/2/3/4 to switch modes")

    def set_level(self, key_char: str):
        if key_char not in self.LEVELS:
            return
        name = self.LEVELS[key_char][0]
        if name == self.current:
            return
        self.current = name
        self._apply_world(name)
        print(f"[Degradation] Switched to: {name}")

    def _apply_world(self, name: str):
        for _key, (n, vr, li, fc) in self.LEVELS.items():
            if n == name:
                if self.fog_node:
                    self.fog_node.getField("visibilityRange").setSFFloat(float(vr))
                    self.fog_node.getField("color").setSFColor(fc)
                if self.sun_node:
                    self.sun_node.getField("intensity").setSFFloat(li)
                return

    def process_rgb(self, image: np.ndarray) -> np.ndarray:
        """后处理RGB图像 — 叠加退化效果"""
        img = image.astype(np.float32)
        h, w = img.shape[:2]

        if self.current == "light_fog":
            img = img * 0.85 + 200 * 0.15

        elif self.current == "heavy_smoke":
            coarse = np.random.rand(max(h // 20, 1), max(w // 20, 1)).astype(np.float32)
            coarse = np.kron(coarse, np.ones((20, 20)))[:h, :w]
            img = img + (coarse[:, :, None] - 0.5) * 50
            img = img + np.random.randn(h, w, 1).astype(np.float32) * 10
            img = img * 0.65 + 128 * 0.35

        elif self.current == "dark":
            img = img * 0.15
            img = img + np.random.randn(h, w, 1).astype(np.float32) * 3

        return np.clip(img, 0, 255).astype(np.uint8)

    def process_ir(self, image: np.ndarray) -> np.ndarray:
        """后处理IR图像 — 红外穿透力强，退化较RGB更轻"""
        img = image.astype(np.float32)
        h, w = img.shape[:2]

        if self.current == "light_fog":
            img = img * 0.9 + 180 * 0.1

        elif self.current == "heavy_smoke":
            img = img + np.random.randn(h, w, 1).astype(np.float32) * 3

        elif self.current == "dark":
            pass  # 红外在黑暗中不受影响

        return np.clip(img, 0, 255).astype(np.uint8)


class RescueRobotController:
    """两栖救灾机器人控制器"""

    # 水域边界（与 disaster_world.wbt 中 water_area 位置对齐）
    WATER_ZONE_X = (-19.0, 19.0)
    WATER_ZONE_Y = (-6.0, 6.0)

    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.step_count = 0

        # === 传感器初始化 ===
        self._init_sensors()

        # === 臂摆电机（位置控制，AI可调角度）===
        self.arm_motors = {}
        default_angles = {"arm_fl": 0.0, "arm_fr": 0.0, "arm_rl": 0.0, "arm_rr": 0.0}
        for name in ["arm_fl", "arm_fr", "arm_rl", "arm_rr"]:
            m = self.robot.getDevice(name)
            if m:
                m.setPosition(default_angles[name])
                m.setVelocity(1.0)
                self.arm_motors[name] = m
        self.arm_passive_mode()  # 默认被动柔顺

        # === 全驱履带：扫描所有驱动电机，按名称分组左/右 ===
        self.left_motors = []
        self.right_motors = []
        for i in range(self.robot.getNumberOfDevices()):
            dev = self.robot.getDeviceByIndex(i)
            name = dev.getName()
            # 驱动电机: wheel_fl/rl + drive_L* = 左侧; wheel_fr/rr + drive_R* = 右侧
            if name.startswith("wheel_f") and name.endswith(("l", "L")):
                dev.setPosition(float('inf')); dev.setVelocity(0.0)
                self.left_motors.append(dev)
            elif name.startswith("wheel_f") and name.endswith(("r", "R")):
                dev.setPosition(float('inf')); dev.setVelocity(0.0)
                self.right_motors.append(dev)
            elif name.startswith("wheel_r") and name.endswith(("l", "L")):
                dev.setPosition(float('inf')); dev.setVelocity(0.0)
                self.left_motors.append(dev)
            elif name.startswith("wheel_r") and name.endswith(("r", "R")):
                dev.setPosition(float('inf')); dev.setVelocity(0.0)
                self.right_motors.append(dev)
            elif name.startswith("drive_L"):
                dev.setPosition(float('inf')); dev.setVelocity(0.0)
                self.left_motors.append(dev)
            elif name.startswith("drive_R"):
                dev.setPosition(float('inf')); dev.setVelocity(0.0)
                self.right_motors.append(dev)
        print(f"[Controller] Arm motors: {list(self.arm_motors.keys())}")
        print(f"[Controller] Left drive: {[m.getName() for m in self.left_motors]}")
        print(f"[Controller] Right drive: {[m.getName() for m in self.right_motors]}")

        # === 信标灯 ===
        self.beacon = self.robot.getDevice("beacon_light")

        # === 键盘 ===
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)

        # === 退化引擎 ===
        self.degradation = DegradationEngine(self.robot)

        # === 状态 ===
        self.mode = "manual"  # manual / ai_local / ai_cloud / beacon / path_follow
        self.current_terrain = "unknown"
        self.in_water = False
        self.water_thrust_L = 0.0  # 左电推 0-1
        self.water_thrust_R = 0.0  # 右电推 0-1
        self._prev_gps = None  # 水流估计用
        self._prev_time = 0.0
        self.water_flow_est = (0.0, 0.0)  # (vx, vy) m/s

        # === 手动驾驶状态 ===
        self._manual_speed = 0.0       # 当前线速度
        self._manual_turn = 0.0        # 当前角速度
        self._manual_gear = 2          # 速度档位 1/2/3
        self._manual_arm_angle = 0.0   # 臂摆角度

        # === 路径跟随状态 ===
        self._path_targets = []        # 待跟随路径点列表 [(x,y), ...]
        self._path_target_idx = 0      # 当前目标点索引
        self._path_speed = 6.0         # 跟随速度

        # === 电池 ===
        self.robot.batterySensorEnable(self.timestep)

        # === 创新点一：风险感知路径规划 ===
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "terrain_seg_best.pt")
        self.risk_mapper = None
        if os.path.exists(model_path):
            try:
                self.risk_mapper = RiskMapper(model_path)
                self._has_ai_planner = True
                print(f"[Controller] RiskMapper loaded — AI path planning ready")
            except Exception as e:
                print(f"[Controller] RiskMapper init failed: {e}")
                self._has_ai_planner = False
        else:
            print(f"[Controller] Model not found at {model_path} — AI planner disabled")
            self._has_ai_planner = False

        # DeepSeek 双层级LLM决策
        self.v4_planner = DeepSeekV4Planner()
        self._v4_enabled = True
        self._r1_url = "http://127.0.0.1:11434/api/generate"
        self._r1_enabled = self._check_r1()

        # 路径规划结果缓存
        self.robot_node = self.robot.getSelf()  # 用于瞬移
        self._last_path = None
        self._last_risk_grid = None
        self._benchmark_results = []

        # 降级模式: 0=正常, 1=R1掉线, 2=V4掉线, 3=全掉线
        self._degrade_level = 0

        # === 创新点三：能源自适应策略 + 极限信标 ===
        # 30步历史缓冲
        self._terrain_hist = deque(maxlen=30)
        self._flow_hist = deque(maxlen=30)
        self._load_hist = deque(maxlen=30)
        self._speed_hist = deque(maxlen=30)
        self._power_hist = deque(maxlen=30)

        # 能源决策模块
        energy_model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                         "models", "energy_lstm_best.pt")
        self.energy_predictor = EnergyPredictor(energy_model_path)
        self.energy_mdp = EnergyMDP()
        self.energy_llm = EnergyLLMDecider(
            r1_url=self._r1_url,
            r1_enabled=self._r1_enabled,
            v4_enabled=self._v4_enabled)
        self._current_load_kg = 0.0
        self._energy_mode = False
        self._energy_interval = 5.0  # 每5秒自动评估
        self._last_energy_time = 0.0
        self._last_energy_result = None

        # === 创新点二：退化环境多模态检测（懒加载，首按T键时初始化）===
        self.detector = None
        self._detector_init_tried = False
        self.v4_detection = None  # V4研判器也懒加载

        print("[Controller] RescueRobot initialized successfully")
        print(f"[Controller] Timestep: {self.timestep}ms")
        print(f"[Controller] Mode: {self.mode}")

    def _safe_get(self, name):
        """安全获取设备，不存在时返回None"""
        dev = self.robot.getDevice(name)
        if dev is None or dev == 0:
            return None
        return dev

    def _init_sensors(self):
        """初始化所有传感器，缺失设备优雅降级"""
        # RGB相机
        self.rgb_camera = self._safe_get("rgb_camera")
        if self.rgb_camera:
            self.rgb_camera.enable(self.timestep)

        # 红外相机
        self.ir_camera = self._safe_get("ir_camera")
        if self.ir_camera:
            self.ir_camera.enable(self.timestep)

        # 深度传感器
        self.depth_sensor = self._safe_get("depth_sensor")
        if self.depth_sensor:
            self.depth_sensor.enable(self.timestep)

        # GPS
        self.gps = self._safe_get("gps")
        if self.gps:
            self.gps.enable(self.timestep)

        # IMU
        self.imu_accel = self._safe_get("imu_accel")
        self.imu_gyro = self._safe_get("imu_gyro")
        if self.imu_accel:
            self.imu_accel.enable(self.timestep)
        if self.imu_gyro:
            self.imu_gyro.enable(self.timestep)

        # 毫米波雷达 (IP2: 穿透烟/雾)
        self.radar = self._safe_get("radar")
        if self.radar:
            self.radar.enable(self.timestep)

        # 激光雷达 (IP2: 3D点云, 主动发光)
        self.lidar = self._safe_get("lidar")
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()

    # ========== 核心API ==========

    def step(self) -> dict:
        """执行一步仿真，返回传感器数据字典"""
        self.robot.step(self.timestep)
        self.step_count += 1
        return self.get_sensor_data()

    def get_sensor_data(self, degraded: bool = True) -> dict:
        """获取所有传感器当前读数

        Args:
            degraded: True=叠加退化效果(IP2/3用), False=原始图像(IP1地形分割用)
        """
        data = {
            "step": self.step_count,
            "timestamp": self.robot.getTime(),
            "battery": self.robot.batterySensorGetValue() if self.robot.batterySensorGetValue() > 0 else 1.0,
            "battery_pct": self.get_battery_pct(),
            "power_w": self._estimate_power(),
        }

        # RGB图像
        if self.rgb_camera:
            img = np.frombuffer(self.rgb_camera.getImage(), dtype=np.uint8)
            img = img.reshape((self.rgb_camera.getHeight(),
                               self.rgb_camera.getWidth(), 4))[:, :, :3]
            data["rgb"] = self.degradation.process_rgb(img) if degraded else img

        # 红外图像
        if self.ir_camera:
            ir_img = np.frombuffer(self.ir_camera.getImage(), dtype=np.uint8)
            ir_img = ir_img.reshape((self.ir_camera.getHeight(),
                                     self.ir_camera.getWidth(), 4))[:, :, :3]
            data["ir"] = self.degradation.process_ir(ir_img) if degraded else ir_img

        # 深度
        if self.depth_sensor:
            data["depth"] = self.depth_sensor.getValue()

        # GPS [x, y, z]
        if self.gps:
            gps_vals = self.gps.getValues()
            data["gps"] = [gps_vals[0], gps_vals[1], gps_vals[2]]

        # IMU
        if self.imu_accel:
            data["accel"] = list(self.imu_accel.getValues())
        if self.imu_gyro:
            data["gyro"] = list(self.imu_gyro.getValues())

        # IMU颠簸强度（用于打滑检测）
        if self.imu_accel:
            data["jolt"] = float(np.linalg.norm(self.imu_accel.getValues()) - 9.81)

        # Radar (IP2: 毫米波雷达, 穿透烟/雾)
        if self.radar:
            targets = self.radar.getTargets()
            data["radar_targets"] = [
                {"distance": t.distance, "angle": t.azimuth,
                 "speed": t.speed, "power": t.signalStrength}
                for t in targets
            ] if targets else []
            data["radar_n"] = len(data["radar_targets"]) if targets else 0

        # Lidar (IP2: 点云, 主动发光不受暗影响)
        if self.lidar:
            try:
                pc = self.lidar.getPointCloud()
                data["lidar_points"] = pc if pc else []
                data["lidar_n"] = len(pc) if pc else 0
            except Exception:
                data["lidar_points"] = []
                data["lidar_n"] = 0

        # 水流速度估计 (GPS漂移差分, 创新点三用)
        if self.gps and self.in_water:
            t = self.robot.getTime()
            dt = t - self._prev_time
            if dt > 0.5 and self._prev_gps is not None:
                dx = data["gps"][0] - self._prev_gps[0]
                dy = data["gps"][1] - self._prev_gps[1]
                self.water_flow_est = (dx / dt, dy / dt)
                data["water_flow"] = list(self.water_flow_est)
            self._prev_gps = (data["gps"][0], data["gps"][1])
            self._prev_time = t
        else:
            self._prev_gps = None
            data["water_flow"] = [0.0, 0.0]

        return data

    # ========== 水陆检测 ==========

    def is_in_water(self) -> bool:
        """根据GPS位置判断机器人是否在水域内"""
        x, y = self.get_position()
        return (self.WATER_ZONE_X[0] <= x <= self.WATER_ZONE_X[1] and
                self.WATER_ZONE_Y[0] <= y <= self.WATER_ZONE_Y[1])

    # ========== 运动控制 ==========

    def set_water_thrust(self, left_pct: float, right_pct: float):
        """水上电推推进 (等价T500): 0.0-1.0 对应 0-100%推力
        实际机器人用T500电推，仿真等效为轮速补偿。
        """
        self.water_thrust_L = max(0.0, min(1.0, left_pct))
        self.water_thrust_R = max(0.0, min(1.0, right_pct))

    def set_track_velocity(self, left_speed: float, right_speed: float):
        """全驱履带 + 水上电推等价"""
        MAX_VEL = 10.0  # 电机maxVelocity上限
        if self.in_water:
            thrust_L = self.water_thrust_L * 0.7 * left_speed
            thrust_R = self.water_thrust_R * 0.7 * right_speed
            wheel_L = left_speed * 0.3
            wheel_R = right_speed * 0.3
            left_speed = wheel_L + thrust_L
            right_speed = wheel_R + thrust_R
        left_speed = max(-MAX_VEL, min(MAX_VEL, left_speed))
        right_speed = max(-MAX_VEL, min(MAX_VEL, right_speed))
        for m in self.left_motors:
            m.setVelocity(left_speed)
        for m in self.right_motors:
            m.setVelocity(right_speed)

    def set_speed(self, linear_vel: float, angular_vel: float = 0.0):
        """设置线速度和角速度，自动转换为差速"""
        wheel_base = 0.5
        if self.in_water:
            angular_vel *= 0.3
        left = linear_vel - angular_vel * wheel_base / 2
        right = linear_vel + angular_vel * wheel_base / 2
        self.set_track_velocity(left, right)

    # ========== 臂摆双模式 + 三自由度姿态控制 ==========

    def arm_passive_mode(self):
        """被动柔顺模式：释放电机扭矩，弹簧+阻尼自然适应地形"""
        for name, m in self.arm_motors.items():
            m.setAvailableTorque(0.0)
        print("[Arm] Passive mode — spring/damping only")

    def set_arm_angle(self, name: str, angle: float):
        """设置单个臂角度 (rad)，自动切到主动模式"""
        if name in self.arm_motors:
            self.arm_motors[name].setAvailableTorque(5.0)
            self.arm_motors[name].setPosition(angle)

    def set_pitch(self, front_angle: float, rear_angle: float):
        """纵向俯仰补偿：前臂倾角, 后臂倾角 (正值=下压, 负值=抬升)"""
        self.set_arm_angle("arm_fl", front_angle)
        self.set_arm_angle("arm_fr", front_angle)
        self.set_arm_angle("arm_rl", rear_angle)
        self.set_arm_angle("arm_rr", rear_angle)

    def stop(self):
        """停止所有驱动"""
        for m in self.left_motors + self.right_motors:
            m.setVelocity(0.0)

    # ========== 模式控制 ==========

    def set_mode(self, mode: str):
        """切换控制模式: manual / ai_local / ai_cloud / beacon / path_follow"""
        valid_modes = ["manual", "ai_local", "ai_cloud", "beacon", "path_follow"]
        if mode in valid_modes:
            self.mode = mode
            print(f"[Controller] Mode switched to: {mode}")
        else:
            print(f"[Controller] Invalid mode: {mode}")

    # ========== 手动驾驶接口 ==========

    GEAR_SPEEDS = {1: 3.0, 2: 6.0, 3: 9.0}  # 三档线速度 (rad/s), 上限留余量给差速转向

    def manual_handle_key(self, key_char: str, pressed: bool = True):
        """处理手动驾驶按键，记录到按下集合

        WASD驾驶:
          W/S = 前进/后退
          A/D = 左转/右转
        R/F = 臂摆升/降
        Z/X/C = 慢/中/快 三档
        Space = 急停
        """
        if key_char in ('W', 'w'): self._manual_speed = self.GEAR_SPEEDS[self._manual_gear] if pressed else 0.0
        elif key_char in ('S', 's'): self._manual_speed = -self.GEAR_SPEEDS[self._manual_gear] if pressed else 0.0
        elif key_char in ('A', 'a'): self._manual_turn = -4.0 if pressed else 0.0
        elif key_char in ('D', 'd'): self._manual_turn = 4.0 if pressed else 0.0
        elif key_char == ' ' and pressed:  # Space 急停
            self._manual_speed = 0.0
            self._manual_turn = 0.0
            self.stop()
        elif pressed:
            if key_char in ('Z', 'z'): self._manual_gear = 1; print(f"[Manual] Gear 1 — SLOW ({self.GEAR_SPEEDS[1]} rad/s)")
            elif key_char in ('X', 'x'): self._manual_gear = 2; print(f"[Manual] Gear 2 — MEDIUM ({self.GEAR_SPEEDS[2]} rad/s)")
            elif key_char in ('C', 'c'): self._manual_gear = 3; print(f"[Manual] Gear 3 — FAST ({self.GEAR_SPEEDS[3]} rad/s)")
            elif key_char in ('R', 'r'): self._manual_arm_angle = min(0.35, self._manual_arm_angle + 0.1); self.set_pitch(self._manual_arm_angle, self._manual_arm_angle); print(f"[Manual] Arm angle: {self._manual_arm_angle:.2f} rad")
            elif key_char in ('F', 'f'): self._manual_arm_angle = max(-0.35, self._manual_arm_angle - 0.1); self.set_pitch(self._manual_arm_angle, self._manual_arm_angle); print(f"[Manual] Arm angle: {self._manual_arm_angle:.2f} rad")

    def manual_drive_step(self):
        """手动驾驶每步更新 — 由主循环调用，能源模式时不干扰"""
        if self.mode != "manual" or self._energy_mode:
            return
        self.set_speed(self._manual_speed, self._manual_turn)

    # ========== 路径跟随接口（纯接口，便于AI模块调用）==========

    def set_path_targets(self, path_points: list, speed: float = None):
        """设置路径跟随目标点序列

        Args:
            path_points: [(x1,y1), (x2,y2), ...] 世界坐标路径点
            speed: 跟随速度 (rad/s), None则用默认值
        """
        if not path_points:
            print("[PathFollow] Empty path — ignored")
            return
        self._path_targets = list(path_points)
        self._path_target_idx = 0
        if speed is not None:
            self._path_speed = speed
        self.mode = "path_follow"
        print(f"[PathFollow] Target set: {len(self._path_targets)} waypoints, "
              f"first=({path_points[0][0]:.1f},{path_points[0][1]:.1f}), "
              f"speed={self._path_speed:.1f}")

    def follow_path_step(self):
        """路径跟随每步更新 — 由主循环调用

        纯接口设计：只负责"朝当前目标点移动 + 到达检测"，
        路径点来源和后续行为由调用方决定（AI规划、手动设点等）。
        """
        if self.mode != "path_follow" or not self._path_targets:
            return

        pos = self.get_position()
        tx, ty = self._path_targets[self._path_target_idx]
        dx, dy = tx - pos[0], ty - pos[1]
        dist = np.hypot(dx, dy)

        waypoint_threshold = 1.5  # 到达阈值 (m)

        if dist < waypoint_threshold:
            # 到达当前目标点
            self._path_target_idx += 1
            if self._path_target_idx >= len(self._path_targets):
                print(f"[PathFollow] All {len(self._path_targets)} waypoints reached — stopping")
                self._path_targets = []
                self._path_target_idx = 0
                self.stop()
                self.mode = "manual"
                return
            else:
                tx, ty = self._path_targets[self._path_target_idx]
                dx, dy = tx - pos[0], ty - pos[1]
                dist = np.hypot(dx, dy)
                print(f"[PathFollow] Waypoint {self._path_target_idx}/{len(self._path_targets)} "
                      f"→ ({tx:.1f},{ty:.1f}), {dist:.1f}m ahead")

        # 朝目标点前进
        target_angle = np.arctan2(dy, dx)
        heading_error = target_angle  # 简化：假设机器人朝向+X，实际应读rotation
        angular = np.clip(heading_error * 1.5, -2.0, 2.0)
        linear = self._path_speed * max(0.3, 1.0 - abs(heading_error) / 2.0)

        self.set_speed(linear, angular)

    def has_path_targets(self) -> bool:
        """是否有待跟随的路径点"""
        return len(self._path_targets) > 0 and self._path_target_idx < len(self._path_targets)

    # ========== 创新点三：能源自适应策略 + 极限信标 ==========

    def _update_energy_history(self, data):
        """每步更新能耗历史缓冲（主循环中调用）"""
        # 地形类别
        terrain_class = 4 if self.in_water else 0  # fallback
        if self.risk_mapper and "rgb" in data:
            try:
                seg = self.risk_mapper.segment(data["rgb"])
                h, w = seg.shape
                bottom = seg[int(h * 0.85):, w // 3:2 * w // 3]
                terrain_class = int(np.bincount(bottom.flatten()).argmax())
            except Exception:
                pass

        speed = abs(self.left_motors[0].getVelocity()) if self.left_motors else 0
        flow = data.get("water_flow", [0.0, 0.0])

        self._terrain_hist.append(terrain_class)
        self._flow_hist.append(tuple(flow))
        self._load_hist.append(self._current_load_kg)
        self._speed_hist.append(speed)
        self._power_hist.append(data["power_w"])

    def energy_decision_step(self, data):
        """能源自适应决策 — 由E键或定时器触发"""
        if len(self._terrain_hist) < 10:
            print(f"[Energy] Buffer filling ({len(self._terrain_hist)}/30)")
            return None

        battery_pct = data["battery_pct"]
        pos = self.get_position()

        # 距岸距离
        dist_to_shore = self._calc_dist_to_shore(pos)

        # Module 1: LSTM预测
        energy_pred = self.energy_predictor.predict(
            list(self._terrain_hist), list(self._flow_hist),
            list(self._load_hist), list(self._speed_hist),
            list(self._power_hist))
        self.energy_predictor.check_returnable(battery_pct, energy_pred)

        # 地形风险
        terrain_risk = 0.0
        if self.risk_mapper and "rgb" in data:
            try:
                seg = self.risk_mapper.segment(data["rgb"])
                risk_grid = self.risk_mapper.base_risk_map(seg)
                terrain_risk = float(np.mean(risk_grid))
            except Exception:
                pass

        # 水流大小
        flow = data.get("water_flow", [0.0, 0.0])
        flow_tuple = (flow[0], flow[1]) if isinstance(flow, list) else (0.0, 0.0)

        # Module 2: MDP决策
        mdp_result = self.energy_mdp.decide(
            battery_pct, terrain_risk, dist_to_shore,
            flow_tuple, self._current_load_kg, energy_pred)

        # Module 3: LLM推理 (V4每10次决策调用一次)
        terrain_summary = self._summarize_terrain_hist()
        use_v4 = (self.energy_mdp._decision_count % 10 == 0)
        llm_result = self.energy_llm.decide(
            battery_pct, mdp_result, energy_pred,
            dist_to_shore, flow_tuple, terrain_summary,
            self._current_load_kg, force_v4=use_v4)

        result = {
            "mdp_action": mdp_result["action_name"],
            "llm_action": llm_result["final_action"],
            "energy_pred": energy_pred,
            "mdp_q": mdp_result["q_values"],
            "dist_to_shore": dist_to_shore,
            "terrain_risk": terrain_risk,
            "decision_source": llm_result["decision_source"],
        }

        print(f"\n[Energy] ===== Energy Decision =====")
        print(f"  Battery: {battery_pct:.1%} | Dist: {dist_to_shore:.0f}m | "
              f"Risk: {terrain_risk:.2f} | Load: {self._current_load_kg:.0f}kg")
        print(f"  LSTM: {energy_pred['energy_pred_j']:.0f}J in 60s "
              f"(need {energy_pred['min_battery_needed_pct']:.1f}% battery)")
        print(f"  MDP: {mdp_result['action_name']} (Q={[f'{q:.0f}' for q in mdp_result['q_values']]})")
        print(f"  LLM: {llm_result['final_action']} (source={llm_result['decision_source']})")
        if energy_pred.get("returnable") == False:
            print(f"  >>> WARNING: Battery insufficient for return! <<<")
        print(f"[Energy] ===============================")

        self._execute_energy_action(llm_result["final_action"])
        self._last_energy_result = result
        return result

    def _execute_energy_action(self, action_name, in_water=None):
        """执行能源决策动作"""
        if in_water is None:
            in_water = self.in_water

        if action_name == "FULL_SPEED":
            if in_water:
                self.set_water_thrust(1.0, 1.0)
                self.set_speed(6.0, 0.0)
            else:
                self.set_speed(10.0, 0.0)
        elif action_name == "CRUISE":
            if in_water:
                self.set_water_thrust(0.5, 0.5)
                self.set_speed(3.0, 0.0)
            else:
                self.set_speed(5.0, 0.0)
        elif action_name == "BEACON":
            self.enter_beacon_mode()
        elif action_name == "HOLD":
            self.stop()
        else:
            # 默认巡航
            self.set_speed(5.0, 0.0)

    def _calc_dist_to_shore(self, pos):
        """距最近岸边的距离 (m)"""
        x, y = pos
        if self.in_water:
            d_north = y - self.WATER_ZONE_Y[1]
            d_south = self.WATER_ZONE_Y[0] - y
            return min(abs(d_north), abs(d_south))
        else:
            if y > self.WATER_ZONE_Y[1]:
                return y - self.WATER_ZONE_Y[1]
            elif y < self.WATER_ZONE_Y[0]:
                return self.WATER_ZONE_Y[0] - y
            return 0.0

    def _summarize_terrain_hist(self):
        """地形历史摘要"""
        if not self._terrain_hist:
            return "未知"
        names = ["硬地", "泥地", "碎石", "浅水", "深水", "陡坡", "废墟"]
        counts = {}
        for c in self._terrain_hist:
            n = names[c % 7]
            counts[n] = counts.get(n, 0) + 1
        return ", ".join(f"{k}{v}步" for k, v in
                        sorted(counts.items(), key=lambda x: -x[1])[:4])

    # ========== 创新点二：退化环境多模态检测 ==========

    def _init_detector(self):
        """懒加载双流检测器 + V4研判器（首按T键时调用，避免启动卡顿）"""
        if self._detector_init_tried:
            return self.detector is not None
        self._detector_init_tried = True

        # V4研判器（轻量，仅测试API连通性）
        try:
            self.v4_detection = DetectionV4Analyzer()
        except Exception as e:
            print(f"[Controller] V4-Detect init failed: {e}")

        # 双流检测器（重载，加载两个YOLO模型 ~7秒）
        try:
            # 加载微调模型（优先），fallback到预训练
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "models", "multimodal_best.pt")
            if not os.path.exists(model_path):
                model_path = None
                print("[Controller] Fine-tuned model not found, using pretrained yolov8n")
            else:
                print(f"[Controller] Using fine-tuned model: multimodal_best.pt")
            print("[Controller] Loading DualStreamDetector...")
            self.detector = DualStreamDetector(rgb_model_path=model_path, ir_model_path=None)
            print("[Controller] DualStreamDetector ready — T key to detect")
            return True
        except Exception as e:
            print(f"[Controller] DualStreamDetector init failed: {e}")
            print("[Controller] Detection disabled — T key unavailable")
            return False

    def _detect_once(self, data):
        """执行一次双流目标检测，打印融合结果"""
        # 懒加载
        if not self._init_detector():
            return None

        rgb = data.get("rgb")
        ir = data.get("ir")
        if rgb is None:
            print("[Detect] No RGB image — skipping")
            return None
        if ir is None:
            print("[Detect] No IR image — skipping")
            return None
        if self.detector is None:
            print("[Detect] Detector not initialized")
            return None

        radar_targets = data.get("radar_targets", [])
        lidar_points = data.get("lidar_points", [])
        if radar_targets:
            print(f"[Radar] {len(radar_targets)} targets: "
                  + ", ".join(f"d={t['distance']:.1f}m a={t['angle']:.2f}rad" for t in radar_targets[:5]))
        result = self.detector.detect(rgb, ir, radar_targets, lidar_points)

        w = result.get("sensor_weights", {})
        deg = result.get("degradation", {})
        print(f"[Detect] ===== Multi-Modal Detection ({self.degradation.current}) =====")
        print(f"  Degradation: brightness={deg.get('d_brightness','?')} "
              f"contrast={deg.get('d_contrast','?')} blur={deg.get('d_blur','?')}")
        print(f"  Sensor Weights: RGB={w.get('w_rgb','?'):.3f} IR={w.get('w_ir','?'):.3f} "
              f"Radar={w.get('w_radar','?'):.3f} Lidar={w.get('w_lidar','?'):.3f}")
        print(f"  Detections: RGB:{result['num_rgb']} IR:{result['num_ir']} "
              f"Radar:{result['num_radar']} Lidar:{result['num_lidar']} → Fused:{result['num_fused']} "
              f"({result['time_ms']:.0f}ms)")
        for i, det in enumerate(result['detections']):
            b = det['bbox']
            src = det.get('src', '?')
            print(f"  Target{i+1}: conf={det['conf']:.3f} src={src} "
                  f"pos=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")
        if result['num_fused'] == 0:
            print(f"  No targets detected")
        print(f"[Detect] ===============================")

        # ── 三模控制级联：本地R1 → 云端V4 → 人工 ──
        # 仅检测到高置信目标(>0.5)或有共识目标时才触发LLM，避免阻塞
        has_quality = any(d['conf'] > 0.5 or '+' in d.get('src','') for d in result['detections'])
        if result['num_fused'] > 0 and has_quality:
            # Tier 1: R1 本地快速初筛
            r1_decision = None
            if self._r1_enabled:
                r1_decision = self._r1_detect_screen(result)
                if r1_decision:
                    # 去掉think标签
                    import re
                    clean = re.sub(r'</?think>', '', r1_decision).strip()
                    print(f"\n[R1-Detect] {clean[:300]}")

            # Tier 2: V4 云端深度研判 (带R1初判，去掉think标签)
            if self.v4_detection and self.v4_detection.enabled:
                import re
                r1_clean = re.sub(r'</?think>', '', r1_decision or '').strip()
                analysis = self.v4_detection.verify(
                    result['detections'], result['d'], self.degradation.current,
                    sensor_weights=result.get('sensor_weights'),
                    deg_vector=result.get('degradation'),
                    r1_opinion=r1_clean)
                print(f"\n[V4-Detect] {analysis}\n")
            # Tier 3: 人工 — 始终可用 (控制台输出+Webots视图)

        return result

    def _capture_screenshot(self, data):
        """保存当前RGB+IR帧到训练数据目录"""
        import cv2
        out_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "detection", "screenshots")
        os.makedirs(out_dir, exist_ok=True)

        # 找下一个可用编号
        idx = 0
        while os.path.exists(os.path.join(out_dir, f"rgb_{idx:04d}.png")):
            idx += 1

        rgb = data.get("rgb")
        ir = data.get("ir")
        if rgb is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            path = os.path.join(out_dir, f"rgb_{idx:04d}.png")
            _, buf = cv2.imencode('.png', bgr)
            if buf is not None:
                with open(path, 'wb') as f: f.write(buf.tobytes())
        if ir is not None:
            ir_bgr = cv2.cvtColor(ir, cv2.COLOR_RGB2BGR) if len(ir.shape) == 3 else ir
            path = os.path.join(out_dir, f"ir_{idx:04d}.png")
            _, buf = cv2.imencode('.png', ir_bgr)
            if buf is not None:
                with open(path, 'wb') as f: f.write(buf.tobytes())

        label = f"{self.degradation.current} pos=({self.get_position()[0]:.1f},{self.get_position()[1]:.1f})"
        print(f"[Capture] #{idx:04d} saved ({label}) → data/detection/screenshots/")

    def _r1_detect_screen(self, result: dict) -> str:
        """R1本地快速初筛：检测结果 → 即时判断"""
        import requests
        n = result['num_fused']
        w = result.get('sensor_weights', {})
        deg = result.get('degradation', {})
        det_list = ", ".join(
            f"conf={d['conf']:.2f}/{d.get('src','?')}" for d in result['detections'][:5])

        prompt = f"""救灾检测初筛: 环境{self.degradation.current} 暗{deg.get('d_brightness',0):.1f} 雾{deg.get('d_contrast',0):.1f} 烟{deg.get('d_blur',0):.1f}
权重(RGB/IR/Radar/Lidar): {w.get('w_rgb',0):.2f}/{w.get('w_ir',0):.2f}/{w.get('w_radar',0):.2f}/{w.get('w_lidar',0):.2f}
{n}个目标: {det_list}

直接回答(跳过推理,15字内):
- 有幸存者吗? (有/疑似/无)
- 信任哪个传感器? (RGB/IR/Radar/Lidar)
- 要V4吗? (要/不要)"""

        try:
            resp = requests.post(self._r1_url, json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 64},
            }, timeout=8)
            return resp.json()["response"].strip()
        except Exception as e:
            return f"[R1 Error] {e}"

    # ========== 悬浮气囊接口（AI入水判断后调用）==========

    def enter_beacon_mode(self):
        """进入信标模式：关闭所有非必要设备"""
        self.mode = "beacon"
        self.stop()
        if self.beacon:
            self.beacon.set(1)
        print("[Controller] === BEACON MODE ACTIVATED ===")
        print("[Controller] WASD to exit beacon, restore manual control")

    def exit_beacon_mode(self):
        """退出信标模式，恢复手动控制"""
        self.mode = "manual"
        self._manual_speed = 0.0
        self._manual_turn = 0.0
        self.stop()
        if self.beacon:
            self.beacon.set(0)
        print("[Controller] === BEACON MODE DEACTIVATED — manual control restored ===")

    # ========== 地形检测辅助 ==========

    def _check_r1(self):
        """检测 Ollama DeepSeek-R1 是否可用，并预热模型"""
        try:
            import requests
            resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                # 预热：发一条空请求让模型加载到内存
                requests.post("http://127.0.0.1:11434/api/generate", json={
                    "model": "deepseek-r1:7b", "prompt": "OK",
                    "stream": False, "options": {"max_tokens": 2}
                }, timeout=30)
                print("[DeepSeek] R1 7B (Ollama) warmed up — local inference ready")
                return True
        except Exception as e:
            print(f"[DeepSeek] R1 warmup: {e}")
        print("[DeepSeek] R1 7B not available — local inference disabled")
        return False

    def get_position(self) -> tuple:
        """获取当前位置 (x, y)"""
        if self.gps:
            gps = self.gps.getValues()
            return (gps[0], gps[1])
        return (0.0, 0.0)

    def _estimate_power(self) -> float:
        """功耗估计 (W) — 基于BOM数据建模
        陆地巡航: ~500W (电机+传感器+Jetson)
        水上电推: +1200-2400W (T500×2, 按推力百分比)
        信标模式: ~5W
        """
        if self.mode == "beacon":
            return 5.0
        base = 80.0  # Jetson + 传感器 + 待机
        # 驱动功耗 (线性项 + 速度平方阻力项, 16个电机)
        s = (abs(self.left_motors[0].getVelocity()) if self.left_motors else 0)
        drive_w = s * 25.0 + s * s * 4.0
        base += drive_w
        # 水上电推 (T500: 1200W×2 满推)
        if self.in_water:
            thrust_avg = (self.water_thrust_L + self.water_thrust_R) / 2
            base += thrust_avg * 2400.0  # T500 ×2
        return round(base, 1)

    def get_battery_pct(self) -> float:
        """获取电量百分比 0-1，batterySensorGetValue()返回焦耳"""
        val = self.robot.batterySensorGetValue()
        return val / 360000.0 if val > 0 else 0.0

    # ========== 数据导出 ==========

    # ========== 创新点一：AI路径规划 ==========

    def ai_plan_path(self):
        """捕获当前相机画面 → 语义分割 → 风险地图 → 路径规划（三种方案对比）"""
        if not self._has_ai_planner:
            print("[AI] RiskMapper not available — skipping")
            return None

        data = self.get_sensor_data(degraded=False)
        rgb = data.get("rgb")
        if rgb is None:
            print("[AI] No RGB image — skipping")
            return None

        jolt = data.get("jolt", 0.0)
        depth = data.get("depth", 10.0)

        print(f"[AI] Planning path (jolt={jolt:.3f}, depth={depth:.2f}m)...")

        # 风险地图
        risk_grid = self.risk_mapper.compute(rgb, jolt=jolt, depth=depth)
        self._last_risk_grid = risk_grid
        summary = self.risk_mapper.risk_summary(risk_grid)
        print(f"[AI] Risk map: mean={summary['mean_risk']:.2f}, "
              f"high_risk={summary['high_risk_pct']:.1f}%, "
              f"impassable={summary['impassable_pct']:.1f}%")

        # 起点（图像底部中央 = 机器人前方近处）
        h, w = risk_grid.shape
        start = (w // 2, h - 20)
        goal = (w // 2, 20)  # 图像顶部中央 = 远处目标

        # 三种方案对比
        results = {}
        for name, method, desc in [
            ("baseline_a", BaselinePlanner.baseline_a, "Binary A*"),
            ("baseline_b", BaselinePlanner.baseline_b, "Fixed Risk A*"),
            ("ours", BaselinePlanner.ours, "Semantic+IMU A*"),
        ]:
            path, stats = method(risk_grid, start, goal)
            success = path is not None
            results[name] = {"success": success, "stats": stats or {}, "path": path}
            status = f"OK ({len(path)} steps, risk={stats.get('avg_risk',0):.3f})" if success else "FAILED"
            print(f"  {desc:20s}: {status}")

        self._last_path = results["ours"]["path"]
        self._benchmark_results.append(results)

        # 导出最新结果
        self._export_benchmark()

        return results

    def _export_benchmark(self):
        """导出对比实验结果到JSON"""
        if not self._benchmark_results:
            return

        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
        os.makedirs(output_dir, exist_ok=True)

        # 完整结果
        path = os.path.join(output_dir, "benchmark_results.json")
        with open(path, "w") as f:
            json.dump(self._benchmark_results, f, indent=2, default=str)

        # 汇总统计
        summary = {"Baseline A": [], "Baseline B": [], "Ours": []}
        key_map = {"baseline_a": "Baseline A", "baseline_b": "Baseline B", "ours": "Ours"}
        for result in self._benchmark_results:
            for k, v in result.items():
                if k in key_map:
                    summary[key_map[k]].append(v["success"])

        summary_path = os.path.join(output_dir, "benchmark_summary.json")
        report = {}
        for method, successes in summary.items():
            report[method] = {
                "success_rate": round(sum(successes) / len(successes), 3) if successes else 0,
                "n_trials": len(successes),
                "n_success": sum(successes),
            }
        with open(summary_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[AI] Results exported to {output_dir}/")

    def terrain_benchmark(self):
        """综合评估：5位置 × (3方案A* + V4分析)"""
        positions = [
            (-12, 18, 0.35, 3.14, "hard_north"),
            (12, 14, 0.35, 3.14, "mud_east"),
            (-10, -9, 0.35, 3.14, "gravel_rubble"),
            (0, 0, 0.20, 3.14, "water_river"),
            (14, -14, 0.35, 3.14, "hard_south"),
        ]
        print("\n" + "=" * 60)
        print("  Full Benchmark: 5 x (3 A* + V4)")
        print("=" * 60)
        self._benchmark_results = []

        for px, py, pz, pyaw, pname in positions:
            print(f"\n--- [{pname}] ---")
            self.robot_node.getField("translation").setSFVec3f([px, py, pz])
            self.robot_node.getField("rotation").setSFRotation([0, 0, 1, pyaw])
            for _ in range(32):
                self.robot.step(self.timestep)
            # A* 三方案
            results = self.ai_plan_path()
            if results:
                results["position"] = pname
            # V4 分析
            self.v4_analyze_path()

        self._export_benchmark()
        self._print_final_summary()

    def _print_final_summary(self):
        if not self._benchmark_results:
            return
        key_map = {"baseline_a": "Baseline A", "baseline_b": "Baseline B", "ours": "Ours"}
        stats = {v: {"success": 0, "total": 0, "risks": [], "lengths": []} for v in key_map.values()}
        for result in self._benchmark_results:
            for k, v in result.items():
                if k in key_map:
                    m = key_map[k]
                    stats[m]["total"] += 1
                    if v["success"]:
                        stats[m]["success"] += 1
                        s = v["stats"]
                        if "avg_risk" in s and "path_length" in s:
                            stats[m]["risks"].append(s["avg_risk"])
                            stats[m]["lengths"].append(s["path_length"])
        print("\n" + "=" * 60)
        print("  BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"  {'Method':20s} {'Success':>8s} {'Avg Risk':>10s} {'Avg Length':>12s}")
        print("  " + "-" * 52)
        for method in ["Baseline A", "Baseline B", "Ours"]:
            s = stats[method]
            rate = f"{s['success']}/{s['total']}"
            avg_risk = f"{np.mean(s['risks']):.3f}" if s['risks'] else "N/A"
            avg_len = f"{np.mean(s['lengths']):.0f}" if s['lengths'] else "N/A"
            print(f"  {method:20s} {rate:>8s} {avg_risk:>10s} {avg_len:>12s}")
        print("=" * 60)

    def v4_analyze_path(self):
        """DeepSeek V4 多模态路径分析"""
        if not self._has_ai_planner:
            print("[V4] RiskMapper not available")
            return

        data = self.get_sensor_data(degraded=False)
        rgb = data.get("rgb")
        if rgb is None:
            return

        seg = self.risk_mapper.segment(rgb)
        risk = self.risk_mapper.base_risk_map(seg)
        summary = self.risk_mapper.risk_summary(risk)
        pos = self.get_position()

        print("\n" + "=" * 60)
        print("  DeepSeek V4 — Terrain Path Analysis")
        print("=" * 60)

        result = self.v4_planner.analyze(seg, summary, pos)
        print(f"\n{result['analysis']}\n")

        # 保存
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
        os.makedirs(output_dir, exist_ok=True)
        self.v4_planner.save_history(os.path.join(output_dir, "v4_analysis.json"))
        print(f"[V4] Analysis saved to results/v4_analysis.json")
        print("=" * 60)

    def r1_quick_decision(self):
        """DeepSeek-R1 7B 本地实时决策 (<1秒)"""
        if not self._r1_enabled:
            print("[R1] Ollama not available")
            return None

        data = self.get_sensor_data()
        seg = self.risk_mapper.segment(data["rgb"])
        summary = self.risk_mapper.risk_summary(
            self.risk_mapper.base_risk_map(seg))
        pos = self.get_position()
        jolt = data.get("jolt", 0.0)

        prompt = f"""你是救灾机器人，需要根据传感器数据做实时路径决策。

参数说明：
- GPS: 当前坐标，用于位置参考
- jolt: IMU颠簸值，单位m/s²。>0.5表示路面粗糙/打滑，>1.0表示严重颠簸
- risk_avg: 前方地形平均风险值(0-1)。0.1=硬质路面(安全), 0.5=泥地(中危), 0.7=深水(高危), 1.0=废墟(不可通行)
- risk_high: 前方高风险区占比(%)

当前数据: GPS({pos[0]:.1f},{pos[1]:.1f}) jolt={jolt:.3f} risk_avg={summary['mean_risk']:.2f} risk_high={summary['high_risk_pct']:.0f}%

决策选项:
- 直行: 前方安全，保持方向
- 左绕: 前方危险，向左绕行
- 右绕: 前方危险，向右绕行
- 停车: 前方不可通行，停止等待

请先分析各参数含义，再给出最终决策。最后一行只输出决策词。"""

        import requests
        try:
            resp = requests.post(self._r1_url, json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 512}
            }, timeout=60)
            decision = resp.json()["response"].strip()
        except Exception as e:
            decision = f"[R1 Error] {e}"

        print(f"\n[R1 Local Decision]\n{decision}\n")
        return decision

    def llm_decision(self):
        """LLM智能决策：根据可用性自动选择决策路径

        优先级: 人工 > R1+V4级联 > R1单独 > V4单独 > 人工接管
        """
        print("\n" + "=" * 60)
        print("  LLM Decision Engine")
        print("=" * 60)

        # 降级模式覆盖
        if self._degrade_level == 0:
            r1_ok = self._r1_enabled
            v4_ok = self._v4_enabled and self._has_ai_planner
        elif self._degrade_level == 1:
            r1_ok, v4_ok = self._r1_enabled, False
        elif self._degrade_level == 2:
            r1_ok, v4_ok = False, (self._v4_enabled and self._has_ai_planner)
        else:
            r1_ok = v4_ok = False

        mode_names = {0: 'NORMAL', 1: 'V4-OFF', 2: 'R1-OFF', 3: 'ALL-OFF'}
        print(f"  R1 Local : {'ONLINE' if r1_ok else 'OFFLINE'}")
        print(f"  V4 Cloud : {'ONLINE' if v4_ok else 'OFFLINE'}")
        print(f"  Mode     : {mode_names[self._degrade_level]} (K to toggle)")

        if r1_ok and v4_ok:
            print("  Path: R1 -> V4 CASCADE\n")
            print("--- R1 7B Local (<1s) ---")
            r1 = self.r1_quick_decision()
            print("\n--- V4 Cloud (~3s, with R1 input) ---")
            data = self.get_sensor_data()
            if data.get("rgb") is not None:
                seg = self.risk_mapper.segment(data["rgb"])
                summary = self.risk_mapper.risk_summary(self.risk_mapper.base_risk_map(seg))
                result = self.v4_planner.analyze_with_r1(seg, summary, self.get_position(), r1)
                print(f"\n{result['analysis']}\n")
                output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
                os.makedirs(output_dir, exist_ok=True)
                self.v4_planner.save_history(os.path.join(output_dir, "v4_analysis.json"))

        elif r1_ok:
            print("  Path: R1 ONLY (V4 degraded)\n")
            self.r1_quick_decision()

        elif v4_ok:
            print("  Path: V4 ONLY (R1 degraded)\n")
            self.v4_analyze_path()

        else:
            print("  Path: MANUAL CONTROL")
            print("  >>> All AI offline — human operator take over <<<")

        print("=" * 60)


# ========== 主循环 ==========

def main():
    controller = RescueRobotController()

    print("\n" + "=" * 60)
    print("  MANUAL DRIVE CONTROLS")
    print("  WASD = 前进/后退/左转/右转")
    print("  Z/X/C = 慢/中/快 三档")
    print("  Space = 急停")
    print("  R/F = 臂摆 升/降")
    print("  IP1: P=路径规划 V=综合评估")
    print("  IP2: T=多模态检测 Y=截图 B=基准测试 1-4=退化级别")
    print("  IP3: E=能源自适应 M=MDP决策")
    print("  L=LLM决策 K=降级切换")
    print("=" * 60 + "\n")

    while controller.robot.step(controller.timestep) != -1:
        data = controller.get_sensor_data()
        controller.step_count += 1

        # ── 键盘控制 ──
        key = controller.keyboard.getKey()
        if key >= 0:
            key_char = chr(key) if 32 <= key <= 126 else ""

            # 手动驾驶按键 (WASD, 1-3, Space, R/F)
            # 信标模式下按WASD自动退出
            if controller.mode == "beacon" and key_char in 'WASDwasd':
                controller.exit_beacon_mode()
            controller.manual_handle_key(key_char, pressed=True)

            # 退化模式 (1-4，但如果已按档位则不触发)
            controller.degradation.set_level(key_char)

            # 路径规划
            if key_char == 'P' or key_char == 'p':
                controller.ai_plan_path()
                # 将规划的路径装入跟随器
                if controller._last_path and len(controller._last_path) > 1:
                    h, w = controller._last_risk_grid.shape if controller._last_risk_grid is not None else (100, 100)
                    pos = controller.get_position()
                    pts = []
                    for py, px in controller._last_path:
                        wx = pos[0] + (px - w/2) * 0.5
                        wy = pos[1] + (w/2 - py) * 0.5
                        pts.append((wx, wy))
                    controller.set_path_targets(pts)

            # LLM 智能决策（自动选择 R1+V4 / R1 / V4 / 人工）
            elif key_char == 'L' or key_char == 'l':
                controller.llm_decision()

            # 综合评估
            elif key_char == 'V' or key_char == 'v':
                controller.terrain_benchmark()

            # 降级模式切换
            elif key_char == 'K' or key_char == 'k':
                controller._degrade_level = (controller._degrade_level + 1) % 4
                names = {0: 'NORMAL', 1: 'V4-OFF (R1 only)', 2: 'R1-OFF (V4 only)', 3: 'ALL-OFF (manual)'}
                print(f"\n[Degrade] Switched to: {names[controller._degrade_level]}\n")

            # 能源自适应决策 (E = 切换模式, M = 手动触发一次)
            elif key_char == 'E' or key_char == 'e':
                controller._energy_mode = not controller._energy_mode
                if controller._energy_mode:
                    controller._energy_interval = 5.0
                    print("\n[Energy] === ENERGY ADAPTIVE MODE ON ===")
                    print("[Energy] Auto-evaluating every 5s. Press E to disable.")
                else:
                    controller.set_speed(0.0, 0.0)
                    print("[Energy] Energy adaptive mode OFF — manual control restored.")
            elif key_char == 'M' or key_char == 'm':
                print("\n[Energy] Manual MDP decision trigger...")
                controller._update_energy_history(data)
                controller.energy_decision_step(data)

            # 目标检测（双流融合）
            elif key_char == 'T' or key_char == 't':
                controller._detect_once(data)

            # 截图采集 (训练数据)
            elif key_char == 'Y' or key_char == 'y':
                controller._capture_screenshot(data)

            # 自动基准测试 (三创新点对比)
            elif key_char == 'B' or key_char == 'b':
                run_benchmark(controller)
        else:
            # 无按键时清除手动驾驶状态（松开即停），能源模式除外
            if controller.mode == "manual" and not controller._energy_mode:
                controller._manual_speed = 0.0
                controller._manual_turn = 0.0

        # ── 模式驱动 ──
        if controller.mode == "manual":
            controller.manual_drive_step()
        elif controller.mode == "path_follow":
            controller.follow_path_step()

        # ── 创新点三：能源历史更新 + 自适应决策 ──
        controller._update_energy_history(data)
        if controller._energy_mode:
            t = data["timestamp"]
            if t - controller._last_energy_time >= controller._energy_interval:
                controller.energy_decision_step(data)
                controller._last_energy_time = t

        # 检测水陆切换，重新应用速度以适配当前地形
        water_now = controller.is_in_water()
        if water_now and not controller.in_water:
            controller.in_water = True
            controller.set_water_thrust(1.0, 1.0)  # 入水自动满推力
            print(f"[t={data['timestamp']:.1f}s] === ENTERED WATER === "
                  f"pos=({data['gps'][0]:.1f},{data['gps'][1]:.1f}) — 电推激活")
        elif not water_now and controller.in_water:
            controller.in_water = False
            controller.set_water_thrust(0.0, 0.0)  # 出水关闭电推
            print(f"[t={data['timestamp']:.1f}s] === EXITED WATER === "
                  f"pos=({data['gps'][0]:.1f},{data['gps'][1]:.1f}) — 电推关闭")

        # 每50步打印一次状态
        if controller.step_count % 50 == 0:
            pos = controller.get_position()
            batt = controller.get_battery_pct()
            terrain = "WATER" if controller.in_water else "LAND"
            thrust_info = f"T=({controller.water_thrust_L:.0f},{controller.water_thrust_R:.0f})" if controller.in_water else ""
            gear_info = f"G{controller._manual_gear}" if controller.mode == "manual" else ""
            pf_info = f"→{controller._path_target_idx}/{len(controller._path_targets)}" if controller.mode == "path_follow" else ""
            print(f"[t={data['timestamp']:.1f}s] "
                  f"pos=({pos[0]:.1f},{pos[1]:.1f}) "
                  f"terrain={terrain} {thrust_info} "
                  f"mode={controller.mode}{gear_info}{pf_info} "
                  f"vis={controller.degradation.current} "
                  f"P={data['power_w']:.0f}W bat={batt:.1%}")

        # 电量保护：自适应模式由LLM/MDP决策, 8%为绝对安全底线
        batt = controller.get_battery_pct()
        if controller._energy_mode:
            if (controller._last_energy_result and
                controller._last_energy_result.get("mdp_action") == "BEACON" and
                controller.mode != "beacon"):
                controller.enter_beacon_mode()
            elif batt < 0.08:
                controller.enter_beacon_mode()
                print("[Energy] SAFETY FLOOR: battery < 8% — forced beacon")
                break
        else:
            if batt < 0.1:
                controller.enter_beacon_mode()
                break

    print("[Controller] Simulation ended")


if __name__ == "__main__":
    main()
