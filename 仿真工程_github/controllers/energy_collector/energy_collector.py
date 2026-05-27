"""energy_collector.py — 独立能耗数据采集控制器
=====================================================
全任务续航测试：按地形区瞬移+直线行驶，覆盖全部7种地形。
每个地形区瞬移到安全起点→直线行驶→记录功耗数据→瞬移到下一区。
一条S键完成全部采集，输出单个连续记录文件。

按键：
  S    — 启动全任务续航采集（~150秒，7地形区）
  R    — 手动记录60秒
  Q    — 保存退出

输出: data/energy/real_scenario_{N:03d}.json
"""

from controller import Supervisor, Keyboard
import numpy as np
import json
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ===================== 全任务续航路线 =====================
# 每段: (名称, 起点(x,y,z,朝向角), 速度, 时长s, 地形标签)
# 瞬移到安全起点→直线行驶→记录→下一段

FULL_MISSION = [
    # 北岸硬地 — 面朝+X（东），沿走廊直线行驶
    ("hard_north",     (-12, 20, 0.35, 0.0),     6.0, 20, 0),
    # 东侧泥地 — 面朝-Y（南），滑泥地直线
    ("mud_east",       (12, 18, 0.35, -math.pi/2), 5.0, 20, 1),
    # 渡河 — 面朝西偏南，直线横渡
    ("water_cross",    (8, 4, 0.20, math.pi*0.7),  4.0, 20, 4),
    # 西侧碎石 — 面朝南偏西
    ("gravel_west",    (-8, -4, 0.35, -math.pi/2), 5.0, 20, 2),
    # 废墟 — 面朝南，慢速
    ("rubble_zone",    (-10, -14, 0.35, -math.pi/2), 3.0, 20, 6),
    # 南岸硬地返航 — 面朝东偏北
    ("hard_south",     (-6, -18, 0.35, 0.0),     7.0, 15, 0),
    # 回北岸 — 面朝北
    ("return_north",   (14, -10, 0.35, math.pi/2), 6.0, 15, 0),
]

GEAR_SPEEDS = {1: 3.0, 2: 6.0, 3: 9.0}
WATER_ZONE_X = (-19.0, 19.0)
WATER_ZONE_Y = (-6.0, 6.0)


class StandaloneCollector:
    """独立能耗采集器 — 全任务续航测试"""

    MAP_X = (-20.0, 20.0)
    MAP_Y = (-20.0, 20.0)

    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.step_count = 0
        self._robot_node = self.robot.getSelf()

        # 驱动电机
        self.left_motors = []
        self.right_motors = []
        for i in range(self.robot.getNumberOfDevices()):
            dev = self.robot.getDeviceByIndex(i)
            name = dev.getName()
            if any(name.startswith(p) for p in ["wheel_f", "wheel_r", "drive_L", "drive_R"]):
                dev.setPosition(float('inf'))
                dev.setVelocity(0.0)
                if name.endswith(("l", "L")) or "drive_L" in name:
                    self.left_motors.append(dev)
                else:
                    self.right_motors.append(dev)
        print(f"[Collector] Motors: L={len(self.left_motors)} R={len(self.right_motors)}")

        # 传感器
        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.timestep)
        self.rgb_camera = self.robot.getDevice("rgb_camera")
        if self.rgb_camera:
            self.rgb_camera.enable(self.timestep)
        self.robot.batterySensorEnable(self.timestep)

        # 键盘
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)

        # 状态
        self._manual_speed = 0.0
        self._manual_turn = 0.0
        self._manual_gear = 2
        self._water_thrust_L = 0.0
        self._water_thrust_R = 0.0

        # 采集
        self._collecting = False
        self._buffer = []
        self._output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "energy")
        os.makedirs(self._output_dir, exist_ok=True)
        self._session_id = self._find_next_id()

        print("\n" + "=" * 55)
        print("  ENERGY DATA COLLECTOR — Full Mission")
        print(f"  {len(FULL_MISSION)} terrain segments, ~150s total")
        print("  S = Start full mission")
        print("  R = Manual record 60s")
        print("  Q = Save & quit")
        print("=" * 55 + "\n")

    def _find_next_id(self):
        sid = 0
        while os.path.exists(os.path.join(self._output_dir, f"real_scenario_{sid:03d}.json")):
            sid += 1
        return sid

    # ===== 传感器 =====
    def get_position(self):
        if self.gps:
            v = self.gps.getValues()
            return (v[0], v[1])
        return (0.0, 0.0)

    def in_water(self):
        x, y = self.get_position()
        return (WATER_ZONE_X[0] <= x <= WATER_ZONE_X[1] and
                WATER_ZONE_Y[0] <= y <= WATER_ZONE_Y[1])

    def get_battery_pct(self):
        val = self.robot.batterySensorGetValue()
        return val / 360000.0 if val > 0 else 1.0

    def _estimate_power(self):
        s = abs(self.left_motors[0].getVelocity()) if self.left_motors else 0
        base = 80.0 + s * 25.0 + s * s * 4.0
        if self.in_water():
            base += (self._water_thrust_L + self._water_thrust_R) / 2 * 2400.0
        return round(base, 1)

    def get_heading(self):
        rot = self._robot_node.getField("rotation").getSFRotation()
        return rot[3] if abs(rot[2]) > 0.9 else 0.0

    # ===== 运动控制 =====
    def set_speed(self, linear, angular=0.0):
        wb = 0.5
        if self.in_water():
            angular *= 0.3
        left = linear - angular * wb / 2
        right = linear + angular * wb / 2
        if self.in_water():
            left = left * 0.3 + self._water_thrust_L * 0.7 * left * 1.5
            right = right * 0.3 + self._water_thrust_R * 0.7 * right * 1.5
        MAX_VEL = 10.0
        for m in self.left_motors:
            m.setVelocity(max(-MAX_VEL, min(MAX_VEL, left)))
        for m in self.right_motors:
            m.setVelocity(max(-MAX_VEL, min(MAX_VEL, right)))

    def drive_straight(self, speed):
        """直线行驶 + 轻微航向修正（维持初始朝向）"""
        self.set_speed(speed, 0.0)

    def stop(self):
        for m in self.left_motors + self.right_motors:
            m.setVelocity(0.0)
        self._water_thrust_L = 0.0
        self._water_thrust_R = 0.0

    # ===== 数据记录 =====
    def _record_step(self, terrain_label):
        pos = self.get_position()
        s = abs(self.left_motors[0].getVelocity()) if self.left_motors else 0
        in_w = self.in_water()
        thrust = (self._water_thrust_L + self._water_thrust_R) / 2 if in_w else 0.0
        self._buffer.append({
            "step": len(self._buffer),
            "t": round(self.robot.getTime(), 1),
            "terrain": terrain_label,
            "speed": round(s, 2),
            "load_kg": 0.0,
            "flow_vx": 0.0,
            "flow_vy": 0.0,
            "power_w": self._estimate_power(),
            "battery_pct": round(self.get_battery_pct(), 4),
            "position": [round(pos[0], 2), round(pos[1], 2)],
            "in_water": in_w,
            "thrust_ratio": round(thrust, 2),
        })

    def _save_session(self):
        if not self._buffer:
            return
        path = os.path.join(self._output_dir, f"real_scenario_{self._session_id:03d}.json")
        powers = [s["power_w"] for s in self._buffer]
        terrains = [s["terrain"] for s in self._buffer]
        tnames = ["hard", "mud", "gravel", "shallow", "deep", "slope", "rubble"]
        record = {
            "scenario_id": self._session_id,
            "route": "full_mission",
            "n_steps": len(self._buffer),
            "duration_s": round(self._buffer[-1]["t"] - self._buffer[0]["t"], 1) if self._buffer else 0,
            "total_energy_j": round(sum(s["power_w"] * self.timestep / 1000 for s in self._buffer), 0),
            "terrain_coverage": {tnames[t]: terrains.count(t) for t in sorted(set(terrains))},
            "power_stats": {
                "mean": round(np.mean(powers), 0),
                "max": round(np.max(powers), 0),
                "min": round(np.min(powers), 0),
            },
            "steps": self._buffer,
        }
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\n[Collector] Saved → {os.path.basename(path)}")
        print(f"  Duration: {record['duration_s']}s | Energy: {record['total_energy_j']}J")
        print(f"  Power: mean={record['power_stats']['mean']:.0f}W "
              f"max={record['power_stats']['max']:.0f}W min={record['power_stats']['min']:.0f}W")
        print(f"  Terrain: {record['terrain_coverage']}")
        self._session_id += 1
        self._buffer = []

    # ===== 全任务执行 =====
    def run_segment(self, name, start, speed, duration_s, terrain_label):
        """执行一段采集：瞬移到起点→面朝指定方向→直线行驶→记录"""
        sx, sy, sz, heading = start
        target_steps = int(duration_s * 1000 / self.timestep)

        print(f"  [{name}] ({sx:.0f},{sy:.0f}) heading={heading:.1f}rad "
              f"speed={speed:.1f} {duration_s}s terrain={terrain_label}")

        # 瞬移 + 设朝向
        self._robot_node.getField("translation").setSFVec3f([sx, sy, sz])
        self._robot_node.getField("rotation").setSFRotation([0, 0, 1, heading])
        for _ in range(32):
            self.robot.step(self.timestep)

        # 水域激活电推
        if terrain_label >= 3:
            self._water_thrust_L = 1.0
            self._water_thrust_R = 1.0
        else:
            self._water_thrust_L = 0.0
            self._water_thrust_R = 0.0

        # 直线行驶
        self.drive_straight(speed)
        seg_start = self.step_count
        while self._collecting and (self.step_count - seg_start) < target_steps:
            self.robot.step(self.timestep)
            self.step_count += 1
            self._record_step(terrain_label)

        self.stop()
        self._water_thrust_L = 0.0
        self._water_thrust_R = 0.0

    def run_full_mission(self):
        """执行完整续航任务"""
        print("\n[Collector] ===== FULL MISSION START =====")
        print(f"[Collector] {len(FULL_MISSION)} segments, ~150s estimated")
        self._buffer = []
        self._collecting = True
        t_start = self.robot.getTime()

        for name, start, speed, duration, terrain_label in FULL_MISSION:
            if not self._collecting:
                break
            self.run_segment(name, start, speed, duration, terrain_label)
            # 段间冷却1秒
            self.stop()
            for _ in range(int(1.0 * 1000 / self.timestep)):
                self.robot.step(self.timestep)
                self.step_count += 1

        self._collecting = False
        elapsed = self.robot.getTime() - t_start
        print(f"\n[Collector] ===== MISSION COMPLETE ({elapsed:.1f}s) =====")
        self._save_session()

    # ===== 手动驾驶 =====
    def handle_key(self, key_char, pressed=True):
        if key_char in ('W', 'w'):
            self._manual_speed = GEAR_SPEEDS[self._manual_gear] if pressed else 0.0
        elif key_char in ('S', 's'):
            self._manual_speed = -GEAR_SPEEDS[self._manual_gear] if pressed else 0.0
        elif key_char in ('A', 'a'):
            self._manual_turn = -4.0 if pressed else 0.0
        elif key_char in ('D', 'd'):
            self._manual_turn = 4.0 if pressed else 0.0
        elif key_char == ' ' and pressed:
            self._manual_speed = 0.0; self._manual_turn = 0.0; self.stop()
        elif pressed:
            if key_char in ('1',): self._manual_gear = 1
            elif key_char in ('2',): self._manual_gear = 2
            elif key_char in ('3',): self._manual_gear = 3


# ===================== 主循环 =====================

def main():
    collector = StandaloneCollector()
    collector.stop()
    for _ in range(32):
        collector.robot.step(collector.timestep)

    print("[Collector] Ready. Press S for full mission, R for manual, Q to quit.")

    while collector.robot.step(collector.timestep) != -1:
        collector.step_count += 1
        key = collector.keyboard.getKey()

        if key >= 0:
            key_char = chr(key) if 32 <= key <= 126 else ""

            if key_char == 'S' or key_char == 's':
                collector.run_full_mission()

            elif key_char == 'R' or key_char == 'r':
                print("\n[Collector] Recording 60s manual...")
                collector._buffer = []
                target = int(60 * 1000 / collector.timestep)
                for _ in range(target):
                    collector.robot.step(collector.timestep)
                    collector.step_count += 1
                    collector._record_step(0)
                collector._save_session()

            elif key_char == 'Q' or key_char == 'q':
                print(f"\n[Collector] Quit. {collector._session_id} sessions saved.")
                break

            if not collector._collecting:
                collector.handle_key(key_char, key >= 0)

        if not collector._collecting:
            collector.set_speed(collector._manual_speed, collector._manual_turn)
        else:
            collector._manual_speed = 0.0; collector._manual_turn = 0.0

        if collector.step_count % 50 == 0:
            pos = collector.get_position()
            tnames = ["hard", "mud", "gravel", "shallow", "deep", "slope", "rubble"]
            print(f"[t={collector.robot.getTime():.1f}s] "
                  f"pos=({pos[0]:.1f},{pos[1]:.1f}) "
                  f"bat={collector.get_battery_pct():.1%} "
                  f"P={collector._estimate_power():.0f}W")


if __name__ == "__main__":
    main()
