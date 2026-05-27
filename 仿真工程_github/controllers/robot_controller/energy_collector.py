"""energy_collector.py — 创新点三 能耗数据采集控制器
=========================================================
在Webots中自动驱动机器人采集功率时序数据。
按键控制（在Webots仿真窗口中操作）：
  S    — 开始自动采集（遍历预设路线）
  R    — 手动记录当前地形的一分钟数据
  1-7  — 设置当前地形标签（手动模式）
  Q    — 退出并保存

输出: data/energy/real_scenario_{N:03d}.json
每步记录: terrain, speed, load_kg, power_w, flow, battery_pct, position
"""

from controller import Robot, Supervisor, GPS, Camera, Motor, Keyboard
import numpy as np
import json
import os
import sys
import time

# 添加控制器路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from robot_controller import RescueRobotController


# ===================== 预设采集路线 =====================

# 每条路线: (名称, 起点(x,y,z), 速度, 持续时间s, 地形标签)
ROUTES = [
    # 北岸硬地
    ("hard_cruise",    (-12, 20, 0.3),  6.0, 30, 0),
    ("hard_fast",      (-12, 20, 0.3),  9.0, 30, 0),
    ("hard_slow",      (-12, 20, 0.3),  3.0, 30, 0),
    # 东侧泥地
    ("mud_cruise",     (12, 18, 0.3),   5.0, 30, 1),
    ("mud_fast",       (12, 18, 0.3),   8.0, 25, 1),
    # 河流 (水域)
    ("water_cruise",   (0, 2, 0.3),     4.0, 30, 3),
    ("water_fast",     (0, 2, 0.3),     7.0, 25, 3),
    # 南岸碎石
    ("gravel_cruise",  (-12, -8, 0.3),  5.0, 30, 2),
    ("gravel_fast",    (-12, -8, 0.3),  8.0, 25, 2),
    # 南岸废墟
    ("rubble_slow",    (-8, -15, 0.3),  3.0, 30, 6),
    ("rubble_cruise",  (-8, -15, 0.3),  5.0, 25, 6),
    # 南岸硬地
    ("hard_south_cruise", (14, -10, 0.3), 6.0, 30, 0),
]


class EnergyCollector(RescueRobotController):
    """能耗数据采集器 — 继承救援机器人控制器，复用传感器和运动接口"""

    def __init__(self):
        super().__init__()
        self._collecting = False
        self._buffer = []
        self._manual_terrain_label = 0
        self._output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "energy")
        os.makedirs(self._output_dir, exist_ok=True)
        self._session_id = 0

        # 找下一个可用的session ID
        while os.path.exists(os.path.join(self._output_dir, f"real_scenario_{self._session_id:03d}.json")):
            self._session_id += 1

        print("\n" + "=" * 60)
        print("  ENERGY DATA COLLECTOR")
        print("  S = Auto-collect all routes")
        print("  R = Record 60s on current terrain")
        print("  1-7 = Set terrain label (0=hard, 1=mud, 2=gravel,")
        print("        3=shallow, 4=deep, 5=slope, 6=rubble)")
        print("  Q = Save & quit")
        print("=" * 60 + "\n")

    def _record_step(self, data, terrain_label):
        """记录一步数据"""
        pos = self.get_position()
        self._buffer.append({
            "step": len(self._buffer),
            "t": round(data["timestamp"], 1),
            "terrain": terrain_label,
            "speed": round(abs(self.left_motors[0].getVelocity()) if self.left_motors else 0, 2),
            "load_kg": 0.0,  # 当前无负载传感器，手动估算
            "flow_vx": round(data.get("water_flow", [0, 0])[0], 3),
            "flow_vy": round(data.get("water_flow", [0, 0])[1], 3),
            "power_w": round(data["power_w"], 1),
            "battery_pct": round(data["battery_pct"], 4),
            "position": [round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)],
        })

    def _save_session(self, route_name, terrain_label, speed):
        """保存当前采集缓冲区到文件"""
        if not self._buffer:
            print("[Collector] Buffer empty — nothing to save")
            return

        filename = f"real_scenario_{self._session_id:03d}.json"
        filepath = os.path.join(self._output_dir, filename)

        record = {
            "scenario_id": self._session_id,
            "route": route_name,
            "terrain_label": terrain_label,
            "speed_setpoint": speed,
            "n_steps": len(self._buffer),
            "steps": self._buffer,
        }

        with open(filepath, "w") as f:
            json.dump(record, f)

        powers = [s["power_w"] for s in self._buffer]
        print(f"[Collector] Saved {len(self._buffer)} steps → {filename}")
        print(f"  Power: mean={np.mean(powers):.0f}W, "
              f"min={np.min(powers):.0f}W, max={np.max(powers):.0f}W")

        self._session_id += 1
        self._buffer = []

    def run_route(self, name, start_pos, speed, duration_s, terrain_label):
        """执行一条采集路线"""
        print(f"\n[Collector] Route: {name} | speed={speed:.1f} rad/s | "
              f"terrain={terrain_label} | {duration_s}s")

        # 瞬移到起点
        node = self.robot.getSelf()
        node.getField("translation").setSFVec3f(start_pos)
        # 稳定物理
        for _ in range(32):
            self.robot.step(self.timestep)

        # 开始驾驶
        self.set_speed(speed, 0.0)
        if start_pos[1] < 4 and start_pos[1] > -4:  # 水中
            self.set_water_thrust(1.0, 1.0)

        self._collecting = True
        start_step = self.step_count
        target_steps = int(duration_s * 1000 / self.timestep)

        while self._collecting and (self.step_count - start_step) < target_steps:
            self.robot.step(self.timestep)
            self.step_count += 1
            data = self.get_sensor_data()
            self._record_step(data, terrain_label)

        self.stop()
        self.set_water_thrust(0.0, 0.0)
        self._collecting = False
        self._save_session(name, terrain_label, speed)

    def manual_record(self, duration_s=60, terrain_label=None):
        """手动记录：保持当前状态记录指定时长"""
        if terrain_label is None:
            terrain_label = self._manual_terrain_label

        print(f"\n[Collector] Manual record: {duration_s}s, terrain={terrain_label}")
        print("[Collector] Keep driving manually during recording...")

        self._collecting = True
        target_steps = int(duration_s * 1000 / self.timestep)

        for _ in range(target_steps):
            self.robot.step(self.timestep)
            self.step_count += 1
            data = self.get_sensor_data()
            self._record_step(data, terrain_label)

        self._collecting = False
        self._save_session("manual", terrain_label, 0)


# ===================== 主循环 =====================

def main():
    collector = EnergyCollector()

    # 启动时先停稳
    collector.stop()
    for _ in range(32):
        collector.robot.step(collector.timestep)

    print("[Collector] Ready. Press S to start auto-collection, R for manual, Q to quit.")

    while collector.robot.step(collector.timestep) != -1:
        collector.step_count += 1

        key = collector.keyboard.getKey()
        if key >= 0:
            key_char = chr(key) if 32 <= key <= 126 else ""

            if key_char == 'S' or key_char == 's':
                print("\n[Collector] === Auto-collecting all routes ===")
                for route in ROUTES:
                    collector.run_route(*route)
                    # 路线间冷却
                    collector.stop()
                    for _ in range(16):
                        collector.robot.step(collector.timestep)
                print(f"\n[Collector] === Done! {len(ROUTES)} routes collected ===")

            elif key_char == 'R' or key_char == 'r':
                collector.manual_record(60, collector._manual_terrain_label)

            elif key_char == 'Q' or key_char == 'q':
                print(f"\n[Collector] Quit. Sessions saved: {collector._session_id}")
                break

            elif key_char in '0123456':
                collector._manual_terrain_label = int(key_char)
                names = ["hard", "mud", "gravel", "shallow", "deep", "slope", "rubble"]
                print(f"[Collector] Terrain label set: {collector._manual_terrain_label} ({names[collector._manual_terrain_label]})")

        # 手动驾驶支持
        if not collector._collecting:
            if key >= 0:
                key_char = chr(key) if 32 <= key <= 126 else ""
                collector.manual_handle_key(key_char, key >= 0)
            collector.manual_drive_step()

    print("[Collector] Session ended.")


if __name__ == "__main__":
    main()
