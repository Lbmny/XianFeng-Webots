"""detection_collector.py — 目标检测训练数据自动采集
=========================================================
自动在幸存者周围多距离×多退化级别截图，无需人工操作。

按键: S = 开始采集, Q = 退出

采集策略:
  - 8个距离 (2m~20m)
  - 4个退化级别 (1/2/3/4)
  - 每位置RGB+IR各1张
  - 总计 8×4×2 = 64张

输出: data/detection/screenshots/rgb_{idx:04d}.png + ir_{idx:04d}.png
"""

from controller import Supervisor, Keyboard
import numpy as np
import cv2
import os
import sys
import math
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ===================== 采集配置 =====================

SURVIVOR_POS = (4.0, 0.0)  # 幸存者位置

# 采集位置: (机器人x, 机器人y, 距离描述)
# 南北两岸 × 多距离 × 多角度，幸存者在(4,0)水域
# 采集位置: 环绕幸存者(4,0), 多距离×8角度
# 角度: 0=N, 45=NE, 90=E, 135=SE, 180=S, 225=SW, 270=W, 315=NW
DISTANCES = [18, 12, 8, 4]
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

COLLECT_POSITIONS = []
for d in DISTANCES:
    for a_deg in ANGLES:
        a = math.radians(a_deg)
        x = 4 + d * math.cos(a)
        y = 0 + d * math.sin(a)
        COLLECT_POSITIONS.append((round(x, 1), round(y, 1), f"d{d}a{a_deg}"))


# 退化级别: level_key, name
DEGRADATION_LEVELS = [
    ("1", "clean"),
    ("2", "light_fog"),
    ("3", "heavy_smoke"),
    ("4", "dark"),
]


class DetectionCollector:
    """自动检测数据采集器"""

    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self._robot_node = self.robot.getSelf()

        # RGB + IR
        self.rgb_camera = self.robot.getDevice("rgb_camera")
        if self.rgb_camera:
            self.rgb_camera.enable(self.timestep)
        self.ir_camera = self.robot.getDevice("ir_camera")
        if self.ir_camera:
            self.ir_camera.enable(self.timestep)

        # Fog + Sun (退化控制)
        self.fog_node = self.robot.getFromDef("FOG")
        self.sun_node = self.robot.getFromDef("SUN")

        # 键盘
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)

        # 输出目录 (用abspath避免中文路径编码问题)
        self._out_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "detection", "screenshots"))
        os.makedirs(self._out_dir, exist_ok=True)
        print(f"[Collector] Output: {self._out_dir}")

        # 找下一个可用起始编号
        self._idx = 0
        while os.path.exists(os.path.join(self._out_dir, f"rgb_{self._idx:04d}.png")):
            self._idx += 1

        # 记录采集日志
        self._log = []

        print("\n" + "=" * 55)
        print("  DETECTION DATA COLLECTOR")
        print(f"  {len(COLLECT_POSITIONS)} distances × {len(DEGRADATION_LEVELS)} degradations")
        print(f"  = {len(COLLECT_POSITIONS) * len(DEGRADATION_LEVELS)} pairs (~{len(COLLECT_POSITIONS) * len(DEGRADATION_LEVELS) * 3}s)")
        print("  S = Start, Q = Quit")
        print("=" * 55 + "\n")

    def _set_degradation(self, level_key: str):
        """设置退化级别 (复用DegradationEngine逻辑)"""
        levels = {
            "1": ("clean",       0,   1.0,  [0.7, 0.7, 0.7]),
            "2": ("light_fog",  50,   0.8,  [0.7, 0.7, 0.7]),
            "3": ("heavy_smoke", 15,  0.5,  [0.4, 0.4, 0.4]),
            "4": ("dark",        8,  0.05,  [0.1, 0.1, 0.1]),
        }
        name, vr, li, fc = levels.get(level_key, levels["1"])
        if self.fog_node:
            self.fog_node.getField("visibilityRange").setSFFloat(float(vr))
            self.fog_node.getField("color").setSFColor(fc)
        if self.sun_node:
            self.sun_node.getField("intensity").setSFFloat(li)
        return name

    def _process_rgb(self, img: np.ndarray, degradation: str) -> np.ndarray:
        """RGB后处理 (与DegradationEngine一致)"""
        img = img.astype(np.float32)
        h, w = img.shape[:2]
        if degradation == "light_fog":
            img = img * 0.85 + 200 * 0.15
        elif degradation == "heavy_smoke":
            coarse = np.random.rand(max(h // 20, 1), max(w // 20, 1)).astype(np.float32)
            coarse = np.kron(coarse, np.ones((20, 20)))[:h, :w]
            img = img + (coarse[:, :, None] - 0.5) * 50
            img = img + np.random.randn(h, w, 1).astype(np.float32) * 10
            img = img * 0.65 + 128 * 0.35
        elif degradation == "dark":
            img = img * 0.15 + np.random.randn(h, w, 1).astype(np.float32) * 3
        return np.clip(img, 0, 255).astype(np.uint8)

    def _process_ir(self, img: np.ndarray, degradation: str) -> np.ndarray:
        """IR后处理 (红外穿透力强)"""
        img = img.astype(np.float32)
        h, w = img.shape[:2]
        if degradation == "light_fog":
            img = img * 0.9 + 180 * 0.1
        elif degradation == "heavy_smoke":
            img = img + np.random.randn(h, w, 1).astype(np.float32) * 3
        # dark: IR不受影响
        return np.clip(img, 0, 255).astype(np.uint8)

    def _save_img(self, img_bgr: np.ndarray, name: str) -> bool:
        """用imencode+文件写入绕过中文路径编码问题"""
        path = os.path.join(self._out_dir, name)
        _, buf = cv2.imencode('.png', img_bgr)
        if buf is not None:
            with open(path, 'wb') as f:
                f.write(buf.tobytes())
            return True
        return False

    def _capture(self, degradation: str, dist_label: str):
        """保存当前帧的RGB和IR"""
        # RGB
        rgb_raw = np.frombuffer(self.rgb_camera.getImage(), dtype=np.uint8)
        h, w = self.rgb_camera.getHeight(), self.rgb_camera.getWidth()
        rgb = rgb_raw.reshape((h, w, 4))[:, :, :3]
        rgb = self._process_rgb(rgb, degradation)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not self._save_img(rgb_bgr, f"rgb_{self._idx:04d}.png"):
            print(f"  [WARN] Failed to write rgb_{self._idx:04d}")

        # IR
        ir_raw = np.frombuffer(self.ir_camera.getImage(), dtype=np.uint8)
        ir = ir_raw.reshape((h, w, 4))[:, :, :3]
        ir = self._process_ir(ir, degradation)
        ir_bgr = cv2.cvtColor(ir, cv2.COLOR_RGB2BGR) if len(ir.shape) == 3 else ir
        if not self._save_img(ir_bgr, f"ir_{self._idx:04d}.png"):
            print(f"  [WARN] Failed to write ir_{self._idx:04d}")

        # 日志
        entry = {
            "idx": self._idx,
            "degradation": degradation,
            "distance": dist_label,
            "rgb": f"rgb_{self._idx:04d}.png",
            "ir": f"ir_{self._idx:04d}.png",
        }
        self._log.append(entry)
        print(f"  [{self._idx:04d}] {degradation:12s} @ {dist_label:12s} ✓")

        self._idx += 1

    def _save_log(self):
        """保存采集日志"""
        log_path = os.path.join(self._out_dir, "collect_log.json")
        with open(log_path, "w") as f:
            json.dump(self._log, f, indent=2)
        print(f"\n[Collector] Log saved: {log_path}")

    def run(self):
        """自动采集主流程"""
        print("[Collector] Starting auto-collection...\n")
        total = len(COLLECT_POSITIONS) * len(DEGRADATION_LEVELS)

        for px, py, dist_label in COLLECT_POSITIONS:
            # 瞬移到采集位置 (面朝幸存者(4,0), -Y是车头)
            heading = math.atan2(4.0 - px, py)  # 车头(-Y)朝向幸存者
            self._robot_node.getField("translation").setSFVec3f([float(px), float(py), 0.35])
            self._robot_node.getField("rotation").setSFRotation([0, 0, 1, heading])

            for _ in range(32):
                self.robot.step(self.timestep)

            for level_key, deg_name in DEGRADATION_LEVELS:
                self._set_degradation(level_key)
                # 退化生效需等几帧
                for _ in range(16):
                    self.robot.step(self.timestep)

                self._capture(deg_name, dist_label)

        self._set_degradation("1")  # 恢复干净
        self._save_log()
        print(f"\n[Collector] Done! {total} pairs → {self._out_dir}")


def main():
    collector = DetectionCollector()

    while collector.robot.step(collector.timestep) != -1:
        key = collector.keyboard.getKey()
        if key >= 0:
            key_char = chr(key) if 32 <= key <= 126 else ""
            if key_char in ('S', 's'):
                collector.run()
            elif key_char in ('Q', 'q'):
                print("[Collector] Quit.")
                break


if __name__ == "__main__":
    main()
