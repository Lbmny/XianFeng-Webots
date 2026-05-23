"""Terrain Semantic Segmentation — Data Collection Controller
==============================================================
Webots Supervisor 控制器：自动遍历各大地形区域，采集 RGB 图像 + 地形类别标注。
输出到 ../../data/terrain/ 目录。

用法：
  1. 在 disaster_world.wbt 中将 Robot.controller 改为 "terrain_collector"
  2. 或在 Webots GUI 中右键 Robot → Edit → controller 选 terrain_collector
  3. 运行仿真，自动采集完成后仿真自动结束

7 类地形标注：
  0 = hard       硬质路面（硬地、沙滩）
  1 = mud        泥地
  2 = gravel     碎石
  3 = shallow    浅水（坑洼积水）
  4 = deep       深水（河流）
  5 = slope      陡坡（>20°）
  6 = rubble     废墟
"""

import os
import sys
import json
import numpy as np
from controller import Supervisor, Camera, GPS
from PIL import Image

# ── 输出路径 ──
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "terrain")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")

# ── 7 类地形 → 采集坐标 (x, y, z, 描述) ──
# 坐标基于 disaster_world.wbt 中各 Solid 节点的 translation，确保机器人站在该类地形上
COLLECTION_POINTS = {
    0: [  # 硬质路面 — hard_north / hard_south / beach / hard_hill(12°)
        (-12, 18, 0.35, "hard_north_center"),
        (-6,  12, 0.35, "hard_north_mid"),
        (-2,  18, 0.35, "hard_north_east"),
        (0,   7,  0.30, "hard_beach"),
        (6,  -5,  0.30, "beach_south"),
        (14, -14, 0.35, "hard_south_center"),
        (10, -10, 0.35, "hard_south_north"),
        (16, -12, 0.35, "hard_south_east"),
        (-9,  12, 0.45, "hard_hill_12deg"),  # 12°缓坡也算硬地
    ],
    1: [  # 泥地 — mud_east
        (14, 18, 0.35, "mud_east_center"),
        (8,  14, 0.35, "mud_east_west"),
        (16, 10, 0.35, "mud_east_north"),
        (12, 12, 0.35, "mud_east_mid"),
        (6,  16, 0.35, "mud_east_south"),
        (10, 8,  0.35, "mud_east_edge"),
        (18, 14, 0.35, "mud_east_far"),
    ],
    2: [  # 碎石 — gravel_west (y: -12~-6, 25×6)
        (-10, -7,  0.35, "gravel_north"),
        (-10, -9,  0.35, "gravel_mid"),
        (-10, -11, 0.35, "gravel_south"),
        (-3,  -9,  0.35, "gravel_east"),
        (-18, -9,  0.35, "gravel_west"),
        (0,   -10, 0.35, "gravel_edge"),
    ],
    3: [  # 浅水 — 积水坑洼（已扩大至3×2m）
        (-14, 8,  0.30, "pothole_1"),
        (14, -3,  0.30, "pothole_2"),
        (-13, 9,  0.30, "pothole_1_edge"),
        (13, -2,  0.30, "pothole_2_edge"),
        (-15, 8,  0.30, "pothole_1_near"),
        (15, -3,  0.30, "pothole_2_near"),
    ],
    4: [  # 深水 — 河流
        (-8,  0,  0.20, "water_west"),
        (0,   0,  0.20, "water_center"),
        (8,   0,  0.20, "water_east"),
        (-4,  2,  0.20, "water_north"),
        (4,  -2,  0.20, "water_south"),
        (-14, 0,  0.20, "water_far_west"),
        (14,  0,  0.20, "water_far_east"),
    ],
    5: [  # 陡坡 — steep_slope(30°, 红棕色) 独立contactMaterial
        (-14, 20, 0.50, "steep_slope_30deg"),
        (-13, 19.5, 0.45, "steep_slope_mid"),
        (-14.5, 20.5, 0.55, "steep_slope_top"),
        (-15, 19, 0.40, "steep_slope_bottom"),
        (-13, 20, 0.48, "steep_slope_center"),
    ],
    6: [  # 废墟 — rubble_zone (y:-18~-12, 25×6) + debris + wall
        (-10, -13, 0.35, "rubble_zone_north"),
        (-10, -15, 0.35, "rubble_zone_mid"),
        (-10, -17, 0.35, "rubble_zone_south"),
        (-3,  -15, 0.35, "rubble_zone_east"),
        (-18, -15, 0.35, "rubble_zone_west"),
        (0,   -16, 0.35, "rubble_zone_edge"),
        (10,  -7,  0.35, "rubble_steps"),
        (-5, -14,  0.35, "collapsed_wall"),
    ],
}

# 8 个朝向角 (rad)
ANGLES = [0.0, 0.785, 1.571, 2.356, 3.142, 3.927, 4.712, 5.498]
ANGLE_NAMES = ["0", "45", "90", "135", "180", "225", "270", "315"]
CLASS_NAMES = ["hard", "mud", "gravel", "shallow", "deep", "slope", "rubble"]


class TerrainDataCollector:
    """地形语义分割数据自动采集器"""

    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot_node = self.robot.getSelf()

        # 相机
        self.camera = self.robot.getDevice("rgb_camera")
        if self.camera is None:
            raise RuntimeError("rgb_camera not found — check world file")
        self.camera.enable(self.timestep)
        self.cam_w = self.camera.getWidth()
        self.cam_h = self.camera.getHeight()

        # GPS
        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.timestep)

        # 禁用退化效果 — 采集干净图像
        self._disable_degradation()

        # 统计
        self.total_captures = 0
        self.manifest = []

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"[Collector] Output: {OUTPUT_DIR}")
        print(f"[Collector] Camera: {self.cam_w}x{self.cam_h}")
        total = sum(len(pts) for pts in COLLECTION_POINTS.values()) * len(ANGLES)
        num_positions = sum(len(pts) for pts in COLLECTION_POINTS.values())
        print(f"[Collector] Plan: {num_positions} positions x {len(ANGLES)} angles = {total} images")
        print(f"[Collector] Classes: {[(c, CLASS_NAMES[c], len(COLLECTION_POINTS[c])) for c in range(7)]}")

    def _disable_degradation(self):
        """关闭雾气/光照退化，采集原始清晰图像"""
        fog = self.robot.getFromDef("FOG")
        sun = self.robot.getFromDef("SUN")
        if fog:
            fog.getField("visibilityRange").setSFFloat(0.0)
        if sun:
            sun.getField("intensity").setSFFloat(1.0)

        # 隐藏品红色边界线（采集时不需要）
        for bdef in ["BOUNDARY_HM", "BOUNDARY_GR"]:
            node = self.robot.getFromDef(bdef)
            if node:
                node.getField("translation").setSFVec3f([0, 0, -100])
        print("[Collector] Degradation disabled, boundaries hidden")

    def teleport(self, x, y, z, yaw=0.0):
        """将机器人瞬移到目标位置+朝向"""
        trans = self.robot_node.getField("translation")
        trans.setSFVec3f([x, y, z])
        rot = self.robot_node.getField("rotation")
        rot.setSFRotation([0, 0, 1, yaw])
        # 让步进几帧让物理稳定（落地）
        for _ in range(16):
            self.robot.step(self.timestep)

    def capture_frame(self):
        """捕获当前帧 RGB 图像 (numpy H×W×3)"""
        img = np.frombuffer(self.camera.getImage(), dtype=np.uint8)
        return img.reshape((self.cam_h, self.cam_w, 4))[:, :, :3].copy()

    def save_sample(self, rgb, class_id, pos_name, angle_idx):
        """保存一张图像 + 标注"""
        angle_name = ANGLE_NAMES[angle_idx]
        fname = f"c{class_id}_{CLASS_NAMES[class_id]}_{pos_name}_a{angle_name}"
        img_path = os.path.join(OUTPUT_DIR, f"{fname}.png")
        lbl_path = os.path.join(OUTPUT_DIR, f"{fname}_label.png")

        # 保存 RGB
        Image.fromarray(rgb).save(img_path)

        # 保存 label（整图 mask，像素值 = 类别 ID）
        label = np.full((self.cam_h, self.cam_w), class_id, dtype=np.uint8)
        Image.fromarray(label, mode="L").save(lbl_path)

        self.total_captures += 1
        self.manifest.append({
            "file": fname,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "position": pos_name,
            "angle": angle_name,
        })

    def run(self):
        """主采集循环"""
        print("\n" + "=" * 60)
        print("  Terrain Data Collection — Starting")
        print("=" * 60)

        for class_id in range(7):
            points = COLLECTION_POINTS[class_id]
            class_name = CLASS_NAMES[class_id]
            print(f"\n── Class {class_id}: {class_name} ({len(points)} positions) ──")

            for px, py, pz, pname in points:
                for ai, (angle, aname) in enumerate(zip(ANGLES, ANGLE_NAMES)):
                    # 瞬移 + 稳定
                    self.teleport(px, py, pz, angle)

                    # 捕获
                    rgb = self.capture_frame()
                    self.save_sample(rgb, class_id, pname, ai)

                print(f"  [{class_name}] {pname} — 8 angles done")

            print(f"  Class {class_id} complete: {len(points) * 8} images")

        # 保存 manifest
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print(f"  Collection Complete!")
        print(f"  Total images: {self.total_captures}")
        print(f"  Output: {OUTPUT_DIR}")
        print(f"  Manifest: {MANIFEST_PATH}")
        print(f"{'=' * 60}")

        # 采集完成，停止仿真
        self.robot.simulationSetMode(0)  # PAUSE


def main():
    collector = TerrainDataCollector()
    collector.run()


if __name__ == "__main__":
    main()
