"""Terrain Semantic Segmentation — Data Collection Controller v3
===================================================================
改进策略（对比 v1 原地旋转）：
  v1: 站在区域中心 × 8 角度旋转 → 376 张，高重复度
  v3: 网格分散采样 × 4-5 角度 → 目标 600+ 张，高覆盖度

核心改进：
  1. 大面积地形（硬地/泥地）：区域内网格撒点，而非单点旋转
  2. 小面积地形（浅水/陡坡）：多点环绕，增加空间多样性
  3. 边界采样：站在地形边缘，拍摄过渡区域
  4. 数据隔离：输出到 data/terrain_v2/ 与 v1 完全分开

用法：
  在 disaster_world_collect.wbt 中修改 controller 为 "terrain_collector_v2"
  或创建新的 disaster_world_collect_v2.wbt
"""

import os
import sys
import json
import numpy as np
from controller import Supervisor, Camera
from PIL import Image

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "terrain_v2")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")

CLASS_NAMES = ["hard", "mud", "gravel", "shallow", "deep", "slope", "rubble"]

# ── v3 采集策略：每类多点分散 + 少量角度 ──
# 格式: (x, y, z, 描述, angles)  — angles 为针对该点的推荐角度列表

COLLECTION_POINTS = {
    # ═══ 硬质路面 (646m²) — 22 点网格覆盖 hard_north + hard_south + beach ═══
    0: [
        # hard_north (20×16m, center at -9,14)
        (-15, 20, 0.35, "hn_nw"),  (-9, 20, 0.35, "hn_n"),   (-3, 20, 0.35, "hn_ne"),
        (-15, 16, 0.35, "hn_w"),   (-9, 16, 0.35, "hn_c"),   (-3, 16, 0.35, "hn_e"),
        (-15, 12, 0.35, "hn_sw1"), (-9, 12, 0.35, "hn_s"),   (-3, 12, 0.35, "hn_se1"),
        (-16, 7,  0.35, "hn_sw2a"), (-9, 8,  0.35, "hn_s2"),  (-3, 8,  0.35, "hn_se2"),
        (-12, 11, 0.35, "hn_sw2b"),  # 原 hn_sw2 移至浅水滩外
        # beach_north (38×2m, y≈5)
        (-12, 5, 0.30, "beach_nw"), (0, 5, 0.30, "beach_nc"), (12, 5, 0.30, "beach_ne"),
        # hard_south (13×12m, center at 12.5,-12)
        (8, -8,  0.35, "hs_nw"),   (12, -8, 0.35, "hs_n"),   (16, -8, 0.35, "hs_ne"),
        (8, -12, 0.35, "hs_c"),    (12,-12, 0.35, "hs_c2"),  (16,-12, 0.35, "hs_e"),
        (8, -16, 0.35, "hs_sw"),   (12,-16, 0.35, "hs_s"),   (16,-16, 0.35, "hs_se"),
        # beach_south
        (-12, -5, 0.30, "beach_sw"), (0, -5, 0.30, "beach_sc"), (12, -5, 0.30, "beach_se"),
    ],
    # ═══ 泥地 (300m²) — 15 点覆盖 mud_east (18×16m, center at 10,14) ═══
    1: [
        (4,  20, 0.35, "mud_nw"),  (10, 20, 0.35, "mud_n"),  (16, 20, 0.35, "mud_ne"),
        (4,  16, 0.35, "mud_w1"),  (10, 16, 0.35, "mud_c1"), (16, 16, 0.35, "mud_e1"),
        (4,  12, 0.35, "mud_w2"),  (10, 12, 0.35, "mud_c2"), (16, 12, 0.35, "mud_e2"),
        (4,  8,  0.35, "mud_sw1"), (10, 8,  0.35, "mud_s1"), (16, 8,  0.35, "mud_se1"),
        (4,  10, 0.35, "mud_sw2"), (10, 10, 0.35, "mud_s2"), (16, 10, 0.35, "mud_se2"),
    ],
    # ═══ 碎石 (164m²) — 12 点覆盖 gravel_west (25×6m, center at -6.5,-9) + mounds ═══
    2: [
        (-16, -7,  0.35, "g_nw"),  (-10, -7, 0.35, "g_n"),   (0, -7, 0.35, "g_ne"),
        (-18, -9,  0.35, "g_w"),   (-10, -9, 0.35, "g_c"),   (0, -9, 0.35, "g_e"),
        (-18, -11, 0.35, "g_sw"),  (-10,-11, 0.35, "g_s"),   (0,-11, 0.35, "g_se"),
        (-12, -10, 0.35, "g_m1"),  (-5,  -8, 0.35, "g_m2"),  (-2, -14, 0.35, "g_m3"),
    ],
    # ═══ 浅水 (50m²) — 12 点覆盖两个浅水滩（机器人可涉水） ═══
    3: [
        # pothole_1 at (-12,8), 6×5m, z=-0.04
        (-12, 8,  0.24, "ph1_c"),  (-10, 8, 0.25, "ph1_e"),
        (-14, 8,  0.25, "ph1_w"),  (-12, 10, 0.25, "ph1_n"),
        (-12, 6,  0.25, "ph1_s"),  (-11, 9, 0.25, "ph1_ne"),
        # pothole_2 at (14,-3), 5×4m, z=-0.04
        (14, -3,  0.24, "ph2_c"),  (16, -3, 0.25, "ph2_e"),
        (12, -3,  0.25, "ph2_w"),  (14, -1, 0.25, "ph2_n"),
        (14, -5,  0.25, "ph2_s"),  (15, -2, 0.25, "ph2_ne"),
    ],
    # ═══ 深水 (304m²) — 14 点横跨 38×8m 河流 ═══
    4: [
        (-16, 2,  0.20, "dw_nw"),  (-8, 2, 0.20, "dw_n1"),  (0, 2, 0.20, "dw_n2"),
        (8,  2,  0.20, "dw_n3"),   (16, 2, 0.20, "dw_ne"),
        (-16, 0,  0.20, "dw_cw"),  (-8, 0, 0.20, "dw_c1"),  (0, 0, 0.20, "dw_cc"),
        (8,  0,  0.20, "dw_c2"),   (16, 0, 0.20, "dw_ce"),
        (-16, -2, 0.20, "dw_sw"),  (-8,-2, 0.20, "dw_s1"),  (0,-2, 0.20, "dw_s2"),
        (8,  -2, 0.20, "dw_s3"),
    ],
    # ═══ 陡坡 (18m²) — 3 块不同位置/角度坡面 ═══
    5: [
        # 30°陡坡 at (-14,20), 3×2m
        (-14, 20, 0.48, "sl30_c"),   (-13, 20, 0.45, "sl30_e"),
        (-15, 20, 0.50, "sl30_w"),   (-14, 21, 0.55, "sl30_n"),
        (-14, 19, 0.40, "sl30_s"),
        # 20°碎石坡 at (-18,-6), 3×2m
        (-18, -6, 0.38, "sl20_c"),   (-17, -6, 0.35, "sl20_e"),
        (-19, -6, 0.40, "sl20_w"),   (-18, -5, 0.42, "sl20_n"),
        # 25°南岸坡 at (18,-12), 3×2m
        (18, -12, 0.42, "sl25_c"),   (19, -12, 0.40, "sl25_e"),
        (17, -12, 0.44, "sl25_w"),   (18, -11, 0.46, "sl25_n"),
    ],
    # ═══ 废墟 (157m²) — 14 点覆盖 rubble_zone (25×6m) + debris + wall ═══
    6: [
        # rubble_zone (center at -6.5,-15, 25×6m)
        (-18, -13, 0.35, "rb_nw"),  (-10, -13, 0.35, "rb_n"),  (0, -13, 0.35, "rb_ne"),
        (-18, -15, 0.35, "rb_w"),   (-10, -15, 0.35, "rb_c"),  (0, -15, 0.35, "rb_e"),
        (-18, -17, 0.35, "rb_sw"),  (-10,-17, 0.35, "rb_s"),   (0, -17, 0.35, "rb_se"),
        # scattered debris
        (10, -8, 0.35, "rb_steps"), (7, -10, 0.35, "rb_debris"),
        (13, -14, 0.35, "rb_debris2"), (6, -16, 0.35, "rb_debris3"),
    ],
}

# 每点使用 5 个角度（0°, 72°, 144°, 216°, 288°）代替 8 个角度
# 减少角度冗余，把"点数"换来更多空间分布
ANGLES = [0.0, 1.257, 2.513, 3.770, 5.027]  # 5 angles, 72° spacing
ANGLE_NAMES = ["0", "72", "144", "216", "288"]


class TerrainDataCollectorV2:
    """v3 地形数据采集器 — 网格分散 + 边界采样"""

    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot_node = self.robot.getSelf()

        self.camera = self.robot.getDevice("rgb_camera")
        if self.camera is None:
            raise RuntimeError("rgb_camera not found")
        self.camera.enable(self.timestep)
        self.cam_w = self.camera.getWidth()
        self.cam_h = self.camera.getHeight()

        self._disable_degradation()
        self.total_captures = 0
        self.manifest = []

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        total = sum(len(pts) for pts in COLLECTION_POINTS.values()) * len(ANGLES)
        num_positions = sum(len(pts) for pts in COLLECTION_POINTS.values())
        print(f"[Collector v3] Output: {OUTPUT_DIR}")
        print(f"[Collector v3] Strategy: grid-distributed, {len(ANGLES)} angles per point")
        print(f"[Collector v3] Plan: {num_positions} positions x {len(ANGLES)} angles = {total} images")
        for c in range(7):
            n = len(COLLECTION_POINTS[c])
            print(f"  Class {c} {CLASS_NAMES[c]:10s}: {n:2d} pos x {len(ANGLES)} ang = {n*len(ANGLES):3d} imgs")

    def _disable_degradation(self):
        fog = self.robot.getFromDef("FOG")
        sun = self.robot.getFromDef("SUN")
        if fog:
            fog.getField("visibilityRange").setSFFloat(0.0)
        if sun:
            sun.getField("intensity").setSFFloat(1.0)

        # 隐藏品红色边界线（采集时不需要边界标记）
        boundary_defs = ["BOUNDARY_HM", "BOUNDARY_GR"]
        for bdef in boundary_defs:
            node = self.robot.getFromDef(bdef)
            if node:
                node.getField("translation").setSFVec3f([0, 0, -100])
        print("[Collector v3] Degradation disabled, boundaries hidden")

    def teleport(self, x, y, z, yaw=0.0):
        trans = self.robot_node.getField("translation")
        trans.setSFVec3f([x, y, z])
        rot = self.robot_node.getField("rotation")
        rot.setSFRotation([0, 0, 1, yaw])
        for _ in range(16):
            self.robot.step(self.timestep)

    def capture_frame(self):
        img = np.frombuffer(self.camera.getImage(), dtype=np.uint8)
        return img.reshape((self.cam_h, self.cam_w, 4))[:, :, :3].copy()

    def save_sample(self, rgb, class_id, pos_name, angle_idx):
        angle_name = ANGLE_NAMES[angle_idx]
        fname = f"c{class_id}_{CLASS_NAMES[class_id]}_{pos_name}_a{angle_name}"
        img_path = os.path.join(OUTPUT_DIR, f"{fname}.png")
        lbl_path = os.path.join(OUTPUT_DIR, f"{fname}_label.png")

        Image.fromarray(rgb).save(img_path)
        label = np.full((self.cam_h, self.cam_w), class_id, dtype=np.uint8)
        Image.fromarray(label, mode="L").save(lbl_path)

        self.total_captures += 1
        self.manifest.append({
            "file": fname, "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "position": pos_name, "angle": angle_name,
        })

    def run(self):
        print("\n" + "=" * 60)
        print("  Terrain Data Collection v3 — Grid-Distributed Sampling")
        print("=" * 60)

        for class_id in range(7):
            points = COLLECTION_POINTS[class_id]
            class_name = CLASS_NAMES[class_id]
            print(f"\n-- Class {class_id}: {class_name} ({len(points)} positions x {len(ANGLES)} angles = {len(points)*len(ANGLES)} imgs) --")

            for px, py, pz, pname in points:
                for ai, (angle, aname) in enumerate(zip(ANGLES, ANGLE_NAMES)):
                    self.teleport(px, py, pz, angle)
                    rgb = self.capture_frame()
                    self.save_sample(rgb, class_id, pname, ai)

                print(f"  [{class_name}] {pname} done")

        # 保存 manifest
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

        # 统计摘要
        from collections import Counter
        counts = Counter(e["class_id"] for e in self.manifest)
        print(f"\n{'=' * 60}")
        print(f"  Collection Complete!")
        print(f"  Total: {self.total_captures} images")
        print(f"  Per class: {dict(sorted(counts.items()))}")
        print(f"  Output: {OUTPUT_DIR}")
        print(f"{'=' * 60}")

        self.robot.simulationSetMode(0)


def main():
    collector = TerrainDataCollectorV2()
    collector.run()


if __name__ == "__main__":
    main()
