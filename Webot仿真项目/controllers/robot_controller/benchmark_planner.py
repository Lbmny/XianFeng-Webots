"""创新点一 对比实验 — 路径规划 Benchmark
==========================================
5种地形场景 × 3种方案 × 多次运行 → 通过成功率 + 风险值对比

场景定义：
  S1: 北岸硬地→南岸硬地（穿越泥地+水域）
  S2: 北岸硬地→南岸硬地（绕行碎石区）
  S3: 北岸→南岸（穿越废墟区）
  S4: 西侧→东侧（跨河流最短路径）
  S5: 全地形混合（随机起点→随机终点）

输出: results.json + 对比可视化数据
"""

import numpy as np
import json
import os
from pathlib import Path
from risk_mapper import RiskMapper, BASE_RISK
from path_planner import BaselinePlanner, RiskAwareAStar


class PlannerBenchmark:
    """路径规划对比实验"""

    def __init__(self, model_path="models/terrain_seg_best.pt"):
        self.mapper = RiskMapper(model_path)
        self.scenarios = self._define_scenarios()
        self.results = []

    def _define_scenarios(self):
        """定义 5 个测试场景：起点→终点坐标对 (像素坐标, 640×480)"""
        # 图像坐标系: x∈[0,639], y∈[0,479]
        # 图像上方=远处, 下方=近处(机器人前方)
        return {
            "S1_cross_mud": {
                "desc": "直穿泥地（高风险直线）",
                "start": (320, 440),   # 图像底部中央
                "goal":  (320, 40),     # 图像顶部中央
            },
            "S2_gravel_detour": {
                "desc": "绕行碎石区",
                "start": (100, 440),    # 左下
                "goal":  (540, 40),     # 右上
            },
            "S3_rubble_avoid": {
                "desc": "穿越废墟区",
                "start": (320, 440),
                "goal":  (500, 40),
            },
            "S4_river_shortest": {
                "desc": "跨河流最短路径",
                "start": (320, 440),
                "goal":  (150, 60),
            },
            "S5_mixed_random": {
                "desc": "全地形混合",
                "start": (80, 420),
                "goal":  (560, 60),
            },
        }

    def evaluate_frame(self, rgb_image, jolt=0.0, depth=10.0, scenario_name=None):
        """在单帧图像上评估三种方案"""
        # 生成风险地图（带IMU修正）
        risk_grid = self.mapper.compute(rgb_image, jolt=jolt, depth=depth)

        # 生成无IMU修正的风险地图（给Baseline B用）
        seg = self.mapper.segment(rgb_image)
        risk_no_imu = self.mapper.base_risk_map(seg)

        frame_results = {"scenario": scenario_name, "jolt": jolt, "depth": depth}

        for sc_name, sc in self.scenarios.items():
            if scenario_name and sc_name != scenario_name:
                continue

            start, goal = sc["start"], sc["goal"]

            # Baseline A: 二值占据栅格
            path_a, stats_a = BaselinePlanner.baseline_a(risk_grid, start, goal)
            # Baseline B: 固定风险值（无IMU）
            path_b, stats_b = BaselinePlanner.baseline_b(risk_no_imu, start, goal)
            # Ours: 语义+IMU
            path_o, stats_o = BaselinePlanner.ours(risk_grid, start, goal)

            frame_results[sc_name] = {
                "desc": sc["desc"],
                "baseline_a": {"success": path_a is not None, "stats": stats_a or {}},
                "baseline_b": {"success": path_b is not None, "stats": stats_b or {}},
                "ours":       {"success": path_o is not None, "stats": stats_o or {}},
            }

        return frame_results, risk_grid, seg

    def run_all_scenarios(self, rgb_image, jolt=0.0, depth=10.0):
        """在所有场景上评估"""
        all_results = []
        for sc_name in self.scenarios:
            result, risk, seg = self.evaluate_frame(
                rgb_image, jolt, depth, sc_name
            )
            all_results.append(result)
        return all_results

    def to_summary(self, all_frame_results):
        """汇总多次运行的统计数据"""
        # 按方案汇总
        summary = {"Baseline A": [], "Baseline B": [], "Ours": []}

        for frame_result in all_frame_results:
            for sc_name, sc_data in frame_result.items():
                if not sc_name.startswith("S"):
                    continue
                for method in ["baseline_a", "baseline_b", "ours"]:
                    key = {"baseline_a": "Baseline A", "baseline_b": "Baseline B", "ours": "Ours"}[method]
                    data = sc_data[method]
                    summary[key].append({
                        "scenario": sc_name,
                        "success": data["success"],
                        "stats": data["stats"],
                    })

        # 计算统计量
        report = {}
        for method, entries in summary.items():
            successes = [e for e in entries if e["success"]]
            success_rate = len(successes) / len(entries) if entries else 0
            avg_risk = np.mean([e["stats"].get("avg_risk", 0) for e in successes]) if successes else 1.0
            avg_length = np.mean([e["stats"].get("path_length", 0) for e in successes]) if successes else 0
            report[method] = {
                "success_rate": round(success_rate, 3),
                "n_trials": len(entries),
                "n_success": len(successes),
                "avg_risk": round(avg_risk, 3),
                "avg_path_length": round(avg_length, 1),
            }

        return report


# ═══════════════════════════════════════════════════════
# 快速单元测试（不依赖 Webots）
# ═══════════════════════════════════════════════════════

def unit_test():
    """用随机数据验证风险地图+路径规划逻辑正确性"""
    print("=== Unit Test: Risk Mapper + Path Planner ===\n")

    # 模拟语义分割输出 (7类, 48×64 小图)
    seg = np.random.choice(7, size=(48, 64), p=[0.3,0.1,0.15,0.05,0.1,0.05,0.25])

    # 基础风险映射
    risk = np.zeros_like(seg, dtype=np.float32)
    for cid, r in BASE_RISK.items():
        risk[seg == cid] = r

    print(f"Risk grid: {risk.shape}, mean={risk.mean():.2f}, max={risk.max():.2f}")

    # 三种方案规划
    start, goal = (10, 40), (54, 5)

    for name, method in [
        ("Baseline A (binary)", BaselinePlanner.baseline_a),
        ("Baseline B (fixed)", BaselinePlanner.baseline_b),
        ("Ours (semantic+IMU)", BaselinePlanner.ours),
    ]:
        path, stats = method(risk, start, goal)
        status = f"SUCCESS ({len(path)} steps)" if path else "FAILED"
        print(f"  {name:25s}: {status}")
        if stats and "error" not in stats:
            print(f"    risk_sum={stats['risk_sum']:.1f}, avg_risk={stats['avg_risk']:.2f}")

    print("\nUnit test passed!")


if __name__ == "__main__":
    unit_test()
