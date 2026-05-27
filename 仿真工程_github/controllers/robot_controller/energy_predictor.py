"""EnergyPredictor — 创新点三 模块一
===============================================
LSTM时序能耗预测 + 物理模型基准估计（混合策略）

输出:
  - power_pred[60]:     未来60步瞬时功率 (W)
  - uncertainty[60]:    预测标准差 σ (W)
  - energy_pred_j:      60步累计能耗 (J)
  - energy_upper_j:     μ+2σ 悲观估计 (J)
  - returnable:         按当前电池能否完成往返
  - min_battery_needed_pct: 所需最低电量%
"""

import numpy as np
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from train.train_energy_lstm import LSTMEnergyModel

FEATURE_DIM = 12
PRED_LEN = 60
SEQ_LEN = 30
BATTERY_TOTAL_J = 360000.0  # 100Wh

# 地形功耗倍率 (基于摩擦系数反比 + 水域推力)
TERRAIN_FACTOR = {
    0: 1.00, 1: 1.50, 2: 1.25, 3: 1.80, 4: 2.00, 5: 1.60, 6: 1.70,
}
# 水流功耗惩罚 (每0.1m/s逆流增加%)
FLOW_PENALTY_PER_01MS = 0.15
# 负载功耗 (W/kg)
LOAD_W_PER_KG = 2.0


class EnergyPredictor:
    """LSTM+物理混合能耗预测器"""

    def __init__(self, model_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self._loaded = False

        if model_path and os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model = LSTMEnergyModel(
                    input_dim=ckpt.get("input_dim", FEATURE_DIM),
                    hidden_dim=ckpt.get("hidden_dim", 96),
                    pred_len=ckpt.get("pred_len", PRED_LEN),
                ).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.eval()
                self._loaded = True
                print(f"[EnergyPredictor] Model loaded (epoch {ckpt.get('epoch','?')}, "
                      f"val_loss={ckpt.get('val_loss',0):.4f}) on {self.device}")
            except Exception as e:
                print(f"[EnergyPredictor] Model load failed: {e} — using physics-based estimation")
        else:
            print(f"[EnergyPredictor] Model not found — using physics-based estimation")

    def predict(self, terrain_history, flow_history, load_history,
                speed_history, power_history):
        """核心预测 — 物理模型为主，LSTM为辅

        Args:
            terrain_history: list[int] 过去N步地形类别 (0-6)
            flow_history: list[(vx,vy)] 过去N步水流速度
            load_history: list[float] 过去N步负载(kg)
            speed_history: list[float] 过去N步速度(rad/s)
            power_history: list[float] 过去N步实测功率(W)

        Returns:
            dict: 完整预测结果
        """
        n = min(30, len(power_history))
        if n < 3:
            return self._fallback_estimate()

        # ── 物理模型 ──
        recent_power = np.mean(list(power_history)[-10:]) if n >= 10 else np.mean(list(power_history))
        recent_terrain = int(np.median(list(terrain_history)[-10:])) if terrain_history else 0
        recent_flow = list(flow_history)[-5:] if flow_history else [(0, 0)]
        recent_load = np.mean(list(load_history)[-5:]) if load_history else 0.0
        recent_speed = np.mean([abs(s) for s in list(speed_history)[-5:]]) if speed_history else 5.0

        # 水流惩罚
        flow_mags = [np.hypot(f[0], f[1]) if isinstance(f, (tuple, list)) else abs(f)
                      for f in recent_flow]
        avg_flow = np.mean(flow_mags)
        flow_penalty = 1.0 + avg_flow * 10 * FLOW_PENALTY_PER_01MS

        # 地形+负载+水流综合倍率
        terrain_mult = TERRAIN_FACTOR.get(recent_terrain, 1.0)
        load_w = recent_load * LOAD_W_PER_KG
        physics_power = recent_power * terrain_mult * flow_penalty + load_w
        physics_power = max(50.0, physics_power)

        # ── LSTM预测 (如果可用) ──
        lstm_mu = None
        if self._loaded and n >= 15:
            try:
                lstm_mu = self._lstm_predict(
                    terrain_history, flow_history, load_history,
                    speed_history, power_history)
            except Exception:
                pass

        # ── 混合预测 ──
        if lstm_mu is not None and len(lstm_mu) == PRED_LEN:
            # LSTM偏差修正：缩放到物理基准附近
            lstm_avg = np.mean(lstm_mu)
            if lstm_avg > 10:
                scale = physics_power / max(lstm_avg, 1.0)
                scale = np.clip(scale, 0.3, 3.0)  # 限制修正幅度
                mu = lstm_mu * scale
            else:
                mu = np.full(PRED_LEN, physics_power, dtype=np.float32)
            sigma = np.abs(mu) * 0.25  # 25%相对不确定度
        else:
            # 纯物理：功率围绕基准小幅波动
            mu = np.full(PRED_LEN, physics_power, dtype=np.float32)
            sigma = mu * 0.25

        # 累计能耗
        energy_pred_j = float(np.sum(mu))
        energy_upper_j = float(np.sum(mu + 2 * sigma))
        energy_lower_j = float(np.sum(np.maximum(0, mu - 2 * sigma)))
        min_battery_needed_pct = round(energy_upper_j / BATTERY_TOTAL_J * 100, 1)

        return {
            "power_pred": mu.tolist(),
            "uncertainty": sigma.tolist(),
            "energy_pred_j": round(energy_pred_j, 1),
            "energy_upper_j": round(energy_upper_j, 1),
            "energy_lower_j": round(energy_lower_j, 1),
            "avg_power_pred": round(float(np.mean(mu)), 1),
            "min_battery_needed_pct": min_battery_needed_pct,
            "physics_power": round(physics_power, 1),
            "returnable": None,  # 由check_returnable()判断
        }

    def _lstm_predict(self, terrain_history, flow_history, load_history,
                      speed_history, power_history):
        """LSTM模型推理"""
        n = min(SEQ_LEN, len(terrain_history))
        terrain = list(terrain_history)[-n:]
        flow = list(flow_history)[-n:]
        load = list(load_history)[-n:]
        speed = list(speed_history)[-n:]
        power = list(power_history)[-n:]

        features = []
        for i in range(n):
            fvx, fvy = flow[i] if isinstance(flow[i], (tuple, list)) else (flow[i], 0.0)
            feat = self._build_feature(
                terrain[i] if i < len(terrain) else 0,
                fvx, fvy,
                load[i] if i < len(load) else 0.0,
                speed[i] if i < len(speed) else 0.0,
                power[i] if i < len(power) else 80.0,
            )
            features.append(feat)

        while len(features) < SEQ_LEN:
            features.insert(0, np.zeros(FEATURE_DIM, dtype=np.float32))

        x = np.stack(features[-SEQ_LEN:], axis=0)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, _ = self.model(x_tensor)
            return mu.squeeze(0).cpu().numpy()

    def _build_feature(self, terrain_class, flow_vx, flow_vy, load_kg, speed, power_w):
        terrain_onehot = np.zeros(7, dtype=np.float32)
        terrain_onehot[int(terrain_class) % 7] = 1.0
        extra = np.array([
            flow_vx, flow_vy,
            min(load_kg, 35.0) / 30.0,
            min(abs(speed), 10.0) / 10.0,
            min(power_w, 3000.0) / 3000.0,
        ], dtype=np.float32)
        return np.concatenate([terrain_onehot, extra]).astype(np.float32)

    def _fallback_estimate(self):
        mu = np.full(PRED_LEN, 500.0, dtype=np.float32)
        sigma = mu * 0.3
        return {
            "power_pred": mu.tolist(),
            "uncertainty": sigma.tolist(),
            "energy_pred_j": round(500.0 * PRED_LEN, 1),
            "energy_upper_j": round(500.0 * PRED_LEN * 1.5, 1),
            "energy_lower_j": round(500.0 * PRED_LEN * 0.5, 1),
            "avg_power_pred": 500.0,
            "min_battery_needed_pct": round(500.0 * PRED_LEN / BATTERY_TOTAL_J * 100, 1),
            "physics_power": 500.0,
            "returnable": None,
        }

    def check_returnable(self, battery_pct, energy_pred):
        """判断当前电量能否支撑预测能耗返航"""
        needed_j = energy_pred.get("energy_upper_j", 0)
        available_j = battery_pct * BATTERY_TOTAL_J
        energy_pred["returnable"] = available_j > needed_j * 1.1
        return energy_pred["returnable"]


# ── 自测 ──

if __name__ == "__main__":
    print("EnergyPredictor Self-Test (Hybrid: Physics + LSTM)")
    print("=" * 55)

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "models", "energy_lstm_best.pt")
    predictor = EnergyPredictor(model_path)

    tests = [
        ("硬地巡航", [0]*30, [(0,0)]*30, [10]*30, [6]*30, [450]*30),
        ("浅水全推", [3]*30, [(0.1,0.05)]*30, [15]*30, [8]*30, [2000]*30),
        ("硬→水过渡", [0]*15+[3]*15, [(0,0)]*15+[(0.15,0)]*15, [10]*30, [8]*15+[5]*15, [500]*15+[1800]*15),
    ]

    for name, terr, flow, load, speed, power in tests:
        result = predictor.predict(terr, flow, load, speed, power)
        print(f"\n[{name}]")
        print(f"  物理基准功率: {result['physics_power']:.0f}W")
        print(f"  预测平均功率: {result['avg_power_pred']:.0f}W")
        print(f"  60秒累计能耗: {result['energy_pred_j']:.0f}J")
        print(f"  悲观估计(μ+2σ): {result['energy_upper_j']:.0f}J")
        print(f"  需电量%: {result['min_battery_needed_pct']:.1f}%")
        for bat in [0.50, 0.25, 0.10]:
            ok = predictor.check_returnable(bat, result)
            print(f"    电量{bat:.0%}: {'✓可返航' if ok else '✗不足'}")

    print("\n[EnergyPredictor] Self-test complete.")
