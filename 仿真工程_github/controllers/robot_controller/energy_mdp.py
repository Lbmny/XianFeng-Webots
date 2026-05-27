"""EnergyMDP — 创新点三 模块二
===============================================
MDP自适应救援策略: 5维状态×4动作 Q-Learning
离散化Q表 ~640 entries, 手工编码专家规则

动作:
  0 = FULL_SPEED 全速突击 (最高功耗)
  1 = CRUISE     巡航省电 (中等功耗)
  2 = BEACON     进入信标 (最低功耗)
  3 = HOLD       保持待援 (低功耗原地待命)

状态离散化:
  battery_pct:   5 bins [0-15%, 15-25%, 25-40%, 40-60%, 60-100%]
  terrain_risk:  4 bins [0-0.2, 0.2-0.5, 0.5-0.8, 0.8-1.0]
  dist_to_shore: 4 bins [<50m, 50-150m, 150-300m, >300m]
  water_flow:    4 bins [顺流/静水(<0.1)/弱逆(0.1-0.3)/强逆(>0.3)] m/s
  load:          2 bins [轻载<15kg, 重载>=15kg]
"""

import numpy as np
import json
import os
from itertools import product


class EnergyMDP:
    """MDP Q表决策器 — 手工编码专家规则"""

    N_ACTIONS = 4
    ACTION_NAMES = {0: "FULL_SPEED", 1: "CRUISE", 2: "BEACON", 3: "HOLD"}
    ACTION_SPEED = {0: 10.0, 1: 5.0, 2: 0.0, 3: 0.0}
    ACTION_THRUST = {0: (1.0, 1.0), 1: (0.5, 0.5), 2: (0.0, 0.0), 3: (0.0, 0.0)}

    # 离散化边界
    BATTERY_BINS = [0.0, 0.15, 0.25, 0.40, 0.60, 1.01]  # 5 bins
    RISK_BINS    = [0.0, 0.20, 0.50, 0.80, 1.01]         # 4 bins
    DIST_BINS    = [0.0, 50.0, 150.0, 300.0, 1e6]        # 4 bins
    FLOW_BINS    = [0.0, 0.05, 0.15, 0.30, 1e6]          # 4 bins (magnitude only)
    LOAD_BINS    = [0.0, 15.0, 1e6]                       # 2 bins

    N_BINS = [5, 4, 4, 4, 2]  # 640 total states

    def __init__(self):
        """构建Q表 — 遍历所有状态组合，按专家规则填充"""
        self.q_table = {}
        self._init_q_table()
        self._decision_count = 0
        self._action_stats = {a: 0 for a in range(4)}
        print(f"[EnergyMDP] Q-table ready: {len(self.q_table)} states × {self.N_ACTIONS} actions")

    # ── 离散化 ──

    def _bin(self, value, bins):
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            if lo <= value < hi:
                return i
        return len(bins) - 2

    def _discretize(self, battery_pct, terrain_risk, dist_to_shore,
                    water_flow_mag, load_kg):
        """连续→离散状态元组"""
        bat = self._bin(battery_pct, self.BATTERY_BINS)
        risk = self._bin(terrain_risk, self.RISK_BINS)
        dist = self._bin(dist_to_shore, self.DIST_BINS)
        flow = self._bin(water_flow_mag, self.FLOW_BINS)
        load = self._bin(load_kg, self.LOAD_BINS)
        return (bat, risk, dist, flow, load)

    def _decode_state(self, state):
        """状态元组→人类可读描述"""
        bat_labels = ["极低(<15%)", "低(15-25%)", "中(25-40%)", "较高(40-60%)", "充裕(>60%)"]
        risk_labels = ["安全(<0.2)", "轻度(0.2-0.5)", "中度(0.5-0.8)", "高危(>0.8)"]
        dist_labels = ["很近(<50m)", "近(50-150m)", "中(150-300m)", "远(>300m)"]
        flow_labels = ["静水/顺流", "弱流(<0.1m/s)", "中流(0.1-0.3)", "强逆流(>0.3)"]
        load_labels = ["轻载", "重载"]
        return {
            "battery": bat_labels[state[0]],
            "risk": risk_labels[state[1]],
            "distance": dist_labels[state[2]],
            "flow": flow_labels[state[3]],
            "load": load_labels[state[4]],
        }

    # ── Q表构建 ──

    def _init_q_table(self):
        """遍历640个状态，按专家规则初始化Q值"""
        for state in product(range(5), range(4), range(4), range(4), range(2)):
            q = np.zeros(4)

            bat, risk, dist, flow, load = state
            bat_mid = (self.BATTERY_BINS[bat] + self.BATTERY_BINS[bat + 1]) / 2
            dist_mid = (self.DIST_BINS[dist] + self.DIST_BINS[dist + 1]) / 2
            flow_mid = (self.FLOW_BINS[flow] + self.FLOW_BINS[flow + 1]) / 2
            load_heavy = (load == 1)

            # 估算所需电量: 基础距离能耗 + 水流惩罚 + 负载惩罚
            base_energy_needed = dist_mid * 30.0  # ~30W/m at cruise
            flow_penalty = 1.0 + flow_mid * 8.0  # 逆流大幅增加能耗
            load_penalty = 1.5 if load_heavy else 1.0
            energy_needed_pct = (base_energy_needed * flow_penalty * load_penalty) / 360000.0

            # ── 规则1a: 电量<15% + 距离>300m → 必须BEACON ──
            if bat <= 0 and dist >= 3:
                q[2] = 100  # BEACON
                q[0], q[1], q[3] = -50, -20, 30
            # ── 规则1b: 电量15-25% + 距离>150m → CRUISE优先, BEACON备选 ──
            elif bat <= 1 and dist >= 2:
                q[1] = 90   # CRUISE
                q[2] = 60   # BEACON (backup)
                q[0], q[3] = 30, 20
            # ── 规则2: 电量充裕 + 距离近 → FULL_SPEED ──
            elif bat >= 3 and dist <= 1:
                q[0] = 100  # FULL_SPEED
                q[1], q[2], q[3] = 60, -50, -30
            # ── 规则3: 电量<10% + 距离<50m → 突击（拼一把）──
            elif bat == 0 and dist <= 0:
                q[0] = 90  # FULL_SPEED (close enough)
                q[1], q[2], q[3] = 70, 30, 10
            # ── 规则4: 电量不足支撑返航 → BEACON ──
            elif bat_mid < energy_needed_pct * 1.2:
                q[2] = 100  # BEACON
                q[1], q[0], q[3] = 40, 10, 50
            # ── 规则5: 强逆流 → CRUISE 省电 ──
            elif flow >= 3:
                q[1] = 100  # CRUISE (conservative)
                q[0], q[2], q[3] = 40, -10, 20
            # ── 规则6: 高风险地形 → CRUISE 减速 ──
            elif risk >= 3:
                q[1] = 90  # CRUISE (slow on risky terrain)
                q[0], q[2], q[3] = 30, 10, 0
            # ── 规则7: 电量中等 + 距离中等 → CRUISE 省电 ──
            elif bat <= 2 and dist >= 2:
                q[1] = 100  # CRUISE
                q[0], q[2], q[3] = 50, 10, 30
            # ── 规则8: 重载 + 远距离 → CRUISE 省电 ──
            elif load_heavy and dist >= 2:
                q[1] = 100  # CRUISE
                q[0], q[2], q[3] = 40, -10, 20
            # ── 规则9: 静水/顺流 + 充裕电量 → FULL_SPEED ──
            elif bat >= 3 and flow <= 0:
                q[0] = 100  # FULL_SPEED (favorable)
                q[1], q[2], q[3] = 70, -50, -40
            # ── 规则10: 低电量 + 弱流 + 中等距离 → CRUISE ──
            elif bat <= 1 and dist <= 1 and flow <= 1:
                q[1] = 80  # CRUISE (still feasible)
                q[0], q[2], q[3] = 40, 30, 10
            # ── 规则11: 极低电量 + 高危地形 + 重载 → HOLD+BEACON ──
            elif bat <= 0 and risk >= 3 and load_heavy:
                q[2] = 60  # BEACON
                q[3] = 100  # HOLD (优先原地待援, 不冒险)
                q[0], q[1] = -50, -20
            # ── 默认: 巡航为主 ──
            else:
                q[1] = 80  # CRUISE default
                q[0], q[2], q[3] = 60, 10, 20

            # 添加小随机噪声避免平局 (确定性种子)
            seed = sum(s * (10 ** (4 - i)) for i, s in enumerate(state))
            rng = np.random.RandomState(abs(seed) % 2**31)
            q += rng.uniform(-2, 2, size=4)

            self.q_table[state] = q

    # ── 决策 ──

    def decide(self, battery_pct, terrain_risk, dist_to_shore,
               water_flow=None, load_kg=0.0, energy_pred=None):
        """核心决策方法

        Args:
            battery_pct: 电量百分比 0-1
            terrain_risk: 地形平均风险 0-1
            dist_to_shore: 距最近岸距离 (m)
            water_flow: 水流速度 (vx, vy) tuple 或 float magnitude
            load_kg: 负载重量 (kg)
            energy_pred: 可选，LSTM预测结果 (含 returnable, min_battery_needed_pct)

        Returns:
            dict: action, action_name, q_values, confidence, state, state_desc
        """
        # 水流处理
        if water_flow is None:
            flow_mag = 0.0
        elif isinstance(water_flow, (tuple, list)):
            flow_mag = float(np.linalg.norm(water_flow))
        else:
            flow_mag = float(water_flow)

        state = self._discretize(battery_pct, terrain_risk, dist_to_shore,
                                 flow_mag, load_kg)
        q_values = self.q_table.get(state, np.zeros(4))
        action = int(np.argmax(q_values))

        # LSTM预测修正 (如果可用)
        final_action = action
        override_reason = ""
        if energy_pred is not None:
            if energy_pred.get("returnable") == False and action in [0, 1]:
                final_action = 2  # BEACON
                override_reason = "LSTM预测电量不足以完成返航，覆盖MDP建议"
            elif energy_pred.get("min_battery_needed_pct", 0) < battery_pct * 0.3 and action == 2:
                final_action = 1  # CRUISE (电量绰绰有余, 不触发信标)
                override_reason = "LSTM预测能耗远低于当前电量，取消信标"

        self._decision_count += 1
        self._action_stats[final_action] += 1

        result = {
            "action": final_action,
            "action_name": self.ACTION_NAMES[final_action],
            "q_values": q_values.tolist(),
            "confidence": float(q_values[final_action] / max(1.0, q_values.sum())),
            "state": state,
            "state_desc": self._decode_state(state),
            "mdp_action": self.ACTION_NAMES[action],
            "override": override_reason if override_reason else None,
        }
        return result

# ── 自测 ──

if __name__ == "__main__":
    mdp = EnergyMDP()

    print("\n" + "=" * 60)
    print("  MDP Self-Test — 典型场景决策")
    print("=" * 60)

    tests = [
        (0.55, 0.15, 80,  0.02, 5,  "充裕电量, 近岸, 静水, 轻载"),
        (0.20, 0.40, 200, 0.35, 20, "低电量, 中风险, 远岸, 强逆流, 重载"),
        (0.08, 0.70, 400, 0.25, 30, "极低电量, 高危, 超远, 逆流, 重载"),
        (0.40, 0.10, 30,  0.00, 0,  "中电量, 安全, 极近, 静水, 空载"),
        (0.12, 0.60, 60,  0.08, 10, "低电量, 中高危, 近距离, 弱流"),
        (0.30, 0.30, 120, 0.40, 25, "边界电量, 中风险, 中距离, 强逆流, 重载"),
    ]

    actions = {0: 0, 1: 0, 2: 0, 3: 0}
    for bat, risk, dist, flow, load, desc in tests:
        result = mdp.decide(bat, risk, dist, flow, load)
        actions[result["action"]] += 1
        print(f"\n  [{desc}]")
        print(f"    State: {result['state_desc']}")
        print(f"    Q:     {[f'{q:.0f}' for q in result['q_values']]}")
        print(f"    -> {result['action_name']} (confidence={result['confidence']:.2f})")
        if result['override']:
            print(f"    override: {result['override']}")

    print(f"\n  Decisions: FULL_SPEED={actions[0]} CRUISE={actions[1]} BEACON={actions[2]} HOLD={actions[3]}")
    print("[EnergyMDP] Self-test complete.")
