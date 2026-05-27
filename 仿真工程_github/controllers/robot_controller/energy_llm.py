"""EnergyLLMDecider — 创新点三 模块三
===============================================
DeepSeek双层级推理: R1实时决策 + V4深度分析
R1: 本地Ollama <1s, 快速判断"全速/巡航/信标/待援"
V4: 云端API ~3s, 能耗曲线推演 + 多因素量化 + 风险提示

复用 deepseek_planner.py 的API调用模式
"""

import os, json
import requests
import numpy as np

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/v1/chat/completions"


class EnergyLLMDecider:
    """双层级LLM能源决策器"""

    def __init__(self, r1_url="http://127.0.0.1:11434/api/generate",
                 r1_enabled=True, v4_enabled=True):
        self.r1_url = r1_url
        self.r1_enabled = r1_enabled
        self.v4_enabled = v4_enabled
        self.history = []
        print(f"[EnergyLLM] R1={'ON' if r1_enabled else 'OFF'}, "
              f"V4={'ON' if v4_enabled else 'OFF'}")

    # ========== R1 快速决策 ==========

    def r1_quick_decide(self, battery_pct, mdp_result, energy_pred,
                        dist_to_shore, water_flow, terrain_seq, load_kg):
        """DeepSeek-R1 7B 本地实时决策 (<1秒)

        Args:
            battery_pct: 电量百分比 0-1
            mdp_result: MDP.decide() 返回的字典
            energy_pred: EnergyPredictor.predict() 返回的字典
            dist_to_shore: 距岸距离 (m)
            water_flow: (vx, vy) 水流速度
            terrain_seq: 地形序列摘要字符串
            load_kg: 负载 (kg)
        """
        if not self.r1_enabled:
            return {"decision": "MDP_FALLBACK", "action": mdp_result["action_name"],
                    "reasoning": "R1 offline — using MDP"}

        flow_mag = np.hypot(water_flow[0], water_flow[1]) if isinstance(water_flow, (tuple, list)) else 0.0
        flow_desc = "静水" if flow_mag < 0.05 else f"逆流{flow_mag:.2f}m/s"

        avail_j = battery_pct * 360000.0
        needed_j = energy_pred.get("energy_upper_j", 0)
        surplus_pct = (avail_j - needed_j) / 3600.0 if needed_j > 0 else 99

        prompt = f"""你是救灾机器人能源决策系统。根据传感器数据快速决定行动。

当前状态:
- 电量: {battery_pct*100:.0f}% ({avail_j:.0f}J)
- 距岸: {dist_to_shore:.0f}m
- 水流: {flow_desc}
- 负载: {load_kg:.0f}kg
- 前方地形: {terrain_seq}

能耗预测:
- 60秒预测能耗: {needed_j:.0f}J (悲观估计)
- 返航余量: {surplus_pct:.0f}%

MDP建议: {mdp_result['action_name']} (置信度{mdp_result['confidence']:.2f})

决策选项: 全速 / 巡航 / 信标 / 待援
请分析数据后给出最终决策。最后一行只输出决策词。"""

        try:
            resp = requests.post(self.r1_url, json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 256}
            }, timeout=60)
            response_text = resp.json().get("response", "").strip()
        except Exception as e:
            response_text = f"[R1 Error] {e}"

        # 解析决策
        action = self._parse_action(response_text, mdp_result["action_name"])

        result = {
            "decision": "R1",
            "action": action,
            "reasoning": response_text[:150],
            "source": "R1_LOCAL",
        }
        self.history.append(result)
        return result

    # ========== V4 深度分析 ==========

    def v4_deep_analyze(self, battery_pct, mdp_result, energy_pred,
                        dist_to_shore, water_flow, terrain_seq, load_kg,
                        r1_result=None):
        """DeepSeek V4 云端深度分析 (~3秒)

        包含: 量化对比 + 逆流修正 + 安全余量 + 风险提示
        """
        if not self.v4_enabled:
            return {"analysis": "V4 offline", "recommendation": mdp_result["action_name"]}

        flow_mag = np.hypot(water_flow[0], water_flow[1]) if isinstance(water_flow, (tuple, list)) else 0.0
        avail_j = battery_pct * 360000.0
        needed_j = energy_pred.get("energy_upper_j", 0)
        surplus_j = avail_j - needed_j * 1.1

        # LSTM预测数据摘要
        power_pred = energy_pred.get("power_pred", [])
        pred_summary = ""
        if power_pred and len(power_pred) >= 10:
            p10 = np.mean(power_pred[:10])
            p60 = np.mean(power_pred)
            trend = "上升" if p60 > p10 * 1.05 else ("下降" if p60 < p10 * 0.95 else "平稳")
            pred_summary = f"功率趋势: 前10秒均值{p10:.0f}W, 60秒均值{p60:.0f}W, 趋势{trend}"

        r1_info = ""
        if r1_result:
            r1_info = f"\nR1本地快速决策: {r1_result.get('action', '?')}\n理由: {r1_result.get('reasoning', '')[:100]}"

        prompt = f"""你是救灾机器人能源决策专家（深度分析层）。请基于以下数据做量化分析：

[状态数据]
- 电量: {battery_pct*100:.1f}% (可用{avail_j:.0f}J)
- 距岸距离: {dist_to_shore:.0f}m
- 水流速度: {flow_mag:.2f}m/s
- 负载: {load_kg:.0f}kg
- 地形序列: {terrain_seq}

[能耗预测]
- 60秒预测能耗: {needed_j:.0f}J (μ+2σ悲观估计)
- 安全返航余量: {surplus_j:.0f}J ({'+' if surplus_j > 0 else ''}{surplus_j/3600:.0f}%)
- {pred_summary}

[MDP Q值]
- 全速: {mdp_result['q_values'][0]:.0f}
- 巡航: {mdp_result['q_values'][1]:.0f}
- 信标: {mdp_result['q_values'][2]:.0f}
- 待援: {mdp_result['q_values'][3]:.0f}
- MDP建议: {mdp_result['action_name']}{r1_info}

请输出:
1) 全速vs巡航vs信标各方案能耗量化对比
2) 水流逆流修正计算
3) 安全余量评估与风险提示
4) 最终建议 (全速/巡航/信标/待援 四选一)

中文，200字内。"""

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 400,
        }
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

        print("[EnergyLLM] V4 analyzing...")
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                analysis = resp.json()["choices"][0]["message"]["content"]
            else:
                analysis = f"[V4 HTTP {resp.status_code}] {resp.text[:200]}"
        except Exception as e:
            analysis = f"[V4 Error] {e}"

        recommendation = self._parse_action(analysis, mdp_result["action_name"])

        result = {
            "analysis": analysis,
            "recommendation": recommendation,
            "source": "V4_CLOUD",
        }
        self.history.append(result)
        return result

    # ========== 主入口 ==========

    def decide(self, battery_pct, mdp_result, energy_pred,
               dist_to_shore, water_flow, terrain_seq, load_kg,
               force_v4=False):
        """双层级LLM决策主入口

        决策优先级:
          正常: R1快速→V4深度(级联)
          仅R1: R1单独
          仅V4: V4单独
          全离线: MDP兜底

        Returns:
            dict: final_action, r1_result, v4_result, decision_source
        """
        r1_result = None
        v4_result = None
        final_action = mdp_result["action_name"]

        if self.r1_enabled:
            r1_result = self.r1_quick_decide(
                battery_pct, mdp_result, energy_pred,
                dist_to_shore, water_flow, terrain_seq, load_kg)
            final_action = r1_result["action"]

        if self.v4_enabled and (force_v4 or self.r1_enabled):
            v4_result = self.v4_deep_analyze(
                battery_pct, mdp_result, energy_pred,
                dist_to_shore, water_flow, terrain_seq, load_kg, r1_result)
            # V4修正R1决策
            if v4_result.get("recommendation"):
                final_action = v4_result["recommendation"]

        if self.r1_enabled and self.v4_enabled:
            source = "R1+V4_CASCADE"
        elif self.r1_enabled:
            source = "R1_ONLY"
        elif self.v4_enabled:
            source = "V4_ONLY"
        else:
            source = "MDP_FALLBACK"

        return {
            "final_action": final_action,
            "r1_result": r1_result,
            "v4_result": v4_result,
            "decision_source": source,
        }

    # ========== 输出解析 ==========

    def _parse_action(self, text, default):
        """从LLM输出文本中解析动作"""
        if not text:
            return default
        text_lower = text.lower()
        if "全速" in text or "full" in text_lower:
            return "FULL_SPEED"
        elif "巡航" in text or "cruise" in text_lower:
            return "CRUISE"
        elif "信标" in text or "beacon" in text_lower:
            return "BEACON"
        elif "待援" in text or "hold" in text_lower:
            return "HOLD"
        return default

    def save_history(self, path):
        """保存决策历史到JSON"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{
                "source": h.get("source", h.get("decision", "")),
                "action": h.get("action", h.get("recommendation", "")),
                "reasoning": h.get("reasoning", h.get("analysis", ""))[:300],
            } for h in self.history], f, indent=2, ensure_ascii=False)


# ── 自测 ──

if __name__ == "__main__":
    print("EnergyLLMDecider Self-Test")
    print("=" * 50)

    # 不实际调用API，仅测试本地逻辑
    decider = EnergyLLMDecider(r1_enabled=True, v4_enabled=True)

    # 模拟输入
    mdp_result = {
        "action_name": "CRUISE",
        "q_values": [50.0, 100.0, 10.0, 20.0],
        "confidence": 0.55,
    }
    energy_pred = {
        "energy_upper_j": 45000,
        "energy_pred_j": 35000,
        "avg_power_pred": 650,
        "power_pred": [600] * 60,
    }

    # 测试解析逻辑
    tests = [
        ("建议全速返航，电量充足", "FULL_SPEED"),
        ("建议巡航模式省电", "CRUISE"),
        ("电量不足，建议进入信标模式等待救援", "BEACON"),
        ("原地待援，不要冒险", "HOLD"),
        ("综合建议：CRUISE", "CRUISE"),
        ("", "CRUISE"),  # 默认
    ]
    for text, expected in tests:
        result = decider._parse_action(text, "CRUISE")
        status = "OK" if result == expected else f"FAIL (got {result})"
        print(f"  [{status}] '{text[:40]}' -> {result}")

    print("\n[EnergyLLM] Self-test complete.")
    print("  Note: R1/V4 API calls skipped — test in Webots with 'L' key")
