"""DeepSeek V4 路径分析 — 创新点一 模块四
==============================================
将语义分割统计 + 风险地图数据发送给 DeepSeek V4，
获取自然语言路径分析与风险推理链。

注：DeepSeek API 暂不支持多模态图片输入（2026.05）。
    因此发送详细地形统计报告代替，推理效果等价。
"""

import json
import os
import numpy as np
import requests

API_KEY = "sk-42d877250eac44cc8634fed801d8f773"
API_URL = "https://api.deepseek.com/v1/chat/completions"

CLASS_NAMES_CN = ["硬质路面", "泥地", "碎石", "浅水", "深水", "陡坡", "废墟"]
RISK_VALUES = [0.10, 0.50, 0.30, 0.40, 0.70, 0.90, 1.00]


class DeepSeekV4Planner:
    """DeepSeek V4 路径分析器"""

    def __init__(self):
        self.history = []
        print("[DeepSeek] V4 client initialized")

    def analyze(self, seg: np.ndarray, risk_summary: dict,
                position: tuple = None) -> dict:
        h, w = seg.shape
        total = h * w

        # 各类别占比
        class_lines = []
        for cid in range(7):
            pct = (seg == cid).sum() / total * 100
            if pct > 0.5:
                class_lines.append(
                    f"  {CLASS_NAMES_CN[cid]}: {pct:.1f}% (风险值{RISK_VALUES[cid]})")

        # 三区分析
        zones = [("上方远处", seg[:h//3, :]),
                 ("中部", seg[h//3:2*h//3, :]),
                 ("下方近处", seg[2*h//3:, :])]
        zone_lines = []
        for zname, zseg in zones:
            counts = np.bincount(zseg.flatten(), minlength=7)
            top = int(np.argmax(counts))
            zrisk = sum(RISK_VALUES[c] * counts[c] for c in range(7)) / max(counts.sum(), 1)
            zone_lines.append(f"  {zname}: 主{CLASS_NAMES_CN[top]}, 局部风险={zrisk:.2f}")

        pos_str = f"GPS({position[0]:.1f},{position[1]:.1f})" if position else "GPS未知"

        report = f"""[AI语义感知报告] 位置:{pos_str} 视野:{w}×{h}
地形分布:
{chr(10).join(class_lines)}
分区:
{chr(10).join(zone_lines)}
全局: 均风险{risk_summary.get('mean_risk',0):.2f} 高风险区{risk_summary.get('high_risk_pct',0):.0f}% 不可通行{risk_summary.get('impassable_pct',0):.0f}%"""

        prompt = f"""你是救灾机器人路径规划专家。AI语义分割模块感知到前方地形：

{report}

风险参考: 硬质路面0.10, 泥地0.50, 碎石0.30, 浅水0.40, 深水0.70, 陡坡0.90, 废墟1.00

请分析：1)危险区域识别 2)推荐最优安全路线 3)各段风险评估 4)整体安全评分(1-10)
中文，200字内。"""

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
        }
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

        print("[DeepSeek] Analyzing terrain data...")
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                analysis = resp.json()["choices"][0]["message"]["content"]
            else:
                analysis = f"[V4 HTTP {resp.status_code}] {resp.text[:200]}"
        except Exception as e:
            analysis = f"[V4 Error] {e}"

        result = {"analysis": analysis, "report": report}
        self.history.append(result)
        return result

    def analyze_with_r1(self, seg: np.ndarray, risk_summary: dict,
                         position: tuple, r1_decision: str = None) -> dict:
        """级联分析：接收 R1 初判，V4 做最终决策"""
        h, w = seg.shape
        total = h * w

        class_lines = []
        for cid in range(7):
            pct = (seg == cid).sum() / total * 100
            if pct > 0.5:
                class_lines.append(
                    f"  {CLASS_NAMES_CN[cid]}: {pct:.1f}% (风险{RISK_VALUES[cid]})")

        zones = [("上方远处", seg[:h//3, :]),
                 ("中部", seg[h//3:2*h//3, :]),
                 ("下方近处", seg[2*h//3:, :])]
        zone_lines = []
        for zname, zseg in zones:
            counts = np.bincount(zseg.flatten(), minlength=7)
            top = int(np.argmax(counts))
            zrisk = sum(RISK_VALUES[c] * counts[c] for c in range(7)) / max(counts.sum(), 1)
            zone_lines.append(f"  {zname}: 主{CLASS_NAMES_CN[top]}, 风险={zrisk:.2f}")

        pos_str = f"GPS({position[0]:.1f},{position[1]:.1f})" if position else ""

        r1_info = f"\nR1本地快速决策建议: {r1_decision}\n请评估R1的建议是否合理，如不合理请给出修正。" if r1_decision else ""

        prompt = f"""你是救灾机器人路径规划专家（最终决策层）。请综合分析：

[AI感知报告] {pos_str}
地形: {chr(10).join(class_lines)}
分区: {chr(10).join(zone_lines)}
全局: 均风险{risk_summary.get('mean_risk',0):.2f} 高风险区{risk_summary.get('high_risk_pct',0):.0f}%
{r1_info}
风险参考: 硬质路面0.1, 泥地0.5, 碎石0.3, 浅水0.4, 深水0.7, 陡坡0.9, 废墟1.0

请输出: 1)评估R1建议 2)最终决策(直行/左绕/右绕/停车) 3)理由。中文100字内。"""

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": 300,
        }
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

        print("[DeepSeek] V4 analyzing with R1 input...")
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                analysis = resp.json()["choices"][0]["message"]["content"]
            else:
                analysis = f"[V4 HTTP {resp.status_code}] {resp.text[:200]}"
        except Exception as e:
            analysis = f"[V4 Error] {e}"

        result = {"analysis": analysis, "r1_input": r1_decision}
        self.history.append(result)
        return result

    def save_history(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"analysis": h["analysis"], "report": h.get("report", "")}
                       for h in self.history], f, indent=2, ensure_ascii=False)
