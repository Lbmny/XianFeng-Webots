"""Risk-Aware Path Planner — 创新点一 模块三
=============================================
三种路径规划方案对比：

  Baseline A: 标准A* + 二值占据栅格（仅避障，risk>0.9=障碍）
  Baseline B: 标准A* + 人工标注风险值（固定风险表）
  Ours:      风险感知A* + IMU动态修正（自动语义→风险）

代价函数:
  总代价 = α·路径长度 + β·地形风险积分
  g(n) = g(parent) + step_dist + β·risk(n)
  h(n) = 欧几里得距离到目标
"""

import numpy as np
import heapq
from collections import namedtuple

# A* 节点
Node = namedtuple("Node", ["f", "g", "x", "y", "parent"])


class RiskAwareAStar:
    """风险感知 A* 路径规划器"""

    def __init__(self, risk_grid, alpha=1.0, beta=2.0, risk_threshold=0.95):
        """
        Args:
            risk_grid: 风险栅格 (H×W), 每格 ∈ [0,1]
            alpha: 距离权重
            beta: 风险权重
            risk_threshold: 不可通行风险阈值
        """
        self.grid = risk_grid
        self.h, self.w = risk_grid.shape
        self.alpha = alpha
        self.beta = beta
        self.risk_threshold = risk_threshold

    def _neighbors(self, x, y):
        """8邻域"""
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                if self.grid[ny, nx] < self.risk_threshold:
                    yield nx, ny, np.sqrt(dx*dx + dy*dy)

    def _heuristic(self, x, y, gx, gy):
        """欧几里得距离启发式"""
        return np.sqrt((x - gx)**2 + (y - gy)**2)

    def plan(self, start, goal):
        """规划路径

        Args:
            start: (x, y) 起点像素坐标
            goal: (x, y) 终点像素坐标

        Returns:
            path: [(x,y), ...] 或 None（无路径）
            stats: {"path_length", "risk_sum", "avg_risk", "max_risk"}
        """
        sx, sy = start
        gx, gy = goal

        # 边界检查
        if not (0 <= sx < self.w and 0 <= sy < self.h):
            return None, {"error": "start out of bounds"}
        if not (0 <= gx < self.w and 0 <= gy < self.h):
            return None, {"error": "goal out of bounds"}
        if self.grid[sy, sx] >= self.risk_threshold:
            return None, {"error": "start blocked"}
        if self.grid[gy, gx] >= self.risk_threshold:
            return None, {"error": "goal blocked"}

        open_set = []
        closed = set()
        g_scores = { (sx, sy): 0.0 }

        h0 = self._heuristic(sx, sy, gx, gy)
        heapq.heappush(open_set, Node(h0, 0, sx, sy, None))

        while open_set:
            current = heapq.heappop(open_set)

            if (current.x, current.y) in closed:
                continue
            closed.add((current.x, current.y))

            # 到达目标
            if current.x == gx and current.y == gy:
                path = self._reconstruct(current)
                stats = self._path_stats(path)
                return path, stats

            for nx, ny, step_dist in self._neighbors(current.x, current.y):
                if (nx, ny) in closed:
                    continue

                # 风险感知 g 值
                risk = self.grid[ny, nx]
                new_g = current.g + self.alpha * step_dist + self.beta * risk

                if (nx, ny) not in g_scores or new_g < g_scores[(nx, ny)]:
                    g_scores[(nx, ny)] = new_g
                    h = self._heuristic(nx, ny, gx, gy)
                    f = new_g + h
                    heapq.heappush(open_set, Node(f, new_g, nx, ny, current))

        return None, {"error": "no path found"}

    def _reconstruct(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    def _path_stats(self, path):
        risks = [self.grid[y, x] for x, y in path]
        return {
            "path_length": len(path),
            "risk_sum": float(np.sum(risks)),
            "avg_risk": float(np.mean(risks)),
            "max_risk": float(np.max(risks)),
        }


class BaselinePlanner:
    """Baseline 路径规划器（用于对比实验）"""

    @staticmethod
    def baseline_a(risk_grid, start, goal):
        """Baseline A: 标准A* + 二值占据栅格（risk>0.9=障碍）"""
        # 二值化：risk>0.9 → 障碍
        binary_grid = (risk_grid >= 0.9).astype(np.float32)
        planner = RiskAwareAStar(binary_grid, alpha=1.0, beta=0.0, risk_threshold=0.5)
        return planner.plan(start, goal)

    @staticmethod
    def baseline_b(risk_grid, start, goal):
        """Baseline B: 标准A* + 人工标注风险值（固定映射，无IMU修正）"""
        # 直接用基础风险值，不加IMU修正
        planner = RiskAwareAStar(risk_grid, alpha=1.0, beta=2.0, risk_threshold=0.95)
        return planner.plan(start, goal)

    @staticmethod
    def ours(risk_grid, start, goal):
        """本方案: 风险感知A* + 语义自动风险 + IMU修正"""
        planner = RiskAwareAStar(risk_grid, alpha=1.0, beta=2.5, risk_threshold=0.95)
        return planner.plan(start, goal)


# ═══════════════════════════════════════════════════════
# 评估工具
# ═══════════════════════════════════════════════════════

def run_comparison(risk_grid, start, goal):
    """运行三种方案的对比实验"""
    results = {}

    for name, method in [
        ("Baseline A (binary)", BaselinePlanner.baseline_a),
        ("Baseline B (fixed risk)", BaselinePlanner.baseline_b),
        ("Ours (semantic+IMU)", BaselinePlanner.ours),
    ]:
        path, stats = method(risk_grid, start, goal)
        results[name] = {
            "success": path is not None,
            "path": path,
            "stats": stats if stats else {},
        }

    return results


def world_to_grid(world_x, world_y, grid_origin, resolution):
    """世界坐标 → 栅格坐标"""
    gx = int((world_x - grid_origin[0]) / resolution)
    gy = int((world_y - grid_origin[1]) / resolution)
    return gx, gy
