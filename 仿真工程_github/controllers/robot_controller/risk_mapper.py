"""Risk Cost Map Generator — 创新点一 模块二
==============================================
多模态风险代价地图：
  RGB → MobileNetV3分割(7类) → 基础风险值
  IMU jolt → 动态修正（颠簸/打滑→风险↑）
  Depth   → 障碍检测（近距离→风险↑）

输出：概率风险栅格地图 grid[x][y] ∈ [0,1]
  0.0 = 完全安全
  1.0 = 不可通行
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# 7类地形 → 基础风险映射
BASE_RISK = {
    0: 0.10,  # hard 硬质路面 — 安全
    1: 0.50,  # mud 泥地 — 中等风险
    2: 0.30,  # gravel 碎石 — 较低风险
    3: 0.40,  # shallow 浅水 — 可涉水
    4: 0.70,  # deep 深水 — 高风险
    5: 0.90,  # slope 陡坡 — 极高风险
    6: 1.00,  # rubble 废墟 — 不可通行
}

CLASS_NAMES = ["hard", "mud", "gravel", "shallow", "deep", "slope", "rubble"]


class RiskMapper:
    """多模态风险代价地图生成器"""

    def __init__(self, model_path="models/terrain_seg_best.pt", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 加载语义分割模型
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        from train.train_seg import build_model

        self.seg_model = build_model(num_classes=7, pretrained=True).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.seg_model.load_state_dict(ckpt["model_state_dict"])
        self.seg_model.eval()

        self.transform = T.Compose([
            T.Resize((480, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 风险参数
        self.imu_jolt_threshold = 0.15   # 颠簸超过此值视为打滑
        self.imu_risk_multiplier = 1.6    # 打滑时风险倍率
        self.obstacle_depth_threshold = 0.5  # 深度<0.5m 视为障碍
        self.grid_resolution = 0.5        # 栅格分辨率 (m/cell)

        print(f"[RiskMapper] Model loaded on {self.device}, grid res={self.grid_resolution}m")

    def segment(self, rgb_image: np.ndarray) -> np.ndarray:
        """RGB图像 → 7类语义分割图 (H×W)"""
        img = Image.fromarray(rgb_image)
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.seg_model(x)["out"]
            seg = out.argmax(dim=1).squeeze(0).cpu().numpy()
        return seg

    def base_risk_map(self, seg: np.ndarray) -> np.ndarray:
        """语义分割图 → 基础风险栅格"""
        risk = np.zeros_like(seg, dtype=np.float32)
        for class_id, base_risk in BASE_RISK.items():
            risk[seg == class_id] = base_risk
        return risk

    def apply_imu_correction(self, risk: np.ndarray, seg: np.ndarray, jolt: float) -> np.ndarray:
        """IMU颠簸反馈 → 动态调整泥地/碎石风险"""
        if jolt < self.imu_jolt_threshold:
            return risk

        corrected = risk.copy()
        # 泥地(1)和碎石(2)在颠簸时风险升高
        for terrain_id in [1, 2]:
            mask = (seg == terrain_id)
            corrected[mask] = np.clip(corrected[mask] * self.imu_risk_multiplier, 0.0, 1.0)

        return corrected

    def apply_depth_obstacle(self, risk: np.ndarray, depth: float) -> np.ndarray:
        """深度传感器 → 近距障碍标记为不可通行"""
        if depth < self.obstacle_depth_threshold:
            # 前方有障碍，视野下半部分标记高风险
            h, w = risk.shape
            obstacle_zone = risk[int(h * 0.6):, :]
            obstacle_zone[:] = np.maximum(obstacle_zone, 0.9)
        return risk

    def compute(self, rgb: np.ndarray, jolt: float = 0.0, depth: float = 10.0) -> np.ndarray:
        """完整多模态风险地图计算

        Args:
            rgb: RGB图像 (H×W×3)
            jolt: IMU颠簸强度
            depth: DistanceSensor测距值 (m)

        Returns:
            风险栅格 (H×W), 每格 ∈ [0,1]
        """
        seg = self.segment(rgb)
        risk = self.base_risk_map(seg)
        risk = self.apply_imu_correction(risk, seg, jolt)
        risk = self.apply_depth_obstacle(risk, depth)
        return risk

    def risk_summary(self, risk: np.ndarray) -> dict:
        """风险地图统计摘要"""
        return {
            "mean_risk": float(np.mean(risk)),
            "max_risk": float(np.max(risk)),
            "high_risk_pct": float(np.mean(risk > 0.7) * 100),  # 高风险区占比
            "impassable_pct": float(np.mean(risk > 0.95) * 100),  # 不可通行区占比
        }
