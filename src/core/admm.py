"""
ADMM 工具函数

软阈值算子（L1 正则化的近端算子），供分布式 ADMM 协调器使用。
"""

import numpy as np


def soft_threshold(u: np.ndarray, kappa: float) -> np.ndarray:
    """
    软阈值算子：S_κ(u) = sign(u) · max(|u| - κ, 0)

    Args:
        u:     输入向量
        kappa: 阈值参数（必须 >= 0）
    """
    if kappa < 0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    return np.sign(u) * np.maximum(np.abs(u) - kappa, 0.0)
