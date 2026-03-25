"""
量化方法实现

提供Γ₁和Γ₂量化方案，用于将浮点数转换为整数以支持同态加密运算。

量化方案：
- Γ₁量化：用于α_k，量化范围[0, Δ²]，Δ² = 2^16
- Γ₂量化：用于z, v和B̄_k，量化范围[0, Δ]，Δ = 2^12

量化过程：
1. 归一化到[0, 1]: scaled = (vec - vmin) / (vmax - vmin)
2. 缩放到量化范围: q = round(scaled * delta)
3. 裁剪到有效范围: q = clip(q, 0, delta)

反量化过程：
1. 归一化: scaled = q / delta
2. 恢复原始范围: vec = scaled * (vmax - vmin) + vmin
"""

import numpy as np
from typing import Tuple, Optional, Union


def quantize_gamma1(
    vec: Union[np.ndarray, list, float],
    delta2: int = 2**16,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Γ₁量化方法

    将浮点向量量化到[0, Δ²]范围的整数，用于量化α_k。

    量化公式：
        q = round((vec - vmin) / (vmax - vmin) * Δ²)
        q = clip(q, 0, Δ²)

    Args:
        vec: 输入向量（浮点数）
        delta2: 量化范围Δ²（默认2^16 = 65536）
        vmin: 最小值（如果为None则自动计算）
        vmax: 最大值（如果为None则自动计算）

    Returns:
        (q, vmin, vmax)元组
        - q: 量化后的整数向量
        - vmin: 使用的最小值
        - vmax: 使用的最大值

    Examples:
        >>> vec = np.array([0.1, 0.5, 0.9])
        >>> q, vmin, vmax = quantize_gamma1(vec, delta2=100)
        >>> q
        array([  0,  50, 100])
    """
    vec = np.array(vec, dtype=float)

    # 计算最小值和最大值
    if vmin is None:
        vmin = float(vec.min())
    if vmax is None:
        vmax = float(vec.max())

    # 处理常数向量的情况
    if vmax == vmin:
        return np.zeros_like(vec, dtype=int), vmin, vmax

    # 归一化到[0, 1]
    scaled = (vec - vmin) / (vmax - vmin)

    # 量化到[0, delta2]
    q = np.round(scaled * delta2).astype(int)

    # 裁剪到有效范围
    q = np.clip(q, 0, delta2)

    return q, vmin, vmax


def quantize_gamma2(
    vec: Union[np.ndarray, list, float],
    delta: int = 2**12,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Γ₂量化方法

    将浮点向量量化到[0, Δ]范围的整数，用于量化z, v和B̄_k。

    量化公式：
        q = round((vec - vmin) / (vmax - vmin) * Δ)
        q = clip(q, 0, Δ)

    Args:
        vec: 输入向量（浮点数）
        delta: 量化范围Δ（默认2^12 = 4096）
        vmin: 最小值（如果为None则自动计算）
        vmax: 最大值（如果为None则自动计算）

    Returns:
        (q, vmin, vmax)元组
        - q: 量化后的整数向量
        - vmin: 使用的最小值
        - vmax: 使用的最大值

    Examples:
        >>> vec = np.array([0.0, 0.5, 1.0])
        >>> q, vmin, vmax = quantize_gamma2(vec, delta=100)
        >>> q
        array([  0,  50, 100])
    """
    vec = np.array(vec, dtype=float)

    # 计算最小值和最大值
    if vmin is None:
        vmin = float(vec.min())
    if vmax is None:
        vmax = float(vec.max())

    # 处理常数向量的情况
    if vmax == vmin:
        return np.zeros_like(vec, dtype=int), vmin, vmax

    # 归一化到[0, 1]
    scaled = (vec - vmin) / (vmax - vmin)

    # 量化到[0, delta]
    q = np.round(scaled * delta).astype(int)

    # 裁剪到有效范围
    q = np.clip(q, 0, delta)

    return q, vmin, vmax


def dequantize(
    q: Union[np.ndarray, list, int],
    vmin: float,
    vmax: float,
    delta: int,
) -> np.ndarray:
    """
    反量化方法

    将量化的整数向量恢复为浮点数。

    反量化公式：
        scaled = q / delta
        vec = scaled * (vmax - vmin) + vmin

    Args:
        q: 量化后的整数向量
        vmin: 量化时使用的最小值
        vmax: 量化时使用的最大值
        delta: 量化范围（Δ或Δ²）

    Returns:
        反量化后的浮点向量

    Examples:
        >>> q = np.array([0, 50, 100])
        >>> vec = dequantize(q, vmin=0.1, vmax=0.9, delta=100)
        >>> vec
        array([0.1, 0.5, 0.9])
    """
    q = np.array(q, dtype=float)

    # 处理常数情况
    if vmax == vmin:
        return np.full_like(q, vmin, dtype=float)

    # 归一化到[0, 1]
    scaled = q / delta

    # 恢复到原始范围
    vec = scaled * (vmax - vmin) + vmin

    return vec


def quantization_error(
    vec_original: np.ndarray,
    vec_quantized: np.ndarray,
) -> float:
    """
    计算量化误差

    使用均方误差(MSE)衡量量化前后的差异。

    Args:
        vec_original: 原始浮点向量
        vec_quantized: 量化后恢复的浮点向量

    Returns:
        均方误差

    Examples:
        >>> original = np.array([0.1, 0.5, 0.9])
        >>> q, vmin, vmax = quantize_gamma2(original, delta=100)
        >>> recovered = dequantize(q, vmin, vmax, delta=100)
        >>> error = quantization_error(original, recovered)
        >>> error < 1e-3  # 误差应该很小
        True
    """
    return float(np.mean((vec_original - vec_quantized) ** 2))


def get_quantization_params(delta_type: str = "gamma2") -> int:
    """
    获取标准量化参数

    Args:
        delta_type: 量化类型，"gamma1"或"gamma2"

    Returns:
        对应的delta值

    Raises:
        ValueError: 如果delta_type不是"gamma1"或"gamma2"

    Examples:
        >>> get_quantization_params("gamma1")
        65536
        >>> get_quantization_params("gamma2")
        4096
    """
    if delta_type == "gamma1":
        return 2**16  # Δ² = 65536
    elif delta_type == "gamma2":
        return 2**12  # Δ = 4096
    else:
        raise ValueError(f"未知的量化类型：{delta_type}，应该是'gamma1'或'gamma2'")
