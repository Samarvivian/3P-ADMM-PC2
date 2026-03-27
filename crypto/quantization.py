import numpy as np

def gamma1(v, delta):
    """
    量化函数Γ1，用于加密 Bk(Ak^T y)
    映射到 [0, delta^2/(max-min)] 的整数
    对应论文公式(14a)
    """
    v = np.array(v, dtype=np.float64)
    v_min = v.min()
    v_max = v.max()

    if v_max == v_min:
        return np.zeros(len(v), dtype=object), v_min, v_max

    quantized = np.floor(
        delta**2 * (v - v_min) / (v_max - v_min)**2
    ).astype(object)

    return quantized, v_min, v_max


def gamma2(v, delta):
    """
    量化函数Γ2，用于加密 z, -v, Bk*rho
    映射到 [0, delta] 的整数
    对应论文公式(14b)(14c)(14d)
    """
    v = np.array(v, dtype=np.float64)
    v_min = v.min()
    v_max = v.max()

    if v_max == v_min:
        return np.zeros(len(v), dtype=object), v_min, v_max

    quantized = np.floor(
        delta * (v - v_min) / (v_max - v_min)
    ).astype(object)

    return quantized, v_min, v_max


def inv_quantize(q, z_min, z_max, delta, B=None, z_prev=None, v_prev=None):
    """
    逆量化，对应论文公式(30)
    还原 x_k 的近似值
    """
    q = np.array(q, dtype=np.float64)
    scale = (z_max - z_min)**2 / delta**2
    result = q * scale

    if B is not None and z_prev is not None and v_prev is not None:
        correction = (2 * B @ np.ones(len(z_prev)) +
                     (z_prev - v_prev) + 1) * z_min - 2 * z_min**2
        result += correction

    return result
