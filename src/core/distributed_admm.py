"""
分布式ADMM — Master协调器

真实联邦场景：
  - Master 持有 y 和 Paillier 私钥
  - Edge  持有私有 A_k，本地计算 B_k / alpha_k / Bbar_k

Master 的职责：
  1. 生成 Paillier 密钥对（加密模式）
  2. 维护 ADMM 全局状态变量 z_k / v_k / x_k
  3. 每轮迭代执行全局 z / v 更新（软阈值）
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .admm import soft_threshold
from .crypto import PaillierKeypair


class FederatedADMMCoordinator:
    """
    Master 节点 ADMM 协调器。

    不持有 A_k，只维护 ADMM 状态变量和密钥。

    Args:
        rho:       ADMM 惩罚参数
        lamb:      L1 正则化参数
        max_iter:  最大迭代次数
        tol:       收敛容差（相对残差变化）
        encrypted: 是否使用 Paillier 加密
        key_bits:  Paillier 密钥位数
        delta:     Γ₂ 量化参数（默认 2^12）
        delta2:    Γ₁ 量化参数（默认 2^16）
    """

    def __init__(
        self,
        rho: float = 1.0,
        lamb: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        encrypted: bool = False,
        key_bits: int = 512,
        delta: int = 2**12,
        delta2: int = 2**16,
    ):
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        if lamb < 0:
            raise ValueError(f"lamb must be >= 0, got {lamb}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")

        self.rho = rho
        self.lamb = lamb
        self.max_iter = max_iter
        self.tol = tol
        self.encrypted = encrypted
        self.delta = delta
        self.delta2 = delta2

        self.keypair: Optional[PaillierKeypair] = None
        if encrypted:
            self.keypair = PaillierKeypair(bits=key_bits)

        # Set after setup()
        self.K: int = 0
        self.Ns: List[int] = []
        self.N: int = 0
        self.x_parts: List[np.ndarray] = []
        self.z_parts: List[np.ndarray] = []
        self.v_parts: List[np.ndarray] = []
        # (a_min, a_max) per partition — needed for encrypted decryption
        self.alpha_params: List[tuple] = []

    def setup(self, Ns: List[int]) -> Dict[str, Any]:
        """
        Initialize ADMM state variables given partition sizes.

        Args:
            Ns: list of column counts per edge, e.g. [500, 500] for K=2 edges

        Returns:
            dict with public key info (encrypted mode) or empty dict (plain mode)
        """
        if not Ns:
            raise ValueError("Ns must not be empty")
        self.K = len(Ns)
        self.Ns = list(Ns)
        self.N = sum(Ns)
        self.x_parts = [np.zeros(n) for n in Ns]
        self.z_parts = [np.zeros(n) for n in Ns]
        self.v_parts = [np.zeros(n) for n in Ns]
        self.alpha_params = [(0.0, 0.0)] * self.K

        info: Dict[str, Any] = {}
        if self.encrypted and self.keypair:
            info["n"] = self.keypair.n
            info["g"] = self.keypair.g
            info["delta"] = self.delta
            info["delta2"] = self.delta2
        return info

    def z_v_update(self, x_new_parts: List[np.ndarray]) -> None:
        """
        Global z and v update (soft thresholding).
        Updates self.z_parts, self.v_parts, self.x_parts in-place.
        """
        x_full = np.concatenate(x_new_parts)
        v_full = np.concatenate(self.v_parts)
        z_full = soft_threshold(v_full + x_full, self.lamb / self.rho)
        v_full = v_full + x_full - z_full

        offset = 0
        for k in range(self.K):
            n = self.Ns[k]
            self.z_parts[k] = z_full[offset:offset + n]
            self.v_parts[k] = v_full[offset:offset + n]
            self.x_parts[k] = x_new_parts[k]
            offset += n

    def get_stats(self) -> Dict[str, Any]:
        return {
            "K": self.K,
            "N": self.N,
            "Ns": self.Ns,
            "rho": self.rho,
            "lamb": self.lamb,
            "encrypted": self.encrypted,
        }
