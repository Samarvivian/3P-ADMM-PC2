import numpy as np
from numpy.linalg import inv
import sys
import pickle
import subprocess
import os
sys.path.append('/mnt/3p-admm-pc2')
from crypto.paillier import encrypt, decrypt, homo_add, homo_mul_const

def quantize2(v, delta, zmin, zmax):
    return np.floor(delta * (np.array(v) - zmin) / (zmax - zmin)).astype(object)

def run(node_id, A_block, pub, priv, alpha_hat, B_bar_q, delta, zmin, zmax, zk, vk):
    """
    边缘节点计算一轮迭代
    返回加密后的 x_hat_k
    """
    Nk = len(zk)
    Bk = inv(A_block.T @ A_block + 1.0 * np.eye(Nk))
    Bk_diag = np.diag(Bk)

    q_z = quantize2(zk,  delta, zmin, zmax)
    q_v = quantize2(-vk, delta, zmin, zmax)

    c_z = [encrypt(int(qi), pub) for qi in q_z]
    c_v = [encrypt(int(qi), pub) for qi in q_v]

    x_hat_k = []
    for i in range(Nk):
        c_sum   = homo_add(c_z[i], c_v[i], pub)
        c_mul   = homo_mul_const(c_sum, int(B_bar_q[i]), pub)
        c_final = homo_add(alpha_hat[i], c_mul, pub)
        x_hat_k.append(c_final)

    return x_hat_k
