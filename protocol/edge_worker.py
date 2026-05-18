import sys
import pickle
import numpy as np
sys.path.append('/mnt/3p-admm-pc2')
from crypto.paillier import encrypt, homo_add, homo_mul_const

def quantize2_safe(v, delta, zmin, zmax):
    v = np.array(v, dtype=np.float64)
    v = np.clip(v, zmin, zmax)
    result = np.floor(delta * (v - zmin) / (zmax - zmin)).astype(object)
    result = np.clip(result, 0, int(delta))
    return result

k = int(sys.argv[1])
with open(f'/mnt/edge_data_k{k}.pkl', 'rb') as f:
    data = pickle.load(f)

pub       = data['pub']
alpha_hat = data['alpha_hat']
B_k       = data['B_k']
rho       = data['rho']
delta     = data['delta']
ZMIN      = data['ZMIN']
ZMAX      = data['ZMAX']

with open(f'/mnt/iter_data_k{k}.pkl', 'rb') as f:
    iter_data = pickle.load(f)

# 直接使用主节点发来的密文（论文正确实现）
c_z = iter_data['zk_hat']
c_v = iter_data['vk_hat']
Nk = len(c_z)

# 计算 c_zv[j] = encrypt(q_z[j] + q_v[j])
c_zv = [homo_add(c_z[j], c_v[j], pub) for j in range(Nk)]

is_full_matrix = (B_k.ndim == 2)
x_hat_k = []

if is_full_matrix:
    # 完整矩阵同态乘法（小规模精确版）
    B_bar_k = quantize2_safe(B_k.flatten(), delta, ZMIN, ZMAX).reshape(Nk, Nk)
    for i in range(Nk):
        acc = None
        for j in range(Nk):
            b_ij = int(B_bar_k[i, j])
            if b_ij == 0:
                continue
            term = homo_mul_const(c_zv[j], b_ij, pub)
            acc = term if acc is None else homo_add(acc, term, pub)
        if acc is None:
            x_hat_k.append(alpha_hat[i])
        else:
            x_hat_k.append(homo_add(alpha_hat[i], acc, pub))
else:
    # 对角近似（大规模快速版）
    B_bar_diag = quantize2_safe(B_k, delta, ZMIN, ZMAX)
    for i in range(Nk):
        b_ii = int(B_bar_diag[i])
        term = homo_mul_const(c_zv[i], b_ii, pub)
        c_final = homo_add(alpha_hat[i], term, pub)
        x_hat_k.append(c_final)

with open(f'/mnt/result_k{k}.pkl', 'wb') as f:
    pickle.dump({'x_hat_k': x_hat_k}, f)
print(f"边缘节点 {k} 计算完成 ({'完整矩阵' if is_full_matrix else '对角近似'})")
