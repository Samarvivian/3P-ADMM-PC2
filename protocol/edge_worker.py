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
alpha_hat = data['alpha_hat']  # 加密的Bk*Ak^T*y，长度Nk
B_k       = data['B_k']        # 完整矩阵Bk，Nk×Nk，明文
rho       = data['rho']
delta     = data['delta']
ZMIN      = data['ZMIN']
ZMAX      = data['ZMAX']

with open(f'/mnt/iter_data_k{k}.pkl', 'rb') as f:
    iter_data = pickle.load(f)

zk = iter_data['zk']   # 已经乘以rho
vk = iter_data['vk']   # 已经乘以rho
Nk = len(zk)

# 量化z和v
q_z = quantize2_safe(zk,  delta, ZMIN, ZMAX)
q_v = quantize2_safe(-vk, delta, ZMIN, ZMAX)

# 加密z和v
c_z = [encrypt(int(qi), pub) for qi in q_z]
c_v = [encrypt(int(qi), pub) for qi in q_v]

# 计算 c_zv[j] = encrypt(q_z[j] + q_v[j])
c_zv = [homo_add(c_z[j], c_v[j], pub) for j in range(Nk)]

# 量化Bk（整个矩阵）
# B_bar_k[i,j] = Γ2(Bk[i,j])
B_bar_k = quantize2_safe(B_k.flatten(), delta, ZMIN, ZMAX).reshape(Nk, Nk)

# 同态矩阵向量乘法：
# x_hat[i] = alpha_hat[i] + sum_j(B_bar_k[i,j] * c_zv[j])
n2 = pub[0] * pub[0]
x_hat_k = []
for i in range(Nk):
    # 先算 B_bar_k[i,:] * c_zv
    # 第一个非零项
    acc = None
    for j in range(Nk):
        b_ij = int(B_bar_k[i, j])
        if b_ij == 0:
            continue
        term = homo_mul_const(c_zv[j], b_ij, pub)
        if acc is None:
            acc = term
        else:
            acc = homo_add(acc, term, pub)

    if acc is None:
        # 所有B_bar_k[i,:]都是0，直接用alpha_hat
        x_hat_k.append(alpha_hat[i])
    else:
        c_final = homo_add(alpha_hat[i], acc, pub)
        x_hat_k.append(c_final)

with open(f'/mnt/result_k{k}.pkl', 'wb') as f:
    pickle.dump({'x_hat_k': x_hat_k}, f)
print(f"边缘节点 {k} 计算完成")
