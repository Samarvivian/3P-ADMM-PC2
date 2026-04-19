import sys
import pickle
import numpy as np
sys.path.append('/mnt/3p-admm-pc2')
from crypto.paillier import encrypt, homo_add, homo_mul_const

def quantize2_safe(v, delta, zmin, zmax):
    v = np.array(v, dtype=np.float64)
    # 先clip到范围内
    v = np.clip(v, zmin, zmax)
    result = np.floor(delta * (v - zmin) / (zmax - zmin)).astype(object)
    result = np.clip(result, 0, int(delta))
    return result

k = int(sys.argv[1])

with open(f'/mnt/edge_data_k{k}.pkl', 'rb') as f:
    data = pickle.load(f)

pub       = data['pub']
alpha_hat = data['alpha_hat']
B_bar_q   = data['B_bar_q']
delta     = data['delta']
ZMIN      = data['ZMIN']
ZMAX      = data['ZMAX']

with open(f'/mnt/iter_data_k{k}.pkl', 'rb') as f:
    iter_data = pickle.load(f)

zk = iter_data['zk']
vk = iter_data['vk']
Nk = len(zk)

q_z = quantize2_safe(zk,  delta, ZMIN, ZMAX)
q_v = quantize2_safe(-vk, delta, ZMIN, ZMAX)

c_z = [encrypt(int(qi), pub) for qi in q_z]
c_v = [encrypt(int(qi), pub) for qi in q_v]

x_hat_k = []
for i in range(Nk):
    c_sum   = homo_add(c_z[i], c_v[i], pub)
    c_mul   = homo_mul_const(c_sum, int(B_bar_q[i]), pub)
    c_final = homo_add(alpha_hat[i], c_mul, pub)
    x_hat_k.append(c_final)

with open(f'/mnt/result_k{k}.pkl', 'wb') as f:
    pickle.dump({'x_hat_k': x_hat_k}, f)

print(f"边缘节点 {k} 计算完成")
