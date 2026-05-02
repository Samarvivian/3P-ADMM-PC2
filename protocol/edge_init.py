"""
边缘节点初始化：计算Bk = (Ak^T Ak + rho*I)^{-1} 和 alpha_k = Bk * Ak^T * y
"""
import sys
import pickle
import numpy as np
from numpy.linalg import inv

k = int(sys.argv[1])

with open(f'/mnt/init_data_k{k}.pkl', 'rb') as f:
    data = pickle.load(f)

Ak = data['Ak']
y  = data['y']
rho = data['rho']

print(f'边缘节点{k}: 计算Bk ({Ak.shape[1]}x{Ak.shape[1]})...')
Nk = Ak.shape[1]
Bk = inv(Ak.T @ Ak + rho * np.eye(Nk))
alpha_k = Bk @ (Ak.T @ y)
print(f'边缘节点{k}: Bk计算完成')

with open(f'/mnt/init_result_k{k}.pkl', 'wb') as f:
    pickle.dump({'Bk': Bk, 'alpha_k': alpha_k}, f)
print(f'边缘节点{k}: 结果已保存')
