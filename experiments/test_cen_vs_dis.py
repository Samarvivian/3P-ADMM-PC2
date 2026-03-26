import numpy as np
import sys
sys.path.append('/mnt/3p-admm-pc2')
from admm.centralized import cen_admm
from admm.distributed import dis_admm

# 生成测试数据
np.random.seed(42)
M, N, K = 50, 99, 3  # N必须能被K整除
sparsity = 0.1

x_true = np.zeros(N)
idx = np.random.choice(N, int(N*sparsity), replace=False)
x_true[idx] = np.random.randn(int(N*sparsity))

A = np.random.randn(M, N) / np.sqrt(M)
y = A @ x_true + 0.01 * np.random.randn(M)

print("="*50)
print("运行 Cen-ADMM...")
x_cen, mse_cen = cen_admm(A, y, rho=1.0, lam=0.05, max_iter=100)

print("="*50)
print("运行 Dis-ADMM...")
x_dis, mse_dis = dis_admm(A, y, K=K, rho=1.0, lam=0.05, max_iter=100)

print("="*50)
print(f"Cen-ADMM 最终MSE: {mse_cen[-1]:.6f}")
print(f"Dis-ADMM 最终MSE: {mse_dis[-1]:.6f}")
print(f"MSE差值: {abs(mse_dis[-1] - mse_cen[-1]):.6f}")
print(f"（论文中差值约为0.07）")
