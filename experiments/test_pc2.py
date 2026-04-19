import numpy as np
import sys
sys.path.append('/mnt/3p-admm-pc2')
from admm.distributed import dis_admm
from protocol.admm_pc2 import admm_pc2

np.random.seed(42)
M, N, K = 50, 99, 3
sparsity = 0.1

x_true = np.zeros(N)
idx = np.random.choice(N, int(N*sparsity), replace=False)
x_true[idx] = np.random.randn(int(N*sparsity))

A = np.random.randn(M, N) / np.sqrt(M)
y = A @ x_true + 0.01 * np.random.randn(M)

print("="*50)
print("运行 Dis-ADMM（基准）...")
_, mse_dis = dis_admm(A, y, K=K, rho=1.0, lam=0.05, max_iter=100)

print("="*50)
print("运行 3P-ADMM-PC2...")
_, mse_pc2 = admm_pc2(A, y, K=K, rho=1.0, lam=0.05, max_iter=100,
                       delta=10**10, bits=1024)

print("="*50)
print(f"Dis-ADMM   最终MSE: {mse_dis[-1]:.8f}")
print(f"3P-ADMM-PC2最终MSE: {mse_pc2[-1]:.8f}")
print(f"MSE差值: {abs(mse_pc2[-1]-mse_dis[-1]):.2e}")
print(f"论文期望差值约为: 1e-14")
