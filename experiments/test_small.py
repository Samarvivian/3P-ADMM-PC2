import numpy as np
import sys
sys.path.append('/mnt/3p-admm-pc2')
from admm.centralized import cen_admm

# 生成小规模测试数据
np.random.seed(42)
M, N = 50, 100
sparsity = 0.1

# 生成真实稀疏信号
x_true = np.zeros(N)
k = int(N * sparsity)
idx = np.random.choice(N, k, replace=False)
x_true[idx] = np.random.randn(k)

# 生成观测矩阵和观测值
A = np.random.randn(M, N) / np.sqrt(M)
y = A @ x_true + 0.01 * np.random.randn(M)

print(f"矩阵维度: A={A.shape}, y={y.shape}")
print(f"真实信号非零元素个数: {np.sum(x_true != 0)}")
print("="*50)

# 运行Cen-ADMM
x_est, mse_list = cen_admm(A, y, rho=1.0, lam=0.05, max_iter=100)

# 验证结果
final_mse = mse_list[-1]
recovery_error = np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true)

print("="*50)
print(f"最终MSE: {final_mse:.8f}")
print(f"信号恢复误差: {recovery_error:.6f}")
print(f"迭代轮数: {len(mse_list)}")

if recovery_error < 0.1:
    print("✓ 算法验证通过！恢复误差小于10%")
else:
    print("✗ 恢复误差偏大，需要调整参数")
